"""A-only sampled planner audio with a biased gradient. / A만 이용한 생성 음성과 근사 기울기."""

from contextlib import contextmanager

import torch
import torch.nn.functional as F

from dataset.quality_speech_dataset import person_a_only
from .quality import ALPHABET
from .tensor_ops import counts, mask_from_lengths
from .units import MaskedUnitPlanner


def _frozen_eval(module, name):
    if any(parameter.requires_grad for parameter in module.parameters()):
        raise ValueError(f'{name} must have frozen parameters')
    if any(child.training for child in module.modules()):
        raise ValueError(f'{name} must remain in eval mode through backward')


@contextmanager
def _inference_modes(model):
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        # Preserve mixed frozen/trainable modes exactly. / 혼합된 고정·학습 모드를 정확히 복구합니다.
        for module, training in modes:
            module.training = training


def planner_a_item(batch, index=0):
    """Select one unpadded A-only item; never inspect B keys. / B 키를 읽지 않고 A 하나만 선택합니다."""
    selected = person_a_only(batch)
    if type(index) is not int or not 0 <= index < len(selected['mel']):
        raise ValueError('Waveform index outside batch')
    selected = {key: value[index:index + 1] for key, value in selected.items()}
    for name in ('mel', 'dmm', 'au', 'speech_a'):
        length = int(selected[name + '_len'][0])
        if not 1 <= length <= selected[name].shape[1]:
            raise ValueError(f'Invalid A length: {name}')
        selected[name] = selected[name][:, :length].detach()
    return selected


def sampled_planner_waveform(model, batch, codec, seed=42, index=0):
    """Return raw 1-D audio and predicted timing; B targets are never read.

    Forward follows generate_batch with checkpoint-configured step budgets.
    The discrete planner uses its existing final-step straight-through gradient:
    this is a biased surrogate, not an exact gradient through argmax or ranking.
    The frozen acoustic sampler differentiates every integration step.
    / 전방 생성은 추론과 같고, 이산 계획기의 역전파는 편향된 마지막 단계 근사입니다.
    """
    if torch.is_inference_mode_enabled():
        raise RuntimeError('Planner waveform gradients cannot run under inference_mode')
    if not isinstance(model.semantic_planner, MaskedUnitPlanner):
        raise ValueError('This helper requires the discrete masked-unit planner')
    if model.config.sample_rate != 32000 or codec.sample_rate != 32000:
        raise ValueError('Planner waveform requires 32 kHz audio')
    # Checkpoint recomputation must see the same dropout mode. / 재계산 중 드롭아웃 모드가 같아야 합니다.
    _frozen_eval(model.codec_generator, 'Acoustic generator')
    _frozen_eval(codec.model.decoder, 'Codec decoder')
    inputs = planner_a_item(batch, index)
    device = inputs['mel'].device
    with _inference_modes(model), torch.autocast(device_type=device.type, enabled=False):
        with torch.no_grad():
            encoded = model.encode_batch(inputs)
            context, mask, affect = encoded['context'], encoded['context_mask'], encoded['affect']
            style, speaker = model.embeddings(inputs['style_id'], inputs['speaker_id'], 1)
            duration, _ = model.length_predictor(context, affect, mask, style)
            if duration.shape != (1,) or not torch.isfinite(duration).all() or (duration <= 0).any():
                raise ValueError('Invalid predicted response duration')
            semantic_lengths = counts(duration, model.config.semantic_hz)
            codec_lengths = counts(duration, model.config.codec_hz)
            semantic_mask = mask_from_lengths(semantic_lengths)
            codec_mask = mask_from_lengths(codec_lengths)
            memory = model.planner_inputs(encoded)
        generator = torch.Generator(device=device).manual_seed(seed)
        semantic = model.semantic_planner.sample(semantic_mask, style=style,
            steps=model.config.semantic_steps, generator=generator, last_step_grad=True, **memory)
        latents = model.codec_generator.sample(codec_mask, semantic, semantic_mask, context, mask,
            affect, style + speaker, model.config.codec_steps, generator, checkpoint_grad=True)
        latents = latents * model.codec_std + model.codec_mean
        # Native LSTM backward supports the frozen eval decoder. / 기본 LSTM은 고정 평가 디코더의 역전파를 지원합니다.
        with torch.backends.cudnn.flags(enabled=False):
            waveform = codec.model.decoder(latents.float().transpose(1, 2))[0, 0]
        wanted = int((duration * 32000).round().long()[0])
        if wanted < 1 or wanted > waveform.numel():
            raise ValueError('Predicted sample length exceeds decoded audio')
        waveform = waveform[:wanted]
        if not torch.isfinite(waveform).all():
            raise RuntimeError('Non-finite sampled planner waveform')
    # Match generate_batch: no extra waveform clipping or normalization. / 추론과 같이 추가 클리핑·정규화하지 않습니다.
    return {'waveform': waveform, 'duration': duration.detach(), 'audio_samples': wanted,
            '_semantic': semantic, '_codec_latents': latents,
            'audio_seconds': wanted / 32000, 'semantic_frames': int(semantic_lengths[0]),
            'codec_frames': int(codec_lengths[0]), 'semantic_steps': model.config.semantic_steps,
            'codec_steps': model.config.codec_steps, 'sample_rate': 32000,
            'gradient_estimator': 'biased final-step straight-through discrete planner; full acoustic sampler'}


def frozen_content_ctc(teacher, waveform, labels):
    """Score already generated audio; labels enter only here. / 생성이 끝난 음성에만 정답 문장을 사용합니다."""
    _frozen_eval(teacher, 'Content evaluator')
    if waveform.ndim != 1 or not waveform.numel() or not torch.isfinite(waveform).all():
        raise ValueError('Expected a finite nonempty waveform')
    if (labels.ndim != 1 or labels.dtype != torch.long or not labels.numel() or
            ((labels < 1) | (labels > len(ALPHABET))).any()):
        raise ValueError('Expected nonempty supported transcript label IDs')
    text = ''.join(ALPHABET[index - 1] for index in labels.detach().cpu().tolist())
    tokenizer = teacher.processor.tokenizer
    ids = tokenizer(text.upper(), add_special_tokens=False).input_ids
    blank = teacher.recognizer.config.pad_token_id
    if not ids or tokenizer.unk_token_id in ids or blank in ids:
        raise ValueError('Content evaluator cannot represent this transcript')
    with torch.autocast(device_type=waveform.device.type, enabled=False):
        logits = teacher.waveform_logits(waveform.float()).float()
        if logits.ndim != 3 or logits.shape[0] != 1 or not torch.isfinite(logits).all():
            raise ValueError('Content evaluator produced invalid logits')
        if not 0 <= blank < logits.shape[-1] or any(not 0 <= value < logits.shape[-1] for value in ids):
            raise ValueError('Content evaluator vocabulary and logits disagree')
        required = len(ids) + sum(first == second for first, second in zip(ids, ids[1:]))
        frames = logits.shape[1]
        if frames < required:
            raise ValueError(f'CTC target requires {required} frames, but predicted audio has {frames}')
        target = torch.tensor(ids, device=waveform.device, dtype=torch.long)
        loss = F.ctc_loss(logits.log_softmax(-1).transpose(0, 1), target,
            torch.tensor([frames], device=waveform.device),
            torch.tensor([len(ids)], device=waveform.device), blank=blank, zero_infinity=False)
    if not torch.isfinite(loss):
        raise RuntimeError('Non-finite sampled planner CTC loss')
    return {'loss': loss, 'ctc_frames': frames, 'ctc_target_tokens': len(ids),
            'ctc_required_frames': required, 'predicted_audio_samples': waveform.numel(),
            'predicted_audio_seconds': waveform.numel() / 32000,
            'target_text': text, 'reference_target_is_not_unique_valid_reply': True}
