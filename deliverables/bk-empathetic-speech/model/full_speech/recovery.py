"""Fixed-condition recovery stages. / 조건을 고정한 복구 학습 단계."""

import math
import torch
from torch import nn
import torch.nn.functional as F

from .quality import ALPHABET, QualitySpeechSystem


ASR_MODEL = 'facebook/wav2vec2-base-960h'
ASR_REVISION = '22aad52d435eb6dbaf354bdad9b0da84ce7d6156'


def downsample_speech(waveform):
    # Anti-alias before 32→16 kHz decimation; preserve gradients. / 미분을 유지하며 저역통과 후 다운샘플링합니다.
    positions = torch.arange(-31, 32, device=waveform.device, dtype=waveform.dtype)
    kernel = .48 * torch.sinc(.48 * positions) * torch.hann_window(63, periodic=False,
                                                               device=waveform.device, dtype=waveform.dtype)
    kernel = kernel / kernel.sum()
    padded = F.pad(waveform[None, None], (31, 31), mode='reflect')
    return F.conv1d(padded, kernel[None, None], stride=2)[0, 0]


class FrozenWaveformCTC(nn.Module):
    def __init__(self):
        super().__init__()
        from huggingface_hub import hf_hub_download
        from transformers import AutoProcessor, Wav2Vec2Config, Wav2Vec2ForCTC
        self.processor = AutoProcessor.from_pretrained(ASR_MODEL, revision=ASR_REVISION)
        config = Wav2Vec2Config.from_pretrained(ASR_MODEL, revision=ASR_REVISION)
        config.mask_time_prob = config.mask_feature_prob = 0.
        self.recognizer = Wav2Vec2ForCTC(config)
        path = hf_hub_download(ASR_MODEL, 'pytorch_model.bin', revision=ASR_REVISION)
        state = torch.load(path, map_location='cpu', weights_only=True)
        # Translate old weight-norm names without reinitializing speech weights. / 기존 정규화 이름을 변환해 가중치를 보존합니다.
        expected = self.recognizer.state_dict()
        for suffix, replacement in [('weight_g', 'parametrizations.weight.original0'),
                                    ('weight_v', 'parametrizations.weight.original1')]:
            old = 'wav2vec2.encoder.pos_conv_embed.conv.' + suffix
            new = 'wav2vec2.encoder.pos_conv_embed.conv.' + replacement
            if old in state and new in expected:
                state[new] = state.pop(old)
        self.recognizer.load_state_dict(state, strict=True)
        self.recognizer.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(False)
        return self

    def forward(self, waveform, labels):
        text = ''.join(ALPHABET[int(x) - 1] for x in labels)
        ids = self.processor.tokenizer(text.upper(), add_special_tokens=False).input_ids
        if self.processor.tokenizer.unk_token_id in ids:
            raise ValueError('ASR teacher cannot represent this transcript; normalize it explicitly')
        logits = self.waveform_logits(waveform)
        target = torch.tensor(ids, device=waveform.device, dtype=torch.long)
        required = len(ids) + sum(a == b for a, b in zip(ids, ids[1:]))
        if logits.shape[1] < required:
            raise ValueError('ASR teacher target cannot align to waveform')
        return F.ctc_loss(logits.log_softmax(-1).transpose(0, 1), target,
                         torch.tensor([logits.shape[1]], device=waveform.device),
                         torch.tensor([len(ids)], device=waveform.device),
                         blank=self.recognizer.config.pad_token_id, zero_infinity=False)

    def waveform_logits(self, waveform):
        # Frozen weights still transmit gradients to speech. / 고정 가중치도 음성으로 기울기를 전달합니다.
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            audio = downsample_speech(waveform.float())
            audio = (audio - audio.mean()) / (audio.var(unbiased=False) + 1e-7).sqrt()
            return self.recognizer(audio[None]).logits.float()


class RecoverySpeechSystem(QualitySpeechSystem):
    def configure_recovery(self, phase, predicted_fraction=.2, teacher=None, teacher_weight=.05):
        self.unit_prior_training = False
        # Stage changes must clear an earlier experimental objective. / 단계 변경 시 이전 실험 목적함수를 해제합니다.
        self.unit_objective_recipe = None
        self.planner_audio_recipe = None
        if phase not in ('acoustic', 'planner', 'joint') or not 0 <= predicted_fraction <= .8:
            raise ValueError('Unknown recovery stage or missing reference anchor')
        if self.config.semantic_steps != self.config.predicted_semantic_steps:
            raise ValueError('Training and inference semantic budgets must match')
        self.recovery_phase = phase
        self.training_fraction = 1. if phase == 'planner' else predicted_fraction
        self.validation_fraction = 1.0
        self.teacher_weight = teacher_weight
        object.__setattr__(self, '_waveform_teacher', teacher)
        self.requires_grad_(False)
        names = {'acoustic': ['codec_generator', 'codec_content'],
                 'planner': ['semantic_planner', 'length_predictor'],
                 'joint': ['semantic_planner', 'codec_generator', 'length_predictor']}[phase]
        for name in names:
            getattr(self, name).requires_grad_(True)
        self.trainable_components = names
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, 'trainable_components'):
            # Frozen conditioning stays deterministic. / 고정 조건망은 결정적으로 실행합니다.
            for name, module in self.named_children():
                if name not in self.trainable_components:
                    module.eval()
        if getattr(self, 'sampled_audio_training', False):
            self.codec_generator.eval()  # Match inference dropout without disabling gradients. / 추론과 같은 드롭아웃 설정에서 미분합니다.
        return self

    def configure_sampled_audio(self, flow_weight=1., waveform_samples=1):
        if self.recovery_phase != 'acoustic' or self._waveform_teacher is None:
            raise ValueError('Sampled audio requires acoustic training and a frozen teacher')
        if not math.isfinite(flow_weight) or flow_weight < 0:
            raise ValueError('Flow weight must be finite and nonnegative')
        if type(waveform_samples) is not int or waveform_samples < 1:
            raise ValueError('Waveform sample count must be a positive integer')
        self.sampled_flow_weight = flow_weight
        self.waveform_samples = waveform_samples
        self.sampled_audio_training = True
        self.codec_content.requires_grad_(False)
        self.trainable_components = ['codec_generator']
        self._waveform_teacher.requires_grad_(False).eval()
        self.train(self.training)

    def sampled_waveform(self, batch, seed=42, index=0):
        if not 0 <= index < len(batch['mel']):
            raise ValueError('Waveform index outside batch')
        # Keep all inputs and targets paired with this utterance. / 입력과 타깃을 같은 발화로 선택합니다.
        batch = {key: value[index:index + 1] for key, value in batch.items()}
        for key, value in list(batch.items()):
            length_key = ('affect' if key == 'affect_weight' else key) + '_len'
            if length_key in batch and value.ndim >= 2:
                batch[key] = value[:, :int(batch[length_key][0])]
        # Use the real full sampler, not a one-step clean estimate. / 한 단계 추정 대신 실제 전체 샘플러를 씁니다.
        with torch.autocast(device_type=batch['mel'].device.type, enabled=False):
            with torch.no_grad():
                encoded = self.encode_batch(batch)
                style, speaker = self.embeddings(batch['style_id'], batch['speaker_id'], len(batch['mel']))
                semantic, semantic_mask = self.target(batch, 'semantic', 768)
                semantic = (semantic - self.semantic_mean) / self.semantic_std
            length = int(batch['codec_len'][0])
            mask = torch.ones(1, length, device=semantic.device, dtype=torch.bool)
            sem_length = int(batch['semantic_len'][0])
            context_length = int(encoded['context_mask'][0].sum())
            generator = torch.Generator(device=semantic.device).manual_seed(seed)
            latents = self.codec_generator.sample(mask, semantic[:1, :sem_length], semantic_mask[:1, :sem_length],
                encoded['context'][:1, :context_length], encoded['context_mask'][:1, :context_length],
                encoded['affect'][:1, :context_length], style[:1] + speaker[:1],
                self.config.codec_steps, generator, checkpoint_grad=True)
            values = latents * self.codec_std + self.codec_mean
            # Frozen decoder remains differentiable; avoid eval-mode cuDNN backward. / 고정 디코더를 기본 커널로 미분합니다.
            with torch.backends.cudnn.flags(enabled=False):
                wave = self._acoustic_codec.model.decoder(values.transpose(1, 2))[0, 0]
            wanted = int(batch['waveform_len'][0])
            if wanted > len(wave):
                raise ValueError('Sampled waveform shorter than reference')
            return wave[:wanted]

    def sampled_audio_losses(self, batch):
        from .quality import multiresolution_spectral
        with torch.no_grad():
            encoded = self.encode_batch(batch)
            style, speaker = self.embeddings(batch['style_id'], batch['speaker_id'], len(batch['mel']))
            semantic, semantic_mask = self.target(batch, 'semantic', 768)
            codec, codec_mask = self.target(batch, 'codec', 128)
            semantic = (semantic - self.semantic_mean) / self.semantic_std
            codec = (codec - self.codec_mean) / self.codec_std
        anchor = self.sampled_flow_weight * self.codec_generator.loss(codec, codec_mask, semantic, semantic_mask,
            encoded['context'], encoded['context_mask'], encoded['affect'], style + speaker)
        ctc, spectral = anchor * 0, anchor * 0
        if not self.training or self.current_step % self.config.acoustic_loss_every == 0:
            size = len(batch['mel'])
            count = min(self.waveform_samples, size) if self.training else size
            start = self.current_step * count if self.training else 0
            # Rotate distinct training items; validate every item. / 학습 발화는 순환하고 검증은 전부 평가합니다.
            ctc_values, spectral_values = [], []
            for offset in range(count):
                index = (start + offset) % size
                seed = 100000 + self.current_step * size + index if self.training else 42
                waveform = self.sampled_waveform(batch, seed=seed, index=index)
                labels = batch['text_b'][index, :int(batch['text_b_len'][index])]
                ctc_values.append(self._waveform_teacher(waveform, labels))
                spectral_values.append(multiresolution_spectral(waveform, batch['waveform'][index, :len(waveform)]))
            ctc = self.teacher_weight * torch.stack(ctc_values).mean()
            spectral = self.config.spectral_weight * torch.stack(spectral_values).mean()
        return {'codec_flow': anchor, 'sampled_ctc': ctc, 'sampled_spectral': spectral,
                'total': anchor + ctc + spectral}

    def stage(self):
        # Planner training includes fully sampled content supervision. / 계획기 학습에 완전 생성 내용 감독을 포함합니다.
        return 'acoustic' if self.recovery_phase == 'acoustic' else 'joint'

    def predicted_fraction(self):
        return self.training_fraction if self.training else self.validation_fraction

    def codec_times(self, batch_size, device):
        # Cover both near-noise and near-clean states. / 잡음·복원 양 끝 부근을 모두 학습합니다.
        return torch.rand(batch_size, device=device)

    def waveform_objectives(self, waveform, batch):
        teacher = self._waveform_teacher
        if teacher is None:
            return {}
        labels = batch['text_b'][0, :int(batch['text_b_len'][0])]
        return {'waveform_ctc': self.teacher_weight * teacher(waveform, labels)}

    def losses(self, batch):
        if getattr(self, 'sampled_audio_training', False):
            return self.sampled_audio_losses(batch)
        values = super().losses(batch)
        allowed = {'acoustic': ('codec_flow', 'target_ctc', 'codec_ctc', 'spectral', 'waveform_ctc'),
                   'planner': ('semantic', 'semantic_ctc', 'duration', 'codec_ctc', 'waveform_ctc'),
                   'joint': ('semantic', 'semantic_ctc', 'duration', 'codec_flow',
                             'codec_ctc', 'spectral', 'waveform_ctc')}[self.recovery_phase]
        # Report excluded terms, but do not optimize them. / 제외한 항은 기록만 하고 최적화하지 않습니다.
        values['total'] = sum(values[name] for name in allowed if name in values)
        return values
