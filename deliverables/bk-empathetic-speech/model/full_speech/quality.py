"""Content-supervised speech architecture v2. / 내용 감독 음성 구조 v2."""

from dataclasses import dataclass
import math
import re
import unicodedata

import torch
from torch import nn
import torch.nn.functional as F

from .config import SpeechConfig
from .system import EmpatheticSpeechSystem
from .planners import SemanticResponsePlanner
from .tensor_ops import align, counts, mask_from_lengths, masked_mean, masked_mse, sinusoidal


ALPHABET = " abcdefghijklmnopqrstuvwxyz'0123456789"
VOCAB_SIZE = len(ALPHABET) + 1  # Zero is CTC blank. / 0은 CTC 공백입니다.


def normalize_text(text):
    text = unicodedata.normalize('NFKD', text.replace('’', "'")).encode('ascii', 'ignore').decode().lower()
    text = re.sub(r"[^a-z0-9' ]", ' ', text)
    return ' '.join(text.split())


def text_ids(text):
    text = normalize_text(text)
    if not text:
        raise ValueError('Transcript has no supported characters')
    return torch.tensor([ALPHABET.index(char) + 1 for char in text], dtype=torch.long)


@dataclass
class QualityConfig(SpeechConfig):
    semantic_dim: int = 768
    semantic_hz: float = 50.0
    num_speakers: int = 2
    max_duration: float = 120.0
    acoustic_epochs: int = 5
    semantic_epochs: int = 5
    mix_ramp_epochs: int = 20
    predicted_semantic_steps: int = 4
    acoustic_loss_every: int = 8
    content_weight: float = .1
    spectral_weight: float = .1
    relevance_weight: float = .05
    planner_memory_mode: str = 'fused'

    def __post_init__(self):
        super().__post_init__()
        if (self.semantic_dim, self.semantic_hz) != (768, 50.0):
            raise ValueError('Quality v2 requires native-width HuBERT at 50 Hz')
        if self.planner_memory_mode not in ('fused', 'native_speech', 'resampled_speech'):
            raise ValueError('Unknown planner memory mode')
        if min(self.acoustic_epochs, self.semantic_epochs) < 0 or min(
                self.mix_ramp_epochs, self.predicted_semantic_steps, self.acoustic_loss_every) < 1:
            raise ValueError('Invalid curriculum schedule')

    def target_contract(self):
        return {**super().target_contract(), 'representation': 'hubert_native_width_v2',
                'text_alphabet': ALPHABET, 'person_a_speech_dim': 768}


class ContentHead(nn.Module):
    def __init__(self, dimension, hidden=256):
        super().__init__()
        self.input = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, hidden))
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden, 4, hidden * 2,
            dropout=0.0, batch_first=True), 1, enable_nested_tensor=False)
        self.output = nn.Linear(hidden, VOCAB_SIZE)

    def forward(self, values, mask):
        positions = sinusoidal(torch.arange(values.shape[1], device=values.device), self.input[1].out_features)
        hidden = self.input(values) + positions[None].to(values.dtype)
        return self.output(self.encoder(hidden, src_key_padding_mask=~mask))


def content_ctc(head, values, mask, labels, label_lengths):
    # Repeated letters also need blank frames. / 반복 글자에도 공백 프레임이 필요합니다.
    lengths = mask.sum(1)
    repeats = ((labels[:, 1:] == labels[:, :-1]) &
               (torch.arange(labels.shape[1] - 1, device=labels.device)[None] < label_lengths[:, None] - 1)).sum(1)
    if ((label_lengths < 1) | (lengths < label_lengths + repeats)).any():
        raise ValueError('CTC target exceeds available frames; fix data rather than zeroing the loss')
    logits = head(values, mask).float().log_softmax(-1).transpose(0, 1)
    return F.ctc_loss(logits, labels, lengths, label_lengths, blank=0, reduction='mean', zero_infinity=False)


class ContentSemanticPlanner(SemanticResponsePlanner):
    def estimate(self, target, mask, context, context_mask, affect, style, *, affect_mask=None):
        # Velocity targets avoid dividing by near-zero alpha. / 속도 타깃으로 작은 알파 나눗셈을 피합니다.
        time = torch.rand(target.shape[0], device=target.device) * .998 + .001
        alpha, sigma = self.schedule(time)
        noise = torch.randn_like(target)
        noisy = alpha * target + sigma * noise
        affect_mask = context_mask if affect_mask is None else affect_mask
        prediction = self.denoiser(noisy, time, align(affect, affect_mask, mask), context, style, mask, context_mask)
        clean = (alpha * noisy - sigma * prediction).masked_fill(~mask[..., None], 0)
        return masked_mse(prediction, alpha * noise - sigma * target, mask), clean

    def sample(self, mask, context, context_mask, affect, style, steps, generator=None, last_step_grad=False, *,
               affect_mask=None):
        if steps < 1:
            raise ValueError('Need positive semantic sampling steps')
        x = torch.randn((*mask.shape, self.config.semantic_dim), device=context.device, generator=generator)
        condition = align(affect, context_mask if affect_mask is None else affect_mask, mask)
        times = torch.linspace(1, 0, steps + 1, device=context.device)
        for index, (current, following) in enumerate(zip(times[:-1], times[1:])):
            # Joint training differentiates the final denoising step. / 공동 학습은 마지막 복원 단계를 미분합니다.
            enabled = torch.is_grad_enabled() and (not last_step_grad or index == steps - 1)
            with torch.set_grad_enabled(enabled):
                alpha, sigma = self.schedule(current.expand(x.shape[0]))
                velocity = self.denoiser(x, current.expand(x.shape[0]), condition, context, style, mask, context_mask)
                clean = alpha * x - sigma * velocity
                noise = sigma * x + alpha * velocity
                next_alpha, next_sigma = self.schedule(following.expand(x.shape[0]))
                x = (next_alpha * clean + next_sigma * noise).masked_fill(~mask[..., None], 0)
        return x


def multiresolution_spectral(predicted, reference):
    losses = []
    for fft in (256, 512, 1024):
        window = torch.hann_window(fft, device=predicted.device)
        first = torch.stft(predicted.float(), fft, fft // 4, window=window, return_complex=True).abs().clamp_min(1e-5)
        second = torch.stft(reference.float(), fft, fft // 4, window=window, return_complex=True).abs().clamp_min(1e-5)
        losses.append((first - second).norm() / second.norm().clamp_min(1e-5) +
                      F.l1_loss(first.log(), second.log()))
    return sum(losses) / len(losses)


class QualitySpeechSystem(EmpatheticSpeechSystem):
    def __init__(self, config=None):
        super().__init__(config or QualityConfig())
        cfg = self.config
        self.semantic_planner = ContentSemanticPlanner(cfg)
        self.speech_projection = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 512))
        layer = nn.TransformerEncoderLayer(512, 4, 1024, dropout=cfg.dropout, batch_first=True)
        self.speech_context = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.context_norm = nn.LayerNorm(512)
        self.affect_style = nn.Linear(cfg.hidden_dim, 512, bias=False)
        self.input_content = ContentHead(512)
        self.semantic_content = ContentHead(768)
        self.codec_content = ContentHead(128)
        self.relevance = nn.Linear(512, 768)
        self.current_epoch = 0
        self.current_step = 0
        # External decoder is injected without checkpointing its frozen weights. / 고정 외부 코덱은 체크포인트에서 제외합니다.
        object.__setattr__(self, '_acoustic_codec', None)

    def stage(self):
        if self.current_epoch < self.config.acoustic_epochs:
            return 'acoustic'
        if self.current_epoch < self.config.acoustic_epochs + self.config.semantic_epochs:
            return 'semantic'
        return 'joint'

    def predicted_fraction(self):
        start = self.config.acoustic_epochs + self.config.semantic_epochs
        return min(1.0, max(0.0, (self.current_epoch - start + 1) / self.config.mix_ramp_epochs))

    def codec_times(self, batch_size, device):
        # Keep the original experiment reproducible. / 기존 실험의 재현성을 유지합니다.
        return torch.rand(batch_size, device=device) * .8 + .1

    def waveform_objectives(self, waveform, batch):
        return {}

    def encode_batch(self, batch):
        encoded = super().encode_batch(batch)
        speech = batch['speech_a']
        speech_mask = mask_from_lengths(batch['speech_a_len'])
        positions = sinusoidal(torch.arange(speech.shape[1], device=speech.device), 512)
        hidden = self.speech_projection(speech) + positions[None].to(speech.dtype)
        hidden = self.speech_context(hidden, src_key_padding_mask=~speech_mask)
        context = self.context_norm(encoded['context'] + align(hidden, speech_mask, encoded['context_mask']))
        context = context.masked_fill(~encoded['context_mask'][..., None], 0)
        style = self.style_embedding(batch['style_id'])
        affect = self.encoder.transport(context + self.affect_style(style)[:, None], batch['au'],
                                        encoded['context_mask'], batch.get('au_len'))
        encoded.update(context=context, affect=affect.trajectory, affect_summary=affect.mean_pool(),
                       speech_hidden=hidden, speech_mask=speech_mask)
        return encoded

    def planner_inputs(self, encoded):
        # Only Module 5 changes memory; retain the original affect timeline. / 모듈 5의 문맥만 바꾸고 감정 시간축은 유지합니다.
        mode = self.config.planner_memory_mode
        memory, memory_mask = encoded['context'], encoded['context_mask']
        if mode in ('native_speech', 'resampled_speech'):
            memory, memory_mask = encoded['speech_hidden'], encoded['speech_mask']
            if mode == 'resampled_speech':
                # Match production downsampling before returning to the native length. / 실제 다운샘플링 후 원래 길이로 되돌립니다.
                downsampled = align(memory, memory_mask, encoded['context_mask'])
                memory = align(downsampled, encoded['context_mask'], memory_mask)
            else:
                memory = memory.masked_fill(~memory_mask[..., None], 0)
        elif mode != 'fused':
            raise ValueError('Unknown planner memory mode')
        return {'context': memory, 'context_mask': memory_mask, 'affect': encoded['affect'],
                'affect_mask': encoded['context_mask']}

    def losses(self, batch):
        cfg, stage = self.config, self.stage()
        encoded = self.encode_batch(batch)
        context, mask, affect = encoded['context'], encoded['context_mask'], encoded['affect']
        style, speaker = self.embeddings(batch['style_id'], batch['speaker_id'], context.shape[0])
        semantic, semantic_mask = self.target(batch, 'semantic', 768)
        codec, codec_mask = self.target(batch, 'codec', 128)
        semantic = ((semantic - self.semantic_mean) / self.semantic_std).masked_fill(~semantic_mask[..., None], 0)
        codec = ((codec - self.codec_mean) / self.codec_std).masked_fill(~codec_mask[..., None], 0)
        target_affect, target_mask = self.target(batch, 'affect', 6)
        weights = align(batch['affect_weight'], target_mask, mask).clamp_min(0)
        target_affect = align(target_affect * batch['affect_weight'], target_mask, mask) / weights.clamp_min(1e-8)
        affect_loss = (((affect - target_affect).square() * weights).sum((1, 2)) / weights.sum((1, 2)).clamp_min(1)).mean()
        # Unlabeled dimensions stay neutral until supervised. / 미라벨 차원은 감독 전까지 중립으로 규제합니다.
        neutral = affect.new_tensor([0.0, .5, .5])
        affect_prior = (affect[..., [0, 1, 5]] - neutral).square()[mask].mean()
        _, log_duration = self.length_predictor(context, affect, mask, style)
        planner_inputs = self.planner_inputs(encoded)
        semantic_loss, clean_semantic = self.semantic_planner.estimate(semantic, semantic_mask,
                                                                      style=style, **planner_inputs)
        ctc_args = (batch['text_b'], batch['text_b_len'])
        target_ctc = content_ctc(self.semantic_content, semantic.detach(), semantic_mask, *ctc_args)
        predicted_ctc = content_ctc(self.semantic_content, clean_semantic, semantic_mask, *ctc_args)
        input_ctc = content_ctc(self.input_content, encoded['speech_hidden'], encoded['speech_mask'],
                               batch['text_a'], batch['text_a_len'])
        local_semantic = semantic
        generated_ctc = semantic_loss * 0
        if stage == 'joint':
            generated = self.semantic_planner.sample(semantic_mask, style=style,
                steps=cfg.predicted_semantic_steps, last_step_grad=True, **planner_inputs)
            choose = torch.rand(semantic.shape[0], 1, 1, device=semantic.device) < self.predicted_fraction()
            # Whole-sequence replacement matches inference better than averaging. / 전체 시퀀스 교체로 추론 조건을 모방합니다.
            local_semantic = torch.where(choose, generated, semantic)
            generated_ctc = content_ctc(self.semantic_content, generated, semantic_mask, *ctc_args)
        elif stage == 'acoustic':
            local_semantic = semantic + .05 * torch.randn_like(semantic)
        time = self.codec_times(codec.shape[0], codec.device)
        t = time[:, None, None]
        noise = torch.randn_like(codec)
        mixed = ((1 - t) * noise + t * codec).masked_fill(~codec_mask[..., None], 0)
        conditions = self.codec_generator.conditions(local_semantic, semantic_mask, affect, mask, codec_mask)
        velocity = self.codec_generator.velocity(mixed, time, conditions, context, style + speaker, codec_mask, mask)
        flow_loss = masked_mse(velocity, codec - noise, codec_mask)
        clean_codec = mixed + (1 - t) * velocity
        codec_ctc = content_ctc(self.codec_content, codec.detach(), codec_mask, *ctc_args)
        output_ctc = content_ctc(self.codec_content, clean_codec, codec_mask, *ctc_args)
        # Duplicate conversations are multiple positive answers. / 같은 대화의 응답은 모두 양성입니다.
        query = F.normalize(self.relevance(masked_mean(context, mask)), dim=-1)
        target = F.normalize(masked_mean(semantic, semantic_mask).detach(), dim=-1)
        similarities = query @ target.T / .1
        positives = batch['conversation_key'][:, None] == batch['conversation_key'][None, :]
        relevance = (similarities.logsumexp(1) - similarities.masked_fill(~positives, -torch.inf).logsumexp(1)).mean()
        spectral = clean_codec.sum() * 0
        waveform_terms = {}
        if self.training and stage != 'semantic' and self._acoustic_codec is not None and self.current_step % cfg.acoustic_loss_every == 0:
            length = int(batch['codec_len'][0])
            values = clean_codec[:1, :length].float() * self.codec_std + self.codec_mean
            # Frozen parameters still permit gradients into the latents. / 고정 가중치도 잠재값으로의 기울기는 허용합니다.
            # cuDNN forbids eval-mode LSTM backward; native kernels keep the codec frozen. / 고정 LSTM 미분은 기본 커널을 사용합니다.
            with torch.autocast(device_type=codec.device.type, enabled=False), torch.backends.cudnn.flags(enabled=False):
                waveform = self._acoustic_codec.model.decoder(values.transpose(1, 2))[0, 0]
                wanted = min(waveform.numel(), int(batch['waveform_len'][0]))
                spectral = multiresolution_spectral(waveform[:wanted], batch['waveform'][0, :wanted])
                waveform_terms = self.waveform_objectives(waveform[:wanted], batch)
        losses = {'affect': affect_loss, 'affect_prior': .01 * affect_prior,
                  'duration': F.smooth_l1_loss(log_duration.float(), batch['duration'].float().log()),
                  'semantic': semantic_loss * (stage != 'acoustic'),
                  'codec_flow': flow_loss * (stage != 'semantic'),
                  'input_ctc': cfg.content_weight * input_ctc,
                  'target_ctc': cfg.content_weight * (target_ctc + codec_ctc),
                  'semantic_ctc': cfg.content_weight * (predicted_ctc + generated_ctc) * (stage != 'acoustic'),
                  'codec_ctc': cfg.content_weight * output_ctc * (stage != 'semantic'),
                  'relevance': cfg.relevance_weight * relevance,
                  'spectral': cfg.spectral_weight * spectral}
        losses.update(waveform_terms)
        losses['total'] = sum(losses.values())
        return losses

    @torch.inference_mode()
    def generate_batch(self, batch, codec, seed=42, oracle_semantic=None, oracle_duration=None):
        if self.training:
            raise RuntimeError('Generation requires eval mode')
        encoded = self.encode_batch(batch)
        context, mask, affect = encoded['context'], encoded['context_mask'], encoded['affect']
        style, speaker = self.embeddings(batch['style_id'], batch['speaker_id'], len(context))
        duration, _ = self.length_predictor(context, affect, mask, style)
        duration = duration if oracle_duration is None else oracle_duration
        sem_mask = mask_from_lengths(counts(duration, self.config.semantic_hz))
        codec_lengths = counts(duration, self.config.codec_hz)
        codec_mask = mask_from_lengths(codec_lengths)
        generator = torch.Generator(device=context.device).manual_seed(seed)
        semantic = self.semantic_planner.sample(sem_mask, style=style, steps=self.config.semantic_steps,
            generator=generator, **self.planner_inputs(encoded)) if oracle_semantic is None else oracle_semantic
        latents = self.codec_generator.sample(codec_mask, semantic, sem_mask, context, mask, affect,
                                             style + speaker, self.config.codec_steps, generator)
        latents = latents * self.codec_std + self.codec_mean
        audio, lengths = codec.decode(latents.float(), codec_lengths, (duration * 32000).round().long())
        return {**encoded, 'waveform': audio, 'audio_lengths': lengths, 'duration': duration,
                'semantic': semantic, 'codec_latents': latents}

    def checkpoint(self, training_steps=0, **metadata):
        result = super().checkpoint(training_steps, **metadata)
        result.update(architecture='llm_free_speech_quality_v2', current_epoch=self.current_epoch)
        return result

    @classmethod
    def from_checkpoint(cls, path):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload.get('architecture') != 'llm_free_speech_quality_v2':
            raise ValueError('Expected quality-v2 checkpoint')
        model = cls(QualityConfig(**payload['config']))
        model.load_state_dict(payload['state_dict'], strict=True)
        model.current_epoch = payload.get('current_epoch', 0)
        return model.eval(), payload
