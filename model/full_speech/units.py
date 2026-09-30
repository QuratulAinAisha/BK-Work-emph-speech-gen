"""Experimental discrete speech units. / 실험용 이산 음성 단위."""

import math
import torch
from torch import nn
import torch.nn.functional as F

from .dit import ConditionalDiT
from .quality import QualityConfig
from .recovery import RecoverySpeechSystem
from .tensor_ops import align


class SpeechCodebook(nn.Module):
    def __init__(self, centers):
        super().__init__()
        if centers.ndim != 2 or centers.shape[1] != 768 or not torch.isfinite(centers).all():
            raise ValueError('Expected finite K by 768 normalized centers')
        self.register_buffer('centers', centers.float().clone())

    @torch.no_grad()
    def encode(self, values):
        # Full precision keeps nearest-unit assignments stable. / 정밀 계산으로 최근접 단위를 고정합니다.
        shape = values.shape[:-1]
        with torch.autocast(device_type=values.device.type, enabled=False):
            flat, centers = values.reshape(-1, 768).float(), self.centers.float()
            result = []
            for chunk in flat.split(2048):
                distance = chunk.square().sum(1, keepdim=True) + centers.square().sum(1)[None] - 2 * chunk @ centers.T
                result.append(distance.argmin(1))
        return torch.cat(result).reshape(shape)

    def forward(self, values):
        return self.centers[self.encode(values)]


def load_codebook(path, model):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('architecture') != 'bk_speech_codebook_v1':
        raise ValueError('Unknown speech codebook')
    for name in ('semantic_mean', 'semantic_std'):
        if not torch.equal(payload[name], getattr(model, name).detach().cpu()):
            raise ValueError('Codebook normalization differs from acoustic checkpoint')
    return SpeechCodebook(payload['centers']), payload


class MaskedUnitPlanner(nn.Module):
    """Parallel masked-unit prediction. / 마스킹된 음성 단위를 병렬 예측합니다."""

    def __init__(self, config, centers):
        super().__init__()
        self.config = config
        self.codebook = SpeechCodebook(centers)
        self.mask_embedding = nn.Parameter(torch.zeros(768))
        self.denoiser = ConditionalDiT(768, 6, config, config.planner_layers)
        self.denoiser.output[-1] = nn.Linear(config.hidden_dim, len(centers))

    def logits(self, ids, hidden, mask, context, context_mask, affect, style, *, affect_mask=None):
        values = self.codebook.centers[ids]
        values = torch.where(hidden[..., None], self.mask_embedding.to(values.dtype), values)
        fraction = hidden.sum(1).float() / mask.sum(1).clamp_min(1)
        affect_mask = context_mask if affect_mask is None else affect_mask
        return self.denoiser(values, fraction, align(affect, affect_mask, mask),
                             context, style, mask, context_mask)

    def estimate(self, target, mask, context, context_mask, affect, style, *,
                 fully_masked_probability=.5, max_mask_ratio=1., denoising_eval=False, affect_mask=None):
        if not 0 <= fully_masked_probability <= 1 or not .2 <= max_mask_ratio <= 1:
            raise ValueError('Invalid unit masking probabilities')
        if mask.ndim != 2 or mask.dtype != torch.bool or not mask.any(1).all():
            raise ValueError('Every unit sequence needs a valid token')
        ids = self.codebook.encode(target)
        # Include A-only, fully masked sequences during learning. / 학습에도 A만 보는 전체 마스킹을 포함합니다.
        if max_mask_ratio < 1 and (self.training or denoising_eval):
            lengths = mask.sum(1)
            minimum = (lengths.float() * .2).ceil().long().clamp_min(1)
            maximum = (lengths.float() * max_mask_ratio).floor().long()
            maximum = torch.minimum(maximum, lengths - 1).clamp_min(1)
            if self.training:
                ratio = .2 + (max_mask_ratio - .2) * torch.rand(len(mask), device=target.device)
                count = (ratio * lengths).round().long()
                count = torch.minimum(torch.maximum(count, minimum), maximum)
                count = torch.where(torch.rand_like(ratio) < fully_masked_probability, lengths, count)
                order = torch.rand(mask.shape, device=target.device).masked_fill(~mask, 2.)
                hidden = (order.argsort(1).argsort(1) < count[:, None]) & mask
            else:
                # Fixed partial masks measure denoising, not A-only replies. / 고정 부분 마스크로 복원 능력을 검증합니다.
                count = torch.minimum(torch.maximum(lengths // 2, minimum), maximum)
                ordinal = mask.long().cumsum(1) - 1
                hidden = (((ordinal + 1) * count[:, None] // lengths[:, None]) >
                          (ordinal * count[:, None] // lengths[:, None])) & mask
        elif self.training:
            ratio = .2 + .8 * torch.rand(len(mask), 1, device=target.device)
            ratio = torch.where(torch.rand_like(ratio) < fully_masked_probability, 1., ratio)
            hidden = (torch.rand(mask.shape, device=target.device) < ratio) & mask
            hidden.scatter_(1, mask.long().argmax(1, keepdim=True), True)
        else:
            hidden = mask
        logits = self.logits(ids, hidden, mask, context, context_mask, affect, style, affect_mask=affect_mask)
        loss = F.cross_entropy(logits[hidden].float(), ids[hidden])
        probabilities = logits.float().softmax(-1)
        soft = probabilities @ self.codebook.centers
        hard = self.codebook.centers[logits.argmax(-1)]
        clean = hard + soft - soft.detach()
        clean = torch.where(hidden[..., None], clean, self.codebook.centers[ids])
        return loss, clean.masked_fill(~mask[..., None], 0)

    def sample(self, mask, context, context_mask, affect, style, steps, generator=None, last_step_grad=False, *,
               affect_mask=None):
        if steps < 1:
            raise ValueError('Need positive unit refinement steps')
        ids = torch.zeros(mask.shape, dtype=torch.long, device=context.device)
        hidden = mask.clone()
        lengths = mask.sum(1)
        for index in range(steps):
            # Discrete choices are detached; final soft gradients are a surrogate. / 이산 선택은 분리하고 마지막 확률로 근사 미분합니다.
            with torch.set_grad_enabled(torch.is_grad_enabled() and index == steps - 1):
                logits = self.logits(ids, hidden, mask, context, context_mask, affect, style, affect_mask=affect_mask)
                probabilities = logits.float().softmax(-1)
                confidence, proposed = probabilities.max(-1)
                ids = torch.where(hidden, proposed, ids)
                if index < steps - 1:
                    remain = (lengths.float() * math.cos(math.pi / 2 * (index + 1) / steps)).floor().long()
                    remain = torch.minimum(remain, (hidden.sum(1) - 1).clamp_min(0))
                    ranking = confidence.masked_fill(~hidden, float('inf')).argsort(1).argsort(1)
                    hidden = (ranking < remain[:, None]) & hidden
        hard = self.codebook.centers[ids]
        if torch.is_grad_enabled():
            soft = probabilities @ self.codebook.centers
            # Cancel before addition so forward units stay exact. / 먼저 상쇄해 전방 단위값을 정확히 유지합니다.
            hard = hard + (soft - soft.detach())
        return hard.masked_fill(~mask[..., None], 0)


class UnitSpeechSystem(RecoverySpeechSystem):
    def __init__(self, config, centers):
        super().__init__(config)
        self.semantic_planner = MaskedUnitPlanner(config, centers)

    def configure_unit_objective(self, name, total_steps, rehearsal_weight=0.):
        if name not in ('balanced', 'curriculum', 'fully_masked') or total_steps < 1 or not 0 <= rehearsal_weight <= 1:
            raise ValueError('Invalid controlled unit objective')
        self.unit_objective_recipe = {'name': name, 'total_steps': total_steps,
                                      'rehearsal_weight': rehearsal_weight}

    def configure_planner_audio(self, weight=.02, every=4):
        import math
        if (self.recovery_phase != 'planner' or getattr(self, 'unit_prior_training', False)
                or self._waveform_teacher is None or self._acoustic_codec is None):
            raise ValueError('Planner audio requires conditional units and frozen codec/evaluator')
        if not math.isfinite(weight) or weight <= 0 or type(every) is not int or every < 1:
            raise ValueError('Invalid sampled planner loss settings')
        self.length_predictor.requires_grad_(False)
        self.trainable_components = ['semantic_planner']
        self._waveform_teacher.requires_grad_(False).eval()
        self._acoustic_codec.requires_grad_(False).eval()
        self.planner_audio_recipe = {'weight': weight, 'every': every}
        self.train(self.training)

    def controlled_unit_losses(self, batch, prior):
        from .unit_objectives import controlled_unit_loss
        semantic, mask = super().target(batch, 'semantic', 768)
        ids = self.semantic_planner.codebook.encode((semantic - self.semantic_mean) / self.semantic_std)
        size = len(ids)
        null_memory = {'context': semantic.new_zeros(size, 1, 512),
                       'context_mask': torch.ones(size, 1, device=semantic.device, dtype=torch.bool),
                       'affect': semantic.new_zeros(size, 1, 6)}
        null_style = semantic.new_zeros(size, self.config.hidden_dim)
        if prior:
            memory, style = null_memory, null_style
        else:
            encoded = self.encode_batch(batch)
            memory = self.planner_inputs(encoded)
            style, _ = self.embeddings(batch['style_id'], batch['speaker_id'], size)
        recipe = self.unit_objective_recipe
        args = (recipe['name'], getattr(self, 'current_step', 0), recipe['total_steps'])
        values = controlled_unit_loss(self.semantic_planner, ids, mask, memory, style, *args, prior, self.training)
        result = {'unit_ce': values['ce'], 'partial_ce_diagnostic': values['partial_ce'].detach(),
                  'full_ce_diagnostic': values['full_ce'].detach(),
                  'full_sequence_fraction': values['full_sequence_fraction'].detach(),
                  'hidden_token_fraction': values['hidden_token_fraction'].detach()}
        total = values['ce']
        if not prior:
            _, log_duration = self.length_predictor(encoded['context'], encoded['affect'],
                                                   encoded['context_mask'], style)
            result['duration'] = F.smooth_l1_loss(log_duration.float(), batch['duration'].float().log())
            total = total + result['duration']
            if recipe['rehearsal_weight']:
                # Keep B reconstruction during A-conditioned learning. / A 조건 학습 중에도 B 복원 연습을 유지합니다.
                rehearsal_args = ('balanced' if recipe['name'] == 'fully_masked' else recipe['name'],
                                  getattr(self, 'current_step', 0), recipe['total_steps'])
                rehearsal = controlled_unit_loss(self.semantic_planner, ids, mask, null_memory, null_style,
                                                  *rehearsal_args, True, self.training)['ce']
                result['prior_rehearsal_ce'] = rehearsal
                total = total + recipe['rehearsal_weight'] * rehearsal
        return dict(result, total=total)

    def configure_unit_prior(self):
        # Learn B-unit structure before learning A-to-B responses. / A→B 응답 전에 B 단위 구조를 배웁니다.
        self.configure_recovery('planner', teacher=None, teacher_weight=0.)
        self.length_predictor.requires_grad_(False)
        self.trainable_components = ['semantic_planner']
        self.unit_prior_training = True
        self.train(self.training)

    def unit_prior_losses(self, batch):
        semantic, mask = super().target(batch, 'semantic', 768)
        semantic = (semantic - self.semantic_mean) / self.semantic_std
        batch_size = len(semantic)
        # No A input, style, voice or duration labels enter this objective. / 이 목적에는 A·스타일·화자·길이 정답을 쓰지 않습니다.
        context = semantic.new_zeros(batch_size, 1, 512)
        context_mask = torch.ones(batch_size, 1, device=semantic.device, dtype=torch.bool)
        affect = semantic.new_zeros(batch_size, 1, 6)
        style = semantic.new_zeros(batch_size, self.config.hidden_dim)
        unit_ce, _ = self.semantic_planner.estimate(semantic, mask, context, context_mask, affect, style,
            fully_masked_probability=0., max_mask_ratio=.8, denoising_eval=True)
        return {'unit_prior_ce': unit_ce, 'total': unit_ce}

    def target(self, batch, name, dimension):
        values, mask = super().target(batch, name, dimension)
        if name == 'semantic':
            # Keep the existing 768-D acoustic interface. / 기존 768차원 음향 인터페이스를 유지합니다.
            normalized = (values - self.semantic_mean) / self.semantic_std
            values = self.semantic_planner.codebook(normalized) * self.semantic_std + self.semantic_mean
        return values.masked_fill(~mask[..., None], 0), mask

    def unit_base_losses(self, batch):
        if getattr(self, 'unit_objective_recipe', None):
            return self.controlled_unit_losses(batch, getattr(self, 'unit_prior_training', False))
        if getattr(self, 'unit_prior_training', False):
            return self.unit_prior_losses(batch)
        if self.recovery_phase != 'planner':
            return super().losses(batch)
        encoded = self.encode_batch(batch)
        context, mask, affect = encoded['context'], encoded['context_mask'], encoded['affect']
        style, _ = self.embeddings(batch['style_id'], batch['speaker_id'], len(context))
        semantic, semantic_mask = super().target(batch, 'semantic', 768)
        semantic = (semantic - self.semantic_mean) / self.semantic_std
        unit_ce, _ = self.semantic_planner.estimate(semantic, semantic_mask, style=style,
                                                  **self.planner_inputs(encoded))
        _, log_duration = self.length_predictor(context, affect, mask, style)
        duration = F.smooth_l1_loss(log_duration.float(), batch['duration'].float().log())
        # First prove unit prediction without optimizing a trainable ASR proxy. / 먼저 학습형 ASR 대리 지표 없이 단위 예측을 검증합니다.
        return {'unit_ce': unit_ce, 'duration': duration, 'total': unit_ce + duration}

    def losses(self, batch):
        result = self.unit_base_losses(batch)
        recipe = getattr(self, 'planner_audio_recipe', None)
        if recipe is None:
            return result
        from .planner_waveform import sampled_planner_waveform, frozen_content_ctc
        loss = result['total'] * 0
        if not self.training or self.current_step % recipe['every'] == 0:
            # Rotate training items; validation uses a fixed sparse subset. / 학습 발화는 순환하고 검증은 고정된 일부를 사용합니다.
            index = (self.current_step // recipe['every']) % len(batch['mel']) if self.training else 0
            seed = 100000 + self.current_step if self.training else 42
            generated = sampled_planner_waveform(self, batch, self._acoustic_codec, seed, index)
            labels = batch['text_b'][index, :int(batch['text_b_len'][index])]
            loss = frozen_content_ctc(self._waveform_teacher, generated['waveform'], labels)['loss']
        result['sampled_planner_ctc'] = loss
        result['total'] = result['total'] + recipe['weight'] * loss
        return result

    def generate_batch(self, batch, codec, seed=42, oracle_semantic=None, oracle_duration=None):
        if oracle_semantic is not None:
            oracle_semantic = self.semantic_planner.codebook(oracle_semantic)
        return super().generate_batch(batch, codec, seed, oracle_semantic, oracle_duration)

    def checkpoint(self, training_steps=0, **metadata):
        result = super().checkpoint(training_steps, **metadata)
        result['architecture'] = 'llm_free_speech_units_v1'
        return result

    @classmethod
    def from_checkpoint(cls, path):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload.get('architecture') != 'llm_free_speech_units_v1':
            raise ValueError('Expected discrete-unit checkpoint')
        model = cls(QualityConfig(**payload['config']), payload['state_dict']['semantic_planner.codebook.centers'])
        model.load_state_dict(payload['state_dict'], strict=True)
        model.current_epoch = payload.get('current_epoch', 0)
        return model.eval(), payload


def initialize_unit_system(config, payload, codebook_path=None):
    if payload['architecture'] == 'llm_free_speech_units_v1':
        model = UnitSpeechSystem(config, payload['state_dict']['semantic_planner.codebook.centers'])
        model.load_state_dict(payload['state_dict'], strict=True)
        if codebook_path is not None:
            book, _ = load_codebook(codebook_path, model)
            if not torch.equal(book.centers, model.semantic_planner.codebook.centers):
                raise ValueError('Checkpoint and requested unit codebook differ')
        return model
    if payload['architecture'] != 'llm_free_speech_quality_v2' or codebook_path is None:
        raise ValueError('Unit initialization needs an acoustic checkpoint and a codebook')
    original = RecoverySpeechSystem(config)
    original.load_state_dict(payload['state_dict'], strict=True)
    codebook, _ = load_codebook(codebook_path, original)
    model = UnitSpeechSystem(config, codebook.centers)
    state = model.state_dict()
    # Preserve every component except the deliberately replaced planner. / 교체하는 계획기 외의 모든 구성요소를 보존합니다.
    for name, value in payload['state_dict'].items():
        if not name.startswith('semantic_planner.'):
            state[name] = value
    model.load_state_dict(state, strict=True)
    return model
