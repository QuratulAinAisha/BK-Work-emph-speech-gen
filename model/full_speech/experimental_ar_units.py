"""Isolated causal speech-unit feasibility model. / 분리된 인과 음성 단위 타당성 모델."""

import math
import torch
from torch import nn
import torch.nn.functional as F

from dataset.quality_speech_dataset import person_a_only
from .dit import ConditionalDiT, DiTBlock
from .quality import QualityConfig
from .tensor_ops import align, mask_from_lengths, sinusoidal
from .units import MaskedUnitPlanner, UnitSpeechSystem


AR_ARCHITECTURE = 'bk_experimental_ar_units_v1'


class CausalDiTBlock(DiTBlock):
    def forward(self, x, memory, global_condition, mask, memory_mask):
        parameters = self.modulation(global_condition).chunk(9, dim=-1)
        causal = torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool).triu(1)
        for index, operation in enumerate((self.self_attention, self.cross_attention, self.mlp)):
            shift, scale, gate = parameters[index * 3:index * 3 + 3]
            query = self.norms[index](x) * (1 + scale[:, None]) + shift[:, None]
            if index == 0:
                update = operation(query, query, query, key_padding_mask=~mask,
                                   attn_mask=causal, need_weights=False)[0]
            elif index == 1:
                update = operation(query, memory, memory, key_padding_mask=~memory_mask, need_weights=False)[0]
            else:
                update = operation(query)
            x = (x + gate[:, None].tanh() * update).masked_fill(~mask[..., None], 0)
        return x


class CausalDiT(ConditionalDiT):
    def __init__(self, config, clusters):
        super().__init__(768, 6, config, config.planner_layers)
        self.blocks = nn.ModuleList(CausalDiTBlock(config.hidden_dim, config.num_heads, config.dropout)
                                   for _ in range(config.planner_layers))
        self.output[-1] = nn.Linear(config.hidden_dim, clusters)

    def forward(self, values, local, context, style, mask, context_mask, planned_lengths):
        # Keep total planned length fixed while prefixes grow. / 접두사가 늘어도 전체 계획 길이는 고정합니다.
        time = values.new_ones(len(values))
        timing = self.time_projection(sinusoidal(time * 1000, self.dim).to(values.dtype))
        condition = style + timing + self.length_projection(planned_lengths.float().log()[:, None].to(values.dtype))
        positions = sinusoidal(torch.arange(values.shape[1], device=values.device), self.dim).to(values.dtype)
        hidden = self.input_projection(values) + self.local_projection(local) + positions[None]
        hidden = hidden.masked_fill(~mask[..., None], 0)
        memory = self.context_projection(context.masked_fill(~context_mask[..., None], 0))
        for block in self.blocks:
            hidden = block(hidden, memory, condition, mask, context_mask)
        return self.output(hidden).masked_fill(~mask[..., None], 0)


class CausalUnitPlanner(MaskedUnitPlanner):
    def __init__(self, config, centers):
        super().__init__(config, centers)
        self.denoiser = CausalDiT(config, len(centers))
        self.configure_sampling()
        # The prior mask vector becomes BOS; parameter names stay compatible. / 기존 마스크 벡터를 BOS로 쓰고 이름을 보존합니다.

    def configure_sampling(self, mode='greedy', temperature=.8, top_k=20, seed=42):
        if mode not in ('greedy', 'categorical') or not math.isfinite(temperature) or temperature <= 0 or top_k < 1:
            raise ValueError('Invalid experimental AR sampling settings')
        self.sampling = {'mode': mode, 'temperature': temperature, 'top_k': top_k, 'seed': seed}

    def _validate(self, ids, mask):
        if (ids.ndim != 2 or ids.dtype != torch.long or mask.ndim != 2 or mask.dtype != torch.bool
                or len(ids) != len(mask) or not mask.any(1).all()
                or not torch.equal(mask, mask_from_lengths(mask.sum(1)))):
            raise ValueError('Expected integer IDs and nonempty contiguous valid prefixes')
        if ids.numel() and ((ids < 0).any() or (ids >= len(self.codebook.centers)).any()):
            raise ValueError('Unit ID outside the fixed vocabulary')

    def prefix_logits(self, previous_ids, full_mask, context, context_mask, affect, style, *, affect_mask=None):
        self._validate(previous_ids, full_mask)
        width = previous_ids.shape[1] + 1
        if width > full_mask.shape[1]:
            raise ValueError('Prefix exceeds the planned response length')
        mask = full_mask[:, :width]
        bos = self.mask_embedding[None, None].expand(len(previous_ids), 1, -1)
        values = torch.cat((bos, self.codebook.centers[previous_ids]), dim=1)
        values = values.masked_fill(~mask[..., None], 0)
        source_mask = context_mask if affect_mask is None else affect_mask
        local = align(affect, source_mask, full_mask)[:, :width]
        return self.denoiser(values, local, context, style, mask, context_mask, full_mask.sum(1))

    def teacher_logits(self, target_ids, mask, context, context_mask, affect, style, *, affect_mask=None):
        if target_ids.shape != mask.shape:
            raise ValueError('Teacher targets and valid mask must have the same shape')
        # Predict y[t] from BOS,y[:t]; y[t:] never enters its attention. / BOS와 y[:t]로 y[t]를 예측합니다.
        return self.prefix_logits(target_ids[:, :-1], mask, context, context_mask, affect, style,
                                  affect_mask=affect_mask)

    @torch.inference_mode()
    def sample_ids(self, mask, context, context_mask, affect, style, *, affect_mask=None):
        if self.training:
            raise RuntimeError('Free generation requires eval mode')
        ids = torch.zeros(mask.shape, dtype=torch.long, device=mask.device)
        # Keep planner draws separate from acoustic Gaussian noise. / 계획기 난수와 음향 가우시안 잡음을 분리합니다.
        rng = torch.Generator(device=mask.device).manual_seed(self.sampling['seed'])
        for position in range(mask.shape[1]):
            logits = self.prefix_logits(ids[:, :position], mask, context, context_mask, affect, style,
                                        affect_mask=affect_mask)
            if self.sampling['mode'] == 'greedy':
                proposed = logits[:, -1].argmax(-1)
            else:
                values, candidates = logits[:, -1].float().topk(min(self.sampling['top_k'], logits.shape[-1]), dim=-1)
                choice = torch.multinomial((values / self.sampling['temperature']).softmax(-1), 1, generator=rng)
                proposed = candidates.gather(1, choice).squeeze(1)
            ids[:, position] = proposed.masked_fill(~mask[:, position], 0)
        return ids

    def sample(self, mask, context, context_mask, affect, style, steps, generator=None, last_step_grad=False, *, affect_mask=None):
        if steps < 1 or last_step_grad or self.training:
            raise ValueError('AR sampling is eval-only; sampled-waveform gradients are unsupported')
        ids = self.sample_ids(mask, context, context_mask, affect, style, affect_mask=affect_mask)
        return self.codebook.centers[ids].masked_fill(~mask[..., None], 0)


class ExperimentalARSystem(UnitSpeechSystem):
    def __init__(self, config, centers):
        super().__init__(config, centers)
        self.semantic_planner = CausalUnitPlanner(config, centers)
        self.ar_phase = 'conditional'

    def configure_ar_training(self, phase):
        if phase not in ('prior', 'conditional'):
            raise ValueError('Unknown AR adaptation phase')
        self.ar_phase = phase
        self.requires_grad_(False)
        self.semantic_planner.requires_grad_(True)
        self.trainable_components = ['semantic_planner']
        self.train(self.training)

    def planner_conditions(self, batch, size, phase=None):
        phase = self.ar_phase if phase is None else phase
        if phase == 'prior':
            values = self.semantic_planner.codebook.centers
            return {'context': values.new_zeros(size, 1, 512),
                'context_mask': torch.ones(size, 1, dtype=torch.bool, device=values.device),
                'affect': values.new_zeros(size, 1, 6), 'style': values.new_zeros(size, self.config.hidden_dim)}
        if phase != 'conditional':
            raise ValueError('Unknown AR adaptation phase')
        with torch.no_grad():
            inputs = person_a_only(batch)
            encoded = self.encode_batch(inputs)
            style, _ = self.embeddings(inputs['style_id'], inputs['speaker_id'], size)
        return {**self.planner_inputs(encoded), 'style': style}

    def unit_targets(self, batch):
        if 'unit_ids' in batch:
            ids = batch['unit_ids']
            mask = mask_from_lengths(batch['unit_len'])
            if ids.shape != mask.shape:
                raise ValueError('Cached unit lengths disagree')
            return ids, mask
        semantic = batch['semantic']
        mask = mask_from_lengths(batch['semantic_len'])
        if semantic.shape != (*mask.shape, 768) or not torch.isfinite(semantic[mask]).all():
            raise ValueError('Expected finite B semantic targets with 768 dimensions')
        with torch.no_grad():
            ids = self.semantic_planner.codebook.encode((semantic - self.semantic_mean) / self.semantic_std)
        return ids.masked_fill(~mask, 0), mask

    def teacher_outputs(self, batch, phase=None):
        ids, mask = self.unit_targets(batch)
        logits = self.semantic_planner.teacher_logits(ids, mask, **self.planner_conditions(batch, len(ids), phase))
        losses = F.cross_entropy(logits.float().transpose(1, 2), ids, reduction='none').masked_fill(~mask, 0)
        return {'loss': (losses.sum(1) / mask.sum(1)).mean(), 'ce_sum': losses.sum(),
                'correct': ((logits.argmax(-1) == ids) & mask).sum(), 'frames': mask.sum(), 'logits': logits}

    def losses(self, batch):
        loss = self.teacher_outputs(batch)['loss']
        return {'next_unit_ce': loss, 'total': loss}

    def generate_batch(self, batch, codec, seed=42, oracle_semantic=None, oracle_duration=None):
        allowed = person_a_only(batch)
        if set(batch) != set(allowed):
            raise ValueError('AR inference accepts only Person A inputs and requested style/voice')
        return super().generate_batch(allowed, codec, seed, oracle_semantic, oracle_duration)

    def checkpoint(self, training_steps=0, **metadata):
        payload = super().checkpoint(training_steps, **metadata)
        payload.update(architecture=AR_ARCHITECTURE, experimental_ar=ar_descriptor(self), ar_phase=self.ar_phase)
        return payload


def ar_descriptor(model):
    return {'version': 1, 'representation': 'fixed 50Hz units', 'semantic_dim': 768,
        'clusters': len(model.semantic_planner.codebook.centers), 'hidden_dim': model.config.hidden_dim,
        'layers': model.config.planner_layers, 'heads': model.config.num_heads,
        'BOS': 'source mask_embedding', 'teacher_input': 'BOS plus targets shifted right by one',
        'attention': 'strict causal self-attention; A cross-attention unchanged',
        'time_condition': 1., 'length_condition': 'fixed full planned unit count during every prefix',
        'generation': 'uncached next-unit decisions; greedy default or explicit categorical evaluation; no EOS',
        'limitation': 'Extra-adaptation feasibility experiment, not an equal-compute architecture comparison.'}


def ar_system_from_payload(payload, allow_masked_initialization=False):
    architecture = payload.get('architecture')
    if architecture != AR_ARCHITECTURE and not (allow_masked_initialization and architecture == 'llm_free_speech_units_v1'):
        raise ValueError('Expected an explicit experimental AR checkpoint')
    model = ExperimentalARSystem(QualityConfig(**payload['config']),
                                payload['state_dict']['semantic_planner.codebook.centers'])
    model.load_state_dict(payload['state_dict'], strict=True)
    if architecture == AR_ARCHITECTURE:
        if payload.get('experimental_ar') != ar_descriptor(model):
            raise ValueError('AR descriptor and checkpoint configuration differ')
        model.ar_phase = payload.get('ar_phase', 'conditional')
        if model.ar_phase not in ('prior', 'conditional'):
            raise ValueError('Unknown saved AR adaptation phase')
    model.current_epoch = payload.get('current_epoch', 0)
    return model.eval()


def load_ar_checkpoint(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    return ar_system_from_payload(payload), payload
