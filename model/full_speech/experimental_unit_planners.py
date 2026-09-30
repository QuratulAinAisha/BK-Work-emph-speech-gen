"""Explicit tiny-diagnostic planner variants. / 명시적인 소규모 진단용 계획기 변형."""

import torch
from torch import nn

from .quality import QualityConfig
from .tensor_ops import align, sinusoidal
from .units import MaskedUnitPlanner, SpeechCodebook, UnitSpeechSystem


EXPERIMENTAL_ARCHITECTURE = 'bk_experimental_unit_planner_tiny_v1'
VARIANTS = ('existing', 'plain_transformer', 'learned_ids')


class LearnedIdUnitPlanner(MaskedUnitPlanner):
    """Same DiT with trainable visible-unit embeddings. / 같은 DiT에 학습 가능한 단위 임베딩을 씁니다."""

    def __init__(self, config, centers):
        super().__init__(config, centers)
        self.unit_embedding = nn.Embedding(len(centers), centers.shape[1])
        with torch.no_grad():
            self.unit_embedding.weight.copy_(centers)

    def logits(self, ids, hidden, mask, context, context_mask, affect, style, *, affect_mask=None):
        values = self.unit_embedding(ids)
        values = torch.where(hidden[..., None], self.mask_embedding.to(values.dtype), values)
        fraction = hidden.sum(1).float() / mask.sum(1).clamp_min(1)
        affect_mask = context_mask if affect_mask is None else affect_mask
        return self.denoiser(values, fraction, align(affect, affect_mask, mask),
                             context, style, mask, context_mask)


class PlainTransformerUnitPlanner(MaskedUnitPlanner):
    """B-only bidirectional encoder; no A conditioning. / A 조건 없는 B 전용 양방향 인코더."""

    def __init__(self, config, centers):
        nn.Module.__init__(self)
        self.config = config
        self.codebook = SpeechCodebook(centers)
        self.mask_embedding = nn.Parameter(torch.zeros(768))
        self.input_projection = nn.Linear(768, config.hidden_dim)
        layer = nn.TransformerEncoderLayer(config.hidden_dim, config.num_heads,
            dim_feedforward=4 * config.hidden_dim, dropout=config.dropout, activation='gelu',
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, config.planner_layers, enable_nested_tensor=False)
        self.output = nn.Sequential(nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, len(centers)))

    def logits(self, ids, hidden, mask, context, context_mask, affect, style, *, affect_mask=None):
        # Reject accidental use as a conditional response model. / 조건부 응답 모델로 잘못 쓰는 것을 막습니다.
        if any(bool(torch.any(value != 0)) for value in (context, affect, style)):
            raise ValueError('Plain transformer is a B-only diagnostic and requires null conditioning')
        values = self.codebook.centers[ids]
        values = torch.where(hidden[..., None], self.mask_embedding.to(values.dtype), values)
        positions = sinusoidal(torch.arange(ids.shape[1], device=ids.device), self.config.hidden_dim)
        projected = self.input_projection(values) + positions[None].to(values.dtype)
        projected = projected.masked_fill(~mask[..., None], 0)
        output = self.encoder(projected, src_key_padding_mask=~mask)
        return self.output(output).masked_fill(~mask[..., None], 0)


def replace_tiny_planner(model, variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown tiny planner variant')
    if variant == 'existing':
        return model
    original = model.semantic_planner
    if variant == 'learned_ids':
        replacement = LearnedIdUnitPlanner(model.config, original.codebook.centers)
        missing, unexpected = replacement.load_state_dict(original.state_dict(), strict=False)
        if missing != ['unit_embedding.weight'] or unexpected:
            raise ValueError('Learned-ID initialization must preserve every source DiT tensor')
    else:
        replacement = PlainTransformerUnitPlanner(model.config, original.codebook.centers)
    model.semantic_planner = replacement
    return model


def experimental_descriptor(model, variant):
    if variant not in VARIANTS or variant == 'existing':
        raise ValueError('Descriptor requires an experimental variant')
    return {'version': 1, 'variant': variant, 'scope': 'B-only tiny denoising diagnostic',
            'hidden_dim': model.config.hidden_dim, 'layers': model.config.planner_layers,
            'heads': model.config.num_heads, 'input_unit_dim': 768,
            'clusters': len(model.semantic_planner.codebook.centers),
            'initialization': ('Source DiT copied exactly; trainable ID embeddings initialized from centers.'
                if variant == 'learned_ids' else 'Fresh plain encoder, input projection and classifier; fixed source centers.'),
            'limitation': 'Architectures differ in conditioning and parameter count; this is not a production response checkpoint.'}


def experimental_system_from_payload(payload):
    if payload.get('architecture') != EXPERIMENTAL_ARCHITECTURE:
        raise ValueError('Expected an experimental tiny-planner checkpoint')
    descriptor = payload.get('experimental_planner', {})
    variant = descriptor.get('variant')
    if descriptor.get('version') != 1 or variant not in VARIANTS[1:]:
        raise ValueError('Unknown experimental planner descriptor')
    model = UnitSpeechSystem(QualityConfig(**payload['config']),
                             payload['state_dict']['semantic_planner.codebook.centers'])
    replace_tiny_planner(model, variant)
    model.load_state_dict(payload['state_dict'], strict=True)
    if descriptor != experimental_descriptor(model, variant):
        raise ValueError('Experimental descriptor disagrees with model configuration')
    model.current_epoch = payload.get('current_epoch', 0)
    return model.eval()


def load_experimental_unit_checkpoint(path):
    """Load a research artifact explicitly; production loaders stay unchanged. / 연구 산출물을 명시적으로 읽습니다."""
    payload = torch.load(path, map_location='cpu', weights_only=True)
    return experimental_system_from_payload(payload), payload
