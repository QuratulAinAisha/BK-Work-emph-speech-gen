"""Controlled unit-denoising objectives. / 조건을 통제한 음성 단위 복원 목적함수."""

import torch
import torch.nn.functional as F


def draw_unit_masks(mask, maximum=.8, fully_probability=0., span_probability=0.):
    if mask.ndim != 2 or mask.dtype != torch.bool or not mask.any(1).all():
        raise ValueError('Expected a nonempty boolean sequence mask')
    if not .2 <= maximum <= 1 or not 0 <= fully_probability <= 1 or not 0 <= span_probability <= 1:
        raise ValueError('Invalid masking probabilities')
    hidden = torch.zeros_like(mask)
    for index, valid_mask in enumerate(mask):
        valid = valid_mask.nonzero(as_tuple=True)[0]
        length = len(valid)
        ratio = float(.2 + (maximum - .2) * torch.rand((), device=mask.device))
        count = max(1, min(length - 1, round(length * ratio)))
        if float(torch.rand((), device=mask.device)) < fully_probability:
            count = length
        if float(torch.rand((), device=mask.device)) < span_probability:
            start = int(torch.randint(length - count + 1, (), device=mask.device))
            chosen = valid[start:start + count]
        else:
            chosen = valid[torch.randperm(length, device=mask.device)[:count]]
        hidden[index, chosen] = True
    return hidden


def equal_sequence_loss(logits, ids, hidden, mask):
    if logits.shape[:2] != ids.shape or ids.shape != hidden.shape or hidden.shape != mask.shape:
        raise ValueError('Unit objective shapes differ')
    if (hidden & ~mask).any() or not hidden.any(1).all():
        raise ValueError('Each sequence needs valid hidden targets')
    # Each utterance contributes equally despite different hidden counts. / 숨긴 개수와 무관하게 발화별 가중치를 맞춥니다.
    values = F.cross_entropy(logits.float().transpose(1, 2), ids, reduction='none')
    per_sequence = (values * hidden).sum(1) / hidden.sum(1)
    fully = hidden.sum(1) == mask.sum(1)
    zero = per_sequence.sum() * 0
    return {'ce': per_sequence.mean(),
            'partial_ce': per_sequence[~fully].mean() if (~fully).any() else zero,
            'full_ce': per_sequence[fully].mean() if fully.any() else zero,
            'full_sequence_fraction': fully.float().mean(),
            'hidden_token_fraction': hidden.sum().float() / mask.sum()}


def objective_settings(name, step, total_steps, prior):
    if name not in ('balanced', 'curriculum', 'fully_masked') or total_steps < 1:
        raise ValueError('Unknown controlled unit objective')
    if name == 'fully_masked':
        if prior:
            raise ValueError('Fully hidden B-only training has no conversational input')
        return {'maximum': 1., 'fully_probability': 1., 'span_probability': 0.}
    progress = min(1., max(0., step / max(1., total_steps * .5)))
    if name == 'balanced':
        return {'maximum': .8 if prior else 1., 'fully_probability': 0. if prior else .5,
                'span_probability': 0.}
    return {'maximum': .5 + .3 * progress, 'fully_probability': 0. if prior else .25 + .25 * progress,
            'span_probability': .5 * progress}


def controlled_unit_loss(planner, ids, mask, memory, style, name, step, total_steps, prior, training):
    if training:
        hidden = draw_unit_masks(mask, **objective_settings(name, step, total_steps, prior))
    elif prior:
        # Validation stays fixed across curriculum stages. / 교육 단계와 무관하게 검증 마스크를 고정합니다.
        ordinal = mask.long().cumsum(1) - 1
        lengths = mask.sum(1)
        count = (lengths // 2).clamp_min(1)
        hidden = ((((ordinal + 1) * count[:, None] // lengths[:, None]) >
                   (ordinal * count[:, None] // lengths[:, None])) & mask)
    else:
        hidden = mask.clone()
    inputs = ids.masked_fill(hidden | ~mask, 0)
    logits = planner.logits(inputs, hidden, mask, style=style, **memory)
    return equal_sequence_loss(logits, ids, hidden, mask)
