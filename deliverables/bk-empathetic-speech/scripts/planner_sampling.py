"""Inference-only planner ablations. / 추론 전용 계획기 비교."""

import math
import torch


@torch.inference_mode()
def sample_units(planner, mask, context, context_mask, affect, style, *, steps=8,
                 mode='greedy', seed=42, temperature=.8, top_k=20,
                 initial_ids=None, initial_hidden=None, affect_mask=None):
    if steps < 1 or mode not in ('greedy', 'categorical', 'revisable', 'random_remask'):
        raise ValueError('Invalid diagnostic sampler')
    if mask.dtype != torch.bool or mask.ndim != 2 or not mask.any(1).all():
        raise ValueError('Every sequence needs valid positions')
    if temperature <= 0 or not math.isfinite(temperature) or top_k < 1:
        raise ValueError('Invalid sampling temperature or top-k')
    ids = torch.zeros_like(mask, dtype=torch.long) if initial_ids is None else initial_ids.clone()
    hidden = mask.clone() if initial_hidden is None else initial_hidden.clone()
    if ids.shape != mask.shape or hidden.shape != mask.shape or hidden.dtype != torch.bool:
        raise ValueError('Hint shapes must match the unit mask')
    if (hidden & ~mask).any() or ids.min() < 0 or ids.max() >= len(planner.codebook.centers):
        raise ValueError('Invalid unit hints')
    mutable = hidden.clone()
    lengths = mutable.sum(1)
    rng = torch.Generator(device=context.device).manual_seed(seed)
    trace = []
    for index in range(steps):
        if not hidden.any(): break
        previous, was_hidden = ids.clone(), hidden.clone()
        kwargs = {'affect_mask': affect_mask} if affect_mask is not None else {}
        logits = planner.logits(ids, hidden, mask, context, context_mask, affect, style, **kwargs).float()
        if not torch.isfinite(logits[mask]).all(): raise ValueError('Non-finite unit logits')
        probabilities = logits.softmax(-1)
        confidence, proposed = probabilities.max(-1)
        if mode == 'categorical':
            values, choices = (logits / temperature).topk(min(top_k, logits.shape[-1]), dim=-1)
            picked = torch.multinomial(values.softmax(-1).reshape(-1, values.shape[-1]), 1,
                                       generator=rng).reshape(*mask.shape, 1)
            proposed = choices.gather(-1, picked).squeeze(-1)
            confidence = probabilities.gather(-1, proposed[..., None]).squeeze(-1)
        ids = torch.where(hidden, proposed, ids)
        if index < steps - 1:
            remain = (lengths.float() * math.cos(math.pi / 2 * (index + 1) / steps)).floor().long()
            if mode == 'random_remask':
                # Uniformly reopen predictions, never true hints. / 정답 힌트는 고정하고 예측 위치를 균등하게 다시 가립니다.
                remain = torch.where(lengths > 0, remain.clamp_min(1), 0)
                confidence = torch.rand(mask.shape, device=context.device, generator=rng)
                eligible = mutable
            else:
                remain = torch.minimum(remain, (hidden.sum(1) - 1).clamp_min(0))
                if mode == 'revisable':
                    # Reconsider accepted predictions, but never supplied hints. / 예측은 재검토하되 제공된 힌트는 고정합니다.
                    confidence = probabilities.gather(-1, ids[..., None]).squeeze(-1)
                    eligible = mutable
                else:
                    eligible = hidden
            ranking = confidence.masked_fill(~eligible, float('inf')).argsort(1).argsort(1)
            hidden = (ranking < remain[:, None]) & eligible
        else:
            hidden = torch.zeros_like(hidden)
        adjacent = mask[:, 1:] & mask[:, :-1]
        trace.append({'round': index + 1, 'predicted_positions': int(was_hidden.sum()),
            'still_hidden': int(hidden.sum()),
            'reopened_accepted_positions': int((hidden & ~was_hidden).sum()),
            'changed_positions': int(((ids != previous) & mask).sum()),
            'distinct_units': len(ids[mask].unique()),
            'adjacent_repeat_fraction': float(((ids[:, 1:] == ids[:, :-1]) & adjacent).sum()
                                               / adjacent.sum().clamp_min(1)),
            'mean_max_probability': float(probabilities[mask].max(-1).values.mean())})
    return ids.masked_fill(~mask, 0), trace
