"""Masked sequence operations. / 마스크가 있는 시퀀스 연산."""

import math
import torch
from torch import Tensor

from model.affective_response_transport import align_valid_prefixes, valid_mask


def masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    clean = x.masked_fill(~mask[..., None], 0)
    return clean.sum(1) / mask.sum(1).clamp_min(1)[:, None].to(x.dtype)


def masked_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    # Equal sample weighting despite unequal durations. / 길이가 달라도 샘플 가중치는 같습니다.
    error = (prediction.float() - target.float()).square().masked_fill(~mask[..., None], 0)
    return (error.sum((1, 2)) / (mask.sum(1) * prediction.shape[-1]).clamp_min(1)).mean()


def align(x: Tensor, source_mask: Tensor, target_mask: Tensor) -> Tensor:
    x = x.masked_fill(~source_mask[..., None], 0)
    return align_valid_prefixes(x, source_mask.sum(1), target_mask.sum(1), target_mask.shape[1])


def counts(duration: Tensor, hz: float, sample_rate: int = 32000) -> Tensor:
    # Quantize duration to audio samples before counting frames. / 프레임 계산 전에 길이를 오디오 샘플로 양자화합니다.
    samples = torch.round(duration.double() * sample_rate)
    return torch.ceil(samples * hz / sample_rate - 1e-9).long().clamp_min(1)


def mask_from_lengths(lengths: Tensor) -> Tensor:
    return valid_mask(lengths, int(lengths.max().item()))


def sinusoidal(values: Tensor, dim: int) -> Tensor:
    frequencies = torch.exp(torch.arange(0, dim, 2, device=values.device).float()
                            * (-math.log(10000.0) / dim))
    angles = values.float()[..., None] * frequencies
    return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)
