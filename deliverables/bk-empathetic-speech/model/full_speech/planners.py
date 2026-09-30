"""Modules 4, 5 and 6. / 모듈 4, 5, 6."""

import math
import torch
from torch import nn

from .dit import ConditionalDiT
from .tensor_ops import align, masked_mean, masked_mse


class ResponseLengthPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.network = nn.Sequential(nn.Linear(512 + 6 + config.hidden_dim, config.hidden_dim),
                                     nn.SiLU(), nn.Linear(config.hidden_dim, 1))

    def forward(self, context, affect, mask, style):
        pooled = torch.cat((masked_mean(context, mask), masked_mean(affect, mask), style), dim=-1)
        log_duration = self.network(pooled).squeeze(-1)
        duration = log_duration.clamp(math.log(self.config.min_duration),
                                      math.log(self.config.max_duration)).exp()
        return duration, log_duration


class SemanticResponsePlanner(nn.Module):
    """Continuous-token diffusion, all positions in parallel. / 연속 토큰을 병렬 확산 생성합니다."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.denoiser = ConditionalDiT(config.semantic_dim, 6, config, config.planner_layers)

    @staticmethod
    def schedule(time):
        angle = (time.float() * 0.999 + 0.008) / 1.008 * math.pi / 2
        alpha_bar = (angle.cos().square() / math.cos(0.008 / 1.008 * math.pi / 2) ** 2).clamp(1e-5, 1)
        return alpha_bar.sqrt()[:, None, None], (1 - alpha_bar).sqrt()[:, None, None]

    def loss(self, target, mask, context, context_mask, affect, style):
        time = torch.rand(target.shape[0], device=target.device) * 0.999 + 0.001
        noise = torch.randn_like(target)
        alpha, sigma = self.schedule(time)
        noisy = (alpha * target + sigma * noise).masked_fill(~mask[..., None], 0)
        condition = align(affect, context_mask, mask)
        prediction = self.denoiser(noisy, time, condition, context, style, mask, context_mask)
        return masked_mse(prediction, noise, mask)

    def sample(self, mask, context, context_mask, affect, style, steps, generator=None):
        if steps < 1:
            raise ValueError("semantic steps must be positive")
        x = torch.randn((*mask.shape, self.config.semantic_dim), device=context.device,
                        generator=generator).masked_fill(~mask[..., None], 0)
        condition = align(affect, context_mask, mask)
        times = torch.linspace(1, 0, steps + 1, device=context.device)
        for current, following in zip(times[:-1], times[1:]):
            time = current.expand(x.shape[0])
            alpha, sigma = self.schedule(time)
            epsilon = self.denoiser(x, time, condition, context, style, mask, context_mask)
            clean = ((x - sigma * epsilon) / alpha).clamp(-5, 5)
            next_alpha, next_sigma = self.schedule(following.expand(x.shape[0]))
            x = (next_alpha * clean + next_sigma * epsilon).masked_fill(~mask[..., None], 0)
        return x


class ConditionalCodecDecoder(nn.Module):
    """Rectified flow from Gaussian noise to codec latents. / 가우시안 잡음에서 코덱 잠재값으로 흐릅니다."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.velocity = ConditionalDiT(config.codec_dim, config.semantic_dim + 6,
                                       config, config.decoder_layers)

    def conditions(self, semantics, semantic_mask, affect, context_mask, mask):
        return torch.cat((align(semantics, semantic_mask, mask),
                          align(affect, context_mask, mask)), dim=-1)

    def loss(self, target, mask, semantics, semantic_mask, context, context_mask, affect, style_speaker):
        time = torch.rand(target.shape[0], device=target.device)
        noise = torch.randn_like(target)
        t = time[:, None, None]
        mixed = ((1 - t) * noise + t * target).masked_fill(~mask[..., None], 0)
        local = self.conditions(semantics, semantic_mask, affect, context_mask, mask)
        prediction = self.velocity(mixed, time, local, context, style_speaker, mask, context_mask)
        return masked_mse(prediction, target - noise, mask)

    def sample(self, mask, semantics, semantic_mask, context, context_mask, affect,
               style_speaker, steps, generator=None, checkpoint_grad=False):
        if steps < 1:
            raise ValueError("codec steps must be positive")
        x = torch.randn((*mask.shape, self.config.codec_dim), device=context.device,
                        generator=generator).masked_fill(~mask[..., None], 0)
        local = self.conditions(semantics, semantic_mask, affect, context_mask, mask)
        def velocity(*args):
            # Recompute activations, preserving all sampler gradients. / 모든 샘플러 기울기를 유지하며 활성값을 재계산합니다.
            if checkpoint_grad and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                return checkpoint(self.velocity, *args, use_reentrant=False)
            return self.velocity(*args)
        # Midpoint integration improves the flow estimate. / 중점 적분으로 흐름 추정을 개선합니다.
        for step in range(steps):
            time = torch.full((x.shape[0],), step / steps, device=x.device)
            first = velocity(x, time, local, context, style_speaker, mask, context_mask)
            middle = x + first * (0.5 / steps)
            second = velocity(middle, time + 0.5 / steps, local, context,
                                   style_speaker, mask, context_mask)
            x = (x + second / steps).masked_fill(~mask[..., None], 0)
        return x
