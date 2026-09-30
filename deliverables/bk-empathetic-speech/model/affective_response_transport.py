"""Module 3: context + A's emotion -> B's affect. / 모듈 3: 문맥과 A 감정 -> B 정서."""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import NamedTuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# Normalized controls, not physical units. / 물리 단위가 아닌 정규화 제어값입니다.
AFFECT_FEATURES = ("valence", "arousal", "pitch", "energy", "speaking_rate", "dominance")
AFFECT_RANGES = ((-1.0, 1.0),) + ((0.0, 1.0),) * 5


@dataclass(frozen=True)
class AffectiveTransportConfig:
    """Lightweight Transformer settings. / 경량 Transformer 설정."""

    context_dim: int = 512
    emotion_dim: int = 25
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    feedforward_dim: int = 256
    dropout: float = 0.1

    def __post_init__(self):
        for name in ("context_dim", "emotion_dim", "hidden_dim", "num_heads",
                     "num_layers", "feedforward_dim"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_dim % self.num_heads or self.hidden_dim % 2:
            raise ValueError("hidden_dim must be even and divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


def valid_mask(lengths: Tensor, width: int) -> Tensor:
    """True means valid, not padding. / True는 유효 프레임입니다."""
    return torch.arange(width, device=lengths.device)[None, :] < lengths[:, None]


def checked_lengths(x: Tensor, lengths: Tensor | None, name: str) -> Tensor:
    """Reject empty or invalid sequences. / 비어 있거나 잘못된 길이를 거부합니다."""
    if x.ndim != 3 or min(x.shape) < 1 or not x.is_floating_point():
        raise ValueError(f"{name} must be a nonempty floating tensor [B, T, D]")
    if lengths is None:
        return torch.full((x.shape[0],), x.shape[1], device=x.device, dtype=torch.long)
    if lengths.shape != (x.shape[0],) or lengths.dtype not in (
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
    ):
        raise ValueError(f"{name} lengths must be an integer tensor [B]")
    lengths = lengths.to(device=x.device, dtype=torch.long)
    if ((lengths < 1) | (lengths > x.shape[1])).any():
        raise ValueError(f"{name} lengths must be within [1, {x.shape[1]}]")
    return lengths


def clean_padding(x: Tensor, mask: Tensor, name: str) -> Tensor:
    """Ignore padding, including NaN padding. / NaN을 포함한 패딩을 제외합니다."""
    clean = x.masked_fill(~mask.unsqueeze(-1), 0.0)
    if not torch.isfinite(clean).all():
        raise ValueError(f"{name} contains non-finite values in valid frames")
    return clean


def align_valid_prefixes(x: Tensor, source_lengths: Tensor,
                         target_lengths: Tensor, width: int) -> Tensor:
    """Match SBE's valid-prefix interpolation. / SBE의 유효 구간 보간을 따릅니다."""
    rows = []
    for row, source_len, target_len in zip(x, source_lengths.tolist(), target_lengths.tolist()):
        resized = F.interpolate(
            row[:source_len].T.unsqueeze(0), size=target_len,
            mode="linear", align_corners=False,
        ).squeeze(0).T
        rows.append(F.pad(resized, (0, 0, 0, width - target_len)))
    return torch.stack(rows)


class AffectiveTransportOutput(NamedTuple):
    """Affect [B,T,6], mask [B,T]. / 정서 궤적과 유효 마스크."""

    trajectory: Tensor
    mask: Tensor

    @property
    def lengths(self) -> Tensor:
        return self.mask.sum(dim=1)

    def mean_pool(self) -> Tensor:
        """Summary for future length planning. / 향후 길이 계획용 요약."""
        values = self.trajectory.masked_fill(~self.mask.unsqueeze(-1), 0.0)
        return values.sum(dim=1) / self.lengths.unsqueeze(-1).to(values.dtype)


class AffectiveResponseTransport(nn.Module):
    """Bidirectional temporal transport. / 양방향 시간축 정서 전달.

    context: [B,T,512]; speaker_emotion: [B,Te,25]. / 문맥과 화자 감정 입력.
    emotion_lengths defaults to full Te. / 감정 길이 생략 시 Te 전체를 사용합니다.
    New weights are untrained. / 새 가중치는 미학습 상태입니다.
    """

    def __init__(self, config: AffectiveTransportConfig | None = None):
        super().__init__()
        self.config = config or AffectiveTransportConfig()
        cfg = self.config
        self.context_projection = nn.Sequential(
            nn.LayerNorm(cfg.context_dim), nn.Linear(cfg.context_dim, cfg.hidden_dim)
        )
        self.emotion_projection = nn.Sequential(
            nn.LayerNorm(cfg.emotion_dim), nn.Linear(cfg.emotion_dim, cfg.hidden_dim)
        )
        self.input_norm = nn.LayerNorm(cfg.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim, nhead=cfg.num_heads,
            dim_feedforward=cfg.feedforward_dim, dropout=cfg.dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            layer, num_layers=cfg.num_layers, norm=nn.LayerNorm(cfg.hidden_dim),
            enable_nested_tensor=False,
        )
        # Give cloned layers distinct weights. / 복제 층의 초기 가중치를 다르게 합니다.
        for parameter in self.temporal_transformer.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)
        self.affect_head = nn.Linear(cfg.hidden_dim, len(AFFECT_FEATURES))

    def _positions(self, length: int, reference: Tensor) -> Tensor:
        # Dynamic positions avoid a fixed clip limit. / 동적 위치값으로 길이 제한을 피합니다.
        position = torch.arange(length, device=reference.device, dtype=torch.float32)[:, None]
        scales = torch.exp(torch.arange(
            0, self.config.hidden_dim, 2, device=reference.device, dtype=torch.float32
        ) * (-math.log(10000.0) / self.config.hidden_dim))
        angles = position * scales
        return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1).to(reference.dtype)

    def forward(self, context: Tensor, speaker_emotion: Tensor,
                context_mask: Tensor | None = None,
                emotion_lengths: Tensor | None = None) -> AffectiveTransportOutput:
        context_lengths = checked_lengths(context, None, "context")
        emotion_lengths = checked_lengths(speaker_emotion, emotion_lengths, "speaker_emotion")
        if context.shape[-1] != self.config.context_dim:
            raise ValueError(f"context last dimension must be {self.config.context_dim}")
        if speaker_emotion.shape[-1] != self.config.emotion_dim:
            raise ValueError(f"speaker_emotion last dimension must be {self.config.emotion_dim}")
        if context.shape[0] != speaker_emotion.shape[0]:
            raise ValueError("context and speaker_emotion batch sizes must match")
        if context.device != speaker_emotion.device or context.dtype != speaker_emotion.dtype:
            raise ValueError("context and speaker_emotion must have the same device and dtype")
        if context_mask is None:
            context_mask = valid_mask(context_lengths, context.shape[1])
        else:
            if context_mask.shape != context.shape[:2] or context_mask.dtype != torch.bool:
                raise ValueError("context_mask must be a boolean tensor [B, T]")
            context_mask = context_mask.to(context.device)
            context_lengths = context_mask.sum(dim=1)
            if (context_lengths == 0).any() or not torch.equal(
                context_mask, valid_mask(context_lengths, context.shape[1])
            ):
                raise ValueError("context_mask must contain a nonempty valid prefix per sample")

        context = clean_padding(context, context_mask, "context")
        emotion = clean_padding(speaker_emotion, valid_mask(
            emotion_lengths, speaker_emotion.shape[1]), "speaker_emotion")
        # Align A's emotion to c; never use B's targets. / A 감정을 c에 정렬하며 B 정답은 쓰지 않습니다.
        emotion = align_valid_prefixes(emotion, emotion_lengths, context_lengths, context.shape[1])
        hidden = self.input_norm(self.context_projection(context) + self.emotion_projection(emotion))
        hidden = hidden + self._positions(context.shape[1], hidden).unsqueeze(0)
        hidden = hidden.masked_fill(~context_mask.unsqueeze(-1), 0.0)
        hidden = self.temporal_transformer(hidden, src_key_padding_mask=~context_mask)
        logits = self.affect_head(hidden)
        # Bound valid controls; set padding to zero. / 유효값 범위를 제한하고 패딩은 0으로 만듭니다.
        trajectory = torch.cat((logits[..., :1].tanh(), logits[..., 1:].sigmoid()), dim=-1)
        trajectory = trajectory.masked_fill(~context_mask.unsqueeze(-1), 0.0)
        return AffectiveTransportOutput(trajectory, context_mask)

    def save_checkpoint(self, path: str | Path) -> None:
        """Save weights and their contract. / 가중치와 입출력 계약을 저장합니다."""
        torch.save({
            "format_version": 1,
            "config": asdict(self.config),
            "feature_names": list(AFFECT_FEATURES),
            "state_dict": self.state_dict(),
        }, path)

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "AffectiveResponseTransport":
        """Load exactly matching weights on CPU. / 정확히 일치하는 가중치를 CPU에 로드합니다."""
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("format_version") != 1:
            raise ValueError("Expected an affect transport checkpoint with format_version=1")
        if payload.get("feature_names") != list(AFFECT_FEATURES):
            raise ValueError("Checkpoint affect feature order does not match")
        model = cls(AffectiveTransportConfig(**payload["config"]))
        model.load_state_dict(payload["state_dict"], strict=True)
        return model.eval()
