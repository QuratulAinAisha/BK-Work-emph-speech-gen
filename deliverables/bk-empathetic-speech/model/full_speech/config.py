"""Shared architecture contract. / 공통 구조 계약."""

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path


@dataclass
class SpeechConfig:
    sbe: dict = field(default_factory=lambda: {
        "dmm_dim": 486, "mel_frame_hz": 22050 / 256,
        "video_frame_hz": 25.0, "target_frame_hz": 25.0,
    })
    hidden_dim: int = 256
    num_heads: int = 4
    planner_layers: int = 4
    decoder_layers: int = 6
    affect_hidden_dim: int = 128
    affect_layers: int = 2
    dropout: float = 0.1
    semantic_dim: int = 256
    codec_dim: int = 128
    semantic_hz: float = 12.5
    codec_hz: float = 50.0
    sample_rate: int = 32000
    num_styles: int = 6
    num_speakers: int = 32
    min_duration: float = 0.2
    max_duration: float = 20.0
    semantic_steps: int = 24
    codec_steps: int = 32
    codec_model: str = "facebook/encodec_32khz"
    semantic_model: str = "facebook/hubert-base-ls960"
    codec_revision: str = "d0c45384f6c44db055f78200cfdcb9c1c8706727"
    semantic_revision: str = "dba3bb02fda4248b6e082697eee756de8fe8aa8a"
    projection_seed: int = 42

    def __post_init__(self):
        for name in ("hidden_dim", "num_heads", "planner_layers", "decoder_layers",
                     "affect_hidden_dim", "affect_layers", "semantic_dim", "codec_dim",
                     "num_styles", "num_speakers", "semantic_steps", "codec_steps", "sample_rate"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_dim % self.num_heads or self.hidden_dim % 2 or self.affect_hidden_dim % 4:
            raise ValueError("Hidden dimensions must be even and divisible by their head counts")
        if (not 0 <= self.dropout < 1 or not 0 < self.min_duration < self.max_duration
                or not all(math.isfinite(v) for v in (self.min_duration, self.max_duration))):
            raise ValueError("Invalid dropout or duration bounds")
        if not all(math.isfinite(v) and v > 0 for v in (self.semantic_hz, self.codec_hz)):
            raise ValueError("Frame rates must be positive and finite")
        if (self.codec_dim, self.codec_hz, self.sample_rate) != (128, 50.0, 32000):
            raise ValueError("This codec adapter requires 128D, 50 Hz, 32 kHz EnCodec")
        if self.semantic_dim > 768:
            raise ValueError("semantic_dim cannot exceed the HuBERT teacher's 768 dimensions")
        if self.sbe.get("embed_dim", 512) != 512:
            raise ValueError("The existing SBE context dimension is 512")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self):
        return asdict(self)

    def target_contract(self):
        # Cache identity prevents incompatible teachers. / 서로 다른 교사의 캐시 혼용을 막습니다.
        return {name: getattr(self, name) for name in (
            "semantic_dim", "codec_dim", "semantic_hz", "codec_hz", "sample_rate",
            "codec_model", "semantic_model", "projection_seed",
            "codec_revision", "semantic_revision",
        )}
