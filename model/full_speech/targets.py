"""Frozen speech-semantic target extraction. / 고정된 음성 의미 타깃 추출."""

import math
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
import torch
from torch import nn
import torch.nn.functional as F


def resample_audio(waveform, source_rate, target_rate):
    if source_rate == target_rate:
        return np.asarray(waveform, dtype=np.float32)
    divisor = math.gcd(int(source_rate), int(target_rate))
    return resample_poly(waveform, target_rate // divisor, source_rate // divisor).astype(np.float32)


class FrozenSemanticTeacher(nn.Module):
    def __init__(self, config):
        super().__init__()
        from transformers import HubertConfig, HubertModel
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError
        from safetensors.torch import load_file
        teacher_config = HubertConfig.from_pretrained(config.semantic_model, revision=config.semantic_revision)
        self.model = HubertModel(teacher_config)
        local = Path(config.semantic_model)
        if local.is_dir():
            path = local / "model.safetensors"
            if not path.is_file():
                path = local / "pytorch_model.bin"
        else:
            try:
                path = Path(hf_hub_download(config.semantic_model, "model.safetensors", revision=config.semantic_revision))
            except EntryNotFoundError:
                path = Path(hf_hub_download(config.semantic_model, "pytorch_model.bin", revision=config.semantic_revision))
        state = (load_file(str(path)) if path.suffix == ".safetensors"
                 else torch.load(path, map_location="cpu", weights_only=True))
        # Map legacy weight norm without random fallbacks. / 임의 초기화 없이 구형 가중치 정규화 키를 변환합니다.
        prefix = "encoder.pos_conv_embed.conv."
        expected = self.model.state_dict()
        for old, new in (("weight_g", "parametrizations.weight.original0"),
                         ("weight_v", "parametrizations.weight.original1")):
            if prefix + old in state and prefix + new in expected:
                state[prefix + new] = state.pop(prefix + old)
        self.model.load_state_dict(state, strict=True)
        self.model.eval().requires_grad_(False)
        self.config = config
        # Fixed orthogonal reduction prevents target collapse. / 고정 직교 축소로 타깃 붕괴를 막습니다.
        generator = torch.Generator().manual_seed(config.projection_seed)
        basis = torch.randn(self.model.config.hidden_size, config.semantic_dim, generator=generator)
        projection = torch.linalg.qr(basis, mode="reduced").Q
        self.register_buffer("projection", projection)

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.inference_mode()
    def encode(self, waveform_16k, duration):
        if waveform_16k.ndim != 1 or waveform_16k.numel() < 400:
            raise ValueError("HuBERT needs at least 400 mono samples at 16 kHz")
        values = waveform_16k.float()
        values = (values - values.mean()) / (values.var(unbiased=False) + 1e-7).sqrt()
        features = self.model(values[None]).last_hidden_state[0] @ self.projection
        audio_samples = round(duration * self.config.sample_rate)
        length = max(1, math.ceil(audio_samples * self.config.semantic_hz / self.config.sample_rate - 1e-9))
        return F.interpolate(features.T[None], size=length, mode="linear",
                             align_corners=False)[0].T
