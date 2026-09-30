"""Frozen pretrained neural codec. / 고정된 사전학습 신경 코덱."""

import math
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


class FrozenEncodec(nn.Module):
    def __init__(self, model_id="facebook/encodec_32khz",
                 revision="d0c45384f6c44db055f78200cfdcb9c1c8706727"):
        super().__init__()
        from transformers import EncodecModel
        self.model, loading = EncodecModel.from_pretrained(model_id, revision=revision, output_loading_info=True)
        if loading["missing_keys"] or loading["unexpected_keys"] or loading.get("mismatched_keys"):
            raise ValueError(f"Codec checkpoint did not load exactly: {loading}")
        self.model.requires_grad_(False).eval()
        cfg = self.model.config
        self.sample_rate = cfg.sampling_rate
        self.hop_length = math.prod(cfg.upsampling_ratios)
        self.frame_hz = self.sample_rate / self.hop_length
        self.latent_dim = cfg.hidden_size
        if (self.sample_rate, self.frame_hz, self.latent_dim) != (32000, 50.0, 128):
            raise ValueError("Expected EnCodec 32 kHz / 50 Hz / 128D")
        if cfg.normalize or cfg.chunk_length_s is not None:
            raise ValueError("Chunked or normalized codecs need a different adapter")

    def train(self, mode=True):
        # Remain frozen even when the parent trains. / 상위 모델 학습 중에도 고정합니다.
        super().train(False)
        self.model.eval()
        return self

    @torch.no_grad()
    def encode(self, audio, lengths):
        rows = []
        for index, length in enumerate(lengths.tolist()):
            if length < self.hop_length or length > audio.shape[-1]:
                raise ValueError("Audio length must be at least one codec frame")
            encoded = self.model.encode(audio[index:index + 1, None, :length])
            codes = encoded.audio_codes[0].transpose(0, 1)
            quantized = self.model.quantizer.decode(codes)
            rows.append(quantized[0].T)
        return pad_sequence(rows, batch_first=True), torch.tensor(
            [row.shape[0] for row in rows], device=audio.device, dtype=torch.long)

    @torch.no_grad()
    def decode(self, latents, lengths, sample_lengths=None):
        rows = []
        for index, length in enumerate(lengths.tolist()):
            if length < 1 or length > latents.shape[1]:
                raise ValueError("Invalid codec latent length")
            # Decode each valid prefix to avoid boundary padding leakage. / 패딩 누출 없이 유효 구간만 복원합니다.
            signal = self.model.decoder(latents[index:index + 1, :length].transpose(1, 2))[0, 0]
            wanted = length * self.hop_length if sample_lengths is None else int(sample_lengths[index])
            if wanted < 1 or wanted > signal.numel():
                raise ValueError("Requested sample length exceeds decoded audio")
            signal = signal[:wanted]
            if not torch.isfinite(signal).all():
                raise RuntimeError("Codec generated non-finite audio")
            rows.append(signal)
        return pad_sequence(rows, batch_first=True), torch.tensor(
            [row.numel() for row in rows], device=latents.device, dtype=torch.long)
