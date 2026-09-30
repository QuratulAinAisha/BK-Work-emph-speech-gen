"""Complete feature-to-waveform model. / 특징에서 파형까지 연결하는 전체 모델."""

import torch
from torch import nn
import torch.nn.functional as F

from model.affective_pipeline import SBEWithAffectiveTransport
from model.affective_response_transport import (
    AFFECT_FEATURES, AFFECT_RANGES, AffectiveResponseTransport, AffectiveTransportConfig,
    checked_lengths, clean_padding, valid_mask,
)
from model.speaker_behavior_encoder import SpeakerBehaviorEncoder
from .config import SpeechConfig
from .planners import ResponseLengthPredictor, SemanticResponsePlanner, ConditionalCodecDecoder
from .tensor_ops import align, counts, mask_from_lengths, masked_mse


class EmpatheticSpeechSystem(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or SpeechConfig()
        cfg = self.config
        transport = AffectiveResponseTransport(AffectiveTransportConfig(
            hidden_dim=cfg.affect_hidden_dim, num_layers=cfg.affect_layers,
            feedforward_dim=cfg.affect_hidden_dim * 2, dropout=cfg.dropout,
            emotion_dim=cfg.sbe.get("au_dim", 25),
        ))
        self.encoder = SBEWithAffectiveTransport(SpeakerBehaviorEncoder(**cfg.sbe), transport)
        self.style_embedding = nn.Embedding(cfg.num_styles, cfg.hidden_dim)
        self.speaker_embedding = nn.Embedding(cfg.num_speakers, cfg.hidden_dim)
        self.length_predictor = ResponseLengthPredictor(cfg)
        self.semantic_planner = SemanticResponsePlanner(cfg)
        self.codec_generator = ConditionalCodecDecoder(cfg)
        for name, size in (("semantic", cfg.semantic_dim), ("codec", cfg.codec_dim)):
            self.register_buffer(name + "_mean", torch.zeros(size))
            self.register_buffer(name + "_std", torch.ones(size))
        self.freeze_visual = False

    def set_visual_frozen(self, frozen):
        self.freeze_visual = bool(frozen)
        for branch in (self.encoder.sbe.app_encoder, self.encoder.sbe.emo_encoder):
            branch.requires_grad_(not frozen)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_visual:
            self.encoder.sbe.app_encoder.eval()
            self.encoder.sbe.emo_encoder.eval()
        return self

    def embeddings(self, style_id, speaker_id, batch_size):
        for name, ids, limit in (("style_id", style_id, self.config.num_styles),
                                 ("speaker_id", speaker_id, self.config.num_speakers)):
            if ids.shape != (batch_size,) or ids.dtype != torch.long or ((ids < 0) | (ids >= limit)).any():
                raise ValueError(f"{name} must be int64 [B] within [0, {limit})")
        return self.style_embedding(style_id), self.speaker_embedding(speaker_id)

    def encode_batch(self, batch):
        return self.encoder(**{key: batch[key] for key in
            ("mel", "dmm", "au", "mel_len", "dmm_len", "au_len") if key in batch})

    def target(self, batch, name, dimension):
        value = batch[name]
        lengths = checked_lengths(value, batch.get(name + "_len"), name)
        if value.shape[-1] != dimension:
            raise ValueError(f"{name} requires {dimension} features")
        mask = valid_mask(lengths, value.shape[1])
        return clean_padding(value, mask, name), mask

    def losses(self, batch):
        encoded = self.encode_batch(batch)
        context, mask, affect = encoded["context"], encoded["context_mask"], encoded["affect"]
        style, speaker = self.embeddings(batch["style_id"], batch["speaker_id"], context.shape[0])
        duration = batch["duration"].float()
        if duration.shape != (context.shape[0],) or not torch.isfinite(duration).all():
            raise ValueError("duration must be finite [B]")
        if ((duration < self.config.min_duration) | (duration > self.config.max_duration)).any():
            raise ValueError("Target duration is outside configured bounds")
        semantic, semantic_mask = self.target(batch, "semantic", self.config.semantic_dim)
        codec, codec_mask = self.target(batch, "codec", self.config.codec_dim)
        affect_gt, affect_mask = self.target(batch, "affect", 6)
        if not torch.equal(semantic_mask.sum(1), counts(duration, self.config.semantic_hz)):
            raise ValueError("Semantic lengths do not match duration * semantic_hz")
        if not torch.equal(codec_mask.sum(1), counts(duration, self.config.codec_hz)):
            raise ValueError("Codec lengths do not match duration * codec_hz")
        for index, (low, high) in enumerate(AFFECT_RANGES):
            values = affect_gt[..., index][affect_mask]
            if ((values < low) | (values > high)).any():
                raise ValueError(f"affect channel {AFFECT_FEATURES[index]} is outside its normalized range")
        semantic = ((semantic - self.semantic_mean) / self.semantic_std).masked_fill(~semantic_mask[..., None], 0)
        codec = ((codec - self.codec_mean) / self.codec_std).masked_fill(~codec_mask[..., None], 0)
        _, log_duration = self.length_predictor(context, affect, mask, style)
        weights = batch.get("affect_weight", torch.ones_like(affect_gt))
        if weights.shape != affect_gt.shape:
            raise ValueError("affect_weight must match affect shape")
        weights = clean_padding(weights, affect_mask, "affect_weight")
        if ((weights < 0) | (weights > 1)).any():
            raise ValueError("affect_weight must be in [0,1]")
        weighted_targets = align(affect_gt * weights, affect_mask, mask)
        weights = align(weights, affect_mask, mask).masked_fill(~mask[..., None], 0)
        # Missing frames must not enter interpolated labels. / 누락 프레임이 보간 라벨에 섞이지 않게 합니다.
        affect_aligned = weighted_targets / weights.clamp_min(1e-8)
        # Audio-only labels do not supervise missing affect channels. / 음성 라벨로 미상 정서 채널을 감독하지 않습니다.
        affect_loss = (((affect.float() - affect_aligned.float()).square() * weights).sum((1, 2))
                       / weights.sum((1, 2)).clamp_min(1)).mean()
        # GT lengths only allocate training targets. / 정답 길이는 학습 타깃 구성에만 씁니다.
        losses = {
            "affect": affect_loss,
            "duration": F.smooth_l1_loss(log_duration.float(), duration.log()),
            "semantic": self.semantic_planner.loss(semantic, semantic_mask, context, mask, affect, style),
            "codec_flow": self.codec_generator.loss(codec, codec_mask, semantic, semantic_mask,
                                                    context, mask, affect, style + speaker),
        }
        losses["total"] = sum(losses.values())
        return losses

    @torch.inference_mode()
    def generate(self, mel, dmm, au, style_id, speaker_id, mel_len=None, dmm_len=None,
                 au_len=None, seed=42, semantic_steps=None, codec_steps=None, codec=None):
        if self.training:
            raise RuntimeError("Call model.eval() before generation")
        encoded = self.encoder(mel, dmm, au, mel_len, dmm_len, au_len)
        context, mask, affect = encoded["context"], encoded["context_mask"], encoded["affect"]
        style, speaker = self.embeddings(style_id, speaker_id, mel.shape[0])
        # Response length depends only on Person A and style. / 응답 길이는 A와 스타일로만 정합니다.
        duration, _ = self.length_predictor(context, affect, mask, style)
        semantic_lengths = counts(duration, self.config.semantic_hz)
        codec_lengths = counts(duration, self.config.codec_hz)
        semantic_mask, codec_mask = mask_from_lengths(semantic_lengths), mask_from_lengths(codec_lengths)
        generator = torch.Generator(device=mel.device).manual_seed(seed)
        semantics = self.semantic_planner.sample(
            semantic_mask, context, mask, affect, style,
            self.config.semantic_steps if semantic_steps is None else semantic_steps, generator,
        )
        latents = self.codec_generator.sample(
            codec_mask, semantics, semantic_mask, context, mask, affect, style + speaker,
            self.config.codec_steps if codec_steps is None else codec_steps, generator,
        )
        latents = (latents * self.codec_std + self.codec_mean).masked_fill(~codec_mask[..., None], 0)
        result = {
            **encoded, "duration": duration, "semantic": (
                semantics * self.semantic_std + self.semantic_mean).masked_fill(~semantic_mask[..., None], 0),
            "semantic_mask": semantic_mask, "semantic_lengths": semantic_lengths,
            "codec_latents": latents, "codec_mask": codec_mask, "codec_lengths": codec_lengths,
        }
        if codec is not None:
            if (codec.sample_rate, codec.frame_hz, codec.latent_dim) != (
                self.config.sample_rate, self.config.codec_hz, self.config.codec_dim
            ):
                raise ValueError("Codec and model contracts do not match")
            sample_lengths = torch.round(duration * self.config.sample_rate).long().clamp_min(1)
            waveform, audio_lengths = codec.decode(latents.float(), codec_lengths, sample_lengths)
            result.update(waveform=waveform, audio_lengths=audio_lengths)
        return result

    def checkpoint(self, training_steps=0, **metadata):
        return {
            "architecture": "llm_free_speech_v1", "format_version": 1,
            "config": self.config.to_dict(), "feature_names": list(AFFECT_FEATURES),
            "state_dict": self.state_dict(), "training_steps": training_steps,
            "freeze_visual": self.freeze_visual, "metadata": metadata,
        }

    @classmethod
    def from_checkpoint(cls, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("architecture") != "llm_free_speech_v1" or payload.get("format_version") != 1:
            raise ValueError("Not a supported full-speech checkpoint")
        if payload.get("feature_names") != list(AFFECT_FEATURES):
            raise ValueError("Affect feature order mismatch")
        model = cls(SpeechConfig(**payload["config"]))
        model.load_state_dict(payload["state_dict"], strict=True)
        model.set_visual_frozen(payload["freeze_visual"])
        return model.eval(), payload
