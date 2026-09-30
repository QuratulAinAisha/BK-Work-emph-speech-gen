"""Cached paired speech targets and leakage checks. / 캐시된 음성 쌍과 누출 검사."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from model.affective_response_transport import AFFECT_FEATURES, AFFECT_RANGES


class FullSpeechDataset(Dataset):
    def __init__(self, manifest, config, split, allow_synthetic=False):
        self.path = Path(manifest)
        self.metadata = json.loads(self.path.read_text(encoding="utf-8"))
        self.config = config
        if self.metadata.get("schema_version") != 1:
            raise ValueError("Unsupported prepared-data schema")
        if self.metadata.get("target_contract") != config.target_contract():
            raise ValueError("Prepared targets do not match the model/teacher contract")
        if self.metadata.get("feature_names") != list(AFFECT_FEATURES):
            raise ValueError("Prepared affect feature order does not match")
        if self.metadata.get("synthetic") and not allow_synthetic:
            raise ValueError("Synthetic data requires --allow-synthetic; it cannot establish speech quality")
        seen, paths = {}, set()
        for row in self.metadata["records"]:
            if row["split"] not in ("train", "val", "test"):
                raise ValueError("Every record needs a train/val/test split")
            conversation = str(row["conversation_id"])
            if conversation in seen and seen[conversation] != row["split"]:
                raise ValueError(f"Conversation leakage across splits: {conversation}")
            seen[conversation] = row["split"]
            resolved = (self.path.parent / row["path"]).resolve()
            if resolved in paths:
                raise ValueError("Duplicate prepared sample path")
            paths.add(resolved)
        self.records = [row for row in self.metadata["records"] if row["split"] == split]
        if not self.records:
            raise ValueError(f"No {split} records in manifest")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        dimensions = {
            "mel": self.config.sbe.get("mel_dim", 80), "dmm": self.config.sbe.get("dmm_dim", 486),
            "au": self.config.sbe.get("au_dim", 25), "affect": 6,
            "semantic": self.config.semantic_dim, "codec": self.config.codec_dim,
        }
        result = {}
        with np.load(self.path.parent / row["path"], allow_pickle=False) as data:
            for name, dim in dimensions.items():
                array = data[name]
                if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != dim:
                    raise ValueError(f"{row['path']}: {name} must be nonempty [T,{dim}]")
                if array.dtype.kind != "f" or not np.isfinite(array).all():
                    raise ValueError(f"{name} must be finite floating features")
                result[name] = torch.from_numpy(array.copy()).float()
            for name, limit in (("style_id", self.config.num_styles), ("speaker_id", self.config.num_speakers)):
                value = data[name]
                if value.shape != () or value.dtype.kind not in "iu" or not 0 <= int(value) < limit:
                    raise ValueError(f"{name} must be a scalar integer in [0,{limit})")
                result[name] = torch.tensor(int(value), dtype=torch.long)
            # Unknown labels have zero weight, not invented values. / 미상 라벨은 가중치 0으로 처리합니다.
            weights = data["affect_weight"] if "affect_weight" in data else np.ones_like(data["affect"])
            if (weights.shape != data["affect"].shape or not np.isfinite(weights).all()
                    or ((weights < 0) | (weights > 1)).any()):
                raise ValueError("affect_weight must match affect with finite values in [0,1]")
            result["affect_weight"] = torch.from_numpy(weights.copy()).float()
            duration = float(data["duration"])
            if not np.isfinite(duration) or not self.config.min_duration <= duration <= self.config.max_duration:
                raise ValueError("Response duration is outside the configured bounds")
            result["duration"] = torch.tensor(duration, dtype=torch.float32)
        for name, hz in (("semantic", self.config.semantic_hz), ("codec", self.config.codec_hz)):
            audio_samples = round(duration * self.config.sample_rate)
            if result[name].shape[0] != int(np.ceil(audio_samples * hz / self.config.sample_rate - 1e-9)):
                raise ValueError(f"{name} length disagrees with the response duration")
        for channel, (low, high) in enumerate(AFFECT_RANGES):
            values = result["affect"][:, channel]
            if ((values < low) | (values > high)).any():
                raise ValueError(f"Affect channel {AFFECT_FEATURES[channel]} is out of range")
        if result["dmm"].shape[0] > self.config.sbe.get("app_max_len", 2000):
            raise ValueError("3DMM sequence exceeds the SBE positional-encoding limit")
        return result


def collate_full_speech(samples):
    result = {}
    for name in ("mel", "dmm", "au", "semantic", "codec", "affect"):
        values = [sample[name] for sample in samples]
        result[name] = pad_sequence(values, batch_first=True)
        result[name + "_len"] = torch.tensor([value.shape[0] for value in values], dtype=torch.long)
    for name in ("style_id", "speaker_id", "duration"):
        result[name] = torch.stack([sample[name] for sample in samples])
    result["affect_weight"] = pad_sequence(
        [sample.get("affect_weight", torch.ones_like(sample["affect"])) for sample in samples], batch_first=True)
    return result


def fit_target_statistics(dataset, runtime=None):
    # Fit on train only, never validation/test. / 검증·테스트를 제외한 학습 데이터만 사용합니다.
    dimensions = {"semantic": dataset.config.semantic_dim, "codec": dataset.config.codec_dim}
    sums = {name: [torch.zeros(dim, dtype=torch.float64), torch.zeros(dim, dtype=torch.float64), 0]
            for name, dim in dimensions.items()}
    indices = range(len(dataset)) if runtime is None else range(runtime.rank, len(dataset), runtime.world_size)
    for index in indices:
        sample = dataset[index]
        for name in dimensions:
            values = sample[name].double()
            sums[name][0] += values.sum(0)
            sums[name][1] += values.square().sum(0)
            sums[name][2] += values.shape[0]
    stats = {}
    for name, dim in dimensions.items():
        total, squares, count = sums[name]
        if runtime is not None and runtime.distributed:
            import torch.distributed as dist
            packed = torch.cat((total, squares, torch.tensor([count], dtype=torch.float64))).to(runtime.device)
            dist.all_reduce(packed)
            total, squares, count = packed[:dim].cpu(), packed[dim:2 * dim].cpu(), float(packed[-1])
        if count == 0:
            raise ValueError("No training frames for normalization")
        mean = total / count
        std = (squares / count - mean.square()).clamp_min(0).sqrt().clamp_min(0.05)
        stats[name + "_mean"], stats[name + "_std"] = mean.float(), std.float()
    return stats
