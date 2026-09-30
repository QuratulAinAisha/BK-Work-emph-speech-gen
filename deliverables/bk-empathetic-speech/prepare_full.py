"""Prepare full-model targets on CPU or explicit CUDA. / CPU 또는 지정한 CUDA로 타깃을 준비합니다."""

import argparse
import hashlib
from functools import lru_cache
import json
import math
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch

from model.affective_response_transport import AFFECT_FEATURES
from model.full_speech.config import SpeechConfig
from model.full_speech.codec import FrozenEncodec
from model.full_speech.targets import FrozenSemanticTeacher, resample_audio
from utils.distributed_training import DistributedRuntime


@lru_cache(maxsize=48)
def load_feature(path):
    path = Path(path)
    if path.is_dir():
        from dataset.empathy_dataset import load_dmm_frames
        value = load_dmm_frames(str(path))
        if value is None:
            raise ValueError(f"No 3DMM features in {path}")
        return value.numpy()
    if path.suffix in (".pt", ".pth"):
        return torch.load(path, map_location="cpu", weights_only=True).float().numpy()
    return np.load(path, allow_pickle=False).astype(np.float32)


def synthetic_record(index, config):
    rng = np.random.default_rng(900 + index)
    duration = (0.32, 0.48, 0.64, 0.40, 0.56)[index % 5]
    time = np.arange(round(duration * config.sample_rate)) / config.sample_rate
    waveform = (0.12 * np.sin(2 * np.pi * (150 + index * 35) * time)
                * np.sin(np.pi * time / duration) ** 2).astype(np.float32)
    count = max(2, math.ceil(duration * 25))
    affect = np.full((count, 6), 0.5, dtype=np.float32)
    affect[:, 0] = np.linspace(-0.2, 0.3, count)
    return {
        "mel": rng.normal(size=(40 + index * 4, config.sbe.get("mel_dim", 80))).astype(np.float32),
        "dmm": rng.normal(size=(12 + index, config.sbe.get("dmm_dim", 486))).astype(np.float32),
        "au": rng.normal(size=(12 + index, config.sbe.get("au_dim", 25))).astype(np.float32),
        "affect": affect, "style_id": index % config.num_styles,
        "speaker_id": index % min(config.num_speakers, 2),
    }, waveform, config.sample_rate


def main(argv=None):
    parser = argparse.ArgumentParser(description="Cache codec, semantic and affect targets")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", type=Path, help="Paired-record JSON, with relative paths resolved beside it")
    group.add_argument("--synthetic", action="store_true", help="Five clearly labeled tone/feature examples")
    parser.add_argument("--config", type=Path, default=Path("configs/full_speech.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/prepared_full"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true", help="Reuse completed samples from the identical source/config")
    args = parser.parse_args(argv)
    runtime = DistributedRuntime.initialize(args.device)
    try:
        prepare(args, runtime)
    finally:
        runtime.close()


def prepare(args, runtime):
    args.device = runtime.device
    config = SpeechConfig.load(args.config)
    source = {}
    if args.synthetic:
        records = [{"conversation_id": f"synthetic_{i}", "split": "train" if i < 3 else "val" if i == 3 else "test"}
                   for i in range(5)]
    else:
        source = json.loads(args.source.read_text(encoding="utf-8"))
        records = source["records"]
    # Validate split identity before expensive extraction. / 비싼 추출 전에 분할 식별자를 검사합니다.
    seen = {}
    for row in records:
        if row["split"] not in ("train", "val", "test"):
            raise ValueError("Unknown split")
        key = str(row["conversation_id"])
        if key in seen and seen[key] != row["split"]:
            raise ValueError(f"Conversation leakage: {key}")
        seen[key] = row["split"]
    args.output.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(json.dumps({"source": source, "config": config.to_dict(), "synthetic": args.synthetic},
                                         sort_keys=True).encode()).hexdigest()
    identity_path = args.output / "preparation_identity.json"
    if runtime.primary:
        if identity_path.exists():
            if not args.resume or json.loads(identity_path.read_text())["sha256"] != identity:
                raise ValueError("Existing cache needs --resume and an identical source/config")
        else:
            identity_path.write_text(json.dumps({"sha256": identity}) + "\n")
    if runtime.distributed:
        torch.distributed.barrier()
    codec = FrozenEncodec(config.codec_model, config.codec_revision).to(args.device)
    teacher = FrozenSemanticTeacher(config).to(args.device)
    prepared = [{"conversation_id": str(row["conversation_id"]), "split": row["split"],
                 "path": f"sample_{index:06d}.npz"} for index, row in enumerate(records)]
    started = time.monotonic()
    # Keep a conversation on one GPU to reuse its input features. / 입력 특징 재사용을 위해 대화별로 GPU를 배정합니다.
    owners = {key: index % runtime.world_size for index, key in
              enumerate(dict.fromkeys(str(row["conversation_id"]) for row in records))}
    indices = [index for index, row in enumerate(records) if owners[str(row["conversation_id"])] == runtime.rank]
    for local_index, index in enumerate(indices):
        row = records[index]
        filename = prepared[index]["path"]
        if args.resume and (args.output / filename).is_file():
            continue
        if args.synthetic:
            values, waveform, source_rate = synthetic_record(index, config)
        else:
            values = {key: load_feature(args.source.parent / row[key])
                      for key in ("mel", "dmm", "au")}
            values.update(style_id=int(row["style_id"]), speaker_id=int(row["speaker_id"]))
            waveform, source_rate = sf.read(args.source.parent / row["response_audio"], dtype="float32", always_2d=True)
            waveform = waveform.mean(axis=1)
            if row.get("affect_mode") == "audio_prosody_only":
                from dataset.audio_affect_targets import audio_affect_targets
                values["affect"], values["affect_weight"] = audio_affect_targets(waveform, source_rate, row["response_text"])
            else:
                values["affect"] = load_feature(args.source.parent / row["affect"])
                if "affect_weight" in row:
                    values["affect_weight"] = load_feature(args.source.parent / row["affect_weight"])
        if not np.isfinite(waveform).all():
            raise ValueError("Response waveform contains non-finite samples")
        signal = resample_audio(waveform, source_rate, config.sample_rate)
        duration = len(signal) / config.sample_rate
        if not config.min_duration <= duration <= config.max_duration:
            raise ValueError(f"Record {index} response exceeds configured duration bounds")
        audio = torch.from_numpy(signal).to(args.device)
        with torch.inference_mode():
            codec_values, codec_lengths = codec.encode(audio[None], torch.tensor([len(signal)], device=args.device))
            signal_16k = torch.from_numpy(resample_audio(signal, config.sample_rate, 16000)).to(args.device)
            semantic = teacher.encode(signal_16k, duration)
        values.update(codec=codec_values[0, :int(codec_lengths[0])].cpu().numpy(),
                      semantic=semantic.cpu().numpy(), duration=np.float32(duration))
        # Rename only fully written samples. / 완전히 저장된 샘플만 이름을 바꿉니다.
        temporary = args.output / (filename + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **values)
        temporary.replace(args.output / filename)
        if local_index == 0 or (local_index + 1) % 25 == 0:
            progress = {"rank": runtime.rank, "completed": local_index + 1, "assigned": len(indices),
                        "seconds": time.monotonic() - started, "sample": filename}
            (args.output / f"progress_rank{runtime.rank}.json").write_text(json.dumps(progress, indent=2) + "\n")
            print(json.dumps(progress), flush=True)
    if runtime.distributed:
        torch.distributed.barrier()
    manifest = {
        "schema_version": 1, "target_contract": config.target_contract(),
        "feature_names": list(AFFECT_FEATURES), "synthetic": args.synthetic,
        "teacher_revision": getattr(teacher.model.config, "_commit_hash", None),
        "codec_revision": getattr(codec.model.config, "_commit_hash", None),
        "records": prepared,
        "provenance": source.get("provenance", {}),
    }
    if runtime.primary:
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if runtime.distributed:
        torch.distributed.barrier()
    # Exercise the same validation used by training. / 학습과 같은 데이터 검증을 수행합니다.
    from dataset.full_speech_dataset import FullSpeechDataset
    for split in sorted({row["split"] for row in prepared}):
        dataset = FullSpeechDataset(args.output / "manifest.json", config, split, allow_synthetic=args.synthetic)
        for index in range(runtime.rank, len(dataset), runtime.world_size):
            dataset[index]
    if runtime.distributed:
        torch.distributed.barrier()
    if runtime.primary:
        (args.output / "preparation_complete.json").write_text(json.dumps(
            {"sha256": identity, "records": len(prepared), "validated": True}, indent=2) + "\n")
    print(f"Manifest: {(args.output / 'manifest.json').resolve()}")


if __name__ == "__main__":
    main()
