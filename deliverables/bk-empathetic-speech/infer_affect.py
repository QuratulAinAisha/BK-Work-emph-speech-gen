"""Run modules 1-3 or cached-context transport. / 모듈 1-3 또는 저장된 문맥으로 실행합니다."""

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from model.affective_response_transport import (
    AFFECT_FEATURES, AFFECT_RANGES, AffectiveResponseTransport, AffectiveTransportConfig,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Affective response transport (no training)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true", help="Synthetic SBE -> fusion -> affect demo")
    source.add_argument("--input", type=Path, help="NPZ with cached context or Person-A features")
    parser.add_argument("--checkpoint", type=Path, help="Module-3 checkpoint")
    parser.add_argument("--sbe-checkpoint", type=Path, help="Full SBE or baseline Stage-2 checkpoint")
    parser.add_argument("--config", type=Path, help="Transport JSON config when no checkpoint is used")
    parser.add_argument("--sbe-config", type=Path, help="Optional SBE constructor settings as JSON")
    parser.add_argument("--allow-untrained", action="store_true", help="Explicitly allow random weights")
    parser.add_argument("--output", type=Path, default=Path("outputs/affect"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.checkpoint and args.config:
        parser.error("--checkpoint already contains its config; omit --config")
    if not args.demo and not args.checkpoint and not args.allow_untrained:
        parser.error("Provide --checkpoint, or explicitly use --allow-untrained for a wiring check")
    return args


def read_inputs(path: Path, device: str):
    # Never load Python objects from NPZ. / NPZ의 파이썬 객체는 로드하지 않습니다.
    with np.load(path, allow_pickle=False) as archive:
        names = set(archive.files)
        cached = {"context", "speaker_emotion"}.issubset(names)
        features = {"mel", "dmm", "au"}.issubset(names)
        if cached == features:
            raise ValueError("NPZ must contain exactly one input route: context/speaker_emotion OR mel/dmm/au")
        required = ("context", "speaker_emotion") if cached else ("mel", "dmm", "au")
        optional = ("context_mask", "emotion_lengths") if cached else ("mel_len", "dmm_len", "au_len")
        values = {}
        for name in (*required, *optional):
            if name not in archive:
                continue
            value = torch.from_numpy(archive[name])
            if name in required:
                if not value.is_floating_point():
                    raise ValueError(f"{name} must contain floating-point features")
                value = value.float()
            values[name] = value.to(device)
    return values, cached


def load_sbe(checkpoint: Path | None, config: dict, device: str):
    from model.speaker_behavior_encoder import SpeakerBehaviorEncoder

    sbe = SpeakerBehaviorEncoder(**config)
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = payload.get("state_dict", payload)
        # Support the baseline's Stage-2 prefix. / 기존 Stage-2 접두사를 지원합니다.
        state = {name.removeprefix("module."): value for name, value in state.items()}
        if any(name.startswith("sbe.") for name in state):
            state = {name.removeprefix("sbe."): value for name, value in state.items()
                     if name.startswith("sbe.")}
        sbe.load_state_dict(state, strict=True)
    return sbe.to(device).eval()


def demo_inputs(sbe, device: str):
    # Unequal lengths exercise padding and alignment. / 서로 다른 길이로 패딩과 정렬을 확인합니다.
    return {
        "mel": torch.randn(2, 40, sbe.mel_encoder.mel_proj.in_features, device=device),
        "dmm": torch.randn(2, 12, sbe.app_encoder.transformer.embed_layer.in_features, device=device),
        "au": torch.randn(2, 12, sbe.emo_encoder.rnn_encoder.x_rnn.input_dim, device=device),
        "mel_len": torch.tensor([40, 26], device=device),
        "dmm_len": torch.tensor([12, 8], device=device),
        "au_len": torch.tensor([12, 8], device=device),
    }


def write_outputs(directory: Path, result: dict, metadata: dict):
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {name: value.detach().cpu().numpy() for name, value in result.items()}
    np.savez_compressed(directory / "affect.npz", **arrays, feature_names=np.array(AFFECT_FEATURES))
    with (directory / "affect.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("sample", "frame", *AFFECT_FEATURES))
        for sample, length in enumerate(arrays["affect_lengths"]):
            for frame in range(int(length)):
                writer.writerow((sample, frame, *arrays["affect"][sample, frame].tolist()))
    (directory / "summary.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def run(args):
    torch.manual_seed(args.seed)
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    transport = (AffectiveResponseTransport.from_checkpoint(args.checkpoint) if args.checkpoint
                 else AffectiveResponseTransport(AffectiveTransportConfig(**config)))
    transport = transport.to(args.device).eval()
    inputs, cached = (None, False) if args.demo else read_inputs(args.input, args.device)
    untrained = [] if args.checkpoint else ["affective_transport"]
    frame_hz = None

    with torch.inference_mode():
        if cached:
            if args.sbe_checkpoint or args.sbe_config:
                raise ValueError("SBE options cannot be used with cached context")
            output = transport(**inputs)
            result = {
                "context": inputs["context"].masked_fill(~output.mask.unsqueeze(-1), 0.0),
                "context_mask": output.mask,
                "affect": output.trajectory,
                "affect_mask": output.mask,
                "affect_lengths": output.lengths,
                "affect_summary": output.mean_pool(),
            }
        else:
            from model.affective_pipeline import SBEWithAffectiveTransport

            if not args.sbe_checkpoint and not (args.demo or args.allow_untrained):
                raise ValueError("Person-A feature input requires --sbe-checkpoint or --allow-untrained")
            sbe_config = json.loads(args.sbe_config.read_text(encoding="utf-8")) if args.sbe_config else {}
            sbe = load_sbe(args.sbe_checkpoint, sbe_config, args.device)
            if not args.sbe_checkpoint:
                untrained.append("speaker_encoder_and_fusion")
            frame_hz = sbe.target_frame_hz
            if args.demo:
                inputs = demo_inputs(sbe, args.device)
            result = SBEWithAffectiveTransport(sbe, transport).eval()(**inputs)

    if not all(torch.isfinite(value).all() for value in result.values()):
        raise RuntimeError("Inference produced non-finite outputs")
    metadata = {
        "mode": "synthetic_demo" if args.demo else "cached_context" if cached else "person_a_features",
        "untrained_components": untrained,
        "quality_note": ("Structural check only; random weights do not predict meaningful empathy."
                         if untrained else "Weights loaded; this run does not validate affect quality."),
        "training_performed": False,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "sbe_checkpoint": str(args.sbe_checkpoint) if args.sbe_checkpoint else None,
        "context_frame_hz": frame_hz,
        "affect_shape": list(result["affect"].shape),
        "valid_lengths": result["affect_lengths"].cpu().tolist(),
        "feature_names": list(AFFECT_FEATURES),
        "feature_ranges": {name: list(bounds) for name, bounds in zip(AFFECT_FEATURES, AFFECT_RANGES)},
        "units": "normalized controls; pitch/energy/speaking_rate are not Hz/dB/syllables per second",
        "config": asdict(transport.config),
        "parameter_count": sum(p.numel() for p in transport.parameters()),
        "seed": args.seed,
        "torch_version": str(torch.__version__),
    }
    write_outputs(args.output, result, metadata)
    print(json.dumps(metadata, indent=2))
    print(f"Saved: {args.output.resolve()}")
    return metadata


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
        raise SystemExit(f"Affect inference failed: {error}") from error


if __name__ == "__main__":
    main()
