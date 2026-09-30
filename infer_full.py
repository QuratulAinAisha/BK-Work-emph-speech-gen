"""Run all seven modules and write audio. / 일곱 모듈을 실행하고 음성을 저장합니다."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model.full_speech import EmpatheticSpeechSystem, SpeechConfig
from model.full_speech.codec import FrozenEncodec


def read_person_a(path, device, style, speaker):
    # Read only A's inputs; ignore every B target. / A 입력만 읽고 B 정답은 모두 무시합니다.
    with np.load(path, allow_pickle=False) as archive:
        batch = {}
        for name in ("mel", "dmm", "au"):
            value = torch.from_numpy(archive[name]).float()
            batch[name] = (value[None] if value.ndim == 2 else value).to(device)
        size = batch["mel"].shape[0]
        for name in ("mel_len", "dmm_len", "au_len"):
            if name in archive:
                batch[name] = torch.from_numpy(np.atleast_1d(archive[name])).to(device)
        for name, default in (("style_id", style), ("speaker_id", speaker)):
            value = np.atleast_1d(archive[name]) if name in archive else np.full(size, default, dtype=np.int64)
            batch[name] = torch.from_numpy(value).to(device)
    return batch


def main(argv=None):
    import soundfile as sf

    parser = argparse.ArgumentParser(description="Full LLM-free feature-to-speech inference")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true")
    source.add_argument("--input", type=Path, help="NPZ with Person-A mel/dmm/au")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--allow-untrained", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--style", type=int, default=0)
    parser.add_argument("--speaker", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--semantic-steps", type=int)
    parser.add_argument("--codec-steps", type=int)
    parser.add_argument("--output", type=Path, default=Path("outputs/full_example"))
    args = parser.parse_args(argv)
    if args.checkpoint and args.config:
        parser.error("The checkpoint already contains its config")
    if not args.checkpoint and not (args.demo or args.allow_untrained):
        parser.error("Use --checkpoint, --demo, or explicit --allow-untrained")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this Python environment")
    torch.manual_seed(args.seed)
    payload = None
    if args.checkpoint:
        model, payload = EmpatheticSpeechSystem.from_checkpoint(args.checkpoint)
    else:
        config = SpeechConfig.load(args.config or "configs/full_speech_small.json")
        model = EmpatheticSpeechSystem(config)
    model.to(args.device).eval()
    if args.demo:
        from prepare_full import synthetic_record
        values, _, _ = synthetic_record(0, model.config)
        batch = {name: torch.from_numpy(values[name])[None].to(args.device) for name in ("mel", "dmm", "au")}
        batch.update(style_id=torch.tensor([args.style], device=args.device),
                     speaker_id=torch.tensor([args.speaker], device=args.device))
    else:
        batch = read_person_a(args.input, args.device, args.style, args.speaker)
    codec = FrozenEncodec(model.config.codec_model, model.config.codec_revision).to(args.device)
    result = model.generate(**batch, seed=args.seed, semantic_steps=args.semantic_steps,
                            codec_steps=args.codec_steps, codec=codec)
    if not all(torch.isfinite(value).all() for value in result.values()):
        raise RuntimeError("Non-finite inference output")
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "stages.npz",
                        **{name: value.cpu().numpy() for name, value in result.items()})
    files, peaks = [], []
    for index, length in enumerate(result["audio_lengths"].tolist()):
        audio = result["waveform"][index, :length].cpu().numpy()
        peak = float(np.max(np.abs(audio)))
        # Attenuate clipping without amplifying quiet output. / 작은 소리는 증폭하지 않고 클리핑만 줄입니다.
        audio = audio / max(1.0, peak / 0.95)
        name = f"response_{index:03d}.wav"
        sf.write(args.output / name, audio, model.config.sample_rate, subtype="PCM_16")
        files.append(name)
        peaks.append(peak)
    report = {
        "example_type": "synthetic" if args.demo else "person_a_features",
        "untrained_generator": payload is None or payload.get("training_steps", 0) == 0,
        "synthetic_training_checkpoint": bool(payload and payload.get("metadata", {}).get("synthetic")),
        "quality_note": "A structural example is not evidence of intelligible or empathetic speech.",
        "device": args.device,
        "training_performed": bool(payload and payload.get("training_steps", 0) > 0),
        "generator_training_steps": 0 if payload is None else payload.get("training_steps", 0),
        "training_data_provenance": {} if payload is None else payload.get("metadata", {}).get("data_provenance", {}),
        "pretrained_codec": model.config.codec_model,
        "codec_revision": getattr(codec.model.config, "_commit_hash", None),
        "codec_frozen": all(not p.requires_grad for p in codec.parameters()),
        "sample_rate": model.config.sample_rate,
        "semantic_steps_used": model.config.semantic_steps if args.semantic_steps is None else args.semantic_steps,
        "codec_steps_used": model.config.codec_steps if args.codec_steps is None else args.codec_steps,
        "context_shape": list(result["context"].shape),
        "affect_shape": list(result["affect"].shape),
        "semantic_shape": list(result["semantic"].shape),
        "codec_shape": list(result["codec_latents"].shape),
        "predicted_durations": result["duration"].cpu().tolist(),
        "semantic_lengths": result["semantic_lengths"].cpu().tolist(),
        "codec_lengths": result["codec_lengths"].cpu().tolist(),
        "audio_lengths": result["audio_lengths"].cpu().tolist(),
        "raw_audio_peaks": peaks, "finite_outputs": True, "audio_files": files,
        "config": model.config.to_dict(),
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Example saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
