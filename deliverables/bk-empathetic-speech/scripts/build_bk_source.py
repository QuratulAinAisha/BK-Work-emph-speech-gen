"""Audit the existing paired BK corpus. / 기존 BK 대화 쌍을 검사합니다."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import soundfile as sf

from dataset.empathy_dataset import build_input_paths, build_output_paths, build_split_manifest, STYLE2IDX
from model.full_speech import SpeechConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/full_speech.json"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-conversations", type=int, help="Explicitly limit a smoke-test corpus")
    parser.add_argument("--defer-audio-validation", action="store_true",
                        help="Validate waveform contents during target preparation, avoiding a second disk read")
    args = parser.parse_args()
    root = args.dataset.resolve()
    source = json.loads((root / "generated_text/train_final_with_reference_images.json").read_text())
    if args.max_conversations is not None:
        source = source[:args.max_conversations]
    split = build_split_manifest(source)
    assignment = {key: name for name, keys in split["splits"].items() for key in keys}
    audio_names = set(path.name for path in (root / "generated_output_audio").iterdir())
    def audit_item(item):
        records, rejected, durations = [], [], []
        prefix = item["conv_id"].replace(":", "_")
        paths = build_input_paths(str(root), prefix, item["input_gender"], item["input_context"]["emotion"],
                                  Path(item["input_context"]["reference_img"]).stem)
        # Normalize the observed reference-image filename alias. / 확인된 참조 이미지 파일명 별칭을 정규화합니다.
        for key in ("dmm", "au"):
            path = Path(paths[key])
            if not path.exists():
                stem = path.stem if key == "au" else path.name
                if stem.endswith("_surprised"):
                    alternate = path.with_name(stem[:-len("surprised")] + "surprise" + (".npy" if key == "au" else ""))
                    if alternate.exists():
                        paths[key] = str(alternate)
        missing = [name for name in ("mel", "dmm", "au") if not Path(paths[name]).exists()]
        for response in item["responses"]:
            output = build_output_paths(str(root), prefix, response["style"], response["output_gender"],
                                        response["predicted_emotion"], Path(response["reference_img"]).stem)
            audio = root / "generated_output_audio" / (Path(output["mel_gt"]).stem + ".wav")
            try:
                if missing:
                    raise ValueError("Missing Person-A features: " + ",".join(missing))
                if audio.name not in audio_names:
                    raise ValueError(f"Missing response audio: {audio.name}")
                duration = None
                if not args.defer_audio_validation:
                    info = sf.info(audio)
                    duration = info.frames / info.samplerate
                    if not 0.2 <= duration <= 120:
                        raise ValueError(f"Unsupported response duration: {duration}")
                gender = response["output_gender"].lower()
                if gender not in ("female", "male"):
                    raise ValueError("Unknown voice group")
            except (ValueError, OSError, RuntimeError) as exc:
                rejected.append({"conversation_id": item["conv_id"], "style": response["style"], "reason": str(exc)})
                continue
            if duration is not None:
                durations.append(duration)
            records.append({"conversation_id": item["conv_id"], "split": assignment[item["conv_id"]],
                            **{key: paths[key] for key in ("mel", "dmm", "au")},
                            "response_audio": str(audio), "response_text": response["output"],
                            "style_id": STYLE2IDX[response["style"]], "speaker_id": ("female", "male").index(gender),
                            "affect_mode": "audio_prosody_only"})
        return records, rejected, durations
    records, rejected, durations = [], [], []
    # Read headers concurrently; preserve metadata order. / 헤더를 병렬로 읽고 메타데이터 순서를 유지합니다.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, (rows, failures, seconds) in enumerate(pool.map(audit_item, source)):
            records.extend(rows)
            rejected.extend(failures)
            durations.extend(seconds)
            if index == 0 or (index + 1) % 100 == 0:
                print(json.dumps({"conversations_audited": index + 1, "accepted": len(records), "rejected": len(rejected)}), flush=True)
    if not records:
        raise ValueError("No usable paired responses")
    config = SpeechConfig.load(args.config)
    config.num_speakers = 2
    config.max_duration = max(config.max_duration, float(math.ceil(max(durations)))) if durations else 120.0
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"source_conversations": len(source), "accepted_responses": len(records), "rejected": rejected,
              "split_counts": dict(Counter(row["split"] for row in records)),
              "min_seconds": min(durations) if durations else None, "max_seconds": max(durations) if durations else None,
              "total_hours": sum(durations) / 3600 if durations else None,
              "audio_validation": "deferred_to_target_preparation" if args.defer_audio_validation else "headers_checked",
              "speaker_groups": {"0": "female", "1": "male"},
              "limitations": ["Voice groups are not verified individual identities.",
                              "Only measured pitch, energy and utterance word rate supervise affect.",
                              "Valence, arousal and dominance have no direct labels.",
                              "The corpus contains generated paired audio/video."]}
    for name, value in (("source.json", {"records": records, "provenance": report}),
                        ("audit.json", report), ("splits.json", split), ("config.json", config.to_dict())):
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rejected"}, indent=2), flush=True)
    if len(rejected) > 0.05 * (len(records) + len(rejected)):
        raise RuntimeError("More than 5% of responses were rejected; inspect audit before preparation")


if __name__ == "__main__":
    main()
