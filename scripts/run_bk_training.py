"""Prepare, train requested epochs, then generate an example. / 준비·요청 에포크 학습·예제 생성."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("outputs/bk_source/source.json"))
    parser.add_argument("--config", type=Path, default=Path("outputs/bk_source/config.json"))
    parser.add_argument("--prepared", type=Path, default=Path("outputs/bk_prepared"))
    parser.add_argument("--output", type=Path, default=Path("outputs/bk_5epochs"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--training-gpus", default="0,1,2,3")
    parser.add_argument("--wait-preparation-pid", type=int)
    parser.add_argument("--use-completed-preparation", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    args.output.mkdir(parents=True, exist_ok=True)
    training_gpus = [int(value) for value in args.training_gpus.split(",")]
    if not training_gpus or len(set(training_gpus)) != len(training_gpus):
        raise ValueError("Training GPU IDs must be unique")
    # This process owns its descendants; SSH disconnection does not stop them. / SSH 종료 후에도 하위 작업을 유지합니다.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    status = {"pid": os.getpid(), "epochs_requested": args.epochs, "gpu_ids": [0, 1],
              "training_gpu_ids": training_gpus, "batch_size_per_gpu": args.batch_size,
              "global_batch_size": args.batch_size * len(training_gpus)}
    def save_status(stage, **extra):
        status.update(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
        temporary = args.output / "run_status.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(args.output / "run_status.json")
        print(json.dumps(status), flush=True)
    def run(stage, command):
        save_status(stage, command=command)
        with (args.output / (stage + ".log")).open("a") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            save_status(stage, child_pid=process.pid)
            code = process.wait()
            if code:
                raise RuntimeError(f"{stage} failed with exit code {code}; see {stage}.log")
    launch = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2"]
    def validate_completed_preparation():
        # Reuse only the validated matching cache. / 검증된 동일 캐시만 재사용합니다.
        complete = json.loads((args.prepared / "preparation_complete.json").read_text())
        identity = json.loads((args.prepared / "preparation_identity.json").read_text())
        source_count = len(json.loads(args.source.read_text())["records"])
        if (not complete.get("validated") or complete["sha256"] != identity["sha256"]
                or complete["records"] != source_count):
            raise RuntimeError("Existing preparation did not complete validation")
    try:
        if args.use_completed_preparation:
            validate_completed_preparation()
        elif args.wait_preparation_pid:
            # Adopt the existing preparation without restarting it. / 기존 준비 작업을 재시작 없이 이어받습니다.
            save_status("preparation", child_pid=args.wait_preparation_pid)
            process_stat = Path(f"/proc/{args.wait_preparation_pid}/stat")
            while process_stat.exists():
                try:
                    if process_stat.read_text().rsplit(")", 1)[1].split()[0] == "Z":
                        break
                except FileNotFoundError:
                    break
                time.sleep(10)
            validate_completed_preparation()
        else:
            run("preparation", launch + ["prepare_full.py", "--source", str(args.source), "--config", str(args.config),
                                        "--output", str(args.prepared), "--device", "cuda", "--resume"])
        # Expand GPU use only after preparation finishes. / 준비 완료 후에만 GPU 사용을 확대합니다.
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, training_gpus))
        status["gpu_ids"] = training_gpus
        launch[-1] = f"--nproc_per_node={len(training_gpus)}"
        checkpoint = args.output / "last.pt"
        initialization = (["--resume", str(checkpoint)] if checkpoint.is_file() else
                          ["--config", str(args.config), "--train-sbe-from-scratch"])
        completed = 0
        if checkpoint.is_file():
            import torch
            completed = torch.load(checkpoint, map_location="cpu", weights_only=True)["epoch"] + 1
        if completed < args.epochs:
            run("training", launch + ["train_full.py", "--manifest", str(args.prepared / "manifest.json"),
                                      *initialization, "--device", "cuda", "--amp", "--amp-dtype", "bfloat16",
                                      "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
                                      "--workers", str(args.workers), "--output", str(args.output)])
        manifest = json.loads((args.prepared / "manifest.json").read_text())
        example = next(row for row in manifest["records"] if row["split"] == "test")
        source_rows = json.loads(args.source.read_text())["records"]
        index = manifest["records"].index(example)
        paired = source_rows[index]
        example_dir = args.output / "heldout_example"
        run("inference", [sys.executable, "infer_full.py", "--input", str(args.prepared / example["path"]),
                          "--checkpoint", str(args.output / "best.pt"), "--device", "cuda:0", "--output", str(example_dir)])
        reference = {"conversation_id": example["conversation_id"], "split": "test", "style_id": paired["style_id"],
                     "voice_group_id": paired["speaker_id"], "reference_text": paired["response_text"],
                     "reference_audio": paired["response_audio"], "person_a_features": example["path"],
                     "note": "Reference response was not passed into inference. Quality requires listening/evaluation."}
        (example_dir / "reference.json").write_text(json.dumps(reference, indent=2) + "\n")
        metrics = [json.loads(line) for line in (args.output / "metrics.jsonl").read_text().splitlines()]
        if metrics[-1]["epoch"] != args.epochs or metrics[-1]["world_size"] != len(training_gpus):
            raise RuntimeError("Training completion did not match the requested epoch/GPU count")
        save_status("complete", completed_epochs=metrics[-1]["epoch"], training_steps=metrics[-1]["step"],
                    final_validation=metrics[-1]["validation"], example=str(example_dir))
    except Exception as exc:
        save_status("failed", error=str(exc))
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
