"""GPU-ready supervised training and CPU preflight. / GPU 학습과 CPU 사전 점검."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset.full_speech_dataset import FullSpeechDataset, collate_full_speech, fit_target_statistics
from infer_affect import load_sbe
from model.affective_response_transport import AffectiveResponseTransport
from model.full_speech import SpeechConfig, EmpatheticSpeechSystem
from utils.distributed_training import DistributedRuntime, ExactDistributedEvalSampler, LossForward


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the full LLM-free empathetic speech model")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--sbe-checkpoint", type=Path)
    parser.add_argument("--affect-checkpoint", type=Path)
    parser.add_argument("--train-sbe-from-scratch", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="One forward/backward check; no optimizer update")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/full_speech_training"))
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.lr <= 0 or args.log_every < 1:
        parser.error("epochs, batch-size and lr must be positive; workers must be nonnegative")
    if args.resume and (args.sbe_checkpoint or args.affect_checkpoint or args.train_sbe_from_scratch):
        parser.error("--resume restores initialization; omit SBE/affect initialization flags")
    if not args.resume and bool(args.sbe_checkpoint) == args.train_sbe_from_scratch:
        parser.error("Choose --sbe-checkpoint (freeze visual/emotion) OR --train-sbe-from-scratch")
    if args.amp and not args.device.startswith("cuda"):
        parser.error("--amp is supported here only with --device cuda")
    return args


def move_batch(batch, device):
    return {name: value.to(device, non_blocking=device.type == "cuda") for name, value in batch.items()}


def evaluate(model, loader, device, seed, runtime=None):
    model.eval()
    sums, count = {}, 0
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    # Fixed validation noise makes epoch comparisons meaningful. / 고정 검증 잡음으로 에포크를 비교합니다.
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        torch.manual_seed(seed)
        for batch in loader:
            size = batch["mel"].shape[0]
            losses = model.losses(move_batch(batch, device))
            for name, value in losses.items():
                if not torch.isfinite(value):
                    raise RuntimeError(f"Non-finite validation {name}")
                sums[name] = sums.get(name, 0.0) + float(value) * size
            count += size
    if runtime is not None:
        return runtime.mean_losses(sums, count)
    return {name: value / count for name, value in sums.items()}


def atomic_save(payload, path):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run_training(args, runtime):
    device = runtime.device
    amp_dtype = getattr(torch, args.amp_dtype)
    if args.amp and amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU needs --amp-dtype float16")
    torch.manual_seed(args.seed)
    digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    restored = None
    if args.resume:
        model, restored = EmpatheticSpeechSystem.from_checkpoint(args.resume)
        if args.config and SpeechConfig.load(args.config).to_dict() != model.config.to_dict():
            raise ValueError("Resume config differs from checkpoint")
        if restored["metadata"].get("manifest_sha256") != digest:
            raise ValueError("Resume manifest differs from the saved training manifest")
        if restored.get("world_size", 1) != runtime.world_size:
            raise ValueError("Resume requires the same distributed world size")
    else:
        config = SpeechConfig.load(args.config or "configs/full_speech.json")
        model = EmpatheticSpeechSystem(config)
        if args.sbe_checkpoint:
            model.encoder.sbe = load_sbe(args.sbe_checkpoint, config.sbe, "cpu")
            model.set_visual_frozen(True)
        if args.affect_checkpoint:
            affect = AffectiveResponseTransport.from_checkpoint(args.affect_checkpoint)
            if affect.config != model.encoder.transport.config:
                raise ValueError("Affect checkpoint config differs from the full model")
            model.encoder.transport.load_state_dict(affect.state_dict(), strict=True)
    config = model.config
    training = FullSpeechDataset(args.manifest, config, "train", args.allow_synthetic)
    validation = FullSpeechDataset(args.manifest, config, "val", args.allow_synthetic)
    if restored is None:
        for name, value in fit_target_statistics(training, runtime).items():
            getattr(model, name).copy_(value)
    model.to(device)
    generator = torch.Generator().manual_seed(args.seed + runtime.rank)
    sampler = (DistributedSampler(training, num_replicas=runtime.world_size, rank=runtime.rank,
                                  shuffle=True, seed=args.seed) if runtime.distributed else None)
    eval_sampler = ExactDistributedEvalSampler(validation, runtime.rank, runtime.world_size) if runtime.distributed else None
    train_loader = DataLoader(training, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler, generator=generator,
                              num_workers=args.workers, collate_fn=collate_full_speech,
                              pin_memory=device.type == "cuda")
    val_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, sampler=eval_sampler, num_workers=args.workers,
                            collate_fn=collate_full_speech, pin_memory=device.type == "cuda")
    args.output.mkdir(parents=True, exist_ok=True)
    forward_model = LossForward(model)
    if runtime.distributed:
        # Legacy SBE contains unused decoder parameters. / 기존 SBE에는 미사용 디코더 파라미터가 있습니다.
        forward_model = DistributedDataParallel(
            forward_model, device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True, broadcast_buffers=False,
        )
    torch.manual_seed(args.seed + runtime.rank)
    if args.check_only:
        model.train()
        losses = forward_model(move_batch(next(iter(train_loader)), device))
        losses["total"].backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise RuntimeError("Missing or non-finite preflight gradients")
        report = {"optimizer_steps": 0, "synthetic": training.metadata["synthetic"],
                  "world_size": runtime.world_size, "rank": runtime.rank,
                  "losses": {name: float(value.detach()) for name, value in losses.items()},
                  "gradient_tensors": len(gradients), "finite_gradients": True, "device": str(device)}
        filename = "preflight.json" if runtime.primary else f"preflight_rank{runtime.rank}.json"
        (args.output / filename).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and amp_dtype == torch.float16)
    start_epoch, step, best = 0, 0, float("inf")
    if restored:
        optimizer.load_state_dict(restored["optimizer"])
        scheduler.load_state_dict(restored["scheduler"])
        scaler.load_state_dict(restored["scaler"])
        start_epoch, step, best = restored["epoch"] + 1, restored["training_steps"], restored["best_validation"]
        if "rank_rng_states" in restored:
            state = restored["rank_rng_states"][runtime.rank]
            generator.set_state(state["loader_rng"])
            torch.set_rng_state(state["torch_rng"])
            if device.type == "cuda" and state["cuda_rng"] is not None:
                torch.cuda.set_rng_state(state["cuda_rng"], device)
        else:
            generator.set_state(restored["loader_rng"])
            torch.set_rng_state(restored["torch_rng"])
            if device.type == "cuda" and restored.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all(restored["cuda_rng"])
    if start_epoch >= args.epochs:
        raise ValueError("--epochs must exceed the last completed checkpoint epoch")
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        started = time.monotonic()
        sums, count = {}, 0
        for batch_index, batch in enumerate(train_loader):
            size = batch["mel"].shape[0]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=args.amp):
                losses = forward_model(move_batch(batch, device))
            if not torch.isfinite(losses["total"]):
                raise RuntimeError(f"Non-finite training loss at step {step}")
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            for name, value in losses.items():
                sums[name] = sums.get(name, 0.0) + float(value.detach()) * size
            count += size
            if runtime.primary and ((batch_index + 1) % args.log_every == 0 or batch_index == 0):
                progress = {"epoch": epoch + 1, "epochs": args.epochs, "step": step,
                            "batch": batch_index + 1, "batches_per_rank": len(train_loader),
                            "world_size": runtime.world_size, "seconds": time.monotonic() - started,
                            "local_losses": {name: float(value.detach()) for name, value in losses.items()}}
                (args.output / "progress.json").write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
                print(json.dumps(progress), flush=True)
        train_metrics = runtime.mean_losses(sums, count)
        # Raw-model validation has no per-batch collectives. / 원본 모델 검증은 배치별 집단 통신이 없습니다.
        metrics = evaluate(model, val_loader, device, args.seed + 1000, runtime)
        scheduler.step(metrics["total"])
        improved = metrics["total"] < best
        best = min(best, metrics["total"])
        report = {"epoch": epoch + 1, "step": step,
                  "train": train_metrics, "world_size": runtime.world_size,
                  "validation": metrics, "lr": optimizer.param_groups[0]["lr"],
                  "epoch_seconds": time.monotonic() - started}
        states = runtime.gather_rng(generator)
        if not runtime.primary:
            continue
        with (args.output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(report) + "\n")
        payload = model.checkpoint(step, manifest_sha256=digest, synthetic=training.metadata["synthetic"],
                                   target_contract=config.target_contract(), device=str(device),
                                   data_provenance=training.metadata.get("provenance", {}),
                                   amp_dtype=args.amp_dtype if args.amp else None)
        payload.update(epoch=epoch, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                       scaler=scaler.state_dict(), best_validation=best, loader_rng=generator.get_state(),
                       torch_rng=torch.get_rng_state(),
                       cuda_rng=[states[0]["cuda_rng"]] if device.type == "cuda" else None,
                       world_size=runtime.world_size, rank_rng_states=states)
        atomic_save(payload, args.output / "last.pt")
        if improved:
            atomic_save(payload, args.output / "best.pt")
        print(json.dumps(report), flush=True)


def main(argv=None):
    args = parse_args(argv)
    runtime = DistributedRuntime.initialize(args.device)
    try:
        run_training(args, runtime)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
