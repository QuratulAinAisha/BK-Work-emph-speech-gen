"""Staged, resumable quality training. / 단계별 재개 가능한 품질 학습."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from dataset.full_speech_dataset import fit_target_statistics
from dataset.quality_speech_dataset import QualitySpeechDataset, collate_quality
from model.full_speech.quality import QualityConfig, QualitySpeechSystem
from model.full_speech.codec import FrozenEncodec
from train_full import atomic_save, move_batch
from utils.distributed_training import DistributedRuntime, ExactDistributedEvalSampler, LossForward


def mean_losses(sums, count, runtime):
    names = ('affect', 'affect_prior', 'duration', 'semantic', 'codec_flow', 'input_ctc',
             'target_ctc', 'semantic_ctc', 'codec_ctc', 'relevance', 'spectral', 'total')
    packed = torch.tensor([sums.get(name, 0.) for name in names] + [count], dtype=torch.float64, device=runtime.device)
    if runtime.distributed:
        torch.distributed.all_reduce(packed)
    if packed[-1] == 0:
        raise ValueError('No examples in split')
    return {name: float(packed[index] / packed[-1]) for index, name in enumerate(names)}


def evaluate(model, loader, runtime):
    model.eval()
    sums, count = {}, 0
    devices = [runtime.device.index] if runtime.device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        torch.manual_seed(1042 + runtime.rank)
        for batch in loader:
            batch = move_batch(batch, runtime.device)
            values = model.losses(batch)
            if not all(torch.isfinite(value) for value in values.values()):
                raise RuntimeError('Non-finite validation losses')
            size = len(batch['mel'])
            count += size
            for key, value in values.items():
                sums[key] = sums.get(key, 0.) + float(value) * size
    return mean_losses(sums, count, runtime)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--initialize-baseline', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--evaluate-every', type=int, default=5)
    parser.add_argument('--evaluation-count', type=int, default=6)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.resume and args.initialize_baseline:
        parser.error('Choose resume or baseline initialization')
    runtime = DistributedRuntime.initialize(args.device)
    torch.set_num_threads(2)
    try:
        train(args, runtime)
    finally:
        runtime.close()


def train(args, runtime):
    config = QualityConfig.load(args.config)
    fingerprint = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    torch.manual_seed(42)
    restored = None
    if args.resume:
        model, restored = QualitySpeechSystem.from_checkpoint(args.resume)
        if restored['config'] != config.to_dict() or restored['metadata']['manifest_sha256'] != fingerprint:
            raise ValueError('Resume requires identical data and configuration')
        if restored['world_size'] != runtime.world_size:
            raise ValueError('Resume requires the same GPU count')
    else:
        model = QualitySpeechSystem(config)
        if args.initialize_baseline:
            baseline = torch.load(args.initialize_baseline, map_location='cpu', weights_only=True)
            # Width/sampler changes require fresh semantic weights. / 차원·샘플러 변경으로 의미 가중치는 새로 학습합니다.
            allowed = ('encoder.sbe.', 'length_predictor.', 'style_embedding.', 'speaker_embedding.',
                       'codec_generator.velocity.blocks.', 'codec_generator.velocity.input_projection.',
                       'codec_generator.velocity.output.', 'codec_generator.velocity.time_projection.',
                       'codec_generator.velocity.length_projection.', 'codec_generator.velocity.context_projection.')
            own = model.state_dict()
            kept = {key: value for key, value in baseline['state_dict'].items()
                    if key.startswith(allowed) and key in own and own[key].shape == value.shape}
            model.load_state_dict(kept, strict=False)
            print(json.dumps({'initialized_tensors': len(kept), 'baseline_epoch': baseline['epoch'] + 1}), flush=True)
    train_data = QualitySpeechDataset(args.manifest, config, 'train')
    val_data = QualitySpeechDataset(args.manifest, config, 'val')
    if not restored:
        for name, value in fit_target_statistics(train_data, runtime).items():
            getattr(model, name).copy_(value)
    model.to(runtime.device)
    codec = FrozenEncodec(config.codec_model, config.codec_revision).to(runtime.device)
    object.__setattr__(model, '_acoustic_codec', codec)
    generator = torch.Generator().manual_seed(42 + runtime.rank)
    sampler = DistributedSampler(train_data, num_replicas=runtime.world_size, rank=runtime.rank, seed=42)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.workers, collate_fn=collate_quality, generator=generator, pin_memory=runtime.device.type == 'cuda')
    val_loader = DataLoader(val_data, batch_size=args.batch_size,
        sampler=ExactDistributedEvalSampler(val_data, runtime.rank, runtime.world_size),
        num_workers=args.workers, collate_fn=collate_quality)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=.5)
    start, step, best, previous_stage = 0, 0, float('inf'), None
    if restored:
        optimizer.load_state_dict(restored['optimizer'])
        scheduler.load_state_dict(restored['scheduler'])
        start, step, best = restored['epoch'] + 1, restored['training_steps'], restored['best_validation']
        previous_stage = restored['stage']
        state = restored['rank_rng_states'][runtime.rank]
        generator.set_state(state['loader_rng'])
        torch.set_rng_state(state['torch_rng'])
        if runtime.device.type == 'cuda':
            torch.cuda.set_rng_state(state['cuda_rng'], runtime.device)
    else:
        torch.manual_seed(42 + runtime.rank)
    forward = LossForward(model)
    if runtime.distributed:
        forward = DistributedDataParallel(forward, device_ids=[runtime.device.index] if runtime.device.type == 'cuda' else None,
                                          find_unused_parameters=True, broadcast_buffers=False)
    args.output.mkdir(parents=True, exist_ok=True)
    for epoch in range(start, args.epochs):
        model.current_epoch = epoch
        stage = model.stage()
        if previous_stage != stage:
            # Loss scales differ between stages. / 단계마다 손실 크기가 달라 최솟값을 재설정합니다.
            best = float('inf')
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=.5)
        previous_stage = stage
        sampler.set_epoch(epoch)
        model.train()
        sums, count, started = {}, 0, time.monotonic()
        for batch_index, batch in enumerate(train_loader):
            batch = move_batch(batch, runtime.device)
            optimizer.zero_grad(set_to_none=True)
            model.current_step = step
            with torch.autocast(device_type=runtime.device.type, dtype=torch.bfloat16,
                                enabled=runtime.device.type == 'cuda'):
                losses = forward(batch)
            if not all(torch.isfinite(value) for value in losses.values()):
                raise RuntimeError(f'Non-finite losses at step {step}')
            losses['total'].backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            if args.check_only:
                checks = {'optimizer_steps': 0, 'rank': runtime.rank, 'world_size': runtime.world_size,
                          'batch_size': len(batch['mel']), 'gradient_norm': float(norm),
                          'peak_gpu_bytes': torch.cuda.max_memory_allocated() if runtime.device.type == 'cuda' else None,
                          'losses': {key: float(value) for key, value in losses.items()}}
                (args.output / f'preflight_rank{runtime.rank}.json').write_text(json.dumps(checks, indent=2))
                print(json.dumps(checks), flush=True)
                return
            optimizer.step()
            step += 1
            count += len(batch['mel'])
            for key, value in losses.items():
                sums[key] = sums.get(key, 0.) + float(value.detach()) * len(batch['mel'])
            if runtime.primary and (batch_index % 25 == 0):
                progress = {'epoch': epoch + 1, 'epochs': args.epochs, 'stage': stage, 'step': step,
                    'batch': batch_index + 1, 'batches': len(train_loader), 'world_size': runtime.world_size,
                    'predicted_fraction': model.predicted_fraction() if stage == 'joint' else 0,
                    'seconds': time.monotonic() - started}
                (args.output / 'progress.json').write_text(json.dumps(progress, indent=2))
                print(json.dumps(progress), flush=True)
        training = mean_losses(sums, count, runtime)
        validation = evaluate(model, val_loader, runtime)
        scheduler.step(validation['total'])
        improved = validation['total'] < best
        best = min(best, validation['total'])
        rng = runtime.gather_rng(generator)
        if runtime.primary:
            metric = {'epoch': epoch + 1, 'step': step, 'stage': stage, 'world_size': runtime.world_size,
                      'train': training, 'validation': validation, 'epoch_seconds': time.monotonic() - started,
                      'lr': optimizer.param_groups[0]['lr']}
            payload = model.checkpoint(step, manifest_sha256=fingerprint, data_provenance=train_data.metadata['provenance'])
            payload.update(epoch=epoch, stage=stage, best_validation=best, optimizer=optimizer.state_dict(),
                           scheduler=scheduler.state_dict(), rank_rng_states=rng, world_size=runtime.world_size)
            atomic_save(payload, args.output / 'last.pt')
            if improved:
                atomic_save(payload, args.output / f'best_{stage}.pt')
                atomic_save(payload, args.output / 'best.pt')
            with (args.output / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(metric) + '\n')
            print(json.dumps(metric), flush=True)
        if runtime.distributed:
            torch.distributed.barrier()
        if (epoch + 1) % args.evaluate_every == 0 or epoch + 1 == args.epochs:
            if runtime.primary:
                # Use validation for development; reserve test for the final run. / 개발에는 검증 집합을 사용합니다.
                subprocess.run([sys.executable, 'evaluate_quality.py', '--checkpoint', str(args.output / 'last.pt'),
                    '--manifest', str(args.manifest), '--split', 'val', '--count', str(args.evaluation_count),
                    '--output', str(args.output / f'eval_epoch_{epoch + 1:03d}')], check=True)
            if runtime.distributed:
                torch.distributed.barrier()


if __name__ == '__main__':
    main()
