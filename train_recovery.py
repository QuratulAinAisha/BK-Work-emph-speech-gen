"""Bounded four-GPU recovery pilot. / 제한된 4-GPU 복구 실험."""

import argparse
from dataclasses import replace
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
import time

import torch
from torch.nn.parallel import DistributedDataParallel

from dataset.quality_speech_dataset import QualitySpeechDataset, collate_quality
from model.full_speech.quality import QualityConfig
from model.full_speech.recovery import RecoverySpeechSystem, FrozenWaveformCTC, ASR_MODEL, ASR_REVISION
from model.full_speech.codec import FrozenEncodec
from prepare_quality import atomic_json
from train_full import move_batch, atomic_save
from utils.distributed_training import DistributedRuntime, LossForward


def load_selected(manifest, config, split, selection):
    source = QualitySpeechDataset(manifest, config, split)
    wanted = set(selection[split])
    chosen = [i for i, row in enumerate(source.records) if row['path'] in wanted]
    if len(chosen) != len(wanted):
        raise ValueError('Selected records missing or duplicated')
    return [source[i] for i in chosen]


def average(values, count, runtime):
    keys = sorted(values)
    packed = torch.tensor([values[k] for k in keys] + [count], dtype=torch.float64, device=runtime.device)
    if runtime.distributed:
        torch.distributed.all_reduce(packed)
    if not packed[-1]:
        raise ValueError('Empty evaluation')
    return {key: float(packed[i] / packed[-1]) for i, key in enumerate(keys)}


def validate(model, samples, runtime, batch_size):
    model.eval()
    sums, count = {}, 0
    devices = [runtime.device.index] if runtime.device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        torch.manual_seed(1042 + runtime.rank)
        local = samples[runtime.rank::runtime.world_size]
        for offset in range(0, len(local), batch_size):
            batch = move_batch(collate_quality(local[offset:offset + batch_size]), runtime.device)
            with torch.autocast(device_type=runtime.device.type, dtype=torch.bfloat16,
                                enabled=runtime.device.type == 'cuda'):
                losses = model.losses(batch)
            if not all(bool(torch.isfinite(v)) for v in losses.values()):
                raise RuntimeError('Non-finite validation')
            n = len(batch['mel']); count += n
            for k, value in losses.items():
                sums[k] = sums.get(k, 0.) + float(value) * n
    return average(sums, count, runtime)


def train(args, runtime):
    torch.set_num_threads(2)
    selection = json.loads(args.selection.read_text())
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    selection_hash = hashlib.sha256(args.selection.read_bytes()).hexdigest()
    if selection['manifest_sha256'] != manifest_hash:
        raise ValueError('Pilot manifest changed')
    payload = torch.load(args.resume or args.initialize, map_location='cpu', weights_only=True)
    config = replace(QualityConfig(**payload['config']), semantic_steps=args.semantic_steps,
                     predicted_semantic_steps=args.semantic_steps, acoustic_loss_every=args.waveform_every)
    if args.planner_memory is not None:
        config = replace(config, planner_memory_mode=args.planner_memory)
    recipe = {'stage': args.stage, 'steps': args.steps, 'batch_size': args.batch_size,
              'world_size': runtime.world_size, 'lr': args.lr, 'semantic_steps': args.semantic_steps,
              'waveform_every': args.waveform_every, 'teacher_weight': args.teacher_weight,
              'training_predicted_fraction': 1. if args.stage == 'planner' else args.predicted_fraction,
              'validation_predicted_fraction': 1.,
              'flow_time_range': [0., 1.], 'manifest_sha256': manifest_hash, 'selection_sha256': selection_hash,
              'waveform_teacher': ASR_MODEL, 'waveform_teacher_revision': ASR_REVISION,
              'scheduler': 'fixed-step warmup and cosine; independent of changing training loss'}
    if args.seed != 42:
        recipe['seed'] = args.seed
    units = args.unit_codebook is not None or payload.get('architecture') == 'llm_free_speech_units_v1'
    if args.planner_memory is not None or args.freeze_duration or config.planner_memory_mode != 'fused':
        if not units or args.stage != 'planner' or args.unit_prior:
            raise ValueError('Planner memory experiments require conditional discrete planner training')
        recipe['planner_memory'] = {'mode': config.planner_memory_mode,
                                    'freeze_duration': args.freeze_duration}
    if args.sampled_audio:
        recipe['sampled_audio'] = {'version': 2, 'codec_steps': config.codec_steps,
            'frozen_codec_content': True, 'generator_eval_mode': True, 'full_sampler_gradients': True,
            'flow_weight': args.sampled_flow_weight, 'waveform_samples_per_rank': args.waveform_samples,
            'validation_waveform_coverage': 'all examples', 'rotating_training_items': True}
    if args.planner_sampled_audio:
        if not units or args.stage != 'planner' or args.unit_prior or not args.freeze_duration:
            raise ValueError('Sampled planner audio needs a conditional unit planner and frozen duration')
        recipe['sampled_planner_audio'] = {'version': 1, 'weight': args.planner_audio_weight,
            'every': args.waveform_every, 'uses_b_for_generation': False, 'uses_predicted_duration': True,
            'gradient_estimator': 'biased final-step straight-through planner; full frozen acoustic sampler',
            'validation_waveform_coverage': 'one fixed item per rank/batch; independent audio evaluation required'}
    if units:
        recipe['planner_representation'] = 'masked_discrete_units_v1'
        recipe['codebook_sha256'] = (hashlib.sha256(args.unit_codebook.read_bytes()).hexdigest()
            if args.unit_codebook else payload['metadata']['recovery_recipe']['codebook_sha256'])
    if args.unit_prior:
        recipe['unit_prior'] = {'version': 1, 'mask_ratio': [.2, .8], 'uses_a': False,
                                'validation': 'deterministic partial masks'}
    if args.unit_objective != 'legacy':
        if not units or args.stage != 'planner':
            raise ValueError('Controlled unit objectives require a discrete planner')
        recipe['controlled_unit_objective'] = {'version': 1, 'name': args.unit_objective,
            'rehearsal_weight': args.unit_rehearsal, 'total_steps': args.steps,
            'loss_reduction': 'equal sequence weighting', 'validation': 'fixed partial prior / fully hidden conditional'}
    elif args.unit_rehearsal:
        raise ValueError('Rehearsal requires a controlled unit objective')
    if args.audio_selection:
        recipe['epoch_audio'] = {'selection_sha256': hashlib.sha256(args.audio_selection.read_bytes()).hexdigest(),
                                 'count': 8, 'semantic_steps': args.semantic_steps}
    if args.resume and payload['metadata']['recovery_recipe'] != recipe:
        raise ValueError('Resume recipe changed')
    torch.manual_seed(args.seed)  # Same new parameters on every rank. / 모든 순위에서 새 가중치를 동일하게 만듭니다.
    if units:
        from model.full_speech.units import initialize_unit_system
        model = initialize_unit_system(config, payload, args.unit_codebook)
    else:
        model = RecoverySpeechSystem(config)
        model.load_state_dict(payload['state_dict'], strict=True)
    model.to(runtime.device)
    if args.stage != 'planner' or not units or args.planner_sampled_audio:
        codec = FrozenEncodec().to(runtime.device)
        object.__setattr__(model, '_acoustic_codec', codec)
    teacher = FrozenWaveformCTC().to(runtime.device) if args.teacher_weight or args.planner_sampled_audio else None
    model.configure_recovery(args.stage, args.predicted_fraction, teacher, args.teacher_weight)
    if args.freeze_duration:
        # Hold response timing fixed across memory arms. / 메모리 실험 간 응답 길이 예측을 고정합니다.
        model.length_predictor.requires_grad_(False)
        model.trainable_components = [name for name in model.trainable_components if name != 'length_predictor']
        model.train(model.training)
    if args.unit_prior:
        if not units:
            raise ValueError('Unit prior requires a discrete-unit model')
        model.configure_unit_prior()
    if args.unit_objective != 'legacy':
        model.configure_unit_objective(args.unit_objective, args.steps, args.unit_rehearsal)
    if args.sampled_audio:
        model.configure_sampled_audio(args.sampled_flow_weight, args.waveform_samples)
    if args.planner_sampled_audio:
        model.configure_planner_audio(args.planner_audio_weight, args.waveform_every)
    training = load_selected(args.manifest, config, 'train', selection)
    validation = load_selected(args.manifest, config, 'val', selection)
    global_batch = runtime.world_size * args.batch_size
    if len(training) % global_batch or len(validation) < runtime.world_size:
        raise ValueError('Pilot needs full unique global batches and validation on every rank')
    batches = len(training) // global_batch
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=.01)
    def schedule(step):
        warmup = min(50, max(1, args.steps // 10))
        if step < warmup:
            return .1 + .9 * step / warmup
        return .1 + .9 * .5 * (1 + math.cos(math.pi * min(1., (step - warmup) / max(1, args.steps - warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    start, best = 0, float('inf')
    generator = torch.Generator().manual_seed(args.seed + runtime.rank)
    if args.resume:
        optimizer.load_state_dict(payload['optimizer']); scheduler.load_state_dict(payload['scheduler'])
        start, best = payload['recovery_step'], payload['best_validation']
        rng = payload['rank_rng_states'][runtime.rank]
        torch.set_rng_state(rng['torch_rng']); generator.set_state(rng['loader_rng'])
        if runtime.device.type == 'cuda':
            torch.cuda.set_rng_state(rng['cuda_rng'], runtime.device)
    else:
        torch.manual_seed(args.seed + runtime.rank)
    wrapped = LossForward(model)
    if runtime.distributed:
        wrapped = DistributedDataParallel(wrapped, device_ids=[runtime.device.index] if runtime.device.type == 'cuda' else None,
                                         find_unused_parameters=True, broadcast_buffers=False)
    args.output.mkdir(parents=True, exist_ok=True)
    if runtime.primary:
        atomic_json(args.output / 'recipe.json', recipe)
    initial = validate(model, validation, runtime, args.batch_size)
    if runtime.primary:
        atomic_json(args.output / ('resume_validation.json' if args.resume else 'initial_validation.json'), initial)
    if not args.resume and not args.check_only:
        # Retain initialization if training makes validation worse. / 검증이 악화되면 초기 가중치를 보존합니다.
        best = initial['total']
        rng = runtime.gather_rng(generator)
        if runtime.primary:
            checkpoint = model.checkpoint(0, recovery_recipe=recipe, manifest_sha256=manifest_hash,
                                         initialized_from=str(args.initialize))
            checkpoint.update(epoch=-1, recovery_step=0, stage=args.stage, best_validation=best,
                optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                rank_rng_states=rng, world_size=runtime.world_size)
            atomic_save(checkpoint, args.output / 'best.pt')
    model.train()
    since, sums, count = time.monotonic(), {}, 0
    limit = min(args.steps, args.stop_after or args.steps)
    for step in range(start, limit):
        # Index-only shuffling makes resume independent of loader iteration. / 인덱스 셔플로 재개 순서를 고정합니다.
        epoch, offset = divmod(step, batches)
        order = torch.randperm(len(training), generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()
        begin = offset * global_batch + runtime.rank * args.batch_size
        batch = move_batch(collate_quality([training[i] for i in order[begin:begin + args.batch_size]]), runtime.device)
        model.current_step = step
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=runtime.device.type, dtype=torch.bfloat16,
                            enabled=runtime.device.type == 'cuda'):
            losses = wrapped(batch)
        if not all(bool(torch.isfinite(v)) for v in losses.values()):
            raise RuntimeError(f'Non-finite training at step {step}')
        losses['total'].backward()
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
        if args.check_only:
            result = {'rank': runtime.rank, 'gradient_norm': float(norm),
                      'teacher_frozen': teacher is None or all(p.grad is None for p in teacher.parameters()),
                      'trainable': model.trainable_components,
                      'losses': {k: float(v.detach()) for k, v in losses.items()},
                      'peak_gpu_bytes': torch.cuda.max_memory_allocated(runtime.device) if runtime.device.type == 'cuda' else 0}
            atomic_json(args.output / f'preflight_rank{runtime.rank}.json', result)
            print(json.dumps(result), flush=True)
            return
        optimizer.step(); scheduler.step()
        count += args.batch_size
        for k, value in losses.items():
            sums[k] = sums.get(k, 0.) + float(value.detach()) * args.batch_size
        if runtime.primary and (step + 1) % 10 == 0:
            status = {'stage': args.stage, 'step': step + 1, 'steps': args.steps, 'world_size': runtime.world_size,
                      'elapsed_seconds': time.monotonic() - since, 'loss': float(losses['total']),
                      'lr': optimizer.param_groups[0]['lr']}
            atomic_json(args.output / 'progress.json', status); print(json.dumps(status), flush=True)
        if (step + 1) % args.evaluate_every == 0 or step + 1 == limit:
            # Waveform loss is sparse; aggregate only available terms across identical rank schedules. / 희소 파형 손실도 동일 일정으로 집계합니다.
            train_values = average(sums, count, runtime)
            val_values = validate(model, validation, runtime, args.batch_size)
            improved = val_values['total'] < best
            best = min(best, val_values['total'])
            rng = runtime.gather_rng(generator)
            if runtime.primary:
                result = {'step': step + 1, 'stage': args.stage, 'train': train_values, 'validation': val_values,
                          'elapsed_seconds': time.monotonic() - since, 'world_size': runtime.world_size}
                checkpoint = model.checkpoint(step + 1, recovery_recipe=recipe, manifest_sha256=manifest_hash)
                checkpoint.update(epoch=epoch, recovery_step=step + 1, stage=args.stage,
                    best_validation=best, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                    rank_rng_states=rng, world_size=runtime.world_size)
                atomic_save(checkpoint, args.output / 'last.pt')
                # Keep candidates for audio-based selection. / 음성 기준 선택을 위해 후보를 보존합니다.
                if args.save_candidates:
                    atomic_save(checkpoint, args.output / f'step_{step + 1:04d}.pt')
                if improved:
                    atomic_save(checkpoint, args.output / 'best.pt')
                with (args.output / 'metrics.jsonl').open('a') as f:
                    f.write(json.dumps(result) + '\n')
                print(json.dumps(result), flush=True)
                if args.audio_selection:
                    # Evaluate this saved epoch before continuing updates. / 다음 갱신 전에 저장된 에포크의 음성을 검사합니다.
                    audio_out = args.output / f'audio_step_{step + 1:04d}'
                    if not (audio_out / 'report.json').exists():
                        with (args.output / f'audio_step_{step + 1:04d}.log').open('a') as log:
                            subprocess.run([sys.executable, 'scripts/diagnose_quality.py',
                                '--checkpoint', str(args.output / 'last.pt'), '--manifest', str(args.manifest),
                                '--selection', str(args.audio_selection), '--output', str(audio_out),
                                '--split', 'val', '--count', '8', '--steps', str(args.semantic_steps),
                                '--device', str(runtime.device)], stdout=log, stderr=subprocess.STDOUT, check=True)
            if runtime.distributed:
                torch.distributed.barrier()
            model.train(); sums, count = {}, 0
    if runtime.primary and limit == args.steps:
        atomic_json(args.output / 'complete.json', {'steps': args.steps, 'world_size': runtime.world_size,
                                                  'stage': args.stage, 'quality_passed': False})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--initialize', type=Path)
    p.add_argument('--unit-codebook', type=Path, help='Initialize the experimental masked-unit planner')
    p.add_argument('--resume', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage', choices=['acoustic', 'planner', 'joint'], required=True)
    p.add_argument('--steps', type=int, default=400)
    p.add_argument('--stop-after', type=int, help='Checkpoint at this total step without marking complete')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--evaluate-every', type=int, default=100)
    p.add_argument('--semantic-steps', type=int, default=8)
    p.add_argument('--waveform-every', type=int, default=4)
    p.add_argument('--teacher-weight', type=float, default=.05)
    p.add_argument('--predicted-fraction', type=float, default=.2)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--save-candidates', action='store_true')
    p.add_argument('--sampled-audio', action='store_true', help='Differentiate the full inference sampler with frozen evaluators')
    p.add_argument('--planner-sampled-audio', action='store_true', help='A-only sampled audio with a frozen content evaluator')
    p.add_argument('--planner-audio-weight', type=float, default=.02)
    p.add_argument('--sampled-flow-weight', type=float, default=1.)
    p.add_argument('--waveform-samples', type=int, default=1)
    p.add_argument('--unit-prior', action='store_true', help='Train a B-unit denoising prior without A inputs')
    p.add_argument('--unit-objective', choices=['legacy', 'balanced', 'curriculum', 'fully_masked'], default='legacy')
    p.add_argument('--unit-rehearsal', type=float, default=0., help='B-only reconstruction weight in conditional training')
    p.add_argument('--audio-selection', type=Path, help='Eight fixed A-only audio examples after each saved epoch')
    p.add_argument('--planner-memory', choices=['fused', 'native_speech', 'resampled_speech'],
                   help='Checkpoint-persisted input memory for the conditional unit planner')
    p.add_argument('--freeze-duration', action='store_true', help='Update only Module 5 in the memory comparison')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if bool(args.initialize) == bool(args.resume):
        p.error('Choose initialization or resume')
    if args.unit_prior and (args.stage != 'planner' or args.sampled_audio or args.audio_selection):
        p.error('Unit prior needs planner stage, without sampled audio or A-only epoch evaluation')
    if args.unit_prior and (args.unit_objective == 'fully_masked' or args.planner_sampled_audio):
        p.error('B-only prior needs visible hints and cannot use conditional waveform supervision')
    if args.sampled_audio and args.planner_sampled_audio:
        p.error('Choose acoustic or planner waveform training, not both')
    if args.audio_selection and args.stage != 'planner':
        p.error('Epoch audio selection is for planner training')
    if min(args.steps, args.batch_size, args.evaluate_every, args.waveform_every) < 1 or not 0 <= args.teacher_weight:
        p.error('Invalid training limits')
    if (args.output / 'metrics.jsonl').exists() and not args.resume:
        p.error('Existing training requires --resume')
    runtime = DistributedRuntime.initialize(args.device)
    try:
        train(args, runtime)
    finally:
        runtime.close()


if __name__ == '__main__':
    main()
