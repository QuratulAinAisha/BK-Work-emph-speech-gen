"""Bounded B-only planner overfit diagnostic. / 제한된 B 전용 계획기 과적합 진단."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from model.full_speech.quality import QualityConfig
from model.full_speech.units import initialize_unit_system, load_codebook
from model.full_speech.experimental_unit_planners import (EXPERIMENTAL_ARCHITECTURE, VARIANTS,
    experimental_descriptor, experimental_system_from_payload, replace_tiny_planner)
from prepare_quality import atomic_json
from train_full import atomic_save


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def make_hidden(length, seed, kind='random', ratio=None):
    if length < 2 or kind not in ('random', 'gap'):
        raise ValueError('Masking needs at least two units and a known mask kind')
    generator = torch.Generator().manual_seed(seed)
    ratio = .2 + .6 * float(torch.rand((), generator=generator)) if ratio is None else ratio
    if not 0 < ratio < 1:
        raise ValueError('Tiny denoising requires both visible and hidden units')
    count = min(length - 1, max(1, math.ceil(length * ratio)))
    hidden = torch.zeros(length, dtype=torch.bool)
    if kind == 'gap':
        start = int(torch.randint(length - count + 1, (), generator=generator))
        hidden[start:start + count] = True
    else:
        hidden[torch.randperm(length, generator=generator)[:count]] = True
    return hidden


def collate_units(records, indices, masks, device):
    rows = [records[index]['ids'] for index in indices]
    if len(rows) != len(masks) or any(len(row) != len(mask) for row, mask in zip(rows, masks)):
        raise ValueError('Unit and mask lengths differ')
    ids = pad_sequence(rows, batch_first=True).to(device)
    lengths = torch.tensor([len(row) for row in rows], device=device)
    valid = torch.arange(ids.shape[1], device=device)[None] < lengths[:, None]
    hidden = pad_sequence(masks, batch_first=True).to(device)
    if hidden.dtype != torch.bool or (hidden & ~valid).any() or not hidden.any(1).all() or not (valid & ~hidden).any(1).all():
        raise ValueError('Each sequence needs valid visible and hidden positions')
    if (ids[~valid] != 0).any():
        raise ValueError('Padding must contain zero IDs')
    return ids, valid, hidden


def b_only_logits(model, ids, valid, hidden):
    planner = model.semantic_planner
    values = planner.codebook.centers
    if ids.dtype != torch.long or ids.min() < 0 or ids.max() >= len(values):
        raise ValueError('Invalid target unit IDs')
    # No A input, duration, style or voice enters this diagnostic. / A·길이·스타일·화자는 이 진단에 들어가지 않습니다.
    context = values.new_zeros(len(ids), 1, 512)
    context_mask = torch.ones(len(ids), 1, device=ids.device, dtype=torch.bool)
    affect = values.new_zeros(len(ids), 1, 6)
    style = values.new_zeros(len(ids), model.config.hidden_dim)
    initial_ids = ids.masked_fill(hidden | ~valid, 0)
    logits = planner.logits(initial_ids, hidden, valid, context, context_mask, affect, style)
    if not torch.isfinite(logits).all() or (logits[~valid] != 0).any():
        raise RuntimeError('Non-finite output or nonzero padding')
    return logits


def nearest_visible_copy(ids, valid, hidden):
    output = torch.zeros_like(ids)
    for index in range(len(ids)):
        visible_positions = (valid[index] & ~hidden[index]).nonzero(as_tuple=True)[0]
        hidden_positions = hidden[index].nonzero(as_tuple=True)[0]
        if not len(visible_positions):
            raise ValueError('Nearest-copy requires visible hints')
        # Sorted positions make equal-distance ties choose left. / 정렬된 위치로 같은 거리는 왼쪽을 택합니다.
        nearest = (hidden_positions[:, None] - visible_positions[None]).abs().argmin(1)
        output[index, hidden_positions] = ids[index, visible_positions[nearest]]
        output[index, visible_positions] = ids[index, visible_positions]
    return output


def fixed_masks(records, indices, seed, ratio):
    return [make_hidden(len(records[i]['ids']), seed + i * 1009, ratio=ratio) for i in indices]


@torch.inference_mode()
def evaluate(model, records, device, batch_size, seed, fixed_ratio):
    model.eval()
    conditions = [('fixed', 'random', fixed_ratio, [None])]
    for kind, label in [('random', 'fresh_random'), ('gap', 'contiguous_gap')]:
        conditions.extend((f'{label}_{round(ratio * 100):03d}', kind, ratio, [0, 1])
                          for ratio in (.25, .5, .75))
    report = {}
    for label, kind, ratio, repeats in conditions:
        total_ce, total_correct, total_copy, total_frames = 0., 0, 0, 0
        tail_correct, tail_copy, tail_frames = 0, 0, 0
        patterns = hashlib.sha256()
        for repeat in repeats:
            for offset in range(0, len(records), batch_size):
                indices = list(range(offset, min(offset + batch_size, len(records))))
                masks = fixed_masks(records, indices, seed, ratio) if repeat is None else [
                    make_hidden(len(records[i]['ids']), seed + 100000 + 9973 * repeat + 1009 * i,
                                kind=kind, ratio=ratio) for i in indices]
                for mask in masks:
                    patterns.update(mask.numpy().tobytes())
                ids, valid, hidden = collate_units(records, indices, masks, device)
                logits = b_only_logits(model, ids, valid, hidden)
                prediction = logits.argmax(-1)
                copied = nearest_visible_copy(ids, valid, hidden)
                total_ce += float(F.cross_entropy(logits[hidden].float(), ids[hidden], reduction='sum'))
                total_correct += int(((prediction == ids) & hidden).sum())
                total_copy += int(((copied == ids) & hidden).sum())
                total_frames += int(hidden.sum())
                prefix = (valid.sum(1).float() * .2).ceil().long()
                tail = hidden & (torch.arange(ids.shape[1], device=device)[None] >= prefix[:, None])
                tail_correct += int(((prediction == ids) & tail).sum())
                tail_copy += int(((copied == ids) & tail).sum())
                tail_frames += int(tail.sum())
        report[label] = {'hidden_ce': total_ce / total_frames, 'hidden_unit_accuracy': total_correct / total_frames,
            'nearest_copy_accuracy': total_copy / total_frames, 'hidden_frames': total_frames,
            'neural_advantage_percentage_points': 100 * (total_correct - total_copy) / total_frames,
            'tail_hidden_accuracy': tail_correct / tail_frames if tail_frames else None,
            'tail_nearest_copy_accuracy': tail_copy / tail_frames if tail_frames else None,
            'tail_hidden_frames': tail_frames, 'mask_sha256': patterns.hexdigest(),
            'mask_seed_repeats': len(repeats), 'hidden_ratio': ratio}
    return report


@torch.inference_mode()
def extract_training_units(manifest_path, selection_path, count, model, device):
    manifest = json.loads(manifest_path.read_text())
    selection = json.loads(selection_path.read_text())
    if selection.get('manifest_sha256') != digest(manifest_path):
        raise ValueError('Selection manifest changed')
    if manifest['target_contract'] != model.config.target_contract():
        raise ValueError('B feature contract differs from checkpoint')
    wanted = selection['train']
    if len(wanted) != len(set(wanted)):
        raise ValueError('Duplicate training paths')
    mapping = {row['path']: row for row in manifest['records']}
    chosen, seen = [], set()
    for path in wanted:
        if path not in mapping or mapping[path]['split'] != 'train':
            raise ValueError('Only original training rows are allowed')
        row = mapping[path]
        if row['conversation_id'] in seen:
            continue
        chosen.append(row); seen.add(row['conversation_id'])
        if len(chosen) == count:
            break
    if len(chosen) != count:
        raise ValueError('Insufficient distinct training conversations')
    records = []
    for row in chosen:
        # Read only B semantics; A caches and held-out audio stay unopened. / B 의미 특징만 읽고 A·미사용 음성은 열지 않습니다.
        with np.load(manifest_path.parent / row['path'], allow_pickle=False) as cache:
            values = torch.from_numpy(cache['semantic'].astype(np.float32)).to(device)
        if values.ndim != 2 or values.shape[1] != 768 or len(values) < 2 or not torch.isfinite(values).all():
            raise ValueError('Invalid B semantic features')
        normalized = (values - model.semantic_mean) / model.semantic_std
        if not torch.isfinite(normalized).all():
            raise ValueError('Invalid semantic normalization')
        ids = model.semantic_planner.codebook.encode(normalized).cpu()
        records.append({'path': row['path'], 'conversation_id': row['conversation_id'], 'ids': ids})
    return records


def configure_training(model, dropout_mode):
    model.configure_unit_prior()
    model.train()
    if dropout_mode == 'off':
        model.semantic_planner.eval()  # Deterministic dropout, gradients still enabled. / 드롭아웃만 끄고 미분은 유지합니다.


def frozen_snapshot(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            if not name.startswith('semantic_planner.') or name == 'semantic_planner.codebook.centers'}


def audit_state(model, frozen):
    state = model.state_dict()
    changed = [name for name, before in frozen.items() if not torch.equal(before, state[name].detach().cpu())]
    nonfinite = [name for name, value in state.items() if not torch.isfinite(value).all()]
    bad_gradients = [name for name, parameter in model.named_parameters()
                    if not name.startswith('semantic_planner.') and parameter.grad is not None]
    if changed or nonfinite or bad_gradients:
        raise RuntimeError(f'Tiny checkpoint audit failed: {changed}, {nonfinite}, {bad_gradients}')
    return {'passed': True, 'frozen_tensors_verified': len(frozen), 'frozen_changed': changed,
            'nonfinite_tensors': nonfinite, 'nonplanner_gradients': bad_gradients,
            'trainable_components': model.trainable_components, 'codebook_frozen': True}


def train(args):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    payload = torch.load(args.resume or args.initialize, map_location='cpu', weights_only=True)
    if payload.get('architecture') == EXPERIMENTAL_ARCHITECTURE:
        if payload['experimental_planner']['variant'] != args.planner_variant:
            raise ValueError('Pass the matching --planner-variant for this experimental checkpoint')
        model = experimental_system_from_payload(payload)
        if args.unit_codebook:
            requested, _ = load_codebook(args.unit_codebook, model)
            if not torch.equal(requested.centers, model.semantic_planner.codebook.centers):
                raise ValueError('Experimental checkpoint and requested codebook differ')
    else:
        model = initialize_unit_system(QualityConfig(**payload['config']), payload, args.unit_codebook)
        replace_tiny_planner(model, args.planner_variant)
    model.to(device).float()
    records = extract_training_units(args.manifest, args.selection, args.count, model, device)
    target_hash = hashlib.sha256()
    for row in records:
        target_hash.update(row['path'].encode()); target_hash.update(row['ids'].numpy().tobytes())
    initial_hash = payload['tiny_recipe']['initialize_sha256'] if args.resume else digest(args.initialize)
    codebook_hash = digest(args.unit_codebook) if args.unit_codebook else \
        payload.get('metadata', {}).get('recovery_recipe', {}).get('codebook_sha256')
    if not isinstance(codebook_hash, str) or len(codebook_hash) != 64:
        raise ValueError('Supply a codebook or initialize from a checkpoint with its recorded codebook SHA256')
    recipe = {'version': 1, 'manifest_sha256': digest(args.manifest), 'selection_sha256': digest(args.selection),
        'initialize_sha256': initial_hash, 'targets_sha256': target_hash.hexdigest(),
        'unit_codebook_sha256': codebook_hash,
        'count': args.count, 'batch_size': args.batch_size, 'fixed_updates': args.fixed_updates,
        'fresh_updates': args.fresh_updates, 'fixed_mask_ratio': args.fixed_mask_ratio,
        'lr': args.lr, 'dropout_mode': args.dropout_mode, 'seed': args.seed,
        'precision': 'float32; TF32 disabled', 'optimizer': 'AdamW, weight_decay=0, constant LR, clip_norm=1',
        'scope': 'B-only prior; semantic planner only; no A, duration, style or speaker supervision',
        'fresh_training_masks': 'New per-update random masks with requested ratio sampled uniformly in[.2,.8]',
        'evaluation': 'Same training conversations; fixed evaluation seeds; no held-out conversation test',
        'evaluate_every': args.evaluate_every, 'save_every': args.save_every}
    if args.planner_variant != 'existing':
        recipe['experimental_planner'] = experimental_descriptor(model, args.planner_variant)
    if args.resume and payload.get('tiny_recipe') != recipe:
        raise ValueError('Tiny resume recipe changed')
    configure_training(model, args.dropout_mode)
    frozen = frozen_snapshot(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.)
    start = int(payload['tiny_step']) if args.resume else 0
    if args.resume:
        optimizer.load_state_dict(payload['optimizer'])
        torch.set_rng_state(payload['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state(payload['cuda_rng'], device)
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.resume and (args.output / 'curve.jsonl').exists():
        raise ValueError('Existing tiny run requires --resume')
    atomic_json(args.output / 'recipe.json', recipe)
    model_summary = {'planner_variant': args.planner_variant,
        'production_loader_compatible': args.planner_variant == 'existing',
        'total_model_parameters': sum(p.numel() for p in model.parameters()),
        'planner_parameters': sum(p.numel() for p in model.semantic_planner.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'hidden_dim': model.config.hidden_dim, 'layers': model.config.planner_layers,
        'heads': model.config.num_heads, 'device': str(device), 'torch_version': str(torch.__version__),
        'comparison_limit': 'Same data/masks/budget; architecture parameter counts and initialization histories differ.'}
    atomic_json(args.output / 'model_summary.json', model_summary)
    atomic_save({'records': records, 'targets_sha256': target_hash.hexdigest(),
                 'manifest_sha256': recipe['manifest_sha256']}, args.output / 'targets.pt')
    total = args.fixed_updates + args.fresh_updates
    limit = min(total, args.stop_after or total)
    if start >= limit:
        raise ValueError('Resume step already reaches requested stopping point')
    started = time.monotonic()
    runtime = dict(payload.get('tiny_runtime', {})) if args.resume else {}
    runtime.setdefault('measured_updates', 0)
    runtime.setdefault('optimizer_update_wall_seconds', 0.)

    def measure(step, phase):
        result = {'step': step, 'phase': phase, 'elapsed_seconds': time.monotonic() - started,
                  'metrics': evaluate(model, records, device, args.batch_size, args.seed, args.fixed_mask_ratio)}
        result['audit'] = audit_state(model, frozen)
        with (args.output / 'curve.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(result) + '\n')
        atomic_json(args.output / 'latest_evaluation.json', result)
        print(json.dumps({'step': step, 'phase': phase, 'fixed': result['metrics']['fixed'],
                          'fresh_random_050': result['metrics']['fresh_random_050']}), flush=True)
        configure_training(model, args.dropout_mode)
        return result

    def save(step, phase):
        audit = audit_state(model, frozen)
        checkpoint = model.checkpoint(step, tiny_recipe=recipe, recovery_recipe={
            'stage': 'planner', 'planner_representation': 'masked_discrete_units_v1',
            'codebook_sha256': codebook_hash, 'tiny_diagnostic': True, 'tiny_recipe': recipe})
        if args.planner_variant != 'existing':
            # Experimental headers prevent silent production loading. / 실험 헤더로 잘못된 운영 로딩을 막습니다.
            checkpoint['architecture'] = EXPERIMENTAL_ARCHITECTURE
            checkpoint['experimental_planner'] = experimental_descriptor(model, args.planner_variant)
        checkpoint.update(tiny_recipe=recipe, tiny_step=step, tiny_phase=phase,
            recovery_step=step, stage='unit_prior_tiny', optimizer=optimizer.state_dict(),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
            frozen_audit=audit, total_updates=total, tiny_runtime=runtime)
        atomic_save(checkpoint, args.output / 'last.pt')
        atomic_save(checkpoint, args.output / f'step_{step:04d}.pt')
        atomic_json(args.output / f'audit_step_{step:04d}.json', audit)

    if not args.resume:
        measure(0, 'initial'); save(0, 'initial')
    last_evaluation = None
    for step in range(start, limit):
        update_started = time.monotonic()
        phase = 'fixed' if step < args.fixed_updates else 'fresh'
        order = torch.randperm(len(records), generator=torch.Generator().manual_seed(args.seed + step)).tolist()
        indices = order[:args.batch_size]
        seeds = [args.seed + i * 1009 if phase == 'fixed' else args.seed + 1000000 + step * 100003 + i * 1009
                 for i in indices]
        masks = [make_hidden(len(records[i]['ids']), seed, ratio=args.fixed_mask_ratio if phase == 'fixed' else None)
                 for i, seed in zip(indices, seeds)]
        ids, valid, hidden = collate_units(records, indices, masks, device)
        optimizer.zero_grad(set_to_none=True)
        logits = b_only_logits(model, ids, valid, hidden)
        loss = F.cross_entropy(logits[hidden].float(), ids[hidden])
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite tiny loss')
        loss.backward()
        parameters = [p for p in model.parameters() if p.requires_grad]
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        if float(norm) == 0. and float(loss) > 1e-6:
            raise RuntimeError('Nonzero loss has no planner gradient')
        optimizer.step()
        audit = {'step': step + 1, 'phase': phase, 'loss': float(loss.detach()),
            'gradient_norm_before_clip': float(norm), 'hidden_frames': int(hidden.sum()),
            'batch_indices': indices, 'mask_seeds': seeds,
            'mask_sha256': hashlib.sha256(hidden.cpu().numpy().tobytes()).hexdigest(),
            'training_hidden_accuracy': float((logits.argmax(-1)[hidden] == ids[hidden]).float().mean())}
        audit['update_wall_seconds'] = time.monotonic() - update_started
        runtime['measured_updates'] += 1
        runtime['optimizer_update_wall_seconds'] += audit['update_wall_seconds']
        with (args.output / 'updates.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(audit) + '\n')
        if (step + 1) % 10 == 0 or step + 1 == limit:
            atomic_json(args.output / 'progress.json', {**audit, 'total_updates': total,
                        'elapsed_seconds': time.monotonic() - started})
        boundary = step + 1 in (args.fixed_updates, limit)
        if (step + 1) % args.evaluate_every == 0 or boundary:
            last_evaluation = measure(step + 1, phase)
        if (step + 1) % args.save_every == 0 or boundary:
            save(step + 1, phase)
    if limit == total:
        atomic_json(args.output / 'runtime.json', {**model_summary, **runtime,
            'mean_update_wall_seconds': runtime['optimizer_update_wall_seconds'] / runtime['measured_updates'],
            'this_process_elapsed_seconds': time.monotonic() - started,
            'timing_note': 'Update timings include batch construction and synchronization, exclude evaluation/checkpoint writes.'})
        atomic_json(args.output / 'complete.json', {'updates': total, 'fixed_updates': args.fixed_updates,
            'fresh_updates': args.fresh_updates, 'training_conversations': len(records),
            'fixed_mask_accuracy_at_endpoint': last_evaluation['metrics']['fixed']['hidden_unit_accuracy'],
            'fresh_half_mask_accuracy_at_endpoint': last_evaluation['metrics']['fresh_random_050']['hidden_unit_accuracy'],
            'all_frozen_audits_passed': True, 'held_out_generalization_tested': False,
            'production_candidate_promoted': False})


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    for name in ('manifest', 'selection', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--initialize', type=Path)
    source.add_argument('--resume', type=Path)
    parser.add_argument('--unit-codebook', type=Path)
    parser.add_argument('--planner-variant', choices=VARIANTS, default='existing')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--count', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--fixed-updates', type=int, default=300)
    parser.add_argument('--fresh-updates', type=int, default=300)
    parser.add_argument('--fixed-mask-ratio', type=float, default=.5)
    parser.add_argument('--evaluate-every', type=int, default=50)
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--dropout-mode', choices=('off', 'train'), default='off')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args(argv)
    if not 8 <= args.count <= 32 or not 1 <= args.batch_size <= args.count:
        parser.error('Use8–32 conversations and a batch no larger than the subset')
    if min(args.fixed_updates, args.fresh_updates, args.evaluate_every, args.save_every) < 1 or \
            not math.isfinite(args.lr) or args.lr <= 0 or not 0 < args.fixed_mask_ratio < 1 or \
            (args.stop_after is not None and args.stop_after < 1):
        parser.error('Invalid bounded training settings')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        parser.error('Tiny diagnostic runs on one GPU; omit torchrun')
    return args


if __name__ == '__main__':
    train(parse_args())
