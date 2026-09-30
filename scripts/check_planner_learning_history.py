"""Compare saved B-unit learning stages. / 저장된 B 음성 단위 학습 단계를 비교합니다."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F

from dataset.quality_speech_dataset import QualitySpeechDataset, collate_quality, person_a_only
from model.full_speech.tensor_ops import counts, mask_from_lengths
from model.full_speech.units import UnitSpeechSystem
from prepare_quality import atomic_json
from scripts.check_planner_hints import nested_hidden_mask, mask_metadata
from scripts.planner_sampling import sample_units
from train_full import move_batch


DEFAULT_CHECKPOINTS = {
    'prior': 'outputs/planner_stages_v1/unit_prior/best.pt',
    'conditional': 'outputs/planner_stages_v1/prior_finetune/best.pt',
    'current': 'outputs/planner_memory_v1/fused/step_0320.pt',
}
MASK_TYPES = ('random', 'contiguous')
CONDITIONS = ('null_prior', 'production_a')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def interface_identity(model):
    return {'centers_sha256': tensor_hash(model.semantic_planner.codebook.centers),
            'semantic_mean_sha256': tensor_hash(model.semantic_mean),
            'semantic_std_sha256': tensor_hash(model.semantic_std),
            'unit_count': len(model.semantic_planner.codebook.centers)}


def gap_mask(mask, ratio, seed, kind):
    if kind not in MASK_TYPES:
        raise ValueError('Unknown gap-mask kind')
    if kind == 'random':
        return nested_hidden_mask(mask, ratio, seed)
    if mask.ndim != 2 or mask.dtype != torch.bool or not mask.any(1).all() or not 0 <= ratio <= 1:
        raise ValueError('Invalid valid-position mask or hidden ratio')
    hidden = torch.zeros_like(mask)
    for index, valid in enumerate(mask):
        positions = valid.nonzero(as_tuple=True)[0]
        length = math.ceil(len(positions) * ratio)
        if not length:
            continue
        rng = torch.Generator(device='cpu').manual_seed(seed + index)
        start = int(torch.randint(len(positions) - length + 1, (1,), generator=rng))
        hidden[index, positions[start:start + length]] = True
    return hidden


def nearest_visible_ids(initial_ids, hidden, mask):
    """Hidden IDs must already be cleared. / 숨긴 ID는 미리 지워야 합니다."""
    if initial_ids.shape != hidden.shape or hidden.shape != mask.shape:
        raise ValueError('Hint shapes differ')
    if hidden.dtype != torch.bool or mask.dtype != torch.bool or (hidden & ~mask).any():
        raise ValueError('Invalid hidden/valid masks')
    if bool(initial_ids[hidden | ~mask].any()):
        raise ValueError('Hidden/padded IDs must be zero, never hidden targets')
    predicted = initial_ids.clone()
    for index in range(len(mask)):
        visible = (mask[index] & ~hidden[index]).nonzero(as_tuple=True)[0]
        wanted = hidden[index].nonzero(as_tuple=True)[0]
        if not len(wanted):
            continue
        if not len(visible):
            raise ValueError('Nearest-copy prediction requires a visible B unit')
        # Copy only public hints; choose left on ties. / 보이는 힌트만 복사하며 동률은 왼쪽을 택합니다.
        right = torch.searchsorted(visible, wanted)
        left = (right - 1).clamp_min(0)
        right = right.clamp_max(len(visible) - 1)
        use_left = (wanted - visible[left]).abs() <= (visible[right] - wanted).abs()
        source = visible[torch.where(use_left, left, right)]
        predicted[index, wanted] = initial_ids[index, source]
    return predicted.masked_fill(~mask, 0)


def frequency_groups(frequencies):
    frequencies = torch.as_tensor(frequencies, dtype=torch.long).cpu()
    if frequencies.ndim != 1 or len(frequencies) < 1 or (frequencies < 0).any() or not frequencies.sum():
        raise ValueError('Need nonnegative training-only unit frequencies')
    seen = sorted((index for index, count in enumerate(frequencies.tolist()) if count),
                  key=lambda index: (-int(frequencies[index]), index))
    edge = math.ceil(len(seen) / 4)
    labels = torch.full_like(frequencies, 3)
    for rank, index in enumerate(seen):
        labels[index] = 0 if rank < edge else 2 if rank >= len(seen) - edge else 1
    return labels


def target_strata(targets, mask, groups):
    if targets.shape != mask.shape or targets.dtype != torch.long or mask.dtype != torch.bool:
        raise ValueError('Target and valid-mask shapes/types differ')
    interior = torch.zeros_like(mask)
    if targets.shape[1] >= 3:
        interior[:, 1:-1] = (mask[:, :-2] & mask[:, 1:-1] & mask[:, 2:] &
            (targets[:, :-2] == targets[:, 1:-1]) & (targets[:, 1:-1] == targets[:, 2:]))
    ordinal = mask.long().cumsum(1) - 1
    prefix = (ordinal < (mask.sum(1).float() * .2).ceil().long()[:, None]) & mask
    types = groups.to(targets.device)[targets]
    return {'all_hidden': mask, 'prefix_20pct_time': prefix, 'remainder_80pct_time': mask & ~prefix,
        'run_interior': interior, 'run_boundary_or_singleton': mask & ~interior,
        **{'frequency_' + name: mask & (types == index) for index, name in
           enumerate(('high', 'middle', 'low', 'unseen_in_training'))}}


def score_predictions(predicted, targets, hidden, strata, logits=None):
    if predicted.shape != targets.shape or targets.shape != hidden.shape:
        raise ValueError('Prediction shapes differ')
    if logits is not None and (logits.shape[:2] != targets.shape or not torch.isfinite(logits).all()):
        raise ValueError('Invalid logits')
    result = {}
    for name, region in strata.items():
        chosen = hidden & region
        frames = int(chosen.sum())
        correct = int((predicted[chosen] == targets[chosen]).sum())
        ce = float(F.cross_entropy(logits[chosen].float(), targets[chosen], reduction='sum')) if logits is not None and frames else None
        result[name] = {'hidden_frames': frames, 'correct_units': correct,
            'hidden_only_accuracy': correct / frames if frames else None,
            'ce_sum': ce, 'hidden_only_ce': ce / frames if ce is not None else None}
    return result


def summarize(examples):
    bins = {}
    for example in examples:
        for trial in example['trials']:
            common = f'{trial["mask_type"]}/{trial["requested_hidden_ratio"]}'
            predictions = {'nearest_copy': trial['nearest_copy']}
            for condition, methods in trial['conditions'].items():
                for method, score in methods.items():
                    predictions[condition + '/' + method] = score
            for name, scores in predictions.items():
                if scores is None:
                    continue
                key = common + '/' + name
                regions = bins.setdefault(key, {})
                for region, score in scores.items():
                    current = regions.setdefault(region, {'hidden_frames': 0, 'correct_units': 0,
                        'ce_sum': 0., 'ce_frames': 0, 'measurements': 0, 'conversation_ids': set()})
                    current['hidden_frames'] += score['hidden_frames']
                    current['correct_units'] += score['correct_units']
                    if score['ce_sum'] is not None:
                        current['ce_sum'] += score['ce_sum']
                        current['ce_frames'] += score['hidden_frames']
                    current['measurements'] += 1
                    current['conversation_ids'].add(example['conversation_id'])
    for regions in bins.values():
        for score in regions.values():
            frames = score['hidden_frames']
            score['hidden_only_accuracy'] = score['correct_units'] / frames if frames else None
            score['hidden_only_ce'] = score['ce_sum'] / score['ce_frames'] if score['ce_frames'] else None
            score['conversations'] = len(score.pop('conversation_ids'))
    return bins


def selected_indices(data, selection, split, count=None):
    wanted = selection[split]
    mapping = {row['path']: index for index, row in enumerate(data.records)}
    if not wanted or len(wanted) != len(set(wanted)) or any(path not in mapping for path in wanted):
        raise ValueError('Selection contains missing, duplicate, or wrong-split records')
    chosen = [mapping[path] for path in wanted[:count]]
    ids = [data.records[index]['conversation_id'] for index in chosen]
    if len(ids) != len(set(ids)):
        raise ValueError('Repeated conversations in selected examples')
    return chosen


@torch.inference_mode()
def fit_frequency_counts(model, manifest, selection, device):
    data = QualitySpeechDataset(manifest, model.config, 'train')
    chosen = selected_indices(data, selection, 'train')
    frequency = torch.zeros(len(model.semantic_planner.codebook.centers), device=device, dtype=torch.long)
    for position, index in enumerate(chosen, 1):
        row = data.records[index]
        # Frequency fitting reads only selected training B features. / 빈도는 선택한 학습 B 특징으로만 계산합니다.
        with np.load(Path(manifest).parent / row['path'], allow_pickle=False) as stored:
            semantic = torch.from_numpy(stored['semantic'].astype(np.float32)).to(device)
        if semantic.ndim != 2 or semantic.shape[1] != 768 or not len(semantic) or not torch.isfinite(semantic).all():
            raise ValueError('Invalid B semantic features while fitting frequencies')
        ids = model.semantic_planner.codebook.encode((semantic - model.semantic_mean) / model.semantic_std)
        frequency += torch.bincount(ids, minlength=len(frequency))
        if position % 256 == 0:
            print(json.dumps({'frequency_training_conversations': position, 'total': len(chosen)}), flush=True)
    return {'fit_split': 'train', 'training_conversations': len(chosen), 'frames': int(frequency.sum()),
        'counts': frequency.cpu().tolist(), 'group_ids': frequency_groups(frequency.cpu()).tolist(),
        'groups': {'0': 'highest-frequency quarter of seen unit types', '1': 'middle seen unit types',
                   '2': 'lowest-frequency quarter of seen unit types', '3': 'unseen in selected training'},
        'tie_break': 'ascending unit ID', 'interface': interface_identity(model)}


def checkpoint_arguments(values):
    result = dict(DEFAULT_CHECKPOINTS) if not values else {}
    for value in values or []:
        name, separator, path = value.partition('=')
        if not separator or not name or any(char not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for char in name) or not path:
            raise ValueError('Use --checkpoint name=path with a simple unique label')
        if name in result or name in ('identity', 'summary', 'training_unit_frequencies'):
            raise ValueError('Repeated checkpoint label')
        result[name] = path
    return {name: Path(path) for name, path in result.items()}


@torch.inference_mode()
def evaluate_model(model, data, chosen, frequencies, args, label):
    examples = []
    groups = torch.tensor(frequencies['group_ids'], device=args.device, dtype=torch.long)
    for position, index in enumerate(chosen, 1):
        row = data.records[index]
        batch = move_batch(collate_quality([data[index]]), torch.device(args.device))
        encoded = model.encode_batch(person_a_only(batch))
        production = model.planner_inputs(encoded)
        style, _ = model.embeddings(batch['style_id'], batch['speaker_id'], 1)
        # This exactly matches the saved B-only prior's null condition. / 저장된 B 사전학습의 무조건 입력과 정확히 맞춥니다.
        null = {'context': style.new_zeros(1, 1, 512),
                'context_mask': torch.ones(1, 1, device=args.device, dtype=torch.bool),
                'affect': style.new_zeros(1, 1, 6)}
        contexts = {'null_prior': (null, torch.zeros_like(style)), 'production_a': (production, style)}
        mask = mask_from_lengths(batch['semantic_len'])
        if not torch.equal(batch['semantic_len'], counts(batch['duration'], model.config.semantic_hz)):
            raise ValueError('Oracle B duration and target-unit length differ')
        target = model.semantic_planner.codebook.encode((batch['semantic'] - model.semantic_mean) / model.semantic_std)
        strata = target_strata(target, mask, groups)
        trials = []
        for kind in MASK_TYPES:
            for ratio in args.ratios:
                for seed in args.mask_seeds:
                    hidden = gap_mask(mask, ratio, seed, kind)
                    visible = mask & ~hidden
                    initial_ids = target.masked_fill(hidden | ~mask, 0)
                    copied = nearest_visible_ids(initial_ids, hidden, mask) if visible.any() else None
                    trial = {**mask_metadata(hidden, mask, ratio, seed), 'mask_type': kind,
                        'nearest_copy': score_predictions(copied, target, hidden, strata) if copied is not None else None,
                        'copy_unavailable_without_visible_hints': copied is None, 'conditions': {}}
                    for condition, (memory, conditional_style) in contexts.items():
                        logits = model.semantic_planner.logits(initial_ids, hidden, mask,
                            style=conditional_style, **memory)
                        trial['conditions'][condition] = {'first_pass': score_predictions(logits.argmax(-1),
                            target, hidden, strata, logits)}
                        if args.sample_steps:
                            generated, _ = sample_units(model.semantic_planner, mask, style=conditional_style,
                                steps=args.sample_steps, initial_ids=initial_ids, initial_hidden=hidden, **memory)
                            if not torch.equal(generated[visible], target[visible]):
                                raise ValueError('Sampler changed a supplied B-unit hint')
                            trial['conditions'][condition]['sampled'] = score_predictions(generated, target, hidden, strata)
                    trials.append(trial)
        examples.append({'path': row['path'], 'conversation_id': row['conversation_id'],
            'frames': int(mask.sum()), 'target_unit_sha256': tensor_hash(target[mask]),
            'trials': trials})
        if position % 8 == 0:
            print(json.dumps({'checkpoint': label, 'evaluated': position, 'total': len(chosen)}), flush=True)
    return examples


def main():
    parser = argparse.ArgumentParser()
    for name in ('manifest', 'selection', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--checkpoint', action='append', help='Repeat name=path; defaults to saved prior, conditional and current fused')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--split', choices=('train', 'val'), default='val')
    parser.add_argument('--count', type=int, default=128)
    parser.add_argument('--ratios', type=float, nargs='+', default=[.25, .5, .75, 1.])
    parser.add_argument('--mask-seeds', type=int, nargs='+', default=[42, 43])
    parser.add_argument('--sample-steps', type=int, default=0, help='Optional greedy refinement in addition to first-pass metrics')
    args = parser.parse_args()
    if (args.count < 1 or args.sample_steps < 0 or any(not 0 < ratio <= 1 for ratio in args.ratios) or
            len(args.ratios) != len(set(args.ratios)) or len(args.mask_seeds) != len(set(args.mask_seeds))):
        parser.error('Use positive count, unique valid ratios/seeds and nonnegative sample steps')
    checkpoints = checkpoint_arguments(args.checkpoint)
    selection = json.loads(args.selection.read_text())
    manifest_hash = file_hash(args.manifest)
    if selection.get('manifest_sha256') != manifest_hash:
        raise ValueError('Selection does not match the manifest')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    args.output.mkdir(parents=True, exist_ok=True)
    identity = {'version': 1, 'manifest_sha256': manifest_hash, 'selection_sha256': file_hash(args.selection),
        'checkpoints': {name: {'path': str(path), 'sha256': file_hash(path)} for name, path in checkpoints.items()},
        'split': args.split, 'count_limit': args.count, 'ratios': args.ratios, 'mask_seeds': args.mask_seeds,
        'sample_steps': args.sample_steps, 'source_sha256': {name: file_hash(Path(__file__).resolve().parents[1] / name)
            for name in ('scripts/check_planner_learning_history.py', 'scripts/check_planner_hints.py',
                         'scripts/planner_sampling.py', 'model/full_speech/units.py', 'model/full_speech/quality.py',
                         'dataset/quality_speech_dataset.py', 'model/full_speech/tensor_ops.py')}}
    identity_path = args.output / 'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Existing experiment identity differs; use another output directory')
    atomic_json(identity_path, identity)
    frequency_path = args.output / 'training_unit_frequencies.json'
    frequencies = json.loads(frequency_path.read_text()) if frequency_path.exists() else None
    reports = {}
    for label, checkpoint in checkpoints.items():
        model, payload = UnitSpeechSystem.from_checkpoint(checkpoint)
        model.to(args.device).requires_grad_(False).eval()
        if frequencies is None:
            frequencies = fit_frequency_counts(model, args.manifest, selection, args.device)
            frequencies['manifest_sha256'] = manifest_hash
            frequencies['selection_sha256'] = identity['selection_sha256']
            atomic_json(frequency_path, frequencies)
        if (frequencies['interface'] != interface_identity(model) or
                frequencies['manifest_sha256'] != manifest_hash or
                frequencies['selection_sha256'] != identity['selection_sha256']):
            raise ValueError('Checkpoints do not share the frequency cache unit interface or selection')
        output = args.output / (label + '.json')
        if output.exists():
            report = json.loads(output.read_text())
            if report.get('identity') != identity or report.get('checkpoint_label') != label:
                raise ValueError('Saved checkpoint report provenance differs')
        else:
            data = QualitySpeechDataset(args.manifest, model.config, args.split)
            chosen = selected_indices(data, selection, args.split, args.count)
            examples = evaluate_model(model, data, chosen, frequencies, args, label)
            if file_hash(checkpoint) != identity['checkpoints'][label]['sha256']:
                raise ValueError('Checkpoint changed while being evaluated')
            report = {'identity': identity, 'checkpoint_label': label, 'checkpoint_step': payload.get('recovery_step'),
                'planner_memory_mode': model.config.planner_memory_mode,
                'checkpoint_trained_as_prior': bool(payload.get('metadata', {}).get('recovery_recipe', {}).get('unit_prior')),
                'frequency_counts_sha256': file_hash(frequency_path), 'count': len(examples),
                'oracle_B_length': True, 'A_encoder_inputs': 'person_a_only', 'normal_inference': False,
                'production_weights_modified': False,
                'conditions': {'null_prior': 'Single zero context token, zero affect and zero style; exact B-only prior recipe.',
                    'production_a': 'Checkpoint-selected A context and affect plus requested style; unseen conditioning for the B-only prior.'},
                'mask_definitions': {'random': 'Nested CPU randperm(seed) with ceil(ratio*N) hidden positions.',
                    'contiguous': 'One span of ceil(ratio*N) consecutive valid positions, uniformly seeded start; not nested across ratios.'},
                'stratum_definitions': {'run_interior': 'Valid frame has both adjacent valid neighbors with the same GT unit ID.',
                    'run_boundary_or_singleton': 'All other valid frames, including sequence endpoints.',
                    'prefix_20pct_time': 'First ceil(.2*N) frames; time proxy, not word alignment.',
                    'frequency': 'Ranks by counts in selected training B units only; high/low quarters of seen types, ID breaks ties.'},
                'limitations': ['Every gap task supplies true B length and visible B hints; this is not A-only response generation.',
                    'Null conditioning matches prior training but is outside ordinary conditional fine-tuning; production conditioning is untrained for the B-only prior.',
                    'Fully hidden B sequences are outside the B-only prior training mask range of 20–80%.',
                    'A prior-versus-conditional score difference is descriptive and is not alone a causal forgetting estimate.',
                    'Run/frequency strata use hidden targets for scoring only, never for predictions.',
                    'Repeated mask seeds are dependent observations of the same conversations.',
                    'Nearest-copy needs a visible hint and is undefined when all B units are hidden.',
                    'Random gaps favor nearby repeated units; contiguous gaps test a different, harder problem.',
                    'No response relevance, empathy, or human listening score is measured.'],
                'summary': summarize(examples), 'examples': examples}
            atomic_json(output, report)
        reports[label] = {key: value for key, value in report.items() if key != 'examples'}
        del model
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
    atomic_json(args.output / 'summary.json', {'identity': identity, 'checkpoints': reports,
        'automatic_winner_selected': False, 'next_action': 'Review matched conditions, gap types and training/validation separately.'})
    print(json.dumps({'completed_checkpoints': list(reports), 'output': str(args.output / 'summary.json')}), flush=True)


if __name__ == '__main__':
    main()
