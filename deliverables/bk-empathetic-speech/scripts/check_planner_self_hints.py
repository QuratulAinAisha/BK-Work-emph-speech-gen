"""Disjoint self-generated B-hint diagnostic. / 서로 분리한 B 생성 힌트 진단."""

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F

from dataset.quality_speech_dataset import QualitySpeechDataset, collate_quality
from model.full_speech.units import UnitSpeechSystem
from prepare_quality import atomic_json
from scripts.check_planner_hints import chosen_indices, prepare_pair
from scripts.check_planner_learning_history import file_hash, gap_mask, interface_identity, tensor_hash
from scripts.evaluate_planner_controls import next_different_conversation
from train_full import move_batch


CONDITIONS = ('correct_a', 'shuffled_a', 'null_prior')
MASK_TYPES = ('random', 'contiguous')
LIMITATIONS = [
    'Oracle B length and partial correct B units are supplied. This is not A-only response generation.',
    'H remains hidden in all three forwards; only the separate C region is predicted and later made visible.',
    'The first prediction sees roughly half of the correct B units, unlike all-hidden production startup.',
    'Random H and contiguous H test different tasks. C is random within the positions outside H in both.',
    'H accuracy is reference reconstruction, not conversational relevance, empathy, or audio quality.',
    'The C prediction is produced separately for each A condition; the A comparison includes that upstream change.',
    'Null conditioning matches B-only prior training; production A conditioning may be untrained for a prior checkpoint.',
    'Mask seeds repeat each conversation and are not independent examples.',
    'No model weights are updated and no automatic winner is selected.',
]


def disjoint_masks(mask, seed, kind):
    if mask.ndim != 2 or mask.dtype != torch.bool or (mask.sum(1) < 4).any():
        raise ValueError('Need boolean masks with at least four valid frames per sequence')
    expected = torch.arange(mask.shape[1], device=mask.device)[None] < mask.sum(1)[:, None]
    if not torch.equal(mask, expected):
        raise ValueError('Valid frames must form an unpadded prefix')
    held_out = gap_mask(mask, .25, seed, kind)
    corruption = torch.zeros_like(mask)
    for index, row in enumerate(mask):
        candidates = (row & ~held_out[index]).nonzero(as_tuple=True)[0]
        count = math.ceil(int(row.sum()) * .25)
        generator = torch.Generator(device='cpu').manual_seed(seed + 700001 + index)
        order = torch.randperm(len(candidates), generator=generator).to(mask.device)
        corruption[index, candidates[order[:count]]] = True
    return held_out, corruption


def _validate_masks(targets, mask, held_out, corruption):
    if targets.dtype != torch.long or targets.shape != mask.shape or mask.ndim != 2:
        raise ValueError('Targets must be int64 [B,T] with matching valid masks')
    if any(value.shape != mask.shape or value.dtype != torch.bool for value in (mask, held_out, corruption)):
        raise ValueError('All masks must be boolean and match the targets')
    if (held_out & corruption).any() or ((held_out | corruption) & ~mask).any():
        raise ValueError('H and C must be disjoint valid positions')
    if any(not value.any(1).all() for value in (held_out, corruption, mask & ~(held_out | corruption))):
        raise ValueError('Each sequence needs H, C, and remaining correct visible hints')


def _score(logits, targets, region):
    frames = int(region.sum())
    correct = int((logits.argmax(-1)[region] == targets[region]).sum())
    ce = float(F.cross_entropy(logits[region].float(), targets[region], reduction='sum'))
    return {'frames': frames, 'correct_units': correct, 'accuracy': correct / frames,
            'ce_sum': ce, 'ce': ce / frames}


@torch.no_grad()
def evaluate_self_hints(planner, targets, mask, held_out, corruption, memory, style):
    """Targets at H never enter any planner call. / H의 정답은 계획기에 전혀 전달하지 않습니다."""
    _validate_masks(targets, mask, held_out, corruption)
    if planner.training:
        raise ValueError('Self-hint diagnostics require eval mode')
    if (targets[mask] < 0).any() or (targets[mask] >= len(planner.codebook.centers)).any():
        raise ValueError('Target unit IDs outside codebook')

    def predict(ids, hidden):
        if bool(ids[hidden | ~mask].any()):
            raise AssertionError('Hidden and padded IDs must be cleared before prediction')
        logits = planner.logits(ids, hidden, mask, style=style, **memory).float()
        if logits.shape != (*targets.shape, len(planner.codebook.centers)) or not torch.isfinite(logits).all():
            raise ValueError('Planner returned invalid logits')
        return logits

    # C is predicted while H remains unavailable. / H를 계속 숨긴 채 C를 예측합니다.
    first_hidden = held_out | corruption
    first_ids = targets.masked_fill(first_hidden | ~mask, 0)
    first_logits = predict(first_ids, first_hidden)
    predicted_c = first_logits.argmax(-1).detach()
    clean_ids = targets.masked_fill(held_out | ~mask, 0)
    self_ids = torch.where(corruption, predicted_c, clean_ids).masked_fill(held_out | ~mask, 0)
    clean_logits = predict(clean_ids, held_out)
    self_logits = predict(self_ids, held_out)
    unchanged = bool(torch.equal(clean_ids, self_ids))
    if unchanged and not torch.equal(clean_logits, self_logits):
        raise RuntimeError('Identical visible hints produced different deterministic logits')
    clean_score, self_score = _score(clean_logits, targets, held_out), _score(self_logits, targets, held_out)
    c_score = _score(first_logits, targets, corruption)
    # Actual IDs and hashes make every diagnostic prediction auditable. / 실제 ID와 해시로 진단 예측을 확인할 수 있습니다.
    return {'first_C': c_score, 'initial_H_with_H_and_C_hidden': _score(first_logits, targets, held_out),
        'H_with_true_C': clean_score, 'H_with_predicted_C': self_score,
        'H_accuracy_drop': clean_score['accuracy'] - self_score['accuracy'],
        'H_ce_increase': self_score['ce'] - clean_score['ce'],
        'H_prediction_changed_fraction': float((clean_logits.argmax(-1)[held_out] !=
                                               self_logits.argmax(-1)[held_out]).float().mean()),
        'C_error_count': c_score['frames'] - c_score['correct_units'],
        'all_C_predictions_correct': unchanged,
        'first_input_sha256': tensor_hash(first_ids), 'true_C_input_sha256': tensor_hash(clean_ids),
        'predicted_C_input_sha256': tensor_hash(self_ids),
        'predicted_C_ids': predicted_c[corruption].cpu().tolist(),
        'predicted_H_with_true_C_ids': clean_logits.argmax(-1)[held_out].cpu().tolist(),
        'predicted_H_with_predicted_C_ids': self_logits.argmax(-1)[held_out].cpu().tolist(),
        'grad_enabled_during_evaluation': torch.is_grad_enabled()}


def summarize(examples):
    grouped = {}
    for row in examples:
        for trial in row['trials']:
            for condition, value in trial['conditions'].items():
                grouped.setdefault((trial['mask_type'], condition), []).append((row['conversation_id'], value))
    output = {}
    for (kind, condition), measurements in grouped.items():
        groups = {}
        for conversation, value in measurements:
            groups.setdefault(conversation, []).append(value)
        per_case = []
        for conversation, values in groups.items():
            h_frames = sum(value['H_with_true_C']['frames'] for value in values)
            c_frames = sum(value['first_C']['frames'] for value in values)
            clean = sum(value['H_with_true_C']['correct_units'] for value in values) / h_frames
            predicted = sum(value['H_with_predicted_C']['correct_units'] for value in values) / h_frames
            ce_increase = sum(value['H_with_predicted_C']['ce_sum'] - value['H_with_true_C']['ce_sum']
                              for value in values) / h_frames
            per_case.append({'conversation_id': conversation, 'measurements': len(values),
                'H_frames': h_frames, 'C_frames': c_frames, 'H_true_C_accuracy': clean,
                'H_predicted_C_accuracy': predicted, 'H_accuracy_drop': clean - predicted,
                'H_ce_increase': ce_increase,
                'C_accuracy': sum(value['first_C']['correct_units'] for value in values) / c_frames})
        output[kind + '/' + condition] = {'measurements': len(measurements), 'conversations': len(groups),
            'mean_conversation_C_accuracy': sum(value['C_accuracy'] for value in per_case) / len(per_case),
            'mean_conversation_H_true_C_accuracy': sum(value['H_true_C_accuracy'] for value in per_case) / len(per_case),
            'mean_conversation_H_predicted_C_accuracy': sum(value['H_predicted_C_accuracy'] for value in per_case) / len(per_case),
            'mean_conversation_H_accuracy_drop': sum(value['H_accuracy_drop'] for value in per_case) / len(per_case),
            'mean_conversation_H_ce_increase': sum(value['H_ce_increase'] for value in per_case) / len(per_case),
            'all_C_correct_measurements': sum(value['all_C_predictions_correct'] for _, value in measurements),
            'per_conversation': per_case}
    return output


def main():
    parser = argparse.ArgumentParser()
    for name in ('checkpoint', 'manifest', 'selection', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--split', choices=('train', 'val'), default='val')
    parser.add_argument('--count', type=int, default=32)
    parser.add_argument('--mask-seeds', type=int, nargs='+', default=[42, 43])
    parser.add_argument('--conditions', choices=CONDITIONS, nargs='+', default=list(CONDITIONS))
    args = parser.parse_args()
    if args.count < 1 or len(set(args.mask_seeds)) != len(args.mask_seeds) or len(set(args.conditions)) != len(args.conditions):
        parser.error('Use a positive count and unique seeds/conditions')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    source_root = Path(__file__).resolve().parents[1]
    source_names = ('scripts/check_planner_self_hints.py', 'scripts/check_planner_hints.py',
        'scripts/check_planner_learning_history.py', 'scripts/evaluate_planner_controls.py',
        'model/full_speech/units.py', 'model/full_speech/quality.py', 'model/full_speech/dit.py',
        'model/full_speech/tensor_ops.py', 'dataset/quality_speech_dataset.py')
    identity = {'version': 1, 'checkpoint': str(args.checkpoint), 'checkpoint_sha256': file_hash(args.checkpoint),
        'manifest_sha256': file_hash(args.manifest), 'selection_sha256': file_hash(args.selection),
        'source_sha256': {name: file_hash(source_root / name) for name in source_names},
        'split': args.split, 'count_limit': args.count, 'mask_seeds': args.mask_seeds,
        'conditions': args.conditions, 'H_fraction': .25, 'C_fraction': .25,
        'mask_types': list(MASK_TYPES), 'device': args.device, 'torch_version': str(torch.__version__)}
    args.output.mkdir(parents=True, exist_ok=True)
    identity_path, report_path = args.output / 'identity.json', args.output / 'units_report.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Existing experiment identity differs; use a new output directory')
    if report_path.exists():
        if json.loads(report_path.read_text()).get('identity') != identity:
            raise ValueError('Saved report identity differs')
        print(json.dumps({'already_completed': str(report_path)}))
        return
    atomic_json(identity_path, identity)
    selection = json.loads(args.selection.read_text())
    model, payload = UnitSpeechSystem.from_checkpoint(args.checkpoint)
    model.to(args.device).requires_grad_(False).eval()
    data = QualitySpeechDataset(args.manifest, model.config, args.split)
    chosen = chosen_indices(data, selection, identity['manifest_sha256'], args.split)
    records = [data.records[index] for index in chosen]
    donors = next_different_conversation(records)
    examples = []
    for position, index in enumerate(chosen[:args.count]):
        row = data.records[index]
        batch = move_batch(collate_quality([data[index]]), torch.device(args.device))
        donor = move_batch(collate_quality([data[chosen[donors[position]]]]), torch.device(args.device))
        targets, mask, contexts, style = prepare_pair(model, batch, donor)
        null_mask = torch.ones(1, 1, dtype=torch.bool, device=targets.device)
        contexts['null_prior'] = {'context': style.new_zeros(1, 1, 512), 'context_mask': null_mask,
                                  'affect': style.new_zeros(1, 1, 6), 'affect_mask': null_mask}
        trials = []
        for kind in MASK_TYPES:
            for seed in args.mask_seeds:
                held_out, corruption = disjoint_masks(mask, seed, kind)
                trial = {'mask_type': kind, 'mask_seed': seed,
                    'H_frames': int(held_out.sum()), 'C_frames': int(corruption.sum()),
                    'initial_visible_frames': int((mask & ~(held_out | corruption)).sum()),
                    'H_actual_fraction': float(held_out.sum() / mask.sum()),
                    'C_actual_fraction': float(corruption.sum() / mask.sum()),
                    'H_mask_sha256': tensor_hash(held_out), 'C_mask_sha256': tensor_hash(corruption),
                    'H_positions': held_out[0].nonzero(as_tuple=True)[0].cpu().tolist(),
                    'C_positions': corruption[0].nonzero(as_tuple=True)[0].cpu().tolist(),
                    'target_H_ids': targets[held_out].cpu().tolist(), 'target_C_ids': targets[corruption].cpu().tolist(),
                    'conditions': {}}
                for condition in args.conditions:
                    condition_style = torch.zeros_like(style) if condition == 'null_prior' else style
                    trial['conditions'][condition] = evaluate_self_hints(model.semantic_planner, targets, mask,
                        held_out, corruption, contexts[condition], condition_style)
                trials.append(trial)
        examples.append({'path': row['path'], 'conversation_id': row['conversation_id'],
            'shuffled_path': records[donors[position]]['path'],
            'shuffled_conversation_id': records[donors[position]]['conversation_id'],
            'frames': int(mask.sum()), 'target_units_sha256': tensor_hash(targets[mask]), 'trials': trials})
        if len(examples) % 8 == 0:
            print(json.dumps({'completed': len(examples), 'total': min(args.count, len(chosen))}), flush=True)
    if file_hash(args.checkpoint) != identity['checkpoint_sha256']:
        raise ValueError('Checkpoint changed during evaluation')
    report = {'identity': identity, 'checkpoint_step': payload.get('recovery_step'),
        'checkpoint_trained_as_prior': bool(payload.get('metadata', {}).get('recovery_recipe', {}).get('unit_prior')),
        'interface': interface_identity(model), 'count': len(examples), 'oracle_B_length': True,
        'partial_correct_B_hints': True, 'A_only_generation': False, 'production_weights_modified': False,
        'planner_memory_mode': model.config.planner_memory_mode, 'automatic_winner_selected': False,
        'H_stays_hidden_in_all_forwards': True, 'C_prediction': 'single-pass argmax, detached',
        'summary': summarize(examples), 'limitations': LIMITATIONS, 'examples': examples}
    atomic_json(report_path, report)
    print(json.dumps({'completed_report': str(report_path), 'count': len(examples)}), flush=True)


if __name__ == '__main__':
    main()
