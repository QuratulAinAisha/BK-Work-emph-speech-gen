"""Verify planner-only updates and checkpoint finiteness. / 계획기 갱신 범위와 체크포인트 유한성을 검증합니다."""

import argparse
from collections import Counter
import json
from pathlib import Path

import torch


UNIT_ARCHITECTURE = 'llm_free_speech_units_v1'
CONTINUOUS_ARCHITECTURE = 'llm_free_speech_quality_v2'
CODEBOOK_KEY = 'semantic_planner.codebook.centers'


def tensor_audit(value, path):
    """Inspect state, optimizer and RNG tensors without changing them. / 상태·옵티마이저·난수 텐서를 변경 없이 검사합니다."""
    total, nonfinite = 0, []
    if isinstance(value, torch.Tensor):
        total = 1
        if not bool(torch.isfinite(value).all()):
            nonfinite.append(path)
    elif isinstance(value, dict):
        for key, child in value.items():
            count, failures = tensor_audit(child, f'{path}.{key}')
            total += count
            nonfinite.extend(failures)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            count, failures = tensor_audit(child, f'{path}[{index}]')
            total += count
            nonfinite.extend(failures)
    return total, nonfinite


def audit_checkpoints(initial, candidate, unit_prior=False, planner_only=False):
    errors = []
    initial_architecture = initial.get('architecture')
    target_architecture = candidate.get('architecture')
    if initial_architecture not in (CONTINUOUS_ARCHITECTURE, UNIT_ARCHITECTURE):
        errors.append('Unsupported initialization architecture')
    if target_architecture != UNIT_ARCHITECTURE:
        errors.append('Planner checkpoint must use the discrete-unit architecture')
    fresh_planner = initial_architecture == CONTINUOUS_ARCHITECTURE
    allowed = {'semantic_planner'} if unit_prior or planner_only else {'semantic_planner', 'length_predictor'}
    source = initial.get('state_dict', {})
    target = candidate.get('state_dict', {})
    if not isinstance(source, dict) or not isinstance(target, dict) or not source or not target:
        raise ValueError('Both checkpoints need nonempty tensor state dictionaries')
    invalid = [name for name, state in [('initialize', source), ('checkpoint', target)]
               if any(not isinstance(value, torch.Tensor) for value in state.values())]
    if invalid:
        raise ValueError('Non-tensor state dictionary values: ' + ', '.join(invalid))
    removed = sorted(set(source) - set(target))
    added = sorted(set(target) - set(source))
    # Only replacement of the continuous planner permits key changes. / 연속 계획기 교체 때만 키 변경을 허용합니다.
    unexpected_removed = [key for key in removed if not (fresh_planner and key.startswith('semantic_planner.'))]
    unexpected_added = [key for key in added if not (fresh_planner and key.startswith('semantic_planner.'))]
    if unexpected_removed:
        errors.append('Unexpected missing state keys')
    if unexpected_added:
        errors.append('Unexpected added state keys')
    changed, frozen_changed, shape_changes, dtype_changes = [], [], [], []
    for key in sorted(set(source) & set(target)):
        before, after = source[key], target[key]
        if before.shape != after.shape:
            shape_changes.append(key)
        if before.dtype != after.dtype:
            dtype_changes.append(key)
        if before.dtype != after.dtype or not torch.equal(before, after):
            changed.append(key)
            if key.split('.', 1)[0] not in allowed:
                frozen_changed.append(key)
    if frozen_changed:
        errors.append('Frozen shared tensors changed')
    if any(not (fresh_planner and key.startswith('semantic_planner.')) for key in shape_changes):
        errors.append('Unexpected shared tensor shape changes')
    if dtype_changes:
        errors.append('Shared tensor dtypes changed')
    codebook_unchanged = None
    if CODEBOOK_KEY not in target:
        errors.append('Discrete checkpoint is missing codebook centers')
    elif initial_architecture == UNIT_ARCHITECTURE:
        codebook_unchanged = (CODEBOOK_KEY in source and source[CODEBOOK_KEY].dtype == target[CODEBOOK_KEY].dtype
                              and torch.equal(source[CODEBOOK_KEY], target[CODEBOOK_KEY]))
        if not codebook_unchanged:
            errors.append('Unit codebook centers changed or are absent in initialization')
    initial_tensors, initial_nonfinite = tensor_audit(initial, 'initialize')
    checkpoint_tensors, checkpoint_nonfinite = tensor_audit(candidate, 'checkpoint')
    nonfinite = initial_nonfinite + checkpoint_nonfinite
    if nonfinite:
        errors.append('Non-finite tensors in initialization or checkpoint')
    all_changes = sorted(set(changed) | set(added) | set(removed))
    return {
        'passed': not errors, 'errors': errors,
        'initialize_architecture': initial_architecture, 'checkpoint_architecture': target_architecture,
        'unit_prior': unit_prior, 'planner_only': planner_only, 'fresh_planner_replacement': fresh_planner,
        'allowed_changed_prefixes': sorted(allowed),
        'changed_prefix_counts': dict(sorted(Counter(key.split('.', 1)[0] for key in all_changes).items())),
        'changed_shared_keys': changed, 'added_keys': added, 'removed_keys': removed,
        'unexpected_added_keys': unexpected_added, 'unexpected_removed_keys': unexpected_removed,
        'frozen_changed_keys': frozen_changed, 'shape_changed_keys': shape_changes,
        'dtype_changed_keys': dtype_changes, 'codebook_unchanged': codebook_unchanged,
        'all_tensors_finite': not nonfinite, 'nonfinite_tensor_paths': nonfinite,
        'initialize_tensor_count': initial_tensors, 'checkpoint_tensor_count': checkpoint_tensors,
        'initialize_state_tensor_count': len(source), 'checkpoint_state_tensor_count': len(target),
        'unchanged_shared_tensor_count': len(set(source) & set(target)) - len(changed),
        'limitation': 'An integrity pass establishes the update boundary, not response relevance or audio quality.',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--initialize', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--unit-prior', action='store_true')
    parser.add_argument('--planner-only', action='store_true', help='Also require a bitwise-frozen duration predictor')
    args = parser.parse_args()
    torch.set_num_threads(2)
    initial = torch.load(args.initialize, map_location='cpu', weights_only=True)
    candidate = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    report = audit_checkpoints(initial, candidate, args.unit_prior, args.planner_only)
    report.update(initialize=str(args.initialize), checkpoint=str(args.checkpoint),
                  checkpoint_step=candidate.get('recovery_step'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    temporary.replace(args.output)
    print(json.dumps(report), flush=True)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
