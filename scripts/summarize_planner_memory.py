"""Summarize fixed planner-memory endpoints without selection. / 선택 없이 고정 계획기 결과를 요약합니다."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ARMS = ('fused', 'resampled_speech', 'native_speech')
COMPARISONS = (('native_speech', 'resampled_speech', 'primary'),
               ('native_speech', 'fused', 'secondary'))
STEP = 320
TAIL = 'remainder_80pct_time'
ATTESTATION_CHECKS = ('model_tensors_bitwise_equal', 'configs_equal', 'architectures_equal',
                      'recovery_recipes_equal', 'both_steps_are_320', 'report_names_verified_alias')


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint_identity(fixed, alias, arm):
    import torch
    if arm not in ARMS or fixed.get('recovery_step') != STEP or alias.get('recovery_step') != STEP:
        raise ValueError('Both checkpoint payloads must be the fixed step-320 endpoint')
    if fixed.get('architecture') != 'llm_free_speech_units_v1' or alias.get('architecture') != fixed['architecture']:
        raise ValueError('Alias architecture differs from the discrete-unit endpoint')
    if fixed.get('config') != alias.get('config') or fixed['config'].get('planner_memory_mode') != arm:
        raise ValueError('Alias config or memory mode differs from the endpoint')
    fixed_recipe = fixed.get('metadata', {}).get('recovery_recipe')
    alias_recipe = alias.get('metadata', {}).get('recovery_recipe')
    if not isinstance(fixed_recipe, dict) or fixed_recipe != alias_recipe:
        raise ValueError('Alias training recipe differs from the endpoint')
    first, second = fixed.get('state_dict'), alias.get('state_dict')
    if not isinstance(first, dict) or not first or not isinstance(second, dict) or set(first) != set(second):
        raise ValueError('Alias model tensor keys differ from the endpoint')
    for key in first:
        a, b = first[key], second[key]
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            raise ValueError('Model state contains non-tensors')
        if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            raise ValueError(f'Alias model tensor differs from the endpoint: {key}')
        if not torch.isfinite(a).all():
            raise ValueError(f'Non-finite endpoint model tensor: {key}')
    return len(first)


def attest_audio_alias(root, arm):
    import torch
    if arm not in ARMS:
        raise ValueError('Unknown memory arm')
    root = Path(root)
    folder = root / arm
    fixed_path, alias_path = folder / 'step_0320.pt', folder / 'last.pt'
    report_path = folder / 'audio_step_0320/report.json'
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    if (report.get('recovery_step') != STEP or report.get('planner_memory_mode') != arm or
            report.get('split') != 'val' or Path(report['checkpoint']).resolve() != alias_path.resolve()):
        raise ValueError('Audio report does not name this arm\'s step-320 last.pt alias')
    fixed_hash, alias_hash = file_sha256(fixed_path), file_sha256(alias_path)
    torch.set_num_threads(2)
    fixed = torch.load(fixed_path, map_location='cpu', weights_only=True)
    alias = torch.load(alias_path, map_location='cpu', weights_only=True)
    count = verify_checkpoint_identity(fixed, alias, arm)
    # Bind the report to verified endpoint tensors without regenerating audio. / 음성을 다시 만들지 않고 보고서를 검증한 가중치에 연결합니다.
    if (file_sha256(fixed_path) != fixed_hash or file_sha256(alias_path) != alias_hash or
            report_path.read_bytes() != report_bytes):
        raise ValueError('Checkpoint or report changed during alias verification')
    result = {'version': 1, 'arm': arm, 'endpoint_step': STEP,
        'report_relative_path': f'{arm}/audio_step_0320/report.json',
        'source_report_sha256': hashlib.sha256(report_bytes).hexdigest(),
        'reported_checkpoint': report['checkpoint'],
        'fixed_checkpoint': {'path': str(fixed_path), 'sha256': fixed_hash},
        'alias_checkpoint': {'path': str(alias_path), 'sha256': alias_hash},
        'verified_model_tensor_count': count,
        'checks': {name: True for name in ATTESTATION_CHECKS},
        'verification_method': 'CPU torch.equal for every model tensor; exact architecture/config/recipe/step checks',
        'limitation': 'Attests the current alias and fixed endpoint plus the recorded report metadata; does not independently reproduce audio generation.'}
    output = folder / 'endpoint_identity.json'
    if output.exists():
        if json.loads(output.read_text(encoding='utf-8')) != result:
            raise ValueError('Refusing to replace a different endpoint attestation')
    else:
        temporary = output.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        temporary.replace(output)
    return result


def validate_alias_attestation(report, arm, attestation, report_sha256):
    if not isinstance(attestation, dict) or not report_sha256:
        raise ValueError('last.pt audio needs an endpoint identity attestation bound to its report hash')
    if (attestation.get('version') != 1 or attestation.get('arm') != arm or
            attestation.get('endpoint_step') != STEP or
            attestation.get('report_relative_path') != f'{arm}/audio_step_0320/report.json' or
            attestation.get('source_report_sha256') != report_sha256 or
            attestation.get('reported_checkpoint') != report['checkpoint']):
        raise ValueError('Endpoint alias attestation does not match this report')
    checks = attestation.get('checks', {})
    if any(checks.get(name) is not True for name in ATTESTATION_CHECKS):
        raise ValueError('Endpoint alias identity checks are incomplete')
    count = attestation.get('verified_model_tensor_count')
    if type(count) is not int or count < 1:
        raise ValueError('Endpoint alias attestation has no verified model tensors')
    for name, expected_filename in (('fixed_checkpoint', 'step_0320.pt'), ('alias_checkpoint', 'last.pt')):
        entry = attestation.get(name, {})
        path = Path(entry.get('path', '').replace('\\', '/'))
        hashed = entry.get('sha256', '')
        if (path.name != expected_filename or path.parent.name != arm or len(hashed) != 64 or
                any(char not in '0123456789abcdef' for char in hashed)):
            raise ValueError('Invalid attested checkpoint path or hash')


def unique_examples(report):
    examples = report['examples']
    rows = {row['conversation_id']: row for row in examples}
    if not rows or len(rows) != len(examples):
        raise ValueError('Missing examples or repeated conversation IDs')
    if report.get('count', len(rows)) != len(rows):
        raise ValueError('Report count differs from conversation count')
    return rows


def validate_fraction(numerator, denominator):
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0:
        raise ValueError('Expected finite counts and a positive denominator')
    if not 0 <= numerator <= denominator:
        raise ValueError('Correct-unit count is outside the valid frame count')


def control_cases(report):
    result = {}
    for key, row in unique_examples(report).items():
        denominator = row['frames']
        numerator = row['conditions']['correct_a']['correct_units']
        validate_fraction(numerator, denominator)
        result[key] = {'correct': numerator, 'frames': denominator,
                       'accuracy': numerator / denominator}
    return result


def tail_cases(report, ratio=.25):
    result = {}
    wanted_seeds = set(report['mask_seeds'])
    if not wanted_seeds or len(wanted_seeds) != len(report['mask_seeds']):
        raise ValueError('Mask seeds must be nonempty and unique')
    for key, row in unique_examples(report).items():
        trials = [trial for trial in row['trials'] if trial['requested_hidden_ratio'] == ratio]
        if len(trials) != len(wanted_seeds) or {trial['mask_seed'] for trial in trials} != wanted_seeds:
            raise ValueError('Each conversation must contain exactly one measurement per mask seed')
        numerator, denominator, signature = 0, 0, []
        for trial in sorted(trials, key=lambda item: item['mask_seed']):
            region = trial['conditions']['correct_a'][TAIL]
            validate_fraction(region['correct_units'], region['hidden_frames'])
            numerator += region['correct_units']
            denominator += region['hidden_frames']
            signature.append((trial['mask_seed'], trial['mask_sha256'], region['hidden_frames']))
        # Pool mask repetitions within a conversation before resampling. / 대화별 마스크 반복을 합친 뒤 재표집합니다.
        result[key] = {'correct': numerator, 'frames': denominator,
            'accuracy': numerator / denominator, 'mask_seed_count': len(trials), 'mask_signature': signature}
    return result


def summarize_cases(cases):
    return {'conversations': len(cases),
        'mean_conversation_accuracy': float(np.mean([row['accuracy'] for row in cases.values()])),
        'frame_accuracy': sum(row['correct'] for row in cases.values()) / sum(row['frames'] for row in cases.values()),
        'correct_frames': sum(row['correct'] for row in cases.values()),
        'scored_frames_including_repeated_masks': sum(row['frames'] for row in cases.values())}


def paired_bootstrap(first, second, resamples=10000, seed=42):
    if set(first) != set(second) or not first or resamples < 1:
        raise ValueError('Paired bootstrap requires identical nonempty conversations and positive resamples')
    keys = sorted(first)
    for key in keys:
        if first[key]['frames'] != second[key]['frames']:
            raise ValueError('Paired conversations have different scoring frame counts')
        if first[key].get('mask_signature') != second[key].get('mask_signature'):
            raise ValueError('Paired hidden masks differ')
        for row in (first[key], second[key]):
            validate_fraction(row['correct'], row['frames'])
    a = np.asarray([first[key]['correct'] for key in keys], dtype=np.float64)
    b = np.asarray([second[key]['correct'] for key in keys], dtype=np.float64)
    frames = np.asarray([first[key]['frames'] for key in keys], dtype=np.float64)
    differences = (a - b) / frames
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(keys), size=(resamples, len(keys)))
    case_deltas = differences[sample_indices].mean(axis=1)
    frame_deltas = ((a - b)[sample_indices].sum(axis=1) / frames[sample_indices].sum(axis=1))
    def interval(point, samples):
        low, high = np.quantile(samples, [.025, .975])
        return {'difference': float(point), 'difference_percentage_points': float(100 * point),
                'percentile_95_ci': [float(low), float(high)],
                'percentile_95_ci_percentage_points': [float(100 * low), float(100 * high)]}
    # A positive difference only indicates higher measured unit accuracy. / 양수는 측정 단위 정확도가 높다는 뜻일 뿐입니다.
    return {'paired_conversations': len(keys), 'resamples': resamples, 'seed': seed,
        'resampling_unit': 'conversation; mask seeds already pooled within each conversation',
        'difference_direction': 'first arm minus second arm',
        'mean_conversation_accuracy': interval(differences.mean(), case_deltas),
        'frame_accuracy': interval((a - b).sum() / frames.sum(), frame_deltas),
        'conversation_wins': int((differences > 0).sum()),
        'conversation_ties': int((differences == 0).sum()),
        'conversation_losses': int((differences < 0).sum())}


def endpoint_check(report, arm, audio=False, split='val', attestation=None, report_sha256=None):
    field = 'recovery_step' if audio else 'checkpoint_step'
    if report.get(field) != STEP or report.get('planner_memory_mode') != arm:
        raise ValueError(f'Report is not the fixed step-{STEP} {arm} endpoint')
    if report.get('split') != split or split not in ('train', 'val'):
        raise ValueError('Unexpected diagnostic split')
    filename = Path(report['checkpoint'].replace('\\', '/')).name
    if audio and filename == 'last.pt':
        validate_alias_attestation(report, arm, attestation, report_sha256)
    elif filename != f'step_{STEP:04d}.pt':
        raise ValueError('Report must use the fixed endpoint filename, not best.pt')


def audio_summary(report):
    rows = unique_examples(report)
    names = set(next(iter(rows.values()))['paths'])
    if any(set(row['paths']) != names for row in rows.values()):
        raise ValueError('Audio paths differ across conversations')
    metrics, examples = {}, []
    for name in sorted(names):
        paths = [row['paths'][name] for row in rows.values()]
        errors = [float(path['reference_wer']) for path in paths]
        if not all(math.isfinite(value) and value >= 0 for value in errors):
            raise ValueError('Invalid audio reference WER')
        metrics[name] = {'count': len(paths), 'mean_reference_wer': float(np.mean(errors)),
            'median_reference_wer': float(np.median(errors)),
            'asr_token_limit_cases': sum(bool(path.get('asr_token_limit_reached', False)) for path in paths),
            'raw_clipping_cases': sum(path.get('raw_clip_fraction', 0) > 0 for path in paths)}
        metrics[name]['input_contract'] = ('A only, predicted duration' if name.startswith('predicted_length_')
            else 'A predicted units with oracle B duration' if name.startswith('reference_length_')
            else 'Oracle B units and B duration' if name in ('oracle_semantics', 'oracle_units')
            else 'Ground-truth B recording or codec reconstruction')
    for key, row in rows.items():
        examples.append({'conversation_id': key, 'input_text': row['input_text'],
            'reference_text': row['reference_text'],
            'paths': {name: {field: path[field] for field in ('asr', 'reference_wer', 'seconds',
                'asr_token_limit_reached', 'raw_clip_fraction', 'file') if field in path}
                for name, path in row['paths'].items()}})
    return {'reference_agreement_only': True, 'automated_relevance_score': None,
            'summary': metrics, 'examples': examples}


def build_summary(root, resamples=10000, seed=42):
    root = Path(root)
    plan_path = root / 'plan.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if plan.get('steps_per_arm') != STEP or set(plan.get('arms', [])) != set(ARMS):
        raise ValueError('This summary requires the preregistered three-arm step-320 experiment')
    sources = {'plan.json': hashlib.sha256(plan_path.read_bytes()).hexdigest()}
    def read(relative):
        path = root / relative
        if not path.exists():
            return None
        data = path.read_bytes()
        sources[str(relative).replace('\\', '/')] = hashlib.sha256(data).hexdigest()
        return json.loads(data)
    output = {'fixed_endpoint': STEP, 'checkpoint_selection_performed': False,
        'candidate_promoted': False, 'bootstrap_seed': seed, 'bootstrap_resamples': resamples,
        'primary_comparison': ['native_speech', 'resampled_speech'],
        'secondary_comparison': ['native_speech', 'fused'], 'source_sha256': sources,
        'limitations': [
            'All endpoints are fixed before this summary; no checkpoint is selected by these values.',
            'Confidence intervals resample paired conversations, not repeated masks or individual frames.',
            'Intervals describe variation across these cases conditional on one trained checkpoint/seed per arm; they exclude training-seed and optimization variability.',
            'Percentile intervals are exploratory and are not adjusted for multiple comparisons.',
            'Exact-reference unit accuracy, CE and audio WER do not establish response relevance or empathy.',
            'The remainder-80%-time region is a temporal proxy, not word-aligned content.',
            'Unit controls and B-hint measurements use oracle B length; only predicted_length audio is A-only inference.',
            'Confirmation is separately reserved validation with historical aggregate exposure, not a pristine final test.',
            'Missing confirmation stays unavailable; this tool never triggers or requires opening it.'
        ]}
    for phase in ('development', 'confirmation'):
        arm_results, measures, report_ids = {}, {}, {}
        expected = plan['validation_conversations'] if phase == 'development' else 32
        lock = read(Path('confirmation/locked_candidates.json')) if phase == 'confirmation' else None
        for arm in ARMS:
            folder = Path(arm) if phase == 'development' else Path('confirmation') / arm
            controls = read(folder / ('controls_val.json' if phase == 'development' else 'controls.json'))
            audio_relative = folder / ('audio_step_0320/report.json' if phase == 'development' else 'report.json')
            audio = read(audio_relative)
            hints = read(folder / 'hints/units_report.json') if phase == 'development' else None
            train_hints = read(folder / 'train_hints/units_report.json') if phase == 'development' else None
            result = {'available': {}, 'missing': []}
            measures[arm] = {}
            if controls is not None:
                endpoint_check(controls, arm)
                selection_key = 'selection_sha256' if phase == 'development' else 'confirmation_selection_sha256'
                if (controls['manifest_sha256'] != plan['manifest_sha256'] or
                        controls['selection_sha256'] != plan[selection_key]):
                    raise ValueError('Control report provenance differs from the plan')
                cases = control_cases(controls)
                if len(cases) != expected:
                    raise ValueError('Unexpected control conversation count')
                measures[arm]['sampled_unit_accuracy'] = cases
                report_ids[(arm, 'controls')] = set(cases)
                result['available']['unit_controls'] = {'count': len(cases), 'conditions': controls['conditions'],
                    'correct_a_case_summary': summarize_cases(cases)}
            else:
                result['missing'].append('unit_controls')
            if hints is not None:
                endpoint_check(hints, arm)
                if (hints['manifest_sha256'] != plan['manifest_sha256'] or
                        hints['selection_sha256'] != plan['selection_sha256']):
                    raise ValueError('Hint report provenance differs from the plan')
                cases = tail_cases(hints)
                if len(cases) != expected:
                    raise ValueError('Unexpected hint conversation count')
                if controls is not None and set(cases) != report_ids[(arm, 'controls')]:
                    raise ValueError('Hints and controls use different conversations')
                measures[arm]['quarter_hidden_tail_accuracy'] = cases
                result['available']['hidden_tail'] = {'hidden_ratio': .25, 'region': TAIL,
                    'within_conversation_mask_aggregation': 'sum correct frames / sum hidden frames across both seeds',
                    **summarize_cases(cases), 'all_ratio_summary': hints['summary']}
                if 'visible_hint_summary' in hints:
                    result['available']['hidden_tail']['visible_hint_summary'] = hints['visible_hint_summary']
            elif phase == 'development':
                result['missing'].append('hidden_tail')
            if audio is not None:
                attestation = read(Path(arm) / 'endpoint_identity.json') if phase == 'development' else None
                endpoint_check(audio, arm, audio=True, attestation=attestation,
                               report_sha256=sources[str(audio_relative).replace('\\', '/')])
                audio_ids = set(unique_examples(audio))
                expected_audio = 8 if phase == 'development' else expected
                if len(audio_ids) != expected_audio:
                    raise ValueError('Unexpected audio conversation count')
                if controls is not None and not audio_ids <= report_ids[(arm, 'controls')]:
                    raise ValueError('Audio cases are outside the matching unit controls')
                report_ids[(arm, 'audio')] = audio_ids
                result['available']['audio'] = audio_summary(audio)
                if Path(audio['checkpoint'].replace('\\', '/')).name == 'last.pt':
                    result['available']['audio']['verified_endpoint_alias'] = attestation
            else:
                result['missing'].append('audio')
            if train_hints is not None:
                endpoint_check(train_hints, arm, split='train')
                if (train_hints['manifest_sha256'] != plan['manifest_sha256'] or
                        train_hints['selection_sha256'] != plan['selection_sha256']):
                    raise ValueError('Training hint provenance differs from the plan')
                cases = tail_cases(train_hints)
                if len(cases) != 64:
                    raise ValueError('Expected the fixed 64-case training diagnostic')
                if controls is not None and set(cases) & report_ids[(arm, 'controls')]:
                    raise ValueError('Training and validation hint conversations overlap')
                result['available']['train_hidden_tail'] = {'split': 'train', 'hidden_ratio': .25,
                    'region': TAIL, 'optimization_performed': False, **summarize_cases(cases),
                    'all_ratio_summary': train_hints['summary']}
            if phase == 'confirmation' and lock is not None:
                for candidate_report in (controls, audio):
                    if candidate_report is not None and candidate_report['checkpoint'] != lock[arm]['checkpoint']:
                        raise ValueError('Confirmation report checkpoint differs from the locked candidate')
            arm_results[arm] = result
        paired = {}
        for first, second, priority in COMPARISONS:
            name = first + '_minus_' + second
            paired[name] = {'priority': priority, 'metrics': {}}
            for metric in ('sampled_unit_accuracy', 'quarter_hidden_tail_accuracy'):
                if metric in measures[first] and metric in measures[second]:
                    paired[name]['metrics'][metric] = paired_bootstrap(measures[first][metric],
                        measures[second][metric], resamples=resamples, seed=seed)
            if (first, 'audio') in report_ids and (second, 'audio') in report_ids:
                if report_ids[(first, 'audio')] != report_ids[(second, 'audio')]:
                    raise ValueError('Audio arms have different conversation memberships')
        present = any(result['available'] for result in arm_results.values())
        if phase == 'confirmation' and present and lock is None:
            raise ValueError('Confirmation reports require the preregistered locked_candidates.json')
        if phase == 'confirmation' and lock is not None and set(lock) != set(ARMS):
            raise ValueError('Confirmation checkpoint lock has unexpected arms')
        output[phase] = {'status': 'complete' if present and all(not result['missing'] for result in arm_results.values())
                         else 'partial' if present else 'not_available',
                         'arms': arm_results, 'paired_comparisons': paired}
        if phase == 'confirmation':
            output[phase]['locked_candidates'] = lock
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--attest-audio-alias', choices=ARMS,
                        help='Verify last.pt model identity against step_0320.pt and bind the epoch audio report')
    args = parser.parse_args()
    if args.attest_audio_alias:
        if args.output:
            parser.error('--attest-audio-alias writes arm/endpoint_identity.json; omit --output')
        print(json.dumps(attest_audio_alias(args.root, args.attest_audio_alias)), flush=True)
        return
    if args.output is None:
        parser.error('--output is required when summarizing')
    report = build_summary(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(args.output)
    print(json.dumps({phase: report[phase]['status'] for phase in ('development', 'confirmation')}), flush=True)


if __name__ == '__main__':
    main()
