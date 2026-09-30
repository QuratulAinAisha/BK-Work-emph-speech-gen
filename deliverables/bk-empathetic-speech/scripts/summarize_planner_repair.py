"""Summarize saved experiment evidence, not live jobs. / 실시간 작업이 아닌 저장된 실험 근거를 요약합니다."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


TITLES = ('Prior history', 'Tiny B-only learnability', 'Target and paired-data audit',
    'Masking/loss recipes', 'Conditional architecture branch', 'Stronger speech prior',
    'A-controlled response generation', 'Sampled-audio objective', 'Acoustic robustness',
    'Length and stopping', 'Affect controls', 'Conditional representation diagnostics',
    'Independent confirmation')
LIMITATIONS = (
    'This is a snapshot of available files. Missing local artifacts may still exist on the server; no server was contacted.',
    'Completed means the listed computational evidence exists, not that the model produces good replies.',
    'Training completion, integrity checks, gradient checks and audio quality are separate claims.',
    'Oracle B hints/duration and tiny reused training examples do not establish normal A-only generalization.',
    'ASR WER, unit accuracy and prosody proxies do not establish empathy, relevance or human-perceived quality.',
    'Report-file hashes are recomputed here; embedded checkpoint/data hashes and review decisions remain reported provenance.',
    'Human independence and untouched-test protocol cannot be certified by generated forms or this script.',
    'The isolated autoregressive branch adds prior/conditional adaptation compute; it is not an equal-compute architecture comparison.',
    'Autoregressive teacher-forced scores use the correct B prefix and B length; they are not free-generation quality scores.',
)
AR_ARCHITECTURE = 'bk_experimental_ar_units_v1'


def requirements():
    result = {step: [] for step in range(1, 14)}

    def add(step, label, kind, *paths):
        result[step].append({'label': label, 'kind': kind, 'paths': list(paths)})

    for label in ('prior', 'conditional', 'current'):
        add(1, label + ' matched reconstruction', 'units', f'step01_history/{label}.json')
    add(2, 'fixed/fresh training endpoint', 'complete', 'step02_tiny/complete.json')
    add(2, 'reconstruction evaluation', 'evaluation', 'step02_tiny/latest_evaluation.json')
    add(2, 'frozen audit', 'audit', 'step02_tiny/audit_step_*.json')
    add(3, 'cache and transcript audit', 'data', 'step03_data.json')
    for arm in ('legacy_low_lr', 'legacy', 'balanced', 'curriculum'):
        add(4, arm + ' endpoint', 'complete', f'step04_recipes/{arm}/complete.json')
        add(4, arm + ' matched gaps', 'units', f'step04_recipes/{arm}/gaps/{arm}.json')
        add(4, arm + ' frozen audits', 'audit', f'step04_recipes/{arm}/audit_step_*.json')
    add(5, 'optional architecture comparison', 'evaluation', 'step05_architecture/summary.json')
    add(6, 'stronger prior endpoint', 'complete', 'step06_prior/complete.json')
    add(6, 'held-out reconstruction', 'units', 'step06_prior/gaps/strong_prior.json')
    add(6, 'oracle reconstruction audio and ASR', 'audio', 'step06_prior/audio/report.json')
    for arm in ('paired_tiny', 'balanced', 'fully_masked', 'rehearsal'):
        base = f'step07_conditional/{arm}'
        add(7, arm + ' endpoint', 'complete', base + '/complete.json')
        for split in ('train', 'val'):
            add(7, arm + ' ' + split + ' A controls', 'controls',
                base + f'/{split}_controls.json', base + f'/controls_{split}.json')
        add(7, arm + ' generated audio and ASR', 'audio', base + '/audio/report.json')
    add(8, 'real sampled-waveform gradient preflight', 'audit', 'step08_waveform/real_gradient.json')
    for arm in ('ce_control', 'sampled_ctc'):
        add(8, arm + ' endpoint', 'complete', f'step08_waveform/{arm}/complete.json')
        add(8, arm + ' independent ASR audio', 'audio', f'step08_waveform/{arm}/audio/report.json')
    add(9, 'correct/corrupted/predicted unit audio', 'corrupt_audio', 'step06_prior/audio/report.json')
    add(10, 'duration-policy audio and metrics', 'duration', 'steps10_11/report.json')
    add(11, 'affect-condition audio and metrics', 'affect', 'steps10_11/report.json')
    add(12, 'sampler unit comparison', 'units', 'step12_sampling/units_report.json')
    add(12, 'sampler audio comparison', 'audio', 'step12_sampling/audio/report.json')
    add(12, 'self-hint reconstruction diagnostic', 'units', 'step12_self_hints/units_report.json', 'step12_self_hints/report.json')
    add(12, '256-unit frozen-acoustic compatibility', 'vocabulary', 'step12_vocabulary/audio/report.json', 'step12_vocabulary/report.json')
    add(13, 'blank blinded listening package', 'listening', 'step13_listening/answer_key.json')
    return result


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def reported_failures(data):
    failures = []
    for key in ('passed', 'hard_checks_passed', 'all_frozen_audits_passed', 'all_modules_unchanged'):
        if data.get(key) is False:
            failures.append(key + '=false')
    if data.get('error'):
        failures.append(str(data['error']))
    # Quality failure is not a failed training process. / 품질 미달은 학습 실행 실패와 다릅니다.
    return failures


def compact_metrics(data):
    scalar_keys = ('count', 'steps', 'updates', 'training_conversations', 'fixed_updates', 'fresh_updates',
        'fixed_mask_accuracy_at_endpoint', 'fresh_half_mask_accuracy_at_endpoint', 'all_frozen_audits_passed',
        'held_out_generalization_tested', 'quality_passed', 'test_examples_opened', 'hard_checks_passed',
        'normal_inference', 'oracle_B_length', 'A_only_generation', 'partial_correct_B_hints',
        'all_modules_unchanged', 'passed', 'checkpoint_step', 'phase', 'architecture', 'step',
        'example_mean_next_unit_ce', 'frame_next_unit_ce', 'teacher_forced_unit_accuracy',
        'uses_correct_B_prefix', 'frames', 'examples', 'feasibility_only', 'world_size',
        'batch_size_per_gpu', 'production_candidate_promoted', 'free_generation_evaluation_required')
    # Example arrays stay in source reports. / 예제 배열은 원본 보고서에 둡니다.
    result = {key: data[key] for key in scalar_keys if key in data}
    if isinstance(result.get('examples'), list):
        result.pop('examples')
    for key in ('metrics', 'conditions', 'duration_summary', 'distortion_summary', 'forward_parity',
                'teacher_forced', 'teacher_forced_validation', 'planner_sampling', 'frozen_audit'):
        if isinstance(data.get(key), dict):
            result[key] = data[key]
    if isinstance(data.get('free_running'), dict):
        result['free_running'] = {key: data['free_running'][key] for key in
            ('count', 'summary', 'A_encoder_inputs', 'duration_locked_across_controls', 'limitations')
            if key in data['free_running']}
    if isinstance(data.get('records'), list):
        result['record_count'] = len(data['records'])
        result['latest_record'] = data['records'][-1] if data['records'] else None
    if isinstance(data.get('summary'), dict):
        summary = data['summary']
        if any('/' in key for key in summary):
            result['summary'] = {key: {name: values[name] for name in
                ('all_hidden', 'remainder_80pct_time', 'run_boundary_or_singleton') if name in values}
                for key, values in summary.items() if isinstance(values, dict)
                and ('/nearest_copy' in key or '/null_prior/first_pass' in key or '/production_a/first_pass' in key)}
        else:
            result['summary'] = summary
    if isinstance(data.get('splits'), dict):
        result['audited_samples'] = {key: {name: value[name] for name in
            ('selected_transcript_count', 'sampled_cache_count') if name in value}
            for key, value in data['splits'].items()}
    return result


class Evidence:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.cache = {}
        self.documents = {}

    def paths(self, patterns):
        found = set()
        for pattern in patterns:
            for path in self.root.glob(pattern):
                resolved = path.resolve()
                if path.is_file() and resolved.is_relative_to(self.root):
                    found.add(path.relative_to(self.root).as_posix())
        return sorted(found)

    def read(self, relative):
        if relative in self.cache:
            return self.cache[relative], self.documents.get(relative)
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Evidence escapes the selected experiment tree')
        raw = path.read_bytes()
        entry = {'path': relative, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
                 'failures': [], 'metrics': {}}
        try:
            decoded = raw.decode('utf-8-sig')
            if path.suffix == '.jsonl':
                records = [json.loads(line) for line in decoded.splitlines() if line.strip()]
                if not all(isinstance(row, dict) for row in records):
                    raise ValueError('Expected JSON objects in each nonempty JSONL line')
                data = {'records': records}
            else:
                data = json.loads(decoded)
            if not isinstance(data, dict):
                raise ValueError('Expected a JSON object')
            entry['failures'] = reported_failures(data)
            entry['metrics'] = compact_metrics(data)
            entry['provenance'] = {key: data[key] for key in
                ('checkpoint', 'checkpoint_sha256', 'manifest_sha256', 'selection_sha256', 'split',
                 'source_checkpoint_sha256', 'source_sha256', 'architecture', 'phase') if key in data}
            self.documents[relative] = data
        except (ValueError, UnicodeError) as error:
            entry['failures'] = ['Invalid artifact: ' + str(error)]
        self.cache[relative] = entry
        return entry, self.documents.get(relative)

    def validate(self, relative, kind):
        entry, data = self.read(relative)
        if data is None:
            return False, entry['failures']
        missing = []
        if kind == 'complete':
            endpoint = data.get('steps', data.get('updates'))
            if type(endpoint) is not int or endpoint <= 0:
                missing.append('No positive recorded update endpoint')
        elif kind == 'audit' and data.get('passed') is not True:
            missing.append('No passing check')
        elif kind == 'data' and data.get('hard_checks_passed') is not True:
            missing.append('No passing hard data checks')
        elif kind == 'evaluation' and not data.get('metrics', data.get('summary')):
            missing.append('No evaluation metrics')
        elif kind in ('units', 'controls'):
            if not data.get('examples') or not data.get('summary', data.get('conditions')):
                missing.append('No completed example-level unit/control metrics')
        if kind in ('audio', 'corrupt_audio', 'duration', 'affect', 'vocabulary'):
            examples = data.get('examples', [])
            if not examples or not data.get('summary'):
                missing.append('No completed audio/ASR summary')
            for row in examples:
                paths = row.get('paths', {})
                if not paths:
                    missing.append('Example has no audio paths')
                for value in paths.values():
                    name = value.get('file')
                    audio = ((self.root / relative).parent / str(name)).resolve()
                    if (not name or not audio.is_relative_to((self.root / relative).parent.resolve())
                            or not audio.is_file()):
                        missing.append('Missing or unsafe referenced audio: ' + str(name))
                    if not isinstance(value.get('asr'), str):
                        missing.append('Missing ASR transcript')
            names = set(examples[0].get('paths', {})) if examples else set()
            if kind == 'corrupt_audio' and not ('oracle_units' in names and
                    any('random_hidden' in x for x in names) and any('iterative_hidden' in x for x in names)):
                missing.append('Correct, corrupted and predicted unit arms are not all present')
            if kind == 'duration' and not data.get('duration_summary'):
                missing.append('No duration summary')
            if kind == 'affect' and not all(any(condition in x for x in names) for condition in
                    ('predicted_affect', 'zero_affect', 'shuffled_affect')):
                missing.append('Predicted/zero/shuffled affect conditions are incomplete')
            if kind == 'vocabulary' and (not data.get('distortion_summary') or data.get('all_modules_unchanged') is not True):
                missing.append('No distortion summary or frozen-module audit')
        if kind == 'listening':
            if not data.get('trials') or not (self.root / relative).parent.joinpath('rater/index.html').is_file():
                missing.append('No usable blinded package')
        return not missing and not entry['failures'], sorted(set(missing))


def listening_status(evidence, key_path='step13_listening/answer_key.json'):
    result = {'status': 'pending_human', 'package_prepared': False, 'valid_rating_cells': 0,
              'rating_files': [], 'independence_verified': False, 'problems': [], 'package_id': None}
    if not evidence.paths([key_path]):
        return result
    valid, _ = evidence.validate(key_path, 'listening')
    _, key = evidence.read(key_path)
    result['package_prepared'] = valid
    if not key:
        return result
    result['package_id'] = key.get('package_id')
    trials = {row['trial_id'] for row in key.get('trials', [])}
    fields = ('intelligibility', 'relevance', 'emotional_appropriateness', 'naturalness')
    base = Path(key_path).parent
    for relative in evidence.paths([(base / '**/*.csv').as_posix(), (base / '*.csv').as_posix()]):
        if Path(relative).name == 'ratings_template.csv':
            continue
        cells, seen = 0, set()
        problems = []
        try:
            with (evidence.root / relative).open(encoding='utf-8-sig', newline='') as stream:
                reader = csv.DictReader(stream)
                if not {'package_id', 'trial_id', *fields}.issubset(reader.fieldnames or []):
                    raise ValueError('Missing rating columns')
                for row in reader:
                    identity = row['trial_id']
                    if row['package_id'] != key.get('package_id') or identity not in trials or identity in seen:
                        problems.append('Unknown package/trial or duplicate trial')
                    seen.add(identity)
                    for field in fields:
                        if row[field] not in ('', '1', '2', '3', '4', '5'):
                            problems.append('Invalid rating value')
                        elif row[field]:
                            cells += 1
        except (ValueError, UnicodeError, csv.Error) as error:
            problems.append(str(error))
        result['rating_files'].append({'path': relative, 'sha256': file_hash(evidence.root / relative),
            'valid_rating_cells': cells if not problems else 0, 'problems': sorted(set(problems))})
        result['problems'].extend(problems)
        if not problems:
            result['valid_rating_cells'] += cells
    if result['valid_rating_cells']:
        result['status'] = 'ratings_recorded_independent_review_pending'
    return result


def external_listening(evidence, packages):
    """Read only explicitly selected package roots. / 명시한 평가 묶음 경로만 읽습니다."""
    artifacts, checks, states = [], [], []
    seen = {(evidence.root / 'step13_listening').resolve()}
    for package in packages or []:
        root = Path(package).resolve()
        if root in seen:
            continue
        seen.add(root)
        selected = Evidence(root)
        status = listening_status(selected, 'answer_key.json')
        key = root / 'answer_key.json'
        for rating in status['rating_files']:
            rating['path'] = (root / rating['path']).as_posix()
        states.append({'root': str(root), **status})
        if selected.paths(['answer_key.json']):
            valid, missing = selected.validate('answer_key.json', 'listening')
            entry = selected.cache['answer_key.json']
            artifacts.append({**entry, 'path': key.as_posix(), 'external_listening_package': True})
            checks.append({'path': key.as_posix(), 'valid': valid, 'missing': missing})
        else:
            checks.append({'path': key.as_posix(), 'valid': False, 'missing': ['No selected listening answer key']})
    return artifacts, checks, states


def merge_listening(states):
    result = {'status': 'pending_human', 'package_prepared': False, 'valid_rating_cells': 0,
        'rating_files': [], 'independence_verified': False, 'problems': [], 'packages': states}
    seen = set()
    for state in states:
        identity = state.get('package_id')
        if identity and identity in seen:
            result['problems'].append('Repeated package ID; ratings counted once: ' + identity)
            continue
        if identity:
            seen.add(identity)
        result['package_prepared'] |= state['package_prepared']
        result['valid_rating_cells'] += state['valid_rating_cells']
        result['rating_files'].extend(state['rating_files'])
        result['problems'].extend(state['problems'])
    if result['valid_rating_cells']:
        result['status'] = 'ratings_recorded_independent_review_pending'
    return result


def teacher_metrics_valid(metrics):
    """Teacher scores require explicit oracle scope. / 교사 점수에는 정답 입력 범위가 필요합니다."""
    if not isinstance(metrics, dict):
        return False
    numeric = ('example_mean_next_unit_ce', 'frame_next_unit_ce', 'teacher_forced_unit_accuracy')
    return (all(type(metrics.get(key)) in (int, float) and math.isfinite(metrics[key]) for key in numeric)
        and metrics['example_mean_next_unit_ce'] >= 0 and metrics['frame_next_unit_ce'] >= 0
        and 0 <= metrics['teacher_forced_unit_accuracy'] <= 1
        and all(type(metrics.get(key)) is int and metrics[key] > 0 for key in ('frames', 'examples'))
        and metrics.get('uses_correct_B_prefix') is True and metrics.get('oracle_B_length') is True
        and metrics.get('normal_inference') is False)


def ar_observed(evidence, patterns):
    paths = evidence.paths(patterns)
    documents, failures = {}, []
    for relative in paths:
        entry, data = evidence.read(relative)
        documents[relative] = data or {}
        failures.extend({'path': relative, 'reason': reason} for reason in entry['failures'])
    return paths, documents, failures


def ar_training(evidence, phase):
    base = f'step12_ar/{phase}'
    names = ('recipe.json', 'complete.json', 'progress.json', 'initial_validation.json',
             'resume_validation.json', 'metrics.jsonl', 'data_profile.json', 'audit_step_*.json')
    paths, documents, failures = ar_observed(evidence, [base + '/' + name for name in names]
        + [f'step12_ar/{phase}_preflight/preflight_rank*.json'])
    data = lambda name: documents.get(base + '/' + name, {})
    recipe, complete = data('recipe.json'), data('complete.json')
    curve = data('metrics.jsonl').get('records', [])
    initial = data('initial_validation.json')
    planned, endpoint = recipe.get('steps'), complete.get('steps')
    positive = lambda value: type(value) is int and value > 0
    pending = []
    if not (positive(planned) and recipe.get('phase') == phase
            and recipe.get('feasibility_only') is True and recipe.get('duration_and_acoustics_frozen') is True):
        pending.append('Complete experimental adaptation recipe')
    if not positive(endpoint):
        pending.append('Training completion endpoint')
    elif endpoint != planned or complete.get('phase') != phase:
        failures.append({'path': base + '/complete.json', 'reason': 'Completion does not match recipe phase/update budget'})
    if not teacher_metrics_valid(initial):
        pending.append('Initial held-out teacher-forced validation')
    if not curve or any(not teacher_metrics_valid(row.get('teacher_forced_validation'))
                        or not positive(row.get('step')) for row in curve):
        pending.append('Valid teacher-forced training curve')
    if positive(endpoint):
        audit_path = base + f'/audit_step_{endpoint:04d}.json'
        if documents.get(audit_path, {}).get('passed') is not True:
            pending.append('Passing frozen audit at the completed endpoint')
        if not curve or curve[-1].get('step') != endpoint:
            pending.append('Teacher-forced validation at the completed endpoint')
    if any(documents[path].get('passed') is not True for path in paths
           if '/audit_step_' in path or '/preflight_rank' in path):
        pending.append('All observed frozen/preflight audits must pass')
    status = 'failed' if failures else ('completed' if not pending else ('partial' if paths else 'pending'))
    timing_names = ('elapsed_seconds', 'this_process_batch_loading_seconds',
                    'this_process_optimizer_update_seconds', 'this_process_update_count')
    return {'phase': phase, 'status': status, 'evidence_paths': paths, 'failed_checks': failures,
        'pending_evidence': pending, 'planned_updates': planned if positive(planned) else None,
        'completed_updates_reported': endpoint if positive(endpoint) else 0,
        'completed_updates_with_required_evidence': endpoint if status == 'completed' else 0,
        'recipe': recipe, 'data_profile': data('data_profile.json'), 'latest_progress_snapshot': data('progress.json'),
        'teacher_forced': {'initial': initial, 'curve': curve,
            'latest': curve[-1].get('teacher_forced_validation') if curve else None,
            'scope': 'Held-out next-unit prediction with correct B prefix and oracle B length; not free generation.'},
        'latest_process_timing': {key: curve[-1][key] for key in timing_names if key in curve[-1]} if curve else {},
        'timing_limit': 'Cumulative values belong to the latest process, may reset on resume, and are not summed across rows.'}


def ar_evaluation(evidence, label, phase, mode, require_audio, base_override=None):
    base = base_override or f'step12_ar/{label}_evaluation'
    unit_path, audio_path = base + '/units_report.json', base + '/audio/report.json'
    paths, documents, failures = ar_observed(evidence, [unit_path, audio_path])
    report, audio = documents.get(unit_path, {}), documents.get(audio_path, {})
    pending = []
    if not report:
        pending.append('AR unit evaluation report')
    for relative, data in ((unit_path, report), (audio_path, audio)):
        if not data:
            continue
        for key, expected in (('architecture', AR_ARCHITECTURE), ('phase', phase)):
            if data.get(key) != expected:
                failures.append({'path': relative, 'reason': f'Unexpected {key}: {data.get(key)}'})
        if data.get('planner_sampling', {}).get('mode') != mode:
            failures.append({'path': relative, 'reason': f'Expected explicit {mode} sampling metadata'})
        if data.get('production_candidate_promoted') is not False:
            pending.append('Explicit experimental/no-promotion provenance')
    teacher = report.get('teacher_forced', {})
    teacher_ok = teacher_metrics_valid(teacher)
    if not teacher_ok:
        pending.append('Teacher-forced oracle-prefix evaluation')
    free = report.get('free_running', {})
    expected = {'null_prior'} if phase == 'prior' else {'correct_a', 'shuffled_a'}
    free_ok = (bool(free.get('examples')) and expected <= set(free.get('summary', {}))
        and all(expected <= set(row.get('conditions', {})) for row in free.get('examples', [])))
    if not free_ok:
        pending.append('Free-running examples and condition summaries')
    if phase == 'conditional' and free_ok:
        if free.get('A_encoder_inputs') != 'person_a_only' or free.get('duration_locked_across_controls') is not True:
            pending.append('A-only inputs and fixed-duration conditioning-control provenance')
        for row in free['examples']:
            if (row.get('A_only_generation') is not True or row.get('uses_B_unit_inputs') is not False
                    or row.get('duration_source') != 'A_predicted_duration'
                    or row.get('style_and_speaker_held_fixed') is not True):
                pending.append('A-only predicted-length generation provenance for every example')
                break
    audit = report.get('frozen_audit', {})
    if audit.get('passed') is not True:
        pending.append('Final frozen-module evaluation audit')
    failures.extend({'path': unit_path, 'reason': 'frozen_audit: ' + reason} for reason in reported_failures(audit))
    audio_ok = False
    if require_audio or audio:
        if audio:
            audio_ok, missing = evidence.validate(audio_path, 'audio')
            pending.extend(missing)
            if not all('predicted_length_ar' in row.get('paths', {}) for row in audio.get('examples', [])):
                pending.append('Predicted-length AR audio for every example')
                audio_ok = False
            if not report.get('checkpoint_sha256') or not audio.get('checkpoint_sha256'):
                pending.append('Unit/audio checkpoint identities')
            elif audio['checkpoint_sha256'] != report['checkpoint_sha256']:
                failures.append({'path': audio_path, 'reason': 'Unit/audio checkpoint identities differ'})
        else:
            pending.append('Completed generated audio and independent ASR report')
    status = 'failed' if failures else ('completed' if not pending else ('partial' if paths else 'pending'))
    return {'phase': phase, 'sampling_mode': mode, 'status': status,
        'optional_if_unobserved': label == 'categorical', 'evidence_paths': paths,
        'failed_checks': failures, 'pending_evidence': sorted(set(pending)),
        'checkpoint_sha256': report.get('checkpoint_sha256'), 'checkpoint_step': report.get('checkpoint_step'),
        'planner_sampling': report.get('planner_sampling'), 'acoustic_seed': report.get('acoustic_seed'),
        'teacher_forced': {'status': 'recorded' if teacher_ok else 'pending', 'metrics': teacher,
                          'normal_inference': False, 'uses_correct_B_prefix': True, 'oracle_B_length': True},
        'free_running': {'status': 'recorded' if free_ok else 'pending', 'summary': free.get('summary', {}),
            'count': free.get('count'), 'quality_status': 'not_established',
            'duration_scope': 'A-predicted; locked across A controls' if phase == 'conditional' else 'Oracle B length; null A prior'},
        'audio': {'status': 'recorded' if audio_ok else ('pending' if require_audio else 'not_requested'),
                  'summary': audio.get('summary', {}), 'human_quality_status': 'not_established'},
        'frozen_audit': audit}


def ar_branch(evidence):
    training = {phase: ar_training(evidence, phase) for phase in ('prior', 'conditional')}
    evaluations = {label: ar_evaluation(evidence, label, phase, mode, audio) for label, phase, mode, audio in (
        ('prior', 'prior', 'greedy', False), ('conditional', 'conditional', 'greedy', True),
        ('categorical', 'conditional', 'categorical', True))}
    components = list(training.values()) + list(evaluations.values())
    required = components[:-1] + ([evaluations['categorical']] if evaluations['categorical']['evidence_paths'] else [])
    paths = sorted({path for row in components for path in row['evidence_paths']})
    failures = [failure for row in components for failure in row['failed_checks']]
    status = ('failed' if failures else 'completed' if all(row['status'] == 'completed' for row in required)
              else 'partial' if paths else 'pending')
    greedy, sampled = evaluations['conditional'], evaluations['categorical']
    paired = bool(greedy['checkpoint_sha256'] and sampled['checkpoint_sha256']
                  and greedy['checkpoint_sha256'] == sampled['checkpoint_sha256'])
    return {'status': status, 'architecture': AR_ARCHITECTURE, 'isolated_experimental_branch': True,
        'production_candidate_promoted': False, 'quality_status': 'not_established',
        'training': training, 'evaluations': evaluations, 'evidence_paths': paths, 'failed_checks': failures,
        'categorical_matches_greedy_checkpoint': paired if sampled['evidence_paths'] else None,
        'additional_adaptation_compute': {
            'planned_updates_in_available_recipes': sum(row['planned_updates'] or 0 for row in training.values()),
            'completed_updates_reported': sum(row['completed_updates_reported'] for row in training.values()),
            'completed_updates_with_required_evidence': sum(row['completed_updates_with_required_evidence'] for row in training.values()),
            'includes_source_masked_prior_training': False, 'gpu_hours_established': False,
            'comparison_limit': 'Extra prior/conditional adaptation plus serial decoding; not an equal-compute architecture winner.'},
        'pending_evidence': [label + ': ' + item for label, row in
            list(training.items()) + [(name + ' evaluation', row) for name, row in evaluations.items()]
            if not (row.get('optional_if_unobserved') and not row['evidence_paths']) for item in row['pending_evidence']]}


def confirmation_phase(evidence, phase):
    """Confirm saved work, not a live process. / 실행 중 상태가 아닌 저장된 작업을 확인합니다."""
    base = 'step13_confirmation'
    protocol_path, status_path = f'{base}/protocol_{phase}.json', f'{base}/status_{phase}.json'
    patterns = [protocol_path, status_path, f'{base}/receipts/{phase}/*.json']
    patterns += ([f'{base}/seeds/*/complete.json', f'{base}/seeds/*/audit_step_*.json',
                  f'{base}/seeds/gaps/*.json', f'{base}/seeds/fresh_audio/*/report.json'] if phase == 'seeds' else
                 [f'{base}/fresh/*/val_controls.json', f'{base}/fresh/*/units_report.json', f'{base}/fresh/*/audio/report.json'])
    paths, documents, failures = ar_observed(evidence, patterns)
    protocol, controller = documents.get(protocol_path, {}), documents.get(status_path, {})
    pending, jobs, comparisons = [], {}, {}
    protocol_hash = evidence.cache.get(protocol_path, {}).get('sha256')
    if not protocol or not protocol.get('source_sha256') or not protocol.get('input_sha256'):
        pending.append('Immutable protocol with source/input hashes')
    if protocol and protocol.get('phase') != phase:
        failures.append({'path': protocol_path, 'reason': 'Protocol phase differs'})
    if controller.get('stage') != 'complete_pending_review':
        pending.append('Controller completion pending review')
    if controller.get('stage') == 'failed' and not controller.get('error'):
        failures.append({'path': status_path, 'reason': 'Controller recorded failure'})
    if controller and protocol_hash and controller.get('protocol_sha256') != protocol_hash:
        failures.append({'path': status_path, 'reason': 'Controller protocol hash differs'})
    fresh_ids = protocol.get('fresh_conversation_ids', [])
    audio_paths = protocol.get('fresh_audio_paths', [])

    def check_cases(data, audio=False, ar=False):
        rows = data.get('free_running', {}).get('examples', []) if ar else data.get('examples', [])
        key, expected = ('path', audio_paths) if audio else ('conversation_id', fresh_ids)
        actual = [row.get(key) for row in rows]
        return bool(expected) and len(actual) == len(expected) and set(actual) == set(expected)

    def require_job(name, outputs):
        checks = []
        for relative, kind in outputs:
            valid, missing = evidence.validate(relative, kind) if relative in documents else (False, ['Missing artifact'])
            data = documents.get(relative, {})
            if kind in ('units', 'controls', 'audio') and data and not check_cases(data, audio=kind == 'audio'):
                valid = False
                missing.append('Fresh selected case set is incomplete or differs')
            checks.append({'path': relative, 'valid': valid, 'missing': missing})
        definition_path = f'{base}/receipts/{phase}/{name}_definition.json'
        completion_path = f'{base}/receipts/{phase}/{name}_complete.json'
        definition, receipt = documents.get(definition_path, {}), documents.get(completion_path, {})
        receipt_ok = bool(definition and receipt and receipt.get('output_sha256') and definition.get('outputs'))
        if definition and protocol_hash and definition.get('protocol_sha256') != protocol_hash:
            failures.append({'path': definition_path, 'reason': 'Job definition protocol hash differs'})
            receipt_ok = False
        if receipt and definition_path in evidence.cache:
            if receipt.get('definition_sha256') != evidence.cache[definition_path]['sha256']:
                failures.append({'path': completion_path, 'reason': 'Receipt definition hash differs'})
                receipt_ok = False
        output_hashes = receipt.get('output_sha256', {})
        if definition and not set(definition.get('outputs', [])) <= set(output_hashes):
            receipt_ok = False
        def local_relative(saved):
            parts = Path(saved.replace('\\', '/')).parts
            return Path(*parts[parts.index(base):]).as_posix() if base in parts and '..' not in parts else None
        declared = {local_relative(saved) for saved in definition.get('outputs', [])}
        if not {relative for relative, _ in outputs} <= declared:
            receipt_ok = False
        verified, unavailable = 0, []
        # Remote checkpoint bytes need not be downloaded. / 원격 체크포인트 전체 다운로드는 요구하지 않습니다.
        for saved, checksum in output_hashes.items():
            relative = local_relative(saved)
            if relative is None:
                unavailable.append(saved)
                continue
            local = (evidence.root / relative).resolve()
            if not local.is_relative_to(evidence.root) or local.suffix == '.pt' or not local.is_file():
                unavailable.append(saved)
                continue
            actual = evidence.cache[relative]['sha256'] if relative in evidence.cache else file_hash(local)
            if actual != checksum:
                failures.append({'path': completion_path, 'reason': 'Receipt output hash differs: ' + relative})
                receipt_ok = False
            else:
                verified += 1
        satisfied = bool(checks) and all(row['valid'] for row in checks) and receipt_ok
        if not satisfied:
            pending.append(name + ': reports/audits and bound completion receipt')
        jobs[name] = {'completed': satisfied, 'artifacts': checks, 'receipt_valid': receipt_ok,
            'locally_verified_output_hashes': verified, 'reported_output_hashes_not_recomputed': unavailable}

    if phase == 'seeds':
        seed_spec = protocol.get('seed_replication', {})
        new_seeds = seed_spec.get('new_training_seeds', [43, 44])
        rates = seed_spec.get('learning_rates', {'low_lr': .0001, 'high_lr': .001})
        endpoints = seed_spec.get('frozen_audit_endpoints', [160, 320, 480, 640])
        for seed in new_seeds:
            for rate in rates:
                label = f'{rate}_seed{seed}'
                out = f'{base}/seeds/{label}'
                require_job(label, [(out + '/complete.json', 'complete')])
                complete = documents.get(out + '/complete.json', {})
                if (complete.get('steps') != seed_spec.get('total_updates', 640)
                        or complete.get('world_size') != 4 or complete.get('stage') != 'planner'):
                    pending.append(label + ': prescribed 640-update, four-GPU planner endpoint')
                for step in endpoints:
                    require_job(label + f'_step_{step:04d}_audit', [(out + f'/audit_step_{step:04d}.json', 'audit')])
        labels = [f'{rate}_seed{seed}' for seed in seed_spec.get('seeds', [42, 43, 44]) for rate in rates]
        require_job('six_endpoint_fresh_gaps', [(f'{base}/seeds/gaps/{label}.json', 'units') for label in labels])
        summary_path = f'{base}/seeds/gaps/summary.json'
        if set(documents.get(summary_path, {}).get('checkpoints', {})) != set(labels):
            pending.append('Gap summary containing all prescribed seed/rate endpoints')
        for label in labels:
            report = documents.get(f'{base}/seeds/gaps/{label}.json', {})
            comparisons[label] = {name: values for name, values in compact_metrics(report).get('summary', {}).items()
                                  if name.startswith(('random/0.5/', 'contiguous/0.5/', 'random/1.0/'))}
        for rate in rates:
            label = f'{rate}_seed42'
            path = f'{base}/seeds/fresh_audio/{label}/report.json'
            require_job(label + '_fresh_hinted_audio', [(path, 'audio')])
            comparisons[label + '_hinted_audio'] = documents.get(path, {}).get('summary', {})
        scope = 'Training RNG/masking/order seeds from one saved initializer; independent initialization is not replicated. Audio covers seed 42 with B hints, not multi-seed A-only responses.'
    else:
        candidates = protocol.get('fresh_conditional', {}).get('masked_candidates', {
            'balanced_step1600': None, 'ce_control_step160': None, 'sampled_ctc_step160': None})
        for label in candidates:
            out = f'{base}/fresh/{label}'
            require_job(label + '_fresh_controls', [(out + '/val_controls.json', 'controls')])
            require_job(label + '_fresh_audio', [(out + '/audio/report.json', 'audio')])
            comparisons[label] = {'controls': compact_metrics(documents.get(out + '/val_controls.json', {})),
                                 'audio': documents.get(out + '/audio/report.json', {}).get('summary', {})}
        modes = ['greedy']
        if protocol.get('ar_categorical_requested_before_fresh_results') or evidence.paths([f'{base}/fresh/ar_categorical/*.json']):
            modes.append('categorical')
        for mode in modes:
            out = f'{base}/fresh/ar_{mode}'
            evaluation = ar_evaluation(evidence, mode, 'conditional', mode, True, base_override=out)
            failures.extend(evaluation['failed_checks'])
            require_job('ar_' + mode + '_fresh', [(out + '/units_report.json', 'ar_units'), (out + '/audio/report.json', 'audio')])
            data = documents.get(out + '/units_report.json', {})
            if (evaluation['status'] != 'completed' or not check_cases(data, ar=True)
                    or data.get('teacher_forced', {}).get('examples') != len(fresh_ids)):
                pending.append('ar_' + mode + ': complete fresh teacher/free evaluation and frozen audit')
            comparisons['ar_' + mode] = {key: evaluation[key] for key in ('teacher_forced', 'free_running', 'audio')}
        scope = 'Fresh development conversations outside the recorded detailed exposure inventory; not an untouched final test. Human relevance/empathy remain unmeasured.'
    status = ('failed' if failures else 'completed' if not pending else 'partial' if paths else 'pending')
    return {'phase': phase, 'status': status, 'evidence_paths': paths, 'failed_checks': failures,
        'pending_evidence': sorted(set(pending)), 'protocol_path': protocol_path,
        'protocol_sha256': protocol_hash, 'controller_reported_snapshot': controller,
        'jobs': jobs, 'comparisons': comparisons, 'scope': scope,
        'source_and_input_hashes_independently_recomputed': False,
        'production_candidate_promoted': False, 'quality_status': 'not_established'}


def build_ledger(root, listening_packages=None):
    evidence = Evidence(root)
    if not evidence.root.is_dir():
        raise ValueError('Experiment tree does not exist')
    external_artifacts, external_checks, external_states = external_listening(evidence, listening_packages)
    review_paths = {'initial': 'initial_review.json', 'recipe': 'recipe_review.json', 'prior': 'prior_review.json',
                    'conditional': 'conditional_review.json', 'waveform': 'waveform_review.json',
                    'representation': 'representation_review.json', 'ar_prior': 'ar_prior_review.json',
                    'ar_conditional': 'ar_conditional_review.json', 'ar': 'ar_review.json'}
    reviews = {}
    for label, relative in review_paths.items():
        if evidence.paths([relative]):
            _, data = evidence.read(relative)
            reviews[label] = {'path': relative, 'reported_decision': data}
    steps = []
    for number, groups in requirements().items():
        required, references, failures = [], set(), []
        for group in groups:
            matches = evidence.paths(group['paths'])
            checks = []
            for relative in matches:
                valid, missing = evidence.validate(relative, group['kind'])
                checks.append({'path': relative, 'valid': valid, 'missing': missing})
                references.add(relative)
                failures.extend({'path': relative, 'reason': reason} for reason in evidence.cache[relative]['failures'])
            if number == 13 and group['kind'] == 'listening':
                checks.extend(external_checks)
                references.update(entry['path'] for entry in external_artifacts)
                failures.extend({'path': entry['path'], 'reason': reason}
                                for entry in external_artifacts for reason in entry['failures'])
            # Wildcard audits must all pass; aliases may have one valid result. / 여러 감사는 모두 통과하고 별칭은 하나면 됩니다.
            satisfied = bool(checks) and (all(row['valid'] for row in checks) if group['kind'] == 'audit'
                                          else any(row['valid'] for row in checks))
            endpoint_audit = None
            if any(pattern.endswith('audit_step_*.json') for pattern in group['paths']):
                base = Path(group['paths'][0]).parent
                completion = (base / 'complete.json').as_posix()
                if evidence.paths([completion]):
                    _, endpoint = evidence.read(completion)
                    updates = (endpoint or {}).get('updates', (endpoint or {}).get('steps'))
                    if isinstance(updates, int):
                        endpoint_audit = (base / f'audit_step_{updates:04d}.json').as_posix()
                satisfied = satisfied and endpoint_audit in matches
            required.append({**group, 'satisfied': satisfied, 'observed': checks})
            if endpoint_audit:
                required[-1]['endpoint_audit_required'] = endpoint_audit
        status = 'completed' if all(item['satisfied'] for item in required) else ('partial' if references else 'pending')
        notes = []
        if failures:
            status = 'failed'
        if number == 5 and not references:
            decision = (reviews.get('initial', {}).get('reported_decision') or {}).get('step5_decision')
            if isinstance(decision, str) and 'defer' in decision.lower():
                status = 'deferred'
                notes.append('Recorded gate decision: ' + decision)
        if number == 8 and evidence.paths(['step08_preflight_on_prior.json']):
            entry, _ = evidence.read('step08_preflight_on_prior.json')
            references.add(entry['path'])
            if entry['failures']:
                notes.append('Historical prior preflight failed; kept separately from the later conditional preflight.')
                if not required[0]['satisfied']:
                    failures.extend({'path': entry['path'], 'reason': why} for why in entry['failures'])
                    status = 'failed'
        if number == 9:
            notes.append('Controlled corruptions reuse step 6 audio; no acoustic fine-tuning is inferred.')
        if number == 10:
            notes.append('Silence and ASR-rate metrics are proxies; sentence completion and truncation need listening.')
        if number == 11:
            notes.append('Zero affect is an ablation; pitch/energy/rate changes do not validate unsupported emotion labels.')
        if number == 12:
            notes.append('Sampling/self-hint/acoustic compatibility diagnostics do not imply a newly trained representation.')
        branch = None
        if number == 12:
            branch = ar_branch(evidence)
            decision = reviews.get('representation', {}).get('reported_decision') or {}
            branch['requested_in_recorded_review'] = decision.get('run_ar_feasibility') is True
            references.update(branch['evidence_paths'])
            failures.extend(branch['failed_checks'])
            if branch['evidence_paths'] or branch['requested_in_recorded_review']:
                status = ('failed' if failures else 'completed' if status == 'completed'
                    and branch['status'] == 'completed' else 'partial' if references else 'pending')
            notes.append('Isolated AR adaptation is tracked separately. Teacher-forced prefix scores do not establish free-response quality; categorical decoding is optional until observed.')
        if number == 13:
            status = ('failed' if failures else 'prepared' if all(item['satisfied'] for item in required)
                      else 'partial' if references else 'pending')
            notes.append('A prepared form is not completed human listening. Multiple-seed confirmation and untouched-test protocol still require explicit review.')
        steps.append({'step': number, 'name': TITLES[number - 1], 'status': status,
            'quality_status': 'not_established', 'requirements': required,
            'pending_evidence': [item['label'] for item in required if not item['satisfied']],
            'evidence_paths': sorted(references), 'failed_checks': failures, 'notes': notes})
        if branch is not None:
            steps[-1]['autoregressive_feasibility'] = branch
            if branch['evidence_paths'] or branch['requested_in_recorded_review']:
                steps[-1]['pending_evidence'].extend('AR ' + item for item in branch['pending_evidence'])
    local_listening = {'root': str(evidence.root / 'step13_listening'), **listening_status(evidence)}
    listening = merge_listening([local_listening, *external_states])
    steps[-1]['human_listening'] = listening
    confirmation = {phase: confirmation_phase(evidence, phase) for phase in ('seeds', 'fresh')}
    steps[-1]['computational_confirmation'] = confirmation
    steps[-1]['final_test'] = {'status': 'deferred', 'test_split_evaluated': False,
        'reason': 'Development confirmation does not authorize a final-test quality claim; candidate qualification and recipe lock remain required.'}
    for phase, detail in confirmation.items():
        steps[-1]['evidence_paths'] = sorted(set(steps[-1]['evidence_paths']) | set(detail['evidence_paths']))
        steps[-1]['failed_checks'].extend(detail['failed_checks'])
        if detail['status'] != 'completed':
            steps[-1]['pending_evidence'].append(('training-RNG seed replication' if phase == 'seeds'
                                                 else 'fresh development confirmation') + ': ' + detail['status'])
    if steps[-1]['failed_checks']:
        steps[-1]['status'] = 'failed'
    elif steps[-1]['status'] == 'pending' and steps[-1]['evidence_paths']:
        steps[-1]['status'] = 'partial'
    steps[-1]['pending_evidence'].extend(['untouched final-test evaluation (deferred)',
                                         'independently reviewed human listening ratings'])
    controller = None
    if evidence.paths(['status.json']):
        _, controller = evidence.read('status.json')
    return {'schema': 'planner_repair_evidence_ledger_v1', 'generated_utc': datetime.now(timezone.utc).isoformat(),
        'root': str(evidence.root), 'snapshot_only': True, 'network_or_gpu_used': False,
        'independent_confirmation_complete': False, 'human_listening_status': listening['status'],
        'computational_confirmation_complete': all(row['status'] == 'completed' for row in confirmation.values()),
        'final_test_status': 'deferred',
        'controller_reported_snapshot': controller, 'status_counts': dict(Counter(row['status'] for row in steps)),
        'limitations': list(LIMITATIONS), 'steps': steps, 'reviews': reviews,
        'artifacts': [evidence.cache[key] for key in sorted(evidence.cache)] + external_artifacts}


def markdown(ledger):
    def clean(value):
        return str(value).replace('|', '\\|').replace('\n', ' ')

    lines = ['# Planner repair evidence snapshot', '',
        'This report inventories saved computational evidence. It does not promote a model or certify good response audio.', '',
        f'Generated: {ledger["generated_utc"]}. Source tree: `{ledger["root"]}`.', '',
        '| Step | Experiment | Observed status | Missing evidence groups |', '|---|---|---|---|']
    for row in ledger['steps']:
        lines.append(f'| {row["step"]} | {row["name"]} | **{row["status"]}** | {len(row["pending_evidence"])} |')
    lines.extend(['', 'Human listening: **' + ledger['human_listening_status'] + '**. Independent confirmation remains pending.', ''])
    artifacts = {row['path']: row for row in ledger['artifacts']}
    for row in ledger['steps']:
        lines.extend([f'## {row["step"]}. {row["name"]}', '', f'Status: **{row["status"]}**. Quality claim: not established.', ''])
        for note in row['notes']:
            lines.append(note + '\n')
        if 'computational_confirmation' in row:
            lines.extend(['| Confirmation | Computational status | Completed bound jobs |', '|---|---|---|'])
            for phase, value in row['computational_confirmation'].items():
                completed = sum(job['completed'] for job in value['jobs'].values())
                lines.append(f'| {phase} | {value["status"]} | {completed}/{len(value["jobs"])} |')
            lines.append('')
            lines.extend(value['scope'] + '\n' for value in row['computational_confirmation'].values())
            lines.append('Final test: **deferred**. Human ratings and quality qualification remain separate.\n')
            receipts = sum('/receipts/' in path for path in row['evidence_paths'])
            if receipts:
                lines.append(f'{receipts} receipt/definition artifacts and their hashes are indexed in ledger.json.\n')
        if 'autoregressive_feasibility' in row:
            branch = row['autoregressive_feasibility']
            compute = branch['additional_adaptation_compute']
            lines.extend([f'AR feasibility branch: **{branch["status"]}**; no production promotion.', '',
                'Additional adaptation updates: ' + str(compute['completed_updates_reported'])
                + ' reported completed; ' + str(compute['completed_updates_with_required_evidence'])
                + ' have the required endpoint evidence. Source masked-prior training is excluded. '
                + compute['comparison_limit'], '',
                '| AR phase | Status | Planned updates | Completed updates reported | Latest teacher-forced accuracy |',
                '|---|---|---|---|---|'])
            for phase, value in branch['training'].items():
                latest = value['teacher_forced'].get('latest') or {}
                accuracy = latest.get('teacher_forced_unit_accuracy')
                score = f'{accuracy:.2%}' if isinstance(accuracy, (float, int)) else 'pending'
                lines.append(f'| {phase} | {value["status"]} | {value["planned_updates"] or "pending"} | '
                    f'{value["completed_updates_reported"]} | {score} |')
            lines.extend(['', 'Teacher-forced scores use the correct B prefix and oracle B length. '
                          'The following free-running scores use no B unit inputs; prior-only generation still uses oracle B length.', '',
                '| Free evaluation | Status | Condition | Unit edit distance / reference unit | Adjacent repeat fraction |',
                '|---|---|---|---|---|'])
            for name, value in branch['evaluations'].items():
                summary = value['free_running']['summary']
                if not summary:
                    lines.append(f'| {name} ({value["sampling_mode"]}) | {value["status"]} | pending | pending | pending |')
                for condition, metrics in summary.items():
                    lines.append(f'| {name} ({value["sampling_mode"]}) | {value["status"]} | {clean(condition)} | '
                        f'{clean(metrics.get("unit_edit_distance_per_reference_unit", "pending"))} | '
                        f'{clean(metrics.get("adjacent_repeat_fraction", "pending"))} |')
            if branch['categorical_matches_greedy_checkpoint'] is False:
                lines.extend(['', 'Greedy and categorical checkpoint identities do not match or are missing; do not interpret them as a paired decoding comparison.'])
            lines.append('')
        for path in row['evidence_paths']:
            if '/receipts/' in path:
                continue
            artifact = artifacts[path]
            target = (Path(ledger['root']) / path).as_posix()
            lines.append(f'- [{path}](<{target}>), SHA256 `{artifact["sha256"]}`.')
            metrics = artifact['metrics']
            endpoint = {key: value for key, value in metrics.items() if not isinstance(value, (dict, list))}
            if endpoint:
                lines.append('  Recorded values: ' + clean(json.dumps(endpoint, ensure_ascii=False)) + '.')
            summary = metrics.get('summary', {})
            for name, value in summary.items():
                if not isinstance(value, dict):
                    continue
                if 'mean_reference_wer' in value:
                    lines.append(f'  {clean(name)}: ASR reference WER {value["mean_reference_wer"]:.4f} (n={value.get("count", "?")}).')
                elif name in ('random/0.5/null_prior/first_pass', 'random/0.5/nearest_copy'):
                    metric = value.get('all_hidden', {}).get('hidden_only_accuracy')
                    if metric is not None:
                        lines.append(f'  {name}: hidden-unit accuracy {100 * metric:.2f}% (oracle hints/length).')
        if row['failed_checks']:
            lines.extend(['', 'Recorded failed checks:'])
            lines.extend('- ' + clean(item['path'] + ': ' + item['reason']) for item in row['failed_checks'])
        if row['pending_evidence']:
            lines.extend(['', 'Pending evidence:'])
            lines.extend('- ' + clean(item) for item in row['pending_evidence'])
        lines.append('')
    lines.extend(['## Limits of this snapshot', ''])
    lines.extend('- ' + value for value in ledger['limitations'])
    lines.extend(['', 'Detailed metrics, recorded review decisions and hashes are in ledger.json. Source reports remain the full evidence.', ''])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--listening-package', type=Path, action='append', default=[],
        help='Optional external package directory containing answer_key.json and rater/; repeatable.')
    args = parser.parse_args()
    ledger = build_ledger(args.root, args.listening_package)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'ledger.json').write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (args.output / 'results.md').write_text(markdown(ledger), encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'status_counts': ledger['status_counts'],
                      'human_listening_status': ledger['human_listening_status']}), flush=True)


if __name__ == '__main__':
    main()
