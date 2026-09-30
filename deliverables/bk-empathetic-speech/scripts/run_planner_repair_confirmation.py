"""Run bounded replication and fresh-development checks. / 제한된 반복 실험과 새 개발 검사를 실행합니다."""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
REPAIR = Path('outputs/planner_repair_v1')
ROOT = REPAIR / 'step13_confirmation'
MANIFEST = Path('outputs/bk_quality_prepared/manifest.json')
SELECTION = Path('outputs/planner_stages_v1/selection.json')
FRESH = REPAIR / 'step13_fresh_v2/confirmation_selection.json'
AUDIO = ROOT / 'fresh_audio_selection.json'
PRIOR = Path('outputs/planner_stages_v1/unit_prior/best.pt')
SEED42 = {'low_lr_seed42': REPAIR / 'step04_recipes/legacy_low_lr/step_0640.pt',
          'high_lr_seed42': REPAIR / 'step04_recipes/legacy/step_0640.pt'}
MASKED = {'balanced_step1600': REPAIR / 'step07_conditional/balanced/step_1600.pt',
          'ce_control_step160': REPAIR / 'step08_waveform/ce_control/step_0160.pt',
          'sampled_ctc_step160': REPAIR / 'step08_waveform/sampled_ctc/step_0160.pt'}
AR = REPAIR / 'step12_ar/conditional/step_1600.pt'
ENDPOINTS = (160, 320, 480, 640)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode('utf-8')


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def fingerprints(paths):
    return {str(Path(path)): file_hash(path) for path in sorted(set(map(Path, paths)), key=str)}


def verify_hashes(expected):
    for path, checksum in expected.items():
        if not Path(path).is_file() or file_hash(path) != checksum:
            raise ValueError('Bound input/output changed or disappeared: ' + path)


def read_json(path):
    return json.loads(Path(path).read_bytes())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_bytes(json_bytes(value))
    temporary.replace(path)


def freeze_json(path, value):
    path = Path(path)
    expected = json_bytes(value)
    if path.exists():
        if path.read_bytes() != expected:
            raise ValueError('Immutable protocol/recipe differs: ' + str(path))
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(expected)
    return False


def prepare_selection():
    manifest, original, fresh = map(read_json, (MANIFEST, SELECTION, FRESH))
    manifest_hash = file_hash(MANIFEST)
    lookup = {row['path']: row for row in manifest['records']}
    if len(lookup) != len(manifest['records']):
        raise ValueError('Manifest paths must be unique')
    for name, selected, count in [('original', original, 128), ('fresh', fresh, 32)]:
        if selected.get('manifest_sha256') != manifest_hash:
            raise ValueError(name + ' selection has another manifest')
        if len(selected['train']) != 2048 or len(selected['val']) != count:
            raise ValueError(name + ' selection has an unexpected training/development count')
        ids_by_split = {}
        for split in ('train', 'val'):
            paths = selected[split]
            if len(paths) != len(set(paths)) or any(path not in lookup for path in paths):
                raise ValueError('Missing or repeated selected paths')
            rows = [lookup[path] for path in paths]
            ids = [row['conversation_id'] for row in rows]
            if len(ids) != len(set(ids)) or any(row['split'] != split for row in rows):
                raise ValueError('Selection repeats conversations or changes original splits')
            if set(ids) != set(selected['conversations'][split]) or len(ids) != len(selected['conversations'][split]):
                raise ValueError('Selection conversation metadata differs')
            if any(row['style_id'] != 0 or row['speaker_id'] != 1 for row in rows):
                raise ValueError('Expected style 0 and voice group 1')
            ids_by_split[split] = set(ids)
        if ids_by_split['train'] & ids_by_split['val']:
            raise ValueError('Training/development conversation overlap')
    if fresh['train'] != original['train'] or fresh['conversations']['train'] != original['conversations']['train']:
        raise ValueError('Fresh evaluation must preserve exact original training membership/order')
    if set(fresh['conversations']['val']) & set(original['conversations']['val']):
        raise ValueError('Fresh conversations overlap original development')
    audit = read_json(FRESH.parent / 'confirmation_data_audit.json')
    exposure = read_json(FRESH.parent / 'used_validation_exposure.json')
    if (audit.get('status') != 'selected' or audit.get('eligible_selected') != 32 or
            audit.get('confirmation_selection_sha256') != file_hash(FRESH) or
            audit.get('exclusion_inventory_sha256') != file_hash(FRESH.parent / 'used_validation_exposure.json') or
            not audit.get('report_snapshot_verified_unchanged')):
        raise ValueError('Fresh selection provenance audit is incomplete or changed')
    if set(fresh['conversations']['val']) & set(exposure['exposed_validation_ids']):
        raise ValueError('Fresh IDs appear in the recorded previous exposure inventory')
    audio = copy.deepcopy(fresh)
    audio['val'] = fresh['val'][:8]
    audio['conversations']['val'] = [lookup[path]['conversation_id'] for path in audio['val']]
    audio['selection_note'] = ('First eight paths in the immutable fresh32 selection; fixed before evaluation. '
                               'No ranking or outcome-based filtering. Development confirmation, not a final test.')
    freeze_json(AUDIO, audio)
    return fresh, audio


def source_files():
    explicit = ['scripts/run_planner_repair_confirmation.py', 'train_recovery.py', 'train_full.py',
        'prepare_quality.py', 'scripts/audit_planner_checkpoint.py', 'scripts/check_planner_learning_history.py',
        'scripts/check_unit_audio_robustness.py', 'scripts/evaluate_planner_controls.py',
        'scripts/diagnose_quality.py', 'scripts/diagnose_speech.py', 'scripts/train_planner_ar.py',
        'scripts/check_planner_hints.py', 'scripts/planner_sampling.py', 'scripts/check_unit_vocabulary_audio.py']
    paths = list(map(Path, explicit))
    for directory in ('model', 'dataset', 'utils'):
        paths.extend(Path(directory).rglob('*.py'))
    return paths


def protocol(phase, fresh, audio, ar_categorical=False):
    initializers = [PRIOR, *SEED42.values()] if phase == 'seeds' else [*MASKED.values(), AR]
    inputs = [MANIFEST, SELECTION, FRESH, AUDIO, FRESH.parent / 'confirmation_data_audit.json',
              FRESH.parent / 'used_validation_exposure.json', *initializers]
    return {'version': 1, 'phase': phase, 'python': sys.executable,
        'ar_categorical_requested_before_fresh_results': bool(ar_categorical) if phase == 'fresh' else False,
        'source_sha256': fingerprints(source_files()), 'input_sha256': fingerprints(inputs),
        'training_gpus': [0, 1, 2, 3], 'batch_per_gpu': 16, 'evaluation_device': 'cuda:0',
        'training_selection': str(SELECTION), 'training_examples': 2048, 'training_validation_examples': 128,
        'fresh_selection': str(FRESH), 'fresh_validation_examples': 32,
        'fresh_audio_selection': str(AUDIO), 'fresh_audio_paths': audio['val'],
        'fresh_conversation_ids': fresh['conversations']['val'],
        'seed_replication': {'seeds': [42, 43, 44], 'new_training_seeds': [43, 44],
            'learning_rates': {'low_lr': .0001, 'high_lr': .001}, 'total_updates': 640,
            'objective': 'legacy unit prior', 'semantic_steps': 8, 'teacher_weight': 0,
            'evaluation_and_save_every': 160, 'frozen_audit_endpoints': list(ENDPOINTS),
            'shared_initializer': str(PRIOR), 'initialization_seeds_replicated': False,
            'interpretation': 'Training RNG/masking/order replication from one identical saved initializer; not independent initialization seeds.',
            'fresh_gaps': {'count': 32, 'mask_seeds': [42, 43], 'ratios': [.25, .5, .75, 1.]},
            'fresh_audio': {'training_seeds': [42], 'count': 8, 'hidden_ratio': .5,
                'planner_condition': 'null_prior', 'interpretation': 'B-hinted reconstruction; not multi-seed audio or A-only response evidence.'}},
        'fresh_conditional': {'masked_candidates': {key: str(path) for key, path in MASKED.items()},
            'controls_count': 32, 'masked_audio_steps': 8, 'audio_count': 8,
            'ar_checkpoint': str(AR), 'ar_teacher_and_free_count': 32, 'ar_sampling': 'greedy',
            'optional_ar_categorical': {'activation': '--ar-categorical', 'temperature': .8,
                                      'top_k': 20, 'sampling_seed': 42, 'separate_report': True}},
        'test_split_evaluated': False, 'automatic_quality_promotion': False,
        'limitations': ['Fresh means outside the recorded detailed development exposure, not a pristine final test.',
            'The original development split may have had earlier aggregate exposure.',
            'Exact-reference unit accuracy and ASR WER do not establish response relevance or empathy.',
            'Human listening ratings are not generated by this runner.']}


def report_audio_files(path):
    paths = []
    value = read_json(path)
    for row in value.get('examples', []):
        for audio in row.get('paths', {}).values():
            if isinstance(audio, dict) and isinstance(audio.get('file'), str):
                relative = Path(audio['file'])
                candidate = (path.parent / relative).resolve()
                if relative.is_absolute() or relative.drive or '..' in relative.parts or not candidate.is_relative_to(path.parent.resolve()):
                    raise ValueError('Report audio path escapes output: ' + str(path))
                if candidate.suffix.lower() != '.wav':
                    raise ValueError('Expected WAV report audio')
                paths.append(candidate)
    return paths


class Runner:
    def __init__(self, phase, definition, lock_fd):
        self.phase, self.definition, self.lock_fd = phase, definition, lock_fd
        self.protocol_hash = hashlib.sha256(json_bytes(definition)).hexdigest()
        freeze_json(ROOT / ('protocol_' + phase + '.json'), definition)
        self.receipts = ROOT / 'receipts' / phase
        self.receipts.mkdir(parents=True, exist_ok=True)

    def verify_protocol(self):
        verify_hashes(self.definition['source_sha256'])
        verify_hashes(self.definition['input_sha256'])

    def status(self, stage, **details):
        write_json(ROOT / ('status_' + self.phase + '.json'), {
            'phase': self.phase, 'stage': stage, 'pid': os.getpid(), 'updated_utc': stamp(),
            'protocol_sha256': self.protocol_hash, 'production_candidate_promoted': False,
            'test_split_evaluated': False, **details})

    def run(self, name, command, outputs, inputs, validate, resume=None, new_output=None):
        self.verify_protocol()
        command, outputs = list(map(str, command)), list(map(Path, outputs))
        binding = {'protocol_sha256': self.protocol_hash, 'logical_command': command,
                   'input_sha256': fingerprints(inputs), 'outputs': list(map(str, outputs))}
        binding_path = self.receipts / (name + '_definition.json')
        existed = binding_path.exists()
        if not existed and (any(path.exists() for path in outputs) or
                (resume is not None and Path(resume).exists()) or
                (new_output is not None and Path(new_output).exists())):
            raise ValueError('Untracked preexisting job outputs cannot acquire a new definition: ' + name)
        freeze_json(binding_path, binding)
        completion = self.receipts / (name + '_complete.json')
        if completion.exists():
            saved = read_json(completion)
            if saved['definition_sha256'] != file_hash(binding_path):
                raise ValueError('Job completion has another definition: ' + name)
            verify_hashes(saved['output_sha256'])
            validate()
            self.status(name + '_verified_cached')
            return
        adopted = existed and all(path.is_file() for path in outputs)
        if not adopted:
            if new_output is not None and Path(new_output).exists():
                raise ValueError('Incomplete AR evaluation cannot overwrite its directory; preserve and inspect it before retry: ' + str(new_output))
            actual = list(command)
            if resume is not None and Path(resume).is_file():
                if not existed:
                    raise ValueError('Untracked training checkpoint cannot be resumed')
                position = actual.index('--initialize')
                actual[position:position + 2] = ['--resume', str(resume)]
            attempts = self.receipts / (name + '_attempts.jsonl')
            with attempts.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'started_utc': stamp(), 'command': actual,
                    'resume_checkpoint_sha256': file_hash(resume) if resume and Path(resume).is_file() else None}) + '\n')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
            with (self.receipts / (name + '.log')).open('a', encoding='utf-8') as log:
                # Children retain the shared lock if the controller exits. / 제어기가 종료되어도 자식은 공유 잠금을 유지합니다.
                child = subprocess.Popen(actual, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         pass_fds=(self.lock_fd,))
                self.status(name, child_pid=child.pid, command=actual)
                code = child.wait()
            if code:
                raise RuntimeError(f'{name} exited {code}; inspect its receipt log')
        if not all(path.is_file() for path in outputs):
            raise RuntimeError('Missing completion artifacts: ' + name)
        validate()
        self.verify_protocol()
        verify_hashes(binding['input_sha256'])
        files = list(outputs)
        for output in outputs:
            if output.name == 'report.json':
                files.extend(report_audio_files(output))
        write_json(completion, {'definition_sha256': file_hash(binding_path), 'completed_utc': stamp(),
            'adopted_complete_artifacts_after_interruption': adopted, 'output_sha256': fingerprints(files),
            'quality_passed': False, 'automatic_quality_promotion': False})
        self.status(name + '_complete', child_pid=None)


def validate_training(out, seed, lr):
    import torch
    complete = read_json(out / 'complete.json')
    if complete.get('steps') != 640 or complete.get('world_size') != 4 or complete.get('stage') != 'planner':
        raise ValueError('Training completion differs from the prescribed budget')
    for step in ENDPOINTS:
        value = torch.load(out / f'step_{step:04d}.pt', map_location='cpu', weights_only=True)
        recipe = value.get('metadata', {}).get('recovery_recipe', {})
        if (value.get('recovery_step') != step or value.get('world_size') != 4 or
                recipe.get('seed', 42) != seed or recipe.get('lr') != lr or
                recipe.get('steps') != 640 or recipe.get('batch_size') != 16 or
                not recipe.get('unit_prior') or recipe.get('controlled_unit_objective') or
                recipe.get('manifest_sha256') != file_hash(MANIFEST) or
                recipe.get('selection_sha256') != file_hash(SELECTION)):
            raise ValueError('Saved training checkpoint does not match the fixed recipe')
        del value


def validate_audit(path):
    if not read_json(path).get('passed'):
        raise ValueError('Frozen checkpoint audit failed: ' + str(path))


def validate_cases(path, selection, variant=None):
    report = read_json(path)
    rows = report.get('examples', [])
    expected = set(read_json(selection)['conversations']['val'])
    actual = [row['conversation_id'] for row in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Evaluation conversation set differs: ' + str(path))
    if variant is not None and any(variant not in row.get('paths', {}) for row in rows):
        raise ValueError('Evaluation omitted the required audio variant')


def run_seeds(runner):
    common = ['--manifest', MANIFEST, '--selection', SELECTION]
    checkpoints = dict(SEED42)
    for seed in (43, 44):
        for label, rate in [('low_lr', .0001), ('high_lr', .001)]:
            name = f'{label}_seed{seed}'
            out = ROOT / 'seeds' / name
            command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                'train_recovery.py', *common, '--initialize', PRIOR, '--output', out, '--stage', 'planner',
                '--unit-prior', '--unit-objective', 'legacy', '--steps', 640, '--evaluate-every', 160,
                '--batch-size', 16, '--semantic-steps', 8, '--teacher-weight', 0, '--lr', rate,
                '--seed', seed, '--save-candidates']
            endpoints = [out / f'step_{step:04d}.pt' for step in ENDPOINTS]
            runner.run(name, command, [out / 'complete.json', *endpoints], [MANIFEST, SELECTION, PRIOR],
                lambda out=out, seed=seed, rate=rate: validate_training(out, seed, rate), resume=out / 'last.pt')
            for endpoint in endpoints:
                audit = out / ('audit_' + endpoint.stem + '.json')
                runner.run(name + '_' + endpoint.stem + '_audit', [sys.executable,
                    'scripts/audit_planner_checkpoint.py', '--initialize', PRIOR, '--checkpoint', endpoint,
                    '--output', audit, '--unit-prior'], [audit], [PRIOR, endpoint],
                    lambda audit=audit: validate_audit(audit))
            checkpoints[name] = out / 'step_0640.pt'
    gap_out = ROOT / 'seeds/gaps'
    arguments = [item for label, path in checkpoints.items() for item in ('--checkpoint', label + '=' + str(path))]
    def check_gaps():
        report = read_json(gap_out / 'summary.json')
        if set(report['checkpoints']) != set(checkpoints):
            raise ValueError('The gap report must contain all six endpoints')
        for label in checkpoints:
            validate_cases(gap_out / (label + '.json'), FRESH)
    runner.run('six_endpoint_fresh_gaps', [sys.executable, 'scripts/check_planner_learning_history.py',
        '--manifest', MANIFEST, '--selection', FRESH, *arguments, '--output', gap_out,
        '--split', 'val', '--count', 32, '--mask-seeds', 42, 43, '--ratios', .25, .5, .75, 1.,
        '--device', 'cuda:0'], [gap_out / 'summary.json', *[gap_out / (label + '.json') for label in checkpoints]],
        [MANIFEST, FRESH, *checkpoints.values()], check_gaps)
    for label, checkpoint in SEED42.items():
        out = ROOT / 'seeds/fresh_audio' / label
        runner.run(label + '_fresh_hinted_audio', [sys.executable, 'scripts/check_unit_audio_robustness.py',
            '--manifest', MANIFEST, '--selection', AUDIO, '--checkpoint', checkpoint, '--output', out,
            '--planner-condition', 'null_prior', '--ratios', .5, '--count', 8, '--device', 'cuda:0'],
            [out / 'report.json'], [MANIFEST, AUDIO, checkpoint],
            lambda out=out: validate_cases(out / 'report.json', AUDIO, 'iterative_hidden_050'))


def run_fresh(runner, ar_categorical=False):
    for label, checkpoint in MASKED.items():
        out = ROOT / 'fresh' / label
        controls = out / 'val_controls.json'
        runner.run(label + '_fresh_controls', [sys.executable, 'scripts/evaluate_planner_controls.py',
            '--manifest', MANIFEST, '--selection', FRESH, '--checkpoint', checkpoint, '--output', controls,
            '--split', 'val', '--count', 32, '--device', 'cuda:0'], [controls], [MANIFEST, FRESH, checkpoint],
            lambda controls=controls: validate_cases(controls, FRESH))
        audio = out / 'audio'
        runner.run(label + '_fresh_audio', [sys.executable, 'scripts/diagnose_quality.py',
            '--manifest', MANIFEST, '--selection', AUDIO, '--checkpoint', checkpoint, '--output', audio,
            '--split', 'val', '--count', 8, '--steps', 8, '--device', 'cuda:0'],
            [audio / 'report.json'], [MANIFEST, AUDIO, checkpoint],
            lambda audio=audio: validate_cases(audio / 'report.json', AUDIO, 'predicted_length_steps_8'))
    modes = ('greedy', 'categorical') if ar_categorical else ('greedy',)
    for mode in modes:
        out = ROOT / 'fresh' / ('ar_' + mode)
        command = [sys.executable, 'scripts/train_planner_ar.py', '--phase', 'evaluate', '--checkpoint', AR,
            '--manifest', MANIFEST, '--selection', FRESH, '--audio-selection', AUDIO, '--output', out,
            '--free-count', 32, '--batch-size', 16, '--conditional-cache', 'inputs_units',
            '--sampling', mode, '--seed', 42, '--device', 'cuda:0']
        if mode == 'categorical':
            command += ['--temperature', .8, '--top-k', 20, '--sampling-seed', 42]
        def check_ar(out=out):
            value = read_json(out / 'units_report.json')
            if not value.get('frozen_audit', {}).get('passed'):
                raise ValueError('AR evaluation lacks a passing frozen audit')
            if value.get('teacher_forced', {}).get('examples') != 32:
                raise ValueError('AR teacher-forced evaluation must use all fresh 32 conversations')
            free = value['free_running']['examples']
            if len(free) != 32 or {row['conversation_id'] for row in free} != set(read_json(FRESH)['conversations']['val']):
                raise ValueError('AR free-running evaluation must use all fresh 32 conversations')
            validate_cases(out / 'audio/report.json', AUDIO, 'predicted_length_ar')
        runner.run('ar_' + mode + '_fresh', command, [out / 'units_report.json', out / 'audio/report.json'],
            [MANIFEST, FRESH, AUDIO, AR], check_ar, new_output=out)


def main():
    parser = argparse.ArgumentParser(description='Bounded replication and fresh-development confirmation; no automatic promotion.')
    parser.add_argument('--phase', choices=('seeds', 'fresh'), required=True)
    parser.add_argument('--ar-categorical', action='store_true', help='Also run a separate categorical AR diagnostic in the fresh phase.')
    args = parser.parse_args()
    if args.ar_categorical and args.phase != 'fresh':
        parser.error('--ar-categorical applies only to --phase fresh')
    os.chdir(REPO)
    # One shared Linux lock prevents overlapping GPU controllers. / Linux 공유 잠금으로 GPU 제어기 중복을 막습니다.
    import fcntl
    lock_path = Path('outputs/quality_recovery/controller.lock')
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fresh, audio = prepare_selection()
        runner = Runner(args.phase, protocol(args.phase, fresh, audio, args.ar_categorical), lock.fileno())
        try:
            runner.status('protocol_frozen_before_execution')
            if args.phase == 'seeds':
                run_seeds(runner)
            else:
                run_fresh(runner, args.ar_categorical)
            runner.status('complete_pending_review', child_pid=None, quality_passed=False)
        except Exception as error:
            runner.status('failed', error=str(error), child_pid=None)
            raise


if __name__ == '__main__':
    main()
