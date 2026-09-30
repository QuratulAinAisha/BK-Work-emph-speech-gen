"""Matched planner-memory experiments. / 조건을 맞춘 계획기 메모리 실험."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_quality import atomic_json


ARMS = ('fused', 'resampled_speech', 'native_speech')
STEPS, INTERVAL = 320, 64
ROOT = Path('outputs/planner_memory_v1')
INITIAL = Path('outputs/planner_stages_v1/prior_finetune/best.pt')
MANIFEST = Path('outputs/bk_quality_prepared/manifest.json')
SELECTION = Path('outputs/planner_stages_v1/selection.json')
AUDIO = Path('outputs/planner_stages_v1/audio_selection.json')
EXPECTED_INITIAL = '7d4f86662197ed128635801a6b1319639eef5801dfe177500746262b367c5bd3'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def immutable_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Experiment identity changed: {path}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, value)


def experiment_plan():
    if digest(INITIAL) != EXPECTED_INITIAL:
        raise ValueError('The shared initializer changed')
    confirmation = ROOT / 'confirmation_selection.json'
    return {'version': 1, 'arms': list(ARMS), 'primary_comparison': ['native_speech', 'resampled_speech'],
        'extra_training_control': 'fused', 'initial_checkpoint': str(INITIAL),
        'initial_sha256': digest(INITIAL), 'manifest_sha256': digest(MANIFEST),
        'selection_sha256': digest(SELECTION), 'audio_selection_sha256': digest(AUDIO),
        'confirmation_selection_sha256': digest(confirmation), 'steps_per_arm': STEPS,
        'epochs_per_arm': 10, 'training_conversations': 2048, 'validation_conversations': 128,
        'world_size': 4, 'batch_per_gpu': 16, 'global_batch': 64, 'lr': .0001,
        'fresh_optimizer_each_arm': True, 'seed': 42, 'audio_every_updates': INTERVAL,
        'primary_endpoint': 'step_0320.pt in every arm; never best.pt',
        'trainable_components': ['semantic_planner'], 'frozen_duration_predictor': True,
        'unit_vocabulary_unchanged': True, 'objective': 'existing unit CE plus constant frozen-duration loss',
        'confirmation': '32 preselected validation conversations; all three fixed final checkpoints',
        'test_split_read': False,
        'acceptance': 'Review actual A-only reply content, audio and unit completion; no automatic promotion by loss or WER.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['train', 'confirmation'], required=True)
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parents[1])
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = Path('outputs/quality_recovery/controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    immutable_json(ROOT / 'plan.json', experiment_plan())
    source_paths = ['train_recovery.py', 'model/full_speech/units.py', 'model/full_speech/quality.py',
        'model/full_speech/recovery.py', 'scripts/run_planner_memory.py', 'scripts/diagnose_quality.py',
        'scripts/evaluate_planner_controls.py', 'scripts/check_planner_hints.py',
        'scripts/planner_sampling.py', 'scripts/audit_planner_checkpoint.py']
    immutable_json(ROOT / 'source_hashes.json', {path: digest(path) for path in source_paths})
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    state = {'pid': os.getpid(), 'phase': args.phase, 'gpu_ids': [0, 1, 2, 3], 'batch_per_gpu': 16,
             'production_checkpoint_replaced': False, 'candidate_promoted': False}

    def status(stage, **values):
        state.update(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(), **values)
        atomic_json(ROOT / 'status.json', state)

    def run(name, command):
        with (ROOT / (name + '.log')).open('a') as log:
            child = subprocess.Popen([str(x) for x in command], env=env, stdout=log, stderr=subprocess.STDOUT)
            status(name, child_pid=child.pid)
            if child.wait():
                raise RuntimeError(name + ' failed; inspect its log')

    def audio(name, checkpoint, selection, output, count):
        if not (output / 'report.json').exists():
            run(name, [sys.executable, 'scripts/diagnose_quality.py', '--checkpoint', checkpoint,
                '--manifest', MANIFEST, '--selection', selection, '--output', output,
                '--split', 'val', '--count', count, '--steps', '8', '--device', 'cuda:0'])

    def train_command(arm, output, check=False):
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
            'train_recovery.py', '--manifest', MANIFEST, '--selection', SELECTION, '--output', output,
            '--stage', 'planner', '--steps', STEPS, '--evaluate-every', INTERVAL, '--batch-size', 16,
            '--semantic-steps', 8, '--teacher-weight', 0, '--lr', '.0001', '--planner-memory', arm,
            '--freeze-duration', '--save-candidates']
        if check:
            return command + ['--initialize', INITIAL, '--check-only']
        command += ['--audio-selection', AUDIO]
        return command + (['--resume', output / 'last.pt'] if (output / 'last.pt').exists()
                          else ['--initialize', INITIAL])

    try:
        if args.phase == 'train':
            import torch
            torch.set_num_threads(2)
            # All arms start from identical tensors. / 모든 조건은 동일한 텐서에서 시작합니다.
            payload = torch.load(INITIAL, map_location='cpu', weights_only=True)
            for arm in ARMS:
                output = ROOT / arm
                output.mkdir(exist_ok=True)
                initial = {key: value for key, value in payload.items()
                           if key not in ('optimizer', 'scheduler', 'rank_rng_states')}
                initial['config'] = dict(payload['config'], planner_memory_mode=arm)
                initial['recovery_step'] = 0
                initial['epoch'] = -1
                if not (output / 'initial.pt').exists():
                    from train_full import atomic_save
                    atomic_save(initial, output / 'initial.pt')
                else:
                    saved = torch.load(output / 'initial.pt', map_location='cpu', weights_only=True)
                    if saved['config'] != initial['config'] or saved['state_dict'].keys() != initial['state_dict'].keys():
                        raise ValueError('Existing initialization identity differs')
                    if any(not torch.equal(value, saved['state_dict'][key]) for key, value in initial['state_dict'].items()):
                        raise ValueError('Existing initialization weights differ')
                    del saved
                preflight = ROOT / 'preflight' / arm
                if not all((preflight / f'preflight_rank{rank}.json').exists() for rank in range(4)):
                    run('preflight_' + arm, train_command(arm, preflight, check=True))
                for rank in range(4):
                    result = json.loads((preflight / f'preflight_rank{rank}.json').read_text())
                    if result['trainable'] != ['semantic_planner'] or not result['teacher_frozen']:
                        raise ValueError('Unexpected preflight update boundary')
            del payload, initial
            for arm in ARMS:
                output = ROOT / arm
                audio(arm + '_initial_audio', output / 'initial.pt', AUDIO, output / 'initial_audio', 8)
                if not (output / 'complete.json').exists():
                    run('train_' + arm, train_command(arm, output))
                for step in range(INTERVAL, STEPS + 1, INTERVAL):
                    checkpoint = output / f'step_{step:04d}.pt'
                    # Recover a save interrupted after last.pt was written. / last.pt 저장 직후 중단된 후보를 복구합니다.
                    if not checkpoint.exists():
                        last = torch.load(output / 'last.pt', map_location='cpu', weights_only=True)
                        if last['recovery_step'] != step or last['config']['planner_memory_mode'] != arm:
                            raise ValueError(f'Missing unrecoverable candidate: {checkpoint}')
                        from train_full import atomic_save
                        atomic_save(last, checkpoint)
                        del last
                    audio(arm + f'_audio_{step:04d}', checkpoint, AUDIO, output / f'audio_step_{step:04d}', 8)
                    audit = output / f'audit_step_{step:04d}.json'
                    if not audit.exists():
                        run(arm + f'_audit_{step:04d}', [sys.executable, 'scripts/audit_planner_checkpoint.py',
                            '--initialize', INITIAL, '--checkpoint', checkpoint, '--output', audit, '--planner-only'])
                    if not json.loads(audit.read_text())['passed']:
                        raise ValueError('Frozen-weight audit failed')
                checkpoint = output / 'step_0320.pt'
                for split, count in [('train', 64), ('val', 128)]:
                    report = output / f'controls_{split}.json'
                    if not report.exists():
                        run(arm + '_controls_' + split, [sys.executable, 'scripts/evaluate_planner_controls.py',
                            '--checkpoint', checkpoint, '--manifest', MANIFEST, '--selection', SELECTION,
                            '--output', report, '--split', split, '--count', count, '--device', 'cuda:0'])
                hints = output / 'hints'
                if not (hints / 'audio/report.json').exists():
                    run(arm + '_hints', [sys.executable, 'scripts/check_planner_hints.py', '--checkpoint', checkpoint,
                        '--manifest', MANIFEST, '--selection', SELECTION, '--audio-selection', AUDIO,
                        '--output', hints, '--device', 'cuda:0'])
            status('development_review_required', next_action='Review all three fixed endpoints before confirmation')
        else:
            import torch
            review = json.loads((ROOT / 'development_review.json').read_text())
            if not review.get('review_complete'):
                raise ValueError('Record the development review before opening confirmation')
            for arm in ARMS:
                completed = json.loads((ROOT / arm / 'complete.json').read_text())
                final = torch.load(ROOT / arm / 'step_0320.pt', map_location='cpu', weights_only=True)
                if completed['steps'] != STEPS or final['recovery_step'] != STEPS or final['config']['planner_memory_mode'] != arm:
                    raise ValueError('Confirmation requires matching completed final endpoints')
                del final
            candidates = {arm: {'checkpoint': str(ROOT / arm / 'step_0320.pt'),
                'sha256': digest(ROOT / arm / 'step_0320.pt')} for arm in ARMS}
            immutable_json(ROOT / 'confirmation/locked_candidates.json', candidates)
            for arm, candidate in candidates.items():
                checkpoint = Path(candidate['checkpoint'])
                output = ROOT / 'confirmation' / arm
                selection = ROOT / 'confirmation_selection.json'
                audio(arm + '_confirmation_audio', checkpoint, selection, output, 32)
                report = output / 'controls.json'
                if not report.exists():
                    run(arm + '_confirmation_controls', [sys.executable, 'scripts/evaluate_planner_controls.py',
                        '--checkpoint', checkpoint, '--manifest', MANIFEST, '--selection', selection,
                        '--output', report, '--split', 'val', '--count', 32, '--device', 'cuda:0'])
            status('confirmation_review_required', next_action='Review content and finalize; no automatic promotion')
    except Exception as error:
        status('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
