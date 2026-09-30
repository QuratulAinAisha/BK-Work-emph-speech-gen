"""Run bounded stages with explicit gates. / 명시적 기준으로 제한된 단계를 실행합니다."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_quality import atomic_json


def acoustic_gate(report):
    rows = report['examples']
    reference = [r['paths']['reference']['reference_wer'] for r in rows]
    oracle = [r['paths']['oracle_semantics']['reference_wer'] for r in rows]
    count = len(rows)
    if count < 8 or not all(math.isfinite(x) for x in reference + oracle):
        return {'passed': False, 'reason': 'Need eight finite controlled examples'}
    reference_mean, oracle_mean = sum(reference) / count, sum(oracle) / count
    good = sum(a <= b + .2 for a, b in zip(oracle, reference)) / count
    passed = reference_mean <= .2 and oracle_mean <= .2 and oracle_mean - reference_mean <= .1 and good >= .75
    return {'passed': passed, 'count': count, 'reference_wer': reference_mean,
            'oracle_wer': oracle_mean, 'excess_wer': oracle_mean - reference_mean,
            'within_20_points_fraction': good,
            'criterion': 'Reference mean <=20%; oracle mean <=20%; excess <=10 percentage points; 75% within20 points.',
            'limitation': 'Development ASR gate; not a human quality or empathy score.'}


def memorization_gate(report):
    rows = report['examples']
    if len(rows) < 8:
        return {'passed': False, 'reason': 'Need eight training examples'}
    errors = [r['paths']['predicted_length_steps_8']['reference_wer'] for r in rows]
    refs = [r['paths']['reference']['reference_wer'] for r in rows]
    passed = all(math.isfinite(x) for x in errors) and sum(errors) / len(errors) <= .25 and sum(refs) / len(refs) <= .2
    return {'passed': passed, 'training_example_wer': sum(errors) / len(errors),
            'criterion': 'Eight training examples, A-only generation, mean reference WER <=25%.',
            'limitation': 'Memorization check only. Does not establish held-out response relevance.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('outputs/quality_recovery'))
    parser.add_argument('--acoustic-steps', type=int, default=400)
    parser.add_argument('--planner-steps', type=int, default=600)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    import fcntl
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = {'pid': os.getpid(), 'gpu_ids': [0, 1, 2, 3], 'batch_size_per_gpu': 16,
              'global_batch_size': 64, 'full_training_quality_gate_passed': False}
    def save(stage, **values):
        status.update(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(), **values)
        atomic_json(args.output / 'run_status.json', status)
        print(json.dumps(status), flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    def run(stage, command):
        with (args.output / f'{stage}.log').open('a') as log:
            proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            save(stage, child_pid=proc.pid, command=command)
            code = proc.wait()
        if code:
            raise RuntimeError(f'{stage} exited {code}; inspect its log')
    manifest = Path('outputs/bk_quality_prepared/manifest.json')
    selection = args.output / 'selection.json'
    initial = args.output / 'v2_initialization.pt'
    launch = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4']
    def training(phase, steps, initializer, output, check=False):
        command = launch + ['train_recovery.py', '--manifest', str(manifest), '--selection', str(selection),
            '--output', str(output), '--stage', phase, '--steps', str(steps), '--batch-size', '16',
            '--semantic-steps', '8', '--waveform-every', '4', '--teacher-weight', '.05']
        checkpoint = output / 'last.pt'
        command += ['--resume', str(checkpoint)] if checkpoint.exists() and not check else ['--initialize', str(initializer)]
        if check:
            command += ['--check-only']
        run('preflight' if check else phase + '_training', command)
    def evaluate(phase, checkpoint, split, oracle=False):
        target = args.output / phase
        if (target / 'report.json').exists():
            return json.loads((target / 'report.json').read_text())
        command = [sys.executable, '-u', 'scripts/diagnose_quality.py', '--checkpoint', str(checkpoint),
                   '--manifest', str(manifest), '--selection', str(selection), '--split', split,
                   '--output', str(target), '--count', '8', '--steps', '8']
        if oracle:
            command += ['--oracle-only']
        run(phase, command)
        return json.loads((target / 'report.json').read_text())
    try:
        old = json.loads(Path('outputs/bk_quality_v2_100/run_status.json').read_text())
        if old['stage'] != 'stopped_for_recovery' or not old.get('gpu_workers_stopped'):
            raise RuntimeError('Old training has not been safely stopped')
        if not json.loads((args.output / 'teacher_validation.json').read_text())['passed']:
            raise RuntimeError('Waveform teacher has not passed controls')
        evaluate('pilot_initial_val', initial, 'val', True)
        preflight = args.output / 'preflight'
        if not all((preflight / f'preflight_rank{i}.json').exists() for i in range(4)):
            training('acoustic', args.acoustic_steps, initial, preflight, True)
        for i in range(4):
            row = json.loads((preflight / f'preflight_rank{i}.json').read_text())
            if not row['teacher_frozen'] or not math.isfinite(row['gradient_norm']) or row['gradient_norm'] <= 0:
                raise RuntimeError('GPU preflight failed')
        acoustic = args.output / 'acoustic'
        if not (acoustic / 'complete.json').exists():
            training('acoustic', args.acoustic_steps, initial, acoustic)
        evaluate('acoustic_train', acoustic / 'best.pt', 'train', True)
        result = evaluate('acoustic_val', acoustic / 'best.pt', 'val', True)
        gate = acoustic_gate(result)
        atomic_json(args.output / 'acoustic_gate.json', gate)
        if not gate['passed']:
            save('acoustic_gate_failed', gate=gate, next_action='Review reconstruction failure before training the planner')
            return
        planner = args.output / 'planner'
        if not (planner / 'complete.json').exists():
            training('planner', args.planner_steps, acoustic / 'best.pt', planner)
        result = evaluate('planner_train', planner / 'best.pt', 'train')
        evaluate('planner_val', planner / 'best.pt', 'val')
        gate = memorization_gate(result)
        atomic_json(args.output / 'planner_gate.json', gate)
        if not gate['passed']:
            save('planner_gate_failed', gate=gate, next_action='Test discrete-unit planner before scaling')
            return
        # A proxy cannot certify conversational quality. / 대리 지표로 대화 품질을 확정하지 않습니다.
        save('pilot_complete_review_required', gate=gate,
             next_action='Review held-out audio and relevance before joint curriculum or full-data training')
    except Exception as error:
        save('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
