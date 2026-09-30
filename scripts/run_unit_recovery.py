"""Evaluate a bounded discrete-unit candidate. / 제한된 이산 단위 후보를 평가합니다."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_quality import atomic_json
from scripts.run_recovery import acoustic_gate, memorization_gate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=Path('outputs/quality_recovery/units'))
    p.add_argument('--acoustic-steps', type=int, default=400)
    p.add_argument('--planner-steps', type=int, default=1200)
    args = p.parse_args()
    os.chdir(Path(__file__).resolve().parents[1])
    import fcntl
    parent = args.output.parent
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (parent / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if json.loads((parent / 'planner_gate.json').read_text())['passed']:
        raise RuntimeError('Conditional redesign requires failed continuous-planner evidence')
    if not json.loads((parent / 'teacher_validation.json').read_text())['passed']:
        raise RuntimeError('Waveform teacher has not passed controls')
    book = json.loads((args.output / 'codebook.json').read_text())
    if book['fit_split'] != 'train':
        raise RuntimeError('Codebook fitting leaked other splits')
    status = {'pid': os.getpid(), 'gpu_ids': [0, 1, 2, 3], 'batch_size_per_gpu': 16,
              'global_batch_size': 64, 'full_training_quality_gate_passed': False,
              'candidate': 'masked_discrete_units_v1'}
    def save(stage, **values):
        status.update(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(), **values)
        atomic_json(args.output / 'run_status.json', status)
        atomic_json(parent / 'run_status.json', status)
        print(json.dumps(status), flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    def run(stage, command):
        with (args.output / f'{stage}.log').open('a') as log:
            proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            save(stage, child_pid=proc.pid, command=command)
            code = proc.wait()
        if code:
            raise RuntimeError(f'{stage} exited {code}')
    launch = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4']
    def train(stage, steps, initializer, output, stop_after=None):
        command = launch + ['train_recovery.py', '--manifest', 'outputs/bk_quality_prepared/manifest.json',
            '--selection', str(parent / 'selection.json'), '--output', str(output), '--stage', stage,
            '--steps', str(steps), '--batch-size', '16', '--semantic-steps', '8', '--waveform-every', '4',
            '--teacher-weight', '.05' if stage == 'acoustic' else '0',
            '--lr', '.0001' if stage == 'acoustic' else '.0003', '--unit-codebook', str(args.output / 'codebook.pt')]
        command += ['--resume', str(output / 'last.pt')] if (output / 'last.pt').exists() else ['--initialize', str(initializer)]
        if stop_after:
            command += ['--stop-after', str(stop_after)]
        run('units_' + output.name, command)
    def evaluate(name, checkpoint, split, oracle=False):
        output = args.output / name
        if not (output / 'report.json').exists():
            command = [sys.executable, '-u', 'scripts/diagnose_quality.py', '--checkpoint', str(checkpoint),
                '--manifest', 'outputs/bk_quality_prepared/manifest.json', '--selection', str(parent / 'selection.json'),
                '--split', split, '--output', str(output), '--count', '8', '--steps', '8']
            if oracle:
                command += ['--oracle-only']
            run('units_' + name, command)
        return json.loads((output / 'report.json').read_text())
    try:
        initial = parent / 'acoustic/best.pt'
        smoke = args.output / 'resume_smoke'
        if not (smoke / 'complete.json').exists():
            if not (smoke / 'last.pt').exists():
                train('acoustic', 4, initial, smoke, 2)
            train('acoustic', 4, initial, smoke)
        acoustic = args.output / 'acoustic'
        if not (acoustic / 'complete.json').exists():
            train('acoustic', args.acoustic_steps, initial, acoustic)
        report = evaluate('acoustic_val', acoustic / 'best.pt', 'val', True)
        gate = acoustic_gate(report)
        atomic_json(args.output / 'acoustic_gate.json', gate)
        if not gate['passed']:
            save('units_acoustic_gate_failed', gate=gate,
                 next_action='Review unit representation before spending planner-training compute')
            return
        planner = args.output / 'planner'
        if not (planner / 'complete.json').exists():
            train('planner', args.planner_steps, acoustic / 'best.pt', planner)
        # Latest checks memorization; validation-selected best checks generalization. / 최신은 암기, 검증 최선은 일반화를 검사합니다.
        report = evaluate('planner_train_last', planner / 'last.pt', 'train')
        evaluate('planner_val_best', planner / 'best.pt', 'val')
        gate = memorization_gate(report)
        atomic_json(args.output / 'planner_gate.json', gate)
        save('units_pilot_complete_review_required' if gate['passed'] else 'units_planner_gate_failed', gate=gate,
             next_action='Review matched audio, unit learning and held-out relevance; do not auto-start full training')
    except Exception as error:
        save('units_failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
