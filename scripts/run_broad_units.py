"""Gated broader unit experiment. / 품질 기준이 있는 확장 단위 실험."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_quality import atomic_json
from scripts.run_recovery import acoustic_gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('outputs/broad_units_v1'))
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parents[1])
    import fcntl
    root = args.output
    lock = Path('outputs/quality_recovery/controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    initial = json.loads((root / 'initial/result.json').read_text())
    book = json.loads((root / 'codebook.json').read_text())
    if book['training_conversations'] != 2048 or book['fit_split'] != 'train':
        raise ValueError('Need the validated broader training-only codebook')
    if book['selection_sha256'] != hashlib.sha256((root / 'selection.json').read_bytes()).hexdigest():
        raise ValueError('Codebook and selected data differ')
    state = {'pid': os.getpid(), 'gpu_ids': [0, 1, 2, 3], 'batch_size_per_gpu': 16,
             'global_batch_size': 64, 'train_conversations': 2048, 'validation_conversations': 128,
             'full_training_started': False, 'broad_planner_epochs': 5}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    def save(stage, **values):
        state.update(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(), **values)
        atomic_json(root / 'run_status.json', state)
        print(json.dumps(state), flush=True)
    def run(stage, command):
        with (root / f'{stage}.log').open('a') as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            save(stage, child_pid=child.pid, command=command)
            code = child.wait()
        if code:
            raise RuntimeError(f'{stage} exited {code}')
    launch = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4']
    def train(phase, initializer, stop_after=None):
        out = root / phase
        command = launch + ['train_recovery.py', '--manifest', 'outputs/bk_quality_prepared/manifest.json',
            '--selection', str(root / 'selection.json'), '--output', str(out), '--stage', phase,
            '--steps', '160', '--evaluate-every', '32', '--batch-size', '16', '--semantic-steps', '8',
            '--waveform-every', '4', '--teacher-weight', '.05' if phase == 'acoustic' else '0',
            '--lr', '.00003' if phase == 'acoustic' else '.0001', '--unit-codebook', str(root / 'codebook.pt')]
        command += ['--resume', str(out / 'last.pt')] if (out / 'last.pt').exists() else ['--initialize', str(initializer)]
        if stop_after:
            command += ['--stop-after', str(stop_after)]
        run(phase + '_training', command)
    def oracle(checkpoint):
        out = root / 'adapted_oracle'
        out.mkdir(exist_ok=True)
        children = []
        for rank in range(4):
            target = out / f'rank{rank}'
            if (target / 'report.json').exists():
                continue
            command = [sys.executable, '-u', 'scripts/diagnose_quality.py', '--checkpoint', str(checkpoint),
                '--manifest', 'outputs/bk_quality_prepared/manifest.json',
                '--selection', str(root / f'initial/controls_rank{rank}.json'), '--output', str(target),
                '--split', 'val', '--count', '8', '--oracle-only', '--device', f'cuda:{rank}']
            with (out / f'rank{rank}.log').open('a') as log:
                children.append(subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT))
        save('adapted_oracle_checks', evaluation_pids=[c.pid for c in children])
        if any([c.wait() for c in children]):
            raise RuntimeError('Adapted oracle check failed')
        rows = []
        for rank in range(4):
            report = json.loads((out / f'rank{rank}/report.json').read_text())
            expected = set(json.loads((root / f'initial/controls_rank{rank}.json').read_text())['val'])
            if {r['path'] for r in report['examples']} != expected or len(report['examples']) != 8:
                raise RuntimeError('Acoustic controls changed')
            rows.extend(report['examples'])
        gate = acoustic_gate({'examples': rows})
        atomic_json(out / 'result.json', {'checkpoint': str(checkpoint), 'unit_gate': gate, 'examples': rows})
        return gate
    def evaluate_epoch(epoch):
        out = root / f'planner_eval_epoch_{epoch:02d}'
        if not (out / 'report.json').exists():
            run(f'planner_audio_epoch_{epoch:02d}', [sys.executable, '-u', 'scripts/diagnose_quality.py',
                '--checkpoint', str(root / 'planner/last.pt'), '--manifest', 'outputs/bk_quality_prepared/manifest.json',
                '--selection', str(root / 'initial/controls_rank0.json'), '--split', 'val', '--count', '8',
                '--steps', '8', '--output', str(out)])
    try:
        acoustic = root / 'acoustic'
        if initial['unit_gate']['passed']:
            # Judge the actual unit path; the continuous path is a comparator. / 실제 단위 경로로 판단하고 연속 경로는 비교에 씁니다.
            if not (acoustic / 'best.pt').exists():
                run('preserve_acoustic_interface', [sys.executable, 'scripts/initialize_unit_interface.py',
                    '--checkpoint', 'outputs/quality_recovery/acoustic/best.pt', '--codebook', str(root / 'codebook.pt'),
                    '--report', str(root / 'initial/result.json'), '--output', str(acoustic)])
            gate = initial['unit_gate']
        else:
            if not (acoustic / 'complete.json').exists():
                train('acoustic', Path('outputs/quality_recovery/acoustic/best.pt'))
            gate = oracle(acoustic / 'best.pt')
        atomic_json(root / 'unit_interface_gate.json', gate)
        if not gate['passed']:
            save('broad_acoustic_gate_failed', gate=gate,
                 next_action='Review broader unit reconstruction; planner has not been trained')
            return
        # Five epochs = 5 × 2048 / 64 = 160 updates. / 5 에포크는 총 160회 갱신입니다.
        for epoch in range(1, 6):
            if (root / f'planner_eval_epoch_{epoch:02d}/report.json').exists():
                continue
            train('planner', acoustic / 'best.pt', epoch * 32)
            evaluate_epoch(epoch)
        save('broad_five_epochs_evaluated', completed_planner_epochs=5,
             next_action='Review unseen-input replies and validation trends before any extension or joint training')
    except Exception as error:
        save('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
