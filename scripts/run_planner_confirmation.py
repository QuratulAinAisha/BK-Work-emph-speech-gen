"""Evaluate a recorded planner comparison without reselection. / 재선택 없이 기록된 계획기를 확인합니다."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_quality import atomic_json


def main():
    os.chdir(Path(__file__).resolve().parents[1])
    root = Path('outputs/planner_stages_v1')
    lock = Path('outputs/quality_recovery/controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan_bytes = (root / 'confirmation_plan.json').read_bytes()
    plan = json.loads(plan_bytes)
    if not plan.get('development_review_complete') or not plan.get('candidates'):
        raise ValueError('Record the development decision before confirmation')
    manifest = Path('outputs/bk_quality_prepared/manifest.json')
    selection = root / 'confirmation_selection.json'
    if hashlib.sha256(selection.read_bytes()).hexdigest() != plan['selection_sha256']:
        raise ValueError('Confirmation selection changed')
    out = root / 'confirmation'; out.mkdir(exist_ok=True)
    recorded = out / 'locked_plan.json'
    if recorded.exists() and recorded.read_bytes() != plan_bytes:
        raise ValueError('Do not reselect candidates against opened confirmation results')
    if not recorded.exists(): recorded.write_bytes(plan_bytes)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    for name, candidate in plan['candidates'].items():
        if name not in ('baseline', 'prior_finetune'):
            raise ValueError('Unexpected planner candidate')
        checkpoint = root / name / candidate['file']
        if checkpoint.name not in {f'step_{step:04d}.pt' for step in range(32, 161, 32)}:
            raise ValueError('Expected an existing epoch candidate')
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != candidate['sha256']:
            raise ValueError('Selected checkpoint changed')
        commands = [
            ('audio', out / name / 'report.json', [sys.executable, 'scripts/diagnose_quality.py',
             '--checkpoint', str(checkpoint), '--manifest', str(manifest), '--selection', str(selection),
             '--output', str(out / name), '--split', 'val', '--count', '32', '--steps', '8', '--device', 'cuda:0']),
            ('controls', out / f'{name}_controls.json', [sys.executable, 'scripts/evaluate_planner_controls.py',
             '--checkpoint', str(checkpoint), '--manifest', str(manifest), '--selection', str(selection),
             '--output', str(out / f'{name}_controls.json'), '--split', 'val', '--count', '32', '--device', 'cuda:0']),
        ]
        for stage, result, command in commands:
            if result.exists(): continue
            # The locked candidates are never chosen again from these results. / 확인 결과로 후보를 다시 고르지 않습니다.
            atomic_json(out / 'status.json', {'stage': name + '_' + stage, 'pid': os.getpid()})
            with (out / f'{name}_{stage}.log').open('a') as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    atomic_json(out / 'status.json', {'stage': 'complete', 'candidate_names': list(plan['candidates']),
                                    'quality_passed': False, 'requires_content_review': True})


if __name__ == '__main__': main()
