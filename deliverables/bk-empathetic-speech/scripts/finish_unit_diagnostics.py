"""Audit both checkpoint choices after the pilot. / 실험 후 두 체크포인트 선택을 검사합니다."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_quality import atomic_json


def main():
    import fcntl
    os.chdir(Path(__file__).resolve().parents[1])
    parent = Path('outputs/quality_recovery')
    root = parent / 'units_1024'
    lock = (parent / 'controller.lock').open('a')
    # Wait without competing for the training GPUs. / 학습 GPU와 경쟁하지 않고 기다립니다.
    fcntl.flock(lock, fcntl.LOCK_EX)
    state = json.loads((root / 'run_status.json').read_text())
    if state['stage'] not in ('units_planner_gate_failed', 'units_pilot_complete_review_required'):
        raise RuntimeError('Pilot did not complete its controlled evaluation')
    state.update(stage='units_final_diagnostics', pid=os.getpid())
    atomic_json(root / 'run_status.json', state); atomic_json(parent / 'run_status.json', state)
    base = [sys.executable, '-u']
    common = ['--manifest', 'outputs/bk_quality_prepared/manifest.json', '--selection', str(parent / 'selection.json')]
    for choice in ('last', 'best'):
        output = root / f'unit_prediction_{choice}.json'
        if not output.exists():
            subprocess.run(base + ['scripts/evaluate_unit_prediction.py', '--checkpoint', str(root / f'planner/{choice}.pt'),
                                  '--output', str(output)] + common, check=True)
    output = root / 'planner_val_last'
    if not (output / 'report.json').exists():
        subprocess.run(base + ['scripts/diagnose_quality.py', '--checkpoint', str(root / 'planner/last.pt'),
                              '--output', str(output), '--split', 'val', '--count', '8', '--steps', '8'] + common, check=True)
    state.update(stage='units_evaluated_review_required', updated_utc=datetime.now(timezone.utc).isoformat(),
                 next_action='Review training reproduction and held-out responses; full training remains gated')
    atomic_json(root / 'run_status.json', state); atomic_json(parent / 'run_status.json', state)


if __name__ == '__main__':
    main()
