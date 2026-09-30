"""Queue the improved experiment behind the baseline. / 기존 실험 뒤에 개선 실험을 실행합니다."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_quality import atomic_json


def baseline_released(status, accept_stopped=False):
    # An authorized stop must also confirm worker exit. / 승인된 중단 후 작업자 종료도 확인합니다.
    if status['stage'] == 'complete':
        return True
    if status['stage'] == 'stopped' and accept_stopped:
        if not status.get('gpu_workers_stopped'):
            raise RuntimeError('Baseline workers have not been confirmed stopped')
        return True
    if status['stage'] == 'failed':
        raise RuntimeError('Baseline failed; inspect it before allocating its GPUs')
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('outputs/bk_quality_v2_100'))
    parser.add_argument('--prepared', type=Path, default=Path('outputs/bk_quality_prepared'))
    parser.add_argument('--baseline-output', type=Path, default=Path('outputs/bk_5epochs'))
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--accept-stopped-baseline', action='store_true',
                        help='Use a deliberately stopped baseline after its workers have exited')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    args.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (args.output / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = {'pid': os.getpid(), 'epochs_requested': args.epochs, 'gpu_ids': [0, 1, 2, 3],
              'batch_size_per_gpu': args.batch_size, 'global_batch_size': 4 * args.batch_size}
    def save(stage, **extra):
        status.update(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
        atomic_json(args.output / 'run_status.json', status)
        print(json.dumps(status), flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    launch = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4']
    def run(stage, command):
        with (args.output / f'{stage}.log').open('a') as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            save(stage, child_pid=process.pid, command=command)
            code = process.wait()
        if code:
            raise RuntimeError(f'{stage} exited {code}; inspect {stage}.log')
    try:
        save('waiting_for_baseline')
        while True:
            baseline = json.loads((args.baseline_output / 'run_status.json').read_text())
            if baseline_released(baseline, args.accept_stopped_baseline):
                break
            time.sleep(20)
        initializer = args.output / 'baseline_initialization.pt'
        if not initializer.exists():
            shutil.copyfile(args.baseline_output / 'best.pt', initializer)
        complete = args.prepared / 'complete.json'
        if not complete.exists():
            run('preparation', launch + ['prepare_quality.py', '--output', str(args.prepared)])
        marker = json.loads(complete.read_text())
        if not marker.get('validated') or marker['records'] != 60000:
            raise ValueError('Full quality cache is not validated')
        checkpoint = args.output / 'last.pt'
        completed = 0
        if checkpoint.exists():
            import torch
            completed = torch.load(checkpoint, map_location='cpu', weights_only=True)['epoch'] + 1
        if completed < args.epochs:
            initialization = ['--resume', str(checkpoint)] if checkpoint.exists() else ['--initialize-baseline', str(initializer)]
            run('training', launch + ['train_quality.py', '--manifest', str(args.prepared / 'manifest.json'),
                '--config', str(args.prepared / 'config.json'), '--output', str(args.output),
                '--epochs', str(args.epochs), '--batch-size', str(args.batch_size), '--workers', str(args.workers),
                '--evaluate-every', '5', '--evaluation-count', '6', *initialization])
        metrics = [json.loads(line) for line in (args.output / 'metrics.jsonl').read_text().splitlines()]
        if [row['epoch'] for row in metrics] != list(range(1, args.epochs + 1)):
            raise RuntimeError('Epoch history is incomplete or duplicated')
        run('evaluation', [sys.executable, 'evaluate_quality.py', '--checkpoint', str(args.output / 'best.pt'),
            '--manifest', str(args.prepared / 'manifest.json'), '--split', 'test', '--count', '100',
            '--output', str(args.output / 'final_evaluation')])
        save('complete', completed_epochs=args.epochs, final_validation=metrics[-1]['validation'])
    except Exception as exc:
        save('failed', error=str(exc))
        raise


if __name__ == '__main__':
    main()
