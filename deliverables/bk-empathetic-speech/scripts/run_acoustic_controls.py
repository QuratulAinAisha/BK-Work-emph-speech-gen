"""Matched acoustic ablations with a reserved confirmation set. / 확인 집합을 분리한 음향 비교."""
import fcntl
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_quality import atomic_json
from scripts.run_recovery import acoustic_gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampled-experiment', action='store_true')
    parser.add_argument('--balanced-experiment', action='store_true')
    args = parser.parse_args()
    args.sampled_experiment = args.sampled_experiment or args.balanced_experiment
    os.chdir(Path(__file__).resolve().parents[1])
    lock = Path('outputs/quality_recovery/controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    root = Path('outputs/balanced_audio_v1' if args.balanced_experiment else
                'outputs/sampled_audio_v1' if args.sampled_experiment else 'outputs/acoustic_controlled_v1')
    root.mkdir(exist_ok=True)
    original = Path('outputs/broad_units_v1')
    initializer = 'outputs/quality_recovery/acoustic/best.pt'
    manifest = 'outputs/bk_quality_prepared/manifest.json'
    book = str(original / 'codebook.pt')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    state = {'pid': os.getpid(), 'gpu_ids': [0, 1, 2, 3], 'batch_per_gpu': 16}
    def status(stage, **kw):
        state.update(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(), **kw)
        atomic_json(root / 'run_status.json', state)
    def run(name, cmd):
        with (root / (name + '.log')).open('a') as f:
            child = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
            status(name, child_pid=child.pid)
            if child.wait():
                raise RuntimeError(name + ' failed; see log')
    selected = json.loads((original / 'selection.json').read_text())
    dev = [r['path'] for r in json.loads((original / 'initial/result.json').read_text())['examples']]
    # Reserve before training; do not use confirmation losses for selection. / 학습 전에 확인 집합을 분리합니다.
    remaining = sorted(set(selected['val']) - set(dev), key=lambda p: hashlib.sha256(p.encode()).hexdigest())
    if args.sampled_experiment:
        previous = json.loads(Path('outputs/acoustic_controlled_v1/protocol.json').read_text())
        remaining = [p for p in remaining if p not in previous['confirmation']]
    if args.balanced_experiment:
        previous = json.loads(Path('outputs/sampled_audio_v1/protocol.json').read_text())
        remaining = [p for p in remaining if p not in previous['confirmation']]
    if len(remaining) < 32:
        raise RuntimeError('Not enough unused confirmation recordings')
    confirm = remaining[:32]
    selection = {**selected, 'val': dev}
    atomic_json(root / 'selection.json', selection)
    for label, paths in [('dev', dev), ('confirm', confirm)]:
        for rank in range(4):
            atomic_json(root / f'{label}_{rank}.json', {**selected, 'val': paths[rank::4]})
    atomic_json(root / 'protocol.json', {'train_count': len(selected['train']), 'dev': dev, 'confirmation': confirm,
        'learning_rate': 3e-6, 'teacher_weights': [.05] if args.sampled_experiment else [0., .05], 'epochs': 5,
        'sampled_audio': args.sampled_experiment,
        'waveform_every': 1 if args.balanced_experiment else 4,
        'waveform_samples': 4 if args.balanced_experiment else 1,
        'initial_checkpoint': initializer, 'codebook': book,
        'note': 'Confirmation audio not previously evaluated, but these validation records contributed to older aggregate losses; not a pristine final test.'})
    def evaluate(checkpoint, out, subset, continuous=False):
        out.mkdir(parents=True, exist_ok=True)
        children = []
        for rank in range(4):
            target = out / f'rank{rank}'
            if (target / 'report.json').exists():
                continue
            cmd = [sys.executable, 'scripts/diagnose_quality.py', '--checkpoint', str(checkpoint),
                   '--manifest', manifest, '--selection', str(root / f'{subset}_{rank}.json'),
                   '--output', str(target), '--split', 'val', '--count', '8', '--oracle-only', '--device', f'cuda:{rank}']
            if continuous:
                cmd += ['--unit-codebook', book]
            with (out / f'rank{rank}.log').open('a') as f:
                children.append(subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env))
        status('evaluate_' + str(out.name), evaluation_pids=[c.pid for c in children])
        if any([c.wait() for c in children]):
            raise RuntimeError('Audio evaluation failed')
        rows = []
        for rank in range(4):
            values = json.loads((out / f'rank{rank}/report.json').read_text())['examples']
            expected = set(json.loads((root / f'{subset}_{rank}.json').read_text())['val'])
            if {r['path'] for r in values} != expected or len(values) != 8:
                raise RuntimeError('Evaluation membership changed')
            if continuous:
                for row in values:
                    row['paths']['oracle_semantics'] = row['paths']['oracle_units']
            rows.extend(values)
        result = {'checkpoint': str(checkpoint), 'gate': acoustic_gate({'examples': rows}), 'examples': rows}
        atomic_json(out / 'result.json', result)
        return result
    try:
        if args.sampled_experiment:
            check_path = Path('outputs/sampled_audio_v1/real_gradient_check.json') if args.balanced_experiment else root / 'real_gradient_check.json'
            check = json.loads(check_path.read_text())
            if not (check['generator_gradient_l1'] > 0 and all(check[k] for k in
                    ['generator_gradients_finite', 'other_model_gradients_absent', 'teacher_gradients_absent', 'codec_gradients_absent'])):
                raise RuntimeError('Real sampled-audio gradient preflight required')
            audit = json.loads(Path('outputs/acoustic_controlled_v1/alignment_audit.json').read_text())
            if not audit['passed']:
                raise RuntimeError('Data alignment audit failed')
            atomic_json(root / 'alignment_audit.json', audit)
        else:
            run('alignment_audit', [sys.executable, 'scripts/audit_acoustic_alignment.py'])
        candidates = []
        trials = [('sampled', '.05')] if args.sampled_experiment else [('no_teacher', '0'), ('teacher', '.05')]
        for label, teacher in trials:
            out = root / label
            if not (out / 'complete.json').exists():
                cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                    'train_recovery.py', '--manifest', manifest, '--selection', str(root / 'selection.json'),
                    '--output', str(out), '--stage', 'acoustic', '--steps', '160', '--evaluate-every', '32',
                    '--batch-size', '16', '--semantic-steps', '8', '--waveform-every', '4',
                    '--teacher-weight', teacher, '--lr', '.000003', '--unit-codebook', book, '--save-candidates']
                cmd += ['--resume', str(out / 'last.pt')] if (out / 'last.pt').exists() else ['--initialize', initializer]
                if args.sampled_experiment:
                    cmd += ['--sampled-audio']
                if args.balanced_experiment:
                    cmd[cmd.index('--waveform-every') + 1] = '1'
                    cmd += ['--waveform-samples', '4', '--sampled-flow-weight', '1']
                run('train_' + label, cmd)
            for epoch in range(1, 6):
                ckpt = out / f'step_{epoch * 32:04d}.pt'
                result = evaluate(ckpt, root / f'{label}_epoch_{epoch}', 'dev')
                candidates.append({'trial': label, 'epoch': epoch, 'checkpoint': str(ckpt), 'gate': result['gate']})
                atomic_json(root / 'candidates.json', candidates)
        initial = json.loads((original / 'initial/result.json').read_text())
        winner = min(candidates, key=lambda c: c['gate']['oracle_wer'])
        atomic_json(root / 'selected_candidate.json', winner)
        baseline = evaluate(initializer, root / 'confirmation_baseline', 'confirm', True)
        confirmed = evaluate(winner['checkpoint'], root / 'confirmation_candidate', 'confirm')
        passed = (winner['gate']['passed'] and confirmed['gate']['passed'] and
                  winner['gate']['oracle_wer'] <= initial['unit_gate']['oracle_wer'] and
                  confirmed['gate']['oracle_wer'] <= baseline['gate']['oracle_wer'])
        atomic_json(root / 'decision.json', {'passed': passed, 'selected': winner,
            'confirmation_baseline': baseline['gate'], 'confirmation_candidate': confirmed['gate'],
            'planner_ready': passed, 'note': 'If passed, run bounded broader planner; otherwise retain original acoustic checkpoint. ASR is not human listening or empathy.'})
        status('acoustic_controls_complete', passed=passed, planner_ready=passed)
    except Exception as exc:
        status('failed', error=str(exc))
        raise


if __name__ == '__main__':
    main()
