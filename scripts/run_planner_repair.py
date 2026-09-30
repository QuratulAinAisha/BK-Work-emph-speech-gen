"""Bounded, reviewed planner repair stages. / 검토 단계가 있는 제한된 계획기 복구 실행."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_quality import atomic_json

ROOT = Path('outputs/planner_repair_v1')
MANIFEST = Path('outputs/bk_quality_prepared/manifest.json')
SELECTION = Path('outputs/planner_stages_v1/selection.json')
AUDIO = Path('outputs/planner_stages_v1/audio_selection.json')
PRIOR = Path('outputs/planner_stages_v1/unit_prior/best.pt')
CONDITIONAL = Path('outputs/planner_stages_v1/prior_finetune/best.pt')
CURRENT = Path('outputs/planner_memory_v1/fused/step_0320.pt')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['initial', 'copy_audio', 'recipes', 'audits', 'prior', 'paired_tiny', 'conditional', 'waveform', 'duration_affect', 'sampling', 'vocabulary', 'ar_prior', 'ar_conditional', 'ar_categorical'], required=True)
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parents[1])
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = Path('outputs/quality_recovery/controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'pid': os.getpid(), 'phase': args.phase, 'production_candidate_promoted': False,
             'test_split_read': False, 'large_run_gpus': [0, 1, 2, 3], 'batch_per_gpu': 16}
    def status(stage, **values):
        state.update(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(), **values)
        atomic_json(ROOT / 'status.json', state)
    def run(name, command, marker):
        if Path(marker).exists():
            return
        command = [str(item) for item in command]
        atomic_json(ROOT / (name + '_command.json'), {'command': command})
        env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='0,1,2,3')
        with (ROOT / (name + '.log')).open('a') as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            status(name, child_pid=process.pid)
            if process.wait():
                raise RuntimeError(name + ' failed; inspect its log')
        if not Path(marker).exists():
            raise RuntimeError(name + ' did not create its completion artifact')
    identity = {'manifest_sha256': sha(MANIFEST), 'selection_sha256': sha(SELECTION),
                'prior_sha256': sha(PRIOR), 'conditional_sha256': sha(CONDITIONAL),
                'current_sha256': sha(CURRENT), 'audio_selection_sha256': sha(AUDIO)}
    path = ROOT / 'data_identity.json'
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError('Existing repair experiment data/checkpoint identity changed')
    atomic_json(path, identity)
    common = ['--manifest', MANIFEST, '--selection', SELECTION]
    try:
        if args.phase == 'initial':
            run('step01_history', [sys.executable, 'scripts/check_planner_learning_history.py', *common,
                '--output', ROOT / 'step01_history'], ROOT / 'step01_history/summary.json')
            out = ROOT / 'step02_tiny'
            source = ['--resume', out / 'last.pt'] if (out / 'last.pt').exists() else ['--initialize', PRIOR]
            run('step02_tiny', [sys.executable, 'scripts/train_planner_tiny.py', *common, *source,
                '--output', out, '--count', 16, '--batch-size', 16, '--fixed-updates', 300,
                '--fresh-updates', 300, '--lr', '.001', '--evaluate-every', 50], out / 'complete.json')
            run('step03_data', [sys.executable, 'scripts/audit_planner_learning_data.py', *common,
                '--checkpoint', CURRENT, '--output', ROOT / 'step03_data.json'], ROOT / 'step03_data.json')
        elif args.phase == 'copy_audio':
            run('copy_audio', [sys.executable, 'scripts/check_unit_audio_robustness.py', '--manifest', MANIFEST,
                '--selection', AUDIO, '--checkpoint', CURRENT, '--output', ROOT / 'copy_audio', '--count', 8],
                ROOT / 'copy_audio/report.json')
        elif args.phase == 'recipes':
            review = json.loads((ROOT / 'initial_review.json').read_text())
            if not review.get('run_recipes'):
                raise ValueError('Record the initial evidence before dependent recipe comparisons')
            # Separate extra updates from the higher learning rate. / 추가 학습 효과와 학습률 효과를 분리합니다.
            for name, objective, lr in [('legacy_low_lr', 'legacy', '.0001'), ('legacy', 'legacy', '.001'),
                                       ('balanced', 'balanced', '.001'), ('curriculum', 'curriculum', '.001')]:
                out = ROOT / 'step04_recipes' / name
                source = ['--resume', out / 'last.pt'] if (out / 'last.pt').exists() else ['--initialize', PRIOR]
                run('recipe_' + name, [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                    '--nproc_per_node=4', 'train_recovery.py', *common, *source, '--output', out,
                    '--stage', 'planner', '--unit-prior', '--unit-objective', objective, '--steps', 640,
                    '--evaluate-every', 160, '--batch-size', 16, '--semantic-steps', 8,
                    '--teacher-weight', 0, '--lr', lr, '--save-candidates'], out / 'complete.json')
                run('recipe_' + name + '_gaps', [sys.executable, 'scripts/check_planner_learning_history.py',
                    *common, '--checkpoint', name + '=' + str(out / 'step_0640.pt'),
                    '--output', out / 'gaps', '--count', 128], out / 'gaps/summary.json')
        elif args.phase == 'audits':
            runs = [(ROOT / 'step04_recipes' / name, PRIOR, (160, 320, 480, 640))
                    for name in ('legacy_low_lr', 'legacy', 'balanced', 'curriculum')]
            if (ROOT / 'step06_prior/complete.json').exists():
                review = json.loads((ROOT / 'recipe_review.json').read_text())
                runs.append((ROOT / 'step06_prior', Path(review['initializer']), (800, 1600, 2400, 3200)))
            if (ROOT / 'prior_review.json').exists():
                review = json.loads((ROOT / 'prior_review.json').read_text())
                for name in ('paired_tiny', 'balanced', 'fully_masked', 'rehearsal'):
                    out = ROOT / 'step07_conditional' / name
                    if (out / 'complete.json').exists():
                        steps = 640 if name == 'paired_tiny' else 1600
                        runs.append((out, Path(review['initializer']), tuple(range(steps // 4, steps + 1, steps // 4))))
            if (ROOT / 'conditional_review.json').exists():
                review = json.loads((ROOT / 'conditional_review.json').read_text())
                for name in ('ce_control', 'sampled_ctc'):
                    out = ROOT / 'step08_waveform' / name
                    if (out / 'complete.json').exists():
                        runs.append((out, Path(review['initializer']), (80, 160)))
            for out, initializer, endpoints in runs:
                for step in endpoints:
                    audit = out / f'audit_step_{step:04d}.json'
                    run(out.name + f'_audit_{step:04d}', [sys.executable, 'scripts/audit_planner_checkpoint.py',
                        '--initialize', initializer, '--checkpoint', out / f'step_{step:04d}.pt',
                        '--output', audit, '--planner-only'], audit)
                    if not json.loads(audit.read_text()).get('passed'):
                        raise ValueError('A saved planner audit failed')
        elif args.phase == 'prior':
            review = json.loads((ROOT / 'recipe_review.json').read_text())
            if not review.get('run_stronger_prior'):
                raise ValueError('Review the masking comparison before scaling')
            selection = ROOT / 'scale_selection.json'
            run('prepare_scale', [sys.executable, 'scripts/prepare_planner_repair_scale.py', *common,
                '--output', selection, '--count', 3072], selection)
            out = ROOT / 'step06_prior'
            source = ['--resume', out / 'last.pt'] if (out / 'last.pt').exists() else ['--initialize', review['initializer']]
            run('step06_prior', [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                'train_recovery.py', '--manifest', MANIFEST, '--selection', selection, *source, '--output', out,
                '--stage', 'planner', '--unit-prior', '--unit-objective', review['objective'], '--steps', 3200,
                '--evaluate-every', 800, '--batch-size', 16, '--semantic-steps', 8, '--teacher-weight', 0,
                '--lr', '.0003', '--save-candidates'], out / 'complete.json')
            run('step06_gaps', [sys.executable, 'scripts/check_planner_learning_history.py', *common,
                '--checkpoint', 'strong_prior=' + str(out / 'step_3200.pt'), '--output', out / 'gaps',
                '--count', 128], out / 'gaps/summary.json')
            run('step06_audio', [sys.executable, 'scripts/check_unit_audio_robustness.py', '--manifest', MANIFEST,
                '--selection', AUDIO, '--checkpoint', out / 'step_3200.pt', '--output', out / 'audio',
                '--planner-condition', 'null_prior', '--count', 8], out / 'audio/report.json')
        elif args.phase in ('paired_tiny', 'conditional'):
            review = json.loads((ROOT / 'prior_review.json').read_text())
            if not review.get('run_conditional'):
                raise ValueError('Review held-out prior reconstruction and audio first')
            small = args.phase == 'paired_tiny'
            selection = SELECTION
            if small:
                selected = json.loads(SELECTION.read_text())
                selected['train'] = selected['train'][:64]
                selected['conversations']['train'] = selected['conversations']['train'][:64]
                selected['selection_note'] = '64-conversation fully hidden A-to-B memorization diagnostic; not generalization.'
                selection = ROOT / 'paired_tiny_selection.json'
                if selection.exists() and json.loads(selection.read_text()) != selected:
                    raise ValueError('Tiny paired selection changed')
                atomic_json(selection, selected)
            arms = [('paired_tiny', 'fully_masked', 0.)] if small else [
                ('balanced', 'balanced', 0.), ('fully_masked', 'fully_masked', 0.),
                ('rehearsal', 'fully_masked', .5)]
            steps = 640 if small else 1600
            for name, objective, rehearsal in arms:
                out = ROOT / 'step07_conditional' / name
                source = ['--resume', out / 'last.pt'] if (out / 'last.pt').exists() else ['--initialize', review['initializer']]
                run('step07_' + name, [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                    'train_recovery.py', '--manifest', MANIFEST, '--selection', selection, *source, '--output', out,
                    '--stage', 'planner', '--unit-objective', objective, '--unit-rehearsal', rehearsal,
                    '--steps', steps, '--evaluate-every', steps // 4, '--batch-size', 16, '--semantic-steps', 8,
                    '--teacher-weight', 0, '--freeze-duration', '--lr', '.001' if small else '.0003',
                    '--save-candidates'], out / 'complete.json')
                for split, count in [('train', 64), ('val', 128)]:
                    run('step07_' + name + '_' + split, [sys.executable, 'scripts/evaluate_planner_controls.py',
                        '--manifest', MANIFEST, '--selection', selection, '--checkpoint', out / f'step_{steps:04d}.pt',
                        '--output', out / (split + '_controls.json'), '--split', split, '--count', count],
                        out / (split + '_controls.json'))
                run('step07_' + name + '_audio', [sys.executable, 'scripts/diagnose_quality.py', '--manifest', MANIFEST,
                    '--selection', AUDIO, '--checkpoint', out / f'step_{steps:04d}.pt', '--output', out / 'audio',
                    '--count', 8, '--steps', 8], out / 'audio/report.json')
            if not small:
                checkpoints = ['--checkpoint', 'strong_prior=' + review['initializer']]
                for name, _, _ in arms:
                    checkpoints += ['--checkpoint', name + '=' + str(ROOT / 'step07_conditional' / name / 'step_1600.pt')]
                run('step07_retention', [sys.executable, 'scripts/check_planner_learning_history.py', *common,
                    *checkpoints, '--output', ROOT / 'step07_conditional/retention', '--count', 32],
                    ROOT / 'step07_conditional/retention/summary.json')
        elif args.phase == 'waveform':
            review = json.loads((ROOT / 'conditional_review.json').read_text())
            if not review.get('run_sampled_audio'):
                raise ValueError('Review A-only conditional generation before sampled-audio training')
            initializer = Path(review['initializer'])
            preflight = ROOT / 'step08_waveform/real_gradient.json'
            run('step08_gradient', [sys.executable, 'scripts/calibrate_planner_waveform_parity.py', *common,
                '--checkpoint', initializer, '--output', preflight], preflight)
            measured = json.loads(preflight.read_text())
            if not measured.get('passed') or measured.get('checkpoint_sha256') != sha(initializer):
                raise ValueError('Real sampled-audio gradient preflight failed')
            check = ROOT / 'step08_waveform/ddp_preflight'
            run('step08_ddp_preflight', [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                '--nproc_per_node=4', 'train_recovery.py', *common, '--initialize', initializer,
                '--output', check, '--stage', 'planner', '--unit-objective', review['objective'],
                '--unit-rehearsal', review['rehearsal_weight'], '--steps', 160, '--evaluate-every', 80,
                '--batch-size', 16, '--teacher-weight', 0, '--freeze-duration', '--lr', '.00003',
                '--planner-sampled-audio', '--planner-audio-weight', '.02', '--waveform-every', 4,
                '--check-only'], check / 'preflight_rank0.json')
            # Verify every rank before updating weights. / 가중치 갱신 전에 모든 순위를 확인합니다.
            for rank in range(4):
                evidence = json.loads((check / f'preflight_rank{rank}.json').read_text())
                if not evidence['teacher_frozen'] or evidence['gradient_norm'] <= 0:
                    raise ValueError('A distributed waveform preflight failed')
            for name in ('ce_control', 'sampled_ctc'):
                out = ROOT / 'step08_waveform' / name
                source = ['--resume', out / 'last.pt'] if (out / 'last.pt').exists() else ['--initialize', initializer]
                command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                    'train_recovery.py', *common, *source, '--output', out, '--stage', 'planner',
                    '--unit-objective', review['objective'], '--unit-rehearsal', review['rehearsal_weight'],
                    '--steps', 160, '--evaluate-every', 80, '--batch-size', 16, '--semantic-steps', 8,
                    '--teacher-weight', 0, '--freeze-duration', '--lr', '.00003', '--save-candidates']
                if name == 'sampled_ctc':
                    command += ['--planner-sampled-audio', '--planner-audio-weight', '.02', '--waveform-every', 4]
                run('step08_' + name, command, out / 'complete.json')
                run('step08_' + name + '_audio', [sys.executable, 'scripts/diagnose_quality.py', '--manifest', MANIFEST,
                    '--selection', AUDIO, '--checkpoint', out / 'step_0160.pt', '--output', out / 'audio',
                    '--count', 8, '--steps', 8], out / 'audio/report.json')
        elif args.phase == 'duration_affect':
            review = json.loads((ROOT / 'waveform_review.json').read_text())
            run('steps10_11', [sys.executable, 'scripts/check_response_duration_affect.py', '--manifest', MANIFEST,
                '--selection', AUDIO, '--checkpoint', review['diagnostic_checkpoint'],
                '--output', ROOT / 'steps10_11', '--count', 8], ROOT / 'steps10_11/report.json')
        elif args.phase == 'sampling':
            review = json.loads((ROOT / 'waveform_review.json').read_text())
            checkpoint = review['diagnostic_checkpoint']
            run('step12_sampling', [sys.executable, 'scripts/check_planner_repair_sampling.py', *common,
                '--checkpoint', checkpoint, '--audio-selection', AUDIO, '--output', ROOT / 'step12_sampling',
                '--count', 32], ROOT / 'step12_sampling/audio/report.json')
            run('step12_self_hints', [sys.executable, 'scripts/check_planner_self_hints.py', *common,
                '--checkpoint', checkpoint, '--output', ROOT / 'step12_self_hints', '--count', 32],
                ROOT / 'step12_self_hints/units_report.json')
        elif args.phase == 'vocabulary':
            review = json.loads((ROOT / 'representation_review.json').read_text())
            if not review.get('run_vocabulary_gate'):
                raise ValueError('Review conditional and sampler evidence before changing vocabulary')
            checkpoint = review['diagnostic_checkpoint']
            book = ROOT / 'step12_vocabulary/units256.pt'
            run('step12_fit_vocabulary', [sys.executable, 'scripts/fit_speech_units.py', *common,
                '--checkpoint', checkpoint, '--output', book, '--clusters', 256, '--iterations', 40], book)
            run('step12_vocabulary_audio', [sys.executable, 'scripts/check_unit_vocabulary_audio.py',
                '--manifest', MANIFEST, '--selection', AUDIO, '--checkpoint', checkpoint, '--codebook', book,
                '--output', ROOT / 'step12_vocabulary/audio', '--count', 8],
                ROOT / 'step12_vocabulary/audio/report.json')
        elif args.phase in ('ar_prior', 'ar_conditional'):
            review = json.loads((ROOT / 'representation_review.json').read_text())
            if not review.get('run_ar_feasibility'):
                raise ValueError('Record evidence for the causal planner feasibility branch')
            prior = args.phase == 'ar_prior'
            name = 'prior' if prior else 'conditional'
            out = ROOT / 'step12_ar' / name
            initializer = Path(review['ar_initializer']) if prior else ROOT / 'step12_ar/prior/step_1600.pt'
            selection = ROOT / 'scale_selection.json' if prior else SELECTION
            if not prior and not json.loads((ROOT / 'ar_prior_review.json').read_text()).get('run_conditional'):
                raise ValueError('Review causal-prior learning and free generation first')
            fixed = ['--manifest', MANIFEST, '--selection', selection, '--phase', name, '--steps', 1600,
                     '--evaluate-every', 400, '--batch-size', 16, '--seed', 42, '--lr', '.0003',
                     '--conditional-cache', 'inputs_units']
            check = ROOT / 'step12_ar' / (name + '_preflight')
            run('step12_ar_' + name + '_preflight', [sys.executable, '-m', 'torch.distributed.run',
                '--standalone', '--nproc_per_node=4', 'scripts/train_planner_ar.py', *fixed,
                '--initialize', initializer, '--output', check, '--check-only'], check / 'preflight_rank0.json')
            for rank in range(4):
                if not json.loads((check / f'preflight_rank{rank}.json').read_text()).get('passed'):
                    raise ValueError('An AR distributed preflight failed')
            source = ['--resume', out / 'last.pt'] if (out / 'last.pt').exists() else ['--initialize', initializer]
            run('step12_ar_' + name, [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                '--nproc_per_node=4', 'scripts/train_planner_ar.py', *fixed, *source, '--output', out],
                out / 'complete.json')
            evaluation = ROOT / 'step12_ar' / (name + '_evaluation')
            command = [sys.executable, 'scripts/train_planner_ar.py', '--phase', 'evaluate',
                '--checkpoint', out / 'step_1600.pt', '--manifest', MANIFEST, '--selection', selection,
                '--output', evaluation, '--free-count', 8, '--conditional-cache', 'inputs_units']
            if not prior:
                command += ['--audio-selection', AUDIO]
            run('step12_ar_' + name + '_evaluate', command,
                evaluation / ('units_report.json' if prior else 'audio/report.json'))
        elif args.phase == 'ar_categorical':
            run('step12_ar_categorical', [sys.executable, 'scripts/train_planner_ar.py', '--phase', 'evaluate',
                '--checkpoint', ROOT / 'step12_ar/conditional/step_1600.pt', *common, '--audio-selection', AUDIO,
                '--output', ROOT / 'step12_ar/categorical_evaluation', '--free-count', 8,
                '--conditional-cache', 'inputs_units', '--sampling', 'categorical', '--temperature', '.8',
                '--top-k', 20, '--sampling-seed', 42], ROOT / 'step12_ar/categorical_evaluation/audio/report.json')
        status(args.phase + '_review_required', child_pid=None,
               next_action='Read the saved evidence and record the next bounded experiment before proceeding.')
    except Exception as error:
        status('failed', error=str(error)); raise


if __name__ == '__main__':
    main()
