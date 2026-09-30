"""Staged planner experiments with frozen acoustics. / 음향을 고정한 단계별 계획기 실험."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_quality import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['baseline', 'prior'], required=True)
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parents[1])
    lock = Path('outputs/quality_recovery/controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    root = Path('outputs/planner_stages_v1');root.mkdir(exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    manifest = 'outputs/bk_quality_prepared/manifest.json'
    book = 'outputs/broad_units_v1/codebook.pt'
    original = 'outputs/quality_recovery/acoustic/best.pt'
    state = {'pid': os.getpid(), 'phase': args.phase, 'gpu_ids': [0,1,2,3], 'batch_per_gpu':16,
             'acoustic_acceptance': 'unchanged; planner experiment does not override failed acoustic gate'}
    def status(stage, **values):
        state.update(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(), **values)
        atomic_json(root/'run_status.json',state)
    def run(name, command):
        with (root/f'{name}.log').open('a') as log:
            child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            status(name,child_pid=child.pid)
            if child.wait():raise RuntimeError(name+' failed; inspect log')
    def train(name, initializer, prior=False):
        out=root/name
        if (out/'complete.json').exists():
            if not prior: ensure_audio(name)
            return
        command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
            'train_recovery.py','--manifest',manifest,'--selection',str(root/'selection.json'),
            '--unit-codebook',book,'--output',str(out),'--stage','planner','--steps','160',
            '--evaluate-every','32','--batch-size','16','--semantic-steps','8','--teacher-weight','0',
            '--lr','.0001','--save-candidates']
        command+=['--resume',str(out/'last.pt')] if (out/'last.pt').exists() else ['--initialize',str(initializer)]
        command+=['--unit-prior'] if prior else ['--audio-selection',str(root/'audio_selection.json')]
        run('train_'+name,command)
        if not prior: ensure_audio(name)
    def ensure_audio(name):
        # Replay interrupted epoch evaluations from the matching weights. / 중단된 에포크 평가는 해당 가중치로 다시 실행합니다.
        for step in range(32, 161, 32):
            out=root/name/f'audio_step_{step:04d}'
            if not (out/'report.json').exists():
                run(name+f'_audio_step_{step:04d}',[sys.executable,'scripts/diagnose_quality.py',
                    '--checkpoint',str(root/name/f'step_{step:04d}.pt'),'--manifest',manifest,
                    '--selection',str(root/'audio_selection.json'),'--output',str(out),
                    '--split','val','--count','8','--steps','8','--device','cuda:0'])
    def controls(name):
        for split,count in [('train',64),('val',128)]:
            out=root/name/f'controls_{split}.json'
            if not out.exists():
                run(name+'_controls_'+split,[sys.executable,'scripts/evaluate_planner_controls.py',
                    '--checkpoint',str(root/name/'best.pt'),'--manifest',manifest,
                    '--selection',str(root/'selection.json'),'--output',str(out),'--split',split,
                    '--count',str(count),'--device','cuda:0'])
    try:
        run('prepare_selection',[sys.executable,'scripts/prepare_planner_experiment.py','--manifest',manifest,
            '--selection','outputs/broad_units_v1/selection.json','--pilot','outputs/quality_recovery/selection.json',
            '--output',str(root)])
        if args.phase=='baseline':
            train('baseline',original)
            controls('baseline')
            status('baseline_review_required',next_action='Review A-only epoch audio and conditioning controls before prior warm-up')
        else:
            # Prior phase requires recorded review, not an automatic guess. / 사전학습 단계는 기록된 검토 후 실행합니다.
            review=root/'baseline_review.json'
            if not review.exists() or not json.loads(review.read_text()).get('run_prior'):
                raise RuntimeError('Baseline review must authorize the conditional prior comparison')
            train('unit_prior',original,prior=True)
            train('prior_finetune',root/'unit_prior/best.pt')
            controls('prior_finetune')
            status('prior_review_required',next_action='Compare relevance and unit controls; assess sequence compression without claiming a working response model')
    except Exception as error:
        status('failed',error=str(error));raise


if __name__=='__main__':main()
