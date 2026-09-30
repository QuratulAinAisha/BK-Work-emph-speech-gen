"""Run approved diagnostic stages in order. / 승인된 진단 단계를 순서대로 실행합니다."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prepare_quality import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['decoding','information','hints'],required=True)
    p.add_argument('--probe-epochs',type=int,default=20);args=p.parse_args()
    os.chdir(Path(__file__).resolve().parents[1])
    root=Path('outputs/planner_diagnostics_v1');root.mkdir(exist_ok=True)
    lock=Path('outputs/quality_recovery/controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    common=['--checkpoint','outputs/planner_stages_v1/prior_finetune/best.pt',
        '--manifest','outputs/bk_quality_prepared/manifest.json','--selection','outputs/planner_stages_v1/selection.json']
    audio=['--audio-selection','outputs/planner_stages_v1/audio_selection.json']
    state={'pid':os.getpid(),'production_weights_updated':False}
    def status(stage,**extra):
        state.update(stage=stage,updated_utc=datetime.now(timezone.utc).isoformat(),**extra);atomic_json(root/'status.json',state)
    def run_parallel(commands):
        children=[]
        for name,command in commands:
            with (root/(name+'.log')).open('a') as log:
                child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            children.append((name,child))
        status(args.stage,children={name:child.pid for name,child in children})
        failures=[]
        for name,child in children:
            if child.wait():failures.append(name)
        if failures:raise RuntimeError('Failed diagnostic processes: '+', '.join(failures))
    try:
        if args.stage=='decoding':
            run_parallel([('step1_decoding',[sys.executable,'-u','scripts/check_planner_decoding.py',*common,*audio,
                '--output',str(root/'step1_decoding'),'--device','cuda:0'])])
            status('step1_review_required')
        elif args.stage=='information':
            if not (root/'step1_review.json').exists():raise ValueError('Review Step1 before Step2')
            commands=[]
            if not (root/'step2_a_transcripts.json').exists():
                commands.append(('step2_a_transcripts',[sys.executable,'-u','scripts/check_a_transcripts.py',
                    '--manifest',common[3],'--selection',common[5],'--output',str(root/'step2_a_transcripts.json'),
                    '--count','128','--device','cuda:0']))
            if not (root/'step2_information/cache/complete.json').exists():
                commands.append(('step2_extract',[sys.executable,'-u','scripts/check_a_information.py',*common,
                    '--output',str(root/'step2_information'),'--extract','--device','cuda:1']))
            if commands:run_parallel(commands)
            commands=[]
            for gpu,view in enumerate(['raw','projected','speech','fused']):
                folder=root/'step2_information/probes'/view
                if (folder/'complete.json').exists() and json.loads((folder/'complete.json').read_text())['epochs']==args.probe_epochs:
                    continue
                command=[sys.executable,'-u','scripts/check_a_information.py',*common,
                    '--output',str(root/'step2_information'),'--train-view',view,'--device',f'cuda:{gpu}',
                    '--epochs',str(args.probe_epochs),'--batch-size','16','--lr','.001']
                if (folder/'last.pt').exists():command+=['--resume']
                commands.append(('step2_probe_'+view,command))
            if commands:run_parallel(commands)
            run_parallel([('step2_summary',[sys.executable,'scripts/check_a_information.py',*common,
                '--output',str(root/'step2_information'),'--summarize'])])
            status('step2_review_required',probe_epochs=args.probe_epochs)
        else:
            if not (root/'step2_review.json').exists():raise ValueError('Review Step2 before Step3')
            run_parallel([('step3_hints',[sys.executable,'-u','scripts/check_planner_hints.py',*common,*audio,
                '--output',str(root/'step3_hints'),'--device','cuda:0'])])
            status('step3_review_required')
    except Exception as error:
        status('failed',error=str(error));raise


if __name__=='__main__':main()
