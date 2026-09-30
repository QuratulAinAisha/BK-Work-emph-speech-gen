"""Measure competing acoustic gradients. / 경쟁하는 음향 기울기를 측정합니다."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from model.full_speech.quality import QualityConfig
from model.full_speech.units import initialize_unit_system
from model.full_speech.recovery import FrozenWaveformCTC
from model.full_speech.codec import FrozenEncodec
from dataset.quality_speech_dataset import QualitySpeechDataset, collate_quality
from train_full import move_batch
from prepare_quality import atomic_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rank', type=int, default=0)
    args = p.parse_args()
    torch.set_num_threads(2)
    device = torch.device(f'cuda:{args.rank}')
    torch.cuda.set_device(device)
    torch.manual_seed(42)
    payload = torch.load('outputs/quality_recovery/acoustic/best.pt', map_location='cpu', weights_only=True)
    cfg = QualityConfig(**payload['config']);cfg.semantic_steps = cfg.predicted_semantic_steps = 8
    model = initialize_unit_system(cfg, payload, Path('outputs/broad_units_v1/codebook.pt')).to(device)
    teacher = FrozenWaveformCTC().to(device)
    object.__setattr__(model, '_acoustic_codec', FrozenEncodec().to(device))
    model.configure_recovery('acoustic', teacher=teacher)
    model.configure_sampled_audio();model.train()
    data = QualitySpeechDataset('outputs/bk_quality_prepared/manifest.json', cfg, 'train')
    wanted = set(json.loads(Path('outputs/broad_units_v1/selection.json').read_text())['train'])
    indices = [i for i,r in enumerate(data.records) if r['path'] in wanted]
    order = torch.randperm(len(indices), generator=torch.Generator().manual_seed(42)).tolist()
    params = [v for v in model.codec_generator.parameters() if v.requires_grad]
    results = []
    for step in [0, 4, 8, 12]:
        model.current_step = step
        chosen = [indices[j] for j in order[step*64+args.rank*16:step*64+args.rank*16+16]]
        batch = move_batch(collate_quality([data[i] for i in chosen]), device)
        torch.manual_seed(1000 + args.rank * 100 + step)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            losses = model.losses(batch)
        gradients = {}
        for name in ['codec_flow','sampled_ctc','sampled_spectral']:
            g = torch.autograd.grad(losses[name], params, retain_graph=True, allow_unused=True)
            gradients[name] = torch.cat([(torch.zeros_like(v) if q is None else q).detach().flatten().float() for v,q in zip(params,g)])
        def dot(a,b):return float(torch.dot(a,b))
        total = sum(gradients.values())
        ctc = gradients['sampled_ctc']
        results.append({'step':step,'rank':args.rank,'path':data.records[chosen[0]]['path'],
            'losses':{k:float(v.detach()) for k,v in losses.items()},
            'norms':{k:float(v.norm()) for k,v in gradients.items()},
            'cosine_with_ctc':{k:dot(v,ctc)/max(1e-12,float(v.norm()*ctc.norm())) for k,v in gradients.items()},
            'ctc_dot_full_update':dot(ctc,total),
            'ctc_dot_flow_only':dot(ctc,gradients['codec_flow'])})
        del gradients, losses, total, ctc, batch
    out=Path('outputs/balanced_audio_v1');out.mkdir(exist_ok=True)
    atomic_json(out/f'gradient_rank{args.rank}.json',results)
    print(json.dumps(results),flush=True)


if __name__=='__main__':main()
