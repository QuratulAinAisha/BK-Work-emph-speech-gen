"""Fit units on selected training data only. / 선택한 학습 자료만으로 단위를 만듭니다."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from model.full_speech.units import SpeechCodebook
from prepare_quality import atomic_json
from train_full import atomic_save


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--clusters', type=int, default=512)
    p.add_argument('--iterations', type=int, default=40)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    if args.output.exists():
        p.error('Refusing to replace an existing codebook')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    manifest = json.loads(args.manifest.read_text())
    selection = json.loads(args.selection.read_text())
    digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if selection['manifest_sha256'] != digest:
        raise ValueError('Manifest changed')
    wanted = set(selection['train'])
    rows = [r for r in manifest['records'] if r['path'] in wanted]
    if len(rows) != len(wanted) or any(r['split'] != 'train' for r in rows):
        raise ValueError('Codebook must use selected training rows only')
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    mean, std = (payload['state_dict'][n] for n in ('semantic_mean', 'semantic_std'))
    frames = []
    for index, row in enumerate(rows):
        with np.load(args.manifest.parent / row['path'], allow_pickle=False) as data:
            frames.append((torch.from_numpy(data['semantic'].astype(np.float32)) - mean) / std)
        if (index + 1) % 256 == 0:
            print(json.dumps({'loaded_conversations': index + 1, 'total': len(rows)}), flush=True)
    values = torch.cat(frames).to(args.device)
    if not 1 < args.clusters < len(values) or not torch.isfinite(values).all():
        raise ValueError('Invalid codebook or frames')
    centers = values[torch.randperm(len(values), device=args.device)[:args.clusters]].clone()
    book = SpeechCodebook(centers).to(args.device)
    for iteration in range(args.iterations):
        ids = book.encode(values)
        sums = torch.zeros_like(centers).index_add_(0, ids, values)
        counts = torch.bincount(ids, minlength=args.clusters)
        centers = torch.where(counts[:, None] > 0, sums / counts[:, None].clamp_min(1), book.centers)
        book.centers.copy_(centers)
        if (iteration + 1) % 10 == 0:
            print(json.dumps({'iteration': iteration + 1, 'mse': float((values - centers[ids]).square().mean())}), flush=True)
    ids = book.encode(values)
    report = {'architecture': 'bk_speech_codebook_v1', 'clusters': args.clusters,
              'frames': len(values), 'training_conversations': len(rows),
              'fit_split': 'train', 'manifest_sha256': digest,
              'selection_sha256': hashlib.sha256(args.selection.read_bytes()).hexdigest(),
              'normalized_mse': float((values - centers[ids]).square().mean()),
              'used_clusters': int(ids.unique().numel()), 'seed': 42,
              'feature': 'cached HuBERT final layer, 768 dimensions, 50 Hz; experimental local clustering'}
    # Create the destination before the atomic checkpoint write. / 원자적 저장 전에 출력 폴더를 만듭니다.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_save({**report, 'centers': centers.cpu(), 'semantic_mean': mean, 'semantic_std': std}, args.output)
    atomic_json(args.output.with_suffix('.json'), report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
