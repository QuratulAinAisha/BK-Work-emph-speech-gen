"""Separate unit prediction from audio rendering. / 단위 예측과 오디오 생성을 분리 검사합니다."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from dataset.quality_speech_dataset import QualitySpeechDataset, collate_quality, person_a_only
from model.full_speech.units import UnitSpeechSystem
from model.full_speech.tensor_ops import mask_from_lengths
from prepare_quality import atomic_json
from train_full import move_batch


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    torch.set_num_threads(2)
    model, payload = UnitSpeechSystem.from_checkpoint(args.checkpoint)
    model.to(args.device).eval()
    selection = json.loads(args.selection.read_text())
    report = {'checkpoint': str(args.checkpoint), 'step': payload['recovery_step'],
              'limitation': 'Correct target length is supplied here to isolate unit prediction; this is not A-only audio quality.',
              'splits': {}}
    for split in ('train', 'val'):
        data = QualitySpeechDataset(args.manifest, model.config, split)
        wanted = set(selection[split])
        rows, predicted_all, target_all = [], [], []
        chosen = [i for i, row in enumerate(data.records) if row['path'] in wanted]
        for offset in range(0, len(chosen), 8):
            indices = chosen[offset:offset + 8]
            batch = move_batch(collate_quality([data[i] for i in indices]), torch.device(args.device))
            encoded = model.encode_batch(person_a_only(batch))
            context, cmask, affect = encoded['context'], encoded['context_mask'], encoded['affect']
            style, _ = model.embeddings(batch['style_id'], batch['speaker_id'], len(indices))
            mask = mask_from_lengths(batch['semantic_len'])
            planner = model.semantic_planner
            targets = planner.codebook.encode((batch['semantic'] - model.semantic_mean) / model.semantic_std)
            generated = planner.sample(mask, context, cmask, affect, style, model.config.semantic_steps)
            predicted = planner.codebook.encode(generated)
            # Swap A's context only; keep the length control fixed. / 길이를 고정한 채 A 문맥만 바꿉니다.
            changed = planner.sample(mask, context.roll(1, 0), cmask.roll(1, 0), affect.roll(1, 0), style,
                                     model.config.semantic_steps)
            changed_ids = planner.codebook.encode(changed)
            for j, index in enumerate(indices):
                valid = mask[j]
                rows.append({'path': data.records[index]['path'], 'frames': int(valid.sum()),
                             'unit_accuracy': float((predicted[j, valid] == targets[j, valid]).float().mean()),
                             'a_context_swap_changed_fraction': float((predicted[j, valid] != changed_ids[j, valid]).float().mean())})
                predicted_all.append(predicted[j, valid].cpu()); target_all.append(targets[j, valid].cpu())
        first, second = torch.cat(predicted_all), torch.cat(target_all)
        mode = torch.bincount(second).argmax()
        report['splits'][split] = {'count': len(rows),
            'frame_unit_accuracy': float((first == second).float().mean()),
            'mean_example_unit_accuracy': sum(r['unit_accuracy'] for r in rows) / len(rows),
            'majority_unit_accuracy': float((second == mode).float().mean()),
            'generated_unique_units': int(first.unique().numel()), 'target_unique_units': int(second.unique().numel()),
            'mean_a_context_swap_changed_fraction': sum(r['a_context_swap_changed_fraction'] for r in rows) / len(rows),
            'examples': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    print(json.dumps({k: {n: v for n, v in row.items() if n != 'examples'} for k, row in report['splits'].items()}), flush=True)


if __name__ == '__main__':
    main()
