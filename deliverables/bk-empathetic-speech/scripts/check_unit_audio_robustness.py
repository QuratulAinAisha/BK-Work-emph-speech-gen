"""Decode controlled unit errors with fixed acoustics. / 음향을 고정하고 단위 오류의 영향을 확인합니다."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from dataset.quality_speech_dataset import QualitySpeechDataset, collate_quality, person_a_only
from model.full_speech.units import UnitSpeechSystem
from model.full_speech.codec import FrozenEncodec
from prepare_quality import atomic_json
from train_full import move_batch
from scripts.check_planner_hints import chosen_indices, nested_hidden_mask, prepare_pair, mask_metadata
from scripts.diagnose_quality import recognize, save_wave
from scripts.planner_sampling import sample_units


def nearest_visible(ids, hidden, mask):
    # Only visible values may enter the predictor. / 예측에는 보이는 값만 사용합니다.
    answer = ids.masked_fill(hidden | ~mask, 0).clone()
    for row in range(len(ids)):
        visible = (mask[row] & ~hidden[row]).nonzero(as_tuple=True)[0]
        missing = hidden[row].nonzero(as_tuple=True)[0]
        if not len(visible):
            raise ValueError('Nearest-visible control requires B hints')
        nearest = (missing[:, None] - visible[None]).abs().argmin(1)
        answer[row, missing] = answer[row, visible[nearest]]
    return answer


@torch.inference_mode()
def run(args):
    torch.set_num_threads(2)
    device = torch.device(args.device)
    model, payload = UnitSpeechSystem.from_checkpoint(args.checkpoint)
    model.to(device).requires_grad_(False).eval()
    codec = FrozenEncodec().to(device).eval()
    data = QualitySpeechDataset(args.manifest, model.config, 'val')
    chosen = chosen_indices(data, json.loads(args.selection.read_text()),
                            hashlib.sha256(args.manifest.read_bytes()).hexdigest())[:args.count]
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'checkpoint': str(args.checkpoint), 'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'selection_sha256': hashlib.sha256(args.selection.read_bytes()).hexdigest(),
              'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              'mask_seed': 42, 'acoustic_seed': 42, 'examples': [], 'asr_model': 'openai/whisper-base.en',
              'planner_condition': args.planner_condition,
              'limitations': ['All completions receive B units and true B length; not A-only response generation.',
                  'Only missing positions change; acoustics and random noise are matched.',
                  'First-pass accuracy and iterative audio reconstruction are different measurements.',
                  'Reference WER measures reconstruction, not response relevance or empathy.']}
    centers = model.semantic_planner.codebook.centers
    distance = torch.cdist(centers.float(), centers.float()).fill_diagonal_(float('inf'))
    confusable = distance.argmin(1)
    for position, index in enumerate(chosen):
        row, sample = data.records[index], data[index]
        batch = move_batch(collate_quality([sample]), device)
        targets, mask, contexts, style = prepare_pair(model, batch, batch)
        memory = contexts['correct_a']
        if args.planner_condition == 'null_prior':
            # Prior completion uses its trained null condition; acoustics retain A. / 사전학습 복원에는 무조건 입력을 쓰고 음향에는 A를 유지합니다.
            memory = {'context': style.new_zeros(1, 1, 512),
                      'context_mask': torch.ones(1, 1, dtype=torch.bool, device=device),
                      'affect': style.new_zeros(1, 1, 6)}
            style = torch.zeros_like(style)
        entry = {'path': row['path'], 'conversation_id': row['conversation_id'],
                 'input_text': row['input_text'], 'reference_text': row['response_text'], 'paths': {}}
        entry['paths']['reference'] = save_wave(args.output, f'{position:02d}_reference.wav', sample['waveform'].numpy())
        variants = {'oracle_units': (targets.clone(), None)}
        for ratio in args.ratios:
            hidden = nested_hidden_mask(mask, ratio, 42)
            inputs = targets.masked_fill(hidden | ~mask, 0)
            first = model.semantic_planner.logits(inputs, hidden, mask, style=style, **memory).argmax(-1)
            generated, _ = sample_units(model.semantic_planner, mask, style=style, **memory,
                steps=8, mode='greedy', seed=42, initial_ids=inputs, initial_hidden=hidden)
            random = torch.randint(len(centers), targets.shape, device=device,
                                    generator=torch.Generator(device=device).manual_seed(42))
            for name, missing_ids in [('copy', nearest_visible(inputs, hidden, mask)),
                    ('firstpass', first), ('iterative', generated), ('random', random),
                    ('similar_unit', confusable[targets])]:
                ids = torch.where(hidden, missing_ids, targets)
                variants[f'{name}_hidden_{round(ratio * 100):03d}'] = (ids, hidden)
        for name, (ids, hidden) in variants.items():
            if hidden is not None and not torch.equal(ids[mask & ~hidden], targets[mask & ~hidden]):
                raise AssertionError('A visible hint changed')
            generated = model.generate_batch(person_a_only(batch), codec, seed=42,
                oracle_semantic=centers[ids], oracle_duration=batch['duration'])
            length = int(generated['audio_lengths'][0])
            value = save_wave(args.output, f'{position:02d}_{name}.wav', generated['waveform'][0, :length].cpu().numpy())
            value['unit_ids_sha256'] = hashlib.sha256(ids[mask].cpu().numpy().tobytes()).hexdigest()
            if hidden is not None:
                value.update(mask_metadata(hidden, mask, float(hidden.sum() / mask.sum()), 42))
                value['hidden_unit_accuracy'] = float((ids[hidden] == targets[hidden]).float().mean())
            entry['paths'][name] = value
        report['examples'].append(entry)
        atomic_json(args.output / 'generated_report.json', report)
        print(json.dumps({'audio_cases_done': position + 1, 'total': len(chosen)}), flush=True)
    del model, codec, centers, distance
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    report = recognize(report, args.output, str(device))
    atomic_json(args.output / 'report.json', report)
    print(json.dumps(report['summary']), flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ('checkpoint', 'manifest', 'selection', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--count', type=int, default=8)
    parser.add_argument('--ratios', type=float, nargs='+', default=[.25, .5])
    parser.add_argument('--planner-condition', choices=['production_a', 'null_prior'], default='production_a')
    args = parser.parse_args()
    if args.count < 1 or any(not 0 < x < 1 for x in args.ratios):
        parser.error('Positive count and strictly partial masks required')
    run(args)


if __name__ == '__main__':
    main()
