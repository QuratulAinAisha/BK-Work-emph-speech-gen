"""Reproduce two final-experiment responses. / 마지막 실험의 응답 두 개를 재생성합니다."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from dataset.quality_speech_dataset import person_a_only
from model.full_speech.codec import FrozenEncodec
from model.full_speech.experimental_ar_units import load_ar_checkpoint
from prepare_quality import atomic_json
from scripts.check_unit_vocabulary_audio import file_hash, module_hashes
from scripts.diagnose_quality import recognize, save_wave
from scripts.train_planner_ar import SelectedSamples


def measurements(wave):
    wave = wave.detach().float().cpu().numpy().reshape(-1)
    return {'rms': float(np.sqrt(np.mean(wave ** 2))), 'dc_offset': float(wave.mean()),
            'finite': bool(np.isfinite(wave).all()), 'samples': len(wave)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.output.exists():
        raise ValueError('Use a new output directory')
    torch.set_num_threads(2)
    previous = Path('outputs/planner_repair_v1/step13_confirmation/fresh/ar_categorical/audio/report.json')
    selection = Path('outputs/planner_repair_v1/step13_confirmation/fresh_audio_selection.json')
    manifest = Path('outputs/bk_quality_prepared/manifest.json')
    source = json.loads(previous.read_text())
    checkpoint = Path(source['checkpoint'])
    if file_hash(checkpoint) != source['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from the last experiment')
    model, payload = load_ar_checkpoint(checkpoint)
    if model.ar_phase != 'conditional' or payload['ar_step'] != 1600:
        raise ValueError('Expected the final conditional response checkpoint')
    model.to(args.device).requires_grad_(False).eval()
    sampling = source['planner_sampling']
    model.semantic_planner.configure_sampling(sampling['mode'], sampling['temperature'], sampling['top_k'], sampling['seed'])
    data = SelectedSamples(manifest, selection, model, 'val', 'conditional')
    selected = data.records[:2]
    if [r['path'] for r in selected] != [r['path'] for r in source['examples'][:2]]:
        raise ValueError('First two predefined examples changed')
    codec = FrozenEncodec().to(args.device).requires_grad_(False).eval()
    before = {'model': module_hashes(model), 'codec': module_hashes(codec)}
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'checkpoint': str(checkpoint), 'checkpoint_sha256': file_hash(checkpoint),
        'checkpoint_step': 1600, 'source_report': str(previous), 'source_report_sha256': file_hash(previous),
        'selection_sha256': file_hash(selection), 'manifest_sha256': file_hash(manifest),
        'planner_sampling': sampling, 'acoustic_seed': source['acoustic_seed'],
        'selection_rule': 'First two entries of the last predeclared fresh-audio selection; no outcome-based choice.',
        'new_training_performed': False, 'new_independent_evaluation_cases': False,
        'A_only_generation': True, 'test_split_evaluated': False,
        'asr_model': source['asr_model'], 'examples': [],
        'limitations': ['These are regenerations of previously evaluated development examples.',
            'ASR and waveform measurements do not replace human listening.',
            'WER against one recorded reply cannot measure response relevance or empathy.']}
    with torch.inference_mode():
        for index, row in enumerate(selected):
            batch = data.batch([index], device)
            # Only A features and requested style/voice enter generation. / 생성에는 A 특징과 요청한 스타일·음성만 사용합니다.
            inputs = person_a_only(batch)
            began = time.monotonic()
            generated = model.generate_batch(inputs, codec, seed=source['acoustic_seed'])
            torch.cuda.synchronize() if str(args.device).startswith('cuda') else None
            seconds = time.monotonic() - began
            response = generated['waveform'][0, :int(generated['audio_lengths'][0])]
            reference = batch['waveform'][0, :int(batch['waveform_len'][0])]
            entry = {key: row[key] for key in ('path', 'conversation_id', 'input_text')}
            entry.update(reference_text=row['response_text'], generation_seconds=seconds, paths={})
            for name, wave in [('generated_response', response), ('recorded_response', reference)]:
                value = save_wave(args.output, f'{index + 1:02d}_{name}.wav', wave.cpu().numpy())
                value.update(measurements(wave), normal_inference=name == 'generated_response')
                value['sha256'] = file_hash(args.output / value['file'])
                entry['paths'][name] = value
            previous_wave = previous.parent / source['examples'][index]['paths']['predicted_length_ar']['file']
            entry['identical_to_previous_generated_wav'] = entry['paths']['generated_response']['sha256'] == file_hash(previous_wave)
            report['examples'].append(entry)
            atomic_json(args.output / 'generated_report.json', report)
            print(json.dumps({'generated_example': index + 1, 'seconds': seconds,
                              'identical_to_last_run': entry['identical_to_previous_generated_wav']}), flush=True)
    after = {'model': module_hashes(model), 'codec': module_hashes(codec)}
    if before != after or file_hash(checkpoint) != source['checkpoint_sha256']:
        raise RuntimeError('Frozen generation state changed')
    report['all_model_and_codec_weights_unchanged'] = True
    model.to('cpu')
    del model, codec
    torch.cuda.empty_cache() if str(args.device).startswith('cuda') else None
    report = recognize(report, args.output, args.device)
    atomic_json(args.output / 'report.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
