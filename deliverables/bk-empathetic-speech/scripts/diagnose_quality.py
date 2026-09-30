"""Isolate improved speech stages. / 개선 음성 단계를 분리 검사합니다."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from dataset.quality_speech_dataset import QualitySpeechDataset, collate_quality, person_a_only
from model.full_speech.loading import load_response_model
from model.full_speech.codec import FrozenEncodec
from train_full import move_batch
from scripts.diagnose_speech import word_error


def save_wave(output, name, wave):
    wave = np.asarray(wave, dtype=np.float32).reshape(-1)
    if not wave.size or not np.isfinite(wave).all():
        raise ValueError('Empty or non-finite waveform')
    peak = float(np.abs(wave).max())
    gain = min(1., .95 / max(peak, 1e-8))
    sf.write(output / name, wave * gain, 32000)
    return {'file': name, 'seconds': len(wave) / 32000, 'raw_peak': peak,
            'raw_clip_fraction': float((np.abs(wave) >= 1).mean()), 'output_gain': gain}


def recognize(report, output, device):
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    processor = WhisperProcessor.from_pretrained('openai/whisper-base.en')
    recognizer = WhisperForConditionalGeneration.from_pretrained('openai/whisper-base.en').to(device).eval()
    for row in report['examples']:
        for entry in row['paths'].values():
            audio, rate = sf.read(output / entry['file'], dtype='float32')
            assert rate == 32000
            inputs = processor(resample_poly(audio, 1, 2), sampling_rate=16000,
                               return_tensors='pt', return_attention_mask=True)
            with torch.inference_mode():
                ids = recognizer.generate(inputs.input_features.to(device),
                    attention_mask=inputs.attention_mask.to(device), max_new_tokens=128)
            entry['asr'] = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            entry['reference_wer'] = word_error(row['reference_text'], entry['asr'])
            entry['asr_token_limit_reached'] = ids.shape[-1] >= 128
        (output / 'partial_report.json').write_text(json.dumps(report, indent=2) + '\n')
    report['summary'] = {name: {
        'mean_reference_wer': float(np.mean([r['paths'][name]['reference_wer'] for r in report['examples']])),
        'median_reference_wer': float(np.median([r['paths'][name]['reference_wer'] for r in report['examples']])),
        'count': len(report['examples'])} for name in report['examples'][0]['paths']}
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'val'], default='val')
    parser.add_argument('--count', type=int, default=6)
    parser.add_argument('--steps', type=int, nargs='+', default=[4, 8, 24])
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--oracle-only', action='store_true')
    parser.add_argument('--unit-codebook', type=Path, help='Explicit quantized-oracle diagnostic')
    args = parser.parse_args()
    if args.count < 1 or any(x < 1 for x in args.steps):
        parser.error('Positive count and steps required')
    torch.set_num_threads(2)
    model, payload = load_response_model(args.checkpoint)
    model.to(args.device).eval()
    codebook = None
    if args.unit_codebook:
        from model.full_speech.units import load_codebook
        codebook, _ = load_codebook(args.unit_codebook, model)
        codebook.to(args.device)
    codec = FrozenEncodec().to(args.device)
    data = QualitySpeechDataset(args.manifest, model.config, args.split)
    if args.selection:
        selected = set(json.loads(args.selection.read_text())[args.split])
        chosen = [i for i, r in enumerate(data.records) if r['path'] in selected][:args.count]
    else:
        chosen, seen = [], set()
        for style in range(6):
            for i, row in enumerate(data.records):
                if row['style_id'] == style and row['conversation_id'] not in seen:
                    chosen.append(i); seen.add(row['conversation_id']); break
            if len(chosen) == args.count:
                break
    if not chosen:
        raise ValueError('No diagnostic examples selected')
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'checkpoint': str(args.checkpoint), 'checkpoint_epoch': payload.get('epoch', -1) + 1,
              'architecture': payload.get('architecture'), 'recovery_step': payload.get('recovery_step'),
              'planner_memory_mode': getattr(model.config, 'planner_memory_mode', 'fused'),
              'length_and_acoustic_memory': 'original_fused',
              'split': args.split, 'examples': [], 'asr_model': 'openai/whisper-base.en',
              'limitations': ['Oracle paths deliberately use B targets for diagnosis only.',
                  'Full predicted-length paths use only A inputs and requested style/voice.',
                  'Reference WER measures reconstruction; free replies may use different valid words.',
                  'ASR can hallucinate and does not establish human-perceived quality.']}
    if args.unit_codebook:
        report['unit_codebook_sha256'] = hashlib.sha256(args.unit_codebook.read_bytes()).hexdigest()
    for index in chosen:
        row, sample = data.records[index], data[index]
        batch = move_batch(collate_quality([sample]), torch.device(args.device))
        inputs = person_a_only(batch)
        entry = {k: row[k] for k in ['conversation_id', 'style_id', 'speaker_id', 'input_text']}
        entry.update(reference_text=row['response_text'], path=row['path'], paths={})
        waves = {'reference': sample['waveform'].numpy()}
        with torch.inference_mode():
            lengths = batch['codec_len']
            wanted = batch['waveform_len']
            reconstructed, _ = codec.decode(batch['codec'], lengths, wanted)
            waves['codec_reconstruction'] = reconstructed[0, :int(wanted[0])].cpu().numpy()
            semantics = (batch['semantic'] - model.semantic_mean) / model.semantic_std
            oracle = model.generate_batch(inputs, codec, seed=42, oracle_semantic=semantics,
                                          oracle_duration=batch['duration'])
            waves['oracle_semantics'] = oracle['waveform'][0, :int(oracle['audio_lengths'][0])].cpu().numpy()
            if codebook is not None:
                quantized = model.generate_batch(inputs, codec, seed=42, oracle_semantic=codebook(semantics),
                                                oracle_duration=batch['duration'])
                waves['oracle_units'] = quantized['waveform'][0, :int(quantized['audio_lengths'][0])].cpu().numpy()
            if not args.oracle_only:
                for steps in args.steps:
                    model.config.semantic_steps = steps
                    for name, duration in [('reference_length', batch['duration']), ('predicted_length', None)]:
                        generated = model.generate_batch(inputs, codec, seed=42, oracle_duration=duration)
                        waves[f'{name}_steps_{steps}'] = generated['waveform'][0, :int(generated['audio_lengths'][0])].cpu().numpy()
        for name, wave in waves.items():
            entry['paths'][name] = save_wave(args.output, f'{index:06d}_{name}.wav', wave)
        report['examples'].append(entry)
        print(json.dumps({'generated_case': index, 'paths': list(entry['paths'])}), flush=True)
    del model, codec
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()
    report = recognize(report, args.output, args.device)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    main()
