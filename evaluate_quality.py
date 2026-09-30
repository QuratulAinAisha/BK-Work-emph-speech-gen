"""Generated validation audio and content checks. / 생성 검증 음성과 내용 검사."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from dataset.quality_speech_dataset import QualitySpeechDataset, collate_quality, person_a_only
from model.full_speech.loading import load_response_model
from model.full_speech.codec import FrozenEncodec
from train_full import move_batch
from scripts.diagnose_speech import word_error
from dataset.audio_affect_targets import audio_affect_targets


def assess(report, output, device):
    from scipy.signal import resample_poly
    from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoTokenizer, AutoModel
    for name in ('openai/whisper-tiny.en', 'openai/whisper-base.en'):
        processor = WhisperProcessor.from_pretrained(name)
        model = WhisperForConditionalGeneration.from_pretrained(name).to(device).eval()
        for row in report['examples']:
            # Reference transcription is a positive control. / 참조 음성 인식은 양성 대조군입니다.
            for key in ('generated', 'reference'):
                wave, _ = sf.read(output / row[key + '_file'], dtype='float32')
                inputs = processor(resample_poly(wave, 1, 2), sampling_rate=16000,
                                   return_tensors='pt', return_attention_mask=True)
                with torch.no_grad():
                    ids = model.generate(inputs.input_features.to(device), attention_mask=inputs.attention_mask.to(device))
                row.setdefault(key + '_asr', {})[name] = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        del model
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    encoder = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').eval()
    for row in report['examples']:
        first, second = row['generated_asr'].values()
        row['asr_disagreement_wer'] = word_error(first, second)
        row['reference_wer_diagnostic'] = word_error(row['reference_text'], second)
        row['reference_control_wer'] = word_error(row['reference_text'], list(row['reference_asr'].values())[-1])
        inputs = tokenizer([second, row['reference_text'], row['input_text']], return_tensors='pt', padding=True)
        with torch.no_grad():
            hidden = encoder(**inputs).last_hidden_state
        mask = inputs['attention_mask'][..., None]
        values = torch.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1), dim=-1)
        row['reference_cosine_diagnostic'] = float(values[0] @ values[1])
        row['input_cosine_diagnostic'] = float(values[0] @ values[2])
        audio, _ = sf.read(output / row['generated_file'], dtype='float32')
        affect, weights = audio_affect_targets(audio, 32000, second)
        row['measured_pitch_energy_rate'] = [float((affect[:, i] * weights[:, i]).sum() /
            max(1, weights[:, i].sum())) for i in (2, 3, 4)]
        row['word_rate_note'] = 'ASR-derived; unreliable when the transcript is wrong.'
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--count', type=int, default=6)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--skip-asr', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, payload = load_response_model(args.checkpoint)
    model.to(args.device).eval()
    codec = FrozenEncodec().to(args.device)
    dataset = QualitySpeechDataset(args.manifest, model.config, args.split)
    args.output.mkdir(parents=True, exist_ok=True)
    chosen, seen = [], set()
    # Round-robin styles while keeping conversations unique. / 대화 중복 없이 스타일을 순환 선택합니다.
    queues = {style: iter([i for i, row in enumerate(dataset.records) if row['style_id'] == style]) for style in range(6)}
    while len(chosen) < args.count:
        before = len(chosen)
        for style in range(6):
            for i in queues[style]:
                row = dataset.records[i]
                if row['conversation_id'] not in seen:
                    chosen.append(i); seen.add(row['conversation_id'])
                    break
            if len(chosen) >= args.count:
                break
        if len(chosen) == before:
            break
    report = {'checkpoint_epoch': payload['epoch'] + 1, 'split': args.split,
              'examples': [], 'limitations': ['ASR and embedding scores are diagnostics, not calibrated quality gates.',
              'No human quality or verified speaker-identity scores are claimed.']}
    for i in chosen:
        row = dataset.records[i]
        sample = dataset[i]
        batch = move_batch(collate_quality([sample]), torch.device(args.device))
        inputs = person_a_only(batch)
        generated = model.generate_batch(inputs, codec, seed=42)
        wave = generated['waveform'][0, :int(generated['audio_lengths'][0])].cpu().numpy()
        if not np.isfinite(wave).all():
            raise RuntimeError('Non-finite generated audio')
        peak = float(np.abs(wave).max())
        audio_file, reference_file = f'{i:06d}_generated.wav', f'{i:06d}_reference.wav'
        gain = 1 / max(1., peak / .95)
        sf.write(args.output / audio_file, wave * gain, 32000)
        ref = sample['waveform'].numpy()
        sf.write(args.output / reference_file, ref / max(1., float(np.abs(ref).max()) / .95), 32000)
        entry = {key: row[key] for key in ('conversation_id', 'input_text', 'response_text', 'style_id', 'speaker_id')}
        entry['reference_text'] = entry.pop('response_text')
        frames = wave[:len(wave) // 1280 * 1280].reshape(-1, 1280)
        entry.update(generated_file=audio_file, reference_file=reference_file, duration_seconds=len(wave) / 32000,
                     raw_peak=peak, output_gain=gain, raw_clip_fraction=float((np.abs(wave) >= 1.).mean()),
                     quiet_frame_fraction=float((np.sqrt((frames ** 2).mean(1)) < .005).mean()),
                     predicted_affect_mean=generated['affect_summary'][0].cpu().tolist())
        # Shuffle A while keeping requested style/voice fixed. / 스타일·음색을 고정하고 A만 교체합니다.
        other = move_batch(collate_quality([dataset[(i + 6) % len(dataset)]]), torch.device(args.device))
        wrong = person_a_only(other)
        wrong['style_id'], wrong['speaker_id'] = inputs['style_id'], inputs['speaker_id']
        if dataset.records[(i + 6) % len(dataset)]['conversation_id'] != row['conversation_id']:
            swapped = model.generate_batch(wrong, codec, seed=42)
            a = generated['semantic'].mean(1); b = swapped['semantic'].mean(1)
            entry['swapped_input_semantic_mean_distance'] = float((a - b).square().mean())
        report['examples'].append(entry)
    # Same A, same voice and seed isolate requested style effects. / A·음색·시드를 고정해 스타일 효과를 분리합니다.
    fixed = person_a_only(move_batch(collate_quality([dataset[chosen[0]]]), torch.device(args.device)))
    report['controlled_style_sweep'] = []
    for style_id in range(6):
        fixed['style_id'] = torch.tensor([style_id], device=args.device)
        result = model.generate_batch(fixed, codec, seed=42)
        audio = result['waveform'][0, :int(result['audio_lengths'][0])].cpu().numpy()
        filename = f'controlled_style_{style_id}.wav'
        peak = float(np.abs(audio).max())
        sf.write(args.output / filename, audio / max(1., peak / .95), 32000)
        report['controlled_style_sweep'].append({'style_id': style_id, 'file': filename,
            'duration_seconds': len(audio) / 32000, 'rms': float(np.sqrt((audio ** 2).mean())),
            'predicted_affect_mean': result['affect_summary'][0].cpu().tolist(),
            'note': 'Differences establish sensitivity, not correct style realization.'})
    del model, codec
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()
    if not args.skip_asr:
        report = assess(report, args.output, args.device)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    with (args.output / 'human_ratings.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['file', 'clarity_1_to_5', 'relevance_1_to_5', 'empathy_1_to_5', 'naturalness_1_to_5', 'notes'])
        for row in report['examples']:
            writer.writerow([row['generated_file'], '', '', '', '', ''])
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
