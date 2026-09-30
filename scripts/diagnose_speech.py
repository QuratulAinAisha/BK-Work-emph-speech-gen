"""Isolate codec, acoustic generator and planner. / 코덱·음향 생성기·계획기를 분리 검사합니다."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from infer_full import read_person_a
from model.full_speech import EmpatheticSpeechSystem
from model.full_speech.codec import FrozenEncodec
from model.full_speech.tensor_ops import mask_from_lengths


def word_error(reference, hypothesis):
    import re
    a = re.findall(r"[a-z]+(?:'[a-z]+)?", reference.lower())
    b = re.findall(r"[a-z]+(?:'[a-z]+)?", hypothesis.lower())
    previous = list(range(len(b) + 1))
    for i, word in enumerate(a, 1):
        current = [i]
        for j, other in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (word != other)))
        previous = current
    return previous[-1] / max(1, len(a))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, default=Path('outputs/bk_prepared'))
    parser.add_argument('--source', type=Path, default=Path('outputs/bk_source/source.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--examples', type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, payload = EmpatheticSpeechSystem.from_checkpoint(args.checkpoint)
    model.cuda().eval()
    codec = FrozenEncodec().cuda()
    rows = json.loads(args.source.read_text())['records']
    chosen, seen = [], set()
    for index, row in enumerate(rows):
        if row['split'] == 'test' and row['conversation_id'] not in seen:
            chosen.append(index)
            seen.add(row['conversation_id'])
        if len(chosen) == args.examples:
            break
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'epoch': payload['epoch'] + 1, 'examples': []}
    for index in chosen:
        source = rows[index]
        path = args.prepared / f'sample_{index:06d}.npz'
        batch = read_person_a(path, 'cuda', 0, 0)
        with np.load(path) as data:
            semantic = torch.from_numpy(data['semantic'].copy())[None].cuda()
            latents = torch.from_numpy(data['codec'].copy())[None].cuda()
            duration = float(data['duration'])
        lengths = torch.tensor([latents.shape[1]], device='cuda')
        wanted = torch.tensor([round(duration * 32000)], device='cuda')
        semantic_mask = torch.ones(semantic.shape[:2], device='cuda', dtype=torch.bool)
        with torch.inference_mode():
            encoded = model.encode_batch(batch)
            style, voice = model.embeddings(batch['style_id'], batch['speaker_id'], 1)
            oracle, _ = codec.decode(latents, lengths, wanted)
            generated = model.codec_generator.sample(mask_from_lengths(lengths),
                (semantic - model.semantic_mean) / model.semantic_std, semantic_mask,
                encoded['context'], encoded['context_mask'], encoded['affect'], style + voice,
                32, torch.Generator(device='cuda').manual_seed(42))
            acoustic, _ = codec.decode(generated * model.codec_std + model.codec_mean, lengths, wanted)
            full = model.generate(**batch, codec=codec, seed=42)['waveform']
        record = {'sample_index': index, 'conversation_id': source['conversation_id'],
                  'reference_text': source['response_text'], 'files': {}}
        reference, rate = sf.read(source['response_audio'], dtype='float32')
        if reference.ndim == 2:
            reference = reference.mean(1)
        import math
        reference = resample_poly(reference, 32000 // math.gcd(rate, 32000), rate // math.gcd(rate, 32000))
        for name, value in [('reference', reference), ('codec_reconstruction', oracle[0].cpu().numpy()),
                            ('oracle_semantics', acoustic[0].cpu().numpy()), ('full_generation', full[0].cpu().numpy())]:
            filename = f'{index:06d}_{name}.wav'
            peak = float(np.abs(value).max())
            sf.write(args.output / filename, value / max(1.0, peak / .95), 32000)
            record['files'][name] = {'file': filename, 'seconds': len(value) / 32000, 'raw_peak': peak}
        report['examples'].append(record)
    del model, codec
    torch.cuda.empty_cache()
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    processor = WhisperProcessor.from_pretrained('openai/whisper-base.en')
    recognizer = WhisperForConditionalGeneration.from_pretrained('openai/whisper-base.en').cuda().eval()
    for record in report['examples']:
        for name, entry in record['files'].items():
            audio, _ = sf.read(args.output / entry['file'], dtype='float32')
            inputs = processor(resample_poly(audio, 1, 2), sampling_rate=16000, return_tensors='pt', return_attention_mask=True)
            with torch.no_grad():
                ids = recognizer.generate(inputs.input_features.cuda(), attention_mask=inputs.attention_mask.cuda())
            entry['transcript'] = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            # WER is definitive only for a known spoken reference. / WER는 정해진 발화의 검사에만 직접 적용합니다.
            entry['reference_wer_diagnostic'] = word_error(record['reference_text'], entry['transcript'])
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
