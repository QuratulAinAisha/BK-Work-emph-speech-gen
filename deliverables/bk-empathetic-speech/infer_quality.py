"""Response inference from Person A only. / A의 정보만 사용하는 응답 추론."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from infer_full import read_person_a
from model.full_speech.loading import load_response_model
from model.full_speech.codec import FrozenEncodec
from model.full_speech.targets import FrozenSemanticTeacher, resample_audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True, help='Person-A mel/dmm/au NPZ')
    parser.add_argument('--audio', type=Path, required=True, help='Person-A waveform')
    parser.add_argument('--style', type=int, default=0)
    parser.add_argument('--speaker', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, payload = load_response_model(args.checkpoint)
    model.to(args.device).eval()
    teacher = FrozenSemanticTeacher(model.config).to(args.device)
    wave, rate = sf.read(args.audio, dtype='float32')
    if wave.ndim == 2:
        wave = wave.mean(1)
    if not np.isfinite(wave).all() or len(wave) / rate < .025:
        raise ValueError('Invalid Person-A audio')
    length = int(np.ceil(len(wave) / rate * 50 - 1e-9))
    values = torch.from_numpy(resample_audio(wave, rate, 16000)).to(args.device)
    values = (values - values.mean()) / (values.var(unbiased=False) + 1e-7).sqrt()
    with torch.inference_mode():
        hidden = teacher.model(values[None]).last_hidden_state
        speech_a = F.interpolate(hidden.transpose(1, 2), size=length, mode='linear', align_corners=False).transpose(1, 2)
    del teacher
    batch = read_person_a(args.features, args.device, args.style, args.speaker)
    # CLI choices override metadata; B targets are never loaded. / CLI 선택을 적용하며 B 정답은 읽지 않습니다.
    batch.update(style_id=torch.tensor([args.style], device=args.device),
                 speaker_id=torch.tensor([args.speaker], device=args.device), speech_a=speech_a,
                 speech_a_len=torch.tensor([length], device=args.device))
    codec = FrozenEncodec().to(args.device)
    result = model.generate_batch(batch, codec, seed=args.seed)
    wave = result['waveform'][0, :int(result['audio_lengths'][0])].cpu().numpy()
    peak = float(np.abs(wave).max())
    args.output.mkdir(parents=True, exist_ok=True)
    sf.write(args.output / 'response.wav', wave / max(1., peak / .95), 32000)
    np.savez_compressed(args.output / 'stages.npz', **{k: v.cpu().numpy() for k, v in result.items()})
    summary = {'epoch': payload.get('epoch', -1) + 1, 'duration': len(wave) / 32000, 'raw_peak': peak,
               'architecture': payload['architecture'], 'recovery_step': payload.get('recovery_step'),
               'style_id': args.style, 'speaker_group_id': args.speaker,
               'person_b_targets_used': False, 'quality_note': 'Requires listening and independent evaluation.'}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
