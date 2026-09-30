"""Check cached response alignment and paired audio errors. / 응답 정렬과 음성 오류 검사."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import soundfile as sf
from model.full_speech.targets import resample_audio
from prepare_quality import atomic_json


def main():
    root = Path('outputs/acoustic_controlled_v1')
    root.mkdir(exist_ok=True)
    source = json.loads(Path('outputs/bk_source/source.json').read_text())['records']
    mp = Path('outputs/bk_quality_prepared/manifest.json')
    manifest = json.loads(mp.read_text())
    bp = Path(manifest['base_manifest'])
    base = {r['path']: r for r in json.loads(bp.read_text())['records']}
    selection = json.loads(Path('outputs/broad_units_v1/selection.json').read_text())
    wanted = set(selection['train'] + selection['val'])
    errors, lengths, wave_errors = [], [], []
    for row in manifest['records']:
        if row['path'] not in wanted:
            continue
        original = source[row['source_index']]
        if any(row[k] != original[k] for k in ('conversation_id', 'split', 'style_id', 'speaker_id', 'response_text')):
            errors.append([row['path'], 'source metadata mismatch'])
        if row['reference_audio'] != original['response_audio'] or base[row['base_path']]['conversation_id'] != row['conversation_id']:
            errors.append([row['path'], 'audio owner mismatch'])
        with np.load(mp.parent / row['path']) as data, np.load(bp.parent / row['base_path']) as old:
            frames, samples = len(data['semantic']), len(data['waveform'])
            if frames != len(old['codec']) or frames != int(np.ceil(samples / 640)):
                errors.append([row['path'], 'frame length mismatch'])
            lengths.append(abs(frames / 50 - samples / 32000))
            raw, rate = sf.read(original['response_audio'], dtype='float32')
            if raw.ndim == 2:
                raw = raw.mean(1)
            wave = resample_audio(raw, rate, 32000)
            difference = float(np.max(np.abs(wave - data['waveform']))) if wave.shape == data['waveform'].shape else float('inf')
            wave_errors.append(difference)
            if difference > 1e-6:
                errors.append([row['path'], 'cached waveform differs from source'])
    result = {'count': len(lengths), 'passed': not errors, 'errors': errors,
              'max_frame_rounding_seconds': max(lengths), 'max_waveform_absolute_difference': max(wave_errors),
              'limitation': 'Checks owners, waveform identity and frame lengths; not phonetic alignment. HuBERT boundary frames are linearly interpolated to codec length.'}
    atomic_json(root / 'alignment_audit.json', result)
    print(json.dumps(result), flush=True)
    if errors:
        raise RuntimeError('Alignment audit failed')


if __name__ == '__main__':
    main()
