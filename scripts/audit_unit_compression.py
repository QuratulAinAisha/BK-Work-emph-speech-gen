"""Measure lossless adjacent-unit compression. / 인접 단위의 무손실 압축을 측정합니다."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from model.full_speech.units import SpeechCodebook
from prepare_quality import atomic_json


def integer_vector(value, name):
    value = np.asarray(value)
    if value.ndim != 1 or value.dtype.kind not in 'iu':
        raise ValueError(f'{name} must be a one-dimensional integer array')
    return value


def compress_units(units):
    units = integer_vector(units, 'units')
    if (units < 0).any():
        raise ValueError('Unit IDs must be nonnegative')
    if not len(units):
        return units.copy(), np.empty(0, dtype=np.int64)
    # Keep run lengths so expansion exactly restores every frame. / 길이를 보존해 모든 프레임을 정확히 복원합니다.
    starts = np.r_[0, np.flatnonzero(units[1:] != units[:-1]) + 1]
    return units[starts], np.diff(np.r_[starts, len(units)])


def expand_units(values, lengths):
    values = integer_vector(values, 'values')
    lengths = integer_vector(lengths, 'lengths')
    if len(values) != len(lengths) or (lengths <= 0).any() or (values < 0).any():
        raise ValueError('Each nonnegative unit requires one positive run length')
    return np.repeat(values, lengths)


def distribution(values):
    values = np.asarray(values)
    if not len(values):
        raise ValueError('Cannot summarize an empty distribution')
    return {'min': float(values.min()), 'mean': float(values.mean()),
            'p50': float(np.percentile(values, 50)), 'p90': float(np.percentile(values, 90)),
            'p95': float(np.percentile(values, 95)), 'max': float(values.max())}


def summarize_rows(rows, run_lengths):
    if not rows:
        raise ValueError('No selected rows')
    duration = sum(row['duration_seconds'] for row in rows)
    raw = sum(row['raw_units'] for row in rows)
    runs = sum(row['compressed_runs'] for row in rows)
    if min(duration, raw, runs) <= 0 or sum(run_lengths) != raw or len(run_lengths) != runs:
        raise ValueError('Inconsistent unit compression totals')
    return {'count': len(rows), 'duration_seconds': duration, 'raw_units': raw,
            'compressed_runs': runs, 'compressed_to_raw_ratio': runs / raw,
            'reduction_fraction': 1 - runs / raw, 'raw_units_per_second': raw / duration,
            'compressed_runs_per_second': runs / duration, 'exact_roundtrip_all': True,
            'duration_distribution_seconds': distribution([r['duration_seconds'] for r in rows]),
            'compressed_ratio_distribution': distribution([r['compressed_to_raw_ratio'] for r in rows]),
            'compressed_rate_distribution_hz': distribution([r['compressed_runs_per_second'] for r in rows]),
            'examples_at_or_below_diagram_12_5_hz': sum(r['compressed_runs_per_second'] <= 12.5 for r in rows),
            'run_length_distribution_frames': distribution(run_lengths),
            'run_length_histogram_frames': dict(sorted(Counter(map(int, run_lengths)).items())),
            'examples': rows}


def selected_rows(manifest, selection, split):
    wanted = selection[split]
    if not wanted or len(wanted) != len(set(wanted)):
        raise ValueError(f'{split} selection is empty or duplicated')
    wanted_set = set(wanted)
    rows = [r for r in manifest['records'] if r['path'] in wanted_set]
    if len(rows) != len(wanted) or any(r['split'] != split for r in rows):
        raise ValueError(f'{split} selection must retain original split membership')
    return rows


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--codebook', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(2)
    manifest = json.loads(args.manifest.read_text())
    selection = json.loads(args.selection.read_text())
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if selection['manifest_sha256'] != manifest_hash:
        raise ValueError('Selection manifest changed')
    if set(selection['train']) & set(selection['val']):
        raise ValueError('Train/validation selections overlap')
    payload = torch.load(args.codebook, map_location='cpu', weights_only=True)
    if payload.get('architecture') != 'bk_speech_codebook_v1' or payload.get('fit_split') != 'train':
        raise ValueError('Expected training-only fitted speech codebook')
    if payload['manifest_sha256'] != manifest_hash:
        raise ValueError('Codebook manifest differs')
    mean, std = (payload[name].to(args.device).float() for name in ('semantic_mean', 'semantic_std'))
    if mean.shape != (768,) or std.shape != (768,) or not torch.isfinite(mean).all() or \
            not torch.isfinite(std).all() or not (std > 0).all():
        raise ValueError('Invalid codebook normalization')
    book = SpeechCodebook(payload['centers']).to(args.device)
    sample_rate = manifest['target_contract']['sample_rate']
    if not isinstance(sample_rate, (int, float)) or sample_rate <= 0:
        raise ValueError('Invalid waveform sample rate')
    report = {'architecture_changed': False, 'test_split_read': False,
              'manifest_sha256': manifest_hash,
              'selection_sha256': hashlib.sha256(args.selection.read_bytes()).hexdigest(),
              'codebook_sha256': hashlib.sha256(args.codebook.read_bytes()).hexdigest(),
              'codebook_fit_selection_sha256': payload.get('selection_sha256'),
              'clusters': len(book.centers), 'waveform_sample_rate': sample_rate,
              'duration_source': 'Exact cached B waveform sample count divided by manifest sample rate.',
              'limitation': 'Adjacent identical units only. Exact recovery requires run lengths. '
                            'Rates do not establish semantic quality, learned duration accuracy, or improved response audio.',
              'splits': {}}
    for split in ('train', 'val'):
        examples, all_lengths, used_units = [], [], set()
        rows = selected_rows(manifest, selection, split)
        for index, row in enumerate(rows):
            # Read only selected B caches; never open test audio. / 선택한 B 캐시만 읽고 테스트 음성은 열지 않습니다.
            with np.load(args.manifest.parent / row['path'], allow_pickle=False) as data:
                semantic = torch.from_numpy(data['semantic'].astype(np.float32)).to(args.device)
                wave = data['waveform']
                if wave.ndim != 1 or not len(wave) or not np.isfinite(wave).all():
                    raise ValueError(f'Invalid waveform: {row["path"]}')
                duration = len(wave) / sample_rate
            if semantic.ndim != 2 or semantic.shape[1] != 768 or not len(semantic) or \
                    not torch.isfinite(semantic).all():
                raise ValueError(f'Invalid semantic frames: {row["path"]}')
            units = book.encode((semantic - mean) / std).cpu().numpy()
            values, lengths = compress_units(units)
            if not np.array_equal(expand_units(values, lengths), units):
                raise AssertionError(f'Unit roundtrip failed: {row["path"]}')
            all_lengths.extend(lengths.tolist())
            used_units.update(values.tolist())
            examples.append({'path': row['path'], 'conversation_id': row['conversation_id'],
                'duration_seconds': duration, 'raw_units': len(units), 'compressed_runs': len(values),
                'compressed_to_raw_ratio': len(values) / len(units),
                'raw_units_per_second': len(units) / duration,
                'compressed_runs_per_second': len(values) / duration, 'exact_roundtrip': True})
            if (index + 1) % 256 == 0:
                print(json.dumps({'split': split, 'completed': index + 1, 'total': len(rows)}), flush=True)
        report['splits'][split] = {**summarize_rows(examples, all_lengths),
            'unique_conversations': len({row['conversation_id'] for row in rows}),
            'unique_units': len(used_units)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    print(json.dumps({split: {key: value for key, value in result.items() if key not in
        ('examples', 'run_length_histogram_frames')} for split, result in report['splits'].items()}), flush=True)


if __name__ == '__main__':
    main()
