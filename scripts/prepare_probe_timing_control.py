"""Keep speech features but match fusion's frame counts. / 음성 특징을 유지하며 융합 프레임 수만 맞춥니다."""

import argparse
import copy
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from model.full_speech.tensor_ops import align
from scripts.check_a_information import file_hash, labels_text, required_ctc_frames, save_tensor_file, write_json


VIEW = 'speech_resampled'


@torch.no_grad()
def resample_to_fused(features, target_length):
    if features.ndim != 2 or features.shape[1] != 512 or len(features) < 1 or target_length < 1:
        raise ValueError('Expected nonempty [frames,512] speech and positive fusion length')
    if features.dtype != torch.float32 or not bool(torch.isfinite(features).all()):
        raise ValueError('Speech features must be finite float32 values')
    source_mask = torch.ones(1, len(features), dtype=torch.bool, device=features.device)
    target_mask = torch.ones(1, target_length, dtype=torch.bool, device=features.device)
    # Use the exact production valid-prefix interpolation. / 실제 모델의 유효 구간 보간 함수를 그대로 씁니다.
    return align(features[None], source_mask, target_mask)[0].detach().cpu().clone()


def length_summary(rows):
    ratios = [row['target_frames'] / row['source_frames'] for row in rows]
    return {'count': len(rows), 'different_frame_counts': sum(row['target_frames'] != row['source_frames'] for row in rows),
            'source_frames_total': sum(row['source_frames'] for row in rows),
            'target_frames_total': sum(row['target_frames'] for row in rows),
            'target_over_source_ratio_mean': statistics.mean(ratios),
            'target_over_source_ratio_median': statistics.median(ratios),
            'target_over_source_ratio_min': min(ratios), 'target_over_source_ratio_max': max(ratios)}


def immutable_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Refusing to replace a different timing-control artifact: {path}')
    else:
        write_json(path, value)


def prepare_timing_cache(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Timing-control output must be a separate directory tree')
    source_cache, target_cache = source / 'cache', output / 'cache'
    completion = json.loads((source_cache / 'complete.json').read_text())
    source_index_hash = file_hash(source_cache / 'index.json')
    if source_index_hash != completion['index_sha256']:
        raise ValueError('Source cache index changed after extraction')
    source_index = json.loads((source_cache / 'index.json').read_text())
    original_provenance = json.loads((source_cache / 'provenance.json').read_text())
    if source_index['provenance'] != original_provenance:
        raise ValueError('Source provenance does not match its cache index')
    lengths = {split: [{'path': row['path'], 'conversation_id': row['conversation_id'],
                       'source_frames': row['lengths']['speech'], 'target_frames': row['lengths']['fused']}
                      for row in source_index['splits'][split]] for split in ('train', 'val')}
    for rows in lengths.values():
        if not rows or any(row['source_frames'] < 1 or row['target_frames'] < 1 for row in rows):
            raise ValueError('Invalid source or target frame counts')
    audit = {'source_output': str(source), 'source_index_sha256': source_index_hash,
             'source_provenance_sha256': file_hash(source_cache / 'provenance.json'),
             'source_view': 'speech', 'target_timing_view': 'fused', 'derived_view': VIEW,
             'splits': {split: length_summary(rows) for split, rows in lengths.items()},
             'method': 'model.full_speech.tensor_ops.align: linear interpolation, align_corners=False',
             'output_dtype': 'float32', 'original_cache_modified': False,
             'limitations': ['Only timing is changed; no fused features or B response targets enter the new features.',
                             'Equal frame counts remove this timing confound, not differences in probe optimization.']}
    audit['needed'] = any(row['different_frame_counts'] for row in audit['splits'].values())
    if not audit['needed']:
        audit['result'] = 'All frame counts are already equal; a separate timing probe is unnecessary.'
        immutable_json(output / 'timing_audit.json', audit)
        return audit
    # Preserve source provenance fields so original checkpoints remain comparable. / 원본 체크포인트 비교를 위해 이력 필드를 보존합니다.
    provenance = copy.deepcopy(original_provenance)
    provenance['derived_feature_dimensions'] = {VIEW: 512}
    provenance['derivation'] = {key: audit[key] for key in ('source_index_sha256', 'source_provenance_sha256',
        'source_view', 'target_timing_view', 'derived_view', 'method')}
    immutable_json(target_cache / 'provenance.json', provenance)
    index = {'provenance': provenance, 'splits': {}, 'ctc_infeasible': {VIEW: []}}
    for split in ('train', 'val'):
        index['splits'][split] = []
        for position, row in enumerate(source_index['splits'][split]):
            for view in ('speech', 'fused'):
                if file_hash(source_cache / row['files'][view]) != row['sha256'][view]:
                    raise ValueError(f'Source {view} cache changed: {row["path"]}')
            item = torch.load(source_cache / row['files']['speech'], map_location='cpu', weights_only=True)
            if (item['path'], item['conversation_id'], item['split'], item['view']) != (
                    row['path'], row['conversation_id'], split, 'speech'):
                raise ValueError('Source feature identity differs from its index')
            if item['features'].shape != (row['lengths']['speech'], 512) or item['length'] != len(item['features']):
                raise ValueError('Source speech dimensions differ from its index')
            labels = item['text_a'].detach().cpu().clone()
            if labels_text(labels) != row['reference_text']:
                raise ValueError('Cached A transcript differs from its source index')
            target_length = row['lengths']['fused']
            feature = resample_to_fused(item['features'], target_length)
            relative = f'{VIEW}/{split}/{position:06d}.pt'
            derived = {'features': feature, 'text_a': labels, 'length': target_length,
                       'path': row['path'], 'conversation_id': row['conversation_id'], 'split': split, 'view': VIEW}
            target = target_cache / relative
            if target.exists():
                previous = torch.load(target, map_location='cpu', weights_only=True)
                if any(not torch.equal(previous[key], value) if isinstance(value, torch.Tensor)
                       else previous[key] != value for key, value in derived.items()):
                    raise ValueError(f'Existing timing-control features differ: {target}')
            else:
                save_tensor_file(target, derived)
            needed = required_ctc_frames(labels)
            entry = {'path': row['path'], 'conversation_id': row['conversation_id'], 'split': split,
                     'reference_text': row['reference_text'], 'required_ctc_frames': needed,
                     'files': {VIEW: relative}, 'sha256': {VIEW: file_hash(target)}, 'lengths': {VIEW: target_length},
                     'source_speech_frames': item['length'], 'source_speech_sha256': row['sha256']['speech'],
                     'source_fused_sha256': row['sha256']['fused']}
            index['splits'][split].append(entry)
            if needed > target_length:
                index['ctc_infeasible'][VIEW].append({'split': split, 'path': row['path'],
                                                    'frames': target_length, 'required': needed})
            if (position + 1) % 64 == 0:
                print(json.dumps({'stage': 'derive_timing_control', 'split': split,
                                  'examples': position + 1, 'total': len(source_index['splits'][split])}), flush=True)
    if file_hash(source_cache / 'index.json') != source_index_hash:
        raise ValueError('Source index changed during derivation')
    immutable_json(target_cache / 'index.json', index)
    audit['ctc_infeasible_count'] = len(index['ctc_infeasible'][VIEW])
    immutable_json(output / 'timing_audit.json', audit)
    immutable_json(target_cache / 'complete.json', {'counts': {key: len(rows) for key, rows in index['splits'].items()},
        'index_sha256': file_hash(target_cache / 'index.json'),
        'ctc_infeasible_counts': {VIEW: audit['ctc_infeasible_count']}})
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True, help='Original probe output root containing cache/')
    parser.add_argument('--output', type=Path, required=True, help='Separate timing-control probe output root')
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = prepare_timing_cache(args.source, args.output)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
