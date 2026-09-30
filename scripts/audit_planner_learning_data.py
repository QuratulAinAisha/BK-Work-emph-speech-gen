"""Read-only planner target, masking and data audit. / 계획기 타깃·마스크·자료를 읽기 전용으로 검사합니다."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from dataset.quality_speech_dataset import QualitySpeechDataset, collate_quality
from model.full_speech.quality import normalize_text, text_ids
from model.full_speech.recovery import RecoverySpeechSystem
from model.full_speech.tensor_ops import counts, mask_from_lengths
from model.full_speech.units import UnitSpeechSystem
from prepare_quality import atomic_json
from scripts.analyze_planner_repetition import sequence_distribution
from train_full import move_batch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def top_counts(counter, total, limit=20):
    return [{'value': key, 'count': count, 'fraction': count / max(1, total)}
            for key, count in sorted(counter.items(), key=lambda item: (-item[1], str(item[0])))[:limit]]


def duplicate_groups(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        key = tuple(normalize_text(row[name]) for name in keys)
        groups[key].append(row['path'])
    duplicates = [{'text': list(key), 'paths': paths} for key, paths in groups.items() if len(paths) > 1]
    return {'groups': len(duplicates), 'records_in_duplicate_groups': sum(len(row['paths']) for row in duplicates),
            'examples': duplicates[:20]}


def transcript_summary(rows):
    if not rows:
        raise ValueError('Transcript audit requires selected records')
    responses = [normalize_text(row['response_text']).split() for row in rows]
    prefixes = {}
    for width in (2, 3, 4, 5):
        frequencies = Counter(' '.join(words[:width]) for words in responses if len(words) >= width)
        prefixes[str(width)] = {'eligible_records': sum(frequencies.values()),
            'top_prefix_fraction_of_all_records': max(frequencies.values(), default=0) / len(rows),
            'top_prefixes': top_counts(frequencies, len(rows))}
    copied = [row['path'] for row in rows if normalize_text(row['input_text']) == normalize_text(row['response_text'])]
    return {'count': len(rows), 'input_duplicates': duplicate_groups(rows, ('input_text',)),
            'response_duplicates': duplicate_groups(rows, ('response_text',)),
            'pair_duplicates': duplicate_groups(rows, ('input_text', 'response_text')),
            'identical_a_and_b_text_paths': copied,
            'response_prefixes': prefixes,
            'voice_groups': dict(Counter(str(row.get('speaker_id')) for row in rows)),
            'styles': dict(Counter(str(row.get('style_id')) for row in rows))}


def cross_split_duplicates(train, validation):
    result = {}
    for name, fields in [('input', ('input_text',)), ('response', ('response_text',)),
                         ('pair', ('input_text', 'response_text'))]:
        lookup = defaultdict(list)
        for row in train:
            lookup[tuple(normalize_text(row[field]) for field in fields)].append(row['path'])
        matches = []
        for row in validation:
            key = tuple(normalize_text(row[field]) for field in fields)
            if key in lookup:
                matches.append({'validation_path': row['path'], 'training_paths': lookup[key], 'text': list(key)})
        result[name] = {'matched_validation_records': len(matches), 'examples': matches[:20]}
    return result


def unit_details(ids):
    if not ids or any(type(value) is not int or value < 0 for value in ids):
        raise ValueError('Expected nonempty nonnegative unit IDs')
    runs, length = [], 1
    for before, after in zip(ids, ids[1:]):
        if before == after:
            length += 1
        else:
            runs.append(length)
            length = 1
    runs.append(length)
    return {'frames': len(ids), 'unit_changes': len(runs) - 1, 'unit_runs': len(runs),
            'longest_run_frames': max(runs), 'mean_run_frames': len(ids) / len(runs),
            'first_unit': ids[0], 'last_unit': ids[-1]}


def numeric_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {'min': float(values.min()), 'median': float(np.median(values)),
            'mean': float(values.mean()), 'p95': float(np.quantile(values, .95)), 'max': float(values.max())}


def summarize_units(sequences, vocabulary_size):
    histogram = Counter(value for row in sequences for value in row)
    details = [unit_details(row) for row in sequences]
    prefix = [row[:max(1, math.ceil(len(row) * .2))] for row in sequences]
    suffix = [row[max(1, math.ceil(len(row) * .2)):] for row in sequences]
    return {**sequence_distribution(sequences), 'vocabulary_size': vocabulary_size,
            'unused_vocabulary_count': vocabulary_size - len(histogram),
            'unit_frequency': [histogram.get(index, 0) for index in range(vocabulary_size)],
            'top_units': top_counts(histogram, sum(histogram.values())),
            'first_units': top_counts(Counter(row['first_unit'] for row in details), len(details)),
            'last_units': top_counts(Counter(row['last_unit'] for row in details), len(details)),
            'longest_run_frames_max': max(row['longest_run_frames'] for row in details),
            'mean_changes_per_sequence': sum(row['unit_changes'] for row in details) / len(details),
            'prefix_20pct_time': sequence_distribution(prefix),
            'remainder_80pct_time': sequence_distribution([row for row in suffix if row]) if any(suffix) else None}


@torch.inference_mode()
def check_target_batch(model, samples, device):
    batch = move_batch(collate_quality(samples), device)
    raw, mask = RecoverySpeechSystem.target(model, batch, 'semantic', 768)
    lengths = batch['semantic_len']
    if not torch.equal(mask, mask_from_lengths(lengths)):
        raise ValueError('Target masks do not match true lengths')
    if not torch.equal(lengths, counts(batch['duration'], model.config.semantic_hz)):
        raise ValueError('B unit frame counts disagree with response duration')
    if not torch.isfinite(model.semantic_mean).all() or not torch.isfinite(model.semantic_std).all() or (model.semantic_std <= 0).any():
        raise ValueError('Invalid checkpoint normalization statistics')
    normalized = (raw - model.semantic_mean) / model.semantic_std
    book = model.semantic_planner.codebook
    ids = book.encode(normalized)
    quantized, quantized_mask = model.target(batch, 'semantic', 768)
    round_trip = book.encode((quantized - model.semantic_mean) / model.semantic_std)
    if not torch.equal(mask, quantized_mask) or not torch.equal(ids[mask], round_trip[mask]):
        raise ValueError('Normalized quantized targets change valid unit IDs on round trip')
    if bool(batch['semantic'][~mask].any()) or bool(quantized[~mask].any()):
        raise ValueError('Collation or quantized-target padding is nonzero')
    changed = dict(batch)
    changed['semantic'] = batch['semantic'].clone()
    changed['semantic'][~mask] = 12345.
    changed_raw, changed_mask = RecoverySpeechSystem.target(model, changed, 'semantic', 768)
    changed_ids = book.encode((changed_raw - model.semantic_mean) / model.semantic_std)
    if not torch.equal(ids[mask], changed_ids[changed_mask]):
        raise ValueError('Padding mutation changed valid target units')
    records = []
    centers = book.centers.detach().cpu().double()
    for index, length in enumerate(lengths.tolist()):
        valid = normalized[index, :length]
        positions = sorted({0, length // 2, length - 1})
        # Independent double-precision distances check sampled assignments. / 독립적인 배정밀도 거리로 일부 단위 배정을 검사합니다.
        selected = valid[positions].cpu().double()
        distances = (selected[:, None] - centers[None]).square().sum(-1)
        independent = distances.argmin(1)
        actual = ids[index, positions].cpu()
        errors = []
        for offset, position in enumerate(positions):
            if actual[offset] != independent[offset]:
                errors.append({'frame': position, 'production_id': int(actual[offset]), 'float64_id': int(independent[offset]),
                    'squared_distance_excess': float(distances[offset, actual[offset]] - distances[offset, independent[offset]])})
        values = ids[index, :length].cpu().tolist()
        records.append({'target_unit_ids': values, **unit_details(values),
            'float64_assignment_checks': len(positions), 'float64_assignment_differences': errors,
            'normalized_rms': float(valid.float().square().mean().sqrt()),
            'normalized_quantization_mse': float((valid - book.centers[ids[index, :length]]).float().square().mean())})
    return records, normalized, mask, {'padding_frames_checked': int((~mask).sum()),
        'cached_padding_zero': True, 'unit_target_padding_zero': True, 'padding_invariance': True,
        'normalization_round_trip': True, 'duration_frames_consistent': True}


@torch.no_grad()
def check_mask_path(planner, normalized, mask):
    previous_training = planner.training
    captured = []
    original_logits = planner.logits
    expected_ids = planner.codebook.encode(normalized)
    current = {}

    def logits(ids, hidden, valid, *args, **kwargs):
        if not torch.equal(ids[valid], expected_ids[valid]) or bool((hidden & ~valid).any()):
            raise ValueError('Masking selected padding or changed target indices')
        if not hidden.any(1).all():
            raise ValueError('A training example has no supervised hidden positions')
        current.update(hidden=hidden, valid=valid)
        return original_logits(ids, hidden, valid, *args, **kwargs)

    def inert_denoiser(values, fraction, local, context, style, valid, context_mask):
        hidden = current['hidden']
        visible = valid & ~hidden
        expected = planner.mask_embedding.to(values.dtype).expand_as(values)
        if not torch.equal(values[hidden], expected[hidden]):
            raise ValueError('Hidden B targets entered the denoiser inputs')
        if not torch.equal(values[visible], planner.codebook.centers[expected_ids][visible]):
            raise ValueError('Visible B hint IDs are misaligned')
        if not torch.equal(fraction, hidden.sum(1).float() / valid.sum(1)):
            raise ValueError('DiT time does not equal the actual hidden fraction')
        captured.append({'hidden_frames': int(hidden.sum()), 'valid_frames': int(valid.sum()),
                         'fully_hidden_examples': int((hidden.sum(1) == valid.sum(1)).sum()),
                         'examples': len(valid), 'hidden_targets_masked': True})
        # Inert logits isolate data wiring, not model quality. / 무의미한 로짓으로 품질이 아닌 자료 연결만 검사합니다.
        return values.new_zeros(*values.shape[:2], len(planner.codebook.centers))

    device = normalized.device
    devices = [device.index] if device.type == 'cuda' else []
    context = normalized.new_zeros(len(mask), 1, 512)
    cmask = torch.ones(len(mask), 1, dtype=torch.bool, device=device)
    affect = normalized.new_zeros(len(mask), 1, 6)
    style = normalized.new_zeros(len(mask), planner.config.hidden_dim)
    try:
        with torch.random.fork_rng(devices=devices), patch.object(planner, 'logits', side_effect=logits), \
                patch.object(planner.denoiser, 'forward', side_effect=inert_denoiser):
            for training in (True, False):
                planner.train(training)
                for seed in (42, 43):
                    torch.manual_seed(seed)
                    before = len(captured)
                    planner.estimate(normalized, mask, context, cmask, affect, style)
                    captured[before].update(training=training, seed=seed)
                    if not training and captured[before]['hidden_frames'] != int(mask.sum()):
                        raise ValueError('Default validation is not fully masked')
    finally:
        planner.train(previous_training)
    return {'passed': True, 'measurements': captured, 'inert_logits_used': True,
            'scope': 'Actual estimate/logits mask and embedding path; no model-quality scores.'}


def main():
    parser = argparse.ArgumentParser()
    for name in ('checkpoint', 'manifest', 'selection', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--train-count', type=int, default=64)
    parser.add_argument('--val-count', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    if min(args.train_count, args.val_count, args.batch_size) < 1:
        parser.error('Sample counts and batch size must be positive')
    torch.set_num_threads(2)
    selection = json.loads(args.selection.read_text())
    manifest_hash = sha256(args.manifest)
    if selection.get('manifest_sha256') != manifest_hash:
        raise ValueError('Selection and manifest hashes differ')
    metadata = json.loads(args.manifest.read_text())
    # Never open or analyze test records. / 테스트 레코드는 열거나 분석하지 않습니다.
    lookup = {row['path']: row for row in metadata['records'] if row['split'] in ('train', 'val')}
    selected = {}
    for split in ('train', 'val'):
        paths = selection[split]
        if not paths or len(set(paths)) != len(paths) or any(path not in lookup for path in paths):
            raise ValueError('Selected paths missing, repeated or outside train/development')
        selected[split] = [lookup[path] for path in paths]
        if any(row['split'] != split for row in selected[split]):
            raise ValueError('Selected paths cross split boundaries')
        if len({row['conversation_id'] for row in selected[split]}) != len(paths):
            raise ValueError('Selected conversation IDs are repeated')
    if {row['conversation_id'] for row in selected['train']} & {row['conversation_id'] for row in selected['val']}:
        raise ValueError('Train/development conversation overlap')
    model, payload = UnitSpeechSystem.from_checkpoint(args.checkpoint)
    model.to(args.device).requires_grad_(False).eval()
    report = {'checkpoint': str(args.checkpoint), 'checkpoint_sha256': sha256(args.checkpoint),
        'checkpoint_step': payload.get('recovery_step'), 'manifest_sha256': manifest_hash,
        'selection_sha256': sha256(args.selection), 'test_examples_opened': 0,
        'shared_manifest_metadata_read': True, 'normalization_refit': False,
        'target_contract': 'Cache stores continuous B HuBERT features; unit IDs are derived using checkpoint normalization and codebook.',
        'normalization': {'mean_min': float(model.semantic_mean.min()), 'mean_max': float(model.semantic_mean.max()),
                         'std_min': float(model.semantic_std.min()), 'std_max': float(model.semantic_std.max())},
        'transcripts': {split: transcript_summary(rows) for split, rows in selected.items()},
        'cross_split_exact_duplicates': cross_split_duplicates(selected['train'], selected['val']),
        'splits': {}, 'hard_checks_passed': True,
        'limitations': ['Automated checks cannot judge empathy, conversation appropriateness or actual voice identities.',
            'Shared manifest metadata contains all splits; only selected train/development example files are opened.',
            'Exact transcript duplicates and frequent prefixes are descriptive; they do not establish wrong labels.',
            'Unit repeats and run boundaries are not spoken repetitions, phoneme boundaries or word alignment.',
            'Target statistics describe the sampled examples, not the entire dataset.',
            'Mask-path checks use inert logits and provide no evidence of reconstruction or response quality.',
            'Source-A timing was checked separately; this audit does not repeat alignment or listen to audio.']}
    for split, maximum in [('train', args.train_count), ('val', args.val_count)]:
        data = QualitySpeechDataset(args.manifest, model.config, split)
        positions = {row['path']: index for index, row in enumerate(data.records)}
        rows = selected[split][:maximum]
        records, batch_checks, anomalies = [], [], []
        mask_check = None
        for begin in range(0, len(rows), args.batch_size):
            group = rows[begin:begin + args.batch_size]
            samples = []
            for row in group:
                index = positions[row['path']]
                value = data[index]
                base_row = data.base.records[data.indices[index]]
                if (base_row['path'], base_row['conversation_id'], base_row['split']) != (
                        row['base_path'], row['conversation_id'], split):
                    raise ValueError('Quality/base dataset indices identify different conversations')
                with np.load(args.manifest.parent / row['path'], allow_pickle=False) as cache:
                    if not torch.equal(torch.from_numpy(cache['semantic'].astype(np.float32)), value['semantic']):
                        raise ValueError('Loader semantic targets differ from cached values')
                if not torch.equal(text_ids(row['input_text']), value['text_a']) or not torch.equal(text_ids(row['response_text']), value['text_b']):
                    raise ValueError('A/B cached transcript IDs differ from paired manifest text')
                if (int(value['style_id']), int(value['speaker_id'])) != (row['style_id'], row['speaker_id']):
                    raise ValueError('Style or voice-group labels differ from manifest')
                samples.append(value)
            measured, normalized, mask, checks = check_target_batch(model, samples, torch.device(args.device))
            batch_checks.append(checks)
            if mask_check is None:
                mask_check = check_mask_path(model.semantic_planner, normalized, mask)
            for row, sample, metrics in zip(group, samples, measured):
                duration = float(sample['duration'])
                waveform_duration = len(sample['waveform']) / model.config.sample_rate
                words = len(normalize_text(row['response_text']).split())
                input_words = len(normalize_text(row['input_text']).split())
                flags = []
                if abs(duration - waveform_duration) > 1 / model.config.sample_rate:
                    flags.append('cached_waveform_duration_differs_by_more_than_one_sample')
                if not .5 <= words / max(duration, 1e-9) <= 5.:
                    flags.append('response_words_per_second_outside_descriptive_0.5_to_5_range')
                if not .1 <= words / max(1, input_words) <= 4.:
                    flags.append('response_input_word_ratio_outside_descriptive_0.1_to_4_range')
                if metrics['float64_assignment_differences']:
                    flags.append('float32_float64_nearest_center_difference')
                metrics.update(path=row['path'], conversation_id=row['conversation_id'],
                    input_text=row['input_text'], response_text=row['response_text'], duration_seconds=duration,
                    waveform_seconds=waveform_duration, response_words=words, response_words_per_second=words / duration,
                    input_words=input_words, response_input_word_ratio=words / max(1, input_words),
                    a_speech_frames=len(sample['speech_a']), a_characters=len(sample['text_a']), b_characters=len(sample['text_b']),
                    style_id=row['style_id'], voice_group_id=row['speaker_id'], flags=flags)
                records.append(metrics)
                if flags:
                    anomalies.append({key: metrics[key] for key in ('path', 'conversation_id', 'flags', 'duration_seconds', 'waveform_seconds')})
            if len(records) % 32 == 0:
                print(json.dumps({'split': split, 'audited': len(records), 'total': len(rows)}), flush=True)
        report['splits'][split] = {'selected_transcript_count': len(selected[split]), 'sampled_cache_count': len(records),
            'sampling': 'First requested count in the predeclared selection order',
            'batch_checks': batch_checks, 'mask_path_check': mask_check,
            'length_summaries': {key: numeric_summary([row[key] for row in records]) for key in (
                'frames', 'a_speech_frames', 'a_characters', 'b_characters', 'duration_seconds',
                'response_words_per_second', 'response_input_word_ratio')},
            'independent_assignment_checks': sum(row['float64_assignment_checks'] for row in records),
            'independent_assignment_difference_count': sum(len(row['float64_assignment_differences']) for row in records),
            'units': summarize_units([row['target_unit_ids'] for row in records], len(model.semantic_planner.codebook.centers)),
            'anomalies': anomalies, 'representative_pairs': [{key: row[key] for key in (
                'path', 'conversation_id', 'input_text', 'response_text', 'duration_seconds', 'flags')} for row in records[:8]],
            'examples': records}
    if sha256(args.checkpoint) != report['checkpoint_sha256']:
        raise ValueError('Checkpoint changed during audit')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    print(json.dumps({'hard_checks_passed': True, 'cache_counts': {key: value['sampled_cache_count'] for key, value in report['splits'].items()},
                      'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
