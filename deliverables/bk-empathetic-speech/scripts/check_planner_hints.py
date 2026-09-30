"""Oracle B-unit hints isolate planner failure. / B 정답 단위 힌트로 계획기 문제를 분리합니다."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F

from dataset.quality_speech_dataset import QualitySpeechDataset, collate_quality, person_a_only
from model.full_speech.tensor_ops import counts, mask_from_lengths
from model.full_speech.units import UnitSpeechSystem
from prepare_quality import atomic_json
from scripts.evaluate_planner_controls import next_different_conversation, planner_inputs
from train_full import move_batch


RATIOS = (.25, .5, .75, 1.)
MASK_SEEDS = (42, 43)
REGIONS = ('all_hidden', 'prefix_20pct_time', 'remainder_80pct_time')
LIMITATIONS = [
    'B units are deliberate diagnostic hints; they never enter the A encoder.',
    'Correct B duration/sequence length is supplied in every condition, including 100% hidden.',
    'The first 20% of frames is a temporal proxy, not a word boundary or verified generic opening.',
    'Unit metrics score hidden positions only; visible correct hints are excluded.',
    'Permuted visible B hints preserve the same positions, number and unit histogram; only visible IDs are shuffled.',
    'Visible-hint order sensitivity tests completion ability, not A-to-B response understanding.',
    'Two mask seeds are repeated measurements of the same conversations, not independent examples.',
    'Reference WER does not establish relevance or empathy; valid free replies may differ from B.',
    'ASR can hallucinate and is not a substitute for listening.',
]


def nested_hidden_mask(mask, ratio, seed):
    if mask.ndim != 2 or mask.dtype != torch.bool or not mask.any(1).all() or not 0 <= ratio <= 1:
        raise ValueError('Expected nonempty valid masks and hidden ratio within [0, 1]')
    hidden = torch.zeros_like(mask)
    # CPU permutations make masks repeatable across devices. / CPU 순열로 장치 간 마스크를 재현합니다.
    for index, row in enumerate(mask):
        valid = row.nonzero(as_tuple=True)[0]
        generator = torch.Generator(device='cpu').manual_seed(seed + index)
        order = torch.randperm(len(valid), generator=generator).to(mask.device)
        hidden[index, valid[order[:math.ceil(len(valid) * ratio)]]] = True
    return hidden


def mask_metadata(hidden, mask, ratio, seed):
    return {'requested_hidden_ratio': ratio, 'mask_seed': seed, 'hidden_frames': int(hidden.sum()),
            'actual_hidden_ratio': float(hidden.sum()) / int(mask.sum()),
            'mask_sha256': hashlib.sha256(hidden.cpu().numpy().tobytes()).hexdigest()}


def score_hidden(logits, targets, hidden, mask):
    if logits.shape[:2] != targets.shape or targets.shape != hidden.shape or hidden.shape != mask.shape:
        raise ValueError('Unit metric shapes differ')
    if (hidden & ~mask).any() or not torch.isfinite(logits).all():
        raise ValueError('Invalid hidden positions or logits')
    lengths = mask.sum(1)
    ordinal = mask.long().cumsum(1) - 1
    prefix = (ordinal < (lengths.float() * .2).ceil().long()[:, None]) & mask
    regions = {'all_hidden': hidden, 'prefix_20pct_time': hidden & prefix,
               'remainder_80pct_time': hidden & ~prefix}
    result = {}
    for name, chosen in regions.items():
        frames = int(chosen.sum())
        ce_sum = float(F.cross_entropy(logits[chosen].float(), targets[chosen], reduction='sum')) if frames else 0.
        correct = int((logits.argmax(-1)[chosen] == targets[chosen]).sum())
        result[name] = {'hidden_frames': frames, 'ce_sum': ce_sum, 'correct_units': correct,
                        'hidden_only_ce': ce_sum / frames if frames else None,
                        'hidden_only_argmax_accuracy': correct / frames if frames else None}
    return result


@torch.inference_mode()
def prepare_pair(model, batch, donor_batch):
    if model.training or len(batch['style_id']) != 1 or len(donor_batch['style_id']) != 1:
        raise ValueError('Use eval mode and one conversation per pair')
    original, donor = person_a_only(batch), person_a_only(donor_batch)
    for key in ('style_id', 'speaker_id'):
        donor[key] = original[key]
    # A sees only A inputs; B units are handled separately below. / A 인코더에는 A만 주고 B 단위는 따로 다룹니다.
    contexts = {}
    for name, inputs in (('correct_a', original), ('shuffled_a', donor)):
        encoded = model.encode_batch(inputs)
        contexts[name] = planner_inputs(model, encoded)
    style, _ = model.embeddings(original['style_id'], original['speaker_id'], 1)
    mask = mask_from_lengths(batch['semantic_len'])
    if not torch.equal(batch['semantic_len'], counts(batch['duration'], model.config.semantic_hz)):
        raise ValueError('Oracle duration and B unit length differ')
    targets = model.semantic_planner.codebook.encode((batch['semantic'] - model.semantic_mean) / model.semantic_std)
    return targets, mask, contexts, style


@torch.inference_mode()
def evaluate_hint_logits(planner, targets, mask, contexts, style, ratio, seed):
    hidden = nested_hidden_mask(mask, ratio, seed)
    # Zero hidden IDs explicitly; fully hidden inputs contain no B units. / 숨긴 ID를 지워 전체 마스킹 시 B 단위를 없앱니다.
    initial_ids = targets.masked_fill(hidden | ~mask, 0)
    conditions = {}
    for name, memory in contexts.items():
        logits = planner.logits(initial_ids, hidden, mask, style=style, **memory)
        conditions[name] = score_hidden(logits, targets, hidden, mask)
    # Keep mask fraction/time identical while corrupting visible hint order. / 마스크 비율·시간은 고정하고 보이는 힌트 순서만 바꿉니다.
    permutation_seed = seed + 100003
    shuffled_ids = permute_visible_hints(initial_ids, hidden, mask, permutation_seed)
    logits = planner.logits(shuffled_ids, hidden, mask, style=style, **contexts['correct_a'])
    shuffled_score = score_hidden(logits, targets, hidden, mask)
    visible = mask & ~hidden
    visible_frames = int(visible.sum())
    changed = int(((initial_ids != shuffled_ids) & visible).sum())
    if not visible_frames and shuffled_score != conditions['correct_a']:
        raise AssertionError('Fully hidden hint permutation changed the diagnostic')
    return {**mask_metadata(hidden, mask, ratio, seed), 'conditions': conditions,
        'visible_hint_control': {'correct_a_held_fixed': True, 'permutation_seed': permutation_seed,
            'visible_frames': visible_frames, 'changed_visible_frames': changed,
            'changed_visible_fraction': changed / visible_frames if visible_frames else 0.,
            'same_mask_and_fraction': True, 'visible_unit_histogram_preserved': True,
            'conditions': {'correct_visible_b': conditions['correct_a'], 'permuted_visible_b': shuffled_score}}}


def permute_visible_hints(initial_ids, hidden, mask, seed):
    if initial_ids.shape != hidden.shape or hidden.shape != mask.shape or hidden.dtype != torch.bool or mask.dtype != torch.bool:
        raise ValueError('Visible-hint permutation shapes differ')
    if (hidden & ~mask).any() or bool(initial_ids[hidden | ~mask].any()):
        raise ValueError('Hidden and padded IDs must be zero before visible-hint permutation')
    result = initial_ids.clone()
    # Only visible IDs enter the permutation; hidden B targets are unavailable. / 순열에는 보이는 ID만 쓰며 숨긴 B 정답은 쓰지 않습니다.
    for index in range(len(mask)):
        positions = (mask[index] & ~hidden[index]).nonzero(as_tuple=True)[0]
        order = torch.randperm(len(positions), generator=torch.Generator(device='cpu').manual_seed(seed + index)).to(mask.device)
        result[index, positions] = initial_ids[index, positions[order]]
    return result


def summarize_unit_examples(examples):
    result = {}
    for ratio in RATIOS:
        trials = [trial for row in examples for trial in row['trials'] if trial['requested_hidden_ratio'] == ratio]
        conditions = {}
        for condition in ('correct_a', 'shuffled_a'):
            regions = {}
            for region in REGIONS:
                entries = [trial['conditions'][condition][region] for trial in trials]
                frames = sum(row['hidden_frames'] for row in entries)
                ce = sum(row['ce_sum'] for row in entries)
                correct = sum(row['correct_units'] for row in entries)
                regions[region] = {'hidden_frames': frames, 'hidden_only_ce': ce / frames if frames else None,
                    'hidden_only_argmax_accuracy': correct / frames if frames else None,
                    'nonempty_measurements': sum(row['hidden_frames'] > 0 for row in entries)}
            conditions[condition] = regions
        result[str(ratio)] = {'measurements': len(trials), 'conditions': conditions,
            'shuffled_ce_increase_all_hidden': conditions['shuffled_a']['all_hidden']['hidden_only_ce'] -
                                             conditions['correct_a']['all_hidden']['hidden_only_ce']}
    return result


def summarize_visible_hints(examples):
    result = {}
    for ratio in RATIOS:
        trials = [trial['visible_hint_control'] for row in examples for trial in row['trials']
                  if trial['requested_hidden_ratio'] == ratio]
        conditions = {}
        for name in ('correct_visible_b', 'permuted_visible_b'):
            regions = {}
            for region in REGIONS:
                entries = [trial['conditions'][name][region] for trial in trials]
                frames = sum(row['hidden_frames'] for row in entries)
                regions[region] = {'hidden_frames': frames,
                    'hidden_only_ce': sum(row['ce_sum'] for row in entries) / frames if frames else None,
                    'hidden_only_argmax_accuracy': sum(row['correct_units'] for row in entries) / frames if frames else None}
            conditions[name] = regions
        visible = sum(row['visible_frames'] for row in trials)
        changed = sum(row['changed_visible_frames'] for row in trials)
        result[str(ratio)] = {'measurements': len(trials), 'visible_frames': visible,
            'changed_visible_frames': changed, 'changed_visible_fraction': changed / visible if visible else 0.,
            'conditions': conditions,
            'permuted_visible_ce_increase_all_hidden': conditions['permuted_visible_b']['all_hidden']['hidden_only_ce'] -
                                                     conditions['correct_visible_b']['all_hidden']['hidden_only_ce'],
            'correct_visible_accuracy_advantage': conditions['correct_visible_b']['all_hidden']['hidden_only_argmax_accuracy'] -
                                                 conditions['permuted_visible_b']['all_hidden']['hidden_only_argmax_accuracy']}
    return result


def chosen_indices(data, selection, manifest_hash, split='val'):
    if selection.get('manifest_sha256') != manifest_hash:
        raise ValueError('Selection and manifest hashes differ')
    wanted = selection[split]
    mapping = {row['path']: index for index, row in enumerate(data.records)}
    if not wanted or len(wanted) != len(set(wanted)) or any(path not in mapping for path in wanted):
        raise ValueError('Selected paths are empty, missing or duplicated')
    return [mapping[path] for path in wanted]


def unit_report(model, data, chosen, output, metadata, device, count=None):
    records = [data.records[index] for index in chosen]
    donors = next_different_conversation(records)
    examples = []
    evaluated = chosen[:count] if count is not None else chosen
    for position, index in enumerate(evaluated):
        batch = move_batch(collate_quality([data[index]]), device)
        donor = move_batch(collate_quality([data[chosen[donors[position]]]]), device)
        targets, mask, contexts, style = prepare_pair(model, batch, donor)
        row = records[position]
        trials = [evaluate_hint_logits(model.semantic_planner, targets, mask, contexts, style, ratio, seed)
                  for ratio in RATIOS for seed in MASK_SEEDS]
        examples.append({'path': row['path'], 'conversation_id': row['conversation_id'],
                         'shuffled_path': records[donors[position]]['path'],
                         'frames': int(mask.sum()), 'trials': trials})
        if len(examples) % 8 == 0:
            print(json.dumps({'units_completed': len(examples), 'total': len(evaluated)}), flush=True)
    report = {**metadata, 'measurement': 'First-pass logits on hidden positions; no iterative sampling.',
              'count': len(examples), 'mask_seeds': MASK_SEEDS, 'hidden_ratios': RATIOS,
              'summary': summarize_unit_examples(examples), 'visible_hint_summary': summarize_visible_hints(examples),
              'visible_hint_permutation': 'Within-example visible IDs only; deterministic order permutation, same positions and histogram.',
              'examples': examples}
    atomic_json(output / 'units_report.json', report)
    print(json.dumps({'unit_summary': report['summary']}), flush=True)


@torch.inference_mode()
def audio_report(model, data, chosen, output, metadata, device):
    from model.full_speech.codec import FrozenEncodec
    from scripts.diagnose_quality import recognize, save_wave
    from scripts.planner_sampling import sample_units

    audio_output = output / 'audio'
    audio_output.mkdir(parents=True, exist_ok=True)
    codec = FrozenEncodec().to(device).eval()
    report = {**metadata, 'asr_model': 'openai/whisper-base.en', 'mask_seed': 42,
              'acoustic_seed': 42, 'refinement_steps': 8, 'sampler_mode': 'greedy', 'examples': []}
    for position, index in enumerate(chosen):
        row, sample = data.records[index], data[index]
        batch = move_batch(collate_quality([sample]), device)
        # Donor is irrelevant for audio: only the correct A condition is rendered. / 음성은 올바른 A 조건만 생성합니다.
        targets, mask, contexts, style = prepare_pair(model, batch, batch)
        memory = contexts['correct_a']
        entry = {'path': row['path'], 'conversation_id': row['conversation_id'],
                 'input_text': row['input_text'], 'reference_text': row['response_text'],
                 'oracle_duration_seconds': float(batch['duration'][0]),
                 'target_unit_ids': targets[mask].cpu().tolist(), 'paths': {}}
        entry['paths']['reference'] = save_wave(audio_output, f'{position:02d}_reference.wav', sample['waveform'].numpy())
        for ratio in (0., *RATIOS):
            hidden = nested_hidden_mask(mask, ratio, 42)
            initial_ids = targets.masked_fill(hidden | ~mask, 0)
            ids, trace = sample_units(model.semantic_planner, mask, style=style, **memory,
                steps=8, mode='greedy', seed=42, initial_ids=initial_ids, initial_hidden=hidden)
            visible = mask & ~hidden
            if not torch.equal(ids[visible], targets[visible]):
                raise AssertionError('Visible B-unit hints changed during sampling')
            generated = model.generate_batch(person_a_only(batch), codec, seed=42,
                oracle_semantic=model.semantic_planner.codebook.centers[ids], oracle_duration=batch['duration'])
            length = int(generated['audio_lengths'][0])
            name = f'hidden_{round(ratio * 100):03d}'
            entry['paths'][name] = {**save_wave(audio_output, f'{position:02d}_{name}.wav',
                generated['waveform'][0, :length].cpu().numpy()), **mask_metadata(hidden, mask, ratio, 42),
                'generated_unit_ids': ids[mask].cpu().tolist(), 'visible_hints_preserved': True,
                'sampling_trace': trace}
        report['examples'].append(entry)
        atomic_json(audio_output / 'generated_report.json', report)
        print(json.dumps({'audio_generated': position + 1, 'total': len(chosen)}), flush=True)
    del codec
    model.to('cpu')
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    report = recognize(report, audio_output, str(device))
    atomic_json(audio_output / 'report.json', report)
    print(json.dumps({'audio_summary': report['summary']}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ('checkpoint', 'manifest', 'selection', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--audio-selection', type=Path)
    parser.add_argument('--split', choices=('train', 'val'), default='val')
    parser.add_argument('--count', type=int)
    parser.add_argument('--device', default='cuda:0')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--units-only', action='store_true')
    mode.add_argument('--audio-only', action='store_true')
    args = parser.parse_args()
    if args.count is not None and args.count < 1:
        parser.error('--count must be positive')
    if not args.units_only and (args.split != 'val' or args.audio_selection is None):
        parser.error('Audio evaluation requires --split val and --audio-selection')
    torch.set_num_threads(2)
    device = torch.device(args.device)
    model, payload = UnitSpeechSystem.from_checkpoint(args.checkpoint)
    model.to(device).requires_grad_(False).eval()
    data = QualitySpeechDataset(args.manifest, model.config, args.split)
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    selection = json.loads(args.selection.read_text())
    chosen = chosen_indices(data, selection, manifest_hash, args.split)
    audio_chosen = []
    if not args.units_only:
        audio_selection = json.loads(args.audio_selection.read_text())
        audio_chosen = chosen_indices(data, audio_selection, manifest_hash)
        if len(audio_chosen) != 8 or not set(audio_chosen) <= set(chosen):
            raise ValueError('Audio selection must contain exactly eight of the selected validation examples')
    metadata = {'checkpoint': str(args.checkpoint), 'checkpoint_step': payload.get('recovery_step'),
        'manifest_sha256': manifest_hash, 'selection_sha256': hashlib.sha256(args.selection.read_bytes()).hexdigest(),
        'audio_selection_sha256': hashlib.sha256(args.audio_selection.read_bytes()).hexdigest() if args.audio_selection else None,
        'split': args.split, 'encoder_input_contract': 'person_a_only', 'test_split_read': False,
        'planner_memory_mode': getattr(model.config, 'planner_memory_mode', 'fused'),
        'length_and_acoustic_memory': 'original_fused',
        'oracle_B_duration_all_conditions': True, 'style_and_speaker_held_fixed': True,
        'prefix_definition': 'First ceil(20% * valid B unit frames); temporal proxy, not word alignment.',
        'mask_algorithm': 'Nested CPU randperm(seed) over valid positions; hidden count=ceil(ratio*length).',
        'limitations': LIMITATIONS}
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.audio_only:
        unit_report(model, data, chosen, args.output, metadata, device, args.count)
    if not args.units_only:
        audio_report(model, data, audio_chosen, args.output, metadata, device)


if __name__ == '__main__':
    main()
