"""Bounded sampler retest after planner repair. / 계획기 개선 후 샘플러를 제한적으로 재검사합니다."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from dataset.quality_speech_dataset import QualitySpeechDataset, collate_quality, person_a_only
from model.full_speech.codec import FrozenEncodec
from model.full_speech.tensor_ops import counts, mask_from_lengths
from model.full_speech.units import UnitSpeechSystem
from prepare_quality import atomic_json
from scripts.analyze_planner_repetition import sequence_distribution
from scripts.diagnose_quality import recognize, save_wave
from scripts.evaluate_planner_controls import next_different_conversation, planner_inputs
from scripts.planner_sampling import sample_units
from train_full import move_batch


VARIANTS = {
    'greedy_8': {'mode': 'greedy', 'steps': 8},
    'greedy_1': {'mode': 'greedy', 'steps': 1},
    'categorical_8_seed42': {'mode': 'categorical', 'steps': 8, 'seed': 42, 'temperature': .8, 'top_k': 20},
    'categorical_8_seed43': {'mode': 'categorical', 'steps': 8, 'seed': 43, 'temperature': .8, 'top_k': 20},
    'random_remask_8_seed42': {'mode': 'random_remask', 'steps': 8, 'seed': 42},
    'random_remask_8_seed43': {'mode': 'random_remask', 'steps': 8, 'seed': 43},
}
LIMITATIONS = [
    'Unit scoring supplies correct B length but no B-unit hints. It is separate from normal A-only audio generation.',
    'Normal audio uses only A features, requested style/voice, and the same A-predicted duration across samplers.',
    'Two sampler seeds are repeated measurements of the same conversations, not independent examples.',
    'Different units or more diverse units do not establish understandable, relevant or empathetic replies.',
    'Shuffled A measures input sensitivity, not correct semantic understanding; original style/voice remain fixed.',
    'ASR can hallucinate, and reference WER against a single B reply does not measure free-reply appropriateness.',
    'Token-limit flags describe the ASR decoder, not verified spoken repetition; uncapped ASR can still be wrong.',
    'Random remasking uses argmax proposals; only the revisited positions are sampled uniformly, without confidence ranking.',
    'Random remasking keeps at least one mutable position until the final pass; existing samplers may finish early on tiny sequences.',
    'Greedy one-pass keeps the fully hidden input used in fully-masked conditional training. Iterative rounds reveal predicted units and change the mask fraction.',
    'The one-pass control uses one forward, while the other arms use up to eight; this is a training-input-match diagnostic, not an equal-compute comparison.',
    'Greedy parity checks use additional forwards solely for verification; each evaluated sampler uses at most eight forwards.',
]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def selected_indices(records, selection, manifest_hash):
    if selection.get('manifest_sha256') != manifest_hash:
        raise ValueError('Selection and manifest hashes differ')
    mapping = {row['path']: index for index, row in enumerate(records)}
    paths = selection['val']
    if not paths or len(paths) != len(set(paths)) or any(path not in mapping for path in paths):
        raise ValueError('Validation selection is empty, duplicated or missing')
    chosen = [mapping[path] for path in paths]
    if len({records[index]['conversation_id'] for index in chosen}) != len(chosen):
        raise ValueError('Repeated conversations in validation selection')
    return chosen


@torch.inference_mode()
def evaluate_unit_pair(model, batch, donor_batch, verify_parity=False):
    if model.training or len(batch['style_id']) != 1 or len(donor_batch['style_id']) != 1:
        raise ValueError('Use eval mode and one example per batch')
    if int(batch['conversation_key'][0]) == int(donor_batch['conversation_key'][0]):
        raise ValueError('Shuffled A requires another conversation')
    inputs, swapped = person_a_only(batch), person_a_only(donor_batch)
    swapped.update(style_id=inputs['style_id'], speaker_id=inputs['speaker_id'])
    encoded = {'correct_a': model.encode_batch(inputs), 'shuffled_a': model.encode_batch(swapped)}
    style, _ = model.embeddings(inputs['style_id'], inputs['speaker_id'], 1)
    if not torch.equal(batch['semantic_len'], counts(batch['duration'], model.config.semantic_hz)):
        raise ValueError('Reference duration and semantic length differ')
    mask = mask_from_lengths(batch['semantic_len'])
    targets = model.semantic_planner.codebook.encode((batch['semantic'] - model.semantic_mean) / model.semantic_std)
    result = {'frames': int(mask.sum()), 'target_unit_ids': targets[mask].cpu().tolist(),
              'greedy_production_parity_checked': verify_parity, 'conditions': {}}
    saved_ids = {}
    for condition, enc in encoded.items():
        memory = planner_inputs(model, enc)
        for name, settings in VARIANTS.items():
            ids, trace = sample_units(model.semantic_planner, mask, style=style, **memory, **settings)
            if verify_parity and name in ('greedy_8', 'greedy_1'):
                expected = model.semantic_planner.sample(mask, style=style, steps=settings['steps'], **memory)
                actual = model.semantic_planner.codebook.centers[ids].masked_fill(~mask[..., None], 0)
                if not torch.equal(actual, expected):
                    raise AssertionError('Greedy units differ from production sampling')
            key = condition + '/' + name
            saved_ids[key] = ids[mask]
            result['conditions'][key] = {'generated_unit_ids': ids[mask].cpu().tolist(),
                'correct_units': int((ids[mask] == targets[mask]).sum()),
                'unit_accuracy': float((ids[mask] == targets[mask]).float().mean()),
                'planner_memory_frames': int(memory['context_mask'].sum()), 'trace': trace,
                'planner_forward_count': len(trace)}
    for key, ids in saved_ids.items():
        condition, name = key.split('/')
        result['conditions'][key].update(
            changed_units_vs_greedy=int((ids != saved_ids[condition + '/greedy_8']).sum()),
            changed_units_vs_onepass=int((ids != saved_ids[condition + '/greedy_1']).sum()),
            changed_units_vs_correct_a=int((ids != saved_ids['correct_a/' + name]).sum()))
    return result


def summarize_units(examples):
    frames = sum(row['frames'] for row in examples)
    summary = {}
    for key in examples[0]['conditions']:
        values = [row['conditions'][key] for row in examples]
        summary[key] = {
            'frame_unit_accuracy': sum(row['correct_units'] for row in values) / frames,
            'mean_example_unit_accuracy': sum(row['unit_accuracy'] for row in values) / len(values),
            'changed_unit_fraction_vs_greedy': sum(row['changed_units_vs_greedy'] for row in values) / frames,
            'changed_unit_fraction_vs_onepass': sum(row['changed_units_vs_onepass'] for row in values) / frames,
            'changed_unit_fraction_vs_correct_a': sum(row['changed_units_vs_correct_a'] for row in values) / frames,
            'planner_forwards': sum(row['planner_forward_count'] for row in values),
            **sequence_distribution([row['generated_unit_ids'] for row in values]),
        }
    return {'count': len(examples), 'frames': frames, 'summary': summary,
            'target_distribution': sequence_distribution([row['target_unit_ids'] for row in examples])}


@torch.inference_mode()
def generate_audio_variants(model, batch, codec):
    if model.training or len(batch['style_id']) != 1:
        raise ValueError('Use eval mode and one example per batch')
    inputs = person_a_only(batch)
    encoded = model.encode_batch(inputs)
    style, _ = model.embeddings(inputs['style_id'], inputs['speaker_id'], 1)
    duration, _ = model.length_predictor(encoded['context'], encoded['affect'], encoded['context_mask'], style)
    if not torch.isfinite(duration).all() or (duration <= 0).any():
        raise ValueError('Invalid A-predicted duration')
    mask = mask_from_lengths(counts(duration, model.config.semantic_hz))
    result = {'predicted_duration_seconds': float(duration.item()), 'semantic_frames': int(mask.sum()),
              'greedy_production_parity_checked': True, 'variants': {}}
    for name, settings in VARIANTS.items():
        ids, trace = sample_units(model.semantic_planner, mask, style=style, **planner_inputs(model, encoded), **settings)
        # Supplied semantics and duration are predictions from A, not B labels. / 제공 의미와 길이는 B 정답이 아닌 A 예측입니다.
        generated = model.generate_batch(inputs, codec, seed=42,
            oracle_semantic=model.semantic_planner.codebook.centers[ids], oracle_duration=duration)
        if not torch.equal(generated['duration'], duration):
            raise AssertionError('Audio sampler changed the locked duration')
        if name in ('greedy_8', 'greedy_1'):
            # Restore config even if parity fails. / 동일성 검사 실패 시에도 설정을 복원합니다.
            with patch.object(model.config, 'semantic_steps', settings['steps']):
                normal = model.generate_batch(inputs, codec, seed=42)
            for key in ('semantic', 'waveform', 'audio_lengths', 'duration'):
                if not torch.equal(normal[key], generated[key]):
                    raise AssertionError('Greedy audio differs from production: ' + key)
        length = int(generated['audio_lengths'][0])
        result['variants'][name] = {'waveform': generated['waveform'][0, :length].cpu().numpy(),
                                   'generated_unit_ids': ids[mask].cpu().tolist(), 'sampling_trace': trace,
                                   'planner_forward_count': len(trace)}
    return result


def summarize_asr(report):
    """Keep common uncapped cases matched across all samplers. / 모든 샘플러의 공통 비제한 사례를 맞춥니다."""
    names = list(VARIANTS)
    shared = [row for row in report['examples'] if all(not row['paths'][name]['asr_token_limit_reached'] for name in names)]
    report['shared_uncapped_asr'] = {'count': len(shared), 'conversation_ids': [row['conversation_id'] for row in shared],
        'mean_reference_wer': {name: float(np.mean([row['paths'][name]['reference_wer'] for row in shared]))
                               if shared else None for name in names}}
    for name in report['examples'][0]['paths']:
        paths = [row['paths'][name] for row in report['examples']]
        uncapped = [entry for entry in paths if not entry['asr_token_limit_reached']]
        report['summary'][name].update(
            asr_token_limit_cases=sum(entry['asr_token_limit_reached'] for entry in paths),
            uncapped_count=len(uncapped),
            uncapped_mean_reference_wer=float(np.mean([entry['reference_wer'] for entry in uncapped])) if uncapped else None,
            raw_clipping_cases=sum(entry['raw_clip_fraction'] > 0 for entry in paths))
    return report


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    for name in ('checkpoint', 'manifest', 'selection', 'audio-selection', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--count', type=int, default=32)
    parser.add_argument('--device', default='cuda:0')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--units-only', action='store_true')
    mode.add_argument('--audio-only', action='store_true')
    args = parser.parse_args()
    if args.count < 1:
        parser.error('--count must be positive')
    torch.set_num_threads(2)
    manifest_hash = file_hash(args.manifest)
    model, payload = UnitSpeechSystem.from_checkpoint(args.checkpoint)
    model.to(args.device).eval().requires_grad_(False)
    data = QualitySpeechDataset(args.manifest, model.config, 'val')
    chosen = selected_indices(data.records, json.loads(args.selection.read_text()), manifest_hash)
    audio_chosen = selected_indices(data.records, json.loads(args.audio_selection.read_text()), manifest_hash)
    if args.count > len(chosen):
        raise ValueError('--count exceeds selected development examples')
    if len(audio_chosen) != 8 or not set(audio_chosen) <= set(chosen):
        raise ValueError('Audio selection must be exactly eight of the selected development examples')
    root = Path(__file__).resolve().parents[1]
    identity = {
        'version': 2, 'checkpoint': str(args.checkpoint), 'checkpoint_sha256': file_hash(args.checkpoint),
        'checkpoint_step': payload.get('recovery_step'), 'manifest_sha256': manifest_hash,
        'selection_sha256': file_hash(args.selection), 'audio_selection_sha256': file_hash(args.audio_selection),
        'unit_count': args.count, 'audio_count': len(audio_chosen), 'split': 'val',
        'unit_paths': [data.records[index]['path'] for index in chosen[:args.count]],
        'audio_paths': [data.records[index]['path'] for index in audio_chosen],
        'encoder_input_contract': 'person_a_only', 'test_examples_opened': 0,
        'variants': VARIANTS, 'shared_acoustic_seed': 42,
        'planner_memory_mode': getattr(model.config, 'planner_memory_mode', 'fused'),
        'length_and_acoustic_memory': 'original_fused', 'checkpoint_semantic_steps': model.config.semantic_steps,
        'diagnostic_semantic_steps_by_variant': {name: settings['steps'] for name, settings in VARIANTS.items()},
        'greedy_parity_variants': ['greedy_8', 'greedy_1'], 'codec_steps': model.config.codec_steps,
        'shuffle_policy': 'Next distinct conversation in full selection order before count truncation; cyclic wraparound.',
        'source_sha256': {name: file_hash(root / name) for name in (
            'scripts/check_planner_repair_sampling.py', 'scripts/planner_sampling.py',
            'scripts/diagnose_quality.py', 'model/full_speech/units.py', 'model/full_speech/quality.py')},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    old = args.output / 'identity.json'
    if old.exists():
        if json.loads(old.read_text()) != identity:
            raise ValueError('Diagnostic recipe changed; use a new output directory')
    elif any(args.output.iterdir()):
        raise ValueError('Existing output lacks provenance; use a new output directory')
    atomic_json(old, identity)

    def load(index):
        return move_batch(collate_quality([data[index]]), torch.device(args.device))

    units_path = args.output / 'units_report.json'
    if not args.audio_only:
        if units_path.exists():
            if json.loads(units_path.read_text())['identity'] != identity:
                raise ValueError('Saved unit report identity differs')
        else:
            donors = next_different_conversation([data.records[index] for index in chosen])
            examples = []
            for position, index in enumerate(chosen[:args.count]):
                row, donor_row = data.records[index], data.records[chosen[donors[position]]]
                result = evaluate_unit_pair(model, load(index), load(chosen[donors[position]]), verify_parity=position == 0)
                result.update(path=row['path'], conversation_id=row['conversation_id'],
                              shuffled_path=donor_row['path'], shuffled_conversation_id=donor_row['conversation_id'])
                examples.append(result)
                print(json.dumps({'unit_cases': position + 1, 'total': args.count}), flush=True)
            atomic_json(units_path, {'identity': identity, 'oracle_B_length': True, 'B_unit_hints_supplied': False,
                'greedy_production_parity': 'One-pass and eight-pass checked for both A conditions on first selected case.',
                'limitations': LIMITATIONS, **summarize_units(examples), 'examples': examples})

    audio_output = args.output / 'audio'
    if not args.units_only:
        audio_output.mkdir(exist_ok=True)
        generated_path = audio_output / 'generated_report.json'
        if generated_path.exists():
            report = json.loads(generated_path.read_text())
            if report['identity'] != identity or len(report['examples']) != 8:
                raise ValueError('Saved audio generation is incomplete or its identity differs')
        else:
            codec = FrozenEncodec().to(args.device).eval()
            report = {'identity': identity, 'examples': [], 'asr_model': 'openai/whisper-base.en',
                      'normal_variants_use_only_a': True, 'oracle_B_length': False,
                      'limitations': LIMITATIONS}
            for position, index in enumerate(audio_chosen):
                row, batch = data.records[index], load(index)
                generated = generate_audio_variants(model, batch, codec)
                entry = {key: row[key] for key in ('path', 'conversation_id', 'input_text', 'style_id', 'speaker_id')}
                entry.update(reference_text=row['response_text'],
                    predicted_duration_seconds=generated['predicted_duration_seconds'],
                    semantic_frames=generated['semantic_frames'], greedy_production_parity_checked=True, paths={})
                entry['paths']['reference'] = save_wave(audio_output, f'{position:02d}_reference.wav',
                    batch['waveform'][0, :int(batch['waveform_len'][0])].cpu().numpy())
                for name, variant in generated['variants'].items():
                    waveform = variant.pop('waveform')
                    entry['paths'][name] = {**save_wave(audio_output, f'{position:02d}_{name}.wav', waveform), **variant}
                report['examples'].append(entry)
                atomic_json(audio_output / 'generation_progress.json', report)
                print(json.dumps({'audio_cases': position + 1, 'total': len(audio_chosen)}), flush=True)
                del batch, generated
            atomic_json(generated_path, report)
            del codec
    del model
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()
    # Load ASR only after releasing the response model. / 응답 모델을 해제한 뒤 ASR을 읽습니다.
    if not args.units_only:
        report_path = audio_output / 'report.json'
        if report_path.exists():
            if json.loads(report_path.read_text())['identity'] != identity:
                raise ValueError('Saved ASR report identity differs')
        else:
            report = summarize_asr(recognize(report, audio_output, args.device))
            atomic_json(report_path, report)
    for field, path in (('checkpoint_sha256', args.checkpoint), ('manifest_sha256', args.manifest),
                        ('selection_sha256', args.selection), ('audio_selection_sha256', args.audio_selection)):
        if file_hash(path) != identity[field]:
            raise RuntimeError('Diagnostic input changed during evaluation: ' + field)
    print(json.dumps({'output': str(args.output), 'checkpoint_unchanged': True,
                      'units_complete': units_path.exists(), 'audio_complete': (audio_output / 'report.json').exists()}), flush=True)


if __name__ == '__main__':
    main()
