"""Measure the planner's A-context dependence. / 계획기의 A 문맥 의존성을 검사합니다."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F

from dataset.quality_speech_dataset import QualitySpeechDataset, collate_quality, person_a_only
from model.full_speech.tensor_ops import mask_from_lengths
from model.full_speech.units import UnitSpeechSystem
from prepare_quality import atomic_json
from train_full import move_batch


CONDITIONS = ('correct_a', 'shuffled_a', 'zero_a')


def planner_inputs(model, encoded):
    """Use the checkpoint's intended memory, with old-model fallback. / 체크포인트의 계획기 메모리를 사용합니다."""
    selector = getattr(model, 'planner_inputs', None)
    if callable(selector):
        return selector(encoded)
    return {'context': encoded['context'], 'context_mask': encoded['context_mask'], 'affect': encoded['affect']}


def next_different_conversation(records):
    """Use global cyclic donors, including single-item evaluation batches. / 전체 순환 순서로 다른 대화를 고릅니다."""
    if len({row['conversation_id'] for row in records}) < 2:
        raise ValueError('A shuffled control needs at least two distinct conversations')
    donors = []
    for index, row in enumerate(records):
        for shift in range(1, len(records)):
            donor = (index + shift) % len(records)
            if records[donor]['conversation_id'] != row['conversation_id']:
                donors.append(donor)
                break
    return donors


@torch.inference_mode()
def evaluate_pair(model, batch, donor_batch):
    """B units/length are scoring targets, never encoder inputs. / B 단위·길이는 평가 타깃으로만 씁니다."""
    if model.training or len(batch['style_id']) != 1 or len(donor_batch['style_id']) != 1:
        raise ValueError('Control evaluation requires eval mode and one example per batch')
    inputs = person_a_only(batch)
    donor_inputs = person_a_only(donor_batch)
    # Keep requested style/identity fixed while replacing A. / A를 바꿔도 요청 스타일·정체성은 고정합니다.
    for key in ('style_id', 'speaker_id'):
        donor_inputs[key] = inputs[key]
    encoded = model.encode_batch(inputs)
    shuffled = model.encode_batch(donor_inputs)
    correct = planner_inputs(model, encoded)
    controls = {
        'correct_a': correct,
        'shuffled_a': planner_inputs(model, shuffled),
        # A valid mask avoids undefined attention on an empty sequence. / 유효 마스크로 빈 어텐션을 방지합니다.
        'zero_a': {**correct, 'context': torch.zeros_like(correct['context']), 'affect': torch.zeros_like(correct['affect'])},
    }
    # Length prediction retains its original fused inputs. / 길이 예측은 원래 융합 입력을 유지합니다.
    duration_controls = {'correct_a': encoded, 'shuffled_a': shuffled,
        'zero_a': {**encoded, 'context': torch.zeros_like(encoded['context']), 'affect': torch.zeros_like(encoded['affect'])}}
    style, _ = model.embeddings(inputs['style_id'], inputs['speaker_id'], 1)
    mask = mask_from_lengths(batch['semantic_len'])
    planner = model.semantic_planner
    normalized = (batch['semantic'] - model.semantic_mean) / model.semantic_std
    targets = planner.codebook.encode(normalized)
    empty_ids = torch.zeros_like(targets)
    output, predictions = {}, {}
    for name, memory in controls.items():
        # Every valid position is hidden: no B token is supplied. / 모든 유효 위치를 가려 B 토큰을 주지 않습니다.
        logits = planner.logits(empty_ids, mask, mask, style=style, **memory)
        ce = F.cross_entropy(logits[mask].float(), targets[mask], reduction='sum')
        generated = planner.sample(mask, style=style, steps=8, **memory)
        predicted = planner.codebook.encode(generated)[mask]
        duration_input = duration_controls[name]
        duration, _ = model.length_predictor(duration_input['context'], duration_input['affect'], duration_input['context_mask'], style)
        correct = int((predicted == targets[mask]).sum())
        frames = int(mask.sum())
        if not torch.isfinite(ce) or not torch.isfinite(duration).all():
            raise ValueError(f'Non-finite planner metric for {name}')
        predictions[name] = predicted
        output[name] = {
            'fully_masked_ce': float(ce) / frames,
            'unit_accuracy': correct / frames,
            'correct_units': correct,
            'predicted_duration_seconds': float(duration.item()),
            'duration_abs_error_seconds': abs(float(duration.item()) - float(batch['duration'].item())),
            'generated_unit_ids': predicted.cpu().tolist(),
            'planner_memory_frames': int(memory['context_mask'].sum()),
            'affect_frames': int(memory.get('affect_mask', memory['context_mask']).sum()),
        }
    for name in CONDITIONS:
        changed = int((predictions[name] != predictions['correct_a']).sum())
        output[name]['changed_units_vs_correct'] = changed
        output[name]['changed_unit_fraction_vs_correct'] = changed / frames
    return {'frames': frames, 'target_duration_seconds': float(batch['duration'].item()),
            'target_unit_ids': targets[mask].cpu().tolist(), 'conditions': output}


def summarize_examples(examples):
    if not examples:
        raise ValueError('No examples to summarize')
    frames = sum(row['frames'] for row in examples)
    summary = {}
    for name in CONDITIONS:
        values = [row['conditions'][name] for row in examples]
        summary[name] = {
            'mean_example_fully_masked_ce': sum(row['fully_masked_ce'] for row in values) / len(values),
            'frame_fully_masked_ce': sum(row['conditions'][name]['fully_masked_ce'] * row['frames']
                                        for row in examples) / frames,
            'mean_example_unit_accuracy': sum(row['unit_accuracy'] for row in values) / len(values),
            'frame_unit_accuracy': sum(row['correct_units'] for row in values) / frames,
            'mean_changed_unit_fraction_vs_correct': sum(row['changed_unit_fraction_vs_correct']
                                                        for row in values) / len(values),
            'frame_changed_unit_fraction_vs_correct': sum(row['changed_units_vs_correct'] for row in values) / frames,
            'mean_duration_abs_error_seconds': sum(row['duration_abs_error_seconds'] for row in values) / len(values),
            'mean_predicted_duration_seconds': sum(row['predicted_duration_seconds'] for row in values) / len(values),
        }
    correct = summary['correct_a']
    for name in ('shuffled_a', 'zero_a'):
        summary[name]['mean_ce_increase_vs_correct'] = (summary[name]['mean_example_fully_masked_ce']
                                                       - correct['mean_example_fully_masked_ce'])
        summary[name]['frame_accuracy_drop_vs_correct'] = (correct['frame_unit_accuracy']
                                                          - summary[name]['frame_unit_accuracy'])
    return {'count': len(examples), 'frames': frames, 'conditions': summary}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    for name in ('checkpoint', 'manifest', 'selection', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--split', default='val', choices=('train', 'val', 'test'))
    parser.add_argument('--count', type=int)
    args = parser.parse_args()
    if args.count is not None and args.count < 1:
        parser.error('--count must be positive')
    torch.set_num_threads(2)
    selection = json.loads(args.selection.read_text())
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if selection.get('manifest_sha256') != manifest_hash:
        raise ValueError('Selection and manifest hashes differ')
    model, payload = UnitSpeechSystem.from_checkpoint(args.checkpoint)
    model.to(args.device).eval()
    data = QualitySpeechDataset(args.manifest, model.config, args.split)
    wanted = selection[args.split]
    mapping = {row['path']: index for index, row in enumerate(data.records)}
    if len(wanted) != len(set(wanted)) or any(path not in mapping for path in wanted):
        raise ValueError('Selected records missing or duplicated')
    chosen = [mapping[path] for path in wanted]
    records = [data.records[index] for index in chosen]
    donors = next_different_conversation(records)
    examples = []
    count = min(len(chosen), args.count or len(chosen))
    for position in range(count):
        donor_position = donors[position]
        batch = move_batch(collate_quality([data[chosen[position]]]), torch.device(args.device))
        donor_batch = move_batch(collate_quality([data[chosen[donor_position]]]), torch.device(args.device))
        metrics = evaluate_pair(model, batch, donor_batch)
        metrics.update(path=records[position]['path'], conversation_id=records[position]['conversation_id'],
                       shuffled_path=records[donor_position]['path'],
                       shuffled_conversation_id=records[donor_position]['conversation_id'])
        examples.append(metrics)
        if len(examples) % 16 == 0:
            print(json.dumps({'evaluated': len(examples), 'total': count}), flush=True)
    report = {
        'checkpoint': str(args.checkpoint), 'checkpoint_step': payload.get('recovery_step'),
        'manifest_sha256': manifest_hash,
        'selection_sha256': hashlib.sha256(args.selection.read_bytes()).hexdigest(),
        'split': args.split, 'refinement_steps': 8, 'encoder_input_contract': 'person_a_only',
        'planner_memory_mode': getattr(model.config, 'planner_memory_mode', 'fused'),
        'length_predictor_memory': 'original_fused',
        'target_length_supplied_for_unit_diagnostic': True,
        'style_and_speaker_held_fixed': True,
        'shuffle_policy': 'Next different conversation in full selection order, with cyclic wraparound',
        'limitations': [
            'Correct B length is supplied for unit diagnostics; this is not normal A-only audio generation.',
            'B units and B duration are scoring targets only; the A encoder receives person_a_only inputs.',
            'Unit accuracy and CE against one recorded reply do not measure response appropriateness.',
            'Zero context is out of distribution; its result alone cannot establish that A is ignored.',
            'Changing units under shuffled A demonstrates sensitivity, not correct semantic understanding.',
        ],
        **summarize_examples(examples), 'examples': examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != 'examples'}), flush=True)


if __name__ == '__main__':
    main()
