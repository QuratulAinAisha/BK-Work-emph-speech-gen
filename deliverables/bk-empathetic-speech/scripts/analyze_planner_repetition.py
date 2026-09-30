"""Inspect saved planner unit distributions. / 저장된 계획기 단위 분포를 검사합니다."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def sequence_distribution(sequences):
    if not sequences or any(not row for row in sequences):
        raise ValueError('Expected nonempty unit sequences')
    if any(type(unit) is not int or unit < 0 for row in sequences for unit in row):
        raise ValueError('Unit IDs must be nonnegative integers')
    counts = Counter(unit for row in sequences for unit in row)
    frames = sum(map(len, sequences))
    adjacent_pairs = sum(len(row) - 1 for row in sequences)
    repeats = [sum(a == b for a, b in zip(row, row[1:])) for row in sequences]
    fractions = [count / (len(row) - 1) if len(row) > 1 else 0.
                 for count, row in zip(repeats, sequences)]
    mode, maximum = min(counts.items(), key=lambda item: (-item[1], item[0]))
    # Count adjacent repeats inside each utterance only. / 인접 반복은 발화 내부에서만 셉니다.
    return {'count': len(sequences), 'frames': frames, 'vocabulary_distinct_count': len(counts),
            'most_common_unit_id': mode, 'most_common_pooled_fraction': maximum / frames,
            'pooled_unit_entropy_bits': -sum((count / frames) * math.log2(count / frames)
                                            for count in counts.values()),
            'adjacent_pair_count': adjacent_pairs, 'adjacent_repeat_count': sum(repeats),
            'frame_weighted_adjacent_repeat_fraction': sum(len(row) * value for row, value in
                                                           zip(sequences, fractions)) / frames,
            'example_mean_adjacent_repeat_fraction': sum(fractions) / len(fractions),
            'pooled_adjacent_repeat_fraction': sum(repeats) / adjacent_pairs if adjacent_pairs else 0.,
            'singleton_count': sum(len(row) == 1 for row in sequences)}


def analyze_controls(controls):
    examples = controls['examples']
    if not examples:
        raise ValueError('Controls contain no examples')
    names = set(examples[0]['conditions'])
    if not names or any(set(row['conditions']) != names for row in examples):
        raise ValueError('Control conditions must match across examples')
    for row in examples:
        if len(row['target_unit_ids']) != row['frames'] or any(
                len(row['conditions'][name]['generated_unit_ids']) != row['frames'] for name in names):
            raise ValueError('Saved unit lengths do not match evaluated frames')
    return {'checkpoint': controls.get('checkpoint'), 'checkpoint_step': controls.get('checkpoint_step'),
            'split': controls.get('split'), 'selection_sha256': controls.get('selection_sha256'),
            'target_length_supplied_for_unit_diagnostic': controls.get('target_length_supplied_for_unit_diagnostic'),
            'interpretation': 'Output-distribution diagnostic only; not an empathy, relevance, or audio-quality score.',
            'weighting': 'Each utterance repeat fraction is repeats/(length-1), or 0 for a singleton. '
                         'Frame-weighted uses utterance length; pooled uses the total adjacent-pair count. '
                         'Utterance boundaries never count as adjacent pairs.',
            'target': sequence_distribution([row['target_unit_ids'] for row in examples]),
            'conditions': {name: sequence_distribution([row['conditions'][name]['generated_unit_ids']
                                                        for row in examples]) for name in sorted(names)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--controls', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.controls.read_bytes()
    report = {**analyze_controls(json.loads(raw)), 'controls_sha256': hashlib.sha256(raw).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Replace the report only after the complete JSON is written. / 전체 JSON을 쓴 뒤 보고서를 교체합니다.
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    temporary.replace(args.output)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
