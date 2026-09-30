"""Summarize audio-selected acoustic trials. / 음성 기준으로 선택한 음향 실험 요약."""
import argparse
import json
from pathlib import Path
import numpy as np


def interval(first, second):
    # Paired bootstrap is descriptive, not a population guarantee. / 대응 부트스트랩은 기술 통계입니다.
    a = {r['path']: r['paths']['oracle_semantics']['reference_wer'] for r in first}
    b = {r['path']: r['paths']['oracle_semantics']['reference_wer'] for r in second}
    if a.keys() != b.keys():
        raise ValueError('Matched samples required')
    delta = np.array([b[k] - a[k] for k in sorted(a)])
    rng = np.random.default_rng(42)
    means = delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(1)
    return {'mean_wer_change': float(delta.mean()), 'paired_bootstrap_95_percent_interval':
            np.quantile(means, [.025, .975]).tolist(), 'count': len(delta)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    baseline = json.loads((args.root / 'confirmation_baseline/result.json').read_text())
    candidate = json.loads((args.root / 'confirmation_candidate/result.json').read_text())
    result = {'confirmation_difference': interval(baseline['examples'], candidate['examples']),
              'teacher_minus_no_teacher_by_epoch': {}}
    for epoch in range(1, 6):
        off = json.loads((args.root / f'no_teacher_epoch_{epoch}/result.json').read_text())
        on = json.loads((args.root / f'teacher_epoch_{epoch}/result.json').read_text())
        result['teacher_minus_no_teacher_by_epoch'][str(epoch)] = interval(off['examples'], on['examples'])
    (args.root / 'paired_comparisons.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
