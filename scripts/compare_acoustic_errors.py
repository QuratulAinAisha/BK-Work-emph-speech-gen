"""Describe matched reconstruction errors. / 동일 표본의 재구성 오류 설명."""
import argparse
import json
import re
from pathlib import Path


def edits(reference, hypothesis):
    a, b = [re.findall(r"[a-z]+(?:'[a-z]+)?", t.lower()) for t in (reference, hypothesis)]
    grid = [[(0, 0, 0)] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        grid[i][0] = (0, i, 0)
    for j in range(len(b) + 1):
        grid[0][j] = (0, 0, j)
    for i, word in enumerate(a, 1):
        for j, other in enumerate(b, 1):
            s, d, n = grid[i-1][j-1]
            diag = (s + (word != other), d, n)
            s, d, n = grid[i-1][j]
            deletion = (s, d+1, n)
            s, d, n = grid[i][j-1]
            insertion = (s, d, n+1)
            grid[i][j] = min((diag, deletion, insertion), key=sum)
    return dict(zip(('substitutions', 'deletions', 'insertions'), grid[-1][-1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    before = json.loads((args.root / 'initial/result.json').read_text())['examples']
    after = {r['path']: r for r in json.loads((args.root / 'adapted_oracle/result.json').read_text())['examples']}
    rows = []
    for row in before:
        a, b = row['paths']['oracle_units'], after[row['path']]['paths']['oracle_semantics']
        rows.append({'path': row['path'], 'reference': row['reference_text'], 'before': a['asr'], 'after': b['asr'],
            'before_errors': edits(row['reference_text'], a['asr']), 'after_errors': edits(row['reference_text'], b['asr']),
            'wer_change': b['reference_wer'] - a['reference_wer'],
            'duration_change_seconds': b['seconds'] - a['seconds'],
            'before_clip_fraction': a['raw_clip_fraction'], 'after_clip_fraction': b['raw_clip_fraction'],
            'before_token_limit': a['asr_token_limit_reached'], 'after_token_limit': b['asr_token_limit_reached']})
    result = {'count': len(rows), 'worse': sum(r['wer_change'] > 1e-9 for r in rows),
              'better': sum(r['wer_change'] < -1e-9 for r in rows),
              'unchanged': sum(abs(r['wer_change']) <= 1e-9 for r in rows),
              'max_duration_change': max(abs(r['duration_change_seconds']) for r in rows),
              'before_edit_totals': {k: sum(r['before_errors'][k] for r in rows) for k in rows[0]['before_errors']},
              'after_edit_totals': {k: sum(r['after_errors'][k] for r in rows) for k in rows[0]['after_errors']},
              'max_before_clip_fraction': max(r['before_clip_fraction'] for r in rows),
              'max_after_clip_fraction': max(r['after_clip_fraction'] for r in rows),
              'asr_limit_hits': sum(r['before_token_limit'] or r['after_token_limit'] for r in rows),
              'examples': sorted(rows, key=lambda r: -r['wer_change']),
              'limitation': 'ASR edit counts are diagnostic; they do not establish human pronunciation or perceptual audio quality. Edit tie breaks prefer substitutions.'}
    (args.root / 'paired_error_analysis.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != 'examples'}, indent=2))
    print(json.dumps(result['examples'][:3], indent=2))


if __name__ == '__main__':
    main()
