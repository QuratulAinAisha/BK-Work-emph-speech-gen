"""Combine predeclared reconstruction controls. / 미리 정한 복원 대조군 결과를 합칩니다."""

import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_quality import atomic_json
from scripts.run_recovery import acoustic_gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('outputs/generalization_check'))
    args = parser.parse_args()
    selection = json.loads((args.output / 'selection.json').read_text())
    rows, books, checkpoints = [], set(), set()
    for rank in range(4):
        report = json.loads((args.output / f'oracle_rank{rank}/report.json').read_text())
        expected = set(json.loads((args.output / f'controls_rank{rank}.json').read_text())['val'])
        if {r['path'] for r in report['examples']} != expected or len(report['examples']) != 8:
            raise ValueError('Incomplete predeclared acoustic controls')
        rows.extend(report['examples'])
        books.add(report['unit_codebook_sha256']); checkpoints.add(report['checkpoint'])
    if len({r['path'] for r in rows}) != 32 or len(books) != 1 or len(checkpoints) != 1:
        raise ValueError('Duplicated examples or incompatible evaluation models')
    continuous = acoustic_gate({'examples': rows})
    units = copy.deepcopy(rows)
    for row in units:
        row['paths']['oracle_semantics'] = row['paths']['oracle_units']
    quantized = acoustic_gate({'examples': units})
    summary = {name: sum(r['paths'][name]['reference_wer'] for r in rows) / len(rows)
               for name in rows[0]['paths']}
    if continuous['passed'] and quantized['passed']:
        decision = 'Acoustic/data prerequisites passed for a bounded broader planner experiment; response generalization remains untested.'
    elif continuous['passed']:
        decision = 'Current unit vocabulary/interface failed broader controls; repair that interface before planner scaling.'
    else:
        decision = 'Continuous acoustic generator failed broader controls too; improve broad acoustic reconstruction before interpreting planner-generated speech.'
    result = {'count': len(rows), 'selected_train': len(selection['train']), 'selected_val': len(selection['val']),
              'checkpoint': next(iter(checkpoints)), 'unit_codebook_sha256': next(iter(books)),
              'mean_reference_wer': summary, 'continuous_gate': continuous, 'unit_gate': quantized,
              'ready_for_broader_planner_training': continuous['passed'] and quantized['passed'],
              'decision': decision, 'examples': rows,
              'limitation': 'Development controls in one style and unverified voice group; not human quality ratings or final-test performance.'}
    atomic_json(args.output / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'examples'}), flush=True)


if __name__ == '__main__':
    main()
