"""Preserve acoustics after a passed unit control. / 단위 대조군 통과 후 음향망을 보존합니다."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from model.full_speech.quality import QualityConfig
from model.full_speech.units import initialize_unit_system
from scripts.run_recovery import acoustic_gate
from train_full import atomic_save
from prepare_quality import atomic_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--codebook', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if (args.output / 'best.pt').exists():
        p.error('Refusing to overwrite an existing acoustic interface')
    report = json.loads(args.report.read_text())
    digest = hashlib.sha256(args.codebook.read_bytes()).hexdigest()
    if report.get('unit_codebook_sha256') != digest or Path(report['checkpoint']).resolve() != args.checkpoint.resolve():
        raise ValueError('Oracle report does not identify this codebook and acoustic checkpoint')
    control = copy.deepcopy(report)
    for row in control['examples']:
        row['paths']['oracle_semantics'] = row['paths']['oracle_units']
    gate = acoustic_gate(control)
    if not gate['passed']:
        raise ValueError('Unit reconstruction gate did not pass')
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    torch.manual_seed(42)
    model = initialize_unit_system(QualityConfig(**payload['config']), payload, args.codebook)
    state = model.state_dict()
    if not all(torch.equal(value, state[name]) for name, value in payload['state_dict'].items()
               if not name.startswith('semantic_planner.')):
        raise RuntimeError('Acoustic or conditioning weights changed unexpectedly')
    result = model.checkpoint(0, recovery_recipe={'codebook_sha256': digest}, initialized_from=str(args.checkpoint))
    result.update(epoch=-1, recovery_step=0, stage='frozen_acoustic_unit_interface')
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_save(result, args.output / 'best.pt')
    atomic_json(args.output / 'initial_unit_gate.json', gate)
    atomic_json(args.output / 'complete.json', {'steps': 0, 'stage': 'frozen_acoustic_unit_interface',
        'quality_passed': True, 'reason': 'Passed oracle units with unchanged acoustic and conditioning weights'})


if __name__ == '__main__':
    main()
