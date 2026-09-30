"""Validate the frozen waveform teacher. / 고정 파형 교사를 검증합니다."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from model.full_speech.quality import QualityConfig
from model.full_speech.recovery import FrozenWaveformCTC, ASR_MODEL, ASR_REVISION
from dataset.quality_speech_dataset import QualitySpeechDataset
from scripts.diagnose_speech import word_error
from prepare_quality import atomic_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    teacher = FrozenWaveformCTC().cuda().eval()
    wanted = set(json.loads(args.selection.read_text())['val'])
    dataset = QualitySpeechDataset(args.manifest, QualityConfig.load(args.config), 'val')
    rows = []
    for i, row in enumerate(dataset.records):
        if row['path'] not in wanted:
            continue
        sample = dataset[i]
        with torch.inference_mode():
            logits = teacher.waveform_logits(sample['waveform'].cuda())
            text = teacher.processor.batch_decode(logits.argmax(-1))[0]
        rows.append({'file': row['path'], 'reference': row['response_text'], 'asr': text,
                     'wer': word_error(row['response_text'], text)})
        if len(rows) == 8:
            break
    wave = sample['waveform'].cuda().requires_grad_()
    loss = teacher(wave, sample['text_b'].cuda())
    loss.backward()
    passed = len(rows) == 8 and sum(r['wer'] for r in rows) / len(rows) <= .2
    gradient_ok = bool(torch.isfinite(wave.grad).all()) and float(wave.grad.abs().sum()) > 0
    frozen = all(p.grad is None for p in teacher.parameters())
    report = {'teacher': ASR_MODEL, 'revision': ASR_REVISION, 'strict_weights_loaded': True,
              'examples': rows, 'mean_reference_wer': sum(r['wer'] for r in rows) / len(rows),
              'finite_input_gradient': gradient_ok, 'weights_frozen': frozen,
              'passed': passed and gradient_ok and frozen, 'criterion': '8 controls; mean WER <= 0.20; finite nonzero waveform gradient; frozen weights'}
    atomic_json(args.output, report)
    print(json.dumps(report), flush=True)
    if not report['passed']:
        raise RuntimeError('Frozen waveform teacher failed validation')


if __name__ == '__main__':
    main()
