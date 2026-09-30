"""Audit A audio against its stored text. / A 음성과 저장 문장을 대조합니다."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import time
import unicodedata


ASR_MODEL = 'openai/whisper-base.en'
DECODER_BUDGET = 448


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def normalize(text):
    text = unicodedata.normalize('NFKD', text.replace('’', "'")).encode('ascii', 'ignore').decode().lower()
    return ' '.join(re.sub(r"[^a-z0-9' ]", ' ', text).split())


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for index, item in enumerate(reference, 1):
        current = [index]
        for position, other in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[position] + 1,
                               previous[position - 1] + (item != other)))
        previous = current
    return previous[-1]


def text_metrics(reference, hypothesis):
    reference, hypothesis = normalize(reference), normalize(hypothesis)
    if not reference:
        raise ValueError('Stored A transcript is empty after normalization')
    words = reference.split()
    word_errors = edit_distance(words, hypothesis.split())
    char_errors = edit_distance(reference, hypothesis)
    return {'normalized_input_text': reference, 'normalized_recognized_text': hypothesis,
            'word_errors': word_errors, 'reference_words': len(words), 'wer': word_errors / len(words),
            'character_errors': char_errors, 'reference_characters': len(reference),
            'cer': char_errors / len(reference)}


def resolve_a_audio(row, source):
    # Match preparation's source-index mapping exactly. / 전처리의 원본 인덱스 매핑을 그대로 확인합니다.
    index = row.get('source_index')
    if type(index) is not int or not 0 <= index < len(source['records']):
        raise ValueError(f'Invalid source_index for {row.get("path")}')
    item = source['records'][index]
    if item.get('response_audio') != row.get('reference_audio') or not item.get('response_audio'):
        raise ValueError(f'Response audio mapping differs at source index {index}')
    for name in ('style_id', 'speaker_id', 'conversation_id', 'split'):
        if name in item and name in row and item[name] != row[name]:
            raise ValueError(f'{name} differs at source index {index}')
    response = Path(item['response_audio'])
    if len(response.parents) < 2 or not item.get('mel'):
        raise ValueError(f'Missing dataset root or mel path at source index {index}')
    root = response.parents[1]
    return root / 'generated_input_audio' / (Path(item['mel']).stem + '.wav')


def select_rows(manifest, selection, split, count):
    if split not in ('train', 'val') or count < 1:
        raise ValueError('Only positive-count train/val audits are allowed')
    selected = selection[split]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError('Selection is empty or contains duplicate paths')
    records = [row for row in manifest['records'] if row['split'] == split]
    mapping = {row['path']: row for row in records}
    if len(mapping) != len(records) or any(path not in mapping for path in selected):
        raise ValueError('Manifest paths duplicated, absent, or from a different split')
    return [mapping[path] for path in selected[:count]]


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def build_recognizer(device):
    import torch
    from transformers import (AutomaticSpeechRecognitionPipeline, StoppingCriteria,
                              StoppingCriteriaList, WhisperForConditionalGeneration, WhisperProcessor)

    class BudgetAudit(StoppingCriteria):
        max_seen = 0

        def __call__(self, input_ids, scores, **kwargs):
            self.max_seen = max(self.max_seen, input_ids.shape[-1])
            return torch.zeros(input_ids.shape[0], device=input_ids.device, dtype=torch.bool)

    class AuditedPipeline(AutomaticSpeechRecognitionPipeline):
        def _forward(self, model_inputs, **generate_kwargs):
            self.budget_audit.max_seen = 0
            stride = model_inputs.get('stride')
            result = super()._forward(model_inputs, **generate_kwargs)
            # Observe complete decoder length, including prefix tokens. / 접두 토큰을 포함한 디코더 길이를 기록합니다.
            length = self.budget_audit.max_seen
            self.chunk_audit.append({'input_stride_samples': stride,
                'decoder_tokens_with_prefix': length,
                'decoder_budget_reached': length >= DECODER_BUDGET,
                'returned_tokens': int(result['tokens'].shape[-1])})
            return result

    processor = WhisperProcessor.from_pretrained(ASR_MODEL, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(ASR_MODEL, local_files_only=True)
    model.requires_grad_(False).eval()
    if model.config.max_target_positions != DECODER_BUDGET:
        raise ValueError('Unexpected Whisper decoder capacity')
    recognizer = AuditedPipeline(model=model, tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor, device=device,
        chunk_length_s=30, stride_length_s=5, return_timestamps=True)
    recognizer.budget_audit = BudgetAudit()
    recognizer.chunk_audit = []
    # 448 new tokens plus a prefix would overflow Whisper. / 새 토큰 448개에 접두부를 더하면 한도를 넘습니다.
    kwargs = {'max_length': DECODER_BUDGET, 'max_new_tokens': None, 'do_sample': False,
              'stopping_criteria': StoppingCriteriaList([recognizer.budget_audit])}
    return recognizer, kwargs, getattr(model.config, '_commit_hash', None)


def summarize(examples):
    return {'count': len(examples), 'mean_example_wer': statistics.mean(row['wer'] for row in examples),
            'median_example_wer': statistics.median(row['wer'] for row in examples),
            'corpus_wer': sum(row['word_errors'] for row in examples) / sum(row['reference_words'] for row in examples),
            'mean_example_cer': statistics.mean(row['cer'] for row in examples),
            'corpus_cer': sum(row['character_errors'] for row in examples) / sum(row['reference_characters'] for row in examples),
            'seconds': sum(row['seconds'] for row in examples),
            'recordings_over_30_seconds': sum(row['seconds'] > 30 for row in examples),
            'recordings_at_decoder_budget': sum(row['possible_truncation'] for row in examples),
            'empty_recognitions': sum(not row['normalized_recognized_text'] for row in examples)}


def main():
    parser = argparse.ArgumentParser()
    for name in ('manifest', 'selection', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--source', type=Path, default=Path('outputs/bk_source/source.json'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--count', type=int, default=128)
    parser.add_argument('--split', choices=('train', 'val'), default='val')
    args = parser.parse_args()
    if args.count < 1:
        parser.error('--count must be positive')
    if args.output.exists():
        raise ValueError('Output exists; preserve it and choose a new output path')
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    selection = json.loads(args.selection.read_text(encoding='utf-8'))
    source = json.loads(args.source.read_text(encoding='utf-8'))
    manifest_hash = sha256(args.manifest)
    if selection.get('manifest_sha256') != manifest_hash:
        raise ValueError('Selection and manifest hashes differ')
    rows = select_rows(manifest, selection, args.split, args.count)
    paths = [resolve_a_audio(row, source) for row in rows]
    if any(not path.is_file() for path in paths):
        raise ValueError('A source audio file is absent; check dataset paths')
    # ASR is independent of our encoder and planner. / ASR은 우리 인코더·계획기와 독립적입니다.
    import numpy as np
    import soundfile as sf
    import torch
    import transformers
    from scipy.signal import resample_poly
    torch.set_num_threads(2)
    recognizer, generate_kwargs, revision = build_recognizer(args.device)
    report = {'manifest_sha256': manifest_hash, 'source_sha256': sha256(args.source),
        'selection_sha256': sha256(args.selection), 'split': args.split, 'requested_count': args.count,
        'selected_count': len(rows), 'asr_model': ASR_MODEL, 'asr_model_revision': revision,
        'transformers_version': transformers.__version__, 'frozen_asr': True,
        'chunk_length_seconds': 30, 'stride_seconds_each_side': 5,
        'chunk_assembly': 'Transformers ASR pipeline Whisper timestamp-aware overlap stitching',
        'decoder_max_length_with_prefix': DECODER_BUDGET, 'sampling': False,
        'normalization': 'NFKD ASCII, lowercase, letters/digits/apostrophe retained, collapsed whitespace; CER includes spaces',
        'limitations': [
            'ASR errors, accents, numbers, and spelling differences do not prove a wrong A/text pairing.',
            'Independent ASR checks source audio, not whether the model encoder retains its content.',
            'Timestamp-aware chunk stitching can lose or duplicate words at overlaps; all chunk metadata are retained.',
            'Decoder-budget flags identify possible truncation; absence of a flag cannot rule out early stopping or ASR omissions.',
            'The 448-token budget includes prefix tokens; max_new_tokens=448 would exceed Whisper capacity.',
            'Training/validation only; no final-test audio or labels are evaluated.'
        ], 'examples': []}
    started = time.monotonic()
    for index, (row, path) in enumerate(zip(rows, paths), 1):
        wave, rate = sf.read(path, dtype='float32')
        channels = 1 if wave.ndim == 1 else wave.shape[1]
        if wave.ndim == 2:
            wave = wave.mean(axis=1)
        if wave.ndim != 1 or not len(wave) or rate <= 0 or not np.isfinite(wave).all():
            raise ValueError(f'Invalid A waveform: {path}')
        seconds = len(wave) / rate
        divisor = math.gcd(rate, 16000)
        audio = resample_poly(wave, 16000 // divisor, rate // divisor).astype(np.float32)
        recognizer.chunk_audit = []
        with torch.inference_mode():
            recognized = recognizer({'array': audio, 'sampling_rate': 16000},
                                   generate_kwargs=generate_kwargs, batch_size=1)
        text = recognized['text'].strip()
        entry = {'path': row['path'], 'conversation_id': row['conversation_id'],
            'source_index': row['source_index'], 'a_audio': str(path), 'a_audio_sha256': sha256(path),
            'sample_rate': rate, 'source_channels': channels, 'seconds': seconds,
            'input_text': row['input_text'], 'recognized_text': text,
            'chunks': recognized.get('chunks', []), 'chunk_audit': list(recognizer.chunk_audit),
            'possible_truncation': any(chunk['decoder_budget_reached'] for chunk in recognizer.chunk_audit),
            **text_metrics(row['input_text'], text)}
        report['examples'].append(entry)
        report.update(summary=summarize(report['examples']), elapsed_seconds=time.monotonic() - started)
        atomic_json(args.output.with_name(args.output.stem + '.partial.json'), report)
        print(json.dumps({'completed': index, 'total': len(rows), 'wer': entry['wer'],
                          'seconds': seconds, 'chunks': len(recognizer.chunk_audit)}), flush=True)
    atomic_json(args.output, report)
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    main()
