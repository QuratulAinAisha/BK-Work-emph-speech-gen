"""Select a small unchanged-split pilot. / 기존 분할을 유지한 소규모 실험을 선택합니다."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import soundfile as sf

from model.full_speech.quality import normalize_text
from prepare_quality import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-count', type=int, default=64)
    parser.add_argument('--val-count', type=int, default=24)
    parser.add_argument('--style', type=int, default=0)
    parser.add_argument('--speaker', type=int, default=1)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    result = {'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              'style_id': args.style, 'speaker_id': args.speaker, 'train': [], 'val': [],
              'selection_note': 'Deterministic, one style/voice group, 3–8 seconds, no ASR-unsupported digits.',
              'conversations': {}}
    for split, count in [('train', args.train_count), ('val', args.val_count)]:
        rows = [r for r in manifest['records'] if r['split'] == split and
                r['style_id'] == args.style and r['speaker_id'] == args.speaker]
        rows.sort(key=lambda r: hashlib.sha256(('42:' + r['conversation_id']).encode()).hexdigest())
        conversations = set()
        for row in rows:
            if row['conversation_id'] in conversations or any(c.isdigit() for c in normalize_text(row['response_text'])):
                continue
            info = sf.info(row['reference_audio'])
            if not 3 <= info.duration <= 8:
                continue
            result[split].append(row['path'])
            conversations.add(row['conversation_id'])
            if len(result[split]) == count:
                break
        if len(result[split]) != count:
            raise ValueError(f'Insufficient {split} records')
        result['conversations'][split] = sorted(conversations)
    if set(result['conversations']['train']) & set(result['conversations']['val']):
        raise ValueError('Conversation leakage')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and json.loads(args.output.read_text()) != result:
        raise ValueError('Refusing to change an existing pilot selection')
    atomic_json(args.output, result)
    print(json.dumps({k: len(result[k]) for k in ('train', 'val')}), flush=True)


if __name__ == '__main__':
    main()
