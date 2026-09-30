"""Audit conversation splits and select broader data. / 대화 분할을 검사하고 확장 자료를 선택합니다."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import soundfile as sf
from model.full_speech.quality import normalize_text
from prepare_quality import atomic_json


def shingles(text):
    words = text.split()
    return set(zip(words, words[1:]))


def related(first, second):
    if first == second:
        return True
    a, b = first.split(), second.split()
    if min(len(a), len(b)) < 6 or min(len(a), len(b)) / max(len(a), len(b)) < .8:
        return False
    x, y = shingles(first), shingles(second)
    return len(x & y) / max(1, len(x | y)) >= .8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if (args.output / 'selection.json').exists():
        raise ValueError('Selection already exists; inspect instead of overwriting')
    manifest = json.loads(args.manifest.read_text())
    pilot = json.loads(args.pilot.read_text())
    conversations = {}
    for row in manifest['records']:
        key = row['conversation_id']
        value = (row['split'], normalize_text(row['input_text']))
        if key in conversations and conversations[key] != value:
            raise ValueError('Conversation split or input text changed across variants')
        conversations[key] = value
    # Detect lexical duplicates without fitting on held-out audio. / 미사용 음성으로 학습하지 않고 문장 중복을 검사합니다.
    exact, inverted, near = defaultdict(list), defaultdict(set), []
    for key, (split, text) in conversations.items():
        candidates = set()
        for token in shingles(text):
            candidates.update(inverted[token])
        candidates.update(exact[text])
        for other in candidates:
            if conversations[other][0] != split and related(text, conversations[other][1]):
                near.append((key, other))
        exact[text].append(key)
        for token in shingles(text):
            inverted[token].add(key)
    train_excluded = {key for pair in near for key in pair if conversations[key][0] == 'train'}
    # Existing checkpoints have seen the original training split. / 기존 체크포인트의 학습 이력도 고려합니다.
    val_excluded = {key for pair in near if any(conversations[k][0] == 'train' for k in pair)
                    for key in pair if conversations[key][0] == 'val'}
    old_val = set(pilot['conversations']['val'])
    selected = {'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                'style_id': 0, 'speaker_id': 1, 'train': [], 'val': [], 'conversations': {},
                'selection_note': '2048 distinct train / 128 val; style0 voice-group1; 3–8 sec; lexical duplicate exclusions; original splits preserved.'}
    duration_rejected, duplicate_rejected = defaultdict(int), defaultdict(int)
    for split, count in [('val', 128), ('train', 2048)]:
        rows = [r for r in manifest['records'] if r['split'] == split and r['style_id'] == 0 and r['speaker_id'] == 1]
        rows.sort(key=lambda r: hashlib.sha256(('generalization42:' + r['conversation_id']).encode()).hexdigest())
        seen, texts = [], []
        for row in rows:
            key, text = row['conversation_id'], normalize_text(row['input_text'])
            if key in seen or (split == 'train' and key in train_excluded) or (split == 'val' and key in old_val | val_excluded):
                duplicate_rejected[split] += 1
                continue
            if any(related(text, other) for other in texts):
                duplicate_rejected[split] += 1
                continue
            if any(c.isdigit() for c in normalize_text(row['response_text'])) or not 3 <= sf.info(row['reference_audio']).duration <= 8:
                duration_rejected[split] += 1
                continue
            selected[split].append(row['path']); seen.append(key); texts.append(text)
            if len(seen) == count:
                break
        if len(seen) != count:
            raise ValueError(f'Only {len(seen)} eligible {split} conversations; need {count}')
        selected['conversations'][split] = seen
    if set(selected['conversations']['train']) & set(selected['conversations']['val']):
        raise ValueError('Selected conversation leakage')
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / 'selection.json', selected)
    audit = {'manifest_sha256': selected['manifest_sha256'], 'original_conversations': len(conversations),
             'original_cross_split_conversation_ids': 0, 'cross_split_lexical_pairs': len(near),
             'excluded_training_conversations': len(train_excluded), 'selected_train': 2048, 'selected_val': 128,
             'validation_candidates_excluded_for_original_training_overlap': len(val_excluded),
             'selected_validation_lexical_overlap_original_training': 0,
             'validation_overlap_previous_pilot': 0,
             'selected_training_overlap_previous_pilot': len(set(selected['conversations']['train']) & set(pilot['conversations']['train'])),
             'lexical_rule': 'Exact normalized text, or word-bigram Jaccard >=0.8 with >=6 words and length ratio >=0.8',
             'limitation': 'This detects lexical near duplicates, not all semantic paraphrases. Test metadata used only for leakage exclusion; test audio and metrics untouched.',
             'duration_or_transcript_rejections': dict(duration_rejected), 'duplicate_rejections': dict(duplicate_rejected),
             'cross_split_pair_examples': near[:20]}
    atomic_json(args.output / 'data_audit.json', audit)
    # Predeclare 32 unseen acoustic controls across four GPUs. / 4개 GPU에서 검사할 새 음성 대조군 32개를 미리 정합니다.
    for rank in range(4):
        atomic_json(args.output / f'controls_rank{rank}.json', {**selected, 'val': selected['val'][rank:32:4]})
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
