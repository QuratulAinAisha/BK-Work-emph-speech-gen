"""Freeze planner data and reserve new confirmation cases. / 계획기 자료와 확인 사례를 고정합니다."""

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import soundfile as sf
from model.full_speech.quality import normalize_text
from scripts.check_generalization_data import related, shingles


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2) + '\n').encode('utf-8')


class TextIndex:
    """Find lexical neighbors without scanning every text. / 모든 문장을 훑지 않고 중복을 찾습니다."""

    def __init__(self):
        self.texts = {}
        self.exact = defaultdict(set)
        self.inverted = defaultdict(set)

    def add(self, key, text):
        self.texts[key] = text
        self.exact[text].add(key)
        for token in shingles(text):
            self.inverted[token].add(key)

    def has_related(self, text):
        candidates = set(self.exact.get(text, ()))
        for token in shingles(text):
            candidates.update(self.inverted.get(token, ()))
        return any(related(text, self.texts[key]) for key in candidates)


def subset(selected, rows, note):
    result = copy.deepcopy(selected)
    result['val'] = [row['path'] for row in rows]
    result['conversations']['val'] = [row['conversation_id'] for row in rows]
    result['selection_note'] = note
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    selection_bytes = args.selection.read_bytes()
    pilot_bytes = args.pilot.read_bytes()
    manifest, selected, pilot = map(json.loads, (manifest_bytes, selection_bytes, pilot_bytes))
    manifest_hash = sha256(manifest_bytes)
    for name, value in [('selection', selected), ('pilot', pilot)]:
        if value.get('manifest_sha256') != manifest_hash:
            raise ValueError(f'{name} belongs to a different manifest')
    if len(selected['train']) != 2048 or len(selected['val']) != 128:
        raise ValueError('Expected the existing 2048 training / 128 development selection')
    if selected.get('style_id') != 0 or selected.get('speaker_id') != 1:
        raise ValueError('Expected style 0 / voice-group 1')

    by_path, conversations = {}, {}
    for row in manifest['records']:
        if row['path'] in by_path:
            raise ValueError('Duplicate manifest record path')
        by_path[row['path']] = row
        key = row['conversation_id']
        value = (row['split'], normalize_text(row['input_text']))
        if key in conversations and conversations[key] != value:
            raise ValueError('Conversation split or input text differs across variants')
        conversations[key] = value
    for name, value in [('selection', selected), ('pilot', pilot)]:
        for split in ('train', 'val'):
            rows = [by_path[path] for path in value[split]]
            ids = [row['conversation_id'] for row in rows]
            listed = value['conversations'][split]
            if len(set(ids)) != len(ids) or len(listed) != len(ids) or set(ids) != set(listed):
                raise ValueError(f'{name} {split} membership is inconsistent')
            if any(row['split'] != split for row in rows):
                raise ValueError(f'{name} crosses original splits')
            if name == 'selection' and any(row['style_id'] != 0 or row['speaker_id'] != 1 for row in rows):
                raise ValueError('Selected row has a different style or voice group')
    if set(selected['conversations']['train']) & set(selected['conversations']['val']):
        raise ValueError('Training/development conversation overlap')

    # Preserve the exact source bytes for codebook provenance. / 코드북 이력을 위해 원본 바이트를 보존합니다.
    dev_rows = [by_path[path] for path in selected['val']]
    dev_rows.sort(key=lambda row: sha256(('planner-development42:' + row['conversation_id']).encode()))
    audio = subset(selected, dev_rows[:8],
                   'Eight predeclared development audio cases from the fixed 128 development conversations; not a final test.')
    training_index = TextIndex()
    for key, (split, text) in conversations.items():
        if split == 'train':
            training_index.add(key, text)
    excluded_ids = set(selected['conversations']['val']) | set(pilot['conversations']['val'])
    candidates = [row for row in manifest['records']
                  if row['split'] == 'val' and row['style_id'] == 0 and row['speaker_id'] == 1]
    candidates.sort(key=lambda row: sha256(('planner-confirmation42:' + row['conversation_id']).encode()))
    accepted, accepted_index, rejected = [], TextIndex(), Counter()
    seen_candidates = set()
    for row in candidates:
        key, text = row['conversation_id'], normalize_text(row['input_text'])
        if key in seen_candidates:
            rejected['repeated_conversation'] += 1
            continue
        seen_candidates.add(key)
        if key in excluded_ids:
            rejected['previous_development_or_pilot'] += 1
            continue
        if training_index.has_related(text):
            rejected['lexically_related_to_original_training'] += 1
            continue
        if accepted_index.has_related(text):
            rejected['lexically_related_within_confirmation'] += 1
            continue
        if any(char.isdigit() for char in normalize_text(row['response_text'])):
            rejected['response_contains_digits'] += 1
            continue
        if not 3 <= sf.info(row['reference_audio']).duration <= 8:
            rejected['response_duration_outside_3_to_8_seconds'] += 1
            continue
        accepted.append(row)
        accepted_index.add(key, text)
        if len(accepted) == 32:
            break
    if len(accepted) != 32:
        raise ValueError(f'Only {len(accepted)} eligible new confirmation conversations; need 32')
    note = ('32 confirmation conversations outside the selected 128 development and previous pilot validation; '
            'no lexical overlap with any original training input or within this confirmation set. '
            'Older quality runs may have used the original validation split for aggregate validation, '
            'so this is not a pristine final test. Original test audio and metrics remain untouched.')
    confirmation = subset(selected, accepted, note)
    files = {'selection.json': selection_bytes,
             'audio_selection.json': json_bytes(audio),
             'confirmation_selection.json': json_bytes(confirmation)}
    audit = {'manifest_sha256': manifest_hash, 'source_selection_sha256': sha256(selection_bytes),
             'pilot_selection_sha256': sha256(pilot_bytes),
             'artifact_sha256': {name: sha256(data) for name, data in files.items()},
             'original_conversations': len(conversations), 'original_training_conversations': len(training_index.texts),
             'training_conversations': 2048, 'development_conversations': 128,
             'development_audio_conversations': 8, 'confirmation_conversations': 32,
             'confirmation_overlap_selected_development': 0, 'confirmation_overlap_previous_pilot': 0,
             'confirmation_lexical_overlap_original_training': 0, 'confirmation_internal_lexical_duplicates': 0,
             'candidate_pool': len(candidates), 'candidates_examined_until_32_selected': len(seen_candidates),
             'rejections_before_selection_completed': dict(sorted(rejected.items())),
             'lexical_rule': 'Exact normalized text, or word-bigram Jaccard >=0.8 with >=6 words and length ratio >=0.8',
             'selection_note': note,
             'limitation': 'Lexical checks do not identify all semantic paraphrases; voice-group identity is undocumented.'}
    files['data_audit.json'] = json_bytes(audit)
    # Check every existing artifact before writing anything. / 쓰기 전에 기존 파일을 모두 확인합니다.
    for name, data in files.items():
        target = args.output / name
        if target.exists() and target.read_bytes() != data:
            raise ValueError(f'Refusing to overwrite different experiment artifact: {target}')
    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        target = args.output / name
        if not target.exists():
            temporary = target.with_suffix('.tmp')
            temporary.write_bytes(data)
            temporary.replace(target)
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
