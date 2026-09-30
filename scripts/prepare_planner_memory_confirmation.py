"""Reserve fresh memory-ablation confirmation cases. / 메모리 비교용 새 확인 사례를 고정합니다."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.full_speech.quality import normalize_text
from scripts.prepare_planner_experiment import TextIndex, json_bytes, subset


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate_membership(value, label, lookup, splits=('val',)):
    result = set()
    for split in splits:
        paths = value[split]
        if not paths or len(paths) != len(set(paths)) or any(path not in lookup for path in paths):
            raise ValueError(f'{label} has missing or duplicated {split} paths')
        rows = [lookup[path] for path in paths]
        ids = [row['conversation_id'] for row in rows]
        if len(ids) != len(set(ids)) or any(row['split'] != split for row in rows):
            raise ValueError(f'{label} repeats conversations or crosses split boundaries')
        if set(value['conversations'][split]) != set(ids) or len(value['conversations'][split]) != len(ids):
            raise ValueError(f'{label} conversation metadata differs from paths')
        result.update(ids)
    return result


def select_confirmation(manifest, selected, pilot, previous, duration_seconds, count=32):
    if count < 1 or not previous:
        raise ValueError('Positive count and previous confirmation selections are required')
    lookup = {row['path']: row for row in manifest['records']}
    if len(lookup) != len(manifest['records']):
        raise ValueError('Duplicate manifest path')
    conversations = {}
    for row in manifest['records']:
        value = (row['split'], normalize_text(row['input_text']))
        key = row['conversation_id']
        if key in conversations and conversations[key] != value:
            raise ValueError('Conversation text or split differs across variants')
        conversations[key] = value
    validate_membership(selected, 'development selection', lookup, ('train', 'val'))
    excluded_groups = {'development': validate_membership(selected, 'development', lookup),
                       'pilot': validate_membership(pilot, 'pilot', lookup)}
    for index, value in enumerate(previous):
        excluded_groups[f'previous_confirmation_{index}'] = validate_membership(
            value, f'previous confirmation {index}', lookup)
    excluded_ids = set().union(*excluded_groups.values())
    original_train, prior_validation = TextIndex(), TextIndex()
    for key, (split, text) in conversations.items():
        if split == 'train':
            original_train.add(key, text)
        if key in excluded_ids:
            prior_validation.add(key, text)
    candidates = [row for row in manifest['records']
                  if row['split'] == 'val' and row['style_id'] == 0 and row['speaker_id'] == 1]
    candidates.sort(key=lambda row: digest(('planner-memory-confirmation42:' + row['conversation_id']).encode()))
    accepted, accepted_index, rejected, seen = [], TextIndex(), Counter(), set()
    durations = {}
    for row in candidates:
        key, text = row['conversation_id'], normalize_text(row['input_text'])
        if key in seen:
            rejected['duplicate_conversation'] += 1
            continue
        seen.add(key)
        if key in excluded_ids:
            rejected['previous_validation_conversation'] += 1
            continue
        if original_train.has_related(text):
            rejected['lexically_related_to_original_training'] += 1
            continue
        if prior_validation.has_related(text):
            rejected['lexically_related_to_previous_validation'] += 1
            continue
        if accepted_index.has_related(text):
            rejected['lexically_related_within_confirmation'] += 1
            continue
        if any(char.isdigit() for char in normalize_text(row['response_text'])):
            rejected['response_contains_digits'] += 1
            continue
        # Read duration metadata only; never recognize or rank candidates. / 길이 메타데이터만 읽고 후보 음성을 평가하지 않습니다.
        duration = float(duration_seconds(row))
        if not 3 <= duration <= 8:
            rejected['response_duration_outside_3_to_8_seconds'] += 1
            continue
        accepted.append(row)
        accepted_index.add(key, text)
        durations[row['path']] = duration
        if len(accepted) == count:
            break
    if len(accepted) != count:
        raise ValueError(f'Only {len(accepted)} eligible fresh confirmation cases; need {count}')
    note = ('Fresh for the native50/roundtrip50/fused-continuation planner comparison: excludes all '
            'specified prior development, pilot and confirmation IDs and their lexical near duplicates. '
            'Original validation was exposed to aggregate validation in older runs; this is not a pristine final test. '
            'Original test audio/metrics remain untouched. Selection reads B duration headers only.')
    confirmation = subset(selected, accepted, note)
    accepted_ids = {row['conversation_id'] for row in accepted}
    audit = {'confirmation_conversations': count, 'style_id': 0, 'speaker_id': 1,
        'selection_order': 'SHA256(planner-memory-confirmation42:conversation_id), ascending',
        'candidate_pool_records': len(candidates), 'candidate_conversations_examined': len(seen),
        'excluded_prior_validation_ids': len(excluded_ids),
        'prior_group_counts': {name: len(ids) for name, ids in excluded_groups.items()},
        'confirmation_overlap_prior_groups': {name: len(ids & accepted_ids) for name, ids in excluded_groups.items()},
        'original_training_conversations': len(original_train.texts),
        'confirmation_lexical_overlap_original_training': 0,
        'confirmation_lexical_overlap_previous_validation': 0,
        'confirmation_internal_lexical_duplicates': 0,
        'lexical_rule': 'Exact normalized input, or word-bigram Jaccard >=0.8 with >=6 words and length ratio >=0.8',
        'response_duration_seconds': durations, 'rejections_before_selection_complete': dict(sorted(rejected.items())),
        'model_outputs_consulted': False, 'audio_samples_decoded': False,
        'test_audio_or_metrics_evaluated': False, 'selection_note': note,
        'limitations': ['Lexical exclusions cannot identify every semantic paraphrase.',
                       'The synthesized voice identity within voice-group 1 remains undocumented.',
                       'Fresh detailed confirmation does not remove earlier aggregate validation exposure.']}
    return confirmation, audit


def write_immutable(output, files):
    # Verify every conflict before creating any artifact. / 파일을 만들기 전에 모든 충돌을 확인합니다.
    for name, data in files.items():
        target = output / name
        if target.exists() and target.read_bytes() != data:
            raise ValueError(f'Refusing to replace different confirmation artifact: {target}')
    output.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        target = output / name
        if not target.exists():
            temporary = target.with_suffix('.tmp')
            temporary.write_bytes(data)
            temporary.replace(target)


def main():
    parser = argparse.ArgumentParser()
    for name in ('manifest', 'selection', 'pilot', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--previous-confirmation', type=Path, action='append', required=True)
    parser.add_argument('--count', type=int, default=32)
    args = parser.parse_args()
    if args.count < 1:
        parser.error('--count must be positive')
    paths = {'manifest': args.manifest, 'selection': args.selection, 'pilot': args.pilot,
             **{f'previous_confirmation_{index}': path for index, path in enumerate(args.previous_confirmation)}}
    source_bytes = {name: path.read_bytes() for name, path in paths.items()}
    values = {name: json.loads(data) for name, data in source_bytes.items()}
    hashes = {name: digest(data) for name, data in source_bytes.items()}
    for name, value in values.items():
        if name != 'manifest' and value.get('manifest_sha256') != hashes['manifest']:
            raise ValueError(f'{name} uses a different manifest')
    selected = values['selection']
    if len(selected['train']) != 2048 or len(selected['val']) != 128:
        raise ValueError('Expected existing 2048 training / 128 development selection')
    if selected.get('style_id') != 0 or selected.get('speaker_id') != 1:
        raise ValueError('Expected original style0 / voice-group1 selection')
    import soundfile as sf
    confirmation, audit = select_confirmation(values['manifest'], selected, values['pilot'],
        [values[f'previous_confirmation_{index}'] for index in range(len(args.previous_confirmation))],
        lambda row: sf.info(row['reference_audio']).duration, args.count)
    data = json_bytes(confirmation)
    audit.update(source_sha256=hashes, source_paths={name: str(path) for name, path in paths.items()},
                 confirmation_selection_sha256=digest(data))
    write_immutable(args.output, {'confirmation_selection.json': data,
                                 'confirmation_data_audit.json': json_bytes(audit)})
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
