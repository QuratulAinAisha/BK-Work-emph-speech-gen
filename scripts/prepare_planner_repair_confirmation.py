"""Reserve fresh development cases using report history. / 보고서 이력으로 새 개발 사례를 확보합니다.

Pass experiment/report trees to --used-root, never raw dataset/cache inventory
trees such as outputs/bk_quality_prepared. / 원시 캐시가 아닌 실험 보고서 폴더를 지정합니다.
Active trees must be snapshotted externally or remain unchanged during this run.
Only JSON/JSONL files are scanned; plain-text logs and Markdown are not scanned.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.full_speech.quality import normalize_text
from scripts.prepare_planner_experiment import TextIndex, json_bytes, subset
from scripts.prepare_planner_memory_confirmation import validate_membership


PATH_FIELDS = ('path', 'base_path', 'person_a_path', 'reference_audio')
FEATURE_FIELDS = ('path', 'base_path', 'person_a_path')


class InsufficientPoolError(ValueError):
    def __init__(self, audit):
        self.audit = audit
        super().__init__('Insufficient clean development pool; no filters were relaxed. ' + json.dumps(audit))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical_path(value):
    return re.sub('/+', '/', value.replace('\\', '/')).removeprefix('./').casefold()


def manifest_index(manifest):
    lookup, conversations = {}, {}
    for row in manifest['records']:
        if row['path'] in lookup:
            raise ValueError('Duplicate manifest path: ' + row['path'])
        if row['split'] not in ('train', 'val', 'test'):
            raise ValueError('Unknown original split')
        lookup[row['path']] = row
        identity = row['split'], None if row['split'] == 'test' else normalize_text(row['input_text'])
        cid = row['conversation_id']
        if cid in conversations and conversations[cid] != identity:
            raise ValueError('Conversation text/split differs across variants: ' + cid)
        conversations[cid] = identity
    if not lookup:
        raise ValueError('Empty manifest')
    return lookup, conversations


class ExposureMatcher:
    def __init__(self, manifest):
        self.ids, self.aliases, self.feature_basenames = set(), defaultdict(set), defaultdict(set)
        for row in manifest['records']:
            if row['split'] != 'val':
                continue
            cid = row['conversation_id']
            self.ids.add(cid)
            for field in PATH_FIELDS:
                if isinstance(row.get(field), str):
                    self.aliases[canonical_path(row[field])].add(cid)
                    if field in FEATURE_FIELDS:
                        self.feature_basenames[canonical_path(row[field]).rsplit('/', 1)[-1]].add(cid)

    def strings(self, value):
        # Nested examples, selections and dictionary keys all count. / 중첩 예시·선택 목록·사전 키를 모두 확인합니다.
        pending = [value]
        while pending:
            current = pending.pop()
            if isinstance(current, str):
                yield current
            elif isinstance(current, dict):
                pending.extend(current.keys())
                pending.extend(current.values())
            elif isinstance(current, list):
                pending.extend(current)

    def match(self, value):
        ids, paths = set(), set()
        for text in self.strings(value):
            if text in self.ids:
                ids.add(text)
            aliases = self.aliases.get(canonical_path(text), set())
            basename = canonical_path(text).rsplit('/', 1)[-1]
            aliases = aliases | self.feature_basenames.get(basename, set())
            if aliases:
                ids.update(aliases)
                paths.add(text)
            # Also catch BK IDs/cache filenames embedded in notes. / 메모에 포함된 BK ID와 캐시 파일명도 찾습니다.
            for token in re.findall(r'[\w][\w:.-]*', text):
                token = token.rstrip('.-:')
                # BK IDs contain colons; notes may prepend a label. / BK ID의 콜론과 메모 접두사를 처리합니다.
                candidates = [token] + [token[index + 1:] for index, char in enumerate(token) if char == ':']
                ids.update(candidate for candidate in candidates if candidate in self.ids)
                matched = self.feature_basenames.get(token.casefold(), set())
                if matched:
                    ids.update(matched)
                    paths.add(token)
        return ids, paths


def report_files(roots):
    found = set()
    for supplied in roots:
        root = Path(supplied).resolve()
        if not root.is_dir():
            raise ValueError('Used-report root is not a directory: ' + str(root))
        for path in root.rglob('*'):
            if path.suffix.lower() not in ('.json', '.jsonl') or not path.is_file():
                continue
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                raise ValueError('Report symlink leaves the explicitly supplied root: ' + str(path))
            found.add(resolved)
    return sorted(found, key=lambda path: str(path))


def scan_exposures(manifest, roots):
    matcher = ExposureMatcher(manifest)
    sources, exposed_paths, hashes = defaultdict(set), set(), []
    files = report_files(roots)
    if not files:
        raise ValueError('No JSON/JSONL history found in the supplied experiment roots')
    for path in files:
        raw = path.read_bytes()
        try:
            decoded = raw.decode('utf-8-sig')
            values = [json.loads(line) for line in decoded.splitlines() if line.strip()] if path.suffix.lower() == '.jsonl' else [json.loads(decoded)]
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError('Unreadable report history; use a stable snapshot: ' + str(path)) from error
        matched_ids, matched_paths = set(), set()
        for value in values:
            if isinstance(value, dict) and 'target_contract' in value and isinstance(value.get('records'), list):
                raise ValueError('Used root contains raw manifest inventory; specify experiment/report roots instead: ' + str(path))
            ids, paths = matcher.match(value)
            matched_ids.update(ids)
            matched_paths.update(paths)
        for cid in matched_ids:
            sources[cid].add(str(path))
        exposed_paths.update(matched_paths)
        hashes.append({'path': str(path), 'sha256': digest(raw), 'bytes': len(raw),
                       'exposed_validation_conversations': len(matched_ids), 'matched_path_strings': len(matched_paths)})
    return {'used_roots': sorted({str(Path(root).resolve()) for root in roots}), 'scanned_files': hashes,
            'scanned_file_formats': ['.json', '.jsonl'],
            'format_limit': 'Plain-text logs, Markdown and other formats are not scanned.',
            'exposed_validation_ids': sorted(sources), 'exposed_path_strings': sorted(exposed_paths),
            'sources_by_conversation': {cid: sorted(paths) for cid, paths in sorted(sources.items())}}


def verify_snapshot(exposures):
    expected = {entry['path']: entry['sha256'] for entry in exposures['scanned_files']}
    current = {str(path) for path in report_files(exposures['used_roots'])}
    if current != set(expected):
        raise ValueError('Report tree membership changed; take an immutable snapshot and retry')
    for path, sha in expected.items():
        if digest(Path(path).read_bytes()) != sha:
            raise ValueError('Report changed during selection; take an immutable snapshot: ' + path)


def select_fresh(manifest, selected, exposures, duration_seconds, count=32, seed=42):
    if not 1 <= count <= 32:
        raise ValueError('Use one to 32 fresh development conversations')
    lookup, conversations = manifest_index(manifest)
    validate_membership(selected, 'source selection', lookup, ('train', 'val'))
    if selected.get('style_id') != 0 or selected.get('speaker_id') != 1:
        raise ValueError('Source selection must use style0 / voice-group1')
    if any(lookup[path]['style_id'] != 0 or lookup[path]['speaker_id'] != 1
           for split in ('train', 'val') for path in selected[split]):
        raise ValueError('Source selected row has a different style or voice group')
    current_ids = {lookup[path]['conversation_id'] for path in selected['val']}
    exposed_ids = set(exposures['exposed_validation_ids'])
    if any(cid not in conversations or conversations[cid][0] != 'val' for cid in exposed_ids):
        raise ValueError('Exposure inventory contains unknown or non-validation IDs')
    excluded = current_ids | exposed_ids
    train_index, exposed_index = TextIndex(), TextIndex()
    for cid, (split, text) in conversations.items():
        if split == 'train':
            train_index.add(cid, text)
        if cid in excluded:
            exposed_index.add(cid, text)
    grouped = defaultdict(list)
    for row in manifest['records']:
        if row['split'] == 'val' and row['style_id'] == 0 and row['speaker_id'] == 1:
            grouped[row['conversation_id']].append(row)
    order = sorted(grouped, key=lambda cid: (digest(f'planner-repair-step13:{seed}:{cid}'.encode()), cid))
    accepted, accepted_index, rejected, durations = [], TextIndex(), Counter(), {}
    examined = 0
    for cid in order:
        examined += 1
        rows = sorted(grouped[cid], key=lambda row: row['path'])
        rejected['extra_same_conversation_variants'] += max(0, len(rows) - 1)
        if cid in excluded:
            rejected['previously_exposed_conversation'] += 1
            continue
        text = normalize_text(rows[0]['input_text'])
        if train_index.has_related(text):
            rejected['lexically_related_to_any_original_training_A'] += 1
            continue
        if exposed_index.has_related(text):
            rejected['lexically_related_to_exposed_validation_A'] += 1
            continue
        if accepted_index.has_related(text):
            rejected['lexically_related_within_fresh_set'] += 1
            continue
        for row in rows:
            if any(char.isdigit() for char in row['response_text']):
                rejected['B_response_contains_digits_records'] += 1
                continue
            # Read B duration headers only, never decode or rank audio. / B 길이 헤더만 읽고 음성을 복원하거나 평가하지 않습니다.
            seconds = float(duration_seconds(row))
            if not 3 <= seconds <= 8:
                rejected['B_duration_outside_3_to_8_seconds_records'] += 1
                continue
            accepted.append(row)
            accepted_index.add(cid, text)
            durations[row['path']] = seconds
            break
        else:
            rejected['conversation_has_no_eligible_B_response'] += 1
        if len(accepted) == count:
            break
    audit = {'requested': count, 'eligible_selected': len(accepted),
        'candidate_conversations': len(grouped), 'candidate_records': sum(map(len, grouped.values())),
        'candidate_conversations_examined': examined, 'current_development_conversations': len(current_ids),
        'history_exposed_validation_conversations': len(exposed_ids), 'excluded_validation_conversations_union': len(excluded),
        'all_original_training_conversations_indexed': len(train_index.texts),
        'rejections_before_selection_complete': dict(sorted(rejected.items())),
        'selection_order': f'SHA256(planner-repair-step13:{seed}:conversation_id), then conversation_id; response variants by path.',
        'response_duration_seconds': durations, 'style_id': 0, 'speaker_id': 1,
        'lexical_rule': 'Existing TextIndex: exact normalized A text, or word-bigram Jaccard>=0.8 with >=6 words and length ratio>=0.8.',
        'lexical_exclusion_scopes': ['ALL original training A texts', 'current and previously exposed validation A texts', 'accepted fresh A texts'],
        'splits_preserved': True, 'training_selection_preserved': True, 'test_audio_or_feature_files_opened': 0,
        'audio_samples_decoded': False, 'model_outputs_used_for_ranking': False,
        'no_relaxed_filters': True,
        'limitations': ['Fresh means outside recorded detailed development exposure within the supplied roots, not a pristine final test.',
            'Older runs may have used the original validation split in aggregate; this selection does not undo that exposure.',
            'Unrecorded work or report trees outside --used-root cannot be discovered.',
            'Only JSON/JSONL report files are scanned; plain-text logs, Markdown and other formats are outside this exposure inventory.',
            'Lexical filtering cannot detect every semantic paraphrase; all-original-training filtering is deliberately conservative.',
            'Voice-group1 does not establish a verified individual voice identity.',
            'Known IDs and path aliases are excluded conservatively even when only mentioned in report metadata.']}
    if len(accepted) != count:
        raise InsufficientPoolError(audit)
    note = ('Fresh detailed development confirmation; all original splits and training membership preserved. '
            'Excludes current development and validation IDs/paths recorded in the supplied report trees, '
            'plus lexical near duplicates of all original training, exposed validation and newly selected A texts. '
            'Original validation may have prior aggregate exposure; this is not a pristine test set.')
    confirmation = subset(selected, accepted, note)
    confirmation['fresh_development'] = {'version': 1, 'count': count, 'seed': seed,
                                         'exclusion_inventory_sha256': digest(json_bytes(exposures))}
    audit['selected_conversation_ids'] = [row['conversation_id'] for row in accepted]
    audit['selected_original_splits'] = sorted({row['split'] for row in accepted})
    return confirmation, audit


def main():
    parser = argparse.ArgumentParser(description='Reserve fresh development cases using immutable experiment/report history.')
    for name in ('manifest', 'selection', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--used-root', type=Path, action='append', required=True,
                        help='Experiment/report tree only, repeatable; never raw dataset/cache inventories.')
    parser.add_argument('--count', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not 1 <= args.count <= 32:
        parser.error('--count must be between 1 and 32')
    if args.output.exists():
        raise ValueError('Use a new output directory; confirmation artifacts are never overwritten')
    source_bytes = {'manifest': args.manifest.read_bytes(), 'selection': args.selection.read_bytes()}
    manifest, selected = (json.loads(source_bytes[name]) for name in ('manifest', 'selection'))
    if selected.get('manifest_sha256') != digest(source_bytes['manifest']):
        raise ValueError('Source selection and manifest hashes differ')
    if len(selected['val']) != 128:
        raise ValueError('Expected the current 128 development selection, not an audio or previous confirmation subset')
    manifest_index(manifest)
    exposures = scan_exposures(manifest, args.used_root)
    import soundfile as sf
    failure = None
    try:
        confirmation, audit = select_fresh(manifest, selected, exposures,
            lambda row: sf.info(row['reference_audio']).duration, args.count, args.seed)
    except InsufficientPoolError as error:
        failure, confirmation, audit = str(error), None, error.audit
    verify_snapshot(exposures)
    for name, path in (('manifest', args.manifest), ('selection', args.selection)):
        if path.read_bytes() != source_bytes[name]:
            raise ValueError('Source input changed during selection: ' + name)
    selection_bytes = json_bytes(confirmation) if confirmation is not None else None
    audit.update(source_paths={'manifest': str(args.manifest.resolve()), 'selection': str(args.selection.resolve())},
        source_sha256={name: digest(raw) for name, raw in source_bytes.items()},
        scanned_report_hashes=exposures['scanned_files'], used_roots=exposures['used_roots'],
        exposed_validation_ids=exposures['exposed_validation_ids'],
        confirmation_selection_sha256=digest(selection_bytes) if selection_bytes is not None else None,
        exclusion_inventory_sha256=digest(json_bytes(exposures)), report_snapshot_verified_unchanged=True)
    audit['status'] = 'insufficient_clean_pool' if failure else 'selected'
    args.output.mkdir(parents=True, exist_ok=False)
    files = {'confirmation_data_audit.json': json_bytes(audit), 'used_validation_exposure.json': json_bytes(exposures)}
    if selection_bytes is not None:
        files['confirmation_selection.json'] = selection_bytes
    for name, data in files.items():
        with (args.output / name).open('xb') as stream:
            stream.write(data)
    print(json.dumps({'output': str(args.output), 'status': audit['status'], 'count': audit['eligible_selected'],
                      'excluded_validation_conversations': audit['excluded_validation_conversations_union'],
                      'scanned_report_files': len(exposures['scanned_files'])}), flush=True)
    if failure:
        print(failure, file=sys.stderr, flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
