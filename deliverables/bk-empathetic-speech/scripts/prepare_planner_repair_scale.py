"""Select more training conversations without touching test audio. / 테스트 음성 없이 학습 대화를 확장합니다."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import soundfile as sf
from prepare_quality import atomic_json
from model.full_speech.quality import normalize_text
from scripts.prepare_planner_experiment import TextIndex


def main():
    parser = argparse.ArgumentParser()
    for name in ('manifest', 'selection', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--count', type=int, default=4096)
    args = parser.parse_args()
    if args.count < 2048 or args.count % 64:
        parser.error('Use at least2048 conversations and a multiple of64')
    raw = args.manifest.read_bytes()
    manifest, selected = json.loads(raw), json.loads(args.selection.read_text())
    if selected['manifest_sha256'] != hashlib.sha256(raw).hexdigest():
        raise ValueError('Manifest differs from the source selection')
    by_path = {row['path']: row for row in manifest['records']}
    original = [by_path[path] for path in selected['train']]
    if any(row['split'] != 'train' for row in original):
        raise ValueError('Source training selection crosses splits')
    rows, seen = list(original), {row['conversation_id'] for row in original}
    candidates = [row for row in manifest['records'] if row['split'] == 'train'
                  and row['style_id'] == selected['style_id'] and row['speaker_id'] == selected['speaker_id']]
    candidates.sort(key=lambda row: hashlib.sha256(('repair-scale42:' + row['conversation_id']).encode()).hexdigest())
    development = TextIndex()
    for path in selected['val']:
        row = by_path[path]
        development.add(row['conversation_id'], normalize_text(row['input_text']))
    rejected = {'duration': 0, 'digits': 0, 'lexically_related_to_development': 0}
    for row in candidates:
        if len(rows) == args.count:
            break
        if row['conversation_id'] in seen:
            continue
        if development.has_related(normalize_text(row['input_text'])):
            rejected['lexically_related_to_development'] += 1; continue
        if any(char.isdigit() for char in row['response_text']):
            rejected['digits'] += 1; continue
        if not 3 <= sf.info(row['reference_audio']).duration <= 8:
            rejected['duration'] += 1; continue
        rows.append(row); seen.add(row['conversation_id'])
    if len(rows) != args.count or seen & set(selected['conversations']['val']):
        print(json.dumps({'requested': args.count, 'eligible': len(rows), 'candidate_rows': len(candidates),
                          'candidate_conversations': len({row['conversation_id'] for row in candidates}),
                          'development_overlap': len(seen & set(selected['conversations']['val'])),
                          'rejections': rejected}), flush=True)
        raise ValueError('Insufficient eligible unique training conversations or split overlap')
    result = copy.deepcopy(selected)
    result['train'] = [row['path'] for row in rows]
    result['conversations']['train'] = [row['conversation_id'] for row in rows]
    result['selection_note'] = 'Training-only expansion; source development unchanged; fixed style and undocumented voice group.'
    result['repair_scale'] = {'source_selection_sha256': hashlib.sha256(args.selection.read_bytes()).hexdigest(),
        'count': args.count, 'existing_training_preserved': len(original), 'rejections': rejected,
        'test_audio_or_metrics_read': False, 'codebook_refit': False}
    if args.output.exists() and json.loads(args.output.read_text()) != result:
        raise ValueError('Refusing to overwrite a different training selection')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    print(json.dumps(result['repair_scale']))


if __name__ == '__main__':
    main()
