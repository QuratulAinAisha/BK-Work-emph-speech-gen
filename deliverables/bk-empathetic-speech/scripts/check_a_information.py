"""Frozen A-feature CTC probes. / 고정된 A 특징의 문자 복원 능력을 검사합니다."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn.utils.rnn import pad_sequence

from dataset.quality_speech_dataset import QualitySpeechDataset, collate_quality, person_a_only
from model.full_speech.loading import load_response_model
from model.full_speech.quality import ALPHABET, ContentHead, content_ctc, normalize_text
from model.full_speech.tensor_ops import mask_from_lengths
from train_full import move_batch


VIEWS = {'raw': 768, 'projected': 512, 'speech': 512, 'fused': 512}
# Optional controls do not change the original extraction recipe. / 선택 대조 실험은 원래 추출 설정을 바꾸지 않습니다.
PROBE_VIEWS = {**VIEWS, 'speech_resampled': 512}
SEED = 42


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def save_tensor_file(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def labels_text(labels):
    ids = labels.tolist() if isinstance(labels, torch.Tensor) else labels
    if any(not 1 <= value <= len(ALPHABET) for value in ids):
        raise ValueError('A transcript contains an invalid character ID')
    return ''.join(ALPHABET[value - 1] for value in ids)


def greedy_text(ids):
    # Collapse repeats before removing blanks. / 공백 제거 전에 연속 반복을 합칩니다.
    collapsed, previous = [], None
    for value in ids:
        value = int(value)
        if value != previous and value != 0:
            if not 1 <= value <= len(ALPHABET):
                raise ValueError('Invalid predicted character ID')
            collapsed.append(ALPHABET[value - 1])
        previous = value
    return ''.join(collapsed)


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for index, first in enumerate(reference, 1):
        current = [index]
        for position, second in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[position] + 1,
                               previous[position - 1] + (first != second)))
        previous = current
    return previous[-1]


def text_metrics(reference, hypothesis):
    words = reference.split()
    if not reference or not words:
        raise ValueError('A transcript must have supported words')
    chars_error = edit_distance(reference, hypothesis)
    words_error = edit_distance(words, hypothesis.split())
    return {'reference_text': reference, 'prediction': hypothesis,
            'character_errors': chars_error, 'reference_characters': len(reference),
            'word_errors': words_error, 'reference_words': len(words),
            'cer': chars_error / len(reference), 'wer': words_error / len(words)}


def aggregate_text(rows):
    if not rows:
        raise ValueError('No transcripts to evaluate')
    return {'count': len(rows),
            'corpus_cer': sum(row['character_errors'] for row in rows) / sum(row['reference_characters'] for row in rows),
            'corpus_wer': sum(row['word_errors'] for row in rows) / sum(row['reference_words'] for row in rows),
            'mean_example_cer': sum(row['cer'] for row in rows) / len(rows),
            'mean_example_wer': sum(row['wer'] for row in rows) / len(rows)}


def required_ctc_frames(labels):
    if labels.ndim != 1 or len(labels) < 1:
        raise ValueError('Expected a nonempty one-dimensional A transcript')
    labels_text(labels)
    return len(labels) + int((labels[1:] == labels[:-1]).sum())


@torch.inference_mode()
def extract_feature_views(model, batch):
    if model.training:
        raise ValueError('A-feature extraction requires model.eval()')
    inputs = person_a_only(batch)
    encoded = model.encode_batch(inputs)
    speech_mask = mask_from_lengths(inputs['speech_a_len'])
    # Projection here precedes sinusoidal position addition. / 이 투영은 위치 인코딩을 더하기 전입니다.
    projected = model.speech_projection(inputs['speech_a'])
    return {'raw': (inputs['speech_a'], speech_mask), 'projected': (projected, speech_mask),
            'speech': (encoded['speech_hidden'], encoded['speech_mask']),
            'fused': (encoded['context'], encoded['context_mask'])}


def trim_feature(values, mask, index, dimension):
    length = int(mask[index].sum())
    if length < 1 or not bool(mask[index, :length].all()) or bool(mask[index, length:].any()):
        raise ValueError('Feature masks must describe nonempty valid prefixes')
    feature = values[index, :length].detach().to(device='cpu', dtype=torch.float32).clone()
    if feature.shape != (length, dimension) or not bool(torch.isfinite(feature).all()):
        raise ValueError('Invalid extracted A-feature shape or values')
    return feature


def input_provenance(args):
    selection = json.loads(args.selection.read_text())
    result = {'checkpoint_sha256': file_hash(args.checkpoint), 'manifest_sha256': file_hash(args.manifest),
              'selection_sha256': file_hash(args.selection), 'cache_version': 1,
              'feature_dimensions': VIEWS, 'encoder_input_contract': 'person_a_only',
              'cache_dtype': 'float32', 'alphabet': ALPHABET}
    if selection.get('manifest_sha256') != result['manifest_sha256']:
        raise ValueError('Selection does not match the manifest')
    return result, selection


def selected_rows(manifest, selection):
    metadata = json.loads(Path(manifest).read_text())
    lookup = {row['path']: row for row in metadata['records']}
    if len(lookup) != len(metadata['records']):
        raise ValueError('Manifest contains duplicate paths')
    rows = {}
    for split in ('train', 'val'):
        paths = selection[split]
        if not paths or len(paths) != len(set(paths)) or any(path not in lookup for path in paths):
            raise ValueError(f'Missing or repeated selected {split} records')
        rows[split] = [lookup[path] for path in paths]
        if any(row['split'] != split for row in rows[split]):
            raise ValueError('Selected examples cross original split boundaries')
        if len({row['conversation_id'] for row in rows[split]}) != len(paths):
            raise ValueError('Selected conversations must be unique within each split')
    if {row['conversation_id'] for row in rows['train']} & {row['conversation_id'] for row in rows['val']}:
        raise ValueError('Training and validation conversations overlap')
    return rows


def compatible_provenance(current, stored):
    if any(stored.get(key) != value for key, value in current.items()):
        raise ValueError('Cached provenance differs from checkpoint, configuration, manifest or selection')


@torch.inference_mode()
def extract(args):
    provenance, selection = input_provenance(args)
    rows = selected_rows(args.manifest, selection)
    cache = args.output / 'cache'
    model, payload = load_response_model(args.checkpoint)
    config = model.config.to_dict()
    provenance.update(config=config, config_sha256=hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest())
    if (cache / 'provenance.json').exists():
        compatible_provenance(provenance, json.loads((cache / 'provenance.json').read_text()))
    else:
        write_json(cache / 'provenance.json', provenance)
    if (cache / 'complete.json').exists():
        index = json.loads((cache / 'index.json').read_text())
        for entries in index['splits'].values():
            for entry in entries:
                for view in VIEWS:
                    if file_hash(cache / entry['files'][view]) != entry['sha256'][view]:
                        raise ValueError('An existing feature cache file changed')
        print(json.dumps({'stage': 'existing_verified_cache', 'counts': {key: len(value) for key, value in rows.items()}}), flush=True)
        return
    model.to(args.device).requires_grad_(False).eval()
    index = {'provenance': provenance, 'splits': {}, 'ctc_infeasible': {view: [] for view in VIEWS}}
    original_head_rows = []
    for split in ('train', 'val'):
        dataset = QualitySpeechDataset(args.manifest, model.config, split)
        mapping = {row['path']: position for position, row in enumerate(dataset.records)}
        index['splits'][split] = []
        for position, row in enumerate(rows[split]):
            sample = dataset[mapping[row['path']]]
            batch = move_batch(collate_quality([sample]), torch.device(args.device))
            views = extract_feature_views(model, batch)
            labels = sample['text_a'].detach().cpu().long().clone()
            needed = required_ctc_frames(labels)
            if labels_text(labels) != normalize_text(row['input_text']):
                raise ValueError('Cached A transcript differs from the manifest input text')
            entry = {'path': row['path'], 'conversation_id': row['conversation_id'], 'split': split,
                     'reference_text': labels_text(labels), 'required_ctc_frames': needed,
                     'files': {}, 'sha256': {}, 'lengths': {}}
            for view, (values, mask) in views.items():
                feature = trim_feature(values, mask, 0, VIEWS[view])
                length = len(feature)
                relative = f'{view}/{split}/{position:06d}.pt'
                item = {'features': feature, 'text_a': labels, 'length': length,
                        'path': row['path'], 'conversation_id': row['conversation_id'], 'split': split, 'view': view}
                # Never replace a previously cached feature with different values. / 기존 특징을 다른 값으로 바꾸지 않습니다.
                target = cache / relative
                if target.exists():
                    previous = torch.load(target, map_location='cpu', weights_only=True)
                    if any(not torch.equal(previous[key], item[key]) if isinstance(item[key], torch.Tensor)
                           else previous[key] != item[key] for key in item):
                        raise ValueError(f'Existing cached feature differs: {target}')
                else:
                    save_tensor_file(target, item)
                entry['files'][view], entry['sha256'][view], entry['lengths'][view] = relative, file_hash(target), length
                if length < needed:
                    index['ctc_infeasible'][view].append({'split': split, 'path': row['path'],
                                                        'frames': length, 'required': needed})
            if split == 'val':
                values, mask = views['speech']
                logits = model.input_content(values, mask)
                hypothesis = greedy_text(logits[0, :int(mask.sum())].argmax(-1).cpu().tolist())
                result = text_metrics(entry['reference_text'], hypothesis)
                result.update(path=row['path'], conversation_id=row['conversation_id'])
                original_head_rows.append(result)
            index['splits'][split].append(entry)
            if (position + 1) % 32 == 0:
                progress = {'stage': 'extract', 'split': split, 'examples': position + 1, 'total': len(rows[split])}
                write_json(args.output / 'extraction_progress.json', progress)
                print(json.dumps(progress), flush=True)
    if file_hash(args.checkpoint) != provenance['checkpoint_sha256']:
        raise ValueError('Source checkpoint changed during extraction')
    write_json(cache / 'index.json', index)
    write_json(args.output / 'existing_input_head.json', {
        'head': 'Frozen production input_content on speech_hidden', 'trained_here': False,
        'split': 'val', **aggregate_text(original_head_rows), 'examples': original_head_rows,
        'limitation': 'This head may be undertrained; failure alone does not prove the A representation lacks words.'})
    write_json(cache / 'complete.json', {'counts': {key: len(value) for key, value in index['splits'].items()},
        'index_sha256': file_hash(cache / 'index.json'),
        'ctc_infeasible_counts': {view: len(items) for view, items in index['ctc_infeasible'].items()}})
    print(json.dumps({'stage': 'extracted', 'ctc_infeasible_counts': {
        view: len(items) for view, items in index['ctc_infeasible'].items()}}), flush=True)


def load_cached(args, view):
    provenance, _ = input_provenance(args)
    cache = args.output / 'cache'
    completion = json.loads((cache / 'complete.json').read_text())
    if file_hash(cache / 'index.json') != completion['index_sha256']:
        raise ValueError('Cache index changed after extraction')
    index = json.loads((cache / 'index.json').read_text())
    compatible_provenance(provenance, index['provenance'])
    if json.loads((cache / 'provenance.json').read_text()) != index['provenance']:
        raise ValueError('Cache configuration provenance changed')
    if view not in index['ctc_infeasible']:
        raise ValueError(f'The selected cache has no {view} view')
    if index['ctc_infeasible'][view]:
        raise ValueError(f'{view} has {len(index["ctc_infeasible"][view])} CTC-infeasible cases; do not drop them or zero the loss')
    result = {}
    for split in ('train', 'val'):
        result[split] = []
        for entry in index['splits'][split]:
            path = cache / entry['files'][view]
            if file_hash(path) != entry['sha256'][view]:
                raise ValueError(f'Cached feature changed: {path}')
            item = torch.load(path, map_location='cpu', weights_only=True)
            if (item['split'], item['view'], item['path'], item['conversation_id']) != (
                    split, view, entry['path'], entry['conversation_id']):
                raise ValueError('Cache record identity mismatch')
            if item['features'].shape != (entry['lengths'][view], PROBE_VIEWS[view]) or item['length'] != len(item['features']):
                raise ValueError('Cache feature dimensions differ from its index')
            if not bool(torch.isfinite(item['features']).all()) or required_ctc_frames(item['text_a']) > item['length']:
                raise ValueError('Invalid cached features or infeasible transcript')
            result[split].append(item)
    return result, index


def collate_features(samples, device):
    lengths = torch.tensor([item['length'] for item in samples], dtype=torch.long, device=device)
    labels = pad_sequence([item['text_a'] for item in samples], batch_first=True).to(device)
    label_lengths = torch.tensor([len(item['text_a']) for item in samples], dtype=torch.long, device=device)
    features = pad_sequence([item['features'] for item in samples], batch_first=True).to(device)
    return features, mask_from_lengths(lengths), labels, label_lengths


@torch.inference_mode()
def evaluate_probe(head, samples, device, batch_size):
    head.eval()
    rows, loss_sum = [], 0.
    for offset in range(0, len(samples), batch_size):
        batch = samples[offset:offset + batch_size]
        features, mask, labels, label_lengths = collate_features(batch, device)
        loss = content_ctc(head, features, mask, labels, label_lengths)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Non-finite probe evaluation loss')
        loss_sum += float(loss) * len(batch)
        logits = head(features, mask)
        for index, item in enumerate(batch):
            predicted = greedy_text(logits[index, :item['length']].argmax(-1).cpu().tolist())
            row = text_metrics(labels_text(item['text_a']), predicted)
            row.update(path=item['path'], conversation_id=item['conversation_id'])
            rows.append(row)
    return {**aggregate_text(rows), 'ctc_loss': loss_sum / len(rows), 'examples': rows}


def epoch_order(count, epoch, seed=SEED):
    # All views see the same examples in the same order. / 모든 특징 뷰에서 같은 순서로 학습합니다.
    return torch.randperm(count, generator=torch.Generator().manual_seed(seed + epoch)).tolist()


def train_view(args):
    view = args.train_view
    samples, index = load_cached(args, view)
    output = args.output / 'probes' / view
    recipe = {'view': view, 'dimension': PROBE_VIEWS[view], 'head': 'ContentHead', 'hidden': 256,
              'layers': 1, 'heads': 4, 'dropout': 0., 'optimizer': 'AdamW', 'lr': args.lr,
              'weight_decay': .01, 'gradient_clip': 1., 'batch_size': args.batch_size,
              'seed': SEED, 'scheduler': 'constant; identical across views', 'precision': 'float32',
              'provenance': index['provenance'], 'train_count': len(samples['train']),
              'validation_count': len(samples['val']), 'train_metric_count': min(64, len(samples['train'])),
              'train_only_optimization': True, 'production_weights_modified': False}
    if (output / 'last.pt').exists() and not args.resume:
        raise ValueError('Existing probe requires --resume; refusing to overwrite its training')
    if args.resume and not (output / 'last.pt').exists():
        raise ValueError('--resume needs an existing probe checkpoint')
    torch.manual_seed(SEED)
    head = ContentHead(PROBE_VIEWS[view], hidden=256).to(args.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=.01)
    start = 0
    if args.resume:
        payload = torch.load(output / 'last.pt', map_location='cpu', weights_only=True)
        if payload['recipe'] != recipe:
            raise ValueError('Probe resume recipe changed')
        head.load_state_dict(payload['state_dict'], strict=True)
        optimizer.load_state_dict(payload['optimizer'])
        start = payload['epoch']
        torch.set_rng_state(payload['torch_rng'])
        if torch.device(args.device).type == 'cuda':
            torch.cuda.set_rng_state(payload['cuda_rng'], torch.device(args.device))
        if args.epochs < start:
            raise ValueError('Requested epochs precede the resumed checkpoint')
    write_json(output / 'recipe.json', recipe)
    write_json(output / 'run.json', {'requested_epochs': args.epochs, 'resume_epoch': start})
    if args.resume and args.epochs > start and (output / 'complete.json').exists():
        # Preserve the old completion before extending its budget. / 학습 예산을 늘리기 전에 이전 완료 기록을 보존합니다.
        write_json(output / f'completed_epoch_{start:04d}.json', json.loads((output / 'complete.json').read_text()))
        (output / 'complete.json').unlink()
    fixed_training = samples['train'][:64]
    since = time.monotonic()
    if start == 0:
        initial = {split: evaluate_probe(head, subset, args.device, args.batch_size)
                   for split, subset in [('train_fixed64', fixed_training), ('val', samples['val'])]}
        write_json(output / 'epoch_0000.json', {'epoch': 0, 'metrics': initial})
    for epoch in range(start, args.epochs):
        head.train()
        order = epoch_order(len(samples['train']), epoch)
        total_loss, updates = 0., 0
        for offset in range(0, len(order), args.batch_size):
            chosen = [samples['train'][position] for position in order[offset:offset + args.batch_size]]
            features, mask, labels, lengths = collate_features(chosen, args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = content_ctc(head, features, mask, labels, lengths)
            if not bool(torch.isfinite(loss)):
                raise ValueError(f'Non-finite {view} training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach()) * len(chosen)
            updates += 1
        metrics = {split: evaluate_probe(head, subset, args.device, args.batch_size)
                   for split, subset in [('train_fixed64', fixed_training), ('val', samples['val'])]}
        epoch_report = {'epoch': epoch + 1, 'train_loss': total_loss / len(order),
                        'updates': updates, 'elapsed_seconds': time.monotonic() - since, 'metrics': metrics}
        write_json(output / f'epoch_{epoch + 1:04d}.json', epoch_report)
        save_tensor_file(output / 'last.pt', {'architecture': 'bk_frozen_a_ctc_probe_v1', 'recipe': recipe,
            'epoch': epoch + 1, 'state_dict': head.state_dict(), 'optimizer': optimizer.state_dict(),
            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(torch.device(args.device))
            if torch.device(args.device).type == 'cuda' else None})
        concise = {'view': view, 'epoch': epoch + 1, 'epochs': args.epochs,
                   'train_loss': epoch_report['train_loss'], 'elapsed_seconds': epoch_report['elapsed_seconds'],
                   'metrics': {split: {key: value for key, value in report.items() if key != 'examples'}
                               for split, report in metrics.items()}}
        write_json(output / 'progress.json', concise)
        print(json.dumps(concise), flush=True)
    if args.epochs == 0:
        raise ValueError('Need at least one training epoch')
    final = json.loads((output / f'epoch_{args.epochs:04d}.json').read_text())
    write_json(output / 'complete.json', {'epochs': args.epochs, 'recipe': recipe, 'final': final,
        'limitation': 'A fresh probe tests recoverable transcript information under this budget; failure alone does not prove information is absent.'})


def summarize(args):
    summaries = {}
    index_path = args.output / 'cache' / 'index.json'
    views = json.loads(index_path.read_text())['ctc_infeasible'] if index_path.exists() else VIEWS
    for view in views:
        folder = args.output / 'probes' / view
        path = folder / 'complete.json'
        if not path.exists():
            summaries[view] = {'status': 'not_complete'}
            continue
        result = json.loads(path.read_text())
        initial = json.loads((folder / 'epoch_0000.json').read_text())
        summaries[view] = {'status': 'complete', 'epochs': result['epochs'],
            'initial': {split: {key: value for key, value in row.items() if key != 'examples'}
                        for split, row in initial['metrics'].items()},
            'final': {split: {key: value for key, value in row.items() if key != 'examples'}
                      for split, row in result['final']['metrics'].items()}}
    head = args.output / 'existing_input_head.json'
    existing = json.loads(head.read_text()) if head.exists() else None
    report = {'views': summaries, 'existing_input_head': {key: value for key, value in existing.items()
               if key != 'examples'} if existing else None,
        'interpretation': [
            'Compare equally trained probes at the same epoch budget; lower corpus CER/WER means more transcript recovery here.',
            'Raw succeeds but projected fails suggests a projection bottleneck or probe optimization issue.',
            'Projected succeeds but speech fails implicates the frozen speech-context transformation.',
            'Speech succeeds but fused fails points toward fusion/timing loss or difficulty decoding that representation.',
            'Compare speech_resampled with fused at identical frame counts to separate timing reduction from fusion effects.',
            'If every probe fails, do not claim the representations lack information; first check probe learning and source ASR.',
            'Transcript recovery is not sufficient evidence of understanding, reply relevance or empathy.',
            'These heads are diagnostic and are not installed in the production model.',
        ]}
    write_json(args.output / 'summary.json', report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ('checkpoint', 'manifest', 'selection', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--extract', action='store_true')
    mode.add_argument('--train-view', choices=tuple(PROBE_VIEWS))
    mode.add_argument('--summarize', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error('Positive epochs, batch size and finite learning rate required')
    if args.resume and not args.train_view:
        parser.error('--resume is only for probe training')
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if args.extract:
        extract(args)
    elif args.train_view:
        train_view(args)
    else:
        summarize(args)


if __name__ == '__main__':
    main()
