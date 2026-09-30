"""Check a smaller unit book with frozen acoustics. / 작은 단위 사전을 고정 음향 모델로 검사합니다."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from dataset.quality_speech_dataset import QualitySpeechDataset, collate_quality, person_a_only
from model.full_speech.codec import FrozenEncodec
from model.full_speech.quality import QualitySpeechSystem
from model.full_speech.tensor_ops import counts
from model.full_speech.units import UnitSpeechSystem, load_codebook
from prepare_quality import atomic_json
from scripts.diagnose_quality import recognize, save_wave
from train_full import move_batch


SEED = 42
ARMS = ('current_units', 'candidate_units', 'continuous_semantics')
LIMITATIONS = [
    'All generated arms use GT B semantic features and true B duration: oracle diagnosis only, never normal A-only inference.',
    'The A encoder receives person_a_only inputs; no B semantic features, audio or text enter that encoder.',
    'The planner is bypassed. This checks frozen acoustic compatibility, not planner learning or response relevance.',
    'Candidate codebook train-only provenance is declared by fit_speech_units.py metadata; fitting is not repeated here.',
    'Candidate unit centers and continuous features bypass the current unit wrapper to avoid re-quantization.',
    'All arms retain identical A conditioning, requested style/voice, true B duration, acoustic weights and seed 42.',
    'Lower quantization distortion need not imply more intelligible audio, and a coarser book may require acoustic retraining.',
    'Reference WER is an ASR reconstruction diagnostic; ASR can hallucinate and is not human listening or an empathy score.',
    'The selected validation examples are a development gate, not a held-out test-set result.',
]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(str((str(value.dtype), tuple(value.shape))).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def module_hashes(module):
    """Hash parameters and buffers by top-level component. / 상위 구성요소별 매개변수와 버퍼를 해시합니다."""
    grouped = {}
    for name, value in sorted(module.state_dict().items()):
        if not torch.isfinite(value).all():
            raise ValueError(f'Non-finite checkpoint tensor: {name}')
        key = name.split('.', 1)[0] if '.' in name else '_root_buffers'
        digest = grouped.setdefault(key, hashlib.sha256())
        digest.update(name.encode())
        digest.update(tensor_hash(value).encode())
    return {key: digest.hexdigest() for key, digest in grouped.items()}


def load_candidate(path, model, manifest_hash):
    book, payload = load_codebook(path, model)
    for name in ('semantic_mean', 'semantic_std'):
        expected = getattr(model, name).detach().cpu()
        supplied = payload[name]
        if (supplied.shape != (768,) or supplied.dtype != expected.dtype
                or not torch.isfinite(supplied).all() or not torch.equal(supplied, expected)):
            raise ValueError('Codebook normalization must exactly match the frozen model')
    if not (payload['semantic_std'] > 0).all():
        raise ValueError('Codebook standard deviation must be positive')
    if payload.get('fit_split') != 'train' or payload.get('manifest_sha256') != manifest_hash:
        raise ValueError('Candidate codebook must declare train-only fitting on this manifest')
    selection_hash = payload.get('selection_sha256')
    if (not isinstance(selection_hash, str) or len(selection_hash) != 64
            or any(char not in '0123456789abcdef' for char in selection_hash.lower())):
        raise ValueError('Candidate codebook needs fit-selection SHA256 provenance')
    if (payload.get('clusters') != len(book.centers) or len(book.centers) < 2
            or payload.get('training_conversations', 0) < 1 or payload.get('frames', 0) < 1):
        raise ValueError('Invalid fitted codebook counts')
    return book, payload


def selected_indices(records, selection, manifest_hash, count):
    wanted = selection.get('val', [])
    if selection.get('manifest_sha256') != manifest_hash:
        raise ValueError('Audio selection manifest changed')
    if count < 1 or len(wanted) < count or len(wanted) != len(set(wanted)):
        raise ValueError('Need enough unique preselected validation paths')
    if set(wanted) & set(selection.get('train', [])):
        raise ValueError('Audio validation selection overlaps training')
    mapping = {row['path']: index for index, row in enumerate(records)}
    if len(mapping) != len(records) or any(path not in mapping for path in wanted):
        raise ValueError('Selected validation paths are missing or records are duplicated')
    selected = [mapping[path] for path in wanted[:count]]
    if (any(records[index]['split'] != 'val' for index in selected)
            or len({records[index]['conversation_id'] for index in selected}) != count):
        raise ValueError('Audio selection must use unique validation conversations')
    return selected


@torch.inference_mode()
def generate_oracle_arms(model, candidate, batch, codec):
    """Prequantize once, then call the continuous interface. / 한 번 양자화한 후 연속 특징 인터페이스를 호출합니다."""
    if not isinstance(model, UnitSpeechSystem) or model.training or len(batch['duration']) != 1:
        raise ValueError('Expected an eval-mode discrete model and one example')
    duration = batch['duration']
    semantic = batch['semantic']
    length = int(batch['semantic_len'][0])
    if (not torch.isfinite(duration).all() or (duration <= 0).any()
            or semantic.shape != (1, length, 768) or length < 1
            or int(counts(duration, model.config.semantic_hz)[0]) != length
            or not torch.isfinite(semantic).all()):
        raise ValueError('Expected finite 768D B semantics aligned to true B duration without padding')
    target = (semantic - model.semantic_mean) / model.semantic_std
    if not torch.isfinite(target).all():
        raise ValueError('Non-finite normalized B semantics')
    books = {'current_units': model.semantic_planner.codebook, 'candidate_units': candidate}
    vectors, metrics = {}, {}
    for name, book in books.items():
        ids = book.encode(target)
        values = book.centers[ids]
        vectors[name] = values
        metrics[name] = {'clusters': len(book.centers), 'frames': length,
            'normalized_mse': float((values - target).square().mean()),
            'normalized_squared_error_sum': float((values - target).double().square().sum()),
            'feature_elements': target.numel(), 'used_units': int(ids.unique().numel()),
            'unit_ids': ids[0].cpu().tolist(), 'semantic_sha256': tensor_hash(values)}
    vectors['continuous_semantics'] = target
    metrics['continuous_semantics'] = {'clusters': None, 'frames': length, 'normalized_mse': 0.,
        'normalized_squared_error_sum': 0., 'feature_elements': target.numel(),
        'semantic_sha256': tensor_hash(target)}
    inputs = person_a_only(batch)
    results = {}
    for name in ARMS:
        # Bypass UnitSpeechSystem's old-book re-quantization. / 기존 사전으로 다시 양자화하는 래퍼를 건너뜁니다.
        generated = QualitySpeechSystem.generate_batch(model, inputs, codec, seed=SEED,
            oracle_semantic=vectors[name], oracle_duration=duration)
        if not torch.equal(generated['semantic'], vectors[name]) or not torch.equal(generated['duration'], duration):
            raise AssertionError('Oracle semantic vectors or true duration changed')
        if results:
            first = results[ARMS[0]]
            for key in ('context', 'context_mask', 'affect', 'audio_lengths'):
                if not torch.equal(generated[key], first[key]):
                    raise AssertionError(f'Conditioning changed between oracle arms: {key}')
        results[name] = generated
    return {'results': results, 'metrics': metrics, 'duration_seconds': float(duration[0])}


def summarize_distortion(examples):
    result = {}
    for name in ARMS:
        rows = [row['unit_metrics'][name] for row in examples]
        elements = sum(row['feature_elements'] for row in rows)
        result[name] = {'clusters': rows[0]['clusters'], 'count': len(rows),
            'frames': sum(row['frames'] for row in rows),
            'frame_weighted_normalized_mse': sum(row['normalized_squared_error_sum'] for row in rows) / elements,
            'example_mean_normalized_mse': sum(row['normalized_mse'] for row in rows) / len(rows)}
        if name != 'continuous_semantics':
            result[name]['used_units'] = len({unit for row in rows for unit in row['unit_ids']})
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'manifest', 'selection', 'codebook', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--count', type=int, default=8)
    args = parser.parse_args()
    if args.count < 1:
        parser.error('--count must be positive')
    if args.output.exists():
        parser.error('--output must be a new directory')
    torch.set_num_threads(2)
    device = torch.device(args.device)
    manifest_hash = file_hash(args.manifest)
    model, payload = UnitSpeechSystem.from_checkpoint(args.checkpoint)
    candidate, candidate_payload = load_candidate(args.codebook, model, manifest_hash)
    if len(model.semantic_planner.codebook.centers) != 1024 or len(candidate.centers) != 256:
        raise ValueError('This gate requires a current 1024-unit checkpoint and a candidate 256-unit codebook')
    model.to(device).requires_grad_(False).eval()
    candidate.to(device).requires_grad_(False).eval()
    data = QualitySpeechDataset(args.manifest, model.config, 'val')
    selection = json.loads(args.selection.read_text())
    chosen = selected_indices(data.records, selection, manifest_hash, args.count)
    codec = FrozenEncodec().to(device).requires_grad_(False).eval()
    before = {'model': module_hashes(model), 'codec': module_hashes(codec), 'candidate': module_hashes(candidate)}
    current_hash = tensor_hash(model.semantic_planner.codebook.centers)
    report = {'checkpoint': str(args.checkpoint), 'checkpoint_sha256': file_hash(args.checkpoint),
        'checkpoint_step': payload.get('recovery_step'), 'architecture': payload['architecture'],
        'manifest_sha256': manifest_hash, 'selection_sha256': file_hash(args.selection),
        'candidate_codebook': str(args.codebook), 'candidate_codebook_sha256': file_hash(args.codebook),
        'candidate_fit_provenance': {key: candidate_payload.get(key) for key in
            ('fit_split', 'manifest_sha256', 'selection_sha256', 'training_conversations', 'frames',
             'seed', 'feature', 'normalized_mse')},
        'current_codebook': {'clusters': 1024, 'centers_sha256': current_hash,
            'reported_source_sha256': payload.get('metadata', {}).get('recovery_recipe', {}).get('codebook_sha256')},
        'candidate_clusters': 256, 'normalization_exact_match': True, 'semantic_interface_dim': 768,
        'sampling': {'acoustic_seed': SEED, 'codec_steps': model.config.codec_steps,
            'oracle_B_duration': True, 'normal_inference': False, 'planner_bypassed': True,
            'generation_entry_point': 'QualitySpeechSystem.generate_batch', 'double_quantization': False},
        'encoder_input_contract': 'person_a_only', 'split': 'val', 'test_split_read': False,
        'asr_model': 'openai/whisper-base.en', 'module_hashes_before': before,
        'limitations': LIMITATIONS, 'examples': []}
    args.output.mkdir(parents=True)
    for position, index in enumerate(chosen):
        row, sample = data.records[index], data[index]
        batch = move_batch(collate_quality([sample]), device)
        generated = generate_oracle_arms(model, candidate, batch, codec)
        entry = {'path': row['path'], 'conversation_id': row['conversation_id'],
            'input_text': row['input_text'], 'reference_text': row['response_text'],
            'style_id': row['style_id'], 'speaker_id': row['speaker_id'],
            'oracle_duration_seconds': generated['duration_seconds'], 'normal_inference': False,
            'unit_metrics': generated['metrics'], 'paths': {}}
        waves = {'reference': sample['waveform'].numpy()}
        for name, result in generated['results'].items():
            waves[name] = result['waveform'][0, :int(result['audio_lengths'][0])].cpu().numpy()
        for name, waveform in waves.items():
            filename = f'{position:02d}_{name}.wav'
            entry['paths'][name] = {**save_wave(args.output, filename, waveform),
                'wav_sha256': file_hash(args.output / filename), 'normal_inference': False,
                'oracle_only': name != 'reference', 'reference_recording': name == 'reference'}
        report['examples'].append(entry)
        atomic_json(args.output / 'generated_report.json', report)
        print(json.dumps({'generated': position + 1, 'count': len(chosen)}), flush=True)
    after = {'model': module_hashes(model), 'codec': module_hashes(codec), 'candidate': module_hashes(candidate)}
    report.update(module_hashes_after=after, all_modules_unchanged=before == after,
                  distortion_summary=summarize_distortion(report['examples']))
    atomic_json(args.output / 'generated_report.json', report)
    if before != after:
        raise AssertionError('A frozen model, codec or codebook changed during evaluation')
    del model, codec, candidate
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    report = recognize(report, args.output, str(device))
    atomic_json(args.output / 'report.json', report)
    print(json.dumps({'distortion': report['distortion_summary'], 'asr': report['summary']}), flush=True)


if __name__ == '__main__':
    main()
