"""Build a separate native-width speech cache. / 원래 캐시를 보존하며 고차원 음성 캐시를 만듭니다."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from model.full_speech.quality import QualityConfig, text_ids, normalize_text
from model.full_speech.targets import FrozenSemanticTeacher, resample_audio
from utils.distributed_training import DistributedRuntime


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def atomic_npz(path, **values):
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **values)
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('outputs/bk_source/source.json'))
    parser.add_argument('--base-config', type=Path, default=Path('outputs/bk_source/config.json'))
    parser.add_argument('--base-manifest', type=Path, default=Path('outputs/bk_prepared/manifest.json'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, default=Path('outputs/bk_quality_prepared'))
    parser.add_argument('--max-conversations', type=int)
    args = parser.parse_args()
    runtime = DistributedRuntime.initialize('cuda')
    torch.set_num_threads(2)
    try:
        prepare(args, runtime)
    finally:
        runtime.close()


def prepare(args, runtime):
    source = json.loads(args.source.read_text())
    base = json.loads(args.base_manifest.read_text())
    base_config = json.loads(args.base_config.read_text())
    config = QualityConfig.load(args.config) if args.config else QualityConfig(sbe=base_config['sbe'])
    root = Path(source['records'][0]['response_audio']).parents[1]
    text_path = root / 'generated_text/train_final_with_reference_images.json'
    texts = {row['conv_id']: row for row in json.loads(text_path.read_text())}
    owners = {key: i for i, key in enumerate(dict.fromkeys(row['conversation_id'] for row in base['records']))}
    selected = set(owners)
    if args.max_conversations:
        # Keep all splits in the smoke corpus. / 작은 검사 집합에도 모든 분할을 유지합니다.
        selected = set()
        for split in ('train', 'val', 'test'):
            values = list(dict.fromkeys(row['conversation_id'] for row in base['records'] if row['split'] == split))
            selected.update(values[:max(1, args.max_conversations // 3)])
    fingerprint = hashlib.sha256(json.dumps({'source': source, 'config': config.to_dict(),
        'base_manifest': base, 'texts_sha256': hashlib.sha256(text_path.read_bytes()).hexdigest(),
        'selected': sorted(selected)}, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    identity = args.output / 'identity.json'
    if runtime.primary:
        if identity.exists() and json.loads(identity.read_text())['sha256'] != fingerprint:
            raise ValueError('Refusing to mix different quality caches')
        atomic_json(identity, {'sha256': fingerprint})
        atomic_json(args.output / 'config.json', config.to_dict())
    if runtime.distributed:
        torch.distributed.barrier()
    teacher = FrozenSemanticTeacher(config).to(runtime.device).eval()
    @torch.inference_mode()
    def native(wave, rate, target_length):
        values = torch.from_numpy(resample_audio(wave, rate, 16000)).to(runtime.device)
        values = (values - values.mean()) / (values.var(unbiased=False) + 1e-7).sqrt()
        # Keep all 768 channels; only align the natural frame boundary. / 768차원을 유지하며 프레임 경계만 정렬합니다.
        hidden = teacher.model(values[None]).last_hidden_state
        return F.interpolate(hidden.transpose(1, 2), size=target_length, mode='linear',
                             align_corners=False)[0].T.cpu().numpy().astype(np.float16)
    def waveform(path):
        value, rate = sf.read(path, dtype='float32')
        if value.ndim == 2:
            value = value.mean(1)
        if not len(value) or not np.isfinite(value).all():
            raise ValueError(f'Invalid waveform {path}')
        return value, rate
    rows, assigned, done = [], [], 0
    for index, old in enumerate(base['records']):
        if old['conversation_id'] not in selected:
            continue
        paired = source['records'][index]
        row = {**old, 'base_path': old['path'], 'path': f'content_{index:06d}.npz',
               'person_a_path': f'person_a_{owners[old["conversation_id"]]:05d}.npz',
               'source_index': index, 'input_text': texts[old['conversation_id']]['input'],
               'response_text': paired['response_text'], 'style_id': paired['style_id'],
               'speaker_id': paired['speaker_id'], 'reference_audio': paired['response_audio']}
        rows.append(row)
        if owners[old['conversation_id']] % runtime.world_size == runtime.rank:
            assigned.append((row, paired))
    started = time.monotonic()
    for row, paired in assigned:
        a_path = args.output / row['person_a_path']
        if not a_path.exists():
            audio_path = root / 'generated_input_audio' / (Path(paired['mel']).stem + '.wav')
            wave, rate = waveform(audio_path)
            length = max(1, int(np.ceil(len(wave) / rate * 50 - 1e-9)))
            atomic_npz(a_path, speech_a=native(wave, rate, length), text_a=text_ids(row['input_text']).numpy())
        b_path = args.output / row['path']
        if not b_path.exists():
            wave, rate = waveform(paired['response_audio'])
            wave32 = resample_audio(wave, rate, 32000)
            with np.load(args.base_manifest.parent / row['base_path']) as old:
                length = old['codec'].shape[0]
            atomic_npz(b_path, semantic=native(wave32, 32000, length), waveform=wave32,
                       text_b=text_ids(row['response_text']).numpy())
        done += 1
        if done == 1 or done % 25 == 0:
            progress = {'rank': runtime.rank, 'completed': done, 'assigned': len(assigned),
                        'seconds': time.monotonic() - started}
            atomic_json(args.output / f'progress_rank{runtime.rank}.json', progress)
            print(json.dumps(progress), flush=True)
    if runtime.distributed:
        torch.distributed.barrier()
    metadata = {'schema_version': 2, 'target_contract': config.target_contract(), 'records': rows,
                'base_manifest': str(args.base_manifest.resolve()), 'base_config': base_config,
                'base_manifest_sha256': hashlib.sha256(args.base_manifest.read_bytes()).hexdigest(),
                'provenance': source.get('provenance', {}), 'synthetic': False}
    if runtime.primary:
        atomic_json(args.output / 'manifest.json', metadata)
    if runtime.distributed:
        torch.distributed.barrier()
    from dataset.quality_speech_dataset import QualitySpeechDataset
    for split in ('train', 'val', 'test'):
        dataset = QualitySpeechDataset(args.output / 'manifest.json', config, split)
        for i in range(runtime.rank, len(dataset), runtime.world_size):
            dataset[i]
    if runtime.distributed:
        torch.distributed.barrier()
    if runtime.primary:
        atomic_json(args.output / 'complete.json', {'sha256': fingerprint, 'records': len(rows), 'validated': True})


if __name__ == '__main__':
    main()
