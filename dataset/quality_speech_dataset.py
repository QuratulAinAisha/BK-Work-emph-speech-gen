"""Add pretrained A speech and B content targets. / A 사전학습 음성과 B 내용 타깃을 추가합니다."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from dataset.full_speech_dataset import FullSpeechDataset, collate_full_speech
from model.full_speech.config import SpeechConfig


class QualitySpeechDataset(torch.utils.data.Dataset):
    def __init__(self, manifest, config, split):
        self.path = Path(manifest)
        self.metadata = json.loads(self.path.read_text())
        self.config = config
        if self.metadata['target_contract'] != config.target_contract():
            raise ValueError('Quality cache representation mismatch')
        base = json.loads(Path(self.metadata['base_manifest']).read_text())
        if hashlib.sha256(Path(self.metadata['base_manifest']).read_bytes()).hexdigest() != self.metadata['base_manifest_sha256']:
            raise ValueError('Original manifest changed')
        self.base = FullSpeechDataset(self.metadata['base_manifest'], SpeechConfig(**self.metadata['base_config']), split)
        mapping = {row['path']: i for i, row in enumerate(self.base.records)}
        self.records = [row for row in self.metadata['records'] if row['split'] == split]
        self.indices = [mapping[row['base_path']] for row in self.records]
        if not self.records:
            raise ValueError(f'No {split} samples')
        original = {row['path']: row for row in base['records']}
        for row in self.metadata['records']:
            old = original[row['base_path']]
            if (row['split'], row['conversation_id']) != (old['split'], old['conversation_id']):
                raise ValueError('Conversation split changed')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        sample = self.base[self.indices[index]]
        with np.load(self.path.parent / row['path'], allow_pickle=False) as data:
            for name in ('semantic', 'waveform'):
                value = data[name].astype(np.float32)
                if not np.isfinite(value).all():
                    raise ValueError(f'Non-finite {name}')
                sample[name] = torch.from_numpy(value.copy())
            sample['text_b'] = torch.from_numpy(data['text_b'].copy()).long()
        with np.load(self.path.parent / row['person_a_path'], allow_pickle=False) as data:
            sample['speech_a'] = torch.from_numpy(data['speech_a'].astype(np.float32).copy())
            sample['text_a'] = torch.from_numpy(data['text_a'].copy()).long()
        if sample['semantic'].shape != (sample['codec'].shape[0], 768) or sample['speech_a'].shape[-1] != 768:
            raise ValueError('Native speech feature shape mismatch')
        for label, feature in (('text_a', 'speech_a'), ('text_b', 'semantic')):
            text = sample[label]
            needed = len(text) + int((text[1:] == text[:-1]).sum())
            if needed > sample[feature].shape[0]:
                raise ValueError(f'{row["path"]}: transcript too long for CTC')
        number = int(hashlib.sha256(row['conversation_id'].encode()).hexdigest()[:15], 16)
        sample['conversation_key'] = torch.tensor(number, dtype=torch.long)
        return sample


def collate_quality(samples):
    batch = collate_full_speech(samples)
    for key in ('speech_a', 'text_a', 'text_b', 'waveform'):
        batch[key] = pad_sequence([sample[key] for sample in samples], batch_first=True)
        batch[key + '_len'] = torch.tensor([len(sample[key]) for sample in samples], dtype=torch.long)
    batch['conversation_key'] = torch.stack([sample['conversation_key'] for sample in samples])
    return batch


def person_a_only(batch):
    # Never forward B features or text to normal inference. / 일반 추론에 B 특징·문장을 전달하지 않습니다.
    return {key: batch[key] for key in ('mel', 'dmm', 'au', 'mel_len', 'dmm_len', 'au_len',
                                      'speech_a', 'speech_a_len', 'style_id', 'speaker_id')}
