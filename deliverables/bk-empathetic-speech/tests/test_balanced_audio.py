"""Sampled-speech coverage and loss weights. / 생성 음성 감독 범위와 손실 가중치 검증."""

import unittest
from unittest.mock import patch

import torch

from dataset.quality_speech_dataset import collate_quality, person_a_only
from model.full_speech.quality import text_ids
from model.full_speech.recovery import RecoverySpeechSystem
from tests.test_quality_speech import FakeCodec, config, sample
from tests.test_recovery import FakeTeacher


class BalancedAudioTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        cfg = config()
        cfg.acoustic_loss_every = 1
        self.model = RecoverySpeechSystem(cfg)
        self.codec, self.teacher = FakeCodec(), FakeTeacher()
        object.__setattr__(self.model, '_acoustic_codec', self.codec)
        self.model.configure_recovery('acoustic', teacher=self.teacher)
        self.rows = [sample(i % 2) for i in range(3)]
        for row, words in zip(self.rows, ['hello', 'nice day', 'good luck']):
            row['text_b'] = text_ids(words)
        self.batch = collate_quality(self.rows)

    def test_requested_samples_are_distinct_and_rotate_each_update(self):
        self.model.configure_sampled_audio(flow_weight=.1, waveform_samples=2)
        self.model.train()
        indices = []

        def waveform(batch, seed=42, index=0):
            indices.append(index)
            return batch['waveform'][index, :int(batch['waveform_len'][index])]

        # Every update supervises different valid items. / 매 갱신마다 서로 다른 유효 샘플을 감독합니다.
        with patch.object(self.model, 'sampled_waveform', side_effect=waveform):
            for step in range(2):
                self.model.current_step = step
                losses = self.model.losses(self.batch)
                self.assertGreater(float(losses['sampled_ctc']), 0.)
        self.assertEqual(len(indices), 4)
        self.assertEqual(len(set(indices[:2])), 2)
        self.assertEqual(len(set(indices[2:])), 2)
        self.assertGreaterEqual(len(set(indices)), 3)

    def test_validation_averages_every_item_and_uses_matching_targets(self):
        self.model.configure_sampled_audio(flow_weight=.1, waveform_samples=1)
        self.model.eval()
        indices, transcripts, references = [], [], []

        def waveform(batch, seed=42, index=0):
            indices.append(index)
            return torch.full((int(batch['waveform_len'][index]),), float(index + 1))

        def ctc(wave, labels):
            transcripts.append(labels.clone())
            return wave.mean()

        def spectral(wave, reference):
            references.append(reference.clone())
            return 2 * wave.mean()

        with patch.object(self.model, 'sampled_waveform', side_effect=waveform), \
                patch.object(self.teacher, 'forward', side_effect=ctc), \
                patch('model.full_speech.quality.multiresolution_spectral', side_effect=spectral):
            with torch.no_grad():
                losses = self.model.losses(self.batch)
        self.assertEqual(sorted(indices), [0, 1, 2])
        self.assertAlmostEqual(float(losses['sampled_ctc']), self.model.teacher_weight * 2, places=6)
        self.assertAlmostEqual(float(losses['sampled_spectral']), self.model.config.spectral_weight * 4, places=6)
        for index, labels, reference in zip(indices, transcripts, references):
            torch.testing.assert_close(labels, self.rows[index]['text_b'])
            torch.testing.assert_close(reference, self.rows[index]['waveform'])

    def test_zero_flow_weight_still_updates_only_generator_from_audio(self):
        self.model.configure_sampled_audio(flow_weight=0., waveform_samples=2)
        self.model.train()
        losses = self.model.losses(self.batch)
        self.assertEqual(float(losses['codec_flow'].detach()), 0.)
        torch.testing.assert_close(losses['total'], losses['sampled_ctc'] + losses['sampled_spectral'])
        losses['total'].backward()
        gradients = [p.grad for p in self.model.codec_generator.parameters() if p.grad is not None]
        self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))
        self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.)
        self.assertTrue(all(p.grad is None for name, p in self.model.named_parameters()
                            if not name.startswith('codec_generator.')))
        self.assertTrue(all(p.grad is None for p in self.teacher.parameters()))

    def test_indexed_waveform_matches_single_utterance_inference(self):
        self.model.configure_sampled_audio(flow_weight=.1, waveform_samples=2)
        self.model.eval()
        second = self.rows[1]
        second['mel'] = second['mel'][:30]
        second['dmm'] = second['dmm'][:9]
        second['au'] = second['au'][:8]
        second['speech_a'] = second['speech_a'][:20]
        second['semantic'] = second['semantic'][:12]
        second['codec'] = second['codec'][:12]
        second['waveform'] = second['waveform'][:7680]
        second['duration'] = torch.tensor(.24)
        batch = collate_quality(self.rows)
        single = collate_quality([second])
        semantic, _ = self.model.target(single, 'semantic', 768)
        semantic = (semantic - self.model.semantic_mean) / self.model.semantic_std
        expected = self.model.generate_batch(person_a_only(single), self.codec, seed=42,
            oracle_semantic=semantic, oracle_duration=single['duration'])['waveform'][0]
        actual = self.model.sampled_waveform(batch, seed=42, index=1)
        self.assertEqual(len(actual), 7680)
        # Padded batches may round differently in attention. / 패딩 배치의 어텐션 반올림 차이를 허용합니다.
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    def test_invalid_weight_or_coverage_is_rejected(self):
        for weight in [-.1, float('nan'), float('inf')]:
            with self.subTest(flow_weight=weight), self.assertRaises(ValueError):
                self.model.configure_sampled_audio(flow_weight=weight, waveform_samples=1)
        for samples in [0, -1, 1.5, True]:
            with self.subTest(waveform_samples=samples), self.assertRaises(ValueError):
                self.model.configure_sampled_audio(flow_weight=.1, waveform_samples=samples)


if __name__ == '__main__':
    unittest.main()
