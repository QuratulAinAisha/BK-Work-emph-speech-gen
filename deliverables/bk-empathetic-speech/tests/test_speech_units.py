"""Discrete-unit contracts and gradients. / 이산 단위 계약과 기울기 검증."""

import tempfile
from pathlib import Path
import unittest

import torch
from tests.test_quality_speech import config, sample, FakeCodec
from dataset.quality_speech_dataset import collate_quality, person_a_only
from model.full_speech.quality import QualitySpeechSystem
from model.full_speech.loading import load_response_model
from model.full_speech.units import SpeechCodebook, UnitSpeechSystem, initialize_unit_system, load_codebook


class UnitTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.centers = torch.randn(8, 768)
        self.model = UnitSpeechSystem(config(), self.centers)
        self.batch = collate_quality([sample(0), sample(1)])

    def test_quantization_matches_nearest_centers(self):
        book = SpeechCodebook(self.centers)
        values = self.centers[[3, 1, 5]][None] + .001
        self.assertEqual(book.encode(values).tolist(), [[3, 1, 5]])
        torch.testing.assert_close(book(values), self.centers[[3, 1, 5]][None])
        self.assertEqual(list(book.parameters()), [])

    def test_planner_updates_only_planner_and_length(self):
        self.model.configure_recovery('planner')
        losses = self.model.losses(self.batch)
        losses['total'].backward()
        self.assertTrue(all(torch.isfinite(v) for v in losses.values()))
        self.assertGreater(float(self.model.semantic_planner.denoiser.output[-1].weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in self.model.codec_generator.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.encoder.parameters()))

    def test_acoustic_uses_units_and_keeps_planner_frozen(self):
        self.model.configure_recovery('acoustic', teacher_weight=0.)
        values, mask = self.model.target(self.batch, 'semantic', 768)
        normalized = (values - self.model.semantic_mean) / self.model.semantic_std
        torch.testing.assert_close(normalized[mask], self.model.semantic_planner.codebook(normalized)[mask])
        self.model.losses(self.batch)['total'].backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in self.model.codec_generator.parameters() if p.grad is not None), 0)
        self.assertTrue(all(p.grad is None for p in self.model.semantic_planner.parameters()))

    def test_all_masked_prediction_uses_a_context(self):
        self.model.eval()
        planner = self.model.semantic_planner
        mask = torch.tensor([[True, True, False], [True, True, True]])
        ids = torch.zeros_like(mask, dtype=torch.long)
        memory = torch.randn(2, 4, 512)
        memory_mask = torch.ones(2, 4, dtype=torch.bool)
        affect, style = torch.randn(2, 4, 6), torch.randn(2, 32)
        first = planner.logits(ids, mask, mask, memory, memory_mask, affect, style)
        second = planner.logits(ids, mask, mask, memory + 10, memory_mask, affect, style)
        self.assertGreater(float((first - second).abs().sum()), 0)
        result = planner.sample(mask, memory, memory_mask, affect, style, 4)
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(float(result[~mask].abs().sum()), 0.)
        torch.testing.assert_close(result[mask], planner.codebook(result)[mask], atol=1e-6, rtol=1e-6)

    def test_a_only_and_checkpoint_roundtrip(self):
        self.model.eval()
        first = self.model.generate_batch(person_a_only(self.batch), FakeCodec())
        self.batch['semantic'].fill_(999)
        second = self.model.generate_batch(person_a_only(self.batch), FakeCodec())
        torch.testing.assert_close(first['waveform'], second['waveform'], rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'units.pt'
            torch.save(self.model.checkpoint(), path)
            loaded, _ = UnitSpeechSystem.from_checkpoint(path)
            result = loaded.generate_batch(person_a_only(self.batch), FakeCodec())
            torch.testing.assert_close(first['waveform'], result['waveform'], rtol=0, atol=0)
            generic, _ = load_response_model(path)
            result = generic.generate_batch(person_a_only(self.batch), FakeCodec())
            torch.testing.assert_close(first['waveform'], result['waveform'], rtol=0, atol=0)

    def test_initialization_preserves_other_modules_and_rejects_wrong_stats(self):
        base = QualitySpeechSystem(config())
        payload = base.checkpoint()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'codebook.pt'
            book = {'architecture': 'bk_speech_codebook_v1', 'centers': self.centers,
                    'semantic_mean': base.semantic_mean, 'semantic_std': base.semantic_std}
            torch.save(book, path)
            model = initialize_unit_system(base.config, payload, path)
            for key, value in base.state_dict().items():
                if not key.startswith('semantic_planner.'):
                    torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
            book['semantic_mean'] = base.semantic_mean + 1
            torch.save(book, path)
            with self.assertRaisesRegex(ValueError, 'normalization differs'):
                load_codebook(path, base)


if __name__ == '__main__':
    unittest.main()
