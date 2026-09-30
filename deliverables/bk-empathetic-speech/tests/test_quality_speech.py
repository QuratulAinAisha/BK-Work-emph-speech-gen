"""Content gradients and inference separation. / 내용 기울기와 추론 분리 검증."""

import tempfile
from pathlib import Path
import unittest

import torch
from torch import nn

from model.full_speech.quality import QualityConfig, QualitySpeechSystem, content_ctc, text_ids, normalize_text
from dataset.quality_speech_dataset import collate_quality, person_a_only


def config():
    return QualityConfig(sbe={'app_num_layers': 1, 'app_mlp_dim': 128, 'app_max_len': 64},
        hidden_dim=32, num_heads=4, planner_layers=1, decoder_layers=1, affect_hidden_dim=32,
        affect_layers=1, dropout=0., semantic_steps=2, codec_steps=2, acoustic_epochs=1,
        semantic_epochs=1, mix_ramp_epochs=1, predicted_semantic_steps=2, max_duration=2.)


def sample(i=0):
    return {'mel': torch.randn(40, 80), 'dmm': torch.randn(12, 486), 'au': torch.randn(12, 25),
            'affect': torch.rand(8, 6), 'affect_weight': torch.tensor([0, 0, 1, 1, 1, 0]).expand(8, 6).float(),
            'semantic': torch.randn(16, 768), 'codec': torch.randn(16, 128),
            'speech_a': torch.randn(25, 768), 'duration': torch.tensor(.32),
            'style_id': torch.tensor(i), 'speaker_id': torch.tensor(i), 'waveform': torch.randn(10240) * .1,
            'text_a': text_ids('hi'), 'text_b': text_ids('hello'), 'conversation_key': torch.tensor(i)}


class FakeCodec:
    def __init__(self):
        self.model = self

    def decoder(self, values):
        return values.mean(1, keepdim=True).repeat_interleave(640, dim=-1).tanh()

    def decode(self, values, lengths, sample_lengths):
        result = self.decoder(values.transpose(1, 2))[:, 0]
        return result[:, :int(sample_lengths.max())], sample_lengths


class QualityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(3)
        self.model = QualitySpeechSystem(config())
        self.batch = collate_quality([sample(0), sample(1)])

    def test_curriculum_and_joint_content_gradients(self):
        for epoch, stage in [(0, 'acoustic'), (1, 'semantic'), (2, 'joint')]:
            self.model.current_epoch = epoch
            self.assertEqual(self.model.stage(), stage)
            losses = self.model.losses(self.batch)
            self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        self.model.zero_grad()
        losses['codec_flow'].backward()
        gradients = [p.grad for p in self.model.semantic_planner.parameters() if p.grad is not None]
        self.assertGreater(sum(float(value.abs().sum()) for value in gradients), 0)

    def test_input_ctc_reaches_speech_adapter(self):
        self.model.losses(self.batch)['input_ctc'].backward()
        self.assertGreater(float(self.model.speech_projection[1].weight.grad.abs().sum()), 0)

    def test_spectral_loss_reaches_generator(self):
        object.__setattr__(self.model, '_acoustic_codec', FakeCodec())
        losses = self.model.losses(self.batch)
        losses['spectral'].backward()
        gradients = [p.grad for p in self.model.codec_generator.parameters() if p.grad is not None]
        self.assertGreater(sum(float(value.abs().sum()) for value in gradients), 0)

    def test_inference_cannot_read_b_targets_and_roundtrip(self):
        self.model.eval()
        inputs = person_a_only(self.batch)
        self.assertFalse({'text_b', 'semantic', 'codec', 'duration', 'waveform'} & inputs.keys())
        first = self.model.generate_batch(inputs, FakeCodec(), seed=42)
        self.batch['semantic'].fill_(10000)
        second = self.model.generate_batch(person_a_only(self.batch), FakeCodec(), seed=42)
        torch.testing.assert_close(first['waveform'], second['waveform'], rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'quality.pt'
            torch.save(self.model.checkpoint(10), path)
            loaded, _ = QualitySpeechSystem.from_checkpoint(path)
            output = loaded.generate_batch(inputs, FakeCodec(), seed=42)
            torch.testing.assert_close(first['waveform'], output['waveform'], rtol=0, atol=0)

    def test_ctc_rejects_impossible_alignment(self):
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            content_ctc(self.model.semantic_content, torch.randn(1, 2, 768),
                        torch.ones(1, 2, dtype=torch.bool), text_ids('letter')[None], torch.tensor([6]))
        self.assertEqual(normalize_text('It’s 2026!'), "it's 2026")


if __name__ == '__main__':
    unittest.main()
