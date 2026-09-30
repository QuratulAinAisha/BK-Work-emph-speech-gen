"""Recovery invariants and gradient checks. / 복구 조건과 기울기를 검증합니다."""

import unittest
import torch
from torch import nn

from tests.test_quality_speech import config, sample, FakeCodec
from dataset.quality_speech_dataset import collate_quality
from model.full_speech.recovery import RecoverySpeechSystem, downsample_speech
from scripts.run_recovery import acoustic_gate, memorization_gate


class FakeTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.), requires_grad=False)

    def forward(self, waveform, labels):
        return ((downsample_speech(waveform) * self.scale) - .2).square().mean()


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model = RecoverySpeechSystem(config())
        self.batch = collate_quality([sample(0), sample(1)])

    def test_validation_condition_fixed_and_reference_anchor(self):
        self.model.configure_recovery('joint', .2)
        self.model.train(); self.assertEqual(self.model.predicted_fraction(), .2)
        self.model.eval(); self.assertEqual(self.model.predicted_fraction(), 1.)
        self.model.training_fraction = .6
        self.assertEqual(self.model.predicted_fraction(), 1.)
        with self.assertRaises(ValueError):
            self.model.configure_recovery('joint', 1.)

    def test_full_time_range_and_matched_sampling(self):
        torch.manual_seed(42)
        values = self.model.codec_times(10000, 'cpu')
        self.assertLess(float(values.min()), .01)
        self.assertGreater(float(values.max()), .99)
        self.model.config.semantic_steps += 1
        with self.assertRaisesRegex(ValueError, 'budgets must match'):
            self.model.configure_recovery('acoustic')

    def test_acoustic_gradients_and_frozen_context(self):
        teacher = FakeTeacher()
        self.model.configure_recovery('acoustic', teacher=teacher)
        object.__setattr__(self.model, '_acoustic_codec', FakeCodec())
        losses = self.model.losses(self.batch)
        losses['waveform_ctc'].backward()
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.encoder.parameters()))
        self.assertFalse(self.model.encoder.training)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in self.model.codec_generator.parameters() if p.grad is not None), 0)

    def test_planner_updates_with_frozen_acoustics(self):
        self.model.configure_recovery('planner')
        values = self.model.losses(self.batch)
        values['total'].backward()
        self.assertTrue(all(p.grad is None for p in self.model.codec_generator.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.speech_projection.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in self.model.semantic_planner.parameters() if p.grad is not None), 0)

    def test_resampling_rejects_high_frequency_and_preserves_gradient(self):
        time = torch.arange(32000).float() / 32000
        low = torch.sin(2 * torch.pi * 1000 * time).requires_grad_()
        high = torch.sin(2 * torch.pi * 12000 * time)
        first, second = downsample_speech(low), downsample_speech(high)
        self.assertEqual(len(first), 16000)
        self.assertGreater(float(first.detach().square().mean()), .45)
        self.assertLess(float(second[40:-40].square().mean()), .001)
        first.square().mean().backward()
        self.assertGreater(float(low.grad.abs().sum()), 0)

    def test_gates_require_controls_and_reconstruction(self):
        def report(ref, acoustic, full):
            return {'examples': [{'paths': {'reference': {'reference_wer': ref},
                'oracle_semantics': {'reference_wer': acoustic},
                'predicted_length_steps_8': {'reference_wer': full}}} for _ in range(8)]}
        self.assertTrue(acoustic_gate(report(.05, .10, 1.))['passed'])
        self.assertFalse(acoustic_gate(report(.05, .4, 0.))['passed'])
        self.assertFalse(acoustic_gate(report(.5, .1, 0.))['passed'])
        self.assertFalse(memorization_gate(report(.05, .1, 1.))['passed'])
        self.assertTrue(memorization_gate(report(.05, .1, .1))['passed'])


if __name__ == '__main__':
    unittest.main()
