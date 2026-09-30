"""A-only waveform boundaries and planner gradients. / A 전용 경계와 계획기 기울기 검증."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F

from dataset.quality_speech_dataset import collate_quality
from model.full_speech.planner_waveform import (
    frozen_content_ctc, planner_a_item, sampled_planner_waveform,
)
from model.full_speech.quality import text_ids
from model.full_speech.units import SpeechCodebook, UnitSpeechSystem
from scripts.check_planner_waveform_gradient import forward_parity
from tests.test_quality_speech import config, sample


class FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv1d(128, 1, 1)

    def forward(self, values):
        return self.projection(values).repeat_interleave(640, dim=-1)


class FakeCodec(nn.Module):
    sample_rate = 32000

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.decoder = FakeDecoder()
        self.requires_grad_(False).eval()

    @torch.no_grad()
    def decode(self, values, lengths, sample_lengths):
        wave = self.model.decoder(values.transpose(1, 2))[:, 0]
        return wave[:, :int(sample_lengths[0])], sample_lengths


class Tokenizer:
    unk_token_id = 3

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[{'A': 1, 'B': 2}.get(char, 3) for char in text])


class FakeTeacher(nn.Module):
    def __init__(self, fixed_frames=None):
        super().__init__()
        self.projection = nn.Linear(1, 4)
        self.processor = SimpleNamespace(tokenizer=Tokenizer())
        self.recognizer = SimpleNamespace(config=SimpleNamespace(pad_token_id=0))
        self.fixed_frames = fixed_frames
        self.requires_grad_(False).eval()

    def waveform_logits(self, waveform):
        values = F.avg_pool1d(waveform[None, None], 640, 640).transpose(1, 2)
        if self.fixed_frames is not None:
            values = values[:, :self.fixed_frames]
        return self.projection(values)


class AOnlyReads(dict):
    def __getitem__(self, key):
        if key not in ('mel', 'dmm', 'au', 'mel_len', 'dmm_len', 'au_len',
                       'speech_a', 'speech_a_len', 'style_id', 'speaker_id'):
            raise AssertionError('B metadata was read: ' + key)
        return super().__getitem__(key)


class PlannerWaveformTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        cfg = replace(config(), dropout=.15, semantic_steps=8, predicted_semantic_steps=8, codec_steps=32)
        self.model = UnitSpeechSystem(cfg, torch.randn(8, 768))
        self.model.configure_recovery('planner', teacher_weight=0.)
        self.model.train()
        self.batch = collate_quality([sample(0), sample(1)])
        self.codec = FakeCodec()

    def test_forward_parity_and_mode_restore(self):
        before = [module.training for module in self.model.modules()]
        with (patch.object(self.model.semantic_planner, 'sample', wraps=self.model.semantic_planner.sample) as planner,
              patch.object(self.model.codec_generator, 'sample', wraps=self.model.codec_generator.sample) as acoustic):
            result = sampled_planner_waveform(self.model, AOnlyReads(self.batch), self.codec, index=1)
        self.assertEqual(before, [module.training for module in self.model.modules()])
        self.assertEqual(planner.call_args.kwargs['steps'], 8)
        self.assertTrue(planner.call_args.kwargs['last_step_grad'])
        self.assertEqual(acoustic.call_args.args[-2], 32)
        self.assertTrue(acoustic.call_args.kwargs['checkpoint_grad'])
        self.model.eval()
        normal = self.model.generate_batch(planner_a_item(self.batch, 1), self.codec, seed=42)
        torch.testing.assert_close(result['waveform'], normal['waveform'][0], rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(result['duration'], normal['duration'], rtol=0, atol=0)
        self.assertEqual(result['audio_samples'], int(normal['audio_lengths'][0]))
        parity = forward_parity(self.model, normal, result)
        self.assertTrue(parity['passed'], parity['failure_reasons'])
        self.assertTrue(parity['semantic_units']['exact_agreement'])

    def test_only_planner_gets_ctc_gradients_and_b_cannot_change_audio(self):
        first = sampled_planner_waveform(self.model, AOnlyReads(self.batch), self.codec)
        teacher = FakeTeacher()
        scored = frozen_content_ctc(teacher, first['waveform'], text_ids('ab'))
        scored['loss'].backward()
        gradients = [(name, parameter.grad) for name, parameter in self.model.named_parameters()
                     if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(name.startswith('semantic_planner.') for name, _ in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for _, gradient in gradients))
        self.assertGreater(sum(float(gradient.abs().sum()) for _, gradient in gradients), 0.)
        self.assertTrue(all(parameter.grad is None for parameter in self.codec.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))
        for key in ('semantic', 'codec', 'affect', 'waveform', 'duration', 'text_b', 'text_a'):
            self.batch[key].fill_(999)
        second = sampled_planner_waveform(self.model, AOnlyReads(self.batch), self.codec)
        torch.testing.assert_close(first['waveform'], second['waveform'], rtol=0, atol=0)
        self.assertEqual(scored['predicted_audio_samples'], first['audio_samples'])

    def test_mode_restore_on_error_and_strict_acoustic_freeze(self):
        modes = [module.training for module in self.model.modules()]
        with patch.object(self.codec.model.decoder, 'forward', side_effect=RuntimeError('decode failed')):
            with self.assertRaisesRegex(RuntimeError, 'decode failed'):
                sampled_planner_waveform(self.model, self.batch, self.codec)
        self.assertEqual(modes, [module.training for module in self.model.modules()])
        self.model.codec_generator.train()
        with self.assertRaisesRegex(ValueError, 'eval mode through backward'):
            sampled_planner_waveform(self.model, self.batch, self.codec)
        self.model.codec_generator.eval().requires_grad_(True)
        with self.assertRaisesRegex(ValueError, 'frozen parameters'):
            sampled_planner_waveform(self.model, self.batch, self.codec)

    def test_rejects_inference_mode_bad_labels_and_infeasible_ctc(self):
        with torch.inference_mode(), self.assertRaisesRegex(RuntimeError, 'inference_mode'):
            sampled_planner_waveform(self.model, self.batch, self.codec)
        waveform = torch.ones(3200, requires_grad=True)
        teacher = FakeTeacher(fixed_frames=2)
        with self.assertRaisesRegex(ValueError, 'requires 3 frames'):
            frozen_content_ctc(teacher, waveform, text_ids('aa'))
        with self.assertRaisesRegex(ValueError, 'supported transcript'):
            frozen_content_ctc(teacher, waveform, torch.tensor([0]))
        with self.assertRaisesRegex(ValueError, 'cannot represent'):
            frozen_content_ctc(teacher, waveform, text_ids('hi'))
        teacher.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, 'frozen parameters'):
            frozen_content_ctc(teacher, waveform, text_ids('a'))

    def test_rejects_nonfinite_evaluator_and_waveform(self):
        teacher = FakeTeacher()
        waveform = torch.ones(3200, requires_grad=True)
        with patch.object(teacher, 'waveform_logits', return_value=torch.full((1, 5, 4), float('nan'))):
            with self.assertRaisesRegex(ValueError, 'invalid logits'):
                frozen_content_ctc(teacher, waveform, text_ids('ab'))
        with self.assertRaisesRegex(ValueError, 'finite nonempty waveform'):
            frozen_content_ctc(teacher, waveform * float('inf'), text_ids('a'))
        with patch.object(self.codec.model.decoder, 'forward',
                          return_value=torch.full((1, 1, 64000), float('nan'))):
            with self.assertRaisesRegex(RuntimeError, 'Non-finite sampled'):
                sampled_planner_waveform(self.model, self.batch, self.codec)


class PlannerParityGateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.book = SpeechCodebook(torch.stack((torch.zeros(768), torch.ones(768), -torch.ones(768))))
        self.model = SimpleNamespace(semantic_planner=SimpleNamespace(codebook=self.book))
        semantic = self.book.centers[torch.tensor([[1, 2]])]
        self.normal = {'waveform': torch.ones(1, 1000), 'audio_lengths': torch.tensor([1000]),
                       'duration': torch.tensor([.03125]), 'semantic': semantic,
                       'codec_latents': torch.ones(1, 2, 128)}
        self.sampled = {'waveform': self.normal['waveform'][0].clone(),
                        'duration': self.normal['duration'].clone(), 'audio_samples': 1000,
                        'semantic_frames': 2, 'codec_frames': 2,
                        '_semantic': semantic.clone(), '_codec_latents': self.normal['codec_latents'].clone()}

    def test_accepts_small_decoder_drift_only_with_exact_upstream_identity(self):
        self.sampled['waveform'] += .0006
        result = forward_parity(self.model, self.normal, self.sampled)
        self.assertTrue(result['passed'])
        self.assertTrue(result['gates']['exact_semantic_ids'])
        self.assertGreater(result['waveform']['maximum_absolute_error'], .0005)
        self.assertEqual(result['codec_latents']['maximum_absolute_error'], 0.)

    def test_rejects_peak_error_even_when_relative_rms_is_small(self):
        self.sampled['waveform'][0] += .002
        result = forward_parity(self.model, self.normal, self.sampled)
        self.assertFalse(result['passed'])
        self.assertLess(result['waveform']['relative_rms_error'], .001)
        self.assertFalse(result['gates']['waveform_numeric'])

    def test_rejects_relative_rms_error_even_when_peak_is_small(self):
        self.normal['waveform'].fill_(.1)
        self.sampled['waveform'].fill_(.1002)
        result = forward_parity(self.model, self.normal, self.sampled)
        self.assertFalse(result['passed'])
        self.assertLess(result['waveform']['maximum_absolute_error'], .001)
        self.assertFalse(result['gates']['waveform_numeric'])
        self.normal['waveform'].zero_()
        self.sampled['waveform'].fill_(1e-7)
        result = forward_parity(self.model, self.normal, self.sampled)
        self.assertFalse(result['passed'])
        self.assertEqual(result['waveform']['rms_denominator_floor'], 1e-8)

    def test_rejects_semantic_change_despite_identical_audio(self):
        self.sampled['_semantic'][0, 0] = self.book.centers[2]
        result = forward_parity(self.model, self.normal, self.sampled)
        self.assertFalse(result['passed'])
        self.assertFalse(result['gates']['exact_semantic_ids'])
        self.assertEqual(result['semantic_units']['different_positions'], 1)
        self.assertTrue(result['gates']['waveform_numeric'])

    def test_rejects_upstream_latent_or_duration_changes(self):
        self.sampled['_codec_latents'] += .0001
        result = forward_parity(self.model, self.normal, self.sampled)
        self.assertFalse(result['passed'])
        self.assertFalse(result['gates']['codec_latent_numeric'])
        self.sampled['_codec_latents'] = self.normal['codec_latents'].clone()
        self.sampled['duration'] += 1e-7
        result = forward_parity(self.model, self.normal, self.sampled)
        self.assertFalse(result['passed'])
        self.assertFalse(result['gates']['exact_timing'])
        self.assertTrue(result['gates']['waveform_numeric'])


if __name__ == '__main__':
    unittest.main()
