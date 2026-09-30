"""Fixed-input attention path attribution. / 고정 입력 어텐션 경로 분리 검증."""

from dataclasses import replace
import unittest

import torch

from dataset.quality_speech_dataset import collate_quality
from model.full_speech.planner_waveform import planner_a_item, sampled_planner_waveform
from model.full_speech.units import UnitSpeechSystem
from scripts.calibrate_planner_waveform_parity import acoustic_replay, attention_path, calibrate_acoustic_paths
from tests.test_planner_waveform import FakeCodec
from tests.test_quality_speech import config, sample


class PlannerParityCalibrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(72)
        cfg = replace(config(), semantic_steps=8, predicted_semantic_steps=8, codec_steps=32)
        self.model = UnitSpeechSystem(cfg, torch.randn(8, 768))
        self.model.configure_recovery('planner', teacher_weight=0.)
        self.model.length_predictor.requires_grad_(False)
        self.model.trainable_components = ['semantic_planner']
        self.batch = collate_quality([sample(0)])
        self.codec = FakeCodec()
        self.model.eval()
        self.normal = self.model.generate_batch(planner_a_item(self.batch), self.codec)
        self.model.train()
        self.sampled = sampled_planner_waveform(self.model, self.batch, self.codec)
        with torch.no_grad():
            style, speaker = self.model.embeddings(self.batch['style_id'], self.batch['speaker_id'], 1)
            self.condition = style + speaker

    def test_matched_noise_exact_replays_and_math_path(self):
        result = calibrate_acoustic_paths(self.model, self.normal, self.sampled, self.condition, profile=False)
        self.assertTrue(result['passed'], result['failure_reasons'])
        self.assertEqual(len({run['noise_sha256'] for run in result['runs'].values()}), 1)
        self.assertTrue(result['checks']['normal_matches_inference_replay_exact'])
        self.assertTrue(result['checks']['helper_matches_gradient_replay_exact'])
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))

    def test_attention_operator_names_are_recorded(self):
        fixed = {name: self.normal[name].detach().clone() for name in ('semantic', 'context', 'context_mask', 'affect')}
        fixed.update(codec_frames=self.normal['codec_latents'].shape[1], style_speaker=self.condition)
        result = acoustic_replay(self.model, fixed, 42, grad_enabled=False, profile=True)
        self.assertTrue(result['attention_operators'])

    def test_rejects_nonmatching_latents_even_when_other_paths_replay(self):
        tampered = dict(self.sampled)
        tampered['_codec_latents'] = self.sampled['_codec_latents'] + .01
        result = calibrate_acoustic_paths(self.model, self.normal, tampered, self.condition, profile=False)
        self.assertFalse(result['passed'])
        self.assertFalse(result['checks']['helper_matches_gradient_replay_exact'])

    def test_backend_state_restored_even_after_failure(self):
        before = torch.backends.mha.get_fastpath_enabled()
        with self.assertRaisesRegex(RuntimeError, 'test'):
            with attention_path(math_only=True):
                self.assertFalse(torch.backends.mha.get_fastpath_enabled())
                raise RuntimeError('test')
        self.assertEqual(torch.backends.mha.get_fastpath_enabled(), before)


if __name__ == '__main__':
    unittest.main()
