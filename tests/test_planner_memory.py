"""Isolated speech-memory ablation contracts. / 음성 문맥 비교의 분리 조건 검증."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from dataset.quality_speech_dataset import collate_quality, person_a_only
from model.full_speech.loading import load_response_model
from model.full_speech.quality import QualityConfig
from model.full_speech.tensor_ops import align
from model.full_speech.units import UnitSpeechSystem, initialize_unit_system
from tests.test_quality_speech import config, sample, FakeCodec


class PlannerMemoryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(73)
        self.model = UnitSpeechSystem(config(), torch.randn(8, 768)).eval()
        self.batch = collate_quality([sample(0), sample(1)])

    def encoded_fixture(self):
        speech_mask = torch.arange(7)[None] < torch.tensor([7, 5])[:, None]
        context_mask = torch.arange(3)[None] < torch.tensor([3, 2])[:, None]
        return {'speech_hidden': torch.randn(2, 7, 512), 'speech_mask': speech_mask,
                'context': torch.randn(2, 3, 512), 'context_mask': context_mask,
                'affect': torch.randn(2, 3, 6)}

    def test_native_and_resampled_lengths_match_exact_production_roundtrip(self):
        encoded = self.encoded_fixture()
        before = {key: value.clone() for key, value in encoded.items()}
        self.model.config.planner_memory_mode = 'native_speech'
        native = self.model.planner_inputs(encoded)
        self.model.config.planner_memory_mode = 'resampled_speech'
        resampled = self.model.planner_inputs(encoded)
        self.assertEqual(native['context'].shape, resampled['context'].shape)
        torch.testing.assert_close(native['context_mask'], resampled['context_mask'], rtol=0, atol=0)
        expected = align(align(encoded['speech_hidden'], encoded['speech_mask'], encoded['context_mask']),
                         encoded['context_mask'], encoded['speech_mask'])
        torch.testing.assert_close(resampled['context'], expected, rtol=0, atol=0)
        torch.testing.assert_close(native['context'][encoded['speech_mask']],
                                   encoded['speech_hidden'][encoded['speech_mask']], rtol=0, atol=0)
        self.assertTrue((native['context'][~encoded['speech_mask']] == 0).all())
        for inputs in (native, resampled):
            self.assertIs(inputs['affect'], encoded['affect'])
            self.assertIs(inputs['affect_mask'], encoded['context_mask'])
        for key, value in encoded.items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_memory_mode_does_not_resample_the_affect_condition(self):
        encoded = self.encoded_fixture()
        target_mask = torch.arange(9)[None] < torch.tensor([9, 6])[:, None]
        ids = torch.zeros_like(target_mask, dtype=torch.long)
        style = torch.zeros(2, self.model.config.hidden_dim)
        expected = align(encoded['affect'], encoded['context_mask'], target_mask)
        for mode in ('fused', 'native_speech', 'resampled_speech'):
            self.model.config.planner_memory_mode = mode
            with patch.object(self.model.semantic_planner.denoiser, 'forward',
                              wraps=self.model.semantic_planner.denoiser.forward) as spy:
                self.model.semantic_planner.logits(ids, target_mask, target_mask, style=style,
                                                   **self.model.planner_inputs(encoded))
            torch.testing.assert_close(spy.call_args.args[2], expected, rtol=0, atol=0)

    def test_legacy_fused_generation_is_exact_and_legacy_checkpoint_loads(self):
        inputs = person_a_only(self.batch)
        actual = self.model.generate_batch(inputs, FakeCodec(), seed=19)
        # The old planner received only these three encoded values. / 기존 계획기는 이 세 인코딩 값만 받았습니다.
        def legacy(encoded):
            return {'context': encoded['context'], 'context_mask': encoded['context_mask'],
                    'affect': encoded['affect']}
        with patch.object(self.model, 'planner_inputs', side_effect=legacy):
            expected = self.model.generate_batch(inputs, FakeCodec(), seed=19)
        torch.testing.assert_close(actual['waveform'], expected['waveform'], rtol=0, atol=0)
        torch.testing.assert_close(actual['semantic'], expected['semantic'], rtol=0, atol=0)
        payload = self.model.checkpoint()
        payload['config'].pop('planner_memory_mode')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'legacy.pt'
            torch.save(payload, path)
            for loader in (UnitSpeechSystem.from_checkpoint, load_response_model):
                model, _ = loader(path)
                self.assertEqual(model.config.planner_memory_mode, 'fused')
                result = model.generate_batch(inputs, FakeCodec(), seed=19)
                torch.testing.assert_close(actual['waveform'], result['waveform'], rtol=0, atol=0)

    def test_memory_choice_persists_without_new_parameters(self):
        original = self.model.state_dict()
        with tempfile.TemporaryDirectory() as folder:
            for mode in ('native_speech', 'resampled_speech'):
                self.model.config.planner_memory_mode = mode
                path = Path(folder) / (mode + '.pt')
                torch.save(self.model.checkpoint(), path)
                for loader in (UnitSpeechSystem.from_checkpoint, load_response_model):
                    loaded, _ = loader(path)
                    self.assertEqual(loaded.config.planner_memory_mode, mode)
                    self.assertEqual(set(loaded.state_dict()), set(original))
                    for key, value in loaded.state_dict().items():
                        torch.testing.assert_close(value, original[key], rtol=0, atol=0)
        payload = self.model.checkpoint()
        switched = initialize_unit_system(replace(self.model.config, planner_memory_mode='native_speech'), payload)
        self.assertEqual(switched.config.planner_memory_mode, 'native_speech')
        with self.assertRaisesRegex(ValueError, 'memory mode'):
            QualityConfig(planner_memory_mode='unknown')

    def test_oracle_acoustics_affect_and_duration_inputs_remain_identical(self):
        inputs = person_a_only(self.batch)
        oracle = (self.batch['semantic'] - self.model.semantic_mean) / self.model.semantic_std
        first = None
        for mode in ('fused', 'native_speech', 'resampled_speech'):
            self.model.config.planner_memory_mode = mode
            with patch.object(self.model.length_predictor, 'forward',
                              wraps=self.model.length_predictor.forward) as length:
                result = self.model.generate_batch(inputs, FakeCodec(), seed=29,
                    oracle_semantic=oracle, oracle_duration=self.batch['duration'])
            encoded_inputs = length.call_args.args[:3]
            if first is None:
                first = (result, encoded_inputs)
            for name in ('context', 'affect', 'codec_latents', 'waveform'):
                torch.testing.assert_close(result[name], first[0][name], rtol=0, atol=0)
            for value, expected in zip(encoded_inputs, first[1]):
                torch.testing.assert_close(value, expected, rtol=0, atol=0)

    def test_planner_training_uses_selected_memory_with_frozen_other_modules(self):
        self.model.configure_recovery('planner')
        self.model.length_predictor.requires_grad_(False)
        self.model.trainable_components = ['semantic_planner']
        self.model.train()
        for mode in ('native_speech', 'resampled_speech'):
            self.model.config.planner_memory_mode = mode
            self.model.zero_grad(set_to_none=True)
            with patch.object(self.model.semantic_planner, 'estimate',
                              wraps=self.model.semantic_planner.estimate) as planner:
                losses = self.model.losses(self.batch)
            losses['total'].backward()
            self.assertEqual(planner.call_args.kwargs['context'].shape[1], self.batch['speech_a'].shape[1])
            torch.testing.assert_close(planner.call_args.kwargs['context_mask'].sum(1), self.batch['speech_a_len'])
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                for p in self.model.semantic_planner.parameters()))
            self.assertTrue(all(p.grad is None for name, p in self.model.named_parameters()
                                if not name.startswith('semantic_planner.')))

    def test_joint_and_acoustic_paths_use_memory_helper_consistently(self):
        for phase in ('acoustic', 'joint'):
            self.model.configure_recovery(phase, teacher_weight=0.)
            self.model.config.planner_memory_mode = 'native_speech'
            with patch.object(self.model.semantic_planner, 'estimate',
                              wraps=self.model.semantic_planner.estimate) as planner:
                losses = self.model.losses(self.batch)
            self.assertTrue(torch.isfinite(losses['total']))
            self.assertEqual(planner.call_args.kwargs['context'].shape[1], self.batch['speech_a'].shape[1])


if __name__ == '__main__':
    unittest.main()
