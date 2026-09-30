"""Speech-prior isolation and masking. / 음성 사전학습의 분리와 마스킹 검증."""

import unittest
from unittest.mock import patch

import torch

from dataset.quality_speech_dataset import collate_quality
from model.full_speech.units import UnitSpeechSystem
from tests.test_quality_speech import config, sample


class UnitPriorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(29)
        self.model = UnitSpeechSystem(config(), torch.randn(8, 768))
        self.batch = collate_quality([sample(0), sample(1)])

    def test_prior_uses_only_b_units(self):
        self.model.configure_unit_prior()
        target_only = {name: self.batch[name] for name in ('semantic', 'semantic_len')}
        # Missing A inputs must be safe during prior learning. / 사전학습은 A 입력 없이도 동작해야 합니다.
        with patch.object(self.model, 'encode_batch', side_effect=AssertionError('A input read')), \
             patch.object(self.model, 'embeddings', side_effect=AssertionError('Condition labels read')):
            torch.manual_seed(41)
            first = self.model.losses(target_only)['total']
            changed = {key: value.clone() for key, value in self.batch.items()}
            for name in ('mel', 'dmm', 'au', 'speech_a', 'duration'):
                changed[name].fill_(float('nan'))
            changed['style_id'].fill_(-100)
            changed['speaker_id'].fill_(-100)
            torch.manual_seed(41)
            second = self.model.losses(changed)['total']
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_prior_updates_only_unit_planner(self):
        self.model.configure_unit_prior()
        losses = self.model.losses(self.batch)
        losses['total'].backward()
        self.assertEqual(set(losses), {'unit_prior_ce', 'total'})
        self.assertTrue(torch.isfinite(losses['total']))
        self.assertEqual(self.model.trainable_components, ['semantic_planner'])
        self.assertFalse(self.model.length_predictor.training)
        changed = []
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                self.assertTrue(name.startswith('semantic_planner.'))
            if parameter.grad is not None and parameter.grad.abs().sum() > 0:
                changed.append(name)
                self.assertTrue(name.startswith('semantic_planner.'))
        self.assertTrue(changed)

    def test_training_masks_keep_visible_and_hidden_valid_units(self):
        planner = self.model.semantic_planner.train()
        lengths = torch.tensor([1, 2, 3, 5, 10])
        mask = torch.arange(10)[None] < lengths[:, None]
        args = (torch.randn(5, 10, 768), mask, torch.zeros(5, 1, 512),
                torch.ones(5, 1, dtype=torch.bool), torch.zeros(5, 1, 6), torch.zeros(5, 32))
        for _ in range(12):
            with patch.object(planner, 'logits', wraps=planner.logits) as spy:
                loss, _ = planner.estimate(*args, fully_masked_probability=0.,
                                           max_mask_ratio=.8, denoising_eval=True)
            hidden = spy.call_args.args[1]
            self.assertTrue(torch.isfinite(loss))
            self.assertFalse(hidden[~mask].any())
            self.assertEqual(int(hidden[0].sum()), 1)
            counts = hidden.sum(1)[1:]
            self.assertTrue((counts >= (.2 * lengths[1:]).ceil()).all())
            self.assertTrue((counts <= (.8 * lengths[1:]).floor()).all())

    def test_prior_validation_is_fixed_partial_masking(self):
        self.model.configure_unit_prior()
        self.model.eval()
        planner = self.model.semantic_planner
        with patch.object(planner, 'logits', wraps=planner.logits) as spy:
            torch.manual_seed(1)
            first = self.model.losses(self.batch)['total']
            torch.manual_seed(99)
            second = self.model.losses(self.batch)['total']
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        one, two = (call.args[1] for call in spy.call_args_list)
        torch.testing.assert_close(one, two)
        self.assertTrue(one.any(1).all())
        self.assertTrue((~one).any(1).all())

    def test_normal_planner_validation_remains_fully_masked(self):
        self.model.configure_unit_prior()
        self.model.configure_recovery('planner')
        self.assertFalse(self.model.unit_prior_training)
        self.assertTrue(all(p.requires_grad for p in self.model.length_predictor.parameters()))
        self.model.eval()
        planner = self.model.semantic_planner
        with patch.object(planner, 'logits', wraps=planner.logits) as spy:
            losses = self.model.losses(self.batch)
        hidden, mask = spy.call_args.args[1:3]
        torch.testing.assert_close(hidden, mask, rtol=0, atol=0)
        self.assertEqual(set(losses), {'unit_ce', 'duration', 'total'})


if __name__ == '__main__':
    unittest.main()
