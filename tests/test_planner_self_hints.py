"""Disjoint holdout and generated-hint invariants. / 분리된 보류 타깃과 생성 힌트 검증."""

from copy import deepcopy
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from dataset.quality_speech_dataset import collate_quality
from model.full_speech.units import UnitSpeechSystem
from scripts.check_planner_hints import prepare_pair
from scripts.check_planner_self_hints import disjoint_masks, evaluate_self_hints, summarize
from tests.test_quality_speech import config, sample


class RecordingPlanner(nn.Module):
    def __init__(self, constant=None):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.))
        self.codebook = SimpleNamespace(centers=torch.zeros(8, 768))
        self.calls = []
        self.constant = constant
        self.eval()

    def logits(self, ids, hidden, mask, **kwargs):
        self.calls.append((ids.clone(), hidden.clone(), torch.is_grad_enabled()))
        value = (ids.sum(1) % 8) if self.constant is None else torch.full((len(ids),), self.constant)
        logits = torch.zeros(*ids.shape, 8) + self.weight
        return logits.scatter(-1, value[:, None, None].expand(*ids.shape, 1), 5.)


class PlannerSelfHintTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.targets = (torch.arange(20).reshape(1, -1) % 7 + 1).long()
        self.mask = torch.ones_like(self.targets, dtype=torch.bool)
        self.memory = {'context': torch.zeros(1, 1, 512), 'context_mask': torch.ones(1, 1, dtype=torch.bool),
                       'affect': torch.zeros(1, 1, 6)}
        self.style = torch.zeros(1, 32)

    def test_masks_disjoint_reproducible_and_contiguous_only_for_H(self):
        mask = torch.arange(20)[None] < torch.tensor([20, 13])[:, None]
        for kind in ('random', 'contiguous'):
            h, c = disjoint_masks(mask, 42, kind)
            again = disjoint_masks(mask, 42, kind)
            self.assertTrue(torch.equal(h, again[0]) and torch.equal(c, again[1]))
            self.assertFalse((h & c).any())
            self.assertFalse(((h | c) & ~mask).any())
            self.assertEqual(h.sum(1).tolist(), [5, 4])
            self.assertEqual(c.sum(1).tolist(), [5, 4])
            self.assertTrue((mask & ~(h | c)).any(1).all())
            if kind == 'contiguous':
                for row in h:
                    positions = row.nonzero(as_tuple=True)[0]
                    self.assertTrue(torch.equal(positions[1:] - positions[:-1], torch.ones(len(positions) - 1)))

    def test_H_is_never_read_and_C_is_hidden_in_first_pass(self):
        h, c = disjoint_masks(self.mask, 42, 'random')
        planner = RecordingPlanner()
        first = evaluate_self_hints(planner, self.targets, self.mask, h, c, self.memory, self.style)
        self.assertEqual(len(planner.calls), 3)
        for ids, hidden, grad_enabled in planner.calls:
            self.assertFalse(grad_enabled)
            self.assertTrue(hidden[h].all())
            self.assertFalse(ids[h].any())
            self.assertFalse(ids[hidden].any())
        self.assertTrue(planner.calls[0][1][c].all())
        self.assertFalse(planner.calls[0][0][c].any())
        perturbed = self.targets.clone()
        perturbed[h] = (perturbed[h] + 3) % 8
        second = evaluate_self_hints(planner, perturbed, self.mask, h, c, self.memory, self.style)
        for key in ('predicted_C_ids', 'predicted_H_with_true_C_ids', 'predicted_H_with_predicted_C_ids',
                    'first_input_sha256', 'true_C_input_sha256', 'predicted_C_input_sha256'):
            self.assertEqual(first[key], second[key])
        self.assertIsNone(planner.weight.grad)
        self.assertFalse(first['grad_enabled_during_evaluation'])

    def test_C_target_change_does_not_leak_into_self_generated_hints(self):
        h, c = disjoint_masks(self.mask, 43, 'contiguous')
        planner = RecordingPlanner()
        first = evaluate_self_hints(planner, self.targets, self.mask, h, c, self.memory, self.style)
        perturbed = self.targets.clone()
        perturbed[c] = (perturbed[c] + 2) % 8
        second = evaluate_self_hints(planner, perturbed, self.mask, h, c, self.memory, self.style)
        for key in ('predicted_C_ids', 'predicted_H_with_predicted_C_ids',
                    'first_input_sha256', 'predicted_C_input_sha256'):
            self.assertEqual(first[key], second[key])

    def test_correct_generated_C_has_no_H_difference(self):
        targets = torch.ones_like(self.targets)
        h, c = disjoint_masks(self.mask, 42, 'random')
        result = evaluate_self_hints(RecordingPlanner(constant=1), targets, self.mask, h, c, self.memory, self.style)
        self.assertTrue(result['all_C_predictions_correct'])
        self.assertEqual(result['C_error_count'], 0)
        self.assertEqual(result['H_accuracy_drop'], 0.)
        self.assertEqual(result['H_ce_increase'], 0.)
        self.assertEqual(result['H_prediction_changed_fraction'], 0.)

    def test_invalid_masks_and_train_mode_rejected(self):
        with self.assertRaisesRegex(ValueError, 'four valid frames'):
            disjoint_masks(torch.ones(1, 3, dtype=torch.bool), 42, 'random')
        with self.assertRaisesRegex(ValueError, 'prefix'):
            disjoint_masks(torch.tensor([[True, True, False, True, True]]), 42, 'random')
        h, c = disjoint_masks(self.mask, 42, 'random')
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            evaluate_self_hints(RecordingPlanner(), self.targets, self.mask, h, h, self.memory, self.style)
        with self.assertRaisesRegex(ValueError, 'eval mode'):
            evaluate_self_hints(RecordingPlanner().train(), self.targets, self.mask, h, c, self.memory, self.style)

    def test_summary_combines_mask_seeds_within_conversation(self):
        h, c = disjoint_masks(self.mask, 42, 'random')
        result = evaluate_self_hints(RecordingPlanner(), self.targets, self.mask, h, c, self.memory, self.style)
        trial = {'mask_type': 'random', 'conditions': {'correct_a': result}}
        report = summarize([{'conversation_id': 'one', 'trials': [trial, deepcopy(trial)]}])['random/correct_a']
        self.assertEqual(report['measurements'], 2)
        self.assertEqual(report['conversations'], 1)
        self.assertAlmostEqual(report['mean_conversation_H_accuracy_drop'], result['H_accuracy_drop'])

    def test_tiny_real_model_uses_existing_memory_and_no_gradients(self):
        model = UnitSpeechSystem(config(), torch.randn(8, 768)).eval()
        batch, donor = collate_quality([sample(0)]), collate_quality([sample(1)])
        targets, mask, contexts, style = prepare_pair(model, batch, donor)
        h, c = disjoint_masks(mask, 42, 'contiguous')
        for memory in contexts.values():
            result = evaluate_self_hints(model.semantic_planner, targets, mask, h, c, memory, style)
            self.assertEqual(len(result['predicted_C_ids']), int(c.sum()))
            self.assertEqual(len(result['predicted_H_with_predicted_C_ids']), int(h.sum()))
            self.assertFalse(result['grad_enabled_during_evaluation'])
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))


if __name__ == '__main__':
    unittest.main()
