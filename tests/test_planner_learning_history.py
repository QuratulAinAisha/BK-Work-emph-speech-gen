"""Check deterministic gaps and hint-only baselines. / 결정적 공백과 힌트 전용 기준을 검사합니다."""

import unittest
from types import SimpleNamespace

import torch

from scripts.check_planner_learning_history import (gap_mask, nearest_visible_ids, frequency_groups,
    target_strata, score_predictions, summarize, checkpoint_arguments, evaluate_model)


class PlannerLearningHistoryTests(unittest.TestCase):
    def test_random_and_contiguous_masks_are_seeded_and_length_exact(self):
        mask = torch.arange(12)[None] < torch.tensor([12, 7])[:, None]
        for kind in ('random', 'contiguous'):
            first = gap_mask(mask, .5, 42, kind)
            self.assertTrue(torch.equal(first, gap_mask(mask, .5, 42, kind)))
            self.assertEqual(first.sum(1).tolist(), [6, 4])
            self.assertFalse((first & ~mask).any())
            self.assertTrue(torch.equal(gap_mask(mask, 1., 42, kind), mask))
            if kind == 'contiguous':
                for row in first:
                    positions = row.nonzero(as_tuple=True)[0]
                    self.assertTrue(((positions[1:] - positions[:-1]) == 1).all())

    def test_copy_left_tie_edges_and_no_hidden_target_access(self):
        mask = torch.ones(1, 9, dtype=torch.bool)
        hidden = mask.clone()
        hidden[:, [2, 6]] = False
        initial = torch.zeros(1, 9, dtype=torch.long)
        initial[:, [2, 6]] = torch.tensor([8, 9])
        predicted = nearest_visible_ids(initial, hidden, mask)
        self.assertEqual(predicted[0].tolist(), [8, 8, 8, 8, 8, 9, 9, 9, 9])
        initial[0, 0] = 5
        with self.assertRaises(ValueError):
            nearest_visible_ids(initial, hidden, mask)
        with self.assertRaises(ValueError):
            nearest_visible_ids(torch.zeros_like(initial), mask, mask)

    def test_frequency_bins_use_stable_count_ranks_and_mark_unseen(self):
        groups = frequency_groups([10, 10, 8, 6, 4, 3, 2, 1, 0])
        self.assertEqual(groups.tolist(), [0, 0, 1, 1, 1, 1, 2, 2, 3])

    def test_run_and_frequency_strata_cover_only_valid_frames(self):
        target = torch.tensor([[1, 1, 1, 2, 3, 3, 0]])
        mask = torch.tensor([[True, True, True, True, True, True, False]])
        strata = target_strata(target, mask, torch.tensor([3, 0, 1, 2]))
        self.assertEqual(strata['run_interior'].nonzero()[:, 1].tolist(), [1])
        self.assertTrue(torch.equal(strata['run_interior'] | strata['run_boundary_or_singleton'], mask))
        self.assertEqual(strata['prefix_20pct_time'].sum(), 2)
        self.assertFalse(strata['frequency_unseen_in_training'].any())

    def test_visible_frames_never_inflate_hidden_accuracy(self):
        target = torch.tensor([[1, 1, 1, 2]])
        predicted = torch.tensor([[1, 0, 1, 2]])
        mask = torch.ones_like(target, dtype=torch.bool)
        hidden = torch.tensor([[False, True, False, False]])
        strata = target_strata(target, mask, torch.tensor([3, 0, 1]))
        score = score_predictions(predicted, target, hidden, strata)
        self.assertEqual(score['all_hidden']['hidden_frames'], 1)
        self.assertEqual(score['all_hidden']['hidden_only_accuracy'], 0.)
        self.assertIsNone(score['frequency_middle']['hidden_only_accuracy'])

    def test_repeated_mask_measurements_do_not_inflate_conversation_count(self):
        score = {'all_hidden': {'hidden_frames': 4, 'correct_units': 1, 'ce_sum': None}}
        trial = {'mask_type': 'random', 'requested_hidden_ratio': .25,
                 'nearest_copy': score, 'conditions': {'null_prior': {'first_pass': score}}}
        summary = summarize([{'conversation_id': 'one', 'trials': [trial, trial]}])
        entry = summary['random/0.25/nearest_copy']['all_hidden']
        self.assertEqual(entry['measurements'], 2)
        self.assertEqual(entry['conversations'], 1)
        self.assertEqual(entry['hidden_only_accuracy'], .25)

    def test_checkpoint_labels_are_safe_and_unique(self):
        self.assertEqual(set(checkpoint_arguments(None)), {'prior', 'conditional', 'current'})
        self.assertEqual(str(checkpoint_arguments(['prior=a.pt'])['prior']), 'a.pt')
        for values in (['../bad=a.pt'], ['prior=a.pt', 'prior=b.pt'], ['bad'], ['summary=a.pt']):
            with self.assertRaises(ValueError):
                checkpoint_arguments(values)

    def test_tiny_model_evaluates_both_conditions_and_keeps_visible_hints(self):
        from model.full_speech.units import UnitSpeechSystem
        from tests.test_quality_speech import config, sample
        torch.set_num_threads(1)
        torch.manual_seed(7)
        model = UnitSpeechSystem(config(), torch.randn(8, 768)).eval()
        datum = sample()
        class Data:
            records = [{'path': 'one.npz', 'conversation_id': 'one'}]
            def __getitem__(self, index):
                return datum
        args = SimpleNamespace(device='cpu', ratios=[.25, 1.], mask_seeds=[42], sample_steps=2)
        frequencies = {'group_ids': frequency_groups([10, 9, 8, 7, 6, 5, 4, 3]).tolist()}
        examples = evaluate_model(model, Data(), [0], frequencies, args, 'toy')
        self.assertEqual(len(examples[0]['trials']), 4)
        for trial in examples[0]['trials']:
            self.assertEqual(set(trial['conditions']), {'null_prior', 'production_a'})
            self.assertEqual(set(trial['conditions']['null_prior']), {'first_pass', 'sampled'})
            self.assertEqual(trial['nearest_copy'] is None, trial['requested_hidden_ratio'] == 1.)
            for condition in trial['conditions'].values():
                self.assertTrue(0 <= condition['first_pass']['all_hidden']['hidden_only_accuracy'] <= 1)


if __name__ == '__main__':
    unittest.main()
