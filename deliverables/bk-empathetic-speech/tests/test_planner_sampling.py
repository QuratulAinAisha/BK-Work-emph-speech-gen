"""Diagnostic decoding invariants. / 진단 디코딩 불변 조건."""
import math
import unittest
from unittest.mock import patch
import torch
from model.full_speech.units import MaskedUnitPlanner
from tests.test_quality_speech import config
from scripts.planner_sampling import sample_units


class SamplingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(3)
        self.planner = MaskedUnitPlanner(config(), torch.randn(8, 768)).eval()
        self.mask = torch.tensor([[True]*9+[False]*2, [True]*11])
        self.args = (self.mask, torch.randn(2, 5, 512), torch.ones(2, 5, dtype=torch.bool),
                     torch.randn(2, 5, 6), torch.randn(2, self.planner.config.hidden_dim))

    def test_greedy_exactly_reproduces_production(self):
        with torch.inference_mode():
            for steps in (1, 8, 16):
                expected = self.planner.sample(*self.args, steps=steps)
                ids, _ = sample_units(self.planner, *self.args, steps=steps)
                actual = self.planner.codebook.centers[ids].masked_fill(~self.mask[..., None], 0)
                self.assertTrue(torch.equal(expected, actual))

    def test_every_mode_preserves_visible_hints_and_padding(self):
        ids = torch.randint(0, 8, self.mask.shape)
        hidden = self.mask.clone(); hidden[:, ::2] = False
        for mode in ('greedy', 'categorical', 'revisable', 'random_remask'):
            result, trace = sample_units(self.planner, *self.args, mode=mode,
                                         initial_ids=ids, initial_hidden=hidden)
            self.assertTrue(torch.equal(result[self.mask & ~hidden], ids[self.mask & ~hidden]))
            self.assertTrue((result[~self.mask] == 0).all())
            self.assertEqual(trace[-1]['still_hidden'], 0)

    def test_sampling_is_seeded_and_does_not_use_global_rng(self):
        state = torch.get_rng_state()
        first, _ = sample_units(self.planner, *self.args, mode='categorical', seed=42)
        second, _ = sample_units(self.planner, *self.args, mode='categorical', seed=42)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_all_visible_is_exact_roundtrip(self):
        ids = torch.randint(0, 8, self.mask.shape)
        result, trace = sample_units(self.planner, *self.args, initial_ids=ids,
                                     initial_hidden=torch.zeros_like(self.mask))
        self.assertTrue(torch.equal(result[self.mask], ids[self.mask])); self.assertEqual(trace, [])

    def test_separate_affect_timeline_matches_production(self):
        memory_mask = self.args[2]
        affect = torch.randn(2, 3, 6)
        affect_mask = torch.tensor([[True, True, True], [True, True, False]])
        arguments = (self.mask, self.args[1], memory_mask, affect, self.args[-1])
        with torch.inference_mode():
            for steps in (1, 8):
                expected = self.planner.sample(*arguments, steps=steps, affect_mask=affect_mask)
                ids, _ = sample_units(self.planner, *arguments, steps=steps, affect_mask=affect_mask)
                actual = self.planner.codebook.centers[ids].masked_fill(~self.mask[..., None], 0)
                self.assertTrue(torch.equal(actual, expected))

    def test_random_remask_has_eight_forwards_and_cosine_counts(self):
        for width in (1, 5, 100):
            mask = torch.ones(1, width, dtype=torch.bool)
            args = (mask, self.args[1][:1], self.args[2][:1], self.args[3][:1], self.args[4][:1])
            with patch.object(self.planner, 'logits', wraps=self.planner.logits) as logits:
                _, trace = sample_units(self.planner, *args, mode='random_remask', steps=8)
            self.assertEqual(logits.call_count, 8)
            self.assertEqual(len(trace), 8)
            expected = [max(1, math.floor(width * math.cos(math.pi / 2 * (i + 1) / 8))) for i in range(7)] + [0]
            self.assertEqual([row['still_hidden'] for row in trace], expected)
            if width == 100:
                self.assertGreater(sum(row['reopened_accepted_positions'] for row in trace), 0)

    def test_random_remask_is_seeded_independent_of_confidence_and_global_rng(self):
        mask = torch.ones(1, 40, dtype=torch.bool)
        args = (mask, self.args[1][:1], self.args[2][:1], self.args[3][:1], self.args[4][:1])
        state = torch.get_rng_state()

        def masks_with(scale, seed):
            captured = []

            def logits(ids, hidden, valid, *unused, **kwargs):
                captured.append(hidden.clone())
                scores = torch.zeros(1, 40, 8)
                scores[..., 1] = torch.arange(40) * scale + 1
                return scores

            with patch.object(self.planner, 'logits', side_effect=logits):
                result, _ = sample_units(self.planner, *args, mode='random_remask', seed=seed)
            return result, captured

        first, masks = masks_with(1., 42)
        second, same = masks_with(100., 42)
        _, different = masks_with(1., 43)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(masks, same)))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(masks, different)))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_remasked_accepted_positions_receive_actual_mask_embedding(self):
        mask = torch.ones(1, 40, dtype=torch.bool)
        hidden = mask.clone(); hidden[:, ::4] = False
        hints = torch.full(mask.shape, 3, dtype=torch.long)
        args = (mask, self.args[1][:1], self.args[2][:1], self.args[3][:1], self.args[4][:1])
        values = []
        hook = self.planner.denoiser.register_forward_pre_hook(lambda module, inputs: values.append(inputs[0].clone()))
        try:
            with patch.object(self.planner, 'logits', wraps=self.planner.logits) as logits:
                result, trace = sample_units(self.planner, *args, mode='random_remask',
                                             initial_ids=hints, initial_hidden=hidden)
        finally:
            hook.remove()
        self.assertGreater(sum(row['reopened_accepted_positions'] for row in trace), 0)
        for value, call in zip(values, logits.call_args_list):
            current_hidden = call.args[1]
            torch.testing.assert_close(value[current_hidden], self.planner.mask_embedding.expand(int(current_hidden.sum()), -1), rtol=0, atol=0)
            torch.testing.assert_close(value[~hidden], self.planner.codebook.centers[hints[~hidden]], rtol=0, atol=0)
        torch.testing.assert_close(result[~hidden], hints[~hidden])


if __name__ == '__main__': unittest.main()
