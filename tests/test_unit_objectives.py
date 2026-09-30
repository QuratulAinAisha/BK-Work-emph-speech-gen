"""Loss weighting and inference boundaries. / 손실 가중치와 추론 경계를 검증합니다."""

import unittest
from unittest.mock import patch
import torch
import torch.nn.functional as F
from dataset.quality_speech_dataset import collate_quality
from model.full_speech.unit_objectives import draw_unit_masks, equal_sequence_loss, objective_settings
from model.full_speech.units import UnitSpeechSystem
from tests.test_quality_speech import config, sample


class UnitObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(9)

    def test_masks_never_hide_padding_and_prior_keeps_hints(self):
        lengths = torch.tensor([1, 2, 11, 25])
        mask = torch.arange(25)[None] < lengths[:, None]
        for span in (0., 1.):
            for _ in range(8):
                hidden = draw_unit_masks(mask, maximum=.8, span_probability=span)
                self.assertFalse(hidden[~mask].any())
                self.assertTrue(hidden.any(1).all())
                self.assertTrue(((mask & ~hidden).sum(1)[1:] > 0).all())
        self.assertTrue(torch.equal(draw_unit_masks(mask, fully_probability=1), mask))

    def test_sequence_weighting_is_not_hidden_token_weighting(self):
        logits = torch.tensor([[[4., 0.], [0., 4.], [0., 4.]],
                               [[0., 4.], [0., 4.], [0., 4.]]], requires_grad=True)
        ids = torch.zeros(2, 3, dtype=torch.long)
        mask = torch.ones(2, 3, dtype=torch.bool)
        hidden = torch.tensor([[True, False, False], [True, True, True]])
        result = equal_sequence_loss(logits, ids, hidden, mask)
        expected = (F.cross_entropy(logits[0, :1], ids[0, :1]) +
                    F.cross_entropy(logits[1], ids[1])) / 2
        torch.testing.assert_close(result['ce'], expected)
        result['ce'].backward()
        self.assertEqual(float(logits.grad[0, 1:].abs().sum()), 0.)
        self.assertGreater(float(logits.grad[1].abs().sum()), 0.)

    def test_prior_never_reads_a_or_labels(self):
        model = UnitSpeechSystem(config(), torch.randn(8, 768))
        model.configure_unit_prior()
        model.configure_unit_objective('curriculum', 100)
        batch = collate_quality([sample(0), sample(1)])
        batch = {key: batch[key] for key in ('semantic', 'semantic_len')}
        with patch.object(model, 'encode_batch', side_effect=AssertionError('A was used')):
            losses = model.losses(batch)
            losses['total'].backward()
        changed = [n for n, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0]
        self.assertTrue(changed)
        self.assertTrue(all(n.startswith('semantic_planner.') for n in changed))

    def test_conditional_validation_is_all_hidden_and_prior_is_fixed(self):
        model = UnitSpeechSystem(config(), torch.randn(8, 768))
        batch = collate_quality([sample(0), sample(1)])
        model.configure_recovery('planner')
        model.configure_unit_objective('curriculum', 100)
        model.eval()
        with patch.object(model.semantic_planner, 'logits', wraps=model.semantic_planner.logits) as spy:
            first = model.losses(batch)
        torch.testing.assert_close(spy.call_args.args[1], spy.call_args.args[2])
        self.assertEqual(float(first['full_sequence_fraction']), 1.)
        model.configure_unit_prior(); model.eval()
        first = model.losses(batch)['total']
        torch.manual_seed(456); model.current_step = 95
        second = model.losses(batch)['total']
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_rehearsal_adds_null_context_without_changing_eval_full_mask(self):
        model = UnitSpeechSystem(config(), torch.randn(8, 768))
        model.configure_recovery('planner')
        model.configure_unit_objective('balanced', 100, .25)
        model.eval()
        batch = collate_quality([sample(0), sample(1)])
        with patch.object(model.semantic_planner, 'logits', wraps=model.semantic_planner.logits) as spy:
            result = model.losses(batch)
        self.assertEqual(spy.call_count, 2)
        self.assertEqual(float(result['full_sequence_fraction']), 1.)
        self.assertTrue(torch.equal(spy.call_args.kwargs['context'], torch.zeros(2, 1, 512)))
        torch.testing.assert_close(result['total'], result['unit_ce'] + result['duration'] + .25 * result['prior_rehearsal_ce'])

    def test_stage_changes_clear_experimental_objective(self):
        model = UnitSpeechSystem(config(), torch.randn(8, 768))
        model.configure_recovery('planner')
        model.configure_unit_objective('balanced', 100, .25)
        model.configure_recovery('acoustic')
        self.assertIsNone(model.unit_objective_recipe)

    def test_fully_masked_training_retains_partial_prior_rehearsal(self):
        model = UnitSpeechSystem(config(), torch.randn(8, 768))
        model.configure_recovery('planner')
        model.configure_unit_objective('fully_masked', 100, .25)
        batch = collate_quality([sample(0), sample(1)])
        with patch.object(model.semantic_planner, 'logits', wraps=model.semantic_planner.logits) as spy:
            model.losses(batch)['total'].backward()
        first, second = spy.call_args_list
        self.assertTrue(torch.equal(first.args[1], first.args[2]))
        self.assertTrue((second.args[2] & ~second.args[1]).any(1).all())


if __name__ == '__main__':
    unittest.main()
