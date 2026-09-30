"""Oracle-hint isolation and metric checks. / 정답 힌트 분리와 지표 검증."""

import unittest
from unittest.mock import patch
import json
from pathlib import Path
import tempfile

import torch

from dataset.quality_speech_dataset import collate_quality
from model.full_speech.units import UnitSpeechSystem
from scripts.check_planner_hints import (nested_hidden_mask, score_hidden, prepare_pair, evaluate_hint_logits,
                                        audio_report, permute_visible_hints)
from tests.test_quality_speech import config, sample, FakeCodec


class PlannerHintTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_masks_are_nested_repeatable_and_padding_safe(self):
        mask = torch.arange(17)[None] < torch.tensor([17, 8])[:, None]
        previous = torch.zeros_like(mask)
        for ratio in (0., .25, .5, .75, 1.):
            hidden = nested_hidden_mask(mask, ratio, 42)
            torch.testing.assert_close(hidden, nested_hidden_mask(mask, ratio, 42))
            self.assertFalse((previous & ~hidden).any())
            self.assertFalse(hidden[~mask].any())
            torch.testing.assert_close(hidden.sum(1), (mask.sum(1).float() * ratio).ceil().long())
            previous = hidden
        torch.testing.assert_close(previous, mask)
        self.assertFalse(torch.equal(nested_hidden_mask(mask, .5, 42), nested_hidden_mask(mask, .5, 43)))

    def test_metrics_exclude_visible_units_and_handle_empty_prefix(self):
        targets = torch.tensor([[0, 1, 2, 1, 0]])
        mask = torch.ones_like(targets, dtype=torch.bool)
        hidden = torch.tensor([[False, True, False, True, False]])
        logits = torch.zeros(1, 5, 3)
        first = score_hidden(logits, targets, hidden, mask)
        logits[~hidden] = torch.tensor([100., -100., -100.])
        second = score_hidden(logits, targets, hidden, mask)
        self.assertEqual(first, second)
        self.assertEqual(first['all_hidden']['hidden_frames'], 2)
        self.assertEqual(first['prefix_20pct_time']['hidden_frames'], 0)
        self.assertIsNone(first['prefix_20pct_time']['hidden_only_ce'])
        self.assertEqual(first['remainder_80pct_time']['hidden_frames'], 2)
        empty = score_hidden(logits, targets, torch.zeros_like(mask), mask)
        self.assertIsNone(empty['all_hidden']['hidden_only_argmax_accuracy'])

    def test_all_hidden_has_no_b_ids_and_a_encoder_is_target_free(self):
        torch.manual_seed(7)
        model = UnitSpeechSystem(config(), torch.randn(8, 768)).eval()
        batch = collate_quality([sample(0)])
        donor = collate_quality([sample(1)])
        with patch.object(model, 'encode_batch', wraps=model.encode_batch) as encode:
            targets, mask, contexts, style = prepare_pair(model, batch, donor)
        for call in encode.call_args_list:
            self.assertFalse({'semantic', 'codec', 'text_b', 'duration', 'waveform'} & call.args[0].keys())
            torch.testing.assert_close(call.args[0]['style_id'], batch['style_id'])
            torch.testing.assert_close(call.args[0]['speaker_id'], batch['speaker_id'])
        with patch.object(model.semantic_planner, 'logits', wraps=model.semantic_planner.logits) as logits:
            result = evaluate_hint_logits(model.semantic_planner, targets, mask, contexts, style, 1., 42)
        for call in logits.call_args_list:
            self.assertEqual(int(call.args[0].abs().sum()), 0)
            torch.testing.assert_close(call.args[1], mask)
        self.assertEqual(result['hidden_frames'], int(mask.sum()))
        self.assertEqual(set(result['conditions']), {'correct_a', 'shuffled_a'})
        control = result['visible_hint_control']
        self.assertEqual(control['changed_visible_frames'], 0)
        self.assertEqual(control['conditions']['correct_visible_b'], control['conditions']['permuted_visible_b'])

    def test_partial_hints_contain_only_visible_target_ids(self):
        torch.manual_seed(11)
        model = UnitSpeechSystem(config(), torch.randn(8, 768)).eval()
        batch = collate_quality([sample(0)])
        targets, mask, contexts, style = prepare_pair(model, batch, batch)
        with patch.object(model.semantic_planner, 'logits', wraps=model.semantic_planner.logits) as logits:
            evaluate_hint_logits(model.semantic_planner, targets, mask, contexts, style, .5, 42)
        for call in logits.call_args_list[:2]:
            initial, hidden = call.args[:2]
            torch.testing.assert_close(initial[mask & ~hidden], targets[mask & ~hidden])
            self.assertEqual(int(initial[hidden].abs().sum()), 0)
        changed, hidden = logits.call_args.args[:2]
        torch.testing.assert_close(changed[mask & ~hidden].sort().values, targets[mask & ~hidden].sort().values)
        self.assertEqual(int(changed[hidden].abs().sum()), 0)
        for call in logits.call_args_list:
            torch.testing.assert_close(call.args[1], hidden)

    def test_visible_permutation_is_seeded_histogram_preserving_and_hidden_free(self):
        mask = torch.tensor([[True] * 9 + [False]])
        hidden = torch.tensor([[False, True, False, True, False, True, False, False, False, False]])
        ids = torch.tensor([[1, 0, 2, 0, 3, 0, 4, 5, 6, 0]])
        rng = torch.get_rng_state()
        shuffled = permute_visible_hints(ids, hidden, mask, 42)
        torch.testing.assert_close(shuffled, permute_visible_hints(ids, hidden, mask, 42))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertFalse(torch.equal(shuffled, ids))
        torch.testing.assert_close(shuffled[mask & ~hidden].sort().values, ids[mask & ~hidden].sort().values)
        self.assertTrue((shuffled[hidden | ~mask] == 0).all())
        leaked = ids.clone(); leaked[hidden] = 2
        with self.assertRaisesRegex(ValueError, 'must be zero'):
            permute_visible_hints(leaked, hidden, mask, 42)

    def test_memory_and_affect_masks_routed_and_hidden_targets_never_forwarded(self):
        torch.manual_seed(7)
        cfg = config(); cfg.planner_memory_mode = 'native_speech'
        model = UnitSpeechSystem(cfg, torch.randn(8, 768)).eval()
        batch = collate_quality([sample(0)])
        targets, mask, contexts, style = prepare_pair(model, batch, batch)
        self.assertEqual(int(contexts['correct_a']['context_mask'].sum()), 25)
        self.assertNotEqual(int(contexts['correct_a']['affect_mask'].sum()), 25)
        hidden = nested_hidden_mask(mask, .5, 42)
        changed_targets = targets.clone()
        changed_targets[hidden] = (changed_targets[hidden] + 1) % 8
        calls = []
        for values in (targets, changed_targets):
            with patch.object(model.semantic_planner, 'logits', wraps=model.semantic_planner.logits) as logits:
                evaluate_hint_logits(model.semantic_planner, values, mask, contexts, style, .5, 42)
            calls.append(logits.call_args_list)
        for original, changed in zip(*calls):
            torch.testing.assert_close(original.args[0], changed.args[0])
            torch.testing.assert_close(original.kwargs['affect_mask'], contexts['correct_a']['affect_mask'])

    def test_audio_integration_preserves_hints_and_saves_five_conditions(self):
        class Codec(FakeCodec):
            def to(self, device): return self
            def eval(self): return self

        class Data:
            records = [{'path': 'synthetic.npz', 'conversation_id': 'synthetic',
                        'input_text': 'Hi', 'response_text': 'Hello'}]
            def __init__(self): self.value = sample(0)
            def __getitem__(self, index): return self.value

        def recognition(report, output, device):
            report['summary'] = {'synthetic_test': True}
            return report

        torch.manual_seed(19)
        model = UnitSpeechSystem(config(), torch.randn(8, 768)).eval()
        with tempfile.TemporaryDirectory() as folder, \
             patch('model.full_speech.codec.FrozenEncodec', Codec), \
             patch('scripts.diagnose_quality.recognize', recognition):
            audio_report(model, Data(), [0], Path(folder), {}, torch.device('cpu'))
            report = json.loads((Path(folder) / 'audio/report.json').read_text())
            row = report['examples'][0]
            paths = row['paths']
            self.assertEqual(set(paths), {'reference', 'hidden_000', 'hidden_025', 'hidden_050',
                                          'hidden_075', 'hidden_100'})
            self.assertEqual(paths['hidden_000']['generated_unit_ids'], row['target_unit_ids'])
            self.assertEqual(paths['hidden_000']['sampling_trace'], [])
            self.assertEqual(paths['hidden_100']['actual_hidden_ratio'], 1.)
            self.assertEqual(paths['hidden_100']['sampling_trace'][-1]['still_hidden'], 0)
            for name, entry in paths.items():
                self.assertTrue((Path(folder) / 'audio' / entry['file']).is_file())
                if name != 'reference': self.assertTrue(entry['visible_hints_preserved'])


if __name__ == '__main__':
    unittest.main()
