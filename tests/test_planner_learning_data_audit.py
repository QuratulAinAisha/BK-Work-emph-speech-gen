"""Check audit integrity without data downloads. / 자료 다운로드 없이 감사 검사를 검증합니다."""

import unittest

import torch

from model.full_speech.units import UnitSpeechSystem
from scripts.audit_planner_learning_data import (check_mask_path, check_target_batch,
    cross_split_duplicates, summarize_units, transcript_summary, unit_details)
from tests.test_quality_speech import config, sample


def row(path, first, second):
    return {'path': path, 'input_text': first, 'response_text': second, 'speaker_id': 1, 'style_id': 0}


class PlannerLearningDataAuditTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(29)

    def test_template_prefixes_and_exact_duplicates_are_separate(self):
        rows = [row('one', 'A first input', 'That sounds really good'),
                row('two', 'Another input', 'That sounds really bad'),
                row('three', 'Another input', 'I hear you')]
        summary = transcript_summary(rows)
        self.assertEqual(summary['response_prefixes']['3']['top_prefix_fraction_of_all_records'], 2 / 3)
        self.assertEqual(summary['input_duplicates']['groups'], 1)
        self.assertEqual(summary['response_duplicates']['groups'], 0)
        self.assertEqual(summary['pair_duplicates']['groups'], 0)
        validation = [row('heldout', 'new input', 'That sounds really good')]
        overlap = cross_split_duplicates(rows, validation)
        self.assertEqual(overlap['response']['matched_validation_records'], 1)
        self.assertEqual(overlap['input']['matched_validation_records'], 0)
        self.assertEqual(overlap['pair']['matched_validation_records'], 0)

    def test_units_do_not_treat_utterance_boundaries_as_adjacent_frames(self):
        result = summarize_units([[1, 1], [1, 1]], 4)
        self.assertEqual(result['adjacent_pair_count'], 2)
        self.assertEqual(result['adjacent_repeat_count'], 2)
        self.assertEqual(result['unit_frequency'], [0, 4, 0, 0])
        self.assertEqual(result['unused_vocabulary_count'], 3)
        details = unit_details([1, 1, 2, 2, 2, 1])
        self.assertEqual(details['unit_changes'], 2)
        self.assertEqual(details['longest_run_frames'], 3)
        self.assertEqual(details['first_unit'], 1)
        self.assertEqual(details['last_unit'], 1)

    def test_actual_target_padding_and_normalization_paths(self):
        model = UnitSpeechSystem(config(), torch.randn(8, 768)).eval()
        with torch.no_grad():
            model.semantic_mean.copy_(torch.randn(768))
            model.semantic_std.copy_(torch.rand(768) + .5)
        samples = [sample(0), sample(1)]
        samples[1]['semantic'] = samples[1]['semantic'][:12]
        samples[1]['codec'] = samples[1]['codec'][:12]
        samples[1]['waveform'] = samples[1]['waveform'][:12 * 640]
        samples[1]['duration'] = torch.tensor(.24)
        records, normalized, mask, checks = check_target_batch(model, samples, torch.device('cpu'))
        self.assertEqual(checks['padding_frames_checked'], 4)
        self.assertEqual([record['frames'] for record in records], [16, 12])
        self.assertTrue(checks['padding_invariance'])
        self.assertTrue(checks['normalization_round_trip'])
        self.assertEqual(sum(len(record['float64_assignment_differences']) for record in records), 0)
        state = torch.get_rng_state()
        weights = {key: value.clone() for key, value in model.state_dict().items()}
        audit = check_mask_path(model.semantic_planner, normalized, mask)
        self.assertTrue(audit['passed'])
        self.assertEqual(len(audit['measurements']), 4)
        for measurement in audit['measurements']:
            self.assertTrue(measurement['hidden_targets_masked'])
            if not measurement['training']:
                self.assertEqual(measurement['hidden_frames'], measurement['valid_frames'])
        self.assertFalse(model.semantic_planner.training)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(weights[key], value))

    def test_invalid_normalization_and_duration_are_rejected(self):
        model = UnitSpeechSystem(config(), torch.randn(8, 768)).eval()
        bad = sample(0)
        bad['duration'] = torch.tensor(.20)
        with self.assertRaisesRegex(ValueError, 'frame counts disagree'):
            check_target_batch(model, [bad], torch.device('cpu'))
        with torch.no_grad():
            model.semantic_std[0] = 0
        with self.assertRaisesRegex(ValueError, 'normalization statistics'):
            check_target_batch(model, [sample(0)], torch.device('cpu'))


if __name__ == '__main__':
    unittest.main()
