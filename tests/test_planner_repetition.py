"""Unit repetition weighting contracts. / 단위 반복 가중 방식 검증."""

import unittest

from scripts.analyze_planner_repetition import analyze_controls, sequence_distribution


class PlannerRepetitionTests(unittest.TestCase):
    def test_two_unequal_sequences_and_boundary(self):
        result = sequence_distribution([[1, 1, 2], [2, 2, 2, 3, 3]])
        self.assertEqual(result['vocabulary_distinct_count'], 3)
        self.assertEqual(result['most_common_unit_id'], 2)
        self.assertEqual(result['most_common_pooled_fraction'], .5)
        self.assertEqual(result['pooled_unit_entropy_bits'], 1.5)
        self.assertEqual(result['adjacent_pair_count'], 6)
        self.assertEqual(result['adjacent_repeat_count'], 4)
        self.assertEqual(result['frame_weighted_adjacent_repeat_fraction'], .65625)
        self.assertEqual(result['example_mean_adjacent_repeat_fraction'], .625)
        self.assertAlmostEqual(result['pooled_adjacent_repeat_fraction'], 2 / 3)

    def test_singletons_do_not_form_cross_utterance_repeats(self):
        result = sequence_distribution([[5], [5]])
        self.assertEqual(result['pooled_unit_entropy_bits'], 0.)
        self.assertEqual(result['adjacent_pair_count'], 0)
        self.assertEqual(result['frame_weighted_adjacent_repeat_fraction'], 0.)
        self.assertEqual(result['example_mean_adjacent_repeat_fraction'], 0.)
        self.assertEqual(result['pooled_adjacent_repeat_fraction'], 0.)

    def test_reports_targets_and_conditions_separately(self):
        controls = {'examples': [{'frames': 3, 'target_unit_ids': [1, 2, 3],
                                  'conditions': {'correct_a': {'generated_unit_ids': [5, 5, 5]}}}]}
        result = analyze_controls(controls)
        self.assertEqual(result['target']['pooled_adjacent_repeat_fraction'], 0.)
        self.assertEqual(result['conditions']['correct_a']['pooled_adjacent_repeat_fraction'], 1.)
        controls['examples'][0]['frames'] = 4
        with self.assertRaises(ValueError):
            analyze_controls(controls)


if __name__ == '__main__':
    unittest.main()
