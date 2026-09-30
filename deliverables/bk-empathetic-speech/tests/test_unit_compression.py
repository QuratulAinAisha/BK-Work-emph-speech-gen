"""Lossless run compression contracts. / 무손실 연속 단위 압축 검증."""

import unittest
import numpy as np

from scripts.audit_unit_compression import compress_units, expand_units, selected_rows, summarize_rows


class UnitCompressionTests(unittest.TestCase):
    def test_runs_preserve_nonadjacent_repeats_and_boundaries(self):
        units = np.array([4, 4, 4, 2, 2, 4, 7, 7], dtype=np.int64)
        values, lengths = compress_units(units)
        np.testing.assert_array_equal(values, [4, 2, 4, 7])
        np.testing.assert_array_equal(lengths, [3, 2, 1, 2])
        np.testing.assert_array_equal(expand_units(values, lengths), units)
        for units in (np.array([], dtype=np.int64), np.array([5]), np.repeat(5, 10), np.arange(10)):
            np.testing.assert_array_equal(expand_units(*compress_units(units)), units)

    def test_malformed_runs_are_rejected(self):
        for values, lengths in (([1, 2], [2]), ([1], [0]), ([-1], [2]), ([1], [-3])):
            with self.assertRaises(ValueError):
                expand_units(np.array(values), np.array(lengths))
        with self.assertRaises(ValueError):
            compress_units(np.array([1., 2.]))
        with self.assertRaises(ValueError):
            compress_units(np.ones((2, 2), dtype=np.int64))

    def test_aggregate_rate_is_duration_weighted(self):
        rows = [{'duration_seconds': 1., 'raw_units': 4, 'compressed_runs': 2,
                 'compressed_to_raw_ratio': .5, 'compressed_runs_per_second': 2.},
                {'duration_seconds': 3., 'raw_units': 8, 'compressed_runs': 2,
                 'compressed_to_raw_ratio': .25, 'compressed_runs_per_second': 2 / 3}]
        result = summarize_rows(rows, [2, 2, 3, 5])
        self.assertEqual(result['raw_units_per_second'], 3.)
        self.assertEqual(result['compressed_runs_per_second'], 1.)
        self.assertAlmostEqual(result['compressed_to_raw_ratio'], 1 / 3)
        self.assertEqual(result['run_length_histogram_frames'], {2: 2, 3: 1, 5: 1})
        with self.assertRaises(ValueError):
            summarize_rows(rows, [1, 1])

    def test_selection_rejects_test_or_cross_split_rows(self):
        manifest = {'records': [{'path': 'a', 'split': 'train'}, {'path': 'b', 'split': 'test'}]}
        self.assertEqual(selected_rows(manifest, {'train': ['a']}, 'train'), manifest['records'][:1])
        with self.assertRaises(ValueError):
            selected_rows(manifest, {'val': ['b']}, 'val')
        with self.assertRaises(ValueError):
            selected_rows(manifest, {'train': ['a', 'a']}, 'train')


if __name__ == '__main__':
    unittest.main()
