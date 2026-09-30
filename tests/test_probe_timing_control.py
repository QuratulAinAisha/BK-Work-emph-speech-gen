"""Check timing-only features and original-cache preservation. / 시간축만 바꾼 특징과 원본 캐시 보존을 검사합니다."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from model.full_speech.quality import text_ids
from model.full_speech.tensor_ops import align
from scripts.check_a_information import (VIEWS, PROBE_VIEWS, file_hash, load_cached, save_tensor_file, write_json)
from scripts.prepare_probe_timing_control import VIEW, prepare_timing_cache, resample_to_fused


def make_source(root, same_length=False):
    cache = root / 'cache'
    provenance = {'feature_dimensions': VIEWS, 'checkpoint_sha256': 'original', 'cache_dtype': 'float32'}
    write_json(cache / 'provenance.json', provenance)
    index = {'provenance': provenance, 'splits': {}, 'ctc_infeasible': {'speech': [], 'fused': []}}
    for split in ('train', 'val'):
        index['splits'][split] = []
        row = {'path': split + '.npz', 'conversation_id': split + '_conversation', 'split': split,
               'reference_text': 'hi', 'required_ctc_frames': 2, 'files': {}, 'lengths': {}, 'sha256': {}}
        for view, length in [('speech', 8), ('fused', 8 if same_length else 4)]:
            relative = f'{view}/{split}/000000.pt'
            sample = {'features': torch.arange(length * 512).reshape(length, 512).float() / 512,
                      'text_a': text_ids('hi'), 'length': length, 'path': row['path'],
                      'conversation_id': row['conversation_id'], 'split': split, 'view': view}
            save_tensor_file(cache / relative, sample)
            row['files'][view], row['lengths'][view], row['sha256'][view] = relative, length, file_hash(cache / relative)
        index['splits'][split].append(row)
    write_json(cache / 'index.json', index)
    write_json(cache / 'complete.json', {'index_sha256': file_hash(cache / 'index.json')})
    return provenance


class ProbeTimingControlTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(5)

    def test_exact_production_alignment_in_both_directions(self):
        for original, target in [(9, 4), (4, 9), (4, 4), (1, 3), (3, 1)]:
            features = torch.randn(original, 512)
            expected = align(features[None], torch.ones(1, original, dtype=torch.bool),
                             torch.ones(1, target, dtype=torch.bool))[0]
            result = resample_to_fused(features, target)
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
            independent = F.interpolate(features.T[None], size=target, mode='linear', align_corners=False)[0].T
            torch.testing.assert_close(result, independent, rtol=0, atol=0)
            self.assertFalse(result.requires_grad)

    def test_derivation_is_immutable_idempotent_and_loads_as_optional_view(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / 'source', Path(folder) / 'timing'
            provenance = make_source(source)
            original = {str(path.relative_to(source)): file_hash(path) for path in source.rglob('*') if path.is_file()}
            audit = prepare_timing_cache(source, output)
            self.assertTrue(audit['needed'])
            self.assertEqual(audit['splits']['train']['target_over_source_ratio_mean'], .5)
            self.assertEqual(prepare_timing_cache(source, output), audit)
            after = {str(path.relative_to(source)): file_hash(path) for path in source.rglob('*') if path.is_file()}
            self.assertEqual(original, after)
            derived_index = json.loads((output / 'cache/index.json').read_text())
            self.assertEqual(set(derived_index['ctc_infeasible']), {VIEW})
            for key, value in provenance.items():
                self.assertEqual(derived_index['provenance'][key], value)
            self.assertNotIn(VIEW, VIEWS)
            self.assertEqual(PROBE_VIEWS[VIEW], 512)
            with patch('scripts.check_a_information.input_provenance', return_value=(provenance, {})):
                loaded, _ = load_cached(SimpleNamespace(output=output), VIEW)
            self.assertEqual(loaded['train'][0]['features'].shape, (4, 512))
            self.assertEqual(loaded['train'][0]['text_a'].tolist(), text_ids('hi').tolist())

    def test_same_lengths_report_no_needed_probe_and_reject_source_output_overlap(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / 'source', Path(folder) / 'timing'
            make_source(source, same_length=True)
            report = prepare_timing_cache(source, output)
            self.assertFalse(report['needed'])
            self.assertFalse((output / 'cache').exists())
            with self.assertRaisesRegex(ValueError, 'separate directory tree'):
                prepare_timing_cache(source, source / 'derived')

    def test_modified_source_features_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / 'source', Path(folder) / 'timing'
            make_source(source)
            item_path = source / 'cache/speech/train/000000.pt'
            sample = torch.load(item_path, weights_only=True)
            sample['features'].add_(1)
            save_tensor_file(item_path, sample)
            with self.assertRaisesRegex(ValueError, 'Source speech cache changed'):
                prepare_timing_cache(source, output)


if __name__ == '__main__':
    unittest.main()
