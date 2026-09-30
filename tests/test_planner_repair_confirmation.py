"""Check fresh-development exclusions. / 새 개발 집합의 제외 규칙을 검사합니다."""

import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.prepare_planner_experiment import json_bytes
from scripts.prepare_planner_repair_confirmation import (
    ExposureMatcher, digest, main, manifest_index, scan_exposures, select_fresh, verify_snapshot,
)


def row(name, text, split='val', **changes):
    result = {'path': 'features/' + name + '.npz', 'base_path': '/base/' + name + '.npz',
              'conversation_id': name, 'split': split, 'input_text': text,
              'response_text': 'I understand how difficult that feels.', 'style_id': 0,
              'speaker_id': 1, 'reference_audio': '/audio/' + name + '/response.wav'}
    result.update(changes)
    return result


def selection(train, val):
    return {'train': [value['path'] for value in train], 'val': [value['path'] for value in val],
            'style_id': 0, 'speaker_id': 1,
            'conversations': {'train': [value['conversation_id'] for value in train],
                              'val': [value['conversation_id'] for value in val]}}


class FreshDevelopmentTests(unittest.TestCase):
    def setUp(self):
        self.train = row('train', 'The bicycle tire broke', 'train')
        self.other_train = row('unselected_train',
            'Today my brother bought himself a very large house near the coast', 'train')
        self.dev = row('development', 'The rainy wedding was unforgettable')
        self.used = row('used', 'I finally received a job offer')
        self.fresh = row('fresh', 'School begins tomorrow')
        self.records = [self.train, self.other_train, self.dev, self.used, self.fresh,
            row('training_near_duplicate', self.other_train['input_text'] + ' today'),
            row('used_duplicate', self.used['input_text']),
            row('development_duplicate', self.dev['input_text']),
            row('test_only', 'A private future topic', 'test'),
            row('digits', 'Aquariums close at sunset', response_text='You waited ４ hours.'),
            row('short', 'I lost an umbrella'),
            row('long', 'My flight was canceled'),
            row('wrong_style', 'Flowers opened at dawn', style_id=1),
            row('wrong_voice', 'We traveled by ferry', speaker_id=0)]
        self.selected = selection([self.train], [self.dev])
        self.exposures = {'exposed_validation_ids': ['used'], 'scanned_files': []}

    def select(self, records=None, count=1):
        opened = []
        def duration(value):
            opened.append(value['conversation_id'])
            self.assertEqual(value['split'], 'val')
            return {'short': 2., 'long': 9.}.get(value['conversation_id'], 5.)
        result = select_fresh({'records': records or self.records}, self.selected,
                              self.exposures, duration, count=count)
        return result, opened

    def test_excludes_all_training_and_history_without_opening_test(self):
        (chosen, audit), opened = self.select()
        self.assertEqual(chosen['conversations']['val'], ['fresh'])
        self.assertEqual(chosen['train'], self.selected['train'])
        self.assertEqual(chosen['conversations']['train'], self.selected['conversations']['train'])
        self.assertEqual(audit['all_original_training_conversations_indexed'], 2)
        self.assertEqual(audit['selected_original_splits'], ['val'])
        self.assertEqual(audit['test_audio_or_feature_files_opened'], 0)
        self.assertNotIn('test_only', opened)
        self.assertNotIn('digits', opened)
        self.assertTrue(audit['no_relaxed_filters'])

    def test_insufficient_pool_fails_with_counts_and_no_relaxation(self):
        with self.assertRaisesRegex(ValueError, 'Insufficient clean development pool') as caught:
            self.select(count=2)
        details = json.loads(str(caught.exception).split('. ', 1)[1])
        self.assertEqual(details['eligible_selected'], 1)
        self.assertEqual(details['rejections_before_selection_complete'][
            'lexically_related_to_any_original_training_A'], 1)
        self.assertEqual(details['rejections_before_selection_complete'][
            'previously_exposed_conversation'], 2)

    def test_order_independence_duplicate_variants_and_fresh_text_duplicates(self):
        duplicate = copy.deepcopy(self.fresh)
        duplicate['path'] = 'features/fresh_second.npz'
        records = self.records + [duplicate, row('fresh_copy', self.fresh['input_text'])]
        (first, _), _ = self.select(records)
        (second, _), _ = self.select(list(reversed(records)))
        self.assertEqual(first, second)
        with self.assertRaisesRegex(ValueError, 'Insufficient clean development pool'):
            self.select(records, count=2)

    def test_exposure_of_one_variant_excludes_entire_conversation(self):
        variant = copy.deepcopy(self.fresh)
        variant.update(path='features/fresh_other.npz', style_id=2)
        manifest = {'records': self.records + [variant]}
        ids, _ = ExposureMatcher(manifest).match({'examples': [{'path': variant['path']}]})
        self.assertEqual(ids, {'fresh'})
        with self.assertRaisesRegex(ValueError, 'Insufficient clean development pool'):
            select_fresh(manifest, self.selected, {'exposed_validation_ids': ['used', 'fresh']},
                         lambda value: 2. if value['conversation_id'] == 'short' else 9.)

    def test_invalid_selection_and_cross_split_ids_fail(self):
        broken = copy.deepcopy(self.selected)
        broken['conversations']['val'] = ['wrong']
        with self.assertRaisesRegex(ValueError, 'metadata'):
            select_fresh({'records': self.records}, broken, self.exposures, lambda _: 5.)
        repeated = copy.deepcopy(self.fresh)
        repeated.update(path='other.npz', split='train')
        with self.assertRaisesRegex(ValueError, 'text/split'):
            manifest_index({'records': self.records + [repeated]})
        with self.assertRaisesRegex(ValueError, 'non-validation'):
            select_fresh({'records': self.records}, self.selected,
                         {'exposed_validation_ids': ['train']}, lambda _: 5.)

    def test_nested_examples_selection_lists_keys_and_embedded_paths(self):
        values = [row('hit1_conv2', 'Apples'), row('hit2_conv3', 'Bananas'),
                  row('hit3_conv4', 'Cherries'), row('hit4_conv5', 'Dates'),
                  row('hit5_conv6', 'Elderberries'), row('hit6_conv7', 'Figs'),
                  row('train_only', 'Grapes', 'train'), row('test_only', 'Hazelnuts', 'test')]
        matcher = ExposureMatcher({'records': values})
        report = {'nested': [{'examples': [{'conversation_id': 'hit1_conv2'}]}],
                  'selection': ['features/hit2_conv3.npz'],
                  'mapping': {'hit3_conv4': {'score': 1}},
                  'notes': 'Previously inspected:hit4_conv5. Saved /server/cache/hit5_conv6.npz.',
                  'reference': '/audio/hit6_conv7/response.wav',
                  'ignored': ['train_only', 'test_only', '/different/response.wav']}
        ids, paths = matcher.match(report)
        self.assertEqual(ids, {value['conversation_id'] for value in values[:6]})
        self.assertIn('features/hit2_conv3.npz', paths)
        self.assertEqual(matcher.match({'path': r'C:\relocated\HIT2_CONV3.NPZ'})[0], {'hit2_conv3'})

    def test_history_snapshot_hashes_nested_jsonl_and_overlap_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / 'nested'
            nested.mkdir()
            (root / 'report.json').write_text(json.dumps({'examples': [{'id': 'used'}]}))
            (nested / 'selection.jsonl').write_text(json.dumps(['features/development.npz']) + '\n')
            result = scan_exposures({'records': self.records}, [root, nested])
            self.assertEqual(result['exposed_validation_ids'], ['development', 'used'])
            self.assertEqual(len(result['scanned_files']), 2)
            self.assertEqual(result['scanned_file_formats'], ['.json', '.jsonl'])
            self.assertIn('not scanned', result['format_limit'])
            verify_snapshot(result)
            (root / 'report.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'changed'):
                verify_snapshot(result)
            refreshed = scan_exposures({'records': self.records}, [root])
            (root / 'new.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'membership changed'):
                verify_snapshot(refreshed)

    def test_real_bk_colon_ids_are_found_inside_prose(self):
        records = [row('hit:170_conv:340', 'A new concern'),
                   row('hit:1684_conv:3369', 'A different concern')]
        matcher = ExposureMatcher({'records': records})
        ids, _ = matcher.match({'notes': 'Inspected:hit:170_conv:340. Also viewed (hit:1684_conv:3369).',
                                'unknown': 'hit:999_conv:999'})
        self.assertEqual(ids, {value['conversation_id'] for value in records})

    def test_corrupt_history_and_raw_inventory_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            path.write_text('{')
            with self.assertRaisesRegex(ValueError, 'stable snapshot'):
                scan_exposures({'records': self.records}, [directory])
            path.write_text(json.dumps({'target_contract': {}, 'records': self.records}))
            with self.assertRaisesRegex(ValueError, 'raw manifest inventory'):
                scan_exposures({'records': self.records}, [directory])

    def test_cli_hashes_128_current_cases_and_immutable_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = root / 'history'
            history.mkdir()
            (history / 'used.json').write_text(json.dumps({'nested': ['used']}))
            dev = [row('current_' + str(i), 'Current topic ' + str(i)) for i in range(128)]
            manifest_bytes = json_bytes({'records': self.records + dev})
            selected = selection([self.train], dev)
            selected['manifest_sha256'] = digest(manifest_bytes)
            manifest_path, selection_path = root / 'manifest.json', root / 'selection.json'
            manifest_path.write_bytes(manifest_bytes)
            selection_path.write_bytes(json_bytes(selected))
            args = ['prepare', '--manifest', str(manifest_path), '--selection', str(selection_path),
                    '--used-root', str(history), '--output', str(root / 'result'), '--count', '1']
            with patch('sys.argv', args), patch('soundfile.info') as info:
                info.return_value.duration = 5.
                main()
            output = root / 'result'
            chosen = json.loads((output / 'confirmation_selection.json').read_bytes())
            audit = json.loads((output / 'confirmation_data_audit.json').read_bytes())
            self.assertEqual(len(chosen['val']), 1)
            self.assertEqual(chosen['train'], selected['train'])
            self.assertTrue(audit['report_snapshot_verified_unchanged'])
            self.assertEqual(audit['current_development_conversations'], 128)
            self.assertEqual(len(audit['scanned_report_hashes']), 1)
            self.assertEqual(audit['confirmation_selection_sha256'],
                             digest((output / 'confirmation_selection.json').read_bytes()))
            with patch('sys.argv', args), self.assertRaisesRegex(ValueError, 'never overwritten'):
                main()
            failed_args = list(args)
            failed_args[failed_args.index('--output') + 1] = str(root / 'insufficient')
            with patch('sys.argv', failed_args), patch('soundfile.info') as info:
                info.return_value.duration = 2.
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                    main()
            self.assertEqual(caught.exception.code, 1)
            failed = root / 'insufficient'
            self.assertFalse((failed / 'confirmation_selection.json').exists())
            failure_audit = json.loads((failed / 'confirmation_data_audit.json').read_bytes())
            self.assertEqual(failure_audit['status'], 'insufficient_clean_pool')
            self.assertEqual(failure_audit['eligible_selected'], 0)
            self.assertEqual(len(failure_audit['scanned_report_hashes']), 1)
            self.assertIsNone(failure_audit['confirmation_selection_sha256'])


if __name__ == '__main__':
    unittest.main()
