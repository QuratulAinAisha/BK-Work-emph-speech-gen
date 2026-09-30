"""Verify matched comparisons and seed clustering. / 대응 비교와 시드 묶음을 검증합니다."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import compare_planner_repair_candidates as compare


def controls(seeds=(42,), offset=0):
    examples = []
    for cid in ('one', 'two'):
        for index, seed in enumerate(seeds):
            correct = 1 + index + offset
            examples.append({'conversation_id': cid, 'seed': seed, 'path': cid + '.npz', 'frames': 4,
                'target_unit_ids': [0, 1, 2, 3], 'target_duration_seconds': 2.,
                'shuffled_path': 'donor.npz', 'shuffled_conversation_id': 'donor',
                'conditions': {name: {'correct_units': correct, 'unit_accuracy': correct / 4,
                    'fully_masked_ce': 2. - offset, 'duration_abs_error_seconds': .5}
                    for name in ('correct_a', 'shuffled_a', 'zero_a')}})
    return {'manifest_sha256': 'manifest', 'selection_sha256': 'selection', 'refinement_steps': 8,
            'checkpoint': 'fixed.pt', 'checkpoint_step': 640, 'target_length_supplied_for_unit_diagnostic': True,
            'encoder_input_contract': 'person_a_only', 'examples': examples}


def audio(seeds=(42,), text='hello friend'):
    examples = []
    for cid in ('one', 'two'):
        for seed in seeds:
            examples.append({'conversation_id': cid, 'seed': seed, 'path': cid + '.npz',
                'input_text': 'A says hello ' + cid, 'reference_text': 'hello there', 'style_id': 0, 'speaker_id': 1,
                'paths': {'predicted_length_steps_8': {'asr': text, 'reference_wer': compare.word_error('hello there', text),
                    'asr_token_limit_reached': False, 'file': cid + '.wav', 'seconds': 2.}}})
    return {'checkpoint': 'fixed.pt', 'recovery_step': 640, 'asr_model': 'test-asr', 'examples': examples}


def write_candidate(root, label, control=None, sound=None):
    folder = root / label
    folder.mkdir()
    if control is not None:
        (folder / 'val_controls.json').write_text(json.dumps(control), encoding='utf-8')
    if sound is not None:
        (folder / 'audio').mkdir()
        (folder / 'audio/report.json').write_text(json.dumps(sound), encoding='utf-8')
    return label, folder


class CandidateComparisonTests(unittest.TestCase):
    def test_repeated_seeds_pool_within_conversation_before_bootstrap(self):
        first = compare.normalize_controls(controls((42, 43), 1))
        second = compare.normalize_controls(controls((42, 43), 0))
        value = first['cases']['one']['conditions']['correct_a']
        self.assertEqual(value['correct'], 5)
        self.assertEqual(value['frames'], 8)
        result = compare.bootstrap_controls(first['cases'], second['cases'], 'correct_a', 'correct_a', 1000, 42)
        self.assertEqual(result['conversations'], 2)
        interval = result['metrics']['frame_unit_accuracy_delta_pp']
        self.assertEqual(interval['difference'], 25.)
        self.assertEqual(interval['ci95_percentile'], [25., 25.])
        reversed_rows = controls((42, 43), 1)
        reversed_rows['examples'].reverse()
        self.assertEqual(first, compare.normalize_controls(reversed_rows))

    def test_mismatched_targets_and_donors_do_not_silently_compare(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = controls()
            bad_target = copy.deepcopy(base)
            bad_target['examples'][0]['target_unit_ids'][0] = 99
            bad_donor = copy.deepcopy(base)
            bad_donor['examples'][0]['shuffled_path'] = 'different.npz'
            candidates = [write_candidate(root, 'base', base), write_candidate(root, 'target', bad_target),
                          write_candidate(root, 'donor', bad_donor)]
            result = compare.build_comparison(candidates, 1000)
            self.assertEqual(result['comparisons']['base_minus_target']['reports']['val']['status'], 'mismatch')
            value = result['comparisons']['base_minus_donor']['reports']['val']['conditions']
            self.assertIn('metrics', value['correct_a'])
            self.assertEqual(value['shuffled_a']['status'], 'mismatch')
            self.assertFalse(result['automatic_promotion'])
            self.assertIsNone(result['candidate_selected'])

    def test_unlabeled_duplicates_seed_mismatches_and_bad_counts_rejected(self):
        raw = controls()
        for row in raw['examples']:
            row.pop('seed')
        raw['examples'].append(copy.deepcopy(raw['examples'][0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            compare.normalize_controls(raw)
        first = compare.normalize_controls(controls((42, 43)))
        second = compare.normalize_controls(controls((42, 44)))
        self.assertTrue(compare.matching_errors(first['cases'], second['cases']))
        bad = controls()
        bad['examples'][0]['conditions']['correct_a']['generated_unit_ids'] = [0, 1, 2, 3]
        with self.assertRaisesRegex(ValueError, 'saved predictions'):
            compare.normalize_controls(bad)

    def test_audio_caps_exclude_whole_conversation_across_seeds_and_preserve_exact_text(self):
        raw = audio((42, 43), 'noise noise noise noise')
        raw['examples'][0]['paths']['predicted_length_steps_8']['asr_token_limit_reached'] = True
        first = compare.normalize_audio(raw)
        second = compare.normalize_audio(audio((42, 43), 'hello there'))
        all_cases = compare.bootstrap_audio(first['cases'], second['cases'], 'predicted_length_steps_8', 1000)
        shared = compare.bootstrap_audio(first['cases'], second['cases'], 'predicted_length_steps_8', 1000, uncapped=True)
        self.assertEqual(all_cases['conversations'], 2)
        self.assertEqual(shared['conversation_ids'], ['two'])
        self.assertEqual(shared['wer_delta_pp']['difference'], 200.)
        self.assertEqual(first['transcripts'][0]['paths']['predicted_length_steps_8']['asr'], 'noise noise noise noise')
        self.assertEqual(compare.audio_summary(first)['predicted_length_steps_8']['capped_conversations'], 1)

    def test_wrong_wer_is_flagged_without_discarding_valid_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = audio()
            raw['examples'][0]['paths']['predicted_length_steps_8']['reference_wer'] = 999.
            candidate = write_candidate(Path(directory), 'bad_audio', controls(), raw)
            report = compare.build_comparison([candidate], 1000)
            self.assertEqual(report['candidates']['bad_audio']['reports']['audio']['status'], 'invalid')
            self.assertEqual(report['candidates']['bad_audio']['reports']['val']['status'], 'valid')
            self.assertTrue(any(flag['reason'] == 'invalid_report' for flag in report['candidates']['bad_audio']['flags']))

    def test_cli_outputs_tables_transcripts_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = write_candidate(root, 'a', controls(), audio())
            b = write_candidate(root, 'b', controls(offset=1), audio(text='hello there'))
            before = {str(path): path.read_bytes() for path in root.rglob('*.json')}
            output = root / 'comparison'
            argv = ['compare', '--candidate', a[0] + '=' + str(a[1]), '--candidate', b[0] + '=' + str(b[1]),
                    '--output', str(output), '--draws', '1000']
            with patch('sys.argv', argv), patch('builtins.print'):
                compare.main()
                with self.assertRaisesRegex(ValueError, 'no files will be overwritten'):
                    compare.main()
            for name, content in before.items():
                self.assertEqual(Path(name).read_bytes(), content)
            self.assertEqual({path.name for path in output.iterdir()}, {'comparison.json', 'comparison.md', 'summary.csv'})
            report = json.loads((output / 'comparison.json').read_text())
            self.assertIsNone(report['candidate_selected'])
            self.assertIn('hello friend', (output / 'comparison.md').read_text(encoding='utf-8'))
            self.assertIn('shared_uncapped_cases', report['comparisons']['a_minus_b']['reports']['audio']['conditions']['predicted_length_steps_8'])


if __name__ == '__main__':
    unittest.main()
