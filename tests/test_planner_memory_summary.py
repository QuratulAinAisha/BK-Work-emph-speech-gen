"""Verify conversation-level paired comparison. / 대화 단위의 짝지은 비교를 검증합니다."""

import json
import hashlib
from pathlib import Path
import tempfile
import unittest

from scripts.summarize_planner_memory import (ARMS, build_summary, control_cases,
    endpoint_check, paired_bootstrap, tail_cases, attest_audio_alias, verify_checkpoint_identity)


class PlannerMemorySummaryTests(unittest.TestCase):
    def test_bootstrap_pairs_conversations_and_reports_weighted_estimands(self):
        first = {'one': {'correct': 2, 'frames': 10}, 'two': {'correct': 20, 'frames': 100}}
        second = {'one': {'correct': 1, 'frames': 10}, 'two': {'correct': 10, 'frames': 100}}
        result = paired_bootstrap(first, second)
        self.assertEqual(result['paired_conversations'], 2)
        self.assertEqual(result['resamples'], 10000)
        self.assertAlmostEqual(result['mean_conversation_accuracy']['difference'], .1)
        self.assertAlmostEqual(result['frame_accuracy']['difference'], .1)
        for value in result['mean_conversation_accuracy']['percentile_95_ci']:
            self.assertAlmostEqual(value, .1)
        self.assertEqual(result, paired_bootstrap(first, second))

    def test_case_average_is_distinct_from_pooled_frame_average(self):
        first = {'one': {'correct': 1, 'frames': 1}, 'two': {'correct': 0, 'frames': 9}}
        second = {'one': {'correct': 0, 'frames': 1}, 'two': {'correct': 0, 'frames': 9}}
        result = paired_bootstrap(first, second, resamples=100)
        self.assertEqual(result['mean_conversation_accuracy']['difference'], .5)
        self.assertEqual(result['frame_accuracy']['difference'], .1)

    def test_hint_seeds_pool_within_conversation_before_bootstrap(self):
        trials = [{'requested_hidden_ratio': .25, 'mask_seed': seed, 'mask_sha256': str(seed),
                   'conditions': {'correct_a': {'remainder_80pct_time': {
                       'correct_units': correct, 'hidden_frames': frames}}}}
                  for seed, correct, frames in [(42, 1, 2), (43, 0, 3)]]
        report = {'mask_seeds': [42, 43], 'examples': [{'conversation_id': 'case', 'trials': trials}]}
        cases = tail_cases(report)
        self.assertEqual(cases['case']['accuracy'], .2)
        self.assertEqual(cases['case']['mask_seed_count'], 2)
        self.assertEqual(paired_bootstrap(cases, cases)['paired_conversations'], 1)
        report['examples'][0]['trials'].append(trials[0])
        with self.assertRaises(ValueError):
            tail_cases(report)

    def test_mismatched_pair_masks_or_membership_fail(self):
        first = {'case': {'correct': 1, 'frames': 4, 'mask_signature': [(42, 'a', 4)]}}
        second = {'case': {'correct': 2, 'frames': 4, 'mask_signature': [(42, 'b', 4)]}}
        with self.assertRaises(ValueError):
            paired_bootstrap(first, second)
        with self.assertRaises(ValueError):
            paired_bootstrap(first, {})

    def test_endpoint_cannot_substitute_best_or_another_step(self):
        report = {'checkpoint_step': 320, 'planner_memory_mode': 'native_speech',
                  'split': 'val', 'checkpoint': 'outputs/native_speech/step_0320.pt'}
        endpoint_check(report, 'native_speech')
        for changed in ({'checkpoint_step': 160}, {'checkpoint': 'outputs/best.pt'},
                        {'planner_memory_mode': 'fused'}, {'split': 'test'}):
            with self.assertRaises(ValueError):
                endpoint_check({**report, **changed}, 'native_speech')

    def test_missing_confirmation_is_allowed_and_does_not_open_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'plan.json').write_text(json.dumps({'steps_per_arm': 320, 'arms': list(ARMS),
                'validation_conversations': 128}), encoding='utf-8')
            result = build_summary(root)
            self.assertEqual(result['confirmation']['status'], 'not_available')
            self.assertFalse(result['checkpoint_selection_performed'])
            self.assertFalse(result['candidate_promoted'])

    def test_repeated_control_conversations_are_rejected(self):
        row = {'conversation_id': 'same', 'frames': 4, 'conditions': {'correct_a': {'correct_units': 1}}}
        with self.assertRaises(ValueError):
            control_cases({'examples': [row, row]})

    def test_alias_attestation_is_portable_and_bound_to_report_bytes(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'native_speech'
            audio = folder / 'audio_step_0320'
            audio.mkdir(parents=True)
            payload = {'architecture': 'llm_free_speech_units_v1', 'recovery_step': 320,
                'config': {'planner_memory_mode': 'native_speech'},
                'metadata': {'recovery_recipe': {'steps': 320}},
                'state_dict': {'semantic_planner.weight': torch.tensor([1., 2.])}}
            for name in ('last.pt', 'step_0320.pt'):
                torch.save(payload, folder / name)
            report = {'recovery_step': 320, 'planner_memory_mode': 'native_speech',
                      'split': 'val', 'checkpoint': str(folder / 'last.pt')}
            data = json.dumps(report).encode()
            (audio / 'report.json').write_bytes(data)
            attestation = attest_audio_alias(root, 'native_speech')
            self.assertEqual(attestation, attest_audio_alias(root, 'native_speech'))
            hashed = hashlib.sha256(data).hexdigest()
        # Checkpoints need not be downloaded with the verified report. / 검증 보고서와 함께 체크포인트를 내려받을 필요는 없습니다.
        endpoint_check(report, 'native_speech', audio=True, attestation=attestation, report_sha256=hashed)
        with self.assertRaises(ValueError):
            endpoint_check(report, 'native_speech', audio=True)
        with self.assertRaises(ValueError):
            endpoint_check(report, 'native_speech', audio=True, attestation=attestation,
                           report_sha256=hashlib.sha256(data + b' ').hexdigest())
        incomplete = {**attestation, 'checks': {**attestation['checks'], 'configs_equal': False}}
        with self.assertRaises(ValueError):
            endpoint_check(report, 'native_speech', audio=True, attestation=incomplete, report_sha256=hashed)

    def test_alias_payload_identity_rejects_changed_weights_config_step_or_recipe(self):
        import copy
        import torch
        payload = {'architecture': 'llm_free_speech_units_v1', 'recovery_step': 320,
            'config': {'planner_memory_mode': 'native_speech'},
            'metadata': {'recovery_recipe': {'steps': 320}},
            'state_dict': {'weight': torch.tensor([1., 2.])}}
        self.assertEqual(verify_checkpoint_identity(payload, copy.deepcopy(payload), 'native_speech'), 1)
        for change in ('weights', 'config', 'step', 'recipe', 'dtype'):
            other = copy.deepcopy(payload)
            if change == 'weights':
                other['state_dict']['weight'][0] = 3
            elif change == 'config':
                other['config']['planner_memory_mode'] = 'resampled_speech'
            elif change == 'step':
                other['recovery_step'] = 256
            elif change == 'recipe':
                other['metadata']['recovery_recipe']['steps'] = 640
            else:
                other['state_dict']['weight'] = other['state_dict']['weight'].double()
            with self.subTest(change=change), self.assertRaises(ValueError):
                verify_checkpoint_identity(payload, other, 'native_speech')


if __name__ == '__main__':
    unittest.main()
