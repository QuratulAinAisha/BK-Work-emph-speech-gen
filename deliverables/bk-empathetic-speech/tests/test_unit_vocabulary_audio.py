"""Protect the oracle vocabulary comparison. / 정답 기반 단위 사전 비교를 검증합니다."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from dataset.quality_speech_dataset import collate_quality, person_a_only
from model.full_speech.units import SpeechCodebook, UnitSpeechSystem
from scripts.check_unit_vocabulary_audio import (ARMS, generate_oracle_arms, load_candidate,
    module_hashes, selected_indices, summarize_distortion, tensor_hash)
from tests.test_quality_speech import FakeCodec, config, sample


class VocabularyAudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(73)
        self.model = UnitSpeechSystem(config(), torch.stack((torch.zeros(768), torch.full((768,), 10.)))).eval()
        self.book = SpeechCodebook(torch.stack((torch.full((768,), -.25), torch.full((768,), .25))))
        self.batch = collate_quality([sample()])

    def test_no_double_quantization_and_same_noise_duration_and_a_inputs(self):
        target = (self.batch['semantic'] - self.model.semantic_mean) / self.model.semantic_std
        expected_current = self.model.semantic_planner.codebook(target)
        expected_new = self.book(target)
        self.assertFalse(torch.equal(expected_new, self.model.semantic_planner.codebook(expected_new)))
        before = module_hashes(self.model)
        noise, seen_inputs = [], []
        original_noise, original_encode = torch.randn, self.model.encode_batch

        def capture_noise(*args, **kwargs):
            value = original_noise(*args, **kwargs)
            if kwargs.get('generator') is not None:
                noise.append(value.clone())
            return value

        def capture_inputs(received):
            allowed = person_a_only(self.batch)
            self.assertEqual(set(received), set(allowed))
            self.assertTrue(all(torch.equal(received[key], allowed[key]) for key in allowed))
            seen_inputs.append(received)
            return original_encode(received)

        # Forbidden wrappers would snap candidate centers back to the old book. / 금지된 래퍼는 새 중심을 기존 사전으로 되돌립니다.
        with patch.object(self.model, 'generate_batch', side_effect=AssertionError('Old wrapper called')), \
                patch.object(self.model.semantic_planner.codebook, 'forward', side_effect=AssertionError('Double quantization')), \
                patch.object(self.model.semantic_planner, 'sample', side_effect=AssertionError('Planner must be bypassed')), \
                patch.object(self.model, 'encode_batch', side_effect=capture_inputs), \
                patch('torch.randn', side_effect=capture_noise):
            result = generate_oracle_arms(self.model, self.book, self.batch, FakeCodec())
        self.assertEqual(len(noise), 3)
        self.assertEqual(len(seen_inputs), 3)
        for value in noise[1:]:
            torch.testing.assert_close(value, noise[0], rtol=0, atol=0)
        self.assertEqual(noise[0].shape, (1, 16, 128))
        for name, expected in zip(ARMS, (expected_current, expected_new, target)):
            generated = result['results'][name]
            torch.testing.assert_close(generated['semantic'], expected, rtol=0, atol=0)
            self.assertEqual(generated['semantic'].shape[-1], 768)
            torch.testing.assert_close(generated['duration'], self.batch['duration'], rtol=0, atol=0)
            self.assertEqual(int(generated['audio_lengths'][0]), 10240)
            self.assertEqual(result['metrics'][name]['semantic_sha256'], tensor_hash(expected))
            self.assertAlmostEqual(result['metrics'][name]['normalized_mse'], float((expected - target).square().mean()))
        self.assertEqual(before, module_hashes(self.model))

    def test_bad_semantic_width_padding_or_duration_is_rejected(self):
        for update in ({'semantic': self.batch['semantic'][..., :767]},
                       {'semantic_len': torch.tensor([15])},
                       {'duration': torch.tensor([.5])}, {'duration': torch.tensor([float('nan')])}):
            with self.subTest(keys=list(update)), self.assertRaisesRegex(ValueError, '768D'):
                generate_oracle_arms(self.model, self.book, {**self.batch, **update}, FakeCodec())

    def candidate_payload(self):
        return {'architecture': 'bk_speech_codebook_v1', 'centers': self.book.centers,
            'semantic_mean': self.model.semantic_mean.detach().clone(),
            'semantic_std': self.model.semantic_std.detach().clone(), 'fit_split': 'train',
            'manifest_sha256': 'm' * 64, 'selection_sha256': 'a' * 64,
            'clusters': 2, 'training_conversations': 16, 'frames': 256}

    def test_candidate_requires_exact_normalization_and_training_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'candidate.pt'
            torch.save(self.candidate_payload(), path)
            actual, payload = load_candidate(path, self.model, 'm' * 64)
            torch.testing.assert_close(actual.centers, self.book.centers, rtol=0, atol=0)
            self.assertEqual(payload['fit_split'], 'train')
            changes = [
                {'semantic_mean': self.model.semantic_mean + 1e-6},
                {'semantic_std': self.model.semantic_std * 2},
                {'semantic_std': self.model.semantic_std.double()},
                {'centers': torch.zeros(2, 767)}, {'fit_split': 'val'},
                {'manifest_sha256': 'other'}, {'selection_sha256': None},
                {'training_conversations': 0}, {'clusters': 3},
            ]
            for change in changes:
                with self.subTest(keys=list(change)):
                    torch.save({**self.candidate_payload(), **change}, path)
                    with self.assertRaises(ValueError):
                        load_candidate(path, self.model, 'm' * 64)

    def test_selection_is_ordered_validation_only_and_complete(self):
        records = [{'path': 'a', 'conversation_id': 'A', 'split': 'val'},
                   {'path': 'b', 'conversation_id': 'B', 'split': 'val'}]
        selection = {'manifest_sha256': 'hash', 'val': ['b', 'a'], 'train': ['c']}
        self.assertEqual(selected_indices(records, selection, 'hash', 2), [1, 0])
        self.assertEqual(selected_indices(records, selection, 'hash', 1), [1])
        for change in ({'val': ['a', 'a']}, {'val': ['a']}, {'val': ['a', 'missing']},
                       {'manifest_sha256': 'changed'}, {'train': ['a']}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                selected_indices(records, {**selection, **change}, 'hash', 2)
        records[1]['split'] = 'test'
        with self.assertRaises(ValueError):
            selected_indices(records, selection, 'hash', 2)

    def test_module_hash_detects_changes_and_summary_weights_frames(self):
        first = module_hashes(self.model)
        with torch.no_grad():
            self.model.semantic_mean[0] += 1
        second = module_hashes(self.model)
        self.assertNotEqual(first['_root_buffers'], second['_root_buffers'])
        self.assertEqual(first['codec_generator'], second['codec_generator'])
        examples = []
        for length, mse, unit in ((1, 1., 0), (3, 3., 1)):
            metrics = {name: {'clusters': 2, 'frames': length, 'feature_elements': length * 768,
                'normalized_mse': mse, 'normalized_squared_error_sum': mse * length * 768,
                'unit_ids': [unit] * length} for name in ARMS}
            examples.append({'unit_metrics': metrics})
        result = summarize_distortion(examples)
        self.assertEqual(result['current_units']['frame_weighted_normalized_mse'], 2.5)
        self.assertEqual(result['current_units']['example_mean_normalized_mse'], 2.)
        self.assertEqual(result['current_units']['used_units'], 2)


if __name__ == '__main__':
    unittest.main()
