"""Causal shifting, experimental isolation and exact resume. / 인과 이동, 실험 격리와 정확한 재개 검증."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from dataset.quality_speech_dataset import collate_quality, person_a_only
from model.full_speech.experimental_ar_units import (AR_ARCHITECTURE, ar_system_from_payload, load_ar_checkpoint)
from model.full_speech.loading import load_response_model
from model.full_speech.tensor_ops import counts, mask_from_lengths
from model.full_speech.units import UnitSpeechSystem
from scripts.train_planner_ar import (SelectedSamples, edit_distance, evaluate, evaluate_free, file_hash,
    frozen_audit, frozen_hashes, parse_args, selected_records, train)
from tests.test_quality_speech import FakeCodec, config, sample
from utils.distributed_training import DistributedRuntime


class PlannerARTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(14)
        self.source = UnitSpeechSystem(config(), torch.randn(8, 768)).eval()
        self.payload = self.source.checkpoint(0, recovery_recipe={'codebook_sha256': 'a' * 64})
        self.model = ar_system_from_payload(self.payload, allow_masked_initialization=True)
        self.ids = torch.tensor([[1, 2, 3, 4, 5, 6], [3, 4, 5, 0, 0, 0]])
        self.mask = mask_from_lengths(torch.tensor([6, 3]))
        self.conditions = self.model.planner_conditions({}, 2, 'prior')

    def test_masked_prior_weights_copy_exactly_and_causal_shift_has_no_future_leakage(self):
        for key, value in self.source.state_dict().items():
            torch.testing.assert_close(self.model.state_dict()[key], value, rtol=0, atol=0)
        planner = self.model.semantic_planner
        with torch.no_grad():
            baseline = planner.teacher_logits(self.ids, self.mask, **self.conditions)
            for position in range(self.ids.shape[1]):
                changed = self.ids.clone()
                changed[:, position:] = (changed[:, position:] + 3) % 8
                result = planner.teacher_logits(changed, self.mask, **self.conditions)
                torch.testing.assert_close(result[:, :position + 1], baseline[:, :position + 1], rtol=0, atol=0)
            changed = self.ids.clone()
            changed[:, 0] = (changed[:, 0] + 1) % 8
            result = planner.teacher_logits(changed, self.mask, **self.conditions)
            self.assertFalse(torch.equal(result[0, 1], baseline[0, 1]))
            self.assertTrue((baseline[~self.mask] == 0).all())

    def test_teacher_prefix_matches_incremental_eval_with_fixed_total_length(self):
        planner = self.model.semantic_planner
        with torch.no_grad():
            teacher = planner.teacher_logits(self.ids, self.mask, **self.conditions)
            for position in range(self.ids.shape[1]):
                prefix = planner.prefix_logits(self.ids[:, :position], self.mask, **self.conditions)
                torch.testing.assert_close(prefix[:, -1], teacher[:, position], rtol=2e-5, atol=2e-6)
            with patch.object(planner.denoiser, 'forward', wraps=planner.denoiser.forward) as spy:
                generated = planner.sample_ids(self.mask, **self.conditions)
            self.assertEqual(len(spy.call_args_list), 6)
            for call in spy.call_args_list:
                torch.testing.assert_close(call.args[-1], torch.tensor([6, 3]), rtol=0, atol=0)
            self.assertTrue((generated[~self.mask] == 0).all())
            # A free prefix must predict itself consistently. / 자유 생성된 접두사도 같은 다음 단위를 예측해야 합니다.
            teacher_generated = planner.teacher_logits(generated, self.mask, **self.conditions).argmax(-1)
            torch.testing.assert_close(teacher_generated[self.mask], generated[self.mask], rtol=0, atol=0)

    def test_prior_ignores_a_features_and_only_planner_receives_gradients(self):
        self.model.configure_ar_training('prior')
        self.model.eval()
        batch = collate_quality([sample(0), sample(1)])
        with patch.object(self.model, 'encode_batch', side_effect=AssertionError('Prior read A')):
            first = self.model.losses(batch)['total']
            for key in ('mel', 'dmm', 'au', 'speech_a', 'text_a', 'duration', 'text_b', 'affect', 'codec', 'waveform'):
                batch[key].fill_(999)
            second = self.model.losses(batch)['total']
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        before = frozen_hashes(self.model)
        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=.001)
        second.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.semantic_planner.parameters()))
        optimizer.step()
        self.assertTrue(frozen_audit(self.model, before)['passed'])

    def test_normal_generation_is_a_only_and_uses_predicted_unit_count(self):
        self.model.configure_ar_training('conditional')
        self.model.eval()
        batch = collate_quality([sample()])
        inputs = person_a_only(batch)
        seen = []
        encode = self.model.encode_batch

        def inspect(received):
            self.assertEqual(set(received), set(inputs))
            seen.append(received)
            return encode(received)

        with patch.object(self.model, 'encode_batch', side_effect=inspect):
            first = self.model.generate_batch(inputs, FakeCodec())
            for key in ('semantic', 'semantic_len', 'duration', 'text_b', 'codec', 'affect', 'waveform'):
                batch[key].fill_(999)
            second = self.model.generate_batch(person_a_only(batch), FakeCodec())
        self.assertEqual(len(seen), 2)
        for key in ('semantic', 'codec_latents', 'waveform', 'duration', 'audio_lengths'):
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
        self.assertEqual(first['semantic'].shape[1], int(counts(first['duration'], 50)[0]))
        with self.assertRaisesRegex(ValueError, 'only Person A'):
            self.model.generate_batch(batch, FakeCodec())

    def test_conditional_teacher_encodes_only_a_and_keeps_acoustics_frozen(self):
        self.model.configure_ar_training('conditional')
        self.model.train()
        batch = collate_quality([sample()])
        before = frozen_hashes(self.model)
        encode = self.model.encode_batch

        def inspect(received):
            self.assertEqual(set(received), set(person_a_only(batch)))
            self.assertFalse(self.model.encoder.training)
            self.assertFalse(self.model.codec_generator.training)
            return encode(received)

        with patch.object(self.model, 'encode_batch', side_effect=inspect) as spy:
            loss = self.model.losses(batch)['total']
        self.assertEqual(spy.call_count, 1)
        loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in self.model.semantic_planner.parameters()
                               if p.grad is not None), 0)
        self.assertTrue(frozen_audit(self.model, before)['passed'])
        self.assertFalse(any(p.requires_grad for p in self.model.length_predictor.parameters()))

    def test_explicit_checkpoint_roundtrip_and_generic_loader_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'ar.pt'
            torch.save(self.model.checkpoint(7), path)
            loaded, payload = load_ar_checkpoint(path)
            self.assertEqual(payload['architecture'], AR_ARCHITECTURE)
            with self.assertRaisesRegex(ValueError, 'Unsupported response architecture'):
                load_response_model(path)
            with torch.no_grad():
                expected = self.model.semantic_planner.teacher_logits(self.ids, self.mask, **self.conditions)
                actual = loaded.semantic_planner.teacher_logits(self.ids, self.mask, **self.conditions)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            payload['experimental_ar']['BOS'] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'descriptor'):
                ar_system_from_payload(payload)

    def test_categorical_private_rng_is_repeatable_and_preserves_codec_noise(self):
        planner = self.model.semantic_planner
        planner.configure_sampling('categorical', .8, 8, 42)
        with patch.object(planner, 'prefix_logits', side_effect=lambda ids, *args, **kwargs:
                          torch.zeros(len(ids), ids.shape[1] + 1, 8)):
            global_before = torch.get_rng_state()
            a = planner.sample_ids(self.mask, **self.conditions)
            b = planner.sample_ids(self.mask, **self.conditions)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(torch.get_rng_state(), global_before, rtol=0, atol=0)
            planner.configure_sampling('categorical', .8, 8, 43)
            c = planner.sample_ids(self.mask, **self.conditions)
            self.assertFalse(torch.equal(a, c))
        batch = collate_quality([sample()])
        noises = []
        original = torch.randn

        def capture(*args, **kwargs):
            result = original(*args, **kwargs)
            if kwargs.get('generator') is not None:
                noises.append(result.clone())
            return result

        with patch('torch.randn', side_effect=capture):
            for mode, seed in [('greedy', 42), ('categorical', 42), ('categorical', 43)]:
                planner.configure_sampling(mode, .8, 8, seed)
                self.model.generate_batch(person_a_only(batch), FakeCodec(), seed=42)
        self.assertEqual(len(noises), 3)
        for noise in noises[1:]:
            torch.testing.assert_close(noise, noises[0], rtol=0, atol=0)

    def test_free_shuffled_control_locks_original_length_style_and_a_only_inputs(self):
        samples = [sample(0), sample(1)]

        class Data:
            records = [{'path': str(i), 'conversation_id': str(i)} for i in range(2)]
            def __len__(self):
                return 2
            def batch(self, indices, device):
                return collate_quality([samples[i] for i in indices])

        self.model.ar_phase = 'conditional'
        calls = []
        def sample_ids(mask, **conditions):
            calls.append((mask.clone(), conditions))
            return torch.zeros_like(mask, dtype=torch.long)
        with patch.object(self.model.semantic_planner, 'sample_ids', side_effect=sample_ids), \
                patch.object(self.model, 'encode_batch', wraps=self.model.encode_batch) as encoder:
            result = evaluate_free(self.model, Data(), torch.device('cpu'), 1)
        self.assertEqual(len(calls), 2)
        torch.testing.assert_close(calls[0][0], calls[1][0], rtol=0, atol=0)
        torch.testing.assert_close(calls[0][1]['style'], calls[1][1]['style'], rtol=0, atol=0)
        self.assertFalse(torch.equal(calls[0][1]['context'], calls[1][1]['context']))
        for call in encoder.call_args_list:
            self.assertEqual(set(call.args[0]), set(person_a_only(collate_quality([samples[0]]))))
        row = result['examples'][0]
        self.assertEqual(row['shuffled_path'], '1')
        self.assertEqual(row['duration_source'], 'A_predicted_duration')
        self.assertFalse(row['uses_B_unit_inputs'])

    def test_inputs_unit_cache_matches_lazy_teacher_without_caching_b_features(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest, selection = root / 'manifest.json', root / 'selection.json'
            records = [{'path': str(i), 'conversation_id': str(i), 'split': split}
                       for i, split in enumerate(('train', 'val'))]
            manifest.write_text(json.dumps({'records': records}))
            selection.write_text(json.dumps({'manifest_sha256': file_hash(manifest), 'train': ['0'], 'val': ['1']}))
            samples = [sample(0), sample(1)]
            class FakeDataset:
                def __init__(self, path, cfg, split):
                    self.records = [row for row in records if row['split'] == split]
                def __getitem__(self, index):
                    return samples[int(self.records[index]['path'])]
            with patch('scripts.train_planner_ar.QualitySpeechDataset', FakeDataset):
                lazy = SelectedSamples(manifest, selection, self.model, 'train', 'conditional', 'lazy')
                cached = SelectedSamples(manifest, selection, self.model, 'train', 'conditional', 'inputs_units')
            expected = self.model.teacher_outputs(lazy.batch([0], torch.device('cpu')))
            actual = self.model.teacher_outputs(cached.batch([0], torch.device('cpu')))
            torch.testing.assert_close(actual['logits'], expected['logits'], rtol=0, atol=0)
            self.assertEqual(set(cached.cached[0]), {'mel', 'dmm', 'au', 'speech_a', 'style_id', 'speaker_id', 'unit_ids'})
            self.assertGreater(cached.profile['cached_tensor_bytes'], 0)
            records[1]['conversation_id'] = '0'
            manifest.write_text(json.dumps({'records': records}))
            selection.write_text(json.dumps({'manifest_sha256': file_hash(manifest), 'train': ['0'], 'val': ['1']}))
            with self.assertRaisesRegex(ValueError, 'conversations overlap'):
                selected_records(manifest, selection, 'train')

    def test_evaluation_refuses_existing_output_before_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            args = parse_args(['--phase', 'evaluate', '--checkpoint', 'missing.pt', '--manifest', 'missing.json',
                '--selection', 'missing.json', '--output', folder, '--device', 'cpu'])
            with self.assertRaisesRegex(ValueError, 'new output directory'):
                evaluate(args, DistributedRuntime(torch.device('cpu')))

    def test_two_update_training_resume_is_exact_with_frozen_components(self):
        class TinyData:
            def __init__(self, manifest, selection, model, split, phase, conditional_cache='lazy'):
                self.records = [{'path': str(i)} for i in range(4 if split == 'train' else 2)]
                self.profile = {'conditional_cache': conditional_cache}

            def __len__(self):
                return len(self.records)

            def batch(self, indices, device):
                rows = torch.tensor([[1, 2, 3, 4] if i % 2 else [3, 2, 5, 4] for i in indices], device=device)
                return {'unit_ids': rows, 'unit_len': torch.full((len(indices),), 4, device=device)}

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'masked.pt'
            torch.save(self.payload, source)
            for name in ('manifest.json', 'selection.json'):
                (root / name).write_text('{}')
            common = ['--phase', 'prior', '--manifest', str(root / 'manifest.json'),
                '--selection', str(root / 'selection.json'), '--steps', '2', '--batch-size', '2',
                '--evaluate-every', '1', '--device', 'cpu']
            whole = parse_args([*common, '--initialize', str(source), '--output', str(root / 'whole')])
            first = parse_args([*common, '--initialize', str(source), '--output', str(root / 'resumed'), '--stop-after', '1'])
            resume = parse_args([*common, '--resume', str(root / 'resumed/last.pt'), '--output', str(root / 'resumed')])
            runtime = DistributedRuntime(torch.device('cpu'))
            with patch('scripts.train_planner_ar.SelectedSamples', TinyData), \
                    patch('scripts.train_planner_ar.verify_feasibility_shape'), redirect_stdout(io.StringIO()):
                train(whole, runtime)
                train(first, runtime)
                self.assertFalse((root / 'resumed/complete.json').exists())
                train(resume, runtime)
            a = torch.load(root / 'whole/last.pt', weights_only=True)
            b = torch.load(root / 'resumed/last.pt', weights_only=True)
            self.assertEqual(a['ar_step'], 2)
            self.assertEqual(a['ar_frozen_hashes'], b['ar_frozen_hashes'])
            self.assertEqual(a['metadata']['recovery_recipe']['codebook_sha256'], 'a' * 64)
            for name, value in a['state_dict'].items():
                torch.testing.assert_close(b['state_dict'][name], value, rtol=0, atol=0)
            self.assertEqual(edit_distance([1, 2, 3], [1, 4]), 2)


if __name__ == '__main__':
    unittest.main()
