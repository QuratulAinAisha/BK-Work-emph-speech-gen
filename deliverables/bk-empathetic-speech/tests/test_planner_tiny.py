"""Tiny B-only training and restart contracts. / 소규모 B 전용 학습과 재시작 검증."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from model.full_speech.loading import load_response_model
from model.full_speech.units import UnitSpeechSystem
from model.full_speech.experimental_unit_planners import (EXPERIMENTAL_ARCHITECTURE,
    experimental_descriptor, load_experimental_unit_checkpoint, replace_tiny_planner)
from scripts.train_planner_tiny import (make_hidden, collate_units, b_only_logits, nearest_visible_copy,
    configure_training, extract_training_units, parse_args, train, digest)
from tests.test_quality_speech import config


class TinyPlannerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)
        self.model = UnitSpeechSystem(config(), torch.randn(8, 768))
        self.records = [{'ids': torch.tensor([1, 1, 2, 2, 3, 3])},
                        {'ids': torch.tensor([4, 5, 5, 4])}]

    def test_masks_are_repeatable_partial_and_gaps_contiguous(self):
        for length in (2, 6, 31):
            for kind in ('random', 'gap'):
                for ratio in (None, .25, .5, .75):
                    mask = make_hidden(length, 42, kind, ratio)
                    torch.testing.assert_close(mask, make_hidden(length, 42, kind, ratio))
                    self.assertTrue(mask.any())
                    self.assertTrue((~mask).any())
                    if kind == 'gap':
                        selected = mask.nonzero(as_tuple=True)[0]
                        self.assertEqual(int(selected[-1] - selected[0] + 1), int(mask.sum()))
        self.assertFalse(torch.equal(make_hidden(31, 42), make_hidden(31, 43)))

    def test_padding_hidden_targets_and_nonplanner_parameters_are_isolated(self):
        configure_training(self.model, 'off')
        masks = [make_hidden(len(row['ids']), 42) for row in self.records]
        ids, valid, hidden = collate_units(self.records, [0, 1], masks, torch.device('cpu'))
        with patch.object(self.model.semantic_planner, 'logits', wraps=self.model.semantic_planner.logits) as spy:
            logits = b_only_logits(self.model, ids, valid, hidden)
        self.assertTrue((spy.call_args.args[0][hidden | ~valid] == 0).all())
        self.assertTrue((spy.call_args.args[3] == 0).all())
        self.assertTrue((spy.call_args.args[5] == 0).all())
        self.assertTrue((spy.call_args.args[6] == 0).all())
        self.assertTrue((logits[~valid] == 0).all())
        loss = torch.nn.functional.cross_entropy(logits[hidden], ids[hidden])
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in self.model.semantic_planner.parameters()))
        self.assertTrue(all(p.grad is None for name, p in self.model.named_parameters()
                            if not name.startswith('semantic_planner.')))

    def test_nearest_copy_does_not_read_hidden_targets_and_ties_choose_left(self):
        ids = torch.tensor([[0, 0, 8, 0, 0, 0, 9, 0]])
        valid = torch.ones_like(ids, dtype=torch.bool)
        hidden = torch.tensor([[True, True, False, True, True, True, False, True]])
        expected = nearest_visible_copy(ids, valid, hidden)
        ids[hidden] = 100
        torch.testing.assert_close(nearest_visible_copy(ids, valid, hidden), expected)
        self.assertEqual(expected.tolist(), [[8, 8, 8, 8, 8, 9, 9, 9]])

    def test_learned_ids_preserve_initial_logits_and_receive_gradients(self):
        self.model.eval()
        masks = [make_hidden(len(row['ids']), 42) for row in self.records]
        ids, valid, hidden = collate_units(self.records, [0, 1], masks, torch.device('cpu'))
        before = b_only_logits(self.model, ids, valid, hidden).detach()
        replace_tiny_planner(self.model, 'learned_ids')
        configure_training(self.model, 'off')
        logits = b_only_logits(self.model, ids, valid, hidden)
        torch.testing.assert_close(before, logits, rtol=0, atol=0)
        torch.nn.functional.cross_entropy(logits[hidden], ids[hidden]).backward()
        self.assertGreater(float(self.model.semantic_planner.unit_embedding.weight.grad.abs().sum()), 0.)
        self.assertFalse(self.model.semantic_planner.codebook.centers.requires_grad)

    def test_plain_transformer_rejects_conditional_response_use(self):
        replace_tiny_planner(self.model, 'plain_transformer')
        configure_training(self.model, 'off')
        masks = [make_hidden(len(row['ids']), 42) for row in self.records]
        ids, valid, hidden = collate_units(self.records, [0, 1], masks, torch.device('cpu'))
        logits = b_only_logits(self.model, ids, valid, hidden)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue((logits[~valid] == 0).all())
        torch.nn.functional.cross_entropy(logits[hidden], ids[hidden]).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in self.model.semantic_planner.parameters()))
        with self.assertRaisesRegex(ValueError, 'B-only'):
            self.model.semantic_planner.logits(ids, hidden, valid, torch.ones(2, 1, 512),
                torch.ones(2, 1, dtype=torch.bool), torch.zeros(2, 1, 6), torch.zeros(2, 32))

    def test_experimental_headers_require_the_explicit_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            for variant in ('learned_ids', 'plain_transformer'):
                model = UnitSpeechSystem(config(), self.model.semantic_planner.codebook.centers)
                replace_tiny_planner(model, variant)
                model.eval()
                masks = [make_hidden(len(row['ids']), 42) for row in self.records]
                ids, valid, hidden = collate_units(self.records, [0, 1], masks, torch.device('cpu'))
                expected = b_only_logits(model, ids, valid, hidden)
                payload = model.checkpoint(recovery_recipe={'codebook_sha256': 'a' * 64})
                payload['architecture'] = EXPERIMENTAL_ARCHITECTURE
                payload['experimental_planner'] = experimental_descriptor(model, variant)
                path = Path(directory) / (variant + '.pt')
                torch.save(payload, path)
                restored, saved = load_experimental_unit_checkpoint(path)
                torch.testing.assert_close(expected, b_only_logits(restored, ids, valid, hidden), rtol=0, atol=0)
                self.assertEqual(saved['experimental_planner']['variant'], variant)
                with self.assertRaisesRegex(ValueError, 'Unsupported response architecture'):
                    load_response_model(path)

    def write_fixture(self, folder):
        records = []
        centers = self.model.semantic_planner.codebook.centers.cpu().numpy()
        for index in range(8):
            ids = np.array([(index + j // 2) % 8 for j in range(6 + index % 3)])
            name = f'train_{index}.npz'
            np.savez(folder / name, semantic=centers[ids])
            records.append({'path': name, 'split': 'train', 'conversation_id': str(index)})
        records.append({'path': 'do_not_read_test.npz', 'split': 'test', 'conversation_id': 'test'})
        manifest = folder / 'manifest.json'
        manifest.write_text(json.dumps({'records': records, 'target_contract': self.model.config.target_contract()}))
        selection = folder / 'selection.json'
        selection.write_text(json.dumps({'manifest_sha256': digest(manifest),
            'train': [row['path'] for row in records[:-1]], 'val': ['do_not_read_val.npz']}))
        checkpoint = folder / 'initialize.pt'
        torch.save(self.model.checkpoint(recovery_recipe={'codebook_sha256': 'a' * 64}), checkpoint)
        return manifest, selection, checkpoint

    def test_extract_reads_only_train_b_semantics_and_checks_split(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            manifest, selection, _ = self.write_fixture(folder)
            records = extract_training_units(manifest, selection, 8, self.model, torch.device('cpu'))
            self.assertEqual(len(records), 8)
            self.assertEqual(records[0]['ids'].tolist(), [0, 0, 1, 1, 2, 2])
            wrong = json.loads(selection.read_text()); wrong['train'][0] = 'do_not_read_test.npz'
            selection.write_text(json.dumps(wrong))
            with self.assertRaisesRegex(ValueError, 'training rows'):
                extract_training_units(manifest, selection, 8, self.model, torch.device('cpu'))

    def test_checkpoint_load_and_exact_two_stage_resume(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            folder = Path(directory)
            manifest, selection, checkpoint = self.write_fixture(folder)
            base = ['--manifest', str(manifest), '--selection', str(selection), '--device', 'cpu',
                    '--count', '8', '--batch-size', '4', '--fixed-updates', '1', '--fresh-updates', '1',
                    '--evaluate-every', '1', '--save-every', '1']
            full = folder / 'full'
            train(parse_args(base + ['--initialize', str(checkpoint), '--output', str(full)]))
            resumed = folder / 'resumed'
            train(parse_args(base + ['--initialize', str(checkpoint), '--output', str(resumed), '--stop-after', '1']))
            train(parse_args(base + ['--resume', str(resumed / 'last.pt'), '--output', str(resumed)]))
            first, _ = load_response_model(full / 'last.pt')
            second, payload = load_response_model(resumed / 'last.pt')
            for name, value in first.state_dict().items():
                torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)
            self.assertEqual(payload['tiny_step'], 2)
            self.assertEqual(payload['tiny_phase'], 'fresh')
            self.assertEqual(payload['metadata']['recovery_recipe']['codebook_sha256'], 'a' * 64)
            self.assertTrue(json.loads((resumed / 'complete.json').read_text())['all_frozen_audits_passed'])
            updates = [json.loads(line) for line in (resumed / 'updates.jsonl').read_text().splitlines()]
            self.assertEqual([row['phase'] for row in updates], ['fixed', 'fresh'])
            self.assertTrue(all(row['gradient_norm_before_clip'] > 0 for row in updates))

    def test_experimental_script_resume_preserves_descriptor_and_codebook_hash(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            folder = Path(directory)
            manifest, selection, checkpoint = self.write_fixture(folder)
            base = ['--manifest', str(manifest), '--selection', str(selection), '--device', 'cpu',
                    '--count', '8', '--batch-size', '4', '--fixed-updates', '1', '--fresh-updates', '1',
                    '--evaluate-every', '1', '--save-every', '1', '--planner-variant', 'plain_transformer']
            output = folder / 'plain'
            train(parse_args(base + ['--initialize', str(checkpoint), '--output', str(output), '--stop-after', '1']))
            train(parse_args(base + ['--resume', str(output / 'last.pt'), '--output', str(output)]))
            _, payload = load_experimental_unit_checkpoint(output / 'last.pt')
            self.assertEqual(payload['tiny_step'], 2)
            self.assertEqual(payload['experimental_planner']['variant'], 'plain_transformer')
            self.assertEqual(payload['metadata']['recovery_recipe']['codebook_sha256'], 'a' * 64)
            runtime = json.loads((output / 'runtime.json').read_text())
            self.assertEqual(runtime['measured_updates'], 2)
            self.assertGreater(runtime['optimizer_update_wall_seconds'], 0.)
            self.assertFalse(runtime['production_loader_compatible'])


if __name__ == '__main__':
    unittest.main()
