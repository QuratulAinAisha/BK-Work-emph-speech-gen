"""Check frozen feature views and CTC probe measurements. / 고정 특징과 CTC 측정을 검증합니다."""

import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from dataset.quality_speech_dataset import collate_quality, person_a_only
from model.full_speech.quality import ContentHead, content_ctc, text_ids
from model.full_speech.units import UnitSpeechSystem
from scripts.check_a_information import (VIEWS, aggregate_text, collate_features, compatible_provenance,
    epoch_order, evaluate_probe, extract_feature_views, greedy_text, labels_text, required_ctc_frames,
    text_metrics, train_view, trim_feature)
from tests.test_quality_speech import config, sample


class AInformationProbeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)

    def test_ctc_collapse_repeats_and_feasibility(self):
        a, b = text_ids('ab').tolist()
        self.assertEqual(greedy_text([a, a, 0, a, b, b, 0]), 'aab')
        self.assertEqual(required_ctc_frames(text_ids('letter')), 7)
        with self.assertRaises(ValueError):
            required_ctc_frames(torch.tensor([], dtype=torch.long))
        with self.assertRaises(ValueError):
            labels_text(torch.tensor([0]))

    def test_metrics_keep_corpus_and_example_weighting_distinct(self):
        rows = [text_metrics('a', ''), text_metrics('b c d', 'b c d')]
        metrics = aggregate_text(rows)
        self.assertEqual(metrics['mean_example_wer'], .5)
        self.assertEqual(metrics['corpus_wer'], .25)
        self.assertEqual(metrics['corpus_cer'], 1 / 6)

    def test_extraction_uses_a_only_and_preposition_projection(self):
        model = UnitSpeechSystem(config(), torch.randn(8, 768)).eval().requires_grad_(False)
        examples = [sample(0), sample(1)]
        examples[1]['speech_a'] = examples[1]['speech_a'][:19]
        batch = collate_quality(examples)
        state = {key: value.clone() for key, value in model.state_dict().items()}
        original = model.encode_batch

        def checked(inputs):
            self.assertEqual(set(inputs), set(person_a_only(batch)))
            return original(inputs)

        with patch.object(model, 'encode_batch', side_effect=checked):
            views = extract_feature_views(model, batch)
        projected, mask = views['projected']
        torch.testing.assert_close(projected, model.speech_projection(batch['speech_a']))
        self.assertEqual(mask.sum(1).tolist(), [25, 19])
        self.assertFalse(torch.equal(views['fused'][1].sum(1), mask.sum(1)))
        for view, (values, valid) in views.items():
            feature = trim_feature(values, valid, 1, VIEWS[view])
            self.assertEqual(len(feature), int(valid[1].sum()))
            self.assertFalse(feature.requires_grad)
        altered = copy.deepcopy(batch)
        altered['semantic'].fill_(999)
        altered['text_b'].fill_(1)
        second = extract_feature_views(model, altered)
        for view in views:
            torch.testing.assert_close(views[view][0], second[view][0], rtol=0, atol=0)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(state[key], value))

    def test_bad_mask_and_changed_provenance_rejected(self):
        with self.assertRaisesRegex(ValueError, 'valid prefixes'):
            trim_feature(torch.randn(1, 3, 512), torch.tensor([[True, False, True]]), 0, 512)
        with self.assertRaisesRegex(ValueError, 'provenance differs'):
            compatible_provenance({'checkpoint_sha256': 'changed'}, {'checkpoint_sha256': 'original'})
        self.assertEqual(epoch_order(8, 2), epoch_order(8, 2))
        self.assertNotEqual(epoch_order(8, 2), epoch_order(8, 3))

    def test_fresh_probe_gradients_and_training_resume_match(self):
        records = []
        for position in range(4):
            records.append({'features': torch.randn(8 + position, 512), 'text_a': text_ids('hi'),
                'length': 8 + position, 'path': f'row{position}', 'conversation_id': f'conversation{position}'})
        head = ContentHead(512, hidden=256)
        batch = collate_features(records[:2], 'cpu')
        content_ctc(head, *batch).backward()
        self.assertTrue(all(item['features'].grad is None for item in records))
        self.assertGreater(float(head.output.weight.grad.abs().sum()), 0)
        metrics = evaluate_probe(head, records, 'cpu', 2)
        self.assertEqual(metrics['count'], 4)
        self.assertTrue(torch.isfinite(torch.tensor(metrics['ctc_loss'])))
        cached = {'train': records, 'val': records[:2]}
        metadata = {'provenance': {'test': True}}
        with tempfile.TemporaryDirectory() as folder:
            first = SimpleNamespace(output=Path(folder) / 'continuous', train_view='projected',
                lr=.001, batch_size=2, resume=False, device='cpu', epochs=2)
            second = SimpleNamespace(**vars(first))
            second.output, second.epochs = Path(folder) / 'resumed', 1
            with patch('scripts.check_a_information.load_cached', return_value=(cached, metadata)):
                train_view(first)
                train_view(second)
                second.epochs, second.resume = 2, True
                train_view(second)
            full = torch.load(first.output / 'probes/projected/last.pt', weights_only=True)
            resumed = torch.load(second.output / 'probes/projected/last.pt', weights_only=True)
            for key in full['state_dict']:
                torch.testing.assert_close(full['state_dict'][key], resumed['state_dict'][key], rtol=0, atol=0)
            self.assertTrue((second.output / 'probes/projected/completed_epoch_0001.json').exists())


if __name__ == '__main__':
    unittest.main()
