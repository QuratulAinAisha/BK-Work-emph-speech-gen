"""Keep sampler retests matched and A-only. / 샘플러 재검사의 조건과 A 전용 입력을 유지합니다."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from dataset.quality_speech_dataset import collate_quality, person_a_only
from model.full_speech.tensor_ops import mask_from_lengths
from model.full_speech.units import UnitSpeechSystem
from scripts import check_planner_repair_sampling as diagnostic
from tests.test_quality_speech import FakeCodec, config, sample


class RepairSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(20)
        cfg = config()
        cfg.planner_memory_mode = 'native_speech'
        self.model = UnitSpeechSystem(cfg, torch.randn(8, 768)).eval()
        self.batch, self.donor = collate_quality([sample(0)]), collate_quality([sample(1)])

    def test_unit_pair_is_matched_b_free_and_checks_production_parity(self):
        with patch.object(self.model, 'encode_batch', wraps=self.model.encode_batch) as encode:
            result = diagnostic.evaluate_unit_pair(self.model, self.batch, self.donor, verify_parity=True)
        self.assertTrue(result['greedy_production_parity_checked'])
        self.assertEqual(len(result['conditions']), 12)
        for call in encode.call_args_list:
            self.assertFalse({'semantic', 'text_b', 'affect', 'codec', 'duration', 'waveform'} & call.args[0].keys())
            torch.testing.assert_close(call.args[0]['style_id'], self.batch['style_id'])
            torch.testing.assert_close(call.args[0]['speaker_id'], self.batch['speaker_id'])
        for name, entry in result['conditions'].items():
            self.assertEqual(entry['planner_forward_count'], diagnostic.VARIANTS[name.split('/')[1]]['steps'])
            self.assertEqual(entry['planner_memory_frames'], 25)
        for key in ('semantic', 'affect', 'waveform', 'codec', 'text_b'):
            self.batch[key].fill_(123)
            self.donor[key].fill_(321)
        changed = diagnostic.evaluate_unit_pair(self.model, self.batch, self.donor)
        for name, entry in result['conditions'].items():
            self.assertEqual(entry['generated_unit_ids'], changed['conditions'][name]['generated_unit_ids'])
        summary = diagnostic.summarize_units([result])
        self.assertEqual(summary['count'], 1)
        self.assertEqual(summary['summary']['correct_a/greedy_8']['changed_unit_fraction_vs_correct_a'], 0.)
        self.assertEqual(summary['summary']['correct_a/greedy_8']['frame_unit_accuracy'],
                         result['conditions']['correct_a/greedy_8']['correct_units'] / result['frames'])

    def test_audio_variants_share_noise_duration_and_ignore_all_b_targets(self):
        original_steps = self.model.config.semantic_steps
        weights = {key: value.clone() for key, value in self.model.state_dict().items()}
        states = []
        original_sample = self.model.codec_generator.sample

        def capture_noise(*args, **kwargs):
            states.append(args[-1].get_state().clone())
            return original_sample(*args, **kwargs)

        with patch.object(self.model.codec_generator, 'sample', side_effect=capture_noise):
            result = diagnostic.generate_audio_variants(self.model, self.batch, FakeCodec())
        self.assertEqual(len(states), 8)  # Six variants and two parity checks. / 여섯 조건과 동일성 검사 두 번입니다.
        self.assertTrue(all(torch.equal(states[0], state) for state in states))
        self.assertEqual(self.model.config.semantic_steps, original_steps)
        self.assertTrue(result['greedy_production_parity_checked'])
        lengths = {len(value['waveform']) for value in result['variants'].values()}
        self.assertEqual(len(lengths), 1)
        for key in ('semantic', 'affect', 'affect_weight', 'waveform', 'codec', 'text_b', 'duration'):
            self.batch[key].fill_(999)
        changed = diagnostic.generate_audio_variants(self.model, self.batch, FakeCodec())
        self.assertEqual(result['predicted_duration_seconds'], changed['predicted_duration_seconds'])
        for name, entry in result['variants'].items():
            np.testing.assert_array_equal(entry['waveform'], changed['variants'][name]['waveform'])
            self.assertEqual(entry['generated_unit_ids'], changed['variants'][name]['generated_unit_ids'])
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, weights[key], rtol=0, atol=0)

    def test_shared_uncapped_asr_uses_the_same_conversations(self):
        names = list(diagnostic.VARIANTS)
        rows = []
        for index in range(3):
            paths = {name: {'reference_wer': index + .2, 'asr_token_limit_reached': False,
                            'raw_clip_fraction': 0.} for name in names}
            rows.append({'conversation_id': str(index), 'paths': paths})
        rows[0]['paths'][names[1]]['asr_token_limit_reached'] = True
        rows[1]['paths'][names[2]]['asr_token_limit_reached'] = True
        report = diagnostic.summarize_asr({'examples': rows, 'summary': {name: {} for name in names}})
        self.assertEqual(report['shared_uncapped_asr']['conversation_ids'], ['2'])
        self.assertTrue(all(value == 2.2 for value in report['shared_uncapped_asr']['mean_reference_wer'].values()))
        self.assertEqual(report['summary'][names[1]]['uncapped_count'], 2)

    def test_cli_saves_two_unit_cases_and_eight_audio_cases_then_reuses_complete_report(self):
        class Dataset:
            def __init__(self):
                self.records = [{'path': str(index), 'conversation_id': str(index), 'input_text': 'hello',
                                 'response_text': 'hello there', 'style_id': 0, 'speaker_id': 0} for index in range(8)]
                self.samples = [sample(0) for _ in range(8)]
                for index, value in enumerate(self.samples):
                    value['conversation_key'] = torch.tensor(index)

            def __getitem__(self, index):
                return self.samples[index]

        class Codec(FakeCodec):
            def to(self, device):
                return self

            def eval(self):
                return self

        def recognize(report, output, device):
            for row in report['examples']:
                for entry in row['paths'].values():
                    self.assertTrue((output / entry['file']).exists())
                    entry.update(asr='hello there', reference_wer=0., asr_token_limit_reached=False)
            report['summary'] = {name: {'mean_reference_wer': 0.} for name in report['examples'][0]['paths']}
            return report

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, manifest, selection = (root / name for name in ('model.pt', 'manifest.json', 'selection.json'))
            checkpoint.write_bytes(b'fixed checkpoint')
            manifest.write_text('{}')
            selection.write_text(json.dumps({'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
                                             'val': [str(index) for index in range(8)]}))
            output = root / 'output'
            argv = ['check', '--checkpoint', str(checkpoint), '--manifest', str(manifest), '--selection', str(selection),
                    '--audio-selection', str(selection), '--output', str(output), '--count', '2', '--device', 'cpu']
            with patch('sys.argv', argv), patch.object(diagnostic.UnitSpeechSystem, 'from_checkpoint', return_value=(self.model, {})), \
                    patch.object(diagnostic, 'QualitySpeechDataset', return_value=Dataset()), \
                    patch.object(diagnostic, 'FrozenEncodec', Codec), patch.object(diagnostic, 'recognize', side_effect=recognize) as asr, \
                    patch('builtins.print'):
                diagnostic.main()
                self.assertEqual(asr.call_count, 1)
                diagnostic.main()
                self.assertEqual(asr.call_count, 1)
            units = json.loads((output / 'units_report.json').read_text())
            audio = json.loads((output / 'audio/report.json').read_text())
            self.assertEqual(units['count'], 2)
            self.assertEqual(len(audio['examples']), 8)
            self.assertEqual(audio['shared_uncapped_asr']['count'], 8)
            self.assertEqual(len(list((output / 'audio').glob('*.wav'))), 56)

    def test_onepass_matches_fully_hidden_argmax(self):
        with torch.inference_mode():
            inputs = person_a_only(self.batch)
            encoded = self.model.encode_batch(inputs)
            style, _ = self.model.embeddings(inputs['style_id'], inputs['speaker_id'], 1)
            mask = mask_from_lengths(self.batch['semantic_len'])
            logits = self.model.semantic_planner.logits(torch.zeros_like(mask, dtype=torch.long), mask, mask,
                style=style, **self.model.planner_inputs(encoded))
        result = diagnostic.evaluate_unit_pair(self.model, self.batch, self.donor)
        onepass = result['conditions']['correct_a/greedy_1']
        self.assertEqual(onepass['generated_unit_ids'], logits.argmax(-1)[mask].tolist())
        self.assertEqual(onepass['planner_forward_count'], 1)
        self.assertEqual(onepass['trace'][0]['predicted_positions'], result['frames'])
        self.assertEqual(onepass['trace'][0]['still_hidden'], 0)
        self.assertEqual(onepass['changed_units_vs_onepass'], 0)


if __name__ == '__main__':
    unittest.main()
