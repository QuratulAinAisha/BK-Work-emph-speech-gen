"""Protect controlled affect diagnostics from leakage. / 정서 대조 실험의 정답 누출을 방지합니다."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from dataset.audio_affect_targets import audio_affect_targets
from dataset.quality_speech_dataset import collate_quality, person_a_only
from model.full_speech.quality import QualitySpeechSystem
from model.full_speech.tensor_ops import align
from model.full_speech.units import UnitSpeechSystem
from scripts.check_response_duration_affect import (
    CONDITIONS, DURATION_POLICIES, control_waveform_metrics, generate_conditions,
    override_encoded_affect, selected_indices, summarize, waveform_metrics,
)
from scripts import check_response_duration_affect as diagnostic
from tests.test_quality_speech import FakeCodec, config, sample


class DurationAffectTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(31)
        self.batch = collate_quality([sample(0)])
        donor = sample(1)
        donor.update(mel=donor['mel'][:28], dmm=donor['dmm'][:8], au=donor['au'][:8], speech_a=donor['speech_a'][:19])
        self.donor = collate_quality([donor])

    def test_baseline_matches_production_and_only_affect_changes(self):
        for unit in (False, True):
            cfg = config()
            cfg.planner_memory_mode = 'native_speech'
            model = (UnitSpeechSystem(cfg, torch.randn(8, 768)) if unit else QualitySpeechSystem(cfg)).eval()
            state = {key: value.clone() for key, value in model.state_dict().items()}
            inputs = person_a_only(self.batch)
            with torch.inference_mode():
                expected = model.generate_batch(inputs, FakeCodec(), seed=42)
                shifted = person_a_only(self.donor)
                shifted.update(style_id=inputs['style_id'], speaker_id=inputs['speaker_id'])
                donor_encoded = model.encode_batch(shifted)
            original_method = model.encode_batch.__func__
            with patch.object(model.semantic_planner, 'sample', wraps=model.semantic_planner.sample) as planner:
                result = generate_conditions(model, self.batch, self.donor, FakeCodec())
            baseline = result['results']['a_predicted_duration_locked__predicted_affect']
            torch.testing.assert_close(baseline['waveform'], expected['waveform'], rtol=0, atol=0)
            torch.testing.assert_close(result['affects']['shuffled_affect'],
                align(donor_encoded['affect'], donor_encoded['context_mask'], expected['context_mask']), rtol=0, atol=0)
            self.assertFalse(result['affects']['zero_affect'].any())
            self.assertEqual(len(planner.call_args_list), 6)
            for number, (name, generated) in enumerate(result['results'].items()):
                for key in ('context', 'context_mask', 'speech_hidden', 'speech_mask'):
                    torch.testing.assert_close(generated[key], expected[key], rtol=0, atol=0)
                torch.testing.assert_close(planner.call_args_list[number].kwargs['context'], expected['speech_hidden'], rtol=0, atol=0)
                policy, condition = name.split('__')
                torch.testing.assert_close(generated['affect'], result['affects'][condition], rtol=0, atol=0)
                duration = expected['duration'] if policy == DURATION_POLICIES[0] else self.batch['duration']
                torch.testing.assert_close(generated['duration'], duration, rtol=0, atol=0)
            self.assertIs(model.encode_batch.__func__, original_method)
            self.assertNotIn('encode_batch', model.__dict__)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, state[key], rtol=0, atol=0)

    def test_b_target_mutations_cannot_change_normal_variants(self):
        model = UnitSpeechSystem(config(), torch.randn(8, 768)).eval()
        first = generate_conditions(model, self.batch, self.donor, FakeCodec())
        for batch in (self.batch, self.donor):
            for key in ('semantic', 'codec', 'affect', 'affect_weight', 'waveform', 'text_b'):
                batch[key].fill_(999)
        self.batch['duration'].fill_(.52)
        second = generate_conditions(model, self.batch, self.donor, FakeCodec())
        for condition in CONDITIONS:
            name = DURATION_POLICIES[0] + '__' + condition
            torch.testing.assert_close(first['results'][name]['waveform'], second['results'][name]['waveform'], rtol=0, atol=0)
            torch.testing.assert_close(first['predicted_durations'][condition], second['predicted_durations'][condition], rtol=0, atol=0)
        oracle = DURATION_POLICIES[1] + '__predicted_affect'
        self.assertNotEqual(first['results'][oracle]['waveform'].shape, second['results'][oracle]['waveform'].shape)

    def test_scoped_override_restores_on_failure_and_rejects_leakage(self):
        model = QualitySpeechSystem(config()).eval()
        original_method = model.encode_batch.__func__
        inputs = person_a_only(self.batch)
        with torch.inference_mode():
            encoded = model.encode_batch(inputs)
        with self.assertRaisesRegex(RuntimeError, 'codec failed'):
            with override_encoded_affect(model, inputs, encoded, torch.zeros_like(encoded['affect'])):
                with patch.object(FakeCodec, 'decode', side_effect=RuntimeError('codec failed')):
                    model.generate_batch(inputs, FakeCodec())
        self.assertIs(model.encode_batch.__func__, original_method)
        self.assertNotIn('encode_batch', model.__dict__)
        with override_encoded_affect(model, inputs, encoded, encoded['affect']):
            with self.assertRaisesRegex(ValueError, 'B fields'):
                model.encode_batch(dict(inputs, duration=self.batch['duration']))

    def test_same_conversation_donor_and_bad_selection_rejected(self):
        model = QualitySpeechSystem(config()).eval()
        self.donor['conversation_key'] = self.batch['conversation_key']
        with self.assertRaisesRegex(ValueError, 'another conversation'):
            generate_conditions(model, self.batch, self.donor, FakeCodec())
        records = [{'path': 'a'}, {'path': 'b'}]
        self.assertEqual(selected_indices(records, {'val': ['b', 'a']}), [1, 0])
        for wanted in ([], ['a', 'a'], ['x']):
            with self.assertRaises(ValueError):
                selected_indices(records, {'val': wanted})

    def test_known_tone_and_silence_metrics_use_existing_extractor(self):
        rate = 32000
        wave = (.1 * np.sin(2 * np.pi * 220 * np.arange(rate) / rate)).astype(np.float32)
        actual = waveform_metrics(wave)
        self.assertAlmostEqual(actual['pitch_median_hz_voiced'], 220, delta=5)
        self.assertAlmostEqual(actual['raw_rms'], .1 / np.sqrt(2), places=5)
        values, _ = audio_affect_targets(wave, rate, '')
        self.assertEqual(actual['normalized_energy_mean'], float(values[:, 3].mean()))
        affect = torch.tensor(values)[None]
        controls = control_waveform_metrics(affect, torch.ones(1, len(values), dtype=torch.bool), wave)
        self.assertEqual(controls['pitch_normalized_mae'], 0)
        self.assertEqual(controls['energy_normalized_mae'], 0)
        silence = waveform_metrics(np.zeros(641, dtype=np.float32))
        self.assertIsNone(silence['pitch_median_hz_voiced'])
        self.assertEqual(silence['low_energy_fraction'], 1)
        self.assertEqual(silence['leading_low_energy_seconds'], 641 / rate)
        self.assertEqual(silence['trailing_low_energy_seconds'], 641 / rate)
        self.assertEqual(silence['envelope_peaks_per_second'], 0)
        self.assertEqual(silence['raw_clip_fraction'], 0)
        for bad in (np.zeros(0), np.array([np.nan]), np.zeros((1, 2))):
            with self.assertRaises(ValueError):
                waveform_metrics(bad)

    def test_defined_clipping_end_windows_and_summary_counts(self):
        wave = np.concatenate([np.zeros(640), np.ones(640) * 1.2, np.zeros(320)]).astype(np.float32)
        metrics = waveform_metrics(wave)
        self.assertEqual(metrics['raw_clip_fraction'], .4)
        self.assertEqual(metrics['leading_low_energy_seconds'], .02)
        self.assertEqual(metrics['trailing_low_energy_seconds'], .01)
        self.assertTrue(metrics['last_window_low_energy'])
        report = {'examples': [{
            'duration_predictions': {name: {'seconds': 2., 'absolute_error_seconds': 1., 'signed_error_seconds': -1.} for name in CONDITIONS},
            'paths': {'reference': {'waveform_metrics': metrics, 'asr_token_limit_reached': True}},
        }], 'summary': {'reference': {'mean_reference_wer': .5}}}
        result = summarize(report)
        self.assertEqual(result['duration_summary']['zero_affect']['mean_absolute_error_seconds'], 1.)
        self.assertEqual(result['summary']['reference']['waveform_metrics']['any_raw_clipping_cases'], 1)
        self.assertEqual(result['summary']['reference']['waveform_metrics']['asr_token_limit_cases'], 1)

    def test_cli_writes_audio_and_report_with_shared_helpers(self):
        model = QualitySpeechSystem(config()).eval()

        class Dataset:
            records = [{'path': str(i), 'conversation_id': str(i), 'input_text': 'hello',
                        'response_text': 'hello there', 'style_id': i, 'speaker_id': i} for i in range(2)]

            def __getitem__(self, index):
                return sample(index)

        class Codec(FakeCodec):
            def to(self, device):
                return self

            def eval(self):
                return self

        def fake_recognize(report, output, device):
            for row in report['examples']:
                for entry in row['paths'].values():
                    self.assertTrue((output / entry['file']).is_file())
                    entry.update(asr='hello there', reference_wer=0., asr_token_limit_reached=False)
            report['summary'] = {name: {'mean_reference_wer': 0., 'count': len(report['examples'])}
                                 for name in report['examples'][0]['paths']}
            return report

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, manifest, selection = (root / name for name in ('model.pt', 'manifest.json', 'selection.json'))
            checkpoint.write_bytes(b'unchanged checkpoint')
            manifest.write_text('{}')
            selection.write_text(json.dumps({'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(), 'val': ['0', '1']}))
            output = root / 'output'
            argv = ['check', '--checkpoint', str(checkpoint), '--manifest', str(manifest), '--selection', str(selection),
                    '--output', str(output), '--device', 'cpu', '--count', '1']
            with patch('sys.argv', argv), patch.object(diagnostic, 'load_response_model', return_value=(model, {})), \
                    patch.object(diagnostic, 'QualitySpeechDataset', return_value=Dataset()), \
                    patch.object(diagnostic, 'FrozenEncodec', Codec), patch.object(diagnostic, 'recognize', side_effect=fake_recognize), \
                    patch('builtins.print'):
                diagnostic.main()
            report = json.loads((output / 'report.json').read_text())
            self.assertTrue(report['checkpoint_unchanged'])
            self.assertEqual(report['test_examples_opened'], 0)
            self.assertEqual(report['examples'][0]['shuffled_conversation_id'], '1')
            self.assertEqual(len(list(output.glob('*.wav'))), 7)
            self.assertEqual(len(list(output.glob('*.npz'))), 1)
            for name, path in report['examples'][0]['paths'].items():
                self.assertEqual(path['asr_rate_proxy']['words'], 2)
                if name != 'reference':
                    self.assertEqual(path['sample_count_error'], 0)


if __name__ == '__main__':
    unittest.main()
