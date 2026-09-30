"""Sampled planner training cadence and isolation. / 생성 계획기 학습 주기와 분리 검증."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from dataset.quality_speech_dataset import collate_quality
from model.full_speech.planner_waveform import sampled_planner_waveform
from model.full_speech.quality import text_ids
from model.full_speech.units import UnitSpeechSystem
from tests.test_planner_waveform import FakeCodec, FakeTeacher
from tests.test_quality_speech import config, sample


class PlannerAudioTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(28)
        cfg = replace(config(), dropout=.15, semantic_steps=8, predicted_semantic_steps=8, codec_steps=32)
        self.model = UnitSpeechSystem(cfg, torch.randn(8, 768))
        self.codec, self.teacher = FakeCodec(), FakeTeacher()
        object.__setattr__(self.model, '_acoustic_codec', self.codec)
        self.model.configure_recovery('planner', teacher=self.teacher, teacher_weight=0.)
        samples = [sample(index % 2) for index in range(4)]
        for item, text in zip(samples, ('a', 'ab', 'ba', 'b')):
            item['text_b'] = text_ids(text)
        self.batch = collate_quality(samples)

    def base(self, batch):
        self.assertIs(batch, self.batch)
        return {'unit_ce': torch.tensor(4., requires_grad=True),
                'duration': torch.tensor(1.), 'total': torch.tensor(5., requires_grad=True)}

    def test_default_loss_path_unchanged(self):
        original = self.base(self.batch)
        with (patch.object(self.model, 'unit_base_losses', return_value=original),
              patch('model.full_speech.planner_waveform.sampled_planner_waveform') as sampled):
            result = self.model.losses(self.batch)
        self.assertIs(result, original)
        self.assertNotIn('sampled_planner_ctc', result)
        sampled.assert_not_called()

    def test_training_cadence_rotates_every_sampled_call_and_keeps_anchor(self):
        self.model.configure_planner_audio(weight=.1, every=4)
        self.model.train()
        wave = torch.ones(6)
        steps = (0, 1, 2, 3, 4, 8, 12, 16)
        with (patch.object(self.model, 'unit_base_losses', side_effect=self.base) as anchor,
              patch('model.full_speech.planner_waveform.sampled_planner_waveform',
                    return_value={'waveform': wave}) as sampled,
              patch('model.full_speech.planner_waveform.frozen_content_ctc',
                    return_value={'loss': torch.tensor(7.)}) as scored):
            for step in steps:
                self.model.current_step = step
                values = self.model.losses(self.batch)
                active = step % 4 == 0
                self.assertAlmostEqual(float(values['sampled_planner_ctc'].detach()), 7. if active else 0.)
                self.assertAlmostEqual(float(values['total'].detach()), 5.7 if active else 5., places=5)
            self.assertEqual(anchor.call_count, len(steps))
        calls = sampled.call_args_list
        self.assertEqual([call.args[4] for call in calls], [0, 1, 2, 3, 0])
        self.assertEqual([call.args[3] for call in calls], [100000, 100004, 100008, 100012, 100016])
        for generation, scoring in zip(calls, scored.call_args_list):
            index = generation.args[4]
            wanted = self.batch['text_b'][index, :int(self.batch['text_b_len'][index])]
            torch.testing.assert_close(scoring.args[2], wanted, rtol=0, atol=0)

    def test_validation_samples_fixed_index_and_seed_on_every_batch(self):
        self.model.configure_planner_audio(weight=.1, every=4)
        self.model.eval()
        with (patch.object(self.model, 'unit_base_losses', side_effect=self.base) as anchor,
              patch('model.full_speech.planner_waveform.sampled_planner_waveform',
                    return_value={'waveform': torch.ones(6)}) as sampled,
              patch('model.full_speech.planner_waveform.frozen_content_ctc',
                    return_value={'loss': torch.tensor(7.)})):
            with torch.no_grad():
                for step in (1, 999):
                    self.model.current_step = step
                    values = self.model.losses(self.batch)
                    self.assertAlmostEqual(float(values['total']), 5.7, places=5)
        self.assertEqual(anchor.call_count, 2)
        self.assertEqual([call.args[3:] for call in sampled.call_args_list], [(42, 0), (42, 0)])
        self.assertFalse(self.model.training)

    def test_configuration_freezes_acoustics_and_phase_change_clears_recipe(self):
        self.codec.requires_grad_(True).train()
        self.teacher.requires_grad_(True).train()
        self.model.configure_planner_audio(weight=.02, every=4)
        self.model.train()
        trainable = [name for name, parameter in self.model.named_parameters() if parameter.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith('semantic_planner.') for name in trainable))
        self.assertTrue(self.model.semantic_planner.training)
        for module in (self.model.codec_generator, self.model.length_predictor,
                       self.model.encoder, self.codec, self.teacher):
            self.assertTrue(all(not child.training for child in module.modules()))
            self.assertTrue(all(not parameter.requires_grad for parameter in module.parameters()))
        self.model.configure_recovery('acoustic')
        self.assertIsNone(self.model.planner_audio_recipe)
        with (patch.object(self.model, 'unit_base_losses', side_effect=self.base),
              patch('model.full_speech.planner_waveform.sampled_planner_waveform') as sampled):
            self.assertNotIn('sampled_planner_ctc', self.model.losses(self.batch))
        sampled.assert_not_called()

    def test_full_loss_updates_only_planner_and_restores_dropout_modes(self):
        self.model.configure_unit_objective('fully_masked', 160, .25)
        self.model.configure_planner_audio(weight=.02, every=4)
        self.model.train()
        self.model.current_step = 0
        before = {name: parameter.detach().clone() for name, parameter in self.model.named_parameters()}
        modes = [module.training for module in self.model.modules()]
        optimizer = torch.optim.SGD([parameter for parameter in self.model.parameters() if parameter.requires_grad], lr=.01)
        with patch('model.full_speech.planner_waveform.sampled_planner_waveform',
                   wraps=sampled_planner_waveform) as sampled:
            losses = self.model.losses(self.batch)
        self.assertEqual(sampled.call_count, 1)
        self.assertEqual(modes, [module.training for module in self.model.modules()])
        self.assertGreater(float(losses['sampled_planner_ctc'].detach()), 0.)
        torch.testing.assert_close(losses['total'], losses['unit_ce'] + losses['duration'] +
            .25 * losses['prior_rehearsal_ce'] + .02 * losses['sampled_planner_ctc'])
        losses['total'].backward()
        gradients = [(name, parameter.grad) for name, parameter in self.model.named_parameters()
                     if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(name.startswith('semantic_planner.') for name, _ in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for _, gradient in gradients))
        self.assertGreater(sum(float(gradient.abs().sum()) for _, gradient in gradients), 0.)
        self.assertTrue(all(parameter.grad is None for module in (self.codec, self.teacher)
                            for parameter in module.parameters()))
        optimizer.step()
        changed = [name for name, parameter in self.model.named_parameters()
                   if not torch.equal(parameter.detach(), before[name])]
        self.assertTrue(changed)
        self.assertTrue(all(name.startswith('semantic_planner.') for name in changed))
        self.assertEqual(modes, [module.training for module in self.model.modules()])

    def test_rejects_invalid_settings_and_wrong_phase(self):
        for weight, every in ((0., 4), (-1., 4), (float('nan'), 4), (float('inf'), 4), (.02, 0), (.02, True)):
            with self.subTest(weight=weight, every=every), self.assertRaisesRegex(ValueError, 'settings'):
                self.model.configure_planner_audio(weight, every)
        self.model.configure_unit_prior()
        with self.assertRaisesRegex(ValueError, 'conditional units'):
            self.model.configure_planner_audio()
        self.model.configure_recovery('joint', teacher=self.teacher)
        with self.assertRaisesRegex(ValueError, 'conditional units'):
            self.model.configure_planner_audio()
        self.model.configure_recovery('planner', teacher=None)
        with self.assertRaisesRegex(ValueError, 'conditional units'):
            self.model.configure_planner_audio()


if __name__ == '__main__':
    unittest.main()
