"""Guard context-control interpretation and isolation. / 문맥 대조의 해석과 입력 격리를 검증합니다."""

import copy
import unittest
from unittest.mock import patch

import torch

from dataset.quality_speech_dataset import collate_quality, person_a_only
from model.full_speech.units import UnitSpeechSystem
from scripts.evaluate_planner_controls import evaluate_pair, next_different_conversation, summarize_examples, planner_inputs
from tests.test_quality_speech import config, sample, FakeCodec


class PlannerControlTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)

    def test_global_donors_wrap_and_skip_same_conversation(self):
        rows = [{'conversation_id': value} for value in ('a', 'a', 'b', 'c')]
        self.assertEqual(next_different_conversation(rows), [2, 2, 3, 0])
        for source, donor in enumerate(next_different_conversation(rows)):
            self.assertNotEqual(rows[source]['conversation_id'], rows[donor]['conversation_id'])
        with self.assertRaisesRegex(ValueError, 'two distinct'):
            next_different_conversation(rows[:2])

    def test_no_b_inputs_and_style_fixed_with_actual_refinement(self):
        model = UnitSpeechSystem(config(), torch.randn(8, 768)).eval()
        batch, donor = collate_quality([sample(0)]), collate_quality([sample(1)])
        original = model.encode_batch
        seen = []

        def checked(inputs):
            self.assertEqual(set(inputs), set(person_a_only(batch)))
            self.assertEqual(inputs['style_id'].tolist(), [0])
            self.assertEqual(inputs['speaker_id'].tolist(), [0])
            seen.append(inputs)
            return original(inputs)

        with patch.object(model, 'encode_batch', side_effect=checked), \
                patch.object(model.semantic_planner, 'sample', wraps=model.semantic_planner.sample) as sampler:
            first = evaluate_pair(model, batch, donor)
            self.assertEqual(sampler.call_count, 3)
            self.assertTrue(all(call.kwargs['steps'] == 8 for call in sampler.call_args_list))
        self.assertEqual(len(seen), 2)
        self.assertEqual(first['conditions']['correct_a']['changed_units_vs_correct'], 0)
        # Alter B targets: predictions and predicted duration must stay fixed. / B 타깃을 바꿔도 예측은 같습니다.
        altered = copy.deepcopy(batch)
        altered['semantic'].fill_(999)
        altered['duration'].fill_(1.25)
        second = evaluate_pair(model, altered, donor)
        for name in first['conditions']:
            self.assertEqual(first['conditions'][name]['generated_unit_ids'], second['conditions'][name]['generated_unit_ids'])
            self.assertEqual(first['conditions'][name]['predicted_duration_seconds'], second['conditions'][name]['predicted_duration_seconds'])
        self.assertEqual(donor['style_id'].tolist(), [1])

    def test_summary_weights_frames_separately_from_examples(self):
        examples = []
        for frames, correct in ((2, 2), (8, 0)):
            metrics = {'fully_masked_ce': float(frames), 'unit_accuracy': correct / frames,
                       'correct_units': correct, 'changed_units_vs_correct': frames - correct,
                       'changed_unit_fraction_vs_correct': (frames - correct) / frames,
                       'duration_abs_error_seconds': .5, 'predicted_duration_seconds': 3.}
            examples.append({'frames': frames, 'conditions': {name: metrics.copy()
                            for name in ('correct_a', 'shuffled_a', 'zero_a')}})
        result = summarize_examples(examples)['conditions']['correct_a']
        self.assertEqual(result['mean_example_unit_accuracy'], .5)
        self.assertEqual(result['frame_unit_accuracy'], .2)
        self.assertEqual(result['mean_example_fully_masked_ce'], 5.)
        self.assertEqual(result['frame_fully_masked_ce'], 6.8)

    def test_memory_routing_matches_normal_generation_and_preserves_duration_inputs(self):
        cfg = config()
        cfg.semantic_steps = 8
        model = UnitSpeechSystem(cfg, torch.randn(8, 768)).eval()
        batch, donor = collate_quality([sample(0)]), collate_quality([sample(1)])
        with torch.inference_mode():
            encoded = model.encode_batch(person_a_only(batch))
        durations = []
        for mode in ('fused', 'native_speech', 'resampled_speech'):
            model.config.planner_memory_mode = mode
            with patch.object(model.semantic_planner, 'sample', wraps=model.semantic_planner.sample) as sampler, \
                 patch.object(model.length_predictor, 'forward', wraps=model.length_predictor.forward) as length:
                result = evaluate_pair(model, batch, donor)
            expected = planner_inputs(model, encoded)
            torch.testing.assert_close(sampler.call_args_list[0].kwargs['context'], expected['context'], rtol=0, atol=0)
            torch.testing.assert_close(sampler.call_args_list[0].kwargs['affect_mask'], encoded['context_mask'])
            torch.testing.assert_close(length.call_args_list[0].args[0], encoded['context'], rtol=0, atol=0)
            torch.testing.assert_close(length.call_args_list[0].args[1], encoded['affect'], rtol=0, atol=0)
            normal = model.generate_batch(person_a_only(batch), FakeCodec(), oracle_duration=batch['duration'])
            units = model.semantic_planner.codebook.encode(normal['semantic'])[0].tolist()
            self.assertEqual(result['conditions']['correct_a']['generated_unit_ids'], units)
            durations.append(result['conditions']['correct_a']['predicted_duration_seconds'])
        self.assertEqual(durations, [durations[0]] * 3)


if __name__ == '__main__':
    unittest.main()
