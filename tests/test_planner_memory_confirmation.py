"""Verify held-out membership and immutable selection. / 확인 집합 분리와 불변 저장을 검사합니다."""

import tempfile
from pathlib import Path
import unittest

from scripts.prepare_planner_memory_confirmation import select_confirmation, write_immutable


def row(name, text, split='val', response='That sounds difficult.'):
    return {'path': name + '.npz', 'conversation_id': name, 'split': split, 'input_text': text,
            'response_text': response, 'style_id': 0, 'speaker_id': 1, 'reference_audio': name + '.wav'}


def selection(train, val):
    return {'train': [x['path'] for x in train], 'val': [x['path'] for x in val],
            'conversations': {'train': [x['conversation_id'] for x in train],
                              'val': [x['conversation_id'] for x in val]}}


class MemoryConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.training = row('training', 'Today my brother bought himself a very large house', 'train')
        self.dev = row('development', 'My new shoes broke during a marathon')
        self.pilot = row('pilot', 'It rained on my wedding day')
        self.previous = row('previous', 'I found a wallet in the street')
        self.fresh = row('fresh', 'School begins tomorrow')
        self.records = [self.training, self.dev, self.pilot, self.previous, self.fresh,
            row('train_duplicate', self.training['input_text']),
            row('dev_duplicate', self.dev['input_text']),
            row('test_only', 'A different future topic', 'test'),
            row('digits', 'The aquarium closes at sunset', response='You waited 4 hours.'),
            row('too_short', 'I lost my umbrella on the bus')]

    def run_selection(self, records=None, count=1):
        return select_confirmation({'records': records or self.records},
            selection([self.training], [self.dev]), selection([self.training], [self.pilot]),
            [selection([self.training], [self.previous])],
            lambda x: 1. if x['conversation_id'] == 'too_short' else 5., count)

    def test_only_fresh_nonduplicate_validation_is_selected(self):
        selected, audit = self.run_selection()
        self.assertEqual(selected['conversations']['val'], ['fresh'])
        self.assertEqual(audit['confirmation_overlap_prior_groups'],
                         {'development': 0, 'pilot': 0, 'previous_confirmation_0': 0})
        self.assertFalse(audit['model_outputs_consulted'])
        self.assertFalse(audit['test_audio_or_metrics_evaluated'])

    def test_selection_is_order_independent(self):
        first, _ = self.run_selection()
        second, _ = self.run_selection(list(reversed(self.records)))
        self.assertEqual(first, second)

    def test_insufficient_candidates_fail(self):
        with self.assertRaises(ValueError):
            self.run_selection(count=2)

    def test_previous_ids_must_match_their_paths(self):
        invalid = selection([self.training], [self.previous])
        invalid['conversations']['val'] = ['wrong_conversation']
        with self.assertRaises(ValueError):
            select_confirmation({'records': self.records}, selection([self.training], [self.dev]),
                selection([self.training], [self.pilot]), [invalid], lambda _: 5., 1)

    def test_immutable_writes_reject_conflicts_before_new_files(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_immutable(output, {'selection.json': b'original'})
            write_immutable(output, {'selection.json': b'original'})
            with self.assertRaises(ValueError):
                write_immutable(output, {'new.json': b'new', 'selection.json': b'changed'})
            self.assertEqual((output / 'selection.json').read_bytes(), b'original')
            self.assertFalse((output / 'new.json').exists())


if __name__ == '__main__':
    unittest.main()
