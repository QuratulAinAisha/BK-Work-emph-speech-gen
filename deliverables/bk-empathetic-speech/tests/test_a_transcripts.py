"""Check source mapping and transcript metrics. / 원본 매핑과 문장 지표를 검사합니다."""

import unittest

from scripts.check_a_transcripts import resolve_a_audio, select_rows, text_metrics


class ATranscriptTests(unittest.TestCase):
    def setUp(self):
        self.row = {'path': 'content_000001.npz', 'source_index': 1, 'split': 'val',
                    'reference_audio': '/dataset/generated_response_audio/b.wav',
                    'style_id': 0, 'speaker_id': 1, 'conversation_id': 'conversation'}
        self.source = {'records': [{}, {'response_audio': self.row['reference_audio'],
            'mel': '/dataset/mel/original.input.npy', 'style_id': 0, 'speaker_id': 1}]}

    def test_exact_preparation_mapping(self):
        self.assertEqual(resolve_a_audio(self.row, self.source).as_posix(),
                         '/dataset/generated_input_audio/original.input.wav')

    def test_mapping_mismatch_is_rejected(self):
        for field, value in [('reference_audio', '/other.wav'), ('source_index', -1),
                             ('speaker_id', 0), ('style_id', 1)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                resolve_a_audio({**self.row, field: value}, self.source)

    def test_selection_never_reads_other_split(self):
        manifest = {'records': [self.row, {**self.row, 'path': 'held_out.npz', 'split': 'test'}]}
        selected = select_rows(manifest, {'val': [self.row['path']]}, 'val', 128)
        self.assertEqual(selected, [self.row])
        with self.assertRaises(ValueError):
            select_rows(manifest, {'val': ['held_out.npz']}, 'val', 128)
        with self.assertRaises(ValueError):
            select_rows(manifest, {'test': ['held_out.npz']}, 'test', 1)

    def test_error_counts_keep_insertions_and_spaces(self):
        metrics = text_metrics('The cat', 'The bat now')
        self.assertEqual(metrics['word_errors'], 2)
        self.assertEqual(metrics['wer'], 1.)
        self.assertEqual(metrics['character_errors'], 5)
        self.assertEqual(metrics['reference_characters'], 7)
        self.assertEqual(metrics['cer'], 5 / 7)

    def test_normalization_and_empty_reference(self):
        self.assertEqual(text_metrics('Café, I’m 21!', "cafe i'm 21")['wer'], 0.)
        self.assertEqual(text_metrics('two words', '')['wer'], 1.)
        with self.assertRaises(ValueError):
            text_metrics('!!!', 'anything')


if __name__ == '__main__':
    unittest.main()
