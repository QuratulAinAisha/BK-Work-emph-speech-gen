"""Blinding, audio integrity and blank export checks. / 익명성, 오디오 무결성, 빈 내보내기 검증."""

import csv
import argparse
import contextlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest
import wave
from unittest.mock import patch

from scripts.prepare_blind_listening import (CSV_FIELDS, RATING_FIELDS, contained_wav,
    file_hash, main, prepare_package, report_argument, report_variant_argument)


VARIANT = 'predicted_length_steps_8'


class BlindListeningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.specs = []
        self.input_text = 'I had a difficult day. </script><script>alert("x")</script> & 안녕'
        for index in range(2):
            folder = self.root / f'SECRET_SOURCE_{index}'
            folder.mkdir()
            examples = []
            for case in range(2):
                filename = f'SECRET_WAV_{case}.wav'
                with wave.open(str(folder / filename), 'wb') as stream:
                    stream.setnchannels(1)
                    stream.setsampwidth(2)
                    stream.setframerate(16000)
                    stream.writeframes(bytes([case + index, 0]) * 160)
                examples.append({'conversation_id': f'SECRET_CONVERSATION_{case}',
                    'input_text': self.input_text if case == 0 else 'My test went badly.',
                    'reference_text': 'SECRET_REFERENCE', 'path': f'SECRET_EXAMPLE_{case}',
                    'paths': {VARIANT: {'file': filename, 'asr': 'SECRET_ASR'}}})
            report = {'checkpoint': f'/models/SECRET_CHECKPOINT_{index}.pt',
                'checkpoint_sha256': f'SECRET_REPORTED_HASH_{index}', 'checkpoint_step': 320,
                'manifest_sha256': 'manifest-hash', 'selection_sha256': 'selection-hash',
                'examples': examples}
            path = folder / 'report.json'
            path.write_text(json.dumps(report), encoding='utf-8')
            self.specs.append((f'SECRET_MODEL_{index}', path))

    def package(self, name='package', seed=42):
        output = self.root / name
        result = prepare_package(self.specs, VARIANT, output, seed)
        html = (output / 'rater' / 'index.html').read_text(encoding='utf-8')
        public = json.loads(re.search(r'id="trials-data">(.*?)</script>', html, re.S).group(1))
        return output, result, html, public

    def change_report(self, index, operation):
        path = self.specs[index][1]
        report = json.loads(path.read_text(encoding='utf-8'))
        operation(report)
        path.write_text(json.dumps(report), encoding='utf-8')

    def test_public_package_is_blind_blank_and_preserves_actual_audio(self):
        output, result, html, public = self.package()
        self.assertEqual((result['conversations'], result['trials']), (2, 4))
        self.assertEqual(result['rating_status'], 'pending_human')
        self.assertEqual(result['human_ratings_collected'], 0)
        public_files = ''.join(path.read_text(encoding='utf-8-sig') for path in
            (output / 'rater').iterdir() if path.is_file())
        self.assertNotIn('SECRET_', public_files)
        self.assertNotIn('</script><script>alert', html)
        self.assertEqual({row['input_text'] for row in public['trials']},
                         {self.input_text, 'My test went badly.'})
        self.assertEqual(set(public), {'package_id', 'trials'})
        for row in public['trials']:
            self.assertEqual(set(row), {'trial_id', 'input_text', 'audio'})
            self.assertRegex(row['trial_id'], r'^T[0-9a-f]{20}$')
            self.assertEqual(row['audio'], f'audio/{row["trial_id"]}.wav')
        with (output / 'rater' / 'ratings_template.csv').open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(reader.fieldnames, list(CSV_FIELDS))
            ratings = list(reader)
        self.assertEqual([row['trial_id'] for row in ratings], [row['trial_id'] for row in public['trials']])
        for row in ratings:
            self.assertEqual(row['package_id'], public['package_id'])
            self.assertTrue(all(row[field] == '' for field in (*RATING_FIELDS, 'listener_id', 'notes')))
        private = json.loads((output / 'answer_key.json').read_text(encoding='utf-8'))
        self.assertEqual(private['rating_status'], 'pending_human')
        self.assertEqual(private['human_ratings_collected'], 0)
        self.assertEqual(private['variant'], VARIANT)
        self.assertTrue(all(source['selected_variant'] == VARIANT for source in private['sources']))
        self.assertIn('no Person A audio is available here', html)
        self.assertIn('Judge emotional fit only against that provided context', html)
        self.assertEqual({row['source_label'] for row in private['trials']},
                         {label for label, _ in self.specs})
        for row in private['trials']:
            self.assertEqual(row['source_variant'], VARIANT)
            source = Path(row['source_audio'])
            copied = output / 'rater' / row['public_audio']
            self.assertEqual(source.read_bytes(), copied.read_bytes())
            self.assertEqual(file_hash(copied), row['source_audio_sha256'])
            self.assertEqual(file_hash(copied), row['copied_audio_sha256'])
            self.assertEqual(file_hash(Path(row['source_report'])), row['source_report_sha256'])
            self.assertAlmostEqual(row['audio_details']['seconds'], .01)

    def test_seeded_trials_reproduce_and_new_directory_is_required(self):
        output, _, _, first = self.package('first')
        _, _, _, same = self.package('same')
        _, _, _, other = self.package('other', seed=43)
        self.assertEqual(first, same)
        self.assertNotEqual(first, other)
        with self.assertRaisesRegex(ValueError, 'new output directory'):
            prepare_package(self.specs, VARIANT, output)

    def test_changed_audio_gets_new_storage_identity_even_with_same_seed(self):
        _, _, _, first = self.package('first')
        audio = self.specs[0][1].parent / 'SECRET_WAV_0.wav'
        data = bytearray(audio.read_bytes())
        data[-1] = 1
        audio.write_bytes(data)
        _, _, _, changed = self.package('changed')
        self.assertNotEqual(first['package_id'], changed['package_id'])

    def test_rejects_duplicate_ids_before_creating_output(self):
        self.change_report(0, lambda r: r['examples'].append(r['examples'][0]))
        with self.assertRaisesRegex(ValueError, 'unique'):
            self.package()
        self.assertFalse((self.root / 'package').exists())

    def test_rejects_mismatched_conversation_sets(self):
        self.change_report(1, lambda r: r['examples'].pop())
        with self.assertRaisesRegex(ValueError, 'identical conversation ID sets'):
            self.package()
        self.assertFalse((self.root / 'package').exists())

    def test_rejects_mismatched_a_text(self):
        self.change_report(1, lambda r: r['examples'][0].update(input_text='A different utterance.'))
        with self.assertRaisesRegex(ValueError, 'A input text differs'):
            self.package()

    def test_rejects_missing_variant_and_duplicate_labels(self):
        with self.assertRaisesRegex(ValueError, 'labels must be unique'):
            prepare_package([self.specs[0], (self.specs[0][0].lower(), self.specs[1][1])],
                            VARIANT, self.root / 'duplicate-labels')
        self.change_report(1, lambda r: r['examples'][0]['paths'].clear())
        with self.assertRaisesRegex(ValueError, 'missing selected variant'):
            self.package()

    def test_rejects_missing_escaping_and_truncated_audio(self):
        report = self.specs[0][1]
        for filename in ('../outside.wav', str((self.root / 'outside.wav').resolve()),
                         'missing.wav', 'not-audio.txt'):
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                contained_wav(report, filename)
        wav = report.parent / 'SECRET_WAV_0.wav'
        wav.write_bytes(wav.read_bytes()[:-2])
        with self.assertRaisesRegex(ValueError, 'Truncated'):
            self.package()
        self.assertFalse((self.root / 'package').exists())

    def test_argument_preserves_equal_signs_in_path(self):
        self.assertEqual(report_argument(' candidate = a=b/report.json '),
                         ('candidate', Path('a=b/report.json')))

    def test_mixed_variants_preserve_audio_and_private_provenance(self):
        alternative = 'predicted_length_ar'
        folder = self.specs[1][1].parent
        for case in range(2):
            with wave.open(str(folder / f'AR_AUDIO_{case}.wav'), 'wb') as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(16000)
                stream.writeframes(bytes([9 + case, 0]) * 320)
        def use_ar(report):
            for case, example in enumerate(report['examples']):
                example['paths'] = {alternative: {'file': f'AR_AUDIO_{case}.wav'}}
        self.change_report(1, use_ar)
        inputs = {path: path.read_bytes() for _, report in self.specs for path in report.parent.iterdir()}
        variants = [(self.specs[0][0], VARIANT), (self.specs[1][0], alternative)]
        output = self.root / 'mixed'
        prepare_package(self.specs, VARIANT, output, report_variants=variants)
        key = json.loads((output / 'answer_key.json').read_text(encoding='utf-8'))
        self.assertIsNone(key['variant'])
        self.assertEqual(key['default_variant'], VARIANT)
        self.assertEqual(key['report_variants'], dict(variants))
        self.assertEqual({source['label']: source['selected_variant'] for source in key['sources']}, dict(variants))
        for trial in key['trials']:
            self.assertEqual(trial['source_variant'], dict(variants)[trial['source_label']])
            self.assertEqual(file_hash(Path(trial['source_audio'])), trial['copied_audio_sha256'])
            self.assertEqual(file_hash(output / 'rater' / trial['public_audio']), trial['source_audio_sha256'])
            self.assertEqual(file_hash(Path(trial['source_report'])), trial['source_report_sha256'])
            if trial['source_label'] == self.specs[1][0]:
                self.assertTrue(Path(trial['source_audio']).name.startswith('AR_AUDIO_'))
                self.assertEqual(trial['audio_details']['frames'], 320)
        public_text = ''.join(path.read_text(encoding='utf-8-sig') for path in
            (output / 'rater').iterdir() if path.is_file())
        for secret in ('SECRET_', VARIANT, alternative, 'AR_AUDIO_'):
            self.assertNotIn(secret, public_text)
        self.assertTrue(all(path.read_bytes() == original for path, original in inputs.items()))

    def test_override_labels_require_exact_coverage_without_duplicates(self):
        first, second = (label for label, _ in self.specs)
        cases = [([(first, VARIANT)], 'exact label coverage'),
                 ([(first, VARIANT), (second, VARIANT), ('unknown', VARIANT)], 'Unknown'),
                 ([(first, VARIANT), (first, VARIANT)], 'Duplicate'),
                 ([(first, VARIANT), (first.lower(), VARIANT)], 'Duplicate'),
                 ([(first.lower(), VARIANT), (second, VARIANT)], 'match exactly'),
                 ([(first, VARIANT), (second, '')], 'nonempty')]
        for index, (overrides, message) in enumerate(cases):
            output = self.root / f'invalid_{index}'
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, message):
                prepare_package(self.specs, VARIANT, output, report_variants=overrides)
            self.assertFalse(output.exists())

    def test_actual_variant_changes_identity_and_default_matches_explicit_map(self):
        alias = 'same_audio_distinct_variant'
        def add_alias(report):
            for example in report['examples']:
                example['paths'][alias] = dict(example['paths'][VARIANT])
        for index in range(2):
            self.change_report(index, add_alias)
        _, default, _, _ = self.package('default')
        explicit = prepare_package(self.specs, 'unused_default', self.root / 'explicit',
            report_variants=[(label, VARIANT) for label, _ in self.specs])
        different = prepare_package(self.specs, VARIANT, self.root / 'alias',
            report_variants=[(label, alias) for label, _ in self.specs])
        self.assertEqual(default['package_id'], explicit['package_id'])
        self.assertNotEqual(default['package_id'], different['package_id'])

    def test_cli_default_custom_default_and_mixed_overrides(self):
        alternative = 'predicted_length_ar'
        def add_ar(report):
            for example in report['examples']:
                example['paths'][alternative] = dict(example['paths'][VARIANT])
        for index in range(2):
            self.change_report(index, add_ar)
        base = ['prepare_blind_listening']
        for label, path in self.specs:
            base.extend(['--report', f'{label}={path}'])
        cases = [('default', [], {label: VARIANT for label, _ in self.specs}),
                 ('custom', ['--variant', alternative], {label: alternative for label, _ in self.specs}),
                 ('mixed', ['--report-variant', f'{self.specs[0][0]}={VARIANT}',
                            '--report-variant', f'{self.specs[1][0]}={alternative}'],
                  {self.specs[0][0]: VARIANT, self.specs[1][0]: alternative})]
        for name, flags, expected in cases:
            output = self.root / name
            with patch('sys.argv', base + flags + ['--output', str(output)]), contextlib.redirect_stdout(io.StringIO()):
                main()
            key = json.loads((output / 'answer_key.json').read_text(encoding='utf-8'))
            self.assertEqual(key['report_variants'], expected)
            for trial in key['trials']:
                self.assertEqual(trial['source_variant'], expected[trial['source_label']])

    def test_report_variant_argument_rejects_missing_parts(self):
        self.assertEqual(report_variant_argument(' source = predicted_length_ar '),
                         ('source', 'predicted_length_ar'))
        for value in ('source', '=variant', 'source= ', ' = '):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                report_variant_argument(value)

    @unittest.skipUnless(shutil.which('node'), 'Node is optional for browser export smoke test')
    def test_browser_export_is_real_csv_and_starts_blank(self):
        _, _, html, public = self.package()
        # Minimal DOM executes the real form/export script. / 최소 DOM으로 실제 양식과 내보내기를 실행합니다.
        harness = r'''
const vm=require('vm');let blobText='';let downloadName='';
function element(){return {value:'',textContent:'',children:[],events:{},appendChild(v){this.children.push(v);},
setAttribute(){},addEventListener(k,f){this.events[k]=f;},remove(){},click(){if(this.download)downloadName=this.download;}};}
const elements=Object.fromEntries(['trials-data','listener','status','save-status','trials','export'].map(k=>[k,element()]));
elements['trials-data'].textContent=JSON.stringify(PUBLIC_DATA);
const context={document:{getElementById:k=>elements[k],createElement:()=>element(),body:element()},
localStorage:{getItem:()=>null,setItem(){}},Blob:class {constructor(parts){blobText=parts.join('');}},
URL:{createObjectURL:()=> 'blob:test',revokeObjectURL(){}},setTimeout:f=>f()};
vm.createContext(context);vm.runInContext(SCRIPT_DATA,context);elements.export.events.click();const initial=blobText;
vm.runInContext(String.raw`listener.value='=malicious()';ratings[data.trials[0].trial_id].notes='  =formula(),"quoted"\nnext';
ratings[data.trials[0].trial_id].intelligibility='4';`,context);elements.export.events.click();
process.stdout.write(JSON.stringify({initial,edited:blobText,downloadName,status:elements.status.textContent,
cards:elements.trials.children.length}));
'''
        script = html.rsplit('<script>', 1)[1].split('</script>', 1)[0]
        harness = 'const PUBLIC_DATA=' + json.dumps(public) + ';const SCRIPT_DATA=' + json.dumps(script) + ';\n' + harness
        completed = subprocess.run([shutil.which('node'), '-e', harness], check=False,
                                   text=True, encoding='utf-8', capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(result['cards'], 4)
        self.assertEqual(result['status'], '0 of 4 fully rated')
        self.assertTrue(result['initial'].startswith('\ufeff'))
        rows = list(csv.DictReader(io.StringIO(result['initial'].lstrip('\ufeff'))))
        self.assertEqual(len(rows), 4)
        self.assertEqual(list(rows[0]), list(CSV_FIELDS))
        self.assertTrue(all(row[field] == '' for row in rows for field in (*RATING_FIELDS, 'listener_id', 'notes')))
        edited = list(csv.DictReader(io.StringIO(result['edited'].lstrip('\ufeff'))))
        self.assertEqual(edited[0]['listener_id'], "'=malicious()")
        self.assertEqual(edited[0]['notes'], "'  =formula(),\"quoted\"\nnext")
        self.assertEqual(edited[0]['intelligibility'], '4')
        self.assertEqual(result['downloadName'], f'listening_ratings_{public["package_id"]}.csv')


if __name__ == '__main__':
    unittest.main()
