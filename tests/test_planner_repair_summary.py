"""Missing evidence and human ratings must stay explicit. / 누락 근거와 사람 평가 상태를 명시합니다."""

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.summarize_planner_repair import AR_ARCHITECTURE, build_ledger, markdown


class RepairSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')
        return path

    def stage(self, number):
        return build_ledger(self.root)['steps'][number - 1]

    @staticmethod
    def teacher(accuracy=.8):
        return {'example_mean_next_unit_ce': .9, 'frame_next_unit_ce': .9,
            'teacher_forced_unit_accuracy': accuracy, 'frames': 100, 'examples': 2,
            'uses_correct_B_prefix': True, 'oracle_B_length': True, 'normal_inference': False}

    def ar_training(self, phase, steps=1600):
        base = 'step12_ar/' + phase
        self.write(base + '/recipe.json', {'phase': phase, 'steps': steps, 'batch_size_per_gpu': 16,
            'world_size': 4, 'duration_and_acoustics_frozen': True, 'feasibility_only': True})
        self.write(base + '/complete.json', {'phase': phase, 'steps': steps, 'quality_passed': False})
        self.write(base + '/audit_step_%04d.json' % steps, {'passed': True})
        self.write(base + '/initial_validation.json', self.teacher(.2))
        curve = [{'step': steps // 2, 'teacher_forced_validation': self.teacher(.6),
                  'this_process_optimizer_update_seconds': 50., 'this_process_update_count': steps // 2},
                 {'step': steps, 'teacher_forced_validation': self.teacher(),
                  'this_process_optimizer_update_seconds': 100., 'this_process_update_count': steps}]
        (self.root / base / 'metrics.jsonl').write_text('\n'.join(json.dumps(row) for row in curve) + '\n', encoding='utf-8')

    def ar_evaluation(self, phase='conditional', categorical=False, audio=True, audited=True):
        label = 'categorical' if categorical else phase
        base = 'step12_ar/' + label + '_evaluation'
        names = ['correct_a', 'shuffled_a'] if phase == 'conditional' else ['null_prior']
        conditions = {name: {'unit_edit_distance_per_reference_unit': 1.2,
            'adjacent_repeat_fraction': .9, 'generated_unit_ids': [1, 1, 1]} for name in names}
        report = {'architecture': AR_ARCHITECTURE, 'phase': phase, 'checkpoint_sha256': phase + '-hash',
            'checkpoint_step': 1600, 'planner_sampling': {'mode': 'categorical' if categorical else 'greedy'},
            'production_candidate_promoted': False, 'teacher_forced': self.teacher(),
            'free_running': {'count': 1, 'A_encoder_inputs': 'person_a_only',
                'duration_locked_across_controls': True, 'summary': conditions,
                'examples': [{'A_only_generation': phase == 'conditional', 'uses_B_unit_inputs': False,
                    'duration_source': 'A_predicted_duration' if phase == 'conditional' else 'oracle_B_length',
                    'style_and_speaker_held_fixed': True, 'conditions': conditions}]}}
        if audited:
            report['frozen_audit'] = {'passed': True}
        self.write(base + '/units_report.json', report)
        if audio:
            audio_report = {key: value for key, value in report.items() if key not in ('free_running', 'teacher_forced', 'frozen_audit')}
            audio_report.update(examples=[{'paths': {'predicted_length_ar': {'file': 'generated.wav', 'asr': 'bad speech'}}}],
                summary={'predicted_length_ar': {'mean_reference_wer': 1.4, 'count': 1}})
            self.write(base + '/audio/report.json', audio_report)
            (self.root / base / 'audio/generated.wav').write_bytes(b'audio fixture')
        return base

    def confirmation_protocol(self, phase, categorical=False):
        protocol = {'phase': phase, 'source_sha256': {'runner.py': 'reported-source-hash'},
            'input_sha256': {'manifest.json': 'reported-input-hash'},
            'fresh_conversation_ids': list(range(32)), 'fresh_audio_paths': [f'cache/{i}.npz' for i in range(8)],
            'ar_categorical_requested_before_fresh_results': categorical,
            'seed_replication': {'seeds': [42, 43, 44], 'new_training_seeds': [43, 44],
                'learning_rates': {'low_lr': .0001, 'high_lr': .001}, 'total_updates': 640,
                'frozen_audit_endpoints': [160, 320, 480, 640]},
            'fresh_conditional': {'masked_candidates': {'balanced_step1600': 'a.pt',
                'ce_control_step160': 'b.pt', 'sampled_ctc_step160': 'c.pt'}}}
        path = self.write(f'step13_confirmation/protocol_{phase}.json', protocol)
        self.write(f'step13_confirmation/status_{phase}.json', {'phase': phase,
            'stage': 'complete_pending_review', 'protocol_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        return protocol

    def confirmation_receipt(self, phase, name, outputs):
        base = 'step13_confirmation/'
        prefix = 'outputs/planner_repair_v1/'
        protocol_hash = hashlib.sha256((self.root / f'{base}protocol_{phase}.json').read_bytes()).hexdigest()
        definition = self.write(f'{base}receipts/{phase}/{name}_definition.json', {
            'protocol_sha256': protocol_hash, 'outputs': [prefix + path for path in outputs]})
        self.write(f'{base}receipts/{phase}/{name}_complete.json', {
            'definition_sha256': hashlib.sha256(definition.read_bytes()).hexdigest(),
            'output_sha256': {prefix + path: hashlib.sha256((self.root / path).read_bytes()).hexdigest() for path in outputs},
            'quality_passed': False, 'automatic_quality_promotion': False})

    def confirmation_audio(self, path, variant):
        report = {'examples': [{'conversation_id': i, 'path': f'cache/{i}.npz',
            'paths': {variant: {'file': 'example.wav', 'asr': 'audio diagnostic'}}} for i in range(8)],
            'summary': {variant: {'mean_reference_wer': 1.1, 'count': 8}}}
        self.write(path, report)
        (self.root / path).parent.joinpath('example.wav').write_bytes(b'local audio fixture')
        return report

    def confirmation_seeds(self):
        protocol = self.confirmation_protocol('seeds')
        for seed in (43, 44):
            for rate in ('low_lr', 'high_lr'):
                name = f'{rate}_seed{seed}'
                base = 'step13_confirmation/seeds/' + name
                path = base + '/complete.json'
                self.write(path, {'steps': 640, 'world_size': 4, 'stage': 'planner'})
                self.confirmation_receipt('seeds', name, [path])
                for step in (160, 320, 480, 640):
                    path = base + f'/audit_step_{step:04d}.json'
                    self.write(path, {'passed': True})
                    self.confirmation_receipt('seeds', name + f'_step_{step:04d}_audit', [path])
        labels = [f'{rate}_seed{seed}' for seed in (42, 43, 44) for rate in ('low_lr', 'high_lr')]
        gaps = []
        for name in labels:
            path = f'step13_confirmation/seeds/gaps/{name}.json'
            self.write(path, {'examples': [{'conversation_id': i} for i in range(32)],
                'summary': {'random/0.5/null_prior/first_pass': {'all_hidden': {'hidden_only_accuracy': .55}}}})
            gaps.append(path)
        summary = 'step13_confirmation/seeds/gaps/summary.json'
        self.write(summary, {'checkpoints': {name: name + '.pt' for name in labels}})
        self.confirmation_receipt('seeds', 'six_endpoint_fresh_gaps', [summary, *gaps])
        for rate in ('low_lr', 'high_lr'):
            name = rate + '_seed42'
            path = f'step13_confirmation/seeds/fresh_audio/{name}/report.json'
            self.confirmation_audio(path, 'iterative_hidden_050')
            self.confirmation_receipt('seeds', name + '_fresh_hinted_audio', [path])
        return protocol

    def test_empty_tree_has_thirteen_pending_steps_without_invented_success(self):
        ledger = build_ledger(self.root)
        self.assertEqual(ledger['status_counts'], {'pending': 13})
        self.assertEqual(len(ledger['steps']), 13)
        self.assertFalse(ledger['independent_confirmation_complete'])
        self.assertEqual(ledger['human_listening_status'], 'pending_human')
        self.assertIn('Missing local artifacts may still exist on the server', markdown(ledger))
        self.assertTrue(all(row['pending_evidence'] for row in ledger['steps']))

    def test_completion_is_separate_from_quality_and_requires_endpoint_audit(self):
        self.write('step02_tiny/complete.json', {'updates': 600, 'quality_passed': False,
            'fixed_mask_accuracy_at_endpoint': 1., 'fresh_half_mask_accuracy_at_endpoint': 1.})
        self.write('step02_tiny/latest_evaluation.json', {'metrics': {'fixed': {'hidden_unit_accuracy': 1.}}})
        self.write('step02_tiny/audit_step_0000.json', {'passed': True})
        self.assertEqual(self.stage(2)['status'], 'partial')
        self.write('step02_tiny/audit_step_0600.json', {'passed': True})
        stage = self.stage(2)
        self.assertEqual(stage['status'], 'completed')
        self.assertEqual(stage['quality_status'], 'not_established')
        self.assertEqual(stage['failed_checks'], [])
        self.write('step02_tiny/audit_step_0600.json', {'passed': False, 'error': 'Frozen tensor changed'})
        stage = self.stage(2)
        self.assertEqual(stage['status'], 'failed')
        self.assertTrue(any('Frozen tensor changed' in item['reason'] for item in stage['failed_checks']))

    def test_invalid_json_is_failed_evidence_not_a_crash(self):
        (self.root / 'step03_data.json').write_text('{incomplete', encoding='utf-8')
        stage = self.stage(3)
        self.assertEqual(stage['status'], 'failed')
        self.assertTrue(stage['pending_evidence'])

    def test_conditional_branch_requires_explicit_deferred_decision(self):
        self.assertEqual(self.stage(5)['status'], 'pending')
        self.write('initial_review.json', {'step5_decision': 'Architecture replacement deferred: tiny learning passed.'})
        stage = self.stage(5)
        self.assertEqual(stage['status'], 'deferred')
        self.assertFalse(stage['requirements'][0]['satisfied'])
        self.assertTrue(stage['pending_evidence'])
        self.assertEqual(self.stage(12)['status'], 'pending')

    def test_historical_preflight_failure_does_not_hide_later_passing_gate(self):
        self.write('step08_preflight_on_prior.json', {'passed': False, 'error': 'Forward mismatch'})
        self.assertEqual(self.stage(8)['status'], 'failed')
        self.write('step08_waveform/real_gradient.json', {'passed': True})
        stage = self.stage(8)
        self.assertEqual(stage['status'], 'partial')
        self.assertTrue(any('Historical' in note for note in stage['notes']))
        self.assertIn('step08_preflight_on_prior.json', stage['evidence_paths'])
        self.assertTrue(stage['pending_evidence'])

    def test_audio_report_without_local_wav_or_asr_is_pending(self):
        report = {'examples': [{'paths': {'oracle_units': {'file': 'oracle.wav'}}}],
                  'summary': {'oracle_units': {'mean_reference_wer': .2, 'count': 1}}}
        self.write('step06_prior/audio/report.json', report)
        stage = self.stage(6)
        self.assertEqual(stage['status'], 'partial')
        audio = stage['requirements'][-1]
        self.assertFalse(audio['satisfied'])
        self.assertTrue(any('Missing ASR' in item for item in audio['observed'][0]['missing']))
        self.assertTrue(any('audio' in item for item in audio['observed'][0]['missing']))

    def test_blank_package_never_counts_as_human_listening(self):
        self.write('step13_listening/answer_key.json', {'package_id': 'p1', 'rating_status': 'pending_human',
            'human_ratings_collected': 999, 'trials': [{'trial_id': 't1'}]})
        rater = self.root / 'step13_listening/rater'
        rater.mkdir()
        (rater / 'index.html').write_text('blank form', encoding='utf-8')
        fields = ['package_id', 'trial_id', 'intelligibility', 'relevance', 'emotional_appropriateness', 'naturalness']
        for name, score in [('ratings_template.csv', '5'), ('listener1.csv', '')]:
            with (rater / name).open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow({'package_id': 'p1', 'trial_id': 't1', 'intelligibility': score})
        stage = self.stage(13)
        self.assertEqual(stage['status'], 'prepared')
        self.assertEqual(stage['human_listening']['status'], 'pending_human')
        self.assertEqual(stage['human_listening']['valid_rating_cells'], 0)
        with (rater / 'listener1.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerow({'package_id': 'p1', 'trial_id': 't1', 'intelligibility': '4'})
        ledger = build_ledger(self.root)
        self.assertEqual(ledger['human_listening_status'], 'ratings_recorded_independent_review_pending')
        self.assertFalse(ledger['independent_confirmation_complete'])
        self.assertEqual(ledger['steps'][12]['human_listening']['valid_rating_cells'], 1)

    def test_duplicate_or_wrong_package_ratings_do_not_count(self):
        self.write('step13_listening/answer_key.json', {'package_id': 'p1', 'trials': [{'trial_id': 't1'}]})
        rater = self.root / 'step13_listening/rater'
        rater.mkdir()
        (rater / 'index.html').write_text('form', encoding='utf-8')
        (rater / 'bad.csv').write_text('package_id,trial_id,intelligibility,relevance,emotional_appropriateness,naturalness\n'
            'wrong,t1,5,5,5,5\n', encoding='utf-8')
        status = self.stage(13)['human_listening']
        self.assertEqual(status['valid_rating_cells'], 0)
        self.assertTrue(status['problems'])

    def test_explicit_external_listening_package_is_prepared_not_human_completed(self):
        external = tempfile.TemporaryDirectory()
        self.addCleanup(external.cleanup)
        package = Path(external.name)
        (package / 'answer_key.json').write_text(json.dumps({'package_id': 'external1',
            'human_ratings_collected': 999, 'trials': [{'trial_id': 'trial1'}]}), encoding='utf-8')
        (package / 'rater').mkdir()
        (package / 'rater/index.html').write_text('blank form', encoding='utf-8')
        self.assertEqual(self.stage(13)['status'], 'pending')
        ledger = build_ledger(self.root, [package, package])
        stage = ledger['steps'][12]
        self.assertEqual(stage['status'], 'prepared')
        self.assertEqual(stage['human_listening']['status'], 'pending_human')
        self.assertEqual(stage['human_listening']['valid_rating_cells'], 0)
        self.assertFalse(ledger['independent_confirmation_complete'])
        entries = [entry for entry in ledger['artifacts'] if entry.get('external_listening_package')]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]['path'], (package / 'answer_key.json').as_posix())
        self.assertIn('SHA256', markdown(ledger))
        self.assertTrue(stage['requirements'][0]['satisfied'])

    def test_missing_explicit_listening_package_remains_pending(self):
        ledger = build_ledger(self.root, [self.root / 'not_downloaded'])
        stage = ledger['steps'][12]
        self.assertEqual(stage['status'], 'pending')
        self.assertFalse(stage['human_listening']['package_prepared'])
        self.assertEqual(stage['failed_checks'], [])

    def test_missing_ar_branch_is_pending_and_not_a_live_job_claim(self):
        branch = self.stage(12)['autoregressive_feasibility']
        self.assertEqual(branch['status'], 'pending')
        self.assertEqual(branch['additional_adaptation_compute']['completed_updates_reported'], 0)
        self.assertFalse(branch['production_candidate_promoted'])
        self.write('step12_ar/prior/progress.json', {'step': 200, 'steps': 1600, 'phase': 'prior'})
        branch = self.stage(12)['autoregressive_feasibility']
        self.assertEqual(branch['status'], 'partial')
        self.assertEqual(branch['training']['prior']['latest_progress_snapshot']['step'], 200)
        self.assertEqual(branch['additional_adaptation_compute']['completed_updates_reported'], 0)
        self.assertNotIn('running', branch['status'])

    def test_recorded_ar_request_retains_missing_branch_as_pending(self):
        self.write('representation_review.json', {'run_ar_feasibility': True})
        stage = self.stage(12)
        self.assertTrue(stage['autoregressive_feasibility']['requested_in_recorded_review'])
        self.assertEqual(stage['autoregressive_feasibility']['status'], 'pending')
        self.assertTrue(any(item.startswith('AR prior:') for item in stage['pending_evidence']))

    def test_ar_training_curve_is_teacher_only_and_timings_not_summed(self):
        self.ar_training('prior')
        branch = self.stage(12)['autoregressive_feasibility']
        prior = branch['training']['prior']
        self.assertEqual(prior['status'], 'completed')
        self.assertEqual(prior['teacher_forced']['latest']['teacher_forced_unit_accuracy'], .8)
        self.assertEqual(prior['latest_process_timing']['this_process_optimizer_update_seconds'], 100.)
        self.assertEqual(branch['status'], 'partial')
        self.assertEqual(branch['evaluations']['prior']['free_running']['status'], 'pending')
        self.assertEqual(branch['additional_adaptation_compute']['completed_updates_with_required_evidence'], 1600)
        self.assertFalse(branch['additional_adaptation_compute']['includes_source_masked_prior_training'])
        self.assertFalse(branch['additional_adaptation_compute']['gpu_hours_established'])
        self.assertEqual(branch['quality_status'], 'not_established')

    def test_ar_complete_feasibility_keeps_bad_free_audio_separate_from_teacher_accuracy(self):
        for phase in ('prior', 'conditional'):
            self.ar_training(phase)
            self.ar_evaluation(phase, audio=phase == 'conditional')
        ledger = build_ledger(self.root)
        branch = ledger['steps'][11]['autoregressive_feasibility']
        self.assertEqual(branch['status'], 'completed')
        self.assertEqual(branch['additional_adaptation_compute']['completed_updates_reported'], 3200)
        self.assertEqual(branch['evaluations']['conditional']['teacher_forced']['metrics']['teacher_forced_unit_accuracy'], .8)
        self.assertEqual(branch['evaluations']['conditional']['audio']['summary']['predicted_length_ar']['mean_reference_wer'], 1.4)
        self.assertEqual(branch['evaluations']['conditional']['free_running']['quality_status'], 'not_established')
        self.assertEqual(branch['evaluations']['categorical']['status'], 'pending')
        self.assertFalse(branch['production_candidate_promoted'])
        rendered = markdown(ledger)
        self.assertIn('not an equal-compute architecture winner', rendered)
        self.assertIn('correct B prefix and oracle B length', rendered)
        self.assertIn('AR feasibility branch: **completed**', rendered)

    def test_categorical_audio_is_separate_and_requires_correct_metadata(self):
        self.ar_evaluation()
        base = self.ar_evaluation(categorical=True, audio=False)
        branch = self.stage(12)['autoregressive_feasibility']
        self.assertEqual(branch['evaluations']['categorical']['status'], 'partial')
        self.assertEqual(branch['evaluations']['categorical']['audio']['status'], 'pending')
        self.assertTrue(branch['categorical_matches_greedy_checkpoint'])
        self.ar_evaluation(categorical=True)
        self.assertEqual(self.stage(12)['autoregressive_feasibility']['evaluations']['categorical']['status'], 'completed')
        path = self.root / base / 'units_report.json'
        data = json.loads(path.read_text())
        data['planner_sampling']['mode'] = 'greedy'
        self.write(base + '/units_report.json', data)
        self.assertEqual(self.stage(12)['status'], 'failed')

    def test_ar_partial_unit_report_needs_final_frozen_audit(self):
        base = self.ar_evaluation(audited=False)
        evaluation = self.stage(12)['autoregressive_feasibility']['evaluations']['conditional']
        self.assertEqual(evaluation['status'], 'partial')
        self.assertEqual(evaluation['teacher_forced']['status'], 'recorded')
        self.assertIn('Final frozen-module evaluation audit', evaluation['pending_evidence'])
        data = json.loads((self.root / base / 'units_report.json').read_text())
        data['frozen_audit'] = {'passed': False, 'error': 'Frozen codec changed'}
        self.write(base + '/units_report.json', data)
        self.assertEqual(self.stage(12)['status'], 'failed')

    def test_ar_audio_downloaded_before_units_is_pending_not_failed(self):
        base = self.ar_evaluation()
        (self.root / base / 'units_report.json').unlink()
        stage = self.stage(12)
        self.assertEqual(stage['status'], 'partial')
        self.assertEqual(stage['autoregressive_feasibility']['evaluations']['conditional']['status'], 'partial')
        self.assertEqual(stage['failed_checks'], [])

    def test_ar_endpoint_without_matching_curve_or_audit_stays_partial(self):
        self.ar_training('conditional')
        path = self.root / 'step12_ar/conditional/metrics.jsonl'
        path.write_text(json.dumps({'step': 800, 'teacher_forced_validation': self.teacher()}) + '\n', encoding='utf-8')
        prior = self.stage(12)['autoregressive_feasibility']['training']['conditional']
        self.assertEqual(prior['status'], 'partial')
        self.assertIn('Teacher-forced validation at the completed endpoint', prior['pending_evidence'])
        self.assertEqual(prior['completed_updates_with_required_evidence'], 0)

    def test_invalid_ar_jsonl_does_not_hide_failure_behind_completion(self):
        self.ar_training('prior')
        path = self.root / 'step12_ar/prior/metrics.jsonl'
        with path.open('a', encoding='utf-8') as stream:
            stream.write('{invalid\n')
        stage = self.stage(12)
        self.assertEqual(stage['status'], 'failed')
        self.assertTrue(any(row['path'].endswith('metrics.jsonl') for row in stage['failed_checks']))

    def test_confirmation_controller_complete_without_evidence_is_partial(self):
        self.confirmation_protocol('seeds')
        stage = self.stage(13)
        phase = stage['computational_confirmation']['seeds']
        self.assertEqual(phase['status'], 'partial')
        self.assertFalse(any(job['completed'] for job in phase['jobs'].values()))
        self.assertEqual(stage['final_test']['status'], 'deferred')

    def test_completed_seed_replication_is_not_independent_initialization_or_human_confirmation(self):
        self.confirmation_seeds()
        ledger = build_ledger(self.root)
        phase = ledger['steps'][12]['computational_confirmation']['seeds']
        self.assertEqual(phase['status'], 'completed')
        self.assertEqual(sum(job['completed'] for job in phase['jobs'].values()), 23)
        self.assertIn('independent initialization is not replicated', phase['scope'])
        self.assertIn('seed 42 with B hints', phase['scope'])
        self.assertFalse(ledger['independent_confirmation_complete'])
        self.assertFalse(ledger['computational_confirmation_complete'])
        self.assertEqual(ledger['human_listening_status'], 'pending_human')
        self.assertEqual(ledger['final_test_status'], 'deferred')
        self.assertIn('23/23', markdown(ledger))

    def test_missing_confirmation_endpoint_audit_does_not_count_as_complete(self):
        self.confirmation_seeds()
        (self.root / 'step13_confirmation/seeds/high_lr_seed44/audit_step_0640.json').unlink()
        phase = self.stage(13)['computational_confirmation']['seeds']
        self.assertEqual(phase['status'], 'partial')
        self.assertFalse(phase['jobs']['high_lr_seed44_step_0640_audit']['completed'])

    def test_confirmation_receipt_hash_failure_is_not_hidden_by_completed_controller(self):
        self.confirmation_seeds()
        path = self.root / 'step13_confirmation/seeds/gaps/low_lr_seed42.json'
        data = json.loads(path.read_text())
        data['unexpected_change'] = True
        path.write_text(json.dumps(data), encoding='utf-8')
        stage = self.stage(13)
        self.assertEqual(stage['status'], 'failed')
        self.assertTrue(any('Receipt output hash differs' in failure['reason'] for failure in stage['failed_checks']))

    def test_fresh_confirmation_requires_ce_control_and_requested_categorical(self):
        self.confirmation_protocol('fresh', categorical=True)
        phase = self.stage(13)['computational_confirmation']['fresh']
        self.assertIn('ce_control_step160_fresh_controls', phase['jobs'])
        self.assertIn('ce_control_step160_fresh_audio', phase['jobs'])
        self.assertIn('ar_categorical_fresh', phase['jobs'])
        self.assertEqual(phase['status'], 'partial')

    def test_completed_fresh_reports_keep_teacher_free_and_human_claims_separate(self):
        protocol = self.confirmation_protocol('fresh', categorical=True)
        for name in protocol['fresh_conditional']['masked_candidates']:
            base = f'step13_confirmation/fresh/{name}'
            path = base + '/val_controls.json'
            self.write(path, {'examples': [{'conversation_id': i} for i in range(32)],
                'conditions': {'correct_a': {'accuracy': .03}}})
            self.confirmation_receipt('fresh', name + '_fresh_controls', [path])
            path = base + '/audio/report.json'
            self.confirmation_audio(path, 'predicted_length_steps_8')
            self.confirmation_receipt('fresh', name + '_fresh_audio', [path])
        for mode in ('greedy', 'categorical'):
            source = self.ar_evaluation(categorical=mode == 'categorical')
            report = json.loads((self.root / source / 'units_report.json').read_text())
            original = report['free_running']['examples'][0]
            report['free_running']['examples'] = [{**original, 'conversation_id': i} for i in range(32)]
            report['free_running']['count'] = 32
            report['teacher_forced']['examples'] = 32
            base = f'step13_confirmation/fresh/ar_{mode}'
            self.write(base + '/units_report.json', report)
            audio = self.confirmation_audio(base + '/audio/report.json', 'predicted_length_ar')
            audio.update({key: report[key] for key in ('architecture', 'phase', 'checkpoint_sha256',
                'checkpoint_step', 'planner_sampling', 'production_candidate_promoted')})
            self.write(base + '/audio/report.json', audio)
            self.confirmation_receipt('fresh', 'ar_' + mode + '_fresh',
                                      [base + '/units_report.json', base + '/audio/report.json'])
        ledger = build_ledger(self.root)
        phase = ledger['steps'][12]['computational_confirmation']['fresh']
        self.assertEqual(phase['status'], 'completed', phase['pending_evidence'])
        self.assertEqual(sum(job['completed'] for job in phase['jobs'].values()), 8)
        self.assertEqual(phase['comparisons']['ar_greedy']['teacher_forced']['metrics']['teacher_forced_unit_accuracy'], .8)
        self.assertEqual(phase['comparisons']['ar_greedy']['audio']['summary']['predicted_length_ar']['mean_reference_wer'], 1.1)
        self.assertFalse(phase['production_candidate_promoted'])
        self.assertEqual(phase['quality_status'], 'not_established')
        self.assertEqual(ledger['human_listening_status'], 'pending_human')
        self.assertFalse(ledger['independent_confirmation_complete'])


if __name__ == '__main__':
    unittest.main()
