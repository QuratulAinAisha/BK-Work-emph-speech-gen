"""Compare saved repair reports without promotion. / 승격 없이 저장된 개선 보고서를 비교합니다."""

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.diagnose_speech import word_error


REPORTS = {'train': 'train_controls.json', 'val': 'val_controls.json', 'audio': 'audio/report.json'}
LIMITATIONS = [
    'No candidate is automatically selected or promoted; WER alone is not a promotion criterion.',
    'Bootstrap units are conversations; explicitly repeated seeds are pooled inside each conversation before resampling.',
    'These intervals describe case sampling for fixed checkpoints, not uncertainty over independent training runs.',
    'All pairwise and within-candidate comparisons are descriptive, with no multiple-comparison correction.',
    'Unit accuracy uses recorded B targets and their length; it does not measure appropriate free responses.',
    'Reference WER can exceed 100% and is not an empathy, relevance or human listening score.',
    'ASR limit flags refer to the recognizer; uncapped ASR can still hallucinate.',
    'Eight audio conversations give limited precision; shared-uncapped subsets can be smaller and selected by recognizer behavior.',
    'This helper reads reports only: it does not load checkpoints, attest checkpoint aliases, listen to audio, or open dataset/test examples.',
]


def metadata(report, name):
    return report.get(name, report.get('identity', {}).get(name))


def finite(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise ValueError('Invalid ' + name)
    return float(value)


def measurement_seed(row, report):
    # Unlabeled duplicate rows must not masquerade as new seeds. / 라벨 없는 중복 행을 새 시드로 취급하지 않습니다.
    for scope in (row, report):
        values = [scope[key] for key in ('sampling_seed', 'mask_seed', 'seed') if key in scope]
        if values:
            if any(isinstance(value, (list, dict, bool)) or value is None for value in values):
                raise ValueError('Invalid explicit measurement seed')
            if len({str(value) for value in values}) != 1:
                raise ValueError('Conflicting measurement seeds')
            return str(values[0])
    return 'single_unseeded_measurement'


def grouped_rows(report):
    groups = {}
    for row in report.get('examples', []):
        cid = row.get('conversation_id')
        if not isinstance(cid, str) or not cid:
            raise ValueError('Missing conversation ID')
        seed = measurement_seed(row, report)
        if seed in groups.setdefault(cid, {}):
            raise ValueError('Duplicate conversation/seed: ' + cid + '/' + seed)
        groups[cid][seed] = row
    if not groups:
        raise ValueError('Report has no examples')
    seed_sets = {tuple(sorted(rows)) for rows in groups.values()}
    if len(seed_sets) != 1:
        raise ValueError('Conversations have different repeated seed sets')
    return groups


def normalize_controls(report):
    groups = grouped_rows(report)
    if metadata(report, 'target_length_supplied_for_unit_diagnostic') is not True:
        raise ValueError('Control report does not declare target-length scoring')
    names = None
    cases = {}
    for cid, rows in groups.items():
        signature, donor_signature = [], []
        values = {}
        target_identity = None
        for seed, row in sorted(rows.items()):
            frames = row['frames']
            if type(frames) is not int or frames < 1:
                raise ValueError('Invalid control frame count')
            targets = row['target_unit_ids']
            if len(targets) != frames or any(type(unit) is not int or unit < 0 for unit in targets):
                raise ValueError('Invalid target unit IDs')
            identity = (row.get('path'), frames, tuple(targets), finite(row['target_duration_seconds'], 'target duration'))
            if target_identity is not None and identity != target_identity:
                raise ValueError('Repeated seeds use different targets: ' + cid)
            target_identity = identity
            signature.append((seed, identity))
            donor_signature.append((seed, row.get('shuffled_path'), row.get('shuffled_conversation_id')))
            if names is None:
                names = set(row['conditions'])
            if not names or set(row['conditions']) != names:
                raise ValueError('Control condition names differ inside a report')
            for name, entry in row['conditions'].items():
                correct = entry['correct_units']
                if type(correct) is not int or not 0 <= correct <= frames:
                    raise ValueError('Invalid correct-unit count')
                if 'generated_unit_ids' in entry:
                    predicted = entry['generated_unit_ids']
                    if len(predicted) != frames or sum(a == b for a, b in zip(predicted, targets)) != correct:
                        raise ValueError('Correct-unit count disagrees with saved predictions')
                if 'unit_accuracy' in entry and not math.isclose(entry['unit_accuracy'], correct / frames, abs_tol=1e-7):
                    raise ValueError('Unit accuracy disagrees with counts')
                score = values.setdefault(name, {'correct': 0, 'frames': 0, 'ce_sum': 0., 'duration_errors': []})
                score['correct'] += correct
                score['frames'] += frames
                score['ce_sum'] += finite(entry['fully_masked_ce'], 'fully masked CE') * frames
                score['duration_errors'].append(finite(entry['duration_abs_error_seconds'], 'duration absolute error'))
        for score in values.values():
            score['duration_error'] = float(np.mean(score.pop('duration_errors')))
        cases[cid] = {'signature': signature, 'donor_signature': donor_signature, 'conditions': values,
                      'seeds': sorted(rows)}
    return {'cases': cases, 'conditions': sorted(names), 'metadata': {
        key: metadata(report, key) for key in ('manifest_sha256', 'selection_sha256', 'refinement_steps',
            'checkpoint', 'checkpoint_step', 'encoder_input_contract', 'planner_memory_mode')}}


def control_summary(normalized):
    result = {}
    for name in normalized['conditions']:
        values = [row['conditions'][name] for row in normalized['cases'].values()]
        frames = sum(row['frames'] for row in values)
        result[name] = {'conversations': len(values), 'seeds_per_conversation': len(next(iter(normalized['cases'].values()))['seeds']),
            'frame_unit_accuracy': sum(row['correct'] for row in values) / frames,
            'mean_conversation_unit_accuracy': float(np.mean([row['correct'] / row['frames'] for row in values])),
            'frame_fully_masked_ce': sum(row['ce_sum'] for row in values) / frames,
            'mean_conversation_fully_masked_ce': float(np.mean([row['ce_sum'] / row['frames'] for row in values])),
            'duration_mae_seconds': float(np.mean([row['duration_error'] for row in values]))}
    return result


def interval(point, samples):
    return {'difference': float(point), 'ci95_percentile': np.quantile(samples, [.025, .975]).tolist()}


def bootstrap_controls(first, second, first_condition, second_condition, draws=20000, seed=42):
    keys = sorted(first)
    if not keys or set(first) != set(second):
        raise ValueError('Control conversation sets differ')
    a = [first[key]['conditions'][first_condition] for key in keys]
    b = [second[key]['conditions'][second_condition] for key in keys]
    frames = np.asarray([row['frames'] for row in a], dtype=float)
    if not np.array_equal(frames, [row['frames'] for row in b]):
        raise ValueError('Pooled control denominators differ')
    samples = np.random.default_rng(seed).integers(0, len(keys), size=(draws, len(keys)))
    result = {'conversations': len(keys), 'conversation_ids': keys, 'metrics': {}}
    for field, title, scale in (('correct', 'unit_accuracy', 100), ('ce_sum', 'fully_masked_ce', 1)):
        delta = np.asarray([x[field] - y[field] for x, y in zip(a, b)])
        means = delta / frames
        suffix = '_delta_pp' if field == 'correct' else '_delta'
        result['metrics']['frame_' + title + suffix] = interval(scale * delta.sum() / frames.sum(),
            scale * delta[samples].sum(1) / frames[samples].sum(1))
        result['metrics']['mean_conversation_' + title + suffix] = interval(scale * means.mean(), scale * means[samples].mean(1))
    delta = np.asarray([x['duration_error'] - y['duration_error'] for x, y in zip(a, b)])
    result['metrics']['duration_mae_delta_seconds'] = interval(delta.mean(), delta[samples].mean(1))
    return result


def normalize_audio(report):
    groups = grouped_rows(report)
    names = None
    cases, transcripts = {}, []
    for cid, rows in groups.items():
        signatures, pooled = [], {}
        target_identity = None
        for seed, row in sorted(rows.items()):
            if not isinstance(row.get('reference_text'), str) or not isinstance(row.get('input_text'), str):
                raise ValueError('Audio report lacks exact A/B reference text')
            identity = (row.get('path'), row['input_text'], row['reference_text'], row.get('style_id'), row.get('speaker_id'))
            if target_identity is not None and identity != target_identity:
                raise ValueError('Repeated audio seeds use different references: ' + cid)
            target_identity = identity
            signatures.append((seed, identity))
            if names is None:
                names = set(row['paths'])
            if not names or set(row['paths']) != names:
                raise ValueError('Audio path names differ inside a report')
            text_row = {'conversation_id': cid, 'seed': seed, 'input_text': row['input_text'],
                        'reference_text': row['reference_text'], 'paths': {}}
            for name, entry in row['paths'].items():
                if not isinstance(entry.get('asr'), str) or type(entry.get('asr_token_limit_reached')) is not bool:
                    raise ValueError('Missing ASR text or explicit token-limit flag')
                wer = finite(entry['reference_wer'], 'reference WER')
                if not math.isclose(wer, word_error(row['reference_text'], entry['asr']), abs_tol=1e-9):
                    raise ValueError('Reported WER disagrees with exact ASR/reference text: ' + cid + '/' + name)
                value = pooled.setdefault(name, {'wers': [], 'capped_seeds': []})
                value['wers'].append(wer)
                if entry['asr_token_limit_reached']:
                    value['capped_seeds'].append(seed)
                text_row['paths'][name] = {key: entry.get(key) for key in ('asr', 'reference_wer', 'asr_token_limit_reached', 'file', 'seconds')}
            transcripts.append(text_row)
        cases[cid] = {'signature': signatures, 'seeds': sorted(rows), 'paths': {
            name: {'mean_wer': float(np.mean(value['wers'])), 'capped_seeds': value['capped_seeds']}
            for name, value in pooled.items()}}
    return {'cases': cases, 'paths': sorted(names), 'transcripts': transcripts,
            'asr_model': metadata(report, 'asr_model')}


def audio_summary(normalized):
    result = {}
    for name in normalized['paths']:
        rows = [row['paths'][name] for row in normalized['cases'].values()]
        valid = [row['mean_wer'] for row in rows if not row['capped_seeds']]
        result[name] = {'conversations': len(rows), 'mean_reference_wer': float(np.mean([row['mean_wer'] for row in rows])),
            'capped_conversations': sum(bool(row['capped_seeds']) for row in rows),
            'capped_measurements': sum(len(row['capped_seeds']) for row in rows),
            'uncapped_conversations': len(valid), 'uncapped_mean_reference_wer': float(np.mean(valid)) if valid else None}
    return result


def bootstrap_audio(first, second, path, draws=20000, seed=42, uncapped=False):
    if set(first) != set(second):
        raise ValueError('Audio conversation sets differ')
    keys = [key for key in sorted(first) if not uncapped or (
        not first[key]['paths'][path]['capped_seeds'] and not second[key]['paths'][path]['capped_seeds'])]
    result = {'conversations': len(keys), 'conversation_ids': keys, 'wer_delta_pp': None}
    if keys:
        a = np.asarray([first[key]['paths'][path]['mean_wer'] for key in keys])
        b = np.asarray([second[key]['paths'][path]['mean_wer'] for key in keys])
        delta = a - b
        samples = np.random.default_rng(seed).integers(0, len(keys), size=(draws, len(keys)))
        result.update(first_mean_wer=float(a.mean()), second_mean_wer=float(b.mean()),
                      wer_delta_pp=interval(100 * delta.mean(), 100 * delta[samples].mean(1)))
    return result


def matching_errors(first, second, donor=False):
    if set(first) != set(second):
        return [{'reason': 'conversation_sets_differ', 'only_first': sorted(set(first) - set(second)),
                 'only_second': sorted(set(second) - set(first))}]
    errors = []
    for cid in sorted(first):
        if first[cid]['signature'] != second[cid]['signature']:
            errors.append({'conversation_id': cid, 'reason': 'target_reference_or_seed_mismatch'})
        if donor and first[cid]['donor_signature'] != second[cid]['donor_signature']:
            errors.append({'conversation_id': cid, 'reason': 'shuffled_donor_mismatch'})
    return errors


def candidate_argument(value):
    label, separator, folder = value.partition('=')
    if not separator or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', label) or not folder:
        raise argparse.ArgumentTypeError('Use label=directory with a simple unique label')
    return label, Path(folder)


def build_comparison(candidates, draws=20000, seed=42):
    if draws < 1:
        raise ValueError('Positive bootstrap draws required')
    report = {'automatic_promotion': False, 'candidate_selected': None,
        'bootstrap': {'draws': draws, 'seed': seed, 'unit': 'conversation',
                      'seed_pooling': 'Pool explicitly repeated seeds within each conversation; require identical seed sets.'},
        'difference_direction': 'First named candidate minus second. Positive accuracy is higher; positive CE/MAE/WER is higher error.',
        'limitations': LIMITATIONS, 'candidates': {}, 'comparisons': {}, 'summary_table': [], 'source_hashes': {}}
    normalized = {}
    for label, directory in candidates:
        if label in normalized:
            raise ValueError('Duplicate candidate label: ' + label)
        result = {'directory': str(directory.resolve()), 'reports': {}, 'flags': [], 'within_candidate_controls': {}}
        normalized[label] = {}
        checkpoint_paths, checkpoint_steps = set(), set()
        for kind, relative in REPORTS.items():
            path = directory / relative
            if not path.exists():
                result['reports'][kind] = {'status': 'missing'}
                continue
            content = path.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            report['source_hashes'][str(path.resolve())] = digest
            raw = json.loads(content)
            result['reports'][kind] = {'source': str(path.resolve()), 'sha256': digest}
            checkpoint = metadata(raw, 'checkpoint')
            step = metadata(raw, 'checkpoint_step')
            if step is None:
                step = metadata(raw, 'recovery_step')
            if checkpoint:
                checkpoint_paths.add(checkpoint.replace('\\', '/'))
            if step is not None:
                checkpoint_steps.add(step)
            try:
                value = normalize_audio(raw) if kind == 'audio' else normalize_controls(raw)
                normalized[label][kind] = value
                summary = audio_summary(value) if kind == 'audio' else control_summary(value)
                result['reports'][kind].update(status='valid', summary=summary)
                result['reports'][kind]['metadata'] = value.get('metadata', {
                    'checkpoint': checkpoint, 'checkpoint_step': step, 'asr_model': value.get('asr_model')})
                if kind == 'audio':
                    result['reports'][kind]['exact_transcripts'] = value['transcripts']
                    if len(value['cases']) != 8:
                        result['flags'].append({'report': kind, 'reason': 'audio_count_not_8', 'count': len(value['cases'])})
                    for name, scores in summary.items():
                        if scores['capped_conversations']:
                            result['flags'].append({'report': kind, 'path': name, 'reason': 'asr_token_limit_reached',
                                                    'conversations': scores['capped_conversations']})
                else:
                    if kind == 'val' and len(value['cases']) != 128:
                        result['flags'].append({'report': kind, 'reason': 'development_count_not_128', 'count': len(value['cases'])})
                    if metadata(raw, 'encoder_input_contract') != 'person_a_only':
                        result['flags'].append({'report': kind, 'reason': 'A_only_contract_unverified'})
                    if 'correct_a' in value['conditions']:
                        for other in ('shuffled_a', 'zero_a'):
                            if other in value['conditions']:
                                result['within_candidate_controls'][kind + '/correct_a_minus_' + other] = bootstrap_controls(
                                    value['cases'], value['cases'], 'correct_a', other, draws, seed)
                report['summary_table'].extend({'candidate': label, 'report': kind, 'condition': name, **scores}
                                               for name, scores in summary.items())
            except (ValueError, KeyError, TypeError) as error:
                normalized[label].pop(kind, None)
                result['reports'][kind].update(status='invalid', error=str(error))
                result['flags'].append({'report': kind, 'reason': 'invalid_report', 'detail': str(error)})
        if all(value['status'] == 'missing' for value in result['reports'].values()):
            raise ValueError('Candidate contains none of the supported reports: ' + label)
        if len(checkpoint_paths) > 1:
            result['flags'].append({'reason': 'checkpoint_paths_differ_unverified_alias', 'paths': sorted(checkpoint_paths)})
        if len(checkpoint_steps) > 1:
            result['flags'].append({'reason': 'checkpoint_steps_differ', 'steps': sorted(checkpoint_steps)})
        if 'train' in normalized[label] and 'val' in normalized[label]:
            overlap = set(normalized[label]['train']['cases']) & set(normalized[label]['val']['cases'])
            if overlap:
                result['flags'].append({'reason': 'train_val_conversation_overlap', 'conversation_ids': sorted(overlap)})
        if 'audio' in normalized[label] and 'val' in normalized[label]:
            outside = set(normalized[label]['audio']['cases']) - set(normalized[label]['val']['cases'])
            if outside:
                result['flags'].append({'reason': 'audio_cases_outside_control_validation', 'conversation_ids': sorted(outside)})
        report['candidates'][label] = result

    for first_label, second_label in itertools.combinations(normalized, 2):
        pair = {'first': first_label, 'second': second_label, 'reports': {}}
        for kind in REPORTS:
            if kind not in normalized[first_label] or kind not in normalized[second_label]:
                pair['reports'][kind] = {'status': 'unavailable'}
                continue
            first, second = normalized[first_label][kind], normalized[second_label][kind]
            errors = matching_errors(first['cases'], second['cases'])
            warnings = []
            if kind == 'audio':
                if first['asr_model'] != second['asr_model']:
                    errors.append({'reason': 'asr_models_differ'})
                elif first['asr_model'] is None:
                    warnings.append('ASR model identity is not recorded.')
            else:
                a, b = first['metadata']['manifest_sha256'], second['metadata']['manifest_sha256']
                if not a or a != b:
                    errors.append({'reason': 'manifest_identity_missing_or_different'})
                for field in ('selection_sha256', 'refinement_steps', 'encoder_input_contract'):
                    if first['metadata'][field] != second['metadata'][field]:
                        warnings.append(field + ' differs; case/target membership is checked separately.')
            if errors:
                pair['reports'][kind] = {'status': 'mismatch', 'mismatches': errors}
                continue
            metrics = {}
            names = 'paths' if kind == 'audio' else 'conditions'
            mismatch_names = sorted(set(first[names]) ^ set(second[names]))
            for name in sorted(set(first[names]) & set(second[names])):
                if kind == 'audio':
                    metrics[name] = {
                        'all_cases': bootstrap_audio(first['cases'], second['cases'], name, draws, seed),
                        'shared_uncapped_cases': bootstrap_audio(first['cases'], second['cases'], name, draws, seed, True)}
                else:
                    donors = matching_errors(first['cases'], second['cases'], donor=True) if name == 'shuffled_a' else []
                    metrics[name] = {'status': 'mismatch', 'mismatches': donors} if donors else bootstrap_controls(
                        first['cases'], second['cases'], name, name, draws, seed)
            pair['reports'][kind] = {'status': 'matched', 'warnings': warnings,
                                    'conditions_only_in_one_candidate': mismatch_names, 'conditions': metrics}
        report['comparisons'][first_label + '_minus_' + second_label] = pair
    return report


def markdown(report):
    def safe(value):
        return str(value).replace('|', '\\|').replace('\n', '<br>')

    lines = ['# Planner repair candidate comparison', '', 'No candidate is automatically promoted. Accuracy and WER do not establish response appropriateness.', '',
             '| Candidate | Report | Condition | Conversations | Unit accuracy | CE | Duration MAE | Mean WER | ASR-capped cases |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for row in report['summary_table']:
        lines.append('| ' + ' | '.join(safe(value) for value in (
            row['candidate'], row['report'], row['condition'], row['conversations'],
            f"{100 * row['frame_unit_accuracy']:.3f}%" if 'frame_unit_accuracy' in row else '',
            f"{row['frame_fully_masked_ce']:.5f}" if 'frame_fully_masked_ce' in row else '',
            f"{row['duration_mae_seconds']:.3f} s" if 'duration_mae_seconds' in row else '',
            f"{100 * row['mean_reference_wer']:.2f}%" if 'mean_reference_wer' in row else '', row.get('capped_conversations', ''))) + ' |')
    lines += ['', '## Paired confidence intervals', '', 'Differences are first candidate minus second. Seeds are pooled within conversations; intervals do not measure training-seed uncertainty.', '',
              '| Comparison | Report / condition | Metric | Difference | 95% interval | Cases |', '|---|---|---|---:|---|---:|']
    for name, pair in report['comparisons'].items():
        for kind, stage in pair['reports'].items():
            if stage['status'] != 'matched':
                lines.append(f"| {safe(name)} | {kind} | {stage['status']} | | | |")
                continue
            for condition, values in stage['conditions'].items():
                if values.get('status') == 'mismatch':
                    lines.append(f'| {safe(name)} | {kind}/{safe(condition)} | donor mismatch | | | |')
                    continue
                entries = values['metrics'].items() if kind != 'audio' else ((suffix + '/WER_delta_pp', score.get('wer_delta_pp'))
                    for suffix, score in values.items())
                for metric, value in entries:
                    if value is None:
                        continue
                    count = values['conversations'] if kind != 'audio' else values[metric.split('/')[0]]['conversations']
                    lo, hi = value['ci95_percentile']
                    lines.append(f"| {safe(name)} | {kind}/{safe(condition)} | {safe(metric)} | {value['difference']:.4f} | [{lo:.4f}, {hi:.4f}] | {count} |")
    lines += ['', '## Flags', '']
    for label, candidate in report['candidates'].items():
        for flag in candidate['flags']:
            lines.append('- ' + safe(label) + ': ' + safe(json.dumps(flag, ensure_ascii=False)))
    for name, pair in report['comparisons'].items():
        for kind, stage in pair['reports'].items():
            if stage.get('mismatches') or stage.get('conditions_only_in_one_candidate'):
                lines.append('- ' + safe(name + '/' + kind) + ': ' + safe(json.dumps(stage.get('mismatches', stage.get('conditions_only_in_one_candidate')))))
            for warning in stage.get('warnings', []):
                lines.append('- ' + safe(name + '/' + kind) + ': ' + safe(warning))
    lines += ['', '## Exact example transcripts', '']
    for label, candidate in report['candidates'].items():
        for row in candidate['reports']['audio'].get('exact_transcripts', []):
            lines += [f"### {safe(label)} — {safe(row['conversation_id'])} — seed {safe(row['seed'])}", '',
                      '**A:** ' + safe(row['input_text']), '', '**Recorded B:** ' + safe(row['reference_text']), '',
                      '| Path | ASR transcript | WER | ASR cap |', '|---|---|---:|---|']
            for name, entry in row['paths'].items():
                lines.append(f"| {safe(name)} | {safe(entry['asr'])} | {100 * entry['reference_wer']:.2f}% | {entry['asr_token_limit_reached']} |")
            lines.append('')
    lines += ['## Limits', ''] + ['- ' + value for value in report['limitations']]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', action='append', required=True, type=candidate_argument)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--draws', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output already exists; no files will be overwritten')
    if args.draws < 1000:
        parser.error('--draws must be at least 1000')
    report = build_comparison(args.candidate, args.draws, args.seed)
    for filename, expected in report['source_hashes'].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise ValueError('Source report changed during comparison: ' + filename)
    # Create a fresh result directory only after validation. / 검증 후 새 결과 폴더만 만듭니다.
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'comparison.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (args.output / 'comparison.md').write_text(markdown(report), encoding='utf-8')
    fields = sorted({key for row in report['summary_table'] for key in row})
    with (args.output / 'summary.csv').open('x', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report['summary_table'])
    print(json.dumps({'output': str(args.output), 'candidate_selected': None,
                      'flags': {label: value['flags'] for label, value in report['candidates'].items()}}), flush=True)


if __name__ == '__main__':
    main()
