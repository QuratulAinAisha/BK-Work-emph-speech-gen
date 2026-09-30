"""Controlled duration and Module 3 audio checks. / 길이와 모듈 3의 오디오 영향을 검사합니다."""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.signal import find_peaks
import soundfile as sf
import torch

from dataset.audio_affect_targets import audio_affect_targets
from dataset.quality_speech_dataset import QualitySpeechDataset, collate_quality, person_a_only
from model.affective_response_transport import AFFECT_FEATURES
from model.full_speech.codec import FrozenEncodec
from model.full_speech.loading import load_response_model
from model.full_speech.tensor_ops import align, masked_mean
from prepare_quality import atomic_json
from scripts.diagnose_quality import recognize, save_wave
from scripts.evaluate_planner_controls import next_different_conversation
from train_full import move_batch


CONDITIONS = ('predicted_affect', 'zero_affect', 'shuffled_affect')
DURATION_POLICIES = ('a_predicted_duration_locked', 'oracle_b_duration')
SAMPLE_RATE = 32000
FRAME_HZ = 25.
SEED = 42
LIMITATIONS = [
    'Zeroing all six controls is an ablation, not physiological or emotional neutrality; it can be out of distribution.',
    'Shuffled affect comes from a different A, with the original requested style/voice; it is time-resampled to the original A timeline.',
    'All A context, native speech memory, masks, requested style/voice and sampling seeds are held fixed across affect variants.',
    'A-predicted duration is locked across affect variants to isolate trajectory effects; each variant\'s free duration prediction is also reported.',
    'Only oracle_b_duration uses a B label for generation. No B affect, semantic features, text or audio is supplied to the generator.',
    'Changing affect can also change generated semantic units; these are total downstream effects, not a decoder-only prosody experiment.',
    'Pitch is an autocorrelation estimate on voiced frames; it can be wrong on noisy or unvoiced audio.',
    'Envelope peak rate is a waveform activity proxy, not a count of words, syllables or intelligible speech.',
    'ASR word rate is a recognizer-based proxy, not verified speaking rate; token-capped or hallucinated ASR can be misleading.',
    'Low-energy and terminal-energy metrics do not establish silence, sentence completion or truncation.',
    'Valence, arousal and dominance have no verified labels here; this diagnostic cannot validate their emotional meaning.',
    'Reference WER against one recorded B reply does not measure response appropriateness or empathy.',
    'ASR is not human listening; a transcript or audio change alone does not establish improved quality.',
]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def selected_indices(records, selection):
    """Use only predeclared development rows. / 미리 정한 개발 데이터만 사용합니다."""
    wanted = selection['val']
    mapping = {row['path']: index for index, row in enumerate(records)}
    if not wanted or len(wanted) != len(set(wanted)) or any(path not in mapping for path in wanted):
        raise ValueError('Selected validation paths are empty, duplicated or missing')
    return [mapping[path] for path in wanted]


@contextmanager
def override_encoded_affect(model, inputs, encoded, affect):
    """Restore the instance method even on failure. / 실패해도 인스턴스 메서드를 복원합니다."""
    if model.training or affect.shape != encoded['affect'].shape or not torch.isfinite(affect).all():
        raise ValueError('Expected eval mode and a finite, shape-matched trajectory')
    mask = encoded['context_mask']
    replacement = {**encoded, 'affect': affect.masked_fill(~mask[..., None], 0),
                   'affect_summary': masked_mean(affect, mask)}

    def fixed_encoder(received):
        # Reject hidden B inputs and any unintended A changes. / 숨은 B 입력과 의도하지 않은 A 변경을 거부합니다.
        if set(received) != set(person_a_only(inputs)) or any(
                not torch.equal(received[key], inputs[key]) for key in inputs):
            raise ValueError('Controlled generation changed A inputs or forwarded B fields')
        return dict(replacement)

    with patch.object(model, 'encode_batch', side_effect=fixed_encoder):
        yield replacement


@torch.inference_mode()
def generate_conditions(model, batch, donor_batch, codec, seed=SEED):
    """Only trajectory changes within each duration policy. / 각 길이 조건에서는 궤적만 바꿉니다."""
    if model.training or len(batch['style_id']) != 1 or len(donor_batch['style_id']) != 1:
        raise ValueError('Use eval mode and one example per batch')
    if int(batch['conversation_key'][0]) == int(donor_batch['conversation_key'][0]):
        raise ValueError('Shuffled affect requires another conversation')
    inputs, donor_inputs = person_a_only(batch), person_a_only(donor_batch)
    donor_inputs.update(style_id=inputs['style_id'], speaker_id=inputs['speaker_id'])
    encoded, donor = model.encode_batch(inputs), model.encode_batch(donor_inputs)
    affects = {
        'predicted_affect': encoded['affect'],
        'zero_affect': torch.zeros_like(encoded['affect']),
        'shuffled_affect': align(donor['affect'], donor['context_mask'], encoded['context_mask']),
    }
    style, _ = model.embeddings(inputs['style_id'], inputs['speaker_id'], 1)
    durations = {}
    for name, affect in affects.items():
        duration, _ = model.length_predictor(encoded['context'], affect, encoded['context_mask'], style)
        if not torch.isfinite(duration).all() or (duration <= 0).any():
            raise ValueError('Invalid predicted duration')
        durations[name] = duration
    oracle = batch['duration']
    if not torch.isfinite(oracle).all() or (oracle <= 0).any():
        raise ValueError('Invalid reference duration')
    results = {}
    for policy, duration in zip(DURATION_POLICIES, (durations['predicted_affect'], oracle)):
        for name, affect in affects.items():
            # This argument can carry an A prediction; its name does not make it a B label. / 이 인자는 B 정답 대신 A 예측도 전달합니다.
            with override_encoded_affect(model, inputs, encoded, affect):
                result = model.generate_batch(inputs, codec, seed=seed, oracle_duration=duration)
            if not torch.equal(result['duration'], duration):
                raise AssertionError('Generation did not preserve the locked duration')
            results[policy + '__' + name] = result
    return {'results': results, 'affects': affects, 'mask': encoded['context_mask'],
            'predicted_durations': durations, 'reference_duration': oracle,
            'source_affect_frames': int(encoded['context_mask'].sum()),
            'donor_affect_frames': int(donor['context_mask'].sum())}


def waveform_metrics(waveform, sample_rate=SAMPLE_RATE):
    """Use the dataset pitch/energy extractor; no transcript enters waveform metrics. / 데이터셋 피치·에너지 추출기를 씁니다."""
    wave = np.asarray(waveform, dtype=np.float32)
    if wave.ndim != 1 or not len(wave) or not np.isfinite(wave).all() or sample_rate < 1000:
        raise ValueError('Expected finite nonempty mono audio and a valid sample rate')
    values, weights = audio_affect_targets(wave, sample_rate, '', frame_hz=FRAME_HZ)
    voiced = weights[:, 2] > 0
    pitch = 60 * (500 / 60) ** values[voiced, 2]
    peaks, _ = find_peaks(values[:, 3], height=.5, prominence=.08, distance=3)
    duration = len(wave) / sample_rate
    # Nonoverlapping 20 ms windows define low-energy duration. / 겹치지 않는 20 ms 창으로 저에너지 길이를 정의합니다.
    width = max(1, round(sample_rate * .02))
    chunks = [wave[start:start + width].astype(np.float64) for start in range(0, len(wave), width)]
    rms = np.asarray([np.sqrt(np.mean(chunk ** 2)) for chunk in chunks])
    chunk_lengths = np.asarray([len(chunk) for chunk in chunks])
    quiet = rms < 1e-3
    active = np.flatnonzero(~quiet)
    leading = int(chunk_lengths[:active[0]].sum()) if len(active) else len(wave)
    trailing = int(chunk_lengths[active[-1] + 1:].sum()) if len(active) else len(wave)
    final = wave[-max(1, round(sample_rate * .2)):].astype(np.float64)
    total_rms = float(np.sqrt(np.mean(wave.astype(np.float64) ** 2)))
    return {
        'samples': len(wave), 'seconds': duration, 'raw_rms': total_rms,
        'raw_rms_dbfs': float(20 * np.log10(max(total_rms, 1e-12))),
        'raw_peak': float(np.abs(wave).max()), 'raw_clip_fraction': float((np.abs(wave) >= 1).mean()),
        'pitch_median_hz_voiced': float(np.median(pitch)) if len(pitch) else None,
        'pitch_p10_hz_voiced': float(np.quantile(pitch, .1)) if len(pitch) else None,
        'pitch_p90_hz_voiced': float(np.quantile(pitch, .9)) if len(pitch) else None,
        'voiced_frame_fraction': float(voiced.mean()),
        'normalized_energy_mean': float(values[:, 3].mean()),
        'envelope_peak_count': len(peaks), 'envelope_peaks_per_second': len(peaks) / duration,
        'low_energy_fraction': float(chunk_lengths[quiet].sum() / len(wave)),
        'leading_low_energy_seconds': leading / sample_rate, 'trailing_low_energy_seconds': trailing / sample_rate,
        'last_200ms_rms': float(np.sqrt(np.mean(final ** 2))),
        'last_window_low_energy': bool(quiet[-1]), 'last_sample_absolute': float(abs(wave[-1])),
    }


def control_waveform_metrics(affect, mask, waveform):
    """Compare normalized acoustic controls only. / 정규화된 음향 제어값만 비교합니다."""
    values, weights = audio_affect_targets(waveform, SAMPLE_RATE, '', frame_hz=FRAME_HZ)
    destination = torch.ones(1, len(values), dtype=torch.bool, device=affect.device)
    wanted = align(affect, mask, destination)[0].float().cpu().numpy()
    output = {}
    for name, channel in (('pitch', 2), ('energy', 3)):
        valid = weights[:, channel] > 0
        output[name + '_normalized_mae'] = float(np.abs(wanted[valid, channel] - values[valid, channel]).mean()) if valid.any() else None
        output[name + '_measured_frames'] = int(valid.sum())
    return output


def summarize(report):
    summary = {}
    for condition in CONDITIONS:
        values = [row['duration_predictions'][condition] for row in report['examples']]
        summary[condition] = {
            'mean_predicted_seconds': float(np.mean([x['seconds'] for x in values])),
            'mean_absolute_error_seconds': float(np.mean([x['absolute_error_seconds'] for x in values])),
            'mean_signed_error_seconds': float(np.mean([x['signed_error_seconds'] for x in values])),
        }
    report['duration_summary'] = summary
    for name in report['examples'][0]['paths']:
        paths = [row['paths'][name] for row in report['examples']]
        metrics = [entry['waveform_metrics'] for entry in paths]
        measured = {}
        for key in metrics[0]:
            if key == 'last_window_low_energy':
                continue
            valid = [entry[key] for entry in metrics if entry[key] is not None]
            measured['mean_' + key] = float(np.mean(valid)) if valid else None
        measured.update(pitch_measurable_cases=sum(x['pitch_median_hz_voiced'] is not None for x in metrics),
                        any_raw_clipping_cases=sum(x['raw_clip_fraction'] > 0 for x in metrics),
                        entirely_low_energy_cases=sum(x['low_energy_fraction'] == 1 for x in metrics),
                        terminal_window_low_energy_cases=sum(x['last_window_low_energy'] for x in metrics),
                        asr_token_limit_cases=sum(x['asr_token_limit_reached'] for x in paths))
        report['summary'][name]['waveform_metrics'] = measured
    return report


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    for name in ('checkpoint', 'manifest', 'selection', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--count', type=int, default=8)
    args = parser.parse_args()
    if args.count < 1:
        parser.error('--count must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use a new or empty output directory to avoid mixing diagnostic runs')
    torch.set_num_threads(2)
    checkpoint_hash, manifest_hash = file_hash(args.checkpoint), file_hash(args.manifest)
    selection = json.loads(args.selection.read_text())
    if selection.get('manifest_sha256') != manifest_hash:
        raise ValueError('Selection and manifest hashes differ')
    model, payload = load_response_model(args.checkpoint)
    model.to(args.device).eval().requires_grad_(False)
    data = QualitySpeechDataset(args.manifest, model.config, 'val')
    indices = selected_indices(data.records, selection)
    records = [data.records[index] for index in indices]
    # Select donors before count truncation, including --count 1. / count로 자르기 전에 대조 대화를 선택합니다.
    donors = next_different_conversation(records)
    if args.count > len(indices):
        raise ValueError('--count exceeds the predeclared validation selection')
    codec = FrozenEncodec().to(args.device).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        'checkpoint': str(args.checkpoint), 'checkpoint_sha256': checkpoint_hash,
        'checkpoint_step': payload.get('recovery_step'), 'checkpoint_epoch': payload.get('epoch'),
        'manifest_sha256': manifest_hash, 'selection_sha256': file_hash(args.selection),
        'split': 'val', 'count': args.count, 'encoder_input_contract': 'person_a_only',
        'planner_memory_mode': getattr(model.config, 'planner_memory_mode', 'fused'),
        'semantic_steps': model.config.semantic_steps, 'codec_steps': model.config.codec_steps,
        'shared_seed': SEED, 'duration_policies': list(DURATION_POLICIES), 'affect_conditions': list(CONDITIONS),
        'asr_model': 'openai/whisper-base.en', 'test_examples_opened': 0,
        'conditions_note': 'The API oracle_duration argument also carries the frozen A-predicted duration; only oracle_b_duration contains a B target.',
        'metric_definitions': {
            'pitch_and_energy': 'dataset.audio_affect_targets: 60 ms windows, 25 Hz, autocorrelation 60–500 Hz, voiced strength >=0.5 and RMS >=0.001.',
            'envelope_peak_rate': 'Local peaks in normalized frame energy; height >=0.5 (-30 dBFS), prominence >=0.08 (4.8 dB), spacing >=3 frames (120 ms), divided by full audio seconds.',
            'low_energy': 'RMS <0.001 (-60 dBFS) in nonoverlapping 20 ms windows; final partial window uses its actual length.',
            'ending': 'RMS in final min(200 ms, full length), final 20 ms-window low-energy flag, and absolute final sample. None is a semantic completion score.',
            'asr_rate': 'Same word tokenizer as audio_affect_targets, applied to Whisper text, divided by full audio duration; a proxy only.',
            'raw_vs_saved': 'Waveform metrics use raw decoder output. WAVs use diagnose_quality.save_wave peak attenuation; ASR and its word rate use those saved WAVs.',
            'control_mae': 'Predicted/shuffled/zero pitch and energy controls are linearly aligned to measured 25 Hz frames; pitch MAE uses voiced frames only. No emotional labels are scored.',
        },
        'limitations': LIMITATIONS, 'examples': [],
    }
    atomic_json(args.output / 'identity.json', {key: value for key, value in report.items() if key != 'examples'})

    def load(index):
        return move_batch(collate_quality([data[index]]), torch.device(args.device))

    for position, index in enumerate(indices[:args.count]):
        batch, donor = load(index), load(indices[donors[position]])
        generated = generate_conditions(model, batch, donor, codec)
        row = records[position]
        entry = {key: row[key] for key in ('path', 'conversation_id', 'input_text', 'style_id', 'speaker_id')}
        reference_duration = float(batch['duration'].item())
        entry.update(reference_text=row['response_text'], reference_duration_seconds=reference_duration,
                     shuffled_path=records[donors[position]]['path'],
                     shuffled_conversation_id=records[donors[position]]['conversation_id'],
                     source_affect_frames=generated['source_affect_frames'], donor_affect_frames=generated['donor_affect_frames'],
                     duration_predictions={}, affect_controls={}, paths={})
        for name, duration in generated['predicted_durations'].items():
            seconds = float(duration.item())
            entry['duration_predictions'][name] = {'seconds': seconds, 'absolute_error_seconds': abs(seconds - reference_duration),
                                                  'signed_error_seconds': seconds - reference_duration}
            values = generated['affects'][name][generated['mask']].cpu().numpy()
            entry['affect_controls'][name] = {feature: {'mean': float(values[:, j].mean()), 'min': float(values[:, j].min()),
                                                       'max': float(values[:, j].max())} for j, feature in enumerate(AFFECT_FEATURES)}
        affect_file = f'{position:03d}_affect_controls.npz'
        np.savez_compressed(args.output / affect_file, **{name: value[generated['mask']].cpu().numpy()
                            for name, value in generated['affects'].items()})
        entry['affect_controls_file'] = affect_file
        waves = {'reference': batch['waveform'][0, :int(batch['waveform_len'][0])].cpu().numpy()}
        for name, result in generated['results'].items():
            waves[name] = result['waveform'][0, :int(result['audio_lengths'][0])].cpu().numpy()
        for name, wave in waves.items():
            path = save_wave(args.output, f'{position:03d}_{name}.wav', wave)
            path['waveform_metrics'] = waveform_metrics(wave)
            if name != 'reference':
                policy, condition = name.split('__')
                path.update(duration_policy=policy, affect_condition=condition,
                            control_waveform_metrics=control_waveform_metrics(generated['affects'][condition], generated['mask'], wave))
                result = generated['results'][name]
                requested = int((result['duration'] * SAMPLE_RATE).round().item())
                path['requested_samples'] = requested
                path['sample_count_error'] = len(wave) - requested
                baseline = waves[policy + '__predicted_affect']
                if len(wave) != len(baseline):
                    raise AssertionError('Affect variants have different locked waveform lengths')
                path['raw_wave_rms_difference_vs_predicted_affect'] = float(np.sqrt(np.mean((wave.astype(np.float64) - baseline) ** 2)))
            entry['paths'][name] = path
        report['examples'].append(entry)
        atomic_json(args.output / 'generated_report.json', report)
        print(json.dumps({'generated_cases': position + 1, 'total': args.count}), flush=True)
        del generated, batch, donor
    del model, codec
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()
    report = recognize(report, args.output, args.device)
    for row in report['examples']:
        for path in row['paths'].values():
            words = re.findall(r"\b[\w]+(?:['’-][\w]+)*\b", path['asr'], flags=re.UNICODE)
            wave, rate = sf.read(args.output / path['file'], dtype='float32')
            controls, _ = audio_affect_targets(wave, rate, path['asr'], frame_hz=FRAME_HZ)
            path['asr_rate_proxy'] = {'words': len(words), 'words_per_second': len(words) / path['seconds'],
                                      'dataset_normalized_rate': float(controls[0, 4]),
                                      'token_limit_reached': path['asr_token_limit_reached']}
    if (file_hash(args.checkpoint) != checkpoint_hash or file_hash(args.manifest) != manifest_hash or
            file_hash(args.selection) != report['selection_sha256']):
        raise RuntimeError('Checkpoint, manifest or selection changed during evaluation')
    report['checkpoint_unchanged'] = True
    atomic_json(args.output / 'report.json', summarize(report))
    print(json.dumps(report['duration_summary']), flush=True)


if __name__ == '__main__':
    main()
