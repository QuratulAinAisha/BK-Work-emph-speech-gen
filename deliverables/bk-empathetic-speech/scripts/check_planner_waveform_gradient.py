"""Real-model A-only waveform gradient preflight. / 실제 모델의 A 전용 파형 기울기 사전 검사."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from dataset.quality_speech_dataset import QualitySpeechDataset, collate_quality
from model.full_speech.codec import FrozenEncodec
from model.full_speech.loading import load_response_model
from model.full_speech.planner_waveform import frozen_content_ctc, planner_a_item, sampled_planner_waveform
from model.full_speech.recovery import ASR_MODEL, ASR_REVISION, FrozenWaveformCTC
from model.full_speech.units import UnitSpeechSystem
from prepare_quality import atomic_json
from train_full import move_batch


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_errors(actual, reference, rms_floor=1e-8):
    """Report numeric drift without hiding near-zero denominators. / 작은 분모를 명시하며 수치 차이를 기록합니다."""
    same_shape = actual.shape == reference.shape
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    result = {'same_shape': same_shape, 'finite': finite, 'rms_denominator_floor': rms_floor,
              'maximum_absolute_error': None, 'rms_error': None, 'reference_rms': None, 'relative_rms_error': None}
    if not same_shape or not finite or not actual.numel():
        return result
    actual, reference = actual.detach().double(), reference.detach().double()
    difference = actual - reference
    rms = float(difference.square().mean().sqrt())
    reference_rms = float(reference.square().mean().sqrt())
    return {**result, 'maximum_absolute_error': float(difference.abs().max()),
            'rms_error': rms, 'reference_rms': reference_rms,
            'relative_rms_error': rms / max(reference_rms, rms_floor)}


@torch.no_grad()
def forward_parity(model, normal, sampled):
    """Require exact units/timing before allowing decoder-kernel drift. / 디코더 차이 허용 전에 단위·길이를 일치시킵니다."""
    expected_wave = normal['waveform'][0, :int(normal['audio_lengths'][0])]
    semantic = tensor_errors(sampled['_semantic'], normal['semantic'])
    latents = tensor_errors(sampled['_codec_latents'], normal['codec_latents'])
    waveform = tensor_errors(sampled['waveform'], expected_wave)
    tokens = {'exact_agreement': False, 'different_positions': None,
              'normal_unit_ids': None, 'sampled_unit_ids': None}
    if semantic['same_shape'] and semantic['finite']:
        book = model.semantic_planner.codebook
        normal_ids, sampled_ids = book.encode(normal['semantic']), book.encode(sampled['_semantic'])
        tokens = {'exact_agreement': bool(torch.equal(normal_ids, sampled_ids)),
                  'different_positions': int((normal_ids != sampled_ids).sum()),
                  'normal_unit_ids': normal_ids.cpu().tolist(), 'sampled_unit_ids': sampled_ids.cpu().tolist()}
    timing = {'exact_duration': bool(torch.equal(normal['duration'], sampled['duration'])),
              'normal_duration_seconds': normal['duration'].detach().cpu().tolist(),
              'sampled_duration_seconds': sampled['duration'].detach().cpu().tolist(),
              'normal_audio_samples': int(normal['audio_lengths'][0]),
              'sampled_audio_samples': sampled['audio_samples'],
              'exact_audio_length': sampled['audio_samples'] == int(normal['audio_lengths'][0]) == sampled['waveform'].numel(),
              'exact_semantic_length': sampled['semantic_frames'] == normal['semantic'].shape[1] == sampled['_semantic'].shape[1],
              'exact_codec_length': sampled['codec_frames'] == normal['codec_latents'].shape[1] == sampled['_codec_latents'].shape[1]}
    def within(values, maximum, relative):
        return (values['same_shape'] and values['finite'] and values['maximum_absolute_error'] is not None
                and values['maximum_absolute_error'] <= maximum and values['relative_rms_error'] <= relative)
    # Latents remain under tight bounds; only decoder output uses prior kernel tolerance. / 잠재값은 엄격히 검사하고 출력만 기존 커널 허용치를 씁니다.
    gates = {'exact_semantic_ids': tokens['exact_agreement'],
             'exact_timing': all(timing[name] for name in ('exact_duration', 'exact_audio_length',
                                                         'exact_semantic_length', 'exact_codec_length')),
             'semantic_numeric': within(semantic, 1e-5, 1e-6),
             'codec_latent_numeric': within(latents, 1e-5, 1e-6),
             'waveform_numeric': within(waveform, 1e-3, 1e-3)}
    return {'version': 2, 'passed': all(gates.values()), 'gates': gates,
            'failure_reasons': [name for name, passed in gates.items() if not passed],
            'semantic_units': tokens, 'timing': timing, 'semantic_values': semantic,
            'codec_latents': latents, 'waveform': waveform,
            'bounds': {'semantic_and_latents': {'maximum_absolute': 1e-5, 'relative_rms': 1e-6},
                       'waveform': {'maximum_absolute': 1e-3, 'relative_rms': 1e-3}},
            'tolerance_basis': {'script': 'scripts/check_sampled_audio.py',
                'prior_report': 'outputs/sampled_audio_v1/real_gradient_check.json',
                'prior_maximum_absolute_error': .0006255358457565308,
                'prior_relative_rms_error': .0004520832735579461,
                'explanation': 'Prior real acoustic preflight established these waveform bounds for native differentiable versus cuDNN inference decoding. Units and timing must match exactly; semantic values and codec latents have separate tighter gates. Any upstream failure needs investigation, not automatic tolerance relaxation.'}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='JSON report path')
    parser.add_argument('--split', choices=['train', 'val'], default='val')
    parser.add_argument('--index', type=int, default=0, help='Index in selected manifest rows')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.index < 0:
        parser.error('Invalid index')
    if args.output.exists():
        parser.error('Use a new report path; preflight evidence is immutable')
    torch.set_num_threads(2)
    device = torch.device(args.device)
    selection = json.loads(args.selection.read_text())
    manifest_sha = sha(args.manifest)
    if selection['manifest_sha256'] != manifest_sha:
        raise ValueError('Selection manifest hash differs')
    report = {'passed': False, 'parity_protocol_version': 2,
        'checkpoint': str(args.checkpoint), 'checkpoint_sha256': sha(args.checkpoint),
        'manifest_sha256': manifest_sha, 'selection_sha256': sha(args.selection),
        'helper_sha256': sha(Path(__file__).resolve().parents[1] / 'model/full_speech/planner_waveform.py'),
        'script_sha256': sha(__file__), 'split': args.split, 'selected_index': args.index,
        'device': str(device), 'torch_version': str(torch.__version__), 'seed': args.seed,
        'content_evaluator': ASR_MODEL, 'content_evaluator_revision': ASR_REVISION,
        'gradient_estimator': 'biased final-step straight-through discrete choices, full acoustic sampler',
        'limitations': ['A gradient is not evidence of appropriate or intelligible responses.',
            'CTC uses the one recorded B transcript; other replies may also be valid.',
            'This preflight does not update model weights or establish independent evaluation quality.']}
    start = time.monotonic()
    try:
        model, _ = load_response_model(args.checkpoint)
        if not isinstance(model, UnitSpeechSystem):
            raise ValueError('Preflight requires a discrete-unit response checkpoint')
        if (model.config.semantic_steps, model.config.codec_steps) != (8, 32):
            raise ValueError('Preflight requires checkpoint sampling budgets of 8 / 32')
        model.to(device).configure_recovery('planner', teacher_weight=0.)
        model.length_predictor.requires_grad_(False)
        model.trainable_components = ['semantic_planner']
        model.train()
        codec = FrozenEncodec().to(device).eval()
        teacher = FrozenWaveformCTC().to(device).eval()
        data = QualitySpeechDataset(args.manifest, model.config, args.split)
        wanted = set(selection[args.split])
        selected = [i for i, row in enumerate(data.records) if row['path'] in wanted]
        if len(selected) != len(wanted) or args.index >= len(selected):
            raise ValueError('Selected records missing, duplicated, or index outside selection')
        item = selected[args.index]
        row = data.records[item]
        report['record'] = {key: row[key] for key in
            ('path', 'conversation_id', 'style_id', 'speaker_id', 'input_text', 'response_text')}
        batch = move_batch(collate_quality([data[item]]), device)
        model.eval()
        normal = model.generate_batch(planner_a_item(batch), codec, seed=args.seed)
        model.train()
        sampled = sampled_planner_waveform(model, batch, codec, seed=args.seed)
        report['sampling'] = {key: value for key, value in sampled.items()
                              if key not in ('waveform', 'duration') and not key.startswith('_')}
        report['sampling']['predicted_duration'] = float(sampled['duration'][0])
        waveform = sampled['waveform']
        report['forward_parity'] = forward_parity(model, normal, sampled)
        del normal
        if not report['forward_parity']['passed']:
            raise RuntimeError('A-only inference parity failed: ' + ', '.join(report['forward_parity']['failure_reasons']))
        labels = batch['text_b'][0, :int(batch['text_b_len'][0])]
        scored = frozen_content_ctc(teacher, waveform, labels)
        report['content'] = {key: value for key, value in scored.items() if key != 'loss'}
        report['content']['loss'] = float(scored['loss'].detach())
        scored['loss'].backward()
        gradient_names, forbidden = [], []
        absolute_sum, squared_sum = 0., 0.
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError('Non-finite gradient: ' + name)
            if not name.startswith('semantic_planner.'):
                forbidden.append(name)
            gradient_names.append(name)
            absolute_sum += float(parameter.grad.double().abs().sum())
            squared_sum += float(parameter.grad.double().square().sum())
        frozen_gradients = any(parameter.grad is not None for module in (codec, teacher)
                               for parameter in module.parameters())
        report['gradients'] = {'parameter_names': gradient_names, 'absolute_sum': absolute_sum,
            'l2_norm': squared_sum ** .5, 'nonplanner_parameters': forbidden,
            'codec_or_evaluator_has_gradients': frozen_gradients}
        if absolute_sum <= 0 or forbidden or frozen_gradients:
            raise RuntimeError('Expected nonzero finite planner-only gradients')
        # B labels are changed only after the first audio was generated. / 첫 생성 후에만 B 정답을 변경합니다.
        changed = dict(batch)
        for key in ('semantic', 'codec', 'affect', 'duration', 'waveform', 'text_b', 'text_a'):
            changed[key] = torch.full_like(batch[key], 999)
        invariant = sampled_planner_waveform(model, changed, codec, seed=args.seed)
        report['b_metadata_invariance'] = torch.equal(waveform.detach(), invariant['waveform'])
        if not report['b_metadata_invariance']:
            raise RuntimeError('Changing B metadata changed generated waveform')
        report['passed'] = True
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - start
        if device.type == 'cuda' and torch.cuda.is_initialized():
            report['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output, report)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
