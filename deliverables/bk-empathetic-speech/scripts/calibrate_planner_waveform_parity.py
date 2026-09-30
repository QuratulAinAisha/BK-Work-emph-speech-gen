"""Attribute acoustic drift to grad/inference attention paths. / 어텐션 경로별 음향 수치 차이 분리."""

import argparse
from contextlib import contextmanager, nullcontext
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
from model.full_speech.recovery import FrozenWaveformCTC
from model.full_speech.units import UnitSpeechSystem
from prepare_quality import atomic_json
from scripts.check_planner_learning_history import tensor_hash
from scripts.check_planner_waveform_gradient import forward_parity, sha, tensor_errors
from train_full import move_batch


@contextmanager
def attention_path(math_only=False):
    original = torch.backends.mha.get_fastpath_enabled()
    if math_only:
        torch.backends.mha.set_fastpath_enabled(False)
    try:
        # This override is diagnostic only and is always restored. / 진단용 설정만 잠시 바꾸고 복구합니다.
        manager = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True,
            enable_mem_efficient=False, enable_cudnn=False) if math_only else nullcontext()
        with manager:
            yield
    finally:
        torch.backends.mha.set_fastpath_enabled(original)


def acoustic_replay(model, fixed, seed, grad_enabled, math_only=False, profile=True):
    generator_model = model.codec_generator
    if any(module.training for module in generator_model.modules()) or any(
            parameter.requires_grad for parameter in generator_model.parameters()):
        raise ValueError('Acoustic replay requires a frozen eval generator')
    semantic = fixed['semantic'].detach().clone().requires_grad_(grad_enabled)
    device = semantic.device
    semantic_mask = torch.ones(semantic.shape[:2], device=device, dtype=torch.bool)
    codec_mask = torch.ones(1, fixed['codec_frames'], device=device, dtype=torch.bool)
    first_noise = {}
    def capture_noise(module, arguments):
        if not first_noise:
            first_noise['value'] = arguments[0].detach().clone()
    hook = generator_model.velocity.register_forward_pre_hook(capture_noise)
    profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) if profile else nullcontext()
    try:
        with attention_path(math_only), torch.set_grad_enabled(grad_enabled), torch.autocast(device_type=device.type, enabled=False):
            with profiler:
                latent = generator_model.sample(codec_mask, semantic, semantic_mask, fixed['context'],
                    fixed['context_mask'], fixed['affect'], fixed['style_speaker'],
                    model.config.codec_steps, torch.Generator(device=device).manual_seed(seed), checkpoint_grad=True)
                latent = (latent * model.codec_std + model.codec_mean).detach().clone()
    finally:
        hook.remove()
    operators = ({event.key: event.count for event in profiler.key_averages()
                  if 'attention' in event.key.lower()} if profile else {})
    return {'latents': latent, 'noise': first_noise['value'], 'attention_operators': operators,
            'grad_enabled': grad_enabled, 'math_only': math_only,
            'semantic_input_sha256': tensor_hash(semantic)}


def calibrate_acoustic_paths(model, normal, sampled, style_speaker, seed=42, profile=True):
    # Every run receives exactly the same semantic centers and A conditions. / 모든 실행에 같은 단위 중심과 A 조건을 줍니다.
    fixed = {name: normal[name].detach().clone() for name in ('semantic', 'context', 'context_mask', 'affect')}
    fixed.update(codec_frames=normal['codec_latents'].shape[1], style_speaker=style_speaker.detach().clone())
    configurations = [('inference', False, False), ('inference_repeat', False, False),
                      ('gradient', True, False), ('gradient_repeat', True, False),
                      ('math_inference', False, True), ('math_gradient', True, True)]
    runs = {name: acoustic_replay(model, fixed, seed, grad, math_only, profile)
            for name, grad, math_only in configurations}
    ordinary = tensor_errors(runs['gradient']['latents'], runs['inference']['latents'])
    math = tensor_errors(runs['math_gradient']['latents'], runs['math_inference']['latents'])
    reference = tensor_errors(runs['inference']['latents'], normal['codec_latents'])
    helper = tensor_errors(runs['gradient']['latents'], sampled['_codec_latents'])
    standardized = tensor_errors((runs['gradient']['latents'] - model.codec_mean) / model.codec_std,
                                 (runs['inference']['latents'] - model.codec_mean) / model.codec_std)
    checks = {'same_initial_noise': all(torch.equal(run['noise'], runs['inference']['noise']) for run in runs.values()),
              'same_semantic_inputs': len({run['semantic_input_sha256'] for run in runs.values()}) == 1,
              'inference_repeat_exact': torch.equal(runs['inference']['latents'], runs['inference_repeat']['latents']),
              'gradient_repeat_exact': torch.equal(runs['gradient']['latents'], runs['gradient_repeat']['latents']),
              'normal_matches_inference_replay_exact': torch.equal(runs['inference']['latents'], normal['codec_latents']),
              'helper_matches_gradient_replay_exact': torch.equal(runs['gradient']['latents'], sampled['_codec_latents']),
              'ordinary_relative_rms_small': ordinary['relative_rms_error'] is not None and ordinary['relative_rms_error'] <= 1e-6,
              'standardized_maximum_small': standardized['maximum_absolute_error'] is not None and standardized['maximum_absolute_error'] <= 1e-5,
              'math_paths_agree_tightly': math['maximum_absolute_error'] is not None and
                  math['maximum_absolute_error'] <= 1e-6 and math['relative_rms_error'] <= 1e-7}
    return {'passed': all(checks.values()), 'checks': checks,
            'failure_reasons': [name for name, passed in checks.items() if not passed],
            'normal_vs_inference_replay': reference, 'helper_vs_gradient_replay': helper,
            'ordinary_gradient_vs_inference': ordinary, 'standardized_gradient_vs_inference': standardized,
            'math_gradient_vs_inference': math,
            'runs': {name: {key: value for key, value in run.items() if key not in ('latents', 'noise')} |
                           {'noise_sha256': tensor_hash(run['noise']), 'latent_sha256': tensor_hash(run['latents'])}
                     for name, run in runs.items()},
            'acceptance_basis': 'Require exact within-path repeats and exact attribution to the two production paths. '
                'Then require tiny relative/standardized drift and agreement with native MHA fastpath disabled and math SDPA. '
                'All semantic-ID, timing and waveform bounds from preflight v2 still apply. No thresholds are adapted from this result.'}


def main():
    parser = argparse.ArgumentParser()
    for name in ('checkpoint', 'manifest', 'selection', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--split', choices=('train', 'val'), default='val')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.index < 0 or args.output.exists():
        parser.error('Use a nonnegative index and a new immutable report path')
    torch.set_num_threads(2)
    selection = json.loads(args.selection.read_text())
    if selection['manifest_sha256'] != sha(args.manifest):
        raise ValueError('Selection manifest hash differs')
    report = {'passed': False, 'protocol': 'controlled_acoustic_attention_calibration_v1',
        'checkpoint': str(args.checkpoint), 'checkpoint_sha256': sha(args.checkpoint),
        'manifest_sha256': sha(args.manifest), 'selection_sha256': sha(args.selection),
        'source_sha256': {name: sha(Path(__file__).resolve().parents[1] / name) for name in (
            'scripts/calibrate_planner_waveform_parity.py', 'scripts/check_planner_waveform_gradient.py',
            'model/full_speech/planner_waveform.py', 'model/full_speech/units.py',
            'model/full_speech/planners.py', 'model/full_speech/dit.py')},
        'device': args.device, 'torch_version': str(torch.__version__), 'split': args.split,
        'index': args.index, 'seed': args.seed, 'weights_updated': False,
        'initial_mha_fastpath_enabled': torch.backends.mha.get_fastpath_enabled(),
        'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
        'limitation': 'Single-case numerical and gradient preflight, not a quality or multi-GPU execution test.'}
    start = time.monotonic()
    try:
        model, _ = load_response_model(args.checkpoint)
        if not isinstance(model, UnitSpeechSystem) or (model.config.semantic_steps, model.config.codec_steps) != (8, 32):
            raise ValueError('Requires a discrete-unit checkpoint with 8 / 32 inference steps')
        model.to(args.device).configure_recovery('planner', teacher_weight=0.)
        model.length_predictor.requires_grad_(False)
        model.trainable_components = ['semantic_planner']
        model.train()
        codec = FrozenEncodec().to(args.device).eval()
        data = QualitySpeechDataset(args.manifest, model.config, args.split)
        wanted = set(selection[args.split])
        selected = [index for index, row in enumerate(data.records) if row['path'] in wanted]
        if len(selected) != len(wanted) or args.index >= len(selected):
            raise ValueError('Invalid selected record set or index')
        index = selected[args.index]
        report['path'] = data.records[index]['path']
        batch = move_batch(collate_quality([data[index]]), torch.device(args.device))
        inputs = planner_a_item(batch)
        model.eval()
        normal = model.generate_batch(inputs, codec, seed=args.seed)
        model.train()
        sampled = sampled_planner_waveform(model, batch, codec, seed=args.seed)
        report['base_forward_parity'] = forward_parity(model, normal, sampled)
        with torch.no_grad():
            style, speaker = model.embeddings(inputs['style_id'], inputs['speaker_id'], 1)
        report['calibration'] = calibrate_acoustic_paths(model, normal, sampled, style + speaker, args.seed)
        preserved_gates = {name: passed for name, passed in report['base_forward_parity']['gates'].items()
                           if name != 'codec_latent_numeric'}
        if not all(preserved_gates.values()) or not report['calibration']['passed']:
            raise RuntimeError('Controlled parity attribution failed; inspect all recorded gates')
        teacher = FrozenWaveformCTC().to(args.device).eval()
        labels = batch['text_b'][0, :int(batch['text_b_len'][0])]
        scored = frozen_content_ctc(teacher, sampled['waveform'], labels)
        report['content'] = {key: value for key, value in scored.items() if key != 'loss'}
        report['content']['loss'] = float(scored['loss'].detach())
        scored['loss'].backward()
        gradients = [(name, parameter.grad) for name, parameter in model.named_parameters() if parameter.grad is not None]
        forbidden = [name for name, _ in gradients if not name.startswith('semantic_planner.')]
        finite = all(bool(torch.isfinite(gradient).all()) for _, gradient in gradients)
        absolute = sum(float(gradient.double().abs().sum()) for _, gradient in gradients)
        frozen_have_gradients = any(parameter.grad is not None for module in (codec, teacher) for parameter in module.parameters())
        report['gradients'] = {'finite': finite, 'absolute_sum': absolute, 'nonplanner_parameters': forbidden,
                               'codec_or_teacher_has_gradients': frozen_have_gradients}
        if not finite or absolute <= 0 or forbidden or frozen_have_gradients:
            raise RuntimeError('Expected nonzero finite planner-only gradients')
        changed = dict(batch)
        for key in ('semantic', 'codec', 'affect', 'duration', 'waveform', 'text_b', 'text_a'):
            changed[key] = torch.full_like(batch[key], 999)
        invariant = sampled_planner_waveform(model, changed, codec, seed=args.seed)
        report['B_metadata_invariance_exact'] = torch.equal(sampled['waveform'].detach(), invariant['waveform'])
        if not report['B_metadata_invariance_exact']:
            raise RuntimeError('Changing B metadata changed A-only generated audio')
        report['passed'] = True
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - start
        if torch.cuda.is_initialized():
            report['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated(torch.device(args.device))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output, report)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
