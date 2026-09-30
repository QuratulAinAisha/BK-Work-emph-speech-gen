"""Verify real sampled-audio gradients before training. / 학습 전 실제 샘플 음성 기울기 검증."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from model.full_speech.quality import QualityConfig
from model.full_speech.units import initialize_unit_system
from model.full_speech.recovery import FrozenWaveformCTC
from model.full_speech.codec import FrozenEncodec
from dataset.quality_speech_dataset import QualitySpeechDataset, collate_quality, person_a_only
from train_full import move_batch
from prepare_quality import atomic_json


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    p = Path('outputs/sampled_audio_v1');p.mkdir(exist_ok=True)
    payload = torch.load('outputs/quality_recovery/acoustic/best.pt', map_location='cpu', weights_only=True)
    cfg = QualityConfig(**payload['config'])
    cfg.semantic_steps = cfg.predicted_semantic_steps = 8
    model = initialize_unit_system(cfg, payload, Path('outputs/broad_units_v1/codebook.pt')).cuda()
    codec, teacher = FrozenEncodec().cuda(), FrozenWaveformCTC().cuda()
    object.__setattr__(model, '_acoustic_codec', codec)
    model.configure_recovery('acoustic', teacher=teacher)
    model.configure_sampled_audio()
    dataset = QualitySpeechDataset('outputs/bk_quality_prepared/manifest.json', cfg, 'train')
    selected = set(json.loads(Path('outputs/broad_units_v1/selection.json').read_text())['train'])
    index = next(i for i,r in enumerate(dataset.records) if r['path'] in selected)
    batch = move_batch(collate_quality([dataset[index]]), torch.device('cuda:0'))
    model.eval()
    semantic, _ = model.target(batch, 'semantic', 768)
    semantic = (semantic - model.semantic_mean) / model.semantic_std
    expected = model.generate_batch(person_a_only(batch), codec, seed=42,
        oracle_semantic=semantic, oracle_duration=batch['duration'])['waveform'][0]
    waveform = model.sampled_waveform(batch, 42)
    delta = float((waveform.detach() - expected).abs().max())
    relative_rms = float((waveform.detach() - expected).square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-8))
    # Native differentiable decoder and cuDNN inference differ numerically. / 미분용 기본 디코더와 cuDNN 추론의 수치 차이를 검사합니다.
    print(json.dumps({'maximum_error': delta, 'relative_rms_error': relative_rms}), flush=True)
    if delta > .001 or relative_rms > .001:
        raise RuntimeError('Sampled waveform differs materially from inference')
    loss = teacher(waveform, batch['text_b'][0, :int(batch['text_b_len'][0])])
    loss.backward()
    grads = [v.grad for v in model.codec_generator.parameters() if v.grad is not None]
    result = {'inference_max_absolute_difference': delta, 'inference_relative_rms_error': relative_rms,
        'parity_tolerances': {'max_absolute': .001, 'relative_rms': .001}, 'sampled_ctc': float(loss.detach()),
        'generator_gradient_l1': sum(float(g.abs().sum()) for g in grads),
        'generator_gradients_finite': all(bool(torch.isfinite(g).all()) for g in grads),
        'other_model_gradients_absent': all(v.grad is None for n,v in model.named_parameters() if not n.startswith('codec_generator.')),
        'teacher_gradients_absent': all(v.grad is None for v in teacher.parameters()),
        'codec_gradients_absent': all(v.grad is None for v in codec.parameters()),
        'peak_gpu_bytes': torch.cuda.max_memory_allocated(), 'codec_steps': cfg.codec_steps,
        'source': dataset.records[index]['path']}
    atomic_json(p/'real_gradient_check.json', result)
    print(json.dumps(result), flush=True)
    if not (result['generator_gradient_l1'] > 0 and all(result[k] for k in
            ['generator_gradients_finite','other_model_gradients_absent','teacher_gradients_absent','codec_gradients_absent'])):
        raise RuntimeError('Sampled waveform gradient contract failed')


if __name__ == '__main__':
    main()
