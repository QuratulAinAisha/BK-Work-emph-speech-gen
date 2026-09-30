"""Full-sampler equality and gradient isolation. / 전체 샘플러 일치와 기울기 분리."""
import unittest
import torch
from torch import nn
from types import SimpleNamespace
from dataset.quality_speech_dataset import collate_quality, person_a_only
from model.full_speech.recovery import RecoverySpeechSystem, FrozenWaveformCTC
from tests.test_quality_speech import config, sample, FakeCodec
from tests.test_recovery import FakeTeacher


class SampledAudioTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.model = RecoverySpeechSystem(config())
        self.codec = FakeCodec()
        object.__setattr__(self.model, '_acoustic_codec', self.codec)
        self.teacher = FakeTeacher()
        self.model.configure_recovery('acoustic', teacher=self.teacher)
        self.model.configure_sampled_audio()
        self.batch = collate_quality([sample()])

    def test_same_waveform_as_inference(self):
        self.model.eval()
        semantic, _ = self.model.target(self.batch, 'semantic', 768)
        semantic = (semantic - self.model.semantic_mean) / self.model.semantic_std
        expected = self.model.generate_batch(person_a_only(self.batch), self.codec, seed=42,
            oracle_semantic=semantic, oracle_duration=self.batch['duration'])['waveform'][0]
        actual = self.model.sampled_waveform(self.batch, 42)
        # Gradient-enabled attention can round differently from inference kernels. / 미분·추론 커널의 반올림 차이를 허용합니다.
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_full_sampled_loss_gradients_only_reach_generator(self):
        self.model.train()
        values = self.model.losses(self.batch)
        (values['sampled_ctc'] + values['sampled_spectral']).backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in self.model.codec_generator.parameters() if p.grad is not None), 0)
        self.assertTrue(all(p.grad is None for name,p in self.model.named_parameters() if not name.startswith('codec_generator.')))
        self.assertTrue(all(p.grad is None for p in self.teacher.parameters()))
        self.assertFalse(self.model.codec_generator.training)

    def test_validation_includes_actual_audio_objectives(self):
        self.model.eval()
        with torch.no_grad():
            values = self.model.losses(self.batch)
        self.assertGreater(float(values['sampled_ctc']), 0)
        self.assertGreater(float(values['sampled_spectral']), 0)

    def test_teacher_resampling_stays_float32_under_autocast(self):
        class Recognizer(nn.Module):
            def forward(self, audio):
                if audio.dtype != torch.float32:
                    raise AssertionError('Teacher input must stay float32')
                return SimpleNamespace(logits=audio[..., None])
        teacher = FrozenWaveformCTC.__new__(FrozenWaveformCTC)
        nn.Module.__init__(teacher)
        teacher.recognizer = Recognizer()
        wave = torch.randn(10240, requires_grad=True)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            value = teacher.waveform_logits(wave)
        value.square().mean().backward()
        self.assertTrue(torch.isfinite(wave.grad).all())


if __name__ == '__main__':
    unittest.main()
