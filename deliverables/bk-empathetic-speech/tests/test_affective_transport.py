"""Check module-3 contracts without training. / 학습 없이 모듈 3 계약을 검증합니다."""

from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from model.affective_response_transport import (
    AFFECT_FEATURES, AFFECT_RANGES, AffectiveResponseTransport, AffectiveTransportConfig,
)


class AffectiveTransportTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = AffectiveResponseTransport(AffectiveTransportConfig(
            hidden_dim=32, num_layers=2, feedforward_dim=64,
        )).eval()
        self.context = torch.randn(2, 9, 512)
        self.emotion = torch.randn(2, 13, 25)
        self.mask = torch.arange(9)[None, :] < torch.tensor([9, 5])[:, None]
        self.lengths = torch.tensor([13, 7])

    def forward(self, context=None, emotion=None, mask=None, lengths=None):
        with torch.inference_mode():
            return self.model(
                self.context if context is None else context,
                self.emotion if emotion is None else emotion,
                self.mask if mask is None else mask,
                self.lengths if lengths is None else lengths,
            )

    def test_shape_range_and_masked_pool(self):
        output = self.forward()
        self.assertEqual(output.trajectory.shape, (2, 9, 6))
        self.assertEqual(output.lengths.tolist(), [9, 5])
        self.assertTrue(torch.equal(output.mask, self.mask))
        self.assertTrue(torch.isfinite(output.trajectory).all())
        self.assertEqual(output.trajectory[1, 5:].count_nonzero().item(), 0)
        for index, (low, high) in enumerate(AFFECT_RANGES):
            values = output.trajectory[..., index][output.mask]
            self.assertTrue(((values >= low) & (values <= high)).all())
        torch.testing.assert_close(output.mean_pool()[1], output.trajectory[1, :5].mean(0))

    def test_nan_and_large_padding_do_not_change_valid_outputs(self):
        expected = self.forward()
        context, emotion = self.context.clone(), self.emotion.clone()
        context[1, 5:] = float("nan")
        emotion[1, 7:] = 1e20
        actual = self.forward(context=context, emotion=emotion)
        torch.testing.assert_close(actual.trajectory, expected.trajectory)
        emotion[1, 7:] = float("nan")
        torch.testing.assert_close(self.forward(emotion=emotion).trajectory, expected.trajectory)

    def test_short_sample_matches_unpadded_inference(self):
        expected = self.forward().trajectory[1, :5]
        actual = self.forward(self.context[1:2, :5], self.emotion[1:2, :7],
                              torch.ones(1, 5, dtype=torch.bool), torch.tensor([7]))
        torch.testing.assert_close(actual.trajectory[0], expected, atol=2e-6, rtol=1e-5)

    def test_batch_members_do_not_influence_each_other(self):
        context, emotion = self.context.clone(), self.emotion.clone()
        context[0] = torch.randn_like(context[0]) * 5
        emotion[0] = torch.randn_like(emotion[0]) * 5
        torch.testing.assert_close(self.forward(context, emotion).trajectory[1], self.forward().trajectory[1])

    def test_context_and_emotion_both_affect_output(self):
        baseline = self.forward().trajectory
        context = self.context.clone()
        context[:, :, 0] += 10
        emotion = self.emotion.clone()
        emotion[:, :, 0] += 10
        self.assertFalse(torch.allclose(self.forward(context=context).trajectory, baseline))
        self.assertFalse(torch.allclose(self.forward(emotion=emotion).trajectory, baseline))

    def test_temporal_attention_connects_different_frames(self):
        context = self.context.clone()
        context[0, 0, :10] += 20
        delta = (self.forward(context=context).trajectory[0, -1] - self.forward().trajectory[0, -1]).abs()
        self.assertGreater(delta.max().item(), 1e-5)

    def test_positions_make_temporal_order_matter(self):
        context, emotion = self.context[:1], self.emotion[:1, :9]
        with torch.inference_mode():
            original = self.model(context, emotion).trajectory
            reversed_result = self.model(context.flip(1), emotion.flip(1)).trajectory.flip(1)
        self.assertFalse(torch.allclose(original, reversed_result))

    def test_single_frame_and_default_masks(self):
        with torch.inference_mode():
            result = self.model(self.context[:1, :1], self.emotion[:1, :1])
        self.assertEqual(result.trajectory.shape, (1, 1, 6))
        self.assertEqual(result.mask.tolist(), [[True]])

    def test_eval_is_repeatable_and_does_not_update_parameters(self):
        before = {name: value.clone() for name, value in self.model.state_dict().items()}
        torch.testing.assert_close(self.forward().trajectory, self.forward().trajectory, atol=0, rtol=0)
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))

    def test_invalid_masks_lengths_shapes_and_values_are_rejected(self):
        invalid_calls = [
            {"mask": torch.zeros(2, 9, dtype=torch.bool)},
            {"mask": self.mask.float()},
            {"mask": self.mask[:, :8]},
            {"lengths": torch.tensor([13, 0])},
            {"lengths": torch.tensor([14, 7])},
            {"lengths": torch.tensor([13.0, 7.0])},
            {"emotion": self.emotion[:, :, :24]},
            {"context": self.context[:, :, :511]},
            {"context": self.context[:, :0]},
            {"context": self.context[:0]},
            {"emotion": self.emotion[:1], "lengths": torch.tensor([13])},
            {"context": self.context.double()},
            {"context": self.context.long()},
        ]
        hole_mask = self.mask.clone()
        hole_mask[0, 1] = False
        invalid_calls.append({"mask": hole_mask})
        for arguments in invalid_calls:
            with self.subTest(arguments=list(arguments)):
                with self.assertRaises(ValueError):
                    self.forward(**arguments)
        for name in ("context", "emotion"):
            value = getattr(self, name).clone()
            value[0, 0, 0] = float("nan")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                self.forward(**{name: value})

    def test_config_validation_and_custom_input_dimensions(self):
        for changes in ({"hidden_dim": 31}, {"num_heads": 0}, {"num_layers": 0},
                        {"dropout": 1.0}, {"context_dim": 2.5}):
            with self.assertRaises(ValueError):
                replace(self.model.config, **changes)
        model = AffectiveResponseTransport(replace(self.model.config, context_dim=16, emotion_dim=2)).eval()
        with torch.inference_mode():
            result = model(torch.randn(1, 3, 16), torch.randn(1, 5, 2))
        self.assertEqual(result.trajectory.shape, (1, 3, 6))

    def test_checkpoint_roundtrip_and_schema_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "affect.pt"
            self.model.save_checkpoint(path)
            loaded = AffectiveResponseTransport.from_checkpoint(path)
            with torch.inference_mode():
                actual = loaded(self.context, self.emotion, self.mask, self.lengths)
            torch.testing.assert_close(actual.trajectory, self.forward().trajectory, atol=0, rtol=0)
            payload = torch.load(path, weights_only=True)
            payload["feature_names"] = list(reversed(AFFECT_FEATURES))
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "feature order"):
                AffectiveResponseTransport.from_checkpoint(path)

    def test_package_does_not_load_legacy_diffusion_dependencies(self):
        script = (
            "import sys; from model import AffectiveResponseTransport, AudioEmbedder; "
            "assert 'model.diffusion.matchers' not in sys.modules; "
            "assert 'timm' not in sys.modules; assert AudioEmbedder is not None"
        )
        subprocess.run([sys.executable, "-c", script], check=True, capture_output=True,
                       cwd=Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    unittest.main()
