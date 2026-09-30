"""Verify actual SBE -> fusion -> affect flow. / 실제 SBE -> 융합 -> 정서 흐름을 검증합니다."""

from pathlib import Path
import tempfile
import unittest

import torch

from infer_affect import load_sbe
from model.affective_pipeline import SBEWithAffectiveTransport
from model.affective_response_transport import AffectiveResponseTransport, AffectiveTransportConfig
from model.speaker_behavior_encoder import SpeakerBehaviorEncoder


SBE_CONFIG = {"app_num_layers": 1, "app_mlp_dim": 128, "app_max_len": 64}


class AffectivePipelineTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        sbe = SpeakerBehaviorEncoder(**SBE_CONFIG)
        transport = AffectiveResponseTransport(AffectiveTransportConfig(
            hidden_dim=32, num_layers=1, feedforward_dim=64,
        ))
        self.model = SBEWithAffectiveTransport(sbe, transport).eval()
        self.inputs = {
            "mel": torch.randn(2, 40, 80), "dmm": torch.randn(2, 12, 486),
            "au": torch.randn(2, 12, 25), "mel_len": torch.tensor([40, 26]),
            "dmm_len": torch.tensor([12, 8]), "au_len": torch.tensor([12, 8]),
        }

    def test_real_sbe_outputs_are_preserved_and_fed_to_transport(self):
        with torch.inference_mode():
            context, mask = self.model.sbe(**self.inputs, return_mask=True)
            result = self.model(**self.inputs)
            expected = self.model.transport(context, self.inputs["au"], mask, self.inputs["au_len"])
        self.assertEqual(result["context"].shape, (2, 11, 512))
        self.assertEqual(result["affect"].shape, (2, 11, 6))
        self.assertEqual(result["affect_lengths"].tolist(), [11, 7])
        torch.testing.assert_close(result["context"], context)
        torch.testing.assert_close(result["context_mask"], mask)
        torch.testing.assert_close(result["affect"], expected.trajectory)
        torch.testing.assert_close(result["affect_summary"], expected.mean_pool())

    def test_padding_and_batch_size_do_not_change_short_sample(self):
        with torch.inference_mode():
            expected = self.model(**self.inputs)
            changed = {name: value.clone() for name, value in self.inputs.items()}
            for name in ("mel", "dmm", "au"):
                changed[name][1, self.inputs[name + "_len"][1]:] = float("nan")
            actual = self.model(**changed)
            single = {name: self.inputs[name][1:2, :self.inputs[name + "_len"][1]]
                      for name in ("mel", "dmm", "au")}
            one = self.model(**single)
        torch.testing.assert_close(actual["affect"], expected["affect"])
        torch.testing.assert_close(one["affect"][0], expected["affect"][1, :7], atol=1e-5, rtol=1e-5)

    def test_empty_input_is_rejected_before_sbe_clamps_lengths(self):
        self.inputs["au_len"][1] = 0
        with self.assertRaisesRegex(ValueError, "lengths"):
            self.model(**self.inputs)

    def test_loads_standalone_and_stage2_sbe_without_silent_skips(self):
        original = self.model.sbe.state_dict()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sbe.pt"
            for prefix in ("", "sbe.", "module.sbe."):
                state = {prefix + key: value for key, value in original.items()}
                if prefix:
                    state[prefix.removesuffix("sbe.") + "decoder.unused"] = torch.ones(1)
                torch.save({"state_dict": state}, path)
                loaded = load_sbe(path, SBE_CONFIG, "cpu")
                for name, value in loaded.state_dict().items():
                    torch.testing.assert_close(value, original[name], atol=0, rtol=0)
            torch.save({"state_dict": {}}, path)
            with self.assertRaises(RuntimeError):
                load_sbe(path, SBE_CONFIG, "cpu")


if __name__ == "__main__":
    unittest.main()
