"""Verify the inference entry point and saved outputs. / 추론 진입점과 저장 결과를 검증합니다."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from infer_affect import parse_args, read_inputs, run
from model.affective_response_transport import AFFECT_FEATURES, AffectiveResponseTransport


class AffectInferenceTest(unittest.TestCase):
    def test_real_input_requires_checkpoint_or_explicit_untrained_flag(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--input", "features.npz"])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--demo", "--checkpoint", "weights.pt", "--config", "config.json"])

    def test_cached_checkpoint_inference_writes_valid_outputs(self):
        torch.manual_seed(9)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "affect.pt"
            model = AffectiveResponseTransport().eval()
            model.save_checkpoint(checkpoint)
            source = root / "input.npz"
            context, emotion = torch.randn(1, 5, 512), torch.randn(1, 7, 25)
            mask = torch.tensor([[True, True, True, False, False]])
            lengths = torch.tensor([4])
            np.savez(source, context=context.numpy(), speaker_emotion=emotion.numpy(),
                     context_mask=mask.numpy(), emotion_lengths=lengths.numpy())
            args = parse_args(["--input", str(source), "--checkpoint", str(checkpoint),
                               "--output", str(root / "output")])
            with redirect_stdout(io.StringIO()):
                metadata = run(args)
            self.assertEqual(metadata["untrained_components"], [])
            self.assertFalse(metadata["training_performed"])
            self.assertEqual(metadata["valid_lengths"], [3])
            with torch.inference_mode():
                expected = model(context, emotion, mask, lengths)
            with np.load(root / "output" / "affect.npz", allow_pickle=False) as result:
                np.testing.assert_allclose(result["affect"], expected.trajectory.numpy(), atol=1e-6)
                self.assertEqual(result["feature_names"].tolist(), list(AFFECT_FEATURES))
                self.assertEqual(np.count_nonzero(result["context"][0, 3:]), 0)
            report = json.loads((root / "output" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(report, metadata)
            self.assertEqual(len((root / "output" / "affect.csv").read_text().splitlines()), 4)

    def test_raw_features_need_sbe_checkpoint_unless_explicitly_untrained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, checkpoint = root / "features.npz", root / "affect.pt"
            np.savez(source, mel=np.ones((1, 4, 80), dtype=np.float32),
                     dmm=np.ones((1, 2, 486), dtype=np.float32),
                     au=np.ones((1, 2, 25), dtype=np.float32))
            AffectiveResponseTransport().save_checkpoint(checkpoint)
            args = parse_args(["--input", str(source), "--checkpoint", str(checkpoint)])
            with self.assertRaisesRegex(ValueError, "sbe-checkpoint"):
                run(args)

    def test_ambiguous_and_incomplete_npz_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.npz"
            for names in (("context",), ("context", "speaker_emotion", "mel", "dmm", "au")):
                np.savez(source, **{name: np.ones((1, 2, 3), dtype=np.float32) for name in names})
                with self.assertRaisesRegex(ValueError, "exactly one input route"):
                    read_inputs(source, "cpu")


if __name__ == "__main__":
    unittest.main()
