"""Architecture, conditioning and resume tests. / 구조·조건·재개 검증."""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from dataset.full_speech_dataset import FullSpeechDataset, collate_full_speech, fit_target_statistics
from infer_full import read_person_a
from model.affective_response_transport import AFFECT_FEATURES
from model.full_speech import SpeechConfig, EmpatheticSpeechSystem
from model.full_speech.tensor_ops import counts, mask_from_lengths
from train_full import main as train_main


def small_config():
    return SpeechConfig(sbe={"app_num_layers": 1, "app_mlp_dim": 128, "app_max_len": 64},
                        hidden_dim=32, num_heads=4, planner_layers=1, decoder_layers=1,
                        affect_hidden_dim=32, affect_layers=1, dropout=0.0,
                        semantic_steps=2, codec_steps=2, max_duration=2.0)


def sample(index=0):
    generator = torch.Generator().manual_seed(50 + index)
    duration = 0.32 if index % 2 == 0 else 0.48
    return {
        "mel": torch.randn(40 - index, 80, generator=generator),
        "dmm": torch.randn(12 - index, 486, generator=generator),
        "au": torch.randn(12 - index, 25, generator=generator),
        "affect": torch.rand(8, 6, generator=generator),
        "semantic": torch.randn(round(duration * 12.5), 256, generator=generator),
        "codec": torch.randn(round(duration * 50), 128, generator=generator),
        "duration": torch.tensor(duration), "style_id": torch.tensor(index % 6),
        "speaker_id": torch.tensor(index % 2),
    }


def make_cache(root, config):
    rows = []
    for index, split in enumerate(("train", "val", "test")):
        path = f"sample_{index}.npz"
        np.savez(root / path, **{name: value.numpy() for name, value in sample(index).items()})
        rows.append({"conversation_id": str(index), "split": split, "path": path})
    metadata = {"schema_version": 1, "synthetic": True, "target_contract": config.target_contract(),
                "feature_names": list(AFFECT_FEATURES), "records": rows}
    path = root / "manifest.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    return path


class FullArchitectureTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.model = EmpatheticSpeechSystem(small_config()).eval()
        self.batch = collate_full_speech([sample(0), sample(1)])

    def person_a(self):
        return {name: self.batch[name] for name in
                ("mel", "dmm", "au", "mel_len", "dmm_len", "au_len", "style_id", "speaker_id")}

    def test_generation_lengths_masks_finite_and_reproducibility(self):
        result = self.model.generate(**self.person_a(), seed=5)
        repeated = self.model.generate(**self.person_a(), seed=5)
        for name in result:
            self.assertTrue(torch.isfinite(result[name]).all(), name)
            torch.testing.assert_close(result[name], repeated[name], atol=0, rtol=0)
        self.assertEqual(result["affect"].shape[-1], 6)
        self.assertEqual(result["semantic"].shape[-1], 256)
        self.assertEqual(result["codec_latents"].shape[-1], 128)
        torch.testing.assert_close(result["codec_lengths"], counts(result["duration"], 50))
        torch.testing.assert_close(result["semantic_lengths"], counts(result["duration"], 12.5))
        for name, mask_name in (("affect", "affect_mask"), ("semantic", "semantic_mask"),
                                 ("codec_latents", "codec_mask")):
            self.assertEqual(result[name][~result[mask_name]].count_nonzero().item(), 0)

    def test_all_required_trainable_stages_receive_finite_gradients(self):
        self.model.train()
        losses = self.model.losses(self.batch)
        self.assertEqual(set(losses), {"total", "affect", "duration", "semantic", "codec_flow"})
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        losses["total"].backward()
        stages = (
            self.model.encoder.sbe.mel_encoder, self.model.encoder.sbe.fusion_mlp,
            self.model.encoder.transport, self.model.length_predictor, self.model.semantic_planner,
            self.model.codec_generator, self.model.style_embedding, self.model.speaker_embedding,
        )
        for stage in stages:
            gradients = [p.grad for p in stage.parameters() if p.grad is not None]
            self.assertTrue(gradients, type(stage).__name__)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)

    def test_visual_freezing_survives_train_mode(self):
        self.model.set_visual_frozen(True)
        self.model.train()
        self.assertFalse(self.model.encoder.sbe.app_encoder.training)
        self.assertFalse(self.model.encoder.sbe.emo_encoder.training)
        self.assertFalse(any(p.requires_grad for p in self.model.encoder.sbe.app_encoder.parameters()))
        self.assertTrue(self.model.encoder.sbe.mel_encoder.training)
        self.assertTrue(self.model.encoder.transport.training)

    def test_diagram_58d_and_100hz_inputs_are_supported_explicitly(self):
        config = small_config()
        config.sbe.update(dmm_dim=58, mel_frame_hz=100.0)
        model = EmpatheticSpeechSystem(config).eval()
        inputs = self.person_a()
        inputs["dmm"] = inputs["dmm"][..., :58]
        result = model.generate(**inputs)
        self.assertEqual(result["context"].shape, (2, 10, 512))
        self.assertEqual(result["affect"].shape, (2, 10, 6))

    def test_losses_ignore_padded_targets_and_padded_inputs(self):
        changed = {name: value.clone() for name, value in self.batch.items()}
        for name in ("mel", "dmm", "au", "affect", "semantic", "codec"):
            for row, length in enumerate(changed[name + "_len"]):
                changed[name][row, length:] = float("nan")
        torch.manual_seed(99)
        with torch.no_grad():
            expected = self.model.losses(self.batch)
            torch.manual_seed(99)
            actual = self.model.losses(changed)
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name])

    def test_semantic_planner_uses_context_affect_and_style(self):
        mask = torch.ones(1, 5, dtype=torch.bool)
        context_mask = torch.ones(1, 7, dtype=torch.bool)
        values = dict(x=torch.randn(1, 5, 256), time=torch.tensor([0.5]),
                      local_condition=torch.randn(1, 5, 6), context=torch.randn(1, 7, 512),
                      global_condition=torch.randn(1, 32), mask=mask, context_mask=context_mask)
        network = self.model.semantic_planner.denoiser
        with torch.no_grad():
            baseline = network(**values)
            for name in ("local_condition", "context", "global_condition"):
                changed = dict(values)
                changed[name] = values[name] + torch.randn_like(values[name])
                self.assertFalse(torch.allclose(network(**changed), baseline), name)

    def test_missing_affect_labels_are_not_supervised(self):
        self.batch["affect_weight"][..., [0, 1, 5]] = 0
        self.batch["affect_weight"][:, 2:5, 2] = 0
        changed = {name: value.clone() for name, value in self.batch.items()}
        changed["affect"][..., [0, 1, 5]] = 0.9
        changed["affect"][:, 2:5, 2] = 0.9
        torch.manual_seed(99)
        expected = self.model.losses(self.batch)
        torch.manual_seed(99)
        actual = self.model.losses(changed)
        for name in expected:
            torch.testing.assert_close(expected[name], actual[name], atol=0, rtol=0)
        actual["total"].backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in self.model.encoder.transport.parameters()
                               if p.grad is not None), 0)

    def test_codec_flow_uses_semantic_affect_context_and_speaker(self):
        mask = torch.ones(1, 5, dtype=torch.bool)
        values = dict(x=torch.randn(1, 5, 128), time=torch.tensor([0.3]),
                      local_condition=torch.randn(1, 5, 262), context=torch.randn(1, 7, 512),
                      global_condition=torch.randn(1, 32), mask=mask,
                      context_mask=torch.ones(1, 7, dtype=torch.bool))
        network = self.model.codec_generator.velocity
        with torch.no_grad():
            baseline = network(**values)
            for start, end in ((0, 256), (256, 262)):
                changed = dict(values)
                changed["local_condition"] = values["local_condition"].clone()
                changed["local_condition"][..., start:end] += 2
                self.assertFalse(torch.allclose(network(**changed), baseline))
            for name in ("context", "global_condition"):
                changed = dict(values)
                changed[name] = values[name] + 2
                self.assertFalse(torch.allclose(network(**changed), baseline))

    def test_invalid_ids_durations_and_sampling_steps_are_rejected(self):
        inputs = self.person_a()
        inputs["speaker_id"] = torch.tensor([32, 0])
        with self.assertRaisesRegex(ValueError, "speaker_id"):
            self.model.generate(**inputs)
        with self.assertRaisesRegex(ValueError, "steps"):
            self.model.generate(**self.person_a(), codec_steps=0)
        changed = dict(self.batch)
        changed["duration"] = torch.tensor([0.33, 0.48])
        with self.assertRaisesRegex(ValueError, "lengths"):
            self.model.losses(changed)
        self.model.train()
        with self.assertRaisesRegex(RuntimeError, "eval"):
            self.model.generate(**self.person_a())

    def test_checkpoint_restores_outputs_statistics_and_freezing(self):
        self.model.semantic_mean.fill_(0.1)
        self.model.set_visual_frozen(True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save(self.model.checkpoint(0, synthetic=True), path)
            restored, payload = EmpatheticSpeechSystem.from_checkpoint(path)
            expected, actual = self.model.generate(**self.person_a()), restored.generate(**self.person_a())
            self.assertTrue(restored.freeze_visual)
            self.assertEqual(payload["training_steps"], 0)
            torch.testing.assert_close(expected["codec_latents"], actual["codec_latents"], atol=0, rtol=0)

    def test_npz_inference_never_reads_response_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.npz"
            values = {name: value.numpy() for name, value in sample().items()}
            np.savez(path, **values)
            inputs = read_person_a(path, "cpu", 0, 0)
            values.update(semantic=np.full((1, 256), np.nan), codec=np.full((1, 128), np.nan),
                          affect=np.full((1, 6), np.nan), duration=np.float32(999))
            np.savez(path, **values)
            changed = read_person_a(path, "cpu", 0, 0)
            for name in inputs:
                torch.testing.assert_close(inputs[name], changed[name], atol=0, rtol=0)
            self.assertNotIn("duration", inputs)


class FullDataAndTrainingTest(unittest.TestCase):
    def test_frame_counts_are_stable_at_float32_boundaries(self):
        frames = torch.arange(10, 6000)
        durations = (frames.double() / 50).float()
        torch.testing.assert_close(counts(durations, 50), frames)
        above = ((frames.double() * 640 + 1) / 32000).float()
        torch.testing.assert_close(counts(above, 50), frames + 1)

    def test_cache_validation_and_train_only_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = small_config()
            manifest = make_cache(root, config)
            with self.assertRaisesRegex(ValueError, "Synthetic"):
                FullSpeechDataset(manifest, config, "train")
            data = FullSpeechDataset(manifest, config, "train", True)
            stats = fit_target_statistics(data)
            torch.testing.assert_close(stats["codec_mean"], data[0]["codec"].mean(0))
            metadata = json.loads(manifest.read_text())
            metadata["records"][1]["conversation_id"] = "0"
            manifest.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "leakage"):
                FullSpeechDataset(manifest, config, "train", True)

    def test_resume_matches_uninterrupted_cpu_training(self):
        # Temporary optimizer steps test resume only. / 임시 최적화 단계로 재개만 검증합니다.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = small_config()
            manifest = make_cache(root, config)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config.to_dict()))
            common = ["--manifest", str(manifest), "--device", "cpu", "--allow-synthetic", "--batch-size", "1"]
            init = ["--config", str(config_path), "--train-sbe-from-scratch"]
            with redirect_stdout(io.StringIO()):
                train_main(common + init + ["--epochs", "1", "--output", str(root / "resumed")])
                train_main(common + ["--resume", str(root / "resumed" / "last.pt"), "--epochs", "2",
                                     "--output", str(root / "resumed")])
                train_main(common + init + ["--epochs", "2", "--output", str(root / "continuous")])
            resumed, a = EmpatheticSpeechSystem.from_checkpoint(root / "resumed" / "last.pt")
            continuous, b = EmpatheticSpeechSystem.from_checkpoint(root / "continuous" / "last.pt")
            self.assertEqual(a["training_steps"], 2)
            self.assertEqual(b["training_steps"], 2)
            for name, value in resumed.state_dict().items():
                torch.testing.assert_close(value, continuous.state_dict()[name], atol=0, rtol=0)


@unittest.skipUnless(os.environ.get("BK_TEST_PRETRAINED") == "1", "Set BK_TEST_PRETRAINED=1 to load real pretrained weights")
class PretrainedIntegrationTest(unittest.TestCase):
    def test_real_codec_quantized_latents_reconstruct_like_official_decoder(self):
        from model.full_speech.codec import FrozenEncodec
        codec = FrozenEncodec()
        codec.train()
        self.assertFalse(codec.training)
        self.assertTrue(all(not p.requires_grad for p in codec.parameters()))
        waveform = torch.sin(torch.arange(10240) * (2 * torch.pi * 220 / 32000))[None] * 0.1
        latents, lengths = codec.encode(waveform, torch.tensor([10240]))
        decoded, audio_lengths = codec.decode(latents, lengths, torch.tensor([10240]))
        with torch.inference_mode():
            encoded = codec.model.encode(waveform[:, None])
            official = codec.model.decode(encoded.audio_codes, encoded.audio_scales).audio_values
        torch.testing.assert_close(decoded[0], official[0, 0, :10240])
        self.assertEqual(latents.shape, (1, 16, 128))
        self.assertEqual(audio_lengths.tolist(), [10240])
        self.assertTrue(torch.isfinite(decoded).all())

    def test_real_semantic_teacher_is_frozen_and_deterministic(self):
        from model.full_speech.targets import FrozenSemanticTeacher
        teacher = FrozenSemanticTeacher(small_config())
        teacher.train()
        self.assertFalse(teacher.training)
        self.assertTrue(all(not p.requires_grad for p in teacher.parameters()))
        signal = torch.sin(torch.arange(5120) * (2 * torch.pi * 220 / 16000)) * 0.1
        first, second = teacher.encode(signal, 0.32), teacher.encode(signal, 0.32)
        self.assertEqual(first.shape, (4, 256))
        self.assertTrue(torch.isfinite(first).all())
        torch.testing.assert_close(first, second, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
