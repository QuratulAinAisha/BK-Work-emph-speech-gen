"""Measured-target and missing-label checks. / 측정 타깃과 누락 라벨 검사."""

import unittest
import numpy as np

from dataset.audio_affect_targets import audio_affect_targets


class AudioAffectTest(unittest.TestCase):
    def test_tone_pitch_energy_rate_and_unknown_channels(self):
        rate = 16000
        signal = 0.1 * np.sin(2 * np.pi * 220 * np.arange(rate) / rate)
        values, weights = audio_affect_targets(signal, rate, "one two three")
        self.assertEqual(values.shape, (25, 6))
        self.assertEqual(weights[:, [0, 1, 5]].sum(), 0)
        pitch = 60 * (500 / 60) ** values[weights[:, 2] > 0, 2]
        self.assertAlmostEqual(float(np.median(pitch)), 220, delta=5)
        self.assertTrue(np.allclose(values[:, 4], 0.5))
        self.assertTrue(np.isfinite(values).all())

    def test_silence_has_no_pitch_supervision(self):
        values, weights = audio_affect_targets(np.zeros(16000), 16000, "")
        self.assertEqual(weights[:, 2].sum(), 0)
        self.assertEqual(values[:, 3].sum(), 0)
        self.assertEqual(values[:, 4].sum(), 0)


if __name__ == "__main__":
    unittest.main()
