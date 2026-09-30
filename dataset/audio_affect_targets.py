"""Measured prosody with explicit missing labels. / 누락 라벨을 명시한 운율 측정."""

import math
import re

import numpy as np
from scipy.fft import rfft, irfft, next_fast_len


def audio_affect_targets(waveform, sample_rate, transcript, frame_hz=25.0):
    """Pitch, energy and utterance word rate only. / 피치·에너지·발화 단어율만 측정합니다."""
    signal = np.asarray(waveform, dtype=np.float32)
    if signal.ndim != 1 or len(signal) == 0 or not np.isfinite(signal).all():
        raise ValueError("Expected a finite, nonempty mono waveform")
    duration = len(signal) / sample_rate
    count = int(math.ceil(duration * frame_hz - 1e-6))
    width = round(sample_rate * 0.06)
    centers = np.minimum((np.arange(count) * sample_rate / frame_hz).astype(int), len(signal) - 1)
    padded = np.pad(signal, (width // 2, width))
    frames = np.lib.stride_tricks.sliding_window_view(padded, width)[centers].copy()
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    frames -= frames.mean(axis=1, keepdims=True)
    frames *= np.hanning(width).astype(np.float32)
    size = next_fast_len(2 * width - 1)
    spectrum = rfft(frames, n=size, axis=1)
    correlation = irfft(spectrum * spectrum.conj(), n=size, axis=1)[:, :width]
    lo, hi = int(sample_rate / 500), int(sample_rate / 60)
    lag = correlation[:, lo:hi + 1].argmax(axis=1) + lo
    strength = correlation[np.arange(count), lag] / np.maximum(correlation[:, 0], 1e-12)
    voiced = (strength >= 0.5) & (rms >= 1e-3)
    pitch = sample_rate / lag
    values, weights = np.zeros((count, 6), np.float32), np.zeros((count, 6), np.float32)
    values[:, 2] = np.clip(np.log(pitch / 60) / np.log(500 / 60), 0, 1)
    values[:, 3] = np.clip((20 * np.log10(np.maximum(rms, 1e-6)) + 60) / 60, 0, 1)
    # Word rate is an utterance-level proxy, not syllable alignment. / 단어율은 음절 정렬이 아닌 발화 단위 근사치입니다.
    words = re.findall(r"\b[\w]+(?:['’-][\w]+)*\b", transcript, flags=re.UNICODE)
    values[:, 4] = np.clip(len(words) / duration / 6, 0, 1)
    weights[:, 2], weights[:, 3], weights[:, 4] = voiced.astype(np.float32), 1, 1
    return values, weights
