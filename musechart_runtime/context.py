"""Compact whole-song acoustic context used by MuseChartWeaver v1.0."""
from __future__ import annotations

import numpy as np


CONTEXT_BINS = 256
CONTEXT_CHANNELS = 6


def song_context(audio, sample_rate):
    audio = np.asarray(audio, dtype=np.float32)
    size = max(64, int(round(sample_rate * .1)))
    frames = max(1, int(np.ceil(len(audio) / size)))
    chunks = np.pad(audio, (0, frames * size - len(audio))).reshape(frames, size)
    features = np.zeros((CONTEXT_CHANNELS, frames), np.float32)
    features[0] = np.log1p(100 * np.sqrt(np.mean(chunks ** 2, axis=1)))
    frequencies = np.fft.rfftfreq(size, 1 / sample_rate)
    bands = [(20, 150), (150, 600), (600, 2000), (2000, sample_rate / 2 + 1)]
    window = np.hanning(size).astype(np.float32)
    for start in range(0, frames, 128):
        spectrum = abs(np.fft.rfft(chunks[start:start + 128] * window, axis=1))
        for index, (low, high) in enumerate(bands, 1):
            mask = (frequencies >= low) & (frequencies < high)
            if np.any(mask):
                features[index, start:start + len(spectrum)] = np.log1p(
                    spectrum[:, mask].mean(1))
    features[5, 1:] = np.maximum(0, np.diff(features[1:5], axis=1)).mean(0)
    result = np.stack([
        np.interp(np.linspace(0, 1, CONTEXT_BINS), np.linspace(0, 1, frames), row)
        for row in features
    ]).astype(np.float32)
    result = ((result - result.mean(1, keepdims=True))
              / np.maximum(result.std(1, keepdims=True), .05))
    return np.clip(result, -5, 5).astype(np.float32)
