from __future__ import annotations

import unittest

import numpy as np

from musechart_runtime.tempo import _select_canonical_peak, estimate_tempo


def click_track(bpm: float, *, seconds: float = 35.0,
                sample_rate: int = 24000) -> np.ndarray:
    audio = np.zeros(int(seconds * sample_rate), dtype=np.float32)
    length = int(0.025 * sample_rate)
    click = np.hanning(length * 2)[:length].astype(np.float32)
    for index in range(int(seconds * bpm / 60.0)):
        start = int(round((0.15 + index * 60.0 / bpm) * sample_rate))
        if start + length <= len(audio):
            audio[start:start + length] += click
    return audio


class TempoTests(unittest.TestCase):
    def test_estimates_120_bpm_clicks(self):
        result = estimate_tempo(click_track(120.0), 24000)
        self.assertLess(abs(result["bpm"] - 120.0), 1.0)
        self.assertLess(abs(result["beat_offset_seconds"] - 0.15), 0.08)

    def test_estimates_180_bpm_clicks(self):
        result = estimate_tempo(click_track(180.0), 24000)
        self.assertLess(abs(result["bpm"] - 180.0), 1.0)

    def test_hint_is_preserved_and_phase_is_measured(self):
        result = estimate_tempo(click_track(128.0), 24000, bpm_hint=128.0)
        self.assertEqual(result["bpm"], 128.0)
        self.assertEqual(result["confidence"], 1.0)

    def test_promotes_strong_triplet_alias_to_higher_meter(self):
        grid = np.array([100.0, 133.25, 200.0])
        scores = np.array([0.89, 1.0, 0.90])
        selected, reason = _select_canonical_peak([1, 2, 0], scores, grid)
        self.assertEqual(grid[selected], 200.0)
        self.assertEqual(reason, "harmonic_promotion")

    def test_silence_falls_back_to_unknown_bpm(self):
        result = estimate_tempo(np.zeros(24000 * 5, dtype=np.float32), 24000)
        self.assertEqual(result["bpm"], 0.0)
        self.assertEqual(result["selection"], "no_reliable_pulse")


if __name__ == "__main__":
    unittest.main()
