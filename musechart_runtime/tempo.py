"""Dependency-light global tempo estimation for local MuseChart inference."""
from __future__ import annotations

import math

import numpy as np


def _moving_average(values: np.ndarray, size: int) -> np.ndarray:
    size = max(1, int(size))
    left = size // 2
    right = size - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    cumulative = np.cumsum(np.concatenate(([0.0], padded)), dtype=np.float64)
    return ((cumulative[size:] - cumulative[:-size]) / size).astype(np.float32)


def _find_peaks(values: np.ndarray, minimum_distance: int) -> np.ndarray:
    if len(values) < 3:
        return np.array([int(values.argmax())], dtype=np.int64)
    candidates = np.flatnonzero(
        (values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:])) + 1
    selected = []
    for index in sorted(candidates, key=lambda item: float(values[item]), reverse=True):
        if all(abs(int(index) - previous) >= minimum_distance for previous in selected):
            selected.append(int(index))
    return np.asarray(selected, dtype=np.int64)


def _select_canonical_peak(ranked: list[int], scores: np.ndarray,
                           bpm_grid: np.ndarray) -> tuple[int, str]:
    """Prefer a strong higher metrical level over half/dotted-beat aliases."""
    raw = ranked[0]
    selected = raw
    raw_score = float(scores[raw])
    for index in ranked[1:]:
        lower = float(bpm_grid[selected])
        higher = float(bpm_grid[index])
        if higher <= lower or float(scores[index]) < 0.85 * raw_score:
            continue
        ratio = higher / lower
        if min(abs(ratio - 1.5), abs(ratio - 2.0)) <= 0.025:
            selected = index
    return selected, "acoustic_peak" if selected == raw else "harmonic_promotion"


def _spectral_flux(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, float]:
    if audio.ndim != 1 or not len(audio):
        raise ValueError("BPM 检测需要非空单声道音频")
    n_fft = 2048 if sample_rate >= 16000 else 1024
    hop = max(1, int(round(sample_rate / 100.0)))
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)))
    starts = np.arange(0, len(audio) - n_fft + 1, hop, dtype=np.int64)
    if not len(starts):
        starts = np.array([0], dtype=np.int64)
    window = np.hanning(n_fft).astype(np.float32)
    bins = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    use = (bins >= 40.0) & (bins <= min(10000.0, sample_rate / 2.0))
    flux = np.zeros(len(starts), dtype=np.float32)
    previous = None
    offsets = np.arange(n_fft, dtype=np.int64)
    for block_start in range(0, len(starts), 512):
        block = starts[block_start:block_start + 512]
        frames = audio[block[:, None] + offsets[None, :]].astype(np.float32, copy=False)
        magnitude = np.abs(np.fft.rfft(frames * window, axis=1))[:, use]
        magnitude = np.log1p(10.0 * magnitude).astype(np.float32, copy=False)
        if previous is None:
            differences = np.diff(magnitude, axis=0, prepend=magnitude[:1])
        else:
            differences = np.diff(magnitude, axis=0, prepend=previous[None, :])
        flux[block_start:block_start + len(block)] = np.maximum(differences, 0.0).mean(axis=1)
        previous = magnitude[-1]
    envelope_rate = sample_rate / hop
    baseline = _moving_average(
        flux, size=max(3, int(round(envelope_rate * 2.0))))
    flux = np.maximum(flux - 0.55 * baseline, 0.0)
    scale = float(np.percentile(flux, 95))
    if scale > 1.0e-8:
        flux = np.minimum(flux / scale, 4.0)
    return flux.astype(np.float32, copy=False), float(envelope_rate)


def _average_autocorrelation(envelope: np.ndarray, rate: float,
                             max_lag: int) -> np.ndarray:
    segment = max(max_lag * 4, int(round(rate * 30.0)))
    stride = max(1, segment // 2)
    starts = list(range(0, max(1, len(envelope) - segment + 1), stride))
    final = max(0, len(envelope) - segment)
    if not starts or starts[-1] != final:
        starts.append(final)
    accumulated = np.zeros(max_lag + 1, dtype=np.float64)
    used = 0
    for start in starts:
        values = envelope[start:start + segment].astype(np.float64, copy=True)
        if len(values) < max_lag * 2:
            continue
        values -= values.mean()
        norm = float(np.dot(values, values))
        if norm <= 1.0e-10:
            continue
        fft_size = 1 << (2 * len(values) - 1).bit_length()
        spectrum = np.fft.rfft(values, fft_size)
        correlation = np.fft.irfft(
            spectrum * spectrum.conj(), fft_size)[:max_lag + 1].real
        correlation /= np.maximum(1.0, len(values) - np.arange(max_lag + 1))
        correlation /= max(1.0e-12, correlation[0])
        accumulated += correlation
        used += 1
    return accumulated / max(1, used)


def _beat_phase(envelope: np.ndarray, rate: float, bpm: float) -> float:
    period = max(1, int(round(rate * 60.0 / bpm)))
    scores = np.array([
        float(envelope[offset::period].mean()) if offset < len(envelope) else 0.0
        for offset in range(period)
    ])
    return float(int(scores.argmax()) / rate)


def estimate_tempo(audio: np.ndarray, sample_rate: int, *, min_bpm: float = 70.0,
                   max_bpm: float = 240.0, bpm_hint: float | None = None) -> dict:
    """Estimate global BPM and beat phase, returning an auditable result dict."""
    if sample_rate <= 0:
        raise ValueError("sample_rate 必须大于 0")
    if not 20 <= min_bpm < max_bpm <= 1000:
        raise ValueError("BPM 搜索范围无效")
    envelope, rate = _spectral_flux(np.asarray(audio, dtype=np.float32), sample_rate)
    if bpm_hint is not None:
        if not math.isfinite(bpm_hint) or bpm_hint <= 0:
            raise ValueError("bpm_hint 必须是正有限数字")
        return {
            "bpm": round(float(bpm_hint), 3),
            "confidence": 1.0,
            "beat_offset_seconds": round(_beat_phase(envelope, rate, bpm_hint), 6),
            "selection": "provided_hint",
            "raw_best_bpm": round(float(bpm_hint), 3),
            "candidates": [{"bpm": round(float(bpm_hint), 3), "score": 1.0,
                            "selected": True}],
        }

    max_lag = int(math.ceil(rate * 60.0 / min_bpm * 3.0))
    autocorrelation = _average_autocorrelation(envelope, rate, max_lag)
    bpm_grid = np.arange(min_bpm, max_bpm + 0.125, 0.25, dtype=np.float64)
    lag_grid = np.arange(len(autocorrelation), dtype=np.float64)
    lags = rate * 60.0 / bpm_grid
    base = np.interp(lags, lag_grid, autocorrelation)
    double_lag = np.interp(2.0 * lags, lag_grid, autocorrelation, right=0.0)
    triple_lag = np.interp(3.0 * lags, lag_grid, autocorrelation, right=0.0)
    rhythmic = np.maximum(base + 0.45 * double_lag + 0.20 * triple_lag, 0.0)
    # Broad prior resolves many half/double ambiguities while still allowing
    # genuinely slow songs.  162 BPM is the median of the prepared dataset.
    prior = np.exp(-0.5 * (np.log(np.maximum(bpm_grid, 1.0) / 162.0) / 0.50) ** 2)
    scores = rhythmic * (0.78 + 0.22 * prior)
    if float(scores.max()) <= 1.0e-8:
        return {"bpm": 0.0, "confidence": 0.0, "beat_offset_seconds": 0.0,
                "selection": "no_reliable_pulse", "raw_best_bpm": None,
                "candidates": []}
    peak_indices = _find_peaks(
        scores, minimum_distance=max(1, int(round(4.0 / 0.25))))
    if not len(peak_indices):
        peak_indices = np.array([int(scores.argmax())])
    ranked = sorted(peak_indices, key=lambda index: float(scores[index]), reverse=True)
    raw_best_index = ranked[0]
    best_index, selection = _select_canonical_peak(ranked, scores, bpm_grid)
    best_bpm = float(bpm_grid[best_index])
    best_score = float(scores[best_index])
    raw_best_score = float(scores[raw_best_index])
    baseline = float(np.median(scores))
    second_score = float(scores[ranked[1]]) if len(ranked) > 1 else baseline
    prominence = max(0.0, (raw_best_score - baseline) / max(1.0e-8, raw_best_score))
    separation = max(0.0, (raw_best_score - second_score) / max(1.0e-8, raw_best_score))
    confidence = min(1.0, (0.75 * prominence + 0.25 * separation)
                     * best_score / max(1.0e-8, raw_best_score))
    candidates = [
        {"bpm": round(float(bpm_grid[index]), 3),
         "score": round(float(scores[index] / max(1.0e-8, raw_best_score)), 6),
         "selected": bool(index == best_index)}
        for index in ranked[:5]
    ]
    return {
        "bpm": round(best_bpm, 3),
        "confidence": round(confidence, 6),
        "beat_offset_seconds": round(_beat_phase(envelope, rate, best_bpm), 6),
        "selection": selection,
        "raw_best_bpm": round(float(bpm_grid[raw_best_index]), 3),
        "candidates": candidates,
    }
