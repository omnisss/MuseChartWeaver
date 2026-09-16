from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn


def load_audio_segment(
    path: str | Path,
    *,
    start_sample: int,
    sample_count: int,
    expected_sample_rate: int,
) -> np.ndarray:
    """Read one mono segment without loading a complete song into RAM."""
    import soundfile as sf

    with sf.SoundFile(str(path), "r") as audio:
        if audio.samplerate != expected_sample_rate:
            raise ValueError(
                f"采样率不匹配: {path} 是 {audio.samplerate} Hz，"
                f"训练配置要求 {expected_sample_rate} Hz"
            )
        audio.seek(min(max(0, start_sample), len(audio)))
        values = audio.read(sample_count, dtype="float32", always_2d=True)
    mono = values.mean(axis=1, dtype=np.float32)
    if mono.shape[0] < sample_count:
        mono = np.pad(mono, (0, sample_count - mono.shape[0]))
    return np.ascontiguousarray(mono, dtype=np.float32)


def load_audio(path: str | Path, target_sample_rate: int) -> np.ndarray:
    import soundfile as sf

    values, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = values.mean(axis=1, dtype=np.float32)
    if sample_rate != target_sample_rate:
        from scipy.signal import resample_poly
        divisor = math.gcd(int(sample_rate), int(target_sample_rate))
        mono = resample_poly(
            mono,
            target_sample_rate // divisor,
            sample_rate // divisor,
        ).astype(np.float32, copy=False)
    return np.ascontiguousarray(mono, dtype=np.float32)


def _hz_to_mel(frequency: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + frequency / 700.0)


def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)


def make_mel_filter(
    *,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float,
    f_max: float,
) -> torch.Tensor:
    if not 0 <= f_min < f_max <= sample_rate / 2:
        raise ValueError("mel 频率范围必须位于 [0, Nyquist] 内")
    fft_frequencies = torch.linspace(0.0, sample_rate / 2, n_fft // 2 + 1)
    mel_min = _hz_to_mel(torch.tensor(float(f_min)))
    mel_max = _hz_to_mel(torch.tensor(float(f_max)))
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)

    lower = hz_points[:-2, None]
    center = hz_points[1:-1, None]
    upper = hz_points[2:, None]
    up_slope = (fft_frequencies[None, :] - lower) / (center - lower).clamp_min(1.0e-8)
    down_slope = (upper - fft_frequencies[None, :]) / (upper - center).clamp_min(1.0e-8)
    filters = torch.minimum(up_slope, down_slope).clamp_min(0.0)
    filters *= (2.0 / (upper - lower).clamp_min(1.0e-8))
    return filters


class LogMelSpectrogram(nn.Module):
    def __init__(
        self,
        *,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        f_min: float,
        f_max: float,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.register_buffer(
            "mel_filter",
            make_mel_filter(
                sample_rate=sample_rate,
                n_fft=n_fft,
                n_mels=n_mels,
                f_min=f_min,
                f_max=f_max,
            ),
            persistent=True,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # STFT remains float32 under autocast, which avoids low-precision FFT issues.
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            spectrum = torch.stft(
                waveform.float(),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=self.window.float(),
                center=True,
                pad_mode="reflect",
                return_complex=True,
            ).abs().square()
            mel = torch.matmul(self.mel_filter.float(), spectrum)
            features = torch.log1p(10.0 * mel)
            mean = features.mean(dim=(-2, -1), keepdim=True)
            std = features.std(dim=(-2, -1), keepdim=True).clamp_min(1.0e-5)
            return (features - mean) / std


