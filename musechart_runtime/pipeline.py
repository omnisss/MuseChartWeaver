"""Chunked audio encoding followed by one whole-song v3.2 decode."""
from __future__ import annotations

import contextlib
import math

import numpy as np
import torch
from tqdm import tqdm

from .context import song_context
from .decoding import decode_batch


STATIC_OUTPUTS = {"pair_embeddings", "boss_transition_logits"}


@torch.inference_mode()
def predict_outputs(*, model, audio, difficulty, bpm, config, overlap_seconds,
                    device, progress=False, beat_offset=0.0, beat_known=False):
    data = config["data"]
    sample_rate = int(data["sample_rate"])
    hop = int(data["hop_length"])
    segment = int(round(float(data["segment_seconds"]) * sample_rate))
    overlap = int(round(overlap_seconds * sample_rate))
    if not len(audio) or not 0 <= overlap < segment or segment % hop:
        raise ValueError("音频不能为空；重叠范围无效或分块长度非整数帧")
    if not math.isfinite(bpm) or bpm < 0 or not math.isfinite(beat_offset):
        raise ValueError("BPM/beat offset 无效")
    stride = max(hop, ((segment - overlap) // hop) * hop)
    last = max(0, math.ceil((len(audio) - segment) / hop) * hop)
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    edges = ([0] + [(left + segment + right) / 2
                    for left, right in zip(starts, starts[1:])]
             + [len(audio) + 1])
    amp = str(config["train"].get("amp", "bf16"))
    if device.type == "cuda" and amp == "bf16" and not torch.cuda.is_bf16_supported():
        amp = "fp16"
    result = {}
    covered = torch.zeros(len(audio) // hop + 1, dtype=torch.bool)
    global_audio = torch.from_numpy(song_context(audio, sample_rate))[None].to(device)
    model.eval()
    iterator = tqdm(starts, desc=f"difficulty {difficulty}",
                    disable=not progress, leave=False)
    for chunk_index, start in enumerate(iterator):
        chunk = audio[start:start + segment]
        waveform = torch.from_numpy(np.ascontiguousarray(
            np.pad(chunk, (0, segment - len(chunk))))).unsqueeze(0).to(device)

        def tensor(value):
            return torch.tensor([value], dtype=torch.float32, device=device)

        autocast = (torch.autocast(
            "cuda", dtype=torch.bfloat16 if amp == "bf16" else torch.float16)
            if device.type == "cuda" and amp != "off"
            else contextlib.nullcontext())
        with autocast:
            outputs = model(
                waveform, torch.tensor([difficulty], device=device), tensor(bpm),
                segment_start=tensor(start / sample_rate),
                audio_duration=tensor(len(audio) / sample_rate),
                beat_offset=tensor(beat_offset),
                beat_known=tensor(float(beat_known and bpm > 0)),
                audio_context=global_audio)
        indices = torch.arange(outputs["onset_logits"].shape[-1]) + start // hop
        owned = ((indices * hop >= edges[chunk_index])
                 & (indices * hop < edges[chunk_index + 1])
                 & (indices < len(covered)))
        local = torch.where(owned)[0]
        global_indices = indices[owned]
        covered[global_indices] = True
        for key, value in outputs.items():
            value = value.detach().float().cpu()
            if key in STATIC_OUTPUTS:
                result[key] = value
                continue
            if key not in result:
                result[key] = torch.zeros(
                    (*value.shape[:-1], len(covered)), dtype=torch.float32)
            result[key][..., global_indices] = value[..., local]
    if not covered.all():
        raise RuntimeError("分块拼接存在未覆盖帧")
    return result


def decode_song(outputs, *, config, vocabulary, threshold, audio_duration,
                max_duration=None):
    train = config["train"]
    return decode_batch(
        outputs, threshold=threshold,
        frame_seconds=config["data"]["hop_length"] / config["data"]["sample_rate"],
        vocabulary=vocabulary,
        minimum_gap=float(train["minimum_gap_seconds"]),
        max_duration=float(max_duration if max_duration is not None
                           else train["max_duration_seconds"]),
        audio_durations=[audio_duration], initial_off=True,
        boss_state_stride=int(train.get("boss_state_stride", 10)))[0]


def predict_difficulty(*, model, audio, difficulty, bpm, vocabulary, config,
                       threshold, overlap_seconds, max_duration, device,
                       progress=True, beat_offset=0.0, beat_known=False):
    outputs = predict_outputs(
        model=model, audio=audio, difficulty=difficulty, bpm=bpm, config=config,
        overlap_seconds=overlap_seconds, device=device, progress=progress,
        beat_offset=beat_offset, beat_known=beat_known)
    return decode_song(
        outputs, config=config, vocabulary=vocabulary, threshold=threshold,
        audio_duration=len(audio) / config["data"]["sample_rate"],
        max_duration=max_duration)
