"""v3.2 learned structured decoder used by full-song inference."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch

from .model import joint_pair_logits
from .schema import (BOSS_ALLOWED, DURATION_TYPES, EVENT_BUNDLES,
                     GROUND_ONLY_IDS, IBMS_IDS, IBMS_TO_SEMANTIC,
                     STATE_CONTROLS)


def boss_path(logits, transition_logits, boundary_logits, *, stride=10,
              initial_off=True):
    """Decode one legal whole-song Boss state path with Viterbi."""
    logits = torch.as_tensor(logits).float().log_softmax(0).numpy()
    transitions = torch.as_tensor(transition_logits).float().log_softmax(1).numpy()
    boundary = torch.as_tensor(boundary_logits).float().sigmoid().numpy().reshape(-1)
    frames = logits.shape[-1]
    if not frames:
        return np.empty(0, np.int64)
    stride = max(1, int(stride))
    blocks = list(range(0, frames, stride))
    emission = np.stack([logits[:, index:index + stride].mean(1)
                         for index in blocks])
    score = emission[0].copy()
    if initial_off:
        score[1:] = -1e9
    back = np.zeros((len(blocks), 4), np.int64)
    for index in range(1, len(blocks)):
        probability = np.clip(
            boundary[blocks[index]:blocks[index] + stride].max(), 1e-5, 1 - 1e-5)
        edge = transitions + np.where(
            np.eye(4, dtype=bool), math.log1p(-probability), math.log(probability))
        candidates = score[:, None] + edge
        back[index] = candidates.argmax(0)
        score = candidates.max(0) + emission[index]
    path = np.empty(len(blocks), np.int64)
    path[-1] = score.argmax()
    for index in range(len(blocks) - 1, 0, -1):
        path[index - 1] = back[index, path[index]]
    return np.repeat(path, stride)[:frames]


@torch.no_grad()
def decode_batch(outputs, *, threshold, frame_seconds, vocabulary,
                 minimum_gap=0.03, max_duration=20.0, valid_frames=None,
                 audio_durations=None, initial_off=False,
                 boss_state_stride=10, **legacy):
    from .decoding import suppress_onsets

    if legacy:
        raise ValueError("v3.2 Boss 由状态序列解码，不再支持独立 boss_threshold")
    if (not 0 < threshold < 1 or minimum_gap < 0
            or max_duration <= 0 or frame_seconds <= 0):
        raise ValueError("无效的解码阈值、时间步长或时长范围")
    tensors = {key: value.detach().float().cpu() for key, value in outputs.items()}
    onset = tensors["onset_logits"].sigmoid().numpy()[:, 0]
    bundles = tensors["bundle_logits"].numpy()
    fine = tensors["ibms_logits"].numpy()
    ratios = tensors["duration_ratio_logits"].sigmoid().clamp_min(1e-6).numpy()
    raw_duration = tensors["duration_log"].clamp(0, 6).expm1().numpy()
    valid = (valid_frames.detach().cpu().numpy()[:, 0]
             if valid_frames is not None else np.ones_like(onset))
    results = []
    for batch_index, scores in enumerate(onset):
        count = int(np.count_nonzero(valid[batch_index]))
        end = (float(audio_durations[batch_index]) if audio_durations is not None
               else max(0, (count - 1) * frame_seconds))
        path = boss_path(
            tensors["boss_state_logits"][batch_index, :, :count],
            tensors["boss_transition_logits"],
            tensors["boss_boundary_logits"][batch_index, :, :count],
            stride=boss_state_stride, initial_off=initial_off)
        padded = np.pad(scores, 2, constant_values=-np.inf)
        maxima = np.maximum.reduce([padded[index:index + len(scores)]
                                    for index in range(5)])
        frames = np.flatnonzero(
            (scores >= threshold) & (scores == maxima)
            & (valid[batch_index] > 0)
            & (np.arange(len(scores)) * frame_seconds < end))
        events = []
        frame_choices = {}
        for frame in frames:
            state = int(path[frame])
            next_change = np.flatnonzero(path[frame + 1:] != state)
            change_frame = (frame + 1 + int(next_change[0])
                            if len(next_change) else None)
            off_time = (change_frame * frame_seconds
                        if state == 1 and change_frame is not None
                        and path[change_frame] == 0 else None)
            choices = {}
            for lane in (0, 1):
                for semantic_type in range(1, 9):
                    choices[lane, semantic_type] = [
                        index for index, code in enumerate(IBMS_IDS)
                        if IBMS_TO_SEMANTIC[code] == semantic_type
                        and not (lane == 1 and code in GROUND_ONLY_IDS)
                        and (code not in BOSS_ALLOWED or state in BOSS_ALLOWED[code])
                        and (code != "17" or off_time is not None
                             and off_time - frame * frame_seconds <= max_duration)
                    ]
            possible = [
                index for index, bundle in enumerate(EVENT_BUNDLES)
                if all(not semantic_type or choices[lane, semantic_type]
                       for lane, semantic_type in enumerate(bundle))
            ]
            bundle_index = possible[int(
                bundles[batch_index, possible, frame].argmax())]
            bundle = EVENT_BUNDLES[bundle_index]
            frame_choices[int(frame)] = choices
            if all(bundle):
                left = choices[0, bundle[0]]
                right = choices[1, bundle[1]]
                joint = joint_pair_logits(
                    tensors, torch.tensor([batch_index]), torch.tensor([frame]))[0].numpy()
                table = joint[np.ix_(left, right)]
                x, y = np.unravel_index(table.argmax(), table.shape)
                ids = (left[x], right[y])
            else:
                ids = tuple(
                    choices[lane, semantic_type][int(
                        fine[batch_index, lane,
                             choices[lane, semantic_type], frame].argmax())]
                    if semantic_type else -1
                    for lane, semantic_type in enumerate(bundle))
            for lane, (semantic_type, class_index) in enumerate(zip(bundle, ids)):
                if not semantic_type:
                    continue
                probabilities = torch.from_numpy(
                    fine[batch_index, lane, :, frame]).softmax(0)
                events.append({
                    "time": round(frame * frame_seconds, 6),
                    "lane": lane,
                    "semantic_type": semantic_type,
                    "ibms_id": IBMS_IDS[class_index],
                    "ibms_source": "structured_model",
                    "confidence": float(scores[frame]),
                    "ibms_confidence": float(probabilities[class_index]),
                    "duration": 0.0,
                    "raw_duration": (float(raw_duration[batch_index, lane, frame])
                                     if semantic_type in DURATION_TYPES else 0.0),
                    "duration_ratio": float(ratios[batch_index, lane, frame]),
                    "is_hold": semantic_type in DURATION_TYPES,
                    "is_double": all(bundle),
                    "boss_state": state,
                    "_off_time": off_time,
                    "_frame": int(frame),
                })
        events = suppress_onsets(events, minimum_gap)
        for lane in (0, 1):
            lane_events = [event for event in events if event["lane"] == lane]
            for index, event in enumerate(lane_events):
                bound = ((lane_events[index + 1]["time"]
                          if index + 1 < len(lane_events) else end) - event["time"])
                event["_bound"] = max(0, bound)
                if event["semantic_type"] in DURATION_TYPES:
                    event["duration_bound"] = max(0, bound)
                    event["duration"] = min(
                        max_duration, bound * event["duration_ratio"])
                event["tick"] = event["time"]

        groups = defaultdict(list)
        for event in events:
            groups[event["_frame"]].append(event)
        implicit_exits = set()
        for frame, group in groups.items():
            valid_choices = []
            raw_choices = []
            fits_end = []
            for event in group:
                off_time = event["_off_time"]
                fits = (off_time is not None
                        and off_time <= event["time"] + event["_bound"] + 1e-6)
                fits_end.append(fits)
                raw_choices.append([
                    index for index, code in enumerate(IBMS_IDS)
                    if IBMS_TO_SEMANTIC[code] == event["semantic_type"]
                    and not (event["lane"] == 1 and code in GROUND_ONLY_IDS)
                    and (code not in BOSS_ALLOWED
                         or event["boss_state"] in BOSS_ALLOWED[code])
                ])
                valid_choices.append([
                    class_index
                    for class_index in frame_choices[
                        frame][event["lane"], event["semantic_type"]]
                    if IBMS_IDS[class_index] != "17" or fits
                ])
            if len(group) == 2:
                left, right = valid_choices
                joint = joint_pair_logits(
                    tensors, torch.tensor([batch_index]), torch.tensor([frame]))[0].numpy()
                table = joint[np.ix_(left, right)]
                x, y = np.unravel_index(table.argmax(), table.shape)
                selected_ids = (left[x], right[y])
                raw_left, raw_right = raw_choices
                raw_table = joint[np.ix_(raw_left, raw_right)]
                x, y = np.unravel_index(raw_table.argmax(), raw_table.shape)
                raw_ids = (raw_left[x], raw_right[y])
            else:
                choices = valid_choices[0]
                selected_ids = (choices[int(
                    fine[batch_index, group[0]["lane"], choices, frame].argmax())],)
                choices = raw_choices[0]
                raw_ids = (choices[int(
                    fine[batch_index, group[0]["lane"], choices, frame].argmax())],)
            for event, class_index, raw_index, fits in zip(
                    group, selected_ids, raw_ids, fits_end):
                event["raw_boss_end_conflict"] = bool(
                    IBMS_IDS[raw_index] == "17" and not fits)
                event["ibms_id"] = IBMS_IDS[class_index]
                event["ibms_confidence"] = float(torch.from_numpy(
                    fine[batch_index, event["lane"], :, frame]
                ).softmax(0)[class_index])
                if event["ibms_id"] == "17":
                    event["duration"] = event["_off_time"] - event["time"]
                    event["boss_end_conflict"] = False
                    implicit_exits.add(round(event["_off_time"], 6))
                for key in ("_off_time", "_frame", "_bound"):
                    del event[key]
        for frame in np.flatnonzero(np.diff(path) != 0) + 1:
            code = STATE_CONTROLS[(int(path[frame - 1]), int(path[frame]))]
            time = round(frame * frame_seconds, 6)
            if time >= end or code == "1B" and time in implicit_exits:
                continue
            events.append({
                "time": time, "tick": time, "lane": -1,
                "semantic_type": 0, "ibms_id": code, "duration": 0.0,
                "confidence": float(tensors["boss_state_logits"][
                    batch_index, :, frame].softmax(0)[path[frame]]),
                "ibms_source": "model_state_transition",
            })
        results.append(sorted(events, key=lambda event: (
            event["time"], event["lane"])))
    return results
