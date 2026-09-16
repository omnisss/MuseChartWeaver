"""Shared onset suppression plus the v3.2 structured decoder."""
from __future__ import annotations

from bisect import bisect_left, insort

from .schema import BOSS_CONTROL_IDS


def suppress_onsets(events: list[dict], minimum_gap: float) -> list[dict]:
    """NMS over time groups, preserving a predicted double as one group."""
    controls = [event for event in events if event.get("semantic_type") == 0]
    groups: dict[float, list[dict]] = {}
    for event in events:
        if event.get("semantic_type") == 0:
            continue
        groups.setdefault(float(event["time"]), []).append(event)
    selected = []
    times = []
    ordered_groups = sorted(
        groups.items(), key=lambda item: (
            -max(event["confidence"] for event in item[1]), item[0]))
    for time, group in ordered_groups:
        index = bisect_left(times, time)
        if ((index and time - times[index - 1] < minimum_gap - 1e-8)
                or (index < len(times)
                    and times[index] - time < minimum_gap - 1e-8)):
            continue
        insort(times, time)
        by_lane = {}
        for event in sorted(group, key=lambda value: -value["confidence"]):
            by_lane.setdefault(event["lane"], dict(event))
        for event in by_lane.values():
            event["is_double"] = len(by_lane) == 2
            selected.append(event)
    # Independent control streams never suppress gameplay or another control ID.
    for code in BOSS_CONTROL_IDS:
        control_times = []
        candidates = (event for event in controls if event["ibms_id"] == code)
        for event in sorted(candidates,
                            key=lambda value: (-value["confidence"], value["time"])):
            time = float(event["time"])
            index = bisect_left(control_times, time)
            if ((index and time - control_times[index - 1] < minimum_gap - 1e-8)
                    or (index < len(control_times)
                        and control_times[index] - time < minimum_gap - 1e-8)):
                continue
            if time in control_times:
                continue
            insort(control_times, time)
            selected.append(dict(event))
    return sorted(selected, key=lambda event: (event["time"], event["lane"]))


from .structured_decoding import decode_batch  # noqa: E402

__all__ = ["decode_batch", "suppress_onsets"]
