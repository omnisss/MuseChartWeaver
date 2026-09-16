"""Optional phrase-level cleanup for V3.1 ordinary-monster choices."""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict

from .schema import BOSS_PLAY_IDS


# Official dataset_stage1 event counts, summed over both lanes.  These are used
# only as relative probabilities; the checkpoint catalog remains the authority
# for whether a concrete object exists in the requested scene/lane/speed.
NORMAL_MONSTER_WEIGHTS = {
    "01": 94645,
    "02": 28058,
    "03": 28651,
    "04": 44174,
    "05": 51064,
    "06": 53387,
    "07": 36131,
    "08": 47753,
    "09": 47782,
    "0A": 52646,
    "0B": 42035,
    "0C": 15414,
    "0D": 19857,
}
# Per-difficulty counts preserve the official tendency to introduce rarer
# variants more often on harder charts.  Difficulty 4 only has two source
# charts, so it deliberately falls back to the much more reliable level-3
# distribution.
DIFFICULTY_MONSTER_WEIGHTS = {
    1: {"01": 16375, "02": 1989, "03": 1945, "04": 11738, "05": 6873,
        "06": 6884, "07": 9888, "08": 6042, "09": 6221, "0A": 10931,
        "0B": 10336, "0C": 766, "0D": 863},
    2: {"01": 24334, "02": 5616, "03": 5782, "04": 12251, "05": 13553,
        "06": 14232, "07": 9850, "08": 12534, "09": 12511, "0A": 15964,
        "0B": 12935, "0C": 3420, "0D": 4797},
    3: {"01": 44635, "02": 17610, "03": 17759, "04": 15929, "05": 25133,
        "06": 26300, "07": 12981, "08": 24059, "09": 24094, "0A": 20005,
        "0B": 14371, "0C": 9876, "0D": 12434},
}

# Frequent non-self transitions observed in official charts.  A phrase starts
# with a weighted main object, then draws its small palette from these familiar
# neighbours instead of combining arbitrary objects independently.
MONSTER_COMPANIONS = {
    "01": ("0A", "04", "0B", "0D"),
    "02": ("05", "08", "01", "0C"),
    "03": ("06", "09", "01", "0D"),
    "04": ("07", "01", "0A", "0B"),
    "05": ("08", "0A", "01", "0C"),
    "06": ("09", "0A", "01", "0D"),
    "07": ("04", "0A", "0B", "01"),
    "08": ("05", "0A", "0B", "01"),
    "09": ("06", "0A", "0B", "01"),
    "0A": ("0B", "01", "06", "05"),
    "0B": ("0A", "01", "04", "07"),
    "0C": ("0D", "05", "08", "0A"),
    "0D": ("0C", "06", "09", "01"),
}
SAFE_SINGLE_MONSTER_IBMS = frozenset(NORMAL_MONSTER_WEIGHTS)
GEMINI_IBMS = "0E"
STYLE_STRATEGIES = ("patterned", "fixed", "weighted", "varied")


def _catalog_candidates(catalog: dict, *, semantic_type: int, lane: int,
                        scene: str, speed: int) -> dict[str, dict]:
    candidates = [
        item for item in catalog.get("candidates", [])
        if int(item.get("semantic_type", -1)) == semantic_type
        and int(item.get("lane", -1)) == lane
    ]
    local = [item for item in candidates if str(item.get("scene", "")) == scene]
    candidates = local or [item for item in candidates if str(item.get("scene", "")) == "0"]
    if not candidates:
        return {}
    distance = min(abs(int(item.get("note_speed", speed)) - speed) for item in candidates)
    candidates = [
        item for item in candidates
        if abs(int(item.get("note_speed", speed)) - speed) == distance
    ]
    # A catalog can contain duplicate seasonal prefabs for one ibms_id.  Pick a
    # deterministic representative; MDM only consumes ibms_id, while note_uid
    # is retained in the sidecar for auditing.
    result = {}
    for item in sorted(candidates, key=lambda value: (
            str(value.get("ibms_id", "")).upper(), str(value.get("note_uid", "")))):
        ibms_id = str(item.get("ibms_id", "")).strip().upper()
        if ibms_id and ibms_id not in result:
            result[ibms_id] = item
    return result


def _choose_monster_id(available: list[str], *, strategy: str,
                       rng: random.Random, previous: str | None) -> str:
    if strategy == "fixed":
        return "01" if "01" in available else available[0]
    choices = [value for value in available if value != previous] if len(available) > 1 else available
    if strategy == "varied":
        weights = [1.0] * len(choices)
    else:
        weights = [float(NORMAL_MONSTER_WEIGHTS[value]) for value in choices]
    return rng.choices(choices, weights=weights, k=1)[0]


def _weights_for_difficulty(difficulty: int) -> dict[str, int]:
    return DIFFICULTY_MONSTER_WEIGHTS.get(difficulty, DIFFICULTY_MONSTER_WEIGHTS[3])


def _weighted_pick(values: list[str], weights: dict[str, int], rng: random.Random) -> str:
    return rng.choices(values, weights=[float(weights[value]) for value in values], k=1)[0]


def _phrase_palette(available: list[str], *, difficulty: int,
                    rng: random.Random) -> list[str]:
    weights = _weights_for_difficulty(difficulty)
    size = min(len(available), 2 if difficulty <= 2 else 3)
    main = _weighted_pick(available, weights, rng)
    palette = [main]
    familiar = [value for value in MONSTER_COMPANIONS[main]
                if value in available and value not in palette]
    while familiar and len(palette) < size:
        chosen = _weighted_pick(familiar, weights, rng)
        palette.append(chosen)
        familiar.remove(chosen)
    remaining = [value for value in available if value not in palette]
    while remaining and len(palette) < size:
        chosen = _weighted_pick(remaining, weights, rng)
        palette.append(chosen)
        remaining.remove(chosen)
    return palette


def _sample_run_length(difficulty: int, rng: random.Random) -> int:
    # Official same-ID transition rates are roughly 73% for all three main
    # difficulties, corresponding to a mean run close to four events.
    if difficulty <= 1:
        values, weights = (3, 4, 6, 8), (3, 4, 2, 1)
    elif difficulty == 2:
        values, weights = (3, 4, 5, 6), (3, 4, 2, 1)
    else:
        # Dense charts alternate lanes more often, so a slightly longer
        # per-lane run is needed to retain the official ~74% self transition.
        values, weights = (4, 5, 6, 8), (3, 4, 2, 1)
    return rng.choices(values, weights=weights, k=1)[0]


def _motif_roles(slots: list[dict], *, palette_size: int, difficulty: int,
                 rng: random.Random) -> list[int]:
    if not slots:
        return []
    roles = []
    current = 0
    remaining = _sample_run_length(difficulty, rng)
    for index, slot in enumerate(slots):
        if index > 0 and remaining <= 0 and palette_size > 1:
            alternatives = [value for value in range(palette_size) if value != current]
            # Role zero is the phrase's main object and is deliberately more
            # likely when returning from an accent/secondary run.
            role_weights = [3.0 if value == 0 else 1.0 for value in alternatives]
            current = rng.choices(alternatives, weights=role_weights, k=1)[0]
            remaining = _sample_run_length(difficulty, rng)
        roles.append(current)
        remaining -= 1
    return roles


def _timing_slot(time: float, bpm: float, beat_offset: float) -> tuple[float, int, int, int]:
    # With known BPM, a phrase is four 4/4 bars.  Unknown-BPM inference uses a
    # stable 120-BPM structural grid (eight seconds per phrase); this changes
    # only styling and never event time or BMS coordinates.
    effective_bpm = bpm if bpm > 0 else 120.0
    beat = (time - beat_offset) * effective_bpm / 60.0
    bar = int(math.floor(beat / 4.0))
    position = int(math.floor((beat - bar * 4.0) * 4.0 + 0.5))
    if position >= 16:
        bar += 1
        position = 0
    return beat, bar // 4, bar, position


def _patterned_assignments(entries: list[tuple[int, dict]], *, available: list[str],
                           difficulty: int, bpm: float, beat_offset: float,
                           rng: random.Random) -> tuple[dict[int, str], dict]:
    phrases = defaultdict(list)
    for original_index, event in entries:
        time = float(event["time"])
        beat, phrase, bar, position = _timing_slot(time, bpm, beat_offset)
        phrases[phrase].append({"index": original_index, "time": time,
                                "lane": int(event["lane"]), "beat": beat,
                                "bar": bar, "position": position})

    assignments: dict[int, str] = {}
    motif_cache: dict[tuple[tuple[int, int], ...], list[int]] = {}
    repeated_motifs = 0
    palette_report = []
    for phrase, phrase_slots in sorted(phrases.items()):
        palette = _phrase_palette(available, difficulty=difficulty, rng=rng)
        preliminary_roles = {}
        for lane in (0, 1):
            lane_slots = sorted(
                (slot for slot in phrase_slots if slot["lane"] == lane),
                key=lambda value: (value["time"], value["index"]))
            roles = _motif_roles(lane_slots, palette_size=len(palette),
                                 difficulty=difficulty, rng=rng)
            preliminary_roles.update(
                (int(slot["index"]), role) for slot, role in zip(lane_slots, roles))
        bars = defaultdict(list)
        for slot in sorted(phrase_slots, key=lambda value: (
                value["time"], value["lane"], value["index"])):
            bars[slot["bar"]].append(slot)
        for bar_slots in bars.values():
            for lane in (0, 1):
                lane_slots = [slot for slot in bar_slots if slot["lane"] == lane]
                if not lane_slots:
                    continue
                # Eighth-note quantization tolerates the few-millisecond timing
                # variation of model output.  Omitting the lane lets the same
                # rhythm played on the other lane reuse the same role pattern.
                signature = tuple(int(slot["position"]) // 2 for slot in lane_slots)
                roles = motif_cache.get(signature)
                if roles is None:
                    roles = [preliminary_roles[int(slot["index"])] for slot in lane_slots]
                    motif_cache[signature] = roles
                else:
                    repeated_motifs += 1
                for slot, role in zip(lane_slots, roles):
                    assignments[int(slot["index"])] = palette[role]
        palette_report.append({
            "phrase": int(phrase),
            "start_seconds": round(min(slot["time"] for slot in phrase_slots), 6),
            "end_seconds": round(max(slot["time"] for slot in phrase_slots), 6),
            "ibms_ids": palette,
        })
    return assignments, {
        "timing_mode": "musical_bars" if bpm > 0 else "fixed_120bpm_style_grid",
        "beat_offset_seconds": beat_offset,
        "phrase_beats": 16,
        "phrase_count": len(phrases),
        "unique_motifs": len(motif_cache),
        "reused_motifs": repeated_motifs,
        "phrase_palettes": palette_report,
    }


def _transition_report(events: list[dict]) -> dict:
    same = total = 0
    for lane in (0, 1):
        sequence = [str(event.get("ibms_id", "")).upper() for event in sorted(
            (event for event in events
             if int(event.get("lane", -1)) == lane
             and int(event.get("semantic_type", -1)) == 1
             and str(event.get("ibms_id", "")).upper() in SAFE_SINGLE_MONSTER_IBMS),
            key=lambda value: float(value["time"]))]
        same += sum(left == right for left, right in zip(sequence, sequence[1:]))
        total += max(0, len(sequence) - 1)
    return {"same_id_transitions": same, "single_monster_transitions": total,
            "same_id_transition_rate": same / max(1, total)}


def apply_mdm_style(events: list[dict], catalog: dict, *, scene: str, speed: int,
                    difficulty: int = 2, bpm: float = 0.0, beat_offset: float = 0.0,
                    strategy: str = "patterned", seed: int = 20260913,
                    enabled: bool = True) -> tuple[list[dict], dict]:
    """Make ordinary monster IDs coherent without touching learned mechanics.

    V3.1 directly predicts IBMS IDs. When enabled, only ordinary type-1 IDs are
    rewritten into small phrase palettes. Boss gameplay IDs and semantic types
    2-8 are always preserved. Double-to-0E conversion is intentionally handled
    by a separate switch after this function.
    """
    if strategy not in STYLE_STRATEGIES:
        raise ValueError(f"未知 style strategy: {strategy}")
    if not isinstance(seed, int):
        raise TypeError("style seed 必须是整数")
    if not isinstance(speed, int) or speed not in (1, 2, 3):
        raise ValueError("style speed 必须是 1、2 或 3")
    if not scene:
        raise ValueError("style scene 不能为空")
    if not isinstance(difficulty, int) or difficulty < 1:
        raise ValueError("style difficulty 必须是正整数")
    if not math.isfinite(bpm) or bpm < 0:
        raise ValueError("style bpm 必须是非负有限数字")
    if not math.isfinite(beat_offset) or beat_offset < 0:
        raise ValueError("style beat_offset 必须是非负有限数字")

    if not enabled:
        output = [dict(event) for event in events]
        return output, {
            "enabled": False,
            "strategy": None,
            "seed": seed,
            "scene": scene,
            "speed": speed,
            "difficulty": difficulty,
            "optimized_monster_events": 0,
            "changed_ibms_ids": 0,
            "arrangement": None,
            **_transition_report(output),
        }

    ordered = sorted(enumerate(events), key=lambda pair: (
        float(pair[1].get("time", 0.0)), int(pair[1].get("lane", -1)), pair[0]))
    for original_index, event in ordered:
        time = float(event.get("time", 0.0))
        if not math.isfinite(time):
            raise ValueError(f"events[{original_index}] 的 time 不是有限数字")

    rng = random.Random(seed)
    candidates_by_lane = {
        lane: _catalog_candidates(catalog, semantic_type=1, lane=lane,
                                  scene=scene, speed=speed)
        for lane in (0, 1)
    }
    common_available = sorted(set.intersection(*(
        set(candidates).intersection(SAFE_SINGLE_MONSTER_IBMS)
        for candidates in candidates_by_lane.values())))
    patterned_report = None
    patterned_assignments = {}
    if strategy == "patterned" and common_available:
        normal_entries = [
            (index, event) for index, event in ordered
            if int(event.get("semantic_type", -1)) == 1
            and int(event.get("lane", -1)) in (0, 1)
            and str(event.get("ibms_id", "")).upper() not in BOSS_PLAY_IDS
        ]
        patterned_assignments, patterned_report = _patterned_assignments(
            normal_entries, available=common_available, difficulty=difficulty,
            bpm=bpm, beat_offset=beat_offset, rng=rng)
    last_by_lane: dict[int, str] = {}
    styled: list[dict | None] = [None] * len(events)
    counts = Counter()
    fallbacks = Counter()
    changed = 0
    for original_index, source in ordered:
        event = dict(source)
        semantic_type = int(event.get("semantic_type", -1))
        lane = int(event.get("lane", -1))
        source_id = str(event.get("ibms_id", "")).upper()
        if (semantic_type != 1 or lane not in (0, 1)
                or source_id in BOSS_PLAY_IDS):
            styled[original_index] = event
            continue
        candidates = candidates_by_lane[lane]
        available = sorted(SAFE_SINGLE_MONSTER_IBMS.intersection(candidates))
        if not available:
            fallbacks["normal_monster_template_missing"] += 1
            styled[original_index] = event
            continue
        if strategy == "patterned":
            chosen_id = patterned_assignments.get(original_index)
            if chosen_id not in candidates:
                fallbacks["patterned_assignment_missing"] += 1
                chosen_id = _weighted_pick(available, _weights_for_difficulty(difficulty), rng)
        else:
            chosen_id = _choose_monster_id(
                available, strategy=strategy, rng=rng, previous=last_by_lane.get(lane))
        chosen = candidates[chosen_id]
        if chosen_id != source_id:
            event.setdefault("model_ibms_id", source_id or None)
            changed += 1
        event.update(ibms_id=chosen_id,
                     note_uid=str(chosen.get("note_uid", "")),
                     note_speed=int(chosen.get("note_speed", speed)),
                     scene=scene,
                     note_scene=str(chosen.get("scene", scene)),
                     skin_source=("postprocess_patterned" if strategy == "patterned"
                                  else "postprocess"))
        last_by_lane[lane] = chosen_id
        counts[chosen_id] += 1
        styled[original_index] = event

    output = [event for event in styled if event is not None]
    transition_report = _transition_report(output)
    report = {
        "enabled": True,
        "strategy": strategy,
        "seed": seed,
        "scene": scene,
        "speed": speed,
        "difficulty": difficulty,
        "bpm": bpm,
        "beat_offset_seconds": beat_offset,
        "safe_single_monster_ids": sorted(SAFE_SINGLE_MONSTER_IBMS),
        "optimized_monster_events": int(sum(counts.values())),
        "changed_ibms_ids": changed,
        "ibms_counts": dict(sorted(counts.items())),
        "fallbacks": dict(sorted(fallbacks.items())),
        "arrangement": patterned_report,
        **transition_report,
    }
    return output, report
