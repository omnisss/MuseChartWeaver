"""Conservative, independently switchable v1.0 chart repairs."""
from __future__ import annotations

from collections import Counter, defaultdict

from .schema import (BOSS_CONTROL_IDS, BOSS_PLAY_IDS, STATE_CONTROLS)


NORMAL_MONSTER_IDS = frozenset((
    "01", "02", "03", "04", "05", "06", "07",
    "08", "09", "0A", "0B", "0C", "0D",
))
BOSS_DOUBLE_ATTACK_IDS = frozenset(("13", "14", "15"))
OBVIOUS_SINGLETON_TYPES = frozenset((2, 6, 7))  # gear, heart, music note
PICKUP_TYPES = frozenset((6, 7))  # heart, music note


def _copy_sorted(events: list[dict]) -> list[dict]:
    return sorted((dict(event) for event in events),
                  key=lambda event: (float(event["time"]), int(event["lane"])))


def _gameplay_groups(events: list[dict]) -> dict[float, list[dict]]:
    groups: dict[float, list[dict]] = defaultdict(list)
    for event in events:
        if int(event.get("semantic_type", -1)) != 0:
            groups[round(float(event["time"]), 6)].append(event)
    return groups


def _clear_template_fields(event: dict) -> None:
    for key in ("note_uid", "prefab_name", "note_speed", "note_scene", "skin_source"):
        event.pop(key, None)


def _rewrite_id(event: dict, ibms_id: str, source: str) -> bool:
    original = str(event.get("ibms_id", "")).upper()
    if original == ibms_id:
        return False
    event.setdefault("model_ibms_id", original or None)
    event["ibms_id"] = ibms_id
    event["ibms_source"] = source
    _clear_template_fields(event)
    return True


def _event_rank(event: dict) -> tuple[float, float, int]:
    """Deterministic confidence rank; ties keep ground lane 0."""
    return (float(event.get("ibms_confidence", -1.0)),
            float(event.get("confidence", 0.0)),
            -int(event.get("lane", 0)))


def _state_after_gameplay(state: int, code: str) -> int:
    if code == "13":
        return 2
    if code in ("14", "15"):
        return 3
    if code in ("11", "12", "16"):
        return 1
    if code == "17":
        return 0
    return state


def optimize_boss_sequence(
    events: list[dict], *, enabled: bool = True,
    minimum_control_gap: float = 0.30,
) -> tuple[list[dict], dict]:
    """Conservatively filter impossible controls using the v1.0 state graph."""
    if minimum_control_gap < 0:
        raise ValueError("minimum_control_gap 必须大于等于 0")
    ordered = _copy_sorted(events)
    input_controls = sum(
        int(event.get("semantic_type", -1)) == 0 for event in ordered)
    if not enabled:
        return ordered, {
            "enabled": False,
            "minimum_control_gap_seconds": minimum_control_gap,
            "input_boss_controls": input_controls,
            "retained_boss_controls": input_controls,
            "dropped_boss_controls": {},
            "dropped_reasons": {},
            "same_time_control_conflicts": 0,
            "final_boss_state": None,
            "animation_timing_validated": False,
        }

    edges_by_code: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for edge, code in STATE_CONTROLS.items():
        edges_by_code[code].append(edge)
    groups: dict[float, list[dict]] = defaultdict(list)
    for event in ordered:
        groups[round(float(event["time"]), 6)].append(event)

    output = []
    dropped_codes: Counter[str] = Counter()
    dropped_reasons: Counter[str] = Counter()
    state = 0
    last_control_time: float | None = None
    retained_controls = 0
    conflict_groups = 0
    for time, group in sorted(groups.items()):
        controls = [
            event for event in group
            if int(event.get("semantic_type", -1)) == 0
            and str(event.get("ibms_id", "")).upper() in BOSS_CONTROL_IDS
        ]
        gameplay = [event for event in group if event not in controls]
        if len(controls) > 1:
            conflict_groups += 1

        eligible: list[tuple[dict, int]] = []
        for event in controls:
            code = str(event["ibms_id"]).upper()
            matching = [right for left, right in edges_by_code[code] if left == state]
            if not matching:
                dropped_codes[code] += 1
                dropped_reasons["state_mismatch"] += 1
                continue
            if (last_control_time is not None
                    and time - last_control_time < minimum_control_gap - 1e-8):
                dropped_codes[code] += 1
                dropped_reasons["minimum_control_gap"] += 1
                continue
            eligible.append((event, matching[0]))

        chosen = max(eligible, key=lambda item: _event_rank(item[0])) if eligible else None
        for event, _ in eligible:
            if chosen is not None and event is chosen[0]:
                continue
            dropped_codes[str(event["ibms_id"]).upper()] += 1
            dropped_reasons["same_time_conflict"] += 1
        if chosen is not None:
            event, state = chosen
            last_control_time = time
            retained_controls += 1
            output.append(event)

        for event in sorted(gameplay, key=lambda value: int(value["lane"])):
            state = _state_after_gameplay(
                state, str(event.get("ibms_id", "")).upper())
            output.append(event)

    return _copy_sorted(output), {
        "enabled": True,
        "minimum_control_gap_seconds": minimum_control_gap,
        "input_boss_controls": input_controls,
        "retained_boss_controls": retained_controls,
        "dropped_boss_controls": dict(sorted(dropped_codes.items())),
        "dropped_reasons": dict(sorted(dropped_reasons.items())),
        "same_time_control_conflicts": conflict_groups,
        "final_boss_state": ("off", "idle", "phase1", "phase2")[state],
        "animation_timing_validated": False,
    }


def _neighbor_limit(bpm: float) -> float:
    if bpm > 0:
        return max(0.35, min(1.5, 120.0 / bpm))
    return 1.0


def _orphan_replacement(event: dict, candidates: list[dict], limit: float) -> str:
    time = float(event["time"])
    before = [value for value in candidates if float(value["time"]) < time
              and time - float(value["time"]) <= limit]
    after = [value for value in candidates if float(value["time"]) > time
             and float(value["time"]) - time <= limit]
    left = max(before, key=lambda value: float(value["time"]), default=None)
    right = min(after, key=lambda value: float(value["time"]), default=None)
    if left is not None and right is not None:
        left_id = str(left["ibms_id"]).upper()
        right_id = str(right["ibms_id"]).upper()
        if left_id == right_id:
            return left_id
        left_distance = time - float(left["time"])
        right_distance = float(right["time"]) - time
        if abs(left_distance - right_distance) <= 1e-8:
            return str(max((left, right), key=_event_rank)["ibms_id"]).upper()
        return left_id if left_distance < right_distance else right_id
    neighbor = left or right
    return str(neighbor["ibms_id"]).upper() if neighbor is not None else "01"


def order_events(
    events: list[dict], *, enabled: bool = True, bpm: float = 0.0,
    minimum_boss_control_gap: float = 0.30,
) -> tuple[list[dict], dict]:
    """Repair isolated ordinary-monster choices without restyling a whole song."""
    ordered, boss_report = optimize_boss_sequence(
        events, enabled=enabled, minimum_control_gap=minimum_boss_control_gap)
    if not enabled:
        return ordered, {
            "enabled": False,
            "neighbor_limit_seconds": None,
            "isolated_outliers_rewritten": 0,
            "orphan_0e_rewritten": 0,
            "rewritten_objects": 0,
            "boss_sequence": boss_report,
        }

    groups = _gameplay_groups(ordered)
    limit = _neighbor_limit(bpm)
    safe_by_lane = {
        lane: sorted([
            event for event in ordered
            if int(event.get("lane", -1)) == lane
            and int(event.get("semantic_type", -1)) == 1
            and str(event.get("ibms_id", "")).upper() in NORMAL_MONSTER_IDS
        ], key=lambda value: float(value["time"]))
        for lane in (0, 1)
    }

    orphan_count = 0
    for event in ordered:
        if (int(event.get("semantic_type", -1)) != 1
                or str(event.get("ibms_id", "")).upper() != "0E"):
            continue
        group = groups[round(float(event["time"]), 6)]
        full_gemini = (
            len(group) == 2
            and {int(value["lane"]) for value in group} == {0, 1}
            and all(str(value.get("ibms_id", "")).upper() == "0E"
                    and int(value.get("semantic_type", -1)) == 1
                    for value in group)
        )
        # Mixed ordinary doubles belong to the independent user-preference repair.
        if full_gemini or len(group) > 1:
            continue
        replacement = _orphan_replacement(
            event, safe_by_lane[int(event["lane"])], limit)
        orphan_count += int(_rewrite_id(event, replacement, "event_ordering_orphan_0e"))

    # Decisions use one immutable ID snapshot so adjacent corrections cannot cascade.
    groups = _gameplay_groups(ordered)
    outlier_count = 0
    for lane in (0, 1):
        sequence = [
            event for event in ordered
            if int(event.get("lane", -1)) == lane
            and len(groups[round(float(event["time"]), 6)]) == 1
            and int(event.get("semantic_type", -1)) == 1
            and str(event.get("ibms_id", "")).upper() in NORMAL_MONSTER_IDS
        ]
        ids = [str(event["ibms_id"]).upper() for event in sequence]
        decisions = []
        for index in range(1, len(sequence) - 1):
            previous, current, following = (
                sequence[index - 1], sequence[index], sequence[index + 1])
            if (ids[index - 1] == ids[index + 1] != ids[index]
                    and float(current["time"]) - float(previous["time"]) <= limit
                    and float(following["time"]) - float(current["time"]) <= limit):
                decisions.append((current, ids[index - 1]))
        for event, replacement in decisions:
            outlier_count += int(_rewrite_id(
                event, replacement, "event_ordering_isolated_outlier"))

    return _copy_sorted(ordered), {
        "enabled": True,
        "neighbor_limit_seconds": limit,
        "isolated_outliers_rewritten": outlier_count,
        "orphan_0e_rewritten": orphan_count,
        "rewritten_objects": outlier_count + orphan_count,
        "boss_sequence": boss_report,
    }


def _boss_double_target(group: list[dict]) -> str | None:
    states = {int(event.get("boss_state", -1)) for event in group}
    if states == {2}:
        return "13"
    if states == {3}:
        eligible = [event for event in group
                    if str(event.get("ibms_id", "")).upper() in ("14", "15")]
        return (str(max(eligible, key=_event_rank)["ibms_id"]).upper()
                if eligible else "15")
    boss_events = [
        event for event in group
        if str(event.get("ibms_id", "")).upper() in BOSS_DOUBLE_ATTACK_IDS
    ]
    codes = {str(event.get("ibms_id", "")).upper() for event in boss_events}
    if codes == {"13"}:
        return "13"
    if codes and codes <= {"14", "15"}:
        return str(max(boss_events, key=_event_rank)["ibms_id"]).upper()
    if boss_events:
        return str(max(boss_events, key=_event_rank)["ibms_id"]).upper()
    return None


def normalize_double_ids(events: list[dict], enabled: bool = True) -> tuple[list[dict], dict]:
    """Apply the optional classic-double preference to ordinary and Boss attacks."""
    result = _copy_sorted(events)
    groups = _gameplay_groups(result)
    ordinary_onsets = 0
    boss_onsets = 0
    changed = 0
    skipped_boss_pairs = 0
    if enabled:
        for group in groups.values():
            if len(group) != 2 or {int(event["lane"]) for event in group} != {0, 1}:
                continue
            codes = {str(event.get("ibms_id", "")).upper() for event in group}
            ordinary = all(
                int(event.get("semantic_type", -1)) == 1
                and str(event.get("ibms_id", "")).upper() not in BOSS_PLAY_IDS
                for event in group)
            if ordinary:
                ordinary_onsets += 1
                for event in group:
                    changed += int(_rewrite_id(
                        event, "0E", "double_optimization"))
                continue
            boss_attack_pair = (
                all(int(event.get("semantic_type", -1)) == 1 for event in group)
                and bool(codes & BOSS_DOUBLE_ATTACK_IDS)
            )
            if boss_attack_pair:
                target = _boss_double_target(group)
                if target is None:
                    skipped_boss_pairs += 1
                    continue
                boss_onsets += 1
                for event in group:
                    changed += int(_rewrite_id(
                        event, target, "boss_double_optimization"))
    return result, {
        "enabled": enabled,
        "optimized_onsets": ordinary_onsets + boss_onsets,
        "ordinary_double_onsets": ordinary_onsets,
        "boss_double_onsets": boss_onsets,
        "skipped_boss_pairs": skipped_boss_pairs,
        "rewritten_objects": changed,
    }


def repair_obvious_double_errors(
    events: list[dict], enabled: bool = True,
) -> tuple[list[dict], dict]:
    """Repair duplicate/pickup conflicts, sustained conflicts and idle Boss visits."""
    result = _copy_sorted(events)
    if not enabled:
        return result, {
            "enabled": False,
            "repaired_onsets": 0,
            "pickup_double_conflicts": 0,
            "pickup_double_objects_removed": 0,
            "duration_conflicts_repaired": 0,
            "opposite_hold_pickups_removed": 0,
            "hold_mash_conflicts": 0,
            "mash_monster_conflicts": 0,
            "idle_boss_visits_removed": 0,
            "idle_boss_controls_removed": 0,
            "idle_boss_visit_details": [],
            "removed_objects": 0,
            "removed_semantic_types": {},
            "removed_ibms_ids": {},
        }
    groups = _gameplay_groups(result)
    removed_ids = set()
    removed_types: Counter[int] = Counter()
    removed_ibms: Counter[str] = Counter()
    repaired_onsets = 0
    pickup_double_conflicts = 0
    pickup_double_objects_removed = 0

    def _remove_event(event: dict) -> bool:
        identity = id(event)
        if identity in removed_ids:
            return False
        removed_ids.add(identity)
        removed_types[int(event.get("semantic_type", -1))] += 1
        removed_ibms[str(event.get("ibms_id", "")).upper()] += 1
        return True

    for group in groups.values():
        if (len(group) != 2
                or {int(event["lane"]) for event in group} != {0, 1}):
            continue
        semantic_types = {int(event.get("semantic_type", -1)) for event in group}
        pickups = [
            event for event in group
            if int(event.get("semantic_type", -1)) in PICKUP_TYPES
        ]
        if pickups:
            non_pickups = [event for event in group if event not in pickups]
            # A pickup must not act as one half of a double-track obstacle.
            # If both lanes contain pickups, retain only the more confident one
            # so the musical accent is preserved without forcing two inputs.
            removed = pickups if non_pickups else [
                event for event in pickups
                if event is not max(pickups, key=_event_rank)
            ]
            removed_now = sum(int(_remove_event(event)) for event in removed)
            if removed_now:
                keep = non_pickups or [
                    event for event in pickups if id(event) not in removed_ids
                ]
                for event in keep:
                    event["is_double"] = False
                    event["duplicate_repair"] = "removed_conflicting_pickup"
                repaired_onsets += 1
                pickup_double_conflicts += 1
                pickup_double_objects_removed += removed_now
            continue
        if (len(semantic_types) != 1
                or next(iter(semantic_types)) not in OBVIOUS_SINGLETON_TYPES):
            continue
        keep = max(group, key=_event_rank)
        removed_event = next(event for event in group if event is not keep)
        _remove_event(removed_event)
        keep["is_double"] = False
        keep["duplicate_repair"] = "kept_higher_confidence"
        repaired_onsets += 1

    def strictly_inside(owner: dict, time: float) -> bool:
        start = float(owner["time"])
        end = start + max(0.0, float(owner.get("duration", 0.0)))
        return start + 1.0e-6 < time < end - 1.0e-6

    holds = [
        event for event in result
        if id(event) not in removed_ids
        and int(event.get("semantic_type", -1)) == 3
        and float(event.get("duration", 0.0)) > 1.0e-6
    ]
    opposite_hold_pickups_removed = 0
    for pickup in result:
        if (id(pickup) in removed_ids
                or int(pickup.get("semantic_type", -1)) not in PICKUP_TYPES):
            continue
        pickup_time = float(pickup["time"])
        pickup_lane = int(pickup.get("lane", -1))
        if any(int(hold.get("lane", -1)) != pickup_lane
               and strictly_inside(hold, pickup_time) for hold in holds):
            opposite_hold_pickups_removed += int(_remove_event(pickup))

    mashes = [
        event for event in result
        if id(event) not in removed_ids
        and int(event.get("semantic_type", -1)) == 8
        and str(event.get("ibms_id", "")).upper() == "0G"
        and float(event.get("duration", 0.0)) > 1.0e-6
    ]
    hold_mash_conflicts = 0
    surviving_mashes = []
    for mash in mashes:
        if any(strictly_inside(hold, float(mash["time"])) for hold in holds):
            hold_mash_conflicts += int(_remove_event(mash))
        else:
            surviving_mashes.append(mash)

    mash_monster_conflicts = 0
    for event in result:
        if (id(event) in removed_ids
                or int(event.get("semantic_type", -1)) != 1
                or str(event.get("ibms_id", "")).upper() in BOSS_PLAY_IDS):
            continue
        if any(strictly_inside(mash, float(event["time"]))
               for mash in surviving_mashes):
            mash_monster_conflicts += int(_remove_event(event))

    # A visit begins when the Boss leaves the off state and closes when it
    # returns to off. If no Boss gameplay object occurs in the closed span,
    # keeping only its entrance/exit animations creates a visibly idle visit.
    edges_by_code: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for edge, code in STATE_CONTROLS.items():
        edges_by_code[code].append(edge)
    timeline: dict[float, list[dict]] = defaultdict(list)
    for event in result:
        if id(event) not in removed_ids:
            timeline[round(float(event["time"]), 6)].append(event)

    state = 0
    visit_start: float | None = None
    visit_controls: list[dict] = []
    visit_has_action = False
    idle_visit_details = []
    idle_controls_removed = 0
    for time, group in sorted(timeline.items()):
        controls = [
            event for event in group
            if int(event.get("semantic_type", -1)) == 0
            and str(event.get("ibms_id", "")).upper() in BOSS_CONTROL_IDS
        ]
        gameplay = [
            event for event in group
            if str(event.get("ibms_id", "")).upper() in BOSS_PLAY_IDS
        ]
        for event in sorted(controls, key=_event_rank, reverse=True):
            code = str(event["ibms_id"]).upper()
            matching = [right for left, right in edges_by_code[code]
                        if left == state]
            if not matching:
                continue
            previous = state
            state = matching[0]
            if previous == 0 and state != 0 and visit_start is None:
                visit_start = time
                visit_controls = []
                visit_has_action = False
            if visit_start is not None:
                visit_controls.append(event)

        if visit_start is not None and gameplay:
            visit_has_action = True
        for event in gameplay:
            state = _state_after_gameplay(
                state, str(event.get("ibms_id", "")).upper())

        if visit_start is not None and state == 0:
            if not visit_has_action:
                removed_now = sum(int(_remove_event(event)) for event in visit_controls)
                idle_controls_removed += removed_now
                idle_visit_details.append({
                    "start_time": visit_start,
                    "end_time": time,
                    "control_ids": [
                        str(event.get("ibms_id", "")).upper()
                        for event in visit_controls
                    ],
                    "removed_controls": removed_now,
                })
            visit_start = None
            visit_controls = []
            visit_has_action = False

    output = [event for event in result if id(event) not in removed_ids]
    return _copy_sorted(output), {
        "enabled": True,
        "repaired_onsets": repaired_onsets,
        "pickup_double_conflicts": pickup_double_conflicts,
        "pickup_double_objects_removed": pickup_double_objects_removed,
        "duration_conflicts_repaired": (
            opposite_hold_pickups_removed
            + hold_mash_conflicts
            + mash_monster_conflicts),
        "opposite_hold_pickups_removed": opposite_hold_pickups_removed,
        "hold_mash_conflicts": hold_mash_conflicts,
        "mash_monster_conflicts": mash_monster_conflicts,
        "idle_boss_visits_removed": len(idle_visit_details),
        "idle_boss_controls_removed": idle_controls_removed,
        "idle_boss_visit_details": idle_visit_details,
        "removed_objects": len(removed_ids),
        "removed_semantic_types": {
            str(key): value for key, value in sorted(removed_types.items())},
        "removed_ibms_ids": dict(sorted(removed_ibms.items())),
    }


def _refresh_double_flags(events: list[dict]) -> None:
    groups = _gameplay_groups(events)
    for group in groups.values():
        is_double = len(group) == 2 and {int(event["lane"]) for event in group} == {0, 1}
        for event in group:
            event["is_double"] = is_double


def apply_optimizations(
    events: list[dict], *, event_ordering: bool | None = None,
    double_optimization: bool = True, obvious_error_repair: bool = True,
    minimum_boss_control_gap: float = 0.30, bpm: float = 0.0,
    chart_optimization: bool | None = None,
) -> tuple[list[dict], dict]:
    """Apply three independent repairs and return a complete audit report."""
    if event_ordering is None:
        event_ordering = True if chart_optimization is None else chart_optimization
    result, ordering_report = order_events(
        events, enabled=event_ordering, bpm=bpm,
        minimum_boss_control_gap=minimum_boss_control_gap)
    result, double_report = normalize_double_ids(
        result, enabled=double_optimization)
    result, obvious_report = repair_obvious_double_errors(
        result, enabled=obvious_error_repair)
    _refresh_double_flags(result)
    return result, {
        "event_ordering": ordering_report,
        "double_optimization": double_report,
        "obvious_error_repair": obvious_report,
        # Compatibility field for old sidecar readers.
        "chart_optimization": {
            "enabled": event_ordering,
            "boss_sequence": ordering_report["boss_sequence"],
        },
    }
