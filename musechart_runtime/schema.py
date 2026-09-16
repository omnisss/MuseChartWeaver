"""Versioned gameplay contract shared by preprocessing, training and inference."""
from __future__ import annotations

import hashlib
import json


SCHEMA_VERSION = 32
TRAINER_VERSION = "3.2"
MODEL_KIND = "ibms_structured_v3_2"
SEMANTIC_TYPES = tuple(range(1, 9))
# Type 8 is a sustained/multi-hit event, not necessarily an ordinary hold.
DURATION_TYPES = (3, 8)
LANE_PATTERNS = ((0,), (1,), (0, 1))
SEMANTIC_IBMS = {
    1: ("01", "02", "03", "04", "05", "06", "07", "08", "09",
        "0A", "0B", "0C", "0D", "0E", "13", "14", "15"),
    2: ("0H", "18"), 3: ("0F",), 4: ("21",), 5: ("11", "12"),
    6: ("22",), 7: ("23",), 8: ("0G", "16", "17"),
}
IBMS_IDS = tuple(sorted(code for codes in SEMANTIC_IBMS.values() for code in codes))
IBMS_TO_SEMANTIC = {code: typ for typ, codes in SEMANTIC_IBMS.items() for code in codes}
BOSS_CONTROL_IDS = ("1A", "1B", "1C", "1D", "1E", "1F", "1G", "1H")
BOSS_PLAY_IDS = frozenset(("11", "12", "13", "14", "15", "16", "17", "18"))
GROUND_ONLY_IDS = frozenset(("11", "12", "16", "17"))

# Zero denotes an empty lane, not a ninth semantic class. Order is a contract.
EVENT_BUNDLES = tuple([(typ, 0) for typ in SEMANTIC_TYPES]
                      + [(0, typ) for typ in SEMANTIC_TYPES]
                      + [(left, right) for left in SEMANTIC_TYPES
                         for right in SEMANTIC_TYPES])
BOSS_STATES = ("off", "idle", "phase1", "phase2")
CONTROL_STATES = {
    "1A": (0, 1), "1B": (1, 0), "1C": (1, 2), "1D": (2, 1),
    "1E": (1, 3), "1F": (3, 1), "1G": (2, 3), "1H": (3, 2),
}
STATE_CONTROLS = {value: code for code, value in CONTROL_STATES.items()}
# CustomAlbums also permits direct ready/out transitions.
STATE_CONTROLS.update({(0, 2): "1C", (0, 3): "1E", (2, 0): "1B", (3, 0): "1B"})
CONTROL_SOURCES = {
    code: tuple(left for (left, _), edge_code in STATE_CONTROLS.items()
                if edge_code == code)
    for code in BOSS_CONTROL_IDS
}
BOSS_ALLOWED = {
    "11": (1,), "12": (1,), "13": (2,), "14": (3,), "15": (3,),
    "16": (1,), "17": (1,), "18": (2, 3),
}


def require_v3(value: dict, name: str = "数据") -> None:
    if value.get("schema_version") != SCHEMA_VERSION or value.get("model_kind") != MODEL_KIND:
        raise ValueError(
            f"{name} 不是 v3.2 (schema_version=32, model_kind={MODEL_KIND})；"
            "旧 v3.1 checkpoint 不能用于本推理器"
        )
    vocabulary = value.get("vocabulary", value)
    if vocabulary.get("semantic_types") != list(SEMANTIC_TYPES):
        raise ValueError(f"{name} 的 semantic_types 与 v3.2 模型不一致")
    if (vocabulary.get("ibms_ids") != list(IBMS_IDS)
            or vocabulary.get("boss_control_ids") != list(BOSS_CONTROL_IDS)):
        raise ValueError(f"{name} 的 IBMS/Boss 白名单或顺序与 v3.2 不一致")
    lane_counts = vocabulary.get("lane_pattern_train_counts")
    if (not isinstance(lane_counts, list) or len(lane_counts) != 3
            or any(int(count) <= 0 for count in lane_counts)):
        raise ValueError(f"{name} 缺少单轨/双轨训练统计")
    if vocabulary.get("event_bundles") != [list(value) for value in EVENT_BUNDLES]:
        raise ValueError(f"{name} 的 v3.2 联合事件词表不匹配")
    if vocabulary.get("boss_states") != list(BOSS_STATES):
        raise ValueError(f"{name} 的 Boss 状态词表不匹配")
    if len(vocabulary.get("bundle_train_counts", [])) != len(EVENT_BUNDLES):
        raise ValueError(f"{name} 缺少 bundle 训练频次")
    pairs = vocabulary.get("pair_train_counts", [])
    if len(pairs) != len(IBMS_IDS) or any(len(row) != len(IBMS_IDS) for row in pairs):
        raise ValueError(f"{name} 缺少 IBMS pair 训练频次")


require_v32 = require_v3


def fingerprint(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
