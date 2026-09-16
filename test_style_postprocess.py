from __future__ import annotations

import struct
import unittest
import zlib
from pathlib import Path

from generate_mdm import optimization_outputs
from musechart_runtime.mdm import _generated_cover_bytes
from musechart_runtime.postprocessing import (
    apply_optimizations,
    normalize_double_ids,
    optimize_boss_sequence,
    order_events,
    repair_obvious_double_errors,
)


def event(time: float, lane: int, semantic_type: int, ibms_id: str,
          *, confidence: float = 0.8, ibms_confidence: float = 0.8,
          boss_state: int | None = None, duration: float = 0.0) -> dict:
    value = {
        "time": time,
        "tick": time,
        "lane": lane,
        "semantic_type": semantic_type,
        "ibms_id": ibms_id,
        "confidence": confidence,
        "ibms_confidence": ibms_confidence,
        "duration": duration,
        "is_double": False,
    }
    if boss_state is not None:
        value["boss_state"] = boss_state
    return value


class PostprocessTests(unittest.TestCase):
    def test_event_ordering_repairs_only_local_a_b_a_outlier(self):
        events = [
            event(0.0, 0, 1, "01"),
            event(0.25, 0, 1, "02"),
            event(0.50, 0, 1, "01"),
            event(3.00, 0, 1, "03"),
        ]
        ordered, report = order_events(events, bpm=120.0)
        self.assertEqual([item["ibms_id"] for item in ordered],
                         ["01", "01", "01", "03"])
        self.assertEqual(ordered[1]["model_ibms_id"], "02")
        self.assertEqual(report["isolated_outliers_rewritten"], 1)
        self.assertEqual(report["rewritten_objects"], 1)

    def test_event_ordering_repairs_solo_0e_but_preserves_double_0e(self):
        events = [
            event(0.0, 0, 1, "04"),
            event(0.25, 0, 1, "0E"),
            event(0.50, 0, 1, "04"),
            event(1.00, 0, 1, "0E"),
            event(1.00, 1, 1, "0E"),
        ]
        ordered, report = order_events(events, bpm=120.0)
        ids_at = {
            (item["time"], item["lane"]): item["ibms_id"] for item in ordered
        }
        self.assertEqual(ids_at[(0.25, 0)], "04")
        self.assertEqual(ids_at[(1.0, 0)], "0E")
        self.assertEqual(ids_at[(1.0, 1)], "0E")
        self.assertEqual(report["orphan_0e_rewritten"], 1)

    def test_double_preference_repairs_ordinary_and_mixed_boss_pairs(self):
        events = [
            event(1.0, 0, 1, "0E"),
            event(1.0, 1, 1, "02"),
            event(2.0, 0, 1, "13", boss_state=2),
            event(2.0, 1, 1, "03", boss_state=2),
            event(3.0, 0, 1, "14", ibms_confidence=0.6, boss_state=3),
            event(3.0, 1, 1, "15", ibms_confidence=0.9, boss_state=3),
        ]
        optimized, report = normalize_double_ids(events)
        by_time = {}
        for item in optimized:
            by_time.setdefault(item["time"], []).append(item["ibms_id"])
        self.assertEqual(by_time[1.0], ["0E", "0E"])
        self.assertEqual(by_time[2.0], ["13", "13"])
        self.assertEqual(by_time[3.0], ["15", "15"])
        self.assertEqual(report["ordinary_double_onsets"], 1)
        self.assertEqual(report["boss_double_onsets"], 2)

    def test_obvious_repair_removes_one_gear_heart_and_note(self):
        events = []
        for time, semantic_type, ibms_id in (
                (1.0, 2, "0H"), (2.0, 6, "22"), (3.0, 7, "23")):
            events.extend([
                event(time, 0, semantic_type, ibms_id,
                      ibms_confidence=0.6),
                event(time, 1, semantic_type, ibms_id,
                      ibms_confidence=0.9),
            ])
        repaired, report = repair_obvious_double_errors(events)
        self.assertEqual(len(repaired), 3)
        self.assertTrue(all(item["lane"] == 1 for item in repaired))
        self.assertTrue(all(not item["is_double"] for item in repaired))
        self.assertEqual(report["repaired_onsets"], 3)
        self.assertEqual(report["removed_semantic_types"],
                         {"2": 1, "6": 1, "7": 1})

    def test_obvious_repair_removes_sustained_event_conflicts(self):
        events = [
            event(1.0, 1, 3, "0F", duration=2.0),
            event(1.5, 0, 8, "0G", duration=1.0),
            event(1.6, 0, 8, "16", duration=1.0),
            # The monster is retained because the containing mash is itself
            # removed in favour of the already-active hold.
            event(1.75, 0, 1, "01"),
            event(4.0, 1, 8, "0G", duration=1.0),
            event(4.25, 0, 1, "02"),
            event(4.50, 1, 1, "0E"),
            # Boss attacks and exact end-boundary events are not intrusions.
            event(4.75, 0, 1, "13"),
            event(5.0, 0, 1, "03"),
        ]
        repaired, report = repair_obvious_double_errors(events)
        by_time = {item["time"]: item["ibms_id"] for item in repaired}
        self.assertNotIn(1.5, by_time)
        self.assertEqual(by_time[1.6], "16")
        self.assertIn(1.75, by_time)
        self.assertNotIn(4.25, by_time)
        self.assertNotIn(4.50, by_time)
        self.assertEqual(by_time[4.75], "13")
        self.assertEqual(by_time[5.0], "03")
        self.assertEqual(report["hold_mash_conflicts"], 1)
        self.assertEqual(report["mash_monster_conflicts"], 2)
        self.assertEqual(report["duration_conflicts_repaired"], 3)

    def test_obvious_repair_removes_only_closed_idle_boss_visits(self):
        events = [
            event(1.0, -1, 0, "1A"),
            event(2.0, 0, 1, "01"),
            event(3.0, -1, 0, "1B"),
            event(5.0, -1, 0, "1C"),
            event(5.5, 0, 1, "13"),
            event(6.0, -1, 0, "1B"),
            # An unclosed entrance is retained conservatively.
            event(8.0, -1, 0, "1A"),
        ]
        repaired, report = repair_obvious_double_errors(events)
        controls = [(item["time"], item["ibms_id"]) for item in repaired
                    if item["semantic_type"] == 0]
        self.assertEqual(controls, [(5.0, "1C"), (6.0, "1B"), (8.0, "1A")])
        self.assertEqual(report["idle_boss_visits_removed"], 1)
        self.assertEqual(report["idle_boss_controls_removed"], 2)
        self.assertEqual(report["idle_boss_visit_details"][0]["control_ids"],
                         ["1A", "1B"])

    def test_three_repairs_are_independently_switchable(self):
        events = [
            event(0.0, 0, 1, "01"),
            event(0.25, 0, 1, "02"),
            event(0.50, 0, 1, "01"),
            event(1.0, 0, 2, "0H", ibms_confidence=0.9),
            event(1.0, 1, 2, "0H", ibms_confidence=0.8),
            event(2.0, 0, 1, "03"),
            event(2.0, 1, 1, "04"),
        ]
        result, report = apply_optimizations(
            events, event_ordering=False, double_optimization=False,
            obvious_error_repair=True)
        ids = [item["ibms_id"] for item in result]
        self.assertIn("02", ids)
        self.assertEqual(ids.count("0H"), 1)
        self.assertNotIn("0E", ids)
        self.assertFalse(report["event_ordering"]["enabled"])
        self.assertFalse(report["double_optimization"]["enabled"])
        self.assertTrue(report["obvious_error_repair"]["enabled"])

    def test_v32_direct_boss_transitions_are_preserved(self):
        events = [
            event(1.0, -1, 0, "1C"),  # off -> phase1 directly
            event(1.5, 0, 1, "13", boss_state=2),
            event(2.0, -1, 0, "1G"),  # phase1 -> phase2
            event(2.5, 0, 1, "14", boss_state=3),
            event(3.0, -1, 0, "1B"),  # phase2 -> off directly
        ]
        optimized, report = optimize_boss_sequence(events)
        controls = [item["ibms_id"] for item in optimized
                    if item["semantic_type"] == 0]
        self.assertEqual(controls, ["1C", "1G", "1B"])
        self.assertEqual(report["retained_boss_controls"], 3)
        self.assertEqual(report["dropped_boss_controls"], {})
        self.assertEqual(report["final_boss_state"], "off")

    def test_optimization_matrix_has_all_eight_combinations(self):
        variants = optimization_outputs(
            Path("song.mdm"), matrix=True, event_ordering=True,
            double_optimization=True, obvious_error_repair=True)
        self.assertEqual(
            [(item[1], item[2], item[3], item[4]) for item in variants],
            [
                ("opt_none", False, False, False),
                ("opt_ordering", True, False, False),
                ("opt_double", False, True, False),
                ("opt_obvious", False, False, True),
                ("opt_ordering_double", True, True, False),
                ("opt_ordering_obvious", True, False, True),
                ("opt_double_obvious", False, True, True),
                ("opt_all", True, True, True),
            ])

    def test_generated_cover_is_a_circular_rgba_png(self):
        content = _generated_cover_bytes("title\nartist", size=16)
        self.assertTrue(content.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertGreater(len(content), 60)
        offset = 8
        idat = bytearray()
        header = None
        while offset < len(content):
            length = struct.unpack(">I", content[offset:offset + 4])[0]
            kind = content[offset + 4:offset + 8]
            payload = content[offset + 8:offset + 8 + length]
            if kind == b"IHDR":
                header = struct.unpack(">IIBBBBB", payload)
            elif kind == b"IDAT":
                idat.extend(payload)
            offset += 12 + length
        self.assertEqual(header, (16, 16, 8, 6, 0, 0, 0))
        rows = zlib.decompress(bytes(idat))
        stride = 1 + 16 * 4
        self.assertTrue(all(rows[row * stride] == 0 for row in range(16)))
        alpha = lambda x, y: rows[y * stride + 1 + x * 4 + 3]
        self.assertEqual(alpha(0, 0), 0)
        self.assertEqual(alpha(15, 0), 0)
        self.assertEqual(alpha(0, 15), 0)
        self.assertEqual(alpha(15, 15), 0)
        self.assertEqual(alpha(8, 8), 255)


if __name__ == "__main__":
    unittest.main()
