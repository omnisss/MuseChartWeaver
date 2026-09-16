from __future__ import annotations

import unittest

import torch

from musechart_runtime.decoding import decode_batch
from musechart_runtime.schema import EVENT_BUNDLES, IBMS_IDS


class V32DecoderTests(unittest.TestCase):
    def test_decodes_joint_bundle_and_direct_ibms_pair(self):
        frames = 12
        outputs = {
            "onset_logits": torch.full((1, 1, frames), -10.0),
            "bundle_logits": torch.full(
                (1, len(EVENT_BUNDLES), frames), -10.0),
            "ibms_logits": torch.full((1, 2, len(IBMS_IDS), frames), -10.0),
            "duration_ratio_logits": torch.zeros((1, 2, frames)),
            "duration_log": torch.zeros((1, 2, frames)),
            "pair_context": torch.zeros((1, 16, frames)),
            "pair_embeddings": torch.zeros((len(IBMS_IDS), len(IBMS_IDS), 16)),
            "boss_state_logits": torch.full((1, 4, frames), -10.0),
            "boss_boundary_logits": torch.full((1, 1, frames), -10.0),
            "boss_transition_logits": torch.eye(4) * 10.0,
        }
        outputs["boss_state_logits"][:, 0] = 10.0
        outputs["onset_logits"][0, 0, 3] = 10.0
        outputs["bundle_logits"][0, EVENT_BUNDLES.index((1, 1)), 3] = 10.0
        outputs["ibms_logits"][0, 0, IBMS_IDS.index("01"), 3] = 10.0
        outputs["ibms_logits"][0, 1, IBMS_IDS.index("02"), 3] = 10.0
        events = decode_batch(
            outputs, threshold=0.65, frame_seconds=0.01,
            vocabulary={}, audio_durations=[0.12], initial_off=True)[0]
        gameplay = [item for item in events if item["semantic_type"] != 0]
        self.assertEqual([item["ibms_id"] for item in gameplay], ["01", "02"])
        self.assertTrue(all(item["is_double"] for item in gameplay))
        self.assertTrue(all(item["boss_state"] == 0 for item in gameplay))

    def test_legacy_boss_threshold_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "状态序列"):
            decode_batch({}, threshold=0.65, boss_threshold=0.85,
                         frame_seconds=0.01, vocabulary={})


if __name__ == "__main__":
    unittest.main()
