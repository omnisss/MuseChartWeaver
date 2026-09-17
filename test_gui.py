from __future__ import annotations

import shutil
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from gui import (
    ALBUM_DISC_PATH,
    ASSET_DIR,
    DIFFICULTY_STAR_PATHS,
    LEVEL_FONT_PATH,
    LEVEL_STAR_PATH,
    PRODUCT_VERSION,
    SCENE_OPTIONS,
    UI_EVENT_PREFIX,
    GenerationOptions,
    _compose_difficulty_star,
    build_generate_command,
    compose_song_preview,
    compose_output_path,
    parse_ui_event_line,
    preferred_window_size,
    scene_id_from_choice,
)
from musechart_runtime.mdm import (
    _prepare_cover,
    find_default_cover,
    inspect_embedded_cover,
)


class GuiCommandTests(unittest.TestCase):
    def test_ui_assets_are_self_contained_in_inference(self):
        expected = Path(__file__).resolve().parent / "assets"
        self.assertEqual(ASSET_DIR, expected)
        for path in (
            ALBUM_DISC_PATH,
            LEVEL_FONT_PATH,
            *DIFFICULTY_STAR_PATHS.values(),
        ):
            self.assertTrue(path.is_file(), path)
            self.assertTrue(path.is_relative_to(expected), path)

    def test_preview_difficulty_stars_share_geometry_and_anchor(self):
        badges = [_compose_difficulty_star(difficulty, 82)
                  for difficulty in (1, 2, 3)]
        alpha_bounds = [badge.getchannel("A").getbbox()
                        for badge, _ in badges]
        centers = [center for _, center in badges]
        self.assertEqual(alpha_bounds[1:], alpha_bounds[:-1])
        self.assertEqual(centers[1:], centers[:-1])
        self.assertEqual(centers[0], (41.0, 41.0))

    def test_window_size_keeps_720p_controls_on_screen(self):
        self.assertEqual(preferred_window_size(1280, 720), (1240, 650))
        self.assertEqual(preferred_window_size(1366, 768), (1326, 698))
        self.assertEqual(preferred_window_size(1920, 1080), (1440, 900))

    def test_song_preview_uses_game_album_and_font_assets(self):
        self.assertEqual(PRODUCT_VERSION, "1.0")
        self.assertTrue(ALBUM_DISC_PATH.is_file())
        self.assertTrue(LEVEL_FONT_PATH.is_file())
        self.assertEqual(LEVEL_STAR_PATH.parent, ASSET_DIR / "chart")
        self.assertTrue(all(path.is_file()
                            for path in DIFFICULTY_STAR_PATHS.values()))
        self.assertTrue(all(path.parent == ASSET_DIR / "chart"
                            for path in DIFFICULTY_STAR_PATHS.values()))
        preview = compose_song_preview(
            background=SCENE_OPTIONS[2].background,
            cover=None,
            title="Link Up",
            artist="MuseChartWeaver",
            difficulty=3,
            play_level="9",
            size=(720, 430),
        )
        self.assertEqual(preview.size, (720, 430))
        self.assertEqual(preview.mode, "RGBA")
        compact = compose_song_preview(
            background=SCENE_OPTIONS[0].background,
            cover=None,
            title="Compact",
            artist="720p",
            difficulty=1,
            play_level="6",
            size=(520, 180),
        )
        self.assertEqual(compact.size, (520, 180))

    def test_structured_ui_progress_ignores_raw_console_output(self):
        self.assertIsNone(parse_ui_event_line("[1/5] 加载模型"))
        event = parse_ui_event_line(
            UI_EVENT_PREFIX
            + '{"kind":"progress","step":3,"percent":52.5,'
              '"stage":"模型推理","message":"已分析 2/4 个音频分块"}'
        )
        self.assertIsNotNone(event)
        self.assertEqual(event["step"], 3)
        self.assertEqual(event["percent"], 52.5)
        self.assertIsNone(parse_ui_event_line(UI_EVENT_PREFIX + "{broken"))

    def test_scene_choices_are_named_and_have_preview_images(self):
        scene_ids = {option.scene_id for option in SCENE_OPTIONS}
        self.assertEqual(
            scene_ids,
            {*(f"scene_{number:02d}" for number in range(1, 11)), "scene_12"},
        )
        self.assertNotIn("scene_11", scene_ids)
        self.assertNotIn("scene_13", scene_ids)
        for option in SCENE_OPTIONS:
            self.assertEqual(scene_id_from_choice(option.label), option.scene_id)
            self.assertTrue(option.background.is_file(), option.background)
        with self.assertRaisesRegex(ValueError, "普通场景"):
            scene_id_from_choice("scene_13")

    def test_output_name_appends_selected_map_number(self):
        directory = Path("output")
        self.assertEqual(
            compose_output_path(directory, "my_chart", 1).name,
            "my_chart_map1.mdm")
        self.assertEqual(
            compose_output_path(directory, "my_chart_map2.mdm", 3).name,
            "my_chart_map3.mdm")
        with self.assertRaisesRegex(ValueError, "1、2 或 3"):
            compose_output_path(directory, "my_chart", 4)
        with self.assertRaisesRegex(ValueError, "非法字符"):
            compose_output_path(directory, "folder/chart", 2)

    def options(self) -> GenerationOptions:
        return GenerationOptions(
            audio=Path("song.mp3"), checkpoint=Path("best.pt"),
            output=Path("song_map2.mdm"), difficulty=2, play_level="7",
            profile="balanced", threshold=None, bpm_mode="auto", bpm=None,
            title="", artist="", level_designer="MuseChartWeave",
            scene="scene_01", speed=2, device="auto", cpu_threads=4,
            event_ordering=True, double_optimization=True,
            obvious_error_repair=True, optimization_matrix=False,
            boss_control_min_gap=0.3, include_cover=True, cover=None,
            make_demo=True, demo_start=0.0, demo_seconds=15.0,
            keep_sidecars=True, strict=False,
        )

    def test_builds_default_local_command_with_three_repairs(self):
        command = build_generate_command(
            self.options(), python="python", script=Path("generate_mdm.py"))
        self.assertEqual(command[:4],
                         ["python", "-X", "utf8", "generate_mdm.py"])
        self.assertIn("--event-ordering", command)
        self.assertIn("--double-optimization", command)
        self.assertIn("--obvious-error-repair", command)
        self.assertNotIn("--bpm", command)
        self.assertNotIn("--cover", command)

    def test_builds_manual_and_disabled_options(self):
        options = replace(
            self.options(), threshold=0.4, bpm_mode="manual", bpm=173.0,
            event_ordering=False, double_optimization=False,
            obvious_error_repair=False, optimization_matrix=True,
            cover=Path("cover.jpg"), make_demo=False, keep_sidecars=False,
            strict=True)
        command = build_generate_command(
            options, python="python", script=Path("generate_mdm.py"))
        for flag in ("--no-event-ordering", "--no-double-optimization",
                     "--no-obvious-error-repair", "--optimization-matrix",
                     "--cover", "--no-demo", "--no-sidecars", "--strict"):
            self.assertIn(flag, command)
        self.assertEqual(command[command.index("--bpm") + 1], "173")
        self.assertEqual(command[command.index("--threshold") + 1], "0.4")

    def test_image_directory_has_a_deterministic_fallback(self):
        fallback = find_default_cover(Path(__file__).resolve().parent / "image")
        self.assertIsNotNone(fallback)
        self.assertEqual(fallback.stem, "爱咕TV")
        self.assertIn(fallback.suffix.lower(), {".jpg", ".jpeg", ".png", ".webp"})

    @unittest.skipUnless(shutil.which("ffprobe"), "需要 ffprobe")
    def test_embedded_cover_probe_distinguishes_sample_audio(self):
        root = Path(__file__).resolve().parent / "test"
        plain = root / "Bittersweet.mp3"
        embedded = root / "no title.mp3"
        if not plain.is_file() or not embedded.is_file():
            self.skipTest("缺少手工音频测试样本")
        self.assertFalse(inspect_embedded_cover(plain)["present"])
        self.assertTrue(inspect_embedded_cover(embedded)["present"])

    def test_cover_priority_explicit_embedded_image_then_generated(self):
        root = Path(__file__).resolve().parent
        fallback = find_default_cover(root / "image")
        temporary = root / "output" / ".cover-priority-test"
        common = {"include_cover": True, "seed": "test"}
        with (mock.patch("musechart_runtime.mdm._convert_cover"),
              mock.patch("musechart_runtime.mdm._try_extract_embedded_cover",
                         return_value=False)):
            _, source, _, _ = _prepare_cover(
                root / "no title.mp3", fallback, temporary / "explicit",
                fallback_cover=fallback, **common)
            self.assertEqual(source, "command_line")
        with mock.patch("musechart_runtime.mdm._try_extract_embedded_cover",
                        return_value=True):
            _, source, _, _ = _prepare_cover(
                root / "no title.mp3", None, temporary / "embedded",
                fallback_cover=fallback, **common)
            self.assertEqual(source, "audio_embedded")
        with (mock.patch("musechart_runtime.mdm._convert_cover"),
              mock.patch("musechart_runtime.mdm._try_extract_embedded_cover",
                         return_value=False)):
            _, source, _, _ = _prepare_cover(
                root / "Bittersweet.mp3", None, temporary / "image",
                fallback_cover=fallback, **common)
            self.assertEqual(source, "image_fallback")
        with (mock.patch("musechart_runtime.mdm._write_generated_cover"),
              mock.patch("musechart_runtime.mdm._try_extract_embedded_cover",
                         return_value=False)):
            _, source, _, _ = _prepare_cover(
                root / "Bittersweet.mp3", None, temporary / "generated",
                fallback_cover=None, **common)
            self.assertEqual(source, "generated_fallback")


if __name__ == "__main__":
    unittest.main()
