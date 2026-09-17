#!/usr/bin/env python3
"""Local Tk desktop UI for the MuseChart v1.0 inference pipeline."""
from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps, ImageTk

from musechart_runtime.mdm import find_default_cover, inspect_embedded_cover


SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_DIR = SCRIPT_DIR / "image"
ASSET_DIR = SCRIPT_DIR / "assets"
PRODUCT_VERSION = "1.0"
ALBUM_DISC_PATH = ASSET_DIR / "album" / "vinyl-disc.png"
LEVEL_STAR_PATH = ASSET_DIR / "chart" / "star.png"
LEVEL_FONT_PATH = ASSET_DIR / "fonts" / "source-han-sans-cn-heavy.otf"
DIFFICULTY_COLORS = {
    1: (109, 253, 100, 255),
    2: (0, 220, 212, 255),
    3: (255, 55, 217, 255),
}
DIFFICULTY_STAR_PATHS = {
    1: ASSET_DIR / "chart" / "difficulty-easy.png",
    2: ASSET_DIR / "chart" / "difficulty-hard.png",
    3: ASSET_DIR / "chart" / "difficulty-master.png",
}
# The colored difficulty sprite has wide transparent margins.  Rendering its
# canvas slightly larger than the white base makes the visible colored star
# fill the base while ``star.png`` remains as the surrounding white edge.
DIFFICULTY_LAYER_SCALE = 1.45
AUDIO_TYPES = (
    ("音频文件", "*.mp3 *.wav *.ogg *.flac *.m4a *.aac *.opus"),
    ("所有文件", "*.*"),
)
IMAGE_TYPES = (
    ("图片文件", "*.png *.jpg *.jpeg *.webp *.bmp *.gif"),
    ("所有文件", "*.*"),
)

THEME = {
    "background": "#160E26",
    "surface_dark": "#27184D",
    "surface_raised": "#3C286F",
    "primary": "#FF1981",
    "primary_deep": "#B1005F",
    "cyan": "#00B7EE",
    "cyan_deep": "#0075A9",
    "violet": "#7941E0",
    "violet_deep": "#5015A6",
    "text": "#FFFFFF",
    "text_muted": "#C9BFE0",
    "success": "#6DFD64",
}
UI_EVENT_PREFIX = "@@MUSECHART_UI@@"
PROFILE_LABELS = {
    "casual": "休闲 · 较疏",
    "balanced": "均衡 · 推荐",
    "aggressive": "激进 · 较密",
}
COVER_SOURCE_LABELS = {
    "command_line": "显式封面",
    "audio_embedded": "音频内嵌封面",
    "image_fallback": "image 目录兜底",
    "generated_fallback": "自动生成封面",
    "disabled": "未打包封面",
}
BPM_SOURCE_LABELS = {
    "command_line": "手动设置",
    "audio_metadata": "音频标签",
    "audio_analysis": "自动检测",
    "disabled": "未知 BPM",
}


def preferred_window_size(screen_width: int, screen_height: int) -> tuple[int, int]:
    """Leave OS chrome room while keeping the full two-column UI visible."""
    vertical_margin = 70 if screen_height <= 800 else 40
    return (
        min(1440, max(980, screen_width - 40)),
        min(900, max(620, screen_height - vertical_margin)),
    )


def _preview_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(str(LEVEL_FONT_PATH), size=size)
    except OSError:
        return ImageFont.load_default()


def _fit_preview_font(text: str, maximum_width: int, start_size: int,
                      minimum_size: int = 12
                      ) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    for size in range(start_size, minimum_size - 1, -1):
        font = _preview_font(size)
        box = probe.textbbox((0, 0), text, font=font, stroke_width=2)
        if box[2] - box[0] <= maximum_width:
            return font
    return _preview_font(minimum_size)


def _compose_difficulty_star(
    difficulty: int, size: int,
) -> tuple[Image.Image, tuple[float, float]]:
    """Compose the game's white star base and centered difficulty layer."""
    with Image.open(LEVEL_STAR_PATH) as source:
        base_source = source.convert("RGBA")
    star = base_source.resize((size, size), Image.Resampling.LANCZOS)

    layer_path = DIFFICULTY_STAR_PATHS.get(
        difficulty, DIFFICULTY_STAR_PATHS[3])
    with Image.open(layer_path) as source:
        difficulty_source = source.convert("RGBA")
    layer_extent = max(1, round(size * DIFFICULTY_LAYER_SCALE))
    difficulty_layer = ImageOps.contain(
        difficulty_source, (layer_extent, layer_extent),
        method=Image.Resampling.LANCZOS)
    layer_width, layer_height = difficulty_layer.size
    difficulty_offset_x = 1
    difficulty_offset_y = 1

    star.alpha_composite(
        difficulty_layer,
        (
            (size - layer_width) // 2 + difficulty_offset_x,
            (size - layer_height) // 2 + difficulty_offset_y,
        ),
    )
    return star, (size / 2, size / 2)


def _draw_preview_text(
    canvas: Image.Image, text: str, *, center_x: int, top: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    outline: tuple[int, int, int, int], stroke_width: int,
) -> int:
    """Draw centered white text with a colored edge, glow and soft shadow."""
    measure = ImageDraw.Draw(canvas)
    box = measure.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    width, height = box[2] - box[0], box[3] - box[1]
    x = center_x - width // 2 - box[0]
    y = top - box[1]

    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).text(
        (x + 5, y + 7), text, font=font, fill=(0, 0, 0, 220),
        stroke_width=stroke_width + 1, stroke_fill=(0, 0, 0, 180),
    )
    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(5)))

    glow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(glow).text(
        (x, y), text, font=font, fill=(255, 255, 255, 205),
        stroke_width=stroke_width + 2, stroke_fill=outline,
    )
    canvas.alpha_composite(glow.filter(ImageFilter.GaussianBlur(7)))
    ImageDraw.Draw(canvas).text(
        (x, y), text, font=font, fill=(255, 255, 255, 255),
        stroke_width=stroke_width, stroke_fill=outline,
    )
    return height


def compose_song_preview(
    *, background: Path, cover: Path | None, title: str, artist: str,
    difficulty: int, play_level: str, size: tuple[int, int],
) -> Image.Image:
    """Compose the game-like album preview shown in the right workspace."""
    width, height = size
    if width < 320 or height < 180:
        raise ValueError("预览区域尺寸过小")
    try:
        with Image.open(background) as source:
            backdrop = ImageOps.fit(
                source.convert("RGB"), size, method=Image.Resampling.LANCZOS)
    except OSError:
        backdrop = Image.new("RGB", size, THEME["background"])
    backdrop = backdrop.filter(
        ImageFilter.GaussianBlur(max(5, round(min(size) * 0.018))))
    canvas = backdrop.convert("RGBA")
    canvas.alpha_composite(Image.new("RGBA", size, (20, 8, 42, 112)))

    # The original prefab uses a 520px disc with a centered 440px cover.
    disc_size = min(int(height * 0.60), int(width * 0.46), 330)
    content_height = disc_size + 82
    disc_top = max(8, (height - content_height) // 2 - 2)
    disc_left = (width - disc_size) // 2
    center_x = width // 2

    shadow = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).ellipse(
        (disc_left + 13, disc_top + 17,
         disc_left + disc_size + 13, disc_top + disc_size + 17),
        fill=(0, 0, 0, 205),
    )
    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(15)))
    glow = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse(
        (disc_left - 5, disc_top - 5,
         disc_left + disc_size + 5, disc_top + disc_size + 5),
        outline=(255, 255, 255, 185), width=7,
    )
    canvas.alpha_composite(glow.filter(ImageFilter.GaussianBlur(10)))

    try:
        with Image.open(ALBUM_DISC_PATH) as source:
            disc = source.convert("RGBA").resize(
                (disc_size, disc_size), Image.Resampling.LANCZOS)
    except OSError:
        disc = Image.new("RGBA", (disc_size, disc_size), (13, 12, 18, 255))
        ImageDraw.Draw(disc).ellipse(
            (1, 1, disc_size - 2, disc_size - 2),
            outline=(105, 95, 122, 255), width=3)
    canvas.alpha_composite(disc, (disc_left, disc_top))

    cover_size = round(disc_size * (440 / 520))
    if cover is not None and cover.is_file():
        try:
            with Image.open(cover) as source:
                cover_image = ImageOps.fit(
                    source.convert("RGBA"), (cover_size, cover_size),
                    method=Image.Resampling.LANCZOS)
        except OSError:
            cover_image = Image.new(
                "RGBA", (cover_size, cover_size), (60, 40, 111, 255))
    else:
        cover_image = Image.new(
            "RGBA", (cover_size, cover_size), (60, 40, 111, 255))
        cover_draw = ImageDraw.Draw(cover_image)
        for radius, color in (
                (cover_size // 2, (121, 65, 224, 255)),
                (cover_size // 3, (0, 183, 238, 210)),
                (cover_size // 6, (255, 25, 129, 235))):
            center = cover_size // 2
            cover_draw.ellipse(
                (center - radius, center - radius,
                 center + radius, center + radius), fill=color)
    cover_mask = Image.new("L", (cover_size, cover_size), 0)
    ImageDraw.Draw(cover_mask).ellipse(
        (1, 1, cover_size - 2, cover_size - 2), fill=255)
    cover_image.putalpha(Image.composite(
        cover_image.getchannel("A"), Image.new("L", cover_image.size, 0),
        cover_mask))
    cover_left = center_x - cover_size // 2
    cover_top = disc_top + (disc_size - cover_size) // 2
    canvas.alpha_composite(cover_image, (cover_left, cover_top))
    ImageDraw.Draw(canvas).ellipse(
        (cover_left, cover_top,
         cover_left + cover_size - 1, cover_top + cover_size - 1),
        outline=(255, 255, 255, 205), width=max(2, disc_size // 110))

    star_size = max(46, min(82, round(disc_size * 0.31)))
    star_left = disc_left + disc_size - round(star_size * 0.77)
    star_top = disc_top + disc_size - round(star_size * 0.70)
    try:
        star, level_center = _compose_difficulty_star(difficulty, star_size)
    except OSError:
        star = Image.new(
            "RGBA", (star_size, star_size),
            DIFFICULTY_COLORS.get(difficulty, DIFFICULTY_COLORS[2]))
        level_center = (star_size / 2, star_size / 2)

    # Draw the level into the badge itself so changing the colored difficulty
    # layer can never affect its coordinates on the preview canvas.
    level_font = _preview_font(max(22, star_size // 2))
    level_text = str(play_level).strip() or "?"
    level_draw = ImageDraw.Draw(star)
    level_box = level_draw.textbbox((0, 0), level_text, font=level_font,
                                    stroke_width=0)
    level_x = round(
        level_center[0]
        - (level_box[2] - level_box[0]) / 2 - level_box[0])
    level_y = round(
        level_center[1]
        - (level_box[3] - level_box[1]) / 2 - level_box[1])
    level_draw.text(
        (level_x, level_y), level_text, font=level_font,
        fill=(76, 0, 126, 255))
    canvas.alpha_composite(star, (star_left, star_top))

    title_text = title.strip() or "未命名曲目"
    artist_text = artist.strip() or "UNKNOWN ARTIST"
    compact = height < 280
    title_font = _fit_preview_font(
        title_text, int(width * 0.82), 22 if compact else 30,
        14 if compact else 18)
    artist_font = _fit_preview_font(
        artist_text, int(width * 0.72), 16 if compact else 20,
        11 if compact else 13)
    title_top = disc_top + disc_size + (8 if compact else 14)
    title_height = _draw_preview_text(
        canvas, title_text, center_x=center_x, top=title_top,
        font=title_font, outline=(0, 183, 238, 255), stroke_width=2)
    _draw_preview_text(
        canvas, artist_text, center_x=center_x,
        top=title_top + title_height + (4 if compact else 8), font=artist_font,
        outline=(255, 25, 129, 255), stroke_width=2)
    return canvas


def parse_ui_event_line(line: str) -> dict | None:
    """Parse one structured child-process event and ignore ordinary stdout."""
    text = line.strip()
    if not text.startswith(UI_EVENT_PREFIX):
        return None
    try:
        event = json.loads(text[len(UI_EVENT_PREFIX):])
    except (json.JSONDecodeError, TypeError):
        return None
    return event if isinstance(event, dict) else None


@dataclass(frozen=True)
class SceneOption:
    scene_id: str
    label: str
    title: str
    description: str
    background: Path


SCENE_OPTIONS = tuple(
    SceneOption(scene_id, f"{scene_id} · {title}", title, description,
                ASSET_DIR / "backgrounds" / f"scene-{scene_id[-2:]}.png")
    for scene_id, title, description in (
        ("scene_01", "太空站 / Space Station", "工业感太空站与传送带背景"),
        ("scene_02", "复古城市 / Retrocity", "霓虹车站与复古都市背景"),
        ("scene_03", "城堡 / Castle", "暗色城堡与幽灵主题背景"),
        ("scene_04", "雨夜 / Rainy Night", "雨夜街道与城市灯光背景"),
        ("scene_05", "糖果世界 / Candyland", "明亮柔和的糖果世界背景"),
        ("scene_06", "和风 / Oriental", "日式街景与夜色背景"),
        ("scene_07", "Let's Groove / GC", "Groove Coaster 联动游乐园背景"),
        ("scene_08", "幻想乡 / Gensokyo", "东方 Project 联动场景"),
        ("scene_09", "Game Graveyard / DJMAX", "DJMAX 联动游戏墓地背景"),
        ("scene_10", "Museland / Mirrorland", "初音未来与镜音联动舞台"),
        ("scene_12", "翡翠寺 / Jade Temple", "中式寺庙与玉石主题背景"),
    )
)
SCENE_BY_ID = {option.scene_id: option for option in SCENE_OPTIONS}
SCENE_ID_BY_LABEL = {option.label: option.scene_id for option in SCENE_OPTIONS}


def scene_id_from_choice(choice: str) -> str:
    """Translate the human-readable UI choice into the MDM scene id."""
    try:
        return SCENE_ID_BY_LABEL[choice]
    except KeyError as error:
        raise ValueError("请选择列表中的普通场景") from error


@dataclass(frozen=True)
class GenerationOptions:
    audio: Path
    checkpoint: Path
    output: Path
    difficulty: int
    play_level: str
    profile: str
    threshold: float | None
    bpm_mode: str
    bpm: float | None
    title: str
    artist: str
    level_designer: str
    scene: str
    speed: int
    device: str
    cpu_threads: int
    event_ordering: bool
    double_optimization: bool
    obvious_error_repair: bool
    optimization_matrix: bool
    boss_control_min_gap: float
    include_cover: bool
    cover: Path | None
    make_demo: bool
    demo_start: float
    demo_seconds: float
    keep_sidecars: bool
    strict: bool


def compose_output_path(directory: Path, name: str, difficulty: int) -> Path:
    """Compose `<name>_mapN.mdm` while preventing paths inside the name field."""
    if difficulty not in (1, 2, 3):
        raise ValueError("模型难度必须是 1、2 或 3")
    clean = name.strip()
    if clean.lower().endswith(".mdm"):
        clean = clean[:-4].rstrip()
    clean = re.sub(r"_map\d+$", "", clean, flags=re.IGNORECASE).rstrip()
    if not clean:
        raise ValueError("MDM 名称不能为空")
    if Path(clean).name != clean or re.search(r'[<>:"/\\|?*\x00-\x1f]', clean):
        raise ValueError("MDM 名称不能包含路径或文件名非法字符")
    return directory.resolve() / f"{clean}_map{difficulty}.mdm"


def build_generate_command(
    options: GenerationOptions, *, python: str = sys.executable,
    script: Path = SCRIPT_DIR / "generate_mdm.py",
) -> list[str]:
    """Build one shell-free command line from validated UI settings."""
    command = [
        python, "-X", "utf8", str(script), str(options.audio),
        "--checkpoint", str(options.checkpoint),
        "--output", str(options.output),
        "--difficulty", str(options.difficulty),
        "--play-level", options.play_level,
        "--profile", options.profile,
        "--level-designer", options.level_designer,
        "--scene", options.scene,
        "--speed", str(options.speed),
        "--device", options.device,
        "--cpu-threads", str(options.cpu_threads),
        "--boss-control-min-gap", f"{options.boss_control_min_gap:g}",
        "--event-ordering" if options.event_ordering else "--no-event-ordering",
        ("--double-optimization" if options.double_optimization
         else "--no-double-optimization"),
        ("--obvious-error-repair" if options.obvious_error_repair
         else "--no-obvious-error-repair"),
    ]
    if options.threshold is not None:
        command.extend(("--threshold", f"{options.threshold:g}"))
    if options.bpm_mode == "manual":
        command.extend(("--bpm", f"{options.bpm:g}"))
    elif options.bpm_mode == "unknown":
        command.extend(("--bpm", "0"))
    if options.title:
        command.extend(("--title", options.title))
    if options.artist:
        command.extend(("--artist", options.artist))
    if options.optimization_matrix:
        command.append("--optimization-matrix")
    if not options.include_cover:
        command.append("--no-cover")
    elif options.cover is not None:
        command.extend(("--cover", str(options.cover)))
    if not options.make_demo:
        command.append("--no-demo")
    else:
        command.extend(("--demo-start", f"{options.demo_start:g}",
                        "--demo-seconds", f"{options.demo_seconds:g}"))
    if not options.keep_sidecars:
        command.append("--no-sidecars")
    if options.strict:
        command.append("--strict")
    return command


class MuseChartApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(f"MuseChart v{PRODUCT_VERSION} · 谱面生成器")
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width, window_height = preferred_window_size(
            screen_width, screen_height)
        self.root.geometry(f"{window_width}x{window_height}")
        self.root.minsize(min(980, window_width), min(620, window_height))
        self.process: subprocess.Popen[str] | None = None
        self.cancel_requested = False
        self.messages: queue.Queue[tuple[str, object]] = queue.Queue()
        self.cover_probe_serial = 0
        self.embedded_cover: bool | None = None
        self.embedded_cover_preview: Path | None = None
        self.cover_probe_pending = False
        self.embedded_message = "选择音频后自动检测内嵌封面"
        self.fallback_cover = find_default_cover(IMAGE_DIR)
        self.name_is_automatic = True
        self.ui_images: dict[str, tk.PhotoImage] = {}
        self.scene_preview_image: ImageTk.PhotoImage | None = None
        self.song_preview_image: ImageTk.PhotoImage | None = None
        self.active_cover_preview: Path | None = None
        self.song_preview_job: str | None = None
        self.preview_directory = Path(tempfile.mkdtemp(prefix="musechart-ui-"))
        self.last_log_path: Path | None = None
        self.last_result: dict | None = None

        self._make_variables()
        self._load_theme_assets()
        self._configure_style()
        self._build_layout()
        self._bind_events()
        self._update_difficulty_icon()
        self._update_scene_preview()
        self._refresh_cover_labels()
        self._schedule_song_preview()
        self._poll_messages()
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _make_variables(self) -> None:
        checkpoint_dir = SCRIPT_DIR / "checkpoints"
        checkpoint_names = (
            "MuseChart_v1.0.pt",
            "best_playability.pt",
            "best.pt",
            "best_playability-55.pt",
            "best_label.pt",
            "best_timing.pt",
            "last.pt",
        )
        default_model = next(
            (checkpoint_dir / name for name in checkpoint_names
             if (checkpoint_dir / name).is_file()),
            checkpoint_dir / checkpoint_names[0],
        )
        self.audio_var = tk.StringVar()
        self.cover_var = tk.StringVar()
        self.checkpoint_var = tk.StringVar(value=str(default_model))
        self.output_name_var = tk.StringVar()
        self.output_dir_var = tk.StringVar(value=str(SCRIPT_DIR / "output"))
        self.output_preview_var = tk.StringVar(value="最终文件：请先选择音乐")
        self.title_var = tk.StringVar()
        self.artist_var = tk.StringVar()
        self.designer_var = tk.StringVar(value="MuseChartWeave")
        self.difficulty_var = tk.StringVar(value="2")
        self.play_level_var = tk.StringVar(value="7")
        self.profile_var = tk.StringVar(value="balanced")
        self.manual_threshold_var = tk.BooleanVar(value=False)
        self.threshold_var = tk.StringVar(value="0.45")
        self.bpm_mode_var = tk.StringVar(value="auto")
        self.bpm_var = tk.StringVar(value="120")
        self.scene_var = tk.StringVar(value="scene_01")
        self.scene_choice_var = tk.StringVar(value=SCENE_BY_ID["scene_01"].label)
        self.scene_name_var = tk.StringVar()
        self.scene_description_var = tk.StringVar()
        self.speed_var = tk.StringVar(value="2")
        self.device_var = tk.StringVar(value="auto")
        self.cpu_threads_var = tk.StringVar(value=str(min(8, os.cpu_count() or 1)))
        self.event_ordering_var = tk.BooleanVar(value=True)
        self.double_optimization_var = tk.BooleanVar(value=True)
        self.obvious_error_var = tk.BooleanVar(value=True)
        self.matrix_var = tk.BooleanVar(value=False)
        self.boss_gap_var = tk.StringVar(value="0.30")
        self.include_cover_var = tk.BooleanVar(value=True)
        self.make_demo_var = tk.BooleanVar(value=True)
        self.demo_start_var = tk.StringVar(value="0")
        self.demo_seconds_var = tk.StringVar(value="25")
        self.sidecars_var = tk.BooleanVar(value=True)
        self.strict_var = tk.BooleanVar(value=False)
        self.cover_probe_var = tk.StringVar(value=self.embedded_message)
        self.cover_source_var = tk.StringVar()
        self.status_var = tk.StringVar(value="等待生成")
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_percent_var = tk.StringVar(value="0%")
        self.progress_stage_var = tk.StringVar(value="准备就绪")
        self.progress_detail_var = tk.StringVar(value="选择音乐并确认谱面设置")
        self.result_bpm_var = tk.StringVar(value="—")
        self.result_events_var = tk.StringVar(value="—")
        self.result_holds_var = tk.StringVar(value="—")
        self.result_boss_var = tk.StringVar(value="—")
        self.result_repairs_var = tk.StringVar(value="—")
        self.result_elapsed_var = tk.StringVar(value="—")
        self.result_cover_var = tk.StringVar(value="尚未生成")
        self.result_output_var = tk.StringVar(value="生成完成后在这里显示输出文件")

    def _load_png(self, key: str, path: Path, *, subsample: int = 1
                  ) -> tk.PhotoImage | None:
        if not path.is_file():
            return None
        try:
            image = tk.PhotoImage(master=self.root, file=str(path))
            if subsample > 1:
                image = image.subsample(subsample, subsample)
        except tk.TclError:
            return None
        self.ui_images[key] = image
        return image

    def _load_theme_assets(self) -> None:
        icon = self._load_png("window_icon", LEVEL_STAR_PATH)
        if icon is not None:
            self.root.iconphoto(True, icon)
            self.ui_images["header_icon"] = icon.subsample(3, 3)
        for difficulty in ("1", "2", "3"):
            self._load_png(
                f"difficulty_{difficulty}",
                DIFFICULTY_STAR_PATHS[int(difficulty)],
                subsample=2,
            )

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        background = THEME["background"]
        panel = THEME["surface_dark"]
        raised = THEME["surface_raised"]
        text = THEME["text"]
        muted = THEME["text_muted"]
        accent = THEME["primary"]
        soft_edge = "#493568"
        font = ("Microsoft YaHei UI", 9)
        self.root.configure(background=background)
        style.configure(".", font=font, foreground=text)
        style.configure("App.TFrame", background=background)
        style.configure("Panel.TFrame", background=panel)
        style.configure("Raised.TFrame", background=raised)
        style.configure("TLabel", background=panel, foreground=text)
        style.configure("Panel.TLabelframe", background=panel, padding=6,
                        bordercolor=soft_edge, lightcolor=soft_edge,
                        darkcolor=soft_edge, borderwidth=1)
        style.configure("Panel.TLabelframe.Label", background=panel,
                        foreground=text,
                        font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("Title.TLabel", background=background, foreground=text,
                        font=("Microsoft YaHei UI", 20, "bold"))
        style.configure("Subtitle.TLabel", background=background, foreground=muted,
                        font=("Microsoft YaHei UI", 9))
        style.configure("HeaderIcon.TLabel", background=background)
        style.configure("Status.TLabel", background=panel, foreground=muted,
                        font=("Microsoft YaHei UI", 9))
        style.configure("PanelHeading.TLabel", background=panel,
                        foreground=text,
                        font=("Microsoft YaHei UI", 12, "bold"))
        style.configure("SectionNumber.TLabel", background=THEME["primary"],
                        foreground=text, padding=(8, 4),
                        font=("Microsoft YaHei UI", 9, "bold"))
        style.configure("SectionTitle.TLabel", background=panel, foreground=text,
                        font=("Microsoft YaHei UI", 13, "bold"))
        style.configure("SectionSubtitle.TLabel", background=panel,
                        foreground=muted, font=("Microsoft YaHei UI", 8))
        style.configure("AccentText.TLabel", background=panel,
                        foreground=THEME["cyan"],
                        font=("Microsoft YaHei UI", 9, "bold"))
        style.configure("ProgressStage.TLabel", background=panel, foreground=text,
                        font=("Microsoft YaHei UI", 13, "bold"))
        style.configure("ProgressPercent.TLabel", background=panel,
                        foreground=THEME["cyan"],
                        font=("Microsoft YaHei UI", 17, "bold"))
        style.configure("ResultLabel.TLabel", background=raised, foreground=muted,
                        font=("Microsoft YaHei UI", 8))
        style.configure("ResultValue.TLabel", background=raised, foreground=text,
                        font=("Microsoft YaHei UI", 12, "bold"))
        style.configure("SceneName.TLabel", background=panel,
                        foreground=THEME["cyan"],
                        font=("Microsoft YaHei UI", 11, "bold"))
        style.configure("SceneMeta.TLabel", background=panel, foreground=muted,
                        font=("Microsoft YaHei UI", 9))

        style.configure("TEntry", foreground=text, fieldbackground=raised,
                        insertcolor=text, bordercolor=raised,
                        lightcolor=raised, darkcolor=raised, padding=4)
        style.map("TEntry", fieldbackground=[("disabled", panel)],
                  foreground=[("disabled", muted)])
        for widget in ("TCombobox", "TSpinbox"):
            style.configure(widget, foreground=text, fieldbackground=raised,
                            background=raised, arrowcolor=THEME["cyan"],
                            padding=5, bordercolor=raised,
                            lightcolor=raised, darkcolor=raised)
            style.map(
                widget,
                fieldbackground=[("readonly", raised), ("disabled", panel)],
                selectbackground=[("readonly", raised)],
                selectforeground=[("readonly", text)],
            )

        style.configure("TButton", font=("Microsoft YaHei UI", 9, "bold"),
                        foreground=text, background=THEME["violet_deep"],
                        bordercolor=THEME["violet_deep"], padding=(10, 5))
        style.map("TButton",
                  background=[("active", THEME["violet"]),
                              ("pressed", THEME["surface_raised"]),
                              ("disabled", panel)],
                  foreground=[("disabled", "#7F719D")])
        style.configure("Secondary.TButton", foreground=text,
                        background=THEME["cyan_deep"],
                        bordercolor=THEME["cyan_deep"], padding=(10, 5))
        style.map("Secondary.TButton",
                  background=[("active", THEME["cyan"]),
                              ("pressed", THEME["cyan_deep"]),
                              ("disabled", panel)])
        style.configure("Accent.TButton", font=("Microsoft YaHei UI", 10, "bold"),
                        foreground=text, background=accent,
                        bordercolor=accent, padding=(16, 7))
        style.map("Accent.TButton",
                  background=[("active", THEME["primary_deep"]),
                              ("pressed", THEME["primary_deep"]),
                              ("disabled", "#6E2852")])
        style.configure("TCheckbutton", background=panel, foreground=text,
                        indicatorbackground=raised, indicatorforeground=text,
                        padding=2)
        style.map("TCheckbutton", background=[("active", panel)],
                  indicatorbackground=[("selected", THEME["cyan"]),
                                       ("active", THEME["violet"])])
        style.configure("TRadiobutton", background=panel, foreground=text,
                        indicatorbackground=raised, indicatorforeground=text)
        style.map("TRadiobutton", background=[("active", panel)],
                  indicatorbackground=[("selected", THEME["cyan"]),
                                       ("active", THEME["violet"])])
        style.configure("Choice.TRadiobutton", background=raised,
                        foreground=text, indicatorbackground=raised,
                        padding=(5, 4), font=("Microsoft YaHei UI", 9, "bold"))
        style.map("Choice.TRadiobutton",
                  background=[("selected", THEME["violet_deep"]),
                              ("active", THEME["surface_raised"])],
                  foreground=[("selected", THEME["text"])],
                  indicatorbackground=[("selected", THEME["cyan"])])
        style.configure("Repair.TCheckbutton", background=panel,
                        foreground=text,
                        font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("TNotebook", background=panel, borderwidth=0,
                        tabmargins=(0, 0, 0, 0))
        style.configure("TNotebook.Tab", background=background, foreground=muted,
                        padding=(18, 10), borderwidth=0,
                        font=("Microsoft YaHei UI", 9, "bold"))
        style.map("TNotebook.Tab",
                  background=[("selected", raised), ("active", panel)],
                  foreground=[("selected", THEME["cyan"]), ("active", text)])
        style.configure("TPanedwindow", background=background)
        style.configure("Horizontal.TProgressbar", background=THEME["cyan"],
                        troughcolor=raised, bordercolor=raised,
                        lightcolor=THEME["cyan"],
                        darkcolor=THEME["cyan"])

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, style="App.TFrame", padding=12)
        outer.pack(fill="both", expand=True)
        header = ttk.Frame(outer, style="App.TFrame")
        header.pack(fill="x", pady=(0, 8))
        header_icon = self.ui_images.get("header_icon")
        if header_icon is not None:
            ttk.Label(header, image=header_icon, style="HeaderIcon.TLabel").pack(
                side="left", padx=(0, 10))
        title_box = ttk.Frame(header, style="App.TFrame")
        title_box.pack(side="left", fill="x", expand=True)
        ttk.Label(title_box, text=f"MuseChartWeaver v{PRODUCT_VERSION} 谱面生成器",
                  style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            title_box,
            text="从音乐到可玩谱面 · 本地推理 · 可视化生成",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(2, 0))
        tk.Label(
            header, text=f"LOCAL  •  V{PRODUCT_VERSION}",
            background=THEME["cyan_deep"],
            foreground=THEME["text"], padx=12, pady=6,
            font=("Microsoft YaHei UI", 9, "bold"),
        ).pack(side="right")

        workspace = ttk.Frame(outer, style="App.TFrame")
        workspace.pack(fill="both", expand=True)
        workspace.columnconfigure(0, weight=42, minsize=390)
        workspace.columnconfigure(1, weight=58, minsize=520)
        workspace.rowconfigure(0, weight=1)

        left_content = ttk.Frame(
            workspace, style="Panel.TFrame", padding=10)
        left_content.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        right_panel = ttk.Frame(workspace, style="App.TFrame")
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        right_panel.rowconfigure(0, weight=1, minsize=200)
        right_panel.rowconfigure(1, weight=0)
        right_panel.columnconfigure(0, weight=1)
        preview_panel = ttk.Frame(
            right_panel, style="Panel.TFrame", padding=10)
        output_panel = ttk.Frame(
            right_panel, style="Panel.TFrame", padding=10)
        preview_panel.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
        output_panel.grid(row=1, column=0, sticky="ew", pady=(5, 0))

        self._build_source_workspace(left_content)
        self._build_chart_workspace(left_content)
        self._build_song_preview_workspace(preview_panel)
        self._build_output_workspace(output_panel)

    @staticmethod
    def _section_header(parent: ttk.Frame, number: str, title: str,
                        subtitle: str) -> None:
        header = ttk.Frame(parent, style="Panel.TFrame")
        header.pack(fill="x", pady=(0, 6))
        ttk.Label(header, text=number, style="SectionNumber.TLabel").pack(
            side="left", padx=(0, 9))
        copy = ttk.Frame(header, style="Panel.TFrame")
        copy.pack(side="left", fill="x", expand=True)
        ttk.Label(copy, text=title, style="SectionTitle.TLabel").pack(anchor="w")
        ttk.Label(copy, text=subtitle, style="SectionSubtitle.TLabel").pack(
            anchor="w", pady=(1, 0))

    def _build_song_preview_workspace(self, panel: ttk.Frame) -> None:
        self._section_header(
            panel, "LIVE", "选曲效果预览", "场景、唱片、封面与难度会实时更新")
        preview_shell = tk.Frame(
            panel, background=THEME["background"], highlightthickness=0,
            height=300)
        preview_shell.pack(fill="both", expand=True)
        preview_shell.pack_propagate(False)
        self.song_preview_label = tk.Label(
            preview_shell, text="正在准备选曲预览…",
            background=THEME["background"], foreground=THEME["text_muted"],
            font=("Microsoft YaHei UI", 11, "bold"),
        )
        self.song_preview_label.pack(fill="both", expand=True)
        self.song_preview_label.bind(
            "<Configure>", lambda _event: self._schedule_song_preview())

    def _build_source_workspace(self, panel: ttk.Frame) -> None:
        self._section_header(panel, "01", "曲目素材", "音乐、封面与基础信息")

        audio_box = ttk.LabelFrame(
            panel, text="素材文件", style="Panel.TLabelframe")
        audio_box.pack(fill="x")
        audio_box.columnconfigure(1, weight=1)
        ttk.Label(audio_box, text="音乐").grid(
            row=0, column=0, sticky="w", padx=(0, 7))
        self.audio_entry = ttk.Entry(audio_box, textvariable=self.audio_var)
        self.audio_entry.grid(row=0, column=1, columnspan=2, sticky="ew")
        ttk.Button(
            audio_box, text="选择", command=self._choose_audio,
            style="Secondary.TButton",
        ).grid(row=0, column=3, padx=(6, 0))
        ttk.Label(audio_box, text="封面").grid(
            row=1, column=0, sticky="w", padx=(0, 7), pady=(5, 0))
        ttk.Label(
            audio_box, text="封面统一在右侧预览",
            style="Status.TLabel", anchor="w", width=19,
        ).grid(row=1, column=1, sticky="w", pady=(5, 0))
        ttk.Button(
            audio_box, text="选择封面", command=self._choose_cover,
            style="Secondary.TButton",
        ).grid(row=1, column=2, sticky="e", padx=(5, 0), pady=(5, 0))
        ttk.Button(
            audio_box, text="自动", command=self._clear_cover,
        ).grid(row=1, column=3, padx=(6, 0), pady=(5, 0))

        metadata = ttk.LabelFrame(
            panel, text="歌曲与模型", style="Panel.TLabelframe")
        metadata.pack(fill="x", pady=(7, 0))
        metadata.columnconfigure(1, weight=1)
        metadata.columnconfigure(3, weight=1)
        for row, (left_label, left_variable, right_label, right_variable) in enumerate((
            ("标题", self.title_var, "作者", self.artist_var),
            ("谱师", self.designer_var, "", None),
        )):
            ttk.Label(metadata, text=left_label).grid(
                row=row, column=0, sticky="w", pady=2)
            ttk.Entry(metadata, textvariable=left_variable, width=14).grid(
                row=row, column=1, sticky="ew", padx=(6, 10), pady=2)
            if right_variable is not None:
                ttk.Label(metadata, text=right_label).grid(
                    row=row, column=2, sticky="w", pady=2)
                ttk.Entry(metadata, textvariable=right_variable, width=14).grid(
                    row=row, column=3, sticky="ew", padx=(6, 0), pady=2)
        ttk.Label(metadata, text="模型").grid(
            row=2, column=0, sticky="w", pady=(3, 0))
        ttk.Entry(
            metadata, textvariable=self.checkpoint_var, width=24).grid(
                row=2, column=1, columnspan=2, sticky="ew",
                padx=(6, 6), pady=(3, 0))
        ttk.Button(
            metadata, text="选择", command=self._choose_checkpoint,
        ).grid(row=2, column=3, sticky="e", pady=(3, 0))

    def _build_chart_workspace(self, panel: ttk.Frame) -> None:
        ttk.Separator(panel, orient="horizontal").pack(
            fill="x", pady=(8, 7))
        self._section_header(panel, "02", "谱面设计", "难度、场景与修复策略")

        difficulty_box = ttk.LabelFrame(
            panel, text="模型难度", style="Panel.TLabelframe")
        difficulty_box.pack(fill="x")
        for column, (value, title) in enumerate((
            ("1", "萌新 · map1"),
            ("2", "高手 · map2"),
            ("3", "大触 · map3"),
        )):
            difficulty_box.columnconfigure(column, weight=1)
            ttk.Radiobutton(
                difficulty_box, text=title, value=value,
                variable=self.difficulty_var, style="Choice.TRadiobutton",
                image=self.ui_images.get(f"difficulty_{value}"),
                compound="left",
            ).grid(row=0, column=column, sticky="ew",
                   padx=(0 if column == 0 else 4, 0), pady=2)
        self.difficulty_icon_label = ttk.Label(panel)

        density_row = ttk.Frame(panel, style="Panel.TFrame")
        density_row.pack(fill="x", pady=(5, 0))
        ttk.Label(density_row, text="事件密度").pack(side="left")
        for value in ("casual", "balanced", "aggressive"):
            ttk.Radiobutton(
                density_row, text=PROFILE_LABELS[value], value=value,
                variable=self.profile_var,
            ).pack(side="left", padx=(8, 0))
        ttk.Label(density_row, text="等级").pack(side="left", padx=(10, 0))
        ttk.Entry(
            density_row, textvariable=self.play_level_var, width=4,
        ).pack(side="left", padx=(5, 0))

        threshold_row = ttk.Frame(panel, style="Panel.TFrame")
        threshold_row.pack(fill="x", pady=(4, 0))
        ttk.Checkbutton(
            threshold_row, text="手动覆盖 onset 阈值",
            variable=self.manual_threshold_var,
            command=self._toggle_threshold,
        ).pack(side="left")
        self.threshold_entry = ttk.Entry(
            threshold_row, textvariable=self.threshold_var,
            width=8, state="disabled")
        self.threshold_entry.pack(side="left", padx=(8, 0))

        scene_box = ttk.LabelFrame(
            panel, text="游玩场景", style="Panel.TLabelframe")
        scene_box.pack(fill="x", pady=(5, 0))
        scene_box.columnconfigure(0, weight=1)
        self.scene_combobox = ttk.Combobox(
            scene_box, textvariable=self.scene_choice_var,
            values=tuple(option.label for option in SCENE_OPTIONS),
            state="readonly",
        )
        self.scene_combobox.grid(row=0, column=0, sticky="ew")

        runtime = ttk.LabelFrame(
            panel, text="节奏与运行", style="Panel.TLabelframe")
        runtime.pack(fill="x", pady=(5, 0))
        bpm_options = ttk.Frame(runtime, style="Panel.TFrame")
        bpm_options.grid(row=0, column=0, columnspan=6, sticky="w")
        ttk.Radiobutton(
            bpm_options, text="自动 BPM", value="auto",
            variable=self.bpm_mode_var, command=self._toggle_bpm,
        ).pack(side="left")
        ttk.Radiobutton(
            bpm_options, text="手动", value="manual",
            variable=self.bpm_mode_var, command=self._toggle_bpm,
        ).pack(side="left", padx=(10, 0))
        self.bpm_entry = ttk.Entry(
            bpm_options, textvariable=self.bpm_var, width=7, state="disabled")
        self.bpm_entry.pack(side="left", padx=(5, 0))
        ttk.Radiobutton(
            bpm_options, text="未知", value="unknown",
            variable=self.bpm_mode_var, command=self._toggle_bpm,
        ).pack(side="left", padx=(10, 0))
        ttk.Label(runtime, text="物件速度").grid(
            row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Combobox(
            runtime, textvariable=self.speed_var, values=("1", "2", "3"),
            state="readonly", width=7,
        ).grid(row=1, column=1, sticky="w", padx=(5, 12), pady=(4, 0))
        ttk.Button(
            runtime, text="高级运行设置", command=self._open_advanced_settings,
        ).grid(row=1, column=2, sticky="w", pady=(4, 0))

        repair_box = ttk.LabelFrame(
            panel, text="谱面优化", style="Panel.TLabelframe")
        repair_box.pack(fill="x", pady=(5, 0))
        for index, (title, variable) in enumerate((
            ("事件秩序化", self.event_ordering_var),
            ("经典双轨修复", self.double_optimization_var),
            ("明显错误修复", self.obvious_error_var),
        )):
            ttk.Checkbutton(
                repair_box, text=title, variable=variable,
                style="Repair.TCheckbutton",
            ).grid(row=0, column=index, sticky="w",
                   padx=(0 if index == 0 else 8, 0), pady=1)

    def _build_output_workspace(self, panel: ttk.Frame) -> None:
        self._section_header(panel, "03", "生成与结果", "输出、进度与核心摘要")

        controls = ttk.Frame(panel, style="Panel.TFrame")
        controls.pack(fill="x")
        controls.columnconfigure(0, weight=3)
        controls.columnconfigure(1, weight=2)
        target_box = ttk.LabelFrame(
            controls, text="输出目标", style="Panel.TLabelframe")
        target_box.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        target_box.columnconfigure(1, weight=1)
        ttk.Label(target_box, text="名称").grid(row=0, column=0, sticky="w")
        self.output_name_entry = ttk.Entry(
            target_box, textvariable=self.output_name_var)
        self.output_name_entry.grid(
            row=0, column=1, columnspan=2, sticky="ew", padx=(7, 0))
        ttk.Label(target_box, text="路径").grid(
            row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(target_box, textvariable=self.output_dir_var).grid(
            row=1, column=1, sticky="ew", padx=(7, 5), pady=(6, 0))
        ttk.Button(
            target_box, text="选择", command=self._choose_output_directory,
        ).grid(row=1, column=2, pady=(6, 0))

        options_box = ttk.LabelFrame(
            controls, text="打包选项", style="Panel.TLabelframe")
        options_box.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        ttk.Checkbutton(
            options_box, text="封面", variable=self.include_cover_var,
            command=self._refresh_cover_labels,
        ).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(
            options_box, text="试听", variable=self.make_demo_var,
        ).grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Checkbutton(
            options_box, text="报告", variable=self.sidecars_var,
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Button(
            options_box, text="高级设置",
            command=self._open_advanced_settings,
        ).grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(4, 0))

        progress_box = ttk.Frame(
            panel, style="Raised.TFrame", padding=5)
        progress_box.pack(fill="x", pady=(5, 0))
        progress_header = ttk.Frame(progress_box, style="Raised.TFrame")
        progress_header.pack(fill="x")
        ttk.Label(
            progress_header, textvariable=self.progress_stage_var,
            style="ProgressStage.TLabel").pack(side="left")
        ttk.Label(
            progress_header, textvariable=self.progress_detail_var,
            style="Status.TLabel").pack(side="left", padx=(10, 0))
        ttk.Label(
            progress_header, textvariable=self.progress_percent_var,
            style="ProgressPercent.TLabel").pack(side="right")
        self.progress = ttk.Progressbar(
            progress_box, mode="determinate", maximum=100,
            variable=self.progress_var)
        self.progress.pack(fill="x", pady=(3, 0))
        stage_row = tk.Frame(progress_box, background=THEME["surface_dark"])
        stage_row.pack(fill="x", pady=(6, 0))
        self.stage_labels: list[tk.Label] = []
        for index, text in enumerate(("模型", "音频", "推理", "整理", "打包"), 1):
            stage_row.columnconfigure(index - 1, weight=1)
            label = tk.Label(
                stage_row, text=text, background=THEME["surface_raised"],
                foreground=THEME["text_muted"], padx=5, pady=3,
                font=("Microsoft YaHei UI", 8, "bold"))
            label.grid(row=0, column=index - 1, sticky="ew",
                       padx=(0 if index == 1 else 3, 0))
            self.stage_labels.append(label)

        result_box = ttk.Frame(
            panel, style="Panel.TFrame")
        result_box.pack(fill="x", pady=(5, 0))
        metrics = (
            ("BPM", self.result_bpm_var),
            ("事件", self.result_events_var),
            ("长按", self.result_holds_var),
            ("Boss", self.result_boss_var),
            ("修复", self.result_repairs_var),
            ("耗时", self.result_elapsed_var),
        )
        for column in range(3):
            result_box.columnconfigure(column, weight=1)
        for index, (label, variable) in enumerate(metrics):
            row, column = divmod(index, 3)
            card = ttk.Frame(result_box, style="Raised.TFrame", padding=3)
            card.grid(row=row, column=column, sticky="ew",
                      padx=(0 if column == 0 else 2, 0),
                      pady=(0 if row == 0 else 2, 0))
            ttk.Label(card, text=label, style="ResultLabel.TLabel").pack(
                side="left")
            ttk.Label(
                card, textvariable=variable,
                style="ResultValue.TLabel").pack(side="right")
        ttk.Label(
            result_box, textvariable=self.result_cover_var,
            style="AccentText.TLabel").grid(
                row=2, column=0, sticky="w", pady=(5, 0))
        ttk.Label(
            result_box, textvariable=self.result_output_var,
            style="Status.TLabel", wraplength=430, justify="right").grid(
                row=2, column=1, columnspan=2, sticky="e", pady=(5, 0))

        actions = ttk.Frame(panel, style="Panel.TFrame")
        actions.pack(fill="x", pady=(5, 0))
        actions.columnconfigure(0, weight=3)
        for column in range(1, 4):
            actions.columnconfigure(column, weight=1)
        self.generate_button = ttk.Button(
            actions, text="开始生成谱面", style="Accent.TButton",
            command=self._start_generation)
        self.generate_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.cancel_button = ttk.Button(
            actions, text="取消", command=self._cancel_generation,
            state="disabled")
        self.cancel_button.grid(row=0, column=1, sticky="ew")
        ttk.Button(
            actions, text="打开输出", command=self._open_output,
            style="Secondary.TButton").grid(
                row=0, column=2, sticky="ew", padx=5)
        self.open_log_button = ttk.Button(
            actions, text="任务日志", command=self._open_last_log,
            state="disabled")
        self.open_log_button.grid(row=0, column=3, sticky="ew")

    def _open_advanced_settings(self) -> None:
        existing = getattr(self, "advanced_window", None)
        if existing is not None and existing.winfo_exists():
            existing.lift()
            existing.focus_force()
            return
        window = tk.Toplevel(self.root)
        self.advanced_window = window
        window.title("MuseChart · 高级设置")
        window.geometry("480x390")
        window.resizable(False, False)
        window.configure(background=THEME["background"])
        window.transient(self.root)

        outer = ttk.Frame(window, style="App.TFrame", padding=16)
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer, text="高级运行与打包设置",
            style="Title.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            outer, text="日常生成保持默认值即可。",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(2, 12))

        runtime = ttk.LabelFrame(
            outer, text="运行设备", style="Panel.TLabelframe")
        runtime.pack(fill="x")
        ttk.Label(runtime, text="设备").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            runtime, textvariable=self.device_var,
            values=("auto", "cuda", "cpu"), state="readonly", width=10,
        ).grid(row=0, column=1, sticky="w", padx=(8, 22))
        ttk.Label(runtime, text="CPU 线程").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(
            runtime, from_=1, to=max(1, os.cpu_count() or 1),
            textvariable=self.cpu_threads_var, width=6,
        ).grid(row=0, column=3, sticky="w", padx=(8, 0))

        packaging = ttk.LabelFrame(
            outer, text="输出策略", style="Panel.TLabelframe")
        packaging.pack(fill="x", pady=(10, 0))
        ttk.Checkbutton(
            packaging, text="输出全部 8 种优化组合",
            variable=self.matrix_var,
        ).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(
            packaging, text="严格导出检查", variable=self.strict_var,
        ).grid(row=0, column=1, sticky="w", padx=(18, 0))

        timing = ttk.LabelFrame(
            outer, text="时间参数", style="Panel.TLabelframe")
        timing.pack(fill="x", pady=(10, 0))
        ttk.Checkbutton(
            timing, text="生成试听 demo", variable=self.make_demo_var,
            command=self._toggle_demo,
        ).grid(row=0, column=0, columnspan=4, sticky="w")
        ttk.Label(timing, text="试听起点").grid(
            row=1, column=0, sticky="w", pady=(8, 0))
        self.demo_start_entry = ttk.Entry(
            timing, textvariable=self.demo_start_var, width=7)
        self.demo_start_entry.grid(
            row=1, column=1, sticky="w", padx=(7, 18), pady=(8, 0))
        ttk.Label(timing, text="试听长度").grid(
            row=1, column=2, sticky="w", pady=(8, 0))
        self.demo_seconds_entry = ttk.Entry(
            timing, textvariable=self.demo_seconds_var, width=7)
        self.demo_seconds_entry.grid(
            row=1, column=3, sticky="w", padx=(7, 0), pady=(8, 0))
        ttk.Label(timing, text="Boss 控制最小间隔").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Entry(
            timing, textvariable=self.boss_gap_var, width=7,
        ).grid(row=2, column=2, sticky="w", pady=(8, 0))
        ttk.Label(timing, text="秒").grid(
            row=2, column=3, sticky="w", pady=(8, 0))
        self._toggle_demo()

        ttk.Button(
            outer, text="完成", style="Accent.TButton",
            command=window.destroy,
        ).pack(fill="x", side="bottom")


    def _bind_events(self) -> None:
        self.audio_entry.bind("<FocusOut>", lambda _event: self._audio_changed())
        self.output_name_entry.bind("<Key>", lambda _event: self._mark_name_manual())
        self.output_name_var.trace_add("write", lambda *_: self._update_output_preview())
        self.output_dir_var.trace_add("write", lambda *_: self._update_output_preview())
        self.difficulty_var.trace_add("write", lambda *_: self._update_output_preview())
        self.difficulty_var.trace_add("write", lambda *_: self._update_difficulty_icon())
        self.difficulty_var.trace_add("write", lambda *_: self._schedule_song_preview())
        self.play_level_var.trace_add("write", lambda *_: self._schedule_song_preview())
        self.title_var.trace_add("write", lambda *_: self._schedule_song_preview())
        self.artist_var.trace_add("write", lambda *_: self._schedule_song_preview())
        self.scene_choice_var.trace_add("write", lambda *_: self._scene_changed())
        self.cover_var.trace_add("write", lambda *_: self._refresh_cover_labels())

    def _update_difficulty_icon(self) -> None:
        if not hasattr(self, "difficulty_icon_label"):
            return
        image = self.ui_images.get(f"difficulty_{self.difficulty_var.get()}")
        self.difficulty_icon_label.configure(image=image or "")

    def _scene_changed(self) -> None:
        try:
            scene_id = scene_id_from_choice(self.scene_choice_var.get())
        except ValueError:
            return
        self.scene_var.set(scene_id)
        self._update_scene_preview()

    def _update_scene_preview(self) -> None:
        option = SCENE_BY_ID.get(self.scene_var.get())
        if option is None:
            self.scene_preview_image = None
            self.scene_name_var.set("")
            self.scene_description_var.set("")
            self._schedule_song_preview()
            return
        self.scene_preview_image = None
        self.scene_name_var.set(option.title)
        self.scene_description_var.set(
            f"{option.scene_id}  ·  {option.description}")
        self._schedule_song_preview()

    def _schedule_song_preview(self) -> None:
        if not hasattr(self, "song_preview_label"):
            return
        if self.song_preview_job is not None:
            try:
                self.root.after_cancel(self.song_preview_job)
            except tk.TclError:
                pass
        self.song_preview_job = self.root.after(80, self._render_song_preview)

    def _render_song_preview(self) -> None:
        self.song_preview_job = None
        if not hasattr(self, "song_preview_label"):
            return
        width = self.song_preview_label.winfo_width()
        height = self.song_preview_label.winfo_height()
        if width < 100 or height < 100:
            self.song_preview_job = self.root.after(
                100, self._render_song_preview)
            return
        width, height = max(320, width), max(180, height)
        option = SCENE_BY_ID.get(self.scene_var.get(), SCENE_OPTIONS[0])
        title = self.title_var.get().strip()
        if not title and self.audio_var.get().strip():
            title = Path(self.audio_var.get().strip()).stem
        try:
            difficulty = int(self.difficulty_var.get())
        except ValueError:
            difficulty = 2
        try:
            prepared = compose_song_preview(
                background=option.background,
                cover=self.active_cover_preview,
                title=title,
                artist=self.artist_var.get(),
                difficulty=difficulty,
                play_level=self.play_level_var.get(),
                size=(width, height),
            )
            preview = ImageTk.PhotoImage(prepared, master=self.root)
        except (OSError, ValueError, tk.TclError):
            preview = None
        self.song_preview_image = preview
        if preview is None:
            self.song_preview_label.configure(
                image="", text="选曲预览暂时不可用")
        else:
            self.song_preview_label.configure(image=preview, text="")

    @staticmethod
    def _extract_embedded_cover(audio: Path, output: Path) -> bool:
        executable = shutil.which("ffmpeg")
        if executable is None:
            return False
        completed = subprocess.run(
            [
                executable, "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(audio), "-map", "0:v:0", "-frames:v", "1",
                str(output),
            ],
            check=False, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
        return (
            completed.returncode == 0
            and output.is_file()
            and output.stat().st_size > 0
        )

    def _set_cover_preview(self, path: Path | None, _placeholder: str) -> None:
        self.active_cover_preview = (
            path if path is not None and path.is_file() else None)
        self._schedule_song_preview()

    def _choose_audio(self) -> None:
        path = filedialog.askopenfilename(title="选择音乐", filetypes=AUDIO_TYPES)
        if not path:
            return
        self.audio_var.set(path)
        self.name_is_automatic = True
        self._update_default_name()
        self._audio_changed()

    def _choose_cover(self) -> None:
        path = filedialog.askopenfilename(title="选择显式封面", filetypes=IMAGE_TYPES)
        if path:
            self.cover_var.set(path)

    def _clear_cover(self) -> None:
        self.cover_var.set("")

    def _choose_checkpoint(self) -> None:
        path = filedialog.askopenfilename(
            title="选择 v1.0 checkpoint",
            initialdir=str(SCRIPT_DIR / "checkpoints"),
            filetypes=(("PyTorch checkpoint", "*.pt"), ("所有文件", "*.*")))
        if path:
            self.checkpoint_var.set(path)

    def _choose_output_directory(self) -> None:
        initial = (Path(self.output_dir_var.get().strip())
                   if self.output_dir_var.get().strip() else SCRIPT_DIR / "output")
        path = filedialog.askdirectory(
            title="选择输出文件夹", initialdir=str(initial))
        if path:
            self.output_dir_var.set(path)

    def _mark_name_manual(self) -> None:
        self.name_is_automatic = False

    def _update_default_name(self) -> None:
        if not self.name_is_automatic:
            return
        text = self.audio_var.get().strip()
        if not text:
            return
        self.output_name_var.set(Path(text).stem)

    def _update_output_preview(self) -> None:
        try:
            difficulty = int(self.difficulty_var.get())
            output = compose_output_path(
                Path(self.output_dir_var.get().strip() or SCRIPT_DIR / "output"),
                self.output_name_var.get(), difficulty)
        except (ValueError, OSError):
            self.output_preview_var.set("请填写有效名称并选择模型难度")
            return
        self.output_preview_var.set(str(output))

    def _audio_changed(self) -> None:
        text = self.audio_var.get().strip()
        self._update_default_name()
        self.cover_probe_serial += 1
        serial = self.cover_probe_serial
        self.embedded_cover = None
        self.embedded_cover_preview = None
        self.cover_probe_pending = False
        if not text:
            self.embedded_message = "选择音频后自动检测内嵌封面"
            self._refresh_cover_labels()
            return
        audio = Path(text)
        if not audio.is_file():
            self.embedded_message = "音频路径无效，尚未检测"
            self._refresh_cover_labels()
            return
        self.embedded_message = "正在检查内嵌封面…"
        self.cover_probe_pending = True
        self._refresh_cover_labels()

        def worker() -> None:
            result = inspect_embedded_cover(audio)
            preview_path = None
            if result.get("present") is True:
                candidate = self.preview_directory / f"embedded-{serial}.png"
                if self._extract_embedded_cover(audio, candidate):
                    preview_path = str(candidate)
                else:
                    result["present"] = False
                    result["message"] = "内嵌封面无法读取，将使用兜底封面"
            result["preview_path"] = preview_path
            self.messages.put(("cover_probe", (serial, result)))

        threading.Thread(target=worker, daemon=True).start()

    def _refresh_cover_labels(self) -> None:
        self.cover_probe_var.set(self.embedded_message)
        if not self.include_cover_var.get():
            source = "已关闭封面打包"
            preview = None
            placeholder = "本次不打包封面"
        elif self.cover_var.get().strip():
            preview = Path(self.cover_var.get().strip())
            source = f"显式封面 · {preview.name}"
            placeholder = "显式封面无法预览"
        elif self.embedded_cover is True:
            source = "音频内嵌封面"
            preview = self.embedded_cover_preview
            placeholder = "正在读取内嵌封面"
        elif self.cover_probe_pending:
            source = "等待内嵌封面检测结果"
            preview = None
            placeholder = "正在检测内嵌封面…"
        elif self.fallback_cover is not None:
            source = f"image 目录兜底 · {self.fallback_cover.name}"
            preview = self.fallback_cover
            placeholder = "兜底封面无法预览"
        else:
            source = "自动生成圆形渐变封面"
            preview = None
            placeholder = "生成时自动创建封面"
        self.cover_source_var.set(source)
        self._set_cover_preview(preview, placeholder)

    def _toggle_threshold(self) -> None:
        self.threshold_entry.configure(
            state="normal" if self.manual_threshold_var.get() else "disabled")

    def _toggle_bpm(self) -> None:
        self.bpm_entry.configure(
            state="normal" if self.bpm_mode_var.get() == "manual" else "disabled")

    def _toggle_demo(self) -> None:
        state = "normal" if self.make_demo_var.get() else "disabled"
        self.demo_start_entry.configure(state=state)
        self.demo_seconds_entry.configure(state=state)

    @staticmethod
    def _finite_float(text: str, label: str, *, minimum: float | None = None,
                      maximum: float | None = None) -> float:
        try:
            value = float(text)
        except ValueError as error:
            raise ValueError(f"{label} 必须是数字") from error
        if not (float("-inf") < value < float("inf")):
            raise ValueError(f"{label} 必须是有限数字")
        if minimum is not None and value < minimum:
            raise ValueError(f"{label} 不能小于 {minimum:g}")
        if maximum is not None and value > maximum:
            raise ValueError(f"{label} 不能大于 {maximum:g}")
        return value

    def _collect_options(self) -> GenerationOptions:
        audio = Path(self.audio_var.get().strip()).resolve()
        checkpoint = Path(self.checkpoint_var.get().strip()).resolve()
        if not audio.is_file():
            raise ValueError("请选择存在的音乐文件")
        if not checkpoint.is_file():
            raise ValueError("请选择存在的 v1.0 checkpoint")
        difficulty = int(self.difficulty_var.get())
        if difficulty not in (1, 2, 3):
            raise ValueError("模型难度必须是 1、2 或 3")
        output_directory = Path(self.output_dir_var.get().strip())
        if not self.output_dir_var.get().strip():
            raise ValueError("请选择输出路径")
        output = compose_output_path(
            output_directory, self.output_name_var.get(), difficulty)
        threshold = None
        if self.manual_threshold_var.get():
            threshold = self._finite_float(
                self.threshold_var.get(), "onset 阈值", minimum=0.000001,
                maximum=0.999999)
        bpm_mode = self.bpm_mode_var.get()
        bpm = None
        if bpm_mode == "manual":
            bpm = self._finite_float(self.bpm_var.get(), "BPM", minimum=0.000001,
                                     maximum=1000)
        cover = None
        if self.include_cover_var.get() and self.cover_var.get().strip():
            cover = Path(self.cover_var.get().strip()).resolve()
            if not cover.is_file():
                raise ValueError("选择的显式封面不存在")
        demo_start = self._finite_float(
            self.demo_start_var.get(), "试听起点", minimum=0)
        demo_seconds = self._finite_float(
            self.demo_seconds_var.get(), "试听长度", minimum=0.01)
        boss_gap = self._finite_float(
            self.boss_gap_var.get(), "Boss 控制最小间隔", minimum=0)
        cpu_threads = int(self.cpu_threads_var.get())
        if cpu_threads < 1:
            raise ValueError("CPU 线程数必须大于 0")
        if not self.play_level_var.get().strip():
            raise ValueError("显示等级不能为空")
        if not self.designer_var.get().strip():
            raise ValueError("谱师名不能为空")
        scene = scene_id_from_choice(self.scene_choice_var.get())
        self.scene_var.set(scene)
        return GenerationOptions(
            audio=audio, checkpoint=checkpoint, output=output,
            difficulty=difficulty, play_level=self.play_level_var.get().strip(),
            profile=self.profile_var.get(), threshold=threshold,
            bpm_mode=bpm_mode, bpm=bpm,
            title=self.title_var.get().strip(), artist=self.artist_var.get().strip(),
            level_designer=self.designer_var.get().strip(),
            scene=scene, speed=int(self.speed_var.get()),
            device=self.device_var.get(), cpu_threads=cpu_threads,
            event_ordering=self.event_ordering_var.get(),
            double_optimization=self.double_optimization_var.get(),
            obvious_error_repair=self.obvious_error_var.get(),
            optimization_matrix=self.matrix_var.get(),
            boss_control_min_gap=boss_gap,
            include_cover=self.include_cover_var.get(), cover=cover,
            make_demo=self.make_demo_var.get(), demo_start=demo_start,
            demo_seconds=demo_seconds, keep_sidecars=self.sidecars_var.get(),
            strict=self.strict_var.get(),
        )

    def _start_generation(self) -> None:
        if self.process is not None:
            return
        try:
            options = self._collect_options()
        except (ValueError, OSError) as error:
            messagebox.showerror("设置有误", str(error), parent=self.root)
            return
        command = build_generate_command(options)
        log_directory = SCRIPT_DIR / "logs"
        log_directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = log_directory / f"{options.output.stem}-{timestamp}.log"
        self.last_log_path = log_path
        self.last_result = None
        self.open_log_button.configure(state="normal")
        self._reset_result()
        self._update_progress(
            step=1, percent=1, stage="启动任务",
            message="正在创建本地推理进程",
        )
        self.cancel_requested = False
        self.generate_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")

        def worker() -> None:
            environment = os.environ.copy()
            environment["PYTHONUTF8"] = "1"
            environment["PYTHONUNBUFFERED"] = "1"
            environment["MUSECHART_UI_PROGRESS"] = "1"
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            try:
                with log_path.open("w", encoding="utf-8", newline="") as log:
                    log.write("MuseChartWeaver UI task\n")
                    log.write(subprocess.list2cmdline(command) + "\n\n")
                    process = subprocess.Popen(
                        command, cwd=str(SCRIPT_DIR), stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                        errors="replace", bufsize=1, env=environment,
                        creationflags=creationflags,
                    )
                    self.process = process
                    assert process.stdout is not None
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        event = parse_ui_event_line(line)
                        if event is not None:
                            self.messages.put(("ui_event", event))
                    return_code = process.wait()
                self.messages.put(("done", (return_code, options, log_path)))
            except Exception as error:  # UI boundary: render unexpected OS errors.
                self.messages.put(
                    ("failed", (str(error), options, log_path)))

        threading.Thread(target=worker, daemon=True).start()

    def _cancel_generation(self) -> None:
        process = self.process
        if process is None:
            return
        self.cancel_requested = True
        self.status_var.set("正在取消…")
        self.progress_stage_var.set("正在取消")
        self.progress_detail_var.set("正在安全终止推理进程")
        try:
            process.terminate()
        except OSError:
            pass

    def _poll_messages(self) -> None:
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "cover_probe":
                    serial, result = payload  # type: ignore[misc]
                    if serial == self.cover_probe_serial:
                        self.cover_probe_pending = False
                        self.embedded_cover = result["present"]
                        self.embedded_message = result["message"]
                        preview_path = result.get("preview_path")
                        self.embedded_cover_preview = (
                            Path(preview_path) if preview_path else None)
                        self._refresh_cover_labels()
                elif kind == "ui_event":
                    event = payload  # type: ignore[assignment]
                    self._apply_ui_event(event)
                elif kind == "done":
                    code, options, log_path = payload  # type: ignore[misc]
                    self._finish_task(int(code), options, log_path)
                elif kind == "failed":
                    error, options, log_path = payload  # type: ignore[misc]
                    self._finish_task(-1, options, log_path, str(error))
        except queue.Empty:
            pass
        self.root.after(100, self._poll_messages)

    def _reset_result(self) -> None:
        for variable in (
            self.result_bpm_var, self.result_events_var, self.result_holds_var,
            self.result_boss_var, self.result_repairs_var,
            self.result_elapsed_var,
        ):
            variable.set("—")
        self.result_cover_var.set("正在等待生成结果")
        self.result_output_var.set("任务完成后显示输出文件")

    def _update_progress(self, *, step: int, percent: float,
                         stage: str, message: str) -> None:
        percent = max(0.0, min(100.0, float(percent)))
        step = max(1, min(5, int(step)))
        self.progress_var.set(percent)
        self.progress_percent_var.set(f"{percent:.0f}%")
        self.progress_stage_var.set(stage)
        self.progress_detail_var.set(message)
        self.status_var.set(stage)
        if not hasattr(self, "stage_labels"):
            return
        for index, label in enumerate(self.stage_labels, 1):
            if percent >= 100 or index < step:
                background = THEME["cyan_deep"]
                foreground = THEME["text"]
            elif index == step:
                background = THEME["primary"]
                foreground = THEME["text"]
            else:
                background = THEME["surface_raised"]
                foreground = THEME["text_muted"]
            label.configure(background=background, foreground=foreground)

    def _apply_ui_event(self, event: dict) -> None:
        kind = str(event.get("kind", ""))
        if kind in {"progress", "result"}:
            self._update_progress(
                step=int(event.get("step", 1)),
                percent=float(event.get("percent", 0)),
                stage=str(event.get("stage", "正在生成")),
                message=str(event.get("message", "")),
            )
        if kind == "result":
            self.last_result = event
            self._apply_result_summary(event)

    def _apply_result_summary(self, summary: dict) -> None:
        outputs = summary.get("outputs")
        if not isinstance(outputs, list) or not outputs:
            return
        representative = outputs[-1] if len(outputs) > 1 else outputs[0]
        first = representative if isinstance(representative, dict) else {}
        bpm = float(summary.get("bpm") or 0)
        bpm_source = BPM_SOURCE_LABELS.get(
            str(summary.get("bpm_source")), "模型坐标")
        self.result_bpm_var.set(
            f"{bpm:g} · {bpm_source}" if bpm > 0 else "未知")
        packaged = int(first.get("packaged_events") or 0)
        self.result_events_var.set(
            f"{packaged} × {len(outputs)} 版本"
            if len(outputs) > 1 else str(packaged))
        self.result_holds_var.set(str(int(first.get("holds") or 0)))
        self.result_boss_var.set(str(int(first.get("boss_controls") or 0)))
        repairs = sum(int(first.get(key) or 0) for key in (
            "double_repair_onsets",
            "obvious_duplicate_onsets",
            "obvious_duration_conflicts",
            "idle_boss_visits_removed",
        ))
        self.result_repairs_var.set(str(repairs))
        elapsed = float(summary.get("elapsed_seconds") or 0)
        self.result_elapsed_var.set(f"{elapsed:.1f} 秒")
        cover = first.get("cover") if isinstance(first.get("cover"), dict) else {}
        cover_source = COVER_SOURCE_LABELS.get(
            str(cover.get("source")), "封面状态未知")
        self.result_cover_var.set(f"封面 · {cover_source}")
        if len(outputs) > 1:
            self.result_output_var.set(
                f"已生成 {len(outputs)} 个版本 · {Path(first.get('output', '')).parent}")
        else:
            self.result_output_var.set(str(first.get("output", "")))

    def _finish_task(self, return_code: int,
                     options: GenerationOptions | None,
                     log_path: Path | None,
                     error: str | None = None) -> None:
        self.process = None
        self.generate_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        if log_path is not None:
            self.last_log_path = log_path
            self.open_log_button.configure(state="normal")
        if return_code == 0:
            assert options is not None
            if self.last_result is None:
                self._update_progress(
                    step=5, percent=100, stage="生成完成",
                    message="MDM 已写入输出目录",
                )
                self.result_output_var.set(
                    str(options.output.parent if options.optimization_matrix
                        else options.output))
        elif self.cancel_requested and error is None:
            self._update_progress(
                step=1, percent=0, stage="任务已取消",
                message="没有继续生成谱面",
            )
        else:
            self._update_progress(
                step=1, percent=self.progress_var.get(),
                stage="生成失败",
                message="请打开任务日志查看详细原因",
            )
            detail = error or f"推理进程退出码：{return_code}"
            if log_path is not None:
                detail += f"\n\n详细日志：\n{log_path}"
            messagebox.showerror(
                "生成失败", detail, parent=self.root)
        self.cancel_requested = False

    def _open_last_log(self) -> None:
        path = self.last_log_path
        if path is None or not path.is_file():
            messagebox.showinfo(
                "暂无日志", "完成一次生成任务后可查看详细日志。",
                parent=self.root)
            return
        self._open_path(path)

    def _open_path(self, path: Path) -> None:
        try:
            if os.name == "nt":
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(("open", str(path)))
            else:
                subprocess.Popen(("xdg-open", str(path)))
        except OSError as error:
            messagebox.showerror("无法打开", str(error), parent=self.root)

    def _open_output(self) -> None:
        text = self.output_dir_var.get().strip()
        directory = Path(text) if text else SCRIPT_DIR / "output"
        directory.mkdir(parents=True, exist_ok=True)
        self._open_path(directory)

    def _close(self) -> None:
        if self.process is not None:
            if not messagebox.askyesno(
                    "任务仍在运行", "关闭窗口会终止当前推理，确定关闭吗？",
                    parent=self.root):
                return
            try:
                self.process.terminate()
            except OSError:
                pass
        preview_root = self.preview_directory.resolve()
        temp_root = Path(tempfile.gettempdir()).resolve()
        if preview_root.parent == temp_root and preview_root.name.startswith(
                "musechart-ui-"):
            shutil.rmtree(preview_root, ignore_errors=True)
        self.root.destroy()


def main() -> int:
    root = tk.Tk()
    MuseChartApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
