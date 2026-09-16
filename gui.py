#!/usr/bin/env python3
"""Local Tk desktop UI for the MuseChart v3.2 inference pipeline."""
from __future__ import annotations

import os
import queue
import re
import shlex
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox
import tkinter as tk
from tkinter import scrolledtext, ttk

from musechart_runtime.mdm import find_default_cover, inspect_embedded_cover


SCRIPT_DIR = Path(__file__).resolve().parent
AUDIO_TYPES = (
    ("音频文件", "*.mp3 *.wav *.ogg *.flac *.m4a *.aac *.opus"),
    ("所有文件", "*.*"),
)
IMAGE_TYPES = (
    ("图片文件", "*.png *.jpg *.jpeg *.webp *.bmp *.gif"),
    ("所有文件", "*.*"),
)


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
        self.root.title("MuseChart v1.0 · 谱面生成器")
        self.root.geometry("1180x780")
        self.root.minsize(980, 680)
        self.process: subprocess.Popen[str] | None = None
        self.cancel_requested = False
        self.messages: queue.Queue[tuple[str, object]] = queue.Queue()
        self.cover_probe_serial = 0
        self.embedded_cover: bool | None = None
        self.cover_probe_pending = False
        self.embedded_message = "选择音频后自动检测内嵌封面"
        self.fallback_cover = find_default_cover(SCRIPT_DIR / "image")
        self.name_is_automatic = True

        self._make_variables()
        self._configure_style()
        self._build_layout()
        self._bind_events()
        self._refresh_cover_labels()
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
        self.designer_var = tk.StringVar(value="MuseChart AI")
        self.difficulty_var = tk.StringVar(value="2")
        self.play_level_var = tk.StringVar(value="7")
        self.profile_var = tk.StringVar(value="balanced")
        self.manual_threshold_var = tk.BooleanVar(value=False)
        self.threshold_var = tk.StringVar(value="0.45")
        self.bpm_mode_var = tk.StringVar(value="auto")
        self.bpm_var = tk.StringVar(value="120")
        self.scene_var = tk.StringVar(value="scene_01")
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
        self.status_var = tk.StringVar(value="就绪")

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        background = "#f4f6fb"
        panel = "#ffffff"
        accent = "#6c5ce7"
        self.root.configure(background=background)
        style.configure("App.TFrame", background=background)
        style.configure("Panel.TFrame", background=panel)
        style.configure("Panel.TLabelframe", background=panel, padding=12)
        style.configure("Panel.TLabelframe.Label", background=panel,
                        foreground="#22263a", font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("TLabel", font=("Microsoft YaHei UI", 9))
        style.configure("Title.TLabel", background=background, foreground="#22263a",
                        font=("Microsoft YaHei UI", 18, "bold"))
        style.configure("Subtitle.TLabel", background=background, foreground="#6b7280",
                        font=("Microsoft YaHei UI", 9))
        style.configure("Status.TLabel", background=panel, foreground="#475569",
                        font=("Microsoft YaHei UI", 9))
        style.configure("PanelHeading.TLabel", background=panel,
                        foreground="#22263a",
                        font=("Microsoft YaHei UI", 12, "bold"))
        style.configure("Accent.TButton", font=("Microsoft YaHei UI", 10, "bold"),
                        foreground="white", background=accent, padding=(16, 8))
        style.map("Accent.TButton", background=[("active", "#5847d6"),
                                                ("disabled", "#aaa4d9")])
        style.configure("Repair.TCheckbutton", background=panel,
                        font=("Microsoft YaHei UI", 10, "bold"))

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, style="App.TFrame", padding=16)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="MuseChart v3.2 本机谱面生成器",
                  style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            outer,
            text="选择音频与模型，配置三类独立后处理，然后在本机直接生成 MDM。",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(2, 12))

        pane = ttk.Panedwindow(outer, orient="horizontal")
        pane.pack(fill="both", expand=True)
        settings_panel = ttk.Frame(pane, style="Panel.TFrame", padding=8)
        task_panel = ttk.Frame(pane, style="Panel.TFrame", padding=12)
        pane.add(settings_panel, weight=3)
        pane.add(task_panel, weight=2)

        notebook = ttk.Notebook(settings_panel)
        notebook.pack(fill="both", expand=True)
        input_tab = ttk.Frame(notebook, style="Panel.TFrame", padding=14)
        chart_tab = ttk.Frame(notebook, style="Panel.TFrame", padding=14)
        repair_tab = ttk.Frame(notebook, style="Panel.TFrame", padding=14)
        notebook.add(input_tab, text="输入与封面")
        notebook.add(chart_tab, text="谱面设置")
        notebook.add(repair_tab, text="优化与输出")
        self._build_input_tab(input_tab)
        self._build_chart_tab(chart_tab)
        self._build_repair_tab(repair_tab)
        self._build_task_panel(task_panel)

    @staticmethod
    def _entry_row(parent: ttk.Frame, row: int, label: str,
                   variable: tk.StringVar, *, browse=None, clear=None) -> ttk.Entry:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=7)
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", padx=(10, 8), pady=7)
        if browse is not None:
            ttk.Button(parent, text="选择…", command=browse).grid(
                row=row, column=2, sticky="ew", pady=7)
        if clear is not None:
            ttk.Button(parent, text="清除", command=clear).grid(
                row=row, column=3, sticky="ew", padx=(6, 0), pady=7)
        return entry

    def _build_input_tab(self, tab: ttk.Frame) -> None:
        tab.columnconfigure(1, weight=1)
        self.audio_entry = self._entry_row(
            tab, 0, "音乐文件", self.audio_var, browse=self._choose_audio)
        ttk.Label(tab, text="内嵌封面").grid(row=1, column=0, sticky="nw", pady=7)
        ttk.Label(tab, textvariable=self.cover_probe_var, wraplength=490,
                  style="Status.TLabel").grid(
            row=1, column=1, columnspan=3, sticky="w", padx=(10, 0), pady=7)
        self.cover_entry = self._entry_row(
            tab, 2, "显式封面", self.cover_var,
            browse=self._choose_cover, clear=self._clear_cover)
        ttk.Label(tab, text="实际封面来源").grid(row=3, column=0, sticky="nw", pady=7)
        ttk.Label(tab, textvariable=self.cover_source_var, wraplength=490,
                  style="Status.TLabel").grid(
            row=3, column=1, columnspan=3, sticky="w", padx=(10, 0), pady=7)
        self._entry_row(tab, 4, "模型 checkpoint", self.checkpoint_var,
                        browse=self._choose_checkpoint)
        self.output_name_entry = self._entry_row(
            tab, 5, "MDM 名称", self.output_name_var)
        self._entry_row(
            tab, 6, "输出路径", self.output_dir_var,
            browse=self._choose_output_directory)
        ttk.Label(tab, text="最终文件").grid(row=7, column=0, sticky="nw", pady=7)
        ttk.Label(tab, textvariable=self.output_preview_var, wraplength=490,
                  style="Status.TLabel").grid(
            row=7, column=1, columnspan=3, sticky="w", padx=(10, 0), pady=7)

        metadata = ttk.LabelFrame(tab, text="歌曲信息（留空则读取音频标签）",
                                  style="Panel.TLabelframe")
        metadata.grid(row=8, column=0, columnspan=4, sticky="ew", pady=(14, 0))
        metadata.columnconfigure(1, weight=1)
        ttk.Label(metadata, text="标题").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(metadata, textvariable=self.title_var).grid(
            row=0, column=1, sticky="ew", padx=(10, 0), pady=5)
        ttk.Label(metadata, text="作者").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(metadata, textvariable=self.artist_var).grid(
            row=1, column=1, sticky="ew", padx=(10, 0), pady=5)
        ttk.Label(metadata, text="谱师名").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(metadata, textvariable=self.designer_var).grid(
            row=2, column=1, sticky="ew", padx=(10, 0), pady=5)

    def _build_chart_tab(self, tab: ttk.Frame) -> None:
        tab.columnconfigure(1, weight=1)
        ttk.Label(tab, text="模型难度 / mapN").grid(row=0, column=0, sticky="w", pady=7)
        ttk.Combobox(tab, textvariable=self.difficulty_var,
                     values=("1", "2", "3"), state="readonly",
                     width=8).grid(row=0, column=1, sticky="w", padx=(10, 0), pady=7)
        ttk.Label(tab, text="显示等级").grid(row=1, column=0, sticky="w", pady=7)
        ttk.Entry(tab, textvariable=self.play_level_var, width=12).grid(
            row=1, column=1, sticky="w", padx=(10, 0), pady=7)
        ttk.Label(tab, text="事件密度档").grid(row=2, column=0, sticky="w", pady=7)
        ttk.Combobox(tab, textvariable=self.profile_var,
                     values=("casual", "balanced", "aggressive"),
                     state="readonly", width=16).grid(
            row=2, column=1, sticky="w", padx=(10, 0), pady=7)

        threshold_frame = ttk.Frame(tab, style="Panel.TFrame")
        threshold_frame.grid(row=3, column=0, columnspan=2, sticky="w", pady=7)
        ttk.Checkbutton(threshold_frame, text="手动覆盖 onset 阈值",
                        variable=self.manual_threshold_var,
                        command=self._toggle_threshold).pack(side="left")
        self.threshold_entry = ttk.Entry(
            threshold_frame, textvariable=self.threshold_var, width=9,
            state="disabled")
        self.threshold_entry.pack(side="left", padx=(10, 0))

        bpm_box = ttk.LabelFrame(tab, text="BPM 条件", style="Panel.TLabelframe")
        bpm_box.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(12, 8))
        ttk.Radiobutton(bpm_box, text="自动读取/检测", value="auto",
                        variable=self.bpm_mode_var,
                        command=self._toggle_bpm).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(bpm_box, text="手动", value="manual",
                        variable=self.bpm_mode_var,
                        command=self._toggle_bpm).grid(row=0, column=1, sticky="w", padx=(16, 0))
        self.bpm_entry = ttk.Entry(bpm_box, textvariable=self.bpm_var,
                                   width=10, state="disabled")
        self.bpm_entry.grid(row=0, column=2, sticky="w", padx=(8, 0))
        ttk.Radiobutton(bpm_box, text="未知 BPM", value="unknown",
                        variable=self.bpm_mode_var,
                        command=self._toggle_bpm).grid(row=0, column=3, sticky="w", padx=(16, 0))

        ttk.Label(tab, text="场景").grid(row=5, column=0, sticky="w", pady=7)
        ttk.Entry(tab, textvariable=self.scene_var).grid(
            row=5, column=1, sticky="ew", padx=(10, 0), pady=7)
        ttk.Label(tab, text="物件速度").grid(row=6, column=0, sticky="w", pady=7)
        ttk.Combobox(tab, textvariable=self.speed_var, values=("1", "2", "3"),
                     state="readonly", width=8).grid(
            row=6, column=1, sticky="w", padx=(10, 0), pady=7)
        ttk.Label(tab, text="推理设备").grid(row=7, column=0, sticky="w", pady=7)
        ttk.Combobox(tab, textvariable=self.device_var,
                     values=("auto", "cuda", "cpu"), state="readonly",
                     width=12).grid(row=7, column=1, sticky="w", padx=(10, 0), pady=7)
        ttk.Label(tab, text="CPU 线程").grid(row=8, column=0, sticky="w", pady=7)
        ttk.Spinbox(tab, from_=1, to=max(1, os.cpu_count() or 1),
                    textvariable=self.cpu_threads_var, width=8).grid(
            row=8, column=1, sticky="w", padx=(10, 0), pady=7)

    def _build_repair_tab(self, tab: ttk.Frame) -> None:
        tab.columnconfigure(0, weight=1)
        items = (
            ("事件秩序化", self.event_ordering_var,
             "修复局部 A-B-A 跳脱怪、单独双轨怪，并检查 Boss 控制顺序。"),
            ("双轨修复", self.double_optimization_var,
             "将双轨普通怪改为经典双轨，并规范 Boss 双轨攻击；可按个人偏好关闭。"),
            ("双音符/红心/齿轮及冲突修复（强烈推荐开启）", self.obvious_error_var,
             "修剪同刻重复物件、长按/连打区间冲突，并移除无攻击行为的 Boss 空转段。"),
        )
        for row, (title, variable, description) in enumerate(items):
            box = ttk.LabelFrame(tab, text=str(row + 1), style="Panel.TLabelframe")
            box.grid(row=row, column=0, sticky="ew", pady=(0, 10))
            ttk.Checkbutton(box, text=title, variable=variable,
                            style="Repair.TCheckbutton").pack(anchor="w")
            ttk.Label(box, text=description, wraplength=570,
                      style="Status.TLabel").pack(anchor="w", pady=(5, 0))

        output_box = ttk.LabelFrame(tab, text="输出选项", style="Panel.TLabelframe")
        output_box.grid(row=3, column=0, sticky="ew", pady=(6, 0))
        ttk.Checkbutton(output_box, text="生成三项优化的全部 8 种组合",
                        variable=self.matrix_var).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Checkbutton(output_box, text="打包封面", variable=self.include_cover_var,
                        command=self._refresh_cover_labels).grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Checkbutton(output_box, text="生成试听 demo.ogg",
                        variable=self.make_demo_var,
                        command=self._toggle_demo).grid(row=1, column=1, sticky="w", padx=(18, 0), pady=(8, 0))
        ttk.Checkbutton(output_box, text="保留 events/report JSON",
                        variable=self.sidecars_var).grid(row=1, column=2, sticky="w", padx=(18, 0), pady=(8, 0))
        ttk.Checkbutton(output_box, text="严格导出检查",
                        variable=self.strict_var).grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Label(output_box, text="Boss 控制最小间隔").grid(
            row=3, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(output_box, textvariable=self.boss_gap_var, width=8).grid(
            row=3, column=1, sticky="w", padx=(8, 0), pady=(10, 0))
        ttk.Label(output_box, text="秒").grid(row=3, column=1, sticky="w", padx=(78, 0), pady=(10, 0))
        ttk.Label(output_box, text="试听起点 / 长度").grid(
            row=4, column=0, sticky="w", pady=(10, 0))
        demo_values = ttk.Frame(output_box, style="Panel.TFrame")
        demo_values.grid(row=4, column=1, columnspan=2, sticky="w", pady=(10, 0))
        self.demo_start_entry = ttk.Entry(
            demo_values, textvariable=self.demo_start_var, width=8)
        self.demo_start_entry.pack(side="left")
        ttk.Label(demo_values, text=" / ").pack(side="left")
        self.demo_seconds_entry = ttk.Entry(
            demo_values, textvariable=self.demo_seconds_var, width=8)
        self.demo_seconds_entry.pack(side="left")
        ttk.Label(demo_values, text=" 秒").pack(side="left")

    def _build_task_panel(self, panel: ttk.Frame) -> None:
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(3, weight=1)
        ttk.Label(panel, text="任务状态", style="PanelHeading.TLabel").grid(
            row=0, column=0, sticky="w")
        ttk.Label(panel, textvariable=self.status_var, style="Status.TLabel").grid(
            row=1, column=0, sticky="w", pady=(4, 8))
        self.progress = ttk.Progressbar(panel, mode="indeterminate")
        self.progress.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        self.log = scrolledtext.ScrolledText(
            panel, wrap="word", font=("Consolas", 9), background="#171923",
            foreground="#e5e7eb", insertbackground="white", relief="flat",
            padx=10, pady=10, state="disabled")
        self.log.grid(row=3, column=0, sticky="nsew")
        buttons = ttk.Frame(panel, style="Panel.TFrame")
        buttons.grid(row=4, column=0, sticky="ew", pady=(12, 0))
        buttons.columnconfigure(0, weight=1)
        self.generate_button = ttk.Button(
            buttons, text="开始生成", style="Accent.TButton",
            command=self._start_generation)
        self.generate_button.grid(row=0, column=0, sticky="ew")
        self.cancel_button = ttk.Button(
            buttons, text="取消", command=self._cancel_generation, state="disabled")
        self.cancel_button.grid(row=0, column=1, padx=(8, 0))
        ttk.Button(buttons, text="打开输出目录", command=self._open_output).grid(
            row=0, column=2, padx=(8, 0))
        ttk.Button(buttons, text="清空日志", command=self._clear_log).grid(
            row=0, column=3, padx=(8, 0))

    def _bind_events(self) -> None:
        self.audio_entry.bind("<FocusOut>", lambda _event: self._audio_changed())
        self.cover_entry.bind("<FocusOut>", lambda _event: self._refresh_cover_labels())
        self.output_name_entry.bind("<Key>", lambda _event: self._mark_name_manual())
        self.output_name_var.trace_add("write", lambda *_: self._update_output_preview())
        self.output_dir_var.trace_add("write", lambda *_: self._update_output_preview())
        self.difficulty_var.trace_add("write", lambda *_: self._update_output_preview())
        self.cover_var.trace_add("write", lambda *_: self._refresh_cover_labels())

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
            title="选择 v3.2 checkpoint",
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
            self.messages.put(("cover_probe", (serial, result)))

        threading.Thread(target=worker, daemon=True).start()

    def _refresh_cover_labels(self) -> None:
        self.cover_probe_var.set(self.embedded_message)
        if not self.include_cover_var.get():
            source = "已关闭封面打包"
        elif self.cover_var.get().strip():
            source = f"显式封面：{Path(self.cover_var.get().strip()).name}"
        elif self.embedded_cover is True:
            source = "音频内嵌封面"
        elif self.cover_probe_pending:
            source = "等待内嵌封面检测结果"
        elif self.fallback_cover is not None:
            source = f"image 目录兜底：{self.fallback_cover.name}"
        else:
            source = "自动生成圆形渐变封面"
        self.cover_source_var.set(source)

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
            raise ValueError("请选择存在的 v3.2 checkpoint")
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
        if not self.scene_var.get().strip():
            raise ValueError("场景不能为空")
        return GenerationOptions(
            audio=audio, checkpoint=checkpoint, output=output,
            difficulty=difficulty, play_level=self.play_level_var.get().strip(),
            profile=self.profile_var.get(), threshold=threshold,
            bpm_mode=bpm_mode, bpm=bpm,
            title=self.title_var.get().strip(), artist=self.artist_var.get().strip(),
            level_designer=self.designer_var.get().strip(),
            scene=self.scene_var.get().strip(), speed=int(self.speed_var.get()),
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
        self._append_log("\n=== 开始新任务 ===\n")
        display = subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)
        self._append_log(display + "\n\n")
        self.status_var.set("正在加载模型并推理…")
        self.cancel_requested = False
        self.generate_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.progress.start(10)

        def worker() -> None:
            environment = os.environ.copy()
            environment["PYTHONUTF8"] = "1"
            environment["PYTHONUNBUFFERED"] = "1"
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            try:
                process = subprocess.Popen(
                    command, cwd=str(SCRIPT_DIR), stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                    errors="replace", bufsize=1, env=environment,
                    creationflags=creationflags,
                )
                self.process = process
                assert process.stdout is not None
                for line in process.stdout:
                    self.messages.put(("log", line.replace("\r", "")))
                return_code = process.wait()
                self.messages.put(("done", (return_code, options)))
            except Exception as error:  # UI boundary: render unexpected OS errors.
                self.messages.put(("failed", str(error)))

        threading.Thread(target=worker, daemon=True).start()

    def _cancel_generation(self) -> None:
        process = self.process
        if process is None:
            return
        self.cancel_requested = True
        self.status_var.set("正在取消…")
        try:
            process.terminate()
        except OSError:
            pass

    def _poll_messages(self) -> None:
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "cover_probe":
                    serial, result = payload  # type: ignore[misc]
                    if serial == self.cover_probe_serial:
                        self.cover_probe_pending = False
                        self.embedded_cover = result["present"]
                        self.embedded_message = result["message"]
                        self._refresh_cover_labels()
                elif kind == "done":
                    code, options = payload  # type: ignore[misc]
                    self._finish_task(int(code), options)
                elif kind == "failed":
                    self._finish_task(-1, None, str(payload))
        except queue.Empty:
            pass
        self.root.after(100, self._poll_messages)

    def _finish_task(self, return_code: int,
                     options: GenerationOptions | None,
                     error: str | None = None) -> None:
        self.process = None
        self.progress.stop()
        self.generate_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        if return_code == 0:
            self.status_var.set("生成完成")
            assert options is not None
            if options.optimization_matrix:
                location = f"已在以下目录生成 8 种优化组合：\n{options.output.parent}"
            else:
                location = f"MDM 已生成：\n{options.output}"
            sidecars = ("同目录下可查看 events/report JSON。"
                        if options.keep_sidecars else "本次未生成 sidecar JSON。")
            messagebox.showinfo(
                "生成完成",
                f"{location}\n\n{sidecars}",
                parent=self.root)
        elif self.cancel_requested and error is None:
            self.status_var.set("任务已取消")
            self._append_log("\n任务已取消。\n")
        else:
            self.status_var.set("生成失败，请查看日志")
            if error:
                self._append_log(f"\n启动失败：{error}\n")
            messagebox.showerror(
                "生成失败", error or f"推理进程退出码：{return_code}",
                parent=self.root)
        self.cancel_requested = False

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _open_output(self) -> None:
        text = self.output_dir_var.get().strip()
        directory = Path(text) if text else SCRIPT_DIR / "output"
        directory.mkdir(parents=True, exist_ok=True)
        try:
            if os.name == "nt":
                os.startfile(directory)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(("open", str(directory)))
            else:
                subprocess.Popen(("xdg-open", str(directory)))
        except OSError as error:
            messagebox.showerror("无法打开目录", str(error), parent=self.root)

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
        self.root.destroy()


def main() -> int:
    root = tk.Tk()
    MuseChartApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
