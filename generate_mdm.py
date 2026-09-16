#!/usr/bin/env python3
"""One-command local pipeline: audio -> MuseChart events -> playable MDM."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
import numpy as np  # noqa: E402
import torch  # noqa: E402

from musechart_runtime.audio import load_audio  # noqa: E402
from musechart_runtime.mdm import (build_bms_text, build_info_json,  # noqa: E402
                                   find_default_cover, prepare_chart_events,
                                   write_mdm)
from musechart_runtime.model import MuseChartModel  # noqa: E402
from musechart_runtime.pipeline import predict_difficulty  # noqa: E402
from musechart_runtime.postprocessing import apply_optimizations  # noqa: E402
from musechart_runtime.schema import SCHEMA_VERSION, TRAINER_VERSION, require_v3  # noqa: E402
from musechart_runtime.tempo import estimate_tempo  # noqa: E402
from musechart_runtime.utils import write_json  # noqa: E402


PROFILE_THRESHOLD_OFFSETS = {
    "casual": 0.10,
    "balanced": 0.0,
    "aggressive": -0.10,
}
OPTIMIZATION_VARIANTS = (
    ("opt_none", False, False, False),
    ("opt_ordering", True, False, False),
    ("opt_double", False, True, False),
    ("opt_obvious", False, False, True),
    ("opt_ordering_double", True, True, False),
    ("opt_ordering_obvious", True, False, True),
    ("opt_double_obvious", False, True, True),
    ("opt_all", True, True, True),
)


def default_checkpoint() -> Path:
    checkpoint_dir = SCRIPT_DIR / "checkpoints"
    for name in ("MuseChart_v1.0.pt", "best_playability.pt", "best.pt",
                 "best_playability-55.pt", "best_label.pt", "best_timing.pt",
                 "last.pt"):
        candidate = checkpoint_dir / name
        if candidate.is_file():
            return candidate
    return checkpoint_dir / "MuseChart_v1.0.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="本机一键生成：MP3/WAV/FLAC/OGG/... -> 模型推理 -> 可玩性后处理 -> CustomAlbums MDM")
    parser.add_argument("audio", nargs="?", type=Path, default=SCRIPT_DIR / "no title.mp3",
                        help="输入音乐；省略时使用 inference/no title.mp3")
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint(),
                        help="模型 checkpoint；默认使用 checkpoints/MuseChart_v1.0.pt")
    parser.add_argument("--output", type=Path, help="输出 .mdm；默认保存到 inference/output/")
    parser.add_argument("--difficulty", type=int, choices=(1, 2, 3), default=2,
                        help="模型难度条件及 mapN，只能选择 1、2、3，默认 2")
    parser.add_argument("--play-level", default="7", help="选曲界面显示等级，例如 7 或 10+")
    parser.add_argument("--profile", choices=tuple(PROFILE_THRESHOLD_OFFSETS),
                        default="balanced",
                        help="以 checkpoint 校准阈值为中心：casual +0.10，balanced 不变，aggressive -0.10")
    parser.add_argument("--threshold", type=float, help="覆盖 profile 的 onset 阈值")
    parser.add_argument("--boss-threshold", type=float, default=0.85,
                        help="兼容旧命令；v3.2 使用状态序列解码，此参数不再参与推理")
    parser.add_argument("--bpm", type=float,
                        help="省略则自动检测；正数为手动覆盖；0 明确禁用检测和模型 BPM 条件")
    parser.add_argument("--bpm-min", type=float, default=70.0, help="自动检测搜索下限，默认 70")
    parser.add_argument("--bpm-max", type=float, default=240.0, help="自动检测搜索上限，默认 240")
    parser.add_argument("--coordinate-bpm", type=float, default=120.0,
                        help="明确使用 --bpm 0 时的 BMS 时间坐标，默认 120，不改变事件时刻")
    parser.add_argument("--title", help="默认读取音频 title 标签，否则使用文件名")
    parser.add_argument("--romanized-title")
    parser.add_argument("--artist", help="默认读取音频 artist 标签")
    parser.add_argument("--level-designer", default="MuseChart AI")
    parser.add_argument("--scene", default="scene_01")
    parser.add_argument("--speed", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--double-optimization", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="用户喜好项：普通怪双轨改为 0E，并规范 Boss 双轨攻击")
    ordering = parser.add_mutually_exclusive_group()
    ordering.add_argument("--event-ordering", "--chart-optimization",
                          dest="event_ordering", action="store_true",
                          help="美化项：修复孤立普通怪/单独0E并检查Boss控制顺序")
    ordering.add_argument("--no-event-ordering", "--no-chart-optimization",
                          dest="event_ordering", action="store_false",
                          help="关闭事件秩序化；--no-chart-optimization 是旧命令别名")
    parser.set_defaults(event_ordering=True)
    obvious = parser.add_mutually_exclusive_group()
    obvious.add_argument("--obvious-error-repair", "--special-double-repair",
                         dest="obvious_error_repair", action="store_true",
                         help="明显错误项：修复重复物件、持续事件冲突和 Boss 空转")
    obvious.add_argument("--no-obvious-error-repair", "--no-special-double-repair",
                         dest="obvious_error_repair", action="store_false",
                         help="关闭重复物件、持续事件冲突和 Boss 空转修复")
    parser.set_defaults(obvious_error_repair=True)
    parser.add_argument("--boss-control-min-gap", type=float, default=0.30,
                        help="谱面优化启用时 Boss 控制的最小间隔秒数，默认 0.30")
    parser.add_argument("--optimization-matrix", "--eight-variants", "--four-variants",
                        action="store_true",
                        help="一次模型推理生成三项开关的全部八种组合")
    parser.add_argument("--max-hold", type=float, default=8.0)
    parser.add_argument("--resolution", type=int, default=960)
    parser.add_argument("--overlap-seconds", type=float)
    parser.add_argument("--cover", type=Path,
                        help="封面图片；统一居中裁剪为透明背景的 440x440 圆形 PNG")
    parser.add_argument("--no-cover", action="store_true",
                        help="不写入封面；默认依次使用内嵌封面、image目录图片、自动生成封面")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--no-demo", action="store_true")
    parser.add_argument("--demo-start", type=float, default=0.0)
    parser.add_argument("--demo-seconds", type=float, default=15.0)
    parser.add_argument("--no-sidecars", action="store_true",
                        help="只输出 MDM，不保留 events.json 和 report.json")
    parser.add_argument("--strict", action="store_true")
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe_tags(path: Path) -> dict[str, str]:
    executable = shutil.which("ffprobe")
    if executable is None:
        return {}
    completed = subprocess.run([
        executable, "-v", "error", "-show_entries", "format_tags=title,artist,bpm,TBPM",
        "-of", "json", str(path),
    ], check=False, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if completed.returncode:
        return {}
    try:
        tags = json.loads(completed.stdout).get("format", {}).get("tags", {})
    except (json.JSONDecodeError, AttributeError):
        return {}
    return {str(key).lower(): str(value) for key, value in tags.items()}


def _ffmpeg_decode(path: Path, output: Path, sample_rate: int) -> None:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("soundfile 无法读取此音频，且没有找到 ffmpeg 作为后备解码器")
    completed = subprocess.run([
        executable, "-hide_banner", "-loglevel", "error", "-y", "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_f32le", str(output),
    ], check=False, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if completed.returncode:
        raise RuntimeError(f"ffmpeg 音频解码失败: {(completed.stderr or completed.stdout).strip()}")


def load_audio_flexible(path: Path, sample_rate: int, work_dir: Path) -> np.ndarray:
    try:
        return load_audio(path, sample_rate)
    except Exception as first_error:
        temporary = work_dir / f".{path.stem}.{uuid.uuid4().hex}.decode.wav"
        try:
            _ffmpeg_decode(path, temporary, sample_rate)
            return load_audio(temporary, sample_rate)
        except Exception as second_error:
            raise RuntimeError(f"无法读取音频。soundfile: {first_error}; ffmpeg: {second_error}") from second_error
        finally:
            if temporary.exists():
                temporary.unlink()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 --device cuda，但当前 PyTorch 没有可用 CUDA")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def tagged_bpm(tags: dict[str, str]) -> float | None:
    for key in ("tbpm", "bpm"):
        try:
            value = float(tags.get(key, ""))
        except ValueError:
            continue
        if math.isfinite(value) and value > 0:
            return value
    return None


def optimization_outputs(output: Path, *, matrix: bool,
                         event_ordering: bool,
                         double_optimization: bool,
                         obvious_error_repair: bool,
                         ) -> list[tuple[Path, str, bool, bool, bool]]:
    if not matrix:
        return [(output, "custom", event_ordering, double_optimization,
                 obvious_error_repair)]
    return [
        (output.with_name(f"{output.stem}_{name}{output.suffix}"),
         name, ordering_enabled, double_enabled, obvious_enabled)
        for name, ordering_enabled, double_enabled, obvious_enabled
        in OPTIMIZATION_VARIANTS
    ]


def main() -> int:
    args = build_parser().parse_args()
    started = time.perf_counter()
    audio_path = args.audio.resolve()
    checkpoint_path = args.checkpoint.resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"找不到输入音乐: {audio_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"找不到 checkpoint: {checkpoint_path}")
    if args.output is None:
        output = SCRIPT_DIR / "output" / f"{audio_path.stem}_map{args.difficulty}.mdm"
    else:
        output = args.output.resolve()
    if output.suffix.lower() != ".mdm":
        raise ValueError("--output 必须以 .mdm 结尾")
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.bpm is not None and (not 0 <= args.bpm <= 1000 or not math.isfinite(args.bpm)):
        raise ValueError("--bpm 必须是 0..1000 的有限数字")
    if not 20 <= args.bpm_min < args.bpm_max <= 1000:
        raise ValueError("--bpm-min/--bpm-max 搜索范围无效")
    if not 0 < args.coordinate_bpm <= 1000 or not math.isfinite(args.coordinate_bpm):
        raise ValueError("--coordinate-bpm 必须大于 0")
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads 必须大于 0")
    if not 0 < args.boss_threshold < 1:
        raise ValueError("--boss-threshold 必须位于 (0,1)")
    if args.boss_control_min_gap < 0 or not math.isfinite(args.boss_control_min_gap):
        raise ValueError("--boss-control-min-gap 必须是非负有限数字")
    if args.cover is not None and args.no_cover:
        raise ValueError("--cover 与 --no-cover 不能同时使用")

    device = choose_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)
    print(f"[1/5] 加载模型: {checkpoint_path.name} -> {device}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require_v3(checkpoint, "checkpoint")
    config, vocabulary = checkpoint["config"], checkpoint["vocabulary"]
    if args.difficulty > int(config["model"]["max_difficulty"]):
        raise ValueError("difficulty 超出 checkpoint 支持范围")
    calibrated = float(checkpoint["calibrated_threshold"])
    profile_offset = PROFILE_THRESHOLD_OFFSETS[args.profile]
    threshold = (args.threshold if args.threshold is not None
                 else min(0.99, max(0.01, calibrated + profile_offset)))
    if not 0 < threshold < 1:
        raise ValueError("推理阈值必须位于 (0,1)")
    overlap = (args.overlap_seconds if args.overlap_seconds is not None
               else float(config["inference"]["overlap_seconds"]))

    sample_rate = int(config["data"]["sample_rate"])
    print(f"[2/5] 解码音频: {audio_path.name} -> mono {sample_rate} Hz")
    audio = load_audio_flexible(audio_path, sample_rate, output.parent)
    duration_seconds = len(audio) / sample_rate
    if duration_seconds <= 0:
        raise ValueError("输入音乐为空")
    tags = probe_tags(audio_path)
    title = args.title or tags.get("title") or audio_path.stem
    artist = args.artist or tags.get("artist") or "Unknown Artist"
    tempo_analysis = None
    if args.bpm == 0:
        resolved_bpm = 0.0
        bpm_source = "disabled"
        beat_offset = 0.0
        print(f"      BPM 自动检测已禁用：模型使用 unknown BPM；BMS 仅用 {args.coordinate_bpm:g} 作为时间坐标")
    else:
        metadata_bpm = tagged_bpm(tags) if args.bpm is None else None
        hint = float(args.bpm) if args.bpm is not None else metadata_bpm
        tempo_analysis = estimate_tempo(
            audio, sample_rate, min_bpm=args.bpm_min, max_bpm=args.bpm_max,
            bpm_hint=hint)
        resolved_bpm = float(tempo_analysis["bpm"])
        beat_offset = float(tempo_analysis["beat_offset_seconds"])
        bpm_source = ("command_line" if args.bpm is not None else
                      "audio_metadata" if metadata_bpm is not None else "audio_analysis")
        candidates = ", ".join(
            f"{item['bpm']:g}" for item in tempo_analysis["candidates"][:3])
        if resolved_bpm > 0:
            print(f"      BPM={resolved_bpm:g} ({bpm_source}, confidence={tempo_analysis['confidence']:.2f}, "
                  f"candidates=[{candidates}])")
        else:
            print(f"      未检测到可靠 BPM：模型使用 unknown BPM；BMS 仅用 {args.coordinate_bpm:g} 作为时间坐标")
    bms_bpm = resolved_bpm if resolved_bpm > 0 else args.coordinate_bpm

    print(f"[3/5] v3.2 整曲状态解码: difficulty={args.difficulty}, "
          f"profile={args.profile}, threshold={threshold:g}")
    model = MuseChartModel(sample_rate=sample_rate,
                           hop_length=int(config["data"]["hop_length"]),
                           config=config["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    events = predict_difficulty(
        model=model, audio=audio, difficulty=args.difficulty, bpm=resolved_bpm,
        vocabulary=vocabulary, config=config, threshold=threshold,
        overlap_seconds=overlap, max_duration=float(config["train"]["max_duration_seconds"]),
        device=device, progress=True, beat_offset=beat_offset,
        beat_known=resolved_bpm > 0)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not events:
        raise RuntimeError("模型没有生成事件；请降低 --threshold 或改用 --profile aggressive")

    checkpoint_hash = sha256_file(checkpoint_path)
    fallback_cover = find_default_cover(SCRIPT_DIR / "image")
    raw_events = [dict(event) for event in events]
    variants = optimization_outputs(
        output, matrix=args.optimization_matrix,
        event_ordering=args.event_ordering,
        double_optimization=args.double_optimization,
        obvious_error_repair=args.obvious_error_repair)

    info = build_info_json(
        title=title, romanized_title=args.romanized_title or title, artist=artist,
        level_designer=args.level_designer, bpm=bms_bpm, scene=args.scene,
        difficulty=args.difficulty, play_level=str(args.play_level))

    print(f"[4/5] 生成 {len(variants)} 种优化组合: {len(raw_events)} 个原始模型事件")
    summaries = []
    for (variant_output, variant_name, ordering_enabled, double_enabled,
         obvious_enabled) in variants:
        optimized_events, postprocess_report = apply_optimizations(
            raw_events, event_ordering=ordering_enabled,
            double_optimization=double_enabled,
            obvious_error_repair=obvious_enabled,
            minimum_boss_control_gap=args.boss_control_min_gap,
            bpm=resolved_bpm)

        converted, conversion = prepare_chart_events(
            optimized_events, bpm=bms_bpm, resolution=args.resolution,
            confidence_threshold=0.0, boss_confidence_threshold=0.0,
            min_hold_seconds=0.10, max_hold_seconds=args.max_hold,
            audio_duration=duration_seconds, strict=args.strict,
            normalize_doubles=False)
        bms_text, placement = build_bms_text(
            converted, bpm=bms_bpm, scene=args.scene, speed=args.speed,
            difficulty=args.difficulty, play_level=str(args.play_level),
            title=title, artist=artist, level_designer=args.level_designer,
            resolution=args.resolution, strict=args.strict)
        conversion.update(placement)
        notes = [event for event in optimized_events if event["semantic_type"] != 0]
        boss_events = [event for event in optimized_events if event["semantic_type"] == 0]
        generation = {
            "schema_version": SCHEMA_VERSION,
            "format": "MDMods/CustomAlbums MDM-BMS",
            "pipeline": "audio_to_mdm_v3_2",
            "variant": variant_name,
            "source_audio": audio_path.name,
            "audio_duration_seconds": duration_seconds,
            "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_global_step": checkpoint.get("global_step"),
            "checkpoint_selection_dimension": checkpoint.get("selection_dimension"),
            "model_kind": checkpoint.get("model_kind"),
            "device": str(device),
            "torch_version": torch.__version__,
            "difficulty": args.difficulty,
            "profile": args.profile,
            "profile_threshold_offset": profile_offset,
            "threshold": threshold,
            "boss_decoder": "learned_state_viterbi",
            "legacy_boss_threshold_ignored": args.boss_threshold,
            "checkpoint_calibrated_threshold": calibrated,
            "requested_bpm": args.bpm,
            "model_bpm": resolved_bpm,
            "bpm_source": bpm_source,
            "tempo_analysis": tempo_analysis,
            "bms_coordinate_bpm": bms_bpm,
            "scene": args.scene,
            "speed": args.speed,
            "optimizations": {
                "event_ordering": ordering_enabled,
                "double": double_enabled,
                "obvious_error_repair": obvious_enabled,
            },
            "postprocess": postprocess_report,
            "conversion": conversion,
            "seconds_before_packaging": time.perf_counter() - started,
        }
        event_document = {
            "schema_version": SCHEMA_VERSION,
            "trainer_version": TRAINER_VERSION,
            "source": "MuseChart IBMS v3.2 local audio-to-MDM pipeline",
            "model_kind": checkpoint.get("model_kind"),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_global_step": checkpoint.get("global_step"),
            "checkpoint_selection_dimension": checkpoint.get("selection_dimension"),
            "audio_file": audio_path.name,
            "audio_duration_seconds": duration_seconds,
            "sample_rate": sample_rate,
            "map_name": variant_output.stem,
            "difficulty_index": args.difficulty,
            "bpm": resolved_bpm,
            "bpm_source": bpm_source,
            "tempo_analysis": tempo_analysis,
            "inference": {
                "profile": args.profile,
                "profile_threshold_offset": profile_offset,
                "threshold": threshold,
                "boss_decoder": "learned_state_viterbi",
                "legacy_boss_threshold_ignored": args.boss_threshold,
                "overlap_seconds": overlap,
                "scene": args.scene,
                "speed": args.speed,
                "event_ordering": ordering_enabled,
                "double_optimization": double_enabled,
                "obvious_error_repair": obvious_enabled,
                "beat_offset_seconds": beat_offset,
            },
            "postprocess": postprocess_report,
            "raw_predictions": raw_events,
            "note_count": len(notes),
            "notes": notes,
            "boss_event_count": len(boss_events),
            "boss_events": boss_events,
        }

        print(f"[5/5] 打包 {variant_output.name}: ordering={ordering_enabled}, "
              f"double={double_enabled}, obvious={obvious_enabled}")
        cover_report = write_mdm(
            variant_output, bms_text=bms_text, info=info, generation=generation,
            audio=audio_path, difficulty=args.difficulty, cover=args.cover,
            fallback_cover=fallback_cover,
            include_cover=not args.no_cover, cover_seed=f"{title}\n{artist}",
            make_demo=not args.no_demo, demo_start=args.demo_start,
            demo_seconds=args.demo_seconds)
        event_document["cover"] = cover_report
        if not args.no_sidecars:
            write_json(variant_output.with_suffix(".events.json"), event_document)
            write_json(variant_output.with_suffix(".report.json"), generation)
        summaries.append({
            "variant": variant_name,
            "output": str(variant_output),
            "event_ordering": ordering_enabled,
            "double_optimization": double_enabled,
            "obvious_error_repair": obvious_enabled,
            "model_events": len(raw_events),
            "packaged_events": conversion["placed_events"],
            "holds": conversion["hold_events"],
            "double_repair_onsets": postprocess_report[
                "double_optimization"]["optimized_onsets"],
            "ordinary_double_onsets": postprocess_report[
                "double_optimization"]["ordinary_double_onsets"],
            "boss_double_onsets": postprocess_report[
                "double_optimization"]["boss_double_onsets"],
            "obvious_duplicate_onsets": postprocess_report[
                "obvious_error_repair"]["repaired_onsets"],
            "obvious_duration_conflicts": postprocess_report[
                "obvious_error_repair"]["duration_conflicts_repaired"],
            "idle_boss_visits_removed": postprocess_report[
                "obvious_error_repair"]["idle_boss_visits_removed"],
            "boss_controls": len(boss_events),
            "cover": cover_report,
        })

    print(json.dumps({
        "status": "ok",
        "device": str(device),
        "duration_seconds": duration_seconds,
        "threshold": threshold,
        "boss_decoder": "learned_state_viterbi",
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": summaries,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
