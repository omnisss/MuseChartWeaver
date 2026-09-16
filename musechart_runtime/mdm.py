"""Export semantic MuseChart events to the CustomAlbums MDM/BMS format.

The implementation intentionally targets the public MDMods/CustomAlbums loader.
It uses a constant BPM as the coordinate system while preserving the model's
absolute event times to sub-millisecond precision at the default resolution.
"""
from __future__ import annotations

import json
import hashlib
import math
import os
import re
import shutil
import struct
import subprocess
import uuid
import zipfile
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import BOSS_CONTROL_IDS, BOSS_PLAY_IDS, GROUND_ONLY_IDS


SEMANTIC_DEFAULT_IBMS = {
    1: "01",  # small enemy
    2: "0H",  # gear
    3: "0F",  # hold
    4: "21",  # ghost / hidden enemy
    5: "11",  # boss melee (ground only)
    6: "22",  # heart
    7: "23",  # music note
    8: "0G",  # masher
}

SEMANTIC_ALLOWED_IBMS = {
    1: frozenset(("01", "02", "03", "04", "05", "06", "07", "08", "09",
                  "0A", "0B", "0C", "0D", "0E", "13", "14", "15")),
    2: frozenset(("0H", "18")),
    3: frozenset(("0F",)),
    4: frozenset(("21",)),
    5: frozenset(("11", "12")),
    6: frozenset(("22",)),
    7: frozenset(("23",)),
    8: frozenset(("0G", "16", "17")),
}

DURATION_TYPES = frozenset((3, 8))
LANE_CHANNELS = {0: ("12", "14"), 1: ("11", "13")}
BOSS_CHANNELS = ("15", "16", "18")
SCENE_PATTERN = re.compile(r"^scene_\d{2}$")


@dataclass(frozen=True)
class ChartEvent:
    source_index: int
    time: float
    lane: int
    semantic_type: int
    duration: float
    confidence: float
    ibms_id: str
    start_slot: int
    end_slot: int | None


def _finite_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 不是数字: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} 必须是有限数字")
    return number


def _quantize_slot(seconds: float, bpm: float, resolution: int) -> int:
    # A BMS measure is four quarter-note beats in the CustomAlbums loader.
    value = seconds * bpm * resolution / 240.0
    return int(math.floor(value + 0.5))


def _slot_seconds(slot: int, bpm: float, resolution: int) -> float:
    return slot * 240.0 / (bpm * resolution)


def _event_ibms(note: dict, semantic_type: int) -> tuple[str, bool]:
    value = str(note.get("ibms_id") or "").strip().upper()
    if value in SEMANTIC_ALLOWED_IBMS[semantic_type]:
        return value, False
    return SEMANTIC_DEFAULT_IBMS[semantic_type], bool(value)


def prepare_chart_events(
    notes: list[dict], *, bpm: float, resolution: int = 960,
    confidence_threshold: float = 0.0, boss_confidence_threshold: float = 0.0,
    min_hold_seconds: float = 0.10,
    max_hold_seconds: float = 8.0, audio_duration: float | None = None,
    strict: bool = False, normalize_doubles: bool = False,
) -> tuple[list[ChartEvent], dict]:
    """Validate, filter and quantize model events for practical playtesting."""
    bpm = _finite_number(bpm, "bpm")
    if bpm <= 0:
        raise ValueError("bpm 必须大于 0")
    if resolution < 24 or resolution > 7680:
        raise ValueError("resolution 必须位于 24..7680")
    if not 0 <= confidence_threshold < 1:
        raise ValueError("confidence_threshold 必须位于 [0,1)")
    if not 0 <= boss_confidence_threshold < 1:
        raise ValueError("boss_confidence_threshold 必须位于 [0,1)")
    if min_hold_seconds <= 0 or max_hold_seconds < min_hold_seconds:
        raise ValueError("长按长度范围无效")
    if audio_duration is not None:
        audio_duration = _finite_number(audio_duration, "audio_duration")
        if audio_duration <= 0:
            raise ValueError("audio_duration 必须大于 0")

    dropped: Counter[str] = Counter()
    replacements: Counter[str] = Counter()
    candidates: list[ChartEvent] = []
    timing_errors = []
    for index, note in enumerate(notes):
        try:
            time = _finite_number(note.get("time", note.get("tick")), f"notes[{index}].time")
            semantic = int(note["semantic_type"])
            lane = -1 if semantic == 0 else int(note["lane"])
            confidence = _finite_number(note.get("confidence", 1.0), f"notes[{index}].confidence")
            duration = _finite_number(note.get("duration", 0.0), f"notes[{index}].duration")
        except (KeyError, TypeError, ValueError) as exc:
            if strict:
                raise ValueError(f"无效事件 notes[{index}]: {exc}") from exc
            dropped["invalid_event"] += 1
            continue
        control_id = str(note.get("ibms_id") or "").upper()
        is_control = semantic == 0 and control_id in BOSS_CONTROL_IDS
        if time < 0 or (not is_control and (lane not in (0, 1) or semantic not in SEMANTIC_DEFAULT_IBMS)):
            if strict:
                raise ValueError(f"notes[{index}] 的时间、轨道或 semantic_type 无效")
            dropped["unsupported_event"] += 1
            continue
        if (semantic == 5 or control_id in GROUND_ONLY_IDS) and lane != 0:
            if strict:
                raise ValueError(f"notes[{index}] type 5 仅支持地面轨道")
            dropped["illegal_boss_lane"] += 1
            continue
        cutoff = boss_confidence_threshold if is_control else confidence_threshold
        if confidence < cutoff:
            dropped["below_boss_confidence_threshold" if is_control else "below_confidence_threshold"] += 1
            continue
        if audio_duration is not None and time >= audio_duration:
            if strict:
                raise ValueError(f"notes[{index}] 超出音频长度")
            dropped["outside_audio"] += 1
            continue

        if semantic in DURATION_TYPES:
            duration = min(max_hold_seconds, max(min_hold_seconds, duration))
            if audio_duration is not None:
                duration = min(duration, audio_duration - time)
            if duration <= 0:
                dropped["empty_hold"] += 1
                continue
        else:
            duration = 0.0

        start = _quantize_slot(time, bpm, resolution)
        end = None
        if duration:
            end = max(start + 1, _quantize_slot(time + duration, bpm, resolution))
        measure = start // resolution
        if measure > 999 or (end is not None and end // resolution > 999):
            if strict:
                raise ValueError(f"notes[{index}] 超过三位 BMS 小节编号上限")
            dropped["measure_overflow"] += 1
            continue
        ibms, replaced = (control_id, False) if is_control else _event_ibms(note, semantic)
        if replaced:
            replacements["incompatible_source_ibms"] += 1
        timing_errors.append(abs(_slot_seconds(start, bpm, resolution) - time) * 1000)
        candidates.append(ChartEvent(index, time, lane, semantic, duration, confidence,
                                     ibms, start, end))

    # One gameplay event per lane/onset; distinct Boss control IDs remain
    # independent. A quantization collision keeps the most confident source.
    by_lane_slot: dict[tuple[int, int, str], ChartEvent] = {}
    for event in sorted(candidates, key=lambda e: (-e.confidence, e.source_index)):
        key = (event.start_slot, event.lane,
               event.ibms_id if event.semantic_type == 0 else "")
        if key in by_lane_slot:
            if strict:
                raise ValueError(f"BMS 量化后发生同轨冲突: slot={event.start_slot}, lane={event.lane}")
            dropped["same_lane_quantization_collision"] += 1
            continue
        by_lane_slot[key] = event
    events = sorted(by_lane_slot.values(), key=lambda e: (e.start_slot, e.lane))

    # Optional final safeguard. The standalone pipeline normally applies this
    # earlier so events.json and MDM describe the same variant.
    groups: dict[int, list[ChartEvent]] = defaultdict(list)
    for event in events:
        groups[event.start_slot].append(event)
    gemini_slots = set()
    rewritten = []
    for event in events:
        group = groups[event.start_slot]
        gameplay = [item for item in group if item.semantic_type != 0]
        is_monster_double = (
            len(gameplay) == 2
            and {item.lane for item in gameplay} == {0, 1}
            and all(item.semantic_type == 1 and item.ibms_id not in BOSS_PLAY_IDS
                    for item in gameplay)
        )
        if normalize_doubles and is_monster_double and event.semantic_type == 1:
            gemini_slots.add(event.start_slot)
            event = ChartEvent(event.source_index, event.time, event.lane, event.semantic_type,
                               event.duration, event.confidence, "0E",
                               event.start_slot, event.end_slot)
        rewritten.append(event)
    events = rewritten

    report = {
        "input_events": len(notes),
        "retained_events": len(events),
        "dropped": dict(sorted(dropped.items())),
        "replacements": dict(sorted(replacements.items())),
        "gemini_onsets": len(gemini_slots),
        "normalize_doubles": normalize_doubles,
        "boss_controls": sum(event.semantic_type == 0 for event in events),
        "hold_events": sum(e.end_slot is not None for e in events),
        "lane_counts": {str(k): v for k, v in sorted(Counter(e.lane for e in events).items())},
        "semantic_counts": {str(k): v for k, v in sorted(Counter(e.semantic_type for e in events).items())},
        "ibms_counts": dict(sorted(Counter(e.ibms_id for e in events).items())),
        "quantization_resolution_per_measure": resolution,
        "quantization_max_error_ms": max(timing_errors, default=0.0),
        "confidence_threshold": confidence_threshold,
        "boss_confidence_threshold": boss_confidence_threshold,
    }
    return events, report


def _channels_for(event: ChartEvent) -> tuple[str, ...]:
    return (BOSS_CHANNELS if event.semantic_type in (0, 5)
            or event.ibms_id in ("16", "17") else LANE_CHANNELS[event.lane])


def place_bms_objects(events: list[ChartEvent], *, strict: bool = False) -> tuple[dict, dict]:
    """Allocate BMS channels, including paired endpoints for sustained notes."""
    occupied: dict[tuple[int, str], str] = {}
    active_until: dict[tuple[str, str], int] = {}
    placed: dict[tuple[int, str], str] = {}
    kept = []
    dropped: Counter[str] = Counter()

    for event in events:
        chosen = None
        for channel in _channels_for(event):
            if (event.start_slot, channel) in occupied:
                continue
            if event.end_slot is not None:
                if (event.end_slot, channel) in occupied:
                    continue
                if event.start_slot <= active_until.get((channel, event.ibms_id), -1):
                    continue
            chosen = channel
            break
        if chosen is None:
            if strict:
                raise ValueError(f"无法为事件分配 BMS 通道: t={event.time}, lane={event.lane}")
            dropped["channel_capacity_or_overlapping_hold"] += 1
            continue
        occupied[(event.start_slot, chosen)] = event.ibms_id
        placed[(event.start_slot, chosen)] = event.ibms_id
        if event.end_slot is not None:
            occupied[(event.end_slot, chosen)] = event.ibms_id
            placed[(event.end_slot, chosen)] = event.ibms_id
            active_until[(chosen, event.ibms_id)] = event.end_slot
        kept.append(event)

    report = {
        "placed_events": len(kept),
        "bms_objects": len(placed),
        "placement_dropped": dict(sorted(dropped.items())),
    }
    return placed, report


def _compact_data_line(positions: dict[int, str], resolution: int) -> str:
    common = resolution
    for position in positions:
        common = math.gcd(common, position)
    step = max(1, common)
    length = resolution // step
    values = ["00"] * length
    for position, value in positions.items():
        index = position // step
        if values[index] != "00":
            raise ValueError("内部错误：BMS 行位置冲突")
        values[index] = value
    return "".join(values)


def _header_value(value: Any) -> str:
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").split())


def build_bms_text(
    events: list[ChartEvent], *, bpm: float, scene: str, speed: int,
    difficulty: int, play_level: str, title: str, artist: str,
    level_designer: str, resolution: int = 960, strict: bool = False,
) -> tuple[str, dict]:
    if not SCENE_PATTERN.fullmatch(scene):
        raise ValueError("scene 必须形如 scene_01")
    if speed not in (1, 2, 3):
        raise ValueError("speed 必须是 1、2 或 3")
    if difficulty not in range(1, 6):
        raise ValueError("difficulty 必须位于 1..5")
    placed, placement_report = place_bms_objects(events, strict=strict)
    title = _header_value(title)
    artist = _header_value(artist)
    play_level = _header_value(play_level)
    level_designer = _header_value(level_designer)
    lines = [
        "*---------------------- HEADER FIELD",
        f"#PLAYER {speed}",
        f"#GENRE {scene}",
        f"#TITLE {title}",
        f"#ARTIST {artist}",
        f"#BPM {bpm:g}",
        f"#PLAYLEVEL {play_level}",
        f"#RANK {difficulty}",
        "#LNTYPE 1",
        f"#LEVELDESIGN {level_designer}",
        "",
        "*---------------------- MAIN DATA FIELD",
    ]
    grouped: dict[tuple[int, str], dict[int, str]] = defaultdict(dict)
    for (slot, channel), value in placed.items():
        measure, position = divmod(slot, resolution)
        grouped[(measure, channel)][position] = value
    for (measure, channel), positions in sorted(grouped.items()):
        lines.append(f"#{measure:03d}{channel}:{_compact_data_line(positions, resolution)}")
    lines.append("")
    return "\n".join(lines), placement_report


def build_info_json(*, title: str, romanized_title: str, artist: str,
                    level_designer: str, bpm: float, scene: str,
                    difficulty: int, play_level: str) -> dict:
    info = {
        "name": title,
        "name_romanized": romanized_title,
        "author": artist,
        "levelDesigner": level_designer,
        "bpm": f"{bpm:g}",
        "scene": scene,
        "difficulty1": "0",
        "difficulty2": "0",
        "difficulty3": "0",
        "difficulty4": "0",
        "difficulty5": "0",
    }
    info[f"difficulty{difficulty}"] = str(play_level)
    return info


def _run_ffmpeg(arguments: list[str]) -> None:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("需要 ffmpeg 将音频转换为 CustomAlbums 支持的 OGG")
    completed = subprocess.run([executable, "-hide_banner", "-loglevel", "error", "-y", *arguments],
                               check=False, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
    if completed.returncode:
        message = (completed.stderr or completed.stdout or "未知 ffmpeg 错误").strip()
        raise RuntimeError(f"ffmpeg 转换失败: {message}")


def _prepare_audio(audio: Path, temporary_prefix: Path, *, make_demo: bool,
                   demo_start: float, demo_seconds: float) -> tuple[Path, Path | None]:
    if not audio.is_file() or audio.stat().st_size == 0:
        raise FileNotFoundError(f"音频不存在或为空: {audio}")
    suffix = audio.suffix.lower()
    if suffix in (".ogg", ".mp3"):
        music = audio
    else:
        music = Path(str(temporary_prefix) + ".music.ogg")
        _run_ffmpeg(["-i", str(audio), "-vn", "-c:a", "libvorbis", "-q:a", "6", str(music)])
    demo = None
    if make_demo:
        demo = Path(str(temporary_prefix) + ".demo.ogg")
        _run_ffmpeg(["-ss", f"{demo_start:g}", "-i", str(audio), "-t", f"{demo_seconds:g}",
                     "-vn", "-c:a", "libvorbis", "-q:a", "4", str(demo)])
        if not demo.is_file() or demo.stat().st_size == 0:
            raise RuntimeError("未能生成试听音频，请调整 --demo-start")
    return music, demo


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (struct.pack(">I", len(payload)) + kind + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))


COVER_SIZE = 440
COVER_EXTENSIONS = frozenset((
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif",
))


def _circle_alpha(x: int, y: int, size: int) -> int:
    """Return an antialiased circular alpha value for one PNG pixel."""
    center = (size - 1) / 2
    distance = math.hypot(x - center, y - center)
    return max(0, min(255, round((size / 2 - distance) * 255)))


def _generated_cover_bytes(seed: str, size: int = 440) -> bytes:
    """Build a deterministic circular RGBA fallback cover."""
    if size <= 0:
        raise ValueError("封面尺寸必须大于 0")
    digest = hashlib.sha256(seed.encode("utf-8", errors="replace")).digest()
    first = tuple(48 + value % 160 for value in digest[:3])
    second = tuple(48 + value % 160 for value in digest[3:6])
    rows = bytearray()
    denominator = max(1, (size - 1) * 2)
    for y in range(size):
        rows.append(0)
        for x in range(size):
            mix = (x + y) / denominator
            band = 18 if ((x + y) // max(1, size // 8)) % 2 else 0
            for left, right in zip(first, second):
                rows.append(max(0, min(255, round(left * (1 - mix) + right * mix + band))))
            rows.append(_circle_alpha(x, y, size))
    # PNG color type 6 is RGBA. Transparent corners keep the packaged asset
    # genuinely circular instead of baking it onto a square background.
    header = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", header)
            + _png_chunk(b"IDAT", zlib.compress(bytes(rows), 9))
            + _png_chunk(b"IEND", b""))


def _write_generated_cover(path: Path, seed: str, size: int = 440) -> None:
    path.write_bytes(_generated_cover_bytes(seed, size))


def _circular_cover_filter(size: int = COVER_SIZE) -> str:
    center = (size - 1) / 2
    radius = size / 2
    return (
        f"scale={size}:{size}:force_original_aspect_ratio=increase,"
        f"crop={size}:{size},format=rgba,"
        "geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':"
        f"a='clip(({radius:g}-hypot(X-{center:g},Y-{center:g}))*255,0,255)'"
    )


def _convert_cover(source: Path, output: Path) -> None:
    _run_ffmpeg([
        "-i", str(source), "-frames:v", "1",
        "-vf", _circular_cover_filter(),
        str(output),
    ])


def _try_extract_embedded_cover(audio: Path, output: Path) -> bool:
    executable = shutil.which("ffmpeg")
    if executable is None:
        return False
    completed = subprocess.run([
        executable, "-hide_banner", "-loglevel", "error", "-y", "-i", str(audio),
        "-map", "0:v:0", "-frames:v", "1",
        "-vf", _circular_cover_filter(),
        str(output),
    ], check=False, capture_output=True, text=True,
       encoding="utf-8", errors="replace")
    return completed.returncode == 0 and output.is_file() and output.stat().st_size > 0


def inspect_embedded_cover(audio: Path) -> dict:
    """Report whether ffprobe sees an embedded picture stream in the audio."""
    audio = audio.resolve()
    if not audio.is_file() or audio.stat().st_size == 0:
        return {
            "present": None,
            "status": "invalid_audio",
            "message": "音频不存在或为空",
        }
    executable = shutil.which("ffprobe")
    if executable is None:
        return {
            "present": None,
            "status": "ffprobe_unavailable",
            "message": "未找到 ffprobe，无法检查内嵌封面",
        }
    completed = subprocess.run([
        executable, "-v", "error",
        "-show_entries",
        "stream=index,codec_type,codec_name:stream_disposition=attached_pic",
        "-of", "json", str(audio),
    ], check=False, capture_output=True, text=True,
       encoding="utf-8", errors="replace")
    if completed.returncode:
        return {
            "present": None,
            "status": "probe_failed",
            "message": "ffprobe 无法读取该音频的封面信息",
        }
    try:
        streams = json.loads(completed.stdout).get("streams", [])
    except (json.JSONDecodeError, AttributeError):
        streams = []
    present = any(
        stream.get("codec_type") == "video"
        or int(stream.get("disposition", {}).get("attached_pic", 0)) == 1
        for stream in streams
    )
    return {
        "present": present,
        "status": "found" if present else "not_found",
        "message": "检测到可用内嵌封面" if present else "未检测到可用内嵌封面",
    }


def find_default_cover(directory: Path | None) -> Path | None:
    """Choose the first deterministic, non-empty image from a fallback folder."""
    if directory is None:
        return None
    directory = directory.resolve()
    if not directory.is_dir():
        return None
    candidates = sorted(
        (path for path in directory.iterdir()
         if path.is_file() and path.suffix.lower() in COVER_EXTENSIONS
         and path.stat().st_size > 0),
        key=lambda path: (path.name.casefold(), path.name),
    )
    return candidates[0] if candidates else None


def _prepare_cover(audio: Path, requested: Path | None, temporary_prefix: Path,
                   *, include_cover: bool, seed: str,
                   fallback_cover: Path | None = None,
                   ) -> tuple[Path | None, str, bool, str | None]:
    if not include_cover:
        return None, "disabled", False, None
    generated = Path(str(temporary_prefix) + ".cover.png")
    if requested is not None:
        requested = requested.resolve()
        if not requested.is_file() or requested.stat().st_size == 0:
            raise FileNotFoundError(f"封面不存在或为空: {requested}")
        _convert_cover(requested, generated)
        return generated, "command_line", True, str(requested)
    if _try_extract_embedded_cover(audio, generated):
        return generated, "audio_embedded", True, str(audio)
    if generated.exists():
        generated.unlink()
    if fallback_cover is not None:
        fallback_cover = fallback_cover.resolve()
        if (fallback_cover.is_file() and fallback_cover.stat().st_size > 0
                and fallback_cover.suffix.lower() in COVER_EXTENSIONS):
            try:
                _convert_cover(fallback_cover, generated)
            except RuntimeError:
                if generated.exists():
                    generated.unlink()
            else:
                return generated, "image_fallback", True, str(fallback_cover)
    _write_generated_cover(generated, seed)
    return generated, "generated_fallback", True, None


def write_mdm(
    output: Path, *, bms_text: str, info: dict, generation: dict,
    audio: Path, difficulty: int, cover: Path | None = None,
    fallback_cover: Path | None = None,
    include_cover: bool = True, cover_seed: str = "MuseChart AI",
    make_demo: bool = True, demo_start: float = 0.0, demo_seconds: float = 15.0,
) -> dict:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if cover is not None and not include_cover:
        raise ValueError("--cover 与 --no-cover 不能同时使用")
    temporary_prefix = output.parent / f".{output.name}.{uuid.uuid4().hex}"
    music = demo = packaged_cover = None
    remove_packaged_cover = False
    try:
        music, demo = _prepare_audio(audio.resolve(), temporary_prefix, make_demo=make_demo,
                                     demo_start=demo_start, demo_seconds=demo_seconds)
        (packaged_cover, cover_source, remove_packaged_cover,
         cover_source_file) = _prepare_cover(
            audio.resolve(), cover, temporary_prefix,
            include_cover=include_cover, seed=cover_seed,
            fallback_cover=fallback_cover)
        cover_report = {
            "included": packaged_cover is not None,
            "source": cover_source,
            "source_file": cover_source_file,
            "archive_name": "cover.png" if packaged_cover is not None else None,
            "shape": "circle" if packaged_cover is not None else None,
            "size": [COVER_SIZE, COVER_SIZE] if packaged_cover is not None else None,
            "transparent_background": packaged_cover is not None,
        }
        generation["cover"] = cover_report
        temporary = output.with_suffix(output.suffix + ".tmp")
        try:
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=6) as archive:
                archive.writestr("info.json", json.dumps(info, ensure_ascii=False, indent=2) + "\n")
                archive.writestr(f"map{difficulty}.bms", bms_text)
                archive.writestr("generation.json", json.dumps(generation, ensure_ascii=False,
                                                               indent=2, sort_keys=True) + "\n")
                archive.write(music, f"music{music.suffix.lower()}")
                if demo is not None:
                    archive.write(demo, "demo.ogg")
                if packaged_cover is not None:
                    archive.write(packaged_cover, "cover.png")
            os.replace(temporary, output)
        finally:
            if temporary.exists():
                temporary.unlink()
    finally:
        for generated in (music, demo):
            if generated is not None and generated != audio.resolve() and generated.exists():
                generated.unlink()
        if (remove_packaged_cover and packaged_cover is not None
                and packaged_cover.exists()):
            packaged_cover.unlink()
    return cover_report


def write_text_atomic(path: Path, text: str) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
