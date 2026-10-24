"""Final QC gate: aggregate every level's verdict + fresh checks of the output.

This is the G6 / P0-item-4 gate from ``QUALITY-ROADMAP.md`` §3.4. After export
it aggregates the per-level QC sidecars written by the earlier M4 layers
(``transcript_qc.json`` / ``translation_qc.json`` / ``subtitles_qc.json``) into a
single ``quality`` verdict for ``report.json``, and adds fresh measurements of
the finished ``dub_track.wav``: loudness (ffmpeg ``ebur128``), A/V duration
delta, and per-speech-row silence / clipping.

The gate is deliberately **tolerant**: every artifact is optional, every
measurement is best-effort, and a missing/failed check degrades to
``"unknown"``/skip rather than ``fail``. Only hard evidence produces a ``fail``
verdict. Stdlib only (ffmpeg is the pre-existing system dependency used for the
loudness probe).
"""

from __future__ import annotations

import array
import json
import logging
import math
import re
import wave
from pathlib import Path
from typing import Any

from vidub.manifest import JobManifest
from vidub.utils.proc import find_tool, run

__all__ = [
    "SILENT_DBFS",
    "CLIP_LEVEL",
    "CLIP_RATIO_MAX",
    "FAIL_LOUDNESS_LU",
    "REVIEW_LOUDNESS_LU",
    "FAIL_DURATION_S",
    "FAIL_SILENT_RATIO",
    "measure_wav",
    "row_rms_dbfs",
    "measure_loudness",
    "check_audio",
    "run_gate",
]

log = logging.getLogger(__name__)

SILENT_DBFS = -50.0
CLIP_LEVEL = 0.99  # fraction of full scale
CLIP_RATIO_MAX = 0.005  # 0.5 % of samples clipped
FAIL_LOUDNESS_LU = 5.0
REVIEW_LOUDNESS_LU = 2.0
FAIL_DURATION_S = 2.0
FAIL_SILENT_RATIO = 0.2

LEVEL_KEYS = (("transcript", "transcript_qc.json"), ("translation", "translation_qc.json"))
_SUBTITLES_KEY = ("subtitles", "subtitles_qc.json")
_EBUR128_I_RE = re.compile(r"^\s*I:\s*(-?\d+(?:\.\d+)?)\s*LUFS", re.MULTILINE)


# ------------------------------------------------------------------- WAV reads
def _pcm16_samples(frames: bytes) -> array.array:
    """Interpret little-endian 16-bit PCM ``frames`` as an int16 array."""
    samples = array.array("h")
    samples.frombytes(frames)
    return samples


def _read_window(path: str | Path, start: float, end: float) -> tuple[array.array, int, int]:
    """Read the ``[start, end]`` window of a WAV; return ``(samples, rate, channels)``."""
    with wave.open(str(path), "rb") as wav_file:
        rate = wav_file.getframerate() or 1
        channels = wav_file.getnchannels() or 1
        total = wav_file.getnframes()
        if wav_file.getsampwidth() != 2:
            raise ValueError("audio QC requires 16-bit PCM")
        start_frame = max(0, int(float(start) * rate))
        end_frame = min(int(float(end) * rate), total)
        if end_frame <= start_frame:
            return array.array("h"), rate, channels
        wav_file.setpos(start_frame)
        frames = wav_file.readframes(end_frame - start_frame)
    return _pcm16_samples(frames), rate, channels


def _dbfs(samples: array.array) -> float | None:
    """RMS level of ``samples`` in dBFS (``None`` when there is nothing to read)."""
    if not samples:
        return None
    total = 0
    for value in samples:
        total += value * value
    rms = (total / len(samples)) ** 0.5
    if rms <= 0:
        return float("-inf")
    return 20.0 * math.log10(rms / 32768.0)


def measure_wav(path: str | Path) -> dict:
    """Measure a 16-bit PCM WAV: ``{"duration": float, "rms_dbfs": float}``.

    Never raises: an unreadable/absent file yields zeros and ``-inf`` dBFS.
    """
    try:
        with wave.open(str(path), "rb") as wav_file:
            rate = wav_file.getframerate() or 1
            frames = wav_file.getnframes()
            channels = wav_file.getnchannels() or 1
            width = wav_file.getsampwidth()
            raw = wav_file.readframes(frames)
    except (OSError, wave.Error, EOFError) as exc:
        log.warning("final QC: could not read %s (%s)", path, exc)
        return {"duration": 0.0, "rms_dbfs": float("-inf")}

    duration = frames / float(rate) if rate else 0.0
    if width != 2:  # pragma: no cover - mix writes 16-bit PCM
        return {"duration": duration, "rms_dbfs": float("-inf")}
    samples = _pcm16_samples(raw)
    level = _dbfs(samples)
    return {"duration": duration, "rms_dbfs": level if level is not None else 0.0}


def row_rms_dbfs(
    path: str | Path, start: float, end: float, rate: int | None = None
) -> float | None:
    """RMS of the ``[start, end]`` window of a WAV, or ``None`` when unreadable."""
    try:
        samples, _rate, _channels = _read_window(path, start, end)
    except (OSError, wave.Error, EOFError, ValueError) as exc:
        log.warning("final QC: could not read window %s [%s, %s] (%s)", path, start, end, exc)
        return None
    return _dbfs(samples)


def measure_loudness(path: str | Path) -> float | None:
    """Integrated loudness (LUFS) via ffmpeg ``ebur128``; ``None`` on any failure.

    Parses the ``I:`` line from ffmpeg's stderr. A missing ffmpeg, a bad input
    or an unparseable summary all return ``None`` instead of raising.
    """
    ffmpeg = find_tool("ffmpeg")
    if not ffmpeg:
        log.warning("final QC: ffmpeg not found; loudness measurement skipped")
        return None
    try:
        proc = run(
            [
                ffmpeg,
                "-hide_banner",
                "-nostats",
                "-i",
                str(path),
                "-filter_complex",
                "ebur128",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001 - a failed probe must never crash
        log.warning("final QC: loudness probe failed (%s)", exc)
        return None

    stderr = proc.stderr or ""
    matches = _EBUR128_I_RE.findall(stderr)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------- audio
def _clipped_ratio(samples: array.array) -> float:
    if not samples:
        return 0.0
    threshold = int(CLIP_LEVEL * 32768.0)
    clipped = sum(1 for value in samples if abs(value) >= threshold)
    return clipped / len(samples)


def check_audio(
    dub_track: str | Path | None,
    script_rows: list[dict],
    manifest_duration: float,
    loudness_target_i: float,
    artifacts_dir: Path | None = None,
) -> dict:
    """Fresh audio checks of the finished ``dub_track``.

    Returns ``{"duration_delta", "loudness_i", "loudness_delta", "silent_rows",
    "clipped_rows"}``; every field is ``None``/empty when it cannot be measured,
    so the caller can degrade to "unknown".
    """
    result: dict[str, Any] = {
        "duration_delta": None,
        "loudness_i": None,
        "loudness_delta": None,
        "silent_rows": [],
        "clipped_rows": [],
        "missing_rows": [],
        "unreadable_rows": [],
        "empty_rows": [],
        "short_rows": [],
    }
    if dub_track is None or not Path(dub_track).is_file():
        return result

    measured = measure_wav(dub_track)
    if measured["duration"] > 0:
        result["duration_delta"] = round(measured["duration"] - float(manifest_duration), 3)

    loudness = measure_loudness(dub_track)
    if loudness is not None:
        result["loudness_i"] = round(loudness, 2)
        result["loudness_delta"] = round(loudness - float(loudness_target_i), 2)

    for index, row in enumerate(script_rows or []):
        row_id = row.get("idx", index)
        try:
            start = float(row.get("start", 0.0))
            end = float(row.get("end", 0.0))
        except (TypeError, ValueError, AttributeError):
            continue
        if end <= start:
            continue
        try:
            if artifacts_dir is not None and row.get("file"):
                speech_file = artifacts_dir / row["file"]
                if not speech_file.is_file():
                    result["missing_rows"].append(row_id)
                    continue
                with wave.open(str(speech_file), "rb") as handle:
                    duration = handle.getnframes() / max(handle.getframerate(), 1)
                if duration + 0.05 < end - start:
                    result["short_rows"].append(row_id)
                samples, _rate, _channels = _read_window(speech_file, 0.0, end - start)
            else:
                samples, _rate, _channels = _read_window(dub_track, start, end)
        except (OSError, wave.Error, EOFError, ValueError):
            result["unreadable_rows"].append(row_id)
            continue
        if not samples:
            result["empty_rows"].append(row_id)
            continue
        level = _dbfs(samples)
        if artifacts_dir is not None:
            try:
                mixed, _rate, _channels = _read_window(dub_track, start, end)
                mixed_level = _dbfs(mixed)
                if mixed_level is None:
                    result["empty_rows"].append(row_id)
                elif mixed_level < SILENT_DBFS:
                    level = mixed_level
            except (OSError, ValueError, wave.Error, EOFError):
                result["unreadable_rows"].append(row_id)
        if level is not None and level < SILENT_DBFS:
            result["silent_rows"].append(row_id)
        if samples and _clipped_ratio(samples) >= CLIP_RATIO_MAX:
            result["clipped_rows"].append(row_id)
    return result


# ----------------------------------------------------------------------- gate
def _level_verdict(artifact: dict, default: str = "unknown") -> str:
    verdict = artifact.get("verdict")
    return str(verdict) if verdict else default


def _subtitles_verdict(artifact: dict) -> str:
    """Subtitles are ``unknown`` when absent, ``review`` on violations, else ``pass``."""
    if not artifact:
        return "unknown"
    violations = artifact.get("violations")
    if not isinstance(violations, dict):
        return "pass"
    total = sum(int(value or 0) for value in violations.values())
    return "review" if total > 0 else "pass"


def _load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _load_script_rows(path: Path) -> list[dict]:
    """Load a ``script.json`` list (``[]`` when absent/unreadable)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    return [row for row in data if isinstance(row, dict)]


def run_gate(manifest: JobManifest, cfg: Any) -> dict:
    """Aggregate the level verdicts + fresh audio checks into the quality block.

    Returns the ``quality`` dict described in the task spec. Missing sidecars
    become ``"unknown"``; missing/failed measurements are simply omitted from the
    reasons, so an old cache can never be failed by an absent artifact.
    """
    artifacts_dir = manifest.artifacts_dir
    levels: dict[str, str] = {}

    for name, default_name in LEVEL_KEYS:
        path = artifacts_dir / manifest.paths.get(f"{name}_qc", default_name)
        levels[name] = _level_verdict(_load_json(path))

    subtitles_path = artifacts_dir / manifest.paths.get(
        "subtitles_qc", _SUBTITLES_KEY[1]
    )
    subtitles_artifact = _load_json(subtitles_path)
    levels[_SUBTITLES_KEY[0]] = _subtitles_verdict(subtitles_artifact)
    tts = _load_json(artifacts_dir / manifest.paths.get("tts_qc", "tts_qc.json"))
    if tts.get("enabled"):
        unresolved = max(int(tts.get("low_mos", 0)) - int(tts.get("resolved", 0)), 0)
        levels["tts"] = "review" if unresolved else ("pass" if tts.get("scored") else "unknown")

    script_rows = _load_script_rows(
        artifacts_dir / manifest.paths.get("script", "script.json")
    )
    timed_path = artifacts_dir / manifest.paths.get("timed", "timed.json")
    timed_rows = _load_script_rows(timed_path)
    has_timing = timed_path.is_file()
    speech_rows = timed_rows if has_timing else script_rows
    dub_track = artifacts_dir / manifest.paths.get("dub_track", "dub_track.wav")
    audio = check_audio(
        dub_track,
        speech_rows,
        float(manifest.duration or 0.0),
        float(getattr(cfg, "loudness_target_i", -16.0)),
        artifacts_dir if has_timing else None,
    )
    audio["row_basis"] = "placed" if has_timing else "script-fallback"
    audio["rows_checked"] = len(speech_rows)
    if has_timing:
        units = _load_json(artifacts_dir / manifest.paths.get("units", "units.json")).get("units", [])
        expected = {unit.get("index", i) for i, unit in enumerate(units) if isinstance(unit, dict)}
        if not expected and not any(row.get("members") for row in timed_rows):
            expected = set(range(len(script_rows)))
        actual = {row.get("idx", i) for i, row in enumerate(timed_rows)}
        audio["unplaced_rows"] = sorted(expected - actual)

    review_dir = artifacts_dir / "review"
    packets: list[str] = []
    if review_dir.is_dir():
        for packet in sorted(review_dir.rglob("*")):
            if packet.is_file():
                try:
                    packets.append(packet.relative_to(manifest.cache_dir).as_posix())
                except ValueError:
                    packets.append(packet.as_posix())

    # --- verdict rules (ordered, first hit wins) ------------------------------
    reasons: list[str] = []
    fail = False
    review = False

    for name, verdict in levels.items():
        if verdict == "fail":
            fail = True
            reasons.append(f"{name} QC verdict is 'fail'")
        elif verdict == "review":
            review = True
            reasons.append(f"{name} QC verdict is 'review'")

    silent_rows = audio.get("silent_rows") or []
    if speech_rows and len(silent_rows) / len(speech_rows) > FAIL_SILENT_RATIO:
        fail = True
        reasons.append(
            f"{len(silent_rows)}/{len(speech_rows)} speech rows are silent "
            f"(> {FAIL_SILENT_RATIO:.0%})"
        )
    elif silent_rows:
        review = True
        reasons.append(f"silent speech row(s): {silent_rows}")
    for key in ("missing_rows", "unreadable_rows", "empty_rows", "unplaced_rows", "short_rows"):
        if audio.get(key):
            review = True
            reasons.append(f"{key}: {audio[key]}")

    loudness_delta = audio.get("loudness_delta")
    if loudness_delta is not None:
        if abs(loudness_delta) > FAIL_LOUDNESS_LU:
            fail = True
            reasons.append(f"loudness delta {loudness_delta:+.1f} LU exceeds ±{FAIL_LOUDNESS_LU:g}")
        elif abs(loudness_delta) > REVIEW_LOUDNESS_LU:
            review = True
            reasons.append(f"loudness delta {loudness_delta:+.1f} LU exceeds ±{REVIEW_LOUDNESS_LU:g}")

    if audio.get("clipped_rows"):
        review = True
        reasons.append(f"clipped speech row(s): {audio['clipped_rows']}")

    duration_delta = audio.get("duration_delta")
    if duration_delta is not None and abs(duration_delta) > FAIL_DURATION_S:
        review = True
        reasons.append(f"duration delta {duration_delta:+.1f}s exceeds ±{FAIL_DURATION_S:g}s")

    subtitles_violations = (
        sum(int(v or 0) for v in (subtitles_artifact.get("violations") or {}).values())
        if isinstance(subtitles_artifact.get("violations"), dict)
        else 0
    )
    if subtitles_violations > 0:
        review = True
        reasons.append(f"{subtitles_violations} unresolved subtitle violation(s)")

    all_unknown = all(verdict == "unknown" for verdict in levels.values())
    no_audio = (
        audio.get("duration_delta") is None
        and audio.get("loudness_i") is None
        and not script_rows
    )
    if fail:
        verdict = "fail"
    elif review:
        verdict = "review"
    elif all_unknown and no_audio:
        verdict = "unknown"
    else:
        verdict = "pass"

    return {
        "enabled": True,
        "verdict": verdict,
        "levels": levels,
        "audio": audio,
        "review_packets": packets,
        "reasons": reasons,
    }
