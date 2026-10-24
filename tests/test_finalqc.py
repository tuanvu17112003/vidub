"""Tests for the final QC gate (M4-T4, QUALITY-ROADMAP.md §3.4).

Fast/pure where possible: synthetic WAVs via the stdlib ``wave`` module, tmp
sidecars, and a stub manifest/cfg. No models; ffmpeg is only touched by
``measure_loudness`` (which must never raise).
"""

from __future__ import annotations

import array
import json
import wave
from pathlib import Path

import pytest

from vidub.config import JobConfig
from vidub.manifest import JobManifest
from vidub.utils import finalqc
from vidub.utils.finalqc import (
    check_audio,
    measure_loudness,
    measure_wav,
    row_rms_dbfs,
    run_gate,
)


# ------------------------------------------------------------------ WAV helpers
def _write_wav(path: Path, samples: list[int], rate: int = 16000, channels: int = 1) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(array.array("h", samples).tobytes())
    return path


def _silence(seconds: float = 1.0, rate: int = 16000) -> list[int]:
    return [0] * int(seconds * rate)


def _tone(seconds: float = 1.0, rate: int = 16000, amp: float = 0.5) -> list[int]:
    return [int(amp * 32767)] * int(seconds * rate)


def _clipped(seconds: float = 1.0, rate: int = 16000) -> list[int]:
    return [32767 if i % 2 else -32768 for i in range(int(seconds * rate))]


# ------------------------------------------------------------------ measure_wav
def test_measure_wav_silence_is_minus_inf(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "sil.wav", _silence())
    result = measure_wav(path)
    assert result["duration"] == pytest.approx(1.0)
    assert result["rms_dbfs"] == float("-inf")


def test_measure_wav_tone_level(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "tone.wav", _tone(amp=0.5))
    result = measure_wav(path)
    assert result["duration"] == pytest.approx(1.0)
    # 0.5 full scale ~= -6 dBFS
    assert result["rms_dbfs"] == pytest.approx(-6.02, abs=0.1)


def test_measure_wav_clipped_near_zero_dbfs(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "clip.wav", _clipped())
    result = measure_wav(path)
    assert result["rms_dbfs"] > -0.5


def test_measure_wav_missing_file_never_raises(tmp_path: Path) -> None:
    result = measure_wav(tmp_path / "nope.wav")
    assert result["duration"] == 0.0
    assert result["rms_dbfs"] == float("-inf")


def test_measure_wav_stereo(tmp_path: Path) -> None:
    # 1 s of stereo at 16 kHz = 16000 frames -> 32000 interleaved samples
    samples = [int(0.5 * 32767)] * (16000 * 2)
    path = _write_wav(tmp_path / "stereo.wav", samples, channels=2)
    result = measure_wav(path)
    assert result["duration"] == pytest.approx(1.0, abs=0.01)
    assert result["rms_dbfs"] == pytest.approx(-6.02, abs=0.1)


# --------------------------------------------------------------- row_rms_dbfs
def test_row_rms_window(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "tone.wav", _tone(seconds=2.0))
    level = row_rms_dbfs(path, 0.0, 1.0, 16000)
    assert level == pytest.approx(-6.02, abs=0.1)


def test_row_rms_unreadable_is_none(tmp_path: Path) -> None:
    assert row_rms_dbfs(tmp_path / "nope.wav", 0.0, 1.0, 16000) is None


def test_row_rms_empty_window_is_none(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "tone.wav", _tone(seconds=1.0))
    assert row_rms_dbfs(path, 5.0, 6.0, 16000) is None


# ------------------------------------------------------------- measure_loudness
def test_measure_loudness_real_wav(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "tone.wav", _tone(seconds=2.0))
    level = measure_loudness(path)
    # ffmpeg may be absent: then None is correct too; otherwise a real number.
    assert level is None or isinstance(level, float)


def test_measure_loudness_non_audio_never_raises(tmp_path: Path) -> None:
    junk = tmp_path / "junk.bin"
    junk.write_bytes(b"not audio at all")
    assert measure_loudness(junk) is None


def test_measure_loudness_missing_tool_is_none(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(finalqc, "find_tool", lambda name: None)
    assert measure_loudness(tmp_path / "nope.wav") is None


# ------------------------------------------------------------------ check_audio
def test_check_audio_flags_silent_row(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "dub.wav", _silence(seconds=2.0))
    result = check_audio(path, [{"start": 0.0, "end": 1.0}], 2.0, -16.0)
    assert result["silent_rows"] == [0]
    assert result["duration_delta"] == pytest.approx(0.0, abs=0.01)


def test_check_audio_flags_clipped_row(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "dub.wav", _clipped(seconds=1.0))
    result = check_audio(path, [{"start": 0.0, "end": 1.0}], 1.0, -16.0)
    assert result["clipped_rows"] == [0]
    assert result["silent_rows"] == []


def test_check_audio_missing_track_is_empty_result(tmp_path: Path) -> None:
    result = check_audio(tmp_path / "nope.wav", [], 5.0, -16.0)
    assert result["duration_delta"] is None
    assert result["loudness_i"] is None
    assert result["silent_rows"] == []


# -------------------------------------------------------------------- run_gate
def _manifest(tmp_path: Path, **overrides) -> JobManifest:
    cache = tmp_path / "cache"
    artifacts = cache / "h1"
    artifacts.mkdir(parents=True, exist_ok=True)
    fields = dict(
        input_path="x/demo.mp4",
        video_hash="h1",
        cache_dir=str(cache),
        mode="dub",
        duration=4.0,
        paths={"script": "script.json", "dub_track": "dub_track.wav"},
    )
    fields.update(overrides)
    manifest = JobManifest(**fields)
    manifest.artifacts_dir.mkdir(parents=True, exist_ok=True)
    return manifest


def _sidecar(manifest: JobManifest, name: str, payload: dict) -> None:
    (manifest.artifacts_dir / name).write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_gate_unknown_when_everything_absent(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    gate = run_gate(manifest, JobConfig())
    assert gate["verdict"] == "unknown"
    assert gate["levels"]["transcript"] == "unknown"
    assert gate["enabled"] is True


def test_gate_fail_on_transcript_fail(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    _sidecar(manifest, "transcript_qc.json", {"verdict": "fail", "counts": {}})
    gate = run_gate(manifest, JobConfig())
    assert gate["verdict"] == "fail"
    assert any("transcript" in reason for reason in gate["reasons"])


def test_gate_review_on_subtitles_violations(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    _sidecar(
        manifest,
        "subtitles_qc.json",
        {"cues_in": 1, "violations": {"cps": 1, "duration": 0, "line_length": 0, "lines": 0}},
    )
    gate = run_gate(manifest, JobConfig())
    assert gate["levels"]["subtitles"] == "review"
    assert gate["verdict"] == "review"


def test_gate_subtitles_pass_when_all_zero(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    _sidecar(
        manifest,
        "subtitles_qc.json",
        {"cues_in": 1, "violations": {"cps": 0, "duration": 0, "line_length": 0, "lines": 0}},
    )
    gate = run_gate(manifest, JobConfig())
    assert gate["levels"]["subtitles"] == "pass"


def test_gate_fail_on_silent_row_ratio(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    _write_wav(manifest.artifacts_dir / "dub_track.wav", _silence(seconds=4.0))
    # 2 of 5 rows silent = 0.4 > 0.2 -> fail
    rows = [{"start": 0.0, "end": 1.0}, {"start": 1.0, "end": 2.0}]
    rows += [{"start": 2.0, "end": 3.0}, {"start": 3.0, "end": 4.0}]
    rows += [{"start": 4.0, "end": 4.0}]  # zero-length: ignored
    (manifest.artifacts_dir / "script.json").write_text(
        json.dumps(rows), encoding="utf-8"
    )
    gate = run_gate(manifest, JobConfig())
    assert gate["audio"]["silent_rows"] == [0, 1, 2, 3]
    assert gate["verdict"] == "fail"


def test_gate_review_on_partial_silent_rows(tmp_path: Path, monkeypatch) -> None:
    manifest = _manifest(tmp_path)
    monkeypatch.setattr(
        finalqc,
        "check_audio",
        lambda *a, **k: {
            "duration_delta": 0.0,
            "loudness_i": -16.0,
            "loudness_delta": 0.0,
            "silent_rows": [0],
            "clipped_rows": [],
        },
    )
    (manifest.artifacts_dir / "script.json").write_text(
        json.dumps([{"start": 0.0, "end": 1.0}] * 10), encoding="utf-8"
    )
    gate = run_gate(manifest, JobConfig())
    assert gate["verdict"] == "review"
    assert any("silent" in reason for reason in gate["reasons"])


def test_gate_lists_review_packets(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    packet_dir = manifest.artifacts_dir / "review" / "translation"
    packet_dir.mkdir(parents=True)
    (packet_dir / "seg-0.txt").write_text("label: number-error\n", encoding="utf-8")
    gate = run_gate(manifest, JobConfig())
    assert any("seg-0.txt" in path for path in gate["review_packets"])


def test_gate_fail_on_loudness_delta(tmp_path: Path, monkeypatch) -> None:
    manifest = _manifest(tmp_path)
    monkeypatch.setattr(
        finalqc,
        "check_audio",
        lambda *a, **k: {
            "duration_delta": 0.0,
            "loudness_i": -5.0,
            "loudness_delta": 9.0,
            "silent_rows": [],
            "clipped_rows": [],
        },
    )
    gate = run_gate(manifest, JobConfig())
    assert gate["verdict"] == "fail"
    assert any("loudness" in reason for reason in gate["reasons"])


def test_gate_review_on_moderate_loudness_delta(tmp_path: Path, monkeypatch) -> None:
    manifest = _manifest(tmp_path)
    monkeypatch.setattr(
        finalqc,
        "check_audio",
        lambda *a, **k: {
            "duration_delta": 0.0,
            "loudness_i": -13.0,
            "loudness_delta": 3.0,
            "silent_rows": [],
            "clipped_rows": [],
        },
    )
    gate = run_gate(manifest, JobConfig())
    assert gate["verdict"] == "review"


def test_gate_review_on_duration_delta(tmp_path: Path, monkeypatch) -> None:
    manifest = _manifest(tmp_path)
    monkeypatch.setattr(
        finalqc,
        "check_audio",
        lambda *a, **k: {
            "duration_delta": 5.0,
            "loudness_i": None,
            "loudness_delta": None,
            "silent_rows": [],
            "clipped_rows": [],
        },
    )
    gate = run_gate(manifest, JobConfig())
    assert gate["verdict"] == "review"
    assert any("duration" in reason for reason in gate["reasons"])


def test_gate_pass_when_clean_and_levels_present(tmp_path: Path, monkeypatch) -> None:
    manifest = _manifest(tmp_path)
    _sidecar(manifest, "transcript_qc.json", {"verdict": "pass", "counts": {}})
    monkeypatch.setattr(
        finalqc,
        "check_audio",
        lambda *a, **k: {
            "duration_delta": 0.0,
            "loudness_i": -16.0,
            "loudness_delta": 0.0,
            "silent_rows": [],
            "clipped_rows": [],
        },
    )
    gate = run_gate(manifest, JobConfig())
    assert gate["verdict"] == "pass"
    assert gate["reasons"] == []
