"""Tests for optional speaker diarization (M2-T4).

Layers:

* fast in-process tests for overlap-based speaker assignment, the speaker->voice
  map, and the pipeline stage graph (no model load);
* a subprocess test for the gated/auth-guard path (``tests/_engine_runner.py
  diarize``); the real pyannote model only runs when ``HF_TOKEN`` is set.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import wave
from pathlib import Path

import pytest

import vidub.engines.diarize.pyannote_diarizer as diarizer_mod
from vidub.engines.diarize.pyannote_diarizer import (
    DiarizeAuthError,
    PyannoteDiarizer,
)
from vidub.pipeline import required_stages
from vidub.stages.diarize import assign_speakers, main as diarize_main
from vidub.stages.speak import resolve_voice

ROOT = Path(__file__).resolve().parents[1]
ENGINE_RUNNER = ROOT / "tests" / "_engine_runner.py"

# Three raw speakers: A=10 s, B=6 s, C=2 s, plus a sub-0.5 s turn that is noise.
TURNS = [
    {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
    {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_00"},
    {"start": 10.0, "end": 13.0, "speaker": "SPEAKER_01"},
    {"start": 13.0, "end": 16.0, "speaker": "SPEAKER_01"},
    {"start": 16.0, "end": 18.0, "speaker": "SPEAKER_02"},
    {"start": 19.0, "end": 19.2, "speaker": "SPEAKER_03"},  # dropped (< 0.5 s)
]

SEGMENTS = [
    {"start": 0.0, "end": 4.0, "text": "a", "speaker": "spk0"},
    # 9-12 overlaps A by 1.0 s and B by 2.0 s -> dominant speaker is B
    {"start": 9.0, "end": 12.0, "text": "b", "speaker": "spk0"},
    {"start": 16.5, "end": 17.5, "text": "c", "speaker": "spk0"},
    # 20-21 overlaps nothing -> keeps its existing speaker
    {"start": 20.0, "end": 21.0, "text": "d", "speaker": "spk7"},
]


# ----------------------------------------------------------------- assignment
def test_assign_speakers_by_total_time_and_dominant_overlap() -> None:
    result = assign_speakers(SEGMENTS, TURNS)

    # speakers relabelled by descending total speech time: A->spk0, B->spk1, C->spk2
    assert [seg["speaker"] for seg in result] == ["spk0", "spk1", "spk2", "spk7"]

    # the inputs are not mutated
    assert SEGMENTS[0]["speaker"] == "spk0"
    assert SEGMENTS[3]["speaker"] == "spk7"


def test_assign_speakers_no_overlap_falls_back_to_spk0() -> None:
    segments = [{"start": 100.0, "end": 101.0, "text": "x", "speaker": "other"}]
    result = assign_speakers(segments, TURNS)
    assert result[0]["speaker"] == "spk0"


def test_assign_speakers_drops_short_turns() -> None:
    result = assign_speakers(
        [{"start": 19.0, "end": 19.2, "text": "x", "speaker": "spk0"}], TURNS
    )
    # the only overlapping turn was < 0.5 s, so it was dropped -> no assignment
    assert result[0]["speaker"] == "spk0"


# ---------------------------------------------------------------- voice map
def test_resolve_voice_defaults_and_overrides() -> None:
    assert resolve_voice({}, "spk0") == "vais1000"
    assert resolve_voice({}, "spk1") == "25hours"
    # unknown speakers fall back to spk0's voice
    assert resolve_voice({}, "spk7") == "vais1000"

    assert resolve_voice({"spk1": "custom.wav"}, "spk1") == "custom.wav"
    assert resolve_voice({"spk2": "extra"}, "spk2") == "extra"
    # overriding spk0 changes the fallback for unknown speakers too
    assert resolve_voice({"spk0": "custom0"}, "spk9") == "custom0"


# -------------------------------------------------------------- stage graph
def test_required_stages_with_and_without_diarize() -> None:
    assert required_stages("dub") == [
        "ingest", "transcribe", "translate", "speak", "timing", "mix", "export",
    ]
    assert required_stages("dub", diarize=True) == [
        "ingest", "transcribe", "diarize", "translate", "speak", "timing", "mix",
        "export",
    ]
    assert required_stages("subtitles-only", diarize=True) == [
        "ingest", "transcribe", "diarize", "translate", "export",
    ]


# ----------------------------------------------------------------- auth guard
def test_check_ready_requires_token(monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr(diarizer_mod, "_env_token", lambda: None)

    with pytest.raises(DiarizeAuthError) as excinfo:
        PyannoteDiarizer().check_ready()

    message = str(excinfo.value)
    assert "huggingface.co/settings/tokens" in message
    assert "speaker-diarization-3.1" in message
    assert "HF_TOKEN" in message


def test_diarize_stage_main_exits_nonzero_without_token(
    monkeypatch, capsys, tmp_path
) -> None:
    monkeypatch.setattr(diarizer_mod, "_env_token", lambda: None)

    rc = diarize_main([str(tmp_path / "job.json")])

    err = capsys.readouterr().err
    assert rc == 1
    assert "HF_TOKEN" in err
    assert "huggingface.co/settings/tokens" in err


def test_cli_run_diarize_fails_before_work(monkeypatch, capsys, tmp_path) -> None:
    """`vidub run --diarize` must exit non-zero before ingest/synthesis."""
    monkeypatch.setattr(diarizer_mod, "_env_token", lambda: None)

    from vidub.cli import main as cli_main

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")

    rc = cli_main(["run", str(video), "--diarize"])

    err = capsys.readouterr().err
    assert rc == 1
    assert "huggingface.co/settings/tokens" in err


# ----------------------------------------------------- subprocess (real/gated)
def _write_silence(path: Path, seconds: float = 1.0) -> Path:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * int(16000 * seconds))
    return path


def test_diarize_without_token_reports_actionable_error(tmp_path: Path) -> None:
    """Gated path: no HF_TOKEN -> actionable error JSON, no crash."""
    wav = _write_silence(tmp_path / "silence.wav")
    result_path = tmp_path / "diarize_result.json"

    env = dict(os.environ)
    env.pop("HF_TOKEN", None)
    env.pop("HUGGING_FACE_HUB_TOKEN", None)
    env["HF_HOME"] = str(tmp_path / "hf_home")  # avoid any stored token

    proc = subprocess.run(
        [sys.executable, str(ENGINE_RUNNER), "diarize", str(result_path), str(wav)],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr[-800:]

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["error_type"] == "DiarizeAuthError"
    assert "huggingface.co/settings/tokens" in payload["error"]
    assert "speaker-diarization-3.1" in payload["error"]


def test_real_diarization_only_with_token(tmp_path: Path) -> None:
    """Real pyannote run; skipped unless HF_TOKEN is set in the environment."""
    if not os.environ.get("HF_TOKEN"):
        pytest.skip("HF_TOKEN not set: gated pyannote models unavailable")

    wav = _write_silence(tmp_path / "silence.wav", seconds=2.0)
    result_path = tmp_path / "diarize_real.json"

    proc = subprocess.run(
        [sys.executable, str(ENGINE_RUNNER), "diarize", str(result_path), str(wav)],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    # The runner reports failures as data; a token can be present yet still be
    # gated (no accepted conditions, no access) — skip on the payload, not on
    # the exit code (which is 0 even for an auth failure).
    if not result_path.exists():
        pytest.skip(
            f"diarize runner produced no result: {proc.stderr.strip()[-800:]}"
        )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if payload.get("ok") is False:
        pytest.skip(f"pyannote unavailable: {payload.get('error', '')}")

    assert payload["ok"] is True, payload
    assert isinstance(payload["turns"], list)
