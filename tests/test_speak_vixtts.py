"""Tests for the opt-in ViXTTS engine (M2-T3).

Two layers:

* fast in-process tests for the engine registry + non-commercial license guard
  (no model load);
* a real subprocess test that synthesises Vietnamese speech through
  ``tests/_engine_runner.py tts engine=vixtts`` (skipped cleanly if the model
  cannot be downloaded).
"""

from __future__ import annotations

import json
import subprocess
import sys
import wave
from pathlib import Path

import pytest

from vidub.config import (
    TTS_ENGINES,
    JobConfig,
    check_tts_engine_allowed,
    is_nc_tts_engine,
    tts_engine_license,
)
from vidub.engines.tts.base import TTSError
from vidub.engines.tts.piper_tts import PiperTTS
from vidub.engines.tts.vixtts import VixttsTTS
from vidub.manifest import JobManifest
from vidub.stages.speak import build_tts

ROOT = Path(__file__).resolve().parents[1]
ENGINE_RUNNER = ROOT / "tests" / "_engine_runner.py"

VI_TEXT = "Xin chào, đây là giọng nói tiếng Việt tự nhiên."


def _manifest(engine: str) -> JobManifest:
    return JobManifest(input_path="demo.mp4", video_hash="hash1", tts_engine=engine)


# ------------------------------------------------------------------ registry
def test_tts_engine_registry_licenses() -> None:
    assert TTS_ENGINES["piper"]["nc"] is False
    assert TTS_ENGINES["vixtts"]["nc"] is True
    assert tts_engine_license("piper") == "MIT"
    assert "CPML" in tts_engine_license("vixtts")
    assert is_nc_tts_engine("vixtts")
    assert not is_nc_tts_engine("piper")


def test_vixtts_guard_message_points_at_flag() -> None:
    with pytest.raises(ValueError) as excinfo:
        check_tts_engine_allowed("vixtts", allow_nc=False)
    assert "--allow-nc-models" in str(excinfo.value)

    # no exception once the opt-in flag is set
    check_tts_engine_allowed("vixtts", allow_nc=True)


def test_unknown_tts_engine_rejected() -> None:
    with pytest.raises(ValueError):
        check_tts_engine_allowed("bogus", allow_nc=True)
    with pytest.raises(TTSError):
        build_tts(_manifest("bogus"), JobConfig(allow_nc_models=True))


# ------------------------------------------------------------- engine wiring
def test_build_tts_vixtts_blocked_without_flag() -> None:
    with pytest.raises(TTSError) as excinfo:
        build_tts(_manifest("vixtts"), JobConfig())
    assert "--allow-nc-models" in str(excinfo.value)


def test_build_tts_vixtts_allowed_is_lazy() -> None:
    engine = build_tts(_manifest("vixtts"), JobConfig(allow_nc_models=True))

    assert isinstance(engine, VixttsTTS)
    # construction must not load the (1.9 GB) checkpoint or import coqui-tts
    assert engine._model is None
    assert engine._config is None


def test_build_tts_piper_is_default_and_unaffected() -> None:
    engine = build_tts(_manifest("piper"), JobConfig())
    assert isinstance(engine, PiperTTS)
    # aliases still resolve to Piper
    assert isinstance(build_tts(_manifest("piper-tts"), JobConfig()), PiperTTS)


def test_ref_voice_is_passed_through() -> None:
    engine = build_tts(
        _manifest("vixtts"),
        JobConfig(allow_nc_models=True, ref_voice="my_voice.wav"),
    )
    assert isinstance(engine, VixttsTTS)
    assert engine.ref_voice == "my_voice.wav"


# ----------------------------------------------------------------- doctor
def test_doctor_lists_tts_engines(capsys) -> None:
    from vidub.cli import main

    rc = main(["doctor"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "tts engines" in out
    assert "piper" in out
    assert "vixtts" in out
    assert "MIT" in out
    assert "CPML" in out
    assert "doctor complete" in out


# ------------------------------------------------------------- real synthesis
def test_real_vixtts_synth(tmp_path: Path) -> None:
    """Synthesise Vietnamese with ViXTTS in a subprocess (skip if unavailable)."""
    out_wav = tmp_path / "vixtts.wav"
    result_path = tmp_path / "vixtts_result.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(ENGINE_RUNNER),
            "tts",
            str(result_path),
            str(out_wav),
            VI_TEXT,
            "vixtts",
        ],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode != 0:
        pytest.skip(
            "viXTTS unavailable (coqui-tts/model download?): "
            f"{proc.stderr.strip()[-800:]}"
        )

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["engine"] == "vixtts"
    assert payload["duration"] > 2.0, f"unexpectedly short: {payload['duration']}"
    assert payload["channels"] == 1
    assert payload["rate"] == 24000
    # The engine reports where it ran; cuda whenever the torch build sees a GPU.
    assert payload["device"] in {"cuda", "cpu"}, payload
    if payload["device"] == "cuda":
        import torch

        assert torch.cuda.is_available()

    with wave.open(str(out_wav), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 24000
        assert wav_file.getnframes() > 2 * 24000
