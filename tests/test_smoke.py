"""Smoke tests for the M1-T0 ViDub scaffold."""

from __future__ import annotations

import vidub
from vidub.cli import main
from vidub.config import JobConfig


def test_package_imports_and_version() -> None:
    assert vidub.__version__ == "0.11.0"


def test_jobconfig_defaults() -> None:
    cfg = JobConfig()
    assert cfg.src_lang == "en"
    assert cfg.tgt_lang == "vi"
    assert cfg.mode == "dub"
    assert cfg.asr_model == "base"
    assert cfg.mt_engine == "envit5"
    assert cfg.tts_engine == "piper"
    assert cfg.cache_dir == ".vidub-cache"


def test_jobconfig_from_dict_ignores_unknown_keys() -> None:
    cfg = JobConfig.from_dict(
        {
            "src_lang": "ja",
            "mode": "voice-over",
            "unknown_key": 123,
            "another_unknown": "ignored",
        }
    )
    assert cfg.src_lang == "ja"
    assert cfg.mode == "voice-over"
    # unspecified fields keep their defaults
    assert cfg.tgt_lang == "vi"
    assert cfg.asr_model == "base"
    assert not hasattr(cfg, "unknown_key")


def test_doctor_runs_and_exits_zero(capsys) -> None:
    rc = main(["doctor"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "python" in out
    assert "ffmpeg" in out
    assert "ffprobe" in out
    assert "doctor complete" in out


def test_run_requires_input(capsys) -> None:
    rc = main(["run"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "input" in err.lower()


def test_doctor_reports_cuda_and_never_crashes(capsys) -> None:
    rc = main(["doctor"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "cuda dlls:" in out
    assert "cuda available:" in out
    assert "gpu whisper" in out
    assert "doctor complete" in out
