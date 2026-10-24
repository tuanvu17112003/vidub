"""Tests for the S3 speak stage (M1-T4).

* logic tests use a fake TTS writing a 0.5 s sine WAV (no model, no network);
* a real test loads the piper ``vi_VN-vais1000-medium`` voice and synthesises a
  Vietnamese sentence (skipped if the voice can't be downloaded).
"""

from __future__ import annotations

import json
import math
import struct
import subprocess
import sys
import wave
from pathlib import Path

import pytest

from vidub.config import JobConfig
from vidub.engines.tts.base import BaseTTS, TTSError
from vidub.manifest import JobManifest
from vidub.stages import speak as speak_mod
from vidub.stages.speak import SPEECH_DIR, main, speak

ROOT = Path(__file__).resolve().parents[1]
ENGINE_RUNNER = ROOT / "tests" / "_engine_runner.py"

SCRIPT = [
    {"start": 0.0, "end": 2.0, "speaker": "spk0", "en": "Hello", "vi": "Xin chào", "truncated": False},
    {"start": 2.0, "end": 4.0, "speaker": "spk0", "en": "Bye", "vi": "Tạm biệt", "truncated": False},
]


def _write_sine_wav(path: Path, duration: float = 0.5, rate: int = 22050, freq: float = 440.0) -> None:
    """Write a mono 16-bit sine WAV of ``duration`` seconds."""
    frames = bytearray()
    for i in range(int(duration * rate)):
        value = int(0.3 * 32767 * math.sin(2.0 * math.pi * freq * i / rate))
        frames += struct.pack("<h", value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(bytes(frames))


class FakeTTS(BaseTTS):
    """Fake engine: writes a 0.5 s sine WAV and records the calls."""

    def __init__(self, duration: float = 0.5) -> None:
        self.duration = duration
        self.calls: list[tuple[str, str]] = []

    def synth(self, text: str, out_wav: str) -> float:
        self.calls.append((text, out_wav))
        _write_sine_wav(Path(out_wav), self.duration)
        return self.duration


def _make_job(
    tmp_path: Path, script: list[dict] | None, mode: str = "dub"
) -> tuple[Path, Path, JobConfig]:
    cache_dir = tmp_path / "cache"
    artifacts_dir = cache_dir / "hash1"
    artifacts_dir.mkdir(parents=True)
    if script is not None:
        (artifacts_dir / "script.json").write_text(
            json.dumps(script, ensure_ascii=False), encoding="utf-8"
        )
    job_path = artifacts_dir / "job.json"
    JobManifest(
        input_path="demo.mp4",
        video_hash="hash1",
        cache_dir=str(cache_dir),
        mode=mode,
        src_lang="en",
        tgt_lang="vi",
        paths={"script": "script.json"},
        stages={"translate": "done"},
    ).save(job_path)
    return job_path, artifacts_dir, JobConfig(cache_dir=str(cache_dir))


def test_speak_writes_segment_wavs_and_speech_json(tmp_path: Path) -> None:
    job_path, artifacts_dir, cfg = _make_job(tmp_path, SCRIPT)
    fake = FakeTTS(0.5)

    manifest = speak(str(job_path), cfg, tts=fake)

    speech_dir = artifacts_dir / SPEECH_DIR
    assert (speech_dir / "seg_0000.wav").exists()
    assert (speech_dir / "seg_0001.wav").exists()

    speech = json.loads((artifacts_dir / "speech.json").read_text(encoding="utf-8"))
    assert len(speech) == 2
    assert set(speech[0]) == {"idx", "start", "end", "speaker", "vi", "file", "duration"}
    assert speech[0]["idx"] == 0
    assert speech[0]["vi"] == "Xin chào"
    assert speech[0]["file"] == "speech/seg_0000.wav"
    assert speech[0]["duration"] == 0.5
    assert speech[1]["idx"] == 1
    assert speech[1]["file"] == "speech/seg_0001.wav"
    assert speech[1]["vi"] == "Tạm biệt"

    # the engine received the Vietnamese text and wrote mono 16-bit audio
    assert [call[0] for call in fake.calls] == ["Xin chào", "Tạm biệt"]
    with wave.open(str(speech_dir / "seg_0000.wav"), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert abs(wav_file.getnframes() / wav_file.getframerate() - 0.5) < 0.01

    reloaded = JobManifest.load(job_path)
    assert reloaded.paths["speech"] == "speech.json"
    assert reloaded.paths["speech_dir"] == "speech"
    assert reloaded.stages["speak"] == "done"
    assert manifest.video_hash == "hash1"


def test_speak_uses_explicit_spoken_text(tmp_path: Path) -> None:
    script = [dict(row) for row in SCRIPT]
    script[0]["spoken_vi"] = "Tôi đang nói câu này"
    job_path, _artifacts_dir, cfg = _make_job(tmp_path, script)
    fake = FakeTTS()

    speak(str(job_path), cfg, tts=fake)

    speech = json.loads(
        (_artifacts_dir / "speech.json").read_text(encoding="utf-8")
    )
    assert fake.calls[0][0] == "Tôi đang nói câu này"
    assert speech[0]["vi"] == "Tôi đang nói câu này"


def test_speak_uses_authoritative_unit_spoken_text(tmp_path: Path) -> None:
    job_path, artifacts_dir, cfg = _make_job(tmp_path, SCRIPT)
    units = {
        "version": 1,
        "text_contract_version": 1,
        "mode": "sentence",
        "units": [
            {"index": 0, "indices": [0], "vi": "Xin chào", "spoken_vi": "Unit speech one"},
            {"index": 1, "indices": [1], "vi": "Tạm biệt", "spoken_vi": "Unit speech two"},
        ],
    }
    (artifacts_dir / "units.json").write_text(
        json.dumps(units, ensure_ascii=False), encoding="utf-8"
    )
    manifest = JobManifest.load(job_path)
    manifest.paths["units"] = "units.json"
    manifest.save(job_path)
    fake = FakeTTS()

    speak(str(job_path), cfg, tts=fake)

    assert [call[0] for call in fake.calls] == ["Unit speech one", "Unit speech two"]


def test_speak_skipped_for_subtitles_only(tmp_path: Path) -> None:
    """Mode guard: speak() refuses a mode that never synthesises (SU-1-T1)."""
    job_path, artifacts_dir, cfg = _make_job(tmp_path, SCRIPT, mode="subtitles-only")
    fake = FakeTTS(0.5)

    with pytest.raises(RuntimeError) as excinfo:
        speak(str(job_path), cfg, tts=fake)

    message = str(excinfo.value)
    assert "skipped" in message
    assert "subtitles-only" in message
    # the guard runs before any work: nothing is synthesised or written
    assert fake.calls == []
    assert not (artifacts_dir / "speech.json").exists()
    assert "speak" not in JobManifest.load(job_path).stages


def test_missing_script_raises_and_cli_exits_nonzero(tmp_path: Path, capsys) -> None:
    job_path, _artifacts_dir, cfg = _make_job(tmp_path, None)

    with pytest.raises(FileNotFoundError) as excinfo:
        speak(str(job_path), cfg, tts=FakeTTS())
    assert "translate" in str(excinfo.value)

    rc = main([str(job_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "error:" in captured.err


class FailingTTS(BaseTTS):
    """Fake engine that fails on the second segment (idx 1)."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def synth(self, text: str, out_wav: str) -> float:
        self.calls.append(text)
        if len(self.calls) == 2:
            raise RuntimeError("synthetic TTS failure on segment 1")
        _write_sine_wav(Path(out_wav), 0.5)
        return 0.5


def test_speak_aborts_on_segment_tts_failure(tmp_path: Path) -> None:
    """A per-segment TTS error aborts the stage (never skips the segment)."""
    job_path, artifacts_dir, cfg = _make_job(tmp_path, SCRIPT)

    with pytest.raises(TTSError) as excinfo:
        speak(str(job_path), cfg, tts=FailingTTS())

    message = str(excinfo.value)
    assert "segment 1" in message
    assert "seg_0001.wav" in message
    # aborts: no speech.json is written at all (so timing can't build a half dub)
    assert not (artifacts_dir / "speech.json").exists()
    assert "speak" not in JobManifest.load(job_path).stages


def test_speak_cli_exits_nonzero_on_tts_failure(tmp_path: Path, monkeypatch) -> None:
    job_path, artifacts_dir, _cfg = _make_job(tmp_path, SCRIPT)
    monkeypatch.setattr(speak_mod, "PiperTTS", lambda voice_path=None: FailingTTS())

    rc = speak_mod.main([str(job_path)])

    assert rc == 1
    assert not (artifacts_dir / "speech.json").exists()


def test_speak_applies_pronunciation_to_tts_only(tmp_path: Path) -> None:
    """The lexicon rewrites the TTS text; script.json keeps the written form."""
    job_path, artifacts_dir, cfg = _make_job(tmp_path, SCRIPT)
    lexicon = tmp_path / "p.tsv"
    lexicon.write_text("Xin chào\tXin chao\n", encoding="utf-8")
    cfg.pronounce_path = str(lexicon)
    fake = FakeTTS(0.5)

    speak(str(job_path), cfg, tts=fake)

    # the TTS received the phonetic form...
    assert [call[0] for call in fake.calls] == ["Xin chao", "Tạm biệt"]
    # ...while script.json still holds the written Vietnamese text
    script = json.loads((artifacts_dir / "script.json").read_text(encoding="utf-8"))
    assert script[0]["vi"] == "Xin chào"

    sidecar = json.loads(
        (artifacts_dir / "pronunciation.json").read_text(encoding="utf-8")
    )
    assert sidecar == {"replacements": 1}
    reloaded = JobManifest.load(job_path)
    assert reloaded.paths["pronunciation"] == "pronunciation.json"


def test_speak_without_lexicon_writes_zero_sidecar(tmp_path: Path) -> None:
    """The default path is unchanged: TTS text untouched, zero counter."""
    job_path, artifacts_dir, cfg = _make_job(tmp_path, SCRIPT)
    fake = FakeTTS(0.5)

    speak(str(job_path), cfg, tts=fake)

    assert [call[0] for call in fake.calls] == ["Xin chào", "Tạm biệt"]
    sidecar = json.loads(
        (artifacts_dir / "pronunciation.json").read_text(encoding="utf-8")
    )
    assert sidecar == {"replacements": 0}


def test_real_piper_synth(tmp_path: Path) -> None:
    out_wav = tmp_path / "piper.wav"

    # Piper (onnxruntime) runs in its own process so a native hang can't wedge
    # the suite.
    result_path = tmp_path / "tts_result.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(ENGINE_RUNNER),
            "tts",
            str(result_path),
            str(out_wav),
            "Xin chào, đây là bài kiểm tra âm thanh.",
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if proc.returncode != 0:
        pytest.skip(f"piper voice unavailable (offline?): {proc.stderr.strip()[-500:]}")

    duration = json.loads(result_path.read_text(encoding="utf-8"))["duration"]

    assert duration > 0.5, f"unexpectedly short: {duration}"
    with wave.open(str(out_wav), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert abs(wav_file.getnframes() / wav_file.getframerate() - duration) < 0.01
