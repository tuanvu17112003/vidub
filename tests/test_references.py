"""Tests for per-speaker cloning references (M3-T4).

Fast tests build a synthetic ``vocals.wav`` stem + fake diarization turns so the
references stage never runs Demucs or pyannote; the ffmpeg trim is real (skipped
when ffmpeg is unavailable). Resolver/validation tests monkeypatch the engine.
"""

from __future__ import annotations

import json
import math
import struct
import wave
from pathlib import Path

import pytest

import vidub.pipeline as pipeline
from vidub.config import JobConfig, validate_clone_speakers
from vidub.manifest import JobManifest
from vidub.stages import references as references_mod
from vidub.stages.references import (
    REFERENCES_DIR,
    REFERENCES_NAME,
    extract_references,
    select_windows,
)
from vidub.stages.speak import _VoiceResolver
from vidub.utils.audio import wav_duration

RATE = 8000


def _write_wav(
    path: Path, duration: float, freq: float | None = None, rate: int = RATE
) -> None:
    """Mono 16-bit WAV; digital silence when ``freq`` is None."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = bytearray()
    for index in range(round(duration * rate)):
        value = (
            0
            if freq is None
            else int(0.4 * 32767 * math.sin(2 * math.pi * freq * index / rate))
        )
        frames += struct.pack("<h", value)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(bytes(frames))


def _make_job(
    tmp_path: Path,
    speakers: dict[str, list[tuple[float, float]]],
    vocals_seconds: float = 30.0,
) -> tuple[Path, Path, JobConfig]:
    cache_dir = tmp_path / "cache"
    artifacts_dir = cache_dir / "hash1"
    artifacts_dir.mkdir(parents=True)

    _write_wav(artifacts_dir / "audio.wav", vocals_seconds, freq=440.0)
    # Pre-place the Demucs stems so run_separation reuses them (no Demucs run).
    track_dir = artifacts_dir / "sep" / "htdemucs" / "audio"
    _write_wav(track_dir / "vocals.wav", vocals_seconds, freq=440.0)
    _write_wav(track_dir / "no_vocals.wav", vocals_seconds, freq=None)

    segments = [
        {"start": start, "end": end, "text": "hi", "speaker": speaker}
        for speaker, turns in speakers.items()
        for start, end in turns
    ]
    (artifacts_dir / "segments.json").write_text(
        json.dumps(segments), encoding="utf-8"
    )

    job_path = artifacts_dir / "job.json"
    JobManifest(
        input_path="demo.mp4",
        video_hash="hash1",
        cache_dir=str(cache_dir),
        paths={"audio": "audio.wav", "segments": "segments.json"},
        stages={"transcribe": "done", "diarize": "done"},
    ).save(job_path)
    cfg = JobConfig(
        cache_dir=str(cache_dir),
        diarize=True,
        clone_speakers=True,
        tts_engine="vixtts",
        allow_nc_models=True,
    )
    return job_path, artifacts_dir, cfg


# ------------------------------------------------------------- select_windows
def test_select_windows_skips_short_speaker() -> None:
    assert select_windows([(0.0, 2.0)]) == []


def test_select_windows_requires_a_long_enough_turn() -> None:
    # 3 s total but every turn is below the 1.5 s floor -> no windows.
    assert select_windows([(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)]) == []


def test_select_windows_longest_first_and_capped() -> None:
    windows = select_windows([(0.0, 4.0), (5.0, 9.0), (10.0, 14.0), (15.0, 19.0)])
    assert len(windows) == 3  # MAX_TURNS
    assert windows[0] == (0.0, 4.0)


def test_select_windows_caps_total_length() -> None:
    windows = select_windows([(0.0, 25.0)])
    assert windows == [(0.0, 20.0)]  # MAX_REF_SECONDS


# ------------------------------------------------------------ extract_references
def test_extract_references_creates_expected_windows_and_fallback(
    tmp_path: Path, ffmpeg_available: bool
) -> None:
    speakers = {
        # 10 s + 8 s selected (1 s turn dropped), 19 s total.
        "spk0": [(0.0, 10.0), (12.0, 20.0), (21.0, 22.0)],
        # 2 s total -> below MIN_SPEAKER_SECONDS -> fallback.
        "spk1": [(25.0, 27.0)],
    }
    job_path, artifacts_dir, cfg = _make_job(tmp_path, speakers)

    manifest = extract_references(str(job_path), cfg)

    rows = json.loads((artifacts_dir / REFERENCES_NAME).read_text(encoding="utf-8"))
    by_speaker = {row["speaker"]: row for row in rows}

    assert by_speaker["spk0"]["fallback"] is False
    assert by_speaker["spk0"]["file"] == f"{REFERENCES_DIR}/spk0.wav"
    assert by_speaker["spk0"]["seconds"] == pytest.approx(18.0, abs=0.01)

    ref_path = artifacts_dir / REFERENCES_DIR / "spk0.wav"
    assert ref_path.exists()
    assert wav_duration(ref_path) == pytest.approx(18.0, abs=0.1)

    assert by_speaker["spk1"]["fallback"] is True
    assert by_speaker["spk1"]["file"] == ""
    assert not (artifacts_dir / REFERENCES_DIR / "spk1.wav").exists()

    assert manifest.stages["references"] == "done"
    assert manifest.paths["references"] == REFERENCES_DIR
    assert manifest.paths["references_json"] == REFERENCES_NAME


def test_extract_references_is_deterministic(
    tmp_path: Path, ffmpeg_available: bool
) -> None:
    job_path, artifacts_dir, cfg = _make_job(
        tmp_path, {"spk0": [(0.0, 10.0), (12.0, 20.0)]}
    )

    extract_references(str(job_path), cfg)
    ref_path = artifacts_dir / REFERENCES_DIR / "spk0.wav"
    first_bytes = ref_path.read_bytes()
    first_mtime = ref_path.stat().st_mtime_ns

    # Re-running reuses the file (same signature) -> untouched.
    extract_references(str(job_path), cfg)
    assert ref_path.stat().st_mtime_ns == first_mtime
    assert ref_path.read_bytes() == first_bytes

    # A forced re-extract (file deleted) produces identical bytes.
    ref_path.unlink()
    extract_references(str(job_path), cfg)
    assert ref_path.read_bytes() == first_bytes


# ---------------------------------------------------------------- validation
def test_validate_clone_speakers_requires_diarize() -> None:
    cfg = JobConfig(
        clone_speakers=True, tts_engine="vixtts", allow_nc_models=True
    )
    with pytest.raises(ValueError) as excinfo:
        validate_clone_speakers(cfg)
    assert "--diarize" in str(excinfo.value)


def test_validate_clone_speakers_rejects_piper() -> None:
    cfg = JobConfig(
        clone_speakers=True, diarize=True, tts_engine="piper", allow_nc_models=True
    )
    with pytest.raises(ValueError) as excinfo:
        validate_clone_speakers(cfg)
    assert "--tts-engine vixtts" in str(excinfo.value)


def test_validate_clone_speakers_requires_nc_optin() -> None:
    cfg = JobConfig(
        clone_speakers=True, diarize=True, tts_engine="vixtts", allow_nc_models=False
    )
    with pytest.raises(ValueError) as excinfo:
        validate_clone_speakers(cfg)
    assert "--allow-nc-models" in str(excinfo.value)


def test_validate_clone_speakers_ok_and_noop() -> None:
    validate_clone_speakers(
        JobConfig(
            clone_speakers=True, diarize=True, tts_engine="vixtts", allow_nc_models=True
        )
    )
    validate_clone_speakers(JobConfig())  # cloning off -> no-op


def test_required_stages_includes_references() -> None:
    assert pipeline.required_stages("dub", diarize=True, clone_speakers=True) == [
        "ingest",
        "transcribe",
        "diarize",
        "references",
        "translate",
        "speak",
        "timing",
        "mix",
        "export",
    ]


def test_run_pipeline_validates_clone_before_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        pipeline.ingest_mod, "ingest", lambda *args, **kwargs: calls.append("ingest")
    )

    with pytest.raises(ValueError, match="clone-speakers"):
        pipeline.run_pipeline(
            "video.mp4",
            JobConfig(cache_dir=str(tmp_path), clone_speakers=True),
        )

    assert calls == []


# ------------------------------------------------------------------ resolver
class _FakeEngine:
    def __init__(self, ref_voice=None) -> None:
        self.ref_voice = ref_voice


def test_resolver_uses_clone_reference_per_speaker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_dir = tmp_path / "cache"
    artifacts_dir = cache_dir / "hash1"
    ref_path = artifacts_dir / REFERENCES_DIR / "spk0.wav"
    ref_path.parent.mkdir(parents=True)
    ref_path.write_bytes(b"RIFF")
    (artifacts_dir / REFERENCES_NAME).write_text(
        json.dumps(
            [
                {
                    "speaker": "spk0",
                    "file": f"{REFERENCES_DIR}/spk0.wav",
                    "seconds": 12.0,
                    "fallback": False,
                    "signature": "0.000-12.000",
                },
                {
                    "speaker": "spk1",
                    "file": "",
                    "seconds": 0.0,
                    "fallback": True,
                    "signature": "",
                },
            ]
        ),
        encoding="utf-8",
    )
    manifest = JobManifest(
        input_path="demo.mp4",
        video_hash="hash1",
        cache_dir=str(cache_dir),
        tts_engine="vixtts",
    )
    cfg = JobConfig(
        clone_speakers=True,
        diarize=True,
        tts_engine="vixtts",
        allow_nc_models=True,
        ref_voice=None,
    )

    monkeypatch.setattr(
        "vidub.engines.tts.vixtts.VixttsTTS",
        lambda ref_voice=None: _FakeEngine(ref_voice),
    )

    resolver = _VoiceResolver(manifest, cfg)
    cloned = resolver.for_speaker("spk0")
    fallback = resolver.for_speaker("spk1")

    assert cloned.ref_voice == str(ref_path)
    assert fallback.ref_voice is None  # engine default
