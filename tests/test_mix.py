"""Tests for the S5 mix stage (M1-T5).

Logic tests pre-place a fake Demucs output (so separation is never run and no
model is downloaded); a ``slow`` integration test runs the real Demucs htdemucs
model on the ``tiny_clip`` audio.
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
from pydub import AudioSegment

from vidub.config import JobConfig
from vidub.manifest import JobManifest
from vidub.stages import mix as mix_mod
from vidub.stages.ingest import ingest
from vidub.stages.mix import DUB_TRACK_NAME, mix, run_separation

STEM_RATE = 44100
ROOT = Path(__file__).resolve().parents[1]
ENGINE_RUNNER = ROOT / "tests" / "_engine_runner.py"


@pytest.fixture(autouse=True)
def _stub_sep_device(monkeypatch) -> None:
    """Keep the in-process logic tests off the torch import.

    Device selection is exercised by ``test_real_demucs_separation`` (in its own
    subprocess) and by ``test_mix_records_sep_device``; everything else only needs
    a deterministic value.
    """
    monkeypatch.setattr(mix_mod, "select_sep_device", lambda: "cpu")


def _write_wav(path: Path, duration: float, freq: float | None = None, amplitude: float = 0.4) -> None:
    """Mono 16-bit WAV; digital silence when ``freq`` is None."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = bytearray()
    for i in range(round(duration * STEM_RATE)):
        value = 0 if freq is None else int(amplitude * 32767 * math.sin(2 * math.pi * freq * i / STEM_RATE))
        frames += struct.pack("<h", value)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(STEM_RATE)
        wav_file.writeframes(bytes(frames))


def _rms(audio: AudioSegment, start_s: float, end_s: float) -> int:
    return audio[int(start_s * 1000) : int(end_s * 1000)].rms


def _make_job(
    tmp_path: Path,
    mode: str = "dub",
    *,
    bed_freq: float | None = None,
    vocals_freq: float | None = None,
    with_segment: bool = True,
) -> tuple[Path, Path, JobConfig]:
    cache_dir = tmp_path / "cache"
    artifacts_dir = cache_dir / "hash1"
    artifacts_dir.mkdir(parents=True)

    duration = 4.0
    _write_wav(artifacts_dir / "audio.wav", duration)

    # Fake Demucs output in the exact layout the stage expects.
    track_dir = artifacts_dir / "sep" / "htdemucs" / "audio"
    _write_wav(track_dir / "no_vocals.wav", duration, bed_freq)
    _write_wav(track_dir / "vocals.wav", duration, vocals_freq)

    timed: list[dict] = []
    if with_segment:
        freq = 880.0
        seg_path = artifacts_dir / "speech" / "seg_0000_fit.wav"
        _write_wav(seg_path, 1.0, freq, amplitude=0.8)
        timed.append(
            {"idx": 0, "file": "speech/seg_0000_fit.wav", "start": 2.0, "end": 3.0, "slot_start": 2.0, "stretch": 1.0}
        )
    (artifacts_dir / "timed.json").write_text(json.dumps(timed), encoding="utf-8")

    job_path = artifacts_dir / "job.json"
    JobManifest(
        input_path="demo.mp4",
        video_hash="hash1",
        duration=duration,
        cache_dir=str(cache_dir),
        mode=mode,
        paths={"audio": "audio.wav", "timed": "timed.json"},
        stages={"timing": "done"},
    ).save(job_path)
    return job_path, artifacts_dir, JobConfig(cache_dir=str(cache_dir))


def test_mix_places_segments_at_slots(tmp_path: Path) -> None:
    job_path, artifacts_dir, cfg = _make_job(tmp_path, mode="dub", bed_freq=None)

    manifest = mix(str(job_path), cfg)

    dub_track = artifacts_dir / DUB_TRACK_NAME
    assert dub_track.exists()

    audio = AudioSegment.from_wav(str(dub_track))
    assert abs(len(audio) / 1000.0 - 4.0) < 0.1

    inside = _rms(audio, 2.1, 2.9)     # where the segment was overlaid
    outside = _rms(audio, 0.0, 1.0)    # silence, far from the segment
    assert inside > 1000, f"segment too quiet: {inside}"
    assert outside < inside / 4, f"segment not localized: inside={inside} outside={outside}"

    reloaded = JobManifest.load(job_path)
    assert reloaded.paths["dub_track"] == DUB_TRACK_NAME
    assert reloaded.stages["mix"] == "done"
    assert manifest.video_hash == "hash1"


def test_mix_voice_over_includes_original_vocals(tmp_path: Path) -> None:
    job_path, artifacts_dir, cfg = _make_job(
        tmp_path, mode="voice-over", bed_freq=None, vocals_freq=330.0, with_segment=False
    )

    mix(str(job_path), cfg)

    audio = AudioSegment.from_wav(str(artifacts_dir / DUB_TRACK_NAME))
    outside = _rms(audio, 0.0, 1.0)
    assert outside > 0, "voice-over should include the original (ducked) vocals"


def test_mix_subtitles_only_is_skipped(tmp_path: Path) -> None:
    job_path, _artifacts_dir, cfg = _make_job(tmp_path, mode="subtitles-only")

    with pytest.raises(RuntimeError) as excinfo:
        mix(str(job_path), cfg)
    assert "skipped" in str(excinfo.value)
    assert "subtitles-only" in str(excinfo.value)


def test_separation_reuses_existing_stems(tmp_path: Path, monkeypatch) -> None:
    artifacts_dir = tmp_path / "cache" / "hash1"
    audio_path = artifacts_dir / "audio.wav"
    _write_wav(audio_path, 1.0)
    track_dir = artifacts_dir / "sep" / "htdemucs" / "audio"
    _write_wav(track_dir / "no_vocals.wav", 1.0)
    _write_wav(track_dir / "vocals.wav", 1.0)

    def _boom(*_args, **_kwargs):
        raise AssertionError("demucs must not run when the stems already exist")

    monkeypatch.setattr(mix_mod, "run", _boom)

    assert run_separation(audio_path, artifacts_dir / "sep") == track_dir


def test_mix_records_sep_device(tmp_path: Path, monkeypatch) -> None:
    """The resolved separation device is persisted on the manifest (report meta)."""
    job_path, _artifacts_dir, cfg = _make_job(tmp_path, mode="dub", bed_freq=None)
    monkeypatch.setattr(mix_mod, "select_sep_device", lambda: "cuda")

    manifest = mix(str(job_path), cfg)

    assert manifest.sep_device == "cuda"
    assert JobManifest.load(job_path).sep_device == "cuda"


@pytest.mark.slow
def test_real_demucs_separation(tiny_clip: Path, tmp_path: Path, ffmpeg_available: bool) -> None:
    cfg = JobConfig(cache_dir=str(tmp_path / "cache"))
    manifest = ingest(str(tiny_clip), cfg)
    audio_path = manifest.artifacts_dir / manifest.paths["audio"]

    # Demucs runs in its own process so a native hang can't wedge the suite.
    result_path = tmp_path / "demucs_result.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(ENGINE_RUNNER),
            "demucs",
            str(result_path),
            str(audio_path),
            str(manifest.artifacts_dir / "sep"),
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stderr[-1000:]

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    track_dir = Path(payload["track_dir"])
    assert (track_dir / "vocals.wav").exists()
    assert (track_dir / "no_vocals.wav").exists()
    # The runner echoes the device it selected (cuda when torch sees a GPU).
    assert payload["device"] in {"cuda", "cpu"}, payload
    if payload["device"] == "cuda":
        # The whole point of the CUDA torch wheel: Demucs must actually use it.
        import torch

        assert torch.cuda.is_available()


def test_mix_voice_over_uses_configured_gain(tmp_path: Path, monkeypatch) -> None:
    """Voice-over gain must come from ``cfg.voice_over_gain``, not a constant."""
    job_path, _artifacts_dir, cfg = _make_job(
        tmp_path, mode="voice-over", bed_freq=None, vocals_freq=330.0, with_segment=False
    )
    cfg.voice_over_gain = 0.5

    captured: dict[str, float] = {}
    original = mix_mod._linear_gain_db

    def spy(gain: float) -> float:
        captured["gain"] = gain
        return original(gain)

    monkeypatch.setattr(mix_mod, "_linear_gain_db", spy)

    mix(str(job_path), cfg)

    assert captured["gain"] == 0.5


def test_mix_places_per_unit_rows_at_their_starts(tmp_path: Path) -> None:
    """UA-1: per-unit ``timed.json`` rows assemble at their unit starts.

    Two unit rows with a silence gap must both land at their ``start`` with no
    overlap, proving the mix stage is generic over the row cardinality.
    """
    job_path, artifacts_dir, cfg = _make_job(tmp_path, mode="dub", bed_freq=None)

    speech_dir = artifacts_dir / "speech"
    speech_dir.mkdir(parents=True, exist_ok=True)
    _write_wav(speech_dir / "unit_0000.wav", 1.0, 660.0, amplitude=0.8)
    _write_wav(speech_dir / "unit_0001.wav", 1.0, 990.0, amplitude=0.8)
    timed = [
        {
            "idx": 0,
            "members": [0, 1],
            "file": "speech/unit_0000.wav",
            "start": 0.5,
            "end": 1.5,
            "slot_start": 0.5,
            "stretch": 1.0,
        },
        {
            "idx": 1,
            "members": [2],
            "file": "speech/unit_0001.wav",
            "start": 2.5,
            "end": 3.5,
            "slot_start": 2.5,
            "stretch": 1.0,
        },
    ]
    (artifacts_dir / "timed.json").write_text(json.dumps(timed), encoding="utf-8")

    mix(str(job_path), cfg)

    audio = AudioSegment.from_wav(str(artifacts_dir / DUB_TRACK_NAME))
    first = _rms(audio, 0.6, 1.4)
    second = _rms(audio, 2.6, 3.4)
    gap = _rms(audio, 1.6, 2.4)
    assert first > 1000, f"first unit missing at its start: {first}"
    assert second > 1000, f"second unit missing at its start: {second}"
    # the gap between the two units stays silent (no bleed/overlap)
    assert gap < min(first, second) / 4, (first, second, gap)
