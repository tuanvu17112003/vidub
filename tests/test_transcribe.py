"""Tests for the S1 transcribe stage (M1-T2).

Two layers:

* a fast logic test with a stubbed ASR engine (no model, no network), and
* a real end-to-end test that builds/ingests ``samples/demo_en.mp4`` and runs
  the actual faster-whisper ``tiny`` model (skipped if the model can't load,
  e.g. offline).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pysubs2
import pytest

from vidub.config import JobConfig
from vidub.manifest import JobManifest, Segment
from vidub.stages import transcribe as transcribe_mod
from vidub.stages.ingest import ingest
from vidub.stages.transcribe import SEGMENTS_NAME, SRT_NAME, transcribe
from vidub.utils import punct as punct_mod

ROOT = Path(__file__).resolve().parents[1]
ENGINE_RUNNER = ROOT / "tests" / "_engine_runner.py"

KEYWORDS = {"video", "speech", "weather", "vietnamese", "testing", "listening"}


def _load_make_sample():
    """Import ``samples/make_sample.py`` without it being a package."""
    spec = importlib.util.spec_from_file_location(
        "vidub_make_sample", ROOT / "samples" / "make_sample.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _StubASR:
    """Fake engine returning fixed segments; ignores the WAV contents."""

    def __init__(self, model_size: str, device: str = "auto") -> None:
        self.model_size = model_size
        self.device = device

    def transcribe(
        self,
        wav_path: str,
        language: str | None = None,
        word_timestamps: bool = True,
    ) -> list[Segment]:
        first = Segment(start=0.0, end=1.2, text="Hello world from a stub")
        if word_timestamps:
            first.words = [
                {"start": 0.0, "end": 0.3, "word": "Hello"},
                {"start": 0.3, "end": 0.5, "word": "world"},
            ]
        # second segment deliberately has no words (optional-safe path)
        return [first, Segment(start=1.5, end=2.8, text="Second stub segment")]


def _make_stub_job(tmp_path: Path) -> tuple[Path, Path, JobConfig]:
    cache_dir = tmp_path / "cache"
    artifacts_dir = cache_dir / "deadbeef"
    artifacts_dir.mkdir(parents=True)
    # Dummy audio: the stub engine never reads it.
    (artifacts_dir / "audio.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

    job_path = artifacts_dir / "job.json"
    JobManifest(
        input_path="demo.mp4",
        video_hash="deadbeef",
        cache_dir=str(cache_dir),
        asr_model="tiny",
        paths={"audio": "audio.wav", "job": "job.json"},
        stages={"ingest": "done"},
    ).save(job_path)
    return job_path, artifacts_dir, JobConfig(cache_dir=str(cache_dir))


def test_transcribe_writes_artifacts_with_stub(monkeypatch, tmp_path) -> None:
    job_path, artifacts_dir, cfg = _make_stub_job(tmp_path)

    monkeypatch.setattr(
        transcribe_mod,
        "build_asr",
        lambda model, device="auto": _StubASR(model, device),
    )

    result = transcribe(str(job_path), cfg)

    segments_path = artifacts_dir / SEGMENTS_NAME
    srt_path = artifacts_dir / SRT_NAME
    assert segments_path.exists()
    assert srt_path.exists()

    data = json.loads(segments_path.read_text(encoding="utf-8"))
    assert len(data) == 2
    assert data[0]["text"] == "Hello world from a stub"
    assert data[0]["speaker"] == "spk0"
    assert data[0]["start"] == 0.0 and data[0]["end"] == 1.2
    # word-level timing is persisted
    assert data[0]["words"] == [
        {"start": 0.0, "end": 0.3, "word": "Hello"},
        {"start": 0.3, "end": 0.5, "word": "world"},
    ]
    # a segment without words is valid too
    assert data[1]["words"] is None

    subs = pysubs2.load(str(srt_path))
    assert len(subs) == 2
    assert subs[0].text == "Hello world from a stub"
    # cue end tightened to last word end + 0.3 = 0.8 (segment end is 1.2)
    assert subs[0].end == pysubs2.make_time(s=0.8)
    # no words -> the segment end is used unchanged
    assert subs[1].end == pysubs2.make_time(s=2.8)

    reloaded = JobManifest.load(job_path)
    assert reloaded.paths["segments"] == SEGMENTS_NAME
    assert reloaded.paths["srt"] == SRT_NAME
    assert reloaded.stages["transcribe"] == "done"
    assert len(reloaded.segments) == 2
    assert reloaded.segments[0].words[0]["word"] == "Hello"
    assert result.video_hash == "deadbeef"


def test_transcribe_words_disabled_is_optional_safe(monkeypatch, tmp_path) -> None:
    job_path, artifacts_dir, cfg = _make_stub_job(tmp_path)
    cfg.asr_words = False

    monkeypatch.setattr(
        transcribe_mod,
        "build_asr",
        lambda model, device="auto": _StubASR(model, device),
    )

    transcribe(str(job_path), cfg)

    data = json.loads((artifacts_dir / SEGMENTS_NAME).read_text(encoding="utf-8"))
    assert all(entry["words"] is None for entry in data)

    subs = pysubs2.load(str(artifacts_dir / SRT_NAME))
    assert subs[0].end == pysubs2.make_time(s=1.2)


class _VietnameseStubASR:
    """Fake ASR that hallucinates Vietnamese on non-Vietnamese (English) audio.

    Mirrors the M2 bug: a Vietnamese-specialist model (``vi-small``) run on
    English audio returns Vietnamese-diacritic text.
    """

    def __init__(self, model_size: str, device: str = "auto") -> None:
        self.model_size = model_size
        self.device = device

    def transcribe(
        self,
        wav_path: str,
        language: str | None = None,
        word_timestamps: bool = True,
    ) -> list[Segment]:
        return [
            Segment(
                start=0.0,
                end=3.0,
                text="Đây là clip trình diễn bài diễn văn xử lý hàng loạt",
            )
        ]


def test_language_guard_raises_on_vietnamese_transcript(
    monkeypatch, tmp_path
) -> None:
    job_path, artifacts_dir, cfg = _make_stub_job(tmp_path)
    assert cfg.allow_lang_mismatch is False  # default: guard is on
    monkeypatch.setattr(
        transcribe_mod,
        "build_asr",
        lambda model, device="auto": _VietnameseStubASR(model, device),
    )

    with pytest.raises(RuntimeError) as excinfo:
        transcribe(str(job_path), cfg)

    message = str(excinfo.value)
    assert "allow-lang-mismatch" in message
    assert "language mismatch" in message
    # names the model actually used and the measured ratio
    assert "tiny" in message
    assert "ratio" in message
    # the guard runs BEFORE any artifact is written
    assert not (artifacts_dir / SEGMENTS_NAME).exists()
    assert not (artifacts_dir / SRT_NAME).exists()


def test_language_guard_bypassed_with_flag(monkeypatch, tmp_path) -> None:
    job_path, artifacts_dir, cfg = _make_stub_job(tmp_path)
    cfg.allow_lang_mismatch = True

    monkeypatch.setattr(
        transcribe_mod,
        "build_asr",
        lambda model, device="auto": _VietnameseStubASR(model, device),
    )

    result = transcribe(str(job_path), cfg)

    assert (artifacts_dir / SEGMENTS_NAME).exists()
    assert (artifacts_dir / SRT_NAME).exists()
    data = json.loads((artifacts_dir / SEGMENTS_NAME).read_text(encoding="utf-8"))
    assert data[0]["text"].startswith("Đây là clip")
    assert result.stages["transcribe"] == "done"


@pytest.fixture(scope="session")
def demo_audio(tmp_path_factory, ffmpeg_available) -> Path:
    """Build + ingest ``samples/demo_en.mp4`` once per session; return the WAV."""
    make_sample = _load_make_sample()
    try:
        demo = make_sample.generate()
    except Exception as exc:  # noqa: BLE001 - environment problem -> skip
        pytest.skip(f"could not build samples/demo_en.mp4: {exc}")

    cfg = JobConfig(cache_dir=str(tmp_path_factory.mktemp("m1t2_cache")))
    manifest = ingest(str(demo), cfg)
    return manifest.artifacts_dir / manifest.paths["audio"]


def test_real_tiny_transcription(demo_audio: Path, tmp_path: Path) -> None:
    # Run the real model in its own process: a native hang can then only fail
    # this test (timeout) instead of wedging the whole suite.
    result_path = tmp_path / "asr_result.json"
    proc = subprocess.run(
        [sys.executable, str(ENGINE_RUNNER), "asr", str(result_path), str(demo_audio)],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if proc.returncode != 0:
        pytest.skip(
            f"could not load faster-whisper 'tiny' (offline?): "
            f"{proc.stderr.strip()[-500:]}"
        )

    segments = json.loads(result_path.read_text(encoding="utf-8"))["segments"]

    assert len(segments) >= 2, f"only {len(segments)} segment(s)"
    word_count = sum(len(seg["text"].split()) for seg in segments)
    assert word_count >= 6, f"only {word_count} word(s): {[s['text'] for s in segments]}"

    joined = " ".join(seg["text"].lower() for seg in segments)
    assert any(keyword in joined for keyword in KEYWORDS), joined


def test_real_whisper_device(demo_audio: Path, tmp_path: Path) -> None:
    """The runner JSON must state which device the real model resolved to.

    Neither CUDA nor CPU is assumed: the GPU probe in ``vidub.utils.cuda``
    decides, and if the CUDA DLLs are wired the engine must actually use cuda.
    """
    result_path = tmp_path / "asr_device_result.json"
    proc = subprocess.run(
        [sys.executable, str(ENGINE_RUNNER), "asr", str(result_path), str(demo_audio)],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if proc.returncode != 0:
        pytest.skip(
            f"could not load faster-whisper 'tiny' (offline?): "
            f"{proc.stderr.strip()[-500:]}"
        )

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["device"] in {"cuda", "cpu"}, payload
    assert payload["segments"], "no segments transcribed"

    if payload["device"] == "cuda":
        assert payload["compute_type"] == "float16"
    else:
        assert payload["compute_type"] == "int8"


def test_real_phowhisper_words(demo_audio: Path, tmp_path: Path) -> None:
    """PhoWhisper (vi-small) runs via the transformers fallback and yields words.

    ``vinai/PhoWhisper-small`` ships PyTorch weights only (no CTranslate2
    ``model.bin``), so this exercises the Plan-B engine. Skips cleanly when the
    model cannot be downloaded/loaded, logging the error text.
    """
    result_path = tmp_path / "phowhisper_result.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(ENGINE_RUNNER),
            "asr",
            str(result_path),
            str(demo_audio),
            "vinai/PhoWhisper-small",
        ],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode != 0:
        pytest.skip(
            "could not load vinai/PhoWhisper-small (offline/conversion?): "
            f"{proc.stderr.strip()[-800:]}"
        )

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    segments = payload["segments"]
    assert segments, "no segments transcribed"
    assert any(seg.get("words") for seg in segments), (
        f"no segment has word-level timing: {segments}"
    )


def test_asr_model_presets_and_resolution() -> None:
    from vidub.config import ASR_PRESETS, resolve_asr_model

    assert ASR_PRESETS["vi-small"] == "vinai/PhoWhisper-small"
    assert resolve_asr_model("vi-small") == "vinai/PhoWhisper-small"
    assert resolve_asr_model("cpu-base") == "base"
    assert resolve_asr_model("gpu-large") == "large-v3"
    assert resolve_asr_model("tiny") == "tiny"
    assert (
        resolve_asr_model("Systran/faster-whisper-small")
        == "Systran/faster-whisper-small"
    )
    # auto picks a concrete model but must not assume CUDA
    assert resolve_asr_model("auto") in {"large-v3", "base"}


def test_unknown_asr_model_fails_fast(capsys, tmp_path) -> None:
    from vidub.cli import main

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")

    rc = main(["run", str(video), "--asr-model", "bogus"])

    err = capsys.readouterr().err
    assert rc == 1
    assert "unknown ASR model" in err
    assert "vi-small" in err  # the message lists the valid presets


class _MetaStubASR:
    """Fake engine with per-segment QC meta; retries the flagged window clean.

    Returns two segments: a clean one and one whose compression ratio (3.5) puts
    it above ``COMPRESSION_RATIO_MAX`` (2.4), i.e. a hallucination. Exposing
    ``transcribe_window`` lets the default-on QC layer retry that window and
    accept the clean hypothesis.
    """

    CLEAN_META = {
        "avg_logprob": -0.2,
        "compression_ratio": 1.2,
        "no_speech_prob": 0.1,
        "temperature": 0.0,
    }
    FLAGGED_META = {
        "avg_logprob": -0.2,
        "compression_ratio": 3.5,
        "no_speech_prob": 0.1,
        "temperature": 0.0,
    }
    CORRECTED_TEXT = "The weather is sunny today."

    def __init__(self, model_size: str, device: str = "auto") -> None:
        self.model_size = model_size
        self.device = device
        self.window_calls: list[tuple[float, float]] = []

    def transcribe(
        self,
        wav_path: str,
        language: str | None = None,
        word_timestamps: bool = True,
    ) -> list[Segment]:
        return [
            Segment(
                start=0.0, end=1.2, text="Hello world from a meta stub",
                meta=dict(self.CLEAN_META),
            ),
            Segment(
                start=1.5, end=3.0, text="so so so so", meta=dict(self.FLAGGED_META)
            ),
        ]

    def transcribe_window(
        self, wav_path: str, start: float, end: float, language: str | None
    ) -> Segment:
        self.window_calls.append((start, end))
        return Segment(
            start=start, end=end, text=self.CORRECTED_TEXT,
            meta=dict(self.CLEAN_META),
        )


def test_transcribe_persists_segment_meta_and_qc(monkeypatch, tmp_path) -> None:
    job_path, artifacts_dir, cfg = _make_stub_job(tmp_path)
    engine = _MetaStubASR("tiny")
    monkeypatch.setattr(
        transcribe_mod, "build_asr", lambda model, device="auto": engine
    )

    result = transcribe(str(job_path), cfg)

    # (a) every written row carries its segment meta
    data = json.loads((artifacts_dir / SEGMENTS_NAME).read_text(encoding="utf-8"))
    assert data[0]["meta"] == _MetaStubASR.CLEAN_META
    assert isinstance(data[1]["meta"], dict)

    # (b) the QC artifact exists and counts the one (original) hallucination
    qc_path = artifacts_dir / "transcript_qc.json"
    assert qc_path.exists()
    artifact = json.loads(qc_path.read_text(encoding="utf-8"))
    assert "verdict" in artifact
    assert artifact["counts"].get("hallucination") == 1
    assert artifact["counts"].get("ok") == 1

    # (c) the manifest records a valid transcript verdict
    reloaded = JobManifest.load(job_path)
    assert reloaded.transcript_verdict in {"pass", "review", "fail"}
    assert result.transcript_verdict == reloaded.transcript_verdict

    # (d) the flagged window was retried once and the clean text replaced it
    assert len(engine.window_calls) == 1
    assert artifact["segments"][1]["retried"] is True
    assert artifact["segments"][1]["resolved"] is True
    assert data[1]["text"] == _MetaStubASR.CORRECTED_TEXT


# ------------------------------------------- M4-T6 punctuation restoration hook
def test_transcribe_restore_punct_enabled(monkeypatch, tmp_path) -> None:
    job_path, artifacts_dir, cfg = _make_stub_job(tmp_path)
    cfg.restore_punct = True
    monkeypatch.setattr(
        transcribe_mod,
        "build_asr",
        lambda model, device="auto": _StubASR(model, device),
    )

    class _FakeRestorer:
        def restore(self, text: str) -> tuple[str, int]:
            return text + " [restored]", 1

    monkeypatch.setattr(punct_mod, "PunctRestorer", _FakeRestorer)

    transcribe(str(job_path), cfg)

    data = json.loads((artifacts_dir / SEGMENTS_NAME).read_text(encoding="utf-8"))
    assert data[0]["text"] == "Hello world from a stub [restored]"
    assert data[1]["text"] == "Second stub segment [restored]"

    subs = pysubs2.load(str(artifacts_dir / SRT_NAME))
    assert subs[0].text == "Hello world from a stub [restored]"


def test_transcribe_restore_punct_default_off(monkeypatch, tmp_path) -> None:
    job_path, artifacts_dir, cfg = _make_stub_job(tmp_path)
    assert cfg.restore_punct is False  # opt-in flag defaults off
    monkeypatch.setattr(
        transcribe_mod,
        "build_asr",
        lambda model, device="auto": _StubASR(model, device),
    )

    def _boom(*args, **kwargs):  # noqa: ANN002, ANN003 - test double
        raise AssertionError("the restorer must not load with restore_punct off")

    monkeypatch.setattr(punct_mod, "PunctRestorer", _boom)

    transcribe(str(job_path), cfg)

    data = json.loads((artifacts_dir / SEGMENTS_NAME).read_text(encoding="utf-8"))
    assert data[0]["text"] == "Hello world from a stub"
    assert data[1]["text"] == "Second stub segment"

    subs = pysubs2.load(str(artifacts_dir / SRT_NAME))
    assert subs[0].text == "Hello world from a stub"
