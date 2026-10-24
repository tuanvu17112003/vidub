"""Tests for Level-1 transcript QC (M4-T1, QUALITY-ROADMAP.md §3.2).

All fast: stubbed engines, synthetic WAVs, no models/network. Covers the
classification labels and precedence, the verdict rules, the one-retry flow
(and its absence when the engine has no ``transcribe_window``), the review
packet (txt + ``wave``-readable snippet), the transcribe-stage integration and
the manifest ``meta`` / ``transcript_verdict`` roundtrip.
"""

from __future__ import annotations

import array
import json
import wave
from pathlib import Path

import pytest

from vidub.config import JobConfig
from vidub.manifest import JobManifest, Segment
from vidub.stages import transcribe as transcribe_mod
from vidub.stages.transcribe import (
    SEGMENTS_NAME,
    TRANSCRIPT_QC_NAME,
    transcribe,
)
from vidub.utils import qc
from vidub.utils.qc import classify_segment, qc_transcript


# --------------------------------------------------------------------- helpers
def _meta(**overrides) -> dict:
    """A clean faster-whisper meta dict, with optional overrides."""
    base = {
        "avg_logprob": -0.2,
        "compression_ratio": 1.2,
        "no_speech_prob": 0.05,
        "temperature": 0.0,
    }
    base.update(overrides)
    return base


def _write_wav(path: Path, seconds: float = 10.0, rate: int = 16000) -> Path:
    """Write a valid mono 16-bit PCM WAV of ``seconds`` length."""
    frames = array.array("h", ((i * 37) % 2000 - 1000 for i in range(int(seconds * rate))))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(frames.tobytes())
    return path


# -------------------------------------------------------------- classification
def test_classify_ok() -> None:
    label, reasons = classify_segment("This is a clean sentence.", _meta(), "en")
    assert label == qc.LABEL_OK
    assert reasons == []


def test_classify_hallucination_compression_ratio() -> None:
    label, reasons = classify_segment("anything at all", _meta(compression_ratio=3.1), "en")
    assert label == qc.LABEL_HALLUCINATION
    assert "compression_ratio=3.1>2.4" in reasons


def test_classify_hallucination_repetition_text_only() -> None:
    # No meta at all (PhoWhisper / old manifest): the text loop still flags.
    text = " ".join(["the weather"] * 8)  # 16 tokens, 3-gram repeated 7x
    label, reasons = classify_segment(text, None, "en")
    assert label == qc.LABEL_HALLUCINATION
    assert any("repeated_3gram" in r for r in reasons)


def test_classify_short_repetition_is_not_hallucination() -> None:
    # Under REPEAT_MIN_TOKENS tokens: a loop cannot be trusted from so little.
    label, _ = classify_segment("no no no no", None, "en")
    assert label == qc.LABEL_OK


def test_classify_garbled_low_avg_logprob() -> None:
    label, reasons = classify_segment("some words here", _meta(avg_logprob=-1.5), "en")
    assert label == qc.LABEL_GARBLED
    assert "avg_logprob=-1.50<-1.0" in reasons


def test_classify_garbled_high_no_speech_prob() -> None:
    label, _ = classify_segment("some words here", _meta(no_speech_prob=0.7), "en")
    assert label == qc.LABEL_GARBLED


def test_classify_language_mismatch_and_vi_is_never_a_mismatch() -> None:
    text = "Đây là clip trình diễn bài diễn văn xử lý hàng loạt"
    label, reasons = classify_segment(text, _meta(), "en")
    assert label == qc.LABEL_LANG_MISMATCH
    assert reasons  # carries the detect_mismatch explanation
    assert classify_segment(text, _meta(), "vi")[0] == qc.LABEL_OK


def test_classify_segmentation_duration() -> None:
    label, reasons = classify_segment("a normal sentence", _meta(duration=45.0), "en")
    assert label == qc.LABEL_SEGMENTATION
    assert "duration=45.0s>30.0s" in reasons


def test_classify_segmentation_non_monotonic_words() -> None:
    words = [
        {"start": 1.0, "end": 1.5, "word": "a"},
        {"start": 0.2, "end": 0.4, "word": "b"},  # starts decrease by > 0.01
    ]
    label, reasons = classify_segment("a b", _meta(words=words), "en")
    assert label == qc.LABEL_SEGMENTATION
    assert "non-monotonic word timing" in reasons


def test_classify_segmentation_word_end_before_start() -> None:
    words = [{"start": 1.0, "end": 0.5, "word": "a"}]
    label, _ = classify_segment("a", _meta(words=words), "en")
    assert label == qc.LABEL_SEGMENTATION


def test_classify_precedence_and_reasons_accumulate() -> None:
    label, reasons = classify_segment(
        "text", _meta(compression_ratio=9.0, avg_logprob=-3.0, no_speech_prob=0.9), "en"
    )
    assert label == qc.LABEL_HALLUCINATION  # highest precedence
    assert any("compression_ratio" in r for r in reasons)
    assert any("avg_logprob" in r for r in reasons)  # lower issues kept as reasons
    assert any("no_speech_prob" in r for r in reasons)


def test_classify_tolerates_missing_signals() -> None:
    # None / absent keys must never raise, only degrade to text-only stats.
    label, reasons = classify_segment("a plain english sentence", None, "en")
    assert (label, reasons) == (qc.LABEL_OK, [])
    label, _ = classify_segment("plain", {"avg_logprob": None, "compression_ratio": None}, "en")
    assert label == qc.LABEL_OK
    label, _ = classify_segment("plain", {"avg_logprob": "not-a-number"}, "en")
    assert label == qc.LABEL_OK


# --------------------------------------------------------------------- verdict
def test_verdict_pass_when_all_ok() -> None:
    segments = [Segment(0.0, 2.0, "A clean sentence.", meta=_meta())]
    result = qc_transcript(segments, "en")
    assert result["verdict"] == "pass"
    assert result["counts"][qc.LABEL_OK] == 1
    assert result["flagged_ratio_unresolved"] == 0.0
    assert result["segments"][0]["resolved"] is False


def test_verdict_fail_on_one_unresolved_hallucination() -> None:
    segments = [Segment(0.0, 2.0, "loop", meta=_meta(compression_ratio=5.0))]
    result = qc_transcript(segments, "en")  # engine None -> no retry
    assert result["verdict"] == "fail"
    assert result["flagged_ratio_unresolved"] == 1.0


def test_verdict_review_when_unresolved_below_fail_ratio() -> None:
    segments = [Segment(0.0, 2.0, f"clean {i} sentence", meta=_meta()) for i in range(10)]
    segments[0].meta = _meta(avg_logprob=-2.0)  # 1/10 unresolved = 0.1 < 0.3
    result = qc_transcript(segments, "en")
    assert result["verdict"] == "review"
    assert result["flagged_ratio_unresolved"] == pytest.approx(0.1)


def test_verdict_fail_at_ratio_threshold() -> None:
    segments = [Segment(0.0, 2.0, f"clean {i} sentence", meta=_meta()) for i in range(10)]
    for i in range(3):
        segments[i].meta = _meta(avg_logprob=-2.0)  # 3/10 = 0.3 -> fail
    result = qc_transcript(segments, "en")
    assert result["verdict"] == "fail"
    assert result["flagged_ratio_unresolved"] == pytest.approx(0.3)


def test_verdict_pass_with_zero_segments() -> None:
    result = qc_transcript([], "en")
    assert result["verdict"] == "pass"
    assert result["flagged_ratio_unresolved"] == 0.0
    assert result["segments"] == []


def test_qc_composes_duration_from_segment_bounds() -> None:
    segment = Segment(0.0, 40.0, "a normal long sentence", meta=_meta())
    result = qc_transcript([segment], "en")
    assert result["counts"][qc.LABEL_SEGMENTATION] == 1
    # one unresolved segment out of one -> ratio 1.0 >= FAIL_RATIO -> fail
    assert result["verdict"] == "fail"


# --------------------------------------------------------------------- retries
class _RetryEngine:
    """Stub engine exposing ``transcribe_window``; returns fixed retry text/meta."""

    def __init__(self, retry_text: str, retry_meta: dict | None = None) -> None:
        self.device = "cpu"
        self.retry_text = retry_text
        self.retry_meta = retry_meta
        self.calls: list[tuple[float, float, str | None]] = []

    def transcribe_window(
        self, wav_path: str, start: float, end: float, language: str | None
    ) -> Segment:
        self.calls.append((start, end, language))
        meta = dict(self.retry_meta) if self.retry_meta is not None else _meta()
        return Segment(start, end, self.retry_text, meta=meta)


def test_retry_resolves_and_replaces_in_place() -> None:
    segment = Segment(0.0, 2.0, "bad bad bad", meta=_meta(compression_ratio=5.0))
    engine = _RetryEngine("This is the corrected sentence.")
    result = qc_transcript([segment], "en", engine=engine)

    assert engine.calls == [(0.0, 2.0, "en")]
    assert segment.text == "This is the corrected sentence."
    assert result["verdict"] == "pass"
    entry = result["segments"][0]
    assert entry["label"] == qc.LABEL_HALLUCINATION  # original label kept
    assert entry["retried"] is True
    assert entry["resolved"] is True
    assert entry["retry_label"] == qc.LABEL_OK
    assert entry["retry_text"] == "This is the corrected sentence."


def test_retry_still_flagged_stays_unresolved() -> None:
    segment = Segment(0.0, 2.0, "bad bad bad", meta=_meta(compression_ratio=5.0))
    engine = _RetryEngine(
        "still bad bad bad bad bad", retry_meta=_meta(compression_ratio=5.0)
    )
    result = qc_transcript([segment], "en", engine=engine)

    entry = result["segments"][0]
    assert entry["retried"] is True
    assert entry["resolved"] is False
    assert entry["retry_label"] == qc.LABEL_HALLUCINATION
    assert segment.text == "bad bad bad"  # unchanged (not accepted)
    assert result["verdict"] == "fail"


def test_one_retry_maximum_per_segment() -> None:
    segments = [
        Segment(float(i), float(i) + 1.0, f"x{i}", meta=_meta(compression_ratio=5.0))
        for i in range(3)
    ]
    engine = _RetryEngine("still bad bad bad bad bad")
    qc_transcript(segments, "en", engine=engine)
    assert len(engine.calls) == 3  # exactly one window call per flagged segment


def test_ok_segments_are_never_retried() -> None:
    segments = [Segment(0.0, 2.0, "a clean sentence", meta=_meta())]
    engine = _RetryEngine("unused")
    qc_transcript(segments, "en", engine=engine)
    assert engine.calls == []


class _NoWindowEngine:
    """Engine mirroring PhoWhisper: no ``transcribe_window`` method."""

    def __init__(self) -> None:
        self.device = "cpu"


def test_engine_without_transcribe_window_no_retry_no_crash() -> None:
    segment = Segment(0.0, 2.0, "bad bad bad", meta=_meta(compression_ratio=5.0))
    result = qc_transcript([segment], "en", engine=_NoWindowEngine())
    entry = result["segments"][0]
    assert entry["retried"] is False
    assert entry["resolved"] is False
    assert result["verdict"] == "fail"


def test_engine_retry_exception_is_tolerated() -> None:
    class _BrokenEngine:
        def transcribe_window(self, *args, **kwargs):
            raise RuntimeError("boom")

    segment = Segment(0.0, 2.0, "bad bad bad", meta=_meta(compression_ratio=5.0))
    result = qc_transcript([segment], "en", engine=_BrokenEngine())
    entry = result["segments"][0]
    assert entry["retried"] is True and entry["resolved"] is False
    assert result["verdict"] == "fail"


# ---------------------------------------------------------------- review packet
def test_review_packet_txt_and_wav_written(tmp_path: Path) -> None:
    audio = _write_wav(tmp_path / "audio.wav", seconds=10.0)
    segment = Segment(1.0, 3.0, "bad bad bad", meta=_meta(compression_ratio=5.0))
    result = qc_transcript(
        [segment], "en", engine=None, audio_path=audio, out_dir=tmp_path
    )
    assert result["verdict"] == "fail"

    txt = tmp_path / "review" / "transcript" / "seg-0.txt"
    wav = tmp_path / "review" / "transcript" / "seg-0.wav"
    assert txt.exists() and wav.exists()

    content = txt.read_text(encoding="utf-8")
    assert "label: hallucination" in content
    assert "original text:" in content
    assert "bad bad bad" in content

    with wave.open(str(wav), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 16000
        assert handle.getnframes() == pytest.approx(2 * 16000, abs=2)


def test_review_packet_caps_snippet_at_30_seconds(tmp_path: Path) -> None:
    audio = _write_wav(tmp_path / "audio.wav", seconds=60.0)
    segment = Segment(0.0, 50.0, "bad bad bad", meta=_meta(compression_ratio=5.0))
    qc_transcript([segment], "en", audio_path=audio, out_dir=tmp_path)
    wav = tmp_path / "review" / "transcript" / "seg-0.wav"
    with wave.open(str(wav), "rb") as handle:
        assert handle.getnframes() <= int(30 * 16000)


def test_review_packet_includes_retry_hypothesis(tmp_path: Path) -> None:
    audio = _write_wav(tmp_path / "audio.wav", seconds=10.0)
    segment = Segment(0.0, 2.0, "bad bad bad", meta=_meta(compression_ratio=5.0))
    engine = _RetryEngine(
        "still bad bad bad bad bad", retry_meta=_meta(compression_ratio=5.0)
    )
    qc_transcript([segment], "en", engine=engine, audio_path=audio, out_dir=tmp_path)
    content = (tmp_path / "review" / "transcript" / "seg-0.txt").read_text(encoding="utf-8")
    assert "retry label: hallucination" in content
    assert "still bad bad bad bad bad" in content


def test_resolved_segments_are_not_in_the_packet(tmp_path: Path) -> None:
    audio = _write_wav(tmp_path / "audio.wav", seconds=10.0)
    segment = Segment(0.0, 2.0, "bad bad bad", meta=_meta(compression_ratio=5.0))
    engine = _RetryEngine("a clean corrected sentence")
    result = qc_transcript(
        [segment], "en", engine=engine, audio_path=audio, out_dir=tmp_path
    )
    assert result["verdict"] == "pass"
    assert not (tmp_path / "review" / "transcript" / "seg-0.txt").exists()


def test_review_packet_txt_written_even_without_audio(tmp_path: Path) -> None:
    segment = Segment(0.0, 2.0, "bad bad bad", meta=_meta(compression_ratio=5.0))
    result = qc_transcript([segment], "en", audio_path=None, out_dir=tmp_path)
    assert result["verdict"] == "fail"
    assert (tmp_path / "review" / "transcript" / "seg-0.txt").exists()
    assert not (tmp_path / "review" / "transcript" / "seg-0.wav").exists()


def test_unreadable_audio_never_crashes(tmp_path: Path) -> None:
    bad_audio = tmp_path / "audio.wav"
    bad_audio.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")  # not a real WAV
    segment = Segment(0.0, 2.0, "bad bad bad", meta=_meta(compression_ratio=5.0))
    result = qc_transcript([segment], "en", audio_path=bad_audio, out_dir=tmp_path)
    assert result["verdict"] == "fail"
    assert (tmp_path / "review" / "transcript" / "seg-0.txt").exists()


# ------------------------------------------------------ transcribe-stage wiring
def _make_stub_job(tmp_path: Path) -> tuple[Path, Path, JobConfig]:
    cache_dir = tmp_path / "cache"
    artifacts_dir = cache_dir / "deadbeef"
    artifacts_dir.mkdir(parents=True)
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


class _QcStageASR:
    """One clean + one hallucinated segment; retry returns a clean hypothesis."""

    def __init__(self, model: str, device: str = "auto") -> None:
        self.model_size = model
        self.device = "cpu"
        self.window_calls: list[tuple[float, float]] = []

    def transcribe(self, wav_path, language=None, word_timestamps=True):
        return [
            Segment(
                0.0,
                2.0,
                "This is a clean sentence.",
                words=[{"start": 0.0, "end": 2.0, "word": "This is a clean sentence."}],
                meta=_meta(),
            ),
            Segment(
                2.5,
                5.0,
                "bad bad bad bad bad bad bad",
                meta=_meta(compression_ratio=4.0),
            ),
        ]

    def transcribe_window(self, wav_path, start, end, language):
        self.window_calls.append((start, end))
        return Segment(
            start,
            end,
            "The weather is nice today.",
            words=[{"start": start, "end": end, "word": "The weather is nice today."}],
            meta=_meta(),
        )


def test_transcribe_stage_runs_qc_and_applies_correction(monkeypatch, tmp_path) -> None:
    job_path, artifacts_dir, cfg = _make_stub_job(tmp_path)
    engine = _QcStageASR("tiny")
    monkeypatch.setattr(
        transcribe_mod, "build_asr", lambda model, device="auto": engine
    )

    result = transcribe(str(job_path), cfg)

    qc_path = artifacts_dir / TRANSCRIPT_QC_NAME
    assert qc_path.exists()
    artifact = json.loads(qc_path.read_text(encoding="utf-8"))
    assert {"verdict", "counts", "flagged_ratio_unresolved", "segments"} <= set(artifact)
    assert artifact["verdict"] == "pass"
    assert artifact["segments"][1]["label"] == qc.LABEL_HALLUCINATION
    assert artifact["segments"][1]["resolved"] is True

    # corrected text is what downstream stages read from segments.json
    data = json.loads((artifacts_dir / SEGMENTS_NAME).read_text(encoding="utf-8"))
    assert data[1]["text"] == "The weather is nice today."

    # verdict recorded on the manifest + path registered
    assert result.transcript_verdict == "pass"
    assert result.paths["transcript_qc"] == TRANSCRIPT_QC_NAME
    reloaded = JobManifest.load(job_path)
    assert reloaded.transcript_verdict == "pass"
    assert len(engine.window_calls) == 1


class _StillFlaggedStageASR(_QcStageASR):
    """Retry keeps returning a hallucination -> unresolved -> packet + fail."""

    def transcribe_window(self, wav_path, start, end, language):
        self.window_calls.append((start, end))
        return Segment(
            start, end, "bad bad bad bad bad", meta=_meta(compression_ratio=4.0)
        )


def test_transcribe_stage_writes_packet_for_unresolved(monkeypatch, tmp_path) -> None:
    job_path, artifacts_dir, cfg = _make_stub_job(tmp_path)
    engine = _StillFlaggedStageASR("tiny")
    monkeypatch.setattr(
        transcribe_mod, "build_asr", lambda model, device="auto": engine
    )

    result = transcribe(str(job_path), cfg)

    assert result.transcript_verdict == "fail"
    artifact = json.loads((artifacts_dir / TRANSCRIPT_QC_NAME).read_text(encoding="utf-8"))
    assert artifact["verdict"] == "fail"
    entry = artifact["segments"][1]
    assert entry["retried"] is True and entry["resolved"] is False
    # the packet entry is still written (the dummy audio just yields no .wav)
    assert (artifacts_dir / "review" / "transcript" / "seg-1.txt").exists()


def test_transcribe_stage_qc_disabled_skips_everything(monkeypatch, tmp_path) -> None:
    job_path, artifacts_dir, cfg = _make_stub_job(tmp_path)
    cfg.qc_transcript = False
    engine = _QcStageASR("tiny")
    monkeypatch.setattr(
        transcribe_mod, "build_asr", lambda model, device="auto": engine
    )

    result = transcribe(str(job_path), cfg)

    assert not (artifacts_dir / TRANSCRIPT_QC_NAME).exists()
    assert result.transcript_verdict == ""
    assert "transcript_qc" not in result.paths
    assert engine.window_calls == []
    # the original (uncorrected) text is kept when QC is off
    data = json.loads((artifacts_dir / SEGMENTS_NAME).read_text(encoding="utf-8"))
    assert data[1]["text"] == "bad bad bad bad bad bad bad"


def test_cli_no_qc_transcript_flag_reaches_config() -> None:
    from vidub.cli import _build_cfg, build_parser

    parser = build_parser()
    on = _build_cfg(parser.parse_args(["run", "clip.mp4"]))
    assert on.qc_transcript is True
    off = _build_cfg(parser.parse_args(["run", "clip.mp4", "--no-qc-transcript"]))
    assert off.qc_transcript is False


# ------------------------------------------------------------------- manifests
def test_segment_meta_and_verdict_roundtrip(tmp_path: Path) -> None:
    manifest = JobManifest(
        video_hash="abc",
        cache_dir=str(tmp_path),
        segments=[Segment(0.0, 1.0, "hi", meta={"avg_logprob": -0.5})],
        transcript_verdict="review",
    )
    path = manifest.save(tmp_path / "abc" / "job.json")
    loaded = JobManifest.load(path)
    assert loaded.segments[0].meta == {"avg_logprob": -0.5}
    assert loaded.transcript_verdict == "review"


def test_old_manifest_without_meta_or_verdict_loads(tmp_path: Path) -> None:
    data = {
        "video_hash": "abc",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "old", "speaker": "spk0", "words": None}
        ],
    }
    path = tmp_path / "job.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    loaded = JobManifest.load(path)
    assert loaded.segments[0].meta is None
    assert loaded.transcript_verdict == ""
