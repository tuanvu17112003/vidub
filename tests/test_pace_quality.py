"""Regression checks for natural pacing and safe timing-stage translation."""
from pathlib import Path
import json

import pytest

from vidub.config import JobConfig
from vidub.manifest import JobManifest
from vidub.stages.timing import fit_segment
from vidub.stages.export import _unit_subtitle_rows
from vidub.engines.mt.validators import _numbers_ok, _validate
from test_timing import _write_sine, _make_unit_job, _RatePiper, _timed, fit


@pytest.mark.parametrize("source,target,ok", [
    ("16 GB", "160 GB", False),
    ("1 and 6", "16", False),
    ("16 and 16", "16", False),
    ("1.5 GB", "1,5 GB", True),
    ("1.5 GB", "15 GB", False),
    ("1,000 GB", "1.000 GB", True),
    ("16 and 32", "32 và 16", True),
])
def test_numeric_tokens_keep_boundaries_and_multiplicity(
    source: str, target: str, ok: bool
) -> None:
    assert _numbers_ok(source, target) is ok


def test_wrong_numeric_token_explains_validator_rejection() -> None:
    reason = _validate("Bộ nhớ cần 160 GB", "Memory needs 16 GB", "en", "vi")
    assert reason is not None and "16" in reason


@pytest.mark.parametrize("candidate,duration", [
    ("nội dung ngắn gọn", 0.5),  # valid language, but most content omitted
    ("đây là nội dung đầy đủ cho tất cả mọi người", 4.0),  # audio worse
])
def test_concise_retry_cannot_trade_content_or_duration_for_fit(
    tmp_path: Path, candidate: str, duration: float
) -> None:
    src = tmp_path / "source.wav"
    _write_sine(src, 3.0)
    calls: list[str] = []

    def synth(text: str, path: Path) -> float:
        calls.append(text)
        _write_sine(Path(path), duration)
        return duration

    record, _ = fit_segment(
        tmp_path, {"idx": 0}, src, window=2.0, slot_start=0.0,
        prev_stretch=1.0, cfg=JobConfig(tts_rate_fit=False, recenter_audio=False),
        en_text="This is the full explanation for all of the people.",
        vi_text="đây là nội dung đầy đủ cho tất cả mọi người",
        concise_retranslate_fn=lambda source, budget: candidate, synth_fn=synth,
    )
    assert not record.get("retranslated")
    assert record["file"] == "source_clip.wav"
    if duration == 0.5:
        assert calls == []  # reject omissions before spending time on TTS


def test_multi_member_rate_fit_uses_measured_sentence_audio(tmp_path: Path) -> None:
    rows = [
        {"start": 0.0, "end": 1.0, "en": "This sentence", "vi": "câu này", "speaker": "spk0"},
        {"start": 1.0, "end": 2.0, "en": "ends here.", "vi": "kết thúc.", "speaker": "spk0"},
    ]
    job, artifacts, cfg = _make_unit_job(
        tmp_path, rows, video_end=2.0, tail_pad=0.0,
        tts_rate_fit=True, recenter=False,
    )
    tts = _RatePiper(default=2.6)
    cfg.retranslate = False
    fit(str(job), cfg, tts=tts)
    record = _timed(artifacts)[0]
    assert record["members"] == [0, 1]
    assert record["length_scale"] == pytest.approx(1 / 1.3)
    assert not record.get("clipped")
    assert tts.rate_calls[0][1] == 1.0  # natural sentence measured first
    assert tts.rate_calls[1][1] == pytest.approx(1 / 1.3)


def test_subtitles_follow_accepted_timing_translation(tmp_path: Path) -> None:
    manifest = JobManifest(cache_dir=str(tmp_path), video_hash="job")
    manifest.artifacts_dir.mkdir()
    (manifest.artifacts_dir / "units.json").write_text(json.dumps({"units": [
        {"index": 0, "start": 0, "end": 2, "vi": "Bản cũ."}
    ]}), encoding="utf-8")
    (manifest.artifacts_dir / "timed.json").write_text(json.dumps([
        {"idx": 0, "start": 0, "end": 1, "retranslated": True, "vi": "Bản mới."}
    ]), encoding="utf-8")
    assert _unit_subtitle_rows(manifest)[0]["vi"] == "Bản mới."
