"""Tests for the S4 timing stage v2 (M1-T5 / M3-C1)."""

from __future__ import annotations

import json
import logging
import math
import struct
import wave
from pathlib import Path

import pytest

from vidub.config import JobConfig
from vidub.engines.mt.base import BaseTranslator
from vidub.engines.tts.base import BaseTTS
from vidub.engines.tts.piper_tts import PiperTTS
from vidub.manifest import JobManifest, Segment
from vidub.stages import timing as timing_mod
from vidub.stages.speak import PIPER_VOICES
from vidub.stages.timing import (
    _assert_end_boundary,
    choose_smart_cut,
    compute_coverage,
    fit,
    fit_segment,
    main,
)
from vidub.utils.audio import wav_duration
from vidub.utils.fit import (
    RATE_SPEED_MAX,
    RECENTER_MAX_SHIFT,
    SPEED_CAP,
    SPEED_CAP_SHORT,
)
from vidub.utils.sentences import split_translation

RATE = 22050
BLEED_TOLERANCE = 0.001


def _write_sine(path: Path, duration: float, freq: float = 440.0) -> None:
    """Write a mono 16-bit sine WAV of exactly ``duration`` seconds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = bytearray()
    for i in range(round(duration * RATE)):
        value = int(0.4 * 32767 * math.sin(2.0 * math.pi * freq * i / RATE))
        frames += struct.pack("<h", value)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(RATE)
        wav_file.writeframes(bytes(frames))


def _write_sine_with_silence(
    path: Path, tone: float, silence: float, freq: float = 440.0
) -> None:
    """Write ``tone`` seconds of sine followed by ``silence`` seconds of silence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = bytearray()
    for i in range(round(tone * RATE)):
        value = int(0.4 * 32767 * math.sin(2.0 * math.pi * freq * i / RATE))
        frames += struct.pack("<h", value)
    frames += struct.pack("<h", 0) * round(silence * RATE)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(RATE)
        wav_file.writeframes(bytes(frames))


class FakeTTS(BaseTTS):
    """Writes a WAV of a fixed / text-mapped duration; records synth calls."""

    def __init__(self, durations: dict[str, float] | None = None, default: float = 1.0) -> None:
        self.durations = dict(durations or {})
        self.default = default
        self.calls: list[tuple[str, str]] = []

    def synth(self, text: str, out_wav: str) -> float:
        self.calls.append((text, str(out_wav)))
        duration = float(self.durations.get(text, self.default))
        _write_sine(Path(out_wav), duration)
        return duration


class FakeTranslator(BaseTranslator):
    """Returns a fixed (short) Vietnamese string for every input."""

    def __init__(self, value: str = "ngắn") -> None:
        self.value = value
        self.calls: list[list[str]] = []

    def translate(self, texts, src_lang, tgt_lang):
        self.calls.append(list(texts))
        return [self.value for _ in texts]


# ------------------------------------------------------------- v2 job builder
STARTS = [0.0, 2.0, 3.0]
ENDS = [2.0, 3.0, 4.0]
EN = ["hello there friend", "and then some words here", "bye now"]
VI = ["a", "b", "c"]


def _make_v2_job(
    tmp_path: Path,
    durations: list[float],
    *,
    video_end: float = 5.0,
    mode: str = "dub",
    retranslate: bool = True,
    tail_pad: float = 2.0,
    tts_rate_fit: bool = False,
    recenter: bool = False,
) -> tuple[Path, Path, JobConfig]:
    cache_dir = tmp_path / "cache"
    artifacts_dir = cache_dir / "v2"
    speech_dir = artifacts_dir / "speech"
    speech_dir.mkdir(parents=True)

    speech = []
    for i, duration in enumerate(durations):
        _write_sine(speech_dir / f"seg_{i:04d}.wav", duration)
        speech.append(
            {
                "idx": i,
                "start": STARTS[i],
                "end": ENDS[i],
                "speaker": "spk0",
                "vi": VI[i],
                "file": f"speech/seg_{i:04d}.wav",
                "duration": duration,
            }
        )
    (artifacts_dir / "speech.json").write_text(
        json.dumps(speech, ensure_ascii=False), encoding="utf-8"
    )

    script = [
        {
            "start": STARTS[i],
            "end": ENDS[i],
            "speaker": "spk0",
            "en": EN[i],
            "vi": VI[i],
            "truncated": False,
        }
        for i in range(len(durations))
    ]
    (artifacts_dir / "script.json").write_text(
        json.dumps(script, ensure_ascii=False), encoding="utf-8"
    )

    job_path = artifacts_dir / "job.json"
    JobManifest(
        input_path="demo.mp4",
        video_hash="v2",
        duration=video_end,
        cache_dir=str(cache_dir),
        mode=mode,
        paths={"script": "script.json", "speech": "speech.json", "speech_dir": "speech"},
        stages={"speak": "done"},
    ).save(job_path)
    return job_path, artifacts_dir, JobConfig(
        cache_dir=str(cache_dir), retranslate=retranslate, tail_pad=tail_pad,
        qc_tts=False, tts_rate_fit=tts_rate_fit, recenter_audio=recenter,
        sentence_units=False,
    )


def _timed(artifacts_dir: Path) -> dict[int, dict]:
    rows = json.loads((artifacts_dir / "timed.json").read_text(encoding="utf-8"))
    return {int(row["idx"]): row for row in rows}


def _assert_no_bleed(rows: dict[int, dict], video_end: float) -> None:
    for idx, row in rows.items():
        boundary = STARTS[idx + 1] if idx + 1 < len(STARTS) else video_end
        assert row["end"] <= boundary + BLEED_TOLERANCE, (idx, row)


# --------------------------------------------------------------- v2 behaviour
def test_asis_segment_is_byte_identical(tmp_path: Path) -> None:
    job_path, artifacts_dir, cfg = _make_v2_job(tmp_path, [2.0, 1.0, 1.0])
    tts = FakeTTS()

    fit(str(job_path), cfg, tts=tts, translator=FakeTranslator())

    rows = _timed(artifacts_dir)
    # seg 0 fits exactly -> untouched file, no stretch, same placement
    assert rows[0]["action"] == "asis"
    assert rows[0]["stretch"] == 1.0
    assert rows[0]["file"] == "speech/seg_0000.wav"
    assert rows[0]["start"] == 0.0
    assert rows[0]["end"] == pytest.approx(2.0, abs=0.02)
    # and the TTS was never asked to re-synthesise anything
    assert tts.calls == []
    _assert_no_bleed(rows, 5.0)


def test_shorten_retranslates_instead_of_overstretching(tmp_path: Path) -> None:
    # seg 1 speech 1.4 s in a 1.0 s window -> ratio 1.4 (would be a 1.35 rush)
    job_path, artifacts_dir, cfg = _make_v2_job(tmp_path, [2.0, 1.4, 1.0])
    tts = FakeTTS(durations={"ngắn": 1.0})

    fit(str(job_path), cfg, tts=tts, translator=FakeTranslator("ngắn"))

    rows = _timed(artifacts_dir)
    row = rows[1]
    assert row["retranslated"] is True
    # the shorter re-translation now fits comfortably -> as-is, not 1.35
    assert row["action"] == "asis"
    assert row["stretch"] == 1.0
    assert row["file"] == "speech/seg_0001_re.wav"
    assert row["end"] <= 3.0 + BLEED_TOLERANCE
    assert row["stretch"] <= SPEED_CAP
    assert tts.calls and tts.calls[0][0] == "ngắn"
    _assert_no_bleed(rows, 5.0)


def test_shorten_retry_applies_pronunciation_to_resynth(tmp_path: Path) -> None:
    """The re-synthesised retry text gets the same lexicon as the speak stage."""
    job_path, artifacts_dir, cfg = _make_v2_job(tmp_path, [2.0, 1.4, 1.0])
    lexicon = tmp_path / "p.tsv"
    lexicon.write_text("ngắn\tngan\n", encoding="utf-8")
    cfg.pronounce_path = str(lexicon)
    tts = FakeTTS(durations={"ngan": 1.0})

    fit(str(job_path), cfg, tts=tts, translator=FakeTranslator("ngắn"))

    # the retry synthesised the phonetic form, not the written "ngắn"
    assert tts.calls and tts.calls[0][0] == "ngan"
    rows = _timed(artifacts_dir)
    assert rows[1]["retranslated"] is True
    assert rows[1]["action"] == "asis"


def test_clip_when_retranslation_still_too_long(tmp_path: Path) -> None:
    job_path, artifacts_dir, cfg = _make_v2_job(tmp_path, [2.0, 1.4, 1.0])
    # the re-translation is not shorter -> the segment still cannot fit
    tts = FakeTTS(durations={"ngắn": 2.0})

    fit(str(job_path), cfg, tts=tts, translator=FakeTranslator("ngắn"))

    rows = _timed(artifacts_dir)
    row = rows[1]
    assert not row.get("retranslated", False)
    assert row["clipped"] is True
    assert row["action"] == "clip"
    assert row["stretch"] == pytest.approx(SPEED_CAP)
    assert row["file"].endswith("seg_0001_clip.wav")
    assert row["end"] <= 3.0 + BLEED_TOLERANCE

    pacing = json.loads((artifacts_dir / "timing.json").read_text(encoding="utf-8"))
    assert pacing["clipped"] == 1
    assert pacing["clips"][0]["idx"] == 1
    assert pacing["clips"][0]["seconds_trimmed"] > 0
    assert "re-translation" in pacing["clips"][0]["reason"]
    assert pacing["max_stretch"] <= SPEED_CAP
    _assert_no_bleed(rows, 5.0)


def test_no_retranslate_clips_with_reason(tmp_path: Path) -> None:
    job_path, artifacts_dir, cfg = _make_v2_job(
        tmp_path, [2.0, 1.4, 1.0], retranslate=False
    )
    tts = FakeTTS()

    fit(str(job_path), cfg, tts=tts, translator=FakeTranslator())

    rows = _timed(artifacts_dir)
    row = rows[1]
    assert row["clipped"] is True
    assert "retranslated" not in row
    assert tts.calls == []
    pacing = json.loads((artifacts_dir / "timing.json").read_text(encoding="utf-8"))
    assert "disabled" in pacing["clips"][0]["reason"]
    _assert_no_bleed(rows, 5.0)


class FakeChatTranslator(FakeTranslator):
    """A translator that also exposes the chat interface (concise prompt path)."""

    def __init__(self, value: str = "nội dung ngắn gọn") -> None:
        super().__init__(value)
        self.prompts: list[str] = []

    def _chat(self, prompt: str) -> str:  # hunyuan-style single-argument chat
        self.prompts.append(prompt)
        return self.value


def test_short_window_cap_stretches_instead_of_clipping(tmp_path: Path) -> None:
    """Q-1-T3 lever 2: a 1.25x need in a short window stretches, never clips."""
    artifacts_dir = tmp_path / "art"
    artifacts_dir.mkdir(parents=True)
    src = artifacts_dir / "seg.wav"
    _write_sine(src, 1.25)
    measured = wav_duration(src)

    record, info = fit_segment(
        artifacts_dir,
        {"idx": 0, "file": "seg.wav", "duration": measured},
        src,
        window=1.0,  # below SHORT_WINDOW_MAX -> the short cap applies
        slot_start=0.0,
        prev_stretch=1.0,
        cfg=JobConfig(retranslate=False),
        en_text="hello",
    )

    assert info["action"] == "stretch"
    assert info["clipped"] is False
    assert record["stretch"] == pytest.approx(measured, rel=0.02)
    assert SPEED_CAP < record["stretch"] <= SPEED_CAP_SHORT
    assert record["end"] <= 1.0 + BLEED_TOLERANCE


def test_concise_retranslation_rescues_a_clip_bound_unit(tmp_path: Path) -> None:
    """Q-1-T3 lever 1: one concise re-translation rescues a would-be clip."""
    job_path, artifacts_dir, cfg = _make_v2_job(tmp_path, [2.0, 1.4, 1.0])
    concise = "nội dung ngắn gọn"
    tts = FakeTTS(durations={concise: 0.8})
    translator = FakeChatTranslator(concise)

    fit(str(job_path), cfg, tts=tts, translator=translator)

    rows = _timed(artifacts_dir)
    row = rows[1]
    assert row["retranslated"] is True
    assert row.get("clipped") is not True
    assert row["action"] == "asis"
    assert row["file"].endswith("_re.wav")
    assert "concise" in row["retranslate_reason"]
    assert row["retranslate_target"] > 0
    # the explicit character target reached the engine prompt
    assert translator.prompts
    assert "at most" in translator.prompts[0]
    assert str(row["retranslate_target"]) in translator.prompts[0]
    assert row["end"] <= 3.0 + BLEED_TOLERANCE
    _assert_no_bleed(rows, 5.0)


def test_clip_bound_unit_with_no_lever_is_one_minimal_clip(tmp_path: Path) -> None:
    """Q-1-T3 lever 3: when nothing fits, exactly one clip at the window end."""
    job_path, artifacts_dir, cfg = _make_v2_job(tmp_path, [2.0, 1.4, 1.0])
    long_text = "một hai ba bốn năm sáu bảy tám"
    tts = FakeTTS(durations={long_text: 2.0})
    translator = FakeChatTranslator(long_text)

    fit(str(job_path), cfg, tts=tts, translator=translator)

    rows = _timed(artifacts_dir)
    row = rows[1]
    assert row["clipped"] is True
    assert row["action"] == "clip"
    assert row["clipped"] is True
    # trimmed at the window end -> never a bleed, only one clip
    assert row["end"] <= 3.0 + BLEED_TOLERANCE
    pacing = json.loads((artifacts_dir / "timing.json").read_text(encoding="utf-8"))
    assert pacing["clipped"] == 1
    clip = pacing["clips"][0]
    assert clip["idx"] == 1
    assert clip["seconds"] > 0
    assert clip["seconds"] == clip["seconds_trimmed"]
    assert clip["reason"].strip()
    _assert_no_bleed(rows, 5.0)


def test_concise_lever_fires_before_the_short_window_stretch(tmp_path: Path) -> None:
    """Q-1-T3-C1: a would-be clip whose concise text fits uses the concise lever.

    With the old order (short-window stretch first) this unit would have been
    recorded ``action="stretch"`` with no re-translation; the concise-first order
    must fire the concise lever and keep the unit out of the ``stretched`` count.
    """
    artifacts_dir = tmp_path / "art"
    artifacts_dir.mkdir(parents=True)
    src = artifacts_dir / "seg.wav"
    _write_sine(src, 1.25)  # 1.25x in a short 1.0 s window
    concise = "nội dung ngắn gọn"
    synth_calls: list[str] = []

    def synth_fn(text: str, out_wav: str) -> float:
        synth_calls.append(text)
        duration = 0.8 if text == concise else 1.25
        _write_sine(Path(out_wav), duration)
        return duration

    record, info = fit_segment(
        artifacts_dir,
        {"idx": 0, "file": "seg.wav", "duration": 1.25},
        src,
        window=1.0,
        slot_start=0.0,
        prev_stretch=1.0,
        cfg=JobConfig(),
        en_text="hello there friend",
        vi_text="xin chào bạn",
        concise_retranslate_fn=lambda source, target: concise,
        synth_fn=synth_fn,
        src_lang="en",
        tgt_lang="vi",
    )

    assert info["retranslated"] is True
    assert record["retranslate_reason"] == "concise re-translation"
    assert record["retranslate_target"] > 0
    # the concise take fits as-is; the extra stretch must NOT have fired
    assert record["action"] == "asis"
    assert info["clipped"] is False
    assert record["end"] <= 1.0 + BLEED_TOLERANCE
    assert synth_calls and synth_calls[0] == concise


def test_short_window_stretch_fires_when_concise_overflows(tmp_path: Path) -> None:
    """Concise first, but when it still overflows the bounded stretch follows."""
    artifacts_dir = tmp_path / "art"
    artifacts_dir.mkdir(parents=True)
    src = artifacts_dir / "seg.wav"
    _write_sine(src, 1.25)
    concise = "nội dung ngắn gọn"
    synth_calls: list[str] = []

    def synth_fn(text: str, out_wav: str) -> float:
        synth_calls.append(text)
        duration = 1.4 if text == concise else 1.25  # concise does NOT fit
        _write_sine(Path(out_wav), duration)
        return duration

    record, info = fit_segment(
        artifacts_dir,
        {"idx": 0, "file": "seg.wav", "duration": 1.25},
        src,
        window=1.0,
        slot_start=0.0,
        prev_stretch=1.0,
        cfg=JobConfig(),
        en_text="hello there friend",
        vi_text="xin chào bạn",
        concise_retranslate_fn=lambda source, target: concise,
        synth_fn=synth_fn,
        src_lang="en",
        tgt_lang="vi",
    )

    # the concise text was tried first (order), overflowed, and the short-window
    # extra stretch then saved the pre-concise take from a clip
    assert concise in synth_calls
    assert info["action"] == "stretch"
    assert info["clipped"] is False
    assert record["stretch"] == pytest.approx(1.25, rel=0.03)
    assert record["end"] <= 1.0 + BLEED_TOLERANCE


def test_retranslate_off_keeps_the_stretch_only_path(tmp_path: Path) -> None:
    """``retranslate=False`` never calls the engine; the stretch lever still runs."""
    artifacts_dir = tmp_path / "art"
    artifacts_dir.mkdir(parents=True)
    src = artifacts_dir / "seg.wav"
    _write_sine(src, 1.25)
    synth_calls: list[str] = []

    def synth_fn(text: str, out_wav: str) -> float:
        synth_calls.append(text)
        _write_sine(Path(out_wav), 1.25)
        return 1.25

    record, info = fit_segment(
        artifacts_dir,
        {"idx": 0, "file": "seg.wav", "duration": 1.25},
        src,
        window=1.0,
        slot_start=0.0,
        prev_stretch=1.0,
        cfg=JobConfig(retranslate=False),
        en_text="hello there friend",
        vi_text="xin chào bạn",
        concise_retranslate_fn=lambda source, target: "nội dung ngắn gọn",
        synth_fn=synth_fn,
        src_lang="en",
        tgt_lang="vi",
    )

    assert synth_calls == []  # no engine re-translation call
    assert info["action"] == "stretch"
    assert info["retranslated"] is False
    assert info["clipped"] is False
    assert record["end"] <= 1.0 + BLEED_TOLERANCE


def test_adequate_window_run_produces_no_clips(tmp_path: Path) -> None:
    """Regression: adequate windows never clip under the new policy."""
    job_path, artifacts_dir, cfg = _make_v2_job(tmp_path, [2.0, 1.1, 1.0])

    fit(str(job_path), cfg, tts=FakeTTS(), translator=FakeTranslator())

    pacing = json.loads((artifacts_dir / "timing.json").read_text(encoding="utf-8"))
    assert pacing["clipped"] == 0
    assert pacing["stretched"] >= 1
    _assert_no_bleed(_timed(artifacts_dir), 5.0)


def test_asis_overrun_upgraded_to_minimal_stretch(tmp_path: Path) -> None:
    # ratio 1.025 is inside the as-is band but the audio would still bleed
    job_path, artifacts_dir, cfg = _make_v2_job(tmp_path, [2.05, 1.0, 1.0])

    fit(str(job_path), cfg, tts=FakeTTS(), translator=FakeTranslator())

    rows = _timed(artifacts_dir)
    row = rows[0]
    assert row["action"] == "stretch"
    assert 1.0 < row["stretch"] <= SPEED_CAP
    assert row["end"] <= 2.0 + BLEED_TOLERANCE
    _assert_no_bleed(rows, 5.0)


def test_timing_units_use_explicit_spoken_text(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "job1"
    artifacts_dir.mkdir()
    script = [
        {
            "start": 0.0,
            "end": 1.0,
            "en": "hello",
            "vi": "legacy text",
            "spoken_vi": "spoken selection",
        }
    ]
    (artifacts_dir / "units.json").write_text(
        json.dumps(
            {
                "units": [
                    {
                        "index": 0,
                        "indices": [0],
                        "start": 0.0,
                        "end": 1.0,
                        "vi": "legacy unit text",
                        "spoken_vi": "unit spoken selection",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest = JobManifest(
        video_hash="job1", cache_dir=str(tmp_path), paths={"units": "units.json"}
    )

    units, authoritative = timing_mod._timing_units(script, manifest, artifacts_dir)

    assert authoritative is True
    assert units[0].text == "unit spoken selection"
    assert timing_mod._unit_text(script, [0]) == "spoken selection"


def test_zero_bleed_and_neighbor_cap(tmp_path: Path) -> None:
    job_path, artifacts_dir, cfg = _make_v2_job(tmp_path, [2.0, 1.4, 1.0])
    tts = FakeTTS(durations={"ngắn": 2.0})

    fit(str(job_path), cfg, tts=tts, translator=FakeTranslator("ngắn"))

    rows = _timed(artifacts_dir)
    _assert_no_bleed(rows, 5.0)
    pacing = json.loads((artifacts_dir / "timing.json").read_text(encoding="utf-8"))
    assert pacing["max_neighbor_delta"] <= 0.20 + 1e-9
    assert pacing["max_stretch"] <= SPEED_CAP


def test_timed_rows_gain_v2_fields(tmp_path: Path) -> None:
    job_path, artifacts_dir, cfg = _make_v2_job(tmp_path, [2.0, 1.0, 1.0])

    fit(str(job_path), cfg, tts=FakeTTS(), translator=FakeTranslator())

    row = _timed(artifacts_dir)[0]
    assert {"idx", "file", "start", "end", "slot_start", "stretch", "window", "action"} <= set(row)


def test_manifest_records_stretched_segments(tmp_path: Path) -> None:
    job_path, artifacts_dir, cfg = _make_v2_job(tmp_path, [2.05, 1.0, 1.0])

    manifest = fit(str(job_path), cfg, tts=FakeTTS(), translator=FakeTranslator())

    assert manifest.stretched_segments >= 1
    assert JobManifest.load(job_path).stretched_segments == manifest.stretched_segments


def test_fit_skipped_for_subtitles_only(tmp_path: Path) -> None:
    job_path, _artifacts_dir, cfg = _make_v2_job(
        tmp_path, [2.0, 1.0, 1.0], mode="subtitles-only"
    )

    with pytest.raises(RuntimeError) as excinfo:
        fit(str(job_path), cfg)
    assert "skipped" in str(excinfo.value)
    assert "subtitles-only" in str(excinfo.value)

    assert main([str(job_path)]) == 1


# ------------------------------------------------- coverage / invariant (M2-C1)
def _make_short_speech_job(tmp_path: Path) -> tuple[Path, Path, JobConfig]:
    """Short speech inside long slots -> coverage well below 0.6."""
    cache_dir = tmp_path / "cache"
    artifacts_dir = cache_dir / "cov1"
    speech_dir = artifacts_dir / "speech"
    speech_dir.mkdir(parents=True)

    # 2.0 s of speech in 10.0 s slots, twice -> 4.0 / 20.0 = 0.2 coverage.
    _write_sine(speech_dir / "seg_0000.wav", 2.0)
    _write_sine(speech_dir / "seg_0001.wav", 2.0)

    speech = [
        {"idx": 0, "start": 0.0, "end": 10.0, "speaker": "spk0", "vi": "a", "file": "speech/seg_0000.wav", "duration": 2.0},
        {"idx": 1, "start": 10.0, "end": 20.0, "speaker": "spk0", "vi": "b c d", "file": "speech/seg_0001.wav", "duration": 2.0},
    ]
    (artifacts_dir / "speech.json").write_text(
        json.dumps(speech, ensure_ascii=False), encoding="utf-8"
    )
    script = [
        {"start": 0.0, "end": 10.0, "speaker": "spk0", "en": "a", "vi": "a", "truncated": False},
        {"start": 10.0, "end": 20.0, "speaker": "spk0", "en": "b c d", "vi": "b c d", "truncated": False},
    ]
    (artifacts_dir / "script.json").write_text(
        json.dumps(script, ensure_ascii=False), encoding="utf-8"
    )

    job_path = artifacts_dir / "job.json"
    JobManifest(
        input_path="demo.mp4",
        video_hash="cov1",
        duration=20.0,
        cache_dir=str(cache_dir),
        mode="dub",
        paths={"script": "script.json", "speech": "speech.json", "speech_dir": "speech"},
        stages={"speak": "done"},
    ).save(job_path)
    return job_path, artifacts_dir, JobConfig(cache_dir=str(cache_dir), qc_tts=False, tts_rate_fit=False, recenter_audio=False, sentence_units=False)


def test_low_coverage_warns_and_is_recorded(tmp_path: Path, caplog) -> None:
    job_path, artifacts_dir, cfg = _make_short_speech_job(tmp_path)

    with caplog.at_level(logging.WARNING):
        manifest = fit(str(job_path), cfg)

    assert manifest.speech_coverage == pytest.approx(0.2)
    assert JobManifest.load(job_path).speech_coverage == pytest.approx(0.2)
    messages = [record.getMessage() for record in caplog.records]
    assert any("speech coverage" in message for message in messages), messages
    assert any("0.20" in message for message in messages), messages
    assert (artifacts_dir / "timed.json").exists()


def test_compute_coverage_units() -> None:
    script = [{"start": 0.0, "end": 10.0}, {"start": 10.0, "end": 20.0}]
    # 2 + 10(capped at the 10 s slot) = 12 / 20
    speech = {0: {"duration": 2.0}, 1: {"duration": 30.0}}
    assert compute_coverage(script, speech) == pytest.approx(12 / 20)
    assert compute_coverage([], {}) == 1.0
    assert compute_coverage([{"start": 0.0, "end": 0.0}], {0: {"duration": 5.0}}) == 1.0


def test_coverage_is_one_when_no_segments(tmp_path: Path) -> None:
    job_path, artifacts_dir, cfg = _make_v2_job(tmp_path, [2.0, 1.0, 1.0])
    (artifacts_dir / "script.json").write_text("[]", encoding="utf-8")
    (artifacts_dir / "speech.json").write_text("[]", encoding="utf-8")

    manifest = fit(str(job_path), cfg)

    assert manifest.speech_coverage == 1.0
    assert json.loads((artifacts_dir / "timed.json").read_text(encoding="utf-8")) == []


def test_missing_speech_entry_raises(tmp_path: Path) -> None:
    job_path, artifacts_dir, cfg = _make_v2_job(tmp_path, [2.0, 1.0, 1.0])
    # Add a fourth script segment with no matching speech.json entry.
    script = json.loads((artifacts_dir / "script.json").read_text(encoding="utf-8"))
    script.append(
        {"start": 4.0, "end": 5.0, "speaker": "spk0", "en": "c", "vi": "c", "truncated": False}
    )
    (artifacts_dir / "script.json").write_text(
        json.dumps(script, ensure_ascii=False), encoding="utf-8"
    )

    with pytest.raises(RuntimeError) as excinfo:
        fit(str(job_path), cfg)

    message = str(excinfo.value)
    assert "3" in message
    assert "speech.json" in message
    assert not (artifacts_dir / "timed.json").exists()


# ---------------------------------------------------- M3-C1 loop-1 regression
def test_smoothing_never_underfits_after_tall_neighbor(tmp_path: Path) -> None:
    """A high predecessor must not smooth the stretch below the fit requirement.

    seg speech 1.15 s in a 1.00 s window needs stretch 1.15; with prev 1.20 a
    faulty "pull toward prev" could pick 1.05 and bleed 0.1 s. The fitted audio
    must cover the window (stretch >= needed) and never extend past it.
    """
    artifacts_dir = tmp_path / "art"
    artifacts_dir.mkdir(parents=True)
    src = artifacts_dir / "seg.wav"
    _write_sine(src, 1.15)
    measured = wav_duration(src)

    record, info = fit_segment(
        artifacts_dir,
        {"idx": 0, "file": "seg.wav", "duration": measured},
        src,
        window=1.0,
        slot_start=2.0,
        prev_stretch=1.2,
        cfg=JobConfig(),
        en_text="hello",
    )

    assert record["action"] == "stretch"
    assert record["stretch"] >= measured - 1e-9
    assert record["stretch"] <= SPEED_CAP
    # trimmed at the window end -> end <= start + window (no bleed)
    assert record["end"] <= 3.0 + BLEED_TOLERANCE
    assert info["clipped"] is False


def test_stretched_audio_is_trimmed_to_window(tmp_path: Path) -> None:
    """atempo frame rounding must not push the fitted file past the window."""
    artifacts_dir = tmp_path / "art"
    artifacts_dir.mkdir(parents=True)
    src = artifacts_dir / "seg.wav"
    _write_sine(src, 1.42)  # would be a 1.35x rush; retry disabled -> clip path

    record, info = fit_segment(
        artifacts_dir,
        {"idx": 0, "file": "seg.wav", "duration": 1.42},
        src,
        window=1.0,
        slot_start=0.0,
        prev_stretch=1.0,
        cfg=JobConfig(retranslate=False),
        en_text="hello",
    )

    assert info["clipped"] is True
    assert record["end"] <= 1.0 + BLEED_TOLERANCE
    assert wav_duration(artifacts_dir / str(record["file"])) <= 1.0 + BLEED_TOLERANCE


# ------------------------------------------------------ M3-C2 tail safety
def test_choose_smart_cut_picks_latest_pause_near_window() -> None:
    cut, smart = choose_smart_cut([(0.1, 0.2), (0.7, 0.9)], window=1.0)

    assert smart is True
    assert cut == 0.7


def test_choose_smart_cut_falls_back_when_pause_too_early() -> None:
    assert choose_smart_cut([(0.1, 0.3)], window=1.0) == (1.0, False)


def test_choose_smart_cut_accepts_window_inside_a_pause() -> None:
    # the pause starts before the lookback but still covers the window
    assert choose_smart_cut([(0.3, 1.4)], window=1.0) == (1.0, True)


def test_choose_smart_cut_without_silences_is_blind() -> None:
    assert choose_smart_cut([], window=1.0) == (1.0, False)


def test_smart_trim_cuts_at_silence(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "art"
    artifacts_dir.mkdir(parents=True)
    src = artifacts_dir / "seg.wav"
    # 1.0 s of tone then 0.5 s of silence; stretched 1.2x -> pause starts at 0.833
    _write_sine_with_silence(src, tone=1.0, silence=0.5)

    record, info = fit_segment(
        artifacts_dir,
        {"idx": 0, "file": "seg.wav", "duration": 1.5},
        src,
        window=1.0,
        slot_start=0.0,
        prev_stretch=1.0,
        cfg=JobConfig(retranslate=False),
        en_text="hello",
    )

    assert info["smart_trim"] is True
    assert info["clipped"] is False
    assert record["smart_trim"] is True
    assert "clipped" not in record
    assert record["end"] <= 1.0 + BLEED_TOLERANCE
    # the cut landed at the pause (~1.0 / 1.2 s), not at the window end
    assert wav_duration(artifacts_dir / str(record["file"])) == pytest.approx(
        1.0 / 1.2, abs=0.08
    )


def test_smart_trim_applies_fade(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []
    real = timing_mod.run_ffmpeg

    def spy(args: list[str]):
        calls.append(list(args))
        return real(args)

    monkeypatch.setattr(timing_mod, "run_ffmpeg", spy)

    artifacts_dir = tmp_path / "art"
    artifacts_dir.mkdir(parents=True)
    src = artifacts_dir / "seg.wav"
    _write_sine(src, 1.42)  # no pause -> blind cut, but still faded

    fit_segment(
        artifacts_dir,
        {"idx": 0, "file": "seg.wav", "duration": 1.42},
        src,
        window=1.0,
        slot_start=0.0,
        prev_stretch=1.0,
        cfg=JobConfig(retranslate=False),
        en_text="hello",
    )

    joined = [" ".join(call) for call in calls]
    assert any("afade=t=out" in text for text in joined), joined


def test_blind_clip_is_flagged_not_smart(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "art"
    artifacts_dir.mkdir(parents=True)
    src = artifacts_dir / "seg.wav"
    _write_sine(src, 1.42)

    record, info = fit_segment(
        artifacts_dir,
        {"idx": 0, "file": "seg.wav", "duration": 1.42},
        src,
        window=1.0,
        slot_start=0.0,
        prev_stretch=1.0,
        cfg=JobConfig(retranslate=False),
        en_text="hello",
    )

    assert info["clipped"] is True
    assert info["smart_trim"] is False
    assert record["clipped"] is True
    assert "smart_trim" not in record
    assert info["clip_reason"].strip()


def test_last_segment_tail_pad_extends_window(tmp_path: Path) -> None:
    # seg 2 speech 2.6 s would need 1.3x in its 2.0 s base window; the 2.0 s tail
    # pad grows the window to 4.0 s so it fits untouched.
    job_path, artifacts_dir, cfg = _make_v2_job(
        tmp_path, [1.0, 1.0, 2.6], tail_pad=2.0
    )

    fit(str(job_path), cfg, tts=FakeTTS(), translator=FakeTranslator())

    rows = _timed(artifacts_dir)
    row = rows[2]
    assert row["action"] == "asis"
    assert row["window"] == pytest.approx(4.0, abs=0.01)
    assert row["end"] <= 3.0 + 4.0 + BLEED_TOLERANCE
    _assert_no_bleed(rows, 7.0)


def test_last_segment_without_tail_pad_clips(tmp_path: Path) -> None:
    job_path, artifacts_dir, cfg = _make_v2_job(
        tmp_path, [1.0, 1.0, 2.6], tail_pad=0.0, retranslate=False
    )

    fit(str(job_path), cfg, tts=FakeTTS(), translator=FakeTranslator())

    rows = _timed(artifacts_dir)
    row = rows[2]
    assert row["action"] == "clip"
    assert row["clipped"] is True
    assert "smart_trim" not in row
    pacing = json.loads((artifacts_dir / "timing.json").read_text(encoding="utf-8"))
    assert pacing["smart_trims"] == 0
    assert pacing["clips"][-1]["idx"] == 2
    assert pacing["clips"][-1]["smart_trim"] is False
    _assert_no_bleed(rows, 5.0)


def _make_smart_trim_job(tmp_path: Path) -> tuple[Path, Path, JobConfig]:
    """One 1.5 s segment (1.0 s tone + 0.5 s silence) in a 1.0 s window."""
    cache_dir = tmp_path / "cache"
    artifacts_dir = cache_dir / "smart"
    speech_dir = artifacts_dir / "speech"
    speech_dir.mkdir(parents=True)
    _write_sine_with_silence(speech_dir / "seg_0000.wav", tone=1.0, silence=0.5)
    (artifacts_dir / "speech.json").write_text(
        json.dumps(
            [
                {
                    "idx": 0,
                    "start": 0.0,
                    "end": 1.0,
                    "speaker": "spk0",
                    "vi": "a",
                    "file": "speech/seg_0000.wav",
                    "duration": 1.5,
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (artifacts_dir / "script.json").write_text(
        json.dumps(
            [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "speaker": "spk0",
                    "en": "hello there friend",
                    "vi": "a",
                    "truncated": False,
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    job_path = artifacts_dir / "job.json"
    JobManifest(
        input_path="demo.mp4",
        video_hash="smart",
        duration=1.0,
        cache_dir=str(cache_dir),
        mode="dub",
        paths={"script": "script.json", "speech": "speech.json", "speech_dir": "speech"},
        stages={"speak": "done"},
    ).save(job_path)
    return job_path, artifacts_dir, JobConfig(
        cache_dir=str(cache_dir), retranslate=False, tail_pad=0.0, qc_tts=False,
        tts_rate_fit=False, recenter_audio=False, sentence_units=False,
    )


def test_fit_counts_smart_trims_in_pacing(tmp_path: Path) -> None:
    job_path, artifacts_dir, cfg = _make_smart_trim_job(tmp_path)

    fit(str(job_path), cfg, tts=FakeTTS(), translator=FakeTranslator())

    pacing = json.loads((artifacts_dir / "timing.json").read_text(encoding="utf-8"))
    assert pacing["smart_trims"] == 1
    assert pacing["clipped"] == 0
    assert len(pacing["clips"]) == 1
    assert pacing["clips"][0]["smart_trim"] is True
    assert pacing["max_stretch"] <= SPEED_CAP


def test_end_boundary_invariant_allows_smart_or_flagged_clip() -> None:
    timed = [{"idx": 0, "start": 0.0, "end": 1.0}]

    # a non-clipped row that lost almost all speech (retained < 0.05 s) and was
    # neither smart-trimmed nor clipped is rejected
    with pytest.raises(RuntimeError):
        _assert_end_boundary(
            [{"idx": 0, "start": 0.0, "end": 0.02}],
            [
                {
                    "seconds_trimmed": 0.3,
                    "clipped": False,
                    "smart_trim": False,
                    "clip_reason": "",
                }
            ],
        )

    # a non-clipped row that retains >= 0.05 s of speech is tolerated
    _assert_end_boundary(
        timed,
        [
            {
                "seconds_trimmed": 0.3,
                "clipped": False,
                "smart_trim": False,
                "clip_reason": "",
            }
        ],
    )

    # smart trim at a silence is allowed
    _assert_end_boundary(
        timed,
        [
            {
                "seconds_trimmed": 0.3,
                "clipped": False,
                "smart_trim": True,
                "clip_reason": "",
            }
        ],
    )

    # a blind clip with a reason is allowed
    _assert_end_boundary(
        timed,
        [
            {
                "seconds_trimmed": 0.3,
                "clipped": True,
                "smart_trim": False,
                "clip_reason": "re-translation disabled (--no-retranslate)",
            }
        ],
    )

    # a blind clip without a reason is rejected
    with pytest.raises(RuntimeError):
        _assert_end_boundary(
            timed,
            [
                {
                    "seconds_trimmed": 0.3,
                    "clipped": True,
                    "smart_trim": False,
                    "clip_reason": "",
                }
            ],
        )

    # an untrimmed row is never inspected
    _assert_end_boundary(
        timed,
        [
            {
                "seconds_trimmed": 0.0,
                "clipped": False,
                "smart_trim": False,
                "clip_reason": "",
            }
        ],
    )


# ------------------------------------------------- per-clip TTS QC (M4-T7)
TTS_QC_NAME = "tts_qc.json"


def _score_by_name(scores: dict[str, float], retry: float | None = None):
    """A fake ``ttsqc.score_wav``: per-file scores, retry files get ``retry``."""

    def _score(path) -> float | None:
        name = Path(path).name
        if "_qc" in name:
            return retry
        return scores.get(name)

    return _score


def test_tts_qc_retries_low_clip_and_keeps_better(tmp_path, monkeypatch) -> None:
    job_path, artifacts_dir, cfg = _make_v2_job(tmp_path, [2.0, 1.0])
    cfg.qc_tts = True
    cfg.tts_qc_mos_min = 3.0
    tts = FakeTTS()
    monkeypatch.setattr(
        timing_mod.ttsqc,
        "score_wav",
        _score_by_name({"seg_0000.wav": 4.5, "seg_0001.wav": 2.0}, retry=3.9),
    )

    fit(str(job_path), cfg, tts=tts, translator=FakeTranslator())

    rows = _timed(artifacts_dir)
    sidecar = json.loads((artifacts_dir / TTS_QC_NAME).read_text(encoding="utf-8"))
    assert sidecar == {
        "enabled": True,
        "mos_min": 3.0,
        "scored": 2,
        "low_mos": 1,
        "retried": 1,
        "resolved": 1,
        "skipped": 0,
        "mos_mean": 4.2,  # (4.5 + 3.9) / 2
    }
    # the better (retry) clip was kept for the low segment ...
    assert rows[1]["file"] == "speech/seg_0001_qc_fit.wav"
    assert rows[1]["tts_qc_retry"] is True
    # ... and the in-range clip is untouched
    assert rows[0]["file"] == "speech/seg_0000.wav"
    assert "tts_qc_retry" not in rows[0]
    # exactly ONE retry synthesis
    assert len(tts.calls) == 1
    assert tts.calls[0][0] == "b"  # script vi text of segment 1
    assert JobManifest.load(job_path).paths["tts_qc"] == TTS_QC_NAME


def test_tts_qc_keeps_original_when_retry_not_better(tmp_path, monkeypatch) -> None:
    job_path, artifacts_dir, cfg = _make_v2_job(tmp_path, [2.0, 1.0])
    cfg.qc_tts = True
    monkeypatch.setattr(
        timing_mod.ttsqc,
        "score_wav",
        _score_by_name({"seg_0000.wav": 4.5, "seg_0001.wav": 2.0}, retry=1.5),
    )

    fit(str(job_path), cfg, tts=FakeTTS(), translator=FakeTranslator())

    rows = _timed(artifacts_dir)
    # the lower retry score is discarded -> original clip kept
    assert rows[1]["file"] == "speech/seg_0001.wav"
    assert "tts_qc_retry" not in rows[1]

    sidecar = json.loads((artifacts_dir / TTS_QC_NAME).read_text(encoding="utf-8"))
    assert sidecar["scored"] == 2
    assert sidecar["low_mos"] == 1
    assert sidecar["retried"] == 1
    assert sidecar["resolved"] == 0
    assert sidecar["skipped"] == 0
    assert sidecar["mos_mean"] == 3.25  # (4.5 + 2.0) / 2

    # still below the threshold -> a review packet with both MOS values
    packet = artifacts_dir / "review" / "tts"
    assert (packet / "seg-1.txt").exists()
    assert (packet / "seg-1.wav").exists()
    text = (packet / "seg-1.txt").read_text(encoding="utf-8")
    assert "mos original: 2.000" in text
    assert "mos retry: 1.500" in text


def test_tts_qc_disabled_writes_no_sidecar(tmp_path, monkeypatch) -> None:
    job_path, artifacts_dir, cfg = _make_v2_job(tmp_path, [2.0, 1.0])
    assert cfg.qc_tts is False  # the shared builders keep the gate off

    def _boom(path):  # noqa: ANN001 - must never be called
        raise AssertionError("the scorer must not run when qc_tts is disabled")

    monkeypatch.setattr(timing_mod.ttsqc, "score_wav", _boom)

    fit(str(job_path), cfg, tts=FakeTTS(), translator=FakeTranslator())

    assert not (artifacts_dir / TTS_QC_NAME).exists()
    assert "tts_qc" not in JobManifest.load(job_path).paths
    assert all("tts_qc_retry" not in row for row in _timed(artifacts_dir).values())


def test_tts_qc_scorer_unavailable_counts_skipped(tmp_path, monkeypatch) -> None:
    job_path, artifacts_dir, cfg = _make_v2_job(tmp_path, [2.0, 1.0])
    cfg.qc_tts = True
    monkeypatch.setattr(timing_mod.ttsqc, "score_wav", lambda path: None)

    # the pipeline must continue (no crash) even with no scorer at all
    result = fit(str(job_path), cfg, tts=FakeTTS(), translator=FakeTranslator())

    sidecar = json.loads((artifacts_dir / TTS_QC_NAME).read_text(encoding="utf-8"))
    assert sidecar == {
        "enabled": True,
        "mos_min": 3.0,
        "scored": 0,
        "low_mos": 0,
        "retried": 0,
        "resolved": 0,
        "skipped": 2,
        "mos_mean": None,
    }
    assert not (artifacts_dir / "review" / "tts").exists()
    assert result.stages["timing"] == "done"


def test_tts_qc_retry_synth_failure_keeps_original(tmp_path, monkeypatch) -> None:
    job_path, artifacts_dir, cfg = _make_v2_job(tmp_path, [2.0, 1.0])
    cfg.qc_tts = True
    monkeypatch.setattr(
        timing_mod.ttsqc,
        "score_wav",
        _score_by_name({"seg_0000.wav": 4.5, "seg_0001.wav": 2.0}),
    )

    class _BoomTTS(FakeTTS):
        def synth(self, text, out_wav):  # noqa: ANN001
            raise RuntimeError("synth failed")

    fit(str(job_path), cfg, tts=_BoomTTS(), translator=FakeTranslator())

    rows = _timed(artifacts_dir)
    assert rows[1]["file"] == "speech/seg_0001.wav"  # original kept
    sidecar = json.loads((artifacts_dir / TTS_QC_NAME).read_text(encoding="utf-8"))
    assert sidecar["retried"] == 1
    assert sidecar["resolved"] == 0
    # still low -> review packet (no retry MOS available)
    assert (artifacts_dir / "review" / "tts" / "seg-1.txt").exists()


def test_tts_qc_piper_retry_uses_alternate_voice(tmp_path, monkeypatch) -> None:
    job_path, artifacts_dir, cfg = _make_v2_job(tmp_path, [2.0, 1.0])
    cfg.qc_tts = True

    class _FakePiper(PiperTTS):
        def __init__(self, voice_path: str = PIPER_VOICES["vais1000"]) -> None:
            super().__init__(voice_path=voice_path)
            self.seen: list[str] = []

        def synth(self, text: str, out_wav: str) -> float:
            self.seen.append(self.voice_path)
            _write_sine(Path(out_wav), 1.0)
            return 1.0

    engine = _FakePiper()
    seen_configs: list[str] = []

    def _fake_files(eng, voice_path):  # noqa: ANN001 - offline test double
        seen_configs.append(voice_path)
        return Path("alt.onnx"), Path("alt.onnx.json")

    monkeypatch.setattr(timing_mod, "_piper_voice_files", _fake_files)
    monkeypatch.setattr(
        timing_mod.ttsqc,
        "score_wav",
        _score_by_name({"seg_0000.wav": 4.5, "seg_0001.wav": 2.0}, retry=4.0),
    )

    fit(str(job_path), cfg, tts=engine, translator=FakeTranslator())

    # the retry used the OTHER shipped voice (with its own config), then restored
    assert engine.seen == [PIPER_VOICES["25hours"]]
    assert seen_configs == [PIPER_VOICES["25hours"]]
    assert engine.voice_path == PIPER_VOICES["vais1000"]


def test_tts_qc_vixtts_retry_uses_variant_settings(tmp_path, monkeypatch) -> None:
    job_path, artifacts_dir, cfg = _make_v2_job(tmp_path, [2.0, 1.0])
    data = json.loads(job_path.read_text(encoding="utf-8"))
    data["tts_engine"] = "vixtts"
    job_path.write_text(json.dumps(data), encoding="utf-8")
    cfg.qc_tts = True

    import vidub.engines.tts.vixtts as vixtts_mod

    class _FakeVixtts(BaseTTS):
        def __init__(self) -> None:
            self.seen: list[tuple] = []

        def synth(self, text: str, out_wav: str) -> float:
            self.seen.append(
                (
                    vixtts_mod._GEN_KWARGS.get("temperature"),
                    vixtts_mod._GEN_KWARGS.get("top_k"),
                )
            )
            _write_sine(Path(out_wav), 1.0)
            return 1.0

    engine = _FakeVixtts()
    before = dict(vixtts_mod._GEN_KWARGS)
    monkeypatch.setattr(
        timing_mod.ttsqc,
        "score_wav",
        _score_by_name({"seg_0000.wav": 4.5, "seg_0001.wav": 2.0}, retry=4.0),
    )

    fit(str(job_path), cfg, tts=engine, translator=FakeTranslator())

    # one retry synth saw the changed sampling settings ...
    assert engine.seen == [(0.65, 80)]
    # ... and the module globals were restored afterwards
    assert vixtts_mod._GEN_KWARGS == before


def test_tts_qc_threshold_flag_controls_retry(tmp_path, monkeypatch) -> None:
    job_path, artifacts_dir, cfg = _make_v2_job(tmp_path, [2.0, 1.0])
    cfg.qc_tts = True
    cfg.tts_qc_mos_min = 4.9  # nothing passes
    tts = FakeTTS()
    monkeypatch.setattr(
        timing_mod.ttsqc,
        "score_wav",
        _score_by_name({"seg_0000.wav": 4.5, "seg_0001.wav": 2.0}, retry=4.8),
    )

    fit(str(job_path), cfg, tts=tts, translator=FakeTranslator())

    sidecar = json.loads((artifacts_dir / TTS_QC_NAME).read_text(encoding="utf-8"))
    assert sidecar == {
        "enabled": True,
        "mos_min": 4.9,
        "scored": 2,
        "low_mos": 2,
        "retried": 2,
        "resolved": 0,
        "skipped": 0,
        "mos_mean": 4.8,  # both retries scored 4.8 (better than 4.5 / 2.0)
    }
    assert (artifacts_dir / "review" / "tts" / "seg-0.txt").exists()
    assert (artifacts_dir / "review" / "tts" / "seg-1.txt").exists()


# --------------------------------------------------- FIT-1 rate fit + recenter
class _RatePiper(PiperTTS):
    """Offline Piper double whose output duration scales with ``length_scale``.

    ``base`` maps vi text to its natural duration; a synthesis at ``length_scale``
    writes ``base * length_scale`` seconds. Records every ``(text, scale)`` call,
    so a job never needs a model download or the network.
    """

    def __init__(
        self, base: dict[str, float] | None = None, default: float = 1.0
    ) -> None:
        super().__init__()
        self.base = dict(base or {})
        self.default = default
        self.rate_calls: list[tuple[str, float]] = []

    def synth(self, text: str, out_wav: str, *, length_scale: float | None = None):
        scale = 1.0 if length_scale is None else float(length_scale)
        self.rate_calls.append((text, scale))
        duration = float(self.base.get(text, self.default)) * scale
        _write_sine(Path(out_wav), duration)
        return duration


def _make_rate_job(
    tmp_path: Path,
    rows: list[tuple[float, float, float, str]],
    *,
    video_end: float,
    tail_pad: float = 0.0,
    tts_rate_fit: bool = True,
    recenter: bool = True,
) -> tuple[Path, Path, JobConfig]:
    """Job builder for rate tests: ``rows`` are (start, end, natural, vi)."""
    cache_dir = tmp_path / "cache"
    artifacts_dir = cache_dir / "rate"
    speech_dir = artifacts_dir / "speech"
    speech_dir.mkdir(parents=True)
    speech = []
    script = []
    for i, (start, end, natural, vi) in enumerate(rows):
        _write_sine(speech_dir / f"seg_{i:04d}.wav", natural)
        speech.append(
            {
                "idx": i,
                "start": start,
                "end": end,
                "speaker": "spk0",
                "vi": vi,
                "file": f"speech/seg_{i:04d}.wav",
                "duration": natural,
            }
        )
        script.append(
            {
                "start": start,
                "end": end,
                "speaker": "spk0",
                "en": vi,
                "vi": vi,
                "truncated": False,
            }
        )
    (artifacts_dir / "speech.json").write_text(
        json.dumps(speech, ensure_ascii=False), encoding="utf-8"
    )
    (artifacts_dir / "script.json").write_text(
        json.dumps(script, ensure_ascii=False), encoding="utf-8"
    )
    job_path = artifacts_dir / "job.json"
    JobManifest(
        input_path="demo.mp4",
        video_hash="rate",
        duration=video_end,
        cache_dir=str(cache_dir),
        mode="dub",
        paths={"script": "script.json", "speech": "speech.json", "speech_dir": "speech"},
        stages={"speak": "done"},
    ).save(job_path)
    return job_path, artifacts_dir, JobConfig(
        cache_dir=str(cache_dir), retranslate=True, tail_pad=tail_pad, qc_tts=False,
        tts_rate_fit=tts_rate_fit, recenter_audio=recenter, sentence_units=False,
    )


def test_rate_fit_preserves_natural_pace_and_recenter(tmp_path: Path) -> None:
    # seg 2 is 1.0 s of speech in a 4.0 s window: preserve pace + recenter.
    job, artifacts_dir, cfg = _make_v2_job(
        tmp_path, [2.0, 1.0, 1.0], tts_rate_fit=True, recenter=True
    )
    tts = _RatePiper(default=1.0)

    fit(str(job), cfg, tts=tts, translator=FakeTranslator())

    rows = _timed(artifacts_dir)
    row = rows[2]
    assert row["rate_action"] == "keep"
    assert row["length_scale"] == 1.0
    assert row["file"] == "speech/seg_0002.wav"
    assert row["end"] - row["start"] == pytest.approx(1.0)
    assert row["recentered"] is True
    assert 0.0 < row["recenter_seconds"] <= RECENTER_MAX_SHIFT
    assert tts.rate_calls == []
    _assert_no_bleed(rows, 7.0)


def test_rate_fit_speedup_avoids_clip(tmp_path: Path) -> None:
    # seg 1 is 1.4 s in a 1.0 s window (old path: shorten -> clip); the rate pass
    # re-synthesizes it faster so it fits without a translation retry.
    job, artifacts_dir, cfg = _make_v2_job(
        tmp_path, [2.0, 1.4, 1.0], tts_rate_fit=True, recenter=True
    )
    tts = _RatePiper(base={"b": 1.4}, default=1.0)
    translator = FakeTranslator("ngắn")

    fit(str(job), cfg, tts=tts, translator=translator)

    rows = _timed(artifacts_dir)
    row = rows[1]
    assert row["rate_action"] == "resynth"
    assert row["length_scale"] < 1.0
    assert row["action"] in {"asis", "stretch"}
    assert "clipped" not in row
    assert translator.calls == []  # the cheaper rate pass fit it, no retry
    _assert_no_bleed(rows, 7.0)


def test_rate_fit_runs_before_retranslate_when_still_too_long(
    tmp_path: Path,
) -> None:
    # seg 1: 2.0 s in a 1.0 s window. Even the clamped fast rate (1.538 s) still
    # overruns, so the translation retry must run AFTER the rate pass.
    job, artifacts_dir, cfg = _make_v2_job(
        tmp_path, [2.0, 2.0, 1.0], tts_rate_fit=True, recenter=False
    )
    tts = _RatePiper(base={"b": 2.0, "ngắn": 0.5}, default=1.0)

    fit(str(job), cfg, tts=tts, translator=FakeTranslator("ngắn"))

    rows = _timed(artifacts_dir)
    row = rows[1]
    assert row["rate_action"] == "resynth"
    assert row["retranslated"] is True
    # order: the rate-pass synthesis came first, then the retry synthesis
    assert tts.rate_calls[0] == ("b", pytest.approx(1.0 / RATE_SPEED_MAX))
    assert tts.rate_calls[1][0] == "ngắn"


def test_rate_fit_and_recenter_off_is_unchanged(tmp_path: Path) -> None:
    job, artifacts_dir, cfg = _make_v2_job(
        tmp_path, [2.0, 1.4, 1.0], tts_rate_fit=False, recenter=False
    )
    tts = FakeTTS(durations={"ngắn": 2.0})

    fit(str(job), cfg, tts=tts, translator=FakeTranslator("ngắn"))

    rows = _timed(artifacts_dir)
    for row in rows.values():
        assert "length_scale" not in row
        assert "rate_action" not in row
        assert "recentered" not in row
        assert row["start"] == row["slot_start"]
    # legacy shorten -> retry -> clip path preserved
    assert rows[1]["action"] == "clip"
    assert rows[1]["clipped"] is True
    pacing = json.loads(
        (artifacts_dir / "timing.json").read_text(encoding="utf-8")
    )
    assert "rate_fit" not in pacing
    assert set(pacing) == {
        "stretched",
        "retranslated",
        "clipped",
        "smart_trims",
        "clips",
        "max_stretch",
        "max_neighbor_delta",
        "filler_removals",
    }
    _assert_no_bleed(rows, 5.0)


def test_recenter_shift_is_bounded_and_never_crosses_window(
    tmp_path: Path,
) -> None:
    # one 0.5 s take in a 40 s window -> the shift is capped and the clip still
    # ends before the window end (no bleed).
    job, artifacts_dir, cfg = _make_rate_job(
        tmp_path,
        [(0.0, 1.0, 0.5, "a")],
        video_end=20.0,
        tail_pad=20.0,
        tts_rate_fit=False,
        recenter=True,
    )

    fit(str(job), cfg, tts=FakeTTS(), translator=FakeTranslator())

    row = _timed(artifacts_dir)[0]
    assert row["recentered"] is True
    assert row["recenter_seconds"] == pytest.approx(RECENTER_MAX_SHIFT)
    assert row["start"] >= 0.0
    assert row["start"] <= row["slot_start"] + row["window"] - (
        row["end"] - row["start"]
    ) + BLEED_TOLERANCE
    assert row["end"] <= row["slot_start"] + row["window"] + BLEED_TOLERANCE


def test_rate_fit_baseline_is_neutral(tmp_path: Path) -> None:
    # every window is 1.0 s and every natural take is 1/1.15 s -> ratio 1.15 ->
    # the video-wide baseline is the clamped median 1.10.
    natural = 1.0 / 1.15
    job, artifacts_dir, cfg = _make_rate_job(
        tmp_path,
        [(0.0, 1.0, natural, "a"), (1.0, 2.0, natural, "b"), (2.0, 3.0, natural, "c")],
        video_end=3.0,
        tail_pad=0.0,
        tts_rate_fit=True,
        recenter=False,
    )

    fit(str(job), cfg, tts=_RatePiper(default=natural), translator=FakeTranslator())

    pacing = json.loads(
        (artifacts_dir / "timing.json").read_text(encoding="utf-8")
    )
    assert pacing["rate_fit"]["global_length_scale"] == 1.0


def test_rate_fit_needs_a_piper_engine(tmp_path: Path) -> None:
    # a non-Piper engine has no ``length_scale`` knob, so the rate pass is inert
    # even with the flag on: no fitted take is written and no synth is called.
    job, artifacts_dir, cfg = _make_v2_job(
        tmp_path, [2.0, 1.0, 1.0], tts_rate_fit=True, recenter=False
    )
    fake = FakeTTS(default=1.0)

    fit(str(job), cfg, tts=fake, translator=FakeTranslator())

    rows = _timed(artifacts_dir)
    assert fake.calls == []
    assert not list((artifacts_dir / "speech").glob("*_ls.wav"))
    # rows the policy wanted to re-synthesize fall back to "off" (no knob)
    assert rows[2]["rate_action"] == "keep"
    assert rows[2]["length_scale"] == 1.0
    assert rows[2]["file"] == "speech/seg_0002.wav"


# -------------------------------------------- UA-1 sentence-unit timing
def _make_unit_job(
    tmp_path: Path,
    rows: list[dict],
    *,
    video_end: float,
    sentence_units: bool = True,
    tail_pad: float = 2.0,
    tts_rate_fit: bool = False,
    recenter: bool = False,
    units: list[dict] | None = None,
) -> tuple[Path, Path, JobConfig]:
    """Job builder for sentence-unit tests.

    ``rows`` carry ``start``/``end``/``speaker``/``en``/``vi``/``duration``; the
    pass-1 ``speech/seg_XXXX.wav`` files are written at those durations. When
    ``units`` is given it is written as ``units.json`` (the SA-1-T2 authoritative
    unit map) so the timing stage consumes it instead of re-grouping.
    """
    cache_dir = tmp_path / "cache"
    artifacts_dir = cache_dir / "units"
    speech_dir = artifacts_dir / "speech"
    speech_dir.mkdir(parents=True)

    speech = []
    script = []
    for idx, row in enumerate(rows):
        duration = float(row.get("duration", 1.0))
        _write_sine(speech_dir / f"seg_{idx:04d}.wav", duration)
        speaker = row.get("speaker", "spk0")
        speech.append(
            {
                "idx": idx,
                "start": float(row["start"]),
                "end": float(row["end"]),
                "speaker": speaker,
                "vi": row.get("vi", ""),
                "file": f"speech/seg_{idx:04d}.wav",
                "duration": duration,
            }
        )
        script.append(
            {
                "start": float(row["start"]),
                "end": float(row["end"]),
                "speaker": speaker,
                "en": row.get("en", ""),
                "vi": row.get("vi", ""),
                "truncated": False,
            }
        )
    (artifacts_dir / "speech.json").write_text(
        json.dumps(speech, ensure_ascii=False), encoding="utf-8"
    )
    (artifacts_dir / "script.json").write_text(
        json.dumps(script, ensure_ascii=False), encoding="utf-8"
    )
    job_path = artifacts_dir / "job.json"
    paths = {"script": "script.json", "speech": "speech.json", "speech_dir": "speech"}
    if units is not None:
        (artifacts_dir / "units.json").write_text(
            json.dumps(
                {"version": 1, "mode": "sentence", "units": units},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        paths["units"] = "units.json"
    JobManifest(
        input_path="demo.mp4",
        video_hash="units",
        duration=video_end,
        cache_dir=str(cache_dir),
        mode="dub",
        paths=paths,
        stages={"speak": "done"},
    ).save(job_path)
    return job_path, artifacts_dir, JobConfig(
        cache_dir=str(cache_dir),
        retranslate=True,
        tail_pad=tail_pad,
        qc_tts=False,
        tts_rate_fit=tts_rate_fit,
        recenter_audio=recenter,
        sentence_units=sentence_units,
    )


def test_sentence_units_join_dao_tao_into_one_row(tmp_path: Path) -> None:
    """The screenshot case: "…đào" / "tạo…" becomes ONE unit, ONE synthesis row."""
    rows = [
        {
            "start": 0.0,
            "end": 2.0,
            "speaker": "spk0",
            "en": "the model was pre-trained and",
            "vi": "mô hình được đào",
            "duration": 1.0,
        },
        {
            "start": 2.0,
            "end": 4.0,
            "speaker": "spk0",
            "en": "self-supervised on lots of data",
            "vi": "tạo trên nhiều dữ liệu",
            "duration": 1.0,
        },
    ]
    job_path, artifacts_dir, cfg = _make_unit_job(tmp_path, rows, video_end=4.0)
    tts = FakeTTS(default=1.0)

    fit(str(job_path), cfg, tts=tts, translator=FakeTranslator())

    # ONE synthesis, whose text carries the joined "đào tạo" contiguously
    assert len(tts.calls) == 1
    assert "đào tạo" in tts.calls[0][0]
    # ONE placed row spanning both members: no boundary can exist between them
    timed = list(_timed(artifacts_dir).values())
    assert len(timed) == 1
    row = timed[0]
    assert row["members"] == [0, 1]
    assert row["idx"] == 0
    assert row["file"] == "speech/unit_0000.wav"
    assert row["action"] == "asis"


def test_single_segment_units_reuse_pass1_wav(tmp_path: Path) -> None:
    rows = [
        {
            "start": 0.0,
            "end": 1.0,
            "speaker": "spk0",
            "en": "Hello there.",
            "vi": "xin chào.",
            "duration": 1.0,
        },
        {
            "start": 2.0,
            "end": 3.0,
            "speaker": "spk0",
            "en": "Goodbye now.",
            "vi": "tạm biệt.",
            "duration": 1.0,
        },
    ]
    job_path, artifacts_dir, cfg = _make_unit_job(tmp_path, rows, video_end=3.0)
    tts = FakeTTS()

    fit(str(job_path), cfg, tts=tts, translator=FakeTranslator())

    assert tts.calls == []  # no extra synthesis: the pass-1 WAVs are reused
    timed = list(_timed(artifacts_dir).values())
    assert [row["members"] for row in timed] == [[0], [1]]
    assert [row["file"] for row in timed] == [
        "speech/seg_0000.wav",
        "speech/seg_0001.wav",
    ]


def test_unit_audio_never_crosses_next_unit_start(tmp_path: Path) -> None:
    """The no-bleed invariant holds at unit granularity."""
    rows = [
        {
            "start": 0.0,
            "end": 1.5,
            "speaker": "spk0",
            "en": "the first half of the sentence",
            "vi": "nửa đầu của câu",
            "duration": 1.0,
        },
        {
            "start": 1.5,
            "end": 3.0,
            "speaker": "spk0",
            "en": "and the second half here.",
            "vi": "và nửa sau ở đây.",
            "duration": 1.0,
        },
        {
            "start": 3.0,
            "end": 4.0,
            "speaker": "spk0",
            "en": "Next sentence follows.",
            "vi": "câu tiếp theo.",
            "duration": 1.0,
        },
    ]
    job_path, artifacts_dir, cfg = _make_unit_job(tmp_path, rows, video_end=4.0)
    first_unit_text = "nửa đầu của câu và nửa sau ở đây."
    tts = FakeTTS(durations={first_unit_text: 4.0})

    fit(str(job_path), cfg, tts=tts, translator=FakeTranslator())

    timed = list(_timed(artifacts_dir).values())
    assert [row["members"] for row in timed] == [[0, 1], [2]]
    # the first unit's 4.0 s take is trimmed at its 3.0 s window, never reaching
    # the next unit's first member start
    assert timed[0]["end"] <= 3.0 + BLEED_TOLERANCE
    assert timed[1]["start"] >= 3.0 - BLEED_TOLERANCE


def test_speaker_change_splits_units(tmp_path: Path) -> None:
    """A unit never crosses a speaker change: two speakers -> two units."""
    rows = [
        {
            "start": 0.0,
            "end": 1.0,
            "speaker": "spk0",
            "en": "speaker one part a",
            "vi": "một a",
            "duration": 1.0,
        },
        {
            "start": 1.0,
            "end": 2.0,
            "speaker": "spk0",
            "en": "speaker one part b.",
            "vi": "một b.",
            "duration": 1.0,
        },
        {
            "start": 2.0,
            "end": 3.0,
            "speaker": "spk1",
            "en": "speaker two part a",
            "vi": "hai a",
            "duration": 1.0,
        },
        {
            "start": 3.0,
            "end": 4.0,
            "speaker": "spk1",
            "en": "speaker two part b.",
            "vi": "hai b.",
            "duration": 1.0,
        },
    ]
    job_path, artifacts_dir, cfg = _make_unit_job(tmp_path, rows, video_end=4.0)
    tts = FakeTTS(default=1.0)

    fit(str(job_path), cfg, tts=tts, translator=FakeTranslator())

    timed = list(_timed(artifacts_dir).values())
    assert [row["members"] for row in timed] == [[0, 1], [2, 3]]
    # each unit is synthesized separately (its own speaker's text), never merged
    assert sorted(call[0] for call in tts.calls) == ["hai a hai b.", "một a một b."]


def test_unit_text_skips_empty_vi_member(tmp_path: Path) -> None:
    rows = [
        {
            "start": 0.0,
            "end": 1.0,
            "speaker": "spk0",
            "en": "alpha beta",
            "vi": "alpha",
            "duration": 1.0,
        },
        {
            "start": 1.0,
            "end": 2.0,
            "speaker": "spk0",
            "en": "gamma delta",
            "vi": "",
            "duration": 1.0,
        },
        {
            "start": 2.0,
            "end": 3.0,
            "speaker": "spk0",
            "en": "epsilon zeta",
            "vi": "gamma",
            "duration": 1.0,
        },
    ]
    job_path, artifacts_dir, cfg = _make_unit_job(tmp_path, rows, video_end=3.0)
    tts = FakeTTS(default=1.0)

    fit(str(job_path), cfg, tts=tts, translator=FakeTranslator())

    timed = list(_timed(artifacts_dir).values())
    assert len(timed) == 1
    assert timed[0]["members"] == [0, 1, 2]  # the empty member is still a member
    assert tts.calls[0][0] == "alpha gamma"  # but contributes no text to the join


def test_sentence_units_off_is_per_segment(tmp_path: Path) -> None:
    """``--no-sentence-units`` keeps the per-segment rows (no ``members``)."""
    rows = [
        {
            "start": 0.0,
            "end": 2.0,
            "speaker": "spk0",
            "en": "part one and",
            "vi": "phần một",
            "duration": 1.0,
        },
        {
            "start": 2.0,
            "end": 4.0,
            "speaker": "spk0",
            "en": "part two here",
            "vi": "phần hai",
            "duration": 1.0,
        },
    ]
    job_path, artifacts_dir, cfg = _make_unit_job(
        tmp_path, rows, video_end=4.0, sentence_units=False
    )
    legacy_keys = {
        "idx",
        "file",
        "start",
        "end",
        "slot_start",
        "stretch",
        "window",
        "action",
        "retranslated",
        "clipped",
        "smart_trim",
        "length_scale",
        "rate_action",
        "recentered",
        "recenter_seconds",
    }

    fit(str(job_path), cfg, tts=FakeTTS(), translator=FakeTranslator())

    timed = list(_timed(artifacts_dir).values())
    assert len(timed) == 2  # one row per segment, no grouping
    assert [row["file"] for row in timed] == [
        "speech/seg_0000.wav",
        "speech/seg_0001.wav",
    ]
    for row in timed:
        assert "members" not in row
        assert "align" not in row
        assert "member_spans" not in row
        assert set(row) <= legacy_keys


def test_compute_coverage_unit_rows() -> None:
    script = [
        {"start": 0.0, "end": 2.0},
        {"start": 2.0, "end": 4.0},
        {"start": 10.0, "end": 20.0},
    ]
    speech = {0: {"duration": 2.0}, 1: {"duration": 2.0}, 2: {"duration": 2.0}}

    # unit [0, 1] fills its 4 s of member spans; unit [2] fills 2 of its 10 s
    assert compute_coverage(script, speech, [[0, 1], [2]]) == pytest.approx(
        (4.0 + 2.0) / 14.0
    )
    # degenerate units reproduce the per-segment result exactly
    assert compute_coverage(
        script, speech, [[0], [1], [2]]
    ) == pytest.approx(compute_coverage(script, speech))


# ------------------------------------- SA-1-T3 authoritative units + alignment
UNIT_TEXT = "toàn bộ câu dịch hoàn chỉnh."


class FakeASR:
    """ASR stub: records calls and returns the configured words (or none)."""

    def __init__(self, words):
        self.words = words
        self.calls: list[tuple[str, object, object]] = []

    def transcribe(self, wav_path, language=None, word_timestamps=True):
        self.calls.append((str(wav_path), language, word_timestamps))
        return [
            Segment(
                start=0.0,
                end=1.0,
                text="vi",
                speaker="spk0",
                words=None if self.words is None else list(self.words),
            )
        ]


def _unit_entry(index, indices, start, end, vi, *, speaker="spk0") -> dict:
    """One ``units.json`` entry with the exact SA-1-T2 schema."""
    return {
        "index": index,
        "indices": list(indices),
        "start": float(start),
        "end": float(end),
        "speaker": speaker,
        "source": "source words here",
        "vi": vi,
        "words_in": 1,
        "coverage": 1.0,
        "guard": "none",
        "battery": "ok",
        "open": False,
    }


TWO_ROWS = [
    {
        "start": 0.0,
        "end": 1.0,
        "speaker": "spk0",
        "en": "two words",
        "vi": "mảnh một",
        "duration": 1.0,
    },
    {
        "start": 1.0,
        "end": 2.0,
        "speaker": "spk0",
        "en": "two words two",
        "vi": "mảnh hai",
        "duration": 1.0,
    },
]
TWO_UNIT = [_unit_entry(0, [0, 1], 0.0, 2.0, UNIT_TEXT)]


def test_units_json_text_drives_synthesis_verbatim(monkeypatch, tmp_path: Path) -> None:
    """(a) the unit's ``vi`` is synthesized verbatim; group_units is not called."""
    job_path, artifacts_dir, cfg = _make_unit_job(
        tmp_path, TWO_ROWS, video_end=2.0, units=TWO_UNIT
    )

    def _boom(rows):  # pragma: no cover - only runs on a regression
        raise AssertionError(
            "group_units must not run when units.json is authoritative"
        )

    monkeypatch.setattr(timing_mod, "group_units", _boom)
    tts = FakeTTS(default=1.0)

    fit(str(job_path), cfg, tts=tts, translator=FakeTranslator(), asr=FakeASR(None))

    assert tts.calls, "the multi-member unit must be synthesized once"
    assert tts.calls[0][0] == UNIT_TEXT
    assert tts.calls[0][0] != "mảnh một mảnh hai"


def test_align_display_asr_path_snaps_to_word_edges(tmp_path: Path) -> None:
    """(b) a stub ASR yields align=='asr' with word-edge, in-window spans."""
    words = [
        {"start": 0.0, "end": 0.25, "word": "a"},
        {"start": 0.25, "end": 0.5, "word": "b"},
        {"start": 0.5, "end": 0.75, "word": "c"},
        {"start": 0.75, "end": 1.0, "word": "d"},
    ]
    job_path, artifacts_dir, cfg = _make_unit_job(
        tmp_path, TWO_ROWS, video_end=2.0, units=TWO_UNIT
    )
    asr = FakeASR(words)

    fit(
        str(job_path),
        cfg,
        tts=FakeTTS(default=1.0),
        translator=FakeTranslator(),
        asr=asr,
    )

    assert asr.calls and asr.calls[0][1] == "vi" and asr.calls[0][2] is True
    row = _timed(artifacts_dir)[0]
    assert row["align"] == "asr"
    spans = row["member_spans"]
    assert [span[0] for span in spans] == [0, 1]
    edges = {0.0, 0.25, 0.5, 0.75, 1.0}
    previous_end = None
    for _member, start, end, piece in spans:
        assert piece
        assert start <= end
        assert start >= row["start"] - 1e-9
        assert end <= row["end"] + 1e-6
        assert any(abs(start - edge) < 1e-6 for edge in edges)
        assert any(abs(end - edge) < 1e-6 for edge in edges)
        if previous_end is not None:
            assert start >= previous_end - 1e-9
        previous_end = end
    assert spans[0][1] == pytest.approx(row["start"], abs=1e-9)
    assert spans[-1][2] == pytest.approx(row["end"], abs=1e-6)

    # Q-1-T4: the ASR word timeline is persisted, monotone, absolute and inside
    # the placed audio, with the word text carried through.
    align_words = row["align_words"]
    assert [word[2] for word in align_words] == ["a", "b", "c", "d"]
    starts = [word[0] for word in align_words]
    assert starts == pytest.approx([0.0, 0.25, 0.5, 0.75], abs=1e-9)
    assert starts == sorted(starts)
    for start_w, end_w, _word in align_words:
        assert start_w <= end_w
        assert row["start"] - 1e-9 <= start_w
        assert end_w <= row["end"] + 1e-6

    pacing = json.loads((artifacts_dir / "timing.json").read_text(encoding="utf-8"))
    assert pacing["align"] == {"asr": 1, "split": 0, "single": 0, "unavailable": 0}


def test_align_display_split_fallback_when_no_words(tmp_path: Path) -> None:
    """(c) ``words=None`` falls back to split_translation pieces."""
    job_path, artifacts_dir, cfg = _make_unit_job(
        tmp_path, TWO_ROWS, video_end=2.0, units=TWO_UNIT
    )

    fit(
        str(job_path),
        cfg,
        tts=FakeTTS(default=1.0),
        translator=FakeTranslator(),
        asr=FakeASR(None),
    )

    row = _timed(artifacts_dir)[0]
    assert row["align"] == "split"
    # no ``asr`` alignment -> no word timeline persisted (legacy row shape)
    assert "align_words" not in row
    script = json.loads((artifacts_dir / "script.json").read_text(encoding="utf-8"))
    shares = timing_mod._source_shares(script, [0, 1])
    expected = split_translation(UNIT_TEXT, shares)
    assert [span[3] for span in row["member_spans"]] == expected


def test_no_align_display_never_builds_or_calls_asr(
    monkeypatch, tmp_path: Path
) -> None:
    """(d) --no-align-display skips the ASR entirely (still records spans)."""
    job_path, artifacts_dir, cfg = _make_unit_job(
        tmp_path, TWO_ROWS, video_end=2.0, units=TWO_UNIT
    )
    cfg.align_display = False
    asr = FakeASR([(0.0, 0.5), (0.5, 1.0)])

    def _boom(model, device="auto"):  # pragma: no cover - only on a regression
        raise AssertionError("build_asr must not run with --no-align-display")

    monkeypatch.setattr(timing_mod, "build_asr", _boom)

    fit(str(job_path), cfg, tts=FakeTTS(default=1.0), translator=FakeTranslator(), asr=asr)

    assert asr.calls == []
    row = _timed(artifacts_dir)[0]
    assert row["align"] == "split"
    assert row["member_spans"]


def test_single_member_units_are_single(tmp_path: Path) -> None:
    """(e) single-member units get align=='single' and their text equals the vi."""
    rows = [
        {
            "start": 0.0,
            "end": 1.0,
            "speaker": "spk0",
            "en": "Hello there.",
            "vi": "xin chào.",
            "duration": 1.0,
        },
        {
            "start": 2.0,
            "end": 3.0,
            "speaker": "spk0",
            "en": "Goodbye now.",
            "vi": "tạm biệt.",
            "duration": 1.0,
        },
    ]
    units = [
        _unit_entry(0, [0], 0.0, 1.0, "xin chào."),
        _unit_entry(1, [1], 2.0, 3.0, "tạm biệt."),
    ]
    job_path, artifacts_dir, cfg = _make_unit_job(
        tmp_path, rows, video_end=3.0, units=units
    )
    tts = FakeTTS()

    fit(str(job_path), cfg, tts=tts, translator=FakeTranslator(), asr=FakeASR(None))

    assert tts.calls == []  # single-member units reuse the pass-1 WAVs
    script = json.loads((artifacts_dir / "script.json").read_text(encoding="utf-8"))
    doc = json.loads((artifacts_dir / "units.json").read_text(encoding="utf-8"))
    for entry in doc["units"]:
        # the authoritative unit text equals the single member row's vi
        assert entry["vi"] == script[entry["indices"][0]]["vi"]
    timed = list(_timed(artifacts_dir).values())
    assert [row["align"] for row in timed] == ["single", "single"]
    for row in timed:
        assert len(row["member_spans"]) == 1
        assert row["member_spans"][0][3]


def test_units_json_missing_falls_back_with_warning(
    tmp_path: Path, caplog
) -> None:
    """(f) no units.json -> warning + re-grouping, still valid timed rows."""
    job_path, artifacts_dir, cfg = _make_unit_job(tmp_path, TWO_ROWS, video_end=2.0)

    with caplog.at_level(logging.WARNING, logger="vidub.stages.timing"):
        fit(
            str(job_path),
            cfg,
            tts=FakeTTS(default=1.0),
            translator=FakeTranslator(),
            asr=FakeASR(None),
        )

    assert "units.json not found" in caplog.text
    timed = list(_timed(artifacts_dir).values())
    assert len(timed) == 1
    assert timed[0]["members"] == [0, 1]
    assert timed[0]["align"] in {"asr", "split", "unavailable"}


def test_units_json_inconsistent_falls_back_with_warning(
    tmp_path: Path, caplog
) -> None:
    """(f) an incomplete units.json -> warning + re-grouping, valid rows."""
    bad = [_unit_entry(0, [0], 0.0, 1.0, "chỉ một mảnh")]  # row 1 is missing
    job_path, artifacts_dir, cfg = _make_unit_job(
        tmp_path, TWO_ROWS, video_end=2.0, units=bad
    )

    with caplog.at_level(logging.WARNING, logger="vidub.stages.timing"):
        fit(
            str(job_path),
            cfg,
            tts=FakeTTS(default=1.0),
            translator=FakeTranslator(),
            asr=FakeASR(None),
        )

    assert "inconsistent" in caplog.text
    timed = list(_timed(artifacts_dir).values())
    assert len(timed) == 1
    assert timed[0]["members"] == [0, 1]


def test_timing_signature_flags_and_manifest_roundtrip() -> None:
    """(g) each of the four flags changes the timing signature; field round-trips."""
    from vidub.pipeline import timing_signature

    base = timing_signature(JobConfig())
    assert timing_signature(JobConfig(tts_rate_fit=False)) != base
    assert timing_signature(JobConfig(recenter_audio=False)) != base
    assert timing_signature(JobConfig(align_display=False)) != base
    assert timing_signature(JobConfig(sentence_units=False)) != base
    assert timing_signature(JobConfig()) == base

    manifest = JobManifest(timing_signature=base)
    reloaded = JobManifest.from_dict(manifest.to_dict())
    assert reloaded.timing_signature == base


def _pipeline_manifest(tmp_path: Path, timing_signature: str) -> Path:
    import vidub.pipeline as pipeline

    cache_dir = tmp_path / "pcache"
    artifacts_dir = cache_dir / "hash1"
    artifacts_dir.mkdir(parents=True)
    JobManifest(
        input_path="demo.mp4",
        video_hash="hash1",
        cache_dir=str(cache_dir),
        mode="dub",
        stages={name: "done" for name in pipeline.STAGES},
        timing_signature=timing_signature,
    ).save(artifacts_dir / "job.json")
    return cache_dir


def _stub_pipeline(monkeypatch, cache_dir: Path, tmp_path: Path, called: list[str]) -> None:
    import vidub.pipeline as pipeline

    artifacts_dir = cache_dir / "hash1"
    monkeypatch.setattr(
        pipeline, "stage_artifacts_present", lambda stage, manifest: True
    )
    monkeypatch.setattr(pipeline.ingest_mod, "video_hash", lambda video: "hash1")

    def make_stage(name: str):
        def _stage(manifest_path, cfg, **kwargs):
            called.append(name)
            loaded = JobManifest.load(manifest_path)
            loaded.stages[name] = "done"
            loaded.save(manifest_path)
            return loaded

        return _stage

    def fake_ingest(video, cfg):
        called.append("ingest")
        return JobManifest.load(artifacts_dir / "job.json")

    monkeypatch.setattr(pipeline.ingest_mod, "ingest", fake_ingest)
    monkeypatch.setattr(pipeline.transcribe_mod, "transcribe", make_stage("transcribe"))
    monkeypatch.setattr(pipeline.translate_mod, "translate", make_stage("translate"))
    monkeypatch.setattr(pipeline.speak_mod, "speak", make_stage("speak"))
    monkeypatch.setattr(pipeline.timing_mod, "fit", make_stage("timing"))
    monkeypatch.setattr(pipeline.mix_mod, "mix", make_stage("mix"))
    monkeypatch.setattr(pipeline.export_mod, "export", make_stage("export"))
    monkeypatch.setattr(
        pipeline.export_mod, "output_dir_for", lambda manifest: tmp_path / "out"
    )


def test_timing_signature_change_reruns_timing_and_downstream(
    monkeypatch, tmp_path: Path
) -> None:
    """(g) a flipped timing flag re-runs timing + mix + export."""
    import vidub.pipeline as pipeline

    called: list[str] = []
    cache_dir = _pipeline_manifest(
        tmp_path, pipeline.timing_signature(JobConfig(align_display=True))
    )
    _stub_pipeline(monkeypatch, cache_dir, tmp_path, called)

    pipeline.run_pipeline(
        "video.mp4",
        JobConfig(cache_dir=str(cache_dir), align_display=False),
        resume=True,
    )

    assert called == ["timing", "mix", "export"]


def test_matching_timing_signature_skips_timing(
    monkeypatch, tmp_path: Path
) -> None:
    """(g) an unchanged timing signature keeps resume skipping the stage."""
    import vidub.pipeline as pipeline

    called: list[str] = []
    cfg = JobConfig(cache_dir=str(tmp_path / "pcache"))
    cache_dir = _pipeline_manifest(tmp_path, pipeline.timing_signature(cfg))
    _stub_pipeline(monkeypatch, cache_dir, tmp_path, called)

    pipeline.run_pipeline("video.mp4", cfg, resume=True)

    assert called == []
