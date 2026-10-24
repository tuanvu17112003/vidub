"""Tests for the S6 export stage (M1-T6)."""

from __future__ import annotations

import json
import math
import struct
import wave
from pathlib import Path

import pysubs2
import pytest

from vidub.config import JobConfig
from vidub.manifest import JobManifest
from vidub.stages.export import (
    ASS_STYLE_NAME,
    DUBBED_NAME,
    REPORT_NAME,
    SCRIPT_TSV_NAME,
    SUB_POSITION_ALIGNMENT,
    SUB_POSITION_LEGACY_ALIGNMENT,
    SUBTITLES_ASS_NAME,
    SUBTITLES_NAME,
    TTS_SKIPPED_NOTE,
    _unit_subtitle_rows,
    build_mux_command,
    build_report,
    burn_subtitles_filter,
    export,
    write_script_tsv,
    write_subtitles_ass,
    write_subtitles_srt,
)
from vidub.utils.proc import find_tool, run


def test_unit_subtitles_prefer_selection_over_timing_retry(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "job1"
    artifacts_dir.mkdir()
    units = {
        "units": [
            {
                "index": 0,
                "indices": [0],
                "start": 0.0,
                "end": 1.0,
                "vi": "legacy unit text",
                "subtitle_vi": "caption selection",
            }
        ]
    }
    (artifacts_dir / "units.json").write_text(
        json.dumps(units, ensure_ascii=False), encoding="utf-8"
    )
    (artifacts_dir / "timed.json").write_text(
        json.dumps(
            [
                {
                    "idx": 0,
                    "start": 0.0,
                    "end": 1.0,
                    "retranslated": True,
                    "vi": "timing retry text",
                }
            ]
        ),
        encoding="utf-8",
    )
    manifest = JobManifest(
        video_hash="job1",
        cache_dir=str(tmp_path),
        paths={"units": "units.json", "timed": "timed.json"},
    )

    rows = _unit_subtitle_rows(manifest)

    assert rows is not None
    assert rows[0]["vi"] == "caption selection"

    units["units"][0].pop("subtitle_vi")
    (artifacts_dir / "units.json").write_text(
        json.dumps(units, ensure_ascii=False), encoding="utf-8"
    )
    rows = _unit_subtitle_rows(manifest)
    assert rows is not None
    assert rows[0]["vi"] == "timing retry text"


def _write_sine(path: Path, duration: float = 1.0, rate: int = 22050, freq: float = 440.0) -> None:
    """Write a mono 16-bit sine WAV of ``duration`` seconds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = bytearray()
    for i in range(round(duration * rate)):
        value = int(0.4 * 32767 * math.sin(2.0 * math.pi * freq * i / rate))
        frames += struct.pack("<h", value)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(bytes(frames))


def _probe(path: Path) -> list[dict]:
    ffprobe = find_tool("ffprobe")
    assert ffprobe is not None
    proc = run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name:stream_tags=language",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout)["streams"]


def _duration(path: Path) -> float:
    ffprobe = find_tool("ffprobe")
    assert ffprobe is not None
    proc = run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1", str(path)],
        capture_output=True,
        text=True,
    )
    return float(proc.stdout.strip().split("=")[1])


def _by_type(streams: list[dict]) -> dict[str, dict]:
    return {stream["codec_type"]: stream for stream in streams}


def _make_job(
    tmp_path: Path,
    video: Path,
    mode: str = "dub",
    with_dub: bool = True,
    script: list[dict] | None = None,
) -> tuple[Path, Path, JobConfig]:
    cache_dir = tmp_path / "cache"
    artifacts_dir = cache_dir / "h1"
    artifacts_dir.mkdir(parents=True)

    if script is None:
        script = [
            {"start": 0.0, "end": 1.0, "speaker": "spk0", "en": "Hello world", "vi": "Xin chào thế giới", "truncated": False},
            {"start": 1.0, "end": 2.0, "speaker": "spk0", "en": "Bye now", "vi": "Tạm biệt", "truncated": False},
        ]
    (artifacts_dir / "script.json").write_text(
        json.dumps(script, ensure_ascii=False), encoding="utf-8"
    )

    paths = {"script": "script.json"}
    if with_dub:
        _write_sine(artifacts_dir / "dub_track.wav", 2.0)
        paths["dub_track"] = "dub_track.wav"

    job_path = artifacts_dir / "job.json"
    JobManifest(
        input_path=str(video),
        video_hash="h1",
        cache_dir=str(cache_dir),
        mode=mode,
        duration=4.0,
        paths=paths,
        stages={"ingest": "done", "transcribe": "done", "translate": "done"},
    ).save(job_path)
    return job_path, artifacts_dir, JobConfig(cache_dir=str(cache_dir))


def test_export_muxes_dub_and_subtitles(
    tiny_clip: Path, tmp_path: Path, monkeypatch, ffmpeg_available: bool
) -> None:
    monkeypatch.chdir(tmp_path)
    job_path, _artifacts, cfg = _make_job(tmp_path, tiny_clip, mode="dub")

    export(str(job_path), cfg, burn=False)

    out_dir = tmp_path / "output" / tiny_clip.stem
    dubbed = out_dir / DUBBED_NAME
    assert dubbed.exists()
    assert (out_dir / SUBTITLES_NAME).exists()
    assert (out_dir / SCRIPT_TSV_NAME).exists()
    assert (out_dir / REPORT_NAME).exists()

    streams = _by_type(_probe(dubbed))
    assert streams["video"]["codec_name"] == "h264"  # copied, not re-encoded
    assert streams["audio"]["codec_name"] == "aac"
    assert streams["subtitle"]["codec_name"] == "mov_text"
    # MP4 stores the ISO 639-2 code; ffprobe may report "vie".
    assert streams["subtitle"].get("tags", {}).get("language") in {"vi", "vie"}

    assert abs(_duration(dubbed) - _duration(tiny_clip)) <= 0.5

    lines = (out_dir / SCRIPT_TSV_NAME).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[0].split("\t") == [
        "0.0", "1.0", "spk0", "Hello world", "Xin chào thế giới",
    ]


def test_export_subtitles_only_copies_audio(
    tiny_clip: Path, tmp_path: Path, monkeypatch, ffmpeg_available: bool
) -> None:
    monkeypatch.chdir(tmp_path)
    job_path, _artifacts, cfg = _make_job(tmp_path, tiny_clip, mode="subtitles-only", with_dub=False)

    export(str(job_path), cfg)

    out_dir = tmp_path / "output" / tiny_clip.stem
    dubbed = out_dir / DUBBED_NAME
    assert dubbed.exists()
    streams = _by_type(_probe(dubbed))
    assert streams["video"]["codec_name"] == "h264"
    assert streams["audio"]["codec_name"] == "aac"  # copied from the source clip
    assert streams["subtitle"]["codec_name"] == "mov_text"
    assert streams["subtitle"].get("tags", {}).get("language") in {"vi", "vie"}

    # The report states honestly what this mode skipped (SU-1-T1): no TTS engine
    # ran, the three audio stages are skipped, and the outputs name the artifacts
    # this mode actually writes (the vietsub MP4 + the .srt).
    report = json.loads((out_dir / REPORT_NAME).read_text(encoding="utf-8"))
    assert report["mode"] == "subtitles-only"
    assert report["engines"]["tts"] == TTS_SKIPPED_NOTE
    for stage in ("speak", "timing", "mix"):
        assert report["stages"][stage]["status"] == "skipped"
    for key in ("dubbed", "subtitles"):
        assert (tmp_path / report["outputs"][key]).exists(), report["outputs"]

    # SU-1-T2: the styled ASS twin exists, carries the default style, and is
    # named in the report outputs.
    ass_path = out_dir / SUBTITLES_ASS_NAME
    assert ass_path.exists()
    ass = pysubs2.load(str(ass_path))
    style = ass.styles[ASS_STYLE_NAME]
    assert int(style.alignment) == SUB_POSITION_ALIGNMENT["bottom"]
    assert style.fontsize == cfg.sub_font_size
    assert style.marginv == cfg.sub_margin_v
    assert report["outputs"]["subtitles_ass"].endswith(SUBTITLES_ASS_NAME)


def test_export_uses_explicit_script_subtitle_text(
    tiny_clip: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    script = [
        {
            "start": 0.0,
            "end": 1.0,
            "speaker": "spk0",
            "en": "Hello",
            "vi": "legacy text",
            "subtitle_vi": "caption selection",
            "truncated": False,
        }
    ]
    job_path, _artifacts, cfg = _make_job(
        tmp_path, tiny_clip, mode="subtitles-only", with_dub=False, script=script
    )
    cfg.qc_final = False

    def fake_mux(command: list[str]) -> None:
        Path(command[-1]).write_bytes(b"mux")

    monkeypatch.setattr("vidub.stages.export.run_ffmpeg", fake_mux)

    export(str(job_path), cfg)

    srt_path = tmp_path / "output" / tiny_clip.stem / SUBTITLES_NAME
    assert pysubs2.load(str(srt_path), encoding="utf-8")[0].text == "caption selection"


def test_build_report_subtitles_only_marks_tts_and_keeps_outputs(
    tmp_path: Path,
) -> None:
    """``build_report`` alone: engines.tts + outputs + untouched dub wording."""
    stages = {
        "ingest": "done",
        "transcribe": "done",
        "translate": "done",
        "export": "done",
    }
    manifest = JobManifest(
        input_path="x/demo.mp4",
        video_hash="h",
        cache_dir=str(tmp_path),
        mode="subtitles-only",
        tts_engine="piper",
        stages=stages,
    )

    report = build_report(manifest)

    assert report["mode"] == "subtitles-only"
    assert report["engines"]["tts"] == TTS_SKIPPED_NOTE
    for stage in ("speak", "timing", "mix"):
        assert report["stages"][stage]["status"] == "skipped"
    # Existing artifact keys are kept (no rename) and still name this mode's files.
    assert report["outputs"]["dubbed"].endswith(DUBBED_NAME)
    assert report["outputs"]["subtitles"].endswith(SUBTITLES_NAME)
    assert report["outputs"]["subtitles_ass"].endswith(SUBTITLES_ASS_NAME)

    # Other modes keep the previous wording exactly.
    dub = build_report(
        JobManifest(
            input_path="x/demo.mp4",
            video_hash="h",
            cache_dir=str(tmp_path),
            mode="dub",
            tts_engine="piper",
            stages=stages,
        )
    )
    assert dub["engines"]["tts"] == "piper"


# --------------------------------------------------- subtitle styling (SU-1-T2)
_SRT_SCRIPT = [
    {"start": 0.0, "end": 1.0, "vi": "Xin chào"},
    {"start": 1.0, "end": 2.0, "vi": "Tạm biệt"},
]
# Golden SRT bytes captured from the current writer (CRLF, blank line between
# cues). The default style must not change these.
_SRT_GOLDEN = (
    b"1\r\n00:00:00,000 --> 00:00:01,000\r\nXin ch\xc3\xa0o\r\n\r\n"
    b"2\r\n00:00:01,000 --> 00:00:02,000\r\nT\xe1\xba\xa1m bi\xe1\xbb\x87t\r\n\r\n"
)


def test_sub_position_alignment_mapping() -> None:
    # ASS (\an): 2 bottom, 8 top, 5 middle.
    assert SUB_POSITION_ALIGNMENT == {"bottom": 2, "top": 8, "center": 5}
    # legacy SSA (\a), what the SRT-fed burn's force_style uses:
    # 2 bottom, 6 top, 10 middle.
    assert SUB_POSITION_LEGACY_ALIGNMENT == {"bottom": 2, "top": 6, "center": 10}


def test_write_subtitles_srt_default_is_byte_identical(tmp_path: Path) -> None:
    produced = write_subtitles_srt(_SRT_SCRIPT, tmp_path / "s.srt")
    assert produced.read_bytes() == _SRT_GOLDEN


@pytest.mark.parametrize(
    "position,alignment", [("bottom", 2), ("top", 8), ("center", 5)]
)
def test_write_subtitles_ass_style_block(
    tmp_path: Path, position: str, alignment: int
) -> None:
    path = write_subtitles_ass(
        _SRT_SCRIPT,
        tmp_path / f"{position}.ass",
        position=position,
        font_size=30,
        margin_v=12,
    )

    subs = pysubs2.load(str(path))
    style = subs.styles[ASS_STYLE_NAME]
    assert int(style.alignment) == alignment
    assert style.fontsize == 30
    assert style.marginv == 12
    assert style.fontname == "Arial"  # generic default; missing font falls back
    assert style.bold is False
    assert style.outline == 2
    assert style.shadow == 1
    # Every event references the named style, and the text is preserved.
    assert [event.style for event in subs] == [ASS_STYLE_NAME, ASS_STYLE_NAME]
    assert [event.text for event in subs] == ["Xin chào", "Tạm biệt"]


def test_burn_filter_carries_style_with_escaping() -> None:
    # Commas stay inside the single-quoted style list (not filter separators).
    # Top uses the legacy SSA value 6 (the SRT-fed filter's force_style numbering).
    assert (
        burn_subtitles_filter("out/sub.srt", "top", 30, 12)
        == "subtitles=out/sub.srt:force_style='Alignment=6,FontSize=30,MarginV=12'"
    )
    # A Windows path keeps its ':' escaped and the style list intact.
    assert burn_subtitles_filter("C:/cache/sub.srt", "center", 28, 0) == (
        "subtitles=C\\:/cache/sub.srt:"
        "force_style='Alignment=10,FontSize=28,MarginV=0'"
    )
    # Bottom is 2 in both numberings.
    assert burn_subtitles_filter("out/sub.srt", "bottom", 24, 24).endswith(
        "force_style='Alignment=2,FontSize=24,MarginV=24'"
    )


def test_build_mux_command_burn_applies_style(tmp_path: Path) -> None:
    srt = tmp_path / "s.srt"
    write_subtitles_srt(_SRT_SCRIPT, srt)

    command = build_mux_command(
        Path("v.mp4"),
        None,
        srt,
        Path("o.mp4"),
        "subtitles-only",
        True,
        sub_position="top",
        sub_font_size=30,
        sub_margin_v=12,
    )

    vf = command[command.index("-vf") + 1]
    assert vf.startswith("subtitles=")
    assert vf.endswith(":force_style='Alignment=6,FontSize=30,MarginV=12'")
    # Burning still re-encodes video and copies the original audio.
    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-c:a") + 1] == "copy"


def test_build_mux_command_soft_has_no_style(tmp_path: Path) -> None:
    """``burn=False`` behaviour is untouched: soft mov_text, no force_style."""
    srt = tmp_path / "s.srt"
    write_subtitles_srt(_SRT_SCRIPT, srt)

    command = build_mux_command(
        Path("v.mp4"),
        None,
        srt,
        Path("o.mp4"),
        "subtitles-only",
        False,
        sub_position="center",
        sub_font_size=30,
        sub_margin_v=0,
    )

    assert "-vf" not in command
    assert command[command.index("-c:v") + 1] == "copy"
    assert command[command.index("-c:a") + 1] == "copy"
    assert "mov_text" in command


def test_export_burn_reencodes_without_soft_subtitles(
    tiny_clip: Path, tmp_path: Path, monkeypatch, ffmpeg_available: bool
) -> None:
    monkeypatch.chdir(tmp_path)
    job_path, _artifacts, cfg = _make_job(tmp_path, tiny_clip, mode="dub")

    export(str(job_path), cfg, burn=True)

    streams = _probe(tmp_path / "output" / tiny_clip.stem / DUBBED_NAME)
    types = _by_type(streams)
    assert "subtitle" not in types
    assert types["video"]["codec_name"] == "h264"
    assert types["audio"]["codec_name"] == "aac"


def test_export_without_speech_writes_no_subtitle_stream(
    tiny_clip: Path, tmp_path: Path, monkeypatch, ffmpeg_available: bool
) -> None:
    """A 0-segment job must mux without a subtitle stream (M2-T6 bug fix)."""
    monkeypatch.chdir(tmp_path)
    job_path, _artifacts, cfg = _make_job(tmp_path, tiny_clip, mode="dub", script=[])

    export(str(job_path), cfg)

    out_dir = tmp_path / "output" / tiny_clip.stem
    dubbed = out_dir / DUBBED_NAME
    assert dubbed.exists()

    streams = _by_type(_probe(dubbed))
    assert "subtitle" not in streams
    assert streams["video"]["codec_name"] == "h264"
    assert streams["audio"]["codec_name"] == "aac"

    report = json.loads((out_dir / REPORT_NAME).read_text(encoding="utf-8"))
    assert report["subtitles"] == "none (no speech detected)"
    assert report["meta"]["tool_version"] == "0.11.0"
    assert report["meta"]["asr_device"] == ""


def test_export_missing_dub_track_raises(
    tiny_clip: Path, tmp_path: Path, monkeypatch, ffmpeg_available: bool
) -> None:
    monkeypatch.chdir(tmp_path)
    job_path, _artifacts, cfg = _make_job(tmp_path, tiny_clip, mode="dub", with_dub=False)

    import pytest

    with pytest.raises(FileNotFoundError):
        export(str(job_path), cfg)


# --------------------------------------------------------- subtitle QC (M4-T2)
_LONG_VI_LINE = (
    "Đây là một câu rất dài cần phải được tự động ngắt "
    "thành hai dòng cho đúng chuẩn."
)


def test_export_subtitle_qc_splits_wraps_and_reports(
    tiny_clip: Path, tmp_path: Path, monkeypatch, ffmpeg_available: bool
) -> None:
    monkeypatch.chdir(tmp_path)
    script = [
        {
            "start": 0.0, "end": 0.6, "speaker": "spk0", "en": "fast",
            "vi": "Câu một đây nhé. Câu hai đây nhé.", "truncated": False,
        },
        {
            "start": 1.0, "end": 7.0, "speaker": "spk0", "en": "long",
            "vi": _LONG_VI_LINE, "truncated": False,
        },
    ]
    job_path, artifacts_dir, cfg = _make_job(
        tmp_path, tiny_clip, mode="dub", script=script
    )
    assert cfg.qc_subtitles is True

    export(str(job_path), cfg)

    out_dir = tmp_path / "output" / tiny_clip.stem
    subs = pysubs2.load(str(out_dir / SUBTITLES_NAME))
    assert len(subs) == len(script)  # no futile high-CPS-only split
    assert subs[0].end <= subs[1].start
    # the long cue was wrapped (pysubs2 exposes the line break as "\N")
    assert any(("\\N" in event.text or "\n" in event.text) for event in subs)

    report = json.loads((out_dir / REPORT_NAME).read_text(encoding="utf-8"))
    block = report["subtitles_qc"]
    assert block["enabled"] is True
    assert block["cues_in"] == 2
    assert block["cues_out"] == len(subs)
    assert block["split"] == 0
    assert set(block["violations"]) == {"cps", "duration", "line_length", "lines"}
    assert block["cps_max_observed"] > 0

    # the machine-readable sidecar is written next to the other artifacts
    assert (artifacts_dir / "subtitles_qc.json").exists()

    # script.tsv is byte-identical to the pre-QC writer (QC only touches the SRT)
    expected_tsv = tmp_path / "expected.tsv"
    write_script_tsv(script, expected_tsv)
    assert (out_dir / SCRIPT_TSV_NAME).read_bytes() == expected_tsv.read_bytes()


def test_export_subtitle_qc_disabled_srt_is_unchanged(
    tiny_clip: Path, tmp_path: Path, monkeypatch, ffmpeg_available: bool
) -> None:
    monkeypatch.chdir(tmp_path)
    script = [
        {
            "start": 0.0, "end": 0.6, "speaker": "spk0", "en": "fast",
            "vi": "Câu một đây nhé. Câu hai đây nhé.", "truncated": False,
        },
    ]
    job_path, _artifacts, cfg = _make_job(
        tmp_path, tiny_clip, mode="dub", script=script
    )
    cfg.qc_subtitles = False

    export(str(job_path), cfg)

    out_dir = tmp_path / "output" / tiny_clip.stem
    expected = tmp_path / "expected.srt"
    write_subtitles_srt(script, expected)
    assert (out_dir / SUBTITLES_NAME).read_bytes() == expected.read_bytes()

    report = json.loads((out_dir / REPORT_NAME).read_text(encoding="utf-8"))
    assert report["subtitles_qc"]["enabled"] is False
    assert report["subtitles_qc"]["cues_in"] == 0


def test_cli_subtitle_qc_flag_names() -> None:
    from vidub.cli import _build_cfg, build_parser

    parser = build_parser()
    assert _build_cfg(parser.parse_args(["run", "x.mp4"])).qc_subtitles is True
    assert (
        _build_cfg(parser.parse_args(["run", "x.mp4", "--no-qc-subtitles"])).qc_subtitles
        is False
    )
    assert (
        _build_cfg(parser.parse_args(["run", "x.mp4", "--no-subtitle-qc"])).qc_subtitles
        is False
    )
    # batch shares the same parser
    assert (
        _build_cfg(parser.parse_args(["batch", "folder", "--no-subtitle-qc"])).qc_subtitles
        is False
    )


# --------------------------------------------------------- final QC gate (M4-T4)
def test_export_writes_quality_block(
    tiny_clip: Path, tmp_path: Path, monkeypatch, ffmpeg_available: bool
) -> None:
    monkeypatch.chdir(tmp_path)
    job_path, _artifacts, cfg = _make_job(tmp_path, tiny_clip, mode="dub")
    assert cfg.qc_final is True

    export(str(job_path), cfg)

    report = json.loads(
        (tmp_path / "output" / tiny_clip.stem / REPORT_NAME).read_text(encoding="utf-8")
    )
    quality = report["quality"]
    assert set(quality) >= {"enabled", "verdict", "levels", "audio", "review_packets"}
    assert quality["enabled"] is True
    assert quality["verdict"] in {"pass", "review", "fail", "unknown"}
    assert set(quality["levels"]) == {"transcript", "translation", "subtitles"}
    assert set(quality["audio"]) >= {
        "duration_delta",
        "loudness_i",
        "loudness_delta",
        "silent_rows",
        "clipped_rows",
    }


def test_export_no_qc_final_writes_disabled_block(
    tiny_clip: Path, tmp_path: Path, monkeypatch, ffmpeg_available: bool
) -> None:
    monkeypatch.chdir(tmp_path)
    job_path, _artifacts, cfg = _make_job(tmp_path, tiny_clip, mode="dub")
    cfg.qc_final = False

    export(str(job_path), cfg)

    report = json.loads(
        (tmp_path / "output" / tiny_clip.stem / REPORT_NAME).read_text(encoding="utf-8")
    )
    assert report["quality"] == {"enabled": False}


def test_cli_qc_final_flag_names() -> None:
    from vidub.cli import _build_cfg, build_parser

    parser = build_parser()
    assert _build_cfg(parser.parse_args(["run", "x.mp4"])).qc_final is True
    assert (
        _build_cfg(parser.parse_args(["run", "x.mp4", "--no-qc-final"])).qc_final
        is False
    )
    assert (
        _build_cfg(parser.parse_args(["batch", "folder", "--no-qc-final"])).qc_final
        is False
    )


# ------------------------------------------- SA-1-T4 sentence-unit subtitles
_UNIT_SCRIPT = [
    {"start": 0.0, "end": 1.0, "speaker": "spk0", "en": "part one", "vi": "phần một", "truncated": False},
    {"start": 1.0, "end": 2.0, "speaker": "spk0", "en": "part two", "vi": "phần hai", "truncated": False},
    {"start": 2.0, "end": 3.0, "speaker": "spk0", "en": "part three.", "vi": "phần ba.", "truncated": False},
]

_UNITS_FIXTURE = [
    {
        "index": 0, "indices": [0], "start": 0.0, "end": 1.0, "speaker": "spk0",
        "source": "part one", "vi": "Phần một.", "words_in": 2, "coverage": 1.0,
        "guard": "none", "battery": "ok", "open": False,
    },
    {
        "index": 1, "indices": [1, 2], "start": 1.0, "end": 3.0, "speaker": "spk0",
        "source": "part two part three.", "vi": "Phần hai phần ba.", "words_in": 4,
        "coverage": 1.0, "guard": "none", "battery": "ok", "open": False,
    },
]

# ``start``/``end`` are the PLACED unit times (they differ from units.json).
_TIMED_FIXTURE = [
    {
        "idx": 0, "members": [0], "file": "speech/unit_0000.wav", "start": 0.5,
        "end": 1.4, "member_spans": [[0, 0.5, 1.4, "Phần một."]],
    },
    {
        "idx": 1, "members": [1, 2], "file": "speech/unit_0001.wav", "start": 1.4,
        "end": 3.6,
        "member_spans": [[1, 1.4, 2.2, "Phần hai"], [2, 2.2, 3.6, "phần ba."]],
    },
]


def _write_units(artifacts: Path, units: list[dict]) -> None:
    (artifacts / "units.json").write_text(
        json.dumps(
            {"version": 1, "mode": "sentence", "units": units}, ensure_ascii=False
        ),
        encoding="utf-8",
    )


def _write_timed(artifacts: Path, rows: list[dict]) -> None:
    (artifacts / "timed.json").write_text(
        json.dumps(rows, ensure_ascii=False), encoding="utf-8"
    )


def test_export_subtitles_one_cue_per_sentence_unit(
    tiny_clip: Path, tmp_path: Path, monkeypatch, ffmpeg_available: bool
) -> None:
    monkeypatch.chdir(tmp_path)
    job_path, artifacts, cfg = _make_job(tmp_path, tiny_clip, script=_UNIT_SCRIPT)
    _write_units(artifacts, _UNITS_FIXTURE)
    _write_timed(artifacts, _TIMED_FIXTURE)

    export(str(job_path), cfg)

    subs = pysubs2.load(str(tmp_path / "output" / tiny_clip.stem / SUBTITLES_NAME))
    # ONE cue per sentence unit; text = the unit's ``vi``; times = the PLACED
    # timed.json row (not the locked units.json span).
    assert [event.text for event in subs] == ["Phần một.", "Phần hai phần ba."]
    assert [(event.start, event.end) for event in subs] == [(500, 1400), (1400, 3600)]

    report = json.loads(
        (tmp_path / "output" / tiny_clip.stem / REPORT_NAME).read_text(encoding="utf-8")
    )
    assert report["subtitles_qc"]["cues_in"] == 2  # one per unit, not per fragment


def test_export_subtitles_use_locked_unit_times_without_timed(
    tiny_clip: Path, tmp_path: Path, monkeypatch, ffmpeg_available: bool
) -> None:
    monkeypatch.chdir(tmp_path)
    job_path, artifacts, cfg = _make_job(tmp_path, tiny_clip, script=_UNIT_SCRIPT)
    _write_units(artifacts, _UNITS_FIXTURE)  # no timed.json -> locked spans

    export(str(job_path), cfg)

    subs = pysubs2.load(str(tmp_path / "output" / tiny_clip.stem / SUBTITLES_NAME))
    assert [(event.start, event.end) for event in subs] == [(0, 1000), (1000, 3000)]


def test_export_subtitles_legacy_without_units_json_is_byte_identical(
    tiny_clip: Path, tmp_path: Path, monkeypatch, ffmpeg_available: bool
) -> None:
    """No ``units.json`` -> today's script-row path, byte-identically."""
    monkeypatch.chdir(tmp_path)
    job_path, artifacts, cfg = _make_job(tmp_path, tiny_clip, script=_UNIT_SCRIPT)
    # Even a member_spans-bearing timed.json must not change the SRT without units.
    _write_timed(artifacts, _TIMED_FIXTURE)
    cfg.qc_subtitles = False

    export(str(job_path), cfg)

    expected = tmp_path / "expected.srt"
    write_subtitles_srt(_UNIT_SCRIPT, expected)
    produced = tmp_path / "output" / tiny_clip.stem / SUBTITLES_NAME
    assert produced.read_bytes() == expected.read_bytes()


def test_export_subtitle_qc_still_splits_a_long_unit_cue(
    tiny_clip: Path, tmp_path: Path, monkeypatch, ffmpeg_available: bool
) -> None:
    monkeypatch.chdir(tmp_path)
    job_path, artifacts, cfg = _make_job(tmp_path, tiny_clip, script=_UNIT_SCRIPT)
    _write_units(
        artifacts,
        [
            {
                "index": 0, "indices": [0, 1, 2], "start": 0.0, "end": 0.6,
                "speaker": "spk0", "source": "one two three",
                "vi": "Câu một đây nhé. Câu hai đây nhé.", "words_in": 3,
                "coverage": 1.0, "guard": "none", "battery": "ok", "open": False,
            }
        ],
    )
    _write_timed(
        artifacts,
        [
            {
                "idx": 0, "members": [0, 1, 2], "file": "speech/unit_0000.wav",
                "start": 0.0, "end": 0.6,
                "member_spans": [[0, 0.0, 0.6, "Câu một đây nhé."]],
            }
        ],
    )
    assert cfg.qc_subtitles is True

    export(str(job_path), cfg)

    subs = pysubs2.load(str(tmp_path / "output" / tiny_clip.stem / SUBTITLES_NAME))
    assert len(subs) == 1  # adjacent silence supplies readable display time
    report = json.loads(
        (tmp_path / "output" / tiny_clip.stem / REPORT_NAME).read_text(encoding="utf-8")
    )
    assert report["subtitles_qc"]["cues_in"] == 1
    assert report["subtitles_qc"]["split"] == 0
    assert report["subtitles_qc"]["violations"]["cps"] == 0


# Q-1-T4: cue boundaries land on the ASR word edges when timed.json carries them.
_WORD_TEXT_LONG = " ".join(["aaaaaaaa"] * 10)
_WORD_TIMES_LONG = [
    [0.0, 0.4, "w0"],
    [0.4, 0.8, "w1"],
    [0.8, 1.2, "w2"],
    [1.2, 1.6, "w3"],
    [1.6, 2.0, "w4"],
    [3.0, 3.4, "w5"],  # an uneven pause before word 5
    [3.4, 3.8, "w6"],
    [3.8, 4.2, "w7"],
    [4.2, 4.6, "w8"],
    [4.6, 5.0, "w9"],
]


def test_export_split_long_unit_cue_on_word_edges(
    tiny_clip: Path, tmp_path: Path, monkeypatch, ffmpeg_available: bool
) -> None:
    monkeypatch.chdir(tmp_path)
    script = [
        {
            "start": 0.0, "end": 5.0, "speaker": "spk0", "en": "x",
            "vi": _WORD_TEXT_LONG, "truncated": False,
        }
    ]
    job_path, artifacts, cfg = _make_job(tmp_path, tiny_clip, script=script)
    _write_units(
        artifacts,
        [
            {
                "index": 0, "indices": [0], "start": 0.0, "end": 5.0,
                "speaker": "spk0", "source": "x", "vi": _WORD_TEXT_LONG,
                "words_in": 1, "coverage": 1.0, "guard": "none",
                "battery": "ok", "open": False,
            }
        ],
    )
    base_row = {
        "idx": 0, "members": [0], "file": "speech/unit_0000.wav",
        "start": 0.0, "end": 5.0, "member_spans": [[0, 0.0, 5.0, _WORD_TEXT_LONG]],
    }
    out_srt = tmp_path / "output" / tiny_clip.stem / SUBTITLES_NAME

    # (1) no word timeline -> today's proportional cut at 2.5 s
    _write_timed(artifacts, [base_row])
    export(str(job_path), cfg)
    subs = pysubs2.load(str(out_srt))
    assert len(subs) == 2
    assert subs[0].end == 2500

    # (2) ASR reference at 3.0 s; readable scheduling stays within lead/lag bounds.
    _write_timed(artifacts, [{**base_row, "align_words": _WORD_TIMES_LONG}])
    export(str(job_path), cfg)
    subs = pysubs2.load(str(out_srt))
    assert len(subs) == 2
    assert subs[0].start == 0 and 2750 <= subs[0].end <= 4500
    assert subs[1].start == subs[0].end and subs[1].end == 5000
