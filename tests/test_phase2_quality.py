"""Content, timing bounds and missing-speech regressions for the new policy."""
from __future__ import annotations

import array
import json
from pathlib import Path
import random
import wave

import pytest

from vidub.config import JobConfig
from vidub.manifest import JobManifest
from vidub.stages.export import output_dir_for
from vidub.stages import timing
from vidub.utils import finalqc
from vidub.utils.glossary import parse_glossary, splice_source_terms
from vidub.utils.pace_metrics import pace_metrics
from vidub.utils.subqc import qc_rows
from vidub.utils.tqc import _write_packets


def test_alias_is_source_conditioned_and_does_not_translate_fragments() -> None:
    mapping, _ = parse_glossary("pre-training\ttiền huấn luyện\taliases=đào tạo trước\ntransformer\tTransformer\taliases=máy biến áp")

    def forbidden(_text: str) -> str:
        raise AssertionError("Complete wording should survive a local terminology fix")

    result, rebuilt, _ = splice_source_terms("we discuss pre-training.", "Chúng ta bàn về đào tạo trước.", mapping, forbidden)
    assert result == "Chúng ta bàn về tiền huấn luyện." and not rebuilt
    result, _, _ = splice_source_terms("we discuss electricity.", "Đây là máy biến áp.", mapping, forbidden)
    assert result == "Đây là máy biến áp."


def test_export_respects_explicit_destination_before_writing(tmp_path: Path) -> None:
    destination = tmp_path / "isolated"
    manifest = JobManifest(input_path="video.mp4", paths={"output_dir": str(destination)})
    assert output_dir_for(manifest) == destination


def test_high_cps_without_available_time_is_flagged_without_word_explosion() -> None:
    rows = [{"start": 0.0, "end": 0.5, "vi": "Đây là một câu vẫn cần được giữ nguyên đầy đủ."},
            {"start": 0.5, "end": 1.5, "vi": "Tiếp theo."}]
    out, stats = qc_rows(rows, video_duration=1.5)
    assert len(out) == 2
    assert stats["violations"]["cps"] > 0
    assert out[0]["end"] <= out[1]["start"]
    assert " ".join(out[0]["vi"].split()) == rows[0]["vi"]


def test_scheduler_preserves_text_and_bounds_over_random_valid_spans() -> None:
    rng = random.Random(73)
    rows = []
    start = 0.0
    for index in range(40):
        text = " ".join(rng.choice(["mô hình", "ngôn ngữ", "Transformer", "tham số", "huấn luyện"]) for _ in range(rng.randint(2, 18))) + "."
        end = start + rng.uniform(0.4, 11)
        rows.append({"start": start, "end": end, "vi": text, "id": index})
        start = end + rng.uniform(0, 2)
    out, stats = qc_rows(rows, video_duration=start)
    assert " ".join(" ".join(r["vi"].split()) for r in out) == " ".join(r["vi"] for r in rows)
    assert all(a["end"] <= b["start"] + 1e-6 for a, b in zip(out, out[1:]))
    assert all(r["start"] >= 0 and r["end"] > r["start"] for r in out)
    assert stats["scheduling"]["max_lead"] <= 0.25 + 1e-6
    assert stats["scheduling"]["max_lag"] <= 1.5 + 1e-6
    for row in rows:
        cues = [r for r in out if r["id"] == row["id"]]
        assert cues[0]["start"] >= row["start"] - 0.25 - 1e-6
        assert cues[-1]["end"] <= row["end"] + 1.5 + 1e-6


def test_combined_speed_is_not_just_atempo() -> None:
    data = pace_metrics([{"idx": 52, "start": 314.25, "slot_start": 314.25,
                         "length_scale": 0.894713, "stretch": 1.07291},
                        {"idx": 53, "start": 331.0, "slot_start": 329.0}])
    assert data["max_effective_speed"] == pytest.approx(1.199, abs=0.001)
    assert data["max_onset_shift"] == 2.0


def _wav(path: Path, samples: list[int]) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(1000)
        handle.writeframes(array.array("h", samples).tobytes())


def test_placed_qc_removes_false_silence_but_detects_missing_and_empty_speech(tmp_path: Path, monkeypatch) -> None:
    manifest = JobManifest(cache_dir=str(tmp_path), video_hash="test", duration=4)
    cache = manifest.artifacts_dir
    cache.mkdir()
    _wav(cache / "dub_track.wav", [0] * 2000 + [8000] * 2000)
    _wav(cache / "voice.wav", [8000] * 1000)
    _wav(cache / "empty.wav", [])
    _wav(cache / "silent.wav", [0] * 1000)
    (cache / "script.json").write_text(json.dumps([{"start": 0, "end": 1}]), encoding="utf-8")
    (cache / "timed.json").write_text(json.dumps([
        {"idx": 0, "start": 2, "end": 3, "file": "voice.wav"},
        {"idx": 1, "start": 3, "end": 4, "file": "missing.wav"},
        {"idx": 2, "start": 3, "end": 4, "file": "empty.wav"},
        {"idx": 3, "start": 3, "end": 4, "file": "silent.wav"},
    ]), encoding="utf-8")
    monkeypatch.setattr(finalqc, "measure_loudness", lambda path: -16.0)
    gate = finalqc.run_gate(manifest, JobConfig())
    assert gate["audio"]["row_basis"] == "placed"
    assert gate["audio"]["silent_rows"] == [3]
    assert gate["audio"]["missing_rows"] == [1]
    assert gate["audio"]["empty_rows"] == [2]
    assert gate["verdict"] != "pass"


def test_empty_timing_artifact_is_not_a_pass(tmp_path: Path, monkeypatch) -> None:
    manifest = JobManifest(cache_dir=str(tmp_path), video_hash="test", duration=1)
    cache = manifest.artifacts_dir
    cache.mkdir()
    _wav(cache / "dub_track.wav", [8000] * 1000)
    (cache / "script.json").write_text('[{"start":0,"end":1}]', encoding="utf-8")
    (cache / "timed.json").write_text('[]', encoding="utf-8")
    monkeypatch.setattr(finalqc, "measure_loudness", lambda path: -16.0)
    gate = finalqc.run_gate(manifest, JobConfig())
    assert gate["audio"]["unplaced_rows"] == [0]
    assert gate["verdict"] == "review"


def test_short_speech_wav_is_not_hidden_by_background_audio(tmp_path: Path, monkeypatch) -> None:
    _wav(tmp_path / "mix.wav", [8000] * 4000)
    _wav(tmp_path / "short.wav", [8000] * 1000)
    monkeypatch.setattr(finalqc, "measure_loudness", lambda path: -16.0)
    audio = finalqc.check_audio(tmp_path / "mix.wav", [{"idx": 7, "start": 1, "end": 3, "file": "short.wav"}],
                                4, -16, tmp_path)
    assert audio["short_rows"] == [7]


def test_resolved_generated_review_packets_are_cleared_but_notes_survive(tmp_path: Path) -> None:
    directory = tmp_path / "review/translation"
    directory.mkdir(parents=True)
    (directory / "seg-36.txt").write_text("old generated warning", encoding="utf-8")
    (directory / "notes.txt").write_text("reviewer notes", encoding="utf-8")
    _write_packets(tmp_path, [], [], {})
    assert not (directory / "seg-36.txt").exists()
    assert (directory / "notes.txt").read_text(encoding="utf-8") == "reviewer notes"


def test_tts_packets_follow_current_scored_audio(tmp_path: Path, monkeypatch) -> None:
    directory = tmp_path / "review/tts"
    directory.mkdir(parents=True)
    (directory / "seg-1.txt").write_text("stale", encoding="utf-8")
    (directory / "notes.txt").write_text("keep", encoding="utf-8")
    monkeypatch.setattr(timing.ttsqc, "score_wav", lambda path: 4.0)
    timing.run_tts_qc(tmp_path, [{"idx": 1, "file": "voice.wav"}], {}, JobConfig(), lambda *args: 1.0)
    assert not (directory / "seg-1.txt").exists()
    assert (directory / "notes.txt").exists()
