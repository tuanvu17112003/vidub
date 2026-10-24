import hashlib
from pathlib import Path

import pytest

from scripts.eval_asr_rework import _verify_audio_hashes, main, run_asr_cases


def test_run_asr_cases_keeps_windows_ids_and_unicode(tmp_path):
    audio_a = tmp_path / "a.wav"
    audio_b = tmp_path / "b.wav"
    audio_a.write_bytes(b"a")
    audio_b.write_bytes(b"b")
    cases = [
        {
            "id": "video-a:unit:2",
            "audio": str(audio_a),
            "audio_sha256": hashlib.sha256(b"a").hexdigest(),
            "start": 2.5,
            "end": 4.0,
        },
        {
            "id": "video-b:unit:2",
            "audio": str(audio_b),
            "audio_sha256": hashlib.sha256(b"b").hexdigest(),
            "start": 7.0,
            "end": 8.5,
        },
    ]
    seen = []

    def transcribe(audio: Path, start: float, end: float) -> str:
        seen.append((audio, start, end))
        return "Xin chào." if start == 2.5 else "Cảm ơn."

    rows = run_asr_cases(cases, transcribe, "fake-asr")

    assert [(row["id"], row["text"]) for row in rows] == [
        ("video-a:unit:2", "Xin chào."),
        ("video-b:unit:2", "Cảm ơn."),
    ]
    assert [(start, end) for _audio, start, end in seen] == [(2.5, 4.0), (7.0, 8.5)]
    assert [audio for audio, _start, _end in seen] == [audio_a, audio_b]
    assert all(row["engine"] == "fake-asr" and row["elapsed_seconds"] >= 0 for row in rows)
    assert [row["audio_sha256"] for row in rows] == [
        hashlib.sha256(b"a").hexdigest(),
        hashlib.sha256(b"b").hexdigest(),
    ]


def test_run_asr_cases_rejects_blank_output_with_case_id(tmp_path):
    audio = tmp_path / "silent.wav"
    audio.write_bytes(b"audio")
    cases = [{"id": "video:unit:9", "audio": str(audio), "start": 0.0, "end": 1.0}]

    with pytest.raises(ValueError, match="video:unit:9"):
        run_asr_cases(cases, lambda *_: "  ", "fake-asr")


def test_run_asr_cases_rejects_invalid_windows_and_missing_audio(tmp_path):
    cases = [{"id": "video:unit:0", "audio": str(tmp_path / "missing.wav"), "start": 1.0, "end": 1.0}]

    with pytest.raises(ValueError, match="invalid time window"):
        run_asr_cases(cases, lambda *_: "text", "fake-asr")

    cases[0]["end"] = 2.0
    with pytest.raises(FileNotFoundError, match="video:unit:0"):
        run_asr_cases(cases, lambda *_: "text", "fake-asr")


def test_run_asr_cases_rejects_non_finite_windows_and_duplicate_ids(tmp_path):
    audio = tmp_path / "source.wav"
    audio.write_bytes(b"audio")
    invalid = [{"id": "video:unit:0", "audio": str(audio), "start": float("nan"), "end": 2.0}]
    with pytest.raises(ValueError, match="invalid time window"):
        run_asr_cases(invalid, lambda *_: "text", "fake-asr")

    duplicate = [
        {"id": "video:unit:0", "audio": str(audio), "start": 0.0, "end": 1.0},
        {"id": "video:unit:0", "audio": str(audio), "start": 1.0, "end": 2.0},
    ]
    with pytest.raises(ValueError, match="duplicate case ID"):
        run_asr_cases(duplicate, lambda *_: "text", "fake-asr")


def test_audio_hash_verification_catches_changed_source(tmp_path):
    audio = tmp_path / "source.wav"
    audio.write_bytes(b"current audio")
    cases = [
        {
            "id": "video:unit:3",
            "audio": str(audio),
            "audio_sha256": "0" * 64,
        }
    ]

    with pytest.raises(ValueError, match="video:unit:3"):
        _verify_audio_hashes(cases)


def test_audio_hash_verification_accepts_the_frozen_source_hash(tmp_path):
    audio = tmp_path / "source.wav"
    audio.write_bytes(b"audio")
    cases = [
        {
            "id": "video:unit:4",
            "audio": str(audio),
            "audio_sha256": hashlib.sha256(b"audio").hexdigest(),
        }
    ]

    _verify_audio_hashes(cases)


@pytest.mark.parametrize("audio_sha256", [None, "not-a-sha256"])
def test_audio_hash_verification_requires_a_well_formed_hash(tmp_path, audio_sha256):
    audio = tmp_path / "source.wav"
    audio.write_bytes(b"audio")
    case = {"id": "video:unit:4", "audio": str(audio)}
    if audio_sha256 is not None:
        case["audio_sha256"] = audio_sha256

    with pytest.raises(ValueError, match="audio_sha256"):
        _verify_audio_hashes([case])


def test_cli_refuses_existing_output_before_loading_model(tmp_path, capsys):
    cases = tmp_path / "cases.jsonl"
    cases.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "candidate.jsonl"
    output.write_text("previous result\n", encoding="utf-8")

    status = main(
        [
            "--cases",
            str(cases),
            "--engine",
            "qwen3",
            "--model",
            "not-loaded",
            "--out",
            str(output),
        ]
    )

    assert status == 1
    assert "exists" in capsys.readouterr().err.lower()
    assert output.read_text(encoding="utf-8") == "previous result\n"
