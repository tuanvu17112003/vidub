import json
import csv
import hashlib
from pathlib import Path

import pytest

from scripts.quality_benchmark import (
    apply_verified_sources,
    collect_cases,
    make_review_sheet,
    summarize,
    write_cases,
)


def _job(
    root: Path,
    video_hash: str,
    units: list[dict],
    *,
    src_lang: str = "en",
    tgt_lang: str = "vi",
    audio: bytes = b"audio bytes",
) -> Path:
    job_dir = root / video_hash
    job_dir.mkdir()
    (job_dir / "audio.wav").write_bytes(audio)
    (job_dir / "units.json").write_text(
        json.dumps({"version": 1, "mode": "sentence", "units": units}),
        encoding="utf-8",
    )
    job_path = job_dir / "job.json"
    job_path.write_text(
        json.dumps(
            {
                "video_hash": video_hash,
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "paths": {"audio": "audio.wav", "units": "units.json"},
            }
        ),
        encoding="utf-8",
    )
    return job_path


def _unit(index: int, source: str, vi: str) -> dict:
    return {
        "index": index,
        "start": float(index * 2),
        "end": float(index * 2 + 1),
        "speaker": "spk0",
        "source": source,
        "vi": vi,
    }


def _mt_row(case: dict, engine: str, text: str, *, source: str | None = None) -> dict:
    input_text = source
    if input_text is None:
        input_text = case.get("verified_source", case.get("source_asr", ""))
    return {
        "id": case["id"],
        "engine": engine,
        "text": text,
        "input_status": "verified_source" if case.get("verified_source") else "source_asr_unverified",
        "source_sha256": hashlib.sha256(str(input_text).encode("utf-8")).hexdigest(),
    }


def _write_run_metadata(path: Path, engine: str) -> None:
    path.with_suffix(".run.json").write_text(
        json.dumps(
            {
                "engine": engine,
                "model": f"{engine}-model",
                "checkpoint_revision": "test-revision",
            }
        ),
        encoding="utf-8",
    )


def test_collect_cases_uses_video_hash_and_preserves_unit_order_and_unicode(tmp_path):
    first = _job(
        tmp_path,
        "video-a",
        [_unit(0, "First source.", "Câu một."), _unit(1, "Second source.", "Câu hai.")],
    )
    second = _job(
        tmp_path,
        "video-b",
        [_unit(0, "Third source.", "Câu ba.")],
        audio=b"second audio",
    )

    cases = collect_cases([first, second], set())

    assert [case["id"] for case in cases] == [
        "video-a:unit:0",
        "video-a:unit:1",
        "video-b:unit:0",
    ]
    assert [case["source_asr"] for case in cases] == [
        "First source.",
        "Second source.",
        "Third source.",
    ]
    assert [case["split"] for case in cases] == [
        "development",
        "development",
        "development",
    ]
    assert cases[0]["baseline_vi"] == "Câu một."
    assert cases[0]["audio"] == str((tmp_path / "video-a" / "audio.wav").resolve())
    assert len({case["audio_sha256"] for case in cases}) == 2


def test_collect_cases_rejects_missing_audio_with_job_path(tmp_path):
    job = _job(tmp_path, "video-a", [_unit(0, "Source.", "Nguồn.")])
    (job.parent / "audio.wav").unlink()

    with pytest.raises(FileNotFoundError, match="video-a.*audio"):
        collect_cases([job], set())


def test_collect_cases_rejects_invalid_span_duplicate_id_and_non_en_vi(tmp_path):
    malformed = _unit(0, "Source.", "Nguồn.")
    malformed["end"] = malformed["start"]
    bad_span = _job(tmp_path, "bad-span", [malformed])
    with pytest.raises(ValueError, match="span"):
        collect_cases([bad_span], set())

    duplicate_a = _job(tmp_path, "same-hash", [_unit(0, "Source.", "Nguồn.")])
    duplicate_dir = tmp_path / "duplicate"
    duplicate_dir.mkdir()
    (duplicate_dir / "audio.wav").write_bytes((duplicate_a.parent / "audio.wav").read_bytes())
    (duplicate_dir / "units.json").write_text(
        (duplicate_a.parent / "units.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    duplicate_b = duplicate_dir / "job.json"
    duplicate_b.write_text(duplicate_a.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate case ID"):
        collect_cases([duplicate_a, duplicate_b], set())

    unsupported = _job(
        tmp_path,
        "unsupported",
        [_unit(0, "Source.", "Nguồn.")],
        src_lang="ja",
    )
    with pytest.raises(ValueError, match="en.*vi"):
        collect_cases([unsupported], set())


def test_apply_verified_sources_changes_only_matching_case_and_preserves_asr():
    cases = [
        {"id": "video:unit:0", "source_asr": "ASR text."},
        {"id": "video:unit:1", "source_asr": "Other ASR."},
    ]

    corrected = apply_verified_sources(
        cases, {"video:unit:0": "Verified text."}
    )

    assert corrected == [
        {
            "id": "video:unit:0",
            "source_asr": "ASR text.",
            "verified_source": "Verified text.",
        },
        {"id": "video:unit:1", "source_asr": "Other ASR."},
    ]
    assert cases[0] == {"id": "video:unit:0", "source_asr": "ASR text."}
    with pytest.raises(ValueError, match="unknown case ID"):
        apply_verified_sources(cases, {"missing:unit:9": "Text."})


def test_write_cases_writes_utf8_jsonl_once(tmp_path):
    out = tmp_path / "cases.jsonl"
    cases = [{"id": "video:unit:0", "baseline_vi": "Xin chào."}]

    write_cases(cases, out)

    assert json.loads(out.read_text(encoding="utf-8")) == cases[0]
    with pytest.raises(FileExistsError):
        write_cases(cases, out)


def test_review_sheet_is_deterministically_blinded_and_keeps_private_key(tmp_path):
    cases = [
        {"id": "a", "split": "development", "source_asr": "Hello.", "verified_source": "Hello."},
        {"id": "b", "split": "holdout", "source_asr": "Goodbye."},
    ]
    candidates = [
        [_mt_row(cases[0], "envit5", "Xin chào."), _mt_row(cases[1], "envit5", "Tạm biệt.")],
        [_mt_row(cases[0], "hy-mt2", "Chào bạn."), _mt_row(cases[1], "hy-mt2", "Chào.")],
    ]
    first = tmp_path / "review-a.tsv"
    second = tmp_path / "review-b.tsv"

    make_review_sheet(cases, candidates, seed=17, output=first)
    make_review_sheet(cases, candidates, seed=17, output=second)

    assert first.read_bytes() == second.read_bytes()
    rows = list(csv.DictReader(first.open(encoding="utf-8", newline=""), delimiter="\t"))
    assert {row["id"] for row in rows} == {"a", "b"}
    assert {row["split"] for row in rows} == {"development", "holdout"}
    assert "difficulty_reason" in rows[0]
    assert "envit5" not in first.read_text(encoding="utf-8")
    key = json.loads(first.with_suffix(".key.json").read_text(encoding="utf-8"))
    assert set(key["by_case"]) == {"a", "b"}
    assert {
        identity["engine"]
        for row in key["by_case"].values()
        for identity in row.values()
    } == {"envit5", "hy-mt2"}


def test_summarize_rejects_candidate_id_mismatch_and_blank_judgment_is_unreviewed():
    cases = [{"id": "a", "split": "development", "verified_source": "Hello."}]
    with pytest.raises(ValueError, match="ID mismatch"):
        summarize(cases, [[{"id": "different", "engine": "envit5", "text": "Xin chào."}]], [])

    candidates = [[_mt_row(cases[0], "envit5", "Xin chào.")]]
    report = summarize(cases, candidates, [{"id": "a", "engine": "envit5", "decision": ""}])
    assert report["judgments"]["unreviewed_count"] == 1
    assert report["judgments"]["pass_count"] == 0
    assert report["status"] == "provisional"
    assert report["unresolved_ids"] == ["a"]


def test_matched_mt_candidates_must_use_the_same_case_source_text():
    cases = [{"id": "a", "split": "development", "verified_source": "Hello."}]
    candidates = [
        [_mt_row(cases[0], "envit5", "Xin chào.")],
        [_mt_row(cases[0], "hy-mt2", "Chào.", source="Different source.")],
    ]

    with pytest.raises(ValueError, match="source hash mismatch"):
        summarize(cases, candidates, [])


def test_asr_candidate_must_match_frozen_audio_hash():
    cases = [
        {
            "id": "a",
            "split": "development",
            "verified_source": "Hello.",
            "audio_sha256": "a" * 64,
        }
    ]
    candidate = [{"id": "a", "engine": "qwen3", "text": "Hello.", "audio_sha256": "b" * 64}]

    with pytest.raises(ValueError, match="audio hash mismatch"):
        summarize(cases, [candidate], [])


def test_asr_candidate_requires_audio_hash_even_when_text_matches():
    cases = [
        {
            "id": "a",
            "split": "development",
            "verified_source": "Hello.",
            "audio_sha256": "a" * 64,
        }
    ]
    candidate = [{"id": "a", "engine": "qwen3", "text": "Hello."}]

    with pytest.raises(ValueError, match="audio hash mismatch"):
        summarize(cases, [candidate], [])


def test_summarize_uses_verified_source_and_keeps_holdout_out_of_development_metrics():
    cases = [
        {"id": "dev", "split": "development", "verified_source": "Hello world."},
        {"id": "test", "split": "holdout", "verified_source": "Goodbye world."},
    ]
    candidate = [
        {
            "id": "dev",
            "engine": "faster-whisper",
            "text": "Hello word.",
            "audio_sha256": "a" * 64,
        },
        {
            "id": "test",
            "engine": "faster-whisper",
            "text": "Goodbye world.",
            "audio_sha256": "b" * 64,
        },
    ]
    cases[0]["audio_sha256"] = "a" * 64
    cases[1]["audio_sha256"] = "b" * 64

    report = summarize(cases, [candidate], [])

    assert report["split_counts"] == {"development": 1, "holdout": 1}
    metrics = {(row["split"], row["engine"]): row for row in report["asr_error_rates"]}
    assert metrics[("development", "faster-whisper")]["case_count"] == 1
    assert metrics[("holdout", "faster-whisper")]["case_count"] == 1
    assert metrics[("development", "faster-whisper")]["wer"] > 0
    assert metrics[("holdout", "faster-whisper")]["wer"] == 0


def test_report_stays_provisional_without_holdout_and_thirty_difficult_cases():
    cases = [{"id": "one", "split": "development", "verified_source": "Source."}]
    candidate = [_mt_row(cases[0], "mt", "Dịch.")]

    report = summarize(
        cases,
        [candidate],
        [{"id": "one", "engine": "mt", "decision": "pass", "severity": "none", "difficulty_reason": "proper name", "reviewer": "reviewer-1"}],
    )

    assert report["status"] == "provisional"
    assert report["promotion_eligible"] is False
    assert report["holdout_present"] is False
    assert report["difficult_case_count"] == 1
    assert "holdout split is missing" in report["provisional_reasons"]
    assert "fewer than 30 difficult cases are tagged" in report["provisional_reasons"]


@pytest.mark.parametrize("severity", ["", "extreme"])
def test_pass_fail_with_missing_or_invalid_severity_remains_unresolved(severity):
    cases = [{"id": "one", "split": "holdout", "verified_source": "Source."}]
    candidate = [_mt_row(cases[0], "mt", "Dịch.")]

    report = summarize(
        cases,
        [candidate],
        [
            {
                "id": "one",
                "engine": "mt",
                "decision": "pass",
                "severity": severity,
                "reviewer": "reviewer-1",
                "difficulty_reason": "proper name",
            }
        ],
        run_metadata=[{"model": "model", "checkpoint_revision": "revision"}],
    )

    assert report["judgments"]["pass_count"] == 0
    assert report["judgments"]["incomplete_severity_count"] == 1
    assert report["difficult_case_count"] == 0
    assert report["unresolved_ids"] == ["one"]
    assert "completed judgments lack valid severity" in report["provisional_reasons"]


def test_incomplete_candidate_provenance_stays_provisional_and_is_reported():
    cases = [{"id": "one", "split": "development", "verified_source": "Source."}]
    candidate = [_mt_row(cases[0], "envit5", "Dịch.")]

    report = summarize(cases, [candidate], [])

    assert report["candidate_provenance"] == [
        {
            "engine": "envit5",
            "model": None,
            "checkpoint_revision": None,
            "complete": False,
            "run_metadata": {},
        }
    ]
    assert "candidate model/checkpoint provenance is incomplete" in report["provisional_reasons"]


def test_score_cli_reads_blinded_sheet_and_writes_provisional_report(tmp_path):
    from scripts.quality_benchmark import main

    cases = [
        {"id": "dev", "split": "development", "verified_source": "Hello."},
        {"id": "test", "split": "holdout", "verified_source": "Goodbye."},
    ]
    candidate_rows = [
        [
            _mt_row(cases[0], "envit5", "Xin chào."),
            _mt_row(cases[1], "envit5", "Tạm biệt."),
        ],
        [
            _mt_row(cases[0], "hy-mt2", "Chào bạn."),
            _mt_row(cases[1], "hy-mt2", "Chào tạm biệt."),
        ],
    ]
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in cases),
        encoding="utf-8",
    )
    candidate_paths = [tmp_path / "envit5.jsonl", tmp_path / "hy-mt2.jsonl"]
    for path, rows in zip(candidate_paths, candidate_rows):
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        _write_run_metadata(path, str(rows[0]["engine"]))
    review_path = tmp_path / "review.tsv"
    make_review_sheet(cases, candidate_rows, seed=5, output=review_path)
    with review_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        fieldnames = reader.fieldnames
        review_rows = list(reader)
    for row in review_rows:
        row["reviewer"] = "reviewer-1"
        row["difficulty_reason"] = "proper name" if row["id"] == "test" else ""
        for label in ("candidate_a", "candidate_b"):
            row[f"{label}_decision"] = "pass"
            row[f"{label}_severity"] = "none"
    with review_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(review_rows)

    output = tmp_path / "comparison.json"
    status = main(
        [
            "score",
            "--cases",
            str(cases_path),
            "--candidate",
            str(candidate_paths[0]),
            "--candidate",
            str(candidate_paths[1]),
            "--judgments",
            str(review_path),
            "--out",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["judgments"]["pass_count"] == 4
    assert report["difficult_case_count"] == 1
    assert report["promotion_eligible"] is False
    assert all(item["complete"] for item in report["candidate_provenance"])
    assert {item["engine"] for item in report["candidate_provenance"]} == {"envit5", "hy-mt2"}


def test_score_cli_rejects_candidate_text_changed_since_blind_review(tmp_path, capsys):
    from scripts.quality_benchmark import main

    cases = [{"id": "dev", "split": "development", "verified_source": "Hello."}]
    reviewed_rows = [[_mt_row(cases[0], "envit5", "Xin chào.")]]
    candidate_path = tmp_path / "candidate.jsonl"
    candidate_path.write_text(
        json.dumps(_mt_row(cases[0], "envit5", "Xin chào."), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_run_metadata(candidate_path, "envit5")
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(json.dumps(cases[0], ensure_ascii=False) + "\n", encoding="utf-8")
    review_path = tmp_path / "review.tsv"
    make_review_sheet(cases, reviewed_rows, seed=1, output=review_path)
    with review_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        fieldnames = reader.fieldnames
        rows = list(reader)
    rows[0]["reviewer"] = "reviewer-1"
    rows[0]["candidate_a_decision"] = "pass"
    with review_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    candidate_path.write_text(
        json.dumps(_mt_row(cases[0], "envit5", "Changed translation."), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    output = tmp_path / "comparison.json"
    status = main(
        [
            "score",
            "--cases",
            str(cases_path),
            "--candidate",
            str(candidate_path),
            "--judgments",
            str(review_path),
            "--out",
            str(output),
        ]
    )

    assert status == 1
    assert "candidate text changed" in capsys.readouterr().err.lower()
    assert not output.exists()


def test_score_cli_requires_candidate_run_metadata(tmp_path, capsys):
    from scripts.quality_benchmark import main

    cases = [{"id": "dev", "split": "development", "verified_source": "Hello."}]
    candidate = _mt_row(cases[0], "envit5", "Xin chào.")
    candidate_path = tmp_path / "candidate.jsonl"
    candidate_path.write_text(json.dumps(candidate) + "\n", encoding="utf-8")
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(json.dumps(cases[0]) + "\n", encoding="utf-8")
    review_path = tmp_path / "review.tsv"
    make_review_sheet(cases, [[candidate]], seed=1, output=review_path)

    status = main(
        [
            "score",
            "--cases",
            str(cases_path),
            "--candidate",
            str(candidate_path),
            "--judgments",
            str(review_path),
            "--out",
            str(tmp_path / "comparison.json"),
        ]
    )

    assert status == 1
    assert "run metadata not found" in capsys.readouterr().err.lower()


def test_candidate_run_metadata_engine_must_match_output(tmp_path):
    from scripts.quality_benchmark import _read_candidates

    case = {"id": "dev", "verified_source": "Hello."}
    candidate_path = tmp_path / "candidate.jsonl"
    candidate_path.write_text(
        json.dumps(_mt_row(case, "envit5", "Xin chào.")) + "\n",
        encoding="utf-8",
    )
    _write_run_metadata(candidate_path, "hy-mt2")

    with pytest.raises(ValueError, match="metadata engine does not match"):
        _read_candidates(candidate_path)


def test_judgment_summary_is_separated_by_engine_and_split():
    cases = [
        {"id": "dev", "split": "development", "verified_source": "Hello."},
        {"id": "test", "split": "holdout", "verified_source": "Goodbye."},
    ]
    candidates = [
        [_mt_row(cases[0], "envit5", "Xin chào."), _mt_row(cases[1], "envit5", "Tạm biệt.")],
        [_mt_row(cases[0], "hy-mt2", "Chào."), _mt_row(cases[1], "hy-mt2", "Tạm biệt.")],
    ]
    judgments = [
        {"id": "dev", "engine": "envit5", "decision": "pass", "severity": "none", "reviewer": "r1"},
        {"id": "test", "engine": "envit5", "decision": "fail", "severity": "major", "reviewer": "r1"},
        {"id": "dev", "engine": "hy-mt2", "decision": "fail", "severity": "minor", "reviewer": "r1"},
        {"id": "test", "engine": "hy-mt2", "decision": "pass", "severity": "none", "reviewer": "r1"},
    ]

    report = summarize(cases, candidates, judgments)

    assert report["judgments_by_engine"]["envit5"]["by_split"]["development"]["pass_count"] == 1
    assert report["judgments_by_engine"]["envit5"]["by_split"]["holdout"]["fail_count"] == 1
    assert report["judgments_by_engine"]["hy-mt2"]["by_split"]["development"]["fail_count"] == 1
    assert report["judgments_by_engine"]["hy-mt2"]["by_split"]["holdout"]["pass_count"] == 1


def test_unattributed_judgment_and_difficulty_tag_do_not_count_toward_gate():
    cases = [{"id": "a", "split": "holdout", "verified_source": "Hello."}]
    candidates = [[_mt_row(cases[0], "envit5", "Xin chào.")]]

    report = summarize(
        cases,
        candidates,
        [{"id": "a", "engine": "envit5", "decision": "pass", "difficulty_reason": "proper name"}],
    )

    assert report["judgments"]["pass_count"] == 0
    assert report["judgments"]["unattributed_count"] == 1
    assert report["difficult_case_count"] == 0
    assert "completed judgments lack reviewer identity" in report["provisional_reasons"]
