"""Prepare immutable sentence-unit cases for ViDub quality comparisons."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

__all__ = [
    "collect_cases",
    "write_cases",
    "apply_verified_sources",
    "make_review_sheet",
    "summarize",
    "main",
]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_cases(job_paths: list[Path], holdout_paths: set[Path]) -> list[dict]:
    """Read existing jobs into stable, non-mutating evaluation cases."""
    holdouts = {path.resolve() for path in holdout_paths}
    cases: list[dict] = []
    seen_ids: set[str] = set()

    for job_path in job_paths:
        job_path = Path(job_path).resolve()
        job = _read_json(job_path)
        src_lang = str(job.get("src_lang", "")).strip().lower()
        tgt_lang = str(job.get("tgt_lang", "")).strip().lower()
        if (src_lang, tgt_lang) != ("en", "vi"):
            raise ValueError(
                f"only en-to-vi cases are supported; got {src_lang!r}-to-"
                f"{tgt_lang!r} in {job_path}"
            )

        video_hash = str(job.get("video_hash", "")).strip()
        if not video_hash:
            raise ValueError(f"job has no video_hash: {job_path}")
        paths = job.get("paths")
        if not isinstance(paths, dict):
            raise ValueError(f"job has no paths object: {job_path}")

        artifacts_dir = job_path.parent
        audio_path = (artifacts_dir / str(paths.get("audio", "audio.wav"))).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(f"audio file missing for {job_path}: {audio_path}")
        units_path = (artifacts_dir / str(paths.get("units", "units.json"))).resolve()
        units_doc = _read_json(units_path)
        units = units_doc.get("units")
        if not isinstance(units, list):
            raise ValueError(f"units must be a list in {units_path}")
        audio_hash = _sha256(audio_path)
        split = "holdout" if job_path in holdouts else "development"

        for position, unit in enumerate(units):
            if not isinstance(unit, dict):
                raise ValueError(f"unit {position} is not an object in {units_path}")
            try:
                index = int(unit.get("index", position))
                start = float(unit["start"])
                end = float(unit["end"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"unit {position} has invalid index or span in {units_path}"
                ) from exc
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
                raise ValueError(f"unit {index} has invalid span in {units_path}")

            case_id = f"{video_hash}:unit:{index}"
            if case_id in seen_ids:
                raise ValueError(f"duplicate case ID {case_id!r} from {job_path}")
            seen_ids.add(case_id)
            cases.append(
                {
                    "id": case_id,
                    "split": split,
                    "audio": str(audio_path),
                    "audio_sha256": audio_hash,
                    "start": start,
                    "end": end,
                    "speaker": str(unit.get("speaker") or "spk0"),
                    "source_asr": str(unit.get("source", "")),
                    "baseline_vi": str(unit.get("vi", "")),
                }
            )

    if not cases:
        raise ValueError("no sentence units found in the supplied jobs")
    return cases


def write_cases(cases: list[dict], output: Path) -> None:
    """Write UTF-8 JSONL without replacing a previous benchmark input."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as destination:
        for case in cases:
            destination.write(json.dumps(case, ensure_ascii=False) + "\n")


def apply_verified_sources(cases: list[dict], corrections: dict[str, str]) -> list[dict]:
    """Copy cases and attach reviewed source text by stable case ID."""
    ids = {str(case.get("id", "")) for case in cases}
    unknown = sorted(set(corrections) - ids)
    if unknown:
        raise ValueError(f"unknown case ID(s): {', '.join(unknown)}")
    cleaned: dict[str, str] = {}
    for case_id, text in corrections.items():
        value = str(text).strip()
        if not value:
            raise ValueError(f"verified_source is blank for {case_id!r}")
        cleaned[case_id] = value
    return [
        {**case, "verified_source": cleaned[case["id"]]}
        if case["id"] in cleaned
        else dict(case)
        for case in cases
    ]


def _read_cases(path: Path) -> list[dict]:
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read cases from {path}: {exc}") from exc
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"expected one or more JSON objects in {path}")
    return rows


def _read_corrections(path: Path) -> dict[str, str]:
    corrections: dict[str, str] = {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source, delimiter="\t")
            if not {"id", "verified_source"}.issubset(reader.fieldnames or []):
                raise ValueError("corrections TSV requires id and verified_source columns")
            for row in reader:
                case_id = str(row.get("id") or "").strip()
                if not case_id:
                    raise ValueError("corrections TSV contains a blank case ID")
                if case_id in corrections:
                    raise ValueError(f"duplicate case ID in corrections TSV: {case_id}")
                corrections[case_id] = str(row.get("verified_source") or "")
    except OSError as exc:
        raise ValueError(f"could not read corrections TSV {path}: {exc}") from exc
    return corrections


def _candidate_maps(cases: list[dict], candidates: list[list[dict]]) -> tuple[list[str], list[dict[str, dict]]]:
    case_ids = [str(case.get("id", "")).strip() for case in cases]
    if not case_ids or any(not case_id for case_id in case_ids):
        raise ValueError("cases must have non-blank IDs")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("duplicate case ID in cases")
    if not candidates:
        raise ValueError("at least one candidate output is required")

    by_candidate: list[dict[str, dict]] = []
    names: list[str] = []
    expected = set(case_ids)
    for index, rows in enumerate(candidates):
        mapping: dict[str, dict] = {}
        engines: set[str] = set()
        for row in rows:
            case_id = str(row.get("id", "")).strip()
            if not case_id or case_id in mapping:
                raise ValueError(f"candidate {index + 1} has blank or duplicate case ID: {case_id!r}")
            if not str(row.get("text") or "").strip():
                raise ValueError(f"candidate text is blank for case {case_id}")
            mapping[case_id] = row
            engine = str(row.get("engine", "")).strip()
            if not engine:
                raise ValueError(f"candidate {index + 1} has no engine name for case {case_id}")
            engines.add(engine)
        if set(mapping) != expected:
            missing = sorted(expected - set(mapping))
            extra = sorted(set(mapping) - expected)
            raise ValueError(f"candidate {index + 1} ID mismatch; missing={missing}, extra={extra}")
        if len(engines) > 1:
            raise ValueError(f"candidate {index + 1} contains mixed engine names: {sorted(engines)}")
        name = next(iter(engines), f"candidate-{index + 1}")
        if name in names:
            raise ValueError(f"candidate engine name is duplicated: {name}")
        names.append(name)
        by_candidate.append(mapping)
    return case_ids, by_candidate


def _candidate_engines(case_ids: list[str], by_candidate: list[dict[str, dict]]) -> list[str]:
    return [str(mapping[case_ids[0]]["engine"]) for mapping in by_candidate]


def _validate_candidate_sources(
    cases: list[dict], case_ids: list[str], by_candidate: list[dict[str, dict]]
) -> None:
    case_by_id = {str(case["id"]): case for case in cases}
    for mapping in by_candidate:
        engine = str(mapping[case_ids[0]]["engine"])
        if engine in {"faster-whisper", "qwen3"}:
            for case_id in case_ids:
                expected = str(case_by_id[case_id].get("audio_sha256") or "").lower()
                actual = str(mapping[case_id].get("audio_sha256") or "").lower()
                if (
                    not re.fullmatch(r"[0-9a-f]{64}", expected)
                    or not re.fullmatch(r"[0-9a-f]{64}", actual)
                    or actual != expected
                ):
                    raise ValueError(
                        f"audio hash mismatch for candidate {engine!r}, case {case_id}"
                    )
            continue
        for case_id in case_ids:
            case = case_by_id[case_id]
            source = (
                case.get("verified_source")
                if "verified_source" in case
                else case.get("source_asr")
            )
            if not isinstance(source, str) or not source.strip():
                raise ValueError(f"case {case_id} has no translation input text")
            expected = hashlib.sha256(source.encode("utf-8")).hexdigest()
            actual = str(mapping[case_id].get("source_sha256") or "").lower()
            if not re.fullmatch(r"[0-9a-f]{64}", actual) or actual != expected:
                raise ValueError(
                    f"source hash mismatch for candidate {engine!r}, case {case_id}"
                )


def _blinded_label(index: int) -> str:
    """Stable lower-case letter label for a small candidate set."""
    result = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(ord("a") + remainder) + result
    return f"candidate_{result}"


def _key_path(output: Path) -> Path:
    return Path(output).with_suffix(".key.json")


def make_review_sheet(
    cases: list[dict], candidates: list[list[dict]], seed: int, output: Path
) -> None:
    """Write blinded, per-case candidate columns and a separate private key."""
    case_ids, by_candidate = _candidate_maps(cases, candidates)
    _validate_candidate_sources(cases, case_ids, by_candidate)
    output = Path(output)
    key_path = _key_path(output)
    if output.exists() or key_path.exists():
        raise FileExistsError(f"review sheet or private key already exists: {output} / {key_path}")
    output.parent.mkdir(parents=True, exist_ok=True)

    labels = [_blinded_label(index) for index in range(len(candidates))]
    fields = ["id", "split", "source_text", "source_status", "difficulty_reason", "reviewer"]
    for label in labels:
        fields.extend(
            [
                f"{label}_text",
                f"{label}_decision",
                f"{label}_severity",
                f"{label}_error_type",
                f"{label}_notes",
            ]
        )

    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    key: dict[str, dict[str, dict[str, str]]] = {}
    shuffled_cases = list(zip(case_ids, cases))
    rng.shuffle(shuffled_cases)
    for case_id, case in shuffled_cases:
        source = case.get("verified_source") or case.get("source_asr") or ""
        source_status = "verified_source" if case.get("verified_source") else "source_asr_unverified"
        row = {
            "id": case_id,
            "split": str(case.get("split") or "development"),
            "source_text": str(source),
            "source_status": source_status,
            "difficulty_reason": "",
            "reviewer": "",
        }
        candidate_order = list(range(len(candidates)))
        rng.shuffle(candidate_order)
        key[case_id] = {}
        for slot, candidate_index in enumerate(candidate_order):
            label = labels[slot]
            candidate = by_candidate[candidate_index][case_id]
            text = str(candidate.get("text") or "")
            if not text.strip():
                raise ValueError(f"candidate text is blank for case {case_id}")
            row[f"{label}_text"] = text
            row[f"{label}_decision"] = ""
            row[f"{label}_severity"] = ""
            row[f"{label}_error_type"] = ""
            row[f"{label}_notes"] = ""
            key[case_id][label] = {
                "engine": str(candidate["engine"]),
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "source_sha256": str(candidate.get("source_sha256") or ""),
            }
        rows.append(row)

    with output.open("x", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _write_private_key({"seed": seed, "by_case": key}, key_path)


def _write_private_key(value: dict, output: Path) -> None:
    with output.open("x", encoding="utf-8", newline="\n") as destination:
        json.dump(value, destination, ensure_ascii=False, indent=2)
        destination.write("\n")


def _write_json(value: dict, output: Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as destination:
        json.dump(value, destination, ensure_ascii=False, indent=2)
        destination.write("\n")


def _normalize_metric_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(
        "".join(char for char in text if not unicodedata.category(char).startswith("P")).split()
    )


def _edit_distance(left: list[str] | str, right: list[str] | str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, left_value in enumerate(left, start=1):
        current = [row]
        for column, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[column - 1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def _error_rates(references: list[str], hypotheses: list[str]) -> tuple[float, float, str]:
    try:
        from jiwer import cer, wer

        return float(wer(references, hypotheses)), float(cer(references, hypotheses)), "jiwer"
    except ImportError:
        word_edits = word_count = char_edits = char_count = 0
        for reference, hypothesis in zip(references, hypotheses):
            ref = _normalize_metric_text(reference)
            hyp = _normalize_metric_text(hypothesis)
            ref_words, hyp_words = ref.split(), hyp.split()
            word_edits += _edit_distance(ref_words, hyp_words)
            word_count += len(ref_words)
            char_edits += _edit_distance(ref, hyp)
            char_count += len(ref)
        return (
            word_edits / max(word_count, 1),
            char_edits / max(char_count, 1),
            "stdlib-normalized",
        )


def summarize(
    cases: list[dict],
    candidates: list[list[dict]],
    judgments: list[dict],
    *,
    run_metadata: list[dict] | None = None,
) -> dict:
    """Summarize split-aware machine metrics and human review completion."""
    case_ids, by_candidate = _candidate_maps(cases, candidates)
    _validate_candidate_sources(cases, case_ids, by_candidate)
    if run_metadata is not None and len(run_metadata) != len(candidates):
        raise ValueError("run metadata count must match candidate count")
    case_by_id = {str(case["id"]): case for case in cases}
    engines = _candidate_engines(case_ids, by_candidate)
    case_engines = set(engines)
    judgment_by_pair: dict[tuple[str, str], dict] = {}
    for judgment in judgments:
        case_id = str(judgment.get("id", "")).strip()
        engine = str(judgment.get("engine", "")).strip()
        if case_id not in case_by_id or engine not in case_engines:
            raise ValueError(f"judgment references unknown case or candidate: {case_id!r}/{engine!r}")
        pair = (case_id, engine)
        if pair in judgment_by_pair:
            raise ValueError(f"duplicate judgment for case/candidate: {case_id}/{engine}")
        judgment_by_pair[pair] = judgment

    split_counts = {
        split: sum(str(case.get("split") or "development") == split for case in cases)
        for split in ("development", "holdout")
    }
    source_counts = {
        split: sum(
            str(case.get("split") or "development") == split
            and bool(str(case.get("verified_source") or "").strip())
            for case in cases
        )
        for split in ("development", "holdout")
    }

    candidate_summaries: list[dict] = []
    candidate_provenance: list[dict] = []
    asr_metrics: list[dict] = []
    for index, (engine, rows_by_id) in enumerate(zip(engines, by_candidate)):
        rows = [rows_by_id[case_id] for case_id in case_ids]
        metadata = (run_metadata or [{}] * len(candidates))[index]
        model = str(metadata.get("model") or "").strip()
        checkpoint_revision = str(metadata.get("checkpoint_revision") or "").strip()
        provenance_complete = bool(model and checkpoint_revision)
        candidate_provenance.append(
            {
                "engine": engine,
                "model": model or None,
                "checkpoint_revision": checkpoint_revision or None,
                "complete": provenance_complete,
                "run_metadata": metadata,
            }
        )
        per_split: dict[str, dict] = {}
        for split in ("development", "holdout"):
            selected = [
                rows_by_id[case_id]
                for case_id in case_ids
                if str(case_by_id[case_id].get("split") or "development") == split
            ]
            per_split[split] = {"case_count": len(selected)}
            if engine in {"faster-whisper", "qwen3"}:
                pairs = [
                    (case_by_id[row["id"]].get("verified_source"), row.get("text"))
                    for row in selected
                    if str(case_by_id[row["id"]].get("verified_source") or "").strip()
                ]
                if pairs:
                    wer_value, cer_value, metric_tool = _error_rates(
                        [str(reference) for reference, _hypothesis in pairs],
                        [str(hypothesis or "") for _reference, hypothesis in pairs],
                    )
                    asr_metrics.append(
                        {
                            "engine": engine,
                            "split": split,
                            "case_count": len(pairs),
                            "wer": wer_value,
                            "cer": cer_value,
                            "metric_tool": metric_tool,
                        }
                    )
                    per_split[split].update({"asr_metric_case_count": len(pairs)})
                else:
                    per_split[split]["asr_metric_case_count"] = 0

        elapsed_values = [
            float(row["elapsed_seconds"])
            for row in rows
            if row.get("elapsed_seconds") is not None
        ]
        measured_elapsed = sum(elapsed_values) if elapsed_values else None
        peak_memory = metadata.get("peak_gpu_memory_mib")
        if peak_memory is None:
            memory_values = [row.get("peak_gpu_memory_mib") for row in rows]
            available = [int(value) for value in memory_values if value is not None]
            peak_memory = max(available) if available else None
        candidate_summaries.append(
            {
                "engine": engine,
                "model": model or None,
                "checkpoint_revision": checkpoint_revision or None,
                "provenance_complete": provenance_complete,
                "case_count": len(rows),
                "by_split": per_split,
                "elapsed_seconds": metadata.get("elapsed_seconds", measured_elapsed),
                "mean_case_seconds": measured_elapsed / len(rows) if rows and measured_elapsed is not None else None,
                "peak_gpu_memory_mib": peak_memory,
            }
        )

    def empty_judgment_counts() -> dict:
        return {
            "pass_count": 0,
            "fail_count": 0,
            "other_count": 0,
            "unreviewed_count": 0,
            "unattributed_count": 0,
            "incomplete_severity_count": 0,
            "severity_counts": Counter(),
            "error_type_counts": Counter(),
            "reviewers": set(),
        }

    judgment_summaries = {
        engine: {
            "total": empty_judgment_counts(),
            "by_split": {
                split: empty_judgment_counts()
                for split in ("development", "holdout")
            },
        }
        for engine in engines
    }
    difficult_ids: set[str] = set()
    unattributed_difficulty_count = 0
    unresolved_ids: set[str] = set()
    for case_id in case_ids:
        case = case_by_id[case_id]
        split = str(case.get("split") or "development")
        if split not in {"development", "holdout"}:
            raise ValueError(f"invalid split for case {case_id}: {case.get('split')!r}")
        for engine in engines:
            judgment = judgment_by_pair.get((case_id, engine))
            decision = str((judgment or {}).get("decision") or "").strip().lower()
            reviewer = str((judgment or {}).get("reviewer") or "").strip()
            severity = str((judgment or {}).get("severity") or "").strip().lower()
            severity_valid = severity in {"none", "minor", "major", "critical"}
            buckets = (
                judgment_summaries[engine]["total"],
                judgment_summaries[engine]["by_split"][split],
            )
            if not decision:
                for bucket in buckets:
                    bucket["unreviewed_count"] += 1
                unresolved_ids.add(case_id)
            elif not reviewer:
                for bucket in buckets:
                    bucket["unattributed_count"] += 1
                unresolved_ids.add(case_id)
                if decision in {"pass", "fail"} and not severity_valid:
                    for bucket in buckets:
                        bucket["incomplete_severity_count"] += 1
            else:
                for bucket in buckets:
                    bucket["reviewers"].add(reviewer)
                    if decision in {"pass", "fail"} and not severity_valid:
                        bucket["incomplete_severity_count"] += 1
                        unresolved_ids.add(case_id)
                    elif decision == "pass":
                        bucket["pass_count"] += 1
                    elif decision == "fail":
                        bucket["fail_count"] += 1
                    else:
                        bucket["other_count"] += 1
                        unresolved_ids.add(case_id)

            if judgment:
                error_type = str(judgment.get("error_type") or "").strip().lower()
                difficulty = str(judgment.get("difficulty_reason") or "").strip()
                if reviewer and decision in {"pass", "fail"} and severity_valid:
                    for bucket in buckets:
                        bucket["severity_counts"][severity] += 1
                        if error_type:
                            bucket["error_type_counts"][error_type] += 1
                    if difficulty:
                        difficult_ids.add(case_id)
                elif difficulty:
                    unattributed_difficulty_count += 1

    def serialize_judgment_counts(counts: dict) -> dict:
        result = {
            key: value
            for key, value in counts.items()
            if key not in {"severity_counts", "error_type_counts", "reviewers"}
        }
        result["reviewer_count"] = len(counts["reviewers"])
        result["severity_counts"] = dict(sorted(counts["severity_counts"].items()))
        result["error_type_counts"] = dict(sorted(counts["error_type_counts"].items()))
        return result

    judgments_by_engine = {
        engine: {
            "total": serialize_judgment_counts(summary["total"]),
            "by_split": {
                split: serialize_judgment_counts(counts)
                for split, counts in summary["by_split"].items()
            },
        }
        for engine, summary in judgment_summaries.items()
    }
    judgment_totals = empty_judgment_counts()
    for summary in judgment_summaries.values():
        counts = summary["total"]
        for key in (
            "pass_count",
            "fail_count",
            "other_count",
            "unreviewed_count",
            "unattributed_count",
            "incomplete_severity_count",
        ):
            judgment_totals[key] += counts[key]
        judgment_totals["severity_counts"].update(counts["severity_counts"])
        judgment_totals["error_type_counts"].update(counts["error_type_counts"])
        judgment_totals["reviewers"].update(counts["reviewers"])
    judgments_summary = serialize_judgment_counts(judgment_totals)

    unverified_pairs = sum(
        1
        for engine, rows_by_id in zip(engines, by_candidate)
        for case_id in case_ids
        if rows_by_id[case_id].get("input_status") == "source_asr_unverified"
        or not str(case_by_id[case_id].get("verified_source") or "").strip()
    )
    reasons: list[str] = []
    holdout_present = split_counts["holdout"] > 0
    if not holdout_present:
        reasons.append("holdout split is missing")
    if len(difficult_ids) < 30:
        reasons.append("fewer than 30 difficult cases are tagged")
    if source_counts["holdout"] < split_counts["holdout"]:
        reasons.append("holdout cases lack verified source transcripts")
    if unverified_pairs:
        reasons.append("candidate outputs include unverified source transcripts")
    if unresolved_ids:
        reasons.append("human judgments are missing or unresolved")
    if judgments_summary["unattributed_count"]:
        reasons.append("completed judgments lack reviewer identity")
    if judgments_summary["incomplete_severity_count"]:
        reasons.append("completed judgments lack valid severity")
    if unattributed_difficulty_count:
        reasons.append("difficult-case tags lack reviewer-attributed decisions")
    if not judgments_summary["reviewer_count"]:
        reasons.append("reviewer identity is missing")
    if any(not item["complete"] for item in candidate_provenance):
        reasons.append("candidate model/checkpoint provenance is incomplete")
    critical_errors = judgments_summary["severity_counts"].get("critical", 0)
    if critical_errors:
        reasons.append("critical errors remain in the human judgments")
    promotion_eligible = not reasons
    return {
        "status": "ready_for_promotion_review" if promotion_eligible else "provisional",
        "promotion_eligible": promotion_eligible,
        "promoted": False,
        "provisional_reasons": reasons,
        "split_counts": split_counts,
        "verified_source_counts": source_counts,
        "holdout_present": holdout_present,
        "difficult_case_count": len(difficult_ids),
        "candidate_summaries": candidate_summaries,
        "candidate_provenance": candidate_provenance,
        "asr_error_rates": asr_metrics,
        "judgments": {**judgments_summary, "critical_error_count": critical_errors},
        "judgments_by_engine": judgments_by_engine,
        "unattributed_difficulty_tag_count": unattributed_difficulty_count,
        "unverified_source_pair_count": unverified_pairs,
        "unresolved_ids": sorted(unresolved_ids),
    }


def _read_candidates(path: Path) -> tuple[list[dict], dict]:
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read candidate output from {path}: {exc}") from exc
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"expected one or more JSON objects in {path}")
    metadata_path = path.with_suffix(".run.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"candidate run metadata not found: {metadata_path}")
    metadata = _read_json(metadata_path)
    engines = {str(row.get("engine") or "").strip() for row in rows}
    if len(engines) != 1 or not next(iter(engines), ""):
        raise ValueError(f"candidate output must contain exactly one engine: {path}")
    if str(metadata.get("engine") or "").strip() != next(iter(engines)):
        raise ValueError(f"candidate run metadata engine does not match output: {path}")
    return rows, metadata


def _read_review_judgments(
    path: Path, expected_ids: set[str], candidates: list[list[dict]]
) -> list[dict]:
    key_path = _key_path(path)
    if not key_path.is_file():
        raise FileNotFoundError(f"private review key not found: {key_path}")
    key = _read_json(key_path).get("by_case")
    if not isinstance(key, dict):
        raise ValueError(f"invalid private review key: {key_path}")
    candidate_by_engine = {
        str(rows[0]["engine"]): {str(row["id"]): row for row in rows}
        for rows in candidates
    }
    judgments: list[dict] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source, delimiter="\t")
            for row in reader:
                case_id = str(row.get("id") or "").strip()
                if case_id not in expected_ids:
                    raise ValueError(f"review sheet contains unknown case ID: {case_id!r}")
                labels = key.get(case_id)
                if not isinstance(labels, dict):
                    raise ValueError(f"private review key has no mapping for case {case_id}")
                for label, identity in labels.items():
                    if not isinstance(identity, dict):
                        raise ValueError(f"private review key has invalid candidate identity for {case_id}/{label}")
                    engine = str(identity.get("engine") or "")
                    candidate = candidate_by_engine.get(engine, {}).get(case_id)
                    if candidate is None:
                        raise ValueError(f"review key candidate not found for case {case_id}: {engine}")
                    text_hash = hashlib.sha256(
                        str(candidate.get("text") or "").encode("utf-8")
                    ).hexdigest()
                    if text_hash != identity.get("text_sha256"):
                        raise ValueError(f"review key candidate text changed for case {case_id}: {engine}")
                    if str(candidate.get("source_sha256") or "") != str(
                        identity.get("source_sha256") or ""
                    ):
                        raise ValueError(f"review key candidate source changed for case {case_id}: {engine}")
                    judgments.append(
                        {
                            "id": case_id,
                            "engine": engine,
                            "decision": row.get(f"{label}_decision", ""),
                            "severity": row.get(f"{label}_severity", ""),
                            "error_type": row.get(f"{label}_error_type", ""),
                            "notes": row.get(f"{label}_notes", ""),
                            "difficulty_reason": row.get("difficulty_reason", ""),
                            "reviewer": row.get("reviewer", ""),
                        }
                    )
    except OSError as exc:
        raise ValueError(f"could not read review judgments from {path}: {exc}") from exc
    return judgments


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="collect existing sentence units")
    prepare.add_argument("--job", action="append", type=Path, required=True)
    prepare.add_argument("--holdout-job", action="append", type=Path, default=[])
    prepare.add_argument("--out", type=Path, required=True)

    apply_source = commands.add_parser("apply-source", help="attach reviewed transcripts")
    apply_source.add_argument("--cases", type=Path, required=True)
    apply_source.add_argument("--corrections", type=Path, required=True)
    apply_source.add_argument("--out", type=Path, required=True)

    score = commands.add_parser("score", help="summarize matched candidates and human review")
    score.add_argument("--cases", type=Path, required=True)
    score.add_argument("--candidate", type=Path, action="append", required=True)
    score.add_argument("--judgments", type=Path, required=True)
    score.add_argument("--out", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            cases = collect_cases(args.job, set(args.holdout_job))
            write_cases(cases, args.out)
            count = len(cases)
        elif args.command == "apply-source":
            cases = apply_verified_sources(
                _read_cases(args.cases), _read_corrections(args.corrections)
            )
            write_cases(cases, args.out)
            count = len(cases)
        else:
            cases = _read_cases(args.cases)
            candidate_runs = [_read_candidates(path) for path in args.candidate]
            candidate_rows = [rows for rows, _metadata in candidate_runs]
            run_metadata = [metadata for _rows, metadata in candidate_runs]
            judgments = _read_review_judgments(
                args.judgments,
                {case["id"] for case in cases},
                candidate_rows,
            )
            comparison = summarize(cases, candidate_rows, judgments, run_metadata=run_metadata)
            _write_json(comparison, args.out)
            count = len(cases)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.command == "score":
        print(f"wrote {comparison['status']} comparison for {count} case(s): {args.out}")
    else:
        print(f"wrote {count} case(s): {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
