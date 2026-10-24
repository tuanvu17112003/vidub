"""Sequential batch processing for ViDub (``vidub batch <folder>``).

One video at a time (VRAM/memory safety), each with its own cache dir and job.
A single failure is collected and the batch continues; Ctrl-C writes a partial
summary and returns exit code 130. ``--resume-batch`` skips videos whose
previous ``batch-summary.json`` entry is already ``done``.
"""

from __future__ import annotations

import fnmatch
import json
import logging
from pathlib import Path

from vidub.config import JobConfig
from vidub.pipeline import run_pipeline

__all__ = ["SUMMARY_NAME", "run_batch", "summary_path", "load_summary", "write_summary"]

log = logging.getLogger(__name__)

SUMMARY_NAME = "batch-summary.json"
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_INTERRUPTED = 130


def summary_path(folder: str | Path, out_dir: str | Path | None = None) -> Path:
    """Location of ``batch-summary.json`` (``out_dir`` overrides the folder)."""
    return Path(out_dir or folder) / SUMMARY_NAME


def load_summary(path: str | Path) -> dict:
    """Load a summary, returning ``{}`` when missing or unreadable."""
    target = Path(path)
    if not target.exists():
        return {}
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def write_summary(path: str | Path, summary: dict) -> Path:
    """Write ``summary`` as pretty JSON, creating parent dirs, and return it."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return target


def _entry_from_report(name: str, report: dict) -> dict:
    return {
        "input": name,
        "status": "done",
        "duration": report.get("duration"),
        "stages": {
            key: value.get("status")
            for key, value in (report.get("stages") or {}).items()
        },
        "output": (report.get("outputs") or {}).get("dubbed"),
    }


def _build_summary(
    folder: Path, pattern: str, videos: list[dict], counts: dict[str, int]
) -> dict:
    return {
        "folder": str(folder),
        "pattern": pattern,
        "totals": {
            "total": len(videos),
            "done": counts["done"],
            "failed": counts["failed"],
            "skipped": counts["skipped"],
            "interrupted": counts["interrupted"],
        },
        "videos": videos,
    }


def run_batch(
    folder: str | Path,
    cfg: JobConfig,
    pattern: str = "*.mp4",
    out_dir: str | Path | None = None,
    resume_batch: bool = False,
    resume: bool = True,
    force: list[str] | None = None,
    burn: bool = False,
) -> tuple[dict, int]:
    """Process every matching video in ``folder``; return ``(summary, exit_code)``.

    Files that do not match ``pattern`` are recorded as ``skipped``. With
    ``resume_batch`` a video whose previous summary entry is ``done`` is not
    re-run. Any single video's failure is captured and the loop continues.
    """
    folder = Path(folder)
    target = summary_path(folder, out_dir)
    previous = load_summary(target) if resume_batch else {}
    prior = {
        entry.get("input"): entry
        for entry in previous.get("videos", [])
        if isinstance(entry, dict)
    }

    videos: list[dict] = []
    counts = {"done": 0, "failed": 0, "skipped": 0, "interrupted": 0}
    current: str | None = None

    try:
        for path in sorted((p for p in folder.iterdir() if p.is_file()), key=lambda p: p.name):
            name = path.name
            if name == SUMMARY_NAME:
                # never treat our own summary as an input video
                continue
            if not fnmatch.fnmatch(name, pattern):
                videos.append(
                    {
                        "input": name,
                        "status": "skipped",
                        "reason": f"does not match pattern {pattern!r}",
                    }
                )
                counts["skipped"] += 1
                continue

            if resume_batch and prior.get(name, {}).get("status") == "done":
                entry = dict(prior[name])
                entry["resumed"] = True
                videos.append(entry)
                counts["done"] += 1
                log.info("batch: skip %s (already done, --resume-batch)", name)
                continue

            current = name
            log.info("batch: processing %s", name)
            try:
                report = run_pipeline(
                    str(path), cfg, resume=resume, force=force, burn=burn
                )
            except Exception as exc:  # noqa: BLE001 - one failure must not stop the batch
                videos.append(
                    {
                        "input": name,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                counts["failed"] += 1
                log.warning("batch: %s failed: %s", name, exc)
                continue

            videos.append(_entry_from_report(name, report))
            counts["done"] += 1
            current = None
    except KeyboardInterrupt:
        counts["interrupted"] += 1
        videos.append(
            {"input": current or "<unknown>", "status": "interrupted"}
        )
        summary = _build_summary(folder, pattern, videos, counts)
        write_summary(target, summary)
        log.warning("batch interrupted; partial summary written to %s", target)
        return summary, EXIT_INTERRUPTED

    summary = _build_summary(folder, pattern, videos, counts)
    write_summary(target, summary)
    return summary, (EXIT_FAILED if counts["failed"] else EXIT_OK)
