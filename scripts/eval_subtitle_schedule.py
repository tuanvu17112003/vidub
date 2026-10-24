"""Matched fixed-text/audio subtitle and placed-span QC audit of both exports."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vidub.config import JobConfig
from vidub.manifest import JobManifest
from vidub.stages.export import _unit_subtitle_rows, write_subtitles_srt, write_subtitles_ass
from vidub.utils.finalqc import check_audio, run_gate
from vidub.utils.pace_metrics import pace_metrics
from vidub.utils.subqc import measure, qc_rows, split_cue, wrap_text


def legacy(rows: list[dict]) -> dict:
    counts = {"cps": 0, "duration": 0, "line_length": 0, "lines": 0}
    count = 0
    for row in rows:
        text = wrap_text(row["vi"])
        parts = split_cue(text, row["start"], row["end"], row.get("align_words")) if measure(text, row["start"], row["end"])["cps"] > 17 else [(row["start"], row["end"], text)]
        for start, end, part in parts:
            count += 1
            # Reproduce the old strict floating-point comparisons as well as
            # its splitting policy; the new scheduler quantizes to SRT ms.
            metrics = measure(wrap_text(part), start, end)
            counts["cps"] += metrics["cps"] > 17
            counts["duration"] += metrics["duration"] < 5 / 6 or metrics["duration"] > 7
            counts["line_length"] += metrics["max_line_len"] > 42
            counts["lines"] += len(metrics["lines"]) > 2
    return {"cues_out": count, "violations": counts}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    results = {}
    for video, digest in [
        ("youtube-LPZh9BOjkQs", "e53bb23d1989ba4d70a3871d412f01327bfaa74e"),
        ("youtube-5sLYAQS9sWQ", "1a372a6d345535b258048bedfe488989ac45fed8"),
    ]:
        cache = ROOT / ".vidub-cache" / digest
        manifest = JobManifest.load(cache / "job.json")
        manifest.cache_dir = str(ROOT / ".vidub-cache")
        rows = _unit_subtitle_rows(manifest)
        assert rows is not None
        after, stats = qc_rows(rows, video_duration=manifest.duration)
        preserved = " ".join(" ".join(row["vi"].split()) for row in rows) == " ".join(" ".join(row["vi"].split()) for row in after)
        no_overlap = all(a["end"] <= b["start"] + 1e-6 for a, b in zip(after, after[1:]))
        before = legacy(rows)
        timed = json.loads((cache / "timed.json").read_text(encoding="utf-8"))
        script = json.loads((cache / "script.json").read_text(encoding="utf-8"))
        old_audio = check_audio(cache / "dub_track.wav", script, manifest.duration, -16)
        new_audio = run_gate(manifest, JobConfig())["audio"]
        result = {"scope": "fixed original text and audio; no retranslation", "before": before, "after": stats,
                  "text_preserved": preserved, "no_overlap": no_overlap,
                  "violation_reduction": 1 - sum(stats["violations"].values()) / max(sum(before["violations"].values()), 1),
                  "before_silent_rows": old_audio["silent_rows"], "after_audio": new_audio,
                  "combined_pacing": pace_metrics(timed)}
        destination = ROOT / "output/phase2-review" / video
        destination.mkdir(parents=True, exist_ok=True)
        write_subtitles_srt(after, destination / "subtitles.vi.srt")
        write_subtitles_ass(after, destination / "subtitles.vi.ass")
        (destination / "quality.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        results[video] = result
        print(video, "before", before["violations"], "after", stats["violations"],
              "reduction", round(result["violation_reduction"] * 100, 1),
              "preserved", preserved, "overlap", not no_overlap, "drift", stats["scheduling"],
              "silence", old_audio["silent_rows"], "->", new_audio["silent_rows"], flush=True)
    (ROOT / "eval-cache/subtitle-schedule.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
