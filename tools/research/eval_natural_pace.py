"""Matched Piper rate-policy experiment on fixed cached sentence translations.

Reads the source cache only. Writes fresh audio and scores under a new output
directory; refuses an existing output directory. No MT, ASR or new weights.
This isolates the rate policy; it is not an end-to-end dubbing benchmark.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from vidub.engines.tts.piper_tts import PiperTTS
from vidub.utils.fit import (
    RATE_AS_IS_TOL, RATE_SPEED_MIN, RATE_SPEED_MAX,
    available_window, pace_base_for, plan_rate_fit, effective_speed_cap,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.job.resolve()
    output = args.output.resolve()
    if source.parent == output or source.parent in output.parents:
        parser.error("output must be outside the source cache")
    output.mkdir(parents=True, exist_ok=False)
    job = json.loads(source.read_text(encoding="utf-8"))
    units = json.loads((source.parent / "units.json").read_text(encoding="utf-8"))["units"]
    tts = PiperTTS()
    natural: list[float] = []
    windows: list[float] = []
    for index, unit in enumerate(units):
        windows.append(available_window(
            unit["start"], unit["end"],
            units[index + 1]["start"] if index + 1 < len(units) else None,
            job["duration"] + 2.0,
        ))
        natural.append(tts.synth(unit["vi"], str(output / f"natural-{index:03}.wav")))
    raw = [min(max(w / d, 1 / RATE_SPEED_MAX), 1 / RATE_SPEED_MIN)
           for w, d in zip(windows, natural)]
    base = pace_base_for(raw)
    records: dict[str, list[dict]] = {"legacy": [], "natural_pace": []}
    for arm in records:
        for index, unit in enumerate(units):
            duration, window = natural[index], windows[index]
            if arm == "legacy":
                requested = window / duration / base
                scale = (1.0 if abs(requested - 1.0) <= RATE_AS_IS_TOL + 1e-9
                         else min(max(raw[index] / base, 1 / RATE_SPEED_MAX), 1 / RATE_SPEED_MIN))
            else:
                scale, _ = plan_rate_fit(duration, window)
            measured = (duration if scale == 1.0 else
                        tts.synth(unit["vi"], str(output / f"{arm}-{index:03}.wav"), length_scale=scale))
            records[arm].append({
                "index": index, "natural_seconds": duration, "window": window,
                "length_scale": scale, "measured_seconds": measured,
                "speed": 1 / scale,
                "overflow": measured > window + 0.02,
                "clip_risk": measured > window * effective_speed_cap(window) + 0.02,
            })
    scores = {}
    for arm, rows in records.items():
        speeds = [r["speed"] for r in rows]
        scores[arm] = {
            "units": len(rows), "slowed": sum(r["length_scale"] > 1.00001 for r in rows),
            "resynthesized": sum(r["length_scale"] != 1.0 for r in rows),
            "speed_min": min(speeds), "speed_max": max(speeds),
            "speed_cv": statistics.pstdev(speeds) / statistics.mean(speeds),
            "overflow": sum(r["overflow"] for r in rows),
            "clip_risk": sum(r["clip_risk"] for r in rows),
        }
    payload = {"experiment": "fixed translations, sentence-level rate policy only",
               "source_job": str(source), "legacy_baseline": base,
               "scores": scores, "rows": records}
    (output / "scores.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
