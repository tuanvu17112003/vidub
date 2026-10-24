"""Measured synthesis/stretch speed and placement displacement for reports."""
from __future__ import annotations

import math


def pace_metrics(rows: list[dict]) -> dict:
    detail: list[dict] = []
    for position, row in enumerate(rows):
        try:
            scale = float(row.get("length_scale", 1.0))
            stretch = float(row.get("stretch", 1.0))
            start = float(row["start"])
            slot = float(row.get("slot_start", start))
            speed = stretch / scale
            if scale <= 0 or stretch <= 0 or not all(math.isfinite(v) for v in (speed, start, slot)):
                continue
            detail.append({"idx": row.get("idx", position), "effective_speed": round(speed, 6),
                           "onset_shift": round(start - slot, 6)})
        except (ValueError, TypeError, KeyError, ZeroDivisionError):
            continue
    speeds = [row["effective_speed"] for row in detail]
    mean = sum(speeds) / len(speeds) if speeds else 1.0
    cv = (sum((value - mean) ** 2 for value in speeds) / len(speeds)) ** 0.5 / mean if speeds else 0.0
    return {"measured_rows": len(detail), "max_effective_speed": max(speeds, default=1.0),
            "speed_cv": round(cv, 6), "max_onset_shift": max((abs(row["onset_shift"]) for row in detail), default=0.0),
            "detail": detail}
