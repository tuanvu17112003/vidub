"""Bounded subtitle scheduling: readable chunks, adjacent silence, fixed profile."""
from __future__ import annotations

import math
from typing import Any

from vidub.utils.subqc import CPS_MAX, DUR_MAX, DUR_MIN, LINE_MAX, measure, wrap_text

MAX_LEAD = 0.25
MAX_LAG = 1.50
GRID = 0.05
POLICY = "bounded-reading-v1"


def _partition(text: str, span: float) -> list[str]:
    """Minimum viable chunk count, then prefer balanced sentence/clause cuts."""
    words = text.split()
    if not words:
        return [text]
    count = min(len(words), max(1, math.ceil(span / DUR_MAX), math.ceil(len(text) / (2 * LINE_MAX))))
    for chunks in range(count, len(words) + 1):
        paths: dict[tuple[int, int], tuple[float, list[str]]] = {(0, 0): (0.0, [])}
        for k in range(chunks):
            for (start, used), (cost, parts) in list(paths.items()):
                if used != k:
                    continue
                for stop in range(start + 1, len(words) - (chunks - k - 1) + 1):
                    part = " ".join(words[start:stop])
                    wrapped = wrap_text(part)
                    fits = len(wrapped.splitlines()) <= 2 and all(
                        len(line) <= LINE_MAX for line in wrapped.splitlines()
                    )
                    if not fits and stop > start + 1:
                        break
                    # One overlong word is retained and flagged, never sliced.
                    balance = ((len(part) - len(text) / chunks) / max(len(text), 1)) ** 2
                    punctuation = 0.0 if part.endswith((".", "!", "?")) else (0.08 if part.endswith((",", ";", ":")) else 0.16)
                    new_cost = cost + balance + (punctuation if stop < len(words) else 0.0)
                    key = (stop, k + 1)
                    if key not in paths or new_cost < paths[key][0]:
                        paths[key] = (new_cost, parts + [part])
        best = paths.get((len(words), chunks))
        if best:
            return best[1]
    return [text]


def _references(parts: list[str], row: dict) -> list[float]:
    """Speech boundaries, using a word timeline only when counts match."""
    start, end = float(row["start"]), float(row["end"])
    words = row.get("align_words")
    sizes = [len(part.split()) for part in parts]
    if isinstance(words, list) and len(words) == sum(sizes) and words:
        try:
            times = [float(word[0]) for word in words]
            if all(math.isfinite(t) and start <= t <= end for t in times) and times == sorted(times):
                positions = []
                position = 0
                for size in sizes[:-1]:
                    position += size
                    positions.append(times[position])
                return [start, *positions, end]
        except (TypeError, ValueError, IndexError):
            pass
    sizes_chars = [len(part) for part in parts]
    total = max(sum(sizes_chars), 1)
    accumulated = 0
    boundaries = [start]
    for size in sizes_chars[:-1]:
        accumulated += size
        boundaries.append(start + (end - start) * accumulated / total)
    return [*boundaries, end]


def _times(reference: float, low: float, high: float) -> list[float]:
    low = max(low, reference - MAX_LEAD)
    high = min(high, reference + MAX_LAG)
    if high < low:
        return []
    # Schedule on the same millisecond precision as SRT/ASS export. Avoid a
    # rounded boundary borrowing beyond the promised speech-displacement bound.
    values = {math.ceil(low * 1000) / 1000, math.floor(high * 1000) / 1000, round(reference, 3)}
    values.update(i * GRID for i in range(math.ceil(low / GRID), math.floor(high / GRID) + 1))
    return sorted(value for value in values if low - 1e-9 <= value <= high + 1e-9)


def _schedule(parts: list[str], refs: list[float], low: float, high: float) -> list[float]:
    """Dynamic program over bounded word/speech edges; flag infeasible profiles."""
    states = {t: (abs(t - refs[0]) * 0.2, [t]) for t in _times(refs[0], low, high)}
    for index, part in enumerate(parts):
        following: dict[float, tuple[float, list[float]]] = {}
        for end in _times(refs[index + 1], low, high):
            for start, (cost, path) in states.items():
                duration = end - start
                if duration < 0.001:
                    continue
                # Readability dominates drift, but an impossible window stays
                # visible as a violation rather than causing text deletion.
                shortfall = max(len(part) / CPS_MAX - duration, 0.0)
                duration_error = max(DUR_MIN - duration, 0.0) + max(duration - DUR_MAX, 0.0)
                score = cost + 100 * shortfall + 1000 * duration_error + 0.2 * abs(end - refs[index + 1])
                if end not in following or score < following[end][0]:
                    following[end] = (score, path + [end])
        states = following
        if not states:
            return refs
    return min(states.values(), key=lambda value: value[0])[1]


def schedule_rows(rows: list[dict], video_duration: float | None = None) -> tuple[list[dict], dict[str, Any]]:
    """Retain all words and metadata; no overlap for chronological speech spans."""
    if not rows:
        return [], {"policy": POLICY, "split": 0, "retimed": 0, "max_lead": 0.0, "max_lag": 0.0, "infeasible_cues": 0}
    normalized = [{**row, "vi": " ".join(str(row.get("vi", "")).split())} for row in rows]
    starts = [float(row["start"]) for row in normalized]
    ends = [float(row["end"]) for row in normalized]
    left = starts[:]
    right = ends[:]
    need = [max(len(row["vi"]) / CPS_MAX, DUR_MIN) - (end - start)
            for row, start, end in zip(normalized, starts, ends)]
    left[0] = max(0.0, starts[0] - min(MAX_LEAD, max(need[0], 0.0)))
    for i in range(len(rows) - 1):
        gap = max(starts[i + 1] - ends[i], 0.0)
        reserve = min(MAX_LEAD, gap) if need[i + 1] > 0 else 0.0
        lag = min(MAX_LAG, gap - reserve) if need[i] > 0 else 0.0
        lead = min(MAX_LEAD, gap - lag) if need[i + 1] > 0 else 0.0
        right[i] += lag
        left[i + 1] -= lead
    right[-1] += MAX_LAG if need[-1] > 0 else 0.0
    if video_duration is not None:
        right[-1] = max(ends[-1], min(right[-1], float(video_duration)))
    out: list[dict] = []
    retimed = 0
    lead_max = lag_max = 0.0
    infeasible = 0
    split = 0
    for row, low, high in zip(normalized, left, right):
        parts = _partition(row["vi"], max(float(row["end"]) - float(row["start"]), 0.0))
        split += len(parts) > 1
        refs = _references(parts, row)
        boundaries = _schedule(parts, refs, low, high)
        for i, part in enumerate(parts):
            start, end = round(boundaries[i], 3), round(boundaries[i + 1], 3)
            result = {**row, "start": start, "end": end, "vi": wrap_text(part)}
            out.append(result)
            infeasible += bool(measure(result["vi"], start, end)["violations"])
            lead_max = max(lead_max, refs[i] - start, refs[i + 1] - end)
            lag_max = max(lag_max, start - refs[i], end - refs[i + 1])
        retimed += any(abs(a - b) > 1e-5 for a, b in zip(boundaries, refs))
    return out, {"policy": POLICY, "split": split, "retimed": retimed, "max_lead": round(lead_max, 6),
                 "max_lag": round(lag_max, 6), "infeasible_cues": infeasible}
