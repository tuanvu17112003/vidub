"""Vietnamese subtitle standards QC (QUALITY-ROADMAP.md G3 / P0 item 2).

Pure, dependency-free measurement and repair of an SRT-shaped cue list against a
fixed **project readability profile** (informed by timed-text style guides):

* reading speed **<= 17 characters per second** (CPS, spaces included);
* cue duration **>= 5/6 s** (minimum readability) and **<= 7 s** (maximum);
* at most **2 lines** per cue, at most **42 characters per line**.

Readable chunks are scheduled with bounded lead/lag and adjacent silence.
High CPS alone never causes recursive splitting in the active export policy.
Infeasible cues remain flagged; text is never truncated or dropped.

The module is I/O-free: :func:`measure`, :func:`wrap_text`, :func:`split_cue`
and :func:`qc_rows` operate on plain dicts/strings, so the export stage can keep
the QC transform strictly scoped to the ``subtitles.vi.srt`` deliverable.
"""

from __future__ import annotations

import math

__all__ = [
    "CPS_MAX",
    "DUR_MIN",
    "DUR_MAX",
    "LINE_MAX",
    "LINES_MAX",
    "measure",
    "wrap_text",
    "split_cue",
    "qc_rows",
]

# Existing project profile; retained for matched comparisons.
CPS_MAX = 17.0
DUR_MIN = 5 / 6  # 0.833 s minimum cue duration
DUR_MAX = 7.0
LINE_MAX = 42
LINES_MAX = 2

# Characters that close a sentence (highest split preference).
_SENTENCE_ENDERS = ".!?"
# Characters that separate clauses (middle preference).
_CLAUSE_ENDERS = ",;:"


def _normalize(text: str) -> str:
    """Normalize newlines to ``\\n`` (no other change)."""
    return str(text).replace("\r\n", "\n").replace("\r", "\n")


def _chars(text: str) -> int:
    """Displayed character count: everything but line breaks (spaces included)."""
    return len(_normalize(text).replace("\n", ""))


def _split_lines(text: str) -> list[str]:
    """The cue's lines after newline normalization."""
    return _normalize(text).split("\n")


def _cps(text: str, duration: float) -> float:
    """Reading speed in characters/second (duration clamped to >= 1 ms)."""
    return _chars(text) / max(float(duration), 0.001)


def measure(text: str, start: float, end: float) -> dict:
    """Measure one cue against the profile.

    Returns ``{"cps", "duration", "lines", "max_line_len", "violations"}``;
    ``violations`` holds human-readable reasons such as ``"cps=23.4>17.0"``,
    ``"duration=0.4<0.833"``, ``"line_len=57>42"`` or ``"lines=3>2"``.
    """
    duration = float(end) - float(start)
    lines = _split_lines(text)
    max_line_len = max((len(line) for line in lines), default=0)
    cps = _cps(text, duration)

    violations: list[str] = []
    if cps > CPS_MAX + 1e-6:
        violations.append(f"cps={cps:.1f}>{CPS_MAX}")
    if duration < DUR_MIN - 1e-9:
        violations.append(f"duration={duration:.1f}<{DUR_MIN:.3f}")
    elif duration > DUR_MAX + 1e-9:
        violations.append(f"duration={duration:.1f}>{DUR_MAX:.1f}")
    if max_line_len > LINE_MAX:
        violations.append(f"line_len={max_line_len}>{LINE_MAX}")
    if len(lines) > LINES_MAX:
        violations.append(f"lines={len(lines)}>{LINES_MAX}")

    return {
        "cps": cps,
        "duration": duration,
        "lines": lines,
        "max_line_len": max_line_len,
        "violations": violations,
    }


def _balanced_break(text: str) -> tuple[str, str] | None:
    """Split single-line ``text`` at the space nearest its midpoint."""
    target = math.ceil(len(text) / 2)
    best: int | None = None
    for index, char in enumerate(text):
        if char != " " or index == 0 or index == len(text) - 1:
            continue
        if best is None or abs(index - target) < abs(best - target):
            best = index
    if best is None:
        return None
    left = text[:best].strip()
    right = text[best + 1 :].strip()
    if not left or not right:
        return None
    return left, right


def wrap_text(text: str) -> str:
    """Wrap ``text`` into at most :data:`LINES_MAX` lines of <= :data:`LINE_MAX`.

    A cue already within the limits is returned unchanged. Otherwise it is
    greedily balanced across two lines (break at the space nearest the midpoint).
    When the limits cannot be met (e.g. a single word longer than a line) the
    original is returned unmodified — the cue is flagged, never truncated.
    """
    text = str(text)
    normalized = _normalize(text)
    lines = normalized.split("\n")
    if len(lines) <= LINES_MAX and all(len(line) <= LINE_MAX for line in lines):
        return text

    flat = " ".join(normalized.split("\n")).strip()
    if len(flat) <= LINE_MAX:
        return flat

    split = _balanced_break(flat)
    if split is None:
        return text
    left, right = split
    if len(left) <= LINE_MAX and len(right) <= LINE_MAX:
        return f"{left}\n{right}"
    return text


def _find_split(text: str, allow_space: bool) -> tuple[int, str] | None:
    """Best cut index for ``text`` and its kind (sentence > clause > space)."""
    target = len(text) / 2

    def nearest(indices: list[int]) -> int:
        return min(indices, key=lambda i: abs(i - target))

    # Sentence enders: cut *after* punctuation (never the trailing one).
    sentence = [
        i
        for i, char in enumerate(text)
        if char in _SENTENCE_ENDERS and i < len(text) - 1
        and (i + 1 == len(text) or text[i + 1].isspace())
    ]
    if sentence:
        return nearest(sentence) + 1, "sentence"

    clause = [i for i, char in enumerate(text) if char in _CLAUSE_ENDERS and i < len(text) - 1]
    if clause:
        return nearest(clause) + 1, "clause"

    if allow_space:
        spaces = [i for i, char in enumerate(text) if char == " " and 0 < i < len(text) - 1]
        if spaces:
            return nearest(spaces), "space"
    return None


def _allocate(left_chars: int, right_chars: int, start: float, end: float) -> float:
    """Proportional cut time, respecting :data:`DUR_MIN` when the span allows."""
    span = float(end) - float(start)
    if span <= 0.002 or left_chars + right_chars <= 0:
        return float(end)
    mid = float(start) + span * (left_chars / (left_chars + right_chars))
    if span >= 2 * DUR_MIN:
        mid = min(max(mid, float(start) + DUR_MIN), float(end) - DUR_MIN)
    else:
        mid = min(max(mid, float(start) + 0.001), float(end) - 0.001)
    return mid


def _clamp_cut(mid: float, start: float, end: float) -> float:
    """Keep a cut time inside ``(start, end)``, honouring :data:`DUR_MIN`."""
    span = float(end) - float(start)
    if span >= 2 * DUR_MIN:
        return min(max(mid, float(start) + DUR_MIN), float(end) - DUR_MIN)
    return min(max(mid, float(start) + 0.001), float(end) - 0.001)


def _word_cut_index(left: str, right: str, words: list) -> int:
    """Word-timeline cut index for a text split, or ``-1`` when unusable.

    ``left`` / ``right`` are the two text halves; the index is the share of the
    ASR word timeline the left half owns (proportional to the halves' word
    counts). ``-1`` means "fall back to the proportional character split" — no
    words, a single word, or a degenerate share at either end.
    """
    if not isinstance(words, list) or len(words) < 2:
        return -1
    n_left = len(left.split())
    n_right = len(right.split())
    total = n_left + n_right
    if total <= 0:
        return -1
    index = int(round(n_left / total * len(words)))
    if index <= 0 or index >= len(words):
        return -1
    return index


def _split_recursive(
    text: str,
    start: float,
    end: float,
    allow_space: bool,
    words: list | None = None,
) -> list[tuple[float, float, str]]:
    text = " ".join(str(text).split())
    span = float(end) - float(start)
    if span < 0.002 or _cps(text, span) <= CPS_MAX:
        return [(float(start), float(end), text)]

    found = _find_split(text, allow_space)
    if found is None:
        # A single clause with nothing left to split on: stop and flag it.
        return [(float(start), float(end), text)]

    cut, kind = found
    left = text[:cut].strip()
    right = text[cut:].strip()
    if not left or not right:
        return [(float(start), float(end), text)]

    # Q-1-T4: when the row carries an ASR word timeline, put the boundary on a
    # real word edge (the start of the word the right half begins with) instead
    # of the proportional character allocation. The children inherit their slice
    # of the timeline, so every internal boundary stays a word edge and the cue
    # windows remain contiguous (they sum to the original).
    index = _word_cut_index(left, right, words) if words else -1
    if index > 0:
        try:
            mid = _clamp_cut(float(words[index][0]), start, end)
        except (IndexError, TypeError, ValueError):
            index = -1
    if index > 0:
        left_words: list | None = words[:index]
        right_words: list | None = words[index:]
    else:
        mid = _allocate(len(left), len(right), start, end)
        left_words = right_words = None
    # A space is the last-resort boundary: after using it, children are treated
    # as single clauses and are no longer split further.
    next_allow_space = allow_space and kind != "space"
    return _split_recursive(
        left, start, mid, next_allow_space, left_words
    ) + _split_recursive(right, mid, end, next_allow_space, right_words)


def split_cue(
    text: str, start: float, end: float, words: list | None = None
) -> list[tuple[float, float, str]]:
    """Re-split an over-CPS cue at sentence/clause/space boundaries.

    Returns ``[(start, end, text), ...]``: contiguous spans that stay inside the
    original ``[start, end]`` and sum to the original duration. A cue that cannot
    be improved (a single clause) is returned unchanged, to be flagged by
    :func:`measure`.

    ``words`` is the optional ASR word timeline ``[[start_abs, end_abs, word],
    ...]`` of the cue (absolute seconds). When given, every internal boundary
    lands on a real word edge (Q-1-T4) so the text tracks the audio; without it
    the historical proportional :func:`_allocate` split is used unchanged.
    """
    return _split_recursive(text, start, end, allow_space=True, words=words)


def _violation_key(reason: str) -> str:
    """Map a violation reason string to its counter key."""
    if reason.startswith("cps="):
        return "cps"
    if reason.startswith("line_len="):
        return "line_length"
    if reason.startswith("lines="):
        return "lines"
    return "duration"


def _zero_stats(enabled: bool) -> dict:
    stats = {
        "enabled": bool(enabled),
        "cues_in": 0,
        "cues_out": 0,
        "split": 0,
        "violations": {"cps": 0, "duration": 0, "line_length": 0, "lines": 0},
        "cps_max_observed": 0.0,
    }
    return stats


def _valid_words(value: object) -> list | None:
    """Coerce a row's ``align_words`` to ``[[start, end, word], ...]`` or ``None``.

    Tolerant by design: any malformed entry (a non-list, a non-numeric time, too
    few fields) makes the whole timeline unusable, so the cue falls back to the
    historical proportional split rather than raising.
    """
    if not isinstance(value, list) or len(value) < 2:
        return None
    cleaned: list[list] = []
    for entry in value:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            return None
        try:
            start = float(entry[0])
            end = float(entry[1])
        except (TypeError, ValueError):
            return None
        cleaned.append([start, end, str(entry[2]) if len(entry) > 2 else ""])
    return cleaned


def qc_rows(
    rows: list[dict], enabled: bool = True, *, video_duration: float | None = None
) -> tuple[list[dict], dict]:
    """Schedule readable script chunks for subtitles and return QC counters.

    ``rows`` are ``script.json``-shaped dicts (``{"start", "end", "vi", ...}``).
    Layout/max-duration constraints choose chunks; bounded dynamic scheduling
    borrows adjacent silence to meet readability where feasible. Split rows
    copy the parent with new ``start``/``end``/``vi``; other keys survive.

    Returns ``(new_rows, stats)`` where ``stats`` counts the final rows. When
    ``enabled`` is False the input rows are returned untouched with zeroed stats
    (``"enabled": False``), so the caller can write a byte-identical SRT.
    """
    if not enabled:
        return rows, _zero_stats(False)

    stats = _zero_stats(True)
    stats["cues_in"] = len(rows)

    from vidub.utils.subtitle_schedule import schedule_rows

    out, scheduling = schedule_rows(rows, video_duration)
    stats["scheduling"] = scheduling
    split_count = scheduling["split"]

    violations_count = {"cps": 0, "duration": 0, "line_length": 0, "lines": 0}
    cps_max = 0.0
    for row in out:
        measured = measure(
            str(row.get("vi", "")),
            float(row.get("start", 0.0)),
            float(row.get("end", 0.0)),
        )
        cps_max = max(cps_max, measured["cps"])
        for reason in measured["violations"]:
            violations_count[_violation_key(reason)] += 1

    stats["cues_out"] = len(out)
    stats["split"] = split_count
    stats["violations"] = violations_count
    stats["cps_max_observed"] = round(cps_max, 2)
    return out, stats
