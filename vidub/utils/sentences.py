"""Sentence-level translation units for the translate stage (QUALITY-ROADMAP.md).

Pure, stdlib-only helpers that turn ordered ASR segment rows into
**sentence-level translation units** and split one unit's translation back onto
its member segments.  There is no engine, no I/O and no logging side effect: the
module is importable and testable in isolation and every function is
deterministic — identical input rows always produce identical units, identical
split positions and identical output strings (no randomness, no locale, no
wall-clock).

An ASR ``segments.json`` row has at least ``text`` / ``start`` / ``end`` /
``speaker`` (the shape :func:`vidub.stages.translate.translate` builds)::

    {"start": 5.8, "end": 11.51, "text": "...", "speaker": "spk0", ...}

:func:`split_into_sentences` turns such rows into one row **per sentence**,
splitting the text at internal terminal punctuation and locking each sentence's
``start``/``end`` to the ASR word edges when ``words`` is usable (interpolating
by character fraction otherwise), so a sentence's span is never an arbitrary
fragment boundary.

:func:`group_units` merges consecutive rows into a unit that ends **only** at a
completed sentence (:func:`ends_sentence` on the last member's text) or a
**speaker change**; a silence gap never ends a unit.  Units are therefore
**sentence-scoped**: each is one complete sentence, or a fragment run ending in
the complete sentence it completes.  The hard ceiling (:data:`HARD_UNIT_WORDS` /
:data:`HARD_UNIT_SECONDS`) is the **sole size guard**; when a single oversized
sentence exceeds it and holds no internal sentence boundary, the unit closes at
the largest internal gap of at least :data:`PAUSE_CUT_GAP_S` seconds (a counted
*pause cut*, the documented last resort) and with no such gap it stays whole and
:func:`unit_cut_stats` counts it ``oversized``.  **No code path takes a
mid-sentence cut**: :func:`unit_cut_stats` reports ``mid_sentence_cuts``, which
is ``0`` for every input.  A unit's closure is classified from the ASR rows, so
:func:`unit_cut_stats` needs them (the rows tell a speaker change, a sentence end
and a timing gap apart).

:func:`split_translation` is the inverse direction: it deterministically cuts one
unit translation back onto its member segments, preferring clause then terminal
then plain word boundaries.  It always returns exactly one string per member and
always preserves the input's non-whitespace characters in order; a token that
contains a digit is not broken whenever an alternative cut exists, and every
piece is non-empty whenever that is achievable without breaking a digit-bearing
token.  See its docstring for the exact guarantee and the unavoidable empty
cases.

Standalone example::

    from vidub.utils.sentences import (
        group_units,
        split_into_sentences,
        split_translation,
    )

    rows = split_into_sentences(asr_rows)
    units = group_units(rows)
    for unit, piece in zip(units, split_translation(text, units[0].shares)):
        ...
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

__all__ = [
    "SENTENCE_GAP_S",
    "HARD_UNIT_WORDS",
    "HARD_UNIT_SECONDS",
    "PAUSE_CUT_GAP_S",
    "TERMINAL_CHARS",
    "CLAUSE_CHARS",
    "Unit",
    "ends_sentence",
    "split_into_sentences",
    "group_units",
    "unit_cut_stats",
    "split_translation",
    "unit_share_summary",
]

# A silence gap at/above this many seconds is a long pause.  It is a public
# threshold for callers that classify pause boundaries (e.g. the sentence-unit
# A/B harness); it never ends a unit (units are sentence-scoped, see
# group_units).
SENTENCE_GAP_S = 0.45
# Hard size ceiling: the sole size guard.  A unit is sentence-scoped, so it only
# closes at a completed sentence or a speaker change; when a single oversized
# sentence exceeds this ceiling and holds no internal sentence boundary, the unit
# closes at the largest internal gap of at least PAUSE_CUT_GAP_S seconds (a
# counted pause cut).  With no such gap it stays whole and is counted
# "oversized".  This bounds a runaway unit without ever cutting mid-word.
HARD_UNIT_WORDS = 150
HARD_UNIT_SECONDS = 45.0
PAUSE_CUT_GAP_S = 0.6
# Terminal (sentence-ending) characters and clause separators.
TERMINAL_CHARS = ".!?…"
CLAUSE_CHARS = ",;:—"

# Characters that may follow a terminal character without changing the fact
# that a sentence ends there (closing quotes / brackets).
_CLOSERS = "'\"”’)"

_TOKEN_RE = re.compile(r"\S+")


# --------------------------------------------------------------------- helpers
def _collapse(text: str) -> str:
    """Whitespace-normalise ``text`` to single spaces (leading/trailing removed)."""
    return " ".join(str(text).split())


def _nonspace_count(text: str) -> int:
    """Number of non-whitespace characters in ``text``."""
    return sum(1 for char in text if not char.isspace())


def _token_ends_sentence(token: str) -> bool:
    """True when whitespace token ``token`` ends a sentence.

    Trailing closing quotes/brackets (``"'”’)``) are skipped first, so ``stop."``
    ends a sentence while ``running``, ``joke,`` and ``data`` do not.
    """
    index = len(token) - 1
    while index >= 0 and token[index] in _CLOSERS:
        index -= 1
    return index >= 0 and token[index] in TERMINAL_CHARS


def _word_edge(entry: Any, key: str) -> float | None:
    """``float`` value of ``entry[key]`` for a word mapping, else ``None``.

    A ``None`` placeholder (the ASR ``words`` list allows holes) or a malformed
    entry simply has no edge, so callers fall back to interpolation.
    """
    if isinstance(entry, Mapping) and key in entry:
        try:
            return float(entry[key])
        except (TypeError, ValueError):
            return None
    return None


def _slice_words(
    words: Any, begin: int, finish: int, token_count: int
) -> Any:
    """The ``words`` sub-list owned by a piece spanning tokens ``[begin, finish)``.

    When ``words`` is a list it is split proportionally onto the token range, so
    every word lands in exactly one piece and no piece shares the whole list. A
    non-list value (``None`` / absent) is returned unchanged.
    """
    if not isinstance(words, list):
        return words
    if not words:
        return []
    count = len(words)
    if count == token_count:
        return list(words[begin:finish])
    first = round(begin * count / token_count)
    last = round(finish * count / token_count)
    return list(words[first:last])


# ------------------------------------------------------------------ unit model
@dataclass(frozen=True)
class Unit:
    """A sentence-level translation unit over consecutive ASR segment rows.

    ``indices`` are the source row indices (in order), ``source`` is the member
    texts joined with a single space (whitespace-normalised), ``start``/``end``
    are the first member's start and the last member's end, and ``shares`` are
    each member's share of the unit's non-whitespace character count (summing to
    ``1.0`` up to floating-point rounding; an empty member gets ``0.0`` unless
    every member is empty).
    """

    indices: tuple[int, ...]
    source: str
    start: float
    end: float
    shares: tuple[float, ...]


def ends_sentence(text: str) -> bool:
    """True when the stripped ``text`` ends with a terminal character.

    Trailing closing quotes/brackets (``"'”’)``) are skipped first, so
    ``'He said "stop."'`` and ``'Done.”'`` both end a sentence while
    ``"Hello,"``, ``"Model 3"`` and the empty string do not.
    """
    stripped = str(text).strip()
    index = len(stripped) - 1
    while index >= 0 and stripped[index] in _CLOSERS:
        index -= 1
    return index >= 0 and stripped[index] in TERMINAL_CHARS


# --------------------------------------------------------- sentence splitting
def split_into_sentences(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Split ASR rows at internal sentence boundaries into one row per sentence.

    ``rows`` is the ASR shape :func:`group_units` takes —
    ``{"text", "start", "end", "speaker", ...}`` with an optional ``words`` list
    of ``{"start", "end", "word"}`` (holes may be ``None``).  ``text`` is
    tokenised on whitespace and a split point is taken after a terminal token
    (its trailing character, closing quotes/brackets skipped, is terminal — see
    :func:`_token_ends_sentence`) **when that token is not the row's last
    token**: a trailing terminal means the row already ends a sentence, so the
    row is returned unchanged.

    Each output row keeps the input's keys (``speaker``, ``meta``, ...) and:

    * ``start``/``end`` are **locked to real word edges** when ``words`` lines up
      with the tokens (``len(words) == len(tokens)`` and every entry carries a
      ``start``/``end``); otherwise they are interpolated by character fraction
      across the row's word timeline (``words[0].start``..``words[-1].end`` when
      those edges exist, else the row's ``start``..``end``).  Either way the
      values are clamped to and monotone inside the row's own ``[start, end]``.
    * ``words`` (when the input had a ``words`` list) is replaced by that piece's
      own filtered sub-list, so no piece shares the whole list.

    A row with **no internal sentence boundary** — or with empty/whitespace text —
    is returned **unchanged** (the very same mapping).  Deterministic, no I/O.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        text = str(row.get("text", ""))
        tokens = text.split()
        if not tokens:
            out.append(row)  # type: ignore[arg-type]  # unchanged (identity)
            continue

        cuts = [
            index + 1
            for index, token in enumerate(tokens)
            if index < len(tokens) - 1 and _token_ends_sentence(token)
        ]
        if not cuts:
            out.append(row)  # type: ignore[arg-type]  # unchanged (identity)
            continue

        boundaries = [0, *cuts, len(tokens)]
        row_start = float(row.get("start", 0.0) or 0.0)
        row_end = float(row.get("end", 0.0) or 0.0)
        low, high = min(row_start, row_end), max(row_start, row_end)

        words = row.get("words")
        word_list = list(words) if isinstance(words, list) else None
        usable = (
            word_list is not None
            and len(word_list) == len(tokens)
            and all(
                _word_edge(entry, "start") is not None
                and _word_edge(entry, "end") is not None
                for entry in word_list
            )
        )
        first_edge = _word_edge(word_list[0], "start") if word_list else None
        last_edge = _word_edge(word_list[-1], "end") if word_list else None
        if first_edge is not None and last_edge is not None:
            span_start, span_end = first_edge, last_edge
        else:
            span_start, span_end = row_start, row_end

        offsets = [0]
        for token in tokens:
            offsets.append(offsets[-1] + len(token))
        total = offsets[-1] or 1

        cursor = low
        for position in range(len(boundaries) - 1):
            begin, finish = boundaries[position], boundaries[position + 1]
            if usable and word_list is not None:
                piece_start = _word_edge(word_list[begin], "start") or 0.0
                piece_end = _word_edge(word_list[finish - 1], "end") or 0.0
            else:
                fraction_start = offsets[begin] / total
                fraction_end = offsets[finish] / total
                piece_start = span_start + fraction_start * (span_end - span_start)
                piece_end = span_start + fraction_end * (span_end - span_start)
            piece_start = min(max(piece_start, cursor), high)
            piece_end = min(max(piece_end, piece_start), high)
            cursor = piece_end

            new_row: dict[str, Any] = dict(row)
            new_row["text"] = " ".join(tokens[begin:finish])
            new_row["start"] = piece_start
            new_row["end"] = piece_end
            if "words" in row:
                new_row["words"] = _slice_words(
                    word_list, begin, finish, len(tokens)
                )
            out.append(new_row)
    return out


# ----------------------------------------------------------------- grouping
def group_units(rows: Sequence[Mapping[str, Any]]) -> list[Unit]:
    """Group ordered ASR rows into sentence-level :class:`Unit` objects.

    A unit ends **only** at a completed sentence (:func:`ends_sentence` on the
    last member's text) or at a **speaker change**; a silence gap never ends a
    unit.  Units are therefore sentence-scoped: each is one complete sentence, or
    a fragment run ending in the complete sentence it completes.

    The hard ceiling (:data:`HARD_UNIT_WORDS` / :data:`HARD_UNIT_SECONDS`) is the
    **sole size guard**.  When a single oversized sentence exceeds it and holds
    no internal sentence boundary, the unit closes at the largest internal gap of
    at least :data:`PAUSE_CUT_GAP_S` seconds (a counted *pause cut*, the
    documented last resort); with no such gap it stays whole and is counted
    ``oversized`` by :func:`unit_cut_stats`.  No path takes a mid-sentence cut.

    Empty ``rows`` yields ``[]``; a single row yields one unit; and a unit never
    spans a speaker change.
    """
    row_list = list(rows)
    count = len(row_list)
    if count == 0:
        return []

    texts = [str(row.get("text", "")) for row in row_list]
    starts = [float(row.get("start", 0.0) or 0.0) for row in row_list]
    ends = [float(row.get("end", 0.0) or 0.0) for row in row_list]
    speakers = [row.get("speaker") for row in row_list]
    normalized = [_collapse(text) for text in texts]
    words = [len(part.split()) for part in normalized]

    def gap(prev: int, cur: int) -> float:
        return starts[cur] - ends[prev]

    def ends_unit(prev: int, cur: int) -> bool:
        """A cut that *must* end the current unit (sentence/speaker boundary)."""
        return speakers[prev] != speakers[cur] or ends_sentence(texts[prev])

    def over_hard(members: list[int]) -> bool:
        if sum(words[member] for member in members) > HARD_UNIT_WORDS:
            return True
        return (ends[members[-1]] - starts[members[0]]) > HARD_UNIT_SECONDS

    def largest_gap_cut(members: list[int]) -> int | None:
        """Latest internal cut at the largest gap >= PAUSE_CUT_GAP_S, else None."""
        best: int | None = None
        best_gap = PAUSE_CUT_GAP_S
        for position in range(1, len(members)):
            current = gap(members[position - 1], members[position])
            if current >= PAUSE_CUT_GAP_S and current >= best_gap:
                best = position
                best_gap = current
        return best

    def build(members: list[int]) -> Unit:
        parts = [part for part in (normalized[member] for member in members) if part]
        source = " ".join(parts)
        counts = [_nonspace_count(texts[member]) for member in members]
        total = sum(counts)
        if total > 0:
            shares = tuple(value / total for value in counts)
        else:
            shares = tuple(1.0 / len(members) for _ in members)
        return Unit(
            indices=tuple(members),
            source=source,
            start=starts[members[0]],
            end=ends[members[-1]],
            shares=shares,
        )

    units: list[Unit] = []
    current: list[int] = [0]
    for index in range(1, count):
        if ends_unit(index - 1, index):
            units.append(build(current))
            current = [index]
            continue
        current.append(index)
        while len(current) > 1 and over_hard(current):
            position = largest_gap_cut(current)
            if position is None:
                break
            units.append(build(current[:position]))
            current = current[position:]

    units.append(build(current))
    return units


def unit_cut_stats(
    units: Sequence[Unit], rows: Sequence[Mapping[str, Any]] | None = None
) -> dict[str, int]:
    """Classify every unit closure; ``mid_sentence_cuts`` is the invariant (0).

    A *closure* is the boundary that ends a unit.  It is a **speaker cut** when
    the next unit's first member has a different speaker; that test is applied
    **first**, so a boundary that is both a speaker change and a sentence end is
    counted as a speaker cut.  Otherwise it is a **sentence cut** when the last
    member's text ends a sentence, a **pause cut** when it is the hard-ceiling
    last resort (same speaker, no terminal text, timing gap at least
    :data:`PAUSE_CUT_GAP_S`), and otherwise a **mid-sentence cut**.  The final
    unit's closure is the end of the transcript and is never a cut.
    ``oversized`` counts units still exceeding the hard ceiling after grouping
    (:data:`HARD_UNIT_WORDS` / :data:`HARD_UNIT_SECONDS`).  Under the shipped
    rules no code path takes a mid-sentence cut, so ``mid_sentence_cuts`` is
    always ``0``.

    ``rows`` are the same ASR rows :func:`group_units` grouped and are
    **required**: they are the only way to tell a speaker change, a sentence end
    and a timing gap apart (the unit text alone cannot).  A missing or ``None``
    ``rows`` raises :class:`TypeError`.
    """
    if rows is None:
        raise TypeError(
            "unit_cut_stats requires the ASR rows: they tell a speaker change, "
            "a sentence end and a timing gap apart"
        )

    stats = {
        "units": len(units),
        "sentence_cuts": 0,
        "pause_cuts": 0,
        "speaker_cuts": 0,
        "mid_sentence_cuts": 0,
        "oversized": 0,
    }

    row_list = list(rows)
    texts = [str(row.get("text", "")) for row in row_list]
    starts = [float(row.get("start", 0.0) or 0.0) for row in row_list]
    ends = [float(row.get("end", 0.0) or 0.0) for row in row_list]
    speakers = [row.get("speaker") for row in row_list]
    for position in range(len(units) - 1):
        prev = units[position].indices[-1]
        cur = units[position + 1].indices[0]
        if speakers[prev] != speakers[cur]:
            stats["speaker_cuts"] += 1
        elif ends_sentence(texts[prev]):
            stats["sentence_cuts"] += 1
        elif starts[cur] - ends[prev] >= PAUSE_CUT_GAP_S:
            stats["pause_cuts"] += 1
        else:
            stats["mid_sentence_cuts"] += 1

    for unit in units:
        if (
            len(unit.source.split()) > HARD_UNIT_WORDS
            or (unit.end - unit.start) > HARD_UNIT_SECONDS
        ):
            stats["oversized"] += 1
    return stats


# --------------------------------------------------------------- translation
def _next_split_index(pieces: list[str]) -> int | None:
    """Index of the token to split next, or ``None`` when none may be split.

    Eligible tokens are non-digit-bearing and at least two characters long; the
    longest eligible token wins (leftmost on a tie).  When no eligible token
    exists — every candidate is a single character or a digit-bearing token —
    ``None`` is returned instead of breaking a number.
    """
    best: int | None = None
    best_len = 0
    for index, token in enumerate(pieces):
        if len(token) < 2 or any(char.isdigit() for char in token):
            continue
        if len(token) > best_len:
            best = index
            best_len = len(token)
    return best


def _split_sparse(tokens: list[str], count: int) -> list[str]:
    """Fallback when there are fewer tokens than members: split non-digit tokens.

    The available tokens are distributed one per member and the remainder is
    filled by splitting eligible tokens — those without a digit and at least two
    characters long — preferring the longest (leftmost on a tie), until there
    are ``count`` pieces.  A token containing a digit is **never** split here: if
    no eligible token remains the tail is padded with empty strings, so the
    result is still exactly ``count`` pieces with every digit token intact and
    the reassembly property preserved.  Those padded empties are the documented
    unavoidable case (see :func:`split_translation`), and a number is never
    traded for a non-empty piece.
    """
    pieces = list(tokens)
    while len(pieces) < count:
        index = _next_split_index(pieces)
        if index is None:
            break
        token = pieces[index]
        middle = len(token) // 2
        pieces[index : index + 1] = [token[:middle], token[middle:]]
    if len(pieces) < count:
        pieces.extend([""] * (count - len(pieces)))
    return pieces


def split_translation(translation: str, shares: Sequence[float]) -> list[str]:
    """Split one unit translation back onto ``len(shares)`` member segments.

    A single share returns ``[translation.strip()]``.  Otherwise the target cut
    offsets are the cumulative ``shares`` of the translation's non-whitespace
    character count (floored); each target is snapped to the nearest boundary
    within ``+/-12`` characters, preferring a **clause** boundary
    (:data:`CLAUSE_CHARS`) over a **terminal** one (:data:`TERMINAL_CHARS`) over
    a plain word boundary, widening to ``+/-24`` when needed.  A cut between two
    adjacent digit-bearing tokens (the space in ``"30 000"``) is avoided
    whenever an alternative cut exists; when every candidate would split such a
    pair it is used (the only boundary available).  **In the main path (at least
    one token per member) every cut is taken between whole tokens**, so it never
    splits a token at all; the sparse fallback is the only place a token may be
    split, and it never splits a digit-bearing one (see :func:`_split_sparse`).

    Guarantees, stated exactly:

    * the result always has exactly ``len(shares)`` strings;
    * concatenating them with a single space and collapsing whitespace always
      reproduces the input's non-whitespace characters in order — nothing is
      dropped, duplicated or reordered;
    * a token containing a digit is not split whenever a cut that avoids it
      exists; the sparse fallback (:func:`_split_sparse`, used when there are
      fewer tokens than members) never splits one at all;
    * every piece is non-empty whenever that is achievable without breaking the
      reassembly property or a digit-bearing token.  Empty pieces can occur in
      exactly three situations: an empty/whitespace translation, a translation
      with fewer non-whitespace characters than members, or a sparse fallback
      whose only remaining splittable tokens contain digits (the digit token is
      kept intact and the tail is padded with empty strings).  A broken number
      is never traded for a non-empty piece.
    """
    count = len(shares)
    if count <= 0:
        return []
    if count == 1:
        return [str(translation).strip()]

    text = str(translation)
    tokens = _TOKEN_RE.findall(text)
    if not tokens:
        # Whitespace-only text carries no content to distribute.
        return [""] * count
    if len(tokens) < count:
        return _split_sparse(tokens, count)

    offsets = [0]
    for token in tokens:
        offsets.append(offsets[-1] + len(token))
    total_chars = offsets[-1]

    def kind(cut: int) -> int:
        """Preference rank of a cut: clause (3) > terminal (2) > plain (1)."""
        left = tokens[cut - 1]
        if left[-1] in CLAUSE_CHARS:
            return 3
        if left[-1] in TERMINAL_CHARS:
            return 2
        return 1

    def digit_unsafe(cut: int) -> bool:
        """True when the cut would separate two adjacent digit-bearing tokens."""
        left, right = tokens[cut - 1], tokens[cut]
        return left[-1].isdigit() and right[0].isdigit()

    def pick(candidates: list[int], target: int) -> int:
        return max(
            candidates,
            key=lambda cut: (kind(cut), -abs(offsets[cut] - target), -cut),
        )

    targets: list[int] = []
    cumulative = 0.0
    for position in range(count - 1):
        cumulative += float(shares[position])
        targets.append(int(total_chars * cumulative))

    cuts: list[int] = []
    previous = 0
    for step, target in enumerate(targets):
        # Leave at least one token for every remaining member.
        max_cut = len(tokens) - (count - 1 - step)
        candidates = list(range(previous + 1, max_cut + 1))
        # Prefer a cut that does not sit between two adjacent digit-bearing
        # tokens; fall back to the candidates themselves when every option
        # would separate a number (the only boundary available).
        allowed = [cut for cut in candidates if not digit_unsafe(cut)] or candidates
        window = [cut for cut in allowed if abs(offsets[cut] - target) <= 12]
        if not window:
            window = [cut for cut in allowed if abs(offsets[cut] - target) <= 24]
        if not window:
            window = allowed
        cut = pick(window, target)
        cuts.append(cut)
        previous = cut

    pieces: list[str] = []
    start = 0
    for cut in cuts:
        pieces.append(" ".join(tokens[start:cut]))
        start = cut
    pieces.append(" ".join(tokens[start:]))
    return [piece.strip() for piece in pieces]


# --------------------------------------------------------------------- summary
def unit_share_summary(units: Sequence[Unit]) -> dict[str, int]:
    """Small caller/log helper: ``{"units", "segments", "merged_segments"}``.

    ``segments`` is the total member count and ``merged_segments`` is
    ``segments - units`` (the number of merges grouping produced).
    """
    total = sum(len(unit.indices) for unit in units)
    return {
        "units": len(units),
        "segments": total,
        "merged_segments": total - len(units),
    }
