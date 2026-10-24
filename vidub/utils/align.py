"""Pure alignment allocators: weights -> fractions -> word cuts -> spans/phrases.

Stdlib-only, deterministic helpers used to map a set of weights (or a whole
unit translation) onto a word timeline.  :func:`share_fractions` turns weights
into cumulative fractions; :func:`snap_word_cuts` snaps those fractions to word
indices; :func:`time_ranges` maps the cuts onto ``(start, end)`` word times;
:func:`phrase_pieces` cuts a string at whitespace-token boundaries; and
:func:`allocate_spans` is the single composition callers use.

Guaranteed (for a well-formed call):

* the fractions end at exactly ``1.0`` and the cut indices are monotone;
* a bucket's time range is monotone and never inverted, the first bucket starts
  at ``times[0][0]`` and the last ends at ``times[-1][1]``;
* ``"".join(phrase_pieces(text, fractions)) == text`` and no cut falls inside a
  token (cuts land between whole whitespace tokens).

**Not** guaranteed: this is not a phoneme-level forced aligner.  It divides by
the given weights and snaps to whole words, so a bucket can be empty when the
weights are degenerate or when there are fewer words than buckets — the latter
is exactly when :func:`allocate_spans` returns ``None`` so the caller can fall
back to a proportional split.
"""

from __future__ import annotations

import re
from typing import Sequence

__all__ = [
    "share_fractions",
    "snap_word_cuts",
    "time_ranges",
    "phrase_pieces",
    "allocate_spans",
]

_TOKEN_RE = re.compile(r"\S+")


def share_fractions(weights: Sequence[float]) -> list[float]:
    """Cumulative fractions of ``weights``, ending at exactly ``1.0``.

    Empty input yields ``[]``.  Non-positive weights contribute nothing; when
    every weight is ``<= 0`` (or the positive total is ``0``) the weights are
    treated as **equal**, so the result is ``1/n, 2/n, ..., 1.0``.  Otherwise
    each entry is the running positive-weight total over the grand total, and
    the last value is forced to exactly ``1.0``.
    """
    count = len(weights)
    if count == 0:
        return []
    positive: list[float] = []
    for weight in weights:
        value = float(weight)
        positive.append(value if value > 0 else 0.0)
    total = sum(positive)
    if total <= 0.0:
        return [(index + 1) / count for index in range(count)]
    fractions: list[float] = []
    running = 0.0
    for weight in positive:
        running += weight
        fractions.append(running / total)
    fractions[-1] = 1.0
    return fractions


def snap_word_cuts(fractions: Sequence[float], n_words: int) -> list[int]:
    """Snap each fraction to the nearest word index over ``n_words``.

    Returns one cut per fraction, monotone non-decreasing and clamped to
    ``[0, n_words]`` (``round`` of a large fraction saturates at ``n_words``).
    Empty ``fractions`` yields ``[]``; ``n_words <= 0`` yields zeros.
    """
    if n_words <= 0:
        return [0] * len(fractions)
    cuts: list[int] = []
    previous = 0
    for fraction in fractions:
        index = int(round(float(fraction) * n_words))
        index = min(max(index, 0), n_words)
        index = max(index, previous)
        cuts.append(index)
        previous = index
    return cuts


def time_ranges(
    fractions: Sequence[float], times: Sequence[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Map each fraction bucket onto the word ``times`` timeline.

    One ``(start, end)`` per fraction.  The first range starts at
    ``times[0][0]`` and the last ends at ``times[-1][1]``; every range is
    monotone (``start <= end``), non-inverted and does not move backwards.  An
    empty ``times`` or ``fractions`` yields ``[]``.
    """
    if not times or not fractions:
        return []

    count = len(times)
    bounds = [0, *snap_word_cuts(fractions, count)]
    ranges: list[tuple[float, float]] = []
    cursor = float(times[0][0])
    for position in range(len(fractions)):
        low = min(bounds[position], count)
        high = min(bounds[position + 1], count)
        if high > low:
            start = float(times[low][0])
            end = float(times[high - 1][1])
        else:
            start = end = float(times[min(low, count - 1)][0])
        start = max(start, cursor)
        end = max(end, start)
        ranges.append((start, end))
        cursor = end

    # Pin the outer edges exactly (the word timeline is ordered, so this cannot
    # invert a range).
    first_start = float(times[0][0])
    first_end = max(ranges[0][1], first_start)
    ranges[0] = (first_start, first_end)
    last_end = float(times[-1][1])
    last_start = min(ranges[-1][0], last_end)
    ranges[-1] = (last_start, last_end)
    return ranges


def phrase_pieces(text: str, fractions: Sequence[float]) -> list[str]:
    """Cut ``text`` into ``len(fractions)`` pieces at whitespace-token boundaries.

    The pieces are exact substrings in order and ``"".join(pieces) == text``:
    nothing is dropped, duplicated or reordered, and no cut falls inside a token
    (a cut lands immediately before a token, so intervening whitespace stays with
    the following piece).  Empty ``fractions`` yields ``[]``; a single bucket
    yields ``[text]``; whitespace-only ``text`` yields one leading piece holding
    the whole string followed by empty pieces.
    """
    count = len(fractions)
    if count <= 0:
        return []
    if count == 1:
        return [text]

    matches = list(_TOKEN_RE.finditer(text))
    if not matches:
        pieces = [""] * count
        pieces[0] = text
        return pieces

    token_count = len(matches)
    # Character position of each cut index k in [0, token_count]: index 0 -> 0,
    # k -> start of token k (1 <= k < token_count), token_count -> end of text.
    cut_chars = [0] + [match.start() for match in matches[1:]] + [len(text)]

    cuts: list[int] = []
    previous = 0
    for position in range(count - 1):
        index = int(round(float(fractions[position]) * token_count))
        index = min(max(index, previous), token_count)
        cuts.append(index)
        previous = index
    bounds = [0, *cuts, token_count]
    return [
        text[cut_chars[bounds[position]] : cut_chars[bounds[position + 1]]]
        for position in range(count)
    ]


def allocate_spans(
    weights: Sequence[float], times: Sequence[tuple[float, float]]
) -> list[tuple[float, float]] | None:
    """Compose the allocator: ``weights`` + word ``times`` -> one span per weight.

    Returns ``None`` when the input is degenerate — empty ``weights`` or
    ``times``, or **fewer word times than buckets** (``len(times) <
    len(weights)``) — so the caller can fall back to a proportional split.
    Otherwise returns :func:`time_ranges` over :func:`share_fractions`: one
    monotone, non-inverted ``(start, end)`` per bucket covering the whole
    timeline (first starts at ``times[0][0]``, last ends at ``times[-1][1]``).
    """
    if not weights or not times:
        return None
    if len(times) < len(weights):
        return None
    fractions = share_fractions(weights)
    if not fractions:
        return None
    return time_ranges(fractions, times)
