"""Hesitation/repetition + filler cleanup for the dubbing text (M3-C1).

Two independent, stdlib-only helpers used before TTS/subtitles:

* :func:`clean_hesitation` runs on the **Vietnamese** MT output. It collapses
  immediate duplicated words (``"một một"`` / ``"một... một"`` -> ``"một"``) and
  turns ellipsis hesitation (``"..."`` / ``"…"``) into a comma, so the TTS never
  reads a stutter or a trailing ``...``.
* :func:`strip_fillers` runs on the **English** source when a segment must be
  re-translated shorter: it drops discourse fillers and parentheticals before
  handing the text back to the translator.

Both return ``(text, n_removed)`` and are pure, so they are cheap to test.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

__all__ = ["FILLERS", "strip_fillers", "clean_hesitation"]

# English filler tokens/phrases dropped before a shortening re-translation. The
# trailing-comma forms intentionally consume the comma ("well," / "so,"). The
# list is deliberately conservative — only meaning-preserving discourse markers,
# never paraphrasing (M3-C1 MA ruling).
FILLERS: tuple[str, ...] = (
    "you know",
    "I mean",
    "actually",
    "basically",
    "literally",
    "kind of",
    "sort of",
    "but also",
    "well,",
    "so,",
    "just",
    "really",
    "and then",
)

_PAREN_RE = re.compile(r"\([^)]*\)")
_WS_RE = re.compile(r"\s+")
_ELLIPSIS_RE = re.compile(r"\s*\.\.\.+\s*")
_LEADING_JUNK_RE = re.compile(r"^[\s,;]+")
_TRAILING_JUNK_RE = re.compile(r"[\s,;]+$")
# Characters stripped from a token before comparing two neighbours.
_DUP_STRIP = ".,;:!?\"'()[]{}…-–—"


def _filler_pattern(fillers: Iterable[str]) -> re.Pattern[str]:
    """Whole-word, case-insensitive alternation of ``fillers`` (longest first)."""
    parts = sorted({re.escape(filler) for filler in fillers}, key=len, reverse=True)
    return re.compile(r"(?<!\w)(?:" + "|".join(parts) + r")(?!\w)", re.IGNORECASE)


def strip_fillers(
    text: str, fillers: Iterable[str] = FILLERS
) -> tuple[str, int]:
    """Drop English fillers and parentheticals; return ``(text, n_removed)``.

    ``n_removed`` counts every removed filler occurrence plus every removed
    ``(...)`` parenthetical. Whitespace is collapsed and leading/trailing
    separators left behind by a removed token are trimmed.
    """
    if not text:
        return text, 0

    removed = 0
    out, count = _PAREN_RE.subn(" ", text)
    removed += count

    filler_list = list(fillers)
    if filler_list:
        out, count = _filler_pattern(filler_list).subn(" ", out)
        removed += count

    out = _WS_RE.sub(" ", out)
    out = _LEADING_JUNK_RE.sub("", out)
    out = _TRAILING_JUNK_RE.sub("", out)
    return out, removed


def _normalized_token(token: str) -> str:
    """Lowercase a token with its surrounding punctuation stripped."""
    return token.strip(_DUP_STRIP).lower()


def _collapse_duplicates(text: str) -> tuple[str, int]:
    """Collapse immediately repeated words; return ``(text, n_collapsed)``.

    Comparison is case- and punctuation-insensitive, so ``"một, một"`` (from
    ``"một... một"``) collapses. When the duplicate carries sentence-final
    punctuation the longer token is kept (``"một một."`` -> ``"một."``).
    """
    tokens = text.split()
    out: list[str] = []
    collapsed = 0
    for token in tokens:
        normalized = _normalized_token(token)
        if out and normalized and normalized == _normalized_token(out[-1]):
            collapsed += 1
            if len(token) >= len(out[-1]) and token[-1:] in ".!?":
                out[-1] = token
            continue
        out.append(token)
    return " ".join(out), collapsed


def clean_hesitation(text: str) -> tuple[str, int]:
    """Clean Vietnamese hesitation artifacts; return ``(text, n_fixed)``.

    Ellipsis runs become a comma, immediate duplicated words are collapsed, and
    leading/trailing commas left over from either are stripped. ``n_fixed``
    counts each ellipsis repaired plus each duplicated word removed. Text with
    no hesitation passes through byte-identical.
    """
    if not text:
        return text, 0

    fixed = 0
    out = text.replace("…", "...")
    out, count = _ELLIPSIS_RE.subn(", ", out)
    fixed += count

    out, count = _collapse_duplicates(out)
    fixed += count

    out = _LEADING_JUNK_RE.sub("", out)
    out = _TRAILING_JUNK_RE.sub("", out)
    out = _WS_RE.sub(" ", out).strip()

    if out == text:
        return text, 0
    return out, fixed
