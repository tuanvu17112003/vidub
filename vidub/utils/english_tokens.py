"""Conservative English grammar-token checks; apostrophized names stay intact."""

from __future__ import annotations

import re
from collections.abc import Iterable

_TOKEN = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)+", re.UNICODE)
_CONTRACTION = re.compile(
    r"(?:(?:i|you|we|they|he|she|it|that|there|here|what|who|where|when|how)"
    r"'(?:m|re|ve|ll|d|s)|(?:is|are|was|were|have|has|had|do|does|did|"
    r"ca|could|wo|would|sha|should|must|might|need|ought|ai)n't|let's)",
    re.IGNORECASE,
)
_ACRONYM_POSSESSIVE = re.compile(r"([A-Z]{2,})['’]s")
_ACRONYM_PLURAL = re.compile(r"([A-Z]{2,})s")


def is_contraction(token: str) -> bool:
    """Recognize grammatical contractions without guessing their expansion."""
    return bool(_CONTRACTION.fullmatch(token.replace("’", "'")))


def entity_form(token: str) -> str:
    """Preserve an acronym itself rather than its English possessive suffix."""
    match = _ACRONYM_POSSESSIVE.fullmatch(token) or _ACRONYM_PLURAL.fullmatch(token)
    return match.group(1) if match else token


def grammar_leaks(text: str, protected: Iterable[str] = ()) -> list[str]:
    """English contractions/possessive acronyms in target text, except opted-in terms."""
    allowed = [term.replace("’", "'") for term in protected if term]
    normalized = text.replace("’", "'")
    spans: list[tuple[int, int]] = []
    for term in allowed:
        pattern = re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", re.IGNORECASE)
        spans.extend((match.start(), match.end()) for match in pattern.finditer(normalized))
    return [
        match.group(0)
        for match in _TOKEN.finditer(text)
        if (is_contraction(match.group(0)) or _ACRONYM_POSSESSIVE.fullmatch(match.group(0)))
        and not any(start <= match.start() and match.end() <= end for start, end in spans)
    ]
