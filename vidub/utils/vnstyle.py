"""Deterministic Vietnamese word-order refinement (M3-C2).

Machine translation from English sometimes keeps English constituent order, which
reads wrong in Vietnamese. This module applies a small, conservative,
**rule-based** pass to the MT/spliced output (never to passthrough rows) *after*
:func:`vidub.utils.textclean.clean_hesitation`:

* **time-adverbial fronting** — the user-reported example
  ``"Bạn có thể sau đó hoàn thành…"`` becomes
  ``"Sau đó, bạn có thể hoàn thành…"``: the time adverbial moves to the sentence
  head (capitalised, followed by ``", "``) and the subject drops to lower case.
  The optional ``"có thể"`` is handled in **both** orders — ``"Họ sau đó có thể
  rời đi."`` is fronted the same way.
* **repeated leading adverbials** collapse: ``"Sau đó, sau đó, bạn…"`` becomes
  ``"Sau đó, bạn…"`` (only *identical* adverbials collapse, so ``"Sau đó, bây
  giờ…"`` is left alone).

Immediate ``"một … một"`` word repeats are *not* handled here: they are already
collapsed by :func:`~vidub.utils.textclean.clean_hesitation`, which runs first.

Every helper is pure and conservative: text with no matching rule is returned
**byte-identical**, so default-path outputs are unchanged. Rules are listed in
the module-level rule table below — add a pattern only together with a unit
test; a wrong reordering is worse than no reordering at all.
"""

from __future__ import annotations

import re

__all__ = ["SUBJECTS", "ADVERBIALS", "apply_vnstyle"]

# ---------------------------------------------------------------- rule table
# The pattern table below is the whole rule set. Extending it means: add the
# subject/adverbial to one of these tuples (or a new compiled pattern), then add
# a unit test in tests/test_vnstyle.py — see the module docstring.
SUBJECTS: tuple[str, ...] = ("Bạn", "Chúng ta", "Chúng tôi", "Người ta", "Họ")
ADVERBIALS: tuple[str, ...] = ("sau đó", "sau đây", "hiện tại", "hiện nay", "bây giờ")

_SUBJECT_ALT = "|".join(
    re.escape(subject) for subject in sorted(SUBJECTS, key=len, reverse=True)
)
_ADVERB_ALT = "|".join(
    re.escape(adverb) for adverb in sorted(ADVERBIALS, key=len, reverse=True)
)

# "Subject [có thể] ADV rest" -> "ADV, subject [có thể] rest". The optional
# "có thể" may sit either side of the adverbial: with it before, group("aux")
# captures it; with it after, it stays in ``rest`` and is re-emitted verbatim.
_TIME_FRONT_RE = re.compile(
    rf"^(?P<subject>(?:{_SUBJECT_ALT}))\s+"
    rf"(?:(?P<aux>có thể)\s+)?"
    rf"(?P<adv>(?:{_ADVERB_ALT}))\b"
    rf"[\s,]*(?P<rest>.*)$",
    re.IGNORECASE,
)

# "ADV, ADV," -> "ADV, " (backreference keeps it to the *same* adverbial).
_REPEATED_ADVERB_RE = re.compile(
    rf"^(?P<first>(?:{_ADVERB_ALT}))\b,?\s+(?P=first)\b,?\s*",
    re.IGNORECASE,
)

_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?…])")
_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Collapse whitespace and the space an inserted comma may leave behind."""
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    return _WS_RE.sub(" ", text).strip()


def _capitalize(word: str) -> str:
    """Upper-case the first character, leaving the rest untouched."""
    return word[:1].upper() + word[1:]


def _front_time_adverbial(text: str) -> tuple[str, int]:
    """Move a leading subject's time adverbial to the sentence head."""
    match = _TIME_FRONT_RE.match(text)
    if match is None:
        return text, 0

    adv = match.group("adv")
    subject = match.group("subject").lower()
    aux = match.group("aux")
    rest = match.group("rest").strip()
    head = f"{subject} {aux}" if aux else subject
    return _normalize(f"{_capitalize(adv)}, {head} {rest}"), 1


def _collapse_repeated_adverbial(text: str) -> tuple[str, int]:
    """Collapse an immediately repeated leading time adverbial (``Sau đó, sau đó``)."""
    fixes = 0
    while True:
        match = _REPEATED_ADVERB_RE.match(text)
        if match is None:
            break
        text = f"{_capitalize(match.group('first'))}, {text[match.end():]}"
        fixes += 1
    return (_normalize(text), fixes) if fixes else (text, 0)


def apply_vnstyle(text: str) -> tuple[str, int]:
    """Apply the Vietnamese word-order rules; return ``(text, n_fixes)``.

    Rules run in a fixed order (repeated-adverbial collapse, then time fronting,
    then a second collapse in case fronting exposed one) so the result is
    deterministic. ``n_fixes`` counts each rule application. Text that matches no
    rule is returned unchanged with ``0``.
    """
    if not text:
        return text, 0

    fixes = 0
    out, count = _collapse_repeated_adverbial(text)
    fixes += count
    out, count = _front_time_adverbial(out)
    fixes += count
    out, count = _collapse_repeated_adverbial(out)
    fixes += count

    if fixes == 0 or out == text:
        return text, 0
    return out, fixes
