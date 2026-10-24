"""Shared Vietnamese-language heuristics (stdlib only, no model needed).

Two stages need the same question answered — *does this text look Vietnamese?*

* :mod:`vidub.stages.transcribe` guards against a Vietnamese-specialist ASR
  model (PhoWhisper ``vi-small``/``vi-medium``) hallucinating Vietnamese on
  non-Vietnamese audio (the M2 demo bug: English audio -> Vietnamese transcript
  -> gibberish dub + long stretches of silence).
* :mod:`vidub.stages.translate` passes an already-Vietnamese segment straight
  through instead of asking envit5 to "translate" Vietnamese to Vietnamese.

The check is a diacritic heuristic: Vietnamese is written in the Latin script
with a distinctive set of letters/tone marks (``ă â ê ô ơ ư đ`` plus tone-marked
vowels) that are extremely rare in English. Counting words that carry at least
one such character gives a robust, dependency-free signal.

The threshold and helpers live here in one place so both callers agree.
"""

from __future__ import annotations

import unicodedata

__all__ = [
    "VN_CHARS",
    "VN_MISMATCH_THRESHOLD",
    "vn_diacritic_ratio",
    "detect_mismatch",
]

# Base Vietnamese-only letters plus every tone-marked vowel, lower and upper.
VN_CHARS: frozenset[str] = frozenset(
    "ăâêôơưđ"
    "áàảãạắằẳẵặấầẩẫậ"
    "éèẻẽẹếềểễệ"
    "íìỉĩị"
    "óòỏõọốồổỗộớờởỡợ"
    "úùủũụứừửữự"
    "ýỳỷỹỵ"
    "ĂÂÊÔƠƯĐ"
    "ÁÀẢÃẠẮẰẲẴẶẤẦẨẪẬ"
    "ÉÈẺẼẸẾỀỂỄỆ"
    "ÍÌỈĨỊ"
    "ÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢ"
    "ÚÙỦŨỤỨỪỬỮỰ"
    "ÝỲỶỸỴ"
)

# A word counts as Vietnamese when it carries at least one VN_CHARS character.
# 0.30 is the mismatch threshold: a handful of Vietnamese loanwords/proper nouns
# can legitimately appear in English text, but a genuine wrong-language ASR run
# saturates — the M2 demo's hallucinated transcript measured 0.52, while mixed
# English + one or two Vietnamese names stays well under 0.30. Requiring >= 30 %
# of words to carry diacritics avoids false positives on scattered names.
VN_MISMATCH_THRESHOLD = 0.30


def _alpha_words(text: str) -> list[str]:
    """Split ``text`` into alphabetic word runs (Unicode-aware, NFC-normalised).

    ``"2nd"`` contributes ``"nd"``; punctuation is not part of a word.
    """
    normalized = unicodedata.normalize("NFC", text or "")
    words: list[str] = []
    current: list[str] = []
    for char in normalized:
        if char.isalpha():
            current.append(char)
        elif current:
            words.append("".join(current))
            current = []
    if current:
        words.append("".join(current))
    return words


def vn_diacritic_ratio(text: str) -> float:
    """Fraction of alphabetic words in ``text`` containing >= 1 Vietnamese char.

    Returns ``0.0`` for text with no alphabetic words.
    """
    words = _alpha_words(text)
    if not words:
        return 0.0
    hits = sum(1 for word in words if any(char in VN_CHARS for char in word))
    return hits / len(words)


def detect_mismatch(text: str, src_lang: str) -> str | None:
    """Describe a likely wrong-language ASR result, or ``None`` when it is fine.

    ``src_lang == "vi"`` is always consistent with Vietnamese output, so only
    non-Vietnamese source languages are flagged. When the diacritic ratio reaches
    :data:`VN_MISMATCH_THRESHOLD` the returned string is a human-readable
    explanation (including the measured ratio) suitable for an error message.
    """
    lang = (src_lang or "").strip().lower()
    if lang == "vi" or lang.startswith("vi-"):
        return None
    ratio = vn_diacritic_ratio(text)
    if ratio >= VN_MISMATCH_THRESHOLD:
        return (
            f"the ASR transcript looks Vietnamese ({ratio:.0%} of words carry "
            f"Vietnamese diacritics) but --src-lang is {src_lang!r}"
        )
    return None
