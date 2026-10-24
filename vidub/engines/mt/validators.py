"""Shared deterministic MT output validators (LLM proposes, validators decide).

These three dependency-free checks -- language, numbers, length -- gate every
candidate an LLM-based MT engine proposes. They were extracted verbatim from
:mod:`vidub.engines.mt.ollama_mt` (M4-T5) in M5-T1 so the Ollama and Hunyuan
engines share exactly one implementation; the names stay private to match the
existing import surface (``vidub.engines.mt.ollama_mt`` re-exports them).

Nothing here talks to a model or the network: given the source segment and a
candidate translation the result is fully deterministic, so a bad LLM proposal
can never reach the dub.
"""

from __future__ import annotations

import re
from collections import Counter

from vidub.utils.langcheck import vn_diacritic_ratio

# Minimum Vietnamese diacritic ratio for an LLM output to look Vietnamese; the
# same 0.30 heuristic the ASR language guard uses (vidub.utils.langcheck).
LANG_RATIO_THRESHOLD = 0.30
# An LLM output may not exceed this many times the source word count.
LENGTH_RATIO = 2.2

_DIGIT_RUN_RE = re.compile(r"\d+")
_NON_DIGIT_RE = re.compile(r"\D")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")


def _missing_numbers(src: str, translated: str) -> list[str]:
    """Missing numeric tokens, with multiplicity and decimal punctuation.

    Dot/comma substitutions allow Vietnamese numeric formatting. Keep token
    boundaries: 16 is not 160, and separate 1 and 6 cannot become 16.
    """
    expected = Counter(n.replace(",", ".") for n in _NUMBER_RE.findall(src or ""))
    actual = Counter(n.replace(",", ".") for n in _NUMBER_RE.findall(translated or ""))
    return list((expected - actual).elements())


def _lang_ok(text: str, tgt: str, src: str = "en") -> bool:
    """True when ``text`` plausibly is in the target language.

    For a Vietnamese target the dependency-free diacritic heuristic is reused
    (:func:`vidub.utils.langcheck.vn_diacritic_ratio` >= 0.30). A Vietnamese
    *source* already yields Vietnamese-looking text, so the check is skipped.
    Any other target passes (no cheaper heuristic exists).
    """
    if (tgt or "").strip().lower() != "vi":
        return True
    if (src or "").strip().lower().startswith("vi"):
        return True
    return vn_diacritic_ratio(text) >= LANG_RATIO_THRESHOLD


def _digits(text: str) -> str:
    """All digits of ``text`` (``"GPU 16 GB"`` -> ``"16"``)."""
    return _NON_DIGIT_RE.sub("", text or "")


def _numbers_ok(src: str, vi: str) -> bool:
    """True when every numeric token survives, including repeated occurrences.

    A source without digits always passes. This stricter engine check keeps
    token boundaries instead of searching concatenated digits.
    """
    return not _missing_numbers(src, vi)


def _length_ok(src: str, vi: str) -> bool:
    """True when ``vi`` is at most 2.2x the source word count."""
    limit = LENGTH_RATIO * max(len((src or "").split()), 1)
    return len((vi or "").split()) <= limit


def _validate(text: str, src: str, src_lang: str, tgt_lang: str) -> str | None:
    """Return a human-readable failure reason, or ``None`` when ``text`` passes.

    Runs the three validators in a fixed order (language, numbers, length) so a
    failed segment gets one precise reason for its targeted re-ask.
    """
    if not (text or "").strip():
        return "the translation is empty"
    if not _lang_ok(text, tgt_lang, src_lang):
        return f"the translation does not look like {tgt_lang} (too few diacritics)"
    if not _numbers_ok(src, text):
        missing = _missing_numbers(src, text)
        return "the translation drops source number(s): " + ", ".join(missing)
    if not _length_ok(src, text):
        return (
            "the translation is far longer than the source "
            f"(over {LENGTH_RATIO}x words)"
        )
    return None
