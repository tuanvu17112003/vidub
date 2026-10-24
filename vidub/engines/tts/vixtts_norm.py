"""Minimal Vietnamese text normalization for ViXTTS.

ViXTTS needs "clean" Vietnamese input. The reference demo
(https://github.com/thinhlpg/vixtts-demo, MPL-2.0) delegates number/date/unit
normalization to the ``vinorm`` package and then applies a few symbol/quote
cleanups. We cannot add ``vinorm`` as a dependency, so this module vendors the
demo's small cleanup step and ships a minimal, dependency-free Vietnamese
number/symbol reader in its place.

Attribution: the replacement chain in :func:`normalize_text` is adapted from
``normalize_vietnamese_text`` in https://github.com/thinhlpg/vixtts-demo
(MPL-2.0). It is intentionally minimal — numbers and a handful of symbols — as
required by the ViDub TTS scope.
"""

from __future__ import annotations

import re

__all__ = [
    "normalize_text",
    "normalize_numbers",
    "normalize_symbols",
    "number_to_vietnamese",
]

_DIGITS = [
    "không",
    "một",
    "hai",
    "ba",
    "bốn",
    "năm",
    "sáu",
    "bảy",
    "tám",
    "chín",
]

# Symbols that should be spoken rather than left for the tokenizer.
_SYMBOL_MAP = {
    "%": " phần trăm",
    "&": " và ",
    "+": " cộng ",
    "=": " bằng ",
    "@": " a còng ",
    "#": " thăng ",
    "$": " đô la ",
    "€": " euro ",
    "°": " độ ",
}

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
_DECIMAL_RE = re.compile(r"(\d+)[.,](\d{1,2})")


def _read_two(n: int) -> str:
    """Read an integer 0..99 in Vietnamese."""
    if n < 10:
        return _DIGITS[n]
    tens, unit = divmod(n, 10)
    head = "mười" if tens == 1 else f"{_DIGITS[tens]} mươi"
    if unit == 0:
        return head
    if unit == 1 and tens > 1:
        tail = "mốt"
    elif unit == 5:
        tail = "lăm"
    else:
        tail = _DIGITS[unit]
    return f"{head} {tail}"


def _read_hundreds(n: int) -> str:
    """Read an integer 0..999 in Vietnamese."""
    hundreds, rest = divmod(n, 100)
    if hundreds == 0:
        return _read_two(rest)
    head = f"{_DIGITS[hundreds]} trăm"
    if rest == 0:
        return head
    if rest < 10:
        return f"{head} lẻ {_DIGITS[rest]}"
    return f"{head} {_read_two(rest)}"


_SCALES = ((10**9, "tỷ"), (10**6, "triệu"), (10**3, "nghìn"))


def number_to_vietnamese(value: int) -> str:
    """Read a non-negative integer aloud in Vietnamese (best effort)."""
    value = int(value)
    if value == 0:
        return "không"
    if value < 0:
        return f"âm {number_to_vietnamese(-value)}"

    parts: list[str] = []
    remaining = value
    for scale, name in _SCALES:
        if remaining >= scale:
            count, remaining = divmod(remaining, scale)
            parts.append(f"{number_to_vietnamese(count)} {name}")
    if remaining:
        parts.append(_read_hundreds(remaining))
    return " ".join(parts)


def normalize_numbers(text: str) -> str:
    """Replace digit runs with their Vietnamese reading (decimals as "phẩy")."""

    def replace(match: re.Match[str]) -> str:
        raw = match.group(0)
        decimal = _DECIMAL_RE.fullmatch(raw)
        if decimal:
            whole, frac = decimal.groups()
            spoken = " ".join(_DIGITS[int(digit)] for digit in frac)
            return f"{number_to_vietnamese(int(whole))} phẩy {spoken}"
        digits = re.sub(r"[.,]", "", raw)
        if not digits.isdigit() or len(digits) > 15:
            return raw
        return number_to_vietnamese(int(digits))

    return _NUMBER_RE.sub(replace, text)


def normalize_symbols(text: str) -> str:
    """Expand a handful of common symbols into Vietnamese words."""
    for symbol, words in _SYMBOL_MAP.items():
        text = text.replace(symbol, words)
    return text


def normalize_text(text: str) -> str:
    """Full normalization pipeline applied before ViXTTS inference."""
    if not text:
        return ""
    text = normalize_numbers(text)
    text = normalize_symbols(text)
    text = (
        text.replace("..", ".")
        .replace("!.", "!")
        .replace("?.", "?")
        .replace(" .", ".")
        .replace(" ,", ",")
        .replace('"', "")
        .replace("'", "")
        .replace("AI", "Ây Ai")
        .replace("A.I", "Ây Ai")
    )
    return " ".join(text.split())
