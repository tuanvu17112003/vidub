"""Translation glossary: terminology enforcement (stdlib only).

Two complementary mechanisms share the same ``term<TAB>translation`` mapping:

* **output-side** (:func:`apply_glossary`) — the original behaviour: every
  whole-word, case-insensitive occurrence of a source term in the MT output is
  replaced with its enforced translation. It cannot help when the model phrases
  the term differently, because the source term simply is not in the output.
* **source-side** (:func:`splice_source_terms`) — for segments whose *source*
  text contains a glossed term but whose MT output lacks the enforced target
  phrase, the source is split at the term and the surrounding chunks are
  translated and joined around the enforced phrase. This is how a term like
  ``AI assistant -> trợ lý AI`` is guaranteed even when the model would phrase
  it differently.

The file is a small UTF-8 TSV::

    vidub<TAB>ViDub
    # lines starting with '#' are comments
    AI assistant<TAB>trợ lý AI

Blank lines and ``#`` comment lines are skipped. Invalid rows (missing tab,
empty term, or empty translation) are skipped with a warning instead of
aborting the run, so :func:`load_glossary` never crashes on a malformed file.
A later row wins when the same term is defined twice.

Terms are tried longest-first so a shorter term can never shadow a longer term
that starts with it (``trí tuệ`` matches before ``trí``).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path

__all__ = [
    "load_glossary",
    "parse_glossary",
    "apply_glossary",
    "find_source_terms",
    "split_source_on_term",
    "splice_translation",
    "splice_source_terms",
    "apply_source_aliases",
]

log = logging.getLogger(__name__)

# TSV field separator. Extra columns are notes unless the third starts with
# ``aliases=``: an explicit pipe-separated list of known MT mistranslations.
_SEP = "\t"


class Glossary(dict[str, str]):
    """Normal term mapping plus optional, explicitly listed MT-output aliases."""

    def __init__(self) -> None:
        super().__init__()
        self.aliases: dict[str, list[str]] = {}


def parse_glossary(text: str) -> tuple[dict[str, str], list[str]]:
    """Parse glossary ``text`` into ``(mapping, problems)``.

    ``problems`` holds one human-readable message per invalid row (line number +
    reason) so callers can warn (CLI) or assert (tests). Comments/blank lines
    never produce a problem; a duplicate term silently overrides the old value.
    """
    mapping = Glossary()
    problems: list[str] = []
    for lineno, raw in enumerate((text or "").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = [field.strip() for field in raw.split(_SEP)]
        term = fields[0] if fields else ""
        translation = fields[1] if len(fields) > 1 else ""
        if len(fields) < 2 or not term or not translation:
            problems.append(
                f"line {lineno}: expected 'term<TAB>translation', got {line!r}"
            )
            continue
        mapping[term] = translation
        # Existing third-column notes retain their meaning. Only this explicit
        # prefix opts into source-conditioned target-side correction.
        if len(fields) > 2 and fields[2].startswith("aliases="):
            mapping.aliases[term] = [
                value.strip() for value in fields[2][8:].split("|") if value.strip()
            ]
        else:
            mapping.aliases.pop(term, None)
    return mapping, problems


def load_glossary(path: str | Path) -> dict[str, str]:
    """Load a UTF-8 TSV glossary, warning about (and skipping) invalid rows.

    Raises :class:`FileNotFoundError` when ``path`` does not exist so the caller
    can abort with a clear message. Everything else is best-effort: a file full
    of bad rows yields an empty mapping rather than an exception.
    """
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"glossary not found: {target}")
    mapping, problems = parse_glossary(target.read_text(encoding="utf-8"))
    for problem in problems:
        log.warning("glossary %s: %s", target, problem)
    if problems:
        log.warning(
            "glossary %s: skipped %d invalid row(s), kept %d term(s)",
            target,
            len(problems),
            len(mapping),
        )
    return mapping


def _compile(mapping: dict[str, str]) -> tuple[re.Pattern[str], dict[str, str]]:
    """Compile the whole-word alternation for ``mapping`` (longest term first)."""
    terms = sorted(mapping, key=len, reverse=True)
    # ``(?<!\w)``/``(?!\w)`` are the Unicode-aware word boundaries; unlike ``\b``
    # they also work when a term itself starts/ends with a non-word character
    # (e.g. ``C++``).
    pattern = re.compile(
        "|".join(f"(?<!\\w){re.escape(term)}(?!\\w)" for term in terms),
        re.IGNORECASE,
    )
    lookup = {term.lower(): mapping[term] for term in terms}
    return pattern, lookup


def apply_glossary(text: str, mapping: dict[str, str]) -> tuple[str, int]:
    """Enforce ``mapping`` on ``text``; return ``(new_text, n_replacements)``.

    Matching is case-insensitive and whole-word, and the replacement is always
    the mapping's value (so ``VIGIB``/``Vigib`` both become ``ViDub``). Longer
    terms are matched before shorter ones.
    """
    if not text or not mapping:
        return text, 0
    pattern, lookup = _compile(mapping)

    def _substitute(match: re.Match[str]) -> str:
        return lookup[match.group(0).lower()]

    return pattern.subn(_substitute, text)


# --------------------------------------------------------------- source-side
_WS_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?…])")


def _normalize(text: str) -> str:
    """Collapse whitespace so phrase matching is whitespace-insensitive."""
    return _WS_RE.sub(" ", (text or "").strip())


def _term_pattern(term: str) -> re.Pattern[str]:
    """Whole-word, case-insensitive pattern for one exact source term."""
    return re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", re.IGNORECASE)


def _source_term_pattern(term: str) -> re.Pattern[str]:
    """Plural-tolerant source pattern for one term (``GPU`` matches ``GPUs``).

    Only source-side *matching* uses this: the optional trailing ``s`` lets a
    singular term find its English plural (``peasant`` matches ``peasants``,
    ``GPU`` matches ``GPUs``) without weakening the output-side enforcement,
    which stays exact. The matched span includes the ``s``, so the spliced
    source skips it and the singular enforced phrase is what gets joined in.
    """
    return re.compile(r"(?<!\w)" + re.escape(term) + r"s?(?!\w)", re.IGNORECASE)


def find_source_terms(
    source: str, mapping: dict[str, str], max_terms: int = 2
) -> list[tuple[str, str, int, int]]:
    """Source-term occurrences in ``source`` as ``(term, target, start, end)``.

    Terms are matched longest-first (so ``AI assistant`` beats ``AI``) and
    occurrences never overlap; the result is ordered by position and capped at
    ``max_terms`` per segment. Matching is **plural-tolerant**: an English plural
    ``s`` may follow the term (``GPU`` matches ``GPUs``, ``peasant`` matches
    ``peasants``); the singular term is what is reported and spliced.
    """
    if not source or not mapping or max_terms <= 0:
        return []

    hits: list[tuple[str, str, int, int]] = []
    used: list[tuple[int, int]] = []
    for term in sorted(mapping, key=len, reverse=True):
        for match in _source_term_pattern(term).finditer(source):
            start, end = match.start(), match.end()
            if any(not (end <= used_start or start >= used_end) for used_start, used_end in used):
                continue
            hits.append((term, mapping[term], start, end))
            used.append((start, end))
    hits.sort(key=lambda hit: hit[2])
    return hits[:max_terms]


def split_source_on_term(source: str, term: str) -> tuple[str, str]:
    """Split ``source`` around the first occurrence of ``term``.

    Returns ``(before, after)`` with surrounding whitespace stripped; either may
    be empty (term at the start/end of the sentence).
    """
    match = _term_pattern(term).search(source or "")
    if match is None:
        return (source or "").strip(), ""
    return source[: match.start()].strip(), source[match.end() :].strip()


def apply_source_aliases(source: str, text: str, mapping: dict[str, str]) -> str:
    """Correct explicitly listed mistranslations only for terms in this source."""
    aliases = getattr(mapping, "aliases", {})
    if not aliases:
        return text
    for term, target, _start, _end in find_source_terms(source, mapping, max_terms=max(len(source.split()), 1)):
        if _term_pattern(target).search(text):
            continue
        for alias in aliases.get(term, []):
            text = _term_pattern(alias).sub(lambda _match: target, text)
    return text


def splice_translation(tr_before: str, target: str, tr_after: str) -> str:
    """Join the translated chunks around the enforced ``target`` phrase.

    Empty chunks are skipped, punctuation keeps its spacing (``" ."`` becomes
    ``"."``) and the result is whitespace-normalized.
    """
    parts: list[str] = []
    if tr_before and tr_before.strip():
        parts.append(tr_before.strip())
    parts.append(target.strip())
    if tr_after and tr_after.strip():
        parts.append(tr_after.strip())
    text = " ".join(parts)
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    return _normalize(text)


def splice_source_terms(
    source: str,
    text: str,
    mapping: dict[str, str],
    translate_fn: Callable[[str], str],
    max_terms: int = 2,
) -> tuple[str, bool, int]:
    """Enforce source terms on ``text``; return ``(new_text, rebuilt, enforced)``.

    ``enforced`` is the number of glossed source terms found in ``source``
    (0-``max_terms``). When every found term's enforced phrase is already present
    in ``text`` (whitespace-normalized) the text is returned unchanged and
    ``rebuilt`` is ``False``; otherwise the source is split at *all* found terms,
    every text chunk is translated with ``translate_fn`` and the parts are joined
    around the enforced phrases (``rebuilt`` is ``True``).
    """
    hits = find_source_terms(source, mapping, max_terms)
    if not hits:
        return text, False, 0

    normalized = _normalize(text)
    normalized = apply_source_aliases(source, normalized, mapping)
    alias_changed = normalized != _normalize(text)
    if all(_normalize(target) in normalized for _term, target, _start, _end in hits):
        # Already enforced: nothing to rebuild.
        return normalized if alias_changed else text, False, len(hits)

    pieces: list[str] = []
    cursor = 0
    for _term, target, start, end in hits:
        if start > cursor:
            chunk = source[cursor:start].strip()
            if chunk:
                pieces.append(str(translate_fn(chunk)).strip())
        pieces.append(target.strip())
        cursor = end
    if cursor < len(source):
        tail = source[cursor:].strip()
        if tail:
            pieces.append(str(translate_fn(tail)).strip())

    new_text = _SPACE_BEFORE_PUNCT_RE.sub(
        r"\1", " ".join(piece for piece in pieces if piece)
    )
    return _normalize(new_text), True, len(hits)
