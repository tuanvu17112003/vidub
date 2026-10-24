"""TTS pronunciation lexicon: write one form, speak another (M3-C3).

A dubbing script keeps the **written** Vietnamese form everywhere a human reads
it (``script.json`` / ``subtitles.vi.srt`` / ``script.tsv``), but a TTS engine
often mangles acronyms (``VRAM`` read letter-by-letter, ``ChatGPT`` spelled out)
and users have no way to steer that. The pronunciation lexicon is a small UTF-8
TSV mapping the written form to the **phonetic** form the TTS should say::

    VRAM<TAB>Vê-ram
    # lines starting with '#' are comments
    GPU<TAB>Gờ-Pê-U

Only :func:`apply_pronunciation` — called by the speak/timing stages on the text
handed to the engine right before synthesis — consumes it. The written form is
never overwritten in any artifact, so subtitles stay readable.

Blank lines and ``#`` comment lines are skipped. Invalid rows (missing tab, empty
written form, or empty spoken form) are warned about and skipped instead of
aborting the run, so :func:`load_pronunciation` never crashes on a malformed
file. A later row wins when the same written form is defined twice. Terms are
tried longest-first (``ChatGPT`` before ``GPT``).
"""

from __future__ import annotations

import logging
from pathlib import Path

from vidub.utils.glossary import apply_glossary

__all__ = ["load_pronunciation", "parse_pronunciation", "apply_pronunciation"]

log = logging.getLogger(__name__)

# TSV field separator; a row is ``written<TAB>spoken`` with optional extra
# columns that are ignored (e.g. a trailing note column).
_SEP = "\t"


def parse_pronunciation(text: str) -> tuple[dict[str, str], list[str]]:
    """Parse lexicon ``text`` into ``(mapping, problems)``.

    ``problems`` holds one human-readable message per invalid row (line number +
    reason) so callers can warn (CLI) or assert (tests). Comments/blank lines
    never produce a problem; a duplicate written form silently overrides the old
    value.
    """
    mapping: dict[str, str] = {}
    problems: list[str] = []
    for lineno, raw in enumerate((text or "").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = [field.strip() for field in raw.split(_SEP)]
        written = fields[0] if fields else ""
        spoken = fields[1] if len(fields) > 1 else ""
        if len(fields) < 2 or not written or not spoken:
            problems.append(
                f"line {lineno}: expected 'written<TAB>spoken', got {line!r}"
            )
            continue
        mapping[written] = spoken
    return mapping, problems


def load_pronunciation(path: str | Path) -> dict[str, str]:
    """Load a UTF-8 pronunciation TSV, warning about (and skipping) bad rows.

    Raises :class:`FileNotFoundError` when ``path`` does not exist so the caller
    can abort with a clear message. Everything else is best-effort: a file full
    of bad rows yields an empty mapping rather than an exception.
    """
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"pronunciation lexicon not found: {target}")
    mapping, problems = parse_pronunciation(target.read_text(encoding="utf-8"))
    for problem in problems:
        log.warning("pronunciation %s: %s", target, problem)
    if problems:
        log.warning(
            "pronunciation %s: skipped %d invalid row(s), kept %d term(s)",
            target,
            len(problems),
            len(mapping),
        )
    return mapping


def apply_pronunciation(text: str, mapping: dict[str, str]) -> tuple[str, int]:
    """Replace written forms in ``text`` with their phonetic forms for the TTS.

    Matching is case-insensitive and whole-word, and terms are tried
    longest-first — the exact rules of
    :func:`vidub.utils.glossary.apply_glossary`, which this delegates to. Returns
    ``(new_text, n_replacements)``; empty ``text``/``mapping`` pass through
    unchanged (``0`` replacements).
    """
    return apply_glossary(text, mapping)
