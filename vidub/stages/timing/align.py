"""Display-span alignment for multi-member timing units (SA-1-T3)."""

from __future__ import annotations

import logging
from pathlib import Path

from vidub.utils.align import allocate_spans
from vidub.utils.sentences import split_translation
from vidub.utils.text_selection import selected_text

from vidub.stages.timing.units import _TimingUnit

log = logging.getLogger(__name__)


def _word_times(segments) -> list[tuple[float, float, str]]:
    """Flatten ``Segment.words`` into ``(start, end, word)`` triples.

    Malformed / missing entries (the ASR word list allows holes) are skipped;
    negative starts are clamped to ``0`` and an inverted pair is dropped. The
    word text is kept so the cue builder can split on real word boundaries
    (Q-1-T4); a missing ``word`` key yields an empty string.
    """
    times: list[tuple[float, float, str]] = []
    for segment in segments or []:
        for word in getattr(segment, "words", None) or []:
            try:
                start = float(word["start"])
                end = float(word["end"])
            except (KeyError, IndexError, TypeError, ValueError):
                continue
            if start < 0:
                start = 0.0
            if end < start:
                continue
            times.append((start, end, str(word.get("word", ""))))
    return times


def _check_align_words(
    align_words: list[list], member_spans: list[list], placed: tuple[float, float]
) -> bool:
    """Sanity-check an ``align_words`` timeline against the member spans.

    True when the timeline is monotone non-decreasing, every word sits inside the
    placed range ``placed``, and each member span contains its share of the
    words (the members partition the timeline in order). Used to gate what the
    stage persists; a false result simply omits ``align_words``.
    """
    if not align_words:
        return False
    placed_start, placed_end = float(placed[0]), float(placed[1])
    previous = placed_start - 1e-6
    for entry in align_words:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            return False
        try:
            start, end = float(entry[0]), float(entry[1])
        except (TypeError, ValueError):
            return False
        if start < previous - 1e-6 or end < start:
            return False
        if start < placed_start - 1e-3 or end > placed_end + 1e-3:
            return False
        previous = start
    if not member_spans:
        return True
    # Members are ordered and contiguous; a member boundary must fall inside the
    # timeline so each member owns a contiguous run of words.
    cursor = 0
    for span in member_spans:
        if not isinstance(span, (list, tuple)) or len(span) < 3:
            return False
        boundary = float(span[2])
        while cursor < len(align_words) and float(align_words[cursor][1]) <= boundary + 1e-3:
            cursor += 1
    return cursor >= len(align_words) - 1


def _align_unit(
    record: dict,
    unit: _TimingUnit,
    script: list[dict],
    unit_text: str,
    wav_path: Path,
    asr_engine,
    tgt_lang: str,
) -> tuple[str, list[list], list[list]]:
    """Return ``(align, member_spans, align_words)`` for one multi-member unit.

    ``member_spans`` is ``[[row_index, start_abs, end_abs, vi_piece], ...]`` in
    the final placed timeline. With an ASR engine the unit's final WAV is
    transcribed and every boundary is snapped to an ASR word edge
    (``allocate_spans`` over the members' SOURCE word counts); otherwise (or when
    the allocator returns ``None``) the members' own locked script spans, clamped
    into the placed range, are used with :func:`split_translation` pieces. Never
    raises.

    ``align_words`` is the unit's ASR word timeline as ``[[start_abs, end_abs,
    word], ...]`` (absolute seconds, monotone) when the ``asr`` path ran and the
    timeline passes :func:`_check_align_words`, else ``[]``. It is what lets the
    subtitle builder split cues on real word edges (Q-1-T4).
    """
    placed_start = float(record.get("start", 0.0))
    placed_end = float(record.get("end", 0.0))
    members = list(unit.members)
    pieces = split_translation(unit_text, unit.shares)
    if len(pieces) != len(members):
        return "unavailable", [], []

    align = "split"
    spans: list[tuple[float, float]] | None = None
    align_words: list[list] = []
    if asr_engine is not None:
        times: list[tuple[float, float, str]] = []
        try:
            segments = asr_engine.transcribe(
                str(wav_path), language=tgt_lang, word_timestamps=True
            )
            times = _word_times(segments)
        except Exception as exc:  # noqa: BLE001 - alignment must never fail a run
            log.warning(
                "unit %s: display alignment ASR failed (%s); using the split "
                "fallback",
                record.get("idx"),
                exc,
            )
            times = []
        if times:
            weights = [
                len(str(script[index].get("en", "")).split()) for index in members
            ]
            absolute = [
                (placed_start + start, placed_start + end)
                for start, end, _word in times
            ]
            allocated = allocate_spans(weights, absolute)
            if allocated is not None:
                spans = allocated
                align = "asr"
                align_words = [
                    [round(placed_start + start, 3), round(placed_start + end, 3), word]
                    for start, end, word in times
                ]

    if spans is None:
        fallback: list[tuple[float, float]] = []
        for index in members:
            start = float(script[index].get("start", placed_start))
            end = float(script[index].get("end", start))
            start = min(max(start, placed_start), placed_end)
            end = min(max(end, start), placed_end)
            fallback.append((start, end))
        spans = fallback
        align = "split"

    member_spans: list[list] = []
    for position, index in enumerate(members):
        start, end = spans[position]
        piece = str(pieces[position]).strip()
        if not piece:
            piece = selected_text(script[index], "spoken_vi").strip()
        if not piece:
            return "unavailable", [], []
        member_spans.append([int(index), float(start), float(end), piece])
    if align != "asr" or not _check_align_words(
        align_words, member_spans, (placed_start, placed_end)
    ):
        align_words = []
    return align, member_spans, align_words
