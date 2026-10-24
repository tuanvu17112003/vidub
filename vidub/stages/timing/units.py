"""Sentence-unit grouping for the timing stage (UA-1 / SA-1)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from vidub.manifest import JobManifest
from vidub.utils.sentences import Unit, group_units
from vidub.utils.text_selection import selected_text

from vidub.stages.timing.constants import UNITS_NAME

log = logging.getLogger(__name__)


def compute_coverage(
    script: list[dict],
    speech_by_idx: dict[int, dict],
    members: list[list[int]] | None = None,
) -> float:
    """Fraction of spoken slot time filled by dubbed speech audio.

    For each script segment ``min(speech_dur, slot_dur)`` is the audible part of
    its slot; the ratio of the sums is the coverage. Returns ``1.0`` when there
    are no (or zero-length) slots.

    ``members`` adapts the same measure to the sentence-unit path: each entry
    lists the script indices of one unit, so a unit's slot is the sum of its
    **member spans** and its speech is the sum of their durations. The fraction
    is then computed per unit; when every unit is a single segment it is
    identical to the per-segment result above.
    """
    groups = (
        members if members is not None else [[index] for index in range(len(script))]
    )
    total_slot = 0.0
    covered = 0.0
    for group in groups:
        slot = 0.0
        speech_dur = 0.0
        for index in group:
            if not 0 <= index < len(script):
                continue
            seg = script[index]
            slot += max(
                float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)), 0.0
            )
            speech = speech_by_idx.get(index)
            if speech is not None:
                speech_dur += float(speech.get("duration", 0.0))
        total_slot += slot
        covered += min(speech_dur, slot)
    if total_slot <= 0.0:
        return 1.0
    return covered / total_slot


def _missing_indices(script_len: int, present) -> list[int]:
    """Script segment indices absent from ``present`` (0-based positions)."""
    seen = set(present)
    return [index for index in range(script_len) if index not in seen]


def _group_script_units(script: list[dict]) -> list[Unit]:
    """Group the translated script rows into the translate stage's sentence units.

    The rows are rebuilt in the shape :func:`vidub.utils.sentences.group_units`
    expects (``text`` is the source ``en`` text), applying the same passthrough
    isolation the translate stage uses (a passthrough row gets a unique speaker
    so it never merges with a translated neighbour). The grouping therefore
    matches the units the translator produced, so joining the member ``vi``
    strings reproduces each unit's translation.
    """
    rows: list[dict] = []
    for index, seg in enumerate(script):
        speaker = seg.get("speaker", "spk0")
        if seg.get("passthrough"):
            speaker = f"__passthrough_{index}__"
        rows.append(
            {
                "text": str(seg.get("en", "")),
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "speaker": speaker,
            }
        )
    return group_units(rows)


def _unit_text(script: list[dict], members: list[int]) -> str:
    """Whitespace-collapsed join of a unit's non-empty member ``vi`` texts.

    A member whose ``vi`` is empty contributes no text but stays in the unit's
    ``members`` list. The join carries the members' written forms; the TTS sees
    the phonetic forms through the same pronunciation pass the speak stage used
    (:func:`apply_pronunciation` inside :func:`fit`'s synth closures).
    """
    pieces = []
    for index in members:
        normalized = " ".join(selected_text(script[index], "spoken_vi").split())
        if normalized:
            pieces.append(normalized)
    return " ".join(pieces)


@dataclass
class _TimingUnit:
    """One timing unit: its member script indices and the text to synthesize.

    ``authoritative`` marks a unit taken verbatim from ``units.json`` (SA-1-T2);
    otherwise it was re-grouped from the script rows (today's behaviour).
    """

    index: int
    members: list[int]
    text: str
    start: float
    end: float
    shares: tuple[float, ...]
    authoritative: bool = False


def _source_shares(script: list[dict], members: list[int]) -> tuple[float, ...]:
    """Member shares by SOURCE (``en``) non-whitespace character count.

    Mirrors :func:`vidub.utils.sentences.group_units`, which derives a unit's
    ``shares`` from the member rows' text (the source text on the translate
    path); the split fallback uses the same fractions.
    """
    counts = [
        sum(1 for char in str(script[index].get("en", "")) if not char.isspace())
        for index in members
    ]
    total = sum(counts)
    if total > 0:
        return tuple(count / total for count in counts)
    return tuple(1.0 / len(members) for _ in members)


def _units_from_file(
    script: list[dict], data: object
) -> list[tuple[list[int], dict]] | None:
    """Parse ``units.json`` into ``(members, entry)`` pairs, or ``None``.

    ``None`` means inconsistent: a malformed entry, an empty ``indices`` list, an
    out-of-range index, or a coverage mismatch (a row index missing, duplicated,
    or out of range). The whole stage then falls back to re-grouping.
    """
    if not isinstance(data, dict):
        return None
    entries = data.get("units")
    if not isinstance(entries, list) or not entries:
        return None
    parsed: list[tuple[list[int], dict]] = []
    seen: list[int] = []
    for entry in entries:
        if not isinstance(entry, dict):
            return None
        indices = entry.get("indices")
        if not isinstance(indices, list) or not indices:
            return None
        try:
            members = [int(index) for index in indices]
        except (TypeError, ValueError):
            return None
        if any(index < 0 or index >= len(script) for index in members):
            return None
        parsed.append((members, entry))
        seen.extend(members)
    if sorted(seen) != list(range(len(script))):
        return None
    return parsed


def _timing_units(
    script: list[dict], manifest: JobManifest, artifacts_dir: Path
) -> tuple[list[_TimingUnit], bool]:
    """Build the timing units from ``units.json`` when authoritative, else regroup.

    ``units.json`` is authoritative when it exists and is consistent (its
    ``indices`` partition every script row exactly once). Then the unit's ``vi``
    is the synthesis text verbatim and its ``start``/``end`` the window source,
    and :func:`group_units` is **not** called. A missing or inconsistent file
    logs a clear warning and falls back to today's re-grouping for the whole
    stage.

    Returns ``(units, authoritative)``.
    """
    units_path = artifacts_dir / manifest.paths.get("units", UNITS_NAME)
    parsed: list[tuple[list[int], dict]] | None = None
    if units_path.exists():
        try:
            data = json.loads(units_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("units.json could not be read (%s); re-grouping", exc)
        else:
            parsed = _units_from_file(script, data)
            if parsed is None:
                log.warning(
                    "units.json is inconsistent with script.json (%d row(s)); "
                    "falling back to re-grouping",
                    len(script),
                )
    else:
        log.warning(
            "units.json not found at %s; falling back to re-grouping (the "
            "translate stage writes it on the sentence-unit path)",
            units_path,
        )

    if parsed is not None:
        units: list[_TimingUnit] = []
        for index, (members, entry) in enumerate(parsed):
            first, last = members[0], members[-1]
            units.append(
                _TimingUnit(
                    index=index,
                    members=members,
                    text=selected_text(entry, "spoken_vi"),
                    start=float(entry.get("start", script[first].get("start", 0.0))),
                    end=float(entry.get("end", script[last].get("end", 0.0))),
                    shares=_source_shares(script, members),
                    authoritative=True,
                )
            )
        return units, True

    units = []
    for index, unit in enumerate(_group_script_units(script)):
        members = list(unit.indices)
        first, last = members[0], members[-1]
        units.append(
            _TimingUnit(
                index=index,
                members=members,
                text=_unit_text(script, members),
                start=float(script[first].get("start", 0.0)),
                end=float(script[last].get("end", 0.0)),
                shares=tuple(unit.shares),
                authoritative=False,
            )
        )
    return units, False
