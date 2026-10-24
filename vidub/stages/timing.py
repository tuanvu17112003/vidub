"""S4 Timing: fit each spoken segment into its original time slot (v2, M3-C2).

The old policy was binary — ``ratio <= 1.25`` untouched, otherwise ``atempo``
capped at ``1.35`` — which produced 4 rushed segments bookended by 1.0
neighbors (0.35 neighbor jitter) and fitted audio that overran its slot and
bled into the next segment. The v2 policy (see :mod:`vidub.utils.fit`):

* a segment is left untouched (``"asis"``) when it is within
  :data:`~vidub.utils.fit.SPEED_AS_IS` of its **availability window**;
* it is time-stretched with ``atempo`` up to
  :data:`~vidub.utils.fit.SPEED_CAP`;
* longer than that, the stage makes **one re-translation retry**
  (:func:`vidub.utils.textclean.strip_fillers` on the source, re-translate,
  :func:`~vidub.utils.textclean.clean_hesitation`, re-synthesise just that
  segment) and, only if it still cannot fit, **trims** the fitted audio at the
  window end with a WARNING.

The availability window runs from the segment start to the **next** segment's
start (or the video end for the last segment), so a segment may use the
following silence instead of being squeezed. Applying the stretch is chained
(smoothed) across neighbours, but smoothing may only choose among stretches that
**fit** the window (it never goes below the needed ``speech_dur / window``).
Fitted audio is trimmed at the window end, and an ``"asis"`` plan that would
still overrun is upgraded to a minimal stretch, so the **no-bleed invariant**
holds: no segment's audio extends past the next segment's start (or the video
end), while truly-fitting rows stay byte-identical.

**Tail safety (M3-C2).** A trim is no longer a blind cut: the final stretch is
analysed with ffmpeg ``silencedetect`` and, when a pause starts at/after
``window - 0.6 s`` (or straddles the window), the cut lands there instead
(``"smart_trim": true``) so no sentence ends mid-word. A cut at a pause is a
smart trim; a blind cut (no silence found) stays flagged ``"clipped"`` with a
reason. Either way a 60 ms fade-out is applied at the cut. The **final**
segment's window additionally extends ``cfg.tail_pad`` seconds past the video
end (``--tail-pad``), because nothing follows it and a mild overrun past the last
frame is acceptable mux behavior.

**Sentence units (UA-1).** With ``cfg.sentence_units`` on (the default) the
stage first groups the script rows into the **same** sentence units the
translate stage used (:func:`vidub.utils.sentences.group_units`). A
multi-segment unit is synthesized **once** from its joined member ``vi`` texts
into ``speech/unit_XXXX.wav`` and fitted as one placed row, so an intra-sentence
pause (e.g. the ``…đào`` / ``tạo…`` split) is structurally impossible: a unit's
``timed.json`` row carries its ``members`` and no boundary can fall inside it.
Single-segment units are the degenerate case (they reuse the member's pass-1
WAV). ``--no-sentence-units`` skips grouping entirely and is byte-identical to
the per-segment path.

Standalone use::

    python -m vidub.stages.timing <cache>/<hash>/job.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from vidub.config import JobConfig
from vidub.engines.asr import build_asr
from vidub.engines.mt.base import TranslationError
from vidub.engines.tts.piper_tts import PiperTTS
from vidub.manifest import JobManifest
from vidub.stages.speak import (
    PIPER_VOICES,
    build_tts,
    pronunciation_map,
    resolve_voice,
)
from vidub.stages.translate import build_translator
from vidub.utils import ttsqc
from vidub.utils.align import allocate_spans
from vidub.utils.audio import run_ffmpeg, wav_duration
from vidub.engines.mt.validators import _validate
from vidub.utils.fit import (
    RATE_SPEED_MAX,
    RATE_SPEED_MIN,
    RECENTER_MIN_SLACK,
    available_window,
    concise_char_target,
    effective_speed_cap,
    plan_rate_fit,
    plan_stretch,
    recenter_shift,
)
from vidub.utils.pronounce import apply_pronunciation
from vidub.utils.sentences import Unit, group_units, split_translation
from vidub.utils.textclean import clean_hesitation, strip_fillers
from vidub.utils.units_guard import guard_signals
from vidub.utils.text_selection import selected_text

__all__ = ["fit", "fit_segment", "compute_coverage", "run_tts_qc", "main"]

log = logging.getLogger(__name__)

SPEECH_NAME = "speech.json"
SCRIPT_NAME = "script.json"
TIMED_NAME = "timed.json"
TIMING_NAME = "timing.json"
# Authoritative sentence-unit map written by the translate stage (SA-1-T2). When
# present and consistent it supersedes ``group_units`` for unit membership/text.
UNITS_NAME = "units.json"
# Per-clip TTS QC sidecar + review packets (M4-T7, QUALITY-ROADMAP.md §3.4/P1.7).
TTS_QC_NAME = "tts_qc.json"
TTS_QC_REVIEW_SUBDIR = "review/tts"
# Changed generation settings for the one low-MOS retry (viXTTS sampling).
VXTTS_RETRY_TEMPERATURE = 0.65
VXTTS_RETRY_TOP_K = 80
SUBTITLES_ONLY = "subtitles-only"
# Below this playback coverage the dub is audibly incomplete (long silences);
# the M2 demo measured 0.17 when the ASR hallucination collapsed the slots.
SPEECH_COVERAGE_WARN = 0.6
# Tolerance (s) for the no-bleed invariant and float comparisons. Fitted audio
# is hard-trimmed at the window end, so real bleeds are sub-millisecond.
BLEED_TOLERANCE = 0.01
# Minimum speech a non-clipped row must keep before a hard cut (M3-C2).
MIN_RETAINED_SPEECH = 0.05
# Tail-safety (M3-C2). A trim must land in a pause when one exists near the end
# of the stretched audio: silences are detected and a pause that starts within
# _SMART_TRIM_LOOKBACK of the window (or straddles it) becomes the cut. Blind
# cuts stay flagged "clipped"; every cut gets a short fade so it cannot click.
_SMART_TRIM_LOOKBACK = 0.6
_FADE_OUT = 0.06
_SILENCE_NOISE = "-35dB"
_SILENCE_MIN_DUR = 0.15
_SILENCE_START_RE = re.compile(r"silence_start:\s*(-?[0-9.]+)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*(-?[0-9.]+)")
# Default padding added to a fitted clip so a frame-boundary round never bleeds.
_RETRANSLATE_SUFFIX = "_re"
_FIT_SUFFIX = "_fit"
_CLIP_SUFFIX = "_clip"
# FIT-1: a segment re-synthesised at a fitted rate lands in ``seg_XXXX_ls.wav``.
_RATE_SUFFIX = "_ls"
# UA-1: a multi-segment sentence unit is synthesised once into
# ``speech/unit_XXXX.wav`` (one WAV, one placed row, no intra-sentence pause).
_UNIT_SYNTH_DIR = "speech"
# Reason recorded when a unit cannot fit its window and the unit path does not
# run the per-segment re-translation retry (see ``fit_unit``).
UNIT_RETRANSLATE_OFF = "sentence unit: no re-translation retry"
# Target-language display names for the concise re-translation prompt (Q-1-T3).
_TGT_LANG_NAMES = {
    "vi": "Vietnamese",
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
}


def _synth_retranslation(
    synth_fn: Callable[[str, Path], float] | None,
    resynth_rate_fn: Callable[[str, Path, float], float] | None,
    text: str,
    out_wav: Path,
    length_scale: float,
) -> float:
    """Synthesize a re-translation take, honouring the take's fitted rate.

    The base take may have been re-synthesised at a fitted ``length_scale``
    (FIT-1). Synthesising the re-translation at natural pace would throw that
    away and re-introduce the clip; when a scale-aware synth is available it is
    used at ``length_scale``, otherwise the plain ``synth_fn`` (natural pace).
    Never raises: a missing/failing synth returns ``0.0`` (candidate dropped).
    """
    if resynth_rate_fn is not None and abs(float(length_scale) - 1.0) > 1e-9:
        try:
            return float(resynth_rate_fn(text, out_wav, float(length_scale)))
        except (TranslationError, RuntimeError, OSError, ValueError):
            pass
    if synth_fn is not None:
        return float(synth_fn(text, out_wav))
    return 0.0


def _clamp_rate(value: float) -> float:
    """Clamp a raw ``length_scale`` into ``[1/RATE_SPEED_MAX, 1/RATE_SPEED_MIN]``.

    ``>1`` speaks slower, ``<1`` faster; the bounds keep the voice intelligible
    (see :mod:`vidub.utils.fit`).
    """
    return min(max(value, 1.0 / RATE_SPEED_MAX), 1.0 / RATE_SPEED_MIN)


def _check_mode(mode: str) -> None:
    if mode == SUBTITLES_ONLY:
        raise RuntimeError(
            f"skipped: mode {mode!r} does not run the timing stage"
        )


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


def _speech_rel_dir(speech_by_idx: dict[int, dict]) -> str:
    """Artifacts-relative directory holding the per-segment speech WAVs."""
    for row in speech_by_idx.values():
        parent = Path(str(row.get("file", ""))).parent.as_posix()
        if parent and parent != ".":
            return parent
    return _UNIT_SYNTH_DIR


def _fit_to_window(src_wav: Path, dst_wav: Path, stretch: float, window: float) -> Path:
    """Time-stretch by ``stretch`` then hard-trim at ``window`` seconds.

    ``atempo`` output can land a few milliseconds past ``input / stretch`` due to
    frame rounding; the ``atrim`` guarantees the written file never extends past
    the segment's availability window (the no-bleed invariant). Used for the
    normal ``"stretch"`` action.
    """
    run_ffmpeg(
        [
            "-y",
            "-i",
            str(src_wav),
            "-filter:a",
            f"atempo={stretch:.6f},atrim=end={window:.6f},asetpts=PTS-STARTPTS",
            str(dst_wav),
        ]
    )
    return dst_wav


def detect_silences(
    path: Path,
    noise: str = _SILENCE_NOISE,
    min_dur: float = _SILENCE_MIN_DUR,
) -> list[tuple[float, float]]:
    """Return ``(start, end)`` silence intervals of the audio at ``path``.

    Thin wrapper over ffmpeg's ``silencedetect`` filter (its findings go to
    stderr). A trailing silence that never ends is closed at the file duration,
    so the result is always a list of complete intervals.
    """
    proc = run_ffmpeg(
        [
            "-i",
            str(path),
            "-af",
            f"silencedetect=noise={noise}:d={min_dur}",
            "-f",
            "null",
            "-",
        ]
    )
    stderr = proc.stderr
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", "ignore")
    starts = [float(match.group(1)) for match in _SILENCE_START_RE.finditer(stderr)]
    ends = [float(match.group(1)) for match in _SILENCE_END_RE.finditer(stderr)]
    total = wav_duration(path)
    return [
        (start, ends[index] if index < len(ends) else total)
        for index, start in enumerate(starts)
    ]


def choose_smart_cut(
    silences: list[tuple[float, float]],
    window: float,
    lookback: float = _SMART_TRIM_LOOKBACK,
) -> tuple[float, bool]:
    """Pick where to trim a too-long segment; return ``(cut, smart)``.

    A pause that starts within ``lookback`` seconds **before** the window (or one
    that straddles the window) lets the cut land in silence, which is a "smart"
    trim; otherwise the cut falls back to the window itself (a blind cut the
    caller must flag as ``clipped``). The chosen ``cut`` is never greater than
    ``window``, so the no-bleed invariant still holds.
    """
    low = window - lookback
    starts = [start for start, _end in silences if low <= start <= window]
    if starts:
        return max(starts), True
    if any(start <= window <= end for start, end in silences):
        return window, True
    return window, False


def _fit_clip(src_wav: Path, dst_wav: Path, stretch: float, window: float) -> tuple[float, bool]:
    """Stretch to ``stretch``, then trim at a smart cut with a fade-out.

    Returns ``(cut, smart)``. The stretched audio is written to a temporary file
    so its silence layout can be probed; the cut then falls in a pause when one
    exists near the window end (:func:`choose_smart_cut`) and always gets a
    60 ms ``afade`` out so the abrupt end never clicks.
    """
    stretched = dst_wav.with_name(f"{dst_wav.stem}_stretch{dst_wav.suffix}")
    run_ffmpeg(
        ["-y", "-i", str(src_wav), "-filter:a", f"atempo={stretch:.6f}", str(stretched)]
    )
    try:
        cut, smart = choose_smart_cut(detect_silences(stretched), window)
    finally:
        stretched.unlink(missing_ok=True)

    fade_start = max(cut - _FADE_OUT, 0.0)
    run_ffmpeg(
        [
            "-y",
            "-i",
            str(src_wav),
            "-filter:a",
            (
                f"atempo={stretch:.6f},atrim=end={cut:.6f},asetpts=PTS-STARTPTS,"
                f"afade=t=out:st={fade_start:.6f}:d={_FADE_OUT:.6f}"
            ),
            str(dst_wav),
        ]
    )
    return cut, smart


def fit_segment(
    artifacts_dir: Path,
    seg: dict,
    src_wav: Path,
    *,
    window: float,
    slot_start: float,
    prev_stretch: float,
    cfg: JobConfig,
    en_text: str = "",
    retranslate_fn: Callable[[str], str] | None = None,
    synth_fn: Callable[[str, Path], float] | None = None,
    resynth_rate_fn: Callable[[str, Path, float], float] | None = None,
    rate_length_scale: float = 1.0,
    rate_decision: str = "off",
    vi_text: str = "",
    retranslate_disabled_reason: str | None = None,
    concise_retranslate_fn: Callable[[str, int], str] | None = None,
    src_lang: str = "en",
    tgt_lang: str = "vi",
) -> tuple[dict, dict]:
    """Fit one segment; return ``(timed_record, stats)``.

    ``retranslate_fn`` / ``synth_fn`` are the lazily-built translator and TTS
    callables used for the single shortening retry (``None`` disables it).

    FIT-1: when ``cfg.tts_rate_fit`` and ``resynth_rate_fn`` (Piper only) are
    given and the pre-pass marked this row ``rate_decision == "resynth"``, the
    row's ``vi_text`` is re-synthesized at ``rate_length_scale`` **before** the
    re-translation retry (the rate pass is cheaper and preserves the
    translation). On success the new take replaces the natural audio and flows
    through the unchanged asis/stretch/shorten logic. When ``cfg.recenter_audio``
    a fitted clip is shifted later into up to ``RECENTER_MAX_SHIFT`` seconds of
    window slack without ever crossing the window end.

    The returned ``timed_record`` is the ``timed.json`` row; ``stats`` carries
    the counters the :func:`fit` stage aggregates.
    """
    idx = int(seg.get("idx", 0))
    speech_dur = wav_duration(src_wav)

    current_wav = src_wav
    current_dur = speech_dur
    length_scale_used = 1.0
    rate_action = "off"

    if (
        cfg.tts_rate_fit
        and resynth_rate_fn is not None
        and rate_decision == "resynth"
    ):
        rate_path = src_wav.with_name(
            f"{src_wav.stem}{_RATE_SUFFIX}{src_wav.suffix}"
        )
        try:
            new_dur = float(resynth_rate_fn(vi_text, rate_path, rate_length_scale))
        except (TranslationError, RuntimeError, OSError, ValueError) as exc:
            log.warning(
                "segment %s: rate re-synthesis failed (%s); keeping the natural "
                "take",
                idx,
                exc,
            )
            new_dur = 0.0
        if new_dur > 0:
            current_wav = rate_path
            current_dur = new_dur
            length_scale_used = float(rate_length_scale)
            rate_action = "resynth"
            log.info(
                "segment %s: re-synthesized at length_scale %.3f (natural %.3fs "
                "-> %.3fs, window %.3fs)",
                idx,
                length_scale_used,
                speech_dur,
                current_dur,
                window,
            )
        else:
            rate_action = "failed"
            log.warning(
                "segment %s: rate re-synthesis produced no audio; keeping the "
                "natural take",
                idx,
            )
    elif cfg.tts_rate_fit and rate_decision == "keep":
        rate_action = "keep"

    stretch, action = plan_stretch(
        current_dur,
        window,
        prev_stretch,
        speed_as_is=cfg.speed_as_is,
        speed_cap=cfg.speed_cap,
    )

    retranslated = False
    spoken_text = vi_text
    filler_removals = 0
    clip_reason = ""
    retranslate_target = 0
    retranslate_reason = ""
    # Short-window rule (Q-1-T3 lever 2): the only case a stretch may exceed
    # ``cfg.speed_cap``, and only to avoid a clip.
    window_cap = effective_speed_cap(window, speed_cap=cfg.speed_cap)
    concise_available = concise_retranslate_fn is not None

    if action == "shorten":
        # Lever order is significant (Q-1-T3-C1): the CONCISE re-translation is
        # attempted FIRST, then the bounded short-window stretch, and only then
        # the minimal trim. Trying the extra stretch before the concise retry
        # turned would-be concise rescues into stretches (inflating the
        # ``stretched`` count); concise-first keeps them out of that counter.
        #
        # (a) Lever 1: ONE dedicated concise re-translation with an explicit
        # character target, accepted only when the new take fits the window at or
        # below the effective cap and is non-degenerate. The legacy filler-strip
        # path (retranslate_fn) is preserved verbatim when no concise callable is
        # injected, so existing behaviour is unchanged.
        pre_concise_wav = current_wav
        pre_concise_dur = current_dur
        if (
            cfg.retranslate
            and synth_fn is not None
            and (concise_available or retranslate_fn is not None)
        ):
            try:
                if concise_available:
                    retranslate_target = concise_char_target(
                        vi_text or en_text, window, cap=window_cap
                    )
                    new_text = str(
                        concise_retranslate_fn(en_text, retranslate_target) or ""
                    )
                else:
                    stripped, filler_removals = strip_fillers(en_text)
                    new_text = str(retranslate_fn(stripped) or "")
                new_text, _fixed = clean_hesitation(new_text or "")
                new_text = new_text.strip()
                # The concise path is gated by the shared deterministic validator
                # (non-empty, target language, no number loss); the legacy path is
                # left exactly as before.
                valid = new_text and (
                    not concise_available
                    or (
                        _validate(new_text, en_text, src_lang, tgt_lang) is None
                        and not guard_signals(en_text, new_text)["flags"]
                    )
                )
                if valid:
                    re_path = src_wav.with_name(
                        f"{src_wav.stem}{_RETRANSLATE_SUFFIX}{src_wav.suffix}"
                    )
                    new_dur = _synth_retranslation(
                        synth_fn,
                        resynth_rate_fn,
                        new_text,
                        re_path,
                        length_scale_used,
                    )
                    # A retry must improve actual audio duration. Text length
                    # alone cannot predict the spoken duration of Vietnamese.
                    if 0 < new_dur < current_dur and (
                        not concise_available or new_dur <= window * window_cap
                    ):
                        retranslated = True
                        spoken_text = new_text
                        current_wav = re_path
                        current_dur = new_dur
                        retranslate_reason = (
                            "concise re-translation"
                            if concise_available
                            else "filler re-translation"
                        )
                        stretch, action = plan_stretch(
                            new_dur,
                            window,
                            prev_stretch,
                            speed_as_is=cfg.speed_as_is,
                            speed_cap=window_cap,
                        )
            except (TranslationError, RuntimeError, OSError, ValueError) as exc:
                log.warning(
                    "segment %s: re-translation failed (%s); falling back to a clip",
                    idx,
                    exc,
                )
        # (b) Lever 2: the bounded short-window stretch, ONLY once the concise
        # attempt could not fit the window at/below the effective cap (or was
        # rejected). It stays available on the stretch-only path when the engine
        # cannot re-translate (``retranslate=False`` / no callable).
        if action == "shorten" and window_cap > cfg.speed_cap:
            short_stretch, short_action = plan_stretch(
                current_dur,
                window,
                prev_stretch,
                speed_as_is=cfg.speed_as_is,
                speed_cap=window_cap,
            )
            if short_action == "shorten" and retranslated:
                # The concise take is worse than the take it replaced: retry the
                # short-window stretch on the pre-concise take before clipping.
                back_stretch, back_action = plan_stretch(
                    pre_concise_dur,
                    window,
                    prev_stretch,
                    speed_as_is=cfg.speed_as_is,
                    speed_cap=window_cap,
                )
                if back_action != "shorten":
                    current_wav = pre_concise_wav
                    current_dur = pre_concise_dur
                    retranslated = False
                    spoken_text = vi_text
                    retranslate_reason = ""
                    retranslate_target = 0
                    short_stretch, short_action = back_stretch, back_action
            if short_action != "shorten":
                stretch, action = short_stretch, short_action
                if not retranslate_reason:
                    retranslate_reason = "short-window extra stretch"
        # (c) Lever 3: the minimal trim, the documented last resort.
        if action == "shorten":
            action = "clip"
            if not cfg.retranslate or (
                retranslate_fn is None and not concise_available
            ):
                clip_reason = (
                    retranslate_disabled_reason
                    if retranslate_disabled_reason is not None
                    else "re-translation disabled (--no-retranslate)"
                )
            else:
                clip_reason = (
                    "re-translation still exceeds the speed cap "
                    f"({current_dur / max(window, 1e-6):.2f}x vs {window_cap:.2f}x)"
                )

    # An "asis" plan can still be a few percent longer than the window; upgrade
    # it to a minimal stretch so the no-bleed invariant always holds. Truly
    # fitting segments stay byte-identical (no stretch, same placement).
    if action == "asis" and current_dur > window + 1e-9:
        action = "stretch"
        stretch = min(max(current_dur / max(window, 1e-6), 1.0), cfg.speed_cap)

    clipped = False
    smart_trim = False
    seconds_trimmed = 0.0
    if action == "asis":
        fitted_path = current_wav
        effective = current_dur
        stretch = 1.0
    elif action == "stretch":
        fitted_path = current_wav.with_name(
            f"{current_wav.stem}{_FIT_SUFFIX}{current_wav.suffix}"
        )
        log.info(
            "segment %s: stretching by %.3f (window %.3fs) -> %s",
            idx,
            stretch,
            window,
            fitted_path.name,
        )
        _fit_to_window(current_wav, fitted_path, stretch, window)
        # Trimmed at the window end, so the measured length can never exceed it.
        effective = min(wav_duration(fitted_path), window)
    else:  # clip (last resort)
        stretch = float(cfg.speed_cap)
        fitted_path = current_wav.with_name(
            f"{current_wav.stem}{_CLIP_SUFFIX}{current_wav.suffix}"
        )
        # A smart trim lands the cut in a pause when one exists near the window
        # end; a blind cut (no silence found) is the documented last resort.
        cut, smart_trim = _fit_clip(current_wav, fitted_path, stretch, window)
        effective = min(wav_duration(fitted_path), window)
        seconds_trimmed = max(current_dur / stretch - cut, 0.0)
        if smart_trim:
            log.info(
                "segment %s cannot fit its %.3fs window (needs %.2fx): "
                "smart-trimmed at a silence (%.3fs, %.3fs removed)",
                idx,
                window,
                current_dur / max(window, 1e-6),
                cut,
                seconds_trimmed,
            )
        else:
            clipped = True
            log.warning(
                "segment %s cannot fit its %.3fs window (needs %.2fx): clipped "
                "%.3fs at the window end [%s]",
                idx,
                window,
                current_dur / max(window, 1e-6),
                seconds_trimmed,
                clip_reason,
            )

    start = max(0.0, float(slot_start))
    recentered = False
    recenter_seconds = 0.0
    if cfg.recenter_audio:
        # Leftover slack is the difference between the window end and where the
        # fitted audio would end if placed at the slot start. Shift the clip later
        # (up to RECENTER_MAX_SHIFT s) and never let it cross the window end.
        slack = (start + window) - (start + effective)
        if slack > RECENTER_MIN_SLACK:
            shift = recenter_shift(slack)
            start = max(min(start + shift, start + window - effective), start)
            recentered = True
            recenter_seconds = shift

    record: dict[str, object] = {
        "idx": idx,
        "file": fitted_path.relative_to(artifacts_dir).as_posix(),
        "start": start,
        "end": start + effective,
        "slot_start": float(slot_start),
        "stretch": float(stretch),
        "window": float(window),
        "action": action,
    }
    if retranslated:
        record["retranslated"] = True
        record["vi"] = spoken_text
        # Q-1-T3: which lever produced the take and the concise character target
        # it was asked for (absent on the legacy filler-only path).
        if retranslate_reason:
            record["retranslate_reason"] = retranslate_reason
        if retranslate_target:
            record["retranslate_target"] = int(retranslate_target)
    if clipped:
        record["clipped"] = True
    if smart_trim:
        record["smart_trim"] = True
    # FIT-1 rate/recenter fields (absent when the flags are off: byte-identical).
    if cfg.tts_rate_fit:
        record["length_scale"] = float(length_scale_used)
        record["rate_action"] = rate_action
    if cfg.recenter_audio:
        record["recentered"] = recentered
        if recentered:
            record["recenter_seconds"] = float(recenter_seconds)

    stats = {
        "action": action,
        "stretch": float(stretch),
        "retranslated": retranslated,
        "clipped": clipped,
        "smart_trim": smart_trim,
        "seconds_trimmed": seconds_trimmed,
        "filler_removals": filler_removals,
        "clip_reason": clip_reason,
        "start": start,
        "end": start + effective,
        "rate_action": rate_action,
        "length_scale": float(length_scale_used),
        "recentered": recentered,
        "recenter_seconds": float(recenter_seconds),
    }
    return record, stats


def _assert_no_bleed(
    timed: list[dict], boundaries: dict[int, float]
) -> None:
    """Raise when a fitted segment extends past its availability boundary."""
    for record in timed:
        idx = int(record.get("idx", 0))
        boundary = boundaries.get(idx)
        if boundary is None or boundary <= 0:
            continue
        end = float(record.get("end", 0.0))
        if end > boundary + BLEED_TOLERANCE:
            raise RuntimeError(
                f"timing bleed invariant violated: segment {idx} ends at "
                f"{end:.3f}s, past its boundary {boundary:.3f}s"
            )


def _assert_end_boundary(timed: list[dict], stats: list[dict]) -> None:
    """Raise when a hard mid-word cut is not the documented last resort.

    A segment whose audio lost time to a trim must either have been
    **smart-trimmed** at a silence (``smart_trim``) or be a **blind clip**
    flagged ``clipped`` with a non-empty reason. A trimmed row that is neither
    would silently drop speech mid-word, so it is rejected. Rows that kept at
    least :data:`MIN_RETAINED_SPEECH` seconds of audible audio are left alone.
    """
    for record, info in zip(timed, stats):
        trimmed = float(info.get("seconds_trimmed", 0.0))
        if trimmed <= BLEED_TOLERANCE:
            continue
        if info.get("smart_trim"):
            continue
        idx = int(record.get("idx", 0))
        retained = float(record.get("end", 0.0)) - float(record.get("start", 0.0))
        if not info.get("clipped"):
            if retained >= MIN_RETAINED_SPEECH:
                continue
            raise RuntimeError(
                f"end-boundary invariant violated: segment {idx} was "
                f"hard-trimmed by {trimmed:.3f}s without being smart-trimmed"
            )
        if not str(info.get("clip_reason", "")).strip():
            raise RuntimeError(
                f"end-boundary invariant violated: segment {idx} was clipped "
                f"without a reason"
            )


def _normalized_engine(manifest: JobManifest) -> str:
    """Lowercased TTS engine name (``piper`` aliases normalised)."""
    name = (manifest.tts_engine or "piper").strip().lower()
    return "piper" if name in {"piper", "piper-tts", "piper_tts"} else name


def _piper_voice_files(engine: PiperTTS, voice_path: str) -> tuple[Path, Path] | None:
    """Cached ``(model, config)`` paths for an alternate Piper voice.

    ``PiperTTS`` downloads only the *default* voice's config, and its load branch
    fires (overwriting the config) whenever ``model_path`` is ``None``. So both
    alternate files are fetched here and set together. ``None`` on failure lets
    ``PiperTTS`` fall back to its own download.
    """
    try:
        from huggingface_hub import hf_hub_download

        model = Path(hf_hub_download(engine.repo_id, voice_path))
        config = Path(hf_hub_download(engine.repo_id, voice_path + ".json"))
        return model, config
    except Exception as exc:  # noqa: BLE001 - optional alternate-voice files
        log.warning("could not fetch the alternate Piper voice files (%s)", exc)
        return None


def _synth_variant(
    engine,
    text: str,
    out_wav: Path,
    engine_name: str,
    cfg: JobConfig,
    speaker: str,
) -> float:
    """Re-synthesize one QC retry with changed generation settings.

    * ``vixtts`` -> higher sampling ``temperature``/``top_k`` (temporarily
      overrides the engine module's ``_GEN_KWARGS``, restored afterwards).
    * ``piper``  -> the *other* shipped voice from :data:`DEFAULT_VOICE_MAP`
      (``25hours`` when the active voice is ``vais1000``, else ``vais1000``) by
      reconfiguring the ``PiperTTS`` instance, restored afterwards.

    Test stub engines are neither, so they are simply asked to synth again.
    """
    if engine_name == "vixtts":
        try:
            from vidub.engines.tts import vixtts as vixtts_mod
        except Exception:  # noqa: BLE001 - optional engine module
            vixtts_mod = None
        saved = getattr(vixtts_mod, "_GEN_KWARGS", None) if vixtts_mod else None
        if saved is not None:
            vixtts_mod._GEN_KWARGS = {
                **saved,
                "temperature": VXTTS_RETRY_TEMPERATURE,
                "top_k": VXTTS_RETRY_TOP_K,
            }
            try:
                return float(engine.synth(text, str(out_wav)))  # type: ignore[attr-defined]
            finally:
                vixtts_mod._GEN_KWARGS = saved
        return float(engine.synth(text, str(out_wav)))  # type: ignore[attr-defined]

    if engine_name == "piper" and isinstance(engine, PiperTTS):
        active = resolve_voice(cfg.voice_map or {}, speaker)
        alternate = "25hours" if active == "vais1000" else "vais1000"
        alternate_path = PIPER_VOICES.get(alternate, PIPER_VOICES["vais1000"])
        state = (
            engine.voice_path,
            engine.model_path,
            engine.config_path,
            engine._voice,
        )
        engine.voice_path = alternate_path
        engine._voice = None
        alternate_files = _piper_voice_files(engine, alternate_path)
        if alternate_files is not None:
            engine.model_path, engine.config_path = alternate_files
        else:
            engine.model_path = None
            engine.config_path = None
        try:
            return float(engine.synth(text, str(out_wav)))
        finally:
            (
                engine.voice_path,
                engine.model_path,
                engine.config_path,
                engine._voice,
            ) = state

    return float(engine.synth(text, str(out_wav)))  # type: ignore[attr-defined]


def _write_tts_review(
    review_dir: Path,
    idx: int,
    record: dict,
    vi_text: str,
    original_mos: float,
    retry_mos: float | None,
    clip_path: Path,
) -> None:
    """Write ``seg-<idx>.txt`` + a copy of the final clip for human review."""
    review_dir.mkdir(parents=True, exist_ok=True)
    retry_str = "n/a" if retry_mos is None else f"{float(retry_mos):.3f}"
    lines = [
        f"segment: {idx}",
        f"window: {record.get('window')}",
        f"start: {record.get('start')}",
        f"end: {record.get('end')}",
        f"action: {record.get('action')}",
        f"mos original: {float(original_mos):.3f}",
        f"mos retry: {retry_str}",
        "",
        "vi text:",
        vi_text,
    ]
    (review_dir / f"seg-{idx}.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    try:
        if clip_path.exists():
            shutil.copyfile(clip_path, review_dir / f"seg-{idx}.wav")
    except OSError as exc:  # noqa: BLE001 - a review copy must not fail the run
        log.warning("could not copy TTS QC review clip %s (%s)", clip_path, exc)


def run_tts_qc(
    artifacts_dir: Path,
    timed: list[dict],
    text_by_idx: dict[int, tuple[str, str]],
    cfg: JobConfig,
    resynth: Callable[[str, Path, str], float],
) -> dict:
    """Score every fitted clip, retry low-MOS ones once; return the sidecar dict.

    ``text_by_idx`` maps each timed row's ``idx`` to ``(vi_text, speaker)`` — the
    segment text on the per-segment path, the joined **unit text** on the
    sentence-unit path. ``resynth(text, out_wav, speaker)`` re-synthesizes that
    text with changed settings (provided by :func:`fit`). A clip whose final MOS
    is still below ``cfg.tts_qc_mos_min`` gets a ``review/tts/seg-<idx>.txt`` +
    ``.wav`` packet (``<idx>`` is the unit index on the unit path).
    """
    scored = low_mos = retried = resolved = skipped = 0
    final_mos: list[float] = []
    review_dir = artifacts_dir / TTS_QC_REVIEW_SUBDIR
    # These are generated summaries of the current fitted clips, not a history.
    # Preserve unrelated notes while discarding packets from an earlier run.
    for packet in review_dir.glob("seg-*"):
        if packet.is_file() and re.fullmatch(r"seg-\d+\.(?:txt|wav)", packet.name):
            packet.unlink()
    min_mos = float(cfg.tts_qc_mos_min)

    for record in timed:
        idx = int(record.get("idx", 0))
        clip_path = artifacts_dir / str(record.get("file", ""))
        original_mos = ttsqc.score_wav(clip_path)
        if original_mos is None:
            skipped += 1
            continue
        scored += 1
        vi_text, speaker = text_by_idx.get(idx, ("", "spk0"))

        kept_mos = float(original_mos)
        kept_path = clip_path
        retry_mos: float | None = None

        if original_mos < min_mos:
            low_mos += 1
            retried += 1
            retry_raw = clip_path.with_name(
                f"{clip_path.stem}_qc{clip_path.suffix}"
            )
            try:
                duration = float(resynth(vi_text, retry_raw, speaker))
            except Exception as exc:  # noqa: BLE001 - a retry must not crash
                log.warning(
                    "segment %s: TTS QC retry synthesis failed (%s)", idx, exc
                )
                duration = 0.0
            if duration > 0 and retry_raw.exists():
                retry_mos = None
                try:
                    stretch = float(record.get("stretch", 1.0)) or 1.0
                    window = float(record.get("window", 0.0))
                    fitted = retry_raw.with_name(
                        f"{retry_raw.stem}{_FIT_SUFFIX}{retry_raw.suffix}"
                    )
                    if str(record.get("action")) == "clip":
                        _fit_clip(retry_raw, fitted, stretch, window)
                    else:
                        _fit_to_window(retry_raw, fitted, stretch, window)
                    retry_mos = ttsqc.score_wav(fitted)
                except Exception as exc:  # noqa: BLE001 - a retry must not crash
                    log.warning(
                        "segment %s: TTS QC retry re-fit failed (%s)", idx, exc
                    )
                # keep the better clip; ties keep the original
                if retry_mos is not None and float(retry_mos) > kept_mos:
                    record["file"] = fitted.relative_to(artifacts_dir).as_posix()
                    record["tts_qc_retry"] = True
                    kept_mos = float(retry_mos)
                    kept_path = fitted

            if kept_mos >= min_mos:
                resolved += 1
            else:
                _write_tts_review(
                    review_dir,
                    idx,
                    record,
                    vi_text,
                    original_mos,
                    retry_mos,
                    kept_path,
                )

        final_mos.append(kept_mos)

    mos_mean = round(sum(final_mos) / len(final_mos), 2) if final_mos else None
    return {
        "enabled": True,
        "mos_min": min_mos,
        "scored": scored,
        "low_mos": low_mos,
        "retried": retried,
        "resolved": resolved,
        "skipped": skipped,
        "mos_mean": mos_mean,
    }


def fit(
    manifest_path: str,
    cfg: JobConfig,
    tts=None,
    translator=None,
    asr=None,
) -> JobManifest:
    """Fit every spoken unit into its window and write ``timed.json``.

    With :attr:`~vidub.config.JobConfig.sentence_units` on (the default) the
    script rows are grouped into sentence units. When the translate stage's
    authoritative ``units.json`` is present and consistent it supplies the unit
    ``indices``, the window ``start``/``end`` and the synthesis ``vi`` verbatim
    (:func:`_timing_units`); otherwise the rows are re-grouped with
    :func:`vidub.utils.sentences.group_units`. Each multi-segment unit is
    synthesized **once** from its unit text into ``speech/unit_XXXX.wav`` and
    fitted as a whole (so no placed-row boundary can fall inside a sentence).
    With ``--no-sentence-units`` the per-segment path runs byte-identically to
    before.

    SA-1-T3: with :attr:`~vidub.config.JobConfig.align_display` on, every
    multi-member unit's FINAL audio is force-aligned with the project ASR and its
    ``timed.json`` row gains ``align`` + ``member_spans`` (a real, word-edge
    snapped span per member). The ASR is built lazily and every error degrades the
    affected unit to the split fallback — never the run.

    ``tts`` / ``translator`` / ``asr`` override the lazily-built engines (used by
    tests).
    """
    manifest = JobManifest.load(manifest_path)
    _check_mode(manifest.mode)

    artifacts_dir = manifest.artifacts_dir

    script_rel = manifest.paths.get("script", SCRIPT_NAME)
    script_path = artifacts_dir / script_rel
    if not script_path.exists():
        raise FileNotFoundError(
            f"script not found: {script_path}. Run the speak stage first, e.g. "
            f"'python -m vidub.stages.speak {manifest_path}'."
        )
    script = json.loads(script_path.read_text(encoding="utf-8"))

    speech_rel = manifest.paths.get("speech", SPEECH_NAME)
    speech_path = artifacts_dir / speech_rel
    if not speech_path.exists():
        raise FileNotFoundError(
            f"speech not found: {speech_path}. Run the speak stage first, e.g. "
            f"'python -m vidub.stages.speak {manifest_path}'."
        )

    speech = json.loads(speech_path.read_text(encoding="utf-8"))
    speech_by_idx = {int(seg.get("idx", 0)): seg for seg in speech}

    # Invariant: the dub must never silently drop a script segment. A missing
    # speech entry means TTS skipped/failed for that segment.
    missing_speech = _missing_indices(len(script), speech_by_idx)
    if missing_speech:
        raise RuntimeError(
            f"speech coverage invariant violated: script segment(s) "
            f"{missing_speech} have no speech.json entry (TTS skipped or "
            f"failed); re-run the speak stage"
        )

    starts = [float(seg.get("start", 0.0)) for seg in script]
    video_end = float(manifest.duration) if manifest.duration > 0 else 0.0
    # The FINAL segment may run past the video end: nothing follows it, so a
    # mild overrun is acceptable mux behavior and prevents a mid-word tail cut.
    tail_pad = max(float(cfg.tail_pad), 0.0)

    def window_for(index: int) -> float:
        slot_start = starts[index]
        slot_end = float(script[index].get("end", 0.0))
        next_start = starts[index + 1] if index + 1 < len(script) else None
        end = video_end if next_start is not None else video_end + tail_pad
        return available_window(slot_start, slot_end, next_start, end)

    boundaries = {
        index: (
            starts[index + 1]
            if index + 1 < len(script)
            else video_end + tail_pad
        )
        for index in range(len(script))
    }
    windows = {index: window_for(index) for index in range(len(script))}

    # UA-1 sentence units (default on). The timing stage uses the translate
    # stage's authoritative ``units.json`` when it is present and consistent,
    # else it re-groups the script rows with the SAME ``group_units`` call the
    # translate stage used. One synthesis and one placed row cover a whole
    # sentence: an intra-sentence pause cannot exist because no row boundary can
    # fall inside a unit. Each unit's window runs from its first member's start
    # to the next unit's first member start (or the video end + tail pad), so the
    # no-bleed invariant holds at unit granularity. With --no-sentence-units this
    # is skipped entirely.
    units: list[_TimingUnit] = (
        _timing_units(script, manifest, artifacts_dir)[0] if cfg.sentence_units else []
    )
    unit_starts: dict[int, float] = {}
    unit_members: dict[int, list[int]] = {}
    unit_texts: dict[int, str] = {}
    unit_windows: dict[int, float] = {}
    unit_boundaries: dict[int, float] = {}
    for unit_index, unit in enumerate(units):
        members = list(unit.members)
        unit_start = float(unit.start)
        slot_end = float(unit.end)
        next_first = (
            units[unit_index + 1].members[0]
            if unit_index + 1 < len(units)
            else None
        )
        next_start = starts[next_first] if next_first is not None else None
        end = video_end if next_start is not None else video_end + tail_pad
        unit_starts[unit_index] = unit_start
        unit_members[unit_index] = members
        unit_texts[unit_index] = unit.text
        unit_windows[unit_index] = available_window(
            unit_start, slot_end, next_start, end
        )
        unit_boundaries[unit_index] = (
            starts[next_first] if next_first is not None else video_end + tail_pad
        )

    # Preserve natural pace for fitting takes; only overflow requests a faster
    # rate. Display fragments cannot define a whole video's speaking baseline.
    rate_base = 1.0
    rate_plans: dict[int, tuple[float, str]] = {}
    n_resynth = 0
    if cfg.tts_rate_fit:
        natural_durs: dict[int, float] = {}
        for index in range(len(script)):
            row = speech_by_idx.get(index)
            rel = str(row.get("file", f"speech/seg_{index:04d}.wav"))
            path = artifacts_dir / rel
            natural_durs[index] = wav_duration(path) if path.exists() else 0.0
        for index in range(len(script)):
            ls_row, decision = plan_rate_fit(
                natural_durs[index], windows[index], rate_base
            )
            rate_plans[index] = (ls_row, decision)
            if decision == "resynth":
                n_resynth += 1
        log.info(
            "rate fit: baseline length_scale %.3f, %d row(s) to re-synthesize",
            rate_base,
            n_resynth,
        )

    # Engines are built at most once, and only when a shortening retry runs.
    tts_holder: dict[str, object] = {"engine": tts}
    translator_holder: dict[str, object] = {"engine": translator}
    # The same TTS-only pronunciation lexicon the speak stage used, loaded once
    # and applied to every re-synthesised retry text too (M3-C3).
    pronunciation = pronunciation_map(cfg)

    def _synth(text: str, out_wav: Path) -> float:
        engine = tts_holder["engine"]
        if engine is None:
            engine = build_tts(manifest, cfg)
            tts_holder["engine"] = engine
        tts_text, _spoken = apply_pronunciation(text, pronunciation)
        return float(engine.synth(tts_text, str(out_wav)))  # type: ignore[attr-defined]

    def _retranslate(text: str) -> str:
        engine = translator_holder["engine"]
        if engine is None:
            engine = build_translator(manifest)
            translator_holder["engine"] = engine
        result = engine.translate(  # type: ignore[attr-defined]
            [text], manifest.src_lang, manifest.tgt_lang
        )
        return str(result[0]) if result else ""

    # Q-1-T3 lever 1: the concise re-translation is only wired when the engine can
    # actually see the instruction (a chat-capable LLM engine: hunyuan / ollama,
    # or an injected stub that exposes ``_chat``). Otherwise the legacy
    # filler-strip retry is used unchanged, so envit5 stays byte-identical.
    mt_engine = (
        (manifest.mt_engine or "").strip().lower()
        or (cfg.mt_engine or "").strip().lower()
        or "envit5"
    )
    use_concise = (
        callable(getattr(translator, "_chat", None))
        if translator is not None
        else mt_engine in {"hunyuan", "ollama"}
    )

    def _concise_retranslate(source: str, target_chars: int) -> str:
        """One dedicated concise translation with an explicit character target.

        Uses the engine's chat interface (hunyuan: ``_chat(user)``; ollama:
        ``_chat(system, user)``) with a length instruction; falls back to the
        plain ``translate`` of the filler-stripped source when no chat interface
        is available. Never raises into the caller (fit_segment wraps it too).
        """
        engine = translator_holder["engine"]
        if engine is None:
            engine = build_translator(manifest)
            translator_holder["engine"] = engine
        language = _TGT_LANG_NAMES.get(
            (manifest.tgt_lang or "").strip().lower(), manifest.tgt_lang
        )
        prompt = (
            f"Translate the following text into {language} concisely: at most "
            f"{int(target_chars)} characters, keep the meaning exactly, and "
            f"output only the translation.\n\n{source}"
        )
        chat = getattr(engine, "_chat", None)
        if callable(chat):
            try:
                out = chat(prompt)
            except TypeError:  # ollama-style _chat(system, user, ...)
                try:
                    out = chat("You are a professional translator.", prompt)
                except Exception:  # noqa: BLE001 - a bad candidate, never fatal
                    out = ""
            except Exception:  # noqa: BLE001 - a bad candidate, never fatal
                out = ""
            if str(out or "").strip():
                return str(out).strip()
        stripped, _removed = strip_fillers(source)
        result = engine.translate(  # type: ignore[attr-defined]
            [stripped], manifest.src_lang, manifest.tgt_lang
        )
        return str(result[0]) if result else ""

    # FIT-1: rate re-synthesis is Piper-only (the ``length_scale`` knob is
    # Piper-specific, like the QC alternate-voice retry). Built when at least one
    # row wants re-synthesis, or when a multi-segment unit needs the baseline
    # length_scale for its single fresh synthesis (UA-1).
    has_multi_units = any(len(members) > 1 for members in unit_members.values())
    resynth_rate_fn: Callable[[str, Path, float], float] | None = None
    if (
        cfg.tts_rate_fit
        and (n_resynth > 0 or has_multi_units)
        and _normalized_engine(manifest) == "piper"
    ):
        engine_for_rate = tts_holder["engine"]
        if engine_for_rate is None:
            engine_for_rate = build_tts(manifest, cfg)
            tts_holder["engine"] = engine_for_rate
        if isinstance(engine_for_rate, PiperTTS):

            def _resynth_rate(text: str, out_wav: Path, scale: float) -> float:
                tts_text, _spoken = apply_pronunciation(text, pronunciation)
                return float(
                    engine_for_rate.synth(  # type: ignore[attr-defined]
                        tts_text, str(out_wav), length_scale=float(scale)
                    )
                )

            resynth_rate_fn = _resynth_rate

    timed: list[dict] = []
    stats: list[dict] = []
    # Per-row ``(vi_text, speaker)`` for the TTS-QC retry: the segment text on the
    # opt-out path, the joined unit text on the sentence-unit path.
    text_by_idx: dict[int, tuple[str, str]] = {}
    prev_stretch = 1.0

    if cfg.sentence_units:
        speech_rel_dir = _speech_rel_dir(speech_by_idx)
        for unit_index, unit in enumerate(units):
            members = unit_members[unit_index]
            first = members[0]
            unit_text = unit_texts[unit_index]
            speaker = str(script[first].get("speaker") or "spk0")
            text_by_idx[unit_index] = (unit_text, speaker)

            if len(members) == 1:
                # Degenerate unit: reuse the member's pass-1 WAV and the exact
                # FIT-1 plan the per-segment path would use (the rate fit may
                # still re-synthesize it, see fit_segment).
                member_row = speech_by_idx.get(first)
                rel = str(
                    member_row.get("file", f"speech/seg_{first:04d}.wav")
                    if member_row
                    else f"speech/seg_{first:04d}.wav"
                )
                src_wav = artifacts_dir / rel
                if not src_wav.exists():
                    raise FileNotFoundError(f"segment audio not found: {src_wav}")
                plan_ls, plan_decision = (
                    plan_rate_fit(wav_duration(src_wav), unit_windows[unit_index])
                    if cfg.tts_rate_fit else (1.0, "off")
                )
                record, info = fit_segment(
                    artifacts_dir,
                    {"idx": unit_index},
                    src_wav,
                    window=unit_windows[unit_index],
                    slot_start=unit_starts[unit_index],
                    prev_stretch=prev_stretch,
                    cfg=cfg,
                    en_text=str(script[first].get("en", "")),
                    retranslate_fn=_retranslate,
                    synth_fn=_synth,
                    resynth_rate_fn=resynth_rate_fn,
                    rate_length_scale=plan_ls,
                    rate_decision=plan_decision,
                    vi_text=unit_text,
                    concise_retranslate_fn=(
                        _concise_retranslate if use_concise else None
                    ),
                    src_lang=manifest.src_lang,
                    tgt_lang=manifest.tgt_lang,
                )
            else:
                # Measure ONE natural take of the full sentence, then apply the
                # same rate policy as a single-member unit. Concise retries are
                # guarded; the legacy fragment translation retry stays disabled.
                src_wav = (
                    artifacts_dir / f"{speech_rel_dir}/unit_{unit_index:04d}.wav"
                )
                new_dur = float(_synth(unit_text, src_wav))
                if new_dur <= 0 or not src_wav.exists():
                    raise RuntimeError(
                        f"unit {unit_index}: synthesis produced no audio for "
                        f"{src_wav}"
                    )
                rate_scale, rate_decision = (
                    plan_rate_fit(new_dur, unit_windows[unit_index])
                    if cfg.tts_rate_fit else (1.0, "off")
                )
                record, info = fit_segment(
                    artifacts_dir,
                    {"idx": unit_index},
                    src_wav,
                    window=unit_windows[unit_index],
                    slot_start=unit_starts[unit_index],
                    prev_stretch=prev_stretch,
                    cfg=cfg,
                    en_text=" ".join(
                        str(script[index].get("en", "")).strip()
                        for index in members
                        if str(script[index].get("en", "")).strip()
                    ),
                    retranslate_fn=None,
                    synth_fn=_synth,
                    resynth_rate_fn=resynth_rate_fn,
                    rate_length_scale=rate_scale,
                    rate_decision=rate_decision,
                    vi_text=unit_text,
                    retranslate_disabled_reason=UNIT_RETRANSLATE_OFF,
                    concise_retranslate_fn=(
                        _concise_retranslate if use_concise else None
                    ),
                    src_lang=manifest.src_lang,
                    tgt_lang=manifest.tgt_lang,
                )
            record["members"] = list(members)
            text_by_idx[unit_index] = (str(record.get("vi", unit_text)), speaker)
            timed.append(record)
            stats.append(info)
            prev_stretch = float(record["stretch"])
    else:
        for seg in sorted(speech, key=lambda row: int(row.get("idx", 0))):
            idx = int(seg.get("idx", 0))
            rel = str(seg.get("file", f"speech/seg_{idx:04d}.wav"))
            src_wav = artifacts_dir / rel
            if not src_wav.exists():
                raise FileNotFoundError(f"segment audio not found: {src_wav}")
            plan_ls, plan_decision = rate_plans.get(idx, (1.0, "off"))
            record, info = fit_segment(
                artifacts_dir,
                seg,
                src_wav,
                window=windows[idx],
                slot_start=starts[idx],
                prev_stretch=prev_stretch,
                cfg=cfg,
                en_text=str(script[idx].get("en", "")),
                retranslate_fn=_retranslate,
                synth_fn=_synth,
                resynth_rate_fn=resynth_rate_fn,
                rate_length_scale=plan_ls,
                rate_decision=plan_decision,
                vi_text=selected_text(script[idx], "spoken_vi"),
                concise_retranslate_fn=(
                    _concise_retranslate if use_concise else None
                ),
                src_lang=manifest.src_lang,
                tgt_lang=manifest.tgt_lang,
            )
            text_by_idx[idx] = (
                str(record.get("vi", selected_text(script[idx], "spoken_vi"))),
                str(script[idx].get("speaker") or "spk0"),
            )
            timed.append(record)
            stats.append(info)
            prev_stretch = float(record["stretch"])

    # Per-clip TTS QC (M4-T7, QUALITY-ROADMAP.md §3.4/P1.7): score every final
    # fitted clip with NISQA-TTS, retry low-MOS clips once with changed settings,
    # keep the better clip and write a review packet when it is still low. Runs
    # after the fit loop (all fitted wavs exist) and BEFORE the bleed asserts.
    # The scorer is optional: an unavailable scorer/weights degrade to skipped
    # counters and never crash the pipeline.
    if cfg.qc_tts:
        engine_name = _normalized_engine(manifest)

        def _qc_resynth(text: str, out_wav: Path, speaker: str) -> float:
            engine = tts_holder["engine"]
            if engine is None:
                engine = build_tts(manifest, cfg)
                tts_holder["engine"] = engine
            tts_text, _spoken = apply_pronunciation(text, pronunciation)
            return _synth_variant(
                engine, tts_text, out_wav, engine_name, cfg, speaker
            )

        tts_qc = run_tts_qc(artifacts_dir, timed, text_by_idx, cfg, _qc_resynth)
        (artifacts_dir / TTS_QC_NAME).write_text(
            json.dumps(tts_qc, ensure_ascii=False), encoding="utf-8"
        )
        manifest.paths["tts_qc"] = TTS_QC_NAME
        log.info(
            "TTS QC: scored=%d low_mos=%d retried=%d resolved=%d skipped=%d "
            "mos_mean=%s (min %.2f)",
            tts_qc["scored"],
            tts_qc["low_mos"],
            tts_qc["retried"],
            tts_qc["resolved"],
            tts_qc["skipped"],
            tts_qc["mos_mean"],
            tts_qc["mos_min"],
        )

    _assert_no_bleed(
        timed, unit_boundaries if cfg.sentence_units else boundaries
    )
    _assert_end_boundary(timed, stats)

    # SA-1-T3 display alignment (sentence-unit path only): transcribe each
    # multi-member unit's FINAL audio with the project ASR and allocate a real,
    # word-edge-snapped span per member row. The ASR is built LAZILY (only when a
    # multi-member unit needs a span and cfg.align_display is on) and every ASR /
    # parse error degrades the affected unit to the split fallback — never the
    # run. Placement (start/end/file) is never touched here.
    align_counts = {"asr": 0, "split": 0, "single": 0, "unavailable": 0}
    if cfg.sentence_units:
        asr_holder: dict[str, object] = {"engine": asr}
        timed_by_idx = {int(row.get("idx", 0)): row for row in timed}

        def _align_engine():
            engine = asr_holder["engine"]
            if engine is None:
                engine = build_asr(manifest.asr_model)
                asr_holder["engine"] = engine
            return engine

        align_engine = None
        if cfg.align_display and has_multi_units:
            try:
                align_engine = _align_engine()
            except Exception as exc:  # noqa: BLE001 - alignment must not fail a run
                log.warning(
                    "display alignment disabled: could not build the ASR (%s)", exc
                )
                align_engine = None

        for unit in units:
            record = timed_by_idx.get(unit.index)
            if record is None:
                continue
            try:
                if len(unit.members) == 1:
                    member = int(unit.members[0])
                    piece = str(split_translation(str(record.get("vi", unit.text)), unit.shares)[0]).strip()
                    if not piece:
                        piece = selected_text(script[member], "spoken_vi").strip()
                    if piece:
                        record["align"] = "single"
                        record["member_spans"] = [
                            [
                                member,
                                float(record["start"]),
                                float(record["end"]),
                                piece,
                            ]
                        ]
                        align_counts["single"] += 1
                    else:
                        record["align"] = "unavailable"
                        record["member_spans"] = []
                        align_counts["unavailable"] += 1
                    continue
                wav_path = artifacts_dir / str(record.get("file", ""))
                align, spans, align_words = _align_unit(
                    record,
                    unit,
                    script,
                    str(record.get("vi", unit.text)),
                    wav_path,
                    align_engine,
                    manifest.tgt_lang,
                )
                record["align"] = align
                record["member_spans"] = spans
                # Q-1-T4: persist the ASR word timeline (sentence-unit path only)
                # so the subtitle builder can split cues on real word edges. Kept
                # off the row when the alignment fell back to the proportional
                # split, so legacy rows keep today's exact shape.
                if align_words:
                    record["align_words"] = align_words
                align_counts[align] = align_counts.get(align, 0) + 1
            except Exception as exc:  # noqa: BLE001 - never fail the run
                log.warning(
                    "unit %s: display alignment failed (%s); recording "
                    "unavailable",
                    unit.index,
                    exc,
                )
                record["align"] = "unavailable"
                record["member_spans"] = []
                align_counts["unavailable"] += 1

    timed_idxs = {int(record.get("idx", 0)) for record in timed}
    if cfg.sentence_units:
        missing_timed = [
            unit_index
            for unit_index in range(len(units))
            if unit_index not in timed_idxs
        ]
    else:
        missing_timed = _missing_indices(len(script), timed_idxs)
    if missing_timed:
        unit_word = "unit(s)" if cfg.sentence_units else "script segment(s)"
        raise RuntimeError(
            f"timing coverage invariant violated: {unit_word} "
            f"{missing_timed} have no timed.json entry"
        )

    stretched = sum(1 for info in stats if info["action"] == "stretch")
    retranslated = sum(1 for info in stats if info["retranslated"])
    # Every clip row (smart or blind) is listed so the report shows *how* it was
    # cut; ``clipped`` counts only the blind cuts, ``smart_trims`` the pause cuts.
    clips = [
        {
            "idx": int(record["idx"]),
            "start": round(float(info["start"]), 3),
            "seconds": round(float(info["seconds_trimmed"]), 3),
            "seconds_trimmed": round(float(info["seconds_trimmed"]), 3),
            "reason": info["clip_reason"],
            "smart_trim": bool(info["smart_trim"]),
        }
        for record, info in zip(timed, stats)
        if info["action"] == "clip"
    ]
    clipped = sum(1 for info in stats if info["clipped"])
    smart_trims = sum(1 for info in stats if info["smart_trim"])
    stretch_values = [float(info["stretch"]) for info in stats]
    max_stretch = max(stretch_values) if stretch_values else 1.0
    max_delta = max(
        (
            abs(curr - prev)
            for prev, curr in zip(stretch_values, stretch_values[1:])
        ),
        default=0.0,
    )
    filler_removals = sum(int(info["filler_removals"]) for info in stats)

    if cfg.sentence_units:
        coverage = compute_coverage(
            script,
            speech_by_idx,
            [unit_members[unit_index] for unit_index in range(len(units))],
        )
    else:
        coverage = compute_coverage(script, speech_by_idx)

    (artifacts_dir / TIMED_NAME).write_text(
        json.dumps(timed, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    pacing = {
        "stretched": stretched,
        "retranslated": retranslated,
        "clipped": clipped,
        "smart_trims": smart_trims,
        "clips": clips,
        "max_stretch": round(max_stretch, 3),
        "max_neighbor_delta": round(max_delta, 3),
        "filler_removals": filler_removals,
    }
    # FIT-1: only present when a new unit ran, so an opt-out job's timing.json is
    # byte-identical to before.
    if cfg.tts_rate_fit or cfg.recenter_audio:
        pacing["rate_fit"] = {
            "resynthesized": sum(
                1 for info in stats if info.get("rate_action") == "resynth"
            ),
            "kept": sum(
                1 for info in stats if info.get("rate_action") == "keep"
            ),
            "failed": sum(
                1 for info in stats if info.get("rate_action") == "failed"
            ),
            "global_length_scale": round(rate_base, 3),
            "recentered": sum(1 for info in stats if info.get("recentered")),
        }
    # SA-1-T3 display-alignment summary (sentence-unit path only, so an opt-out
    # job's timing.json stays byte-identical).
    if cfg.sentence_units:
        pacing["align"] = dict(align_counts)
    (artifacts_dir / TIMING_NAME).write_text(
        json.dumps(pacing, ensure_ascii=False), encoding="utf-8"
    )

    if clipped:
        log.warning(
            "%d segment(s) clipped at the window end (last resort): %s",
            clipped,
            ", ".join(str(clip["idx"]) for clip in clips if not clip["smart_trim"]),
        )
    if smart_trims:
        log.info(
            "%d segment(s) smart-trimmed at a silence: %s",
            smart_trims,
            ", ".join(str(clip["idx"]) for clip in clips if clip["smart_trim"]),
        )

    manifest.speech_coverage = round(coverage, 2)
    manifest.stretched_segments = stretched
    if coverage < SPEECH_COVERAGE_WARN:
        log.warning(
            "speech coverage %.2f is below %.2f: the dub only fills %s of the "
            "spoken time, so long stretches will be silent. Likely causes: "
            "wrong-language ASR (short/hallucinated segments — use the default "
            "auto/large-v3 model for English audio), very short TTS output, or "
            "a TTS segment that failed. Check script.tsv / segments.json.",
            coverage,
            SPEECH_COVERAGE_WARN,
            f"{coverage:.0%}",
        )

    manifest.paths["timed"] = TIMED_NAME
    manifest.paths["timing"] = TIMING_NAME
    manifest.stages["timing"] = "done"
    if tts is None:
        engine = tts_holder["engine"]
        close = getattr(engine, "close", None)
        if callable(close):
            close()
    manifest.save(artifacts_dir / "job.json")
    return manifest


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``python -m vidub.stages.timing <job.json>``."""
    parser = argparse.ArgumentParser(
        prog="vidub.stages.timing",
        description="S4 timing: fit speech segments into their video windows.",
    )
    parser.add_argument("job", help="path to the job.json manifest")
    parser.add_argument(
        "--cache-dir", default=None, help="cache root (default: manifest's)"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = JobConfig()
    if args.cache_dir:
        cfg.cache_dir = args.cache_dir

    try:
        manifest = fit(args.job, cfg)
    except (FileNotFoundError, RuntimeError, TranslationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(str(manifest.artifacts_dir / TIMED_NAME))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
