"""Level-1 transcript QC: classify ASR segments and re-ASR flagged windows.

This is the P0 quality layer from ``QUALITY-ROADMAP.md`` §3.2. For every
segment produced by the ASR stage it:

1. **classifies** the segment from the faster-whisper confidence signals
   (``avg_logprob`` / ``compression_ratio`` / ``no_speech_prob``), cheap text
   statistics (repetition loops) and the existing Vietnamese language guard —
   into ``ok`` / ``hallucination`` / ``garbled`` / ``language-mismatch`` /
   ``segmentation-issue``;
2. **retries once** the ``hallucination``/``garbled`` segments through
   ``engine.transcribe_window`` (a changed-settings re-run) when the engine
   exposes it, and re-classifies the new hypothesis;
3. writes a human **review packet** (``review/transcript/seg-<i>.txt`` + a
   ``.wav`` snippet sliced with the stdlib :mod:`wave` module) for anything that
   stays flagged;
4. returns a machine-readable QC artifact with counts, the unresolved flagged
   ratio and the per-job **verdict** (``pass`` / ``review`` / ``fail``).

The module is stdlib-only and deliberately tolerant: a missing/``None`` signal
(the PhoWhisper path, an old manifest) never raises — classification simply
falls back to text-only statistics.
"""

from __future__ import annotations

import array
import logging
import unicodedata
import wave
from pathlib import Path
from typing import Any

from vidub.manifest import Segment
from vidub.utils.langcheck import detect_mismatch

__all__ = [
    "COMPRESSION_RATIO_MAX",
    "AVG_LOGPROB_MIN",
    "NO_SPEECH_PROB_MAX",
    "SEGMENT_MAX_SECONDS",
    "FAIL_RATIO",
    "REPEAT_NGRAM",
    "REPEAT_MAX",
    "REPEAT_MIN_TOKENS",
    "LABEL_OK",
    "LABEL_HALLUCINATION",
    "LABEL_GARBLED",
    "LABEL_LANG_MISMATCH",
    "LABEL_SEGMENTATION",
    "REVIEW_SUBDIR",
    "classify_segment",
    "qc_transcript",
]

log = logging.getLogger(__name__)

# --------------------------------------------------------------- thresholds
# faster-whisper compression ratio above which a segment is a repetition loop
# (the upstream default that triggers the temperature fallback).
COMPRESSION_RATIO_MAX = 2.4
# Average log-probability below which the decode is low-confidence (garbled).
AVG_LOGPROB_MIN = -1.0
# Probability the segment holds no speech above which we distrust the text.
NO_SPEECH_PROB_MAX = 0.6
# Longer than this and the segment is a run-on (segmentation issue).
SEGMENT_MAX_SECONDS = 30.0
# Fraction of still-flagged segments that turns the job verdict into ``fail``.
FAIL_RATIO = 0.3
# Repetition loop detection: a repeated n-gram of this size, at least this many
# times, in a text of at least this many tokens.
REPEAT_NGRAM = 3
REPEAT_MAX = 3
REPEAT_MIN_TOKENS = 12

# ------------------------------------------------------------------- labels
LABEL_OK = "ok"
LABEL_HALLUCINATION = "hallucination"
LABEL_GARBLED = "garbled"
LABEL_LANG_MISMATCH = "language-mismatch"
LABEL_SEGMENTATION = "segmentation-issue"

# Order encodes precedence: a segment carrying several issues is labelled by
# the first (highest) one that applies.
_LABEL_PRECEDENCE = (
    LABEL_HALLUCINATION,
    LABEL_GARBLED,
    LABEL_LANG_MISMATCH,
    LABEL_SEGMENTATION,
)

REVIEW_SUBDIR = "review/transcript"
# Review-packet audio snippets are capped at this many seconds.
SNIPPET_MAX_SECONDS = 30.0


def _numeric(meta: dict[str, Any], key: str) -> float | None:
    """Return ``meta[key]`` as a float, or ``None`` when missing/unparseable."""
    value = meta.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _tokens(text: str) -> list[str]:
    """Lowercased alphanumeric tokens (Unicode-aware, punctuation stripped)."""
    tokens: list[str] = []
    current: list[str] = []
    for char in unicodedata.normalize("NFC", text or "").lower():
        if char.isalnum():
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def _max_ngram_repeats(tokens: list[str], n: int) -> int:
    """Highest number of times any ``n``-gram of ``tokens`` occurs."""
    if n <= 0 or len(tokens) < n:
        return 0
    counts: dict[tuple[str, ...], int] = {}
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i : i + n])
        counts[gram] = counts.get(gram, 0) + 1
    return max(counts.values(), default=0)


def _non_monotonic_words(words: list[dict[str, Any]]) -> bool:
    """True when word timings overlap or go backwards."""
    prev_start: float | None = None
    for word in words:
        try:
            start = float(word.get("start", 0.0))
            end = float(word.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        if end < start:
            return True
        if prev_start is not None and start < prev_start - 0.01:
            return True
        prev_start = start
    return False


def classify_segment(
    text: str, meta: dict[str, Any] | None, src_lang: str
) -> tuple[str, list[str]]:
    """Classify one segment and return ``(label, reasons)``.

    ``meta`` carries the ASR confidence signals plus, when the caller can supply
    them, ``duration`` (seconds) and ``words`` (word timing) — :func:`qc_transcript`
    composes those from the :class:`~vidub.manifest.Segment`. A ``None``/sparse
    ``meta`` degrades gracefully to text-only statistics.

    ``reasons`` accumulates every triggered explanation; ``label`` is the
    highest-precedence one (hallucination > garbled > language-mismatch >
    segmentation-issue > ``ok``).
    """
    data: dict[str, Any] = meta or {}
    reasons: list[str] = []

    # --- hallucination: a repetition loop -------------------------------------
    hallucination: list[str] = []
    compression = _numeric(data, "compression_ratio")
    if compression is not None and compression > COMPRESSION_RATIO_MAX:
        hallucination.append(
            f"compression_ratio={compression:.1f}>{COMPRESSION_RATIO_MAX}"
        )
    else:
        tokens = _tokens(text)
        if len(tokens) >= REPEAT_MIN_TOKENS:
            repeats = _max_ngram_repeats(tokens, REPEAT_NGRAM)
            if repeats >= REPEAT_MAX:
                hallucination.append(
                    f"repeated_{REPEAT_NGRAM}gram={repeats}>={REPEAT_MAX}"
                )

    # --- garbled: low confidence / likely no speech ---------------------------
    garbled: list[str] = []
    avg_logprob = _numeric(data, "avg_logprob")
    if avg_logprob is not None and avg_logprob < AVG_LOGPROB_MIN:
        garbled.append(f"avg_logprob={avg_logprob:.2f}<{AVG_LOGPROB_MIN}")
    no_speech = _numeric(data, "no_speech_prob")
    if no_speech is not None and no_speech > NO_SPEECH_PROB_MAX:
        garbled.append(f"no_speech_prob={no_speech:.2f}>{NO_SPEECH_PROB_MAX}")

    # --- language-mismatch: reuse the shared Vietnamese diacritic guard -------
    mismatch = detect_mismatch(text, src_lang)
    language = [mismatch] if mismatch else []

    # --- segmentation: overlong run-on or non-monotonic word timing -----------
    segmentation: list[str] = []
    duration = _numeric(data, "duration")
    if duration is not None and duration > SEGMENT_MAX_SECONDS:
        segmentation.append(f"duration={duration:.1f}s>{SEGMENT_MAX_SECONDS}s")
    words = data.get("words")
    if words and _non_monotonic_words(words):
        segmentation.append("non-monotonic word timing")

    for group in (hallucination, garbled, language, segmentation):
        reasons.extend(group)

    for label, group in zip(
        _LABEL_PRECEDENCE, (hallucination, garbled, language, segmentation)
    ):
        if group:
            return label, reasons
    return LABEL_OK, reasons


def _compose_meta(segment: Segment) -> dict[str, Any]:
    """Merge a segment's ASR meta with duration + words for classification."""
    meta = dict(segment.meta or {})
    meta["duration"] = float(segment.end) - float(segment.start)
    meta["words"] = segment.words
    return meta


def _snippet_end(start: float, end: float) -> float:
    """Clamp the snippet end so the slice is at most ``SNIPPET_MAX_SECONDS``."""
    return min(float(end), float(start) + SNIPPET_MAX_SECONDS)


def _write_snippet(
    audio_path: str | Path, target: Path, start: float, end: float
) -> bool:
    """Slice ``[start, end]`` from ``audio_path`` into a mono 16-bit WAV.

    Returns ``True`` on success, ``False`` when the source cannot be read (the
    caller still writes the text entry). Uses only the stdlib :mod:`wave`.
    """
    try:
        with wave.open(str(audio_path), "rb") as src:
            rate = src.getframerate()
            channels = src.getnchannels()
            width = src.getsampwidth()
            total = src.getnframes()
            start_frame = max(0, int(float(start) * rate))
            end_frame = min(int(_snippet_end(start, end) * rate), total)
            if end_frame <= start_frame:
                return False
            src.setpos(start_frame)
            frames = src.readframes(end_frame - start_frame)
    except (OSError, wave.Error, EOFError) as exc:
        log.warning("could not read audio snippet source %s (%s)", audio_path, exc)
        return False

    if width != 2:  # pragma: no cover - ingest always writes 16-bit PCM
        samples = frames
        out_width = width
        out_channels = channels
    else:
        values = array.array("h")
        values.frombytes(frames)
        if channels > 1:  # downmix to mono by keeping the first channel
            values = values[0::channels]
        samples = values.tobytes()
        out_width = 2
        out_channels = 1

    with wave.open(str(target), "wb") as out:
        out.setnchannels(out_channels)
        out.setsampwidth(out_width)
        out.setframerate(rate)
        out.writeframes(samples)
    return True


def _write_packet_entry(
    out_dir: str | Path,
    entry: dict[str, Any],
    segment: Segment,
    original_text: str,
    audio_path: str | Path | None,
) -> None:
    """Write the ``.txt`` + ``.wav`` review-packet files for one segment."""
    packet_dir = Path(out_dir) / REVIEW_SUBDIR
    packet_dir.mkdir(parents=True, exist_ok=True)
    stem = f"seg-{entry['index']}"
    text_target = packet_dir / f"{stem}.txt"

    lines = [
        f"segment: {entry['index']}",
        f"start: {entry['start']:.3f}",
        f"end: {entry['end']:.3f}",
        f"label: {entry['label']}",
        "reasons: " + ("; ".join(entry["reasons"]) or "(none)"),
        "",
        "original text:",
        original_text,
    ]
    if entry["retried"]:
        lines += [
            "",
            f"retry label: {entry['retry_label']}",
            "retry text:",
            entry["retry_text"] or "(no hypothesis)",
        ]
    text_target.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if audio_path is not None:
        _write_snippet(
            audio_path,
            packet_dir / f"{stem}.wav",
            float(entry["start"]),
            float(entry["end"]),
        )


def qc_transcript(
    segments: list[Segment],
    src_lang: str,
    engine: Any = None,
    audio_path: str | Path | None = None,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Classify every segment, retry the flagged ones once, and return the QC.

    ``segments`` is mutated in place: when a retry produces an ``ok`` hypothesis
    the segment's ``text``/``words``/``meta`` are replaced by the retry's. The
    returned artifact is::

        {
          "verdict": "pass" | "review" | "fail",
          "counts": {<label>: <int>, ...},          # original labels
          "flagged_ratio_unresolved": <float>,
          "segments": [
            {"index", "start", "end", "label", "reasons", "retried",
             "resolved", "retry_text", "retry_label"},
            ...
          ],
        }

    When ``out_dir`` is given, every still-flagged segment (unresolved, whether
    or not it was retried) also gets ``review/transcript/seg-<i>.txt`` and — if
    ``audio_path`` is readable — a ``seg-<i>.wav`` snippet.
    """
    counts = {
        LABEL_OK: 0,
        LABEL_HALLUCINATION: 0,
        LABEL_GARBLED: 0,
        LABEL_LANG_MISMATCH: 0,
        LABEL_SEGMENTATION: 0,
    }
    entries: list[dict[str, Any]] = []
    packet: list[tuple[dict[str, Any], Segment, str]] = []
    unresolved = 0
    unresolved_hallucination = 0

    for index, segment in enumerate(segments):
        original_text = segment.text
        label, reasons = classify_segment(
            segment.text, _compose_meta(segment), src_lang
        )
        counts[label] = counts.get(label, 0) + 1

        retried = False
        resolved = False
        retry_text: str | None = None
        retry_label: str | None = None

        flagged = label in (LABEL_HALLUCINATION, LABEL_GARBLED)
        if flagged and engine is not None and hasattr(engine, "transcribe_window"):
            retried = True
            retry_segment = _retry_window(
                engine, audio_path, segment, src_lang
            )
            if retry_segment is not None:
                retry_text = retry_segment.text
                retry_label, _ = classify_segment(
                    retry_segment.text, _compose_meta(retry_segment), src_lang
                )
                if retry_label == LABEL_OK:
                    segment.text = retry_segment.text
                    segment.words = retry_segment.words
                    segment.meta = retry_segment.meta
                    resolved = True

        entry = {
            "index": index,
            "start": float(segment.start),
            "end": float(segment.end),
            "label": label,
            "reasons": list(reasons),
            "retried": retried,
            "resolved": resolved,
            "retry_text": retry_text,
            "retry_label": retry_label,
        }
        entries.append(entry)

        if label != LABEL_OK and not resolved:
            unresolved += 1
            if label == LABEL_HALLUCINATION:
                unresolved_hallucination += 1
            packet.append((entry, segment, original_text))

    total = len(segments)
    flagged_ratio = (unresolved / total) if total else 0.0
    if unresolved_hallucination > 0 or flagged_ratio >= FAIL_RATIO:
        verdict = "fail"
    elif unresolved > 0:
        verdict = "review"
    else:
        verdict = "pass"

    if out_dir is not None and packet:
        for entry, segment, original_text in packet:
            _write_packet_entry(
                out_dir, entry, segment, original_text, audio_path
            )

    return {
        "verdict": verdict,
        "counts": counts,
        "flagged_ratio_unresolved": flagged_ratio,
        "segments": entries,
    }


def _retry_window(
    engine: Any, audio_path: str | Path | None, segment: Segment, src_lang: str
) -> Segment | None:
    """One ``transcribe_window`` retry, tolerating any engine failure."""
    try:
        return engine.transcribe_window(
            str(audio_path) if audio_path is not None else "",
            float(segment.start),
            float(segment.end),
            src_lang or None,
        )
    except Exception as exc:  # noqa: BLE001 - a retry must never crash the run
        log.warning(
            "re-ASR of segment [%.2f, %.2f] failed (%s); leaving it flagged",
            float(segment.start),
            float(segment.end),
            exc,
        )
        return None
