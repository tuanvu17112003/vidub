"""Segment fitting: stretch, smart-trim, re-translation retry, recentering."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from vidub.config import JobConfig
from vidub.engines.mt.base import TranslationError
from vidub.engines.mt.validators import _validate
from vidub.utils.audio import run_ffmpeg, wav_duration
from vidub.utils.fit import (
    RATE_SPEED_MAX,
    RATE_SPEED_MIN,
    RECENTER_MIN_SLACK,
    concise_char_target,
    effective_speed_cap,
    plan_stretch,
    recenter_shift,
)
from vidub.utils.textclean import clean_hesitation, strip_fillers
from vidub.utils.units_guard import guard_signals

from vidub.stages.timing.constants import (
    _CLIP_SUFFIX,
    _FADE_OUT,
    _FIT_SUFFIX,
    _RATE_SUFFIX,
    _RETRANSLATE_SUFFIX,
    _SILENCE_END_RE,
    _SILENCE_MIN_DUR,
    _SILENCE_NOISE,
    _SILENCE_START_RE,
    _SMART_TRIM_LOOKBACK,
    _UNIT_SYNTH_DIR,
    BLEED_TOLERANCE,
    MIN_RETAINED_SPEECH,
)

log = logging.getLogger(__name__)


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
