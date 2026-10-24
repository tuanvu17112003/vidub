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

# Re-exported so ``monkeypatch.setattr(timing.ttsqc, ...)`` keeps patching the
# same module object the clip-QC code calls into (test compatibility).
from vidub.utils import ttsqc  # noqa: F401

from vidub.stages.timing.align import (  # noqa: F401
    _align_unit,
    _check_align_words,
    _word_times,
)
from vidub.stages.timing.clipqc import (  # noqa: F401
    _normalized_engine,
    _piper_voice_files,
    _synth_variant,
    _write_tts_review,
    run_tts_qc,
)
from vidub.stages.timing.constants import (  # noqa: F401
    BLEED_TOLERANCE,
    MIN_RETAINED_SPEECH,
    SPEECH_COVERAGE_WARN,
    SPEECH_NAME,
    SCRIPT_NAME,
    SUBTITLES_ONLY,
    TIMED_NAME,
    TIMING_NAME,
    TTS_QC_NAME,
    TTS_QC_REVIEW_SUBDIR,
    UNIT_RETRANSLATE_OFF,
    UNITS_NAME,
    VXTTS_RETRY_TEMPERATURE,
    VXTTS_RETRY_TOP_K,
)
from vidub.stages.timing.synth import (  # noqa: F401
    _assert_end_boundary,
    _assert_no_bleed,
    _clamp_rate,
    _fit_clip,
    _fit_to_window,
    _synth_retranslation,
    choose_smart_cut,
    detect_silences,
    fit_segment,
)
from vidub.stages.timing.units import (  # noqa: F401
    _TimingUnit,
    _group_script_units,
    _missing_indices,
    _source_shares,
    _timing_units,
    _unit_text,
    _units_from_file,
    compute_coverage,
)
from vidub.stages.timing.run import _check_mode, fit, main  # noqa: F401

__all__ = ["fit", "fit_segment", "compute_coverage", "run_tts_qc", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
