"""S2 Translate: source segments -> Vietnamese dubbing script (``script.json``).

Reads ``job.json`` + ``segments.json``, translates every segment with the
configured MT engine (envit5 by default, or the contextual Ollama LLM engine /
the opt-in Hunyuan HY-MT1.5 specialist) and writes ``script.json`` — a list of
``{start, end, speaker, en, vi, truncated}`` records — records it under
``manifest.paths`` and marks the ``translate`` stage done.

The main batch goes through ``engine.translate_contextual`` so a context-aware
engine (Ollama, M4-T5) can see neighbouring segments; engines without context
(BaseTranslator's default) delegate to ``translate``, keeping envit5 identical.

When ``cfg.glossary_path`` is set, the glossary is loaded once and applied to
each MT output (never to passthrough segments) *before* the length/low-confidence
checks, so enforced terminology is what gets truncated/flagged. Two mechanisms
run: the output-side replace (a term present in the MT output) and the
source-side splice (a glossed term present only in the source is reinforced by
re-translating the surrounding chunks and joining the enforced phrase). The
counts (``terms`` / ``replacements`` / ``spliced`` / ``enforced`` /
``hesitations`` / ``style_fixes``) are written to a small ``glossary.json``
sidecar that the export stage reads for the report.

Every row (translated, spliced and passthrough) is passed through
:func:`vidub.utils.textclean.clean_hesitation` before the length check, so
stutters/repetitions never reach the TTS. MT/spliced rows then get the
deterministic Vietnamese word-order pass
(:func:`vidub.utils.vnstyle.apply_vnstyle`) before the length guard; passthrough
rows are user-language and are left untouched.

Length guard (dubbing rhythm): if a translation's word count exceeds 2.2x the
source word count it is truncated at the last sentence end at/before the limit
and flagged ``"truncated": true``.

Sentence-level translation units (TU-1 / SA-1-T2, default on) make the
**sentence** the ``script.json`` row unit and the **unit** the translation + QC
unit. The ASR rows are immediately split at internal sentence boundaries
(:func:`vidub.utils.sentences.split_into_sentences`) with each sentence's
``start``/``end`` locked to the ASR word edges, and
:func:`vidub.utils.sentences.group_units` merges a fragment run with the
complete sentence it finishes into one unit; a unit closes only at a completed
sentence or a speaker change. Each unit's source is translated as one
``translate_contextual`` batch item and the result is checked by the new
completeness guard (:mod:`vidub.utils.units_guard`): a short/dropped unit is
resampled or handed to a fallback engine, and a unit that reproduced a
neighbour's content (``context-bleed`` / ``sentence-addition``) is retried with
the direct (non-contextual) template, before the deterministic passes
(glossary, hesitation cleanup, vnstyle, length guard) run **on the unit text**;
only then is the final unit text distributed back onto its member sentence rows
with :func:`vidub.utils.sentences.split_translation`. Passthrough (already
target-language) rows are never merged and never translated. The one-per-unit
guard+battery records and the unit counts are written to
``translation_qc.json`` under ``units`` (``detail`` + ``flags``) and the unit
map to ``units.json``. ``--no-sentence-units`` (``cfg.sentence_units = False``)
skips the whole unit path and is byte-identical to the previous per-segment
behavior (no ``units.json``, one ``script.json`` row per ASR segment).

Standalone use::

    python -m vidub.stages.translate <cache>/<hash>/job.json
"""

from __future__ import annotations

from vidub.stages.translate.constants import (  # noqa: F401
    GLOSSARY_NAME,
    LENGTH_RATIO,
    LOW_CONFIDENCE_MIN_WORDS,
    PASSTHROUGH_RATIO,
    SCRIPT_NAME,
    SEGMENTS_NAME,
    TRANSLATION_QC_NAME,
    UNITS_NAME,
    _SENTENCE_END,
)
from vidub.stages.translate.engines import (  # noqa: F401
    _build_refiner,
    _is_low_confidence,
    _refine_engine,
    build_translator,
    truncate_translation,
)
from vidub.stages.translate.qc import _run_qc_battery, _script_row  # noqa: F401
from vidub.stages.translate.run import _build_per_segment_script, main, translate  # noqa: F401
from vidub.stages.translate.units import (  # noqa: F401
    _CONCISE_VARIANTS,
    _UnitPathResult,
    _accepts_budgets,
    _budgeted_repair,
    _chat_candidate,
    _compression_candidates,
    _concise_request,
    _engine_accepts_budgets,
    _passthrough_guard_record,
    _rejection_reason,
    _translate_sentence_units,
    _unit_rows,
)

__all__ = ["translate", "build_translator", "truncate_translation", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
