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

import argparse
import inspect
import json
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from vidub.config import (
    DEFAULT_HUNYUAN_HOST,
    DEFAULT_HUNYUAN_MODEL,
    JobConfig,
)
from vidub.engines.mt.base import BaseTranslator, TranslationError
from vidub.engines.mt.envit5_mt import Envit5Translator
from vidub.engines.mt.hunyuan_mt import _lang_full
from vidub.engines.mt.validators import _validate
from vidub.manifest import JobManifest
from vidub.utils.glossary import apply_glossary, load_glossary, splice_source_terms
from vidub.utils.isometry import (
    accept_shrunk,
    char_count,
    coverage_floor,
    over_budget,
    projected_stretch,
    unit_budget,
)
from vidub.utils.langcheck import VN_CHARS, vn_diacritic_ratio
from vidub.utils.sentences import (
    Unit,
    ends_sentence,
    group_units,
    split_into_sentences,
    split_translation,
    unit_cut_stats,
    unit_share_summary,
)
from vidub.utils.textclean import clean_hesitation
from vidub.utils.tqc import run_battery
from vidub.utils.units_guard import (
    UNIT_COVERAGE_MIN,
    contamination_signals,
    guard_signals,
    guard_units,
)
from vidub.utils.vnstyle import apply_vnstyle

__all__ = ["translate", "build_translator", "truncate_translation", "main"]

log = logging.getLogger(__name__)

SEGMENTS_NAME = "segments.json"
SCRIPT_NAME = "script.json"
# Glossary counters sidecar for report.json ("glossary": {...}); script.json rows
# stay unchanged, so the loaded/applied counts are persisted here instead.
GLOSSARY_NAME = "glossary.json"
# Level-2 translation QC artifact (QUALITY-ROADMAP.md §3.3, M4-T3).
TRANSLATION_QC_NAME = "translation_qc.json"
# SA-1-T2 unit map sidecar: one record per sentence-level unit with the unit's
# authoritative locked span and its member row indices (only on the unit path).
UNITS_NAME = "units.json"
LENGTH_RATIO = 2.2
# A source segment this Vietnamese-looking is already Vietnamese: asking envit5
# to translate it just echoes the input (and can corrupt it), so copy it through.
PASSTHROUGH_RATIO = 0.5
# Below this many source words an output without diacritics is not meaningful.
LOW_CONFIDENCE_MIN_WORDS = 3
_SENTENCE_END = (".", "!", "?")


def build_translator(
    manifest: JobManifest, cfg: JobConfig | None = None
) -> BaseTranslator:
    """Instantiate the MT engine named by ``manifest.mt_engine``.

    ``manifest.mt_engine`` (written by ingest from ``cfg.mt_engine``) selects the
    engine; ``cfg`` optionally supplies engine-specific settings, currently the
    Ollama and Hunyuan model/host. It stays optional so callers that only need
    the default envit5 engine (the timing stage's shortening retry) keep working.
    """
    engine = (
        (manifest.mt_engine or "").strip().lower()
        or ((cfg.mt_engine if cfg else "") or "envit5").strip().lower()
    )
    if engine in {"envit5", "envit5-translation", "vietai/envit5-translation"}:
        return Envit5Translator()
    if engine == "ollama":
        from vidub.engines.mt.ollama_mt import OllamaTranslator

        return OllamaTranslator(
            model=(cfg.ollama_model if cfg else "") or "qwen2.5:7b",
            host=(cfg.ollama_host if cfg else "") or "http://127.0.0.1:11434",
        )
    if engine == "hunyuan":
        from vidub.engines.mt.hunyuan_mt import HunyuanMTTranslator

        return HunyuanMTTranslator(
            model=(cfg.hunyuan_model if cfg else "") or DEFAULT_HUNYUAN_MODEL,
            host=(cfg.hunyuan_host if cfg else "") or DEFAULT_HUNYUAN_HOST,
        )
    raise TranslationError(f"unknown mt_engine: {manifest.mt_engine!r}")


def truncate_translation(text: str, src_words: int) -> tuple[str, bool]:
    """Enforce the 2.2x length rule.

    Returns ``(text, truncated)``: when ``text`` exceeds ``2.2 * src_words``
    words it is cut at the last ``. ! ?`` at/before the limit (falling back to
    the limit itself if there is no sentence end) and flagged.
    """
    if src_words <= 0:
        return text, False

    limit = int(LENGTH_RATIO * src_words)
    words = text.split()
    if len(words) <= limit:
        return text, False

    cut = None
    for index, word in enumerate(words[:limit]):
        if word and word[-1] in _SENTENCE_END:
            cut = index + 1
    if cut is None:
        cut = limit
    return " ".join(words[:cut]), True


def _is_low_confidence(source: str, translated: str, tgt_lang: str) -> bool:
    """Flag a translation that is empty or (for vi) shockingly diacritic-free.

    An empty output is always suspicious. For ``tgt_lang == "vi"``, an output
    with no Vietnamese diacritics at all while the source had at least
    :data:`LOW_CONFIDENCE_MIN_WORDS` words usually means the MT engine echoed or
    dropped the input.
    """
    if not translated:
        return True
    if (tgt_lang or "").strip().lower() == "vi":
        if len(source.split()) >= LOW_CONFIDENCE_MIN_WORDS and not any(
            char in VN_CHARS for char in translated
        ):
            return True
    return False


def _build_refiner(
    cfg: JobConfig, glossary: dict[str, str], manifest: JobManifest
) -> Callable[[list[dict], list[dict]], tuple[list[dict], list[dict], dict]]:
    """Build the M5-T2 refine callback; may raise on a construction failure.

    The Hunyuan generator is built directly from config (independent of the main
    engine, so refine works with any ``--mt-engine``). The Chimera fusion model
    is used only when ``cfg.fusion_model`` is set; the envit5 candidate engine is
    optional (a missing/heavy dependency disables just that candidate).
    """
    from vidub.engines.mt.hunyuan_mt import HunyuanMTTranslator
    from vidub.utils.refine import make_refiner

    generator = HunyuanMTTranslator(
        model=cfg.hunyuan_model or DEFAULT_HUNYUAN_MODEL,
        host=cfg.hunyuan_host or DEFAULT_HUNYUAN_HOST,
    )
    fusion = None
    if cfg.fusion_model:
        fusion = HunyuanMTTranslator(
            model=cfg.fusion_model,
            host=cfg.hunyuan_host or DEFAULT_HUNYUAN_HOST,
        )
    envit5 = None
    try:
        from vidub.engines.mt.envit5_mt import Envit5Translator

        envit5 = Envit5Translator()
    except Exception as exc:  # noqa: BLE001 - optional/heavy dependency
        log.warning("refine: envit5 candidate unavailable (%s)", exc)
        envit5 = None
    return make_refiner(
        generator,
        fusion=fusion,
        envit5=envit5,
        samples=cfg.hunyuan_samples,
        src_lang=manifest.src_lang,
        tgt_lang=manifest.tgt_lang,
        glossary=glossary,
        max_units=cfg.hunyuan_refine_max_units,
    )


def _refine_engine(manifest: JobManifest, cfg: JobConfig) -> str:
    """The MT engine name the refine gate should key on.

    Mirrors :func:`build_translator`'s resolution (manifest first, then config,
    then ``envit5``) so a hunyuan job is recognised whether the engine name comes
    from the manifest (normal runs) or the config (tests / programmatic use).
    """
    return (
        (manifest.mt_engine or "").strip().lower()
        or (cfg.mt_engine or "").strip().lower()
        or "envit5"
    )


@dataclass
class _UnitPathResult:
    """Everything the sentence-unit path hands back to :func:`translate`."""

    script: list[dict]
    counters: dict[str, int]
    qc_artifact: dict | None
    unit_block: dict | None
    units_doc: dict | None


# ------------------------------------------------- ISO-1 isometric budgeting
def _accepts_budgets(method: object) -> bool:
    """True when ``method`` declares a ``budgets`` keyword parameter."""
    try:
        return "budgets" in inspect.signature(method).parameters  # type: ignore[arg-type]
    except (TypeError, ValueError):  # builtins / C callables have no signature
        return False


def _engine_accepts_budgets(engine: BaseTranslator) -> bool:
    """True when the engine's contextual call path understands ``budgets``.

    The real engines accept the keyword (hunyuan/ollama on their own
    ``translate_contextual``; envit5 on ``translate``, reached through the base
    delegating default). A test double or third-party engine that predates the
    budget interface is left on the exact three-arg calls, so wiring ISO-1
    budgeting in can never break it.
    """
    cls = type(engine)
    contextual = cls.__dict__.get("translate_contextual")
    if contextual is not None:
        return _accepts_budgets(contextual)
    translate = getattr(cls, "translate", None)
    return translate is not None and _accepts_budgets(translate)


def _budgeted_repair(
    engine: BaseTranslator, source: str, src_lang: str, tgt_lang: str, budget: int
) -> str | None:
    """ONE budgeted re-translation of ``source``; ``None`` when it fails.

    Uses the engine's plain (non-contextual) path with a single-item
    ``budgets`` hint, wrapped so any failure only keeps the original text (the
    caller records ``unavailable``). The candidate is *proposed*; the caller
    decides with :func:`vidub.utils.isometry.accept_shrunk`.
    """
    try:
        result = engine.translate([source], src_lang, tgt_lang, budgets=[budget])
    except Exception as exc:  # noqa: BLE001 - a repair failure keeps the original
        log.warning(
            "isometric: budgeted re-translation failed (%s); keeping the original",
            exc,
        )
        return None
    if not result:
        return None
    return str(result[0])


# ISO-1-T3-C1 Best-of-K compression lever.
MAX_ISOMETRIC_CANDIDATES = 3
# (variant id, compression_hint): the hint rides INSIDE the proven translate
# request — it never replaces it. Keeping the "Translate the following text into
# {language} concisely: … output only the translation" shape is what makes the
# hunyuan specialist answer with a Vietnamese translation instead of English/meta
# text (the C1 defect).
_CONCISE_VARIANTS: tuple[tuple[str, str], ...] = (
    ("fewer-words", "say the same thing in fewer words"),
    (
        "drop-filler",
        "drop hedges, filler and redundant modifiers, keeping every fact and number",
    ),
    ("shorter-synonyms", "prefer shorter synonyms and tighter word order"),
)


def _concise_request(
    source: str, tgt_lang: str, compression_hint: str, budget: int
) -> str:
    """The proven concise request carrying one ``compression_hint``.

    Always opens with the wording the timing stage uses successfully
    (``Translate the following text into {language} concisely: …``), names the
    explicit language, states the budget, and ends with "output only the
    translation".
    """
    language = _lang_full(tgt_lang)
    return (
        f"Translate the following text into {language} concisely: at most "
        f"{int(budget)} characters, {compression_hint}, keep the meaning exactly, "
        f"and output only the translation.\n\n{source}"
    )


def _chat_candidate(
    engine: BaseTranslator,
    source: str,
    src_lang: str,
    tgt_lang: str,
    compression_hint: str,
    budget: int,
) -> str:
    """One concise candidate via the engine's private ``_chat`` (never raises).

    hunyuan exposes ``_chat(user, temperature)`` and takes a plain prompt; ollama
    exposes ``_chat(system, user, json_mode)``. The signature is detected so the
    stage can request the dynamic concise prompts without touching the engine
    files. Every request is :func:`_concise_request`. Returns ``""`` on failure.
    """
    chat = getattr(engine, "_chat", None)
    if not callable(chat):
        return ""
    request = _concise_request(source, tgt_lang, compression_hint, budget)
    try:
        params = inspect.signature(chat).parameters
        if "json_mode" in params:  # ollama-style: (system, user, json_mode)
            system = (
                f"You are a professional {src_lang}->{tgt_lang} dubbing translator."
            )
            return str(chat(system, request, False) or "").strip()
        return str(chat(request) or "").strip()
    except Exception as exc:  # noqa: BLE001 - a failed candidate is dropped
        log.warning("isometric: concise candidate failed (%s)", exc)
        return ""


def _compression_candidates(
    engine: BaseTranslator,
    source: str,
    src_lang: str,
    tgt_lang: str,
    budget: int,
) -> list[tuple[str, str]]:
    """Up to :data:`MAX_ISOMETRIC_CANDIDATES` ``(variant, text)`` candidates.

    Engines exposing ``_chat`` (hunyuan/ollama) get the three dynamic concise
    prompts. Otherwise one budgeted ``translate`` call plus, when available,
    ``sample_candidates`` fills the cap. Total extra calls per unit <= K.
    """
    candidates: list[tuple[str, str]] = []
    if callable(getattr(engine, "_chat", None)):
        for variant, compression_hint in _CONCISE_VARIANTS:
            if len(candidates) >= MAX_ISOMETRIC_CANDIDATES:
                break
            text = _chat_candidate(
                engine, source, src_lang, tgt_lang, compression_hint, budget
            )
            if text:
                candidates.append((variant, text))
        if candidates:
            return candidates
    # Non-chat engines: one budgeted re-translation, then samples up to the cap.
    text = _budgeted_repair(engine, source, src_lang, tgt_lang, budget)
    if text:
        candidates.append(("budgeted", text))
    if len(candidates) < MAX_ISOMETRIC_CANDIDATES and hasattr(
        engine, "sample_candidates"
    ):
        try:
            sampled = engine.sample_candidates(  # type: ignore[attr-defined]
                source, src_lang, tgt_lang, MAX_ISOMETRIC_CANDIDATES - len(candidates)
            )
        except Exception as exc:  # noqa: BLE001 - a failing sampler is dropped
            log.warning("isometric: sample_candidates failed (%s)", exc)
            sampled = []
        for position, sample in enumerate(sampled or []):
            if len(candidates) >= MAX_ISOMETRIC_CANDIDATES:
                break
            if str(sample or "").strip():
                candidates.append((f"sample-{position}", str(sample)))
    return candidates


def _rejection_reason(
    source: str,
    candidate: str,
    budget: int,
    current: str,
    previous_vi: str,
    next_vi: str,
    floor: float,
    src_lang: str,
    tgt_lang: str,
) -> str:
    """The first ``accept_shrunk`` condition ``candidate`` fails (for the record).

    Mirrors :func:`vidub.utils.isometry.accept_shrunk` so the persisted
    ``isometric_reason`` explains a rejection. ``previous_vi``/``next_vi`` are the
    neighbouring units' texts (never this unit's own ``current``). Returns ``""``
    when the candidate is acceptable.
    """
    if not candidate.strip():
        return "no-candidate-shorter"
    why = _validate(candidate, source, src_lang, tgt_lang)
    if why is not None:
        return f"validator:{why}"
    signals = guard_signals(source, candidate, min_coverage=floor)
    flags = list(signals["flags"])
    coverage = float(signals["coverage"])
    if "omission-risk" in flags or "sentence-loss" in flags or coverage < float(floor):
        return f"omission-risk:{coverage:.3f}<{float(floor):.3f}"
    contamination = contamination_signals(
        candidate, previous_vi, next_vi, source=source
    )
    if "context-bleed" in contamination["flags"]:
        overlap = max(
            float(contamination["overlap_prev"]), float(contamination["overlap_next"])
        )
        return f"context-bleed:{overlap:.2f}"
    if char_count(candidate) > char_count(current):
        return "no-candidate-shorter"
    if projected_stretch(candidate, budget) >= projected_stretch(current, budget):
        return "no-candidate-shorter"
    return ""


def _script_row(
    seg: dict,
    vi: str,
    truncated: bool,
    *,
    passthrough: bool = False,
    spliced: bool = False,
    low_confidence: bool = False,
) -> dict[str, object]:
    """Build one ``script.json`` row, adding the optional flags when set."""
    record: dict[str, object] = {
        "start": float(seg.get("start", 0.0)),
        "end": float(seg.get("end", 0.0)),
        "speaker": seg.get("speaker", "spk0"),
        "en": str(seg.get("text", "")),
        "vi": vi,
        "spoken_vi": vi,
        "subtitle_vi": vi,
        "truncated": truncated,
    }
    if passthrough:
        record["passthrough"] = True
    if spliced:
        record["glossary_spliced"] = True
    if low_confidence:
        record["low_confidence"] = True
    return record


def _run_qc_battery(
    rows: list[dict],
    glossary: dict[str, str],
    engine: BaseTranslator,
    manifest: JobManifest,
    cfg: JobConfig,
    artifacts_dir: Path,
) -> tuple[list[dict], dict | None]:
    """Run the level-2 battery with the M5-T2 refine/pristine-snapshot structure.

    Returns ``(rows, artifact)`` where ``artifact`` is ``None`` when the battery
    is disabled, so the caller writes no ``translation_qc.json`` (today's
    behaviour). ``rows`` are the scored rows — the per-segment rows on the off
    path, one row per unit on the sentence-unit path.
    """
    if not cfg.qc_translation:
        return rows, None
    # M5-T2 refine pass. Default-on for the hunyuan engine (cfg.hunyuan_refine is
    # None = auto); explicit True forces it on, explicit False
    # (--no-hunyuan-refine) restores the previous behaviour. The pass is only
    # ever built for the hunyuan engine, so envit5/ollama runs are byte-identical
    # to before. Construction and the battery call are both guarded so refine can
    # never fail a run (any error -> the refiner is dropped).
    refiner = None
    if cfg.hunyuan_refine is not False and _refine_engine(manifest, cfg) == "hunyuan":
        try:
            refiner = _build_refiner(cfg, glossary, manifest)
        except Exception as exc:  # noqa: BLE001 - refine must never fail a run
            log.warning("refine disabled: could not build refiner (%s)", exc)
            refiner = None
    if refiner is not None:
        # Pristine pre-battery snapshot (shallow per-row copies): if the
        # refiner raises, recompute the deterministic corrections from
        # scratch instead of re-applying them to already-mutated rows.
        battery_rows = [dict(row) for row in rows]
        try:
            rows, qc_artifact = run_battery(
                rows,
                glossary,
                engine,
                manifest.src_lang,
                manifest.tgt_lang,
                out_dir=artifacts_dir,
                refiner=refiner,
            )
        except Exception as exc:  # noqa: BLE001 - refine must never fail a run
            log.warning("refine pass failed (%s); rerunning without refine", exc)
            rows, qc_artifact = run_battery(
                battery_rows,
                glossary,
                engine,
                manifest.src_lang,
                manifest.tgt_lang,
                out_dir=artifacts_dir,
            )
    else:
        # No refiner: unchanged path, so a battery error still propagates.
        rows, qc_artifact = run_battery(
            rows,
            glossary,
            engine,
            manifest.src_lang,
            manifest.tgt_lang,
            out_dir=artifacts_dir,
        )
    return rows, qc_artifact


def _build_per_segment_script(
    engine: BaseTranslator,
    rows: list[dict],
    src_texts: list[str],
    passthrough_idx: set[int],
    translations: list[str],
    glossary: dict[str, str],
    manifest: JobManifest,
) -> tuple[list[dict], dict[str, int]]:
    """The per-segment (``--no-sentence-units``) loop: today's exact behaviour.

    Returns ``(script, counters)``; ``counters`` holds the glossary / hesitation
    / style counts the glossary sidecar records.
    """
    counters = {
        "replacements": 0,
        "spliced": 0,
        "enforced": 0,
        "hesitations": 0,
        "style_fixes": 0,
    }
    translated_iter = iter(translations)
    script: list[dict[str, object]] = []
    low_confidence_idx: list[int] = []

    def _translate_one(chunk: str) -> str:
        """Translate one short source chunk for the source-side glossary splice."""
        result = engine.translate([chunk], manifest.src_lang, manifest.tgt_lang)
        return str(result[0]) if result else ""

    for index, seg in enumerate(rows):
        source = src_texts[index]
        rebuilt = False
        if index in passthrough_idx:
            # Already user-language: the glossary must not touch passthrough rows.
            vi_text = source
        else:
            vi_text = next(translated_iter)
            if glossary:
                # Enforce terminology on the MT output *before* the length and
                # low-confidence checks so those act on the final text.
                vi_text, applied = apply_glossary(vi_text, glossary)
                counters["replacements"] += applied
                # Source-side enforcement: only re-translates when the model did
                # not already emit the enforced phrase.
                if source:
                    vi_text, rebuilt, enforced = splice_source_terms(
                        source, vi_text, glossary, _translate_one, max_terms=2
                    )
                    counters["enforced"] += enforced
                    if rebuilt:
                        counters["spliced"] += 1
        # Hesitation/repetition cleanup runs on every row (MT output, spliced
        # output and passthrough) before the length guard.
        cleaned, fixed = clean_hesitation(vi_text or "")
        counters["hesitations"] += fixed
        # Vietnamese word-order refinement runs on MT/spliced rows only, after
        # the hesitation cleanup and before the length guard. Passthrough rows
        # are user-language and must not be reordered.
        if index not in passthrough_idx:
            cleaned, style_fixed = apply_vnstyle(cleaned)
            counters["style_fixes"] += style_fixed
        cleaned, truncated = truncate_translation(cleaned.strip(), len(source.split()))
        low = _is_low_confidence(source, cleaned, manifest.tgt_lang)
        if low:
            low_confidence_idx.append(index)
        script.append(
            _script_row(
                seg,
                cleaned,
                truncated,
                passthrough=index in passthrough_idx,
                spliced=rebuilt,
                low_confidence=low,
            )
        )

    if low_confidence_idx:
        log.warning(
            "low-confidence translation for segment(s) %s: empty output or no "
            "Vietnamese diacritics on a >= %d-word source",
            low_confidence_idx,
            LOW_CONFIDENCE_MIN_WORDS,
        )
    return script, counters


def _unit_rows(
    rows: list[dict], src_texts: list[str], passthrough_idx: set[int]
) -> tuple[list[Unit], list[dict]]:
    """Adapt the sentence rows into :class:`Unit` objects, isolating passthrough.

    A passthrough row is forced into its own single-row unit (a unique speaker
    makes the cut safe on both sides), so it is never merged with a translated
    neighbour and never sent to the engine.

    Returns ``(units, adapted)``: ``units`` is ``group_units(adapted)`` and
    ``adapted`` is the exact row list it grouped, with the synthetic
    passthrough speakers. ``adapted`` keeps the SAME order and indices as
    ``rows`` (only a passthrough ``speaker`` differs), so callers keep using
    ``rows`` for the original indices while feeding ``adapted`` to
    :func:`vidub.utils.sentences.unit_cut_stats` — the synthetic speaker change
    is what classifies a passthrough boundary as a **speaker** cut instead of a
    mid-sentence cut. The adaptation stays internal bookkeeping: it never reaches
    ``units.json`` or ``script.json``.
    """
    adapted: list[dict] = []
    for index, seg in enumerate(rows):
        speaker = seg.get("speaker", "spk0")
        if index in passthrough_idx:
            speaker = f"__passthrough_{index}__"
        adapted.append(
            {
                "text": src_texts[index],
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "speaker": speaker,
            }
        )
    return group_units(adapted), adapted


def _passthrough_guard_record(source: str) -> dict:
    """The guard record for a unit that is never translated (passthrough)."""
    signals = guard_signals(source, source)
    return {
        "words_in": signals["words_in"],
        "words_out": signals["words_out"],
        "coverage": signals["coverage"],
        "sentences_in": signals["sentences_in"],
        "sentences_out": signals["sentences_out"],
        "flags": signals["flags"],
        "action": "none",
        "candidates": 1,
        # A passthrough unit is never sent to the engine and has no neighbour
        # translation, so its contamination overlaps are exactly zero.
        "overlap_prev": 0.0,
        "overlap_next": 0.0,
        "overlap_prev_after": 0.0,
        "overlap_next_after": 0.0,
    }


def _translate_sentence_units(
    engine: BaseTranslator,
    rows: list[dict],
    src_texts: list[str],
    passthrough_idx: set[int],
    manifest: JobManifest,
    cfg: JobConfig,
    glossary: dict[str, str],
    artifacts_dir: Path,
) -> _UnitPathResult:
    """Translate + QC the sentence-unit path: units in, sentence rows out.

    One unit is translated as one batch item, guarded for completeness, run
    through the deterministic passes **as a unit**, scored by the battery as a
    unit and only then split back onto its member sentence rows.
    """
    units, adapted_rows = _unit_rows(rows, src_texts, passthrough_idx)
    non_pt = [unit for unit in units if unit.indices[0] not in passthrough_idx]
    unit_sources = [unit.source for unit in non_pt]

    # ISO-1: one character budget per unit from its locked window (plus a
    # documented share of the following silence is available but defaults to 0,
    # matching the T1 baseline). Budgets are computed for every unit so the
    # record is uniform; the engine batch passes only the non-passthrough
    # budgets, parallel to ``unit_sources`` (passthrough units are never sent to
    # the engine, so they are excluded from the budget list consistently).
    isometric = bool(cfg.isometric)
    unit_budgets: list[int] = []
    unit_windows = [float(unit.end) - float(unit.start) for unit in units]
    if isometric:
        for index, unit in enumerate(units):
            next_start = units[index + 1].start if index + 1 < len(units) else None
            unit_budgets.append(
                unit_budget(
                    unit.start,
                    unit.end,
                    next_start,
                    headroom=cfg.isometric_headroom,
                    chars_per_second=cfg.isometric_cps,
                )
            )

    # ISO-1-T3-C1 adaptive omission floor: 0.65 below the tiny-window threshold,
    # else the normal 0.75. Only the isometric path uses it; `--no-isometric`
    # keeps UNIT_COVERAGE_MIN so it stays byte-identical. ISO-1-C1-FIX: this SAME
    # resolved floor is handed to guard_units (``min_coverage``), to the rejection
    # reason and to ``accept_shrunk(..., floor=…)``, so the configured
    # ``isometric_short_window`` / ``isometric_short_coverage`` knobs decide what
    # is accepted (not just what is recorded).
    unit_floors = [
        coverage_floor(
            unit_windows[index],
            short_window=cfg.isometric_short_window,
            normal=UNIT_COVERAGE_MIN,
            short=cfg.isometric_short_coverage,
        )
        if isometric
        else UNIT_COVERAGE_MIN
        for index in range(len(units))
    ]

    engine_budgets: list[int] | None = None
    if isometric and _engine_accepts_budgets(engine):
        engine_budgets = [
            unit_budgets[upos]
            for upos, unit in enumerate(units)
            if unit.indices[0] not in passthrough_idx
        ]
    translate_kwargs = {"budgets": engine_budgets} if engine_budgets is not None else {}
    unit_translations = (
        engine.translate_contextual(
            unit_sources, manifest.src_lang, manifest.tgt_lang, **translate_kwargs
        )
        if unit_sources
        else []
    )
    if len(unit_translations) != len(unit_sources):
        raise TranslationError(
            f"translator returned {len(unit_translations)} result(s) for "
            f"{len(unit_sources)} unit source(s)"
        )

    # SA-1-T2 / Q-1-T1 completeness guard: repair a dropping or neighbour-bleeding
    # translation before the deterministic passes act on it. Never raises. The
    # neighbours are the *raw* engine translations (unit_translations[i-1/i+1]),
    # so a unit that reproduced a neighbour's output is flagged against exactly
    # the text the engine produced (empty strings at the two ends).
    neighbours = [
        (
            unit_translations[index - 1] if index - 1 >= 0 else "",
            unit_translations[index + 1] if index + 1 < len(unit_translations) else "",
        )
        for index in range(len(unit_sources))
    ]
    guarded, guard_records = guard_units(
        unit_sources,
        unit_translations,
        engine,
        manifest.src_lang,
        manifest.tgt_lang,
        neighbours=neighbours,
        min_coverage=(
            [
                unit_floors[upos]
                for upos, unit in enumerate(units)
                if unit.indices[0] not in passthrough_idx
            ]
            if isometric
            else None
        ),
    )

    guard_by_unit: dict[int, dict | None] = {}
    texts_by_unit: list[str] = []
    cursor = 0
    for upos, unit in enumerate(units):
        if unit.indices[0] in passthrough_idx:
            guard_by_unit[upos] = None
            texts_by_unit.append(unit.source)
        else:
            guard_by_unit[upos] = guard_records[cursor]
            texts_by_unit.append(guarded[cursor])
            cursor += 1

    # ISO-1-T3-C1 bounded Best-of-K compression: after the completeness guard,
    # every unit still over its character budget gets up to K=3 concise
    # candidates (dynamic prompts for chat engines; one budgeted re-translation
    # plus samples otherwise). A candidate replaces the text ONLY when the
    # deterministic ``accept_shrunk`` predicate passes (with the tiered floor of
    # the unit's window); the SHORTEST passing candidate wins (ties: earliest).
    # Every outcome persists a specific ``isometric_reason``.
    isometric_outcome: dict[int, dict] = {}
    if isometric:
        can_compress = _accepts_budgets(engine.translate) or callable(
            getattr(engine, "_chat", None)
        )
        for upos, unit in enumerate(units):
            budget = unit_budgets[upos]
            window = unit_windows[upos]
            floor = unit_floors[upos]
            current = texts_by_unit[upos]
            outcome = {
                "budget_chars": budget,
                "chars": char_count(current),
                "over_budget": over_budget(current, budget),
                "projected_stretch_before": projected_stretch(current, budget),
                "coverage_floor": floor,
                "short_floor_applied": floor < UNIT_COVERAGE_MIN,
                "isometric": "none",
                "isometric_reason": "budget-ok",
                "isometric_candidates": 0,
                "isometric_variant": None,
            }
            if unit.indices[0] not in passthrough_idx and outcome["over_budget"]:
                if not can_compress:
                    outcome["isometric"] = "unavailable"
                    outcome["isometric_reason"] = "engine-unavailable"
                else:
                    # The contamination check compares against the NEIGHBOURING
                    # units' final translations, never this unit's own text (a
                    # genuine compression shares most tokens with the text it
                    # compresses and would otherwise be false-flagged).
                    prev_vi = texts_by_unit[upos - 1] if upos - 1 >= 0 else ""
                    next_vi = (
                        texts_by_unit[upos + 1] if upos + 1 < len(units) else ""
                    )
                    candidates = _compression_candidates(
                        engine,
                        unit.source,
                        manifest.src_lang,
                        manifest.tgt_lang,
                        budget,
                    )
                    outcome["isometric_candidates"] = len(candidates)
                    best: str | None = None
                    best_variant: str | None = None
                    reject_reason = "no-candidate-shorter"
                    for variant, raw in candidates:
                        cleaned, _fixed = clean_hesitation(str(raw or ""))
                        cleaned = cleaned.strip()
                        if not cleaned:
                            continue
                        if accept_shrunk(
                            unit.source,
                            cleaned,
                            budget,
                            previous_vi=prev_vi,
                            next_vi=next_vi,
                            current_vi=current,
                            src_lang=manifest.src_lang,
                            tgt_lang=manifest.tgt_lang,
                            window=window,
                            # ISO-1-C1-FIX: the acceptance predicate must use the
                            # SAME configured floor the rest of the unit path uses
                            # (``unit_floors`` was built from
                            # ``cfg.isometric_short_window`` /
                            # ``cfg.isometric_short_coverage``), otherwise the
                            # knobs change the record but not what is accepted.
                            floor=floor,
                        ):
                            if best is None or char_count(cleaned) < char_count(best):
                                best = cleaned
                                best_variant = variant
                        else:
                            reason = _rejection_reason(
                                unit.source,
                                cleaned,
                                budget,
                                current,
                                prev_vi,
                                next_vi,
                                floor,
                                manifest.src_lang,
                                manifest.tgt_lang,
                            )
                            reject_reason = reason or reject_reason
                    if best is not None:
                        texts_by_unit[upos] = best
                        outcome["isometric"] = "shrunk"
                        outcome["isometric_reason"] = "shrunk"
                        outcome["isometric_variant"] = best_variant
                    else:
                        outcome["isometric"] = "kept"
                        outcome["isometric_reason"] = reject_reason
            isometric_outcome[upos] = outcome

    counters = {
        "replacements": 0,
        "spliced": 0,
        "enforced": 0,
        "hesitations": 0,
        "style_fixes": 0,
    }

    def _translate_one(chunk: str) -> str:
        """Translate one short source chunk for the source-side glossary splice."""
        result = engine.translate([chunk], manifest.src_lang, manifest.tgt_lang)
        return str(result[0]) if result else ""

    # Deterministic passes run ON THE UNIT TEXT (never on a display fragment).
    final_texts: list[str] = []
    truncated_by_unit: list[bool] = []
    low_by_unit: list[bool] = []
    spliced_by_unit: list[bool] = []
    low_confidence_units: list[int] = []
    for upos, unit in enumerate(units):
        if unit.indices[0] in passthrough_idx:
            final_texts.append(unit.source)
            truncated_by_unit.append(False)
            low_by_unit.append(False)
            spliced_by_unit.append(False)
            continue
        text = texts_by_unit[upos]
        rebuilt = False
        if glossary:
            text, applied = apply_glossary(text, glossary)
            counters["replacements"] += applied
            if unit.source:
                text, rebuilt, enforced = splice_source_terms(
                    unit.source, text, glossary, _translate_one, max_terms=2
                )
                counters["enforced"] += enforced
                if rebuilt:
                    counters["spliced"] += 1
        cleaned, fixed = clean_hesitation(text or "")
        counters["hesitations"] += fixed
        cleaned, style_fixed = apply_vnstyle(cleaned)
        counters["style_fixes"] += style_fixed
        cleaned, truncated = truncate_translation(
            cleaned.strip(), len(unit.source.split())
        )
        low = _is_low_confidence(unit.source, cleaned, manifest.tgt_lang)
        if low:
            low_confidence_units.append(upos)
        final_texts.append(cleaned)
        truncated_by_unit.append(truncated)
        low_by_unit.append(low)
        spliced_by_unit.append(rebuilt)

    if low_confidence_units:
        log.warning(
            "low-confidence translation for unit(s) %s: empty output or no "
            "Vietnamese diacritics on a >= %d-word source",
            low_confidence_units,
            LOW_CONFIDENCE_MIN_WORDS,
        )

    # ONE battery row per unit: the battery scores units, not display fragments.
    battery_rows: list[dict] = []
    for upos, unit in enumerate(units):
        row: dict[str, object] = {
            "start": float(unit.start),
            "end": float(unit.end),
            "speaker": rows[unit.indices[0]].get("speaker", "spk0"),
            "en": unit.source,
            "vi": final_texts[upos],
            "truncated": truncated_by_unit[upos],
        }
        if unit.indices[0] in passthrough_idx:
            row["passthrough"] = True
        if spliced_by_unit[upos]:
            row["glossary_spliced"] = True
        # A unit the guard could not repair (still omission-/sentence-loss) is
        # low-confidence by definition, so the battery can never label it ``ok``
        # (``_score_row`` folds ``low_confidence`` into the ``unnatural``
        # signal). The plain per-unit ``_is_low_confidence`` result is included
        # exactly as on the off path.
        record = guard_by_unit[upos]
        if low_by_unit[upos] or (record is not None and record["action"] == "flagged"):
            row["low_confidence"] = True
        battery_rows.append(row)

    battery_rows, qc_artifact = _run_qc_battery(
        battery_rows, glossary, engine, manifest, cfg, artifacts_dir
    )

    battery_label: dict[int, str] = {}
    if qc_artifact is not None:
        for entry in qc_artifact["segments"]:
            unit_index = int(entry["index"])
            battery_label[unit_index] = str(entry["label"])
            entry["members"] = list(units[unit_index].indices)

    # M5-T2 refine outcomes (one per attempted/budget-skipped unit), keyed by the
    # battery row index == the unit index on this path. Surfaced in units.detail
    # so a report can see which unit the refine pass touched and what it did.
    refine_outcome: dict[int, dict] = {}
    refine_summary = qc_artifact.get("refine") if qc_artifact is not None else None
    if isinstance(refine_summary, dict):
        for outcome in refine_summary.get("outcomes", []):
            refine_outcome[int(outcome["index"])] = outcome

    resolved_records = [
        guard_by_unit[upos] or _passthrough_guard_record(unit.source)
        for upos, unit in enumerate(units)
    ]

    # Sentence rows: every member row gets its piece of the final unit text.
    script: list[dict[str, object]] = []
    for upos, unit in enumerate(units):
        if unit.indices[0] in passthrough_idx:
            for index in unit.indices:
                script.append(
                    _script_row(rows[index], src_texts[index], False, passthrough=True)
                )
            continue
        vi_text = str(battery_rows[upos].get("vi", ""))
        truncated = bool(battery_rows[upos].get("truncated", False))
        low = bool(battery_rows[upos].get("low_confidence", False))
        spliced = bool(battery_rows[upos].get("glossary_spliced", False))
        pieces = split_translation(vi_text, unit.shares)
        for position, index in enumerate(unit.indices):
            script.append(
                _script_row(
                    rows[index],
                    pieces[position],
                    truncated,
                    spliced=spliced,
                    low_confidence=low,
                )
            )

    summary = unit_share_summary(units)
    # SA-1-T2-C1: classify the closures against the SAME adapted rows the units
    # were grouped from, so a passthrough-isolated boundary is a speaker cut and
    # ``mid_sentence_cuts`` is 0 for every input the stage can see.
    cut_stats = unit_cut_stats(units, adapted_rows)
    def _iso_detail(unit_pos: int) -> dict:
        """The ISO-1 fields surfaced on ``units.detail`` (isometric on only)."""
        outcome = isometric_outcome[unit_pos]
        return {
            "isometric": outcome["isometric"],
            "isometric_reason": outcome["isometric_reason"],
            "isometric_candidates": outcome["isometric_candidates"],
            "coverage_floor": outcome["coverage_floor"],
            "short_floor_applied": outcome["short_floor_applied"],
            "budget_chars": outcome["budget_chars"],
            "over_budget": outcome["over_budget"],
        }

    detail = [
        {
            "index": upos,
            "indices": list(unit.indices),
            "words_in": resolved_records[upos]["words_in"],
            "words_out": resolved_records[upos]["words_out"],
            "coverage": resolved_records[upos]["coverage"],
            "sentences_in": resolved_records[upos]["sentences_in"],
            "sentences_out": resolved_records[upos]["sentences_out"],
            "flags": list(resolved_records[upos]["flags"]),
            "candidates": resolved_records[upos]["candidates"],
            "action": resolved_records[upos]["action"],
            # Q-1-T1 neighbour contamination: the before (detected) and after
            # (chosen candidate) containment of each neighbour's content tokens.
            "overlap_prev": resolved_records[upos].get("overlap_prev", 0.0),
            "overlap_next": resolved_records[upos].get("overlap_next", 0.0),
            "overlap_prev_after": resolved_records[upos].get("overlap_prev_after", 0.0),
            "overlap_next_after": resolved_records[upos].get("overlap_next_after", 0.0),
            "battery": battery_label.get(upos, "skipped"),
            # M5-T2 refine action for this unit ("none" when it was never a
            # refine candidate): refine-best / refine-fusion / refine-skipped /
            # refine-invalid / refine-budget.
            "refine": refine_outcome.get(upos, {}).get("action", "none"),
            **(_iso_detail(upos) if isometric else {}),
        }
        for upos, unit in enumerate(units)
    ]
    unit_flags = {
        "guard_flagged": sum(
            1 for record in guard_records if record["action"] == "flagged"
        ),
        "omission": sum(
            1 for record in guard_records if "omission-risk" in record["flags"]
        ),
        "sentence_loss": sum(
            1 for record in guard_records if "sentence-loss" in record["flags"]
        ),
        "addition": sum(
            1 for record in guard_records if "addition-risk" in record["flags"]
        ),
        # Q-1-T1: units the guard saw reproducing a neighbour's content (before
        # repair), so the count documents the defect even when a unit was fixed.
        "contamination": sum(
            1
            for record in guard_records
            if "context-bleed" in record["flags"]
            or "sentence-addition" in record["flags"]
        ),
        "oversized": int(cut_stats["oversized"]),
    }
    if isometric:
        # ISO-1 outcomes: how many units are still over budget after the repair
        # (and the deterministic passes/battery), and how the bounded repair
        # resolved (shrunk = accepted a shorter candidate, kept = rejected it).
        unit_flags["over_budget_after"] = sum(
            1
            for upos in range(len(units))
            if over_budget(str(battery_rows[upos].get("vi", "")), unit_budgets[upos])
        )
        unit_flags["isometric_shrunk"] = sum(
            1
            for outcome in isometric_outcome.values()
            if outcome["isometric"] == "shrunk"
        )
        unit_flags["isometric_kept"] = sum(
            1
            for outcome in isometric_outcome.values()
            if outcome["isometric"] == "kept"
        )
        # ISO-1-T3-C1: how many units used the lowered tiny-window floor (always
        # recorded, never silent).
        unit_flags["isometric_short_floor"] = sum(
            1
            for outcome in isometric_outcome.values()
            if outcome["short_floor_applied"]
        )
    unit_block = {
        "mode": "sentence",
        "units": summary["units"],
        "segments": summary["segments"],
        "merged_segments": summary["merged_segments"],
        "max_unit_words": max(
            (len(unit.source.split()) for unit in units), default=0
        ),
        "max_unit_seconds": max(
            (unit.end - unit.start for unit in units), default=0.0
        ),
        "detail": detail,
        "flags": unit_flags,
    }

    units_doc: dict[str, object] = {
        "version": 1,
        "text_contract_version": 1,
        "mode": "sentence",
        "units": [],
    }
    unit_entries: list[dict] = []
    for upos, unit in enumerate(units):
        final_text = str(battery_rows[upos].get("vi", ""))
        record = resolved_records[upos]
        entry: dict[str, object] = {
            "index": upos,
            "indices": list(unit.indices),
            "start": float(unit.start),
            "end": float(unit.end),
            "speaker": rows[unit.indices[0]].get("speaker", "spk0"),
            "source": unit.source,
            "vi": final_text,
            "spoken_vi": final_text,
            "subtitle_vi": final_text,
            "words_in": record["words_in"],
            "coverage": record["coverage"],
            "guard": record["action"],
            "battery": battery_label.get(upos, "skipped"),
            "open": not ends_sentence(final_text),
        }
        if isometric:
            # ISO-1 per-unit record: the window budget, the pre-repair text's
            # character count / over-budget state / projected stretch, the
            # bounded Best-of-K outcome + reason, the tiered coverage floor and
            # whether the tiny-window exception applied, and the final
            # (post-repair, post-passes) text's character count / stretch.
            outcome = isometric_outcome[upos]
            entry["budget_chars"] = outcome["budget_chars"]
            entry["chars"] = outcome["chars"]
            entry["over_budget"] = outcome["over_budget"]
            entry["isometric"] = outcome["isometric"]
            entry["chars_after"] = char_count(final_text)
            entry["projected_stretch_before"] = outcome["projected_stretch_before"]
            entry["projected_stretch_after"] = projected_stretch(
                final_text, outcome["budget_chars"]
            )
            entry["coverage_floor"] = outcome["coverage_floor"]
            entry["short_floor_applied"] = outcome["short_floor_applied"]
            entry["isometric_reason"] = outcome["isometric_reason"]
            entry["isometric_candidates"] = outcome["isometric_candidates"]
            entry["isometric_variant"] = outcome["isometric_variant"]
        unit_entries.append(entry)
    units_doc["units"] = unit_entries

    log.info(
        "sentence units: %d units from %d sentence rows (%d merged); "
        "mid-sentence cuts: %d, oversized: %d, guard flagged: %d",
        summary["units"],
        summary["segments"],
        summary["merged_segments"],
        cut_stats["mid_sentence_cuts"],
        cut_stats["oversized"],
        unit_flags["guard_flagged"],
    )

    return _UnitPathResult(
        script=script,
        counters=counters,
        qc_artifact=qc_artifact,
        unit_block=unit_block,
        units_doc=units_doc,
    )


def translate(
    manifest_path: str,
    cfg: JobConfig,
    translator: BaseTranslator | None = None,
) -> JobManifest:
    """Translate the job's segments and write ``script.json``.

    ``translator`` may be supplied to override the engine (used by tests);
    otherwise the engine named in the manifest is used. Segments whose source
    text is already Vietnamese (diacritic ratio >= :data:`PASSTHROUGH_RATIO`)
    skip the MT call and are copied through, flagged ``"passthrough": true``.
    """
    manifest = JobManifest.load(manifest_path)
    artifacts_dir = manifest.artifacts_dir

    segments_rel = manifest.paths.get("segments", SEGMENTS_NAME)
    segments_path = artifacts_dir / segments_rel
    if not segments_path.exists():
        raise FileNotFoundError(
            f"segments not found: {segments_path}. "
            "Run the transcribe stage first, e.g. "
            f"'python -m vidub.stages.transcribe {manifest_path}'."
        )

    raw_segments = json.loads(segments_path.read_text(encoding="utf-8"))
    # SA-1-T2: on the sentence-unit path the ROW unit is a *sentence*. The ASR
    # rows (manifest.segments / segments.json) stay untouched; only the working
    # row list downstream is split at internal sentence boundaries, with each
    # sentence's start/end locked to the ASR word edges. Off path: unchanged.
    rows = split_into_sentences(raw_segments) if cfg.sentence_units else raw_segments
    src_texts = [str(seg.get("text", "")) for seg in rows]

    tgt_lang = (manifest.tgt_lang or "").strip().lower()
    passthrough_idx = {
        index
        for index, text in enumerate(src_texts)
        if tgt_lang == "vi" and vn_diacritic_ratio(text) >= PASSTHROUGH_RATIO
    }
    if passthrough_idx:
        log.warning(
            "translation passthrough for already-Vietnamese segment(s) %s "
            "(the ASR language guard should normally have caught this)",
            sorted(passthrough_idx),
        )

    glossary: dict[str, str] = {}
    if cfg.glossary_path:
        glossary = load_glossary(cfg.glossary_path)
        log.info(
            "loaded glossary %s (%d term(s))", cfg.glossary_path, len(glossary)
        )

    engine = translator if translator is not None else build_translator(manifest, cfg)
    # Feed the loaded glossary to context-aware LLM engines so their terminology
    # template sees the enforced pairs (M5-T2); the deterministic enforcement
    # below is unchanged. Fixes the dormant-glossary debt for ollama + hunyuan.
    if glossary and hasattr(engine, "glossary"):
        engine.glossary = glossary

    unit_block: dict | None = None
    units_doc: dict | None = None
    if cfg.sentence_units:
        # TU-1 / SA-1-T2: the sentence is the row unit, the unit is the
        # translation + QC unit. The helper translates each unit, guards its
        # completeness, runs the deterministic passes on the unit text, scores
        # the units with the battery and hands back the sentence rows.
        result = _translate_sentence_units(
            engine,
            rows,
            src_texts,
            passthrough_idx,
            manifest,
            cfg,
            glossary,
            artifacts_dir,
        )
        script = result.script
        counters = result.counters
        qc_artifact = result.qc_artifact
        unit_block = result.unit_block
        units_doc = result.units_doc
    else:
        # --no-sentence-units: the previous per-segment behaviour, unchanged.
        to_translate = [
            text
            for index, text in enumerate(src_texts)
            if index not in passthrough_idx
        ]
        translations = (
            engine.translate_contextual(
                to_translate, manifest.src_lang, manifest.tgt_lang
            )
            if to_translate
            else []
        )
        if len(translations) != len(to_translate):
            raise TranslationError(
                f"translator returned {len(translations)} result(s) for "
                f"{len(to_translate)} input(s)"
            )
        script, counters = _build_per_segment_script(
            engine, rows, src_texts, passthrough_idx, translations, glossary, manifest
        )
        # Level-2 translation QC battery (QUALITY-ROADMAP.md §3.3): score the
        # fully built rows, correct what is safely correctable (glossary splice /
        # one re-translation) and record a translation verdict. Runs after every
        # existing pass and before script.json is written, so corrected text is
        # what downstream stages read.
        script, qc_artifact = _run_qc_battery(
            script, glossary, engine, manifest, cfg, artifacts_dir
        )

    # QC corrections are part of the accepted translation too; initialize both
    # selections from the final script text only after the battery has finished.
    for row in script:
        row["spoken_vi"] = str(row.get("vi", ""))
        row["subtitle_vi"] = str(row.get("vi", ""))

    if glossary:
        log.info(
            "glossary: %d term(s) applied (%d replacement(s), %d source-side "
            "enforcement(s), %d spliced)",
            len(glossary),
            counters["replacements"],
            counters["enforced"],
            counters["spliced"],
        )
    if counters["hesitations"]:
        log.info(
            "cleaned %d hesitation artifact(s) from the translated text",
            counters["hesitations"],
        )
    if counters["style_fixes"]:
        log.info(
            "refined Vietnamese word order in %d place(s)", counters["style_fixes"]
        )

    # Always write the sidecar (zeros when no glossary) so a later glossary-free
    # re-run cannot inherit stale counts from a previous run.
    (artifacts_dir / GLOSSARY_NAME).write_text(
        json.dumps(
            {
                "terms": len(glossary),
                "replacements": counters["replacements"],
                "spliced": counters["spliced"],
                "enforced": counters["enforced"],
                "hesitations": counters["hesitations"],
                "style_fixes": counters["style_fixes"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    if qc_artifact is not None:
        if unit_block is not None:
            # SA-1-T2 unit statistics + the per-unit guard/battery records and
            # flag counts. Only added when the sentence-unit path ran, so
            # --no-sentence-units keeps today's artifact byte-identical.
            qc_artifact["units"] = unit_block
        (artifacts_dir / TRANSLATION_QC_NAME).write_text(
            json.dumps(qc_artifact, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        manifest.paths["translation_qc"] = TRANSLATION_QC_NAME
        manifest.translation_verdict = str(qc_artifact.get("verdict", ""))
        refine_info = qc_artifact.get("refine")
        if refine_info:
            log.info(
                "translation QC: verdict=%s counts=%s (roundtrip=%s, refine=%s)",
                qc_artifact["verdict"],
                qc_artifact["counts"],
                qc_artifact["roundtrip"],
                refine_info,
            )
        else:
            log.info(
                "translation QC: verdict=%s counts=%s (roundtrip=%s)",
                qc_artifact["verdict"],
                qc_artifact["counts"],
                qc_artifact["roundtrip"],
            )

    (artifacts_dir / SCRIPT_NAME).write_text(
        json.dumps(script, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if units_doc is not None:
        # SA-1-T2 unit map: the authoritative unit text + locked span + members.
        (artifacts_dir / UNITS_NAME).write_text(
            json.dumps(units_doc, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        manifest.paths["units"] = UNITS_NAME

    manifest.paths["script"] = SCRIPT_NAME
    manifest.paths["glossary"] = GLOSSARY_NAME
    manifest.text_contract_version = 1
    manifest.stages["translate"] = "done"
    manifest.save(artifacts_dir / "job.json")
    return manifest


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``python -m vidub.stages.translate <job.json>``."""
    parser = argparse.ArgumentParser(
        prog="vidub.stages.translate",
        description="S2 translate: segments.json -> script.json (Vietnamese).",
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
        manifest = translate(args.job, cfg)
    except (FileNotFoundError, TranslationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(str(manifest.artifacts_dir / SCRIPT_NAME))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
