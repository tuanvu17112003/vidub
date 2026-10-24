"""The translate stage entry point: :func:`translate` + CLI (S2)."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from vidub.config import JobConfig
from vidub.engines.mt.base import BaseTranslator, TranslationError
from vidub.manifest import JobManifest
from vidub.utils.glossary import apply_glossary, load_glossary, splice_source_terms
from vidub.utils.langcheck import vn_diacritic_ratio
from vidub.utils.sentences import split_into_sentences
from vidub.utils.textclean import clean_hesitation
from vidub.utils.vnstyle import apply_vnstyle

from vidub.stages.translate.constants import (
    GLOSSARY_NAME,
    LOW_CONFIDENCE_MIN_WORDS,
    PASSTHROUGH_RATIO,
    SCRIPT_NAME,
    SEGMENTS_NAME,
    TRANSLATION_QC_NAME,
    UNITS_NAME,
)
from vidub.stages.translate.qc import _run_qc_battery, _script_row
from vidub.stages.translate.engines import (
    _is_low_confidence,
    build_translator,
    truncate_translation,
)
from vidub.stages.translate.units import (
    _translate_sentence_units,
)

log = logging.getLogger(__name__)


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
