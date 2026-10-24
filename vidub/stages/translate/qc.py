"""Level-2 translation QC battery wrapper + script row shaping."""

from __future__ import annotations

import logging
from pathlib import Path

from vidub.config import JobConfig
from vidub.engines.mt.base import BaseTranslator
from vidub.manifest import JobManifest
from vidub.utils.tqc import run_battery

from vidub.stages.translate.engines import _build_refiner, _refine_engine

log = logging.getLogger(__name__)


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

