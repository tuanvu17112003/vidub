"""MT engine construction + deterministic text guards (build/refine/length)."""

from __future__ import annotations

import logging
from collections.abc import Callable

from vidub.config import (
    DEFAULT_HUNYUAN_HOST,
    DEFAULT_HUNYUAN_MODEL,
    JobConfig,
)
from vidub.engines.mt.base import BaseTranslator, TranslationError
from vidub.engines.mt.envit5_mt import Envit5Translator
from vidub.manifest import JobManifest
from vidub.utils.langcheck import VN_CHARS

from vidub.stages.translate.constants import (
    LENGTH_RATIO,
    LOW_CONFIDENCE_MIN_WORDS,
    _SENTENCE_END,
)

log = logging.getLogger(__name__)


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

