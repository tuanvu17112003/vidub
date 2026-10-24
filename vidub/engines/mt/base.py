"""Machine-translation engine interface.

Pipeline stages depend only on :class:`BaseTranslator`, so the concrete engine
(envit5 in M1, a local LLM in M4) can be swapped without touching stage code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

__all__ = ["BaseTranslator", "TranslationError"]


class TranslationError(RuntimeError):
    """Raised when a translation engine cannot load or produce output."""


def _check_budgets(
    texts: Sequence[str], budgets: Sequence[int] | None
) -> list[int] | None:
    """Validate an optional per-item character ``budgets`` sequence.

    ``budgets`` is parallel to ``texts`` — one target *character* count per
    input, as produced by :func:`vidub.utils.isometry.unit_budget`. ``None`` (the
    default everywhere) keeps today's behaviour exactly. A length mismatch
    raises ``ValueError`` so a caller can never silently misalign or drop
    budgets; the budgets are returned as a plain ``list[int]`` (or ``None``).
    """
    if budgets is None:
        return None
    budget_list = [int(budget) for budget in budgets]
    if len(budget_list) != len(texts):
        raise ValueError(
            "budgets must be parallel to texts: got "
            f"{len(budget_list)} budget(s) for {len(texts)} text(s)"
        )
    return budget_list


class BaseTranslator(ABC):
    """Interface every MT engine must implement."""

    @abstractmethod
    def translate(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
        *,
        budgets: Sequence[int] | None = None,
    ) -> list[str]:
        """Translate ``texts`` from ``src_lang`` to ``tgt_lang``.

        Returns one translated string per input, in the same order.

        ``budgets`` optionally carries one target *character* count per input
        (``None`` — the default — means no length control, today's behaviour).
        A budget is only a target: an engine may ignore it, and acceptance is
        decided by the caller (see :mod:`vidub.utils.isometry`), never by the
        engine. When given it must be parallel to ``texts``; a length mismatch
        raises ``ValueError``.
        """
        raise NotImplementedError

    def translate_contextual(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
        *,
        budgets: Sequence[int] | None = None,
    ) -> list[str]:
        """Translate ``texts`` while taking document context into account.

        The default engine has no context and simply delegates to
        :meth:`translate`, so segment-by-segment engines (envit5) are unchanged.
        Engines that can see neighbouring segments and a terminology table (the
        Ollama LLM engine, :mod:`vidub.engines.mt.ollama_mt`, and the Hunyuan
        HY-MT1.5 specialist, :mod:`vidub.engines.mt.hunyuan_mt`) override this;
        the translate stage prefers it for the main batch and keeps
        :meth:`translate` for the short glossary/batch splices.

        ``budgets`` is the same optional per-input character-budget hint as
        :meth:`translate`. With no budgets the delegation is the exact three-arg
        call that existed before, so unchanged engines keep working untouched.
        """
        checked = _check_budgets(texts, budgets)
        if checked is None:
            return self.translate(texts, src_lang, tgt_lang)
        return self.translate(texts, src_lang, tgt_lang, budgets=checked)
