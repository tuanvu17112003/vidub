"""VietAI ``envit5-translation`` machine-translation engine.

The model is a T5 seq2seq model; each input string must be prefixed with the
*source* language tag (``"en: "`` / ``"vi: "``) and the model returns strings
prefixed with the *target* language tag, which is stripped here.

The heavy dependencies (torch, transformers) are imported lazily on first use so
importing this module never pulls them; model loading failures raise
:class:`~vidub.engines.mt.base.TranslationError`.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

from vidub.engines.mt.base import BaseTranslator, TranslationError, _check_budgets

__all__ = ["Envit5Translator"]

log = logging.getLogger(__name__)

MODEL_NAME = "VietAI/envit5-translation"
BATCH_SIZE = 16
# Tokenizer truncation window (the model's own positional limit).
INPUT_MAX_LENGTH = 512
# Decode settings: scaled/beam search with a repetition guard. max_length caps
# the generated Vietnamese to 256 tokens (the old 512 let the model ramble and
# repeat), and no_repeat_ngram_size stops it echoing the input verbatim.
MAX_LENGTH = 256
NUM_BEAMS = 4
NO_REPEAT_NGRAM_SIZE = 6

# ------------------------------------------------------- budgeted decode (ISO-1)
# Conservative characters-per-subword ratio used to turn a per-unit character
# budget (``vidub.utils.isometry.unit_budget``) into a token cap. Vietnamese
# averages ~4-5 characters per word and the ViT5 sentencepiece vocabulary splits
# many multi-syllable words further; 3.0 therefore *over*-estimates the token
# count for a given character budget, so the resulting ``max_length`` is generous
# and never truncates below what the budget asks for.
VI_CHARS_PER_TOKEN = 3.0
# A budgeted decode is never capped below this many tokens, so a tiny or
# mis-measured character budget cannot truncate the model mid-sentence. It
# mirrors the isometry floor (``MIN_BUDGET_CHARS = 8``); the caller's omission
# floor (coverage >= ``UNIT_COVERAGE_MIN``) is enforced after synthesis, not by
# this cap.
MIN_BUDGET_TOKENS = 8
# Beam-search length bias (< 1.0 favours shorter outputs) applied only when a
# budget is set. A mild 0.6 nudges toward the target without overriding the token
# cap; without budgets no ``length_penalty`` is passed at all.
LENGTH_PENALTY_BUDGETED = 0.6


def _budget_max_length(budget_chars: int) -> int:
    """Decode token cap for one character budget, floored then capped.

    ``ceil(budget_chars / VI_CHARS_PER_TOKEN)`` clamped to
    ``[MIN_BUDGET_TOKENS, MAX_LENGTH]``. Pure.
    """
    tokens = math.ceil(max(int(budget_chars), 0) / VI_CHARS_PER_TOKEN)
    return min(max(tokens, MIN_BUDGET_TOKENS), MAX_LENGTH)


class Envit5Translator(BaseTranslator):
    """envit5-backed translator (En↔Vi), lazily loaded on first use."""

    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None) -> None:
        self.model_name = model_name
        self._requested_device = device
        self.device = device
        self._tokenizer = None
        self._model = None
        self._torch = None

    # -------------------------------------------------------------- lazy loading
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
            from transformers import (
                AutoModelForSeq2SeqLM,
                AutoTokenizer,
                PreTrainedTokenizerFast,
            )
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise TranslationError(
                "transformers/torch are not installed; install them with "
                "'pip install torch --index-url https://download.pytorch.org/whl/cpu' "
                "then 'pip install transformers sentencepiece'"
            ) from exc

        device = self._requested_device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        try:
            tokenizer = self._load_tokenizer(AutoTokenizer, PreTrainedTokenizerFast)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            model.to(device)
            model.eval()
        except Exception as exc:  # noqa: BLE001 - download/load failure
            raise TranslationError(
                f"could not load translation model '{self.model_name}': {exc}. "
                "Check your network connection (the model is downloaded from "
                "Hugging Face on first use)."
            ) from exc

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self.device = device
        log.info("envit5 '%s' loaded on device=%s", self.model_name, device)

    # ------------------------------------------------------------------ helpers
    def _load_tokenizer(self, auto_cls, fast_cls):  # noqa: ANN001 - lazy classes
        """Load the tokenizer, tolerating the transformers>=5 T5 regression.

        transformers 5.x's unified ``T5Tokenizer`` forwards the model's ``vocab``
        dict straight into ``tokenizers.Unigram``, which requires a sequence and
        raises ``TypeError``. The repo ships ``tokenizer.json``, so on failure we
        fall back to the generic fast tokenizer, which loads it unchanged.
        """
        try:
            return auto_cls.from_pretrained(self.model_name)
        except Exception as exc:  # noqa: BLE001 - compatibility fallback
            log.warning(
                "AutoTokenizer failed for '%s' (%s); falling back to "
                "PreTrainedTokenizerFast (tokenizer.json)",
                self.model_name,
                exc,
            )
            return fast_cls.from_pretrained(self.model_name)

    @staticmethod
    def _strip_prefix(text: str, lang: str) -> str:
        """Strip a leading ``"<lang>: "`` tag and surrounding whitespace."""
        text = (text or "").strip()
        prefix = f"{lang}:"
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
        return text

    # ------------------------------------------------------------- translation
    def translate(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
        *,
        budgets: Sequence[int] | None = None,
    ) -> list[str]:
        """Translate ``texts`` from ``src_lang`` to ``tgt_lang``.

        An optional per-item character ``budgets`` sequence (parallel to
        ``texts``; ``None`` = today) steers decoding: each batch is generated
        with ``max_length`` = the largest per-item token estimate (see
        :func:`_budget_max_length`, floored at :data:`MIN_BUDGET_TOKENS` and
        capped at :data:`MAX_LENGTH`) plus a shorter-output bias
        (:data:`LENGTH_PENALTY_BUDGETED`). Without budgets the ``generate`` call
        is byte-identical to before (``max_length=MAX_LENGTH`` and no
        ``length_penalty``).
        """
        if not texts:
            return []

        budget_list = _check_budgets(texts, budgets)

        self._ensure_loaded()
        torch = self._torch
        tokenizer = self._tokenizer
        model = self._model
        assert torch is not None and tokenizer is not None and model is not None

        inputs = [f"{src_lang}: {text}" for text in texts]

        outputs: list[str] = []
        for start in range(0, len(inputs), BATCH_SIZE):
            batch = inputs[start : start + BATCH_SIZE]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=INPUT_MAX_LENGTH,
            ).to(self.device)
            generate_kwargs: dict[str, int | float] = {
                "max_length": MAX_LENGTH,
                "num_beams": NUM_BEAMS,
                "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,
            }
            if budget_list is not None:
                batch_budgets = budget_list[start : start + BATCH_SIZE]
                generate_kwargs["max_length"] = max(
                    _budget_max_length(budget) for budget in batch_budgets
                )
                generate_kwargs["length_penalty"] = LENGTH_PENALTY_BUDGETED
            with torch.no_grad():
                generated = model.generate(**encoded, **generate_kwargs)
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            outputs.extend(self._strip_prefix(text, tgt_lang) for text in decoded)

        return outputs
