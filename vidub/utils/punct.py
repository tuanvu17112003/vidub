"""Opt-in punctuation/casing restoration for the SOURCE transcript (P1.6).

ASR output often lacks punctuation and casing, which hurts both the machine
translation input and sentence-aware subtitle splitting. This module restores
periods/commas/question marks and basic casing with a small token-classification
model:

    oliverguhr/fullstop-punctuation-multilang-large

The model (~2.2 GB) downloads on first use from Hugging Face into the existing
cache; ``transformers``/``torch`` are already installed, so there is no new
dependency. It is **opt-in** (``--restore-punct`` / ``JobConfig.restore_punct``):
when disabled the pipeline is byte-identical to before.

Design notes:

* :data:`PUNCT_DENSITY_MIN` guards against re-writing text that already looks
  punctuated (terminal-punctuation density ``. ! ?`` per word >= 0.1 is left
  untouched).
* The model is loaded lazily (imports and weights only on first :meth:`restore`),
  so importing this module is cheap and stdlib-only.
* Predictions run over word-subword windows (max 128 wordpieces, stride overlap)
  and are merged deterministically — the first window to predict a word wins.
* The label map is read from the model config at load time (the upstream model
  maps labels ``0``/``.``/``,``/``?``/``-``/``:``); :func:`_label_to_mark`
  normalises the common spellings without hardcoding one model's ids.
"""

from __future__ import annotations

import logging

from vidub.manifest import Segment

__all__ = [
    "PUNCT_DENSITY_MIN",
    "MODEL_NAME",
    "WINDOW_WORDPIECES",
    "WINDOW_OVERLAP",
    "PunctRestorer",
    "restore_segments",
]

log = logging.getLogger(__name__)

MODEL_NAME = "oliverguhr/fullstop-punctuation-multilang-large"
# A segment this punctuated (terminal marks per word) is left untouched.
PUNCT_DENSITY_MIN = 0.1
# Model forward passes are capped at this many wordpieces, with this stride
# overlap between consecutive windows (both configurable for tests).
WINDOW_WORDPIECES = 128
WINDOW_OVERLAP = 16
_TERMINAL = (".", "!", "?")
_SENTENCE_END = _TERMINAL


def _punct_density(text: str) -> float:
    """Terminal-punctuation marks per word (``0.0`` for text with no words)."""
    words = (text or "").split()
    if not words:
        return 0.0
    marks = sum(1 for char in text if char in _TERMINAL)
    return marks / len(words)


def _capitalize_first(word: str) -> str:
    """Uppercase the first alphabetic character of ``word`` (case-preserving)."""
    for index, char in enumerate(word):
        if char.isalpha():
            return word[:index] + char.upper() + word[index + 1 :]
    return word


def _label_to_mark(label: str) -> str:
    """Map a model label to the punctuation mark to append (``""`` for none).

    Handles the upstream ids (``"0"`` = no punctuation, ``"."``/``","``/``"?"``/
    ``"-"``/``":"``) as well as the common named spellings, without assuming one
    model's exact id set.
    """
    raw = (label or "").strip()
    if not raw or raw in {"0", "O", "_", "none"}:
        return ""
    if raw in {".", ",", "?", "!", ":", ";", "-", "…"}:
        return raw
    if len(raw) > 2 and raw[1] == "-" and raw[0].upper() in {"B", "I", "S", "E"}:
        # BIO/Style tag: strip the prefix and retry (e.g. "B-PERIOD").
        return _label_to_mark(raw[2:])
    low = raw.lower()
    if low in {"period", "fullstop", "full stop", "dot", "point"}:
        return "."
    if low in {"comma"}:
        return ","
    if low in {"question", "questionmark", "question mark", "qmark"}:
        return "?"
    if low in {
        "no_punct",
        "no-punct",
        "no punctuation",
        "nopunct",
        "empty",
        "none_punct",
    }:
        return ""
    for char in (".", ",", "?", "!", ":", ";"):
        if raw.endswith(char):
            return char
    return ""


def _windows(
    n_tokens: int, size: int = WINDOW_WORDPIECES, overlap: int = WINDOW_OVERLAP
) -> list[tuple[int, int]]:
    """Half-open token windows ``(start, end)`` covering ``[0, n_tokens)``.

    Consecutive windows overlap by ``overlap`` tokens; ``overlap < size``
    guarantees forward progress. Returns ``[]`` for a non-positive token count.
    """
    if n_tokens <= 0:
        return []
    if overlap < 0:
        overlap = 0
    if overlap >= size:
        overlap = size - 1
    windows: list[tuple[int, int]] = []
    start = 0
    while start < n_tokens:
        end = min(start + size, n_tokens)
        windows.append((start, end))
        if end >= n_tokens:
            break
        start = end - overlap
    return windows


class PunctRestorer:
    """Lazily-loaded punctuation/casing restorer around a token-classification model."""

    def __init__(self, model_name: str = MODEL_NAME, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._labels: dict[int, str] = {}

    # ------------------------------------------------------------- lazy load
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import (
                AutoModelForTokenClassification,
                AutoTokenizer,
            )
        except ImportError as exc:  # pragma: no cover - transformers is installed
            raise RuntimeError(
                "punctuation restoration needs transformers/torch, which are not "
                "installed; drop --restore-punct to run without it"
            ) from exc

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForTokenClassification.from_pretrained(self.model_name)
            model.to(self.device)
            model.eval()
        except Exception as exc:  # noqa: BLE001 - download/load failure
            raise RuntimeError(
                f"could not load punctuation-restoration model "
                f"'{self.model_name}': {exc}. It downloads from Hugging Face on "
                "first use (~2.2 GB); check the network, or drop --restore-punct "
                "to run without punctuation restoration."
            ) from exc

        id2label = getattr(getattr(model, "config", None), "id2label", None) or {}
        self._labels = {int(key): str(value) for key, value in id2label.items()}
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        log.info("punctuation model '%s' loaded on device=%s", self.model_name, self.device)

    # ----------------------------------------------------------- inference
    def _tokenize_words(self, words: list[str]) -> tuple[list[int], list[int], list[int | None]]:
        """Tokenize ``words`` into wordpieces, keeping each piece's word index."""
        encoding = self._tokenizer(
            words, is_split_into_words=True, add_special_tokens=False
        )
        input_ids = [int(token) for token in encoding["input_ids"]]
        attention_mask = [int(mask) for mask in encoding["attention_mask"]]
        word_ids = list(encoding.word_ids(0))
        return input_ids, attention_mask, word_ids

    def _predict_window(self, input_ids: list[int], attention_mask: list[int]) -> list[int]:
        """Predicted label id per token for one window (overridable in tests)."""
        torch = self._torch
        assert torch is not None
        with torch.no_grad():
            logits = self._model(
                input_ids=torch.tensor([list(input_ids)]),
                attention_mask=torch.tensor([list(attention_mask)]),
            ).logits
        return [int(value) for value in torch.argmax(logits, dim=-1)[0].tolist()]

    def _predict_marks(self, words: list[str]) -> list[str]:
        """One punctuation mark (``""`` for none) per word.

        Token predictions are merged across overlapping windows; the first
        window to predict a word wins, so the result is deterministic.
        """
        if not words:
            return []
        self._ensure_loaded()
        input_ids, attention_mask, word_ids = self._tokenize_words(words)
        marks = ["" for _ in words]
        assigned = [False for _ in words]
        for start, end in _windows(len(input_ids)):
            label_ids = self._predict_window(
                input_ids[start:end], attention_mask[start:end]
            )
            for label_id, word_index in zip(label_ids, word_ids[start:end]):
                if word_index is None or assigned[word_index]:
                    continue
                assigned[word_index] = True
                marks[word_index] = _label_to_mark(
                    self._labels.get(int(label_id), "")
                )
        return marks

    # -------------------------------------------------------------- public
    def restore(self, text: str) -> tuple[str, int]:
        """Return ``(restored_text, inserted_mark_count)`` for ``text``.

        Predicted marks are appended to their words; the first word and the
        first word after a sentence-ending mark are capitalised.
        """
        words = (text or "").split()
        if not words:
            return "", 0
        marks = self._predict_marks(words)
        output: list[str] = []
        inserted = 0
        capitalize = True
        for word, mark in zip(words, marks):
            if capitalize:
                word = _capitalize_first(word)
            capitalize = mark in _SENTENCE_END
            if mark:
                output.append(word + mark)
                inserted += 1
            else:
                output.append(word)
        return " ".join(output), inserted


def restore_segments(segments: list[Segment], enabled: bool = True) -> int:
    """Restore punctuation/casing on ``segments`` in place; return marks added.

    Segments whose terminal-punctuation density already reaches
    :data:`PUNCT_DENSITY_MIN` are skipped (never re-written). One shared
    :class:`PunctRestorer` is reused across the whole call, and it is only
    constructed when at least one segment actually needs restoring.
    """
    if not enabled or not segments:
        return 0

    restorer: PunctRestorer | None = None
    total = 0
    for segment in segments:
        text = segment.text or ""
        if not text.strip():
            continue
        if _punct_density(text) >= PUNCT_DENSITY_MIN:
            continue
        if restorer is None:
            restorer = PunctRestorer()
        restored, inserted = restorer.restore(text)
        if restored and restored != text:
            segment.text = restored
        total += inserted
    return total
