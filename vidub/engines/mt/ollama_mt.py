"""Ollama-backed LLM machine translation (M4-T5, QUALITY-ROADMAP.md P1.5).

Runs the contextual **translate → reflect → improve** pattern against a local
`Ollama <https://ollama.com>`_ server. Ollama is spoken to with the stdlib only
(``urllib.request`` + ``json``) -- no ``ollama`` or ``requests`` package.

The LLM only ever *proposes*. Every candidate output -- the initial pass, the
improved rewrite and the one targeted retry -- is checked by the shared
deterministic validators in :mod:`vidub.engines.mt.validators` (language,
numbers, length), re-exported here for backward compatibility. A segment that
still fails after exactly one targeted retry falls back to the per-segment
envit5 result, so a bad LLM suggestion can never reach the dub. The counters
live in ``self.stats`` (``llm_used`` / ``fallback_used`` / ``validator_retries``).

The plain :meth:`OllamaTranslator.translate` interface runs the initial pass
only (no reflect/improve), so the batch and timing re-translation splices stay
fast; the translate stage calls :meth:`translate_contextual` for the main batch
(see :mod:`vidub.stages.translate`).
"""

from __future__ import annotations

import json
import logging
import socket
import urllib.error
import urllib.request
from collections.abc import Sequence

from vidub.engines.mt.base import BaseTranslator, TranslationError, _check_budgets
from vidub.engines.mt.validators import (
    LANG_RATIO_THRESHOLD,
    LENGTH_RATIO,
    _digits,
    _lang_ok,
    _length_ok,
    _numbers_ok,
    _validate,
)

__all__ = [
    "OllamaTranslator",
    "BATCH_SIZE",
    "LANG_RATIO_THRESHOLD",
    "LENGTH_RATIO",
    "_digits",
    "_lang_ok",
    "_length_ok",
    "_numbers_ok",
    "_validate",
]

log = logging.getLogger(__name__)

# Segments per LLM call (one initial / reflect / improve call per batch).
BATCH_SIZE = 8


# --------------------------------------------------------------- JSON parsing
def _json_object(raw: str) -> dict | None:
    """Parse the first ``{...}`` object in ``raw``; ``None`` when malformed."""
    text = (raw or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _parse_translations(raw: str, expected: set[int]) -> dict[int, str]:
    """Parse ``{"translations": [{"index": i, "text": "..."}]}``.

    Only entries whose ``index`` is in ``expected`` and whose ``text`` is a
    non-empty string are kept, so missing/malformed entries are detectable.
    """
    data = _json_object(raw)
    if not data:
        return {}
    items = data.get("translations")
    if not isinstance(items, list):
        return {}
    out: dict[int, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        text = item.get("text")
        if index in expected and isinstance(text, str) and text.strip():
            out[index] = text.strip()
    return out


def _parse_critiques(raw: str, expected: set[int]) -> dict[int, list[str]]:
    """Parse the reflect response into ``{index: issues}`` for major issues."""
    data = _json_object(raw)
    if not data:
        return {}
    items = data.get("critiques")
    if not isinstance(items, list):
        return {}
    major: dict[int, list[str]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if index not in expected:
            continue
        if str(item.get("severity", "")).strip().lower() != "major":
            continue
        issues = item.get("issues")
        if not isinstance(issues, list):
            issues = [issues] if issues else []
        major[index] = [str(issue) for issue in issues]
    return major


# --------------------------------------------------------------- prompt text
def _sys_translate(src_lang: str, tgt_lang: str) -> str:
    return (
        f"You are a professional {src_lang}->{tgt_lang} dubbing translator. "
        "Translate ONLY the segments marked TRANSLATE; the PREVIOUS/NEXT lines "
        "are neighbouring context and must never be translated. Rules: keep "
        "every number exactly as written; keep Latin brand and entity names; "
        "keep each translation close in length to its source (at most 2.0x the "
        "source word count); apply the terminology table exactly when a listed "
        "source term occurs. Reply with STRICT JSON only, no prose: "
        '{"translations": [{"index": <segment index>, "text": "<translation>"}]}'
    )


def _sys_reflect(src_lang: str, tgt_lang: str) -> str:
    return (
        f"You are a meticulous bilingual {src_lang}->{tgt_lang} translation "
        "reviewer. For each segment compare the DRAFT against the SOURCE and "
        "report issues under accuracy, terminology, naturalness and length. "
        'Use severity "major" only when the draft is wrong (mistranslation, '
        "changed meaning, dropped/added content, wrong terminology) or far too "
        'long; style nitpicks are "minor". Reply with STRICT JSON only, no '
        'prose: {"critiques": [{"index": <segment index>, "issues": ["..."], '
        '"severity": "major"|"minor"}]}'
    )


def _sys_improve(src_lang: str, tgt_lang: str) -> str:
    return (
        f"You are a professional {src_lang}->{tgt_lang} dubbing translator "
        "revising a draft. Rewrite ONLY the segments listed below, fixing every "
        "listed ISSUE; keep numbers and Latin brand/entity names, keep the "
        "translation close in length to the source (at most 2.0x the source "
        "word count), and never change the meaning. Reply with STRICT JSON "
        'only, no prose: {"translations": [{"index": <segment index>, '
        '"text": "<translation>"}]}'
    )


class OllamaTranslator(BaseTranslator):
    """Ollama-backed contextual translator (LLM proposes, validators decide)."""

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        host: str = "http://127.0.0.1:11434",
        fallback: BaseTranslator | None = None,
        timeout: float = 180.0,
    ) -> None:
        self.model = model
        self.host = (host or "").rstrip("/")
        self.timeout = timeout
        self.fallback = fallback
        # Terminology pairs (source -> target) handed to the translation prompt
        # when set; the stage's glossary pass still enforces them deterministically.
        self.glossary: dict[str, str] = {}
        self.stats: dict[str, int] = {
            "llm_used": 0,
            "fallback_used": 0,
            "validator_retries": 0,
        }
        self._fallback = fallback
        self._fallback_unavailable = False

    # ---------------------------------------------------------------- HTTP
    def _post(self, path: str, payload: dict) -> dict:
        """POST ``payload`` as JSON to ``path`` on the Ollama host.

        Connection failures raise :class:`TranslationError` with an actionable
        message (is Ollama running? start it with 'ollama serve') naming the host.
        """
        url = f"{self.host}{path}"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except (urllib.error.URLError, socket.timeout, OSError) as exc:
            raise TranslationError(
                f"could not reach Ollama at {self.host} ({exc}); is Ollama "
                "running? start it with 'ollama serve'"
            ) from exc
        try:
            result = json.loads(body)
        except json.JSONDecodeError as exc:
            raise TranslationError(
                f"Ollama returned invalid JSON from {path}: {exc}"
            ) from exc
        return result if isinstance(result, dict) else {}

    def _chat(self, system: str, user: str, json_mode: bool = False) -> str:
        """One non-streaming ``/api/chat`` call; returns the assistant content."""
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": 0.2},
        }
        if json_mode:
            payload["format"] = "json"
        result = self._post("/api/chat", payload)
        message = result.get("message")
        if not isinstance(message, dict):
            return ""
        return str(message.get("content", ""))

    # -------------------------------------------------------------- prompts
    def _glossary_lines(self) -> str:
        if not self.glossary:
            return ""
        pairs = "\n".join(
            f"  {source} => {target}" for source, target in self.glossary.items()
        )
        return f"Terminology table (use these exact renderings):\n{pairs}\n\n"

    def _context_lines(self, texts: list[str], batch_idx: list[int]) -> str:
        lines: list[str] = []
        for index in batch_idx:
            lines.append(f"Segment {index}:")
            if index - 1 >= 0:
                lines.append(f"  PREVIOUS (context only): {texts[index - 1]}")
            lines.append(f"  TRANSLATE THIS: {texts[index]}")
            if index + 1 < len(texts):
                lines.append(f"  NEXT (context only): {texts[index + 1]}")
        return "\n".join(lines)

    def _translate_user(
        self,
        texts: list[str],
        batch_idx: list[int],
        budgets: Sequence[int] | None = None,
    ) -> str:
        """User message for one batch; ``budgets`` adds per-segment limits.

        With ``budgets`` set the message states each segment's target character
        count (``At most N characters``); ``None`` (the default) reproduces
        today's message exactly.
        """
        return (
            f"{self._glossary_lines()}"
            f"Translate every segment marked TRANSLATE ({len(batch_idx)} of "
            "them). Return strict JSON with one entry per marked segment.\n\n"
            f"{self._budget_lines(budgets, batch_idx)}"
            f"{self._context_lines(texts, batch_idx)}"
        )

    def _budget_lines(
        self, budgets: Sequence[int] | None, batch_idx: list[int]
    ) -> str:
        """Per-segment character limits block (``""`` when no budgets)."""
        if budgets is None:
            return ""
        lines = [
            f"  Segment {index}: at most {int(budgets[index])} characters"
            for index in batch_idx
        ]
        return (
            "Character limit per segment (keep each translation at or below it):\n"
            + "\n".join(lines)
            + "\n\n"
        )

    # ----------------------------------------------------------- LLM passes
    def _batch_translations(
        self,
        texts: list[str],
        batch_idx: list[int],
        src_lang: str,
        tgt_lang: str,
        note: str | None = None,
        budgets: Sequence[int] | None = None,
    ) -> dict[int, str]:
        """One chat call returning ``{index: translation}`` for ``batch_idx``."""
        user = self._translate_user(texts, batch_idx, budgets)
        if note:
            user = f"{note}\n\n{user}"
        raw = self._chat(_sys_translate(src_lang, tgt_lang), user, json_mode=True)
        return _parse_translations(raw, set(batch_idx))

    def _initial_pass(
        self,
        texts: list[str],
        batch_idx: list[int],
        src_lang: str,
        tgt_lang: str,
        budgets: Sequence[int] | None = None,
    ) -> dict[int, str]:
        """Initial pass + one individual retry for each missing/malformed entry."""
        result = self._batch_translations(
            texts, batch_idx, src_lang, tgt_lang, budgets=budgets
        )
        for index in batch_idx:
            if index in result:
                continue
            single = self._batch_translations(
                texts, [index], src_lang, tgt_lang, budgets=budgets
            )
            result[index] = single.get(index, "")
        return result

    def _reflect(
        self,
        texts: list[str],
        drafts: dict[int, str],
        batch_idx: list[int],
        src_lang: str,
        tgt_lang: str,
    ) -> dict[int, list[str]]:
        """Reflect pass: return ``{index: issues}`` for major-issue segments only."""
        pairs = "\n".join(
            f"Segment {index}:\n"
            f"  SOURCE: {texts[index]}\n"
            f"  DRAFT: {drafts.get(index, '')}"
            for index in batch_idx
        )
        user = f"Review these segments for translation quality.\n\n{pairs}"
        raw = self._chat(_sys_reflect(src_lang, tgt_lang), user, json_mode=True)
        return _parse_critiques(raw, set(batch_idx))

    def _improve(
        self,
        texts: list[str],
        drafts: dict[int, str],
        major: dict[int, list[str]],
        src_lang: str,
        tgt_lang: str,
    ) -> dict[int, str]:
        """Improve pass: rewrite only the major-issue segments."""
        indexes = sorted(major)
        blocks: list[str] = []
        for index in indexes:
            issues = "; ".join(major.get(index, []))
            previous = texts[index - 1] if index - 1 >= 0 else ""
            following = texts[index + 1] if index + 1 < len(texts) else ""
            blocks.append(
                f"Segment {index}:\n"
                f"  PREVIOUS (context only): {previous}\n"
                f"  SOURCE: {texts[index]}\n"
                f"  DRAFT: {drafts.get(index, '')}\n"
                f"  NEXT (context only): {following}\n"
                f"  ISSUES: {issues}"
            )
        user = "Rewrite every listed segment, fixing its ISSUES.\n\n" + "\n".join(
            blocks
        )
        raw = self._chat(_sys_improve(src_lang, tgt_lang), user, json_mode=True)
        return _parse_translations(raw, set(indexes))

    # ----------------------------------------------------------- fallback
    def _resolve_fallback(self) -> BaseTranslator | None:
        """Return the envit5 fallback, constructing it lazily on first use."""
        if self._fallback is not None:
            return self._fallback
        if self._fallback_unavailable:
            return None
        try:
            from vidub.engines.mt.envit5_mt import Envit5Translator
        except Exception as exc:  # noqa: BLE001 - optional/heavy dependency
            log.warning("ollama: envit5 fallback unavailable (%s)", exc)
            self._fallback_unavailable = True
            return None
        self._fallback = Envit5Translator()
        return self._fallback

    def _fallback_one(
        self, src: str, src_lang: str, tgt_lang: str, draft: str
    ) -> str:
        """Delegate one failed segment to envit5, keeping the draft on failure."""
        fallback = self._resolve_fallback()
        if fallback is None:
            log.warning(
                "ollama: no envit5 fallback available; keeping the LLM draft for %r",
                src,
            )
            return draft
        try:
            result = fallback.translate([src], src_lang, tgt_lang)
        except Exception as exc:  # noqa: BLE001 - never crash on a fallback error
            log.warning("ollama: envit5 fallback failed (%s); keeping the draft", exc)
            return draft
        return str(result[0]) if result else draft

    # -------------------------------------------------------- orchestration
    def _accept(
        self,
        draft: str,
        src: str,
        texts: list[str],
        index: int,
        src_lang: str,
        tgt_lang: str,
        budgets: Sequence[int] | None = None,
    ) -> str:
        """Validate ``draft``; one targeted re-ask, then the envit5 fallback."""
        reason = _validate(draft, src, src_lang, tgt_lang)
        if reason is None:
            self.stats["llm_used"] += 1
            return draft

        # Exactly one targeted re-ask, with the failure reason in the prompt.
        self.stats["validator_retries"] += 1
        note = (
            f"The previous translation for segment {index} failed validation "
            f"({reason}). Produce a corrected translation for it."
        )
        retried = self._batch_translations(
            texts, [index], src_lang, tgt_lang, note=note, budgets=budgets
        )
        candidate = retried.get(index, "")
        if _validate(candidate, src, src_lang, tgt_lang) is None:
            self.stats["llm_used"] += 1
            return candidate

        self.stats["fallback_used"] += 1
        return self._fallback_one(src, src_lang, tgt_lang, draft)

    def _run(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
        contextual: bool,
        budgets: Sequence[int] | None = None,
    ) -> list[str]:
        if not texts:
            return []
        budget_list = _check_budgets(texts, budgets)
        outputs: list[str] = ["" for _ in texts]
        for start in range(0, len(texts), BATCH_SIZE):
            batch_idx = list(
                range(start, min(start + BATCH_SIZE, len(texts)))
            )
            drafts = self._initial_pass(
                texts, batch_idx, src_lang, tgt_lang, budgets=budget_list
            )
            if contextual:
                major = self._reflect(texts, drafts, batch_idx, src_lang, tgt_lang)
                if major:
                    improved = self._improve(
                        texts, drafts, major, src_lang, tgt_lang
                    )
                    for index, text in improved.items():
                        if text:
                            drafts[index] = text
            for index in batch_idx:
                outputs[index] = self._accept(
                    drafts.get(index, ""),
                    texts[index],
                    texts,
                    index,
                    src_lang,
                    tgt_lang,
                    budgets=budget_list,
                )
        return outputs

    # ------------------------------------------------------------- interface
    def translate(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
        *,
        budgets: Sequence[int] | None = None,
    ) -> list[str]:
        """Initial pass only (no reflect/improve) -- fast batch/splice path.

        ``budgets`` optionally carries one target character count per ``texts``
        entry; when set the user message states each segment's limit. ``None`` is
        byte-identical to today's prompt.
        """
        return self._run(texts, src_lang, tgt_lang, contextual=False, budgets=budgets)

    def translate_contextual(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
        *,
        budgets: Sequence[int] | None = None,
    ) -> list[str]:
        """Contextual translate → reflect → improve over batched segments.

        ``budgets`` optionally carries one target character count per ``texts``
        entry, threaded into every translate/retry call. ``None`` keeps the
        prompts exactly as before.
        """
        return self._run(texts, src_lang, tgt_lang, contextual=True, budgets=budgets)
