"""Hunyuan HY-MT1.5 translation specialist via a local Ollama server (M5-T1).

Optional, opt-in MT engine (QUALITY-ROADMAP.md §7) sitting alongside ``envit5``
(the default) and ``ollama`` (M4-T5). It runs Tencent's dedicated
``HY-MT1.5-7B`` translation model -- served by the same local `Ollama
<https://ollama.com>`_ server the ``ollama`` engine uses -- with the *official*
prompt templates from the model card. Ollama is spoken to with the stdlib only
(``urllib.request`` + ``json``); no ``ollama`` or ``requests`` package, so this
engine adds zero runtime dependencies.

Unlike the general-purpose ``ollama`` engine, the Hunyuan models are
special-purpose translation models: prompts are plain text (NO JSON mode, NO
system message). The model is asked one segment at a time with the official
template; a terminology table is injected when a glossary is set and the
contextual path adds the neighbouring segments.

LLM proposes, deterministic validators decide, envit5 is the fallback: every
candidate -- the first answer and the single temperature-0.3 re-ask -- is gated
by the shared checks in :mod:`vidub.engines.mt.validators` (language, numbers,
length); a segment that still fails falls back to the per-segment envit5 result,
so a bad suggestion can never reach the dub. The counters live in
``self.stats`` (``llm_used`` / ``fallback_used`` / ``validator_retries``).

Sampling parameters come from the official model card: ``top_k`` 20, ``top_p``
0.6, ``repetition_penalty`` 1.05 and ``temperature`` 0.7 (Ollama names the last
one ``repeat_penalty``).
"""

from __future__ import annotations

import json
import logging
import socket
import urllib.error
import urllib.request
from collections.abc import Sequence

from vidub.engines.mt.base import BaseTranslator, TranslationError, _check_budgets
from vidub.engines.mt.validators import _validate

__all__ = [
    "HunyuanMTTranslator",
    "DEFAULT_MODEL",
    "DEFAULT_HOST",
    "FULL_LANG_NAMES",
    "_lang_full",
    "_budget_clause",
    "_template_direct",
    "_template_contextual",
    "_template_terminology",
    "_template_fusion",
]

log = logging.getLogger(__name__)

# Official GGUF of Tencent's HY-MT1.5-7B translation specialist (Ollama model id).
DEFAULT_MODEL = "hf.co/tencent/HY-MT1.5-7B-GGUF:Q6_K"
# The same local Ollama server the ``ollama`` engine talks to.
DEFAULT_HOST = "http://127.0.0.1:11434"

# Target/source language codes -> names the official templates expect. Unknown
# codes fall back to the raw code so a new pair degrades gracefully.
FULL_LANG_NAMES: dict[str, str] = {
    "vi": "Vietnamese",
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
}


def _lang_full(code: str) -> str:
    """Full language name for a code (``"vi"`` -> ``"Vietnamese"``).

    Unknown codes are returned unchanged, so ``_template_*`` never needs to
    special-case a language the model card does not name explicitly.
    """
    key = (code or "").strip().lower()
    return FULL_LANG_NAMES.get(key, code)


# ------------------------------------------------------- official templates
def _budget_clause(budget_chars: int) -> str:
    """Concise length instruction appended to a template when a budget is set.

    This is the *proven* wording style from the timing stage's concise lever —
    "at most N characters, keep the meaning exactly, and output only the
    translation". It is only appended (with a leading space) when a caller passes
    a character budget, so the official templates stay byte-identical otherwise
    (the M5 contract). The budget is a hint: acceptance is decided by
    :func:`vidub.engines.mt.validators._validate` and the caller.
    """
    return (
        f" At most {int(budget_chars)} characters, keep the meaning exactly, "
        "and output only the translation."
    )


def _template_direct(tgt_full: str, text: str, budget_chars: int | None = None) -> str:
    """Official single-segment template (no context).

    With ``budget_chars`` set the concise clause is appended to the instruction
    sentence (before the source), so the model sees the length limit first.
    """
    instruction = (
        f"Translate the following segment into {tgt_full}, without additional "
        "explanation."
    )
    if budget_chars is not None:
        instruction += _budget_clause(budget_chars)
    return f"{instruction}\n\n{text}"


def _template_contextual(
    tgt_full: str,
    previous: str,
    text: str,
    following: str,
    budget_chars: int | None = None,
) -> str:
    """Official contextual template: neighbours first, then the segment.

    With no neighbours on either side this is exactly
    :func:`_template_direct` (the context sentence is omitted entirely). The
    budget clause, when set, is appended to the closing instruction sentence.
    """
    parts: list[str] = []
    if previous:
        parts.append(f"{previous}")
    if following:
        parts.append(f"{following}")
    if not parts:
        return _template_direct(tgt_full, text, budget_chars)
    context_block = "\n".join(parts)
    instruction = (
        f"{context_block}\nBased on the information above, translate the "
        f"following text into {tgt_full}. Note: do not translate the text "
        "above, and do not add any explanation"
    )
    if budget_chars is not None:
        instruction += _budget_clause(budget_chars)
    return f"{instruction}:\n{text}"


def _template_terminology(
    tgt_full: str,
    glossary: dict[str, str],
    text: str,
    budget_chars: int | None = None,
) -> str:
    """Official terminology template: the enforced pairs, then the segment."""
    pairs = "\n".join(
        f"{source} should be translated as {target}"
        for source, target in glossary.items()
    )
    instruction = (
        "Refer to the following translations:\n"
        f"{pairs}\n\n"
        f"Translate the following text into {tgt_full}. Only output the "
        "translation, without any additional explanation"
    )
    if budget_chars is not None:
        instruction += _budget_clause(budget_chars)
    return f"{instruction}:\n{text}"


def _template_fusion(
    src_full: str, tgt_full: str, source: str, candidates: list[str]
) -> str:
    """Official Chimera fusion template (M5-T2): fuse N candidate translations.

    Used by :meth:`HunyuanMTTranslator.fuse` with the Hunyuan-MT-Chimera-7B
    model; the numbered candidate list is 1-based and triple-backtick fenced,
    exactly as on the model card.
    """
    entries = "\n".join(
        f"{i}. ```{candidate}```"
        for i, candidate in enumerate(candidates, start=1)
    )
    return (
        f"Analyze the following multiple {tgt_full} translations of the "
        f"{src_full} segment surrounded in triple backticks and generate a "
        f"single refined {tgt_full} translation. Only output the refined "
        f"translation, do not explain.\n\nThe {src_full} segment:\n"
        f"```{source}```\n\nThe multiple {tgt_full} translations:\n{entries}"
    )


class HunyuanMTTranslator(BaseTranslator):
    """Hunyuan HY-MT1.5-7B translator (LLM proposes, validators decide)."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        fallback: BaseTranslator | None = None,
        timeout: float = 300.0,
    ) -> None:
        self.model = model
        self.host = (host or "").rstrip("/")
        self.timeout = timeout
        self.fallback = fallback
        # Terminology pairs (source -> target) injected into the official
        # terminology template when set; the stage's glossary pass still
        # enforces them deterministically.
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

    def _chat(self, user: str, temperature: float = 0.7) -> str:
        """One non-streaming ``/api/chat`` call; returns the assistant content.

        Hunyuan translation models take a plain-text user prompt only: there is
        no system message and no ``format: json`` (unlike the general-purpose
        ``ollama`` engine). Sampling uses the official model-card parameters.
        """
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [{"role": "user", "content": user}],
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_k": 20,
                "top_p": 0.6,
                "repeat_penalty": 1.05,
            },
        }
        result = self._post("/api/chat", payload)
        message = result.get("message")
        if not isinstance(message, dict):
            return ""
        return str(message.get("content", ""))

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
            log.warning("hunyuan: envit5 fallback unavailable (%s)", exc)
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
                "hunyuan: no envit5 fallback available; keeping the LLM draft "
                "for %r",
                src,
            )
            return draft
        try:
            result = fallback.translate([src], src_lang, tgt_lang)
        except Exception as exc:  # noqa: BLE001 - never crash on a fallback error
            log.warning(
                "hunyuan: envit5 fallback failed (%s); keeping the draft", exc
            )
            return draft
        return str(result[0]) if result else draft

    # -------------------------------------------------------- orchestration
    def _translate_segment(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        contextual: bool,
        previous: str = "",
        following: str = "",
        budget_chars: int | None = None,
    ) -> str:
        """Translate one segment; validate, re-ask once, then fall back.

        Template choice (contextual mode): with a glossary set the official
        terminology template is used and the neighbour context is *not* mixed
        in -- terminology and context are mutually exclusive per call, keeping
        the prompt identical to the model card. Without a glossary the official
        contextual template is used with the neighbours. Plain mode always uses
        the direct template.

        ``budget_chars`` is an optional target character count; when set the
        concise clause (:func:`_budget_clause`) is appended to the prompt, and
        the same prompt (with the reason) is reused for the one re-ask. ``None``
        keeps the official template byte-identical.
        """
        if not (text or "").strip():
            return ""
        tgt_full = _lang_full(tgt_lang)
        if contextual:
            if self.glossary:
                user = _template_terminology(
                    tgt_full, self.glossary, text, budget_chars
                )
            else:
                user = _template_contextual(
                    tgt_full, previous, text, following, budget_chars
                )
        else:
            user = _template_direct(tgt_full, text, budget_chars)

        candidate = self._chat(user).strip()
        reason = _validate(candidate, text, src_lang, tgt_lang)
        if reason is None:
            self.stats["llm_used"] += 1
            return candidate

        # Exactly one targeted re-ask, cooler temperature, with the reason.
        self.stats["validator_retries"] += 1
        retry_user = (
            f"The previous translation failed validation ({reason}). "
            f"Try again.\n\n{user}"
        )
        retried = self._chat(retry_user, temperature=0.3).strip()
        if _validate(retried, text, src_lang, tgt_lang) is None:
            self.stats["llm_used"] += 1
            return retried

        self.stats["fallback_used"] += 1
        return self._fallback_one(text, src_lang, tgt_lang, retried)

    # ---------------------------------------------------------- refine (M5-T2)
    def sample_candidates(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        k: int,
        *,
        budget_chars: int | None = None,
    ) -> list[str]:
        """Return up to ``k`` distinct non-empty translations of ``text``.

        Each candidate is an independent ``_chat`` call at the official
        temperature 0.7 with the direct template. Blanks and duplicates are
        dropped, preserving first-seen order, so two identical samples collapse
        to one candidate. ``budget_chars`` is an optional target character count
        appended to the direct template (``None`` = the official template).
        """
        if k <= 0 or not (text or "").strip():
            return []
        user = _template_direct(_lang_full(tgt_lang), text, budget_chars)
        out: list[str] = []
        seen: set[str] = set()
        for _ in range(k):
            candidate = self._chat(user).strip()
            if candidate and candidate not in seen:
                seen.add(candidate)
                out.append(candidate)
        return out

    def fuse(
        self, source: str, candidates: list[str], src_lang: str, tgt_lang: str
    ) -> str:
        """Fuse ``candidates`` into one refined translation (Chimera prompt).

        Returns the stripped response. No validation happens here: the caller
        (``vidub.utils.refine``) is responsible for gating the result.
        """
        user = _template_fusion(
            _lang_full(src_lang), _lang_full(tgt_lang), source, candidates
        )
        return self._chat(user).strip()

    # ------------------------------------------------------------- interface
    def translate(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
        *,
        budgets: Sequence[int] | None = None,
    ) -> list[str]:
        """Translate each segment independently with the direct template.

        ``budgets`` optionally carries one target character count per ``texts``
        entry; a set budget appends the concise clause to that segment's direct
        template. ``None`` is byte-identical to the official template.
        """
        budget_list = _check_budgets(texts, budgets)
        return [
            self._translate_segment(
                text,
                src_lang,
                tgt_lang,
                contextual=False,
                budget_chars=None if budget_list is None else budget_list[index],
            )
            for index, text in enumerate(texts)
        ]

    def translate_contextual(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
        *,
        budgets: Sequence[int] | None = None,
    ) -> list[str]:
        """Translate each segment with its neighbours (and glossary) as context.

        ``budgets`` optionally carries one target character count per ``texts``
        entry (the contextual/terminology template gets the concise clause for
        that segment). ``None`` keeps the official prompts exactly.
        """
        budget_list = _check_budgets(texts, budgets)
        outputs: list[str] = []
        for index, text in enumerate(texts):
            previous = texts[index - 1] if index - 1 >= 0 else ""
            following = texts[index + 1] if index + 1 < len(texts) else ""
            outputs.append(
                self._translate_segment(
                    text,
                    src_lang,
                    tgt_lang,
                    contextual=True,
                    previous=previous,
                    following=following,
                    budget_chars=None if budget_list is None else budget_list[index],
                )
            )
        return outputs
