"""Level-2 translation QC battery (QUALITY-ROADMAP.md §3.3, MQM-aligned).

Deterministic, stdlib-only scoring of every translated ``script.json`` row on
independent signals (no single score decides):

* ``number-error``  — a source number whose digits are missing from the vi text;
* ``entity-error``  — a capitalized source entity missing from the vi text;
* ``terminology-error`` — a glossed source term whose enforced target is absent;
* ``omission``      — the vi text is empty or far shorter than the source;
* ``addition``      — the vi text is far longer than the source (not truncated);
* ``inconsistent``  — a repeated source phrase rendered divergently across rows;
* ``mistranslation-risk`` — auxiliary en->vi->en round-trip token-F1 is low;
* ``unnatural``     — low confidence or leaked English grammar in Vietnamese.

Corrections only ever happen through existing helpers: a deterministic glossary
splice (:func:`vidub.utils.glossary.splice_source_terms`, "an entity/number is
forced in; the surroundings are re-translated") or a single re-translation
retry through the MT engine. Passthrough rows (already user-language) are never
touched. The per-job verdict is ``pass`` / ``review`` / ``fail``.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from vidub.utils.glossary import (
    apply_glossary,
    apply_source_aliases,
    find_source_terms,
    splice_source_terms,
)
from vidub.utils.textclean import clean_hesitation
from vidub.utils.vnstyle import apply_vnstyle
from vidub.utils.english_tokens import entity_form, grammar_leaks, is_contraction
from vidub.utils.units_guard import guard_signals

__all__ = [
    "LENGTH_RATIO_MIN",
    "NUMBER_ERROR_DIGITS_ONLY",
    "ROUNDTRIP_F1_MIN",
    "CONSISTENCY_JACCARD_MIN",
    "REPEAT_PHRASE_WORDS",
    "ENTITY_MIN_LEN",
    "ENTITY_STOPLIST",
    "LABELS",
    "extract_numbers",
    "digits_of",
    "vi_number_form",
    "extract_entities",
    "token_f1",
    "score_rows",
    "correct_rows",
    "run_battery",
]

log = logging.getLogger(__name__)

# ------------------------------------------------------------------ constants
LENGTH_RATIO_MIN = 0.4
NUMBER_ERROR_DIGITS_ONLY = True
ROUNDTRIP_F1_MIN = 0.5
CONSISTENCY_JACCARD_MIN = 0.3
REPEAT_PHRASE_WORDS = 3
ENTITY_MIN_LEN = 2
# The existing truncation bound (vidub.stages.translate.LENGTH_RATIO): a vi text
# beyond it that was *not* truncated is an "addition".
ADDITION_RATIO_MAX = 2.2

ENTITY_STOPLIST = frozenset(
    {
        "i", "the", "a", "an", "but", "and", "or", "so", "if", "it", "in", "on",
        "at", "to", "for", "with", "my", "we", "he", "she", "they", "this",
        "that", "these", "those", "what", "how", "why", "when", "where", "who",
        "yes", "no", "now", "then", "ok", "hey", "well", "like", "just",
        "really", "actually", "maybe", "let", "lets", "get", "got", "go",
        "going", "one", "two", "new", "not", "do", "does", "did", "is", "are",
        "was", "were", "be", "been", "being", "have", "has", "had", "will",
        "would", "can", "could", "should", "may", "might", "must", "there",
        "here",
    }
)

# All MQM-aligned labels, worst-first.
LABELS = (
    "number-error",
    "entity-error",
    "terminology-error",
    "omission",
    "addition",
    "inconsistent",
    "mistranslation-risk",
    "unnatural",
    "ok",
)
_LABEL_SEVERITY = {label: index for index, label in enumerate(reversed(LABELS))}

NUMBER_RE = re.compile(
    r"(?<![\w])[$€£]?(?:\d{1,3}(?:[.,]\d{3})+|\d+)(?:[.,]\d+)?%?"
)
_WORD_RE = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*", re.UNICODE)
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_SENTENCE_END_RE = re.compile(r"[.!?]")


# ------------------------------------------------------------------ extractors
def extract_numbers(text: str) -> list[str]:
    """Numeric tokens in ``text``: ``30,000`` / ``2.5`` / ``1999`` / ``50%`` / ``$30,000``."""
    return NUMBER_RE.findall(text or "")


def digits_of(token: str) -> str:
    """All digits of ``token`` (``"30,000"`` -> ``"30000"``, ``"$50%"`` -> ``"50"``)."""
    return re.sub(r"\D", "", token or "")


def vi_number_form(token: str) -> str:
    """En->vi separator convention: ``"30,000"`` -> ``"30.000"``, ``"2.5"`` -> ``"2,5"``."""
    token = token or ""
    prefix = token[:1] if token[:1] in "$€£" else ""
    core = token[len(prefix):]
    suffix = "%" if core.endswith("%") else ""
    core = core[: -len(suffix)] if suffix else core
    if "," in core and "." in core:
        core = core.replace(".", "\x00").replace(",", ".").replace("\x00", ",")
    elif "," in core:
        core = core.replace(",", ".")
    elif "." in core:
        core = core.replace(".", ",")
    return f"{prefix}{core}{suffix}"


def extract_entities(text: str) -> list[str]:
    """Capitalized alphabetic tokens (len >= 2) that are not sentence-initial.

    A token is an entity only when it is (a) capitalized, (b) at least
    :data:`ENTITY_MIN_LEN` characters, (c) not a stoplist word, and (d) **not
    sentence-initial**. Sentence-initial means the text start or immediately
    preceded by ``. ! ?`` (possibly followed by whitespace/quotes) — ordinary
    capitalization there is not an entity signal. This is deliberately
    conservative: a mid-sentence mention (``"we use Vigib daily"``) is flagged
    when dropped, while a sentence-initial ``"Hello"``/``"Vigib"`` is not.
    """
    entities: list[str] = []
    previous_end: int | None = None
    for match in _WORD_RE.finditer(text or ""):
        token = match.group(0)
        start = match.start()
        previous_start = start if previous_end is None else previous_end
        between = text[:start] if previous_end is None else text[previous_end:start]
        previous_end = match.end()
        if is_contraction(token):
            continue
        token = entity_form(token)
        if not token[:1].isupper():
            continue
        if len(token) < ENTITY_MIN_LEN:
            continue
        if token.lower() in ENTITY_STOPLIST:
            continue
        # Sentence-initial: the very start of the text, or a preceding `. ! ?`.
        # A plain space is NOT a sentence boundary (otherwise every word would
        # be treated as sentence-initial).
        if previous_start == 0 or _SENTENCE_END_RE.search(between):
            continue
        entities.append(token)
    return entities


# ------------------------------------------------------------------ similarity
def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def token_f1(a: str, b: str) -> float:
    """Word-level token-set F1 (lowercased); ``1.0`` for two empty strings."""
    set_a = set(_tokens(a))
    set_b = set(_tokens(b))
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    common = len(set_a & set_b)
    if common == 0:
        return 0.0
    precision = common / len(set_b)
    recall = common / len(set_a)
    return 2 * precision * recall / (precision + recall)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


# ----------------------------------------------------------------- per-row score
def _label_for(signals: dict[str, list[str]]) -> str:
    """Worst present class in precedence order (``ok`` when none)."""
    for label in LABELS:
        if label in signals:
            return label
    return "ok"


def _score_row(
    index: int,
    row: dict,
    glossary: dict[str, str],
    src_lang: str,
    tgt_lang: str,
) -> dict:
    """Per-row signals + label (no cross-segment/round-trip pass)."""
    entry: dict[str, Any] = {
        "index": index,
        "label": "ok",
        "signals": {},
        "tokens": {"numbers": [], "entities": [], "terms": []},
    }
    if row.get("passthrough"):
        # Already user-language: skipped by the battery, recorded as ok.
        entry["signals"]["passthrough"] = ["passthrough"]
        entry["passthrough"] = True
        return entry

    source = str(row.get("en", ""))
    vi = str(row.get("vi", ""))
    vi_lower = vi.lower()
    signals: dict[str, list[str]] = {}

    # --- numbers: every source number's digits must appear in the vi text -----
    vi_digits = digits_of(vi)
    missing_numbers = [
        token
        for token in extract_numbers(source)
        if digits_of(token) and digits_of(token) not in vi_digits
    ]
    if missing_numbers:
        signals["number-error"] = [f"missing number(s): {', '.join(missing_numbers)}"]
        entry["tokens"]["numbers"] = missing_numbers

    # --- entities: capitalized source tokens must survive ---------------------
    glossary_keys = {term.lower() for term in glossary}
    # Tokens that are part of a matched glossed term are handled by the
    # terminology signal instead (avoid double-flagging e.g. the "AI" inside
    # "AI assistant").
    term_tokens: set[str] = set()
    if glossary and source:
        for term, _target, _start, _end in find_source_terms(
            source, glossary, max_terms=max(len(glossary), 1)
        ):
            term_tokens.update(token.lower() for token in _tokens(term))
    missing_entities = [
        entity
        for entity in extract_entities(source)
        if entity.lower() not in glossary_keys
        and entity.lower() not in term_tokens
        and entity.lower() not in vi_lower
    ]
    if missing_entities:
        signals["entity-error"] = [f"missing entity: {', '.join(missing_entities)}"]
        entry["tokens"]["entities"] = missing_entities

    # --- terminology: glossed source terms must render as their target --------
    missing_terms: list[tuple[str, str]] = []
    if glossary and source:
        hits = find_source_terms(source, glossary, max_terms=max(len(glossary), 1))
        for term, target, _start, _end in hits:
            if target and target.lower() not in vi_lower:
                missing_terms.append((term, target))
    if missing_terms:
        signals["terminology-error"] = [
            f"missing term {term!r} -> {target!r}" for term, target in missing_terms
        ]
        entry["tokens"]["terms"] = missing_terms

    # --- omission / addition: length ratio ------------------------------------
    src_words = len(source.split())
    vi_words = len(vi.split())
    ratio = vi_words / max(src_words, 1)
    if not vi.strip():
        signals["omission"] = ["empty translation"]
    elif ratio < LENGTH_RATIO_MIN:
        signals["omission"] = [f"length_ratio={ratio:.2f}<{LENGTH_RATIO_MIN}"]
    if ratio > ADDITION_RATIO_MAX and not row.get("truncated"):
        signals["addition"] = [f"length_ratio={ratio:.2f}>{ADDITION_RATIO_MAX}"]

    # --- fluency: fold the existing low-confidence flag -----------------------
    if row.get("low_confidence"):
        signals["unnatural"] = ["low_confidence"]
    if src_lang == "en" and tgt_lang == "vi":
        leaks = grammar_leaks(vi, glossary.values())
        if leaks:
            signals.setdefault("unnatural", []).append(
                f"untranslated English grammar: {', '.join(leaks)}"
            )

    entry["signals"] = signals
    entry["label"] = _label_for(signals)
    return entry


def _apply_consistency(
    rows: list[dict], results: list[dict]
) -> None:
    """Flag the later row of divergent renderings of a repeated source phrase."""
    if len(rows) < 2:
        return
    vi_tokens = [set(_tokens(str(row.get("vi", "")))) for row in rows]
    occurrences: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        words = _tokens(str(row.get("en", "")))
        seen: set[str] = set()
        for i in range(len(words) - REPEAT_PHRASE_WORDS + 1):
            phrase = " ".join(words[i : i + REPEAT_PHRASE_WORDS])
            if phrase in seen:
                continue
            seen.add(phrase)
            occurrences[phrase].append(index)

    flagged: set[int] = set()
    for indices in occurrences.values():
        if len(indices) < 2:
            continue
        for first_pos in range(len(indices) - 1):
            for second_pos in range(first_pos + 1, len(indices)):
                earlier, later = indices[first_pos], indices[second_pos]
                if _jaccard(vi_tokens[earlier], vi_tokens[later]) < CONSISTENCY_JACCARD_MIN:
                    flagged.add(later)

    for index in flagged:
        results[index]["signals"].setdefault(
            "inconsistent",
            ["divergent rendering of a repeated source phrase"],
        )


def score_rows(
    rows: list[dict],
    glossary: dict[str, str],
    engine: Any = None,
    src_lang: str = "en",
    tgt_lang: str = "vi",
) -> tuple[list[dict], dict]:
    """Score every non-passthrough row; return ``(per_row_results, summary)``.

    ``summary`` is ``{"counts": {label: n}, "roundtrip": <status>}``. The
    round-trip is auxiliary (never a fail signal) and runs as ONE batch
    back-translation when ``engine`` is set and ``src_lang == "en"``.
    """
    results = [_score_row(i, row, glossary, src_lang, tgt_lang) for i, row in enumerate(rows)]

    roundtrip = "disabled"
    if engine is not None:
        if (src_lang or "").strip().lower() == "en":
            roundtrip = "ok"
            scored_indices = [
                i for i, row in enumerate(rows) if not row.get("passthrough")
            ]
            vi_texts = [str(rows[i].get("vi", "")) for i in scored_indices]
            back = engine.translate(vi_texts, tgt_lang, src_lang) if vi_texts else []
            for position, index in enumerate(scored_indices):
                if position >= len(back):
                    break
                f1 = token_f1(str(rows[index].get("en", "")), str(back[position]))
                results[index]["roundtrip_f1"] = round(f1, 3)
                if f1 < ROUNDTRIP_F1_MIN:
                    results[index]["signals"]["mistranslation-risk"] = [
                        f"roundtrip_f1={f1:.2f}<{ROUNDTRIP_F1_MIN}"
                    ]
        else:
            roundtrip = "skipped (engine covers en<->vi only)"

    _apply_consistency(rows, results)

    for result in results:
        result["label"] = _label_for(result["signals"])

    counts: dict[str, int] = {}
    for result in results:
        counts[result["label"]] = counts.get(result["label"], 0) + 1
    return results, {"counts": counts, "roundtrip": roundtrip}


# ------------------------------------------------------------------ corrections
def _translate_one(
    engine: Any, src_lang: str, tgt_lang: str
) -> Callable[[str], str]:
    """A one-chunk ``translate_fn`` bound to ``engine`` for the splice helper."""

    def _fn(chunk: str) -> str:
        result = engine.translate([chunk], src_lang, tgt_lang)
        return str(result[0]) if result else ""

    return _fn


def _splice(
    row: dict, mapping: dict[str, str], engine: Any, src_lang: str, tgt_lang: str
) -> str:
    """Re-run the source-side glossary splice with a pseudo-glossary."""
    source = str(row.get("en", ""))
    text = str(row.get("vi", ""))
    new_text, _rebuilt, _enforced = splice_source_terms(
        source, text, mapping, _translate_one(engine, src_lang, tgt_lang), max_terms=3
    )
    return new_text


def _is_better(new_label: str, old_label: str) -> bool:
    """True when ``new_label`` is less severe than ``old_label``."""
    return _LABEL_SEVERITY.get(new_label, 0) < _LABEL_SEVERITY.get(old_label, 0)


def correct_rows(
    rows: list[dict],
    labels: list[dict],
    glossary: dict[str, str],
    engine: Any = None,
    src_lang: str = "en",
    tgt_lang: str = "vi",
) -> tuple[list[dict], list[dict]]:
    """ONE corrective pass over flagged rows; return ``(rows, corrections)``.

    Corrections only ever replace text through the existing helpers — a
    deterministic splice (:func:`splice_source_terms`) or a single
    re-translation. Passthrough rows and ``ok`` rows are never modified.
    """
    corrections: list[dict] = []
    labels_by_index = {entry["index"]: entry for entry in labels}

    for index, row in enumerate(rows):
        entry = labels_by_index.get(index, {"label": "ok", "signals": {}, "tokens": {}})
        label = entry.get("label", "ok")
        if row.get("passthrough") or label == "ok":
            continue

        tokens = entry.get("tokens", {})
        new_row = dict(row)
        action = "none"
        retried = False

        if label == "entity-error":
            entities = list(tokens.get("entities", []))[:3]
            if engine is not None and entities:
                pseudo = {entity: entity for entity in entities}
                new_row["vi"] = _splice(new_row, pseudo, engine, src_lang, tgt_lang)
                action = "splice-entity"
            else:
                action = "review-entity"
        elif label == "number-error":
            missing = list(tokens.get("numbers", []))
            vi_has_digits = bool(re.search(r"\d", str(row.get("vi", ""))))
            if engine is not None and missing and not vi_has_digits:
                pseudo = {token: vi_number_form(token) for token in missing[:3]}
                new_row["vi"] = _splice(new_row, pseudo, engine, src_lang, tgt_lang)
                action = "splice-number"
            else:
                # The translator wrote the number differently -> human review.
                action = "review-number"
        elif label == "terminology-error":
            if glossary:
                vi_text = str(new_row.get("vi", ""))
                vi_text, _applied = apply_glossary(vi_text, glossary)
                if engine is not None:
                    vi_text, _rebuilt, _enforced = splice_source_terms(
                        str(new_row.get("en", "")),
                        vi_text,
                        glossary,
                        _translate_one(engine, src_lang, tgt_lang),
                        max_terms=2,
                    )
                new_row["vi"] = vi_text
                action = "enforce-glossary"
        else:
            if engine is not None:
                retried = True
                outputs = engine.translate(
                    [str(new_row.get("en", ""))], src_lang, tgt_lang
                )
                if outputs:
                    candidate = dict(new_row)
                    # Mirror the stage's final passes so a retry can never
                    # reintroduce hesitation/repetition artifacts or unsorted
                    # Vietnamese word order.
                    cleaned, _fixed = clean_hesitation(str(outputs[0]))
                    cleaned, _styled = apply_vnstyle(cleaned)
                    candidate["vi"] = apply_source_aliases(str(candidate.get("en", "")), cleaned, glossary)
                    before = _score_row(index, new_row, glossary, src_lang, tgt_lang)["label"]
                    after = _score_row(index, candidate, glossary, src_lang, tgt_lang)["label"]
                    # Keep the retry only when it is STRICTLY better; otherwise
                    # the already-processed original text is preserved.
                    if _is_better(after, before):
                        new_row = candidate
                    action = "retranslate"

        # A repair must not introduce another deterministic defect or drop
        # content merely to improve the worst label. Keep the original on ties.
        before_score = _score_row(index, row, glossary, src_lang, tgt_lang)
        new_row["vi"] = apply_source_aliases(str(new_row.get("en", "")), str(new_row.get("vi", "")), glossary)
        final = _score_row(index, new_row, glossary, src_lang, tgt_lang)
        critical = {"number-error", "entity-error", "terminology-error", "omission", "addition"}
        new_defects = (set(final["signals"]) - set(before_score["signals"])) & critical
        source = str(row.get("en", ""))
        old_text, new_text = str(row.get("vi", "")), str(new_row.get("vi", ""))
        new_guard = set(guard_signals(source, new_text)["flags"]) - set(
            guard_signals(source, old_text)["flags"]
        )
        new_leaks = src_lang == "en" and tgt_lang == "vi" and bool(
            set(grammar_leaks(new_text, glossary.values()))
            - set(grammar_leaks(old_text, glossary.values()))
        )
        if new_row != row and (new_defects or new_guard or new_leaks):
            new_row = dict(row)
            final = before_score
            action = f"{action}-rejected"
        # Preserve the auxiliary round-trip signal computed by score_rows: the
        # per-row re-score cannot recompute it (it needs the batch engine call),
        # and it must never be silently dropped by a correction.
        original_signals = entry.get("signals", {})
        if "mistranslation-risk" in original_signals and label in (
            "mistranslation-risk",
            "unnatural",
            "inconsistent",
        ):
            final["signals"].setdefault(
                "mistranslation-risk", original_signals["mistranslation-risk"]
            )
            final["label"] = _label_for(final["signals"])
        if "inconsistent" in original_signals:
            final["signals"].setdefault("inconsistent", original_signals["inconsistent"])
        corrections.append(
            {
                "index": index,
                "action": action,
                "retried": retried,
                "resolved": final["label"] == "ok",
                "label": final["label"],
                "signals": final["signals"],
            }
        )
        rows[index] = new_row

    return rows, corrections


# ------------------------------------------------------------------- battery
def _write_packets(
    out_dir: str | Path,
    rows: list[dict],
    final_entries: list[dict],
    corrections: dict[int, dict],
) -> None:
    """Write ``review/translation/seg-<index>.txt`` for each unresolved flag."""
    packet_dir = Path(out_dir) / "review" / "translation"
    packet_dir.mkdir(parents=True, exist_ok=True)
    active = {f"seg-{entry['index']}.txt" for entry in final_entries if entry["label"] != "ok"}
    for packet in packet_dir.glob("seg-*.txt"):
        if re.fullmatch(r"seg-\d+\.txt", packet.name) and packet.name not in active:
            packet.unlink()  # Remove only this battery's now-resolved packets.
    for entry in final_entries:
        index = entry["index"]
        if entry["label"] == "ok":
            continue
        correction = corrections.get(index, {})
        lines = [
            f"segment: {index}",
            f"label: {entry['label']}",
            "signals:",
        ]
        for name, reasons in entry["signals"].items():
            lines.append(f"  {name}: {'; '.join(reasons)}")
        lines += [
            "",
            "source:",
            str(rows[index].get("en", "")),
            "",
            "vi:",
            str(rows[index].get("vi", "")),
        ]
        if correction.get("retried"):
            lines += ["", "retry: re-translation attempted (still flagged)"]
        if correction.get("action") not in (None, "", "none"):
            lines.append(f"action: {correction['action']}")
        (packet_dir / f"seg-{index}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )


def _verdict(final_entries: list[dict], rows: list[dict]) -> str:
    """Translation verdict under the M4-T9-C1 policy (content-critical only).

    ``fail`` requires an uncorrectable, content-critical defect:

    * an unresolved **``omission``** (empty output, or a length ratio below
      :data:`LENGTH_RATIO_MIN`), or
    * an unresolved **``number-error`` whose vi text has no digits at all** (a
      number wholly dropped, not merely rendered differently).

    Every other unresolved flag — a digit-present number-error, terminology-error,
    entity-error, inconsistent, mistranslation-risk, unnatural — is a ``review``
    (surfaced through the review packets and counters, but exit 0). ``pass`` means
    no unresolved flags. ``mistranslation-risk`` is auxiliary and never fails.
    """
    fail = False
    for entry in final_entries:
        label = entry["label"]
        if label == "omission":
            fail = True
        elif label == "number-error":
            index = int(entry["index"])
            vi_text = (
                str(rows[index].get("vi", "")) if 0 <= index < len(rows) else ""
            )
            if not re.search(r"\d", vi_text):
                fail = True
    if fail:
        return "fail"
    has_other = any(entry["label"] != "ok" for entry in final_entries)
    return "review" if has_other else "pass"


def run_battery(
    rows: list[dict],
    glossary: dict[str, str],
    engine: Any = None,
    src_lang: str = "en",
    tgt_lang: str = "vi",
    out_dir: str | Path | None = None,
    *,
    refiner: Any = None,
) -> tuple[list[dict], dict]:
    """Score -> correct -> re-score the corrected rows; return ``(rows, artifact)``.

    ``rows`` is mutated in place (corrected vi text). The artifact is::

        {"verdict": "pass"|"review"|"fail",
         "counts": {<label>: <int>},
         "roundtrip": <status>,
         "segments": [{"index", "label", "signals", "retried", "resolved",
                       "action"}, ...]}

    Verdict policy (M4-T9-C1): ``fail`` only for content-critical uncorrectable
    defects — an unresolved ``omission`` or an unresolved digit-free
    ``number-error``; any other unresolved flag is ``review``; ``pass`` when
    nothing is flagged. See :func:`_verdict`.

    When ``out_dir`` is given, unresolved flagged rows also get a
    ``review/translation/seg-<index>.txt`` packet.

    ``refiner`` (M5-T2) is an optional callable
    ``refine(rows, final_entries) -> (rows, final_entries, summary)`` invoked
    after the deterministic corrections. The returned entries drive the counts,
    verdict and review packets, and ``summary`` is recorded under the artifact's
    ``"refine"`` key (absent when no refiner runs, so the default path is
    byte-identical). Exceptions propagate to the caller — the stage wraps refine
    so it can never fail a run.
    """
    first, summary = score_rows(rows, glossary, engine, src_lang, tgt_lang)
    rows, corrections = correct_rows(rows, first, glossary, engine, src_lang, tgt_lang)
    correction_by_index = {record["index"]: record for record in corrections}

    final_entries: list[dict] = []
    for entry in first:
        index = entry["index"]
        correction = correction_by_index.get(index)
        if correction is not None:
            final_entries.append(
                {"index": index, "label": correction["label"], "signals": correction["signals"]}
            )
        else:
            final_entries.append(
                {"index": index, "label": entry["label"], "signals": entry["signals"]}
            )

    # Optional refine cycle (M5-T2): the injected refiner may propose better
    # candidates for still-flagged rows. It mutates ``rows`` and returns the
    # updated entries plus a summary; exceptions propagate (the stage guards,
    # so refine can never fail a run).
    refine_summary: dict | None = None
    if refiner is not None:
        rows, final_entries, refine_summary = refiner(rows, final_entries)

    counts: dict[str, int] = {}
    for entry in final_entries:
        counts[entry["label"]] = counts.get(entry["label"], 0) + 1

    verdict = _verdict(final_entries, rows)

    segments = []
    for entry in final_entries:
        correction = correction_by_index.get(entry["index"], {})
        segments.append(
            {
                "index": entry["index"],
                "label": entry["label"],
                "signals": entry["signals"],
                "retried": bool(correction.get("retried", False)),
                "resolved": entry["label"] == "ok",
                "action": correction.get("action", "none"),
            }
        )

    artifact = {
        "verdict": verdict,
        "counts": counts,
        "roundtrip": summary["roundtrip"],
        "segments": segments,
    }
    if refine_summary is not None:
        artifact["refine"] = refine_summary

    if out_dir is not None:
        _write_packets(out_dir, rows, final_entries, correction_by_index)

    return rows, artifact
