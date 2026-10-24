# TransQuest reference-free QE validation (En→Vi) — QUALITY-ROADMAP.md D6

**Status:** evaluation complete · **Verdict: DEFER (do not wire into the pipeline now)**
**Task:** M4-T8 · **Date:** 2026-09-29

## Purpose

QUALITY-ROADMAP.md §2 proposes a reference-free **quality-estimation (QE)** model
as an extra Level-2 translation signal ("Mistranslation / naturalness / register
→ QE model ..."), and decision **D6** says *"TransQuest yes (validate first);
cometkiwi defer"*. This document records that validation: does TransQuest's
reference-free MonotransQuest model separate good En→Vi translations from known
bad ones well enough to adopt as a ViDub signal?

## Setup (scratch venv — the main `.venv` is untouched)

TransQuest is a 2021-era package that pins an old `transformers`; installing it
into the main venv would downgrade the stack. Everything ran in a disposable
`.eval-venv/` (gitignored, **not** in `pyproject.toml`):

```bash
./.venv/Scripts/python.exe -m venv .eval-venv
./.eval-venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
./.eval-venv/Scripts/python.exe -m pip install transquest
# transquest 1.1.1 pulls transformers 5.17.0, which REMOVED
# transformers.convert_graph_to_onnx -> `import transquest...run_model` fails:
#   ModuleNotFoundError: No module named 'transformers.convert_graph_to_onnx'
# Fixed by pinning the transformer stack TransQuest was built against:
./.eval-venv/Scripts/python.exe -m pip install transformers==4.34.1 huggingface_hub==0.16.4 numpy==1.26.4
```

Resulting working set:

| Package | Version |
|---|---|
| `transquest` | 1.1.1 |
| `transformers` | 4.34.1 (pinned; 5.x breaks the import) |
| `huggingface_hub` | 0.16.4 |
| `numpy` | 1.26.4 |
| `torch` | 2.14.0+cpu |
| model | `TransQuest/monotransquest-da-multilingual` (XLM-R large, **2.24 GB** weights) |
| API | `transquest.algo.sentence_level.monotransquest.run_model.MonoTransQuestModel("xlmroberta", ...)`; `predict([[src, mt], ...])` → score in 0–1 |

> **API correction vs the task spec:** the class is `MonoTransQuestModel` (the spec
> said "TransQuest MonteroScaling"); `predict` takes a list of `[source, reference]`
> 2-element lists, not dicts or `InputExample` objects.

## Data construction

`scripts/eval_transquest.py` (runs under `.eval-venv`):

* **30 "good" pairs** — real `(en, vi)` rows from `output/youtube-a7TOameRqoY/script.tsv`
  and `output/demo_en2/script.tsv` (TSV columns stripped; kept only complete,
  diacritic-bearing, non-passthrough sentences).
* **30 "bad" pairs** — one deterministic corruption per good pair, tagged with its
  kind: `number-swap` (change a digit), `clause-drop` (drop a clause/suffix),
  `term-swap` (replace a glossary term's Vietnamese rendering with a wrong one),
  `entity-drop` (remove a Latin entity).
* Both classes are scored with the same model; the pairs share their source, so the
  only variable is translation quality.

Reproduce:

```bash
./.eval-venv/Scripts/python.exe scripts/eval_transquest.py
```

## Results

| Metric | Value |
|---|---|
| pairs | 60 (30 good / 30 bad) |
| **mean score — good** | **0.7603** |
| **mean score — bad** | **0.6806** |
| **separation (good − bad)** | **0.0797** |
| **AUC (2-class ranking)** | **0.7722** |
| **best-separating threshold** | **0.7382** (accuracy 0.7333) |
| misclassified good / bad | 5 / 11 |

Per corruption (mean bad score; lower = more detected):

| Corruption | n | mean score |
|---|---|---|
| `clause-drop` (omission) | 18 | **0.6380** |
| `term-swap` (terminology) | 2 | 0.7017 |
| `number-swap` (number) | 10 | **0.7530** |

Reading: TransQuest reliably lowers the score for **structural omissions**
(`clause-drop`), but a **number swap barely moves the score** (0.753 — essentially
the same as the good mean of 0.760) and terminology swaps are only weakly
penalised. In other words, the errors it detects are the ones ViDub's existing
deterministic M4-T3 battery already catches, while it misses the number errors the
battery catches for free.

## Limitations

* **Synthetic, single-source corruption set** — 60 pairs from one demo/sample
  corpus; the bad class is rule-generated, so it measures *relative* sensitivity,
  not calibrated accuracy on naturally occurring MT errors.
* **Not En→Vi-specific** — `monotransquest-da-multilingual` is a multilingual QE
  checkpoint trained on DA data; En→Vi is out of its explicit training languages,
  so the 0.77 AUC is a lower bound on a domain-adapted model.
* **Cost** — 2.24 GB XLM-R-large weights and a `transformers<5` pin that conflicts
  with the installed (transformers 5.x) stack; CPU inference is ~1.5 s/pair.
* No reference translations are used (by design), so it cannot detect errors that
  a reference would expose.

## Recommended usage

A **soft auxiliary signal only, never a gate**. If ever enabled: flag a segment as
`mistranslation-risk` when `score < 0.74`, and only as one more vote alongside the
deterministic battery — never as the deciding signal, because it demonstrably
ignores number errors.

## Verdict

**DEFER.**

Reasoning: the validation is *positive but weak* — AUC 0.77 is usable as a
tie-breaker, yet the gains are concentrated exactly where ViDub already has
deterministic coverage (omissions/length via the battery), while the model adds a
2.24 GB XLM-R-large download plus an old-`transformers` pin that would regress the
working stack. It is not worth wiring into the pipeline now. Keep
`scripts/eval_transquest.py` as an offline research harness; revisit if/when a
small **En→Vi-specific** QE checkpoint (or an open COMETKiwi alternative) appears.
`cometkiwi` remains deferred per D6.
