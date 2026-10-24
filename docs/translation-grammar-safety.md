# Translation repair phase 1 — 2026-10-01

Implemented the first step of `dubbing-improvement-research.md` without adding
dependencies or changing the default MT model.

## Changes

- Entity extraction excludes grammatical contractions, including curly
  apostrophes. An uppercase acronym possessive such as `AI's` preserves `AI`;
  apostrophized names such as `O'Reilly` and `McDonald's` remain intact.
- English-to-Vietnamese QC flags retained contractions and acronym possessives
  as `unnatural`. The existing one full-source retry can repair these without
  guessing ambiguous expansions such as `I'd`. Explicit glossary targets can
  protect English examples; passthrough rows remain exempt.
- Corrective candidates are rejected if they introduce another deterministic
  critical signal, completeness-guard flag, or grammar leak. The original and
  its unresolved review flag survive rejection. These heuristics do not prove
  semantic fidelity or fluent Vietnamese.
- `mt_signature` includes the grammar-QC policy and glossary content SHA-256.
  Editing a glossary at the same path invalidates translation and downstream
  artifacts on the next pipeline run. Missing glossary paths fail explicitly.
- `samples/glossary-ai-tutorial.tsv` is opt-in through `--glossary`. Its mappings
  for `code` and `transformer` apply to AI/programming tutorials, where those
  words have the reported technical senses. They are not global replacements.

## VERIFY

Run in Git Bash with the project venv:

```bash
./.venv/Scripts/python.exe -m pytest tests/test_translation_grammar_safety.py tests/test_tqc.py tests/test_pipeline.py -q
./.venv/Scripts/python.exe scripts/eval_translation_grammar_safety.py
./.venv/Scripts/python.exe -m pytest -q
```

The evaluation uses only the installed envit5 model with Hugging Face offline
mode. It writes `eval-cache/translation-grammar-safety.json`; it does not edit
the original caches, WAVs, subtitles or exported videos.

| Criterion | Result |
|---|---|
| Contractions excluded, acronyms normalized, real names preserved | MET — focused regressions |
| Observed retained grammar caught and removed | MET — 5/5 detected; retained tokens 5 → 0 in the offline repair check |
| Glossary content edits invalidate cached translation | MET — same-path regression |
| Automated verification | MET — full suite 877 passed / 4 skipped; final focused suite 85 passed |
| Sample terminology constraints satisfied | MET after phase 2 follow-up — 8/8, using source-conditioned aliases without rebuilding the sentence |
| Identified awkward closing sentence and isolated phrase | MET after phase 2 follow-up — explicit reference wording in the opt-in tutorial profile |
| New final videos verified | See `subtitle-scheduling-and-placed-qc.md` for isolated full rerenders and decode checks |

Full-suite VERIFY: **877 passed, 4 skipped** (140.12 s). Two additional negative
contraction cases were added after suite collection; the final grammar regression
file passed **17 tests**. The final focused QC/cache suite passed **85 tests**
(23.24 s).

The terminology sample is the first eight flagged units in cache order from
LPZh9BOjkQs, not a representative benchmark or a balanced comparison of both
videos. A satisfied terminology constraint only means the target term appears.
The second video's five-leak evaluation contribution covers its two observed
contractions. Broader model quality, subtitle scheduling and final placed-span
QC were completed in the phase 2 follow-up; broader model/listener evaluation
remains separate.

The initial 7/8 result is retained as historical evidence in the research notes.
Source-conditioned alias corrections now survive other QC repairs, and plural
acronyms such as `GPUs` protect `GPU` rather than forcing an English suffix back
into Vietnamese. The tutorial profile contains two exact-source reference
sentences; these are explicit editorial corrections, not evidence of general
model improvement. Round-trip heuristics and bilingual/listener assessment
remain separate from these deterministic results.
