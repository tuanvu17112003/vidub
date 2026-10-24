# Isometric length budgeting (ISO-1)

**Status:** closed at **v0.11.0** with an honest negative result. The matched
timing-stage A/B below did not meet its pre-registered targets, so the pass ships
**off by default** and stays available via `--isometric` for further work. Every
number below comes from a run named next to it; nothing is projected, and the
miss is stated plainly.

## The problem

Q-1's pacing work ended with the survives needing **1.30–1.54×** compression
while the timing caps are **1.20×** (normal) and **1.30×** (short windows). The
per-clip reasons are conclusive: the translation itself is too long for the time
its window allows, so raising the caps would only change the perceived pace (the
thing the User complained about earlier). The fix belongs **upstream**, at
translation time: give the MT engine the character budget the window allows, and
when it still overshoots, take one bounded shorter candidate **only** when a
deterministic predicate says it is genuinely better.

The pre-feature baseline on the User's tech video (`326ea9c4…`, 81 sentence
units) is **37 units over budget** at the default headroom 0.9 — measured by the
T1 calibration and re-read by this workstream's harness (`score` on the old job
prints `null` + a reason because that file predates the isometry records). Those
37 units are the same ones Q-1's timing stage later clipped or stretched.

## The design

**One budget function, one constant set** (`vidub/utils/isometry.py`), used by
the translate stage only; the timing stage's own fit is unchanged.

```
window    = max(end - start, 0.0)            # the unit's locked span
silence   = max(next_start - end, 0.0)       # 0 when there is no next unit
share     = clamp(include_following_silence, 0.0, 1.0)   # default 0.0
effective = window + silence * share
budget    = round(effective * chars_per_second * headroom)
budget    = clamp(budget, MIN_BUDGET_CHARS=8, MAX_BUDGET_CHARS=600)
```

* `chars_per_second` is the shared `CONCISE_CHARS_PER_SECOND = 20.0`
  ("range ~18–25", calibrated on this video) — never re-declared.
* `headroom = 0.9` leaves 10 % room so FIT-1's `SPEED_AS_IS = 1.10` band can
  absorb normal variation instead of triggering a stretch.
* The floor and ceiling keep a tiny or mis-measured window from demanding a
  zero-length or absurd translation.

**Three levers, in order:**

1. **Decode/prompt steering (ISO-1-T2).** Every engine accepts an optional
   per-item `budgets` sequence. envit5 turns a character budget into a token cap
   (`max_length`, floored at `MIN_BUDGET_TOKENS = 8`, capped at `MAX_LENGTH`) plus
   `length_penalty = 0.6`; hunyuan/ollama append the proven
   `At most N characters, keep the meaning exactly, and output only the translation.`
   clause. **No budgets = today's prompts/decode byte-for-byte.**
2. **One bounded budgeted repair (ISO-1-T3).** After the completeness guard, each
   still-over-budget unit gets **one** `engine.translate([source], …, budgets=[b])`
   call. The candidate replaces the text **only** when `accept_shrunk` is true.
3. **The predicate.** `accept_shrunk(source, candidate, budget, previous_vi,
   next_vi, …)` requires: non-empty; the engine validators pass (Vietnamese
   diacritics, numbers kept, ≤ 2.2× source words); `guard_signals` reports neither
   `omission-risk` (< 0.75 coverage) nor `sentence-loss`; `contamination_signals`
   reports no `context-bleed`; the candidate is not longer than the current text;
   and its `projected_stretch` is strictly lower. A failure keeps the original and
   records `isometric: "kept"`.

**Records.** `units.json` per unit gains `budget_chars`, `chars`, `over_budget`,
`isometric` ∈ `none`/`shrunk`/`kept`/`unavailable`, `chars_after`,
`projected_stretch_before`/`after`; `translation_qc.json` `units.flags` gains
`over_budget_after`, `isometric_shrunk`, `isometric_kept`. The pass is off by
default, so the default path is byte-identical to before; `--isometric` opts in
and `--no-sentence-units` never uses it.

## The matched A/B (real runs, misses included)

Two real translate+timing runs on the reference tech video, both arms on the
**same input file with identical structure** (74 ASR rows → 113 script rows → 83
sentence units each), `--mt-engine hunyuan`, `--mode subtitles-only`,
`--force translate`, both with `--no-hunyuan-refine` (to isolate the isometric
pass). Run via `scripts/eval_isometry.py run`; the two caches are
`.vidub-cache-iso/run-off-timing` (OFF) and `.vidub-cache-iso/run-on-c1c` (ON).

| metric | isometric **OFF** (`--no-isometric`) | isometric **ON** |
|---|---|---|
| clips | **2** (idx 3: 0.390 s, idx 6: 0.324 s) | **2** (idx 3: 0.341 s, idx 50: 0.113 s) |
| `stretched` | **19** | **27** |
| `retranslated` | 2 | 3 |
| `max_stretch` | 1.20 | 1.20 |

- **OFF (`--no-isometric`, `.vidub-cache-iso/run-off-timing`)**: clips **2**
  (idx 3: 0.390 s, idx 6: 0.324 s), `stretched` **19**, `retranslated` 2,
  `max_stretch` 1.20.
- **ON (isometric, `.vidub-cache-iso/run-on-c1c`)**: clips **2** (idx 3: 0.341 s,
  idx 50: 0.113 s), `stretched` **27**, `retranslated` 3, `max_stretch` 1.20.
- Translation side (ON): 29 units over budget → **17 shrunk** (0 before the
  C1-cont2b fixes), `kept` 12 with principled reasons (`no-candidate-shorter`
  × 7, real neighbour bleed × 3, candidate longer than the source × 2); the
  adaptive short floor (0.65 below 1.8 s) applied to **21** units.

**Verdict: the pre-registered targets are NOT MET — clips 2 → 2 (target 0) and
`stretched` 19 → 27 (target ≤ 12, i.e. worse than the matched control).** Partial
benefit: the surviving trims are smaller (0.454 s total vs 0.714 s).

### Defects the persisted reasons exposed (and which were fixed)

- **D1** — the contamination check compared a compression candidate against the
  unit's **own current text** (self-bleed false positives, 26 rejections at
  0.57–1.00); it now compares the **neighbouring units**.
- **D2** — one compression prompt omitted the target language (31 validator
  rejections "does not look like vi"); all three variants now open with the
  proven request shape.

### Open hypothesis (not a conclusion)

Shorter texts appear to re-shape the fitter's window/slack and pace decisions
instead of removing the squeeze; the suspects are the pace baseline, neighbour
smoothing and the availability-window use.

## Flags and defaults

The pass is **off by default** (v0.11.0): the matched A/B above showed the
pre-registered targets not met — clips unchanged (2 → 2, target 0) and
`stretched` worse (19 → 27, target ≤ 12) — so it is opt-in via `--isometric` and
no path re-enables it silently.

| Flag | Default | Meaning |
|---|---|---|
| `--isometric` / `--no-isometric` | **off** | Compute each sentence unit's character budget, pass it to the engine and run the one bounded budgeted repair (sentence-unit path only; no-op on `--no-sentence-units`). |
| `--isometric-headroom FLOAT` | `0.9` | Fraction of the window's natural-pace capacity a translation may use; must be in `(0, 1]`. |
| `--isometric-cps FLOAT` | `20.0` | Natural Vietnamese pace in characters per second; must be in `[10, 30]`. |

A change to any of these changes `mt_signature` (`iso=`, `iso_head=`,
`iso_cps=`, `iso_cov=`, `iso_win=`), so a cached translation is invalidated
automatically.

## How to reproduce

```bash
# 0) the harness refuses any cache/output path inside the User's caches.
#    score is READ-ONLY; score either arm directly:
./.venv/Scripts/python.exe scripts/eval_isometry.py score \
    --run .vidub-cache-iso/run-off-timing
./.venv/Scripts/python.exe scripts/eval_isometry.py score \
    --run .vidub-cache-iso/run-on-c1c

# 1) isometric ON, hunyuan, no refine (isolates the isometric pass):
./.venv/Scripts/python.exe scripts/eval_isometry.py run \
    --input .vidub-cache/downloads/generic-youtube-a7TOameRqoY.mkv \
    --cache-dir .vidub-cache-iso/run-on-c1c \
    --output-dir .vidub-cache-iso/work-on \
    --mt-engine hunyuan --mode subtitles-only --force translate \
    --extra=--no-hunyuan-refine

# 2) the matched control: same command with --extra=--no-isometric,
#    --cache-dir .vidub-cache-iso/run-off-timing, --output-dir .../work-off.
```
