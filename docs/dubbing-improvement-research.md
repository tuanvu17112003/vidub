# ViDub improvement research — 2026-10-01

## Decision

Implementation follow-up: [phase 1 results](translation-grammar-safety.md).
Subtitle scheduling and final-audio measurement are implemented and verified in
[phase 2 results](subtitle-scheduling-and-placed-qc.md).
The evidence below records the pre-change exports; phase 1 includes a separate
offline repair check rather than a rerendered video benchmark.

Prioritize translation repair and subtitle scheduling. Preserve the new natural
pace policy, but measure combined synthesis/stretch speed and onset displacement.
Benchmark a newer translation candidate only after removing deterministic
corruptions in the pipeline. A better model cannot compensate for an entity
repair that inserts untranslated English or a subtitle algorithm that has no
extra display time.

This is research plus reproducible diagnostics, not a claim that the proposed
changes already improve listener ratings. The diagnostic phase did not download
models or change production algorithms. A separate VoxCPM2 voice preview was
rendered on 2026-10-24; its technical results and listening limits are recorded
in the [quality rework benchmark](quality-rework-benchmark.md).

## Evidence from the two user exports

The diagnostic reads the authoritative sentence text and placed spans. The
17 CPS threshold is the current project profile; it is not asserted here to
be a universal requirement for every distributor or audience.

| Observation | LPZh9BOjkQs | 5sLYAQS9sWQ |
|---|---:|---:|
| Sentence units | 58 | 54 |
| Apostrophe tokens treated as entities and retained in Vietnamese | 3 | 2 |
| Those affected units labeled `ok` by the battery | 3 | 2 |
| Unit captions above 17 CPS using placed audio spans | 54 | 47 |
| Unit captions above 17 CPS using source sentence spans | 43 | 23 |
| Vietnamese characters, including spaces | 7,503 | 4,341 |
| Sum of placed speech durations | 378.93 s | 223.48 s |
| Aggregate reading speed over those durations | 19.80 CPS | 19.42 CPS |
| Display time needed at 17 CPS, before rounding/boundary whitespace | 441.35 s | 255.35 s |

These videos differ: their counts are diagnostic, not a matched A/B comparison.
Source spans are a feasibility probe, not a proposed final subtitle track; they
may precede/follow placed audio and still require readability and overlap checks.

The newer export does apply `natural-pace-v1`: no slowing, no clips, one
rate-resynthesis, two stretches. The closing call to action reaches 1.199×
combined speed despite `max_stretch=1.073`. Translation still includes a broken
GPT expansion, programming `code` rendered as `mật mã`, and `Transformer`
rendered as an electrical transformer. Suspicious ASR text includes `GTP3`,
`GPP`, and `45-10`; the correct recording content has not been verified.

## 1. Repair entity handling before switching MT engines

**Confirmed defect.** `extract_entities` accepts capitalized tokens containing
apostrophes. The ordinary-word stoplist does not include inflected contractions.
`correct_rows` then repairs entity errors with source splicing. The two outputs
retain `AI's` twice, `I'll`, `I've`, and `I'm`; all five units are labeled `ok`.

**Proposed change:**

* Exclude English grammatical contractions from entity preservation. Recognize
  curly and straight apostrophes; do not globally remove apostrophes from names.
* For `AI's`, protect the entity **AI**, while translating the possessive
  relationship. Do not splice the full possessive token into Vietnamese.
* Treat entity repair as a candidate, then revalidate fluency, leakage,
  completeness, terminology, and numbers. Keep the original text if repair is
  worse. Prefer translation-time term constraints over reconstructing a sentence
  from independently translated source fragments.
* Add a target-text English-contraction detector. Allow legitimate names,
  technical abbreviations, and explicitly protected phrases.

**Acceptance:** zero retained grammatical contractions in the five real cases;
preserve AI, GPT, genuine apostrophe names and numeric values; no lost clauses.
Test `I'd` without assuming whether it means “I had” or “I would.”

## 2. Use domain terms and context deliberately

Both runs used zero glossary terms. Start with an explicit technical profile:

| Source term | Vietnamese rendering |
|---|---|
| Transformer | Transformer |
| large language model / LLM | mô hình ngôn ngữ lớn / LLM |
| foundation model | mô hình nền tảng |
| pre-training | tiền huấn luyện |
| fine-tuning | tinh chỉnh |
| feed-forward neural network | mạng nơ-ron truyền thẳng |
| attention | cơ chế attention, or cơ chế chú ý consistently |
| parallel processing | xử lý song song |
| code, in programming context | mã nguồn |

Use contextual profiles: `code` may mean a cipher in another video, and a
physical transformer is correctly `máy biến áp`. Avoid unconditional global
replacement. Feed only relevant terms and a small amount of neighboring source
context; translate the current unit alone. Retain contamination checks because
the project's previous contextual Hunyuan runs copied neighboring material.

**Model experiment:** [Hy-MT2](https://github.com/Tencent-Hunyuan/Hy-MT2)
was released in May 2026, supports Vietnamese and provides terminology, style,
and background-context instructions. Official 1.8B and 7B GGUF checkpoints
exist. The [1.8B GGUF card](https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF)
and [license](https://raw.githubusercontent.com/Tencent-Hunyuan/Hy-MT2/main/LICENSE.txt)
identify Apache-2.0. This makes **Hy-MT2-1.8B the first new MT candidate to
benchmark**, with 7B as a second arm if runtime permits. GPU feasibility and
Vietnamese quality remain unmeasured on this 3080. Extreme low-bit variants
have additional runtime requirements; start with a conventional quantization.

The previous HY-MT1.5 evaluation did not justify promotion over envit5. Its
negative result does not evaluate Hy-MT2. Verify model templates, generation
behavior and installed server compatibility rather than changing only a model
name and assuming the old adapter is correct. Leave defaults unchanged until
human-reviewed results justify promotion.

## 3. Improve transcription independently of translation

Current normal ASR uses **base** in these exports and `beam_size=1` in the
wrapper. Its targeted retry path already uses stronger decoding.

Compare base/beam-1 with a cached larger English-capable Whisper model and
beam-5 on the same suspicious audio excerpts. Supply technical vocabulary such
as GPT, GPT-3, Transformer, LLM, and petabyte as hints. The official
[faster-whisper API](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py)
supports `initial_prompt`, `hotwords`, and beam configuration. Hints are biases,
not ground truth; preserve spoken counterexamples such as an intentionally
incorrect next-word prediction. Do not rewrite `45-10` to a guessed number.

Keep source-transcript fixes and MT experiments separate. Otherwise a better
translation may merely reflect an easier input transcript. Measure name/number
accuracy alongside WER; a small WER change can hide a serious numeric error.

## 4. Replace CPS-only recursive splitting with constrained scheduling

For unchanged text in unchanged total time, splitting cannot reduce aggregate
reading speed:

`aggregate CPS = total displayed characters / total display seconds`.

The current recursion divides over-CPS text inside the same parent span. With
character-proportional allocation, children have approximately the same CPS;
with word-edge allocation, some become much faster. Splitting also creates
short-duration violations. Boundary whitespace removal is too small to solve
the measured capacity problem.

**Proposed scheduler:** first choose readable sentence/clause chunks, then assign
times under constraints: minimum duration, maximum duration, character/line
limits, chronological order, no overlap, and bounded lead/lag relative to speech.
Borrow adjacent silence within those bounds. Use dynamic programming or a
bounded deterministic search to minimize violations and timing displacement.
Split to satisfy text length or maximum duration, not as the sole response to
high CPS. If no feasible schedule exists, flag it; offer a separately reviewed,
shorter **subtitle** wording without silently deleting spoken content.

The [Netflix general requirements](https://partnerhelp.netflixstudios.com/hc/en-us/articles/215758617-Timed-Text-Style-Guide-General-Requirements)
are useful for duration and line-layout constraints, but a distributor's rules
should be an explicit profile. Keep 17 CPS as the existing project target while
testing this algorithm; changing the limit would confound the comparison.

The new export needs roughly 31.87 more display seconds in aggregate to reach
17 CPS with the same text. Source sentence spans reduce over-limit units from
47 to 23, demonstrating usable display time outside placed speech. This does
not prove all remaining captions can be repaired without text changes.

## 5. Fit duration after checking meaning; track total speed

Keep whole-sentence synthesis. Generate at most three shorter candidates only
for real overflow, validate meaning-related constraints first, then synthesize
survivors at natural pace and rank by measured fit. Penalize unnecessary speed
changes and onset drift. A candidate that loses a clause must not win by being
shorter. Avoid blind clipping of translation tokens or words.

[Duration-based Translation](https://aclanthology.org/2025.emnlp-demos.37/)
uses source-duration-driven phoneme targets and iterative translation adjustment;
its experiments cover English, Spanish and Korean, not Vietnamese. Its reported
gains motivate an experiment, not an expected ViDub improvement.
[SSPO](https://aclanthology.org/2025.acl-long.227/) treats segment-level duration
alignment as a preference-training problem. Inference-time candidate selection
is a cheaper first step here; it is not equivalent to reproducing that training.

Compute `effective_speed = atempo / length_scale` in both policy and report.
If a proposed overall cap is 1.20×, synthesis at 1.15× leaves only about 1.043×
for downstream stretching. Choose that cap through listening tests rather than
assuming it is universally natural. Allow silence instead of slowing fitting
speech. Make recentering a separate synchronization decision: preserve source
onsets where possible, and measure displacement instead of reporting pace alone.

## 6. Correct measurement and quality selection

Final audio QC currently probes original script windows; all five silence flags
across these exports have audio in their aligned placed spans. Probe those
spans, confirm file existence/duration and signal energy, and separate missing
speech, silence, clipping and synchronization warnings.

The round-trip lexical score and Vietnamese word-count ratio are risk signals,
not semantic validators. The local TransQuest study also missed many number
swaps. Prefer a small bilingual error corpus, protected-term/number checks and
blinded pairwise review before a learned scorer becomes a gate.

[COMETKiwi](https://huggingface.co/Unbabel/wmt22-cometkiwi-da)
accepts source/translation pairs and lists Vietnamese among its covered
languages, but that checkpoint is **CC-BY-NC-SA-4.0** and gated. It is not an
appropriate automatic addition to the project's commercially usable default.
Architecture/code licensing does not establish checkpoint licensing.

## 7. TTS upgrade only after fidelity is stable

[VoxCPM2](https://github.com/OpenBMB/VoxCPM) explicitly supports Vietnamese;
the [checkpoint card](https://huggingface.co/openbmb/VoxCPM2) identifies
Apache-2.0. The authors report roughly 8 GB VRAM and multilingual voice cloning.
Those facts justified an isolated Vietnamese voice preview on the 10 GB RTX 3080,
not promotion to the production default. The preview used the current reviewed
translation and subtitle contract, but shortened eight spoken forms for timing
and three subtitle forms for reading speed; it is not a fixed-text Piper versus
VoxCPM2 comparison. Published intelligibility metrics are not a matched Piper
comparison on these videos. Listener preference and naturalness remain unscored.
One cold-start sample took 22.47 seconds to load and generated 5.44 seconds of
audio in 4.21 seconds (RTF 0.77); process peak was 5.28 GiB allocated and
5.46 GiB reserved. This is only one sample. Measure technical pronunciation,
duration variance and sustained runtime, then get listener preference before
considering adoption.

## Proposed experiment order and acceptance

These are prospective targets, not achieved results.

| Order | Isolated experiment | Acceptance |
|---|---|---|
| 1 | Contraction/entity repair and technical profile | Five leakage cases repaired; zero name/number/clause regressions; Transformer consistent in technical contexts |
| 2 | Subtitle scheduler on fixed translation/audio | At least 50% fewer violations at the same profile; no overlaps, lost text or new excessive timing displacement |
| 3 | Placed-span QC and combined-speed reporting | Five false silence flags removed without hiding genuinely empty WAVs; combined closing-line speed reported as about 1.199× |
| 4 | ASR base vs stronger cached model/decoding | Suspicious names/numbers assessed against audio; no regression on intended counterexamples |
| 5 | envit5 vs HY-MT1.5 vs Hy-MT2, same repaired input | Blinded bilingual preference, fewer critical errors, acceptable runtime/VRAM; do not promote on proxy score alone |
| 6 | Validated duration candidates | No added fidelity errors or clips; lower combined-speed variation and onset displacement |
| 7 | Piper vs VoxCPM2, fixed text | Listener preference plus verified Vietnamese intelligibility, stable duration and feasible resources |

Use the 112 current sentence units as a regression pool, with at least 30
manually reviewed difficult cases, then hold out another video. Include clean
sentences, negation, quantities, genuine names, contractions, technical terms
and intentional wrong-word examples. Separate transcript fidelity, translation
accuracy, naturalness, timing and subtitle readability. Archive prompts, input
hashes, model/quant versions and candidate rejection reasons for every arm.

## VERIFY and status

```bash
./.venv/Scripts/python.exe scripts/research_dubbing_diagnostics.py
```

* **MET:** diagnostic command completes and produces the table above plus
  `eval-cache/dubbing-research-diagnostics.json`.
* **MET:** contraction leakage and subtitle timing-capacity hypotheses reproduced
  on both exports without loading models.
* **MET:** primary sources checked for proposed API/model capabilities and
  checkpoint licenses.
* **TECHNICAL PREVIEW MET:** VoxCPM2 loaded and rendered the 477.481-second clip
  in an isolated environment; final timing reported zero clipped rows and
  subtitle QC reported zero unresolved violations. A separate cold-start sample
  measured 5.28 GiB peak allocated VRAM. This is not a listener-quality result or
  a matched TTS benchmark.
* **NOT MET / not yet attempted:** bilingual model comparison, listener-quality
  improvement, and implementation acceptance targets. This research does not
  substitute for those evaluations.
