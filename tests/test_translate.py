"""Tests for the S2 translate stage (M1-T3).

* logic tests use stub translators (no model, no network) to check the
  ``script.json`` shape, the 2.2x truncation rule and the error path;
* a real test loads ``VietAI/envit5-translation`` and checks the output
  contains Vietnamese diacritics (skipped if the model can't be downloaded).
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from pathlib import Path

import pytest

from vidub.config import JobConfig
from vidub.engines.mt.base import BaseTranslator
from vidub.manifest import JobManifest
from vidub.stages.export import build_report
from vidub.stages.translate import (
    SCRIPT_NAME,
    UNITS_NAME,
    _unit_rows,
    main,
    translate,
    truncate_translation,
)
from vidub.utils.glossary import (
    find_source_terms,
    splice_source_terms,
    splice_translation,
    split_source_on_term,
)
from vidub.utils.sentences import split_translation, unit_cut_stats

DIACRITIC_RE = re.compile(r"[ăâêôơưđÁ-ỹá-ỹ]", re.IGNORECASE)
ROOT = Path(__file__).resolve().parents[1]
ENGINE_RUNNER = ROOT / "tests" / "_engine_runner.py"


class FakeTranslator(BaseTranslator):
    """Deterministic fake: reverses each string."""

    def translate(self, texts, src_lang, tgt_lang):
        return [text[::-1] for text in texts]


class ExpandingTranslator(BaseTranslator):
    """Returns a fixed over-long translation for every input."""

    def __init__(self, value: str) -> None:
        self.value = value

    def translate(self, texts, src_lang, tgt_lang):
        return [self.value for _ in texts]


class RecordingTranslator(BaseTranslator):
    """Reverses output but records every batch of texts it is asked to translate."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def translate(self, texts, src_lang, tgt_lang):
        self.calls.append(list(texts))
        return [text[::-1] for text in texts]


SEGMENTS = [
    {"start": 0.0, "end": 4.0, "text": "Hello world", "speaker": "spk0"},
    {"start": 4.0, "end": 6.85, "text": "The weather is sunny today", "speaker": "spk0"},
]


def _make_job(
    tmp_path: Path,
    segments: list[dict] | None,
    sentence_units: bool = False,
) -> tuple[Path, Path, JobConfig]:
    """Build a translate job; defaults to the per-segment path (TU-1).

    Existing per-segment assertions run with ``sentence_units=False`` (today's
    byte-identical path); the sentence-unit tests opt in explicitly.
    """
    cache_dir = tmp_path / "cache"
    artifacts_dir = cache_dir / "hash1"
    artifacts_dir.mkdir(parents=True)
    if segments is not None:
        (artifacts_dir / "segments.json").write_text(
            json.dumps(segments, ensure_ascii=False), encoding="utf-8"
        )
    job_path = artifacts_dir / "job.json"
    JobManifest(
        input_path="demo.mp4",
        video_hash="hash1",
        cache_dir=str(cache_dir),
        src_lang="en",
        tgt_lang="vi",
        paths={"segments": "segments.json", "audio": "audio.wav"},
        stages={"transcribe": "done"},
    ).save(job_path)
    return (
        job_path,
        artifacts_dir,
        JobConfig(
            cache_dir=str(cache_dir),
            sentence_units=sentence_units,
            # ISO-1 close (v0.11.0): the production default is now OFF; these
            # tests exercise the sentence-unit pipeline WITH the isometric pass,
            # so the shared fixture opts in explicitly. The off default itself is
            # pinned in tests/test_config.py.
            isometric=True,
        ),
    )


def test_script_fields_with_fake_translator(tmp_path: Path) -> None:
    job_path, artifacts_dir, cfg = _make_job(tmp_path, SEGMENTS)

    result = translate(str(job_path), cfg, translator=FakeTranslator())

    script = json.loads((artifacts_dir / SCRIPT_NAME).read_text(encoding="utf-8"))
    assert len(script) == 2
    assert set(script[0]) == {
        "start", "end", "speaker", "en", "vi", "spoken_vi", "subtitle_vi", "truncated"
    }
    assert script[0]["en"] == "Hello world"
    assert script[0]["vi"] == "dlrow olleH"
    assert script[0]["speaker"] == "spk0"
    assert script[0]["start"] == 0.0 and script[0]["end"] == 4.0
    assert script[0]["truncated"] is False
    assert script[1]["vi"] == "yadot ynnus si rehtaew ehT"

    reloaded = JobManifest.load(job_path)
    assert reloaded.paths["script"] == SCRIPT_NAME
    assert reloaded.stages["translate"] == "done"
    assert result.video_hash == "hash1"


@pytest.mark.parametrize("sentence_units", [False, True])
def test_translation_initializes_spoken_and_subtitle_text(
    tmp_path: Path, sentence_units: bool
) -> None:
    job_path, artifacts_dir, cfg = _make_job(
        tmp_path, SEGMENTS, sentence_units=sentence_units
    )
    cfg.qc_translation = False

    translate(str(job_path), cfg, translator=FakeTranslator())

    script = json.loads((artifacts_dir / SCRIPT_NAME).read_text(encoding="utf-8"))
    assert all(row["spoken_vi"] == row["subtitle_vi"] == row["vi"] for row in script)
    manifest = JobManifest.load(job_path)
    assert manifest.text_contract_version == 1
    if sentence_units:
        units_doc = json.loads((artifacts_dir / UNITS_NAME).read_text(encoding="utf-8"))
        assert units_doc["text_contract_version"] == 1
        assert all(
            unit["spoken_vi"] == unit["subtitle_vi"] == unit["vi"]
            for unit in units_doc["units"]
        )


def test_truncation_rule_flags_overlong_translation(tmp_path: Path) -> None:
    # source "Hello world" (2 words) -> limit int(2.2 * 2) = 4 words.
    job_path, artifacts_dir, cfg = _make_job(
        tmp_path,
        [{"start": 0.0, "end": 1.0, "text": "Hello world", "speaker": "spk0"}],
    )
    long_value = "Một hai ba. Bốn năm sáu bảy tám."  # 8 words, sentence end at #3

    translate(str(job_path), cfg, translator=ExpandingTranslator(long_value))

    script = json.loads((artifacts_dir / SCRIPT_NAME).read_text(encoding="utf-8"))
    assert script[0]["truncated"] is True
    assert script[0]["vi"] == "Một hai ba."


def test_truncate_translation_edge_cases() -> None:
    # within the limit -> untouched, not flagged
    assert truncate_translation("Một hai ba", src_words=2) == ("Một hai ba", False)
    # over the limit but no sentence end -> cut at the limit, still flagged
    assert truncate_translation("a b c d e", src_words=2) == ("a b c d", True)
    # empty source word count -> no truncation
    assert truncate_translation("anything", src_words=0) == ("anything", False)


# ------------------------------------------------------ M2-C1 hardening
PASSTHROUGH_SEGMENTS = [
    {
        "start": 0.0,
        "end": 3.0,
        "text": "thì tuấn triều tư trong văn thì trước triều tiên.",
        "speaker": "spk0",
    },
    {"start": 3.0, "end": 5.0, "text": "Hello world", "speaker": "spk0"},
]


def test_passthrough_skips_mt_for_already_vietnamese(tmp_path: Path) -> None:
    job_path, artifacts_dir, cfg = _make_job(tmp_path, PASSTHROUGH_SEGMENTS)
    engine = RecordingTranslator()

    translate(str(job_path), cfg, translator=engine)

    script = json.loads((artifacts_dir / SCRIPT_NAME).read_text(encoding="utf-8"))
    # the Vietnamese-diacritic segment is copied en->vi and flagged ...
    assert script[0]["vi"] == PASSTHROUGH_SEGMENTS[0]["text"]
    assert script[0]["en"] == PASSTHROUGH_SEGMENTS[0]["text"]
    assert script[0]["passthrough"] is True
    # ... the English one still went through the engine
    assert "passthrough" not in script[1]
    assert script[1]["vi"] == "dlrow olleH"
    # the MT engine never saw the Vietnamese segment in the translation pass
    # (the QC battery may retry the English row afterwards)
    assert engine.calls[0] == ["Hello world"]


def test_low_confidence_flagged_for_diacritic_free_output(tmp_path: Path) -> None:
    job_path, artifacts_dir, cfg = _make_job(
        tmp_path,
        [
            {
                "start": 0.0,
                "end": 3.0,
                "text": "The weather is sunny today",
                "speaker": "spk0",
            }
        ],
    )

    translate(
        str(job_path), cfg, translator=ExpandingTranslator("hello there friend")
    )

    script = json.loads((artifacts_dir / SCRIPT_NAME).read_text(encoding="utf-8"))
    assert script[0]["low_confidence"] is True
    assert script[0]["vi"] == "hello there friend"


def test_low_confidence_flagged_for_empty_output(tmp_path: Path) -> None:
    job_path, artifacts_dir, cfg = _make_job(
        tmp_path,
        [{"start": 0.0, "end": 3.0, "text": "Hello world today", "speaker": "spk0"}],
    )

    translate(str(job_path), cfg, translator=ExpandingTranslator(""))

    script = json.loads((artifacts_dir / SCRIPT_NAME).read_text(encoding="utf-8"))
    assert script[0]["low_confidence"] is True
    assert script[0]["vi"] == ""


# ------------------------------------------------------ envit5 decode config
class _FakeEncoded(dict):
    """Stand-in for a transformers ``BatchEncoding`` (dict + ``.to``)."""

    def to(self, device):  # noqa: ANN001 - test double
        return self


class _FakeTokenizer:
    def __init__(self) -> None:
        self.last_kwargs: dict | None = None

    def __call__(self, batch, **kwargs):  # noqa: ANN001 - test double
        self.last_kwargs = kwargs
        return _FakeEncoded()

    def batch_decode(self, generated, skip_special_tokens=True):  # noqa: ANN001
        return ["vi: Xin chào" for _ in generated]


class _FakeModel:
    def __init__(self) -> None:
        self.kwargs: dict | None = None

    def generate(self, **kwargs):  # noqa: ANN003 - test double
        self.kwargs = kwargs
        return [[1, 2, 3]]


class _FakeTorch:
    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def no_grad(self) -> "_FakeTorch._NoGrad":
        return _FakeTorch._NoGrad()


def test_envit5_decode_config() -> None:
    """The hardened envit5 decode settings reach ``model.generate``."""
    from vidub.engines.mt.envit5_mt import Envit5Translator

    translator = Envit5Translator()
    model = _FakeModel()
    tokenizer = _FakeTokenizer()
    # Bypass _ensure_loaded: set the lazy attributes directly.
    translator._model = model
    translator._tokenizer = tokenizer
    translator._torch = _FakeTorch()
    translator.device = "cpu"

    result = translator.translate(["Hello there"], "en", "vi")

    assert result == ["Xin chào"]
    assert model.kwargs is not None
    assert model.kwargs["num_beams"] == 4
    assert model.kwargs["max_length"] == 256
    assert model.kwargs["no_repeat_ngram_size"] == 6
    # tokenizer truncation window is unchanged (the model's positional limit)
    assert tokenizer.last_kwargs is not None
    assert tokenizer.last_kwargs["max_length"] == 512


def _envit5_with_fakes():
    """An Envit5Translator with the fake model/tokenizer/torch installed."""
    from vidub.engines.mt.envit5_mt import Envit5Translator

    translator = Envit5Translator()
    model = _FakeModel()
    tokenizer = _FakeTokenizer()
    # Bypass _ensure_loaded: set the lazy attributes directly.
    translator._model = model
    translator._tokenizer = tokenizer
    translator._torch = _FakeTorch()
    translator.device = "cpu"
    return translator, model, tokenizer


def test_envit5_no_budgets_omits_length_penalty() -> None:
    """Without budgets the decode call carries no length_penalty (today's)."""
    translator, model, tokenizer = _envit5_with_fakes()

    translator.translate(["Hello there"], "en", "vi")

    assert model.kwargs is not None
    assert model.kwargs["max_length"] == 256
    assert "length_penalty" not in model.kwargs
    assert tokenizer.last_kwargs is not None
    assert tokenizer.last_kwargs["max_length"] == 512


def test_envit5_budgeted_decode_config() -> None:
    """A budget becomes a token-capped ``generate`` call plus a shorter bias."""
    from vidub.engines.mt.envit5_mt import LENGTH_PENALTY_BUDGETED

    translator, model, _tokenizer = _envit5_with_fakes()

    # 60 characters / 3 chars-per-token == 20 tokens
    result = translator.translate(["Hello there"], "en", "vi", budgets=[60])

    assert result == ["Xin chào"]
    assert model.kwargs is not None
    assert model.kwargs["max_length"] == 20
    assert model.kwargs["length_penalty"] == LENGTH_PENALTY_BUDGETED
    assert model.kwargs["num_beams"] == 4
    assert model.kwargs["no_repeat_ngram_size"] == 6


def test_envit5_budget_floor_never_truncates_below_minimum() -> None:
    """A tiny budget still yields at least MIN_BUDGET_TOKENS tokens."""
    from vidub.engines.mt.envit5_mt import MIN_BUDGET_TOKENS

    translator, model, _tokenizer = _envit5_with_fakes()

    translator.translate(["Hello there"], "en", "vi", budgets=[1])

    assert model.kwargs is not None
    assert model.kwargs["max_length"] == MIN_BUDGET_TOKENS


def test_envit5_budget_is_capped_at_max_length() -> None:
    translator, model, _tokenizer = _envit5_with_fakes()

    translator.translate(["Hello there"], "en", "vi", budgets=[100_000])

    assert model.kwargs is not None
    assert model.kwargs["max_length"] == 256


def test_envit5_batch_uses_the_largest_budget() -> None:
    # 90 / 3 == 30, 30 / 3 == 10 -> the batch cap is max(10, 30) == 30
    translator, model, _tokenizer = _envit5_with_fakes()

    translator.translate(["a", "b"], "en", "vi", budgets=[30, 90])

    assert model.kwargs is not None
    assert model.kwargs["max_length"] == 30


def test_envit5_budgets_length_mismatch_raises() -> None:
    from vidub.engines.mt.envit5_mt import Envit5Translator

    translator = Envit5Translator()
    with pytest.raises(ValueError):
        translator.translate(["a", "b"], "en", "vi", budgets=[10])


def test_missing_segments_raises_and_cli_exits_nonzero(
    tmp_path: Path, capsys
) -> None:
    job_path, _artifacts_dir, cfg = _make_job(tmp_path, None)

    with pytest.raises(FileNotFoundError) as excinfo:
        translate(str(job_path), cfg, translator=FakeTranslator())
    assert "transcribe" in str(excinfo.value)

    rc = main([str(job_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "error:" in captured.err


def test_real_envit5_translation_has_diacritics(tmp_path: Path) -> None:
    # envit5 pulls in torch; run it in its own process to keep the suite safe.
    result_path = tmp_path / "mt_result.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(ENGINE_RUNNER),
            "mt",
            str(result_path),
            "Hello, how are you today?",
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if proc.returncode != 0:
        pytest.skip(f"could not load envit5 (offline?): {proc.stderr.strip()[-500:]}")

    payload = json.loads(result_path.read_text(encoding="utf-8"))

    # envit5 must run on the GPU when the torch build exposes one (device-agnostic
    # otherwise): the runner reports the device it resolved.
    assert payload["device"] in {"cuda", "cpu"}, payload
    if payload["device"] == "cuda":
        import torch

        assert torch.cuda.is_available()

    results = payload["translations"]
    assert results, "no translation returned"
    vi = results[0]
    assert vi.strip(), "empty translation"
    assert len(DIACRITIC_RE.findall(vi)) >= 3, f"no Vietnamese diacritics in: {vi!r}"


# ------------------------------------------- M3-C1 cleanup + source-side splice
class MappingTranslator(BaseTranslator):
    """Returns a mapping[input] (fallback ``"??"``) and records every call."""

    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping
        self.calls: list[list[str]] = []

    def translate(self, texts, src_lang, tgt_lang):
        self.calls.append(list(texts))
        return [self.mapping.get(text, "??") for text in texts]


def test_translate_cleans_hesitation_before_length_check(tmp_path: Path) -> None:
    job_path, artifacts_dir, cfg = _make_job(
        tmp_path,
        [{"start": 0.0, "end": 3.0, "text": "He said one one thing", "speaker": "spk0"}],
    )

    translate(str(job_path), cfg, translator=ExpandingTranslator("một... một"))

    script = json.loads((artifacts_dir / SCRIPT_NAME).read_text(encoding="utf-8"))
    assert script[0]["vi"] == "một"

    sidecar = json.loads(
        (artifacts_dir / "glossary.json").read_text(encoding="utf-8")
    )
    assert sidecar["hesitations"] >= 2


def test_translate_splices_source_term(tmp_path: Path) -> None:
    glossary = tmp_path / "g.tsv"
    glossary.write_text("AI assistant\ttrợ lý AI\n", encoding="utf-8")
    source = "and their AI assistant."
    job_path, artifacts_dir, cfg = _make_job(
        tmp_path,
        [{"start": 0.0, "end": 3.0, "text": source, "speaker": "spk0"}],
    )
    cfg.glossary_path = str(glossary)
    engine = MappingTranslator(
        {source: "và cỗ máy của họ.", "and their": "và", ".": "."}
    )

    translate(str(job_path), cfg, translator=engine)

    script = json.loads((artifacts_dir / SCRIPT_NAME).read_text(encoding="utf-8"))
    assert script[0]["vi"] == "và trợ lý AI."
    assert script[0]["glossary_spliced"] is True

    sidecar = json.loads(
        (artifacts_dir / "glossary.json").read_text(encoding="utf-8")
    )
    assert sidecar["spliced"] == 1
    assert sidecar["enforced"] == 1
    # the splice actually called the translator on the surrounding chunks
    assert ["and their"] in engine.calls

    report = build_report(JobManifest.load(job_path))
    assert report["glossary"]["spliced"] == 1
    assert report["glossary"]["enforced"] == 1


def test_translate_counts_enforced_when_phrase_already_present(tmp_path: Path) -> None:
    glossary = tmp_path / "g.tsv"
    glossary.write_text("AI assistant\ttrợ lý AI\n", encoding="utf-8")
    source = "the AI assistant is here"
    job_path, artifacts_dir, cfg = _make_job(
        tmp_path,
        [{"start": 0.0, "end": 3.0, "text": source, "speaker": "spk0"}],
    )
    cfg.glossary_path = str(glossary)
    engine = MappingTranslator({source: "trợ lý AI ở đây."})

    translate(str(job_path), cfg, translator=engine)

    script = json.loads((artifacts_dir / SCRIPT_NAME).read_text(encoding="utf-8"))
    assert script[0]["vi"] == "trợ lý AI ở đây."
    # no rebuild: the model already emitted the enforced phrase
    assert "glossary_spliced" not in script[0]

    sidecar = json.loads(
        (artifacts_dir / "glossary.json").read_text(encoding="utf-8")
    )
    assert sidecar["spliced"] == 0
    assert sidecar["enforced"] == 1


# ------------------------------------------------- glossary splice unit (M3-C1)
def test_find_source_terms_longest_first_non_overlapping() -> None:
    hits = find_source_terms(
        "the AI assistant and AI",
        {"AI": "trí tuệ nhân tạo", "AI assistant": "trợ lý AI"},
    )

    assert [hit[0] for hit in hits] == ["AI assistant", "AI"]
    assert hits[0][2] == 4


def test_find_source_terms_caps_at_max_terms() -> None:
    mapping = {"alpha": "A", "beta": "B", "gamma": "C"}
    hits = find_source_terms("alpha beta gamma", mapping, max_terms=2)

    assert len(hits) == 2


def test_split_source_on_term() -> None:
    assert split_source_on_term("and their AI assistant.", "AI assistant") == (
        "and their",
        ".",
    )
    assert split_source_on_term("AI assistant here", "AI assistant") == ("", "here")
    assert split_source_on_term("nothing", "AI") == ("nothing", "")


def test_splice_translation_punctuation_and_empty_parts() -> None:
    assert splice_translation("và", "trợ lý AI", ".") == "và trợ lý AI."
    assert splice_translation("", "trợ lý AI", "của họ") == "trợ lý AI của họ"
    assert splice_translation("và", "trợ lý AI", "") == "và trợ lý AI"


def test_splice_source_terms_rebuilds_when_missing() -> None:
    mapping = {"AI assistant": "trợ lý AI"}
    calls: list[str] = []

    def translate_chunk(chunk: str) -> str:
        calls.append(chunk)
        return {"and their": "và", ".": "."}.get(chunk, "?")

    new_text, rebuilt, enforced = splice_source_terms(
        "and their AI assistant.", "và cỗ máy.", mapping, translate_chunk
    )

    assert rebuilt is True
    assert enforced == 1
    assert new_text == "và trợ lý AI."
    assert calls == ["and their", "."]


def test_splice_source_terms_present_returns_unchanged() -> None:
    mapping = {"AI assistant": "trợ lý AI"}

    new_text, rebuilt, enforced = splice_source_terms(
        "the AI assistant", "trợ lý AI ở đây", mapping, lambda chunk: "X"
    )

    assert (new_text, rebuilt, enforced) == ("trợ lý AI ở đây", False, 1)


def test_splice_source_terms_multiple_terms() -> None:
    mapping = {"alpha": "A", "beta": "B"}

    new_text, rebuilt, enforced = splice_source_terms(
        "alpha and beta", "x", mapping, lambda chunk: chunk.upper()
    )

    assert rebuilt is True
    assert enforced == 2
    assert new_text == "A AND B"


def test_splice_source_terms_no_match_is_unchanged() -> None:
    assert splice_source_terms(
        "nothing here", "rien", {"AI": "trí tuệ"}, lambda chunk: "X"
    ) == ("rien", False, 0)


# ------------------------------------------- M3-C2 Vietnamese word order
def test_translate_applies_vnstyle_after_clean_hesitation(tmp_path: Path) -> None:
    job_path, artifacts_dir, cfg = _make_job(
        tmp_path,
        [{"start": 0.0, "end": 3.0, "text": "They can finish the task today", "speaker": "spk0"}],
    )

    translate(
        str(job_path),
        cfg,
        translator=ExpandingTranslator("Bạn có thể sau đó hoàn thành một một."),
    )

    script = json.loads((artifacts_dir / SCRIPT_NAME).read_text(encoding="utf-8"))
    # clean_hesitation collapses "một một" first, then vnstyle fronts the adverb
    assert script[0]["vi"] == "Sau đó, bạn có thể hoàn thành một."

    sidecar = json.loads(
        (artifacts_dir / "glossary.json").read_text(encoding="utf-8")
    )
    assert sidecar["hesitations"] >= 1
    assert sidecar["style_fixes"] == 1

    report = build_report(JobManifest.load(job_path))
    assert report["cleanup"]["style_fixes"] == 1


def test_translate_counts_style_fixes_when_rule_matches(tmp_path: Path) -> None:
    job_path, artifacts_dir, cfg = _make_job(
        tmp_path,
        [{"start": 0.0, "end": 3.0, "text": "They then finish it", "speaker": "spk0"}],
    )

    translate(
        str(job_path), cfg, translator=ExpandingTranslator("Sau đó, sau đó, bạn làm.")
    )

    script = json.loads((artifacts_dir / SCRIPT_NAME).read_text(encoding="utf-8"))
    assert script[0]["vi"] == "Sau đó, bạn làm."
    sidecar = json.loads(
        (artifacts_dir / "glossary.json").read_text(encoding="utf-8")
    )
    assert sidecar["style_fixes"] == 1


def test_vnstyle_not_applied_to_passthrough(tmp_path: Path) -> None:
    source = "Bạn có thể sau đó hoàn thành."
    job_path, artifacts_dir, cfg = _make_job(
        tmp_path, [{"start": 0.0, "end": 3.0, "text": source, "speaker": "spk0"}]
    )

    translate(str(job_path), cfg, translator=ExpandingTranslator("X"))

    script = json.loads((artifacts_dir / SCRIPT_NAME).read_text(encoding="utf-8"))
    assert script[0].get("passthrough") is True
    assert script[0]["vi"] == source
    sidecar = json.loads(
        (artifacts_dir / "glossary.json").read_text(encoding="utf-8")
    )
    assert sidecar["style_fixes"] == 0


# ----------------------------------------------- M4-T3 translation QC battery
TRANSLATION_QC_NAME = "translation_qc.json"


class _EntityDroppingTranslator(BaseTranslator):
    """Translates "we use Vigib daily" but drops the mid-sentence entity.

    A mapping translator is not enough here: the battery's entity correction
    re-translates the surrounding chunks, so the echo fallback must be safe.
    """

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def translate(self, texts, src_lang, tgt_lang):
        self.calls.append(list(texts))
        out: list[str] = []
        for text in texts:
            lowered = text.lower()
            if "vigib" in lowered:
                # drop the entity on the first pass, keep it when re-translating
                # the spliced chunks (which no longer contain the entity itself)
                out.append("chúng tôi dùng hằng ngày")
            else:
                out.append(text)  # echo, so the splice can rebuild cleanly
        return out


def test_translate_runs_qc_and_splices_dropped_entity(tmp_path: Path) -> None:
    job_path, artifacts_dir, cfg = _make_job(
        tmp_path,
        [{"start": 0.0, "end": 3.0, "text": "we use Vigib daily", "speaker": "spk0"}],
    )
    assert cfg.qc_translation is True
    engine = _EntityDroppingTranslator()

    result = translate(str(job_path), cfg, translator=engine)

    script = json.loads((artifacts_dir / SCRIPT_NAME).read_text(encoding="utf-8"))
    # the battery spliced the dropped entity back into the written script
    assert "Vigib" in script[0]["vi"]

    # the artifact + manifest verdict are recorded
    qc_path = artifacts_dir / TRANSLATION_QC_NAME
    assert qc_path.exists()
    artifact = json.loads(qc_path.read_text(encoding="utf-8"))
    assert {"verdict", "counts", "segments"} <= set(artifact)
    assert artifact["segments"][0]["action"] == "splice-entity"
    assert artifact["segments"][0]["resolved"] is True
    assert artifact["verdict"] in {"pass", "review", "fail"}
    assert result.translation_verdict == artifact["verdict"]
    assert result.paths["translation_qc"] == TRANSLATION_QC_NAME
    assert JobManifest.load(job_path).translation_verdict in {"pass", "review", "fail"}


def test_translate_qc_disabled_skips_everything(tmp_path: Path) -> None:
    job_path, artifacts_dir, cfg = _make_job(
        tmp_path,
        [{"start": 0.0, "end": 3.0, "text": "we use Vigib daily", "speaker": "spk0"}],
    )
    cfg.qc_translation = False
    engine = _EntityDroppingTranslator()

    result = translate(str(job_path), cfg, translator=engine)

    assert not (artifacts_dir / TRANSLATION_QC_NAME).exists()
    assert result.translation_verdict == ""
    assert "translation_qc" not in result.paths
    # no correction ran: the entity stays dropped
    script = json.loads((artifacts_dir / SCRIPT_NAME).read_text(encoding="utf-8"))
    assert "Vigib" not in script[0]["vi"]


def test_cli_no_translation_qc_flag_reaches_config() -> None:
    from vidub.cli import _build_cfg, build_parser

    parser = build_parser()
    assert _build_cfg(parser.parse_args(["run", "x.mp4"])).qc_translation is True
    assert (
        _build_cfg(
            parser.parse_args(["run", "x.mp4", "--no-translation-qc"])
        ).qc_translation
        is False
    )


# ------------------------------------- M4-T5 contextual engine dispatch
def test_build_translator_ollama_uses_cfg() -> None:
    from vidub.engines.mt.ollama_mt import OllamaTranslator
    from vidub.stages.translate import build_translator

    manifest = JobManifest(mt_engine="ollama")
    cfg = JobConfig(mt_engine="ollama", ollama_model="qwen2.5:7b")

    engine = build_translator(manifest, cfg)

    assert isinstance(engine, OllamaTranslator)
    assert engine.model == "qwen2.5:7b"


def test_translate_prefers_contextual_output(tmp_path: Path) -> None:
    """The main batch goes through ``translate_contextual`` when overridden."""

    class ContextualFake(BaseTranslator):
        def translate(self, texts, src_lang, tgt_lang):
            return [text[::-1] for text in texts]

        def translate_contextual(self, texts, src_lang, tgt_lang):
            return ["CTX " + text for text in texts]

    job_path, artifacts_dir, cfg = _make_job(tmp_path, SEGMENTS)
    cfg.qc_translation = False  # isolate the engine dispatch from the QC battery

    translate(str(job_path), cfg, translator=ContextualFake())

    script = json.loads((artifacts_dir / SCRIPT_NAME).read_text(encoding="utf-8"))
    assert script[0]["vi"] == "CTX Hello world"

    # envit5 (and any engine without an override) still uses the default
    # translate_contextual -> translate delegation (existing tests prove the
    # byte-identical reverse output through FakeTranslator).


# --------------------------------------------- TU-1 sentence-level units
# The real screenshot fragment run (indices 1-3 of the reference job cache):
# the three rows must be translated as ONE sentence and split back onto 3 rows.
SCREENSHOT_FRAGMENTS = [
    {
        "start": 5.8,
        "end": 11.51,
        "text": "Meanwhile, these open-source models are sitting",
        "speaker": "spk0",
    },
    {
        "start": 11.51,
        "end": 17.51,
        "text": (
            "there like some cruel joke. Sure, you can download them, you can "
            "even name them, but actually running"
        ),
        "speaker": "spk0",
    },
    {
        "start": 17.51,
        "end": 23.21,
        "text": "them, that's going to cost you more than a small house.",
        "speaker": "spk0",
    },
]

# A diacritic-bearing Vietnamese sentence (so low_confidence never fires and the
# exact row-key assertion holds). No time adverbial / repetition, so vnstyle and
# clean_hesitation leave it untouched.
VI_TRANSLATION = "Xin chào các bạn trẻ đây là một câu tiếng Việt rất hay"


def _nonspace(text: str) -> str:
    return "".join(text.split())


def test_sentence_units_batches_the_fragment_run(tmp_path: Path) -> None:
    job_path, _artifacts_dir, cfg = _make_job(
        tmp_path, SCREENSHOT_FRAGMENTS, sentence_units=True
    )
    cfg.qc_translation = False  # isolate the batch from the QC retries
    engine = RecordingTranslator()

    translate(str(job_path), cfg, translator=engine)

    # friend/fragment/sentence became one unit per complete sentence: row 1 is
    # split at "joke." so the second unit is [fragment..., completing sentence]
    assert len(engine.calls) == 1
    assert len(engine.calls[0]) == 2
    assert any("actually running them" in text for text in engine.calls[0])


def test_sentence_units_off_sends_the_original_fragments(tmp_path: Path) -> None:
    job_path, _artifacts_dir, cfg = _make_job(
        tmp_path, SCREENSHOT_FRAGMENTS, sentence_units=False
    )
    cfg.qc_translation = False
    engine = RecordingTranslator()

    translate(str(job_path), cfg, translator=engine)

    assert engine.calls[0] == [seg["text"] for seg in SCREENSHOT_FRAGMENTS]


def test_sentence_units_split_back_contract(tmp_path: Path) -> None:
    job_path, artifacts_dir, cfg = _make_job(
        tmp_path, SCREENSHOT_FRAGMENTS, sentence_units=True
    )
    cfg.qc_translation = False

    translate(str(job_path), cfg, translator=ExpandingTranslator(VI_TRANSLATION))

    script = json.loads((artifacts_dir / SCRIPT_NAME).read_text(encoding="utf-8"))
    # SA-1-T2: the row unit is now a SENTENCE (row 1 splits at "joke." -> 4 rows).
    assert len(script) == 4
    allowed = {
        "start", "end", "speaker", "en", "vi", "spoken_vi", "subtitle_vi", "truncated",
        "passthrough", "glossary_spliced", "low_confidence",
    }
    for row in script:
        assert {"start", "end", "speaker", "en", "vi", "truncated"} <= set(row)
        assert set(row) <= allowed
        assert row["vi"].strip()
    # the unit translation was distributed across every member sentence row
    joined = " ".join(row["vi"] for row in script)
    assert "tiếng Việt" in joined


def test_sentence_units_keep_digit_token_once(tmp_path: Path) -> None:
    segments = [
        {"start": 0.0, "end": 1.0, "text": "The price is 300", "speaker": "spk0"},
        {
            "start": 1.0,
            "end": 2.0,
            "text": "dollars for the new GPU",
            "speaker": "spk0",
        },
    ]
    job_path, artifacts_dir, cfg = _make_job(tmp_path, segments, sentence_units=True)
    cfg.qc_translation = False

    translate(
        str(job_path),
        cfg,
        translator=ExpandingTranslator("Giá là 300 đô la cho GPU mới"),
    )

    script = json.loads((artifacts_dir / SCRIPT_NAME).read_text(encoding="utf-8"))
    joined = " ".join(row["vi"] for row in script)
    assert joined.count("300") == 1


def test_sentence_units_passthrough_row_is_isolated(tmp_path: Path) -> None:
    segments = [
        {"start": 0.0, "end": 1.0, "text": "fragment one two", "speaker": "spk0"},
        {
            "start": 1.0,
            "end": 2.0,
            "text": "thì tuấn triều tư trong văn thì trước triều tiên.",
            "speaker": "spk0",
        },
        {"start": 2.0, "end": 3.0, "text": "fragment three four", "speaker": "spk0"},
    ]
    job_path, artifacts_dir, cfg = _make_job(tmp_path, segments, sentence_units=True)
    cfg.qc_translation = False
    engine = RecordingTranslator()

    translate(str(job_path), cfg, translator=engine)

    # the passthrough row is its own unit and is never fed to the engine
    assert engine.calls[0] == ["fragment one two", "fragment three four"]
    assert all(
        segments[1]["text"] not in call for call in engine.calls
    )
    script = json.loads((artifacts_dir / SCRIPT_NAME).read_text(encoding="utf-8"))
    assert script[1]["passthrough"] is True
    assert script[1]["vi"] == segments[1]["text"]


def test_sentence_units_passthrough_boundary_is_a_speaker_cut(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A passthrough-isolated boundary must be a speaker cut, never mid-sentence.

    The stage groups *adapted* rows (a passthrough row gets a unique speaker), so
    it must feed those SAME rows to ``unit_cut_stats``: the synthetic speaker
    change is what classifies the boundary as a speaker cut.  Feeding the
    original rows hides it and reports ``mid_sentence_cuts > 0``.
    """
    segments = [
        {"start": 0.0, "end": 1.0, "text": "fragment one two", "speaker": "spk0"},
        {"start": 1.0, "end": 2.0, "text": "đoạn tiếng Việt", "speaker": "spk0"},
        {"start": 2.0, "end": 3.0, "text": "fragment three four", "speaker": "spk0"},
    ]
    job_path, _artifacts_dir, cfg = _make_job(tmp_path, segments, sentence_units=True)
    assert cfg.qc_translation is True  # the unit report is written on this path

    from vidub.stages import translate as translate_module

    captured: dict = {}
    real_cut_stats = translate_module.unit_cut_stats

    def _spy(units, rows):
        stats = real_cut_stats(units, rows)
        captured["rows"] = rows
        captured["stats"] = stats
        return stats

    monkeypatch.setattr(translate_module, "unit_cut_stats", _spy)

    with caplog.at_level(logging.INFO, logger="vidub.stages.translate"):
        translate(str(job_path), cfg, translator=RecordingTranslator())

    # the unit path fed unit_cut_stats the ADAPTED rows it grouped with
    assert captured, "the unit path never called unit_cut_stats"
    assert captured["rows"][1]["speaker"] == "__passthrough_1__"
    stats = captured["stats"]
    assert stats["mid_sentence_cuts"] == 0
    assert stats["speaker_cuts"] == 2  # both passthrough-isolated boundaries
    # the stage log reports the same invariant
    assert any(
        "mid-sentence cuts: 0" in record.getMessage() for record in caplog.records
    )

    # regression guard: the ORIGINAL rows alone would report two mid cuts
    src_texts = [str(seg.get("text", "")) for seg in segments]
    units, adapted = _unit_rows(segments, src_texts, {1})
    assert unit_cut_stats(units, adapted)["mid_sentence_cuts"] == 0
    assert unit_cut_stats(units, segments)["mid_sentence_cuts"] == 2


def test_sentence_units_artifact_block_when_on(tmp_path: Path) -> None:
    job_path, artifacts_dir, cfg = _make_job(tmp_path, SEGMENTS, sentence_units=True)

    translate(str(job_path), cfg, translator=ExpandingTranslator(VI_TRANSLATION))

    artifact = json.loads(
        (artifacts_dir / TRANSLATION_QC_NAME).read_text(encoding="utf-8")
    )
    assert artifact["units"]["mode"] == "sentence"
    assert artifact["units"]["segments"] == 2
    assert artifact["units"]["units"] == 1
    assert artifact["units"]["merged_segments"] == 1
    assert artifact["units"]["max_unit_words"] > 0
    assert artifact["units"]["max_unit_seconds"] > 0
    # run_battery's own keys are intact
    assert {"verdict", "counts", "segments"} <= set(artifact)


def test_sentence_units_artifact_absent_when_off(tmp_path: Path) -> None:
    job_path, artifacts_dir, cfg = _make_job(tmp_path, SEGMENTS, sentence_units=False)

    translate(str(job_path), cfg, translator=ExpandingTranslator(VI_TRANSLATION))

    artifact = json.loads(
        (artifacts_dir / TRANSLATION_QC_NAME).read_text(encoding="utf-8")
    )
    # the off path omits the block, so today's artifact is byte-identical
    assert "units" not in artifact
    assert {"verdict", "counts", "segments"} <= set(artifact)


def test_sentence_units_default_and_cli_flags() -> None:
    from vidub.cli import _build_cfg, build_parser

    assert JobConfig().sentence_units is True

    parser = build_parser()
    assert _build_cfg(parser.parse_args(["run", "x.mp4"])).sentence_units is True
    assert (
        _build_cfg(
            parser.parse_args(["run", "x.mp4", "--no-sentence-units"])
        ).sentence_units
        is False
    )
    assert (
        _build_cfg(
            parser.parse_args(["run", "x.mp4", "--sentence-units"])
        ).sentence_units
        is True
    )


# --------------------------- TU-1-T5: ON-path coverage of the shared passes
def test_sentence_units_on_path_glossary_splices_split_back_row(
    tmp_path: Path,
) -> None:
    """The source-side glossary splice acts on the UNIT text, not a fragment."""
    glossary = tmp_path / "g.tsv"
    glossary.write_text("AI assistant\ttrợ lý AI\n", encoding="utf-8")
    segments = [
        {
            "start": 0.0,
            "end": 3.0,
            "text": "and their AI assistant",
            "speaker": "spk0",
        },
        {"start": 3.0, "end": 6.0, "text": "is important.", "speaker": "spk0"},
    ]
    job_path, artifacts_dir, cfg = _make_job(tmp_path, segments, sentence_units=True)
    cfg.glossary_path = str(glossary)
    cfg.qc_translation = False  # isolate the stage's glossary pass
    # the merged fragment run is one unit, translated without the enforced phrase
    merged = "and their AI assistant is important."
    engine = MappingTranslator(
        {
            merged: "cỗ máy của họ rất quan trọng.",
            "and their": "và",
            "is important.": "rất quan trọng.",
        }
    )

    translate(str(job_path), cfg, translator=engine)

    script = json.loads((artifacts_dir / SCRIPT_NAME).read_text(encoding="utf-8"))
    # the unit text is spliced, then split onto its two member sentence rows
    joined = " ".join(row["vi"] for row in script)
    assert "trợ lý AI" in joined
    assert all(row.get("glossary_spliced") is True for row in script)

    sidecar = json.loads(
        (artifacts_dir / "glossary.json").read_text(encoding="utf-8")
    )
    assert sidecar["spliced"] == 1
    assert sidecar["enforced"] == 1
    # the splice re-translated both surrounding chunks of the whole UNIT
    assert ["and their"] in engine.calls
    assert ["is important."] in engine.calls


def test_sentence_units_on_path_truncation_guard(tmp_path: Path) -> None:
    """The length guard truncates the UNIT; every member row inherits it."""
    segments = [
        {
            "start": 0.0,
            "end": 3.0,
            "text": "Internationalization globalization",
            "speaker": "spk0",
        },
        {"start": 3.0, "end": 6.0, "text": "x y z", "speaker": "spk0"},
    ]
    job_path, artifacts_dir, cfg = _make_job(tmp_path, segments, sentence_units=True)
    cfg.qc_translation = False

    translate(
        str(job_path),
        cfg,
        translator=ExpandingTranslator(
            "Một hai ba bốn năm sáu bảy tám chín mười mười một mười hai"
        ),
    )

    script = json.loads((artifacts_dir / SCRIPT_NAME).read_text(encoding="utf-8"))
    # the 5-word unit (limit int(2.2 * 5) == 11) is cut at the limit and flagged
    assert len(script) == 2
    assert all(row["truncated"] is True for row in script)
    assert "mười một" in " ".join(row["vi"] for row in script)


def test_sentence_units_on_path_low_confidence_flag(tmp_path: Path) -> None:
    """``low_confidence`` fires on a split-back row exactly as on the off path."""
    segments = [
        {
            "start": 0.0,
            "end": 3.0,
            "text": "The weather is sunny today",
            "speaker": "spk0",
        },
        {"start": 3.0, "end": 6.0, "text": "and tomorrow too", "speaker": "spk0"},
    ]
    job_path, artifacts_dir, cfg = _make_job(tmp_path, segments, sentence_units=True)
    cfg.qc_translation = False

    translate(
        str(job_path), cfg, translator=ExpandingTranslator("hello there friend")
    )

    script = json.loads((artifacts_dir / SCRIPT_NAME).read_text(encoding="utf-8"))
    # a >= 3-word source with a diacritic-free split-back piece is flagged
    assert script[0]["low_confidence"] is True


# ------------------------------------------- SA-1-T2 sentence rows / unit map
def _words(count: int, start: float, step: float = 1.0) -> list[dict]:
    """``count`` synthetic word-timing entries starting at ``start``."""
    return [
        {
            "start": start + index * step,
            "end": start + (index + 1) * step,
            "word": f"w{index}",
        }
        for index in range(count)
    ]


def test_sentence_units_split_asr_row_on_word_edges(tmp_path: Path) -> None:
    """(a) a two-sentence ASR row becomes two sentence rows on word edges."""
    segments = [
        {
            "start": 0.0,
            "end": 5.0,
            "speaker": "spk0",
            "text": "Hello there. How are you",
            "words": _words(5, 0.0),
        }
    ]
    job_path, artifacts_dir, cfg = _make_job(tmp_path, segments, sentence_units=True)
    cfg.qc_translation = False

    translate(str(job_path), cfg, translator=ExpandingTranslator(VI_TRANSLATION))

    script = json.loads((artifacts_dir / SCRIPT_NAME).read_text(encoding="utf-8"))
    assert [row["en"] for row in script] == ["Hello there.", "How are you"]
    # each sentence span is locked to the real ASR word edges
    assert script[0]["start"] == 0.0 and script[0]["end"] == 2.0
    assert script[1]["start"] == 2.0 and script[1]["end"] == 5.0


def test_sentence_units_user_shape_one_unit_and_joint_translation(
    tmp_path: Path,
) -> None:
    """(b) fragment + completing sentence are ONE unit; units.json maps it."""
    segments = [
        {
            "start": 0.0,
            "end": 11.0,
            "speaker": "spk0",
            "text": (
                "generalizable and adaptable output. And large language models "
                "are instances of"
            ),
            "words": _words(11, 0.0),
        },
        {
            "start": 11.0,
            "end": 23.0,
            "speaker": "spk0",
            "text": (
                "foundation models applied specifically to text and text like "
                "things. I'm talking"
            ),
            "words": _words(12, 11.0),
        },
    ]
    joint = (
        "And large language models are instances of foundation models applied "
        "specifically to text and text like things."
    )
    job_path, artifacts_dir, cfg = _make_job(tmp_path, segments, sentence_units=True)
    cfg.qc_translation = False
    engine = MappingTranslator(
        {
            joint: "một bản dịch chung cho cả hai mảnh câu.",
            "generalizable and adaptable output.": "đầu ra có thể khái quát.",
            "I'm talking": "tôi đang nói",
        }
    )

    translate(str(job_path), cfg, translator=engine)

    doc = json.loads((artifacts_dir / UNITS_NAME).read_text(encoding="utf-8"))
    assert set(doc) == {"version", "text_contract_version", "mode", "units"}
    assert doc["version"] == 1 and doc["mode"] == "sentence"
    expected_keys = {
        "index",
        "indices",
        "start",
        "end",
        "speaker",
        "source",
        "vi",
        "spoken_vi",
        "subtitle_vi",
        "words_in",
        "coverage",
        "guard",
        "battery",
        "open",
        # ISO-1 isometric budgeting is enabled by this fixture, so each unit
        # records its window budget and the bounded-repair outcome.
        "budget_chars",
        "chars",
        "over_budget",
        "isometric",
        "chars_after",
        "projected_stretch_before",
        "projected_stretch_after",
        # ISO-1-T3-C1: the adaptive floor and the persisted rejection reason.
        "coverage_floor",
        "short_floor_applied",
        "isometric_reason",
        "isometric_candidates",
        "isometric_variant",
    }
    for entry in doc["units"]:
        assert set(entry) == expected_keys
    holders = [unit for unit in doc["units"] if set(unit["indices"]) >= {1, 2}]
    assert len(holders) == 1
    unit = holders[0]
    assert unit["indices"] == [1, 2]
    assert unit["source"] == joint
    assert unit["vi"] == "một bản dịch chung cho cả hai mảnh câu."
    # the joint span runs from the fragment's first word edge to the sentence end
    assert unit["start"] == 4.0 and unit["end"] == 21.0

    # the member rows' joined vi reproduces the unit's vi (up to whitespace)
    script = json.loads((artifacts_dir / SCRIPT_NAME).read_text(encoding="utf-8"))
    joined = " ".join(script[index]["vi"] for index in unit["indices"])
    assert _nonspace(joined) == _nonspace(unit["vi"])


class _ResamplingTranslator(BaseTranslator):
    """Returns a too-short translation that ``sample_candidates`` repairs."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def translate(self, texts, src_lang, tgt_lang):
        self.calls.append(list(texts))
        return ["ngắn" for _ in texts]

    def sample_candidates(self, text, src_lang, tgt_lang, k):
        count = len(text.split())
        return [" ".join(f"từ{index}" for index in range(count))]


def test_sentence_unit_guard_resamples_a_dropping_engine(tmp_path: Path) -> None:
    """(c) a too-short unit translation is repaired by resampling."""
    segments = [
        {
            "start": 0.0,
            "end": 3.0,
            "speaker": "spk0",
            "text": "The weather is sunny today and tomorrow morning",
        }
    ]
    job_path, artifacts_dir, cfg = _make_job(tmp_path, segments, sentence_units=True)
    cfg.qc_translation = False

    translate(str(job_path), cfg, translator=_ResamplingTranslator())

    doc = json.loads((artifacts_dir / UNITS_NAME).read_text(encoding="utf-8"))
    unit = doc["units"][0]
    assert unit["guard"] == "resampled"
    # the complete candidate (one word per source word) is what the unit holds
    assert len(unit["vi"].split()) == len(segments[0]["text"].split())


def test_sentence_unit_guard_flags_an_unrepairable_drop(tmp_path: Path) -> None:
    """(c) a unit that stays omission-risk is flagged and never labelled ok."""
    segments = [
        {
            "start": 0.0,
            "end": 3.0,
            "speaker": "spk0",
            "text": "The weather is sunny today and tomorrow morning",
        }
    ]
    job_path, artifacts_dir, cfg = _make_job(tmp_path, segments, sentence_units=True)

    translate(str(job_path), cfg, translator=ExpandingTranslator("một hai ba bốn"))

    doc = json.loads((artifacts_dir / UNITS_NAME).read_text(encoding="utf-8"))
    assert doc["units"][0]["guard"] == "flagged"

    artifact = json.loads(
        (artifacts_dir / TRANSLATION_QC_NAME).read_text(encoding="utf-8")
    )
    detail = artifact["units"]["detail"][0]
    assert detail["action"] == "flagged"
    assert "omission-risk" in detail["flags"]
    assert artifact["units"]["flags"]["guard_flagged"] == 1
    labels = {entry["index"]: entry["label"] for entry in artifact["segments"]}
    assert labels[0] != "ok"


# ------------------------------------- Q-1-T1 neighbour contamination
# The User's real unit 24 (source + contaminated vi) and unit 23's vi, reused as
# the bleed context.
BLEED_SOURCE = (
    "Sure, you could do some fancy offloading where you constantly swap "
    "weights in and out of memory."
)
BLEED_VI = (
    "Thật không may cho bạn đấy. Nhưng việc đó cũng giống như cố gắng nấu bữa "
    "tối bằng cách chạy đi mua từng nguyên liệu riêng lẻ vậy."
)
TOUGH_VI = "Thật không may cho bạn đấy."


class _ContextBleedTranslator(BaseTranslator):
    """``translate_contextual`` reproduces the previous unit; ``translate`` is direct.

    Models the hunyuan ``_template_contextual`` failure: the batch output for a
    unit is its neighbour's text.  ``translate`` is the direct (non-contextual)
    template the guard's ``decontext`` repair uses.
    """

    def __init__(self, contextual: list[str], direct: dict[str, str]) -> None:
        self.contextual = list(contextual)
        self.direct = dict(direct)
        self.calls: list[list[str]] = []

    def translate_contextual(self, texts, src_lang, tgt_lang):
        return list(self.contextual)

    def translate(self, texts, src_lang, tgt_lang):
        self.calls.append(list(texts))
        return [self.direct.get(text, text) for text in texts]


def _bleed_segments() -> list[dict]:
    return [
        {"start": 0.0, "end": 3.0, "speaker": "spk0", "text": "Tough luck buddy."},
        {"start": 3.0, "end": 6.0, "speaker": "spk0", "text": BLEED_SOURCE},
    ]


def test_sentence_units_decontext_repairs_a_context_bleed(tmp_path: Path) -> None:
    """A unit that reproduced its neighbour is retried with the direct template."""
    clean_vi = "Bạn có thể hoán đổi trọng số ra vào bộ nhớ khi cần thiết."
    job_path, artifacts_dir, cfg = _make_job(
        tmp_path, _bleed_segments(), sentence_units=True
    )
    engine = _ContextBleedTranslator(
        contextual=[TOUGH_VI, BLEED_VI],
        direct={BLEED_SOURCE: clean_vi},
    )

    translate(str(job_path), cfg, translator=engine)

    artifact = json.loads(
        (artifacts_dir / TRANSLATION_QC_NAME).read_text(encoding="utf-8")
    )
    assert artifact["units"]["flags"]["contamination"] >= 1
    bleed = [
        entry
        for entry in artifact["units"]["detail"]
        if "context-bleed" in entry["flags"]
    ]
    assert bleed
    entry = bleed[0]
    assert entry["action"] == "decontext"
    assert entry["overlap_prev"] == 1.0
    assert entry["overlap_prev_after"] < 0.5
    # the clean direct candidate is what the unit now holds
    doc = json.loads((artifacts_dir / UNITS_NAME).read_text(encoding="utf-8"))
    assert any(unit["vi"] == clean_vi for unit in doc["units"])


def test_sentence_units_flag_an_unrepairable_context_bleed(tmp_path: Path) -> None:
    """A bleed the direct template cannot remove stays ``flagged`` and never ``ok``."""
    job_path, artifacts_dir, cfg = _make_job(
        tmp_path, _bleed_segments(), sentence_units=True
    )
    engine = _ContextBleedTranslator(
        contextual=[TOUGH_VI, BLEED_VI],
        # the direct output still reproduces the previous unit's content
        direct={BLEED_SOURCE: f"{TOUGH_VI} Và đây là nội dung của đơn vị này."},
    )

    translate(str(job_path), cfg, translator=engine)

    artifact = json.loads(
        (artifacts_dir / TRANSLATION_QC_NAME).read_text(encoding="utf-8")
    )
    assert artifact["units"]["flags"]["contamination"] >= 1
    bleed = [
        entry
        for entry in artifact["units"]["detail"]
        if "context-bleed" in entry["flags"]
    ]
    assert bleed
    assert bleed[0]["action"] == "flagged"
    labels = {entry["index"]: entry["label"] for entry in artifact["segments"]}
    assert labels[bleed[0]["index"]] != "ok"


def test_sentence_units_battery_scores_units_with_members(tmp_path: Path) -> None:
    """(d) one translation_qc segments entry per unit, each with members."""
    job_path, artifacts_dir, cfg = _make_job(
        tmp_path, SCREENSHOT_FRAGMENTS, sentence_units=True
    )

    translate(str(job_path), cfg, translator=ExpandingTranslator(VI_TRANSLATION))

    artifact = json.loads(
        (artifacts_dir / TRANSLATION_QC_NAME).read_text(encoding="utf-8")
    )
    units_block = artifact["units"]
    assert {
        "mode",
        "segments",
        "units",
        "merged_segments",
        "max_unit_words",
        "max_unit_seconds",
        "detail",
        "flags",
    } <= set(units_block)
    assert units_block["mode"] == "sentence"
    assert units_block["units"] == 2
    assert len(artifact["segments"]) == 2  # ONE entry per unit
    for entry, detail in zip(artifact["segments"], units_block["detail"]):
        assert entry["index"] == detail["index"]
        assert entry["members"] == detail["indices"]
        assert len(entry["members"]) >= 1
    assert {
        "guard_flagged",
        "omission",
        "sentence_loss",
        "addition",
        "oversized",
    } <= set(units_block["flags"])
    # a unit the guard could not repair can never be labelled ok
    flagged = [d for d in units_block["detail"] if d["action"] == "flagged"]
    assert flagged
    labels = {entry["index"]: entry["label"] for entry in artifact["segments"]}
    for detail in flagged:
        assert labels[detail["index"]] != "ok"


def test_sentence_units_member_rows_are_split_pieces(tmp_path: Path) -> None:
    """(e) every member row's vi is its split_translation piece (post-battery)."""
    job_path, artifacts_dir, cfg = _make_job(
        tmp_path, SCREENSHOT_FRAGMENTS, sentence_units=True
    )

    translate(str(job_path), cfg, translator=ExpandingTranslator(VI_TRANSLATION))

    script = json.loads((artifacts_dir / SCRIPT_NAME).read_text(encoding="utf-8"))
    doc = json.loads((artifacts_dir / UNITS_NAME).read_text(encoding="utf-8"))
    assert doc["units"]
    for unit in doc["units"]:
        member_texts = [script[index]["en"] for index in unit["indices"]]
        counts = [len("".join(text.split())) for text in member_texts]
        total = sum(counts)
        pieces = split_translation(unit["vi"], [count / total for count in counts])
        for position, index in enumerate(unit["indices"]):
            assert script[index]["vi"] == pieces[position]


def test_sentence_units_off_writes_no_units_json(tmp_path: Path) -> None:
    """(f) the off path writes no units.json and keeps today's shapes."""
    job_path, artifacts_dir, cfg = _make_job(tmp_path, SEGMENTS, sentence_units=False)

    translate(str(job_path), cfg, translator=ExpandingTranslator(VI_TRANSLATION))

    assert not (artifacts_dir / UNITS_NAME).exists()
    script = json.loads((artifacts_dir / SCRIPT_NAME).read_text(encoding="utf-8"))
    assert len(script) == len(SEGMENTS)
    for row in script:
        assert set(row) == {
            "start", "end", "speaker", "en", "vi", "spoken_vi", "subtitle_vi", "truncated"
        }
    artifact = json.loads(
        (artifacts_dir / TRANSLATION_QC_NAME).read_text(encoding="utf-8")
    )
    assert "units" not in artifact
    assert "detail" not in artifact


def test_mt_signature_encodes_sentence_unit_pipeline_version() -> None:
    """(g) mt_signature reports units=2 (on) vs units=0 (off), and they differ."""
    from vidub.pipeline import mt_signature

    on = mt_signature(JobConfig(sentence_units=True))
    off = mt_signature(JobConfig(sentence_units=False))
    assert "units=2" in on
    assert "units=0" in off
    assert on != off


# ------------------------------------- M5-T2 refine default on for hunyuan
def _set_mt_engine(job_path: Path, engine: str) -> None:
    """Stamp the engine name onto an existing test manifest (for the gate)."""
    manifest = JobManifest.load(job_path)
    manifest.mt_engine = engine
    manifest.save(job_path)


def test_refine_is_default_on_for_hunyuan_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Auto default: refine is built for hunyuan and never for envit5."""
    import vidub.stages.translate as translate_mod

    built: list[str] = []

    def fake_build(cfg, glossary, manifest):
        built.append(manifest.mt_engine or "")
        return lambda rows, entries: (
            rows,
            entries,
            {"attempted": 0, "improved": 0, "fusion_used": False, "outcomes": []},
        )

    monkeypatch.setattr(translate_mod, "_build_refiner", fake_build)

    # envit5 (the default engine): the refiner is never built
    job_path, _art, cfg = _make_job(tmp_path / "envit5", SEGMENTS, sentence_units=True)
    translate(str(job_path), cfg, translator=ExpandingTranslator(VI_TRANSLATION))
    assert built == []

    # hunyuan: built by default (no flag, cfg.hunyuan_refine is None = auto)
    job_path, _art, cfg = _make_job(
        tmp_path / "hunyuan", SEGMENTS, sentence_units=True
    )
    _set_mt_engine(job_path, "hunyuan")
    translate(str(job_path), cfg, translator=ExpandingTranslator(VI_TRANSLATION))
    assert built == ["hunyuan"]

    # hunyuan + --no-hunyuan-refine: the explicit opt-out wins
    job_path, _art, cfg = _make_job(tmp_path / "off", SEGMENTS, sentence_units=True)
    _set_mt_engine(job_path, "hunyuan")
    cfg.hunyuan_refine = False
    translate(str(job_path), cfg, translator=ExpandingTranslator(VI_TRANSLATION))
    assert built == ["hunyuan"]


def test_refine_records_counts_budget_and_detail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hunyuan refine pass records counts/budget and each unit's action."""
    import vidub.stages.translate as translate_mod

    def fake_build(cfg, glossary, manifest):
        assert cfg.hunyuan_refine_max_units == 40

        def refiner(rows, entries):
            outcomes = []
            for entry in entries:
                if entry["label"] == "ok":
                    continue
                outcomes.append(
                    {
                        "index": entry["index"],
                        "action": "refine-best",
                        "before": entry["label"],
                        "after": "ok",
                    }
                )
                entry["label"] = "ok"
                entry["signals"] = {}
            return (
                rows,
                entries,
                {
                    "attempted": len(outcomes),
                    "improved": len(outcomes),
                    "fusion_used": False,
                    "max_units": 40,
                    "skipped_budget": 0,
                    "outcomes": outcomes,
                },
            )

        return refiner

    monkeypatch.setattr(translate_mod, "_build_refiner", fake_build)

    job_path, artifacts_dir, cfg = _make_job(tmp_path, SEGMENTS, sentence_units=True)
    _set_mt_engine(job_path, "hunyuan")

    # a diacritic-free unit -> the battery labels it ``unnatural`` (not ok)
    translate(str(job_path), cfg, translator=ExpandingTranslator("hello there friend"))

    artifact = json.loads(
        (artifacts_dir / TRANSLATION_QC_NAME).read_text(encoding="utf-8")
    )
    refine = artifact["refine"]
    assert refine["max_units"] == 40
    assert refine["skipped_budget"] == 0
    assert refine["improved"] >= 1
    assert artifact["units"]["detail"]
    assert all("refine" in detail for detail in artifact["units"]["detail"])
    assert any(
        detail["refine"] == "refine-best" for detail in artifact["units"]["detail"]
    )


# =========================================== ISO-1 isometric budgeting (T3)
# A 9-word, one-sentence source; an 18-word "long" translation (55 non-space
# chars) that overflows a 1.5 s window (budget 27), and a 7-word "short"
# candidate (22 chars) that fits it. The candidate shares no content tokens with
# the long text, so the containment guard cannot flag it.
ISO_SOURCE = "The quick brown fox jumps over the lazy dog"
ISO_LONG = "Đây là một bản dịch tiếng Việt rất là dài dòng và lan man cho đơn vị này"
ISO_SHORT = "Con cáo nâu nhảy qua chú chó"


def _nonspace_len(text: str) -> int:
    return len("".join(text.split()))


class _BudgetedTranslator(BaseTranslator):
    """Contextual engine that honours (or ignores) a per-item budget.

    ``translate_contextual`` always returns ``long_value`` (the over-budget
    initial batch). ``translate`` returns ``source_value`` **without** a budget
    (so the battery's round-trip back-translation stays clean and never rewrites
    the row) and ``short_value`` **with** a budget (the repair candidate).
    """

    def __init__(self, long_value: str, short_value: str, source_value: str = "") -> None:
        self.long_value = long_value
        self.short_value = short_value
        self.source_value = source_value
        self.contextual_calls: list[list[int] | None] = []
        self.translate_calls: list[list[int] | None] = []

    def translate(self, texts, src_lang, tgt_lang, *, budgets=None):  # noqa: ANN001
        self.translate_calls.append(None if budgets is None else list(budgets))
        value = self.short_value if budgets is not None else self.source_value
        return [value for _ in texts]

    def translate_contextual(self, texts, src_lang, tgt_lang, *, budgets=None):  # noqa: ANN001
        self.contextual_calls.append(None if budgets is None else list(budgets))
        return [self.long_value for _ in texts]


LEGACY_UNIT_KEYS = {
    "index",
    "indices",
    "start",
    "end",
    "speaker",
    "source",
    "vi",
    "spoken_vi",
    "subtitle_vi",
    "words_in",
    "coverage",
    "guard",
    "battery",
    "open",
}


def test_isometric_shrinks_over_budget_unit(tmp_path: Path) -> None:
    """A budget-honouring engine shrinks the unit and the records show it."""
    segments = [{"start": 0.0, "end": 1.5, "text": ISO_SOURCE, "speaker": "spk0"}]
    job_path, artifacts_dir, cfg = _make_job(tmp_path, segments, sentence_units=True)
    engine = _BudgetedTranslator(ISO_LONG, ISO_SHORT, ISO_SOURCE)

    translate(str(job_path), cfg, translator=engine)

    # the initial batch got the unit's budget (1.5 s * 20 cps * 0.9 headroom)
    assert engine.contextual_calls == [[27]]
    # the one budgeted repair was requested with that same budget
    assert [call for call in engine.translate_calls if call is not None] == [[27]]

    doc = json.loads((artifacts_dir / UNITS_NAME).read_text(encoding="utf-8"))
    unit = doc["units"][0]
    assert unit["isometric"] == "shrunk"
    assert unit["over_budget"] is True
    assert unit["budget_chars"] == 27
    assert unit["chars"] == _nonspace_len(ISO_LONG)
    assert unit["chars_after"] == _nonspace_len(ISO_SHORT)
    assert unit["chars_after"] < unit["budget_chars"]
    assert unit["projected_stretch_after"] < unit["projected_stretch_before"]

    artifact = json.loads(
        (artifacts_dir / TRANSLATION_QC_NAME).read_text(encoding="utf-8")
    )
    flags = artifact["units"]["flags"]
    assert flags["isometric_shrunk"] == 1
    assert flags["isometric_kept"] == 0
    assert flags["over_budget_after"] == 0

    script = json.loads((artifacts_dir / SCRIPT_NAME).read_text(encoding="utf-8"))
    assert script[0]["vi"] == ISO_SHORT


def test_isometric_keeps_when_engine_ignores_the_budget(tmp_path: Path) -> None:
    """A budget-ignoring repair candidate is rejected; the original survives."""
    segments = [{"start": 0.0, "end": 1.5, "text": ISO_SOURCE, "speaker": "spk0"}]
    job_path, artifacts_dir, cfg = _make_job(tmp_path, segments, sentence_units=True)
    # the repair returns the same long text -> no shrink -> kept
    engine = _BudgetedTranslator(ISO_LONG, ISO_LONG, ISO_SOURCE)

    translate(str(job_path), cfg, translator=engine)

    doc = json.loads((artifacts_dir / UNITS_NAME).read_text(encoding="utf-8"))
    unit = doc["units"][0]
    assert unit["isometric"] == "kept"
    assert unit["vi"] == ISO_LONG
    assert unit["chars_after"] == unit["chars"]

    script = json.loads((artifacts_dir / SCRIPT_NAME).read_text(encoding="utf-8"))
    assert script[0]["vi"] == ISO_LONG
    artifact = json.loads(
        (artifacts_dir / TRANSLATION_QC_NAME).read_text(encoding="utf-8")
    )
    assert artifact["units"]["flags"]["isometric_kept"] == 1
    assert artifact["units"]["flags"]["isometric_shrunk"] == 0


def test_isometric_rejects_candidate_below_omission_floor(tmp_path: Path) -> None:
    """``accept_shrunk`` is the only gate: a too-short candidate is kept away."""
    from vidub.utils.isometry import accept_shrunk
    from vidub.utils.units_guard import guard_signals

    # the predicate's reason, asserted directly: "ngắn" (1 word) is omission-risk
    assert "omission-risk" in guard_signals(ISO_SOURCE, "ngắn")["flags"]
    assert accept_shrunk(ISO_SOURCE, "ngắn", 27, previous_vi=ISO_LONG) is False

    segments = [{"start": 0.0, "end": 1.5, "text": ISO_SOURCE, "speaker": "spk0"}]
    job_path, artifacts_dir, cfg = _make_job(tmp_path, segments, sentence_units=True)
    engine = _BudgetedTranslator(ISO_LONG, "ngắn", ISO_SOURCE)

    translate(str(job_path), cfg, translator=engine)

    doc = json.loads((artifacts_dir / UNITS_NAME).read_text(encoding="utf-8"))
    assert doc["units"][0]["isometric"] == "kept"
    script = json.loads((artifacts_dir / SCRIPT_NAME).read_text(encoding="utf-8"))
    assert script[0]["vi"] == ISO_LONG


def test_isometric_only_over_budget_units_get_a_repair(tmp_path: Path) -> None:
    """A mixed fixture: only the over-budget unit is re-sent with a budget."""
    segments = [
        {"start": 0.0, "end": 1.0, "text": "Hello world good morning everyone.", "speaker": "spk0"},
        {"start": 1.0, "end": 8.0, "text": "The weather is nice today.", "speaker": "spk0"},
    ]
    job_path, artifacts_dir, cfg = _make_job(tmp_path, segments, sentence_units=True)
    engine = _BudgetedTranslator(ISO_LONG, ISO_SHORT, ISO_SOURCE)

    translate(str(job_path), cfg, translator=engine)

    # budgets: window 1.0 -> 18 (over for the 55-char text), window 7.0 -> 126
    assert engine.contextual_calls == [[18, 126]]
    budgeted_repairs = [call for call in engine.translate_calls if call is not None]
    assert budgeted_repairs == [[18]]  # unit 1 (under budget) is never re-sent

    doc = json.loads((artifacts_dir / UNITS_NAME).read_text(encoding="utf-8"))
    assert [entry["isometric"] for entry in doc["units"]] == ["shrunk", "none"]
    assert [entry["over_budget"] for entry in doc["units"]] == [True, False]


def test_no_isometric_passes_no_budgets_and_keeps_legacy_keys(tmp_path: Path) -> None:
    """``--no-isometric`` -> zero budgeted calls and today's artifact shapes."""
    segments = [{"start": 0.0, "end": 1.5, "text": ISO_SOURCE, "speaker": "spk0"}]
    job_path, artifacts_dir, cfg = _make_job(tmp_path, segments, sentence_units=True)
    cfg.isometric = False
    engine = _BudgetedTranslator(ISO_LONG, ISO_SHORT, ISO_SOURCE)

    translate(str(job_path), cfg, translator=engine)

    # no budget anywhere: the contextual batch and every plain call are 3-arg
    assert engine.contextual_calls == [None]
    assert engine.translate_calls  # the battery's round-trip still called translate
    assert all(call is None for call in engine.translate_calls)

    doc = json.loads((artifacts_dir / UNITS_NAME).read_text(encoding="utf-8"))
    for entry in doc["units"]:
        assert set(entry) == LEGACY_UNIT_KEYS

    artifact = json.loads(
        (artifacts_dir / TRANSLATION_QC_NAME).read_text(encoding="utf-8")
    )
    flags = artifact["units"]["flags"]
    for key in ("over_budget_after", "isometric_shrunk", "isometric_kept"):
        assert key not in flags


def test_no_sentence_units_ignores_isometric(tmp_path: Path) -> None:
    """The legacy per-segment path never computes or passes a budget."""
    job_path, artifacts_dir, cfg = _make_job(tmp_path, SEGMENTS, sentence_units=False)
    engine = _BudgetedTranslator(ISO_LONG, ISO_SHORT, ISO_SOURCE)

    translate(str(job_path), cfg, translator=engine)

    assert engine.contextual_calls == [None]
    assert all(call is None for call in engine.translate_calls)
    assert not (artifacts_dir / UNITS_NAME).exists()
    script = json.loads((artifacts_dir / SCRIPT_NAME).read_text(encoding="utf-8"))
    for row in script:
        assert "isometric" not in row


def test_mt_signature_encodes_isometric_knobs() -> None:
    from vidub.pipeline import mt_signature

    base = mt_signature(JobConfig())
    # ISO-1 close (v0.11.0): the default is off, so the signature reads iso=0.
    assert "iso=0" in base
    assert mt_signature(JobConfig()) == base  # deterministic
    assert mt_signature(JobConfig(isometric=True)) != base
    assert mt_signature(JobConfig(isometric_headroom=0.5)) != base
    assert mt_signature(JobConfig(isometric_cps=25.0)) != base


# ======================================= ISO-1-T3-C1 correction (adaptive + KoK)
class _ChatTranslator(BaseTranslator):
    """hunyuan-like stub exposing ``_chat`` for the Best-of-K lever.

    ``translate_contextual`` returns ``initial`` (the over-budget first pass).
    ``_chat`` returns the scripted candidates in call order (the three concise
    variants). ``translate`` without a budget returns ``source_value`` so the
    battery's round-trip stays clean in the qc-on cases.
    """

    def __init__(self, initial: str | list[str], candidates: list[str], source_value: str = "") -> None:
        self.initial = initial
        self.candidates = list(candidates)
        self.source_value = source_value
        self.contextual_budgets: list[list[int] | None] = []
        self.chat_prompts: list[str] = []

    def translate(self, texts, src_lang, tgt_lang, *, budgets=None):  # noqa: ANN001
        value = self.initial if budgets is not None else self.source_value
        return [value for _ in texts]

    def translate_contextual(self, texts, src_lang, tgt_lang, *, budgets=None):  # noqa: ANN001
        self.contextual_budgets.append(None if budgets is None else list(budgets))
        if isinstance(self.initial, list):
            return [self.initial[index] for index in range(len(texts))]
        return [self.initial for _ in texts]

    def _chat(self, user, temperature=0.7):  # noqa: ANN001
        self.chat_prompts.append(user)
        index = min(len(self.chat_prompts) - 1, len(self.candidates) - 1)
        return self.candidates[index]


# A Vietnamese text that shares no content tokens with ISO_LONG and is LONGER
# than it (so a budget-ignoring engine yields ``no-candidate-shorter``).
DISTINCT_LONGER = (
    "Những chú gấu trúc màu trắng đen thích ăn tre non vào mỗi sáng sớm tinh mơ"
)
NUMBER_SOURCE = "The GPU has 16 GB of RAM and runs fast today"
NUMBER_ORIGINAL = "GPU cần 16 GB bộ nhớ và chạy rất nhanh trong mọi tác vụ hôm nay"
NUMBER_CANDIDATE = "GPU cần bộ nhớ và chạy rất nhanh trong mọi tác vụ hôm nay"
# 20 source words / no terminal; the candidate keeps 14 -> coverage 0.70.
# Distinct Vietnamese tokens (a repeated token would be collapsed by
# ``clean_hesitation``) and disjoint prefixes (so the containment guard sees no
# neighbour bleed between the current text and the candidate).
FLOOR_SOURCE_TX = " ".join(f"s{chr(ord('a') + index)}" for index in range(20))
FLOOR_PREV = " ".join(f"mã{index}" for index in range(25))
FLOOR_CAND = " ".join(f"bò{index}" for index in range(14))


def test_best_of_k_picks_the_shortest_passing_candidate(tmp_path: Path) -> None:
    segments = [{"start": 0.0, "end": 1.5, "text": ISO_SOURCE, "speaker": "spk0"}]
    job_path, artifacts_dir, cfg = _make_job(tmp_path, segments, sentence_units=True)
    cfg.qc_translation = False  # isolate the lever from the battery
    shortest = ISO_SHORT  # 7 words
    medium = "Con cáo nâu nhanh nhẹn nhảy qua chú chó"  # 9 words
    longest = "Con cáo nâu nhanh nhẹn nhảy qua con chó lười biếng"  # 11 words
    # the shortest is the SECOND variant, so a naive "first wins" would fail
    engine = _ChatTranslator(ISO_LONG, [medium, shortest, longest], ISO_SOURCE)

    translate(str(job_path), cfg, translator=engine)

    # the FIRST translation call carries the per-unit budget (27 chars @ 1.5 s)
    assert engine.contextual_budgets == [[27]]
    # <= K extra calls, each opening with the proven concise translate request
    assert len(engine.chat_prompts) == 3
    for prompt in engine.chat_prompts:
        assert prompt.startswith("Translate the following text into Vietnamese")
        assert "at most 27 characters" in prompt
        assert "output only the translation" in prompt

    unit = json.loads((artifacts_dir / UNITS_NAME).read_text(encoding="utf-8"))["units"][0]
    assert unit["isometric"] == "shrunk"
    assert unit["isometric_reason"] == "shrunk"
    assert unit["isometric_candidates"] == 3
    assert unit["isometric_variant"] == "drop-filler"  # the shortest (2nd) variant
    assert unit["vi"] == shortest
    assert unit["chars_after"] == _nonspace_len(shortest)
    assert unit["coverage_floor"] == 0.65
    assert unit["short_floor_applied"] is True
    assert unit["projected_stretch_after"] < unit["projected_stretch_before"]


def test_isometric_reason_no_candidate_shorter(tmp_path: Path) -> None:
    segments = [{"start": 0.0, "end": 1.5, "text": ISO_SOURCE, "speaker": "spk0"}]
    job_path, artifacts_dir, cfg = _make_job(tmp_path, segments, sentence_units=True)
    cfg.qc_translation = False
    engine = _ChatTranslator(ISO_LONG, [DISTINCT_LONGER] * 3, ISO_SOURCE)

    translate(str(job_path), cfg, translator=engine)

    unit = json.loads((artifacts_dir / UNITS_NAME).read_text(encoding="utf-8"))["units"][0]
    assert unit["isometric"] == "kept"
    assert unit["isometric_reason"] == "no-candidate-shorter"
    assert unit["vi"] == ISO_LONG  # the original survives


def test_isometric_reason_validator_number(tmp_path: Path) -> None:
    segments = [{"start": 0.0, "end": 1.5, "text": NUMBER_SOURCE, "speaker": "spk0"}]
    job_path, artifacts_dir, cfg = _make_job(tmp_path, segments, sentence_units=True)
    cfg.qc_translation = False
    engine = _ChatTranslator(
        NUMBER_ORIGINAL, [NUMBER_CANDIDATE] * 3, NUMBER_SOURCE
    )

    translate(str(job_path), cfg, translator=engine)

    unit = json.loads((artifacts_dir / UNITS_NAME).read_text(encoding="utf-8"))["units"][0]
    assert unit["isometric"] == "kept"
    assert unit["isometric_reason"].startswith("validator:")
    assert "16" in unit["isometric_reason"]


def test_isometric_reason_context_bleed(tmp_path: Path) -> None:
    """D1(b): a candidate that copies its NEIGHBOUR is rejected on context-bleed.

    Two units; only unit 1 is over budget, and its repair candidates copy unit
    0's text (the previous neighbour). The contamination check must still catch
    that (it compares neighbours, not the unit's own text).
    """
    segments = [
        {"start": 0.0, "end": 8.0, "text": "Alpha beta gamma delta epsilon.", "speaker": "spk0"},
        {"start": 8.0, "end": 9.5, "text": "Zeta eta theta iota kappa.", "speaker": "spk0"},
    ]
    job_path, artifacts_dir, cfg = _make_job(tmp_path, segments, sentence_units=True)
    cfg.qc_translation = False
    initial0 = "Nội dung của đơn vị đầu tiên khá riêng biệt"
    initial1 = "Nội dung khác hẳn cho đơn vị thứ hai dài"
    # only unit 1 over budget -> only its 3 candidates are requested, all = initial0
    engine = _ChatTranslator([initial0, initial1], [initial0] * 3, "")

    translate(str(job_path), cfg, translator=engine)

    units = json.loads((artifacts_dir / UNITS_NAME).read_text(encoding="utf-8"))["units"]
    assert units[0]["isometric"] == "none"  # under its 144-char budget
    assert units[1]["isometric"] == "kept"
    assert units[1]["isometric_reason"].startswith("context-bleed:")
    assert units[1]["vi"] == initial1


# The compressed rewrite of ISO_LONG: it shares most content tokens with the
# text it compresses (high self-overlap), which the old self-bleed bug rejected.
ISO_COMPRESSED = "Đây là bản dịch tiếng Việt dài dòng cho đơn vị"


def test_compression_accepts_a_high_self_overlap_candidate(tmp_path: Path) -> None:
    """D1(a): a genuine compression of the CURRENT text is accepted (``shrunk``).

    Fails if the contamination check regresses to comparing the candidate
    against the unit's own current text (self-overlap ~0.64 would false-flag).
    """
    segments = [{"start": 0.0, "end": 1.5, "text": ISO_SOURCE, "speaker": "spk0"}]
    job_path, artifacts_dir, cfg = _make_job(tmp_path, segments, sentence_units=True)
    cfg.qc_translation = False
    engine = _ChatTranslator(ISO_LONG, [ISO_COMPRESSED] * 3, ISO_SOURCE)

    translate(str(job_path), cfg, translator=engine)

    unit = json.loads((artifacts_dir / UNITS_NAME).read_text(encoding="utf-8"))["units"][0]
    assert unit["isometric"] == "shrunk"
    assert unit["isometric_reason"] == "shrunk"
    assert unit["vi"] == ISO_COMPRESSED
    assert unit["chars_after"] < unit["chars"]


def test_accept_shrunk_contamination_uses_neighbours_not_current() -> None:
    """The predicate's contamination check takes neighbours, not the unit text."""
    from vidub.utils.isometry import accept_shrunk

    neighbour = "Nội dung hoàn toàn khác biệt của đơn vị lân cận bên cạnh"
    # a genuine compression of the current text is accepted ...
    assert (
        accept_shrunk(
            ISO_SOURCE,
            ISO_COMPRESSED,
            27,
            previous_vi=neighbour,
            current_vi=ISO_LONG,
        )
        is True
    )
    # ... while a candidate copying the neighbour is rejected on context-bleed
    assert (
        accept_shrunk(
            ISO_SOURCE, neighbour, 27, previous_vi=neighbour, current_vi=ISO_LONG
        )
        is False
    )


def test_tiered_floor_accepts_a_short_window_and_records_it(tmp_path: Path) -> None:
    segments = [{"start": 0.0, "end": 1.6, "text": FLOOR_SOURCE_TX, "speaker": "spk0"}]
    job_path, artifacts_dir, cfg = _make_job(tmp_path, segments, sentence_units=True)
    engine = _BudgetedTranslator(FLOOR_PREV, FLOOR_CAND, FLOOR_SOURCE_TX)

    translate(str(job_path), cfg, translator=engine)

    unit = json.loads((artifacts_dir / UNITS_NAME).read_text(encoding="utf-8"))["units"][0]
    assert unit["isometric"] == "shrunk"
    assert unit["coverage_floor"] == 0.65
    assert unit["short_floor_applied"] is True
    assert unit["vi"] == FLOOR_CAND

    artifact = json.loads(
        (artifacts_dir / TRANSLATION_QC_NAME).read_text(encoding="utf-8")
    )
    flags = artifact["units"]["flags"]
    assert flags["isometric_short_floor"] == 1
    assert flags["isometric_shrunk"] == 1
    detail = artifact["units"]["detail"][0]
    assert detail["coverage_floor"] == 0.65
    assert detail["short_floor_applied"] is True
    assert detail["isometric_reason"] == "shrunk"


def test_tiered_floor_rejects_a_normal_window_with_the_reason(tmp_path: Path) -> None:
    segments = [{"start": 0.0, "end": 2.2, "text": FLOOR_SOURCE_TX, "speaker": "spk0"}]
    job_path, artifacts_dir, cfg = _make_job(tmp_path, segments, sentence_units=True)
    cfg.qc_translation = False
    engine = _BudgetedTranslator(FLOOR_PREV, FLOOR_CAND, FLOOR_SOURCE_TX)

    translate(str(job_path), cfg, translator=engine)

    unit = json.loads((artifacts_dir / UNITS_NAME).read_text(encoding="utf-8"))["units"][0]
    assert unit["isometric"] == "kept"
    assert unit["coverage_floor"] == 0.75
    assert unit["short_floor_applied"] is False
    assert unit["isometric_reason"] == "omission-risk:0.700<0.750"
    assert unit["vi"] == FLOOR_PREV


# 25 source words, no terminal; the candidate keeps 18 distinct words -> coverage
# 0.720, strictly between the configured 0.70 short floor and the normal 0.75
# floor. Distinct digit-bearing tokens survive ``clean_hesitation`` unchanged.
CFGW_SOURCE = " ".join(f"s{chr(ord('a') + index)}" for index in range(25))
CFGW_PREV = " ".join(f"mã{index}" for index in range(30))
CFGW_CAND = " ".join(f"bò{index}" for index in range(18))


def test_configured_short_floor_reaches_acceptance_and_the_record(
    tmp_path: Path,
) -> None:
    """ISO-1-C1-FIX: the configured knobs decide acceptance, not only the record.

    A 2.2 s unit with a 0.72-coverage candidate: with
    ``isometric_short_window = 2.5`` (2.2 < 2.5) and
    ``isometric_short_coverage = 0.70`` the floor is 0.70 and the candidate is
    accepted; with the module default threshold (1.80 s) the same unit is a
    normal window (floor 0.75) and the candidate is rejected.
    """
    segments = [{"start": 0.0, "end": 2.2, "text": CFGW_SOURCE, "speaker": "spk0"}]

    # (1) configured: 2.2 s < 2.5 s -> floor 0.70 -> the 0.72 candidate is accepted
    job_path, artifacts_dir, cfg = _make_job(
        tmp_path / "configured", segments, sentence_units=True
    )
    cfg.qc_translation = False
    cfg.isometric_short_window = 2.5
    cfg.isometric_short_coverage = 0.70
    engine = _BudgetedTranslator(CFGW_PREV, CFGW_CAND, CFGW_SOURCE)

    translate(str(job_path), cfg, translator=engine)

    unit = json.loads((artifacts_dir / UNITS_NAME).read_text(encoding="utf-8"))[
        "units"
    ][0]
    assert unit["coverage_floor"] == 0.70
    assert unit["short_floor_applied"] is True
    assert unit["isometric"] == "shrunk"
    assert unit["vi"] == CFGW_CAND

    # (2) default threshold (1.80 s): 2.2 s is a NORMAL window -> floor 0.75 ->
    # the very same candidate is rejected and the original survives.
    job_path, artifacts_dir, cfg = _make_job(
        tmp_path / "default", segments, sentence_units=True
    )
    cfg.qc_translation = False
    engine = _BudgetedTranslator(CFGW_PREV, CFGW_CAND, CFGW_SOURCE)

    translate(str(job_path), cfg, translator=engine)

    unit = json.loads((artifacts_dir / UNITS_NAME).read_text(encoding="utf-8"))[
        "units"
    ][0]
    assert unit["coverage_floor"] == 0.75
    assert unit["short_floor_applied"] is False
    assert unit["isometric"] == "kept"
    assert unit["vi"] == CFGW_PREV


def test_mt_signature_encodes_isometric_short_floor_knobs() -> None:
    from vidub.pipeline import mt_signature

    base = mt_signature(JobConfig())
    assert "iso_cov=0.65" in base
    assert "iso_win=1.8" in base
    assert mt_signature(JobConfig(isometric_short_coverage=0.6)) != base
    assert mt_signature(JobConfig(isometric_short_window=2.0)) != base


# ------------------------- ISO-1-T3-C1-cont2: the proven concise prompt shape
def test_concise_request_exact_shape_for_all_variants() -> None:
    from vidub.stages.translate import _CONCISE_VARIANTS, _concise_request

    budget = 27
    hints = []
    for _variant, hint in _CONCISE_VARIANTS:
        hints.append(hint)
        prompt = _concise_request(ISO_SOURCE, "vi", hint, budget)
        assert prompt == (
            "Translate the following text into Vietnamese concisely: at most "
            f"{budget} characters, {hint}, keep the meaning exactly, and output "
            "only the translation.\n\n" + ISO_SOURCE
        )
        # the three clauses a future edit must not silently drop
        assert "Vietnamese" in prompt
        assert str(budget) in prompt
        assert "output only the translation" in prompt
    assert len(set(hints)) == 3  # three distinct compression strategies


def test_concise_prompt_rejects_non_vietnamese_and_accepts_valid(
    tmp_path: Path,
) -> None:
    segments = [{"start": 0.0, "end": 1.5, "text": ISO_SOURCE, "speaker": "spk0"}]

    # a non-Vietnamese candidate -> the validator reason is persisted
    job_path, artifacts_dir, cfg = _make_job(
        tmp_path / "rejected", segments, sentence_units=True
    )
    cfg.qc_translation = False
    english = "This is an English sentence, not a Vietnamese translation"
    engine = _ChatTranslator(ISO_LONG, [english] * 3, ISO_SOURCE)

    translate(str(job_path), cfg, translator=engine)

    unit = json.loads((artifacts_dir / UNITS_NAME).read_text(encoding="utf-8"))["units"][0]
    assert unit["isometric"] == "kept"
    assert unit["isometric_reason"].startswith("validator:")
    assert "diacritics" in unit["isometric_reason"]

    # a valid shorter Vietnamese candidate is accepted as ``shrunk``
    job_path, artifacts_dir, cfg = _make_job(
        tmp_path / "accepted", segments, sentence_units=True
    )
    cfg.qc_translation = False
    engine = _ChatTranslator(ISO_LONG, [ISO_SHORT] * 3, ISO_SOURCE)

    translate(str(job_path), cfg, translator=engine)

    unit = json.loads((artifacts_dir / UNITS_NAME).read_text(encoding="utf-8"))["units"][0]
    assert unit["isometric"] == "shrunk"
    assert unit["isometric_reason"] == "shrunk"
    assert unit["vi"] == ISO_SHORT

