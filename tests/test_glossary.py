"""Tests for the translation glossary + language support matrix (M3-T2).

Fast in-process tests: no model is loaded. The translate-with-glossary tests use
a fake translator and assert the enforced term lands in ``script.json`` plus the
``glossary`` counters in the report (assembled with ``build_report`` directly).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

import vidub.pipeline as pipeline
from vidub.config import TTS_LANGS, JobConfig, validate_langs
from vidub.engines.mt.base import BaseTranslator
from vidub.manifest import JobManifest
from vidub.stages.export import build_report
from vidub.stages.translate import GLOSSARY_NAME, SCRIPT_NAME, translate
from vidub.utils.glossary import (
    apply_glossary,
    find_source_terms,
    load_glossary,
    parse_glossary,
    splice_source_terms,
    split_source_on_term,
)


# ----------------------------------------------------------------------- loader
def test_load_glossary_valid_rows_comments_and_blanks(tmp_path: Path) -> None:
    path = tmp_path / "g.tsv"
    path.write_text(
        "# a comment\n"
        "\n"
        "Vigib\tViDub\n"
        "AI\ttrí tuệ nhân tạo\n",
        encoding="utf-8",
    )

    assert load_glossary(path) == {"Vigib": "ViDub", "AI": "trí tuệ nhân tạo"}


def test_parse_glossary_reports_bad_rows() -> None:
    mapping, problems = parse_glossary(
        "ok\tfine\n"
        "missing-tab\n"
        "\tno-term\n"
        "empty-translation\t\n"
    )

    assert mapping == {"ok": "fine"}
    assert len(problems) == 3
    assert all("line" in problem for problem in problems)


def test_load_glossary_skips_bad_rows_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "g.tsv"
    path.write_text("good\tgreat\nbroken-row\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="vidub.utils.glossary"):
        mapping = load_glossary(path)

    assert mapping == {"good": "great"}
    assert any("skipped" in record.getMessage() for record in caplog.records)


def test_load_glossary_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_glossary(tmp_path / "nope.tsv")


# ------------------------------------------------------------------------ apply
def test_apply_glossary_case_insensitive_whole_word() -> None:
    text = "The VIGIB tool is not Vigibish, but vigib is fine."
    mapping = {"Vigib": "ViDub"}

    replaced, count = apply_glossary(text, mapping)

    # both case variants replaced, but the longer word is left untouched
    assert replaced == "The ViDub tool is not Vigibish, but ViDub is fine."
    assert count == 2


def test_apply_glossary_longest_term_first() -> None:
    mapping = {"trí": "mind", "trí tuệ": "intelligence"}

    replaced, count = apply_glossary("trí tuệ nhân tạo", mapping)

    assert replaced == "intelligence nhân tạo"
    assert count == 1


def test_apply_glossary_escapes_regex_terms() -> None:
    replaced, count = apply_glossary("I love C++ a lot", {"C++": "Cpp"})

    assert replaced == "I love Cpp a lot"
    assert count == 1


def test_apply_glossary_zero_match_and_empty_mapping() -> None:
    assert apply_glossary("nothing to see", {"cat": "meow"}) == ("nothing to see", 0)
    assert apply_glossary("anything", {}) == ("anything", 0)


# ------------------------------------------------- plural-tolerant source side
def test_find_source_terms_matches_plural_gpu() -> None:
    source = "We need two GPUs today"

    hits = find_source_terms(source, {"GPU": "GPU"})

    assert len(hits) == 1
    term, target, start, end = hits[0]
    assert (term, target) == ("GPU", "GPU")
    # the matched span includes the plural 's' so the splice skips it
    assert source[start:end] == "GPUs"


def test_find_source_terms_matches_plural_peasant() -> None:
    hits = find_source_terms("peasants unite", {"peasant": "anh em"})

    assert len(hits) == 1
    assert hits[0][0] == "peasant"
    assert hits[0][2:4] == (0, len("peasants"))


def test_apply_glossary_does_not_replace_plural() -> None:
    """Output-side enforcement stays exact: GPUs is not replaced by GPU."""
    replaced, count = apply_glossary("two GPUs today", {"GPU": "GPU"})

    assert replaced == "two GPUs today"
    assert count == 0


def test_splice_source_terms_counts_plural_hit_when_target_present() -> None:
    text, rebuilt, enforced = splice_source_terms(
        "We need two GPUs today",
        "Chúng tôi cần hai GPU hôm nay",
        {"GPU": "GPU"},
        lambda chunk: chunk,
    )

    assert enforced == 1
    assert rebuilt is False
    assert text == "Chúng tôi cần hai GPU hôm nay"


def test_splice_source_terms_rebuilds_around_plural_hit() -> None:
    text, rebuilt, enforced = splice_source_terms(
        "We need two GPUs today",
        "máy cần",
        {"GPU": "GPU"},
        lambda chunk: f"[{chunk}]",
    )

    assert rebuilt is True
    assert enforced == 1
    # the plural 's' is consumed by the match, not left dangling before the term
    assert "We need two" in text
    assert "today" in text
    assert "GPU" in text
    assert "GPUs" not in text


def test_split_source_on_term_stays_exact() -> None:
    # the splitting helper keeps the exact word boundary (no plural tolerance)
    assert split_source_on_term("two GPUs", "GPU") == ("two GPUs", "")


# ------------------------------------------------------ translate + glossary
class FakeTranslator(BaseTranslator):
    """Returns a fixed Vietnamese sentence containing the term ``Vigib``."""

    def __init__(self, template: str = "Đây là công cụ Vigib hôm nay.") -> None:
        self.template = template
        self.calls: list[list[str]] = []

    def translate(self, texts, src_lang, tgt_lang):
        self.calls.append(list(texts))
        return [self.template for _ in texts]


def _make_job(
    tmp_path: Path,
    *,
    source: str = "This is the Vigib tool today",
    glossary_path: str | None = None,
) -> tuple[Path, Path, JobConfig]:
    cache_dir = tmp_path / "cache"
    artifacts_dir = cache_dir / "hash1"
    artifacts_dir.mkdir(parents=True)
    (artifacts_dir / "segments.json").write_text(
        json.dumps([{"start": 0.0, "end": 3.0, "text": source, "speaker": "spk0"}]),
        encoding="utf-8",
    )
    job_path = artifacts_dir / "job.json"
    JobManifest(
        input_path="demo.mp4",
        video_hash="hash1",
        cache_dir=str(cache_dir),
        src_lang="en",
        tgt_lang="vi",
        paths={"segments": "segments.json"},
        stages={"transcribe": "done"},
    ).save(job_path)
    cfg = JobConfig(cache_dir=str(cache_dir), glossary_path=glossary_path)
    return job_path, artifacts_dir, cfg


def test_translate_applies_glossary_and_records_counts(tmp_path: Path) -> None:
    glossary = tmp_path / "g.tsv"
    glossary.write_text("Vigib\tViDub\n", encoding="utf-8")
    job_path, artifacts_dir, cfg = _make_job(tmp_path, glossary_path=str(glossary))

    manifest = translate(str(job_path), cfg, translator=FakeTranslator())

    script = json.loads((artifacts_dir / SCRIPT_NAME).read_text(encoding="utf-8"))
    assert "ViDub" in script[0]["vi"]
    assert "Vigib" not in script[0]["vi"]

    report = build_report(manifest)
    assert report["glossary"] == {
        "terms": 1,
        "replacements": 1,
        "spliced": 0,
        "enforced": 1,
    }


def test_translate_zero_match_still_writes_zero_sidecar(tmp_path: Path) -> None:
    """A glossary term that never occurs reports 0 replacements, not an error."""
    glossary = tmp_path / "g.tsv"
    glossary.write_text("vidub\tViDub\n", encoding="utf-8")
    job_path, artifacts_dir, cfg = _make_job(tmp_path, glossary_path=str(glossary))

    manifest = translate(str(job_path), cfg, translator=FakeTranslator())

    sidecar = json.loads((artifacts_dir / GLOSSARY_NAME).read_text(encoding="utf-8"))
    assert sidecar == {
        "terms": 1,
        "replacements": 0,
        "spliced": 0,
        "enforced": 0,
        "hesitations": 0,
        "style_fixes": 0,
    }
    assert build_report(manifest)["glossary"] == {
        "terms": 1,
        "replacements": 0,
        "spliced": 0,
        "enforced": 0,
    }


def test_translate_without_glossary_writes_zero_sidecar(tmp_path: Path) -> None:
    job_path, artifacts_dir, cfg = _make_job(tmp_path)

    manifest = translate(str(job_path), cfg, translator=FakeTranslator())

    sidecar = json.loads((artifacts_dir / GLOSSARY_NAME).read_text(encoding="utf-8"))
    assert sidecar == {
        "terms": 0,
        "replacements": 0,
        "spliced": 0,
        "enforced": 0,
        "hesitations": 0,
        "style_fixes": 0,
    }
    assert build_report(manifest)["glossary"] == {
        "terms": 0,
        "replacements": 0,
        "spliced": 0,
        "enforced": 0,
    }


def test_glossary_not_applied_to_passthrough_segments(tmp_path: Path) -> None:
    """Already-Vietnamese segments are user-language and must not be replaced."""
    glossary = tmp_path / "g.tsv"
    glossary.write_text("Vigib\tViDub\n", encoding="utf-8")
    job_path, artifacts_dir, cfg = _make_job(
        tmp_path,
        source="Đây là công cụ Vigib tiếng Việt.",
        glossary_path=str(glossary),
    )

    engine = FakeTranslator()
    manifest = translate(str(job_path), cfg, translator=engine)

    script = json.loads((artifacts_dir / SCRIPT_NAME).read_text(encoding="utf-8"))
    assert script[0].get("passthrough") is True
    assert "Vigib" in script[0]["vi"]
    assert "ViDub" not in script[0]["vi"]
    assert engine.calls == []  # the MT engine never saw the passthrough segment
    assert build_report(manifest)["glossary"] == {
        "terms": 1,
        "replacements": 0,
        "spliced": 0,
        "enforced": 0,
    }


# ------------------------------------------------------------- language matrix
def test_tts_langs_matrix_shape() -> None:
    assert TTS_LANGS["piper"] == frozenset({"vi"})
    for lang in ("vi", "en", "es", "zh", "ja"):
        assert lang in TTS_LANGS["vixtts"]


def test_validate_langs_piper_non_vi_raises_naming_pair() -> None:
    with pytest.raises(ValueError) as excinfo:
        validate_langs("en", "en", "piper")

    message = str(excinfo.value)
    assert "unsupported language pair en->en" in message
    assert "--tts-engine" in message


def test_validate_langs_vixtts_non_vi_ok() -> None:
    validate_langs("en", "en", "vixtts")  # must not raise


def test_validate_langs_vi_always_ok() -> None:
    validate_langs("en", "vi", "piper")
    validate_langs("en", "vi", "vixtts")


# ---------------------------------------------------------- fail-fast wiring
def test_run_pipeline_rejects_unsupported_pair_before_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        pipeline.ingest_mod, "ingest", lambda *args, **kwargs: calls.append("ingest")
    )

    with pytest.raises(ValueError, match="unsupported language pair"):
        pipeline.run_pipeline(
            "video.mp4",
            JobConfig(cache_dir=str(tmp_path), tgt_lang="en", tts_engine="piper"),
        )

    assert calls == []


def test_speak_main_rejects_unsupported_pair(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from vidub.stages import speak as speak_mod

    cache_dir = tmp_path / "cache"
    artifacts_dir = cache_dir / "hash1"
    artifacts_dir.mkdir(parents=True)
    job_path = artifacts_dir / "job.json"
    JobManifest(
        input_path="demo.mp4",
        video_hash="hash1",
        cache_dir=str(cache_dir),
        tgt_lang="en",
        tts_engine="piper",
    ).save(job_path)

    rc = speak_mod.main([str(job_path)])

    assert rc == 1
    assert "--tts-engine" in capsys.readouterr().err
