import hashlib

import pytest

from scripts.eval_mt_rework import (
    PROMPT_TEMPLATE_SHA256,
    _ollama_translate,
    _prompt_template_sha256,
    main,
    run_mt_cases,
)


def test_run_mt_cases_preserves_ids_and_gives_engines_identical_verified_source():
    cases = [
        {
            "id": "video-a:unit:2",
            "source_asr": "Raw ASR text.",
            "verified_source": "Verified source text.",
        },
        {
            "id": "video-b:unit:8",
            "source_asr": "Second verified source.",
            "verified_source": "Second verified source.",
        },
    ]
    seen_envit5 = []
    seen_hy_mt2 = []

    envit5 = run_mt_cases(cases, lambda text: seen_envit5.append(text) or "Bản dịch một.", "envit5")
    hy_mt2 = run_mt_cases(cases, lambda text: seen_hy_mt2.append(text) or "Bản dịch hai.", "hy-mt2")

    assert seen_envit5 == seen_hy_mt2 == ["Verified source text.", "Second verified source."]
    assert [row["id"] for row in envit5] == [row["id"] for row in hy_mt2] == [
        "video-a:unit:2",
        "video-b:unit:8",
    ]
    assert [row["input_status"] for row in envit5] == ["verified_source"] * 2
    assert [row["source_sha256"] for row in envit5] == [
        hashlib.sha256(text.encode("utf-8")).hexdigest() for text in seen_envit5
    ]
    assert all(row["elapsed_seconds"] >= 0 for row in envit5 + hy_mt2)


def test_run_mt_cases_labels_unverified_source_asr():
    cases = [{"id": "video:unit:0", "source_asr": "Unreviewed transcript."}]
    seen = []

    rows = run_mt_cases(cases, lambda text: seen.append(text) or "Bản dịch.", "envit5")

    assert seen == ["Unreviewed transcript."]
    assert rows[0]["input_status"] == "source_asr_unverified"


def test_run_mt_cases_rejects_blank_translation_and_duplicate_id():
    case = {"id": "video:unit:4", "verified_source": "Text."}
    with pytest.raises(ValueError, match="video:unit:4"):
        run_mt_cases([case], lambda _text: "  ", "fake-mt")

    with pytest.raises(ValueError, match="duplicate case ID"):
        run_mt_cases([case, case], lambda _text: "Bản dịch.", "fake-mt")


def test_hy_mt2_uses_official_prompt_and_only_relevant_terms(monkeypatch):
    captured = {}

    def fake_request(_url, *, method, payload):
        captured.update(payload)
        return {"message": {"content": "Bản dịch."}}

    monkeypatch.setattr("scripts.eval_mt_rework._request_json", fake_request)
    text = _ollama_translate(
        "http://localhost:11434",
        "hy-mt2-local",
        "The AI assistant is available.",
        {"LLM": "mô hình ngôn ngữ lớn", "AI assistant": "trợ lý AI", "GPU": "GPU"},
    )

    prompt = captured["messages"][0]["content"]
    assert "Translate the following text into Vietnamese." in prompt
    assert "AI assistant translates to trợ lý AI" in prompt
    assert "LLM translates to" not in prompt
    assert "GPU translates to" not in prompt
    assert "The AI assistant is available." in prompt
    assert text == "Bản dịch."


def test_hy_mt2_prompt_provenance_hash_is_immutable_and_matches_template():
    assert PROMPT_TEMPLATE_SHA256 == _prompt_template_sha256()
    assert len(PROMPT_TEMPLATE_SHA256) == 64


def test_cli_refuses_existing_output_before_loading_model(tmp_path, capsys):
    cases = tmp_path / "cases.jsonl"
    cases.write_text('{"id":"case","verified_source":"Hello"}\n', encoding="utf-8")
    output = tmp_path / "candidate.jsonl"
    output.write_text("previous result\n", encoding="utf-8")

    status = main(
        [
            "--cases",
            str(cases),
            "--engine",
            "hy-mt2",
            "--model",
            "not-loaded",
            "--out",
            str(output),
        ]
    )

    assert status == 1
    assert "exists" in capsys.readouterr().err.lower()
    assert output.read_text(encoding="utf-8") == "previous result\n"
