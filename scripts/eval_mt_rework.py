"""Run reproducible, ID-matched En-to-Vi translation comparisons."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from scripts.eval_asr_rework import _GpuMemorySampler

__all__ = ["run_mt_cases", "main"]

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
PROMPT_TEMPLATE_VERSION = "hy-mt2-en-vi-v1"
DEFAULT_PROMPT_TEMPLATE = (
    "Translate the following text into Vietnamese. Note that you should only "
    "output the translated result without any additional explanation:\n\n{source_text}"
)
TERMINOLOGY_PROMPT_TEMPLATE = (
    "Reference the following translations:\n{terms}\n\n" + DEFAULT_PROMPT_TEMPLATE
)


def _prompt_template_sha256() -> str:
    templates = DEFAULT_PROMPT_TEMPLATE + "\n---TERMINOLOGY---\n" + TERMINOLOGY_PROMPT_TEMPLATE
    return hashlib.sha256(templates.encode("utf-8")).hexdigest()


PROMPT_TEMPLATE_SHA256 = _prompt_template_sha256()


def run_mt_cases(
    cases: list[dict], translate: Callable[[str], str], engine_id: str
) -> list[dict]:
    """Translate each case exactly once while keeping the verified source join."""
    rows: list[dict] = []
    seen: set[str] = set()
    for position, case in enumerate(cases):
        case_id = str(case.get("id", "")).strip()
        if not case_id:
            raise ValueError(f"case {position} has no ID")
        if case_id in seen:
            raise ValueError(f"duplicate case ID {case_id!r}")
        seen.add(case_id)
        if "verified_source" in case:
            source = str(case.get("verified_source") or "")
            input_status = "verified_source"
        else:
            source = str(case.get("source_asr") or "")
            input_status = "source_asr_unverified"
        if not source.strip():
            raise ValueError(f"source text is blank for case {case_id}")

        began = time.perf_counter()
        try:
            text = translate(source)
        except Exception as exc:
            raise RuntimeError(f"translation failed for case {case_id}: {exc}") from exc
        elapsed = time.perf_counter() - began
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"translation returned blank text for case {case_id}")
        rows.append(
            {
                "id": case_id,
                "engine": engine_id,
                "text": text.strip(),
                "elapsed_seconds": elapsed,
                "input_status": input_status,
                "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            }
        )
    return rows


def _read_cases(path: Path) -> list[dict]:
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read cases from {path}: {exc}") from exc
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"expected one or more JSON objects in {path}")
    return rows


def _write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(data: dict, path: Path) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(data, output, ensure_ascii=False, indent=2)
        output.write("\n")


def _package_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _load_envit5(model_name: str):
    from huggingface_hub import snapshot_download

    from vidub.engines.mt.envit5_mt import Envit5Translator

    try:
        local_path = snapshot_download(model_name, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"envit5 model {model_name!r} is not cached locally; refusing a network download"
        ) from exc
    engine = Envit5Translator(model_name=local_path)
    return engine, {
        "model": model_name,
        "checkpoint_revision": Path(local_path).name,
        "settings": {"source_language": "en", "target_language": "vi"},
        "runtime_packages": _package_versions(("torch", "transformers", "sentencepiece", "huggingface-hub")),
    }


def _request_json(url: str, *, method: str = "GET", payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            body = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"local Ollama request failed at {url}: {exc}") from exc
    try:
        result = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Ollama returned invalid JSON at {url}: {exc}") from exc
    if not isinstance(result, dict):
        raise RuntimeError(f"Ollama returned a non-object response at {url}")
    return result


def _ollama_model_digest(host: str, model_name: str) -> str | None:
    tags = _request_json(f"{host.rstrip('/')}/api/tags")
    models = tags.get("models")
    if not isinstance(models, list):
        return None
    for model in models:
        if isinstance(model, dict) and model.get("name") == model_name:
            return str(model.get("digest") or "") or None
    raise RuntimeError(
        f"Ollama model {model_name!r} is not installed locally; refusing to pull it"
    )


def _relevant_glossary(source: str, glossary: dict[str, str]) -> dict[str, str]:
    if not glossary:
        return {}
    from vidub.utils.glossary import find_source_terms

    hits = find_source_terms(source, glossary, max_terms=len(glossary))
    return {term: target for term, target, _start, _end in hits}


def _hy_mt2_prompt(source: str, glossary: dict[str, str]) -> str:
    """Build the official English-to-target and terminology prompt from Hy-MT2."""
    if glossary:
        terms = "\n".join(
            f"{term} translates to {translation}"
            for term, translation in glossary.items()
        )
        return TERMINOLOGY_PROMPT_TEMPLATE.format(terms=terms, source_text=source)
    return DEFAULT_PROMPT_TEMPLATE.format(source_text=source)


def _ollama_translate(host: str, model_name: str, source: str, glossary: dict[str, str]) -> str:
    prompt = _hy_mt2_prompt(source, _relevant_glossary(source, glossary))
    response = _request_json(
        f"{host.rstrip('/')}/api/chat",
        method="POST",
        payload={
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.0, "top_p": 1.0, "repeat_penalty": 1.0, "num_predict": 512},
        },
    )
    message = response.get("message")
    if not isinstance(message, dict):
        return ""
    return str(message.get("content") or "")


def _glossary_sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--engine", choices=("envit5", "hy-mt2"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--glossary", type=Path, help="optional UTF-8 project glossary TSV")
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = _build_parser().parse_args(raw_argv)
    sidecar = args.out.with_suffix(".run.json")
    if args.out.exists() or sidecar.exists():
        print(f"error: output or run metadata already exists: {args.out} / {sidecar}", file=sys.stderr)
        return 1
    try:
        cases = _read_cases(args.cases)
        glossary: dict[str, str] = {}
        if args.glossary:
            from vidub.utils.glossary import load_glossary

            glossary = load_glossary(args.glossary)
        args.out.parent.mkdir(parents=True, exist_ok=True)

        if args.engine == "envit5":
            model, metadata = _load_envit5(args.model)

            def translate(source: str) -> str:
                translated = model.translate([source], src_lang="en", tgt_lang="vi")
                return translated[0] if translated else ""

        else:
            host = args.ollama_host.rstrip("/")
            digest = _ollama_model_digest(host, args.model)
            metadata = {
                "model": args.model,
                "checkpoint_revision": digest,
                "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                "prompt_template_sha256": PROMPT_TEMPLATE_SHA256,
                "glossary_sha256": _glossary_sha256(args.glossary),
                "settings": {
                    "source_language": "English",
                    "target_language": "Vietnamese",
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "repeat_penalty": 1.0,
                    "num_predict": 512,
                },
                "ollama_host": host,
                "runtime_packages": {},
            }

            def translate(source: str) -> str:
                return _ollama_translate(host, args.model, source, glossary)

        with _GpuMemorySampler() as gpu:
            rows = run_mt_cases(cases, translate, args.engine)
        run = {
            **metadata,
            "engine": args.engine,
            "command": [sys.executable, str(Path(__file__).resolve()), *raw_argv],
            "case_count": len(rows),
            "eligible_source_count": sum(row["input_status"] == "verified_source" for row in rows),
            "elapsed_seconds": sum(row["elapsed_seconds"] for row in rows),
            "peak_gpu_memory_mib": gpu.peak_mib,
            "gpu_memory_status": "available" if gpu.peak_mib is not None else "unavailable",
        }
        _write_jsonl(rows, args.out)
        _write_json(run, sidecar)
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {len(rows)} MT result(s): {args.out}")
    print(f"wrote run metadata: {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
