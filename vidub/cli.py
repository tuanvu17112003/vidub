"""Command-line interface for ViDub.

``vidub doctor`` checks the local environment; ``vidub run`` orchestrates the
full dubbing pipeline (see :mod:`vidub.pipeline`).
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import logging
import platform
import shutil
import sys
import urllib.request
from pathlib import Path

from vidub import __version__
from vidub.options import OPTION_SPECS, OptionSpec
from vidub.config import (
    ASR_PRESETS,
    DEFAULT_HUNYUAN_HOST,
    DEFAULT_HUNYUAN_MODEL,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    TTS_ENGINES,
    JobConfig,
    SUBTITLES_ONLY,
    check_mt_engine,
    check_tts_engine_allowed,
    normalize_mode,
    resolve_asr_model,
    validate_clone_speakers,
)
from vidub.utils.cuda import cuda_available, gpu_whisper_available, setup_cuda_dlls
from vidub.utils.proc import find_tool
from vidub.utils.ytdlp import download, is_url

__all__ = ["build_parser", "main"]

log = logging.getLogger(__name__)


def _free_disk_gb(path: Path) -> float:
    """Return free space on the volume holding ``path``, in GiB."""
    return shutil.disk_usage(path).free / (1024**3)


def _parse_voice_map(raw: str | None) -> dict[str, str]:
    """Parse a ``--voice-map`` JSON object; raises ``ValueError`` when invalid."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--voice-map is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(
            '--voice-map must be a JSON object, e.g. {"spk1": "25hours"}'
        )
    return {str(key): str(value) for key, value in parsed.items()}


def _parse_force(raw: str | None) -> list[str]:
    """Split a ``--force stage[,stage]`` value into a clean list."""
    return [name for name in (raw.split(",") if raw else []) if name.strip()]


def _build_cfg(args: argparse.Namespace) -> JobConfig:
    """Build a :class:`JobConfig` from parsed args, validating model/engine choices."""
    resolve_asr_model(args.asr_model)  # raises ValueError on an unknown model
    check_mt_engine(args.mt_engine)  # raises ValueError on an unknown engine
    mode = normalize_mode(args.mode)  # canonical spelling (subtitles -> subtitles-only)
    # TTS is irrelevant in subtitles-only (no synthesis runs), so an unusable TTS
    # config must not block that mode; dub/voice-over keep the license guard.
    if mode != SUBTITLES_ONLY:
        check_tts_engine_allowed(args.tts_engine, args.allow_nc_models)
    kwargs: dict[str, object] = {}
    for spec in OPTION_SPECS:
        if not spec.flag:
            continue  # UI-only / programmatic-only fields keep their dataclass default
        raw = getattr(args, spec.dest or spec.field)
        if spec.transform == "voice_map":
            kwargs[spec.field] = _parse_voice_map(raw)
        else:
            kwargs[spec.field] = raw
    # ``mode`` is normalized again by JobConfig.__post_init__; passing the raw
    # spelling keeps the CLI's validation above as the single fail-fast point.
    kwargs["mode"] = mode
    cfg = JobConfig(**kwargs)
    if args.cache_dir:
        cfg.cache_dir = args.cache_dir
    # Cross-flag guardrails (cloning needs diarize + vixtts + NC opt-in).
    validate_clone_speakers(cfg)
    return cfg


def _extras_status(label: str, module: str, dist: str) -> str:
    """One ``vidub doctor`` line for an optional extra (never raises).

    Probes availability with ``importlib`` and reports the installed distribution
    version when metadata is available, so an absent/broken package can never
    crash the doctor. Format: ``"<label>: yes (<dist> <version>)"`` or
    ``"<label>: no"``.
    """
    try:
        found = importlib.util.find_spec(module) is not None
    except Exception:  # noqa: BLE001 - a broken package must not crash doctor
        found = False
    if not found:
        return f"{label}: no"
    try:
        version = importlib.metadata.version(dist)
    except Exception:  # noqa: BLE001 - installed but no distribution metadata
        version = "unknown"
    return f"{label}: yes ({dist} {version})"


def _llm_server_status(label: str, host: str, model: str, note: str = "") -> str:
    """One never-raising ``vidub doctor`` line about a local Ollama server.

    ``GET <host>/api/version`` (2s timeout) proves the server is up; when it is,
    ``GET <host>/api/tags`` says whether ``model`` is pulled. Any failure is
    reported as text, never raised, so a missing server can't crash the doctor.
    ``note`` is appended to a reachable line (e.g. a model-license reminder).
    """
    base = (host or DEFAULT_OLLAMA_HOST).rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/api/version", timeout=2.0) as response:
            version = str(
                json.loads(response.read().decode("utf-8")).get("version", "?")
            )
    except Exception:  # noqa: BLE001 - an absent server must not crash doctor
        return f"{label}: not reachable (start it with 'ollama serve')"

    present = "unknown"
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=2.0) as response:
            tags = json.loads(response.read().decode("utf-8"))
        names = {
            str(item.get("name", ""))
            for item in (tags.get("models") or [])
            if isinstance(item, dict)
        }
        present = "present" if (model in names or f"{model}:latest" in names) else "missing"
    except Exception:  # noqa: BLE001 - version worked; tags is best-effort
        present = "unknown"
    return f"{label}: {version} at {host} (model {model}: {present}){note}"


def _ollama_status(host: str, model: str) -> str:
    """One never-raising ``vidub doctor`` line about the local Ollama server."""
    return _llm_server_status("ollama", host, model)


def _hunyuan_status(host: str, model: str) -> str:
    """One never-raising doctor line about the local Hunyuan HY-MT1.5 model."""
    return _llm_server_status(
        "hunyuan",
        host,
        model,
        note=" — Tencent HY Community License (opt-in; not valid in EU/UK/South Korea)",
    )


def cmd_doctor(args: argparse.Namespace) -> int:
    """Print an environment report. Always succeeds (exit code 0)."""
    print(f"ViDub {__version__} — environment doctor")
    print("-" * 48)

    py_version = platform.python_version()
    py_ok = sys.version_info >= (3, 10)
    print(f"python      : {py_version} {'OK' if py_ok else 'NG (< 3.10)'}")

    ffmpeg = find_tool("ffmpeg")
    print(f"ffmpeg      : {'found ' + ffmpeg if ffmpeg else 'missing'}")

    ffprobe = find_tool("ffprobe")
    print(f"ffprobe     : {'found ' + ffprobe if ffprobe else 'missing'}")

    nvidia_smi = find_tool("nvidia-smi")
    if nvidia_smi:
        print(f"nvidia-smi  : found {nvidia_smi}")
    else:
        print("nvidia-smi  : missing (no NVIDIA GPU detected — CPU fallback)")

    # CUDA report: the DLL wiring status plus two best-effort probes. Every call
    # is guarded so a broken/absent CUDA install can never crash the doctor.
    try:
        dll_status = setup_cuda_dlls()
    except Exception as exc:  # noqa: BLE001 - doctor must never crash
        dll_status = f"unknown ({exc})"
    print(f"cuda dlls: {dll_status}")

    try:
        cuda_ok, cuda_reason = cuda_available()
    except Exception as exc:  # noqa: BLE001 - doctor must never crash
        cuda_ok, cuda_reason = False, f"probe failed: {exc}"
    print(f"cuda available: {'yes' if cuda_ok else 'no'} ({cuda_reason})")

    try:
        gpu_ok, gpu_reason = gpu_whisper_available()
    except Exception as exc:  # noqa: BLE001 - doctor must never crash
        gpu_ok, gpu_reason = False, f"probe failed: {exc}"
    print(f"gpu whisper: {'yes' if gpu_ok else 'no'} ({gpu_reason})")

    # ASR presets + licenses, so the matrix covers ASR as well as TTS/diarize.
    try:
        preset_bits = []
        for name, model in ASR_PRESETS.items():
            license_name = (
                "BSD-3-Clause" if model.startswith("vinai/PhoWhisper") else "MIT"
            )
            preset_bits.append(f"{name}: {model} [{license_name}]")
        print(f"asr presets : {'; '.join(preset_bits)}")
    except Exception as exc:  # noqa: BLE001 - doctor must never crash
        print(f"asr presets : unknown ({exc})")

    # TTS engines + licenses (the license guardrail from DESIGN.md §4.2).
    try:
        engine_bits = []
        for name, spec in TTS_ENGINES.items():
            status = "opt-in (--allow-nc-models)" if spec["nc"] else "active"
            engine_bits.append(f"{name} [{spec['license']}] {status}")
        print(f"tts engines: {'; '.join(engine_bits)}")
    except Exception as exc:  # noqa: BLE001 - doctor must never crash
        print(f"tts engines: unknown ({exc})")

    # Optional extras: URL input (yt-dlp) and the local web UI (gradio). Probed
    # via importlib so an absent/broken package can never crash the doctor.
    print(_extras_status("youtube", "yt_dlp", "yt-dlp"))
    print(_extras_status("ui", "gradio", "gradio"))

    # Local LLM MT engine (M4-T5). Best-effort probe of the Ollama server; the
    # doctor must never raise even when the server is down.
    try:
        print(_ollama_status(DEFAULT_OLLAMA_HOST, DEFAULT_OLLAMA_MODEL))
    except Exception as exc:  # noqa: BLE001 - doctor must never crash
        print(f"ollama: unknown ({exc})")

    # Hunyuan HY-MT1.5-7B translation specialist (M5-T1): opt-in, served by the
    # same Ollama server. Also best-effort; never raise when the server is down.
    try:
        print(_hunyuan_status(DEFAULT_HUNYUAN_HOST, DEFAULT_HUNYUAN_MODEL))
    except Exception as exc:  # noqa: BLE001 - doctor must never crash
        print(f"hunyuan: unknown ({exc})")

    # Per-clip TTS QC scorer (M4-T7): only check that NISQA imports; never
    # download weights or construct the model here.
    try:
        from vidub.utils.ttsqc import nisqa_available

        ready, version = nisqa_available()
        print(
            f"tts-qc: NISQA ready (torchmetrics {version})"
            if ready
            else "tts-qc: unavailable (pip install 'torchmetrics[audio]')"
        )
    except Exception:  # noqa: BLE001 - doctor must never crash
        print("tts-qc: unavailable (pip install 'torchmetrics[audio]')")

    try:
        free_gb = _free_disk_gb(Path.cwd())
        drive = Path.cwd().drive or str(Path.cwd())
        print(f"free disk   : {free_gb:.1f} GB ({drive})")
    except OSError as exc:  # pragma: no cover - platform specific
        print(f"free disk   : unknown ({exc})")

    print("-" * 48)
    if not ffmpeg:
        print("hint: ffmpeg is required — run tools/install_ffmpeg.ps1 (or .sh)")
    print("doctor complete")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run the full dubbing pipeline. Exit 0 only when every stage is done."""
    # Imported lazily so `vidub doctor` stays fast and dependency-light.
    from vidub.pipeline import required_stages, run_pipeline
    from vidub.stages import diarize as diarize_mod
    from vidub.stages import ingest as ingest_mod

    if not args.input:
        print("error: an input video is required", file=sys.stderr)
        return 1

    # A URL is downloaded by the pipeline; a local path must already exist.
    url_input = is_url(args.input)
    video = Path(args.input)
    if not url_input and not video.exists():
        print(f"error: input video not found: {video}", file=sys.stderr)
        return 1
    pipeline_input = args.input if url_input else str(video)

    try:
        cfg = _build_cfg(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # Diarization preflight runs *before* --clean: a --diarize run without
    # credentials must fail fast without wiping any prior artifacts.
    if cfg.diarize:
        try:
            diarize_mod.preflight(cfg)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    if args.clean:
        # A URL has no local hash until downloaded; resolve it first so --clean
        # still targets the cache directory of the downloaded file.
        source = download(args.input, cfg.cache_dir) if url_input else video
        cache_path = Path(cfg.cache_dir) / ingest_mod.video_hash(str(source))
        log.info("cleaning %s", cache_path)
        shutil.rmtree(cache_path, ignore_errors=True)

    try:
        report = run_pipeline(
            pipeline_input,
            cfg,
            resume=args.resume,
            force=_parse_force(args.force),
            burn=args.burn,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.report:
        print(json.dumps(report, indent=2, ensure_ascii=False))

    required = required_stages(
        cfg.mode, cfg.diarize, cfg.clone_speakers
    )
    stages = report.get("stages", {})
    all_done = all(stages.get(name, {}).get("status") == "done" for name in required)
    output_ok = Path(report.get("outputs", {}).get("dubbed", "")).exists()
    if not (all_done and output_ok):
        print("error: pipeline did not produce every required output", file=sys.stderr)
        return 1

    # Final QC gate: a hard "fail" verdict makes the run exit non-zero, with the
    # machine-readable reasons on stderr. "unknown"/"review"/missing all pass.
    quality = report.get("quality") or {}
    if quality.get("verdict") == "fail":
        for reason in quality.get("reasons") or []:
            print(f"error: quality check failed: {reason}", file=sys.stderr)
        if not quality.get("reasons"):
            print("error: quality check failed", file=sys.stderr)
        return 1
    return 0


def _cmd_batch_url(args: argparse.Namespace, cli_batch_mod) -> int:
    """``vidub batch <URL>``: download the URL once and process it as one item.

    Only a direct URL is supported as the batch argument (a ``.txt`` list file
    is out of scope). A ``batch-summary.json`` with a single entry is written to
    ``--out-dir`` (default: the current directory).
    """
    from vidub.pipeline import run_pipeline

    try:
        cfg = _build_cfg(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        local = download(args.folder, cfg.cache_dir)
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    log.info("batch: downloaded %s -> %s", args.folder, local)

    try:
        report = run_pipeline(
            args.folder,
            cfg,
            resume=args.resume,
            force=_parse_force(args.force),
            burn=args.burn,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    entry = {
        "input": args.folder,
        "status": "done",
        "duration": report.get("duration"),
        "stages": {
            key: value.get("status")
            for key, value in (report.get("stages") or {}).items()
        },
        "output": (report.get("outputs") or {}).get("dubbed"),
    }
    summary = {
        "folder": args.folder,
        "pattern": "url",
        "totals": {"total": 1, "done": 1, "failed": 0, "skipped": 0, "interrupted": 0},
        "videos": [entry],
    }
    summary_dir = Path(args.out_dir) if args.out_dir else Path.cwd()
    target = cli_batch_mod.write_summary(
        cli_batch_mod.summary_path(summary_dir), summary
    )
    print(f"batch: 1 done, 0 failed, 0 skipped -> {target}")
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    """Dub every matching video in a folder, sequentially.

    Exit 0 when all videos succeeded, 1 when any failed (summary still written),
    130 on Ctrl-C (partial summary written).
    """
    from vidub import cli_batch

    if is_url(args.folder):
        # A direct URL is a batch of one; a .txt list file is not supported.
        return _cmd_batch_url(args, cli_batch)

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"error: batch folder not found: {folder}", file=sys.stderr)
        return 1

    try:
        cfg = _build_cfg(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    summary, code = cli_batch.run_batch(
        folder,
        cfg,
        pattern=args.pattern,
        out_dir=args.out_dir,
        resume_batch=args.resume_batch,
        resume=args.resume,
        force=_parse_force(args.force),
        burn=args.burn,
    )
    totals = summary.get("totals", {})
    print(
        f"batch: {totals.get('done', 0)} done, {totals.get('failed', 0)} failed, "
        f"{totals.get('skipped', 0)} skipped -> "
        f"{cli_batch.summary_path(folder, args.out_dir)}"
    )
    return code


def cmd_ui(args: argparse.Namespace) -> int:
    """Launch the local Gradio web UI. Needs the optional ``[ui]`` extra."""
    try:
        from vidub.ui import launch_ui
    except ImportError as exc:  # gradio (or a dep) is missing
        print(
            "error: gradio is not installed; the web UI needs the optional "
            '\'ui\' extra. Install it with: pip install -e ".[ui]" '
            f"({exc})",
            file=sys.stderr,
        )
        return 1

    log.info("serving ViDub UI on http://%s:%s (localhost only)", args.host, args.port)
    launch_ui(host=args.host, port=args.port)
    return 0


def _add_shared_pipeline_args(p: argparse.ArgumentParser) -> None:
    """Flags shared by ``vidub run`` and ``vidub batch``.

    Generated from :data:`vidub.options.OPTION_SPECS` (the single option
    registry, spec 2026-10-25 D2). Flag spellings, dests, defaults, aliases
    and help text reproduce the pre-registry argparse byte-for-byte; run
    controls that are not ``JobConfig`` options (--resume/--force/--burn)
    stay hand-written and render where they historically sat.
    """
    for spec in OPTION_SPECS:
        if not spec.flag:
            continue
        if spec.field == "sub_position":
            # Historical help order: the run controls sat between --pronounce
            # and --sub-pos; keep that order.
            _add_run_controls(p)
        _add_spec_args(p, spec)
    p.add_argument(
        "--cache-dir", default=None, help="cache root (default: .vidub-cache)"
    )


def _add_spec_args(p: argparse.ArgumentParser, spec) -> None:
    """Register one :class:`~vidub.options.OptionSpec` on an argparse parser."""
    dest = spec.dest or spec.field
    names = [spec.flag, *spec.aliases]
    default = spec.default if spec.cli_default is OptionSpec._UNSET else spec.cli_default
    if spec.cli_action == "opt":
        p.add_argument(
            *names,
            dest=dest,
            action=argparse.BooleanOptionalAction,
            default=default,
            help=spec.help,
        )
    elif spec.cli_action == "true":
        p.add_argument(*names, dest=dest, action="store_true", help=spec.help)
    elif spec.cli_action == "false":
        p.add_argument(*names, dest=dest, action="store_false", default=True, help=spec.help)
    else:
        kwargs: dict[str, object] = {"default": default, "help": spec.help}
        if spec.metavar:
            kwargs["metavar"] = spec.metavar
        if spec.cli_type is not None:
            kwargs["type"] = spec.cli_type
        elif spec.kind in ("float", "int"):
            kwargs["type"] = float if spec.kind == "float" else int
        if spec.choices and spec.cli_type is None:
            kwargs["choices"] = list(spec.choices)
        p.add_argument(*names, dest=dest, **kwargs)
    for alias in spec.suppressed_aliases:
        p.add_argument(
            alias, dest=dest, action="store_false", help=argparse.SUPPRESS
        )


def _add_run_controls(p: argparse.ArgumentParser) -> None:
    """Resume/force/burn: run controls, not ``JobConfig`` options."""
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="skip stages already done (default: on)",
    )
    p.add_argument(
        "--force",
        default=None,
        metavar="STAGE[,STAGE]",
        help="re-run these pipeline stages",
    )
    p.add_argument(
        "--burn", action="store_true", help="burn subtitles into the video"
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="vidub",
        description="ViDub — video translation + Vietnamese dubbing (in development).",
    )
    parser.add_argument(
        "--version", action="version", version=f"vidub {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_doctor = subparsers.add_parser(
        "doctor", help="check the local environment (python, ffmpeg, GPU, disk)"
    )
    p_doctor.set_defaults(func=cmd_doctor)

    p_run = subparsers.add_parser("run", help="run the dubbing pipeline")
    p_run.add_argument("input", nargs="?", help="input video file")
    _add_shared_pipeline_args(p_run)
    p_run.add_argument(
        "--clean", action="store_true", help="wipe the video's cache dir first"
    )
    p_run.add_argument(
        "--report", action="store_true", help="print report.json at the end"
    )
    p_run.set_defaults(func=cmd_run)

    p_batch = subparsers.add_parser(
        "batch", help="dub every video in a folder (sequential)"
    )
    p_batch.add_argument("folder", help="folder containing the input videos")
    p_batch.add_argument(
        "--pattern",
        default="*.mp4",
        help="glob of files to process (default: *.mp4)",
    )
    p_batch.add_argument(
        "--out-dir",
        default=None,
        help="where to write batch-summary.json (default: the input folder)",
    )
    p_batch.add_argument(
        "--resume-batch",
        action="store_true",
        help="skip videos whose batch-summary entry is already 'done'",
    )
    _add_shared_pipeline_args(p_batch)
    p_batch.set_defaults(func=cmd_batch)

    p_ui = subparsers.add_parser(
        "ui", help="launch the local web UI (needs the optional [ui] extra)"
    )
    p_ui.add_argument(
        "--port", type=int, default=7860, help="port to serve on (default: 7860)"
    )
    p_ui.add_argument(
        "--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)"
    )
    p_ui.set_defaults(func=cmd_ui)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
