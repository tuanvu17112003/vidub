"""S3 Speak: ``script.json`` -> one Vietnamese WAV per segment + ``speech.json``.

Reads ``job.json`` + ``script.json``, synthesises the ``vi`` text of every
segment to ``<artifacts>/speech/seg_XXXX.wav`` and writes ``speech.json`` — a
list of ``{idx, start, end, speaker, vi, file, duration}`` records. Records
``paths["speech"]`` / ``paths["speech_dir"]`` and marks the ``speak`` stage done.

Standalone use::

    python -m vidub.stages.speak <cache>/<hash>/job.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from vidub.config import (
    DEFAULT_VOICE_MAP,
    JobConfig,
    check_tts_engine_allowed,
    validate_langs,
)
from vidub.engines.tts.base import BaseTTS, TTSError
from vidub.engines.tts.piper_tts import PiperTTS
from vidub.manifest import JobManifest
from vidub.utils.pronounce import apply_pronunciation, load_pronunciation
from vidub.utils.sentences import split_translation
from vidub.utils.text_selection import selected_text

__all__ = [
    "speak",
    "build_tts",
    "seg_filename",
    "resolve_voice",
    "pronunciation_map",
    "main",
]

log = logging.getLogger(__name__)

SCRIPT_NAME = "script.json"
SPEECH_NAME = "speech.json"
SPEECH_DIR = "speech"
UNITS_NAME = "units.json"
# Mode in which no synthesis runs; the pipeline never calls speak() then, and a
# direct/standalone call must refuse loudly instead of doing pointless work.
SUBTITLES_ONLY = "subtitles-only"
# TTS-only lexicon counters sidecar for report.json ("pronunciation": {...});
# script.json/subtitles keep the written form, so the spoken replacements are
# persisted here instead.
PRONUNCIATION_NAME = "pronunciation.json"
# Sidecar written by the references stage (clone_speakers); see stages/references.
REFERENCES_NAME = "references.json"
_PIPER_ALIASES = {"piper", "piper-tts", "piper_tts"}

# Shipped Piper vi voices (paths within rhasspy/piper-voices).
PIPER_VOICES: dict[str, str] = {
    "vais1000": "vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx",
    "25hours": "vi/vi_VN/25hours_single/low/vi_VN-25hours_single-low.onnx",
}


def resolve_voice(voice_map: dict[str, str] | None, speaker: str) -> str:
    """Voice name for ``speaker``; unknown speakers fall back to spk0's voice."""
    mapping = dict(DEFAULT_VOICE_MAP)
    mapping.update(voice_map or {})
    return mapping.get(speaker, mapping.get("spk0", "vais1000"))


def _load_clone_refs(manifest: JobManifest) -> dict[str, str | None]:
    """Map speaker -> reference WAV path (``None`` = engine default).

    Reads the ``references.json`` sidecar written by the references stage. A row
    with ``fallback: true`` (or no file) keeps the engine's default reference.
    Raises :class:`TTSError` when the sidecar is missing, so ``--clone-speakers``
    never silently dubs with the wrong voice.
    """
    path = manifest.artifacts_dir / manifest.paths.get(
        "references_json", REFERENCES_NAME
    )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise TTSError(
            "--clone-speakers is on but the references sidecar is missing or "
            f"unreadable: {path}. Run the references stage first (it follows "
            "diarize)."
        ) from exc

    refs: dict[str, str | None] = {}
    if isinstance(data, list):
        for row in data:
            if not isinstance(row, dict) or not row.get("speaker"):
                continue
            file = str(row.get("file") or "")
            if row.get("fallback") or not file:
                refs[str(row["speaker"])] = None
            else:
                refs[str(row["speaker"])] = str(manifest.artifacts_dir / file)
    return refs


class _VoiceResolver:
    """Build one TTS engine per distinct mapped voice, lazily.

    Piper voices load on first use only (so a single-speaker video never loads
    the second voice); ViXTTS maps a *file path* voice value to that reference
    WAV and falls back to the default reference otherwise. With
    ``cfg.clone_speakers`` each speaker gets its own ViXTTS engine built from the
    extracted per-speaker reference (or the engine default on a fallback row).
    """

    def __init__(self, manifest: JobManifest, cfg: JobConfig) -> None:
        self.manifest = manifest
        self.cfg = cfg
        engine = (manifest.tts_engine or "piper").strip().lower()
        self.engine_name = "piper" if engine in _PIPER_ALIASES else engine
        try:
            check_tts_engine_allowed(self.engine_name, cfg.allow_nc_models)
        except ValueError as exc:
            raise TTSError(str(exc)) from exc
        self._engines: dict[str, BaseTTS] = {}
        self._use_clones = bool(cfg.clone_speakers) and self.engine_name == "vixtts"
        self._clone_refs: dict[str, str | None] = (
            _load_clone_refs(manifest) if self._use_clones else {}
        )

    def for_speaker(self, speaker: str) -> BaseTTS:
        if self._use_clones:
            ref = self._clone_refs.get(speaker)
            key = f"clone:{ref or '<default>'}"
            if key not in self._engines:
                self._engines[key] = self._build_clone(ref)
            return self._engines[key]

        voice = (
            "default"
            if self.engine_name == "voxcpm2"
            else resolve_voice(self.cfg.voice_map, speaker)
        )
        if voice not in self._engines:
            self._engines[voice] = self._build(voice)
        return self._engines[voice]

    def _build(self, voice: str) -> BaseTTS:
        if self.engine_name == "piper":
            path = PIPER_VOICES.get(voice, PIPER_VOICES["vais1000"])
            return PiperTTS(voice_path=path)
        if self.engine_name == "vixtts":
            from vidub.engines.tts.vixtts import VixttsTTS

            ref = voice if Path(voice).is_file() else self.cfg.ref_voice
            return VixttsTTS(ref_voice=ref)
        if self.engine_name == "voxcpm2":
            from vidub.engines.tts.voxcpm2 import VoxCPM2TTS

            return VoxCPM2TTS(
                ref_voice=self.cfg.ref_voice,
                voice_profile=self.cfg.voxcpm2_voice,
            )
        raise TTSError(f"unknown tts_engine: {self.manifest.tts_engine!r}")

    def _build_clone(self, ref: str | None) -> BaseTTS:
        """ViXTTS engine for one speaker's extracted reference (``None`` default)."""
        from vidub.engines.tts.vixtts import VixttsTTS

        return VixttsTTS(ref_voice=ref or self.cfg.ref_voice)

    def close(self) -> None:
        for engine in self._engines.values():
            close = getattr(engine, "close", None)
            if callable(close):
                close()


def build_tts(manifest: JobManifest, cfg: JobConfig) -> BaseTTS:
    """Instantiate the TTS engine named by ``manifest.tts_engine``.

    Enforces the non-commercial license guard (``cfg.allow_nc_models``) first.
    ViXTTS is imported lazily so the default Piper path never touches the
    optional coqui-tts stack.
    """
    engine = (manifest.tts_engine or "piper").strip().lower()
    normalized = "piper" if engine in _PIPER_ALIASES else engine
    try:
        check_tts_engine_allowed(normalized, cfg.allow_nc_models)
    except ValueError as exc:
        raise TTSError(str(exc)) from exc

    if normalized == "piper":
        return PiperTTS()
    if normalized == "vixtts":
        from vidub.engines.tts.vixtts import VixttsTTS

        return VixttsTTS(ref_voice=cfg.ref_voice)
    if normalized == "voxcpm2":
        from vidub.engines.tts.voxcpm2 import VoxCPM2TTS

        return VoxCPM2TTS(
            ref_voice=cfg.ref_voice,
            voice_profile=cfg.voxcpm2_voice,
        )
    raise TTSError(f"unknown tts_engine: {manifest.tts_engine!r}")


def seg_filename(idx: int) -> str:
    """Per-segment WAV filename, e.g. ``seg_0000.wav``."""
    return f"seg_{idx:04d}.wav"


def pronunciation_map(cfg: JobConfig) -> dict[str, str]:
    """The TTS pronunciation lexicon for ``cfg`` (``{}`` when unset).

    Loaded once per stage and shared with the timing stage's shortening retry so
    both speak the acronyms the same way. The written forms in ``script.json`` /
    subtitles are never changed — the mapping only feeds :func:`synth`.
    """
    if not cfg.pronounce_path:
        return {}
    return load_pronunciation(cfg.pronounce_path)


def _check_mode(mode: str) -> None:
    if mode == SUBTITLES_ONLY:
        raise RuntimeError(
            f"skipped: mode {mode!r} does not run the speak stage"
        )


def _spoken_texts(script: list[dict], manifest: JobManifest) -> list[str]:
    """Resolve member speech text, projecting explicit sentence-unit text."""
    texts = [selected_text(row, "spoken_vi") for row in script]
    units_path = manifest.artifacts_dir / manifest.paths.get("units", UNITS_NAME)
    try:
        doc = json.loads(units_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return texts
    entries = doc.get("units") if isinstance(doc, dict) else None
    if not isinstance(entries, list) or not entries:
        return texts

    parsed: list[tuple[list[int], dict]] = []
    seen: list[int] = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("indices"), list):
            return texts
        try:
            members = [int(index) for index in entry["indices"]]
        except (TypeError, ValueError):
            return texts
        if not members or any(index < 0 or index >= len(script) for index in members):
            return texts
        parsed.append((members, entry))
        seen.extend(members)
    if sorted(seen) != list(range(len(script))):
        return texts

    for members, entry in parsed:
        raw_text = entry.get("spoken_vi")
        if not isinstance(raw_text, str) or not raw_text.strip():
            continue  # old unit files keep their script-row speech unchanged
        counts = [
            sum(1 for char in str(script[index].get("en", "")) if not char.isspace())
            for index in members
        ]
        total = sum(counts)
        shares = (
            tuple(count / total for count in counts)
            if total
            else tuple(1.0 / len(members) for _ in members)
        )
        pieces = split_translation(raw_text, shares)
        if len(pieces) == len(members):
            for index, piece in zip(members, pieces, strict=True):
                texts[index] = piece
    return texts


def speak(
    manifest_path: str,
    cfg: JobConfig,
    tts: BaseTTS | None = None,
) -> JobManifest:
    """Synthesise every script segment and write ``speech.json``.

    ``tts`` may be supplied to override the engine (used by tests); otherwise the
    engine named in the manifest is used.
    """
    manifest = JobManifest.load(manifest_path)
    _check_mode(manifest.mode)
    artifacts_dir = manifest.artifacts_dir

    script_rel = manifest.paths.get("script", SCRIPT_NAME)
    script_path = artifacts_dir / script_rel
    if not script_path.exists():
        raise FileNotFoundError(
            f"script not found: {script_path}. Run the transcribe and translate "
            f"stages first, e.g. 'python -m vidub.stages.translate {manifest_path}'."
        )

    script = json.loads(script_path.read_text(encoding="utf-8"))

    speech_dir = artifacts_dir / SPEECH_DIR
    speech_dir.mkdir(parents=True, exist_ok=True)

    engine = tts  # single override engine (tests); otherwise resolve per speaker
    resolver = None if tts is not None else _VoiceResolver(manifest, cfg)

    # TTS-only pronunciation lexicon: the written ``vi`` text in script.json /
    # subtitles is untouched; only the string handed to the engine is phonetically
    # rewritten (M3-C3).
    pronunciation = pronunciation_map(cfg)
    replacements = 0
    spoken_texts = _spoken_texts(script, manifest)

    speech: list[dict[str, object]] = []
    try:
        for idx, seg in enumerate(script):
            vi_text = spoken_texts[idx].strip()
            speaker = str(seg.get("speaker") or "spk0")
            filename = seg_filename(idx)
            out_wav = speech_dir / filename
            active = engine if engine is not None else resolver.for_speaker(speaker)
            tts_text, spoken = apply_pronunciation(vi_text, pronunciation)
            replacements += spoken
            try:
                duration = float(active.synth(tts_text, str(out_wav)))
            except Exception as exc:  # noqa: BLE001 - abort loudly on any failure
                # Never skip a failed segment: an incomplete speech.json would let
                # the timing stage build a half-empty dub. Abort the whole stage.
                raise TTSError(
                    f"TTS failed for segment {idx} ({filename}): {exc}. "
                    "Fix the TTS engine/model and re-run the speak stage."
                ) from exc
            speech.append(
                {
                    "idx": idx,
                    "start": float(seg.get("start", 0.0)),
                    "end": float(seg.get("end", 0.0)),
                    "speaker": speaker,
                    "vi": vi_text,
                    "file": f"{SPEECH_DIR}/{filename}",
                    "duration": duration,
                }
            )
    finally:
        if resolver is not None:
            resolver.close()

    (artifacts_dir / SPEECH_NAME).write_text(
        json.dumps(speech, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Always write the sidecar (zeros when no lexicon) so a later lexicon-free
    # re-run cannot inherit stale counts from a previous run.
    (artifacts_dir / PRONUNCIATION_NAME).write_text(
        json.dumps({"replacements": replacements}, ensure_ascii=False),
        encoding="utf-8",
    )

    manifest.paths["speech"] = SPEECH_NAME
    manifest.paths["speech_dir"] = SPEECH_DIR
    manifest.paths["pronunciation"] = PRONUNCIATION_NAME
    manifest.stages["speak"] = "done"
    manifest.save(artifacts_dir / "job.json")
    return manifest


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``python -m vidub.stages.speak <job.json>``."""
    parser = argparse.ArgumentParser(
        prog="vidub.stages.speak",
        description="S3 speak: script.json -> per-segment Vietnamese WAVs.",
    )
    parser.add_argument("job", help="path to the job.json manifest")
    parser.add_argument(
        "--cache-dir", default=None, help="cache root (default: manifest's)"
    )
    parser.add_argument(
        "--allow-nc-models",
        action="store_true",
        help="allow non-commercial TTS weights (e.g. ViXTTS / CPML)",
    )
    parser.add_argument(
        "--ref-voice", default=None, help="reference WAV for voice cloning (ViXTTS)"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = JobConfig(
        allow_nc_models=args.allow_nc_models, ref_voice=args.ref_voice
    )
    if args.cache_dir:
        cfg.cache_dir = args.cache_dir

    try:
        # Shared language guardrail: a standalone speak run must not dub a
        # target language the configured engine cannot speak.
        loaded = JobManifest.load(args.job)
        validate_langs(loaded.src_lang, loaded.tgt_lang, loaded.tts_engine)
        manifest = speak(args.job, cfg)
    except (FileNotFoundError, TTSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(str(manifest.artifacts_dir / SPEECH_NAME))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
