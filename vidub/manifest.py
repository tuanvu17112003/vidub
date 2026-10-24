"""Job manifest: the single source of truth for a ViDub job.

Each job writes a plain-JSON ``job.json`` into its per-video cache directory
(``<cache_dir>/<video_hash>/``). The file is intentionally dependency-free so it
can be read/inspected without ViDub installed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

__all__ = ["Segment", "JobManifest"]

_SEGMENT_FIELDS = {"start", "end", "text", "speaker", "words", "meta"}


@dataclass
class Segment:
    """One timed unit of speech produced by the ASR stage.

    ``words`` carries optional word-level timing as
    ``[{"start": float, "end": float, "word": str}, ...]`` (``None`` when the
    engine did not produce them). Everything downstream must tolerate ``None``.

    ``meta`` carries optional per-segment confidence signals from the ASR
    engine (faster-whisper: ``avg_logprob`` / ``compression_ratio`` /
    ``no_speech_prob`` / ``temperature``); ``None`` when the engine exposes
    none (e.g. the PhoWhisper path). The Level-1 transcript QC layer
    (:mod:`vidub.utils.qc`) consumes it and must tolerate its absence.
    """

    start: float
    end: float
    text: str
    speaker: str = "spk0"
    words: list[dict[str, Any]] | None = None
    meta: dict[str, Any] | None = None


@dataclass
class JobManifest:
    """Everything a stage needs to know about a job.

    ``paths`` maps a logical artifact name (``"audio"``, ``"srt"`` …) to a path
    relative to :attr:`artifacts_dir`. ``stages`` records per-stage status
    (``"done"`` / ``"running"`` / …) so the pipeline can resume.
    """

    input_path: str = ""
    # Original http(s) URL when the input came from yt-dlp (``vidub run <URL>``);
    # empty for a plain local file. ``input_path`` always points at the local
    # media file the pipeline actually ingested.
    source_url: str = ""
    video_hash: str = ""
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    src_lang: str = "en"
    tgt_lang: str = "vi"
    mode: str = "dub"
    asr_model: str = "base"
    # Resolved ASR device the transcribe stage actually used ("cpu"/"cuda");
    # empty when no ASR run has recorded it yet (or the engine reports none).
    asr_device: str = ""
    # Resolved device the mix stage ran Demucs on ("cpu"/"cuda"); empty before a
    # mix run. Recorded so report.json can show where separation happened.
    sep_device: str = ""
    # Fraction of the spoken time slots actually covered by dubbed audio,
    # recorded by the timing stage (0.0-1.0); ``None`` before timing ran (or for
    # modes that skip it). See ``vidub.stages.timing``.
    speech_coverage: float | None = None
    # Number of segments the timing stage time-stretched (stretch > 1.0),
    # recorded for ``report.json``'s ``pacing`` block (M3-C1).
    stretched_segments: int = 0
    mt_engine: str = "envit5"
    tts_engine: str = "piper"
    cache_dir: str = ".vidub-cache"
    paths: dict[str, str] = field(default_factory=dict)
    stages: dict[str, str] = field(default_factory=dict)
    segments: list[Segment] = field(default_factory=list)
    # Level-1 transcript QC verdict ("pass"/"review"/"fail"); empty until the
    # transcribe stage runs QC. See vidub.utils.qc and QUALITY-ROADMAP.md §3.2.
    transcript_verdict: str = ""
    # Level-2 translation QC verdict ("pass"/"review"/"fail"); empty until the
    # translate stage runs the battery. See vidub.utils.tqc and §3.3.
    translation_verdict: str = ""
    # Signature of the MT settings that produced the current translated text
    # (``script.json`` / the ``vi`` rows); recorded by the pipeline right after a
    # real translate run (see ``vidub.pipeline.mt_signature``). Empty means
    # "unknown" — a manifest written before this field existed — and an empty
    # signature NEVER invalidates resume, so pre-change ``job.json`` files keep
    # their skip behavior. Only a non-empty signature that differs from the
    # current config re-runs translate and everything downstream.
    mt_signature: str = ""
    # Signature of the voice reference used to synthesize cached speech. Empty
    # means the old pipeline did not record it; VoxCPM2's generation profile is
    # signed even without a user sample so old variable-seed audio is rebuilt.
    tts_signature: str = ""
    # Signature of the timing-stage shaping settings that produced the current
    # ``timed.json`` (``tts_rate_fit`` / ``recenter_audio`` / ``align_display`` /
    # ``sentence_units``); recorded by the pipeline right after a real timing run
    # (see ``vidub.pipeline.timing_signature``). Same additive "empty means
    # unknown, so never invalidate" contract as :attr:`mt_signature`: only a
    # non-empty recorded value that differs re-runs timing and everything after it.
    timing_signature: str = ""
    # Export-only policy/profile signature. Legacy empty values keep their
    # resume behavior; use --force export once to upgrade an older deliverable.
    export_signature: str = ""
    # Version of the translated-text selection contract. 0 means an older
    # artifact whose consumers should resolve text through the legacy ``vi``.
    text_contract_version: int = 0
    # Effective text selections last consumed successfully by speak/export.
    # Empty values are legacy/unknown and preserve old resume behavior.
    spoken_text_signature: str = ""
    subtitle_text_signature: str = ""

    @property
    def artifacts_dir(self) -> Path:
        """Directory holding this job's artifacts (``<cache_dir>/<hash>/``)."""
        return Path(self.cache_dir) / self.video_hash

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serialisable representation."""
        return asdict(self)

    def to_json(self) -> str:
        """Return the manifest as a pretty-printed JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def save(self, path: str | Path) -> Path:
        """Write the manifest to ``path`` (creating parents), returning it."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")
        return target

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobManifest":
        """Build a manifest from ``data``, ignoring unknown keys.

        Missing keys fall back to the dataclass defaults, so older/newer
        manifests load without error.
        """
        known = {f.name for f in fields(cls)}
        filtered = {key: value for key, value in data.items() if key in known}
        raw_segments = filtered.pop("segments", None) or []
        segments = [
            Segment(**{k: v for k, v in seg.items() if k in _SEGMENT_FIELDS})
            for seg in raw_segments
        ]
        return cls(segments=segments, **filtered)

    @classmethod
    def load(cls, path: str | Path) -> "JobManifest":
        """Load a manifest from a ``job.json`` file (tolerant to missing keys)."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)
