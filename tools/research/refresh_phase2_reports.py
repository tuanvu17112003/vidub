"""Drop stale generated TTS packets proven to belong to superseded WAVs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vidub.config import JobConfig
from vidub.manifest import JobManifest
from vidub.utils.finalqc import run_gate


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    scratch = ROOT / "eval-cache/phase2-rerender"
    results = json.loads((scratch / "results.json").read_text(encoding="utf-8"))
    for video, result in results.items():
        output = Path(result["output"])
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        # Known isolated source caches used by the two audited rerenders.
        digest = "1a372a6d345535b258048bedfe488989ac45fed8" if video.endswith("5sLYAQS9sWQ") else "e53bb23d1989ba4d70a3871d412f01327bfaa74e"
        manifest = JobManifest.load(scratch / digest / "job.json")
        cache = manifest.artifacts_dir
        assert cache.resolve().is_relative_to(scratch.resolve())
        rows = json.loads((cache / "timed.json").read_text(encoding="utf-8"))
        current = {str(row["idx"]): cache / row["file"] for row in rows}
        cleared = []
        for packet in (cache / "review/tts").glob("seg-*.wav"):
            index = packet.stem.removeprefix("seg-")
            if not index.isdigit():
                continue
            fitted = current.get(index)
            if fitted is not None and fitted.is_file() and hashlib.sha256(packet.read_bytes()).digest() == hashlib.sha256(fitted.read_bytes()).digest():
                continue
            packet.unlink()
            packet.with_suffix(".txt").unlink(missing_ok=True)
            cleared.append(int(index))
        report["quality"] = run_gate(manifest, JobConfig())
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        result["quality"] = report["quality"]
        result["stale_tts_packets_removed"] = cleared
        print(video, "stale TTS packets cleared", cleared, flush=True)
    (scratch / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
