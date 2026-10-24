"""Regenerate both user dubs in isolated caches, retaining the original exports."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from vidub.config import JobConfig
from vidub.manifest import JobManifest
from vidub.pipeline import mt_signature, timing_signature
from vidub.stages import translate, speak, timing, mix, export
from vidub.utils.english_tokens import grammar_leaks
from vidub.utils.proc import find_tool, run


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    scratch = ROOT / "eval-cache/phase2-rerender"
    scratch.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=str(scratch / "run.log"), level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", encoding="utf-8")
    results = {}
    for video, digest in [
        ("youtube-5sLYAQS9sWQ", "1a372a6d345535b258048bedfe488989ac45fed8"),
        ("youtube-LPZh9BOjkQs", "e53bb23d1989ba4d70a3871d412f01327bfaa74e"),
    ]:
        source = ROOT / ".vidub-cache" / digest
        destination = scratch / digest
        shutil.copytree(source, destination, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("review", "speech", "dub_track.wav", "_mix_prelim.wav"))
        cfg = JobConfig(cache_dir=str(scratch), glossary_path=str(ROOT / "samples/glossary-ai-tutorial.tsv"))
        manifest = JobManifest.load(destination / "job.json")
        manifest.cache_dir = str(scratch)
        if not Path(manifest.input_path).is_file():
            # Export maps only the video stream from this file; original audio
            # and separated stems remain the unchanged ingested cache assets.
            manifest.input_path = str(ROOT / "output" / video / "dubbed.vi.mp4")
        manifest.paths["output_dir"] = str(ROOT / "output/phase2-improved" / video)
        assert export.output_dir_for(manifest).resolve() == (ROOT / "output/phase2-improved" / video).resolve()
        job = destination / "job.json"
        manifest.save(job)
        durations = {}
        for name, function in [("translate", translate.translate), ("speak", speak.speak),
                               ("timing", timing.fit), ("mix", mix.mix), ("export", export.export)]:
            print(video, name, "started", flush=True)
            started = time.perf_counter()
            manifest = function(str(job), cfg)
            durations[name] = round(time.perf_counter() - started, 2)
            print(video, name, "done", durations[name], "seconds", flush=True)
        manifest.mt_signature = mt_signature(cfg)
        manifest.timing_signature = timing_signature(cfg)
        manifest.save(job)
        output = Path(manifest.paths["output_dir"])
        ffmpeg = find_tool("ffmpeg")
        assert ffmpeg
        decoded = run([ffmpeg, "-v", "error", "-i", str(output / "dubbed.vi.mp4"),
                       "-f", "null", "-"], capture_output=True, text=True, check=False)
        units = json.loads((destination / "units.json").read_text(encoding="utf-8"))["units"]
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        results[video] = {"output": str(output), "stage_seconds": durations,
                          "full_decode_exit": decoded.returncode,
                          "grammar_leaks": [{"unit": unit["index"], "tokens": grammar_leaks(unit["vi"])}
                                            for unit in units if grammar_leaks(unit["vi"])],
                          "subtitle_qc": report["subtitles_qc"], "quality": report["quality"],
                          "pacing": report["pacing"]}
        (scratch / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(video, "verified full decode", decoded.returncode, "output", output, flush=True)


if __name__ == "__main__":
    main()
