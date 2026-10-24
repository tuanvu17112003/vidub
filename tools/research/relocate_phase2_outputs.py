"""Recover the baseline exports and place isolated rerenders at explicit paths."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from vidub.config import JobConfig
from vidub.manifest import JobManifest
from vidub.stages import export
from vidub.utils.finalqc import run_gate
from vidub.utils.proc import find_tool, run
from vidub.utils.subqc import measure, split_cue, wrap_text


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    scratch = ROOT / "eval-cache/phase2-rerender"
    results = json.loads((scratch / "results.json").read_text(encoding="utf-8"))
    ffmpeg = find_tool("ffmpeg")
    assert ffmpeg
    for video, digest in [
        ("youtube-5sLYAQS9sWQ", "1a372a6d345535b258048bedfe488989ac45fed8"),
        ("youtube-LPZh9BOjkQs", "e53bb23d1989ba4d70a3871d412f01327bfaa74e"),
    ]:
        original = ROOT / "output" / video
        improved = ROOT / "output/phase2-improved" / video
        improved.mkdir(parents=True, exist_ok=True)
        for name in [export.DUBBED_NAME, export.SUBTITLES_NAME, export.SUBTITLES_ASS_NAME,
                     export.SCRIPT_TSV_NAME, export.REPORT_NAME]:
            shutil.copy2(original / name, improved / name)
        baseline = JobManifest.load(ROOT / ".vidub-cache" / digest / "job.json")
        baseline.cache_dir = str(ROOT / ".vidub-cache")
        baseline.paths["output_dir"] = str(original)
        rows = export._unit_subtitle_rows(baseline)
        assert rows is not None
        legacy_rows = []
        for row in rows:
            text = wrap_text(row["vi"])
            parts = split_cue(text, row["start"], row["end"], row.get("align_words")) if measure(text, row["start"], row["end"])["cps"] > 17 else [(row["start"], row["end"], text)]
            legacy_rows.extend({**row, "start": start, "end": end, "vi": wrap_text(part)} for start, end, part in parts)
        srt = export.write_subtitles_srt(legacy_rows, original / export.SUBTITLES_NAME)
        export.write_subtitles_ass(legacy_rows, original / export.SUBTITLES_ASS_NAME)
        script = json.loads((baseline.artifacts_dir / "script.json").read_text(encoding="utf-8"))
        export.write_script_tsv(script, original / export.SCRIPT_TSV_NAME)
        source_video = Path(baseline.input_path)
        if not source_video.is_file():
            source_video = improved / export.DUBBED_NAME
        command = export.build_mux_command(source_video, baseline.artifacts_dir / "dub_track.wav", srt,
                                           original / export.DUBBED_NAME, baseline.mode)
        run([ffmpeg, "-hide_banner", "-loglevel", "error", *command], capture_output=True)
        report = export.build_report(baseline)
        report["quality"] = run_gate(baseline, JobConfig())
        report["baseline_restoration"] = "Video rebuilt from unchanged original audio/text cache; report reconstructed with current QC, not an exact byte copy of the previous report."
        (original / export.REPORT_NAME).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(video, "baseline audio and captions restored", flush=True)
        job = scratch / digest / "job.json"
        manifest = JobManifest.load(job)
        manifest.paths["output_dir"] = str(improved)
        manifest.save(job)
        # Verify the corrected destination through the production export path.
        export.export(str(job), JobConfig(glossary_path=str(ROOT / "samples/glossary-ai-tutorial.tsv")))
        report = json.loads((improved / export.REPORT_NAME).read_text(encoding="utf-8"))
        results[video].update({"output": str(improved), "subtitle_qc": report["subtitles_qc"],
                               "quality": report["quality"], "pacing": report["pacing"]})
        for label, directory in [("baseline", original), ("improved", improved)]:
            decoded = run([ffmpeg, "-v", "error", "-i", str(directory / export.DUBBED_NAME), "-f", "null", "-"],
                          capture_output=True, text=True, check=False)
            assert decoded.returncode == 0, decoded.stderr
            print(video, label, "full decode passed", flush=True)
        results[video]["baseline_restored"] = True
        results[video]["full_decode_exit"] = 0
        (scratch / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
