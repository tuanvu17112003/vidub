"""Summarize final isolated outputs separately from the matched caption audit."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vidub.engines.mt.validators import _missing_numbers
from vidub.utils.glossary import load_glossary
from vidub.utils.tqc import score_rows


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    scratch = ROOT / "eval-cache/phase2-rerender"
    results = json.loads((scratch / "results.json").read_text(encoding="utf-8"))
    glossary = load_glossary(ROOT / "samples/glossary-ai-tutorial.tsv")
    for video, digest in [
        ("youtube-5sLYAQS9sWQ", "1a372a6d345535b258048bedfe488989ac45fed8"),
        ("youtube-LPZh9BOjkQs", "e53bb23d1989ba4d70a3871d412f01327bfaa74e"),
    ]:
        cache = scratch / digest
        units = json.loads((cache / "units.json").read_text(encoding="utf-8"))["units"]
        qc = json.loads((cache / "translation_qc.json").read_text(encoding="utf-8"))
        tts = json.loads((cache / "tts_qc.json").read_text(encoding="utf-8"))
        term_flags = []
        numbers = []
        references = []
        for unit in units:
            row = {"en": unit["source"], "vi": unit["vi"]}
            labels = score_rows([row], glossary)[0]
            if "terminology-error" in labels[0]["signals"]:
                term_flags.append({"unit": unit["index"], "signals": labels[0]["signals"]})
            missing = _missing_numbers(unit["source"], unit["vi"])
            if missing:
                numbers.append({"unit": unit["index"], "missing": missing})
            if unit["source"] in glossary and unit["source"].endswith("."):
                references.append({"unit": unit["index"], "exact_reference": unit["vi"] == glossary[unit["source"]]})
        results[video].update({"units": len(units), "remaining_terminology_flags": term_flags,
                               "strict_missing_numbers": numbers, "reference_sentences": references,
                               "translation_counts": qc["counts"], "tts_qc": tts})
        summary = {"units": len(units), "grammar_leaks": len(results[video]["grammar_leaks"]),
                   "terminology_flags": len(term_flags), "missing_numbers": numbers, "references": references,
                   "subtitle_violations": results[video]["subtitle_qc"]["violations"],
                   "audio": results[video]["quality"]["audio"], "pacing_clips": results[video]["pacing"]["clipped"],
                   "translation_counts": qc["counts"], "tts": tts}
        print(video, json.dumps(summary, ensure_ascii=False), flush=True)
    (scratch / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
