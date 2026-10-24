"""Offline repair check on the five observed leaks; never mutates job caches."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from vidub.engines.mt.envit5_mt import Envit5Translator
from vidub.utils.english_tokens import grammar_leaks
from vidub.utils.glossary import load_glossary
from vidub.utils.tqc import correct_rows, score_rows


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    cases = []
    term_cases = []
    glossary = load_glossary(ROOT / "samples/glossary-ai-tutorial.tsv")
    for video, digest in [
        ("youtube-LPZh9BOjkQs", "e53bb23d1989ba4d70a3871d412f01327bfaa74e"),
        ("youtube-5sLYAQS9sWQ", "1a372a6d345535b258048bedfe488989ac45fed8"),
    ]:
        units = json.loads((ROOT / ".vidub-cache" / digest / "units.json").read_text(encoding="utf-8"))["units"]
        for unit in units:
            row = {"en": unit["source"], "vi": unit["vi"]}
            entry = score_rows([row], glossary)[0][0]
            if "terminology-error" in entry["signals"] and len(term_cases) < 8:
                term_cases.append({"video": video, "unit": unit["index"], **row})
            if grammar_leaks(unit["vi"]):
                cases.append({"video": video, "unit": unit["index"],
                              "en": unit["source"], "vi": unit["vi"],
                              "cached_battery": unit.get("battery")})
    rows = [{"en": case["en"], "vi": case["vi"]} for case in cases]
    # Per-row scoring avoids confusing repeated phrases across different videos.
    labels = []
    for index, row in enumerate(rows):
        entry = score_rows([row], glossary)[0][0]
        entry["index"] = index
        labels.append(entry)
    print(f"Checking {len(cases)} cached leaks with the installed envit5 model.", flush=True)
    engine = Envit5Translator()
    repaired, corrections = correct_rows(rows, labels, glossary, engine)
    for case, label, row, correction in zip(cases, labels, repaired, corrections):
        case.update({"before_label": label["label"], "after_vi": row["vi"],
                     "after_leaks": grammar_leaks(row["vi"], glossary.values()),
                     "correction": correction})
        print(case["video"], case["unit"], correction["label"],
              "remaining leaks:", case["after_leaks"], flush=True)
    for case in term_cases:
        row = {"en": case["en"], "vi": case["vi"]}
        entries = score_rows([row], glossary)[0]
        repaired_term, actions = correct_rows([row], entries, glossary, engine)
        after = score_rows(repaired_term, glossary)[0][0]
        case.update({"before_signals": entries[0]["signals"],
                     "after_vi": repaired_term[0]["vi"], "after_signals": after["signals"],
                     "correction": actions[0]})
        print("Terminology", case["video"], case["unit"], after["label"], flush=True)
    result = {"scope": "five cached grammar leaks plus eight first terminology flags; deterministic checks, no listener/semantic approval",
              "cases": cases, "before_leaks": sum(len(grammar_leaks(c["vi"])) for c in cases),
              "after_leaks": sum(len(c["after_leaks"]) for c in cases),
              "terminology_cases": term_cases,
              "remaining_terminology_flags": sum("terminology-error" in c["after_signals"] for c in term_cases)}
    destination = ROOT / "eval-cache/translation-grammar-safety.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(destination, flush=True)


if __name__ == "__main__":
    main()
