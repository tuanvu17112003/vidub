"""Read-only diagnostics for the two audited user exports; no model loading."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vidub.utils.english_tokens import grammar_leaks, is_contraction
from vidub.utils.subqc import measure


def main() -> None:
    results = {}
    for video, digest in [
        ("youtube-LPZh9BOjkQs", "e53bb23d1989ba4d70a3871d412f01327bfaa74e"),
        ("youtube-5sLYAQS9sWQ", "1a372a6d345535b258048bedfe488989ac45fed8"),
    ]:
        cache = ROOT / ".vidub-cache" / digest
        units = json.loads((cache / "units.json").read_text(encoding="utf-8"))["units"]
        timed = {r["idx"]: r for r in json.loads((cache / "timed.json").read_text(encoding="utf-8"))}
        leaks = []
        terms = []
        cps_placed = cps_locked = 0
        chars = seconds = 0.0
        for unit in units:
            index = unit["index"]
            source, vi = unit["source"], unit["vi"]
            # Read retained tokens from the frozen export, independently of
            # today's corrected entity extractor, so baseline audits reproduce.
            for token in grammar_leaks(vi):
                leaks.append({"unit": index, "token": token,
                              "english_contraction": is_contraction(token),
                              "retained_in_vi": True,
                              "battery": unit.get("battery")})
            if "transformer" in source.lower() and "máy biến áp" in vi.lower():
                terms.append({"unit": index, "source": source, "vi": vi,
                              "battery": unit.get("battery")})
            row = timed[index]
            placed = measure(vi, row["start"], row["end"])
            locked = measure(vi, unit["start"], unit["end"])
            cps_placed += int(placed["cps"] > 17)
            cps_locked += int(locked["cps"] > 17)
            chars += len(vi.replace("\n", ""))
            seconds += row["end"] - row["start"]
        results[video] = {"units": len(units), "apostrophe_entities": leaks,
            "electrical_transformer_errors": terms,
            "units_over_17cps_placed": cps_placed,
            "units_over_17cps_locked": cps_locked,
            "total_characters": chars, "total_placed_seconds": seconds,
            "weighted_cps": chars / seconds,
            "minimum_total_display_seconds_at_17cps": chars / 17,
            "interpretation": "Minimum ignores rounding and whitespace removed by splitting; diagnostic, not a repaired subtitle track."}
    dest = ROOT / "eval-cache/dubbing-research-diagnostics.json"
    dest.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    for video, result in results.items():
        print(video, json.dumps({k:v for k,v in result.items() if k not in {"apostrophe_entities", "electrical_transformer_errors"}}, ensure_ascii=False))
        print("apostrophe entities", result["apostrophe_entities"])


if __name__ == "__main__":
    main()
