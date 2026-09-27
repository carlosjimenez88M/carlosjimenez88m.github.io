"""Inspect existing aggregate exports; no model calls or scientific reruns.

Run from any directory with Python 3. Writes a provenance summary and copies
two aggregate tables to static/examples/editorial-review-2026 for the articles.
No lyrics, audio, private receipts, or reviewer identities are exported.
"""
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "static/examples/editorial-review-2026"


def provenance(path):
    return {"path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    source = ROOT / "tidytuesday/data/attention_windows_results.csv"
    grouped = defaultdict(list)
    with source.open(newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["artist"]].append(float(row["attention_window"]))
    summary = {
        "editorial_revision": "2026-09-27",
        "scope": "Inspection of existing files, not a scientific replication or model rerun.",
        "attention_windows": {
            "source": provenance(source),
            "aggregation": "Unweighted arithmetic mean of attention_window by artist over all CSV rows.",
            "groups": {artist: {"rows": len(values), "mean": mean(values)}
                       for artist, values in sorted(grouped.items())},
            "limitation": "CSV lacks model, threshold and run identifiers; the previous headline cannot be reconciled from this file alone.",
        },
        "aquamosh": [],
    }
    exports = ROOT / "tidytuesday/aquamosh-analysis/outputs/exports"
    for name in ("cross_model_invariance.csv", "llm_judge_stratified.csv"):
        path = exports / name
        shutil.copyfile(path, DEST / name)
        summary["aquamosh"].append(provenance(path))
    (DEST / "provenance.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["attention_windows"]["groups"], indent=2))


if __name__ == "__main__":
    main()
