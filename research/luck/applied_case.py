"""A descriptive reference-class case using official R UCBAdmissions counts.

Offline reproduction: MPLCONFIGDIR=/tmp/luck-mpl python3 research/luck/applied_case.py
Refresh the frozen public data: python3 research/luck/applied_case.py --refresh-data
No individual records, randomness attribution, causal estimates, or p-values.
"""
from pathlib import Path
import argparse
import csv
from fractions import Fraction
import hashlib
import json
import re
from urllib.request import urlopen

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "research/luck/data/berkeley-admissions.csv"
RESULT = ROOT / "research/luck/results/applied-case.json"
FIG = ROOT / "static/img/luck/reference-class-case"
SOURCE_URL = "https://svn.r-project.org/R/trunk/src/library/datasets/data/UCBAdmissions.R"
SOURCE_SHA256 = "86a5801b122ec1f172098303de35c9df9a8c4c09f2f81dc9f03bfafdde102d5e"
DOC_URL = "https://search.r-project.org/R/refmans/datasets/html/UCBAdmissions.html"
PAPER_URL = "https://doi.org/10.1126/science.187.4175.398"
BLUE = "#23618b"
ORANGE = "#c5652b"
INK = "#243745"
MUTED = "#5d6b75"
GRID = "#e7edf1"


def refresh_data():
    """Parse the 24 public counts; never execute downloaded source code."""
    with urlopen(SOURCE_URL, timeout=30) as response:
        source = response.read()
    actual_hash = hashlib.sha256(source).hexdigest()
    if actual_hash != SOURCE_SHA256:
        raise ValueError("Official source changed; review it before replacing the frozen cohort.")
    text = source.decode("utf-8")
    match = re.search(r"array\(c\((.*?)\),\s*dim\s*=\s*c\(2,\s*2,\s*6\)", text, re.S)
    if not match:
        raise ValueError("Unexpected UCBAdmissions array format")
    counts = [int(value.strip()) for value in match.group(1).split(",")]
    if len(counts) != 24:
        raise ValueError("Expected 24 aggregate cell counts")
    DATA.parent.mkdir(parents=True, exist_ok=True)
    with DATA.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["department", "recorded_sex", "admitted", "rejected", "applications"], lineterminator="\n")
        writer.writeheader()
        for dept_index, dept in enumerate("ABCDEF"):
            for group_index, group in enumerate(["Male", "Female"]):
                offset = 4 * dept_index + 2 * group_index
                admitted, rejected = counts[offset:offset + 2]
                writer.writerow({"department": dept, "recorded_sex": group,
                                 "admitted": admitted, "rejected": rejected,
                                 "applications": admitted + rejected})


def calculate():
    with DATA.open(newline="") as stream:
        records = list(csv.DictReader(stream))
    cells = {}
    for record in records:
        key = record["department"], record["recorded_sex"]
        if key in cells:
            raise ValueError(f"Duplicated aggregate cell: {key}")
        counts = {name: int(record[name]) for name in ["admitted", "rejected", "applications"]}
        assert counts["admitted"] + counts["rejected"] == counts["applications"]
        assert counts["applications"] > 0
        cells[key] = counts
    assert set(cells) == {(dept, group) for dept in "ABCDEF" for group in ["Male", "Female"]}
    total = sum(cell["applications"] for cell in cells.values())
    assert total == 4526
    totals = {group: {name: sum(cells[dept, group][name] for dept in "ABCDEF")
                      for name in ["admitted", "rejected", "applications"]}
              for group in ["Male", "Female"]}
    assert totals["Male"]["applications"] == 2691 and totals["Male"]["admitted"] == 1198
    assert totals["Female"]["applications"] == 1835 and totals["Female"]["admitted"] == 557
    raw = {group: Fraction(totals[group]["admitted"], totals[group]["applications"])
           for group in totals}
    dept_counts = {dept: sum(cells[dept, group]["applications"] for group in totals) for dept in "ABCDEF"}
    weights = {dept: Fraction(dept_counts[dept], total) for dept in "ABCDEF"}
    assert sum(weights.values()) == 1
    rates = {(dept, group): Fraction(cell["admitted"], cell["applications"])
             for (dept, group), cell in cells.items()}
    standardized = {group: sum(weights[dept] * rates[dept, group] for dept in "ABCDEF")
                    for group in totals}
    # The raw contrast uses each group's own observed department distribution.
    for group in totals:
        reconstructed = sum(Fraction(cells[dept, group]["applications"], totals[group]["applications"])
                            * rates[dept, group] for dept in "ABCDEF")
        assert reconstructed == raw[group]
    raw_difference = raw["Female"] - raw["Male"]
    standardized_difference = standardized["Female"] - standardized["Male"]
    assert raw_difference < 0 < standardized_difference
    return {
        "kind": "Descriptive reanalysis of published historical aggregate counts; not a luck estimate or causal effect",
        "source": {"dataset": "R datasets::UCBAdmissions", "data_url": SOURCE_URL,
                   "documentation_url": DOC_URL, "original_paper_doi": PAPER_URL,
                   "source_r_sha256": SOURCE_SHA256, "source_verified_date": "2026-10-05",
                   "csv_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest()},
        "cohort": {"year": 1973, "departments": 6, "applications": total,
                   "scope": "Six largest departments represented in UCBAdmissions; not all Berkeley applications",
                   "recorded_sex_labels": ["Male", "Female"]},
        "group_totals": totals,
        "observed_mix": {"admission_rates": {group: float(value) for group, value in raw.items()},
                         "exact_rate_fractions": {group: str(value) for group, value in raw.items()},
                         "female_minus_male": float(raw_difference),
                         "female_minus_male_percentage_points": float(100 * raw_difference)},
        "common_pooled_department_mix": {
            "definition": "w_d = applications in department d / 4526, pooled across the two recorded groups",
            "admission_rates": {group: float(value) for group, value in standardized.items()},
            "exact_rate_fractions": {group: str(value) for group, value in standardized.items()},
            "female_minus_male": float(standardized_difference),
            "female_minus_male_percentage_points": float(100 * standardized_difference)},
        "departments": [
            {"department": dept, "pooled_applications": dept_counts[dept], "common_weight": float(weights[dept]),
             "exact_common_weight": str(weights[dept]),
             "rates": {group: float(rates[dept, group]) for group in totals},
             "group_department_shares": {group: cells[dept, group]["applications"] / totals[group]["applications"]
                                          for group in totals},
             "female_minus_male_percentage_points": float(100 * (rates[dept, "Female"] - rates[dept, "Male"]))}
            for dept in "ABCDEF"],
        "uncertainty": "No confidence intervals: these are exact descriptive proportions for the recorded cohort, not sampled future populations.",
        "limitations": [
            "Changing weights changes the descriptive estimand; it does not change the recorded admissions decisions.",
            "Department may mediate prior inequality; conditioning on it cannot establish the absence of discrimination.",
            "Qualifications, application processes, and individual counterfactuals are absent from these aggregate cells.",
            "These records do not identify who was lucky or what fraction of an outcome was earned.",
            "A single admissions cycle cannot identify between-cycle variability or stable applicant-level variance.",
        ],
    }


def render(result):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "axes.edgecolor": "#cad4da",
                         "svg.fonttype": "none", "svg.hashsalt": "luck-berkeley-1973"})
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.6, 7.5),
                                     gridspec_kw={"height_ratios": [1.1, 1]})
    fig.subplots_adjust(left=.13, right=.97, top=.93, bottom=.13, hspace=.58)
    fig.text(.13, .983, "Berkeley 1973 · six departments · 4,526 applications", fontsize=12, color=MUTED)
    positions = [0, 1]
    for offset, group, color in [(-.17, "Male", BLUE), (.17, "Female", ORANGE)]:
        values = [result["observed_mix"]["admission_rates"][group],
                  result["common_pooled_department_mix"]["admission_rates"][group]]
        bars = top.bar([x + offset for x in positions], values, width=.31,
                       color=color, label=f"{group} (recorded label)", zorder=3)
        for bar, value in zip(bars, values):
            top.text(bar.get_x() + bar.get_width()/2, value + .012,
                     f"{value:.1%}", ha="center", fontsize=15, color=color)
    top.set(xlim=(-.55, 1.55), ylim=(0, .58), yticks=[0, .2, .4],
            xticks=positions, xticklabels=["Observed\ndepartment mix", "Common pooled\ndepartment weights"],
            ylabel="Admission rate")
    top.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    top.tick_params(length=0, pad=7)
    top.set_axisbelow(True);top.yaxis.grid(True, color=GRID)
    top.legend(loc="upper left", bbox_to_anchor=(0, 1.15), frameon=False, ncol=2, fontsize=10.5)
    for position, key in zip(positions, ["observed_mix", "common_pooled_department_mix"]):
        gap = result[key]["female_minus_male_percentage_points"]
        top.text(position, -.32, f"Female − male: {gap:+.2f} pp", transform=top.get_xaxis_transform(),
                 ha="center", fontsize=11, color=INK)
    departments = result["departments"]
    differences = [row["female_minus_male_percentage_points"] for row in departments]
    for y, value in zip(range(6), differences):
        color = ORANGE if value > 0 else BLUE
        bottom.plot([0, value], [y, y], color=color, linewidth=2.5, zorder=3)
        bottom.scatter([value], [y], s=40, color=color, zorder=4)
        bottom.text(24.8, y, f"{value:+.2f}", ha="right", va="center", fontsize=11, color=color)
    bottom.set(xlim=(-5.7, 25.7), ylim=(5.55, -.65), yticks=list(range(6)),
               yticklabels=[row["department"] for row in departments], xticks=[-5, 0, 10, 20],
               xlabel="Female − male admission rate (percentage points)", ylabel="Department")
    bottom.axvline(0, color=MUTED, linewidth=.9, linestyle=(0, (3, 3)))
    bottom.text(0, 1.12, "Within-department contrasts remain heterogeneous", transform=bottom.transAxes,
                fontsize=12, fontweight="semibold")
    bottom.set_axisbelow(True);bottom.xaxis.grid(True, color=GRID)
    bottom.tick_params(length=0, pad=7)
    fig.text(.13, .040, "Common weights change the question, not the historical outcomes.", fontsize=10.5, color=MUTED)
    fig.text(.13, .010, "Descriptive rates; no causal attribution or sampling model is imposed.", fontsize=10.5, color=MUTED)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG.with_suffix(".svg"), bbox_inches="tight", pad_inches=.14, facecolor="white", metadata={"Date": None})
    svg = FIG.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(FIG.with_suffix(".png"), bbox_inches="tight", pad_inches=.14, dpi=190, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-data", action="store_true")
    args = parser.parse_args()
    if args.refresh_data:
        refresh_data()
    result = calculate()
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, indent=2) + "\n")
    render(result)
    print(json.dumps({"observed_mix": result["observed_mix"],
                      "common_pooled_department_mix": result["common_pooled_department_mix"]}, indent=2))


if __name__ == "__main__":
    main()
