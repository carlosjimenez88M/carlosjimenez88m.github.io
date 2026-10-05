"""Render the luck essays' editorial figures without rerunning the experiments.

Run after analysis.py: MPLCONFIGDIR=/tmp/luck-mpl python3 research/luck/graphics.py
Exact discrete probability masses are drawn as bars, not smooth densities.
The remaining figures read the fixed synthetic results in results/summary.json.
"""
from pathlib import Path
import json

import numpy as np
from scipy.stats import binom, betabinom
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "static/img/luck"
BLUE = "#23618b"
BLUE_PALE = "#bbd3e2"
ORANGE = "#c5652b"
INK = "#243745"
MUTED = "#5d6b75"
GRID = "#e7edf1"


def setup():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.edgecolor": "#cad4da",
        "axes.linewidth": .8,
        "svg.fonttype": "none",
        "svg.hashsalt": "luck-editorial-2026",
        "savefig.facecolor": "white",
    })


def save(fig, name):
    """Save the same figure for the blog's SVG and email's PNG renderer."""
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight", pad_inches=.14,
                metadata={"Date": None})
    # Strip whitespace introduced by Matplotlib's XML pretty printer.
    path = FIG / f"{name}.svg"
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    fig.savefig(FIG / f"{name}.png", dpi=190, bbox_inches="tight", pad_inches=.14,
                metadata={"Software": "Reproducible statistical graphics: research/luck/graphics.py"})
    plt.close(fig)


def discrete_panel(ax, probability, *, label, tail, variance=None):
    k = np.arange(21)
    ax.bar(k, probability, width=.80,
           color=[ORANGE if value >= 15 else BLUE_PALE for value in k],
           edgecolor="white", linewidth=.7, zorder=3)
    ax.axvline(14.5, color=ORANGE, linewidth=1.2, linestyle=(0, (3, 3)), zorder=4)
    ax.set_xlim(-.8, 20.8)
    ax.set_ylim(0, .225)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_yticks([0, .10, .20])
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.tick_params(axis="both", length=0, pad=7)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=.8)
    ax.set_ylabel("Probability", fontsize=11, labelpad=9)
    ax.text(0, 1.13, label, transform=ax.transAxes,
            fontsize=15.5, fontweight="semibold", va="baseline")
    ax.text(1, 1.13, f"P(K ≥ 15) = {tail:.2%}", transform=ax.transAxes,
            fontsize=15.5, color=ORANGE, ha="right", va="baseline")
    if variance is not None:
        ax.text(.985, .78, f"Variance = {variance:.2f}", transform=ax.transAxes,
                ha="right", fontsize=11, color=MUTED)


def reference_distributions():
    fig, axes = plt.subplots(3, 1, figsize=(7.6, 8.2))
    fig.subplots_adjust(left=.13, right=.97, top=.94, bottom=.115, hspace=.63)
    k = np.arange(21)
    for ax, p in zip(axes, [.5, .6, .7]):
        discrete_panel(ax, binom.pmf(k, 20, p),
                       label=f"Known p = {p:.2f}",
                       tail=binom.sf(14, 20, p))
    axes[-1].set_xlabel("Successes in 20 independent attempts", labelpad=9)
    fig.text(.13, .01, "Orange bars: the same observed threshold, 15 or more successes.",
             fontsize=10.5, color=MUTED)
    save(fig, "reference-distributions")


def posterior_predictive():
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.9))
    fig.subplots_adjust(left=.13, right=.97, top=.91, bottom=.13, hspace=.64)
    k = np.arange(21)
    p = 13 / 22
    distributions = [
        (binom.pmf(k, 20, p), "Fixed p = 13/22", binom.sf(14, 20, p), binom.var(20, p)),
        (betabinom.pmf(k, 20, 13, 9), "p ∼ Beta(13, 9)", betabinom.sf(14, 20, 13, 9), betabinom.var(20, 13, 9)),
    ]
    for ax, (mass, label, tail, variance) in zip(axes, distributions):
        discrete_panel(ax, mass, label=label, tail=tail, variance=variance)
    axes[-1].set_xlabel("Successes in the new batch of 20 attempts", labelpad=9)
    fig.text(.13, .015, "Both means = 11.82. Only the second prediction retains uncertainty about p.",
             fontsize=10.5, color=MUTED)
    save(fig, "posterior-predictive")


def shared_environment():
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.9))
    fig.subplots_adjust(left=.13, right=.97, top=.91, bottom=.13, hspace=.64)
    k = np.arange(21)
    distributions = [
        (binom.pmf(k, 20, .6), "Independent: p = 0.60", binom.sf(14, 20, .6), binom.var(20, .6)),
        (betabinom.pmf(k, 20, 2.4, 1.6), "Shared batch: ρ = 0.20", betabinom.sf(14, 20, 2.4, 1.6), betabinom.var(20, 2.4, 1.6)),
    ]
    for ax, (mass, label, tail, variance) in zip(axes, distributions):
        discrete_panel(ax, mass, label=label, tail=tail, variance=variance)
    axes[-1].set_xlabel("Successes in 20 attempts", labelpad=9)
    fig.text(.13, .015, "Both means = 12. Shared batch conditions make extreme counts more common.",
             fontsize=10.5, color=MUTED)
    save(fig, "shared-environment")


def prediction_and_control():
    """Illustrative source classifications, conditional on an actor and horizon."""
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.set(xlim=(0, 10), ylim=(0, 5.8))
    ax.axis("off")
    for x, heading in [(4.05, "Expected under\nthe reference"),
                       (8.03, "Unexpected under\nthe reference")]:
        ax.text(x, 5.18, heading, ha="center", va="center", fontsize=12.5,
                fontweight="semibold", linespacing=1.45)
    for y, heading, color in [(3.78, "Within\npractical control", BLUE),
                               (2.02, "Outside\npractical control", ORANGE)]:
        ax.text(.04, y, heading, ha="left", va="center", fontsize=11.5,
                color=color, fontweight="semibold", linespacing=1.5)
    def box(x, y, text, color):
        ax.add_patch(FancyBboxPatch((x, y), 3.7, 1.38,
                                   boxstyle="round,pad=0.06,rounding_size=0.10",
                                   edgecolor=color, facecolor="white", linewidth=1.2))
        ax.text(x + 1.85, y + .69, text, ha="center", va="center", fontsize=12,
                linespacing=1.5, color=INK)
    box(2.20, 3.08, "A chosen configuration\nwith known effects", BLUE)
    box(6.18, 3.08, "A chosen configuration;\nits effect was underestimated", BLUE)
    box(2.20, 1.32, "Inherited access to tools\nknown before the trial", ORANGE)
    box(6.18, 1.32, "An unanticipated\nshared outage", ORANGE)
    ax.text(5, .87, "Illustrative inputs, not classifications of complete outcomes.",
            ha="center", fontsize=11, color=MUTED)
    ax.text(5, .47, "Practical control depends on the actor and time horizon.",
            ha="center", fontsize=11, color=MUTED)
    ax.text(5, .07, "Control of an action does not imply control of every consequence.",
            ha="center", fontsize=11, color=MUTED)
    fig.subplots_adjust(left=.02, right=.98, top=.98, bottom=.02)
    save(fig, "prediction-and-control")


def selection_and_regression(result):
    rows = result["selection"]["rows"]
    ms = [row["candidates"] for row in rows]
    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    fig.subplots_adjust(left=.12, right=.97, top=.95, bottom=.23)
    ax.plot(ms, [row["selected_score"] for row in rows], color=ORANGE,
            marker="o", linewidth=2, label="Winning observed score", zorder=4)
    ax.plot(ms, [row["selected_latent_mean"] for row in rows], color=BLUE,
            marker="s", linewidth=2, label="Winner's latent mean", zorder=4)
    ax.plot(ms, [row["independent_repeat_mean"] for row in rows], color="#647d8e",
            marker="o", markerfacecolor="white", linewidth=1.5, linestyle=(0, (3, 2)),
            label="Independent repeat", zorder=5)
    ax.axhline(70, color=MUTED, linewidth=.8, linestyle=":")
    ax.text(25, 68.70, "Population mean: 70", color=MUTED, fontsize=10.5)
    ax.set(xscale="log", xticks=ms, xticklabels=ms, ylim=(68, 92), xlim=(.85, 118),
           xlabel="Candidates inspected before selecting the winner",
           ylabel="Mean score (synthetic units)")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID)
    ax.tick_params(length=0, pad=6)
    ax.fill_between(ms, [row["selected_latent_mean"] for row in rows],
                    [row["selected_score"] for row in rows], color=ORANGE, alpha=.10)
    ax.annotate("Selection optimism", xy=(20, 79.2), xytext=(4.5, 85.8),
                arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 1.1},
                fontsize=11, color=ORANGE)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -.20), ncol=2,
              frameon=False, fontsize=10.5, columnspacing=1.4)
    save(fig, "selection-and-regression")


def reinforcement():
    """Exact count distributions for the two stipulated 500-draw processes."""
    n = 500
    k = np.arange(n + 1)
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.5))
    fig.subplots_adjust(left=.13, right=.97, top=.92, bottom=.17, hspace=.52)
    models = [
        (binom.pmf(k, n, .5), "Independent draws: fixed p = 0.50", BLUE),
        (betabinom.pmf(k, n, 1, 1), "Reinforced urn: equal initial weights (1, 1)", ORANGE),
    ]
    for ax, (mass, label, color) in zip(axes, models):
        ax.bar(k / n, mass, width=.002, color=color, edgecolor="none", rasterized=False)
        ax.set(xlim=(-.01, 1.01), ylim=(0, .04), xticks=[0, .25, .5, .75, 1],
               yticks=[0, .02, .04])
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.text(0, 1.10, label, transform=ax.transAxes, fontsize=12, fontweight="semibold")
        ax.set_ylabel("Probability", fontsize=11)
        ax.tick_params(length=0, pad=6)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=GRID)
    axes[-1].set_xlabel("Final share of A after 500 draws", labelpad=9)
    axes[-1].text(.5, .50, "Every count from 0 to 500 is equally likely",
                  transform=axes[-1].transAxes, ha="center", fontsize=11, color=ORANGE)
    fig.text(.13, .015, "Exact model probabilities. The reinforced count is uniform: 1/501 per count.",
             fontsize=10.5, color=MUTED)
    save(fig, "reinforcement")


def evaluation_units(result):
    ev = result["clustered_evaluation"]
    mean = ev["observed_mean"]
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    fig.subplots_adjust(left=.07, right=.97, top=.86, bottom=.20)
    ax.set(xlim=(-.005, .075), ylim=(-.6, 1.7), yticks=[],
           xticks=[0, .02, .04, .06], xlabel="Mean paired score difference (synthetic units)")
    for y, key, label, color in [
        (1, "cluster_95_interval", "40 independent user averages", BLUE),
        (0, "naive_95_interval", "480 rows treated as independent", ORANGE),
    ]:
        lo, hi = ev[key]
        ax.errorbar([mean], [y], xerr=[[mean - lo], [hi - mean]], fmt="o",
                    color=color, linewidth=2.2, markersize=7, capsize=6, zorder=5)
        ax.text(-.004, y + .33, label, color=color, fontsize=12, fontweight="semibold")
        ax.text(.074, y - .25, f"95% CI: {lo:.4f} to {hi:.4f}",
                color=MUTED, fontsize=10.5, ha="right")
    ax.axvline(ev["true_difference"], color=MUTED, linestyle=(0, (3, 3)), linewidth=1.1)
    ax.axvline(0, color="#aebdc7", linewidth=.8)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID)
    ax.tick_params(length=0, pad=7)
    fig.text(.07, .95, f"Same estimate: {mean:.4f}. Different claims about independence.",
             fontsize=12.5, color=INK)
    fig.text(.07, .015, "Dashed line: known effect = 0.0200. Correct sampling units preserve uncertainty.",
             fontsize=10.5, color=MUTED)
    save(fig, "evaluation-units")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    setup()
    result = json.loads((ROOT / "research/luck/results/summary.json").read_text())
    reference_distributions()
    posterior_predictive()
    shared_environment()
    prediction_and_control()
    selection_and_regression(result)
    reinforcement()
    evaluation_units(result)
    calculated = {
        "plug_in_posterior_mean": {
            "p": 13 / 22,
            "mean": 20 * 13 / 22,
            "probability_15_or_more": float(binom.sf(14, 20, 13 / 22)),
            "variance": float(binom.var(20, 13 / 22)),
        },
        "posterior_predictive": {
            "mean": float(betabinom.mean(20, 13, 9)),
            "probability_15_or_more": float(betabinom.sf(14, 20, 13, 9)),
            "variance": float(betabinom.var(20, 13, 9)),
        },
        "shared_environment": {
            "mean": float(betabinom.mean(20, 2.4, 1.6)),
            "intraclass_correlation": 0.2,
            "probability_15_or_more": float(betabinom.sf(14, 20, 2.4, 1.6)),
            "variance": float(betabinom.var(20, 2.4, 1.6)),
        },
        "figures": ["reference-distributions", "posterior-predictive", "shared-environment",
                    "prediction-and-control", "selection-and-regression", "reinforcement",
                    "evaluation-units"],
    }
    (ROOT / "research/luck/results/graphics.json").write_text(json.dumps(calculated, indent=2) + "\n")
    print(json.dumps(calculated, indent=2))


if __name__ == "__main__":
    main()
