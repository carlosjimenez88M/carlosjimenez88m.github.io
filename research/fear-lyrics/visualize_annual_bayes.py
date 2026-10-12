#!/usr/bin/env python3
"""Offline Beta-binomial graphics for seven interpreted, paired-read motifs.

The success is a song with a consensual fear reference AND consensual motif.
All twelve selected songs remain in each period-language denominator, including
unresolved readings. This operational rate describes the selected corpus; it
does not estimate historical population prevalence. The one-reader scenario is
a decision sensitivity, not an alternative gold standard. No lyrics are read.

Run with .research-cache/fear-lyrics/.venv/bin/python
research/fear-lyrics/visualize_annual_bayes.py
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import brentq
from scipy.special import betainc, betaincc, betaln
from scipy.stats import beta

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
INPUT = ROOT / "results/annual-120"
OUT = INPUT / "bayes"
FIGURES = ROOT / "figures"
PUBLIC = REPO / "static/research/fear-lyrics"
os.environ.setdefault("MPLCONFIGDIR", str(REPO / ".research-cache/fear-lyrics/mpl-annual-bayes"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, PercentFormatter

MOTIFS = {
    "surveillance_pursuit": "Vigilancia y persecución",
    "collective_violence": "Violencia colectiva",
    "loss_abandonment": "Abandono y pérdida",
    "fragility_identity": "Fragilidad e identidad",
    "uncertainty_change": "Incertidumbre y cambio",
    "relational_coercion": "Coerción relacional",
    "supernatural": "Sobrenatural",
}
PERIODS = (1980, 1990, 2000, 2010, 2020)
PERIOD_LABELS = ("1980–89", "1990–99", "2000–09", "2010–19", "2020–26*")
LANGUAGES = ("es", "en")
LANGUAGE_NAMES = {"es": "Español", "en": "Inglés"}
PRIORS = {"Jeffreys": (.5, .5), "uniform": (1., 1.)}
PRIOR_NAMES = {"Jeffreys": "Jeffreys · Beta(½,½)", "uniform": "Uniforme · Beta(1,1)"}
COLORS = {"Jeffreys": "#8054a1", "uniform": "#bd8134"}
SCENARIOS = ("consensus", "one_reader")
INPUT_NAMES = ("annotations.json", "annual_presence.csv", "periods.csv", "endpoint_sensitivity.csv")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "svg.fonttype": "none", "svg.hashsalt": "annual-bayes-seven-motifs-v1",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "savefig.facecolor": "white"})


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_read(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(name, records, outputs):
    path = OUT / f"annual-bayes-{name}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    target = PUBLIC / path.name
    shutil.copy2(path, target)
    outputs.append({"path": str(path.relative_to(REPO)), "public_path": str(target.relative_to(REPO)),
                    "sha256": digest(path), "bytes": path.stat().st_size, "rows": len(records)})


def save(fig, name, outputs):
    for extension in ("svg", "png"):
        path = FIGURES / f"annual-bayes-{name}.{extension}"
        metadata = {"Date": None, "Creator": "visualize_annual_bayes.py"} if extension == "svg" else {"Software": "visualize_annual_bayes.py"}
        fig.savefig(path, dpi=160, metadata=metadata)
        target = PUBLIC / path.name
        shutil.copy2(path, target)
        outputs.append({"path": str(path.relative_to(REPO)), "public_path": str(target.relative_to(REPO)),
                        "sha256": digest(path), "bytes": path.stat().st_size})
    plt.close(fig)


def prepare_counts():
    annotations = json.loads((INPUT / "annotations.json").read_text(encoding="utf-8"))
    annual = csv_read(INPUT / "annual_presence.csv")
    periods = csv_read(INPUT / "periods.csv")
    endpoint = csv_read(INPUT / "endpoint_sensitivity.csv")
    assert len(annotations) == len({r["song_id"] for r in annotations}) == 120
    assert all(len(r["readers"]) == len(set(r["readers"])) == 2 for r in annotations)
    assert Counter(r["fear_reference_consensus"] for r in annotations) == {True: 53, False: 37, None: 30}
    assert all(r["language"] in LANGUAGES and 1980 <= r["year"] <= 2026 for r in annotations)
    assert all(r["decade"] == r["year"] // 10 * 10 for r in annotations)
    cell = defaultdict(list)
    for r in annotations:
        assert set(r["consensus"]["subtopics"]) <= set(MOTIFS)
        assert not r["consensus"]["subtopics"] or r["fear_reference_consensus"] is True
        assert set(r["consensus"]["subtopics"]) <= set(r["one_reader_subtopics"]) <= set(MOTIFS)
        cell[(r["language"], r["decade"])].append(r)
    assert set(cell) == {(language, period) for language in LANGUAGES for period in PERIODS}
    assert all(len(group) == 12 for group in cell.values())
    assert len(periods) == 5 and len(endpoint) == 7
    for row in periods:
        group = [r for r in annotations if r["decade"] == int(row["decade"])]
        assert int(row["songs"]) == len(group) == 24
        assert int(row["fear_songs"]) == sum(r["fear_reference_consensus"] is True for r in group)
        assert int(row["reference_unresolved"]) == sum(r["fear_reference_consensus"] is None for r in group)
        assert int(row["fear_one_reader"]) == sum(r["one_reader_fear"] for r in group)
    annual_index = {(int(r["year"]), r["language"], r["subtopic"]): r for r in annual}
    assert len(annual_index) == len(annual) == 47 * 3 * 7
    for year in range(1980, 2027):
        for language in ("balanced", *LANGUAGES):
            group = [r for r in annotations if r["year"] == year and (language == "balanced" or r["language"] == language)]
            for motif, label in MOTIFS.items():
                row = annual_index[(year, language, motif)]
                assert row["label"] == label and int(row["songs"]) == len(group)
                assert int(row["count"]) == sum(motif in r["consensus"]["subtopics"] for r in group)
                assert int(row["one_reader_count"]) == sum(motif in r["one_reader_subtopics"] for r in group)
    counts = []
    for motif, label in MOTIFS.items():
        for language in LANGUAGES:
            for period, period_label in zip(PERIODS, PERIOD_LABELS):
                group = cell[(language, period)]
                for scenario in SCENARIOS:
                    successes = sum(motif in r["consensus"]["subtopics"] if scenario == "consensus"
                                    else motif in r["one_reader_subtopics"] for r in group)
                    counts.append(dict(subtopic=motif, label=label, language=language, decade=period,
                                       period=period_label, scenario=scenario, songs=len(group), successes=successes,
                                       observed_rate=successes / len(group),
                                       fear_songs=sum(r["fear_reference_consensus"] is True for r in group),
                                       reference_unresolved=sum(r["fear_reference_consensus"] is None for r in group)))
    context = []
    for row in endpoint:
        motif = row["subtopic"]
        first = [r for r in counts if r["subtopic"] == motif and r["decade"] == 1980]
        last = [r for r in counts if r["subtopic"] == motif and r["decade"] == 2020]
        strict_first = sum(r["successes"] for r in first if r["scenario"] == "consensus")
        strict_last = sum(r["successes"] for r in last if r["scenario"] == "consensus")
        strict_difference = (strict_last - strict_first) / 24
        relaxed_difference = (sum(r["successes"] for r in last if r["scenario"] == "one_reader")
                              - sum(r["successes"] for r in first if r["scenario"] == "one_reader")) / 24
        assert int(row["first_count"]) == strict_first and int(row["last_count"]) == strict_last
        assert int(row["songs_each_period"]) == 24
        assert math.isclose(float(row["observed_balanced_difference"]), strict_difference, abs_tol=1e-12)
        assert math.isclose(float(row["one_reader_difference"]), relaxed_difference, abs_tol=1e-12)
        context.append(dict(subtopic=motif, label=MOTIFS[motif], first_count=strict_first, last_count=strict_last,
                            songs_each_period=24, songs_each_language_period=12,
                            observed_balanced_difference=strict_difference, one_reader_difference=relaxed_difference,
                            leave_artist_min=float(row["leave_artist_min"]), leave_artist_max=float(row["leave_artist_max"]),
                            interpretation="Descriptive decision/artist sensitivity; no credible interval"))
    return annotations, counts, context


def beta_moments(a, b):
    return a / (a + b), a * b / ((a + b) ** 2 * (a + b + 1))


def posterior_rates(counts):
    estimates = []
    for row in counts:
        for prior, (a0, b0) in PRIORS.items():
            a, b = a0 + row["successes"], b0 + row["songs"] - row["successes"]
            mean, variance = beta_moments(a, b)
            low, high = beta.ppf([.025, .975], a, b)
            assert abs(mean - beta.mean(a, b)) < 1e-14 and abs(variance - beta.var(a, b)) < 1e-14
            assert 0 < low < mean < high < 1
            estimates.append(row | dict(prior=prior, prior_alpha=a0, prior_beta=b0,
                                         posterior_alpha=a, posterior_beta=b, posterior_mean=mean,
                                         credible_low=float(low), credible_high=float(high),
                                         posterior_variance=variance))
    return estimates


def difference_cdf(d, a_first, b_first, a_last, b_last, tolerance=2e-10):
    """P(theta_last - theta_first <= d), deterministic weighted integration.

    Integrate f_first(x) F_last(x+d). QUADPACK's algebraic endpoint weights
    handle Jeffreys posteriors with a or b=.5 without evaluating infinite PDFs.
    The CDF's interior break at x=-d or x=1-d is split explicitly.
    """
    if d <= -1:
        return 0., 0.
    if d >= 1:
        return 1., 0.
    low, high = max(0., -d), min(1., 1. - d)
    left_exponent = a_first - 1 if low == 0 else 0.
    right_exponent = b_first - 1 if high == 1 else 0.
    normalization = math.exp(-float(betaln(a_first, b_first)))

    def integrand(x):
        left = 1. if low == 0 else x ** (a_first - 1)
        right = 1. if high == 1 else (1 - x) ** (b_first - 1)
        return normalization * left * right * float(betainc(a_last, b_last, min(1., max(0., x + d))))

    with warnings.catch_warnings():
        warnings.simplefilter("error", IntegrationWarning)
        value, error = quad(integrand, low, high, weight="alg", wvar=(left_exponent, right_exponent),
                            epsabs=tolerance, epsrel=tolerance, limit=240)
    if high < 1:
        value += float(betaincc(a_first, b_first, high))
    assert -.00000001 <= value <= 1.00000001 and error < 2e-8
    return min(1., max(0., value)), error


@lru_cache(maxsize=None)
def difference_summary(a_first, b_first, a_last, b_last):
    args = (a_first, b_first, a_last, b_last)
    cdf = lambda d: difference_cdf(d, *args)[0]
    low = brentq(lambda d: cdf(d) - .025, -1., 1., xtol=2e-10)
    high = brentq(lambda d: cdf(d) - .975, -1., 1., xtol=2e-10)
    first_mean, first_var = beta_moments(a_first, b_first)
    last_mean, last_var = beta_moments(a_last, b_last)
    audits = [difference_cdf(d, *args, tolerance=2e-12) for d in (low, 0., high)]
    residual = max(abs(audits[0][0] - .025), abs(audits[2][0] - .975))
    assert residual < 1e-8
    assert abs(cdf(.17) + difference_cdf(-.17, a_last, b_last, a_first, b_first)[0] - 1) < 1e-8
    if a_first == a_last and b_first == b_last:
        assert abs(cdf(0) - .5) < 1e-9 and abs(low + high) < 1e-8
    return dict(posterior_mean_difference=last_mean - first_mean, credible_low=low, credible_high=high,
                probability_difference_positive=1 - audits[1][0], posterior_difference_variance=first_var + last_var,
                max_quadrature_error=max(r[1] for r in audits), quantile_cdf_residual=residual)


def endpoint_estimates(estimates):
    lookup = {(r["subtopic"], r["language"], r["decade"], r["scenario"], r["prior"]): r for r in estimates}
    result = []
    for motif, label in MOTIFS.items():
        for language in LANGUAGES:
            for scenario in SCENARIOS:
                for prior in PRIORS:
                    first = lookup[(motif, language, 1980, scenario, prior)]
                    last = lookup[(motif, language, 2020, scenario, prior)]
                    posterior = difference_summary(first["posterior_alpha"], first["posterior_beta"],
                                                   last["posterior_alpha"], last["posterior_beta"])
                    result.append(dict(subtopic=motif, label=label, language=language, scenario=scenario, prior=prior,
                                       first_period="1980–89", last_period="2020–26*", first_songs=first["songs"],
                                       last_songs=last["songs"], first_count=first["successes"], last_count=last["successes"],
                                       observed_difference=last["observed_rate"] - first["observed_rate"],
                                       first_posterior_alpha=first["posterior_alpha"], first_posterior_beta=first["posterior_beta"],
                                       last_posterior_alpha=last["posterior_alpha"], last_posterior_beta=last["posterior_beta"]) | posterior)
    return result


def temporal_variance(estimates):
    summary = []
    for motif, label in MOTIFS.items():
        for language in LANGUAGES:
            for prior in PRIORS:
                group = [r for r in estimates if r["subtopic"] == motif and r["language"] == language
                         and r["prior"] == prior and r["scenario"] == "consensus"]
                assert len(group) == 5
                summary.append(dict(subtopic=motif, label=label, language=language, prior=prior, periods=5,
                                    songs_each_period=12, total_successes=sum(r["successes"] for r in group),
                                    observed_temporal_variance_ddof0=float(np.var([r["observed_rate"] for r in group], ddof=0)),
                                    posterior_mean_temporal_variance_ddof0=float(np.var([r["posterior_mean"] for r in group], ddof=0)),
                                    mean_posterior_variance=float(np.mean([r["posterior_variance"] for r in group]))))
    return summary


def prior_legend():
    return [Line2D([], [], color=COLORS[p], marker="o", lw=1.8, label=PRIOR_NAMES[p]) for p in PRIORS]


def rates_figure(estimates, outputs):
    fig, axes = plt.subplots(7, 2, figsize=(15, 20), sharex=True, sharey=True)
    lookup = {(r["subtopic"], r["language"], r["decade"], r["prior"]): r for r in estimates if r["scenario"] == "consensus"}
    for i, (motif, label) in enumerate(MOTIFS.items()):
        for j, language in enumerate(LANGUAGES):
            ax = axes[i, j]
            for prior, offset in (("Jeffreys", -.08), ("uniform", .08)):
                rows = [lookup[(motif, language, period, prior)] for period in PERIODS]
                means = np.array([r["posterior_mean"] for r in rows])
                bounds = np.array([[r["credible_low"], r["credible_high"]] for r in rows])
                ax.errorbar(np.arange(5) + offset, means, yerr=[means - bounds[:, 0], bounds[:, 1] - means],
                            fmt="o", color=COLORS[prior], markersize=5, capsize=3, lw=1.5)
            raw = [lookup[(motif, language, period, "Jeffreys")] for period in PERIODS]
            ax.scatter(range(5), [r["observed_rate"] for r in raw], marker="D", s=25,
                       facecolors="white", edgecolors="#203647", linewidths=1, zorder=4)
            for x, r in enumerate(raw):
                ax.text(x, .80, f"{r['successes']}/{r['songs']}", ha="center", color="#52616c", fontsize=10)
            ax.set_title(f"{label} · {LANGUAGE_NAMES[language]}", loc="left", fontsize=12)
            ax.set_ylim(-.025, .865)
            ax.set_xlim(-.4, 4.4)
            ax.set_yticks([0, .25, .50, .75])
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_xticks(range(5), PERIOD_LABELS)
            ax.tick_params(axis="x", labelbottom=True, labelsize=9)
            ax.grid(axis="y", color="#e8ecee", lw=.7)
    fig.suptitle("Miedo y siete motivos: recuentos y dos priors", x=.055, ha="left", y=.987, fontsize=22)
    fig.text(.055, .962, "Éxito = referencia al miedo y motivo con dos votos. Denominador = TODAS las letras de cada celda (12).", fontsize=12)
    handles = prior_legend() + [Line2D([], [], marker="D", linestyle="none", markerfacecolor="white", color="#203647", label="Recuento observado · c/12")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.05, .050), ncol=3, frameon=False)
    fig.text(.055, .042, "Puntos: medias posteriores. Barras: intervalos centrales del 95%. Los números superiores muestran c/12.", fontsize=10)
    fig.text(.055, .029, "120 letras seleccionadas · 53 referencias consensuadas al miedo · 37 ausencias · 30 sin resolver. Motivos superpuestos.", fontsize=10)
    fig.text(.055, .016, "Los casos sin consenso permanecen en n; c cuenta únicamente etiquetas consensuadas. No es prevalencia histórica poblacional.", fontsize=10)
    fig.text(.055, .005, "* 2020–26 termina el 11-10-2026. Cada celda usa su propio prior fijo y no comparte información con otras.", fontsize=9, color="#52616c")
    fig.subplots_adjust(top=.925, bottom=.10, left=.055, right=.975, hspace=.47, wspace=.15)
    save(fig, "priors", outputs)


def variance_figure(summary, outputs):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.8), sharex=True)
    labels = list(MOTIFS.values())
    maximum = max(max(r["observed_temporal_variance_ddof0"], r["mean_posterior_variance"]) for r in summary) * 1000 * 1.25
    for ax, language in zip(axes, LANGUAGES):
        group = {(r["subtopic"], r["prior"]): r for r in summary if r["language"] == language}
        for i, motif in enumerate(MOTIFS):
            observed = group[(motif, "Jeffreys")]["observed_temporal_variance_ddof0"] * 1000
            ax.barh(i - .23, observed, height=.18, color="#263c4e")
            ax.text(observed + maximum * .012, i - .23, f"{observed:.1f}", va="center", fontsize=9, color="#263c4e")
            for prior, offset in (("Jeffreys", 0), ("uniform", .23)):
                variance = group[(motif, prior)]["mean_posterior_variance"] * 1000
                ax.barh(i + offset, variance, height=.18, color=COLORS[prior])
                ax.text(variance + maximum * .012, i + offset, f"{variance:.1f}", va="center", fontsize=9, color=COLORS[prior])
        ax.set_yticks(range(7), labels)
        ax.invert_yaxis()
        ax.set_xlim(0, maximum)
        ax.set_xlabel("Varianza de una tasa × 1.000")
        ax.set_title(LANGUAGE_NAMES[language], loc="left", fontsize=15)
        ax.grid(axis="x", color="#e8ecee", zorder=0)
        ax.set_axisbelow(True)
    fig.suptitle("Variación entre periodos y duda dentro de cada periodo", x=.045, ha="left", y=.985, fontsize=21)
    fig.text(.045, .925, "Consenso estricto · Siete motivos nombrados · Cinco periodos por idioma · 12 letras en cada periodo e idioma.", fontsize=11)
    handles = [Line2D([], [], color="#263c4e", lw=7, label="Variación temporal observada · Var(c/12)")]
    handles += [Line2D([], [], color=COLORS[p], lw=7, label=f"Duda posterior media · {PRIOR_NAMES[p]}") for p in PRIORS]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.04, .092), ncol=1, frameon=False, fontsize=10)
    fig.text(.045, .068, "Oscuro: varianza descriptiva de las cinco tasas observadas, con divisor 5. Color: promedio de las cinco varianzas posteriores Beta.", fontsize=10)
    fig.text(.045, .042, "Varianza posterior exacta = a·b / [(a+b)²·(a+b+1)]. No se calcula a partir del ancho de un intervalo.", fontsize=10)
    fig.text(.045, .017, "Estas cantidades no son intercambiables: variación observada y incertidumbre del modelo no miden una tendencia histórica poblacional.", fontsize=10)
    fig.subplots_adjust(top=.86, bottom=.255, left=.19, right=.96, wspace=.73)
    save(fig, "variance", outputs)


def endpoints_figure(endpoints, context, outputs):
    fig, axes = plt.subplots(1, 3, figsize=(20, 11), gridspec_kw={"width_ratios": [1, 1, .90]})
    lookup = {(r["subtopic"], r["language"], r["scenario"], r["prior"]): r for r in endpoints}
    extent = max(abs(r[k]) for r in endpoints for k in ("credible_low", "credible_high"))
    extent = math.ceil((extent + .025) * 10) / 10
    offsets = {("consensus", "Jeffreys"): -.25, ("consensus", "uniform"): -.09,
               ("one_reader", "Jeffreys"): .09, ("one_reader", "uniform"): .25}
    for ax, language in zip(axes[:2], LANGUAGES):
        labels = []
        for i, (motif, label) in enumerate(MOTIFS.items()):
            strict = lookup[(motif, language, "consensus", "Jeffreys")]
            relaxed = lookup[(motif, language, "one_reader", "Jeffreys")]
            labels.append(f"{label}\nC {strict['first_count']}/12 → {strict['last_count']}/12 · U {relaxed['first_count']}/12 → {relaxed['last_count']}/12")
            for scenario in SCENARIOS:
                for prior in PRIORS:
                    r = lookup[(motif, language, scenario, prior)]
                    y = i + offsets[(scenario, prior)]
                    ax.plot([r["credible_low"], r["credible_high"]], [y, y], color=COLORS[prior], lw=1.6,
                            linestyle="-" if scenario == "consensus" else "--", alpha=.9)
                    ax.scatter(r["posterior_mean_difference"], y, color=COLORS[prior], s=26,
                               marker="o" if scenario == "consensus" else "^", zorder=3)
        ax.set_yticks(range(7), labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(-extent, extent)
        ax.axvline(0, color="#556774", lw=1.0)
        ax.grid(axis="x", color="#e8ecee")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{100 * value:.0f}"))
        ax.set_title(f"{LANGUAGE_NAMES[language]} · posterior e intervalo 95%", loc="left", fontsize=12)
        ax.set_xlabel("Diferencia de tasas · puntos porcentuales")
    ax = axes[2]
    for i, row in enumerate(context):
        ax.plot([row["leave_artist_min"], row["leave_artist_max"]], [i - .08] * 2, color="#74838e", lw=4, alpha=.7)
        ax.scatter(row["observed_balanced_difference"], i - .08, color="#203647", marker="D", s=33, zorder=3)
        ax.scatter(row["one_reader_difference"], i + .15, color="#203647", marker="^", s=32, zorder=3)
    ax.set_yticks(range(7), [f"{r['label']}\nConsenso {r['first_count']}/24 → {r['last_count']}/24" for r in context], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(-extent, extent)
    ax.axvline(0, color="#556774", lw=1)
    ax.grid(axis="x", color="#e8ecee")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{100 * value:.0f}"))
    ax.set_title("Balance EN/ES\nSensibilidad descriptiva", loc="left", fontsize=12)
    ax.set_xlabel("Diferencia observada\n(puntos porcentuales)", fontsize=10)
    fig.suptitle("Entre extremos: 2020–26 menos 1980–89", x=.025, ha="left", y=.982, fontsize=23)
    fig.text(.025, .937, "C = consenso de miedo + motivo. U = al menos una lectura positiva: sensibilidad de decisión, conservando todas las canciones en n.", fontsize=11)
    handles = prior_legend() + [Line2D([], [], marker="o", color="#52616c", label="C · consenso", lw=1.5),
                               Line2D([], [], marker="^", color="#52616c", label="U · una lectura", linestyle="--", lw=1.5)]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.02, .135), ncol=4, frameon=False, fontsize=10)
    fig.text(.025, .113, "ES/EN: diferencia de dos Beta independientes; puntos = medias, barras = intervalos centrales del 95% por integración determinista.", fontsize=10)
    fig.text(.025, .087, "Panel derecho: ◇ diferencia observada C; ▲ diferencia U; trazo gris = rango al retirar un artista. Ese rango NO es un intervalo posterior.", fontsize=10)
    fig.text(.025, .061, "ES y EN mantienen n=12 en cada extremo; el balance descriptivo da igual peso a ambos idiomas (24 canciones por extremo).", fontsize=10)
    fig.text(.025, .035, "Un intervalo que cruza cero deja abiertas ambas direcciones bajo ese prior. Los priors y U son sensibilidades del análisis del corpus seleccionado.", fontsize=10)
    fig.text(.025, .010, "* Corte 11-10-2026. No representa el cambio de una población histórica ni independencia entre motivos, que pueden coexistir.", fontsize=10)
    fig.subplots_adjust(top=.865, bottom=.215, left=.16, right=.985, wspace=1.22)
    save(fig, "endpoints", outputs)


def main():
    for directory in (OUT, FIGURES, PUBLIC):
        directory.mkdir(parents=True, exist_ok=True)
    annotations, counts, context = prepare_counts()
    estimates = posterior_rates(counts)
    # Boundary checks include zero and full counts, even if the actual motifs are sparse.
    for a0, b0 in PRIORS.values():
        for k_first, k_last in ((0, 0), (0, 12), (12, 0), (12, 12), (6, 6)):
            difference_summary(a0 + k_first, b0 + 12 - k_first, a0 + k_last, b0 + 12 - k_last)
    endpoints = endpoint_estimates(estimates)
    variances = temporal_variance(estimates)
    outputs = []
    write_csv("counts", counts, outputs)
    write_csv("posterior-rates", estimates, outputs)
    write_csv("endpoint-posteriors", endpoints, outputs)
    write_csv("endpoint-observed-sensitivity", context, outputs)
    write_csv("variance", variances, outputs)
    rates_figure(estimates, outputs)
    variance_figure(variances, outputs)
    endpoints_figure(endpoints, context, outputs)
    receipt = dict(status="executed", generated_utc=datetime.now(timezone.utc).isoformat(),
                   script_sha256=digest(Path(__file__)), input_hashes={name: digest(INPUT / name) for name in INPUT_NAMES},
                   source_files=list(INPUT_NAMES), public_outputs=outputs,
                   songs=len(annotations), readings=sum(len(r["readers"]) for r in annotations),
                   fear_reference_counts={"consensus_present": 53, "consensus_absent": 37, "unresolved": 30},
                   motifs=MOTIFS, periods=list(PERIOD_LABELS), languages=list(LANGUAGES),
                   songs_each_period_language=12, priors={p: {"alpha": a, "beta": b} for p, (a, b) in PRIORS.items()},
                   estimand="Operational corpus rate: both readings support fear and the named motif, divided by ALL selected songs in the period-language cell.",
                   unresolved_handling="Unresolved cases remain in denominators; absence of a consensus label is coded zero for this operational target, not established absence of real fear.",
                   one_reader_sensitivity="At least one positive fear reading supports the motif; denominator unchanged. Decision sensitivity, not validation or an independent provider.",
                   model="Independent binomial likelihood and fixed Beta prior in each cell; no temporal pooling, no hyperparameter fitting.",
                   finite_corpus_counts="Observed c/n is known for these selected songs. Posterior variance expresses uncertainty conditional on the working binomial model, not uncertainty that a label is true.",
                   rate_posterior="Beta(prior_alpha+c, prior_beta+n-c), exact means/variances and equal-tail .025/.975 quantiles.",
                   variance_definition="Observed: population variance across five c/n values, ddof=0. Posterior uncertainty: mean of five exact a*b/((a+b)^2*(a+b+1)) values.",
                   endpoint_method="CDF of theta_last-theta_first = integral f_first(x)*F_last(x+d) dx; endpoint-weighted deterministic quadrature, no Monte Carlo.",
                   endpoint_variance="Exact sum of two independent Beta variances.",
                   max_endpoint_quadrature_error=max(r["max_quadrature_error"] for r in endpoints),
                   max_endpoint_quantile_cdf_residual=max(r["quantile_cdf_residual"] for r in endpoints),
                   checks={"annotation_unique_ids": 120, "two_readers_each": True, "annual_counts_reconciled": True,
                           "period_counts_reconciled": True, "endpoint_counts_reconciled": True,
                           "boundary_and_symmetry_checks": True, "beta_moments_checked_against_scipy": True,
                           "count_rows": len(counts), "posterior_rows": len(estimates), "endpoint_rows": len(endpoints), "variance_rows": len(variances)},
                   scientific_scope="Selected song corpus, not historical population prevalence. Motifs can cooccur; no human gold labels or external validation are claimed.",
                   lyrics_read=False, project_api_calls=0, downloads=0,
                   dependencies={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "matplotlib")})
    receipt_path = OUT / "annual-bayes-receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    shutil.copy2(receipt_path, PUBLIC / receipt_path.name)
    strict = [r for r in endpoints if r["scenario"] == "consensus"]
    excluded_zero = [r for r in strict if r["credible_low"] > 0 or r["credible_high"] < 0]
    print(json.dumps(dict(songs=120, motifs=7, denominator=12, counts=len(counts), posteriors=len(estimates),
                          endpoints=len(endpoints), variance_rows=len(variances), artifacts=len(outputs),
                          strict_endpoint_intervals_excluding_zero=len(excluded_zero),
                          max_quantile_residual=receipt["max_endpoint_quantile_cdf_residual"], receipt=str(receipt_path.relative_to(REPO))), ensure_ascii=False))


if __name__ == "__main__":
    main()
