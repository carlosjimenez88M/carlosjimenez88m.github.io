#!/usr/bin/env python3
"""Readable Bayesian graphics from executed PUBLIC topic-presence results.

No lyrics, model inference, APIs, new topic fits or Monte Carlo are needed.
The grid is exactly the existing bayes.py quadrature. Posterior variance is
calculated from moments of its Beta mixture, never from interval width.

    .research-cache/fear-lyrics/.venv/bin/python research/fear-lyrics/visualize_bayes_readable.py
"""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(REPO / ".research-cache/fear-lyrics/mpl-bayes-readable"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter, MaxNLocator

from bayes import full_grid, make_key, slug

MODELS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "intfloat/multilingual-e5-small",
    "openai/text-embedding-3-large",
    "openai/text-embedding-3-small",
]
NAMES = dict(zip(MODELS, ["MiniLM multilingüe", "E5-small multilingüe", "OpenAI 3-large", "OpenAI 3-small"]))
COHORT_NAMES = {"matched40": "40 canciones compartidas por cuatro encoders", "expanded120": "120 canciones: ampliación con dos encoders locales"}
METHODS = ["Jeffreys", "uniform", "EB_plugin", "hierarchical_grid"]
METHOD_NAMES = {"Jeffreys": "Jeffreys · Beta(½,½)", "uniform": "Uniforme · Beta(1,1)",
                "EB_plugin": "Bayes empírico (EB)", "hierarchical_grid": "Bayes jerárquico"}
COLORS = {"Jeffreys": "#8a5ea6", "uniform": "#bc8036", "EB_plugin": "#cd5845", "hierarchical_grid": "#137a8b"}
PERIODS = [1980, 1990, 2000, 2010, 2020]
PERIOD_LABELS = ["1980–89", "1990–99", "2000–09", "2010–19", "2020–26*"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none",
                     "axes.titleweight": "bold", "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.facecolor": "white"})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def csv_write(path, records):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)


def context_rows(counts, model, topic, context):
    selected = [r for r in counts if r["model"] == model and r["topic_id"] == topic]
    if context.startswith("language_"):
        selected = [r for r in selected if r["language"] == context.split("_", 1)[1]]
    assert len(selected) == (10 if context == "pooled_languages" else 5)
    return selected


def variance_moments(row, counts):
    """Exact Beta or numerical-quadrature mixture moments and total variance."""
    if row["method"] != "hierarchical_grid":
        a, b = float(row["posterior_alpha"]), float(row["posterior_beta"])
        mean = a / (a + b)
        within = a * b / ((a + b) ** 2 * (a + b + 1))
        return mean, within, within, 0.0
    group = context_rows(counts, row["model"], row["topic_id"], row["prior_context"])
    grid = full_grid(make_key([r["successes"] for r in group], [r["songs"] for r in group]), 64, 81)
    a = grid["a"] + row["successes"]
    b = grid["b"] + row["songs"] - row["successes"]
    w = grid["weights"]
    conditional_mean = a / (a + b)
    mean = float(np.sum(w * conditional_mean))
    within = float(np.sum(w * a * b / ((a + b) ** 2 * (a + b + 1))))
    between = float(np.sum(w * (conditional_mean - mean) ** 2))
    total = within + between
    second = float(np.sum(w * a * (a + 1) / ((a + b) * (a + b + 1))))
    assert abs(total - (second - mean * mean)) < 1e-12
    return mean, total, within, between


def prepare(directory, cohort):
    raw = rows(directory / "bayes/cell_counts.csv")
    for r in raw:
        for k in ("topic_id", "decade", "songs", "successes"): r[k] = int(r[k])
        r["raw_rate"] = float(r["raw_rate"])
    estimates = rows(directory / "bayes/posterior_rates.csv")
    largest_error = 0.0
    for r in estimates:
        for k in ("topic_id", "decade", "songs", "successes"): r[k] = int(r[k])
        for k in ("raw_rate", "posterior_mean", "credible_low", "credible_high"): r[k] = float(r[k])
        mean, variance, within, between = variance_moments(r, raw)
        largest_error = max(largest_error, abs(mean - r["posterior_mean"]))
        assert abs(mean - r["posterior_mean"]) < 1e-10
        assert 0 <= variance <= .25 + 1e-12
        r.update(posterior_variance=variance, conditional_variance_component=within,
                 hyperparameter_mean_variance_component=between, cohort=cohort)
    selected = {r["model"]: int(r["topic_id"]) for r in rows(directory / "bayes/figure_selection.csv")}
    summary = []
    groups = defaultdict(list)
    for r in estimates:
        if r["topic_id"] >= 0 and r["prior_context"] in {"fixed", "pooled_languages"}:
            groups[(r["model"], r["topic_id"], r["language"], r["method"])].append(r)
    for (model, topic, language, method), group in sorted(groups.items()):
        assert len(group) == 5 and {r["decade"] for r in group} == set(PERIODS)
        summary.append(dict(cohort=cohort, model=model, topic_id=topic, language=language, method=method,
            periods=5, observed_temporal_variance_population_ddof0=float(np.var([r["raw_rate"] for r in group], ddof=0)),
            posterior_mean_temporal_variance_population_ddof0=float(np.var([r["posterior_mean"] for r in group], ddof=0)),
            mean_posterior_variance=float(np.mean([r["posterior_variance"] for r in group])),
            mean_conditional_variance_component=float(np.mean([r["conditional_variance_component"] for r in group])),
            mean_hyperparameter_mean_variance_component=float(np.mean([r["hyperparameter_mean_variance_component"] for r in group])),
            songs_per_period=group[0]["songs"], total_topic_presence=sum(r["successes"] for r in group)))
    return raw, estimates, selected, summary, largest_error


def main_records(estimates, model, method):
    context = "fixed" if method in {"Jeffreys", "uniform"} else "pooled_languages"
    return [r for r in estimates if r["model"] == model and r["topic_id"] >= 0 and
            r["method"] == method and r["prior_context"] == context]


def axes_grid(models, height=10):
    cols = 2
    nr = math.ceil(len(models) / cols)
    fig, axes = plt.subplots(nr, cols, figsize=(16, height if nr == 2 else height * .62), squeeze=False)
    for ax in axes.ravel()[len(models):]: ax.set_visible(False)
    return fig, axes.ravel()


def save(fig, name, outputs):
    root_out = ROOT / "figures"
    public_out = REPO / "static/research/fear-lyrics"
    root_out.mkdir(parents=True, exist_ok=True); public_out.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        path = root_out / f"{name}.{ext}"
        fig.savefig(path, dpi=150)
        target = public_out / path.name
        shutil.copy2(path, target)
        outputs.append(dict(path=str(path.relative_to(REPO)), sha256=sha(path), bytes=path.stat().st_size,
                            public_path=str(target.relative_to(REPO))))
    plt.close(fig)


def shrinkage(estimates, models, cohort, outputs):
    fig, axes = axes_grid(models)
    n = next(r["songs"] for r in estimates)
    for ax, model in zip(axes, models):
        post = main_records(estimates, model, "hierarchical_grid")
        eb = {(r["topic_id"], r["decade"], r["language"]): r for r in main_records(estimates, model, "EB_plugin")}
        ax.plot([0, 1], [0, 1], "--", color="#afb7be", lw=1.3, zorder=0)
        for r in post:
            key = (r["topic_id"], r["decade"], r["language"])
            other = eb[key]
            area = 23 + 105 * r["successes"] / n
            ax.plot([r["raw_rate"]] * 2, [other["posterior_mean"], r["posterior_mean"]], color="#c2c8ce", lw=.65, alpha=.6)
            ax.scatter(r["raw_rate"], other["posterior_mean"], s=area, facecolors="none", edgecolors="#78818a", alpha=.38, linewidth=.65)
            ax.scatter(r["raw_rate"], r["posterior_mean"], s=area, c=COLORS["hierarchical_grid"], alpha=.55, edgecolors="white", linewidth=.35, zorder=3)
        extreme = sorted(post, key=lambda r: (-abs(r["posterior_mean"] - r["raw_rate"]), r["topic_id"], r["decade"], r["language"]))[0]
        key = (extreme["topic_id"], extreme["decade"], extreme["language"])
        label = f"T{extreme['topic_id']} · {extreme['decade']} · {extreme['language'].upper()}\n{extreme['successes']}/{n} → {extreme['posterior_mean']:.0%}"
        x, y = extreme["raw_rate"], extreme["posterior_mean"]
        ax.annotate(label, (x, y), xytext=(.40 if x < .5 else .07, .80 if y < .5 else .19),
                    textcoords="axes fraction", fontsize=10, arrowprops=dict(arrowstyle="-", color="#283c4f", lw=.9),
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=3))
        ax.text(.02, .98, f"{len(post)} celdas · {len(post)//10} tópicos", transform=ax.transAxes, va="top", fontsize=10, color="#56636d")
        ax.set(title=NAMES[model], xlim=(-.035, 1.035), ylim=(-.035, 1.035),
               xlabel="Presencia observada en canciones · c/n", ylabel="Estimación de la presencia · media posterior")
        ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xticks([0, .25, .5, .75, 1]); ax.set_yticks([0, .25, .5, .75, 1]); ax.grid(color="#edf0f2", lw=.7)
    handles = [Line2D([], [], marker="o", linestyle="none", color=COLORS["hierarchical_grid"], label="Bayes jerárquico"),
               Line2D([], [], marker="o", linestyle="none", markerfacecolor="none", color="#78818a", label="Bayes empírico"),
               Line2D([], [], linestyle="--", color="#afb7be", label="Sin cambio: y = x")]
    fig.suptitle(f"Del recuento a la estimación: ¿cuánto cambia una tasa?\n{COHORT_NAMES[cohort]} · n={n} por periodo e idioma", x=.06, ha="left", y=.99, fontsize=20)
    fig.text(.06, .90 if len(models) == 4 else .85,
             "Cada punto = tópico × periodo × idioma. Área proporcional a c/n con mínimo para c=0; n es igual en todas las celdas.", fontsize=10)
    fig.legend(handles=handles, ncol=3, loc="lower left", bbox_to_anchor=(.05, .057), frameon=False)
    fig.text(.06, .043, "Se muestran todos los tópicos excepto ruido; los ceros permanecen. Ejemplo anotado: mayor desplazamiento |posterior−c/n| por encoder.", fontsize=9, color="#56636d")
    fig.text(.06, .016, "Pool EN/ES = supuesto de intercambio. Presencia de un tópico ≠ miedo. Los IDs de tópicos pertenecen a cada encoder y no están alineados.", fontsize=9, color="#56636d")
    fig.subplots_adjust(top=.86 if len(models) == 4 else .77, bottom=.16 if len(models) == 4 else .26, left=.07, right=.97, hspace=.37, wspace=.24)
    save(fig, f"bayes-readable-shrinkage-{cohort}", outputs)


def priors(estimates, models, selected, directory, cohort, outputs):
    fig, axes = axes_grid(models, 14)
    offsets = dict(zip(METHODS, [-.28, -.10, .10, .28]))
    for ax, model in zip(axes, models):
        topic = selected[model]
        raw = sorted([r for r in main_records(estimates, model, "hierarchical_grid") if r["topic_id"] == topic], key=lambda r: (r["decade"], r["language"]))
        cell_index = {(r["decade"], r["language"]): i for i, r in enumerate(raw)}
        info = json.loads((directory / slug(model) / "topics.json").read_text())
        terms = next(r["terms"][:3] for r in info["topics"] if int(r["topic_id"]) == topic)
        for method in METHODS:
            for r in main_records(estimates, model, method):
                if r["topic_id"] != topic: continue
                y = cell_index[(r["decade"], r["language"])] + offsets[method]
                dashed = method == "EB_plugin" and r["eb_boundary_diagnostic"] == "True"
                ax.plot([r["credible_low"], r["credible_high"]], [y, y], color=COLORS[method], lw=1.6, alpha=.7, linestyle="--" if dashed else "-")
                ax.scatter(r["posterior_mean"], y, color=COLORS[method], s=20, zorder=3)
        for i, r in enumerate(raw): ax.scatter(r["raw_rate"], i, facecolors="white", edgecolors="#142938", marker="D", s=30, zorder=4)
        ax.set_yticks(range(10), [f"{PERIOD_LABELS[PERIODS.index(r['decade'])]} · {r['language'].upper()} ({r['successes']}/{r['songs']})" for r in raw])
        ax.invert_yaxis(); ax.set_xlim(-.03, 1.03); ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xticks([0, .25, .5, .75, 1]); ax.grid(axis="x", color="#e6eaed")
        ax.set_title(f"{NAMES[model]} · T{topic}\n" + " / ".join(t["term"] for t in terms), loc="left", fontsize=13, pad=13)
        ax.set_xlabel("Presencia: observada ◇ y media posterior ● con intervalo del 95%")
    fig.suptitle(f"El mismo recuento, cuatro supuestos bayesianos\n{COHORT_NAMES[cohort]}", x=.04, ha="left", y=.985, fontsize=21)
    fig.text(.04, .925 if len(models) == 4 else .89, "Un tópico por encoder: el de mayor varianza temporal media de c/n dentro de cada idioma. Selección exploratoria explícita.", fontsize=10)
    handles = [Line2D([], [], color=COLORS[m], marker="o", lw=1.8, label=METHOD_NAMES[m]) for m in METHODS]
    fig.legend(handles=handles, ncol=2, loc="lower left", bbox_to_anchor=(.035, .040), frameon=False)
    fig.text(.04, .031, "EB condiciona al hiperprior ajustado (trazos: límite); jerárquico usa su propio hiperprior débil. Los priors fijos no comparten información entre celdas.", fontsize=9, color="#56636d")
    fig.text(.04, .009, "2020–26* termina el 11-10-2026. Tasas de tópicos superpuestas; su suma no es 100%. IDs y palabras no identifican el mismo tema entre modelos.", fontsize=9, color="#56636d")
    fig.subplots_adjust(top=.86 if len(models) == 4 else .80, bottom=.13 if len(models) == 4 else .20, left=.16, right=.97, hspace=.34, wspace=.57)
    save(fig, f"bayes-readable-priors-{cohort}", outputs)


def variance_plot(summary, models, selected, cohort, outputs):
    fig, axes = axes_grid(models)
    maximum_x = max(r["observed_temporal_variance_population_ddof0"] for r in summary) * 1.12 * 1000
    maximum_y = max(r["mean_posterior_variance"] for r in summary) * 1.15 * 1000
    for ax, model in zip(axes, models):
        own = [r for r in summary if r["model"] == model]
        for method in METHODS:
            for language, marker in [("en", "o"), ("es", "^")]:
                points = [r for r in own if r["method"] == method and r["language"] == language]
                ax.scatter([r["observed_temporal_variance_population_ddof0"] * 1000 for r in points],
                           [r["mean_posterior_variance"] * 1000 for r in points],
                           s=25, color=COLORS[method], marker=marker, alpha=.52, edgecolor="white", linewidth=.35)
        t = selected[model]
        for language in ("en", "es"):
            r = next(r for r in own if r["method"] == "hierarchical_grid" and r["topic_id"] == t and r["language"] == language)
            ax.annotate(f"T{t} · {language.upper()}",
                        (r["observed_temporal_variance_population_ddof0"] * 1000, r["mean_posterior_variance"] * 1000),
                        xytext=(5, 5 if language == "en" else -14), textcoords="offset points", fontsize=9, color="#183c4b")
        ax.set(title=NAMES[model], xlabel="Variación entre cinco periodos · Var(c/n) × 10³",
               ylabel="Varianza posterior media\nVar(θ | datos) × 10³", xlim=(-maximum_x * .035, maximum_x), ylim=(0, maximum_y))
        ax.grid(color="#edf0f2"); ax.xaxis.set_major_locator(MaxNLocator(5)); ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.text(.02, .97, f"{len(own)//4} series tópico–idioma · todos los tópicos sin ruido", transform=ax.transAxes, va="top", fontsize=9, color="#56636d")
    fig.suptitle(f"Una tasa que cambia y una tasa incierta son cosas distintas\n{COHORT_NAMES[cohort]}", x=.06, ha="left", y=.99, fontsize=20)
    fig.text(.06, .91 if len(models) == 4 else .85,
             "Más a la derecha: mayor diferencia observada entre periodos. Más arriba: mayor incertidumbre media sobre θ bajo el modelo.", fontsize=10)
    handles = [Line2D([], [], color=COLORS[m], marker="o", linestyle="none", label=METHOD_NAMES[m]) for m in METHODS]
    handles += [Line2D([], [], color="#384955", marker="o", linestyle="none", label="Inglés"),
                Line2D([], [], color="#384955", marker="^", linestyle="none", label="Español")]
    fig.legend(handles=handles, ncol=3, loc="lower left", bbox_to_anchor=(.05, .048), frameon=False, fontsize=10)
    fig.text(.06, .036, "Var(c/n): varianza poblacional descriptiva de cinco tasas (ddof=0), incluye ruido de recuento. Var(θ|datos): momentos exactos de Beta o su mezcla.", fontsize=9, color="#56636d")
    fig.text(.06, .011, "Ninguna es varianza de embeddings: esa depende de coordenadas/vectores y se estudia con PCA. Los tópicos no se alinean entre encoders.", fontsize=9, color="#56636d")
    fig.subplots_adjust(top=.85 if len(models) == 4 else .77, bottom=.18 if len(models) == 4 else .28, left=.08, right=.97, hspace=.36, wspace=.28)
    save(fig, f"bayes-readable-variance-{cohort}", outputs)


def main():
    start = time.monotonic()
    outputs, receipt = [], {"executed": True, "source": "https://juliasilge.com/blog/bayesian-blues/",
        "scope": "Existing public topic-presence counts; no lyrics, API calls, embedding inference or topic refits.",
        "variance": {"beta": "a*b/((a+b)^2*(a+b+1))",
            "hierarchical": "E[Var(theta|hyper,data)]+Var(E[theta|hyper,data]); existing 64×81 Gauss-Legendre grid",
            "temporal": "population variance (ddof=0) of five observed c/n values within one topic and language",
            "not_embedding_variance": True, "interval_width_used_as_variance": False},
        "pooling": "EB and hierarchical figures use pooled_languages (ten EN/ES cells); fixed priors are independent per cell.",
        "comparability": "Shared songs within cohort, but topic ontologies/IDs are not aligned between encoders or cohort fits.",
        "topic_scope": "All non-noise topics; every zero cell retained. Noise also present in exported moments table for audit.",
        "cohorts": [], "script_sha256": sha(__file__), "bayes_source_script_sha256": sha(ROOT / "bayes.py")}
    all_moments, all_summaries = [], []
    for cohort, subdir in [("matched40", "results"), ("expanded120", "expanded/results")]:
        directory = ROOT / subdir
        counts, estimates, selected, summary, error = prepare(directory, cohort)
        models = [m for m in MODELS if any(r["model"] == m for r in counts)]
        shrinkage(estimates, models, cohort, outputs)
        priors(estimates, models, selected, directory, cohort, outputs)
        variance_plot(summary, models, selected, cohort, outputs)
        all_moments.extend({k: r[k] for k in ["cohort", "model", "method", "prior_context", "topic_id", "decade", "language", "songs", "successes", "raw_rate", "posterior_mean", "posterior_variance", "conditional_variance_component", "hyperparameter_mean_variance_component"]} for r in estimates)
        all_summaries.extend(summary)
        cohort_receipt = dict(cohort=cohort, models=models, counts_rows=len(counts), posterior_rows=len(estimates),
            nonnoise_topic_language_series=len(summary)//4, songs_per_cell=sorted({r["songs"] for r in counts}),
            moments_mean_max_absolute_disagreement_with_executed_posterior=error,
            illustrative_topic_selection=selected,
            input_hashes={str((directory / "bayes" / f).relative_to(ROOT)): sha(directory / "bayes" / f)
                          for f in ["cell_counts.csv", "posterior_rates.csv", "figure_selection.csv", "protocol.json"]},
            model_summaries=[])
        for model in models:
            own = [r for r in summary if r["model"] == model]
            cohort_receipt["model_summaries"].append(dict(model=model,
                topics=len(own)//8,
                descriptive_temporal_variance_median=float(np.median([r["observed_temporal_variance_population_ddof0"] for r in own if r["method"] == "hierarchical_grid"])),
                median_mean_posterior_variance_by_method={m: float(np.median([r["mean_posterior_variance"] for r in own if r["method"] == m])) for m in METHODS},
                maximum_mean_posterior_variance_by_method={m: float(max(r["mean_posterior_variance"] for r in own if r["method"] == m)) for m in METHODS}))
        receipt["cohorts"].append(cohort_receipt)
    for name, records in [("bayes-readable-posterior-moments.csv", all_moments), ("bayes-readable-temporal-variance.csv", all_summaries)]:
        path = ROOT / "figures" / name
        csv_write(path, records)
        target = REPO / "static/research/fear-lyrics" / name
        shutil.copy2(path, target)
        outputs.append(dict(path=str(path.relative_to(REPO)), sha256=sha(path), rows=len(records), public_path=str(target.relative_to(REPO))))
    receipt["outputs"] = outputs
    receipt["runtime_seconds"] = round(time.monotonic() - start, 3)
    path = ROOT / "figures/bayes-readable-receipt.json"
    path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    shutil.copy2(path, REPO / "static/research/fear-lyrics" / path.name)
    print(json.dumps({"executed": True, "cohorts": [r["cohort"] for r in receipt["cohorts"]],
                      "posterior_variance_rows": len(all_moments), "temporal_variance_rows": len(all_summaries),
                      "figures": 6, "receipt": str(path.relative_to(REPO)), "runtime_seconds": receipt["runtime_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
