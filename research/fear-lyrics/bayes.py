#!/usr/bin/env python3
"""Song-level topic-presence sensitivity: empirical and hierarchical Bayes.

This complement reads PUBLIC IDs/assignments, never lyric text. The observed unit
is a selected song with >= one unique block in a fixed topic, not a lyric block,
a person, or an independently validated expression of fear. Different topics can
be present in the same song; presence rates do not form a composition summing to 1.

    python bayes.py --self-test
    python bayes.py --results research/fear-lyrics/results

Inspired critically by https://juliasilge.com/blog/bayesian-blues/ (2016), whose
state-population denominator and exclusion of unmentioned states are not reused.
No model inference, network requests, API calls, MCMC, or paid services are used.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq, minimize, minimize_scalar
from scipy.special import betaln, betainc, expit, gammaln, logit, logsumexp
from scipy.stats import beta, norm

MODELS = ["sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "intfloat/multilingual-e5-small",
          "openai/text-embedding-3-large", "openai/text-embedding-3-small"]
KAPPA_BOUNDS = (.05, 200.0)
MU_EPS = 1e-4
PRIOR_LOG_KAPPA_MEAN = math.log(4.0)
PRIOR_LOG_KAPPA_SD = 1.5
SOURCE_URL = "https://juliasilge.com/blog/bayesian-blues/"
SOURCE_CODE_URL = "https://github.com/juliasilge/old_bloggy_blog/blob/master/_R/2016-09-28-Bayesian-Blues.Rmd"


def slug(model): return model.replace("/", "--")
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n"); temporary = Path(handle.name)
    temporary.replace(path)


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows); temporary = Path(handle.name)
    temporary.replace(path)


def bb_logpmf(k, n, a, b):
    """Beta-binomial log mass, including the binomial normalizer."""
    k, n = np.asarray(k), np.asarray(n)
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1) + betaln(k + a, n - k + b) - betaln(a, b)


def make_key(k, n):
    if len(k) != len(n) or not len(k): raise ValueError("Nonempty aligned cell counts are required.")
    pairs = tuple(sorted((int(x), int(y)) for x, y in zip(k, n)))
    if any(y <= 0 or x < 0 or x > y for x, y in pairs): raise ValueError("Counts must satisfy 0 <= successes <= songs.")
    return pairs


def moment_diagnostic(k, n):
    rates = np.asarray(k) / np.asarray(n)
    mu, variance = float(rates.mean()), float(rates.var(ddof=1)) if len(rates) > 1 else None
    concentration = mu * (1 - mu) / variance - 1 if variance and 0 < mu < 1 else None
    return dict(raw_rate_mean=mu, raw_rate_sample_variance=variance,
                naive_beta_moments_kappa=concentration,
                naive_beta_moments_valid=bool(concentration is not None and concentration > 0),
                warning="Fitting a Beta directly to raw c/n rates confuses binomial observation noise with latent cell variation.")


@lru_cache(maxsize=None)
def fit_eb(key):
    k = np.asarray([p[0] for p in key]); n = np.asarray([p[1] for p in key])
    z_bounds = (float(logit(MU_EPS)), float(logit(1 - MU_EPS)))
    q_bounds = tuple(math.log(v) for v in KAPPA_BOUNDS)
    def loss(x):
        mu, concentration = float(expit(x[0])), math.exp(float(x[1]))
        value = -float(np.sum(bb_logpmf(k, n, mu * concentration, (1 - mu) * concentration)))
        return value if math.isfinite(value) else 1e100
    start_mu = (k.sum() + .5) / (n.sum() + 1)
    attempts = []
    for mu in (start_mu, .25, .75):
        for concentration in (.05, 1.0, 10.0, 200.0):
            result = minimize(loss, [logit(np.clip(mu, MU_EPS, 1 - MU_EPS)), math.log(concentration)],
                              method="L-BFGS-B", bounds=(z_bounds, q_bounds),
                              options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8})
            if np.isfinite(result.fun): attempts.append(result)
    if not attempts: raise FloatingPointError("No finite empirical-Bayes fit.")
    best = min(attempts, key=lambda r: r.fun)
    # A deterministic profile guards against choosing one numerical local start.
    grid = np.unique(np.append(np.geomspace(*KAPPA_BOUNDS, 41), math.exp(best.x[1])))
    profile = []
    for concentration in grid:
        one = minimize_scalar(lambda z: loss([z, math.log(concentration)]), bounds=z_bounds,
                              method="bounded", options={"xatol": 1e-10})
        profile.append(dict(kappa=float(concentration), mu=float(expit(one.x)), log_likelihood=float(-one.fun)))
        if one.fun < best.fun - 1e-8:
            result = minimize(loss, [one.x, math.log(concentration)], method="L-BFGS-B",
                              bounds=(z_bounds, q_bounds), options={"maxiter": 500, "ftol": 1e-12})
            if np.isfinite(result.fun) and result.fun < best.fun: best = result
    mu, concentration = float(expit(best.x[0])), float(math.exp(best.x[1]))
    maximum = float(-best.fun)
    for row in profile: row["delta_log_likelihood"] = row["log_likelihood"] - maximum
    support = [r["kappa"] for r in profile if r["delta_log_likelihood"] >= -1.920729]
    flags = []
    if mu <= MU_EPS * 1.01: flags.append("mu_lower_bound")
    if mu >= 1 - MU_EPS * 1.01: flags.append("mu_upper_bound")
    if concentration <= KAPPA_BOUNDS[0] * 1.001: flags.append("kappa_lower_bound")
    if concentration >= KAPPA_BOUNDS[1] / 1.001: flags.append("kappa_upper_bound")
    if support and (min(support) <= KAPPA_BOUNDS[0] * 1.001 or max(support) >= KAPPA_BOUNDS[1] / 1.001):
        flags.append("profile_support_touches_bound")
    if support and max(support) / min(support) >= 100: flags.append("broad_kappa_profile")
    if np.all(k == 0): flags.append("all_zero_cells")
    if np.all(k == n): flags.append("all_full_cells")
    if len(key) < 10: flags.append("fewer_than_ten_cells")
    return dict(mu=mu, kappa=concentration, alpha=mu * concentration, beta=(1 - mu) * concentration,
                log_likelihood=maximum, converged=bool(best.success), starts=len(attempts), cells=len(key),
                boundary=any(flag in {"mu_lower_bound", "mu_upper_bound", "kappa_lower_bound", "kappa_upper_bound"} for flag in flags), flags=flags,
                identifiability="weak_or_boundary" if flags else "interior_fit_not_proof_of_identifiability",
                profile_support_min=min(support) if support else None, profile_support_max=max(support) if support else None,
                profile=profile, moments=moment_diagnostic(k, n))


@lru_cache(maxsize=None)
def full_grid(key, mu_nodes=64, kappa_nodes=81):
    """Proper hierarchical posterior by Gauss-Legendre quadrature in mu/log-kappa.

    mu ~ Beta(1,1), log kappa ~ Normal(log 4,1.5), truncated to the declared
    kappa bounds. Quadrature weights integrate the actual coordinate measure;
    a log-normal density in kappa would additionally require its Jacobian.
    """
    mu_base, mu_weight = leggauss(mu_nodes)
    mu, mu_weight = (mu_base + 1) / 2, mu_weight / 2
    z_base, z_weight = leggauss(kappa_nodes)
    low, high = (math.log(v) for v in KAPPA_BOUNDS)
    z = low + (z_base + 1) * (high - low) / 2
    z_weight = z_weight * (high - low) / 2
    concentration = np.exp(z)
    a, b = mu[:, None] * concentration[None, :], (1 - mu[:, None]) * concentration[None, :]
    truncation = norm.cdf((high - PRIOR_LOG_KAPPA_MEAN) / PRIOR_LOG_KAPPA_SD) - norm.cdf((low - PRIOR_LOG_KAPPA_MEAN) / PRIOR_LOG_KAPPA_SD)
    prior = np.log(mu_weight[:, None]) + np.log(z_weight[None, :]) + norm.logpdf(z[None, :], PRIOR_LOG_KAPPA_MEAN, PRIOR_LOG_KAPPA_SD) - math.log(truncation)
    logweights = prior.copy()
    for k, n in key: logweights = logweights + bb_logpmf(k, n, a, b)
    normalizer = float(logsumexp(logweights))
    weights = np.exp(logweights - normalizer)
    if not np.isfinite(weights).all() or not np.isclose(weights.sum(), 1): raise FloatingPointError("Invalid grid weights.")
    return dict(a=a.ravel(), b=b.ravel(), weights=weights.ravel(),
                mu=np.broadcast_to(mu[:, None], a.shape).ravel(),
                kappa=np.broadcast_to(concentration[None, :], a.shape).ravel(),
                log_evidence=normalizer, mu_nodes=mu_nodes, kappa_nodes=kappa_nodes)


def weighted_quantile(values, weights, q):
    order = np.argsort(values); v, w = values[order], weights[order]
    return float(v[np.searchsorted(np.cumsum(w), q, side="left")])


def grid_hyper_summary(grid):
    w = grid["weights"]
    return dict(mu_mean=float(np.sum(w * grid["mu"])), mu_low=weighted_quantile(grid["mu"], w, .025),
                mu_high=weighted_quantile(grid["mu"], w, .975),
                kappa_mean=float(np.sum(w * grid["kappa"])), kappa_low=weighted_quantile(grid["kappa"], w, .025),
                kappa_high=weighted_quantile(grid["kappa"], w, .975),
                hyper_prior="mu~Beta(1,1); log(kappa)~Normal(log(4),1.5), truncated kappa [.05,200]",
                log_evidence=grid["log_evidence"])


def beta_posterior(k, n, a, b):
    low, high = beta.ppf([.025, .975], a + k, b + n - k)
    mean = (a + k) / (a + b + n)
    if not all(math.isfinite(float(x)) for x in (low, high, mean)): raise FloatingPointError("Invalid Beta posterior.")
    return dict(posterior_mean=float(mean), credible_low=float(low), credible_high=float(high),
                conditional_interval=True, posterior_alpha=float(a + k), posterior_beta=float(b + n - k),
                numeric_endpoint=bool(low == 0 or high == 1))


def grid_posterior(k, n, grid):
    a, b, weights = grid["a"] + k, grid["b"] + n - k, grid["weights"]
    def cdf_z(z): return float(np.sum(weights * betainc(a, b, expit(z))))
    def quantile(q):
        if cdf_z(-700) >= q: return 0.0
        if cdf_z(36) <= q: return 1.0
        return float(expit(brentq(lambda z: cdf_z(z) - q, -700, 36, xtol=1e-9)))
    mean = float(np.sum(weights * a / (a + b)))
    low, high = quantile(.025), quantile(.975)
    if not (0 <= low <= high <= 1 and math.isfinite(mean)): raise FloatingPointError("Invalid mixture posterior.")
    return dict(posterior_mean=mean, credible_low=low, credible_high=high, conditional_interval=False,
                posterior_alpha=None, posterior_beta=None, numeric_endpoint=bool(low == 0 or high == 1))


def song_presence(metadata, assignments, topics):
    """Aggregate distinct songs, never independent blocks or fractional weights."""
    memberships = defaultdict(set)
    for row in assignments:
        if str(row["exact_repeat"]).lower() in {"true", "1"}: continue
        if row["song_id"] not in metadata: raise ValueError("Assignment has unknown song ID.")
        memberships[row["song_id"]].add(int(row["topic_id"]))
    rows, cells = [], defaultdict(list)
    for song_id, info in sorted(metadata.items()):
        cells[(info["decade"], info["language"])].append(song_id)
        for topic in topics:
            rows.append(dict(song_id=song_id, decade=info["decade"], language=info["language"],
                             topic_id=topic, present=topic in memberships[song_id]))
    counts = []
    for topic in topics:
        for (decade, language), ids in sorted(cells.items()):
            counts.append(dict(decade=decade, language=language, topic_id=topic, songs=len(ids),
                               successes=sum(topic in memberships[song_id] for song_id in ids),
                               raw_rate=sum(topic in memberships[song_id] for song_id in ids) / len(ids)))
    return rows, counts


def load_model(results, model, expected_song_ids=None):
    directory = results / slug(model)
    shares = read_csv(directory / "song_topic_shares_unique_first_occurrence.csv")
    metadata = {}
    for row in shares:
        info = dict(decade=int(row["decade"]), language=row["language"])
        if row["song_id"] in metadata and metadata[row["song_id"]] != info: raise ValueError("Inconsistent song metadata.")
        metadata[row["song_id"]] = info
    if expected_song_ids is not None and set(metadata) != expected_song_ids:
        missing = expected_song_ids - set(metadata); extra = set(metadata) - expected_song_ids
        raise ValueError(f"Model {model} cohort differs from manifest: {len(missing)} missing and {len(extra)} extra song IDs. Do not silently mix topic-fit cohorts.")
    topics = [int(r["topic_id"]) for r in json.loads((directory / "topics.json").read_text())["topics"]]
    presence, counts = song_presence(metadata, read_csv(directory / "section_assignments.csv"), topics)
    return presence, counts


def posterior_predictive(cells, mode, parameters, rng, replicates):
    k = np.asarray([r["successes"] for r in cells]); n = np.asarray([r["songs"] for r in cells])
    if mode == "hierarchical_grid":
        chosen = rng.choice(len(parameters["weights"]), size=replicates, p=parameters["weights"])
        a, b = parameters["a"][chosen, None], parameters["b"][chosen, None]
    else:
        a, b = parameters["alpha"], parameters["beta"]
    probability = rng.beta(a + k[None, :], b + n[None, :] - k[None, :], size=(replicates, len(cells)))
    simulated = rng.binomial(n[None, :], probability)
    hist = [dict(successes_count=c, observed_cells=int(np.sum(k == c)),
                 posterior_predictive_mean_cells=float(np.mean(np.sum(simulated == c, axis=1))),
                 predictive_low=float(np.quantile(np.sum(simulated == c, axis=1), .025)),
                 predictive_high=float(np.quantile(np.sum(simulated == c, axis=1), .975)))
            for c in range(int(n.max()) + 1)]
    stats = []
    for name, observed, rep in [
        ("zero_cell_fraction", float(np.mean(k == 0)), np.mean(simulated == 0, axis=1)),
        ("full_cell_fraction", float(np.mean(k == n)), np.mean(simulated == n, axis=1)),
        ("between_cell_rate_variance", float(np.var(k / n)), np.var(simulated / n[None, :], axis=1)),
    ]:
        tail = min(1., 2 * min((1 + np.sum(rep >= observed)) / (replicates + 1), (1 + np.sum(rep <= observed)) / (replicates + 1)))
        stats.append(dict(statistic=name, observed=observed, predictive_mean=float(rep.mean()),
                          predictive_low=float(np.quantile(rep, .025)), predictive_high=float(np.quantile(rep, .975)),
                          posterior_predictive_tail_probability=float(tail),
                          caution="Resubstitution diagnostic, not a calibrated p-value or held-out validation."))
    return hist, stats


def validate_design(cells, args):
    """Infer a balanced design from actual song denominators; never assume n=4."""
    if not cells or any(r["songs"] <= 0 or not 0 <= r["successes"] <= r["songs"] for r in cells):
        raise ValueError("Every cell needs positive song trials and valid integer successes.")
    if len({(r["decade"], r["language"]) for r in cells}) != len(cells):
        raise ValueError("Duplicate decade/language cells in a topic.")
    if args.allow_unbalanced: return
    if len(cells) != args.expected_cells or len({r["songs"] for r in cells}) != 1:
        raise ValueError(f"Expected {args.expected_cells} balanced decade/language cells; use --allow-unbalanced for a documented alternative.")
    if args.expected_songs_per_cell is not None and any(r["songs"] != args.expected_songs_per_cell for r in cells):
        raise ValueError(f"Expected {args.expected_songs_per_cell} songs per cell; actual denominators disagree.")


def select_panels(counts, models):
    """Fixed exploratory rule: maximum temporal variance WITHIN each language.

    This avoids treating a constant EN/ES split as historical evolution. The
    maximum is selected on these same observations; it is not a trend test.
    """
    panels = []
    for model in models:
        candidates = []
        topics = sorted({r["topic_id"] for r in counts if r["model"] == model and r["topic_id"] >= 0})
        for topic in topics:
            rows = [r for r in counts if r["model"] == model and r["topic_id"] == topic]
            groups = [[r["raw_rate"] for r in rows if r["language"] == lang] for lang in sorted({r["language"] for r in rows})]
            score = float(np.mean([np.var(g) for g in groups]))
            candidates.append((score, topic))
        if candidates:
            score, topic = sorted(candidates, key=lambda item: (-item[0], item[1]))[0]
            panels.append(dict(model=model, topic_id=topic, selection_score=score,
                               selection_rule="maximum_mean_within_language_between_decade_population_variance_of_observed_song_presence; exclude_noise; tie_smallest_topic_id"))
    return panels


def analyze_model(model, counts, args, diagnostic_topics=None):
    estimates, priors, profiles, predictive, predictive_stats, leaveout, grid_diagnostics = [], [], [], [], [], [], []
    topics = sorted({r["topic_id"] for r in counts})
    rng = np.random.default_rng(args.seed)
    for topic in topics:
        cells = [r for r in counts if r["topic_id"] == topic]
        validate_design(cells, args)
        # Fixed priors neither borrow across cells nor use an EB fitted center.
        for method, a, b in (("Jeffreys", .5, .5), ("uniform", 1., 1.)):
            for row in cells:
                estimates.append(dict(model=model, method=method, prior_context="fixed",
                                      **row, **beta_posterior(row["successes"], row["songs"], a, b),
                                      prior_alpha=a, prior_beta=b, prior_mu=.5, prior_kappa=a + b,
                                      hyperprior_uncertainty_included=False, boundary=False, identifiability="fixed_prior_not_fitted"))
        groups = [("pooled_languages", cells)] + [(f"language_{lang}", [r for r in cells if r["language"] == lang])
                  for lang in sorted({r["language"] for r in cells})]
        for context, group in groups:
            key = make_key([r["successes"] for r in group], [r["songs"] for r in group])
            fit = fit_eb(key); grid = full_grid(key, args.mu_nodes, args.kappa_nodes)
            hyper = grid_hyper_summary(grid)
            priors.append(dict(model=model, topic_id=topic, prior_context=context,
                               **{k: v for k, v in fit.items() if k not in {"profile", "moments", "flags"}},
                               flags=";".join(fit["flags"]), **fit["moments"],
                               **{f"full_{k}": v for k, v in hyper.items()}))
            profiles.extend(dict(model=model, topic_id=topic, prior_context=context, **p) for p in fit["profile"])
            # Cache repeated count-specific quantiles within the same fitted group.
            posterior_cache = {}
            for row in group:
                k, n = row["successes"], row["songs"]
                if (k, n) not in posterior_cache: posterior_cache[(k, n)] = grid_posterior(k, n, grid)
                for method, post in (("EB_plugin", beta_posterior(k, n, fit["alpha"], fit["beta"])),
                                     ("hierarchical_grid", posterior_cache[(k, n)])):
                    estimates.append(dict(model=model, method=method, prior_context=context, **row, **post,
                                          prior_alpha=fit["alpha"] if method == "EB_plugin" else None,
                                          prior_beta=fit["beta"] if method == "EB_plugin" else None,
                                          prior_mu=fit["mu"] if method == "EB_plugin" else .5,
                                          prior_kappa=fit["kappa"] if method == "EB_plugin" else None,
                                          hyperposterior_mu_mean=hyper["mu_mean"] if method == "hierarchical_grid" else None,
                                          hyperposterior_kappa_mean=hyper["kappa_mean"] if method == "hierarchical_grid" else None,
                                          hyperprior_uncertainty_included=method == "hierarchical_grid",
                                          hyperparameter_uncertainty_included=method == "hierarchical_grid",
                                          hyperprior_choice_uncertainty_included=False,
                                          boundary=fit["boundary"] if method == "EB_plugin" else None,
                                          eb_boundary_diagnostic=fit["boundary"], identifiability=fit["identifiability"],
                                          eb_fit_converged=fit["converged"],
                                          warning="Language exchangeability is an assumption; topic presence is not fear prevalence."))
            for method, params in (("EB_plugin", fit), ("hierarchical_grid", grid)):
                hist, checks = posterior_predictive(group, method, params, rng, args.predictive_replicates)
                predictive.extend(dict(model=model, topic_id=topic, prior_context=context, method=method, **r) for r in hist)
                predictive_stats.extend(dict(model=model, topic_id=topic, prior_context=context, method=method, **r) for r in checks)
            # Refit only the hyperprior without each cell. Topic discovery itself
            # still used all selected songs: this is not topic-level generalization.
            for i, row in enumerate(group):
                training = group[:i] + group[i + 1:]
                held_key = make_key([r["successes"] for r in training], [r["songs"] for r in training])
                held_fit = fit_eb(held_key); held_grid = full_grid(held_key, args.mu_nodes, args.kappa_nodes)
                k, n = row["successes"], row["songs"]
                post = beta_posterior(k, n, held_fit["alpha"], held_fit["beta"])
                integrated_logprob = float(logsumexp(np.log(held_grid["weights"]) + bb_logpmf(k, n, held_grid["a"], held_grid["b"])))
                leaveout.append(dict(model=model, prior_context=context, **row,
                                     fitted_cells=len(training), leave_cell_out_mu=held_fit["mu"],
                                     leave_cell_out_kappa=held_fit["kappa"], leave_cell_out_boundary=held_fit["boundary"],
                                     leave_cell_out_identifiability=held_fit["identifiability"],
                                     plugin_posterior_mean_after_observing_cell=post["posterior_mean"],
                                     plugin_mean_change=post["posterior_mean"] - beta_posterior(k, n, fit["alpha"], fit["beta"])["posterior_mean"],
                                     plugin_held_cell_log_predictive=float(bb_logpmf(k, n, held_fit["alpha"], held_fit["beta"])),
                                     hierarchical_held_cell_log_predictive=integrated_logprob,
                                     Jeffreys_held_cell_log_predictive=float(bb_logpmf(k, n, .5, .5)),
                                     uniform_held_cell_log_predictive=float(bb_logpmf(k, n, 1., 1.)),
                                     topic_fit_reuses_heldout_songs=True))
            # Deterministic resolution sensitivity for the requested panels.
            if topic in (diagnostic_topics or set()):
                finer = full_grid(key, 96, 121)
                for k, n in sorted(posterior_cache):
                    fine_post = grid_posterior(k, n, finer); main_post = posterior_cache[(k, n)]
                    grid_diagnostics.append(dict(model=model, topic_id=topic, prior_context=context, successes=k, songs=n,
                        main_mu_nodes=args.mu_nodes, main_kappa_nodes=args.kappa_nodes, comparison_mu_nodes=96, comparison_kappa_nodes=121,
                        mean_absolute_difference=abs(main_post["posterior_mean"] - fine_post["posterior_mean"]),
                        low_absolute_difference=abs(main_post["credible_low"] - fine_post["credible_low"]),
                        high_absolute_difference=abs(main_post["credible_high"] - fine_post["credible_high"])))
    return estimates, priors, profiles, predictive, predictive_stats, leaveout, grid_diagnostics


def model_caption(model):
    if "MiniLM" in model: return "MiniLM multilingüe"
    if "multilingual-e5" in model: return "E5 multilingüe"
    if "text-embedding-3-large" in model: return "OpenAI · 3-large"
    if "text-embedding-3-small" in model: return "OpenAI · 3-small"
    return model


def panel_terms(results, model, topic):
    data = json.loads((results / slug(model) / "topics.json").read_text(encoding="utf-8"))
    row = next(r for r in data["topics"] if int(r["topic_id"]) == topic)
    return [r["term"] for r in row.get("terms", [])[:3]]


def figure(estimates, priors, panels, output, private_cache, results):
    os.environ.setdefault("MPLCONFIGDIR", str(private_cache / "mpl-bayes"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    ncols = 2 if len(panels) > 1 else 1
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(19 if ncols == 2 else 11, 8 * nrows + 4), sharex=True, squeeze=False)
    colors = {"EB_plugin": "#d55e40", "hierarchical_grid": "#007f83", "Jeffreys": "#7168a6"}
    offsets = {"EB_plugin": -.22, "hierarchical_grid": 0, "Jeffreys": .22}
    for ax, panel in zip(axes.ravel(), panels):
        model, topic = panel["model"], panel["topic_id"]
        terms = panel_terms(results, model, topic)
        title = f"{model_caption(model)} · T{topic}\n" + " / ".join(term[:28] for term in terms)
        rows = [r for r in estimates if r["model"] == model and r["topic_id"] == topic and
                (r["prior_context"] == "pooled_languages" or r["method"] == "Jeffreys")]
        cells = sorted({(r["decade"], r["language"]) for r in rows})
        raw = {(r["decade"], r["language"]): r for r in rows}
        ymap = {cell: i for i, cell in enumerate(cells)}
        fit = next(r for r in priors if r["model"] == model and r["topic_id"] == topic and r["prior_context"] == "pooled_languages")
        for row in rows:
            method = row["method"]
            if method not in offsets: continue
            y = ymap[(row["decade"], row["language"])] + offsets[method]
            ax.hlines(y, row["credible_low"], row["credible_high"], colors=colors[method], lw=2,
                      linestyles="dashed" if method == "EB_plugin" and fit["boundary"] else "solid", alpha=.82)
            ax.scatter(row["posterior_mean"], y, s=27, color=colors[method], zorder=3)
        for cell, i in ymap.items(): ax.scatter(raw[cell]["raw_rate"], i + .36, s=22, color="#152638", marker="D", zorder=4)
        ax.set_yticks(range(len(cells)), [f"{d} · {lang.upper()}  ({raw[(d,lang)]['successes']}/{raw[(d,lang)]['songs']})" for d, lang in cells])
        ax.invert_yaxis(); ax.set_xlim(-.04, 1.04)
        ax.set_xticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
        ax.grid(axis="x", color="#e6e9ed", lw=.8); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        status = " · máximo en límite" if fit["boundary"] else " · perfil débil" if fit["identifiability"] == "weak_or_boundary" else ""
        ax.set_title(title + f"\nEB κ={fit['kappa']:.3g}" + status, loc="left", pad=15, fontweight="bold")
        ax.set_xlabel("Probabilidad modelada de presencia en una canción")
    for ax in axes.ravel()[len(panels):]: ax.set_visible(False)
    handles = [Line2D([0], [0], color="#152638", marker="D", linestyle="none", label="Frecuencia observada c/n")]
    handles += [Line2D([0], [0], color=colors[m], marker="o", lw=2, label=label) for m, label in
                (("EB_plugin", "EB: intervalo condicionado al hiperprior ajustado"),
                 ("hierarchical_grid", "Bayes jerárquico: integra incertidumbre de los hiperparámetros"),
                 ("Jeffreys", "Prior fijo Jeffreys Beta(½,½)"))]
    denominators = sorted({r["songs"] for r in estimates})
    trial_text = str(denominators[0]) if len(denominators) == 1 else f"{min(denominators)}–{max(denominators)}"
    title_y, subtitle_y, selection_y, axes_top = (.975, .93, .90, .82) if nrows == 1 else (.985, .957, .936, .875)
    fig.suptitle("Cuatro formas de leer la presencia de un tópico", fontsize=22, x=.05, ha="left", y=title_y)
    fig.text(.05, subtitle_y, f"{trial_text} canciones por década e idioma · c/n conocido · intervalos creíbles del 95% bajo distintos supuestos", fontsize=12)
    fig.text(.05, selection_y, "Un tópico por modelo: máxima variación temporal dentro de cada idioma. Selección exploratoria; IDs y palabras no equivalen entre modelos.", fontsize=11, color="#4b5661")
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.04, .055), ncol=2, frameon=False, fontsize=11)
    fig.text(.05, .025, "Presencia de tópicos ≠ prevalencia del miedo. Pooling EN/ES es una hipótesis; selección, artistas y asignación de tópicos quedan fuera de estos intervalos.", fontsize=10, color="#4b5661")
    fig.text(.05, .012, "EB a trazos: ajuste en un límite. 2020–2026 es una ventana incompleta. Las tasas de presencia pueden sumar más del 100%.", fontsize=10, color="#4b5661")
    fig.subplots_adjust(top=axes_top, bottom=.16 if nrows == 1 else .13, left=.10, right=.98, wspace=.40, hspace=.39)
    for suffix in ("svg", "png"): fig.savefig(output / f"beta_binomial_presence.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)


def temporal_contrasts(counts, estimates, args):
    """Earliest vs latest decade bin, conditional on every fixed topic.

    The hierarchical contrast draws ONE shared hyperparameter pair for both
    cells per replicate. Independently sampling their marginal mixtures would
    erase the posterior dependence induced by partial pooling.
    """
    contrasts = []
    models = list(dict.fromkeys(r["model"] for r in counts))
    for model in models:
        for topic in sorted({r["topic_id"] for r in counts if r["model"] == model and r["topic_id"] >= 0}):
            group = [r for r in counts if r["model"] == model and r["topic_id"] == topic]
            key = make_key([r["successes"] for r in group], [r["songs"] for r in group])
            grid = full_grid(key, args.mu_nodes, args.kappa_nodes)
            for language in sorted({r["language"] for r in group}):
                series = sorted((r for r in group if r["language"] == language), key=lambda r: r["decade"])
                if len(series) < 2: continue
                first, last = series[0], series[-1]
                for method in ("EB_plugin", "hierarchical_grid", "Jeffreys", "uniform"):
                    context = "fixed" if method in {"Jeffreys", "uniform"} else "pooled_languages"
                    rows = [r for r in estimates if r["model"] == model and r["topic_id"] == topic and
                            r["language"] == language and r["method"] == method and r["prior_context"] == context]
                    lo = next(r for r in rows if r["decade"] == first["decade"])
                    hi = next(r for r in rows if r["decade"] == last["decade"])
                    token = f"{model}|{topic}|{language}|{method}|{args.seed}".encode()
                    seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "big")
                    rng = np.random.default_rng(seed)
                    if method == "hierarchical_grid":
                        chosen = rng.choice(len(grid["weights"]), size=args.contrast_draws, p=grid["weights"])
                        a, b = grid["a"][chosen], grid["b"][chosen]
                        early = rng.beta(a + first["successes"], b + first["songs"] - first["successes"])
                        late = rng.beta(a + last["successes"], b + last["songs"] - last["successes"])
                    else:
                        early = rng.beta(lo["posterior_alpha"], lo["posterior_beta"], size=args.contrast_draws)
                        late = rng.beta(hi["posterior_alpha"], hi["posterior_beta"], size=args.contrast_draws)
                    delta = late - early
                    if not np.isfinite(delta).all(): raise FloatingPointError("Non-finite temporal contrast draws.")
                    probability = float(np.mean(delta > 0))
                    ci = np.quantile(delta, [.025, .975])
                    contrasts.append(dict(model=model, topic_id=topic, language=language, method=method,
                        prior_context=context, first_decade=first["decade"], last_decade=last["decade"],
                        first_successes=first["successes"], first_songs=first["songs"],
                        last_successes=last["successes"], last_songs=last["songs"],
                        raw_rate_difference=last["raw_rate"] - first["raw_rate"],
                        posterior_mean_difference=hi["posterior_mean"] - lo["posterior_mean"],
                        difference_credible_low=float(ci[0]), difference_credible_high=float(ci[1]),
                        posterior_probability_increase=probability,
                        probability_monte_carlo_se=float(math.sqrt(probability * (1 - probability) / args.contrast_draws)),
                        simulation_draws=args.contrast_draws, simulation_seed=seed,
                        shared_hyperparameter_draw=method == "hierarchical_grid",
                        interval_excludes_zero=bool(ci[0] > 0 or ci[1] < 0),
                        exploratory_multiple_comparisons_uncorrected=True,
                        warning="Latest bin is incomplete; topic assignments and purposive corpus selection are conditioned on. This is not a population trend or an emotion contrast."))
    return contrasts


def sensitivity_summary(counts, estimates, priors, panels, contrasts):
    """Reviewable extremes with model/topic/cell IDs, without naming them fear."""
    summaries = []
    for model in list(dict.fromkeys(r["model"] for r in counts)):
        records = [r for r in estimates if r["model"] == model and r["topic_id"] >= 0 and
                   r["prior_context"] in {"pooled_languages", "fixed"}]
        by_cell = defaultdict(dict)
        for row in records: by_cell[(row["topic_id"], row["decade"], row["language"])][row["method"]] = row
        def extreme(left, right):
            candidates = []
            for (topic, decade, language), rows in by_cell.items():
                first = rows[left]["posterior_mean"] if left != "raw" else rows[right]["raw_rate"]
                second = rows[right]["posterior_mean"]
                candidates.append(dict(topic_id=topic, decade=decade, language=language,
                    raw_rate=rows[right]["raw_rate"], songs=rows[right]["songs"], successes=rows[right]["successes"],
                    compared_values={left: first, right: second}, absolute_difference=abs(first - second)))
            return max(candidates, key=lambda r: r["absolute_difference"]) if candidates else None
        fits = [r for r in priors if r["model"] == model and r["topic_id"] >= 0 and r["prior_context"] == "pooled_languages"]
        own_panels = [r for r in panels if r["model"] == model]
        robust, selected_contrasts = [], []
        keys = sorted({(r["topic_id"], r["language"]) for r in contrasts if r["model"] == model})
        for topic, language in keys:
            rows = [r for r in contrasts if r["model"] == model and r["topic_id"] == topic and r["language"] == language]
            probabilities = [r["posterior_probability_increase"] for r in rows]
            if len(rows) == 4 and (min(probabilities) > .95 or max(probabilities) < .05):
                robust.append(dict(topic_id=topic, language=language, direction="increase" if min(probabilities) > .95 else "decrease",
                                   probability_range=[min(probabilities), max(probabilities)], raw_rate_difference=rows[0]["raw_rate_difference"],
                                   qualification="exploratory_consistency_across_declared_priors_only; no_multiple_comparison_correction"))
            if own_panels and topic == own_panels[0]["topic_id"]: selected_contrasts.extend(rows)
        summaries.append(dict(model=model, semantic_topics=len(fits), selected_panel=own_panels[0] if own_panels else None,
            pooled_eb_boundary_fits=sum(r["boundary"] for r in fits),
            pooled_eb_profiles_touching_bound=sum("profile_support_touches_bound" in r["flags"] for r in fits),
            maximum_absolute_eb_shrinkage=extreme("raw", "EB_plugin"),
            maximum_absolute_eb_vs_hierarchical_shift=extreme("EB_plugin", "hierarchical_grid"),
            maximum_absolute_eb_vs_Jeffreys_shift=extreme("EB_plugin", "Jeffreys"),
            mean_interval_width={method: float(np.mean([r["credible_high"] - r["credible_low"] for r in records if r["method"] == method])) if records else None
                                 for method in ("EB_plugin", "hierarchical_grid", "Jeffreys", "uniform")},
            endpoint_directions_consistent_across_declared_priors=robust,
            selected_panel_endpoint_contrasts=selected_contrasts,
            interpretation="Topic IDs and ontologies differ between encoders. Shrinkage and width are sensitivity diagnostics, not model-quality scores or historical fear prevalence."))
    return summaries


def self_test():
    from scipy.stats import betabinom
    for k in range(5): assert np.isclose(bb_logpmf(k, 4, .5, 1.2), betabinom.logpmf(k, 4, .5, 1.2), atol=1e-12)
    assert np.isclose(np.exp([bb_logpmf(k, 4, .5, .5) for k in range(5)]).sum(), 1)
    assert np.isclose(beta_posterior(4, 4, .5, .5)["posterior_mean"], .9)
    metadata = {"one": dict(decade=1980, language="en"), "two": dict(decade=1980, language="en")}
    assignments = [dict(song_id="one", topic_id=2, exact_repeat=False)] * 10 + [dict(song_id="one", topic_id=2, exact_repeat=True)]
    _, cells = song_presence(metadata, assignments, [2])
    assert cells[0]["songs"] == 2 and cells[0]["successes"] == 1
    degenerate = fit_eb(make_key([0] * 5, [4] * 5))
    assert degenerate["boundary"] and "all_zero_cells" in degenerate["flags"]
    key = make_key([0, 1, 3, 4], [4] * 4)
    grid = full_grid(key, 32, 41)
    assert np.isclose(grid["weights"].sum(), 1)
    assert np.isclose(grid_hyper_summary(grid)["mu_mean"], .5, atol=1e-10)
    left, right = grid_posterior(1, 4, grid), grid_posterior(3, 4, grid)
    assert np.isclose(left["posterior_mean"], 1 - right["posterior_mean"], atol=1e-10)
    assert np.isclose(left["credible_low"], 1 - right["credible_high"], atol=1e-8)
    args = parser().parse_args(["--mu-nodes", "16", "--kappa-nodes", "17", "--predictive-replicates", "100"])
    cells = [dict(decade=1980 + 10 * (i // 2), language="en" if i % 2 == 0 else "es", topic_id=99,
                  songs=4, successes=i % 5, raw_rate=(i % 5) / 4) for i in range(10)]
    result = analyze_model("synthetic_integration", cells, args)
    assert len(result[0]) == 60 and len(result[1]) == 3 and len(result[5]) == 20
    assert all(0 <= r["credible_low"] <= r["credible_high"] <= 1 for r in result[0])
    twelve = [dict(r, songs=12, successes=3 * r["successes"]) for r in cells]
    validate_design(twelve, args)
    try: validate_design([dict(twelve[0], songs=11)] + twelve[1:], args)
    except ValueError: pass
    else: raise AssertionError("Unequal song denominators must be rejected for the balanced design.")
    temporal = [dict(model="m", topic_id=0, decade=1980 + 10 * i, language=lang,
                     songs=12, successes=12 if lang == "es" else 0, raw_rate=1. if lang == "es" else 0.)
                for lang in ("en", "es") for i in range(5)]
    temporal += [dict(model="m", topic_id=1, decade=1980 + 10 * i, language=lang,
                      songs=12, successes=3 * i, raw_rate=i / 4) for lang in ("en", "es") for i in range(5)]
    assert select_panels(temporal, ["m"])[0]["topic_id"] == 1
    same = [dict(r, model="synthetic_integration") for r in cells]
    contrasts = temporal_contrasts(same, result[0], args)
    assert len(contrasts) == 8 and all(r["simulation_draws"] == args.contrast_draws for r in contrasts)
    assert all(r["shared_hyperparameter_draw"] == (r["method"] == "hierarchical_grid") for r in contrasts)
    assert all(-1 <= r["difference_credible_low"] <= r["difference_credible_high"] <= 1 for r in contrasts)
    print("Self-tests passed: count likelihood, song deduplication, dynamic n=12 design, rejection of imbalance, within-language panel selection, boundary flags, grid symmetry and shared-hyperparameter temporal contrasts.")


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=Path("research/fear-lyrics/results"))
    p.add_argument("--output", type=Path, help="Defaults to results/bayes; leaves primary outputs unchanged.")
    p.add_argument("--private-cache", type=Path, default=Path(".research-cache/fear-lyrics"))
    p.add_argument("--corpus-manifest", type=Path, help="Optional public song manifest; assert identical model cohorts instead of silently filtering different topic fits.")
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--mu-nodes", type=int, default=64)
    p.add_argument("--kappa-nodes", type=int, default=81)
    p.add_argument("--predictive-replicates", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20261012)
    p.add_argument("--expected-cells", type=int, default=10)
    p.add_argument("--expected-songs-per-cell", "--expected-cell-songs", dest="expected_songs_per_cell", type=int, default=None, help="Optional strict design check; default infers a balanced actual song count.")
    p.add_argument("--contrast-draws", type=int, default=20000, help="Seeded draws for conditional first-to-last-bin contrasts.")
    p.add_argument("--allow-unbalanced", action="store_true")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.self_test: self_test(); return 0
    if min(args.mu_nodes, args.kappa_nodes) < 16 or args.predictive_replicates < 100 or args.contrast_draws < 100:
        raise ValueError("Use >=16 quadrature nodes and >=100 predictive replicates and contrast draws.")
    start = time.monotonic(); output = args.output or args.results / "bayes"
    output.mkdir(parents=True, exist_ok=True)
    all_presence, all_counts = [], []
    bundles = [[] for _ in range(7)]; sources = {}
    expected_song_ids = None
    if args.corpus_manifest:
        manifest = json.loads(args.corpus_manifest.read_text(encoding="utf-8"))
        songs = manifest.get("songs", manifest) if isinstance(manifest, dict) else manifest
        expected_song_ids = {r["song_id"] for r in songs}
        if len(expected_song_ids) != len(songs): raise ValueError("Duplicate song IDs in corpus manifest.")
    loaded = {}
    first_cohort = None
    first_section_metadata_hash = None
    for model in args.models:
        presence, counts = load_model(args.results, model, expected_song_ids)
        cohort = {r["song_id"] for r in presence}
        if first_cohort is None: first_cohort = cohort
        elif cohort != first_cohort: raise ValueError("Compared models must use exactly the same song IDs.")
        loaded[model] = counts
        all_presence.extend(dict(model=model, **r) for r in presence)
        all_counts.extend(dict(model=model, **r) for r in counts)
        source = args.results / slug(model) / "section_assignments.csv"
        unit_fields = ("document_id", "song_id", "order", "canonical_id", "exact_repeat", "segmentation", "word_count", "line_count")
        units = sorted(({k: row[k] for k in unit_fields} for row in read_csv(source)), key=lambda row: row["document_id"])
        units_hash = hashlib.sha256(json.dumps(units, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        if first_section_metadata_hash is None: first_section_metadata_hash = units_hash
        elif units_hash != first_section_metadata_hash: raise ValueError("Compared encoders must use the same section IDs, canonical repeats, ordering and lengths.")
        sources[model] = dict(section_assignments_sha256=digest(source),
                              unique_song_shares_sha256=digest(args.results / slug(model) / "song_topic_shares_unique_first_occurrence.csv"),
                              songs=len(cohort), song_ids_sha256=hashlib.sha256("\n".join(sorted(cohort)).encode()).hexdigest(),
                              section_metadata_sha256=units_hash)
    panels = select_panels(all_counts, args.models)
    for panel in panels: panel["terms"] = panel_terms(args.results, panel["model"], panel["topic_id"])
    for model in args.models:
        print(f"Song-level Bayes sensitivity: {model}", flush=True)
        selected = {r["topic_id"] for r in panels if r["model"] == model}
        computed = analyze_model(model, loaded[model], args, diagnostic_topics=selected)
        for destination, rows in zip(bundles, computed): destination.extend(rows)
    estimates, priors, profiles, predictive, stats, leaveout, resolution = bundles
    contrasts = temporal_contrasts(all_counts, estimates, args)
    for name, rows in [("song_presence", all_presence), ("cell_counts", all_counts), ("posterior_rates", estimates),
                       ("hyperpriors", priors), ("kappa_profiles", profiles), ("posterior_predictive_counts", predictive),
                       ("posterior_predictive_diagnostics", stats), ("leave_one_cell_out", leaveout), ("grid_resolution", resolution),
                       ("temporal_endpoint_contrasts", contrasts), ("figure_selection", panels)]:
        write_csv(output / f"{name}.csv", rows)
    if not args.no_figure and panels: figure(estimates, priors, panels, output, args.private_cache, args.results)
    pooled = [r for r in priors if r["prior_context"] == "pooled_languages"]
    max_resolution = max((r[field] for r in resolution for field in ("mean_absolute_difference", "low_absolute_difference", "high_absolute_difference")), default=None)
    protocol = dict(
        executed=True, sources=sources, script_sha256=digest(__file__), runtime_seconds=round(time.monotonic() - start, 2),
        source_article=SOURCE_URL, source_code=SOURCE_CODE_URL,
        observation="A selected song contains >=1 unique block assigned to a fixed topic. One Bernoulli per song/topic.",
        estimand="Latent topic-presence propensity under a hypothetical exchangeable song-sampling model. Observed finite-corpus c/n is known exactly; purposive selection prevents external prevalence interpretation.",
        trials="Actual selected-song count per decade/language cell, inferred and validated before fitting; all zero cells retained. Noise -1 is diagnostic.",
        song_denominators=sorted({r["songs"] for r in all_counts}), expected_cells=args.expected_cells,
        expected_songs_per_cell=args.expected_songs_per_cell, models=args.models,
        corpus_manifest_sha256=digest(args.corpus_manifest) if args.corpus_manifest else None,
        cohort_check="Every encoder uses exactly the same song IDs, section IDs, order, canonical repeats and lengths; optional manifest match is strict. This public metadata hash does not itself verify private lyric text bytes. Different fit cohorts are not silently filtered.",
        likelihood="c_g ~ BetaBinomial(n_g, mu*kappa, (1-mu)*kappa); theta_g | c_g,hyper ~ Beta(alpha+c_g,beta+n_g-c_g)",
        eb="Marginal likelihood with 12 deterministic starts and a kappa profile; plug-in intervals condition on fitted hyperparameters.",
        eb_kappa_bounds=list(KAPPA_BOUNDS), eb_mu_bounds=[MU_EPS, 1 - MU_EPS],
        profile="delta log likelihood >= -1.920729 is a heuristic support diagnostic, not a calibrated interval in this tiny boundary-prone design.",
        full_hierarchical="mu~Beta(1,1); log(kappa)~Normal(log(4),1.5) truncated [.05,200]; Gauss-Legendre quadrature over mu/log(kappa).",
        grid=[args.mu_nodes, args.kappa_nodes], grid_resolution_comparison=[96, 121],
        fixed_priors={"Jeffreys": [.5, .5], "uniform": [1., 1.]},
        pooling="Compare all ten EN/ES cells with five cells per language. Neither exchangeability assumption is established by this corpus.",
        predictive_replicates=args.predictive_replicates, simulation_seed=args.seed,
        leave_one_cell_out="Refits only hyperprior; topic discovery still reuses all songs. Predictive scores are a conditional sensitivity, not topic generalization.",
        credible_intervals="95% equal-tail intervals. Grid mixture integrates hyperparameter (alpha/beta) uncertainty; its weak hyperprior specification stays fixed. All methods condition on topic assignments and corpus design.",
        full_vs_eb_difference="The hierarchical grid also introduces explicit weak hyperpriors; differences from EB are not solely an uncertainty-only correction at the same maximum-likelihood point.",
        unresolved="Selection bias, dependent artists, unequal provider coverage, song length/detection opportunity, topic/model uncertainty, and population inference are not solved.",
        presence_not_composition="A song can contain multiple topics; presence percentages may sum above 100% and do not equal primary fractional topic shares.",
        e5_warning="Language association is diagnosed separately for each embedding space; rates of those topics are not fear prevalence.",
        figure_selection="One non-noise topic per supplied model: maximum mean within-language between-decade variance of observed song presence; smallest ID breaks ties. Exploratory selection on this corpus, not an inferential trend test.",
        selected_panels=panels, temporal_contrast_draws=args.contrast_draws,
        temporal_contrasts="Earliest-to-latest decade bin for every non-noise topic and language. Shared hyperparameter draws preserve dependence of the hierarchical cells; all fixed assignments conditioned on; no multiplicity correction or complete-2020s claim.",
        article_adaptation="Keep shrinkage and explicit intervals; replace state population with actual song trials, retain zeros, fit count likelihood, inspect hyperparameter uncertainty.",
        article_critique="The linked Rmd excludes unmentioned states via filter(!is.na(n)); population is not a measured number of Bernoulli song trials.",
        packages={p: importlib.metadata.version(p) for p in ("numpy", "scipy", "matplotlib")},
    )
    write_json(output / "protocol.json", protocol)
    nonconverged = [{k: row[k] for k in ("model", "topic_id", "prior_context", "mu", "kappa", "converged", "flags")}
                    for row in priors if not row["converged"]]
    write_json(output / "optimization_diagnostics.json", dict(
        fits=len(priors), converged_fits=len(priors) - len(nonconverged), nonconverged_fits=nonconverged,
        interpretation="Finite flagged estimates are retained for sensitivity; an optimizer convergence flag does not establish identifiability or calibrated intervals."))
    write_json(output / "summary.json", dict(status="complete", cell_topic_rows=len(all_counts), posterior_rows=len(estimates),
        fitted_hyperpriors=len(priors), pooled_eb_boundary_fits=sum(r["boundary"] for r in pooled), pooled_eb_fits=len(pooled),
        eb_converged_fits=len(priors) - len(nonconverged), eb_nonconverged_fits=len(nonconverged),
        pooled_eb_profiles_touching_bound=sum("profile_support_touches_bound" in r["flags"] for r in pooled),
        maximum_selected_grid_resolution_difference=max_resolution,
        maximum_leave_cell_out_plugin_mean_shift=max(abs(r["plugin_mean_change"]) for r in leaveout),
        model_sensitivity=sensitivity_summary(all_counts, estimates, priors, panels, contrasts),
        selected_panels=panels, models=args.models, song_denominators=sorted({r["songs"] for r in all_counts}),
        mean_interval_width_by_method={method: float(np.mean([r["credible_high"] - r["credible_low"] for r in estimates if r["method"] == method and
            (r["prior_context"] == "pooled_languages" or r["prior_context"] == "fixed")])) for method in ("EB_plugin", "hierarchical_grid", "Jeffreys", "uniform")},
        interpretation="Conditional sensitivity of selected-song topic presence. No historical population change or validated fear prevalence is established."))
    print(f"Complete: {len(all_counts)} cell/topic observations, {len(estimates)} posterior estimates; primary results untouched.")
    return 0


if __name__ == "__main__": raise SystemExit(main())
