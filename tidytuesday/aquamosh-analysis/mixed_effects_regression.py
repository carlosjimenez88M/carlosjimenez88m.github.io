"""Regresión logística con efectos por track.

El chi² del análisis original es marginal: cuenta rupturas agregadas, ignora
que cada track tiene su propio nivel basal de coherencia. *Savage Sucker Boy*
tiene mucho switch Y probablemente discontinuidad temática genuina — eso
confunde el efecto del switch con el efecto del track.

Tres modelos:

  M1 — Marginal:     logit(broken) ~ switch
  M2 — Con controles: logit(broken) ~ switch + position_in_track
                                       + position_in_track² + lang_a + lang_b
  M3 — GEE clustered: M2 + cluster_by=track, cov_struct=exchangeable

GEE da population-averaged odds ratios con SE robustos a la correlación
intra-track. Si el OR del switch sobrevive M3, el efecto NO se explica
por idiosincrasia de track.

Variante:
  M4 — Mixed Linear (LPM): Probabilidad lineal con (1|track) random intercept,
       como sanity check (statsmodels no tiene GLMM completo para binomial
       sin pasar por R; este es un proxy razonable).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.regression.mixed_linear_model import MixedLM
from scipy.stats import chi2

plt.rcParams.update({
    "axes.facecolor": "#0d0d1a", "figure.facecolor": "#0d0d1a",
    "axes.edgecolor": "white", "axes.labelcolor": "white",
    "xtick.color": "white", "ytick.color": "white",
    "text.color": "white", "axes.titlecolor": "white",
    "savefig.facecolor": "#0d0d1a",
})

# === datos ===========================================================
lines = pd.read_parquet("outputs/exports/corpus_lines_v2.parquet")
lines = lines.sort_values(["track_num", "line_num"]).reset_index(drop=True)
emb_oa = np.load("data/embeddings/openai_lyrics_lines.npy")
emb_lb = np.load("data/embeddings/labse_lyrics_lines.npy")

def normalize(emb):
    nrm = np.linalg.norm(emb, axis=1, keepdims=True)
    out = np.where(nrm > 1e-10, emb / np.maximum(nrm, 1e-12), 0.0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

emb_oa_n = normalize(emb_oa)
emb_lb_n = normalize(emb_lb)
theta_oa = 0.3201
theta_lb = 0.3230

# === construir el dataset de pares ===================================
rows = []
for tnum, g in lines.groupby("track_num", sort=False):
    idx = g.index.values
    n_track = len(idx)
    for k in range(n_track - 1):
        i, j = idx[k], idx[k+1]
        if not (np.isfinite(emb_oa_n[i]).all() and np.isfinite(emb_oa_n[j]).all()
                and np.isfinite(emb_lb_n[i]).all() and np.isfinite(emb_lb_n[j]).all()):
            continue
        s_oa = float(emb_oa_n[i] @ emb_oa_n[j])
        s_lb = float(emb_lb_n[i] @ emb_lb_n[j])
        if not (np.isfinite(s_oa) and np.isfinite(s_lb)):
            continue
        rows.append({
            "track_num": int(tnum),
            "track_title": g["title"].iloc[k],
            "pair_index": k,
            "position": (k + 0.5) / (n_track - 1) if n_track > 1 else 0.5,
            "lang_a": g["lang_v2"].iloc[k],
            "lang_b": g["lang_v2"].iloc[k+1],
            "switch": int(g["lang_v2"].iloc[k] != g["lang_v2"].iloc[k+1]),
            "sim_openai": s_oa,
            "sim_labse": s_lb,
            "broken_openai": int(s_oa < theta_oa),
            "broken_labse": int(s_lb < theta_lb),
        })

df = pd.DataFrame(rows)
df["position_sq"] = df["position"] ** 2
df["track_num"] = df["track_num"].astype("category")
# Reduce idiomas a categorías presentes
df["lang_a"] = df["lang_a"].astype("category")
df["lang_b"] = df["lang_b"].astype("category")
print(f"n pares: {len(df)}")
print(f"tracks: {df['track_num'].nunique()}")
print(f"% switches: {df['switch'].mean()*100:.1f}%")
print(f"% broken OpenAI: {df['broken_openai'].mean()*100:.1f}%")
print(f"% broken LaBSE:  {df['broken_labse'].mean()*100:.1f}%")

# === ¿qué tracks contribuyen al efecto switch? =======================
print("\n=== Break rate por track y tipo de transición (OpenAI) ===")
agg = df.groupby(["track_title", "switch"])["broken_openai"].agg(
    ["mean", "count"]).unstack()
agg.columns = ["P_same", "P_switch", "n_same", "n_switch"]
agg["gap"] = agg["P_switch"] - agg["P_same"]
agg = agg.sort_values("gap", ascending=False)
print(agg.round(3).to_string())

# === modelos para OpenAI =============================================
def fit_models(df, target):
    """Fit M1 (marginal), M2 (con controles), M3 (GEE clustered),
    M4 (mixed LPM)."""
    res = {}

    # M1: marginal switch effect
    m1 = smf.logit(f"{target} ~ switch", data=df).fit(disp=False)
    res["M1_marginal"] = m1

    # M2: controles fijos
    m2 = smf.logit(f"{target} ~ switch + position + position_sq + C(lang_a) + C(lang_b)",
                    data=df).fit(disp=False)
    res["M2_with_controls"] = m2

    # M3: GEE con cluster por track
    m3 = smf.gee(f"{target} ~ switch + position + position_sq + C(lang_a) + C(lang_b)",
                   groups="track_num", data=df,
                   family=Binomial(), cov_struct=Exchangeable()).fit()
    res["M3_gee_clustered"] = m3

    # M4: Mixed LM con random intercept por track (LPM, no GLMM real)
    # Tratamos broken como continuo para tener BLUP del track.
    m4 = MixedLM.from_formula(
        f"{target} ~ switch + position + position_sq + C(lang_a) + C(lang_b)",
        groups="track_num", data=df).fit()
    res["M4_mixed_LPM"] = m4

    return res

print("\n" + "=" * 70)
print("MODELOS PARA OPENAI")
print("=" * 70)
res_oa = fit_models(df, "broken_openai")

print("\n" + "=" * 70)
print("MODELOS PARA LABSE")
print("=" * 70)
res_lb = fit_models(df, "broken_labse")

# === extraer odds ratios del switch ==================================
def extract_switch_effect(model, name):
    """Devuelve (OR, CI_low, CI_high, p, coef, SE)."""
    if "Mixed" in name:
        # LPM: coef es marginal probability change, no OR. Lo reportamos
        # separadamente.
        coef = model.params["switch"]
        se = model.bse["switch"]
        # CI
        ci = (coef - 1.96*se, coef + 1.96*se)
        # p-value
        z = coef / se
        p = 2 * (1 - 0.5 * (1 + np.tanh(abs(z) / np.sqrt(2))))
        return {"effect_type": "marginal_prob", "coef": coef,
                "se": se, "ci_low": ci[0], "ci_high": ci[1], "p": p}
    else:
        coef = model.params["switch"]
        se = model.bse["switch"]
        or_ = np.exp(coef)
        ci_low = np.exp(coef - 1.96*se)
        ci_high = np.exp(coef + 1.96*se)
        p = model.pvalues["switch"]
        return {"effect_type": "odds_ratio", "OR": or_,
                "ci_low": ci_low, "ci_high": ci_high, "p": p,
                "coef": coef, "se": se}

print("\n=== EFECTO DEL SWITCH POR MODELO Y MÉTODO ===")
records = []
for model_target, results in [("OpenAI", res_oa), ("LaBSE", res_lb)]:
    for spec_name, model in results.items():
        eff = extract_switch_effect(model, spec_name)
        if eff["effect_type"] == "odds_ratio":
            print(f"  {model_target:7s}  {spec_name:22s}  OR={eff['OR']:5.2f}  "
                  f"95% CI [{eff['ci_low']:.2f}, {eff['ci_high']:.2f}]  p={eff['p']:.3g}")
        else:
            print(f"  {model_target:7s}  {spec_name:22s}  ΔP={eff['coef']:+.3f}  "
                  f"95% CI [{eff['ci_low']:+.3f}, {eff['ci_high']:+.3f}]  p={eff['p']:.3g}")
        records.append({"target": model_target, "spec": spec_name, **eff})

df_effects = pd.DataFrame(records)

# === reporte detallado del modelo final ==============================
print("\n" + "=" * 70)
print("MODELO M3 (GEE clustered by track) — OpenAI, reporte completo")
print("=" * 70)
print(res_oa["M3_gee_clustered"].summary())

print("\n" + "=" * 70)
print("MODELO M3 (GEE clustered by track) — LaBSE, reporte completo")
print("=" * 70)
print(res_lb["M3_gee_clustered"].summary())

# === forest plot =====================================================
fig, ax = plt.subplots(figsize=(11, 6))
or_rows = df_effects[df_effects["effect_type"] == "odds_ratio"].copy()
or_rows["label"] = or_rows["target"] + " · " + or_rows["spec"].str.replace("_", " ")
or_rows = or_rows.sort_values(["target", "spec"]).reset_index(drop=True)

y = np.arange(len(or_rows))
colors = ["#ff6b6b" if t == "OpenAI" else "#4ea1d3" for t in or_rows["target"]]
ax.errorbar(or_rows["OR"], y,
             xerr=[or_rows["OR"] - or_rows["ci_low"],
                    or_rows["ci_high"] - or_rows["OR"]],
             fmt="o", color="white", ecolor="white", capsize=4,
             markersize=8, markerfacecolor="white", markeredgecolor="white")
for i, (_, r) in enumerate(or_rows.iterrows()):
    ax.scatter(r["OR"], i, s=160, color=colors[i], zorder=3,
                edgecolors="white", linewidths=1.5)
    ax.text(r["ci_high"] + 0.15, i,
             f"  OR={r['OR']:.2f}  p={r['p']:.2g}",
             color="white", va="center", fontsize=9)
ax.axvline(1.0, color="#888", linestyle="--", linewidth=1)
ax.set_yticks(y); ax.set_yticklabels(or_rows["label"])
ax.set_xlabel("Odds ratio (switch vs same-lang)")
ax.set_title("Forest plot · efecto del language-switch sobre la ruptura de ventana",
              color="white", fontsize=13)
ax.set_facecolor("#0d0d1a")
plt.tight_layout()
plt.savefig("outputs/figures/mixed_effects_forest.png", dpi=140, bbox_inches="tight")
plt.close()
print("\n[fig] outputs/figures/mixed_effects_forest.png")

# === random effects por track (de M4) ================================
print("\n=== Track-specific random effects (M4 LPM, OpenAI) ===")
re_oa = res_oa["M4_mixed_LPM"].random_effects
re_lb = res_lb["M4_mixed_LPM"].random_effects
# Convertir a DataFrame
track_titles = {int(t): df[df["track_num"]==t]["track_title"].iloc[0]
                  for t in df["track_num"].cat.categories}
re_records = []
for tnum_str in re_oa:
    tnum_int = int(tnum_str)
    re_records.append({
        "track": track_titles.get(tnum_int, str(tnum_int)),
        "re_openai": float(re_oa[tnum_str].iloc[0]),
        "re_labse": float(re_lb[tnum_str].iloc[0]),
    })
df_re = pd.DataFrame(re_records).sort_values("re_openai")
print(df_re.round(3).to_string(index=False))

# Forest de random effects
fig, ax = plt.subplots(figsize=(11, 5))
y = np.arange(len(df_re))
ax.barh(y - 0.18, df_re["re_openai"], 0.36, color="#ff6b6b",
         edgecolor="white", label="OpenAI")
ax.barh(y + 0.18, df_re["re_labse"], 0.36, color="#4ea1d3",
         edgecolor="white", label="LaBSE")
ax.axvline(0, color="#888", linestyle="--", linewidth=1)
ax.set_yticks(y); ax.set_yticklabels(df_re["track"])
ax.set_xlabel("Random intercept (desviación de P(rupture) sobre la media)")
ax.set_title("Variación baseline por track (M4 LPM)", color="white", fontsize=13)
ax.legend(facecolor="#0d0d1a", labelcolor="white", edgecolor="#444")
plt.tight_layout()
plt.savefig("outputs/figures/mixed_effects_random_per_track.png",
             dpi=140, bbox_inches="tight")
plt.close()
print("[fig] outputs/figures/mixed_effects_random_per_track.png")

# === guardar JSON ====================================================
def model_to_dict(model, name):
    eff = extract_switch_effect(model, name)
    return {
        "model_name": name,
        "n_obs": int(model.nobs) if hasattr(model, "nobs") else len(df),
        "switch_effect": eff,
        "summary_text": str(model.summary()),
    }

out = {
    "dataset": {
        "n_pairs": len(df), "n_tracks": int(df["track_num"].nunique()),
        "pct_switch": float(df["switch"].mean()),
        "pct_broken_openai": float(df["broken_openai"].mean()),
        "pct_broken_labse": float(df["broken_labse"].mean()),
    },
    "models": {
        "openai": {k: model_to_dict(v, k) for k, v in res_oa.items()},
        "labse":  {k: model_to_dict(v, k) for k, v in res_lb.items()},
    },
    "random_effects_per_track": df_re.to_dict("records"),
    "break_rate_by_track": agg.round(4).reset_index().to_dict("records"),
}
Path("outputs/exports/mixed_effects.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2, default=str),
    encoding="utf-8")
print("\n[json] outputs/exports/mixed_effects.json")
print("\n--- DONE ---")
