"""Permutation tests para attention windows.

Dos hipótesis nulas distintas:

  H0_A: el idioma y la ruptura son independientes.
    Permutación: barajar etiquetas de idioma DENTRO de track preservando el
    orden de las líneas. El contenido y la posición se mantienen idénticos;
    solo cambia qué etiquetas tocan a qué posiciones.

  H0_B: el orden de las líneas no estructura las rupturas.
    Permutación: barajar el ORDEN de líneas dentro de track preservando
    sus idiomas. La sintaxis temporal del track desaparece.

Si el gap observado (P(break|switch) − P(break|same)) cae fuera del 99 % de
la distribución nula bajo H0_A, rechazamos independencia idioma-ruptura.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "axes.facecolor": "#0d0d1a", "figure.facecolor": "#0d0d1a",
    "axes.edgecolor": "white", "axes.labelcolor": "white",
    "xtick.color": "white", "ytick.color": "white",
    "text.color": "white", "axes.titlecolor": "white",
    "savefig.facecolor": "#0d0d1a",
})

# === datos ============================================================
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

def calibrate_theta(emb_n, n_pairs=5000, seed=42):
    rng = np.random.default_rng(seed)
    N = len(emb_n)
    i = rng.integers(0, N, n_pairs); j = rng.integers(0, N, n_pairs)
    m = i != j
    sims = (emb_n[i[m]] * emb_n[j[m]]).sum(axis=1)
    sims = sims[np.isfinite(sims)]
    return float(np.median(sims) + np.std(sims))

theta_oa = calibrate_theta(emb_oa_n)
theta_lb = calibrate_theta(emb_lb_n)
print(f"θ OpenAI = {theta_oa:.4f}, θ LaBSE = {theta_lb:.4f}")

# === precomputar similaridades consecutivas ===========================
# Una matriz [track, consecutive_pair_idx] = sim — la usamos para todas
# las permutaciones sin recalcular embeddings.
def consecutive_sims(emb_n, lines_df):
    """Devuelve una lista de arrays, uno por track, con sims consecutivas."""
    out = []
    for tnum, g in lines_df.groupby("track_num", sort=False):
        idx = g.index.values
        sims = []
        for k in range(len(idx) - 1):
            i, j = idx[k], idx[k+1]
            if not (np.isfinite(emb_n[i]).all() and np.isfinite(emb_n[j]).all()):
                sims.append(np.nan); continue
            s = float(emb_n[i] @ emb_n[j])
            sims.append(s if np.isfinite(s) else np.nan)
        out.append({"track": tnum, "indices": idx,
                     "sims": np.array(sims, dtype=np.float64)})
    return out

sims_oa = consecutive_sims(emb_oa_n, lines)
sims_lb = consecutive_sims(emb_lb_n, lines)
n_pairs_total = sum(len(t["sims"]) for t in sims_oa)
print(f"Pares consecutivos totales: {n_pairs_total}")

# === observed gap =====================================================
def observed_gap(sims_list, lines_df, theta):
    """P(break|switch) − P(break|same) con las etiquetas reales."""
    same_b = same_t = sw_b = sw_t = 0
    for track in sims_list:
        idx = track["indices"]
        langs = lines_df.loc[idx, "lang_v2"].values
        for k, s in enumerate(track["sims"]):
            if np.isnan(s): continue
            broken = int(s < theta)
            if langs[k] == langs[k+1]:
                same_b += broken; same_t += 1
            else:
                sw_b += broken; sw_t += 1
    p_same = same_b / same_t
    p_switch = sw_b / sw_t
    return p_switch - p_same, p_same, p_switch

gap_oa, ps_oa, psw_oa = observed_gap(sims_oa, lines, theta_oa)
gap_lb, ps_lb, psw_lb = observed_gap(sims_lb, lines, theta_lb)
print(f"\nOpenAI:  P_same={ps_oa:.3f}, P_switch={psw_oa:.3f}, gap={gap_oa:+.3f}")
print(f"LaBSE :  P_same={ps_lb:.3f}, P_switch={psw_lb:.3f}, gap={gap_lb:+.3f}")

# === permutation H0_A: shuffle lang labels within track ===============
def permute_lang_within_track(lines_df, rng):
    """Devuelve una copia de lang_v2 con las etiquetas barajadas dentro de
    cada track, preservando la longitud."""
    new_langs = lines_df["lang_v2"].values.copy()
    for tnum, g in lines_df.groupby("track_num", sort=False):
        idx = g.index.values
        new_langs[idx] = rng.permutation(new_langs[idx])
    return new_langs

def gap_with_langs(sims_list, langs_full, theta):
    same_b = same_t = sw_b = sw_t = 0
    for track in sims_list:
        idx = track["indices"]
        langs = langs_full[idx]
        for k, s in enumerate(track["sims"]):
            if np.isnan(s): continue
            broken = int(s < theta)
            if langs[k] == langs[k+1]:
                same_b += broken; same_t += 1
            else:
                sw_b += broken; sw_t += 1
        if sw_t == 0:
            continue
    if same_t == 0 or sw_t == 0: return np.nan
    return sw_b/sw_t - same_b/same_t

print("\n=== Permutation test H0_A (shuffle lang labels within track) ===")
N_PERM = 10_000
rng = np.random.default_rng(42)
null_oa = []
null_lb = []
for _ in range(N_PERM):
    new_langs = permute_lang_within_track(lines, rng)
    g_oa = gap_with_langs(sims_oa, new_langs, theta_oa)
    g_lb = gap_with_langs(sims_lb, new_langs, theta_lb)
    if not np.isnan(g_oa): null_oa.append(g_oa)
    if not np.isnan(g_lb): null_lb.append(g_lb)
null_oa = np.array(null_oa); null_lb = np.array(null_lb)

def report(name, observed, null_dist):
    mu = null_dist.mean()
    sd = null_dist.std()
    z = (observed - mu) / sd if sd > 0 else float("inf")
    # p-value de dos colas
    p_two_sided = (np.abs(null_dist - mu) >= np.abs(observed - mu)).mean()
    p_one_sided = (null_dist >= observed).mean()
    print(f"  {name}: observed={observed:+.4f}, null μ={mu:+.4f}, σ={sd:.4f}, "
          f"z={z:+.2f}, p_two={p_two_sided:.4f}, p_one_sided_greater={p_one_sided:.4f}")
    return {"observed": float(observed), "null_mean": float(mu),
            "null_std": float(sd), "z_score": float(z),
            "p_two_sided": float(p_two_sided),
            "p_one_sided_greater": float(p_one_sided),
            "n_permutations": int(len(null_dist))}

res_H0A_oa = report("OpenAI H0_A", gap_oa, null_oa)
res_H0A_lb = report("LaBSE  H0_A", gap_lb, null_lb)

# === permutation H0_B: shuffle line order within track ================
print("\n=== Permutation test H0_B (shuffle line order within track) ===")
# Aquí permutamos el ORDEN de las líneas dentro de cada track. La estructura
# del switch sigue siendo la misma (porque las etiquetas se mueven JUNTO con
# las líneas), pero la sintaxis temporal del track se destruye. Las
# similaridades consecutivas cambian.

def gap_with_shuffled_order(emb_n, lines_df, theta, rng):
    """Re-calcula las sims consecutivas tras barajar el orden de líneas
    en cada track (preservando contenido y etiquetas — pero el orden
    altera qué par es consecutivo)."""
    same_b = same_t = sw_b = sw_t = 0
    for tnum, g in lines_df.groupby("track_num", sort=False):
        idx = g.index.values
        new_order = rng.permutation(idx)
        langs = lines_df.loc[new_order, "lang_v2"].values
        for k in range(len(new_order) - 1):
            i, j = new_order[k], new_order[k+1]
            if not (np.isfinite(emb_n[i]).all() and np.isfinite(emb_n[j]).all()):
                continue
            s = float(emb_n[i] @ emb_n[j])
            if not np.isfinite(s): continue
            broken = int(s < theta)
            if langs[k] == langs[k+1]:
                same_b += broken; same_t += 1
            else:
                sw_b += broken; sw_t += 1
    if same_t == 0 or sw_t == 0: return np.nan
    return sw_b/sw_t - same_b/same_t

null_oa_B = []
null_lb_B = []
rng = np.random.default_rng(123)
for it in range(N_PERM):
    g_oa = gap_with_shuffled_order(emb_oa_n, lines, theta_oa, rng)
    g_lb = gap_with_shuffled_order(emb_lb_n, lines, theta_lb, rng)
    if not np.isnan(g_oa): null_oa_B.append(g_oa)
    if not np.isnan(g_lb): null_lb_B.append(g_lb)
    if (it + 1) % 2000 == 0:
        print(f"  ...{it+1}/{N_PERM}")
null_oa_B = np.array(null_oa_B); null_lb_B = np.array(null_lb_B)

res_H0B_oa = report("OpenAI H0_B", gap_oa, null_oa_B)
res_H0B_lb = report("LaBSE  H0_B", gap_lb, null_lb_B)

# === visualización ====================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

def plot_null(ax, null_dist, observed, title):
    ax.hist(null_dist, bins=60, color="#4ea1d3", alpha=0.7,
             edgecolor="white", linewidth=0.4)
    ax.axvline(observed, color="#ff6b6b", linestyle="-", linewidth=2.5,
                label=f"observed = {observed:+.3f}")
    ax.axvline(null_dist.mean(), color="white", linestyle="--", alpha=0.6,
                label=f"null μ = {null_dist.mean():+.3f}")
    ax.set_title(title, color="white", fontsize=11)
    ax.set_xlabel("Δ P(break|switch) − P(break|same)")
    ax.set_ylabel("Frecuencia bajo H0")
    ax.legend(facecolor="#0d0d1a", labelcolor="white", edgecolor="#444",
               fontsize=9)

plot_null(axes[0, 0], null_oa, gap_oa,
           f"OpenAI · H0_A (shuffle lang labels)  ·  z={res_H0A_oa['z_score']:+.2f}")
plot_null(axes[0, 1], null_oa_B, gap_oa,
           f"OpenAI · H0_B (shuffle line order)   ·  z={res_H0B_oa['z_score']:+.2f}")
plot_null(axes[1, 0], null_lb, gap_lb,
           f"LaBSE · H0_A (shuffle lang labels)  ·  z={res_H0A_lb['z_score']:+.2f}")
plot_null(axes[1, 1], null_lb_B, gap_lb,
           f"LaBSE · H0_B (shuffle line order)   ·  z={res_H0B_lb['z_score']:+.2f}")

plt.suptitle("Distribución nula del language-switch effect  ·  10,000 permutaciones",
              color="white", fontsize=13, y=0.99)
plt.tight_layout()
plt.savefig("outputs/figures/permutation_null_distributions.png",
             dpi=140, bbox_inches="tight")
plt.close()
print("[fig] outputs/figures/permutation_null_distributions.png")

# === guardar JSON =====================================================
out = {
    "n_permutations": N_PERM,
    "n_consecutive_pairs": int(n_pairs_total),
    "thresholds": {"openai": theta_oa, "labse": theta_lb},
    "observed": {
        "openai": {"p_same": ps_oa, "p_switch": psw_oa, "gap": gap_oa},
        "labse":  {"p_same": ps_lb, "p_switch": psw_lb, "gap": gap_lb},
    },
    "H0_A_shuffle_lang_labels": {
        "openai": res_H0A_oa, "labse": res_H0A_lb,
    },
    "H0_B_shuffle_line_order": {
        "openai": res_H0B_oa, "labse": res_H0B_lb,
    },
}
Path("outputs/exports/permutation_tests.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("[json] outputs/exports/permutation_tests.json")
print("\n--- DONE ---")
