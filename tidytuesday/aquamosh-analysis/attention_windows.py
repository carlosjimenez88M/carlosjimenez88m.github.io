"""
Attention Windows en Aquamosh — comparación OpenAI vs Google (LaBSE).

MARCO TEÓRICO
=============

Sea una canción una secuencia de líneas L = {l_1, ..., l_n} con embeddings
E = {e_1, ..., e_n} ∈ R^d producidos por algún modelo.

La **ventana de atención** en la posición i se define como el máximo número
k de líneas consecutivas posteriores que mantienen similaridad de coseno
≥ θ con la línea-ancla:

    W_i(θ) = max{ k : sim(e_i, e_{i+j}) ≥ θ ∀ j ∈ [1, k] }

W_i mide cuánto "persiste" un concepto antes de que el discurso cambie de
tema. Es la cantidad operacionalizable de "atención sostenida" que requiere
el oyente para mantener coherencia.

LA PREGUNTA CRÍTICA (heredada del post Beatles vs Floyd)
--------------------------------------------------------
Si dos modelos de embedding distintos producen ventanas de longitud
distinta sobre el mismo texto, ¿quién tiene razón? Ninguno y ambos. La
ventana mide la **persistencia distribucional**, no la persistencia
conceptual. En letras con mucha repetición lexical (estribillos), las
ventanas son largas por accidente. En letras con metáfora variada y tema
unificado, las ventanas son cortas por accidente opuesto.

LA NOVEDAD DE AQUAMOSH
----------------------
*Aquamosh* es cuadrilingüe: las transiciones de idioma (EN → ES, ES →
MIXED) son omnipresentes. Cada cambio de idioma es una discontinuidad
LÉXICA garantizada, aun cuando el sentido sea continuo. Esto convierte al
álbum en un *natural experiment* sobre el sesgo distribucional:

    HIPÓTESIS H1:  los modelos rompen ventanas de atención en cada
    transición de idioma, *no* porque el tema cambie, sino porque las
    palabras lo hacen.

    HIPÓTESIS H2:  un modelo optimizado para alineamiento cross-lingüe
    (LaBSE, entrenado con corpus paralelos) debería romper menos veces
    que un modelo entrenado en mayoría inglés-dominante (OpenAI
    text-embedding-3-large).

Esta es la prueba: si LaBSE muestra ventanas SIGNIFICATIVAMENTE más
largas que OpenAI en transiciones de idioma — y similares en transiciones
monolingües — entonces la métrica de OpenAI está midiendo más superficie
que sentido.

CALIBRACIÓN DEL THRESHOLD θ
---------------------------
Cada modelo tiene una distribución de similaridad propia. Comparar W_i con
el mismo θ entre modelos sería injusto. Calibramos por modelo:

    θ_modelo = mediana( sim(e_i, e_j) | i,j aleatorios ) + 1·SD

de modo que θ corresponde a "similaridad notablemente alta para este modelo".
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, ks_2samp, spearmanr
from dotenv import load_dotenv

# === setup ============================================================
for p in [Path.cwd(), *Path.cwd().parents][:5]:
    if (p / ".env").exists():
        load_dotenv(p / ".env"); break

plt.rcParams.update({
    "axes.facecolor": "#0d0d1a", "figure.facecolor": "#0d0d1a",
    "axes.edgecolor": "white", "axes.labelcolor": "white",
    "xtick.color": "white", "ytick.color": "white",
    "text.color": "white", "axes.titlecolor": "white",
    "savefig.facecolor": "#0d0d1a",
})

DATA = Path("data")
OUT = Path("outputs")

# === 1) cargar corpus =================================================
lines = pd.read_parquet(OUT / "exports/corpus_lines_v2.parquet")
print(f"Líneas en corpus: {len(lines)}")
print(f"Tracks: {lines['track_num'].nunique()}")

# Ordenar por track, line_num — orden de aparición en la canción
lines = lines.sort_values(["track_num", "line_num"]).reset_index(drop=True)
texts = lines["line_text"].astype(str).tolist()

# === 2) embeddings OpenAI (cacheados) =================================
emb_oa = np.load(DATA / "embeddings/openai_lyrics_lines.npy")
print(f"OpenAI embeddings: {emb_oa.shape}")
assert len(texts) == len(emb_oa), \
    f"Desajuste: {len(texts)} líneas vs {len(emb_oa)} embeddings — re-correr clasificación primero"

# Normalizar (filtrar filas con norma cero antes para evitar NaN)
norms_oa = np.linalg.norm(emb_oa, axis=1, keepdims=True)
emb_oa_n = np.where(norms_oa > 1e-10, emb_oa / np.maximum(norms_oa, 1e-12), 0.0)
n_zero_oa = (norms_oa.ravel() <= 1e-10).sum()
if n_zero_oa:
    print(f"⚠️  {n_zero_oa} embeddings OpenAI con norma ~0 (líneas vacías o fallidas)")

# === 3) embeddings Google LaBSE =======================================
labse_path = DATA / "embeddings/labse_lyrics_lines.npy"
if labse_path.exists():
    emb_lb = np.load(labse_path)
    print(f"LaBSE embeddings cargados: {emb_lb.shape}")
else:
    print("Generando embeddings LaBSE (primera vez, ~1-2 min)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/LaBSE")
    emb_lb = model.encode(texts, batch_size=64, show_progress_bar=True,
                            normalize_embeddings=True)
    np.save(labse_path, emb_lb)
    print(f"LaBSE embeddings guardados: {emb_lb.shape}")

# LaBSE ya viene normalizado si normalize_embeddings=True
norms_lb = np.linalg.norm(emb_lb, axis=1, keepdims=True)
emb_lb_n = np.where(norms_lb > 1e-10, emb_lb / np.maximum(norms_lb, 1e-12), 0.0)
n_zero_lb = (norms_lb.ravel() <= 1e-10).sum()
if n_zero_lb:
    print(f"⚠️  {n_zero_lb} embeddings LaBSE con norma ~0")
# Reemplazar cualquier NaN remanente
emb_oa_n = np.nan_to_num(emb_oa_n, nan=0.0, posinf=0.0, neginf=0.0)
emb_lb_n = np.nan_to_num(emb_lb_n, nan=0.0, posinf=0.0, neginf=0.0)

# === 4) calibrar threshold por modelo (mediana + 1 SD de pares aleatorios)
def calibrate(emb_n: np.ndarray, n_pairs: int = 5000, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    N = len(emb_n)
    i = rng.integers(0, N, size=n_pairs)
    j = rng.integers(0, N, size=n_pairs)
    mask = i != j
    sims = (emb_n[i[mask]] * emb_n[j[mask]]).sum(axis=1)
    return float(np.median(sims) + np.std(sims))

theta_oa = calibrate(emb_oa_n)
theta_lb = calibrate(emb_lb_n)
print(f"\nThresholds calibrados:")
print(f"  OpenAI θ = {theta_oa:.4f}  (mediana + 1·SD de pares aleatorios)")
print(f"  LaBSE  θ = {theta_lb:.4f}")

# === 5) calcular ventanas de atención =================================
def attention_windows(emb_n: np.ndarray, lines: pd.DataFrame, theta: float) -> pd.DataFrame:
    """Para cada línea i, cuenta cuántas líneas consecutivas posteriores en
    el MISMO track mantienen sim ≥ theta."""
    rows = []
    for tnum, g in lines.groupby("track_num", sort=False):
        idx = g.index.values
        for pos, i in enumerate(idx):
            k = 0
            for j in idx[pos+1:]:
                s = float(emb_n[i] @ emb_n[j])
                if s >= theta:
                    k += 1
                else:
                    break
            rows.append({
                "track_num": tnum,
                "title": g["title"].iloc[0],
                "line_num": int(g["line_num"].iloc[pos]),
                "language": g["lang_v2"].iloc[pos] if "lang_v2" in g.columns else g["language"].iloc[pos],
                "campo": g["campo"].iloc[pos] if "campo" in g.columns else None,
                "window": k,
                "global_idx": int(i),
            })
    return pd.DataFrame(rows)

w_oa = attention_windows(emb_oa_n, lines, theta_oa)
w_lb = attention_windows(emb_lb_n, lines, theta_lb)
print(f"\nVentanas OpenAI: μ={w_oa['window'].mean():.2f}, σ={w_oa['window'].std():.2f}, max={w_oa['window'].max()}")
print(f"Ventanas LaBSE:  μ={w_lb['window'].mean():.2f}, σ={w_lb['window'].std():.2f}, max={w_lb['window'].max()}")

# Empate por global_idx para comparación pareada
merged = w_oa.merge(w_lb[["global_idx", "window"]], on="global_idx", suffixes=("_oa", "_lb"))
merged["diff_lb_minus_oa"] = merged["window_lb"] - merged["window_oa"]

# === 6) prueba de hipótesis H1 ========================================
# Lineas con anchor MIXED vs monolingüe — ¿tienen ventanas más cortas?
def hipotesis_lang_effect(df, col):
    mono = df[df["language"].isin(["ES", "EN", "FR"])][col]
    mixed = df[df["language"] == "MIXED"][col]
    u, p = mannwhitneyu(mono, mixed, alternative="greater")
    return {
        "modelo": col.replace("window_", "").upper(),
        "mono_mean": float(mono.mean()),
        "mixed_mean": float(mixed.mean()),
        "diff": float(mono.mean() - mixed.mean()),
        "mann_whitney_u": float(u),
        "p_value": float(p),
    }

print("\n=== H1: monolingüe vs MIXED en ventanas de atención ===")
for col in ["window_oa", "window_lb"]:
    res = hipotesis_lang_effect(merged, col)
    print(f"  {res['modelo']}: mono μ={res['mono_mean']:.2f}, mixed μ={res['mixed_mean']:.2f}, "
          f"Δ={res['diff']:+.2f}, p={res['p_value']:.3g}")

# === 7) H2: discrepancia entre modelos según idioma del ancla =========
print("\n=== H2: ¿LaBSE produce ventanas más largas que OpenAI en MIXED? ===")
for lang in ["ES", "EN", "FR", "MIXED"]:
    sub = merged[merged["language"] == lang]
    if len(sub) < 5:
        continue
    delta_mean = sub["diff_lb_minus_oa"].mean()
    print(f"  {lang}: n={len(sub):3d}  Δ(LaBSE - OpenAI) = {delta_mean:+.2f}")

# Test global: ¿LaBSE - OpenAI difiere entre MIXED y monolingüe?
mono_diff = merged[merged["language"].isin(["ES","EN","FR"])]["diff_lb_minus_oa"]
mixed_diff = merged[merged["language"] == "MIXED"]["diff_lb_minus_oa"]
ks_stat, ks_p = ks_2samp(mono_diff, mixed_diff)
print(f"\n  KS-test sobre distribución de (LaBSE - OpenAI):")
print(f"    monolingüe vs MIXED:  KS={ks_stat:.3f}, p={ks_p:.3g}")

# === 8) correlación general entre ventanas ============================
rho, rho_p = spearmanr(merged["window_oa"], merged["window_lb"])
print(f"\nSpearman OpenAI vs LaBSE (ranking línea-por-línea): ρ={rho:.3f} (p={rho_p:.2e})")

# === 9) VISUALIZACIONES ===============================================

# 9.1 — Distribución global de ventanas, ambos modelos
fig, ax = plt.subplots(figsize=(10, 5))
bins = np.arange(0, max(merged["window_oa"].max(), merged["window_lb"].max()) + 2)
ax.hist(merged["window_oa"], bins=bins, alpha=0.7, label="OpenAI 3-large (3072d)",
         color="#ff6b6b", edgecolor="white", linewidth=0.5)
ax.hist(merged["window_lb"], bins=bins, alpha=0.7, label="Google LaBSE (768d)",
         color="#4ea1d3", edgecolor="white", linewidth=0.5)
ax.set_xlabel("Longitud de ventana de atención (líneas)")
ax.set_ylabel("Frecuencia")
ax.set_title("Distribución de ventanas de atención por modelo", color="white", fontsize=13)
ax.legend(facecolor="#0d0d1a", labelcolor="white", edgecolor="#444")
plt.tight_layout()
plt.savefig(OUT / "figures/aw_distribution.png", dpi=140, bbox_inches="tight")
plt.close()
print("\n[fig] aw_distribution.png")

# 9.2 — Ventanas medias por idioma del ancla — comparado por modelo
fig, ax = plt.subplots(figsize=(10, 5))
langs = ["ES", "EN", "FR", "MIXED", "OTHER"]
means_oa, means_lb, errs_oa, errs_lb = [], [], [], []
for l in langs:
    s_oa = merged.loc[merged["language"] == l, "window_oa"]
    s_lb = merged.loc[merged["language"] == l, "window_lb"]
    means_oa.append(s_oa.mean()); means_lb.append(s_lb.mean())
    errs_oa.append(s_oa.std() / max(np.sqrt(len(s_oa)), 1))
    errs_lb.append(s_lb.std() / max(np.sqrt(len(s_lb)), 1))

x = np.arange(len(langs))
w = 0.4
ax.bar(x - w/2, means_oa, w, yerr=errs_oa, color="#ff6b6b", label="OpenAI 3-large",
        edgecolor="white", capsize=4)
ax.bar(x + w/2, means_lb, w, yerr=errs_lb, color="#4ea1d3", label="Google LaBSE",
        edgecolor="white", capsize=4)
ax.set_xticks(x); ax.set_xticklabels(langs)
ax.set_xlabel("Idioma de la línea-ancla")
ax.set_ylabel("Longitud media de ventana (± SE)")
ax.set_title("¿Las transiciones lingüísticas rompen ventanas? — OpenAI vs LaBSE",
              color="white", fontsize=13)
ax.legend(facecolor="#0d0d1a", labelcolor="white", edgecolor="#444")
plt.tight_layout()
plt.savefig(OUT / "figures/aw_by_language.png", dpi=140, bbox_inches="tight")
plt.close()
print("[fig] aw_by_language.png")

# 9.3 — Per-track means: side-by-side
fig, ax = plt.subplots(figsize=(11, 6))
by_track = (merged.groupby("title")
              .agg(window_oa=("window_oa", "mean"),
                    window_lb=("window_lb", "mean"))
              .sort_values("window_oa", ascending=True))
y = np.arange(len(by_track))
ax.barh(y - 0.2, by_track["window_oa"], 0.4, color="#ff6b6b",
         edgecolor="white", label="OpenAI")
ax.barh(y + 0.2, by_track["window_lb"], 0.4, color="#4ea1d3",
         edgecolor="white", label="LaBSE")
ax.set_yticks(y); ax.set_yticklabels(by_track.index)
ax.set_xlabel("Longitud media de ventana (líneas)")
ax.set_title("Ventanas de atención por track  ·  OpenAI vs LaBSE",
              color="white", fontsize=13)
ax.legend(facecolor="#0d0d1a", labelcolor="white", edgecolor="#444")
plt.tight_layout()
plt.savefig(OUT / "figures/aw_per_track.png", dpi=140, bbox_inches="tight")
plt.close()
print("[fig] aw_per_track.png")

# 9.4 — Scatter cross-model: ¿coinciden las ventanas línea por línea?
fig, ax = plt.subplots(figsize=(8, 7))
# Jitter para evitar solapamiento total
rng = np.random.default_rng(42)
jx = merged["window_oa"] + rng.normal(0, 0.15, len(merged))
jy = merged["window_lb"] + rng.normal(0, 0.15, len(merged))
color_map = {"ES": "#4ea1d3", "EN": "#ff6b6b", "FR": "#6bcb77",
              "JA": "#ff9f43", "MIXED": "#a0a0a0", "OTHER": "#666"}
for lang, c in color_map.items():
    mask = merged["language"] == lang
    if mask.sum() == 0:
        continue
    ax.scatter(jx[mask], jy[mask], c=c, label=lang, alpha=0.6, s=24,
                edgecolors="white", linewidths=0.3)
# Línea de identidad
lim = max(merged["window_oa"].max(), merged["window_lb"].max())
ax.plot([0, lim+1], [0, lim+1], "--", color="#777", lw=1, label="y=x")
ax.set_xlabel("Ventana OpenAI"); ax.set_ylabel("Ventana LaBSE")
ax.set_title(f"Acuerdo línea-por-línea  ·  Spearman ρ = {rho:.2f}",
              color="white", fontsize=13)
ax.legend(facecolor="#0d0d1a", labelcolor="white", edgecolor="#444", loc="upper left")
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig(OUT / "figures/aw_cross_model_scatter.png", dpi=140, bbox_inches="tight")
plt.close()
print("[fig] aw_cross_model_scatter.png")

# 9.5 — Distribución de la discrepancia (LaBSE - OpenAI) por idioma
fig, ax = plt.subplots(figsize=(10, 5))
for lang, c in color_map.items():
    sub = merged[merged["language"] == lang]
    if len(sub) < 5:
        continue
    sns.kdeplot(sub["diff_lb_minus_oa"], ax=ax, label=lang, color=c, linewidth=2,
                 bw_adjust=0.7, fill=False)
ax.axvline(0, color="#888", linestyle="--")
ax.set_xlabel("Δ ventana  (LaBSE − OpenAI)")
ax.set_ylabel("Densidad")
ax.set_title("¿Para qué idiomas LaBSE alarga vs acorta las ventanas?",
              color="white", fontsize=13)
ax.legend(facecolor="#0d0d1a", labelcolor="white", edgecolor="#444")
plt.tight_layout()
plt.savefig(OUT / "figures/aw_discrepancy_by_language.png", dpi=140, bbox_inches="tight")
plt.close()
print("[fig] aw_discrepancy_by_language.png")

# 9.6 — Per-track timeline: dónde se rompen las ventanas
fig, axes = plt.subplots(5, 2, figsize=(15, 14), sharex=False)
axes = axes.ravel()
for ax_i, (title, sub) in enumerate(merged.groupby("title", sort=False)):
    ax = axes[ax_i]
    sub = sub.sort_values("line_num")
    x = np.arange(len(sub))
    ax.plot(x, sub["window_oa"].values, label="OpenAI", color="#ff6b6b", linewidth=1.2)
    ax.plot(x, sub["window_lb"].values, label="LaBSE", color="#4ea1d3", linewidth=1.2)
    # Marcar transiciones de idioma
    langs = sub["language"].values
    for i in range(1, len(langs)):
        if langs[i] != langs[i-1]:
            ax.axvline(i - 0.5, color="#666", alpha=0.4, linewidth=0.5)
    ax.set_title(title[:28], color="white", fontsize=9)
    ax.set_xlim(0, len(sub))
    ax.set_facecolor("#0d0d1a")
    ax.tick_params(labelsize=7)
    if ax_i == 0:
        ax.legend(fontsize=7, facecolor="#0d0d1a", labelcolor="white",
                   edgecolor="#444")
plt.suptitle("Cronología de ventanas de atención por track  ·  líneas grises = cambio de idioma",
              color="white", fontsize=12, y=0.995)
plt.tight_layout()
plt.savefig(OUT / "figures/aw_timelines.png", dpi=130, bbox_inches="tight")
plt.close()
print("[fig] aw_timelines.png")

# 9.7 — Effect size: cuánto cambia la ventana ante cambio de idioma vs continuidad
def transition_effect(emb_n, lines_df, theta):
    """Para cada línea i, mide sim(i, i+1) y registra si hubo cambio de idioma."""
    rows = []
    for tnum, g in lines_df.groupby("track_num", sort=False):
        idx = g.index.values
        for k in range(len(idx) - 1):
            i, j = idx[k], idx[k+1]
            s = float(emb_n[i] @ emb_n[j])
            lang_i = g["lang_v2"].iloc[k]
            lang_j = g["lang_v2"].iloc[k+1]
            transition_type = "same" if lang_i == lang_j else "switch"
            rows.append({"sim": s, "transition": transition_type,
                          "broken": int(s < theta)})
    return pd.DataFrame(rows)

trans_oa = transition_effect(emb_oa_n, lines, theta_oa)
trans_lb = transition_effect(emb_lb_n, lines, theta_lb)

print("\n=== Tasa de ruptura de ventana en transiciones ===")
print("                        same-lang   switch-lang   diff")
for name, df in [("OpenAI", trans_oa), ("LaBSE", trans_lb)]:
    same = df[df["transition"] == "same"]["broken"].mean()
    switch = df[df["transition"] == "switch"]["broken"].mean()
    print(f"  {name:6s}                {same:.3f}        {switch:.3f}     {switch-same:+.3f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (name, df, c) in zip(axes, [("OpenAI", trans_oa, "#ff6b6b"),
                                       ("LaBSE",  trans_lb, "#4ea1d3")]):
    same = df[df["transition"] == "same"]["broken"].mean()
    switch = df[df["transition"] == "switch"]["broken"].mean()
    bars = ax.bar(["same lang", "lang switch"], [same, switch],
                    color=c, edgecolor="white")
    bars[0].set_alpha(0.55)
    bars[1].set_alpha(1.0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probabilidad de ruptura de ventana")
    ax.set_title(name, color="white")
    ax.set_facecolor("#0d0d1a")
    for j, v in enumerate([same, switch]):
        ax.text(j, v + 0.02, f"{v:.2f}", ha="center", color="white", fontsize=11)
plt.suptitle("¿Una transición de idioma rompe la ventana con más frecuencia?",
              color="white", fontsize=13)
plt.tight_layout()
plt.savefig(OUT / "figures/aw_break_rates.png", dpi=140, bbox_inches="tight")
plt.close()
print("[fig] aw_break_rates.png")

# === 10) guardar resultados ===========================================
findings = {
    "theta": {"openai": theta_oa, "labse": theta_lb},
    "global_stats": {
        "openai": {"mean": float(w_oa["window"].mean()),
                    "std": float(w_oa["window"].std()),
                    "max": int(w_oa["window"].max())},
        "labse":  {"mean": float(w_lb["window"].mean()),
                    "std": float(w_lb["window"].std()),
                    "max": int(w_lb["window"].max())},
    },
    "cross_model_spearman": float(rho),
    "h1_lang_effect": [hipotesis_lang_effect(merged, c) for c in ["window_oa", "window_lb"]],
    "h2_discrepancy_by_language": {
        lang: {
            "n": int((merged["language"] == lang).sum()),
            "mean_diff_labse_minus_openai": float(merged.loc[merged["language"] == lang, "diff_lb_minus_oa"].mean())
        }
        for lang in ["ES", "EN", "FR", "MIXED", "OTHER"]
        if (merged["language"] == lang).sum() > 0
    },
    "break_rate_by_transition": {
        "openai": {"same_lang": float(trans_oa[trans_oa["transition"]=="same"]["broken"].mean()),
                    "switch_lang": float(trans_oa[trans_oa["transition"]=="switch"]["broken"].mean())},
        "labse":  {"same_lang": float(trans_lb[trans_lb["transition"]=="same"]["broken"].mean()),
                    "switch_lang": float(trans_lb[trans_lb["transition"]=="switch"]["broken"].mean())},
    },
    "per_track_means": (merged.groupby("title")
                         .agg({"window_oa": "mean", "window_lb": "mean"})
                         .round(3).reset_index().to_dict("records")),
}
Path("outputs/exports/attention_windows.json").write_text(
    json.dumps(findings, ensure_ascii=False, indent=2),
    encoding="utf-8")
print("\n[json] outputs/exports/attention_windows.json")

# Tabla full para uso posterior
merged.to_parquet("outputs/exports/attention_windows_per_line.parquet", index=False)
print("[parquet] outputs/exports/attention_windows_per_line.parquet")

print("\n--- DONE ---")
