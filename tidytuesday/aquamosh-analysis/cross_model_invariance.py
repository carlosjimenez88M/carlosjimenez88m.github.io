"""Cross-model invariance: ¿el efecto language-switch es invariante a la arquitectura
del embedding? Probamos 5 modelos:

  1. OpenAI text-embedding-3-large  (3072d, decoder, mostly English-trained)
  2. Google LaBSE                    (768d, encoder, parallel-corpus)
  3. BAAI BGE-M3                     (1024d, encoder, multi-functional)
  4. Multilingual E5 large           (1024d, encoder, weakly supervised)
  5. Paraphrase-multilingual-MPNet   (768d, encoder, paraphrase-trained)

Para cada modelo:
  - calibrar θ con pares aleatorios
  - calcular ventanas de atención por línea
  - tasa de ruptura same-lang vs switch
Salida: matriz modelo × tipo de transición + figura.
"""
import os, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

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

# === corpus ===========================================================
lines = pd.read_parquet("outputs/exports/corpus_lines_v2.parquet")
lines = lines.sort_values(["track_num", "line_num"]).reset_index(drop=True)
texts = lines["line_text"].astype(str).tolist()
print(f"Corpus: {len(texts)} líneas, {lines['track_num'].nunique()} tracks")

# === modelos ==========================================================
MODELS = {
    "OpenAI 3-large": {"type": "openai", "dim": 3072,
                        "cache": "data/embeddings/openai_lyrics_lines.npy"},
    "LaBSE": {"type": "st", "name": "sentence-transformers/LaBSE",
              "dim": 768, "cache": "data/embeddings/labse_lyrics_lines.npy"},
    "BGE-M3": {"type": "st", "name": "BAAI/bge-m3",
                "dim": 1024, "cache": "data/embeddings/bgem3_lyrics_lines.npy"},
    "E5 multilingual large": {"type": "st", "name": "intfloat/multilingual-e5-large",
                                "dim": 1024,
                                "cache": "data/embeddings/e5ml_lyrics_lines.npy"},
    "MPNet multilingual": {"type": "st",
                            "name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                            "dim": 768,
                            "cache": "data/embeddings/mpnet_lyrics_lines.npy"},
}

def load_or_compute(model_name, info):
    cache = Path(info["cache"])
    if cache.exists():
        emb = np.load(cache)
        print(f"  [{model_name}] cargado de cache: {emb.shape}")
        return emb
    print(f"  [{model_name}] computando...")
    if info["type"] == "st":
        m = SentenceTransformer(info["name"])
        # E5 requiere prefijo "query: " o "passage: " para mejor performance
        if "e5" in info["name"]:
            inputs = [f"passage: {t}" for t in texts]
        else:
            inputs = texts
        emb = m.encode(inputs, batch_size=32, normalize_embeddings=True,
                        show_progress_bar=True, convert_to_numpy=True)
    else:
        # OpenAI ya está cacheado, no debería caer aquí
        raise RuntimeError(f"No cache for {model_name}")
    np.save(cache, emb)
    return emb

embeddings = {}
for name, info in MODELS.items():
    embeddings[name] = load_or_compute(name, info)

# === calibrar θ y calcular tasa de ruptura ============================
def normalize(emb):
    nrm = np.linalg.norm(emb, axis=1, keepdims=True)
    out = np.where(nrm > 1e-10, emb / np.maximum(nrm, 1e-12), 0.0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

def calibrate_theta(emb_n, n_pairs=5000, seed=42):
    rng = np.random.default_rng(seed)
    N = len(emb_n)
    i = rng.integers(0, N, n_pairs); j = rng.integers(0, N, n_pairs)
    m = i != j
    sims = (emb_n[i[m]] * emb_n[j[m]]).sum(axis=1)
    sims = sims[np.isfinite(sims)]
    return float(np.median(sims) + np.std(sims))

def transition_break_rate(emb_n, lines_df, theta):
    same_broken = same_total = 0
    switch_broken = switch_total = 0
    for tnum, g in lines_df.groupby("track_num", sort=False):
        idx = g.index.values
        langs = g["lang_v2"].values
        for k in range(len(idx) - 1):
            i, j = idx[k], idx[k+1]
            if not (np.isfinite(emb_n[i]).all() and np.isfinite(emb_n[j]).all()):
                continue
            s = float(emb_n[i] @ emb_n[j])
            if not np.isfinite(s):
                continue
            broken = int(s < theta)
            if langs[k] == langs[k+1]:
                same_broken += broken; same_total += 1
            else:
                switch_broken += broken; switch_total += 1
    p_same = same_broken / same_total if same_total else 0.0
    p_switch = switch_broken / switch_total if switch_total else 0.0
    return p_same, p_switch, same_total, switch_total

print("\n=== Tasa de ruptura por modelo y tipo de transición ===")
results = []
for name, emb in embeddings.items():
    emb_n = normalize(emb)
    theta = calibrate_theta(emb_n)
    p_same, p_switch, n_same, n_switch = transition_break_rate(emb_n, lines, theta)
    results.append({
        "model": name,
        "dim": emb.shape[1],
        "theta": theta,
        "p_break_same": p_same,
        "p_break_switch": p_switch,
        "gap": p_switch - p_same,
        "rel_gap": p_switch / p_same if p_same > 0 else float("nan"),
        "n_same": n_same,
        "n_switch": n_switch,
    })
    print(f"  {name:25s}  θ={theta:.3f}  same={p_same:.3f}  switch={p_switch:.3f}  "
          f"gap={p_switch-p_same:+.3f}  rel={p_switch/p_same:.2f}x")

df_res = pd.DataFrame(results)
df_res.to_csv("outputs/exports/cross_model_invariance.csv", index=False)

# === ¿es el "gap" invariante a la arquitectura? =======================
print(f"\nGap medio entre modelos: {df_res['gap'].mean():.3f} (σ={df_res['gap'].std():.3f})")
print(f"Coeficiente de variación del gap: {df_res['gap'].std()/df_res['gap'].mean():.3f}")
print(f"Rel-gap medio: {df_res['rel_gap'].mean():.2f}x")
print(f"Min/Max rel-gap entre modelos: {df_res['rel_gap'].min():.2f}x / {df_res['rel_gap'].max():.2f}x")

# Test: ¿todos los modelos tienen gap > 0?
print(f"\n¿Todos los modelos rompen más en switches?  {(df_res['gap'] > 0).all()}")
print(f"¿Todos lo hacen al menos 1.5x más?         {(df_res['rel_gap'] >= 1.5).all()}")

# === figura: matriz modelo × transición ===============================
fig, ax = plt.subplots(figsize=(11, 6))
n_models = len(df_res)
x = np.arange(n_models)
w = 0.38
bars1 = ax.bar(x - w/2, df_res["p_break_same"], w, label="same-lang transition",
                color="#4ea1d3", edgecolor="white", alpha=0.85)
bars2 = ax.bar(x + w/2, df_res["p_break_switch"], w, label="language-switch transition",
                color="#ff6b6b", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels([m for m in df_res["model"]], rotation=20, ha="right")
ax.set_ylabel("Probabilidad de ruptura de ventana")
ax.set_ylim(0, 1)
ax.set_title(f"Cross-model invariance: el efecto language-switch sobre 5 modelos\n"
              f"rel-gap medio = {df_res['rel_gap'].mean():.2f}× "
              f"(min {df_res['rel_gap'].min():.2f}, max {df_res['rel_gap'].max():.2f})",
              color="white", fontsize=12)
ax.legend(facecolor="#0d0d1a", labelcolor="white", edgecolor="#444")

for i, r in df_res.iterrows():
    ax.text(i - w/2, r["p_break_same"] + 0.02, f"{r['p_break_same']:.2f}",
             ha="center", color="white", fontsize=8)
    ax.text(i + w/2, r["p_break_switch"] + 0.02, f"{r['p_break_switch']:.2f}",
             ha="center", color="white", fontsize=8)

plt.tight_layout()
plt.savefig("outputs/figures/cross_model_invariance.png", dpi=140, bbox_inches="tight")
plt.close()
print("[fig] outputs/figures/cross_model_invariance.png")

# Tambien guardar el gap como gráfica
fig, ax = plt.subplots(figsize=(9, 5))
order = df_res.sort_values("gap", ascending=True)
ax.barh(order["model"], order["gap"], color="#ff9f43", edgecolor="white")
ax.set_xlabel("Δ probabilidad de ruptura (switch − same)")
ax.set_title("Magnitud del language-switch effect por modelo", color="white", fontsize=13)
for i, (_, r) in enumerate(order.iterrows()):
    ax.text(r["gap"] + 0.01, i, f"  +{r['gap']:.2f}  ({r['rel_gap']:.2f}×)",
             color="white", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("outputs/figures/cross_model_gap.png", dpi=140, bbox_inches="tight")
plt.close()
print("[fig] outputs/figures/cross_model_gap.png")

# JSON con todo
out = {
    "models": df_res.to_dict("records"),
    "summary": {
        "mean_gap": float(df_res["gap"].mean()),
        "std_gap": float(df_res["gap"].std()),
        "cv_gap": float(df_res["gap"].std() / df_res["gap"].mean()),
        "mean_rel_gap": float(df_res["rel_gap"].mean()),
        "all_positive_gap": bool((df_res["gap"] > 0).all()),
        "all_rel_gap_above_1_5": bool((df_res["rel_gap"] >= 1.5).all()),
    },
}
Path("outputs/exports/cross_model_invariance.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("[json] outputs/exports/cross_model_invariance.json")
print("\n--- DONE ---")
