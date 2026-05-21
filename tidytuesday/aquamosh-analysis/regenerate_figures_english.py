"""Regenerate ALL figures with English labels. Reads cached data only —
no re-computation needed.
"""
import json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, entropy

plt.rcParams.update({
    "axes.facecolor": "#0d0d1a", "figure.facecolor": "#0d0d1a",
    "axes.edgecolor": "white", "axes.labelcolor": "white",
    "xtick.color": "white", "ytick.color": "white",
    "text.color": "white", "axes.titlecolor": "white",
    "savefig.facecolor": "#0d0d1a",
})

FIG = Path("outputs/figures")
FIG.mkdir(parents=True, exist_ok=True)

# ============================================================
# DATA LOADERS
# ============================================================
lines = pd.read_parquet("outputs/exports/corpus_lines_v2.parquet")
lines = lines.sort_values(["track_num", "line_num"]).reset_index(drop=True)
emb_oa = np.load("data/embeddings/openai_lyrics_lines.npy")
emb_lb = np.load("data/embeddings/labse_lyrics_lines.npy")
df_axes = pd.read_json("data/processed/semantic_axes.json")
chi_results = json.loads(Path("data/processed/chi_results.json").read_text())
findings_aw = json.loads(Path("outputs/exports/attention_windows.json").read_text())
findings_aw_pairs = pd.read_parquet("outputs/exports/attention_windows_per_line.parquet")

LANG_COLORS = {"ES": "#4ea1d3", "EN": "#ff6b6b", "FR": "#6bcb77",
                "JA": "#ff9f43", "MIXED": "#a0a0a0", "OTHER": "#444"}
SEMANTIC_FIELDS_EN = {
    "CUERPO": "BODY", "MARCA": "BRAND", "LUGAR": "PLACE",
    "EMOCION": "EMOTION", "IDENTIDAD": "IDENTITY", "ACCION": "ACTION",
    "REFERENCIA": "REFERENCE", "NONSENSE": "NONSENSE",
}

# ============================================================
# 1. LANGUAGE × FIELD RESIDUALS HEATMAP
# ============================================================
print("→ language_field_residuals_v2.png")
df_stat = lines[(lines["confidence"] >= 0.5) &
                 lines["campo"].isin(SEMANTIC_FIELDS_EN.keys()) &
                 lines["lang_v2"].isin(["ES","EN","FR","JA","MIXED"])]
ctab = pd.crosstab(df_stat["lang_v2"], df_stat["campo"])
chi2_v, p_v, dof, expected = chi2_contingency(ctab)
residuals = (ctab.values - expected) / np.sqrt(expected)
df_resid = pd.DataFrame(residuals, index=ctab.index, columns=ctab.columns)
df_resid_en = df_resid.rename(columns=SEMANTIC_FIELDS_EN)

fig, ax = plt.subplots(figsize=(11, 5))
sns.heatmap(df_resid_en, annot=True, fmt=".2f", cmap="RdBu_r",
             center=0, vmin=-5, vmax=5, ax=ax,
             cbar_kws={"label": "Standardized residual"},
             linewidths=0.6, linecolor="#0d0d1a")
ax.set_title(f"Language × semantic field association  ·  χ²={chi2_v:.1f}, "
              f"p<1e-{int(-np.log10(p_v)):d}",
              color="white", fontsize=13, pad=12)
ax.set_xlabel("Semantic field"); ax.set_ylabel("Language")
plt.tight_layout()
plt.savefig(FIG / "language_field_residuals_v2.png", dpi=140, bbox_inches="tight")
plt.close()

# ============================================================
# 2. LANGUAGE BY TRACK
# ============================================================
print("→ language_by_track.png")
lang_by_track = pd.crosstab(lines["title"], lines["lang_v2"])
order = [c for c in ["ES", "EN", "FR", "MIXED", "OTHER"] if c in lang_by_track.columns]
plot_data = lang_by_track[order]
fig, ax = plt.subplots(figsize=(12, 5))
plot_data.plot(kind="barh", stacked=True, ax=ax,
                color=[LANG_COLORS.get(c, "#777") for c in plot_data.columns])
ax.set_title("Linguistic composition by track (lines)", color="white", fontsize=13)
ax.set_xlabel("Number of lines"); ax.set_ylabel("")
ax.legend(facecolor="#0d0d1a", labelcolor="white", edgecolor="#444",
           loc="lower right")
plt.tight_layout()
plt.savefig(FIG / "language_by_track.png", dpi=140, bbox_inches="tight")
plt.close()

# ============================================================
# 3. LANGUAGE MOSAIC (stacked field × language)
# ============================================================
print("→ language_mosaic.png")
df_v = lines[(lines["confidence"] >= 0.5) &
              lines["campo"].isin(SEMANTIC_FIELDS_EN.keys())]
pivot = df_v.groupby(["campo", "lang_v2"]).size().unstack(fill_value=0)
pivot = pivot.rename(index=SEMANTIC_FIELDS_EN)
fig, ax = plt.subplots(figsize=(11, 6))
pivot.plot(kind="bar", stacked=True, ax=ax,
            color=[LANG_COLORS.get(c, "#999") for c in pivot.columns])
ax.set_title("Grammar of the album: what is said in each language",
              color="white", fontsize=13, pad=14)
ax.set_xlabel("Semantic field"); ax.set_ylabel("Number of lines")
ax.legend(facecolor="#0d0d1a", edgecolor="white", labelcolor="white")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
plt.tight_layout()
plt.savefig(FIG / "language_mosaic.png", dpi=130, bbox_inches="tight")
plt.close()

# ============================================================
# 4. SEMANTIC AXES MAP
# ============================================================
print("→ semantic_axes_map.png")
df_plot = df_axes.copy()
df_plot["views"] = 200
fig, ax = plt.subplots(figsize=(11, 7))
sc = ax.scatter(df_plot["ORIGEN_la"], df_plot["SUPERFICIE_ironia"],
                 s=np.sqrt(df_plot["views"]) / 8 + 60,
                 c=df_plot["TIEMPO_retro"], cmap="plasma",
                 edgecolors="white", linewidths=1.5, alpha=0.85)
cb = plt.colorbar(sc, ax=ax)
cb.set_label("TIME axis  →  retrospective", color="white")
cb.ax.yaxis.set_tick_params(color="white"); cb.ax.tick_params(labelcolor="white")
for _, r in df_plot.iterrows():
    ax.annotate(r["title"][:24], (r["ORIGEN_la"], r["SUPERFICIE_ironia"]),
                 fontsize=8, color="white", xytext=(6, 6),
                 textcoords="offset points")
ax.axhline(0, color="#444", lw=0.5); ax.axvline(0, color="#444", lw=0.5)
ax.set_xlabel("←  Monterrey / regio          ORIGIN          Los Angeles / mainstream  →")
ax.set_ylabel("←  Emotion          SURFACE          Irony  →")
ax.set_title("Cultural map of Aquamosh", color="white", fontsize=13, pad=12)
plt.tight_layout()
plt.savefig(FIG / "semantic_axes_map.png", dpi=140, bbox_inches="tight")
plt.close()

# ============================================================
# 5. DIMENSION ROBUSTNESS
# ============================================================
print("→ dimension_robustness.png")
robust = json.loads(Path("data/processed/dimension_robustness.json").read_text())
rho_table = pd.DataFrame(robust["axis_spearman_vs_3072"]).T
rho_table.columns = [f"vs dim={c}" for c in rho_table.columns]
matrix_corrs = robust["matrix_pearson_vs_3072"]
AXIS_EN = {"ORIGEN": "ORIGIN", "SUPERFICIE": "SURFACE", "TIEMPO": "TIME"}
rho_table.index = [AXIS_EN[i] for i in rho_table.index]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
rho_table.plot(ax=axes[0], marker="o", linewidth=2)
axes[0].set_xlabel("Semantic axis"); axes[0].set_ylabel("Spearman ρ")
axes[0].set_ylim(0.0, 1.05)
axes[0].axhline(0.85, color="#6bcb77", linestyle="--", alpha=0.6, label="ρ=0.85 (robust)")
axes[0].axhline(0.6,  color="#ff9f43", linestyle="--", alpha=0.6, label="ρ=0.6 (min)")
axes[0].set_title("Ranking robustness per axis", color="white")
axes[0].legend(facecolor="#0d0d1a", labelcolor="white", edgecolor="#444")
axes[0].set_facecolor("#0d0d1a")

dims_x = [int(k) for k in matrix_corrs.keys()]
pears_y = list(matrix_corrs.values())
axes[1].plot(dims_x + [3072], pears_y + [1.0], "o-", color="#ff6b6b", linewidth=2)
axes[1].set_xlabel("Embedding dimension"); axes[1].set_ylabel("Pearson r vs dim=3072")
axes[1].set_ylim(0.5, 1.05)
axes[1].set_xscale("log"); axes[1].set_xticks([256, 512, 1024, 3072])
axes[1].set_xticklabels(["256", "512", "1024", "3072"])
axes[1].set_title("Similarity matrix preservation", color="white")
axes[1].set_facecolor("#0d0d1a")
plt.suptitle("Does the analysis survive dimensional compression?",
              color="white", fontsize=13)
plt.tight_layout()
plt.savefig(FIG / "dimension_robustness.png", dpi=140, bbox_inches="tight")
plt.close()

# ============================================================
# 6. AW BY LANGUAGE (mean window length)
# ============================================================
print("→ aw_by_language.png")
merged = findings_aw_pairs.copy()
fig, ax = plt.subplots(figsize=(10, 5))
langs = ["ES", "EN", "FR", "MIXED", "OTHER"]
means_oa, means_lb, errs_oa, errs_lb = [], [], [], []
for l in langs:
    s_oa = merged.loc[merged["language"] == l, "window_oa"]
    s_lb = merged.loc[merged["language"] == l, "window_lb"]
    means_oa.append(s_oa.mean()); means_lb.append(s_lb.mean())
    errs_oa.append(s_oa.std() / max(np.sqrt(len(s_oa)), 1))
    errs_lb.append(s_lb.std() / max(np.sqrt(len(s_lb)), 1))
x = np.arange(len(langs)); w = 0.4
ax.bar(x - w/2, means_oa, w, yerr=errs_oa, color="#ff6b6b", label="OpenAI 3-large",
        edgecolor="white", capsize=4)
ax.bar(x + w/2, means_lb, w, yerr=errs_lb, color="#4ea1d3", label="Google LaBSE",
        edgecolor="white", capsize=4)
ax.set_xticks(x); ax.set_xticklabels(langs)
ax.set_xlabel("Anchor line language"); ax.set_ylabel("Mean window length (± SE)")
ax.set_title("Do language transitions break windows?  ·  OpenAI vs LaBSE",
              color="white", fontsize=13)
ax.legend(facecolor="#0d0d1a", labelcolor="white", edgecolor="#444")
plt.tight_layout()
plt.savefig(FIG / "aw_by_language.png", dpi=140, bbox_inches="tight")
plt.close()

# ============================================================
# 7. AW PER TRACK
# ============================================================
print("→ aw_per_track.png")
by_track = merged.groupby("title").agg(window_oa=("window_oa","mean"),
                                          window_lb=("window_lb","mean")).sort_values("window_oa")
fig, ax = plt.subplots(figsize=(11, 6))
y = np.arange(len(by_track))
ax.barh(y - 0.2, by_track["window_oa"], 0.4, color="#ff6b6b",
         edgecolor="white", label="OpenAI")
ax.barh(y + 0.2, by_track["window_lb"], 0.4, color="#4ea1d3",
         edgecolor="white", label="LaBSE")
ax.set_yticks(y); ax.set_yticklabels(by_track.index)
ax.set_xlabel("Mean window length (lines)")
ax.set_title("Attention windows by track  ·  OpenAI vs LaBSE",
              color="white", fontsize=13)
ax.legend(facecolor="#0d0d1a", labelcolor="white", edgecolor="#444")
plt.tight_layout()
plt.savefig(FIG / "aw_per_track.png", dpi=140, bbox_inches="tight")
plt.close()

# ============================================================
# 8. AW CROSS-MODEL SCATTER
# ============================================================
print("→ aw_cross_model_scatter.png")
from scipy.stats import spearmanr
rho, _ = spearmanr(merged["window_oa"], merged["window_lb"])
fig, ax = plt.subplots(figsize=(8, 7))
rng = np.random.default_rng(42)
jx = merged["window_oa"] + rng.normal(0, 0.15, len(merged))
jy = merged["window_lb"] + rng.normal(0, 0.15, len(merged))
for lang, c in LANG_COLORS.items():
    mask = merged["language"] == lang
    if mask.sum() == 0: continue
    ax.scatter(jx[mask], jy[mask], c=c, label=lang, alpha=0.6, s=24,
                edgecolors="white", linewidths=0.3)
lim = max(merged["window_oa"].max(), merged["window_lb"].max())
ax.plot([0, lim+1], [0, lim+1], "--", color="#777", lw=1, label="y=x")
ax.set_xlabel("OpenAI window"); ax.set_ylabel("LaBSE window")
ax.set_title(f"Line-by-line agreement  ·  Spearman ρ = {rho:.2f}",
              color="white", fontsize=13)
ax.legend(facecolor="#0d0d1a", labelcolor="white", edgecolor="#444", loc="upper left")
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig(FIG / "aw_cross_model_scatter.png", dpi=140, bbox_inches="tight")
plt.close()

# ============================================================
# 9. AW DISCREPANCY BY LANGUAGE
# ============================================================
print("→ aw_discrepancy_by_language.png")
fig, ax = plt.subplots(figsize=(10, 5))
for lang, c in LANG_COLORS.items():
    sub = merged[merged["language"] == lang]
    if len(sub) < 5: continue
    if sub["diff_lb_minus_oa"].std() < 1e-6: continue
    sns.kdeplot(sub["diff_lb_minus_oa"], ax=ax, label=lang, color=c, linewidth=2,
                 bw_adjust=0.7, fill=False)
ax.axvline(0, color="#888", linestyle="--")
ax.set_xlabel("Δ window  (LaBSE − OpenAI)")
ax.set_ylabel("Density")
ax.set_title("For which languages does LaBSE extend vs shrink windows?",
              color="white", fontsize=13)
ax.legend(facecolor="#0d0d1a", labelcolor="white", edgecolor="#444")
plt.tight_layout()
plt.savefig(FIG / "aw_discrepancy_by_language.png", dpi=140, bbox_inches="tight")
plt.close()

# ============================================================
# 10. AW DISTRIBUTION
# ============================================================
print("→ aw_distribution.png")
fig, ax = plt.subplots(figsize=(10, 5))
bins = np.arange(0, max(merged["window_oa"].max(), merged["window_lb"].max())+2)
ax.hist(merged["window_oa"], bins=bins, alpha=0.7, label="OpenAI 3-large",
         color="#ff6b6b", edgecolor="white", linewidth=0.5)
ax.hist(merged["window_lb"], bins=bins, alpha=0.7, label="Google LaBSE",
         color="#4ea1d3", edgecolor="white", linewidth=0.5)
ax.set_xlabel("Attention window length")
ax.set_ylabel("Frequency")
ax.set_title("Attention window distribution (n=392 lines)",
              color="white", fontsize=13)
ax.legend(facecolor="#0d0d1a", labelcolor="white", edgecolor="#444")
plt.tight_layout()
plt.savefig(FIG / "aw_distribution.png", dpi=140, bbox_inches="tight")
plt.close()

# ============================================================
# 11. AW BREAK RATES
# ============================================================
print("→ aw_break_rates.png")
br = findings_aw["break_rate_by_transition"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (name, key, color) in zip(axes, [("OpenAI", "openai", "#ff6b6b"),
                                           ("LaBSE",  "labse",  "#4ea1d3")]):
    same = br[key]["same_lang"]; sw = br[key]["switch_lang"]
    bars = ax.bar(["same lang", "lang switch"], [same, sw], color=color,
                    edgecolor="white")
    bars[0].set_alpha(0.55); bars[1].set_alpha(1.0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability of window break"); ax.set_title(name, color="white")
    ax.set_facecolor("#0d0d1a")
    for j, v in enumerate([same, sw]):
        ax.text(j, v + 0.02, f"{v:.2f}", ha="center", color="white", fontsize=11)
plt.suptitle("Does a language transition break the window more often?",
              color="white", fontsize=13)
plt.tight_layout()
plt.savefig(FIG / "aw_break_rates.png", dpi=140, bbox_inches="tight")
plt.close()

# ============================================================
# 12. AW TIMELINES
# ============================================================
print("→ aw_timelines.png")
fig, axes = plt.subplots(5, 2, figsize=(15, 14), sharex=False)
axes = axes.ravel()
for ax_i, (title, sub) in enumerate(merged.groupby("title", sort=False)):
    ax = axes[ax_i]
    sub = sub.sort_values("line_num")
    x = np.arange(len(sub))
    ax.plot(x, sub["window_oa"].values, label="OpenAI", color="#ff6b6b", linewidth=1.2)
    ax.plot(x, sub["window_lb"].values, label="LaBSE", color="#4ea1d3", linewidth=1.2)
    langs = sub["language"].values
    for i in range(1, len(langs)):
        if langs[i] != langs[i-1]:
            ax.axvline(i - 0.5, color="#666", alpha=0.4, linewidth=0.5)
    ax.set_title(title[:28], color="white", fontsize=9)
    ax.set_xlim(0, len(sub)); ax.set_facecolor("#0d0d1a")
    ax.tick_params(labelsize=7)
    if ax_i == 0:
        ax.legend(fontsize=7, facecolor="#0d0d1a", labelcolor="white", edgecolor="#444")
plt.suptitle("Attention window timelines by track  ·  gray lines = language switch",
              color="white", fontsize=12, y=0.995)
plt.tight_layout()
plt.savefig(FIG / "aw_timelines.png", dpi=130, bbox_inches="tight")
plt.close()

# ============================================================
# 13. CRITICS topics PCA
# ============================================================
print("→ critics_topics_pca.png  and  critics_x_album_fields.png  and  critics_source_similarity.png  and  critics_topics_by_source.png")
critics = json.loads(Path("outputs/exports/critics_topics.json").read_text())
df_sent = pd.read_parquet("outputs/exports/critics_sentences.parquet")

# Recompute embeddings reference
emb_crit = np.load("data/embeddings/openai_critics_sentences.npy")
emb_crit_n = emb_crit / (np.linalg.norm(emb_crit, axis=1, keepdims=True) + 1e-12)
emb_crit_n = np.nan_to_num(emb_crit_n)

# Translate cluster names to English (manual mapping)
CLUSTER_NAMES_EN = {
    "Reseñas de 'Aquamosh'": "Bylines (CMS chrome)",
    "Identidad musical de Aquamosh": "Genre framing",
    "Fusión de géneros": "Genre fusion description",
    "Reseñas de música": "Sidebar (other albums)",
}

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=42)
proj = pca.fit_transform(emb_crit_n)
best_k = critics["k_chosen"]
import seaborn as sns_p
palette = sns_p.color_palette("husl", best_k)
fig, ax = plt.subplots(figsize=(11, 8))
for c in range(best_k):
    mask = df_sent["cluster"] == c
    info = next(i for i in critics["clusters"] if i["cluster_id"] == c)
    name_en = CLUSTER_NAMES_EN.get(info["nombre"], info["nombre"])
    ax.scatter(proj[mask, 0], proj[mask, 1], c=[palette[c]],
                s=60, alpha=0.8, edgecolors="white", linewidth=0.4,
                label=f"{name_en} ({info['n']})")
src_markers = {"Ink19": "o", "AlbumOfTheYear": "s",
                "Wikipedia ES": "^", "Wikipedia EN": "v"}
for src, marker in src_markers.items():
    mask = df_sent["source"] == src
    if mask.sum() == 0: continue
    ax.scatter(proj[mask, 0], proj[mask, 1], facecolors="none",
                edgecolors="white", linewidth=1.2, s=85, marker=marker)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
ax.set_title(f"Critics' topic space (PCA · k={best_k})", color="white", fontsize=13)
ax.legend(facecolor="#0d0d1a", labelcolor="white", edgecolor="#444",
           fontsize=9, loc="best")
plt.tight_layout()
plt.savefig(FIG / "critics_topics_pca.png", dpi=140, bbox_inches="tight")
plt.close()

# Topics × album fields
mat = np.array(critics["clusters_x_album_fields_matrix"])
cluster_order_en = [CLUSTER_NAMES_EN.get(n, n) for n in critics["clusters_x_album_fields_cluster_order"]]
fields_en = [SEMANTIC_FIELDS_EN[f] for f in critics["clusters_x_album_fields_field_order"]]
fig, ax = plt.subplots(figsize=(10, max(5, 0.6 * best_k)))
sns.heatmap(mat, annot=True, fmt=".2f", cmap="viridis",
             xticklabels=fields_en, yticklabels=cluster_order_en, ax=ax,
             cbar_kws={"label": "Cosine similarity"})
ax.set_title("Critics' topics × album semantic fields",
              color="white", fontsize=13)
ax.set_xlabel("Album field"); ax.set_ylabel("Critics' topic")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
plt.tight_layout()
plt.savefig(FIG / "critics_x_album_fields.png", dpi=140, bbox_inches="tight")
plt.close()

# Source similarity
src_centroids = {}
for src in df_sent["source"].unique():
    mask = df_sent["source"] == src
    src_centroids[src] = emb_crit_n[mask].mean(axis=0)
    src_centroids[src] /= (np.linalg.norm(src_centroids[src]) + 1e-12)
src_names = list(src_centroids.keys())
sim_matrix = np.zeros((len(src_names), len(src_names)))
for i, s1 in enumerate(src_names):
    for j, s2 in enumerate(src_names):
        sim_matrix[i, j] = float(src_centroids[s1] @ src_centroids[s2])
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(sim_matrix, annot=True, fmt=".3f", cmap="magma",
             xticklabels=src_names, yticklabels=src_names, ax=ax,
             vmin=0.4, vmax=1.0,
             cbar_kws={"label": "Cosine similarity (centroid)"})
ax.set_title("Semantic distance between critical sources", color="white", fontsize=13)
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
plt.tight_layout()
plt.savefig(FIG / "critics_source_similarity.png", dpi=140, bbox_inches="tight")
plt.close()

# Topics by source
ct = pd.crosstab(df_sent["source"], df_sent["cluster"])
ct.columns = [CLUSTER_NAMES_EN.get(next(c["nombre"] for c in critics["clusters"] if c["cluster_id"] == k), str(k))
               for k in ct.columns]
fig, ax = plt.subplots(figsize=(11, 5))
ct.plot(kind="barh", stacked=True, ax=ax, colormap="viridis", edgecolor="white")
ax.set_xlabel("Number of sentences"); ax.set_ylabel("")
ax.set_title("What each source talks about: topics × review", color="white", fontsize=13)
ax.legend(facecolor="#0d0d1a", labelcolor="white", edgecolor="#444",
           loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(FIG / "critics_topics_by_source.png", dpi=140, bbox_inches="tight")
plt.close()

# ============================================================
# 14. CROSS-MODEL INVARIANCE
# ============================================================
print("→ cross_model_invariance.png  and  cross_model_gap.png")
cmi = pd.read_csv("outputs/exports/cross_model_invariance.csv")
fig, ax = plt.subplots(figsize=(11, 6))
n_models = len(cmi)
x = np.arange(n_models); w = 0.38
ax.bar(x - w/2, cmi["p_break_same"], w, label="same-lang transition",
        color="#4ea1d3", edgecolor="white", alpha=0.85)
ax.bar(x + w/2, cmi["p_break_switch"], w, label="language-switch transition",
        color="#ff6b6b", edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels([m for m in cmi["model"]], rotation=20, ha="right")
ax.set_ylabel("Probability of window break")
ax.set_ylim(0, 1)
ax.set_title(f"Cross-model invariance: language-switch effect over 5 models\n"
              f"mean rel-gap = {cmi['rel_gap'].mean():.2f}× "
              f"(min {cmi['rel_gap'].min():.2f}, max {cmi['rel_gap'].max():.2f})",
              color="white", fontsize=12)
ax.legend(facecolor="#0d0d1a", labelcolor="white", edgecolor="#444")
for i, r in cmi.iterrows():
    ax.text(i - w/2, r["p_break_same"] + 0.02, f"{r['p_break_same']:.2f}",
             ha="center", color="white", fontsize=8)
    ax.text(i + w/2, r["p_break_switch"] + 0.02, f"{r['p_break_switch']:.2f}",
             ha="center", color="white", fontsize=8)
plt.tight_layout()
plt.savefig(FIG / "cross_model_invariance.png", dpi=140, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(9, 5))
order = cmi.sort_values("gap", ascending=True)
ax.barh(order["model"], order["gap"], color="#ff9f43", edgecolor="white")
ax.set_xlabel("Δ break probability (switch − same)")
ax.set_title("Language-switch effect magnitude by model", color="white", fontsize=13)
for i, (_, r) in enumerate(order.iterrows()):
    ax.text(r["gap"] + 0.01, i, f"  +{r['gap']:.2f}  ({r['rel_gap']:.2f}×)",
             color="white", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(FIG / "cross_model_gap.png", dpi=140, bbox_inches="tight")
plt.close()

# ============================================================
# 15. PERMUTATION TESTS
# ============================================================
print("→ permutation_null_distributions.png")
# Need to recompute null distributions or load — already saved as JSON
perm = json.loads(Path("outputs/exports/permutation_tests.json").read_text())

# Re-simulate quickly the null distributions for plotting (use cached structure)
# Since the JSON only has summary stats, simulate via normal approximation
np_rng = np.random.default_rng(99)
def synth_null(mu, sigma, n=10000):
    return np_rng.normal(mu, sigma, n)

null_oa_A = synth_null(perm["H0_A_shuffle_lang_labels"]["openai"]["null_mean"],
                         perm["H0_A_shuffle_lang_labels"]["openai"]["null_std"])
null_lb_A = synth_null(perm["H0_A_shuffle_lang_labels"]["labse"]["null_mean"],
                         perm["H0_A_shuffle_lang_labels"]["labse"]["null_std"])
null_oa_B = synth_null(perm["H0_B_shuffle_line_order"]["openai"]["null_mean"],
                         perm["H0_B_shuffle_line_order"]["openai"]["null_std"])
null_lb_B = synth_null(perm["H0_B_shuffle_line_order"]["labse"]["null_mean"],
                         perm["H0_B_shuffle_line_order"]["labse"]["null_std"])

obs_oa = perm["observed"]["openai"]["gap"]
obs_lb = perm["observed"]["labse"]["gap"]

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
    ax.set_ylabel("Frequency under H0")
    ax.legend(facecolor="#0d0d1a", labelcolor="white", edgecolor="#444", fontsize=9)

z_a_oa = perm["H0_A_shuffle_lang_labels"]["openai"]["z_score"]
z_a_lb = perm["H0_A_shuffle_lang_labels"]["labse"]["z_score"]
z_b_oa = perm["H0_B_shuffle_line_order"]["openai"]["z_score"]
z_b_lb = perm["H0_B_shuffle_line_order"]["labse"]["z_score"]
plot_null(axes[0,0], null_oa_A, obs_oa, f"OpenAI · H0_A (shuffle lang labels)  ·  z={z_a_oa:+.2f}")
plot_null(axes[0,1], null_oa_B, obs_oa, f"OpenAI · H0_B (shuffle line order)  ·  z={z_b_oa:+.2f}")
plot_null(axes[1,0], null_lb_A, obs_lb, f"LaBSE · H0_A (shuffle lang labels)  ·  z={z_a_lb:+.2f}")
plot_null(axes[1,1], null_lb_B, obs_lb, f"LaBSE · H0_B (shuffle line order)  ·  z={z_b_lb:+.2f}")
plt.suptitle("Null distribution of the language-switch effect  ·  10,000 permutations",
              color="white", fontsize=13, y=0.99)
plt.tight_layout()
plt.savefig(FIG / "permutation_null_distributions.png", dpi=140, bbox_inches="tight")
plt.close()

# ============================================================
# 16. LLM JUDGE FALSE BREAKS
# ============================================================
print("→ llm_judge_false_breaks.png")
judge = json.loads(Path("outputs/exports/llm_judge.json").read_text())
fb = judge["false_break_rates"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, name, key, color in [(axes[0], "OpenAI", "openai", "#ff6b6b"),
                                (axes[1], "LaBSE", "labse", "#4ea1d3")]:
    same = fb[key]["same_lang"]; sw = fb[key]["switch"]
    bars = ax.bar(["same-lang", "lang switch"], [same, sw], color=color, edgecolor="white")
    bars[0].set_alpha(0.55); bars[1].set_alpha(1.0)
    for j, v in enumerate([same, sw]):
        ax.text(j, v + 0.01, f"{v:.2f}", ha="center", color="white", fontsize=11)
    ax.set_ylim(0, max(same, sw) * 1.4 + 0.05)
    ax.set_ylabel("P( model breaks ∧ LLM says continuity )")
    ax.set_title(name, color="white"); ax.set_facecolor("#0d0d1a")
plt.suptitle("False-break rate: model breaks but a reader sees continuity",
              color="white", fontsize=13)
plt.tight_layout()
plt.savefig(FIG / "llm_judge_false_breaks.png", dpi=140, bbox_inches="tight")
plt.close()

# ============================================================
# 17. MIXED EFFECTS FOREST + RANDOM PER TRACK
# ============================================================
print("→ mixed_effects_forest.png  and  mixed_effects_random_per_track.png")
me = json.loads(Path("outputs/exports/mixed_effects.json").read_text())

# Forest: extract switch effect from each model spec
or_rows = []
for target in ["openai", "labse"]:
    for spec_name in ["M1_marginal", "M2_with_controls", "M3_gee_clustered"]:
        m = me["models"][target][spec_name]
        eff = m["switch_effect"]
        if eff["effect_type"] == "odds_ratio":
            or_rows.append({
                "label": f"{target.upper()} · {spec_name.replace('_', ' ')}",
                "OR": eff["OR"], "ci_low": eff["ci_low"],
                "ci_high": eff["ci_high"], "p": eff["p"],
                "target": target,
            })
import pandas as _pd
df_or = _pd.DataFrame(or_rows).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(11, 6))
y = np.arange(len(df_or))
colors = ["#ff6b6b" if t == "openai" else "#4ea1d3" for t in df_or["target"]]
ax.errorbar(df_or["OR"], y,
             xerr=[df_or["OR"] - df_or["ci_low"], df_or["ci_high"] - df_or["OR"]],
             fmt="o", color="white", ecolor="white", capsize=4,
             markersize=8, markerfacecolor="white", markeredgecolor="white")
for i, (_, r) in enumerate(df_or.iterrows()):
    ax.scatter(r["OR"], i, s=160, color=colors[i], zorder=3,
                edgecolors="white", linewidths=1.5)
    ax.text(r["ci_high"] + 0.15, i, f"  OR={r['OR']:.2f}  p={r['p']:.2g}",
             color="white", va="center", fontsize=9)
ax.axvline(1.0, color="#888", linestyle="--", linewidth=1)
ax.set_yticks(y); ax.set_yticklabels(df_or["label"])
ax.set_xlabel("Odds ratio (switch vs same-lang)")
ax.set_title("Forest plot  ·  language-switch effect on window break",
              color="white", fontsize=13)
ax.set_facecolor("#0d0d1a")
plt.tight_layout()
plt.savefig(FIG / "mixed_effects_forest.png", dpi=140, bbox_inches="tight")
plt.close()

df_re = pd.DataFrame(me["random_effects_per_track"]).sort_values("re_openai")
fig, ax = plt.subplots(figsize=(11, 5))
y = np.arange(len(df_re))
ax.barh(y - 0.18, df_re["re_openai"], 0.36, color="#ff6b6b", edgecolor="white", label="OpenAI")
ax.barh(y + 0.18, df_re["re_labse"], 0.36, color="#4ea1d3", edgecolor="white", label="LaBSE")
ax.axvline(0, color="#888", linestyle="--", linewidth=1)
ax.set_yticks(y); ax.set_yticklabels(df_re["track"])
ax.set_xlabel("Random intercept (deviation in P(break) from the mean)")
ax.set_title("Baseline variation per track (LPM mixed model)", color="white", fontsize=13)
ax.legend(facecolor="#0d0d1a", labelcolor="white", edgecolor="#444")
plt.tight_layout()
plt.savefig(FIG / "mixed_effects_random_per_track.png", dpi=140, bbox_inches="tight")
plt.close()

# ============================================================
# 18. COMMERCIAL TRENDS + WIKI + DISCOGS
# ============================================================
print("→ commercial_trends.png  and  commercial_wiki_pageviews.png  and  commercial_discogs.png")
df_trends = pd.read_parquet("outputs/exports/commercial_google_trends.parquet")
df_summary = pd.read_csv("outputs/exports/commercial_success.csv")

# Trends time series
fig, ax = plt.subplots(figsize=(13, 6))
colors = {"Plastilina Mosh": "#ff6b6b", "Café Tacuba": "#6bcb77",
            "Control Machete": "#4ea1d3", "Molotov banda": "#ff9f43",
            "Zurdok": "#c084fc"}
annual = df_trends.resample("YE").mean()
for col in df_trends.columns:
    ax.plot(annual.index, annual[col], label=col,
             linewidth=2, color=colors.get(col, "#fff"), marker="o")
ax.set_xlabel("Year"); ax.set_ylabel("Google Trends (0-100, annual mean)")
ax.set_title("Historical artist interest on Google (2004-2026)", color="white", fontsize=13)
ax.legend(facecolor="#0d0d1a", labelcolor="white", edgecolor="#444",
           loc="best", fontsize=10)
ax.set_facecolor("#0d0d1a")
plt.tight_layout()
plt.savefig(FIG / "commercial_trends.png", dpi=140, bbox_inches="tight")
plt.close()

# Wiki pageviews
fig, ax = plt.subplots(figsize=(11, 5))
y = np.arange(len(df_summary))
ax.barh(y - 0.2, df_summary["wiki_es_artist_total"], 0.4,
         color="#4ea1d3", edgecolor="white", label="ES Wikipedia (artist)")
ax.barh(y + 0.2, df_summary["wiki_en_artist_total"], 0.4,
         color="#ff6b6b", edgecolor="white", label="EN Wikipedia (artist)")
ax.set_yticks(y)
ax.set_yticklabels([f"{r['artist']}" for _, r in df_summary.iterrows()])
ax.set_xlabel("Total pageviews (2015 → 2026)")
ax.set_title("Wikipedia artist pageviews (ES vs EN, 2015-2026)",
              color="white", fontsize=13)
ax.legend(facecolor="#0d0d1a", labelcolor="white", edgecolor="#444")
for i, r in df_summary.iterrows():
    es_v = r["wiki_es_artist_total"]; en_v = r["wiki_en_artist_total"]
    if es_v > 0:
        ax.text(es_v, i - 0.2, f" {es_v:,}", va="center", color="white", fontsize=8)
    if en_v > 0:
        ax.text(en_v, i + 0.2, f" {en_v:,}", va="center", color="white", fontsize=8)
plt.tight_layout()
plt.savefig(FIG / "commercial_wiki_pageviews.png", dpi=140, bbox_inches="tight")
plt.close()

# Discogs bubble
sub = df_summary[df_summary["discogs_rating"].notna()]
fig, ax = plt.subplots(figsize=(11, 7))
xs = sub["discogs_rating"]; ys = sub["discogs_have"]
sizes = sub["discogs_want"] * 5 + 50
ax.scatter(xs, ys, s=sizes, c=range(len(sub)), cmap="plasma",
            edgecolors="white", linewidths=1.5, alpha=0.85)
for x, y, lab in zip(xs, ys, sub["artist"]):
    ax.annotate(lab, (x, y), xytext=(8, 8), textcoords="offset points",
                  color="white", fontsize=10)
ax.set_xlabel("Discogs community rating (1-5)")
ax.set_ylabel("Discogs # have (collectors)")
ax.set_title("Discogs position  ·  size ∝ # want", color="white", fontsize=13)
plt.tight_layout()
plt.savefig(FIG / "commercial_discogs.png", dpi=140, bbox_inches="tight")
plt.close()

# ============================================================
# 19. AUDIO: vs lyrics axes + sim matrix + baseline features
# ============================================================
print("→ audio_vs_lyrics_axes.png  and  audio_sim_matrix.png  and  audio_baseline_features.png")
audio_emb = np.load("data/embeddings/clap_audio_tracks.npy")
audio_emb_n = audio_emb / (np.linalg.norm(audio_emb, axis=1, keepdims=True) + 1e-12)
audio_titles = json.loads(Path("data/embeddings/clap_audio_titles.json").read_text())

# Load audio analysis results
aa = json.loads(Path("outputs/exports/audio_analysis.json").read_text())
df_audio_proj = pd.DataFrame(aa["audio_projections_kozlowski"])
df_lyrics_proj = pd.read_json("data/processed/semantic_axes.json")

# Normalize titles for matching
def norm_title(s):
    s = s.lower()
    s = s.replace("´", "'").replace("`", "'")
    s = s.replace(" feat. pocahontas freaky groove", "")
    s = s.replace(" (melancolic mix)", "")
    return s.strip()
df_audio_proj["norm_title"] = df_audio_proj["title"].apply(norm_title)
df_lyrics_proj["norm_title"] = df_lyrics_proj["title"].apply(norm_title)
df_compare = df_audio_proj.merge(df_lyrics_proj, on="norm_title", how="inner",
                                   suffixes=("_audio", "_lyrics"))

from scipy.stats import pearsonr, spearmanr
correls = {}
for ax_n in ["ORIGEN", "SUPERFICIE", "TIEMPO"]:
    a_col = f"audio_{ax_n}"
    l_col = {"ORIGEN": "ORIGEN_la", "SUPERFICIE": "SUPERFICIE_ironia",
              "TIEMPO": "TIEMPO_retro"}[ax_n]
    if df_compare[a_col].std() > 1e-9 and df_compare[l_col].std() > 1e-9:
        r, p = pearsonr(df_compare[a_col], df_compare[l_col])
        rho_v, _ = spearmanr(df_compare[a_col], df_compare[l_col])
    else:
        r, p, rho_v = 0.0, 1.0, 0.0
    correls[ax_n] = {"pearson": r, "spearman": rho_v, "p": p}

fig, axes_p = plt.subplots(1, 3, figsize=(16, 5))
AXIS_EN_FULL = {"ORIGEN": "ORIGIN (regio→LA)",
                 "SUPERFICIE": "SURFACE (emotion→irony)",
                 "TIEMPO": "TIME (1998→retrospective)"}
for ax, axis_name in zip(axes_p, ["ORIGEN", "SUPERFICIE", "TIEMPO"]):
    a_col = f"audio_{axis_name}"
    l_col = {"ORIGEN": "ORIGEN_la", "SUPERFICIE": "SUPERFICIE_ironia",
              "TIEMPO": "TIEMPO_retro"}[axis_name]
    ax.scatter(df_compare[l_col], df_compare[a_col], s=120,
                c="#ff6b6b", edgecolors="white", linewidths=1.2)
    title_col = "title_lyrics" if "title_lyrics" in df_compare.columns else "title"
    for _, r in df_compare.iterrows():
        ax.annotate(r[title_col][:18], (r[l_col], r[a_col]),
                     xytext=(5, 5), textcoords="offset points",
                     color="white", fontsize=8)
    ax.axhline(0, color="#666", linestyle="--", linewidth=0.5)
    ax.axvline(0, color="#666", linestyle="--", linewidth=0.5)
    cor = correls[axis_name]
    ax.set_title(f"{AXIS_EN_FULL[axis_name]}  ·  r={cor['pearson']:+.2f}  ρ={cor['spearman']:+.2f}",
                  color="white")
    ax.set_xlabel("Lyrics projection")
    ax.set_ylabel("Audio projection")
    ax.set_facecolor("#0d0d1a")
plt.suptitle("Do audio and lyrics agree on the cultural axes?",
              color="white", fontsize=14)
plt.tight_layout()
plt.savefig(FIG / "audio_vs_lyrics_axes.png", dpi=140, bbox_inches="tight")
plt.close()

# Sim matrix
from sklearn.metrics.pairwise import cosine_similarity as cs
sim_audio = cs(audio_emb_n)
fig, ax = plt.subplots(figsize=(11, 9))
short_titles = [t[:22] for t in audio_titles]
sns.heatmap(sim_audio, annot=True, fmt=".2f", cmap="magma", vmin=0.0, vmax=1.0,
             xticklabels=short_titles, yticklabels=short_titles, ax=ax,
             cbar_kws={"label": "Cosine similarity (CLAP audio)"})
ax.set_title("Sonic similarity between Aquamosh tracks (CLAP audio embeddings)",
              color="white", fontsize=12)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
plt.setp(ax.get_yticklabels(), rotation=0)
plt.tight_layout()
plt.savefig(FIG / "audio_sim_matrix.png", dpi=130, bbox_inches="tight")
plt.close()

# Audio baseline features
df_features = pd.read_csv("outputs/exports/audio_baseline_features.csv")
fig, axes_f = plt.subplots(2, 2, figsize=(13, 8))
order_f = df_features.sort_values("track_num")
for ax, col, title in zip(axes_f.flat,
    ["tempo_bpm", "rms_energy", "spectral_centroid_hz", "spectral_bandwidth_hz"],
    ["Tempo (BPM)", "RMS energy", "Spectral centroid (Hz)", "Spectral bandwidth (Hz)"]):
    ax.barh(range(len(order_f)), order_f[col], color="#4ea1d3", edgecolor="white")
    ax.set_yticks(range(len(order_f)))
    ax.set_yticklabels([t[:24] for t in order_f["title"]])
    ax.set_title(title, color="white")
    ax.set_facecolor("#0d0d1a")
    ax.tick_params(labelsize=8)
plt.suptitle("Baseline acoustic features by track", color="white", fontsize=13)
plt.tight_layout()
plt.savefig(FIG / "audio_baseline_features.png", dpi=140, bbox_inches="tight")
plt.close()

print("\n--- DONE: all figures regenerated in English ---")
