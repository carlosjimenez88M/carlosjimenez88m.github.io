"""
Step 7 — figures (static PNGs for the post + interactive Plotly community graph).

Outputs land in outputs/figures/ and are copied to the published Hugo folder
tidytuesday/2026-06-16-beatles-evolution/ referenced by the post.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PROC = ROOT / "data" / "processed"
EMB = ROOT / "data" / "embeddings"
EXP = ROOT / "outputs" / "exports"
FIG = ROOT / "outputs" / "figures"
PUB = REPO / "tidytuesday" / "2026-06-16-beatles-evolution"
FIG.mkdir(parents=True, exist_ok=True)
PUB.mkdir(parents=True, exist_ok=True)

ALBUM_ORDER = ["Rubber Soul", "Revolver",
               "Sgt. Pepper's Lonely Hearts Club Band", "Abbey Road"]
SHORT = ["Rubber Soul", "Revolver", "Sgt. Pepper's", "Abbey Road"]
# chronological, warm->cool palette to read as time
PAL = {"Rubber Soul": "#E8A33D", "Revolver": "#D45B4E",
       "Sgt. Pepper's Lonely Hearts Club Band": "#6F4FA0", "Abbey Road": "#3A7CA5"}
PAL_SHORT = {"Rubber Soul": "#E8A33D", "Revolver": "#D45B4E",
             "Sgt. Pepper's": "#6F4FA0", "Abbey Road": "#3A7CA5"}
sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150, "font.size": 11})


def save(fig, name: str) -> None:
    p = FIG / name
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    shutil.copy(p, PUB / name)
    plt.close(fig)
    print(f"  wrote {name}")


def l2norm(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def fig_topics():
    prof = pd.read_csv(EXP / "album_topic_profile.csv", index_col=0)
    ent = pd.read_csv(EXP / "album_topic_entropy.csv")
    info = json.load(open(EXP / "topic_info.json"))
    label = {t["Topic"]: t["label"] for t in info}
    prof = prof.reindex(ALBUM_ORDER)
    prof.columns = [str(c) for c in prof.columns]
    # keep the 12 most frequent topics for readability
    top = prof.sum(axis=0).sort_values(ascending=False).head(12).index.tolist()
    sub = prof[top]
    sub.columns = [f"{c} · {label.get(int(c), c)}" for c in top]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2),
                                   gridspec_kw={"width_ratios": [3, 1]})
    sns.heatmap(sub, cmap="rocket_r", ax=ax1, cbar_kws={"label": "share of album's clustered lines"},
                yticklabels=SHORT, linewidths=.5, linecolor="white")
    ax1.set_title("Per-album topic profile (shared BERTopic space, top 12 topics)", fontsize=12)
    ax1.set_xlabel("topic"); ax1.set_ylabel("")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=55, ha="right", fontsize=8)
    ax2.plot(SHORT, ent["topic_entropy_norm"], "o-", color="#444", lw=2, ms=9)
    for x, y in zip(SHORT, ent["topic_entropy_norm"]):
        ax2.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 9), fontsize=9)
    ax2.set_title("Normalized topic entropy", fontsize=12)
    ax2.set_ylim(0.6, 0.9); ax2.set_ylabel("entropy (0–1)")
    ax2.set_xticklabels(SHORT, rotation=30, ha="right")
    fig.suptitle("Topic modeling per album — the descriptive entry point", fontweight="bold", y=1.02)
    save(fig, "fig1_topic_profile.png")


def fig_lexical():
    geo = pd.read_csv(EXP / "evolution_geometry.csv")
    songlex = pd.read_csv(EXP / "song_lexical.csv")
    geo = geo.set_index("album").reindex(ALBUM_ORDER).reset_index()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(SHORT, geo["ttr"], "o-", color="#D45B4E", lw=2.4, ms=9, label="Type-token ratio")
    ax1b = ax1.twinx()
    ax1b.plot(SHORT, geo["hapax_ratio"], "s--", color="#3A7CA5", lw=2.4, ms=8, label="Hapax ratio")
    ax1.set_ylabel("Type-token ratio", color="#D45B4E")
    ax1b.set_ylabel("Hapax ratio", color="#3A7CA5")
    ax1.set_title("Album-level lexical richness rises monotonically\n(TTR r=+0.93, hapax r=+0.95 vs release order)", fontsize=11)
    ax1.set_xticklabels(SHORT, rotation=25, ha="right")
    # per-song scatter (the honest caveat)
    order_map = {a: i + 1 for i, a in enumerate(ALBUM_ORDER)}
    songlex["order"] = songlex["album"].map(order_map)
    jitter = np.random.default_rng(1).normal(0, 0.06, len(songlex))
    for a in ALBUM_ORDER:
        m = songlex.album == a
        ax2.scatter(songlex["order"][m] + jitter[m.to_numpy()], songlex["hapax_ratio"][m],
                    color=PAL[a], s=45, alpha=.8, edgecolor="white", linewidth=.5)
    ax2.set_xticks([1, 2, 3, 4]); ax2.set_xticklabels(SHORT, rotation=25, ha="right")
    ax2.set_ylabel("hapax ratio (per song)")
    ax2.set_title("…but per song (n=58) the trend is not significant\n(OLS β_order p=0.15, length-controlled) — an aggregation effect", fontsize=11)
    save(fig, "fig2_lexical_diversification.png")


def fig_semantic_map():
    songs = pd.read_parquet(PROC / "corpus_songs.parquet").reset_index(drop=True)
    emb = l2norm(np.load(EMB / "songs_openai_3large.npy"))
    ts = TSNE(n_components=2, perplexity=10, metric="cosine",
              init="pca", random_state=42).fit_transform(emb)
    cmat = pd.read_csv(EXP / "album_centroid_sim.csv", index_col=0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6),
                                   gridspec_kw={"width_ratios": [1.4, 1]})
    for a in ALBUM_ORDER:
        m = (songs.album == a).to_numpy()
        ax1.scatter(ts[m, 0], ts[m, 1], color=PAL[a], s=80, alpha=.85,
                    edgecolor="white", linewidth=.8, label=SHORT[ALBUM_ORDER.index(a)])
    ax1.set_title("t-SNE of song embeddings — albums intermingle\n(silhouette ≈ 0; detectable but not separable)", fontsize=11)
    ax1.set_xticks([]); ax1.set_yticks([]); ax1.legend(frameon=True, fontsize=9)
    sns.heatmap(cmat, annot=True, fmt=".3f", cmap="mako_r", ax=ax2, vmin=0.85, vmax=1.0,
                cbar_kws={"label": "centroid cosine"}, square=True,
                xticklabels=SHORT, yticklabels=SHORT)
    ax2.set_title("Album centroid similarity\n(steady drift ~0.10 per step)", fontsize=11)
    ax2.set_xticklabels(SHORT, rotation=30, ha="right")
    save(fig, "fig3_semantic_map.png")


def fig_community(nodes, edges):
    comp = pd.read_csv(EXP / "community_album_composition.csv", index_col=0)
    summ = json.load(open(EXP / "graph_summary.json"))
    comp.columns = SHORT
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                   gridspec_kw={"width_ratios": [1.3, 1]})
    comp.plot(kind="bar", stacked=True, ax=ax1,
              color=[PAL_SHORT[c] for c in comp.columns], width=.7)
    ax1.set_title("Louvain communities cut across albums\n"
                  f"(ARI={summ['community_vs_album_ARI']}, NMI={summ['community_vs_album_NMI']})", fontsize=11)
    ax1.set_xlabel("detected community"); ax1.set_ylabel("songs")
    ax1.legend(title="album", fontsize=8); ax1.set_xticklabels(comp.index, rotation=0)
    bars = ["Album partition\n(by record)", "Louvain\n(semantic)"]
    qs = [summ["modularity_album"], summ["modularity_communities"]]
    ax2.bar(bars, qs, color=["#bbb", "#6F4FA0"], width=.6)
    for x, y in zip(bars, qs):
        ax2.annotate(f"Q={y:.3f}", (x, y), textcoords="offset points", xytext=(0, 6),
                     ha="center", fontsize=11, fontweight="bold")
    ax2.set_title("Modularity: album structure is near-zero\nvs real cross-cutting structure", fontsize=11)
    ax2.set_ylabel("modularity Q"); ax2.set_ylim(0, max(qs) * 1.25)
    save(fig, "fig4_community_composition.png")


def fig_graph_static(nodes, edges):
    fig, ax = plt.subplots(figsize=(11, 9))
    for e in edges.itertuples():
        s, d = nodes.iloc[e.src], nodes.iloc[e.dst]
        ax.plot([s.x, d.x], [s.y, d.y], color="#cfcfcf" if not e.cross_album else "#e7c9c9",
                lw=0.4 + e.weight * 0.8, alpha=.5, zorder=1)
    for a in ALBUM_ORDER:
        m = (nodes.album == a)
        ax.scatter(nodes.x[m], nodes.y[m], s=40 + nodes.betweenness[m] * 900,
                   color=PAL[a], edgecolor="white", linewidth=.8, zorder=3,
                   label=SHORT[ALBUM_ORDER.index(a)])
    # annotate the top bridges
    for r in nodes.sort_values("betweenness", ascending=False).head(5).itertuples():
        ax.annotate(r.title, (r.x, r.y), fontsize=8, fontweight="bold",
                    xytext=(4, 4), textcoords="offset points", zorder=4)
    ax.set_title("The Beatles song-similarity graph (node size = betweenness / bridge role)\n"
                 "color = album · layout = semantic spring · 69% of edges cross albums",
                 fontsize=12)
    ax.set_xticks([]); ax.set_yticks([]); ax.legend(frameon=True, loc="upper left")
    save(fig, "fig5_song_graph.png")


def fig_permutation():
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    s = json.load(open(EXP / "stats_robustness.json"))
    cats = ["Silhouette\n(separability)", "Q album\n(modularity)"]
    obs = [s["C2_silhouette_obs"], s["C2b_Q_album_obs"]]
    nul = [s["C2_silhouette_null_mean"], s["C2b_Q_album_null_mean"]]
    x = np.arange(len(cats)); w = .35
    ax.bar(x - w/2, nul, w, label="null (shuffled albums)", color="#bbb")
    ax.bar(x + w/2, obs, w, label="observed", color="#D45B4E")
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_title("Album signal is statistically real but practically tiny\n"
                 "(silhouette p=0.004, Q_album p=0.007 — yet both ≈ 0)", fontsize=11)
    ax.legend()
    save(fig, "fig6_permutation.png")


def interactive_graph(nodes, edges):
    import plotly.graph_objects as go
    edge_x, edge_y = [], []
    for e in edges.itertuples():
        s, d = nodes.iloc[e.src], nodes.iloc[e.dst]
        edge_x += [s.x, d.x, None]; edge_y += [s.y, d.y, None]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(width=0.5, color="#d9d9d9"), hoverinfo="none"))
    palette = ["#E8A33D", "#D45B4E", "#6F4FA0", "#3A7CA5", "#5AAE61", "#A0522D", "#999"]
    for ci in sorted(nodes.community.unique()):
        m = nodes.community == ci
        sub = nodes[m]
        fig.add_trace(go.Scatter(
            x=sub.x, y=sub.y, mode="markers",
            marker=dict(size=8 + sub.betweenness * 120, color=palette[ci % len(palette)],
                        line=dict(width=1, color="white")),
            text=[f"<b>{t}</b><br>{a} ({y})<br>community {c} · betweenness {b:.3f}"
                  for t, a, y, c, b in zip(sub.title, sub.album_short, sub.year,
                                           sub.community, sub.betweenness)],
            hoverinfo="text", name=f"community {ci}"))
    fig.update_layout(
        title="The Beatles, lyric by lyric — semantic communities (node = song, size = bridge role)",
        showlegend=True, template="plotly_white", width=1000, height=720,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        legend=dict(title="Louvain community"))
    out = PUB / "beatles_song_graph.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    shutil.copy(out, FIG / "beatles_song_graph.html")
    print(f"  wrote beatles_song_graph.html (interactive)")


def main():
    nodes = pd.read_csv(EXP / "graph_nodes.csv")
    edges = pd.read_csv(EXP / "graph_edges.csv")
    print("generating figures ...")
    fig_topics()
    fig_lexical()
    fig_semantic_map()
    fig_community(nodes, edges)
    fig_graph_static(nodes, edges)
    fig_permutation()
    interactive_graph(nodes, edges)
    print(f"\nall figures in {FIG} and published to {PUB}")


if __name__ == "__main__":
    main()
