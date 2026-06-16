"""Assemble the study into a single reproducible notebook.

Heavy steps (Genius fetch, OpenAI embeddings, BERTopic) are guarded behind their
cached artifacts so the notebook executes end-to-end in seconds on a warm cache
while still showing the full pipeline source.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent
nb = new_notebook()
cells = []


def md(text): cells.append(new_markdown_cell(text))
def code(text): cells.append(new_code_cell(text))


md("""# The Beatles, Lyric by Lyric — How Four Albums Diffuse Through One Semantic Register

**Rubber Soul (1965) · Revolver (1966) · Sgt. Pepper's (1967) · Abbey Road (1969)**

A computational study of lyrical evolution. Part of the *Computational Musicology / Attention Windows* series.

## Central question (data-driven)
Rather than fix a thesis in advance, we let it emerge from topic modeling + embedding geometry + graph theory.
The pipeline:

1. **Topic modeling per album** (BERTopic, shared topic space) — the descriptive entry point.
2. **Embeddings + vector database** (OpenAI `text-embedding-3-large` → ChromaDB).
3. **Embedding-geometry evolution metrics** (cohesion, drift, NN-purity, lexical diversity).
4. **Graph theory** — the song-similarity network, Louvain communities, bridge songs (the centrepiece).
5. **Statistical robustness** — song-level OLS + permutation tests.

## What emerges
The four albums are **statistically detectable but practically inseparable** in embedding space
(silhouette ≈ 0 though p=0.004; album-partition graph modularity Q=0.06 vs a cross-cutting community
structure at Q=0.35). Lexical richness rises monotonically *per album* (TTR r=+0.93, hapax r=+0.95) but
this is an **aggregation effect**, not a significant per-song enrichment. The catalogue reads as **one
semantic register slowly diffusing** — a continuum with weak album-eddies and strong cross-album bridges
(*A Day in the Life*, *Got to Get You into My Life*).""")

code("""import json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path.cwd()
PROC, EMB, EXP = ROOT/"data"/"processed", ROOT/"data"/"embeddings", ROOT/"outputs"/"exports"
ALBUM_ORDER = ["Rubber Soul","Revolver","Sgt. Pepper's Lonely Hearts Club Band","Abbey Road"]
SHORT = ["Rubber Soul","Revolver","Sgt. Pepper's","Abbey Road"]""")

md("""## 1 · Data collection — Genius API (song by song)

Album search on Genius is unreliable for the Beatles (it merges compilations, singles and
foreign-language versions — `Sgt. Pepper's` came back with 216 "tracks"). We fetch **song by song**
against canonical UK tracklists (`collect_lyrics.py`), clean section headers and CMS noise, and
persist a line-level and a song-level corpus.""")

code("""songs = pd.read_parquet(PROC/"corpus_songs.parquet")
lines = pd.read_parquet(PROC/"corpus_lines.parquet")
summary = (songs.groupby(["album_order","album","year"])
           .agg(songs=("title","nunique"), lines=("n_lines","sum"), words=("n_words","sum"))
           .reset_index())
print(summary.to_string(index=False))
print(f"\\nTotal: {songs.title.nunique()} songs, {len(lines)} lines")""")

md("""## 2 · Topic modeling per album — BERTopic (shared topic space)

The albums are small (~400–540 lines each), so per-album HDBSCAN is unstable. We fit **one** BERTopic
model over all 1,866 lines using the precomputed OpenAI embeddings — a *shared* topic vocabulary in
which albums are directly comparable (`topic_modeling.py`, sklearn PCA+HDBSCAN backend because numba/UMAP
is broken under NumPy 2.4). Topics are auto-labelled with OpenAI `gpt-4o-mini`.

On a corpus this small ~57% of lines fall to HDBSCAN's outlier bucket (-1) — kept out of the per-album
profile rather than force-assigned. **Thematic concentration does not trend monotonically** (Sgt.
Pepper's is the most diverse, Abbey Road the most concentrated): the evolution does not live in the
topic mix. This is exactly why we move to geometry.""")

code("""info = pd.DataFrame(json.load(open(EXP/"topic_info.json")))
info["top_words"] = info["Representation"].apply(lambda x: ", ".join(x[:6]) if isinstance(x,list) else "")
print(info[info.Topic!=-1][["Topic","Count","label","top_words"]].head(14).to_string(index=False))
print("\\n=== per-album thematic concentration ===")
print(pd.read_csv(EXP/"album_topic_entropy.csv").to_string(index=False))""")

md("""### Per-album topic profile
![topic profile](outputs/figures/fig1_topic_profile.png)""")

md("""## 3 · Embeddings + vector database

Each song's full lyric and each line is embedded with OpenAI `text-embedding-3-large` (3072-dim),
cached to `.npy`, and the song vectors are indexed in a persistent **ChromaDB** collection
(`embeddings.py`). The vector DB powers nearest-neighbour retrieval used by the graph step.""")

code("""song_emb = np.load(EMB/"songs_openai_3large.npy")
line_emb = np.load(EMB/"lines_openai_3large.npy")
print(f"song embeddings: {song_emb.shape}   line embeddings: {line_emb.shape}")

import chromadb
client = chromadb.PersistentClient(path=str(ROOT/"data"/"chroma"))
coll = client.get_collection("beatles_songs")
q = coll.query(query_embeddings=[song_emb[0].tolist()], n_results=4)
print(f"\\nNearest neighbours of '{songs.iloc[0]['title']}' ({songs.iloc[0]['album']}):")
for mid, dist in zip(q["ids"][0], q["distances"][0]):
    m = coll.get(ids=[mid])["metadatas"][0]
    print(f"   {m['title']:<32} {m['album'][:18]:<20} cos_dist={dist:.3f}")""")

md("""**Already a clue:** *Drive My Car* (the opener of the earliest album) has its nearest neighbours in
*later* albums, not in Rubber Soul. Album membership is not what organizes the space.""")

md("""## 4 · Embedding-geometry evolution metrics

Far more robust than topic modeling on this tiny corpus. We compute, per album in chronological order:
intra-album cohesion, dispersion, nearest-neighbour album purity, centroid drift, lexical diversity
(TTR, hapax), within-song repetition, and the global album separability (silhouette). `evolution_metrics.py`.""")

code("""geo = pd.read_csv(EXP/"evolution_geometry.csv").set_index("album").reindex(ALBUM_ORDER).reset_index()
cols = ["short","year","intra_cohesion","dispersion","nn_purity_k1","ttr","hapax_ratio","adj_line_cos"]
print(geo[cols].to_string(index=False))
print("\\n--- centroid drift ---")
print(pd.read_csv(EXP/"centroid_drift.csv").to_string(index=False))
print("\\n--- correlation with release order (1..4) ---")
order = np.array([1,2,3,4])
for c in ["intra_cohesion","nn_purity_k1","ttr","hapax_ratio"]:
    print(f"   {c:>14}: r = {np.corrcoef(order, geo[c])[0,1]:+.3f}")
print(f"\\nalbum separability (silhouette, cosine): {json.load(open(EXP/'separability.json'))['silhouette_cosine']:+.4f}")""")

md("""### Lexical diversification — and its honest caveat
![lexical](outputs/figures/fig2_lexical_diversification.png)

### The semantic map — albums intermingle
![map](outputs/figures/fig3_semantic_map.png)

*Rubber Soul ↔ Sgt. Pepper's* are the most distant album centroids (0.861); *Revolver ↔ Abbey Road*
the closest (0.903). Chronology is **not** the dominant axis of the geometry.""")

md("""## 5 · Graph theory — the song-similarity network (centrepiece)

Nodes = 58 songs; edges = mutual-kNN semantic similarity. We detect **Louvain communities**, compare
their modularity to the album partition, measure alignment (ARI/NMI), and find **bridge songs** by
betweenness centrality (`graph_analysis.py`).""")

code("""g = json.load(open(EXP/"graph_summary.json"))
for k,v in g.items(): print(f"   {k:>28}: {v}")
print("\\n--- community composition (rows=community, cols=album) ---")
print(pd.read_csv(EXP/"community_album_composition.csv", index_col=0).to_string())
nodes = pd.read_csv(EXP/"graph_nodes.csv")
print("\\n--- top bridge songs (betweenness) ---")
print(nodes.sort_values("betweenness",ascending=False)
      [["title","album_short","betweenness","cross_album_ratio"]].head(6).to_string(index=False))""")

md("""![community](outputs/figures/fig4_community_composition.png)
![graph](outputs/figures/fig5_song_graph.png)

**Interactive version:** [`beatles_song_graph.html`](../2026-06-16-beatles-evolution/beatles_song_graph.html)
— node = song, size = bridge role (betweenness), colour = Louvain community.

The album partition has modularity **Q = 0.06**; the semantic communities reach **Q = 0.35**, but they
**cut across albums** (ARI = 0.09, NMI = 0.21) and 69% of edges are cross-album. The only structure that
emerges is a *soft* early (community 0: Rubber Soul + Revolver) vs late (community 2: Sgt. Pepper's +
Abbey Road) gradient.""")

md("""## 6 · Statistical robustness

Four album means cannot carry inferential weight, so we test the two load-bearing claims at the song
level (n=58) and via permutation (`stats_robustness.py`).""")

code("""s = json.load(open(EXP/"stats_robustness.json"))
print("C1  lexical diversification (song level, length-controlled OLS):")
print(f"     TTR   ~ order:  beta={s['C1_ttr_beta_order']:+.4f}  p={s['C1_ttr_p']:.3f}")
print(f"     hapax ~ order:  beta={s['C1_hapax_beta_order']:+.4f}  p={s['C1_hapax_p']:.3f}")
print(f"     => album-level trend is an AGGREGATION effect, not significant per song")
print("\\nC2  album separability / structure (permutation vs shuffled labels):")
print(f"     silhouette  obs={s['C2_silhouette_obs']:+.4f}  null={s['C2_silhouette_null_mean']:+.4f}  p={s['C2_silhouette_p']:.4f}")
print(f"     Q_album     obs={s['C2b_Q_album_obs']:.4f}  null={s['C2b_Q_album_null_mean']:+.4f}  p={s['C2b_Q_album_p']:.4f}")
print(f"     cross-edges obs={s['C2c_cross_frac_obs']:.4f}  null={s['C2c_cross_frac_null_mean']:.4f}  (below chance)")
print("\\n     => album identity is statistically REAL but practically NEGLIGIBLE")""")

md("""![permutation](outputs/figures/fig6_permutation.png)""")

md("""## 7 · Synthesis — the arc

**The Beatles 1965–69 do not evolve as a march between separable lyrical worlds.** In OpenAI-embedding
space the four albums are statistically detectable yet practically inseparable: album identity explains
almost none of the semantic geometry (silhouette ≈ 0, p=0.004; album-partition modularity Q=0.06, p=0.007),
against a far stronger community structure (Q=0.35) that **cuts across albums** (ARI=0.09).

What does change is **lexical** — pooled per album, vocabulary richness and hapax rate climb steeply
(r>0.93) — but at the song level (n=58) the trend dissolves (p>0.14): later albums *reuse fewer words
across their songs*, an aggregation-level diffusion rather than per-song enrichment.

So the catalogue is best read as **a single semantic register slowly diffusing**: a continuum with weak
album-eddies and strong cross-album bridges (*A Day in the Life*, *Got to Get You into My Life*), in which
**Sgt. Pepper's is the only album with genuine internal cohesion** (highest NN-purity — the concept-album
geometry) and **Abbey Road is the most internally fragmented** (the medley as a patchwork).

The evolution is real. It is just not where album covers tell us to look.""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
out = ROOT / "beatles_evolution.ipynb"
nbf.write(nb, str(out))
print(f"wrote {out} with {len(cells)} cells")
