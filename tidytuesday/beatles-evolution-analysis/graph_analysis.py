"""
Step 5 — graph theory on the song-similarity network (the centrepiece).

Nodes = 58 songs. Edges = semantic similarity above a calibrated threshold.
We detect communities (Louvain), measure how well they align with album
boundaries (ARI/NMI/modularity), identify bridge songs (betweenness), and export
the layout for an interactive force-directed Plotly graph.

Central falsifiable question (emergent from evolution_metrics): do semantic
communities respect album boundaries, or cut across them?
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
EMB = ROOT / "data" / "embeddings"
EXP = ROOT / "outputs" / "exports"
EXP.mkdir(parents=True, exist_ok=True)

ALBUM_ORDER = ["Rubber Soul", "Revolver",
               "Sgt. Pepper's Lonely Hearts Club Band", "Abbey Road"]
SHORT = {"Rubber Soul": "Rubber Soul", "Revolver": "Revolver",
         "Sgt. Pepper's Lonely Hearts Club Band": "Sgt. Pepper's", "Abbey Road": "Abbey Road"}


def l2norm(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def build_knn_graph(S: np.ndarray, songs: pd.DataFrame, k: int = 5) -> nx.Graph:
    """Mutual-kNN graph: edge i-j kept if either is in the other's top-k, weight=sim."""
    n = len(S)
    Sd = S.copy()
    np.fill_diagonal(Sd, -1)
    nn = np.argsort(-Sd, axis=1)[:, :k]
    G = nx.Graph()
    for i in range(n):
        r = songs.iloc[i]
        G.add_node(i, title=r.title, album=r.album, album_short=SHORT[r.album],
                   album_order=int(r.album_order), year=int(r.year))
    for i in range(n):
        for j in nn[i]:
            w = float(S[i, j])
            if G.has_edge(i, j):
                continue
            G.add_edge(i, int(j), weight=w)
    return G


def main() -> None:
    songs = pd.read_parquet(PROC / "corpus_songs.parquet").reset_index(drop=True)
    song_emb = l2norm(np.load(EMB / "songs_openai_3large.npy"))
    S = cosine_similarity(song_emb)

    G = build_knn_graph(S, songs, k=5)
    print(f"graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"density={nx.density(G):.3f}, components={nx.number_connected_components(G)}")

    # ---- community detection (Louvain) ----
    comms = nx.community.louvain_communities(G, weight="weight", seed=42, resolution=1.0)
    comm_of = {n: ci for ci, c in enumerate(comms) for n in c}
    nx.set_node_attributes(G, comm_of, "community")
    n_comm = len(comms)
    Q_comm = nx.community.modularity(G, comms, weight="weight")
    print(f"\nLouvain: {n_comm} communities, modularity Q = {Q_comm:.4f}")

    # ---- album partition modularity (for comparison) ----
    album_parts = [set(songs.index[songs.album == a]) for a in ALBUM_ORDER]
    Q_album = nx.community.modularity(G, album_parts, weight="weight")
    print(f"Album partition modularity Q = {Q_album:.4f}")

    # ---- alignment of communities with albums ----
    album_lab = songs["album"].map({a: i for i, a in enumerate(ALBUM_ORDER)}).to_numpy()
    comm_lab = np.array([comm_of[i] for i in range(len(songs))])
    ari = adjusted_rand_score(album_lab, comm_lab)
    nmi = normalized_mutual_info_score(album_lab, comm_lab)
    print(f"Community vs album:  ARI = {ari:.4f}   NMI = {nmi:.4f}")

    # community composition by album
    comp = pd.crosstab(comm_lab, songs["album"]).reindex(columns=ALBUM_ORDER).fillna(0).astype(int)
    print("\nCommunity composition (rows=community, cols=album):")
    print(comp.to_string())

    # ---- centrality / bridge songs ----
    btw = nx.betweenness_centrality(G, weight=lambda u, v, d: 1 - d["weight"], normalized=True)
    deg = dict(G.degree(weight="weight"))
    songs["betweenness"] = [btw[i] for i in range(len(songs))]
    songs["w_degree"] = [deg[i] for i in range(len(songs))]
    songs["community"] = comm_lab
    # cross-album edge ratio per node = fraction of a node's edges going to other albums
    cross = []
    for i in range(len(songs)):
        nb = list(G.neighbors(i))
        if not nb:
            cross.append(0.0); continue
        diff = sum(1 for j in nb if songs.iloc[j].album != songs.iloc[i].album)
        cross.append(diff / len(nb))
    songs["cross_album_ratio"] = cross

    print("\nTop bridge songs (betweenness):")
    print(songs.sort_values("betweenness", ascending=False)
          [["title", "album", "betweenness", "cross_album_ratio"]].head(8).to_string(index=False))

    # ---- global cross-album edge fraction ----
    cross_edges = sum(1 for u, v in G.edges() if songs.iloc[u].album != songs.iloc[v].album)
    frac_cross = cross_edges / G.number_of_edges()
    print(f"\nCross-album edges: {cross_edges}/{G.number_of_edges()} = {frac_cross:.3f}")

    # ---- layout for plotting (spring, deterministic) ----
    pos = nx.spring_layout(G, weight="weight", seed=42, k=0.55, iterations=300)
    node_rows = []
    for i in range(len(songs)):
        r = songs.iloc[i]
        node_rows.append({
            "node": i, "title": r.title, "album": r.album, "album_short": SHORT[r.album],
            "album_order": int(r.album_order), "year": int(r.year),
            "community": int(comm_lab[i]), "betweenness": round(float(btw[i]), 4),
            "w_degree": round(float(deg[i]), 4),
            "cross_album_ratio": round(float(cross[i]), 4),
            "x": round(float(pos[i][0]), 5), "y": round(float(pos[i][1]), 5),
        })
    nodes_df = pd.DataFrame(node_rows)
    edge_rows = [{"src": int(u), "dst": int(v), "weight": round(float(d["weight"]), 4),
                  "cross_album": bool(songs.iloc[u].album != songs.iloc[v].album)}
                 for u, v, d in G.edges(data=True)]
    edges_df = pd.DataFrame(edge_rows)

    nodes_df.to_csv(EXP / "graph_nodes.csv", index=False)
    edges_df.to_csv(EXP / "graph_edges.csv", index=False)
    comp.to_csv(EXP / "community_album_composition.csv")
    nx.write_gexf(G, str(EXP / "beatles_song_graph.gexf"))
    json.dump({
        "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(),
        "density": round(nx.density(G), 4), "n_communities": int(n_comm),
        "modularity_communities": round(Q_comm, 4), "modularity_album": round(Q_album, 4),
        "community_vs_album_ARI": round(ari, 4), "community_vs_album_NMI": round(nmi, 4),
        "cross_album_edge_fraction": round(frac_cross, 4),
    }, open(EXP / "graph_summary.json", "w"), indent=2)
    print(f"\nsaved graph artifacts to {EXP}")


if __name__ == "__main__":
    main()
