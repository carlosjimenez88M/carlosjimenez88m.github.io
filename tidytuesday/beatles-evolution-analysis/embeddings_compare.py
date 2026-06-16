"""
Cross-architecture comparison of the findings.

Goal: do the load-bearing claims (album inseparability, cross-cutting community
structure, the A Day in the Life / Got to Get You into My Life bridges) survive a
change of embedding model? We try Gemini first; if its key is unusable we fall
back to an open sentence-transformers model as the second architecture and label
it honestly.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
load_dotenv(REPO / ".env")
PROC, EMB, EXP = ROOT / "data" / "processed", ROOT / "data" / "embeddings", ROOT / "outputs" / "exports"
ALBUM_ORDER = ["Rubber Soul", "Revolver", "Sgt. Pepper's Lonely Hearts Club Band", "Abbey Road"]
SHORT = ["Rubber Soul", "Revolver", "Sgt. Pepper's", "Abbey Road"]
RNG = np.random.default_rng(42)


def l2(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def try_gemini_embeddings(texts: list[str]) -> np.ndarray | None:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API")
    if not key:
        print("Gemini: no key found"); return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        out = []
        for t in texts:
            r = genai.embed_content(model="models/text-embedding-004",
                                    content=t.replace("\n", " "),
                                    task_type="semantic_similarity")
            out.append(r["embedding"])
        print(f"Gemini: embedded {len(out)} texts")
        return np.asarray(out, dtype=np.float32)
    except Exception as e:
        print(f"Gemini embeddings FAILED: {str(e)[:90]}")
        return None


def open_model_embeddings(texts: list[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_name)
    return np.asarray(m.encode(texts, batch_size=32, normalize_embeddings=True,
                               show_progress_bar=False), dtype=np.float32)


def knn_graph(S, labels, k=5):
    n = len(S); Sd = S.copy(); np.fill_diagonal(Sd, -1)
    nn = np.argsort(-Sd, axis=1)[:, :k]
    G = nx.Graph(); G.add_nodes_from(range(n))
    for i in range(n):
        for j in nn[i]:
            G.add_edge(i, int(j), weight=float(S[i, int(j)]))
    return G


def analyse(emb, songs, label):
    emb = l2(emb)
    S = cosine_similarity(emb)
    y = songs["album"].map({a: i for i, a in enumerate(ALBUM_ORDER)}).to_numpy()

    sil = silhouette_score(emb, y, metric="cosine")
    null = np.array([silhouette_score(emb, RNG.permutation(y), metric="cosine") for _ in range(500)])
    p_sil = (np.sum(null >= sil) + 1) / (len(null) + 1)

    G = knn_graph(S, y, k=5)
    album_parts = [set(np.where(y == k)[0]) for k in range(4)]
    Q_alb = nx.community.modularity(G, album_parts, weight="weight")
    comms = nx.community.louvain_communities(G, weight="weight", seed=42)
    Q_com = nx.community.modularity(G, comms, weight="weight")
    comm_lab = np.zeros(len(songs), int)
    for ci, c in enumerate(comms):
        for nidx in c:
            comm_lab[nidx] = ci
    ari = adjusted_rand_score(y, comm_lab)
    nmi = normalized_mutual_info_score(y, comm_lab)
    cross = sum(1 for u, v in G.edges() if y[u] != y[v]) / G.number_of_edges()

    btw = nx.betweenness_centrality(G, weight=lambda u, v, d: 1 - d["weight"], normalized=True)
    bridges = sorted(range(len(songs)), key=lambda i: -btw[i])[:5]
    bridge_titles = [(songs.iloc[i]["title"], round(btw[i], 3)) for i in bridges]

    # centroid drift
    cents = {a: emb[np.where(y == i)[0]].mean(0) for i, a in enumerate(ALBUM_ORDER)}
    drift = []
    for a, b in zip(ALBUM_ORDER, ALBUM_ORDER[1:]):
        ca, cb = cents[a] / np.linalg.norm(cents[a]), cents[b] / np.linalg.norm(cents[b])
        drift.append(round(1 - float(ca @ cb), 4))

    return {"label": label, "dim": emb.shape[1], "silhouette": round(float(sil), 4),
            "silhouette_p": round(float(p_sil), 4), "Q_album": round(float(Q_alb), 4),
            "Q_louvain": round(float(Q_com), 4), "n_comm": len(comms),
            "ARI": round(float(ari), 4), "NMI": round(float(nmi), 4),
            "cross_album_frac": round(float(cross), 4), "drift": drift,
            "bridges": bridge_titles, "_S": S}


def main():
    songs = pd.read_parquet(PROC / "corpus_songs.parquet").reset_index(drop=True)
    texts = songs["full_text"].tolist()
    openai_emb = np.load(EMB / "songs_openai_3large.npy")

    # second architecture
    second = try_gemini_embeddings(texts)
    if second is not None:
        np.save(EMB / "songs_gemini_te004.npy", second)
        second_label = "Gemini text-embedding-004"
    else:
        model_name = "sentence-transformers/all-mpnet-base-v2"
        print(f"\nFalling back to open model: {model_name}")
        second = open_model_embeddings(texts, model_name)
        np.save(EMB / "songs_mpnet.npy", second)
        second_label = "all-mpnet-base-v2 (open, stand-in for Gemini)"

    r1 = analyse(openai_emb, songs, "OpenAI text-embedding-3-large")
    r2 = analyse(second, songs, second_label)

    # cross-model agreement on the song-song similarity geometry
    iu = np.triu_indices(len(songs), k=1)
    rho, _ = spearmanr(r1["_S"][iu], r2["_S"][iu])

    print("\n================ CROSS-ARCHITECTURE COMPARISON ================")
    keys = ["dim", "silhouette", "silhouette_p", "Q_album", "Q_louvain", "ARI",
            "NMI", "cross_album_frac"]
    print(f"{'metric':<18}{'OpenAI':>16}{'2nd model':>16}")
    for k in keys:
        print(f"{k:<18}{str(r1[k]):>16}{str(r2[k]):>16}")
    print(f"{'drift (steps)':<18}{str(r1['drift']):>16}{str(r2['drift']):>16}")
    print(f"\ncross-model similarity-geometry agreement (Spearman): rho = {rho:.3f}")
    print(f"\nOpenAI bridges : {r1['bridges']}")
    print(f"2nd    bridges : {r2['bridges']}")

    for r in (r1, r2):
        r.pop("_S")
    json.dump({"openai": r1, "second": r2, "geometry_spearman": round(float(rho), 4)},
              open(EXP / "embedding_comparison.json", "w"), indent=2)
    print(f"\nsaved -> {EXP/'embedding_comparison.json'}")


if __name__ == "__main__":
    main()
