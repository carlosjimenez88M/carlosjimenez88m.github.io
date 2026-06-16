"""
Step 6 — statistical robustness.

Four-album means (n=4) cannot carry inferential weight on their own, so we test
the two load-bearing claims at the song level (n=58) and via permutation:

  C1  Lexical diversification is monotonic in release order.
        -> song-level OLS: hapax/TTR ~ release_order, controlling for song length.
  C2  Albums are NOT separable as semantic clusters; community structure exists
      but cuts across album boundaries.
        -> permutation tests on silhouette, album-partition modularity, and
           cross-album edge fraction against shuffled album labels.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
EMB = ROOT / "data" / "embeddings"
EXP = ROOT / "outputs" / "exports"

ALBUM_ORDER = ["Rubber Soul", "Revolver",
               "Sgt. Pepper's Lonely Hearts Club Band", "Abbey Road"]
RNG = np.random.default_rng(42)


def l2norm(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def song_lexical(text: str) -> tuple[int, int, float, float]:
    toks = re.findall(r"[a-z']+", text.lower())
    if not toks:
        return 0, 0, 0.0, 0.0
    cnt = Counter(toks)
    types = len(cnt)
    hapax = sum(1 for w, c in cnt.items() if c == 1)
    return len(toks), types, types / len(toks), hapax / types


def main() -> None:
    songs = pd.read_parquet(PROC / "corpus_songs.parquet").reset_index(drop=True)
    song_emb = l2norm(np.load(EMB / "songs_openai_3large.npy"))

    # ---- C1: song-level lexical diversification ----
    lex = songs["full_text"].apply(song_lexical)
    songs["tokens"] = [x[0] for x in lex]
    songs["types"] = [x[1] for x in lex]
    songs["ttr"] = [x[2] for x in lex]
    songs["hapax_ratio"] = [x[3] for x in lex]
    songs["order"] = songs["album_order"].astype(float)

    print("=== C1 — LEXICAL DIVERSIFICATION (song level, n=58) ===")
    # TTR is length-biased, so control for log tokens.
    songs["log_tokens"] = np.log(songs["tokens"])
    m_ttr = smf.ols("ttr ~ order + log_tokens", data=songs).fit()
    m_hap = smf.ols("hapax_ratio ~ order + log_tokens", data=songs).fit()
    for name, m in [("TTR", m_ttr), ("hapax_ratio", m_hap)]:
        b = m.params["order"]; p = m.pvalues["order"]; ci = m.conf_int().loc["order"]
        print(f"   {name:>12} ~ order(+length ctrl):  beta_order={b:+.4f}  "
              f"95%CI=[{ci[0]:+.4f},{ci[1]:+.4f}]  p={p:.4g}  R2={m.rsquared:.3f}")
    # Spearman of song-level value vs order (monotonicity)
    from scipy.stats import spearmanr
    for col in ["ttr", "hapax_ratio"]:
        rho, p = spearmanr(songs["order"], songs[col])
        print(f"   Spearman {col} vs order: rho={rho:+.3f}, p={p:.4g}")

    # ---- C2: album separability permutation ----
    print("\n=== C2 — ALBUM SEPARABILITY (permutation, n=58) ===")
    labels = songs["album"].map({a: i for i, a in enumerate(ALBUM_ORDER)}).to_numpy()
    obs_sil = silhouette_score(song_emb, labels, metric="cosine")
    null_sil = np.array([silhouette_score(song_emb, RNG.permutation(labels), metric="cosine")
                         for _ in range(1000)])
    p_sil = (np.sum(null_sil >= obs_sil) + 1) / (len(null_sil) + 1)
    print(f"   observed silhouette = {obs_sil:+.4f}")
    print(f"   null silhouette     = {null_sil.mean():+.4f} ± {null_sil.std():.4f}")
    print(f"   one-sided p(obs > null) = {p_sil:.4f}  "
          f"(z = {(obs_sil - null_sil.mean())/null_sil.std():+.2f})")

    # ---- C2b: album-partition modularity permutation on fixed graph ----
    S = cosine_similarity(song_emb)
    Sd = S.copy(); np.fill_diagonal(Sd, -1)
    nn = np.argsort(-Sd, axis=1)[:, :5]
    G = nx.Graph()
    G.add_nodes_from(range(len(songs)))
    for i in range(len(songs)):
        for j in nn[i]:
            G.add_edge(i, int(j), weight=float(S[i, int(j)]))

    def album_modularity(lab: np.ndarray) -> float:
        parts = [set(np.where(lab == k)[0]) for k in range(4)]
        return nx.community.modularity(G, parts, weight="weight")

    obs_Q = album_modularity(labels)
    null_Q = np.array([album_modularity(RNG.permutation(labels)) for _ in range(1000)])
    p_Q = (np.sum(null_Q >= obs_Q) + 1) / (len(null_Q) + 1)
    print("\n=== C2b — ALBUM-PARTITION MODULARITY (permutation) ===")
    print(f"   observed Q_album = {obs_Q:.4f}")
    print(f"   null Q          = {null_Q.mean():.4f} ± {null_Q.std():.4f}")
    print(f"   one-sided p(obs > null) = {p_Q:.4f}  "
          f"(z = {(obs_Q - null_Q.mean())/null_Q.std():+.2f})")

    # ---- C2c: cross-album edge fraction vs expected ----
    cross = sum(1 for u, v in G.edges() if labels[u] != labels[v])
    frac = cross / G.number_of_edges()
    null_frac = []
    for _ in range(1000):
        lab = RNG.permutation(labels)
        c = sum(1 for u, v in G.edges() if lab[u] != lab[v])
        null_frac.append(c / G.number_of_edges())
    null_frac = np.array(null_frac)
    print("\n=== C2c — CROSS-ALBUM EDGE FRACTION ===")
    print(f"   observed = {frac:.4f}   null(random labels) = {null_frac.mean():.4f} ± {null_frac.std():.4f}")
    print(f"   => observed is {'BELOW' if frac < null_frac.mean() else 'ABOVE'} chance "
          f"(z = {(frac - null_frac.mean())/null_frac.std():+.2f})")

    json.dump({
        "C1_ttr_beta_order": float(m_ttr.params["order"]),
        "C1_ttr_p": float(m_ttr.pvalues["order"]),
        "C1_hapax_beta_order": float(m_hap.params["order"]),
        "C1_hapax_p": float(m_hap.pvalues["order"]),
        "C2_silhouette_obs": float(obs_sil), "C2_silhouette_null_mean": float(null_sil.mean()),
        "C2_silhouette_p": float(p_sil),
        "C2b_Q_album_obs": float(obs_Q), "C2b_Q_album_null_mean": float(null_Q.mean()),
        "C2b_Q_album_p": float(p_Q),
        "C2c_cross_frac_obs": float(frac), "C2c_cross_frac_null_mean": float(null_frac.mean()),
    }, open(EXP / "stats_robustness.json", "w"), indent=2)
    songs.to_csv(EXP / "song_lexical.csv", index=False)
    print(f"\nsaved to {EXP}")


if __name__ == "__main__":
    main()
