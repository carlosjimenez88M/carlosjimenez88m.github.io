"""
Step 4 — embedding-geometry evolution metrics (song level, 58 songs).

These metrics are far more robust than topic modeling on this tiny corpus and
carry the rigorous part of the evolution arc. We let the thesis emerge from
several convergent signals rather than fixing it in advance.

Signals computed per album (chronological):
  * intra-album cohesion        mean pairwise cosine among an album's songs
  * dispersion                  mean distance of songs to their album centroid
  * NN-album purity             fraction of songs whose nearest neighbour is same-album
  * centroid drift              distance between consecutive album centroids
  * lexical diversity           type-token ratio, hapax ratio
  * within-song repetition      mean adjacent-line cosine (link to attention windows)
  * album separability          silhouette of album labels in embedding space
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
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


def main() -> None:
    songs = pd.read_parquet(PROC / "corpus_songs.parquet").reset_index(drop=True)
    lines = pd.read_parquet(PROC / "corpus_lines.parquet").reset_index(drop=True)
    song_emb = l2norm(np.load(EMB / "songs_openai_3large.npy"))
    line_emb = l2norm(np.load(EMB / "lines_openai_3large.npy"))

    songs["idx"] = np.arange(len(songs))
    S = cosine_similarity(song_emb)

    # ---- per-album geometry ----
    rows = []
    centroids = {}
    for alb in ALBUM_ORDER:
        idx = songs.index[songs.album == alb].to_numpy()
        sub = song_emb[idx]
        centroid = sub.mean(axis=0)
        centroids[alb] = centroid
        # intra-album cohesion (upper triangle of within-album sim)
        sim = cosine_similarity(sub)
        iu = np.triu_indices(len(idx), k=1)
        cohesion = float(sim[iu].mean())
        # dispersion: mean cosine distance to centroid
        c = centroid / np.linalg.norm(centroid)
        dispersion = float(1 - (sub @ c).mean())
        rows.append({"album": alb, "short": SHORT[alb],
                     "year": int(songs[songs.album == alb]["year"].iloc[0]),
                     "n_songs": len(idx),
                     "intra_cohesion": round(cohesion, 4),
                     "dispersion": round(dispersion, 4)})
    geo = pd.DataFrame(rows)

    # ---- nearest-neighbour album purity (k=1 and k=3) ----
    np.fill_diagonal(S, -1)
    nn1 = S.argmax(axis=1)
    same1 = (songs.loc[nn1, "album"].to_numpy() == songs["album"].to_numpy())
    songs["nn_same_album"] = same1
    purity_k1 = (songs.groupby("album")["nn_same_album"].mean()
                 .reindex(ALBUM_ORDER).round(4))
    # k=3 purity
    k = 3
    nn3 = np.argsort(-S, axis=1)[:, :k]
    purity3 = []
    for i in range(len(songs)):
        same = (songs.loc[nn3[i], "album"].to_numpy() == songs.loc[i, "album"]).mean()
        purity3.append(same)
    songs["nn3_same_share"] = purity3
    purity_k3 = (songs.groupby("album")["nn3_same_share"].mean()
                 .reindex(ALBUM_ORDER).round(4))
    geo["nn_purity_k1"] = geo["album"].map(purity_k1).values
    geo["nn_purity_k3"] = geo["album"].map(purity_k3).values

    # ---- centroid drift (consecutive albums) ----
    drift = [{"from": "—", "to": ALBUM_ORDER[0], "centroid_cos": np.nan, "drift": 0.0}]
    for a, b in zip(ALBUM_ORDER, ALBUM_ORDER[1:]):
        ca, cb = centroids[a] / np.linalg.norm(centroids[a]), centroids[b] / np.linalg.norm(centroids[b])
        cs = float(ca @ cb)
        drift.append({"from": SHORT[a], "to": SHORT[b],
                      "centroid_cos": round(cs, 4), "drift": round(1 - cs, 4)})
    drift_df = pd.DataFrame(drift)

    # all-pairs album centroid similarity matrix
    cmat = np.zeros((4, 4))
    for i, a in enumerate(ALBUM_ORDER):
        for j, b in enumerate(ALBUM_ORDER):
            ca, cb = centroids[a], centroids[b]
            cmat[i, j] = (ca @ cb) / (np.linalg.norm(ca) * np.linalg.norm(cb))
    pd.DataFrame(cmat, index=[SHORT[a] for a in ALBUM_ORDER],
                 columns=[SHORT[a] for a in ALBUM_ORDER]).round(4)\
      .to_csv(EXP / "album_centroid_sim.csv")

    # ---- lexical diversity per album ----
    import re
    lex = []
    for alb in ALBUM_ORDER:
        text = " ".join(songs[songs.album == alb]["full_text"]).lower()
        toks = re.findall(r"[a-z']+", text)
        types = set(toks)
        from collections import Counter
        cnt = Counter(toks)
        hapax = sum(1 for w, c in cnt.items() if c == 1)
        lex.append({"album": alb,
                    "tokens": len(toks), "types": len(types),
                    "ttr": round(len(types) / len(toks), 4),
                    "hapax_ratio": round(hapax / len(types), 4)})
    lex_df = pd.DataFrame(lex)
    geo = geo.merge(lex_df, on="album")

    # ---- within-song repetition: mean adjacent-line cosine ----
    rep_rows = []
    for alb in ALBUM_ORDER:
        sims = []
        for title, g in lines[lines.album == alb].groupby("title"):
            gi = g.sort_values("line_num").index.to_numpy()
            if len(gi) < 2:
                continue
            e = line_emb[gi]
            adj = (e[:-1] * e[1:]).sum(axis=1)
            sims.extend(adj.tolist())
        rep_rows.append({"album": alb, "adj_line_cos": round(float(np.mean(sims)), 4)})
    rep_df = pd.DataFrame(rep_rows)
    geo = geo.merge(rep_df, on="album")

    # ---- album separability (silhouette over all songs) ----
    labels = songs["album"].map({a: i for i, a in enumerate(ALBUM_ORDER)}).to_numpy()
    sil = float(silhouette_score(song_emb, labels, metric="cosine"))

    geo = geo.set_index("album").reindex(ALBUM_ORDER).reset_index()
    print("=== PER-ALBUM EMBEDDING GEOMETRY (chronological) ===")
    print(geo.to_string(index=False))
    print("\n=== CENTROID DRIFT ===")
    print(drift_df.to_string(index=False))
    print(f"\nAlbum separability (silhouette, cosine): {sil:.4f}")
    print("\n=== CORRELATIONS WITH TIME (album_order 1..4) ===")
    order = np.array([1, 2, 3, 4])
    for col in ["intra_cohesion", "dispersion", "nn_purity_k1", "nn_purity_k3",
                "ttr", "hapax_ratio", "adj_line_cos"]:
        r = np.corrcoef(order, geo[col].to_numpy())[0, 1]
        print(f"   {col:>16}: Pearson r vs year = {r:+.3f}")

    geo.to_csv(EXP / "evolution_geometry.csv", index=False)
    drift_df.to_csv(EXP / "centroid_drift.csv", index=False)
    songs[["album", "title", "nn_same_album", "nn3_same_share"]].to_csv(
        EXP / "song_nn_purity.csv", index=False)
    json.dump({"silhouette_cosine": sil}, open(EXP / "separability.json", "w"), indent=2)
    print(f"\nsaved to {EXP}")


if __name__ == "__main__":
    main()
