"""
Step 2 — embeddings + vector database.

Computes OpenAI `text-embedding-3-large` (3072-dim) vectors at two granularities:
  * song-level  (full lyric text)   -> 58 vectors
  * line-level  (individual lines)  -> 1866 vectors

Caches everything to disk as .npy and indexes the song-level vectors in a
persistent ChromaDB collection used by the graph and retrieval steps.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
load_dotenv(REPO / ".env")

PROC = ROOT / "data" / "processed"
EMB = ROOT / "data" / "embeddings"
CHROMA = ROOT / "data" / "chroma"
EMB.mkdir(parents=True, exist_ok=True)

MODEL = "text-embedding-3-large"
DIM = 3072


def embed_texts(client: OpenAI, texts: list[str], batch: int = 64) -> np.ndarray:
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        chunk = [t.replace("\n", " ") for t in texts[i:i + batch]]
        resp = client.embeddings.create(input=chunk, model=MODEL, dimensions=DIM)
        out.extend([d.embedding for d in resp.data])
    return np.asarray(out, dtype=np.float32)


def main() -> None:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    df_songs = pd.read_parquet(PROC / "corpus_songs.parquet")
    df_lines = pd.read_parquet(PROC / "corpus_lines.parquet")

    # ---- song-level ----
    song_path = EMB / "songs_openai_3large.npy"
    if song_path.exists():
        song_emb = np.load(song_path)
        print(f"song embeddings cached: {song_emb.shape}")
    else:
        print(f"embedding {len(df_songs)} songs ...")
        song_emb = embed_texts(client, df_songs["full_text"].tolist())
        np.save(song_path, song_emb)
        print(f"song embeddings: {song_emb.shape}")

    # ---- line-level ----
    line_path = EMB / "lines_openai_3large.npy"
    if line_path.exists():
        line_emb = np.load(line_path)
        print(f"line embeddings cached: {line_emb.shape}")
    else:
        print(f"embedding {len(df_lines)} lines ...")
        line_emb = embed_texts(client, df_lines["line_text"].tolist())
        np.save(line_path, line_emb)
        print(f"line embeddings: {line_emb.shape}")

    # ---- vector database (ChromaDB, song-level) ----
    import chromadb
    chroma_client = chromadb.PersistentClient(path=str(CHROMA))
    try:
        chroma_client.delete_collection("beatles_songs")
    except Exception:
        pass
    coll = chroma_client.create_collection(
        "beatles_songs", metadata={"hnsw:space": "cosine"}
    )
    coll.add(
        ids=[f"{r.album_order}_{r.track_num}" for r in df_songs.itertuples()],
        embeddings=[song_emb[i].tolist() for i in range(len(df_songs))],
        metadatas=[{
            "album": r.album, "album_order": int(r.album_order), "year": int(r.year),
            "track_num": int(r.track_num), "title": r.title, "n_words": int(r.n_words),
        } for r in df_songs.itertuples()],
        documents=df_songs["full_text"].tolist(),
    )
    print(f"ChromaDB collection 'beatles_songs' indexed: {coll.count()} vectors at {CHROMA}")

    # sanity retrieval
    q = coll.query(query_embeddings=[song_emb[0].tolist()], n_results=4)
    base = df_songs.iloc[0]["title"]
    print(f"\nNearest neighbours of '{base}':")
    for mid, dist in zip(q["ids"][0], q["distances"][0]):
        meta = coll.get(ids=[mid])["metadatas"][0]
        print(f"   {meta['title']} ({meta['album']})  cos_dist={dist:.3f}")


if __name__ == "__main__":
    main()
