"""
Step 3 — topic modeling with BERTopic (shared topic space).

The four albums are small (~400-540 lines each), so per-album HDBSCAN is
unstable. Instead we fit ONE BERTopic model over all 1866 lines using the
precomputed OpenAI embeddings, giving a *shared* topic vocabulary in which the
albums are directly comparable. We then read each album's distribution over the
shared topics and quantify its thematic concentration via normalized entropy.

Topics are auto-labelled with Gemini (LLM) from their top words + exemplar lines.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
load_dotenv(REPO / ".env")

PROC = ROOT / "data" / "processed"
EMB = ROOT / "data" / "embeddings"
EXP = ROOT / "outputs" / "exports"
EXP.mkdir(parents=True, exist_ok=True)

ALBUM_ORDER = ["Rubber Soul", "Revolver",
               "Sgt. Pepper's Lonely Hearts Club Band", "Abbey Road"]


def _build_prompt(words: list[str], exemplars: list[str]) -> str:
    return (
        "You are analysing Beatles song lyrics. Given the top keywords and a few "
        "representative lines of a topic cluster, return a SHORT (2-4 word) thematic "
        "label in English. Reply with ONLY the label, no punctuation.\n\n"
        f"Keywords: {', '.join(words[:10])}\n"
        f"Lines:\n- " + "\n- ".join(exemplars)
    )


def _gemini_labeler():
    """Return a callable(prompt)->label using Gemini, or None if the key is unusable.

    Reads GEMINI_API_KEY or GEMINI_API. Validates with a tiny probe call so we
    fail fast and fall back to OpenAI instead of erroring on every topic.
    """
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API")
    if not key:
        print("   .. no Gemini key found (GEMINI_API_KEY / GEMINI_API)")
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        model.generate_content("ping")  # probe; raises on invalid key
        print("   .. Gemini key valid — labelling topics with gemini-2.0-flash")
        return lambda p: (model.generate_content(p).text or "").strip()
    except Exception as e:
        print(f"   !! Gemini key unusable ({str(e)[:60]}...) — falling back to OpenAI")
        return None


def label_topics_with_llm(topic_info: pd.DataFrame, rep_docs: dict[int, list[str]]) -> dict[int, str]:
    """Label each topic with Gemini if its key works, else OpenAI gpt-4o-mini."""
    gemini = _gemini_labeler()
    backend = "gemini"
    if gemini is None:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        backend = "openai"

        def call(p: str) -> str:
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": p}],
                temperature=0.0, max_tokens=16,
            )
            return (r.choices[0].message.content or "").strip()
    else:
        call = gemini
    print(f"   .. topic-label backend: {backend}")

    labels: dict[int, str] = {}
    for row in topic_info.itertuples():
        tid = row.Topic
        if tid == -1:
            labels[tid] = "outliers / noise"
            continue
        words = row.Representation if isinstance(row.Representation, list) else []
        prompt = _build_prompt(words, rep_docs.get(tid, [])[:6])
        try:
            labels[tid] = call(prompt).strip('".').splitlines()[0][:40]
        except Exception as e:
            labels[tid] = f"topic {tid}"
            print(f"   !! label failed for {tid}: {str(e)[:60]}")
    return labels


def norm_entropy(counts: np.ndarray) -> float:
    """Shannon entropy of a distribution, normalized to [0,1] over its support."""
    p = counts[counts > 0] / counts.sum()
    if len(p) <= 1:
        return 0.0
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def main() -> None:
    from bertopic import BERTopic
    from sklearn.cluster import HDBSCAN
    from sklearn.decomposition import PCA
    from sklearn.feature_extraction.text import CountVectorizer

    df = pd.read_parquet(PROC / "corpus_lines.parquet").reset_index(drop=True)
    line_emb = np.load(EMB / "lines_openai_3large.npy")
    docs = df["line_text"].tolist()
    assert len(docs) == len(line_emb)

    # NumPy 2.4 in this env is incompatible with numba (and thus UMAP), so we use
    # sklearn-native PCA + HDBSCAN as BERTopic's reduction/clustering backends.
    dimred_model = PCA(n_components=10, random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=12, min_samples=4, metric="euclidean")
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=3)

    topic_model = BERTopic(
        umap_model=dimred_model, hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer, calculate_probabilities=False, verbose=False,
    )
    topics, _ = topic_model.fit_transform(docs, embeddings=line_emb)
    # Keep HDBSCAN outliers as topic -1 (excluded from per-album profiles): on a
    # corpus this small, forcing every line into a topic just inflates one generic
    # bucket and washes out the per-album signal.
    df["topic"] = topics

    info = topic_model.get_topic_info()
    print(info[["Topic", "Count", "Name"]].to_string(index=False))

    # representative docs per topic
    rep_docs: dict[int, list[str]] = {}
    for tid in info["Topic"]:
        sub = df[df["topic"] == tid]["line_text"].tolist()
        rep_docs[int(tid)] = sub[:8]

    labels = label_topics_with_llm(info, rep_docs)
    info["label"] = info["Topic"].map(labels)
    df["topic_label"] = df["topic"].map(labels)
    print("\nLLM labels:")
    for _, r in info.iterrows():
        print(f"   [{r['Topic']:>2}] n={r['Count']:>4}  {r['label']}")

    # ---- per-album topic profile + entropy ----
    valid = df[df["topic"] != -1]
    profile = (pd.crosstab(valid["album"], valid["topic"])
               .reindex(ALBUM_ORDER).fillna(0).astype(int))
    profile_norm = profile.div(profile.sum(axis=1), axis=0)

    entropy_rows = []
    for alb in ALBUM_ORDER:
        counts = profile.loc[alb].values
        entropy_rows.append({
            "album": alb,
            "year": int(valid[valid.album == alb]["year"].iloc[0]),
            "n_lines": int(counts.sum()),
            "n_topics_present": int((counts > 0).sum()),
            "topic_entropy_norm": round(norm_entropy(counts), 4),
            "top_topic": int(np.argmax(counts)),
            "top_topic_share": round(counts.max() / counts.sum(), 4),
        })
    entropy_df = pd.DataFrame(entropy_rows)
    print("\n=== PER-ALBUM THEMATIC CONCENTRATION ===")
    print(entropy_df.to_string(index=False))

    # ---- persist ----
    df.to_parquet(PROC / "corpus_lines_topics.parquet", index=False)
    profile.to_csv(EXP / "album_topic_counts.csv")
    profile_norm.to_csv(EXP / "album_topic_profile.csv")
    entropy_df.to_csv(EXP / "album_topic_entropy.csv", index=False)
    info_out = info.copy()
    info_out["Representation"] = info_out["Representation"].apply(
        lambda x: x if isinstance(x, list) else [])
    info_out[["Topic", "Count", "label", "Representation"]].to_json(
        EXP / "topic_info.json", orient="records", indent=2, force_ascii=False)
    topic_model.save(str(ROOT / "data" / "bertopic_model"),
                     serialization="safetensors", save_ctfidf=True)
    print(f"\nsaved topic artifacts to {EXP}")


if __name__ == "__main__":
    main()
