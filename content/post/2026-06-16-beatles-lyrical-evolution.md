---
author: Carlos Daniel Jiménez
date: 2026-06-16
title: "One Register, Slowly Diffusing: A Topic-Model, Vector-Database and Graph-Theoretic Reading of the Beatles' Lyrical Evolution (1965–1969)"
categories: ["Music Analysis", "LLMs"]
tags: ["llms", "nlp", "music-analysis", "embeddings", "bertopic", "topic-modeling", "vector-database", "graph-theory", "computational-musicology"]
series:
  - NLP
  - LLMs
  - Embeddings
  - Computational Musicology
aliases:
  - /tidytuesday/2026-06-16-beatles-evolution/
---

## Abstract

This study traces the lyrical evolution of the Beatles across four albums — **Rubber Soul (1965)**, **Revolver (1966)**, **Sgt. Pepper's Lonely Hearts Club Band (1967)** and **Abbey Road (1969)** — using a pipeline of LLM-driven topic modeling (BERTopic), a vector database (OpenAI `text-embedding-3-large` indexed in ChromaDB), embedding geometry, and graph theory. Rather than fix a thesis in advance, we let it emerge from the data.

**Core Finding (data-driven, falsifiable):** The four albums are **statistically detectable but practically inseparable** in lyrical-semantic space. A silhouette of album labels over song embeddings is ≈ 0 (−0.011) yet significantly above a shuffled null (null μ = −0.024, p = 0.004); the modularity of the album partition on the song-similarity graph is Q = 0.060 — again above chance (null μ = −0.017, p = 0.007) but near-zero in absolute terms. Against this, **Louvain community detection finds a real cross-cutting structure at Q = 0.348** that does **not** respect album boundaries (community-vs-album ARI = 0.090, NMI = 0.208; 69% of edges connect songs from different albums). The only monotonic signal is **lexical**: pooled per album, type-token ratio rises 0.156 → 0.236 (r = +0.93 vs release order) and the hapax ratio 0.364 → 0.538 (r = +0.95). But this is an **aggregation effect** — at the song level (n = 58, length-controlled OLS) the trend dissolves (β_order p = 0.15 for hapax, p = 0.88 for TTR). The evolution is real, but it is a **single semantic register slowly diffusing**, not a sequence of separable thematic eras.

---

## TL;DR

- **Topic modeling per album** (BERTopic over a shared topic space, OpenAI-embedding backend, `gpt-4o-mini` labels) is the descriptive entry point — but **thematic concentration does not trend monotonically** (Sgt. Pepper's is the *most* topically diverse, Abbey Road the *most* concentrated). The evolution does not live in the topic mix.
- **Vector database + embedding geometry**: songs embedded with `text-embedding-3-large` (3072-dim), indexed in ChromaDB. *Drive My Car* — the opener of the earliest album — has its nearest neighbours in *later* albums. Album membership is not what organizes the space.
- **Albums are detectable but not separable**: silhouette ≈ 0 (p = 0.004), album-partition graph modularity Q = 0.06 (p = 0.007). Real, but negligible.
- **Graph theory (the centrepiece)**: a song-similarity network with Louvain communities at Q = 0.35 that **cut across albums** (ARI = 0.09). The structure that exists is a *soft* early (Rubber Soul + Revolver) vs late (Sgt. Pepper's + Abbey Road) gradient, with **bridge songs** — *A Day in the Life* and *Got to Get You into My Life* — holding the catalogue together by betweenness centrality.
- **Lexical diversification is real per album (r > 0.93) but not per song (p > 0.14)** — later albums *reuse fewer words across their songs*; diffusion, not enrichment.
- **The arc**: Sgt. Pepper's is the only album with genuine internal cohesion (concept-album geometry); Abbey Road is the most internally fragmented (the medley as a patchwork).

---

## What This Post Does

This analysis does four things. **First**, it starts where the brief asked — *topic modeling per album* with **BERTopic** — and shows honestly why, on a corpus this small, topic modeling is a descriptive entry point rather than the load-bearing method. **Second**, it embeds every song and line with OpenAI's `text-embedding-3-large`, indexes them in a **vector database** (ChromaDB), and measures the *geometry* of the evolution. **Third**, it builds the **song-similarity graph** and uses graph theory — Louvain communities, modularity, betweenness — to ask whether semantic structure respects album boundaries (it does not). **Fourth**, it submits the two load-bearing claims to **statistical scrutiny**: song-level regressions and permutation tests, because four album means cannot carry inference on their own.

Throughout, the thesis is allowed to **emerge** from the measurements. The result is more nuanced — and more honest — than a tidy "from simple love songs to psychedelic abstraction" narrative would have been.

---

## Why This Matters

The received story of the Beatles is one of dramatic evolution: the folk-rock introspection of *Rubber Soul*, the studio experimentation of *Revolver*, the psychedelic concept-album peak of *Sgt. Pepper's*, the mature synthesis of *Abbey Road*. That story is true at the level of production, arrangement and cultural reception. The question this post asks is narrower and testable: **does that evolution show up in the lyrics, as semantic structure that separates the albums?**

This matters beyond Beatles fandom. It is a clean test of what distributional embeddings *can* and *cannot* see in a famously evolving body of text — the same epistemic question that runs through this series ([Beatles vs. Pink Floyd](/post/2026-02-10-attention-windows-beatles-floyd); [Aquamosh](/post/2026-05-20-aquamosh-quadrilingual-anatomy)). If even the Beatles' canonical "evolution" is, at the lyrical-semantic level, a slow diffusion within one register rather than a march between separable worlds, that is a substantive finding about both the music and the measuring instrument.

---

## A Note on Method: Letting the Thesis Emerge

We did not pre-register a hypothesis. We ran topic modeling and embedding geometry first, looked at what the data actually showed, and only then fixed the strongest falsifiable claim. This is riskier than confirming a tidy intuition — the data could have surprised us, and in one respect it did: the lexical-diversification signal that looks dramatic at the album level **evaporates** when tested at the song level. We report that rather than hide it.

---

## Methodology

### Data Collection

**Albums (chronological — the spine of the arc):**

| Album | Year | Songs | Lines | Words |
|---|---|---|---|---|
| Rubber Soul | 1965 | 14 | 464 | 2,922 |
| Revolver | 1966 | 14 | 455 | 2,399 |
| Sgt. Pepper's Lonely Hearts Club Band | 1967 | 13 | 544 | 3,094 |
| Abbey Road | 1969 | 17 | 403 | 2,255 |
| **Total** | | **58** | **1,866** | **10,670** |

**Source:** Genius API via the `lyricsgenius` library. Album-level search is unreliable for the Beatles — it merges compilations, singles, and foreign-language versions (`Sgt. Pepper's` came back as a 216-"track" collection mixing in Abbey Road songs, *Hey Jude*, and *Komm, gib mir deine Hand*). We therefore fetch **song by song against canonical UK tracklists**, which is fully reproducible. Lyrics are cleaned of section markers (`[Chorus]`) and CMS noise (`NNNEmbed`, "Contributors").

### Embeddings + Vector Database

Each song's full lyric and each individual line is embedded with OpenAI **`text-embedding-3-large`** (3072-dim), cached to disk, and the 58 song vectors are indexed in a persistent **ChromaDB** collection with cosine space.

```python
def embed_texts(client, texts, batch=64):
    out = []
    for i in range(0, len(texts), batch):
        chunk = [t.replace("\n", " ") for t in texts[i:i+batch]]
        resp = client.embeddings.create(input=chunk, model="text-embedding-3-large", dimensions=3072)
        out.extend([d.embedding for d in resp.data])
    return np.asarray(out, dtype=np.float32)

coll = chromadb.PersistentClient(path=CHROMA).create_collection(
    "beatles_songs", metadata={"hnsw:space": "cosine"})
coll.add(ids=ids, embeddings=song_emb.tolist(), metadatas=metas, documents=texts)
```

A first clue arrives immediately from the vector DB. Querying the nearest neighbours of *Drive My Car* — the opener of the **earliest** album — returns songs from **later** albums:

```
Nearest neighbours of 'Drive My Car' (Rubber Soul):
   And Your Bird Can Sing      Revolver        cos_dist=0.421
   Got to Get You into My Life Revolver        cos_dist=0.438
   A Day in the Life           Sgt. Pepper's   cos_dist=0.449
```

Album membership is not what organizes the space. The rest of the post quantifies that.

> **Engineering note.** Under this environment's NumPy 2.4, `numba` (and therefore UMAP) is unavailable, so BERTopic runs on a scikit-learn **PCA + HDBSCAN** backend and all 2-D projections use scikit-learn t-SNE/PCA. All LLM work uses OpenAI (`gpt-4o-mini`); the Gemini key was non-functional.

---

## 1 · Topic Modeling per Album (BERTopic) — The Descriptive Entry Point

The albums are small (~400–540 lines), so per-album HDBSCAN is unstable. We fit **one** BERTopic model over all 1,866 lines using the precomputed OpenAI embeddings — a **shared topic vocabulary** in which the albums are directly comparable — then read each album's distribution over the shared topics. Topics are auto-labelled by `gpt-4o-mini` from their top words and exemplar lines.

The resulting topic landscape is legible: *Ambition and Heartbreak*, *Sgt. Pepper's Band*, *dreamlike imagery* (Lucy in the Sky), *yellow submarine*, *Doctor theme*, *violence and desperation* (Maxwell's Silver Hammer / Run for Your Life), *Morning greetings*, *unity and togetherness* (Come Together). On a corpus this small, ~57% of lines fall to HDBSCAN's outlier bucket and are kept *out* of the per-album profile rather than force-assigned into a single generic mega-topic.

![Topic modeling per album](/tidytuesday/2026-06-16-beatles-evolution/fig1_topic_profile.png)

**Per-album thematic concentration (normalized topic entropy over clustered lines):**

| Album | Topics present | Top-topic share | Topic entropy (0–1) |
|---|---|---|---|
| Rubber Soul | 14 | 0.346 | 0.795 |
| Revolver | 13 | 0.420 | 0.766 |
| **Sgt. Pepper's** | 17 | 0.330 | **0.822** (most diverse) |
| **Abbey Road** | 19 | 0.429 | **0.719** (most concentrated) |

**Key finding (NEGATIVE, and instructive):** thematic concentration **does not trend monotonically** with time. If the "increasing complexity" intuition were a topic-distribution fact, entropy would climb toward Abbey Road. Instead Sgt. Pepper's is the most topically *diverse* and Abbey Road the most *concentrated* — the latter because the Side-B medley repeats a handful of motifs (*"carry that weight"*, *"love you"*). **Topic modeling tells us the evolution does not live in the topic mix.** This is exactly the limitation flagged in the [Aquamosh post](/post/2026-05-20-aquamosh-quadrilingual-anatomy): topic models need large corpora; 13–19 songs cannot yield a stable thematic trajectory. So we move to geometry.

---

## 2 · Embedding Geometry — Where the Evolution Actually Lives

Far more robust than topic modeling on this corpus. Per album, in chronological order, we measure intra-album cohesion (mean pairwise cosine among its songs), dispersion (distance to centroid), nearest-neighbour album purity, centroid drift, lexical diversity, and within-song repetition.

| Album | Intra-cohesion | NN-purity (k=1) | TTR | Hapax ratio | Adj-line cos |
|---|---|---|---|---|---|
| Rubber Soul | 0.459 | 0.500 | 0.156 | 0.364 | 0.359 |
| Revolver | 0.450 | 0.214 | 0.208 | 0.411 | 0.354 |
| Sgt. Pepper's | **0.481** | **0.615** | 0.231 | 0.531 | 0.335 |
| Abbey Road | **0.399** | **0.118** | 0.235 | 0.538 | **0.414** |

Two things jump out.

**(a) Lexical richness rises monotonically — at the album level.** Type-token ratio climbs 0.156 → 0.208 → 0.231 → 0.236 (Pearson r = **+0.93** vs release order); the hapax ratio (share of words used exactly once) climbs 0.364 → 0.411 → 0.531 → 0.538 (r = **+0.95**). Pooled per album, the Beatles' vocabulary becomes steadily richer and less repetitive.

**(b) Album cohesion is *not* monotonic — it is a Sgt. Pepper's spike.** Sgt. Pepper's has the highest internal cohesion (0.481) *and* the highest nearest-neighbour purity (0.615): its songs are each other's nearest neighbours far more than chance — **the geometry of a concept album**. Abbey Road has the *lowest* purity (0.118): only ~1 in 9 of its songs has its nearest neighbour on the same record — **the most internally heterogeneous album**, exactly as a medley of stylistic fragments should be.

![Lexical diversification and its caveat](/tidytuesday/2026-06-16-beatles-evolution/fig2_lexical_diversification.png)

The right panel above is the honest caveat, developed in §4: the strong album-level lexical trend is an **aggregation** phenomenon.

### The semantic map: albums intermingle

Projecting the 58 song embeddings to 2-D (t-SNE, cosine) and the album centroids to a similarity matrix:

![Semantic map and centroid drift](/tidytuesday/2026-06-16-beatles-evolution/fig3_semantic_map.png)

The albums **do not occupy distinct territories** — the colours interleave. The centroid-similarity matrix shows steady drift of ~0.10 per album step (Rubber Soul→Revolver 0.094, Revolver→Sgt. Pepper's 0.109, Sgt. Pepper's→Abbey Road 0.101) with **no acceleration**. Tellingly, the two most *distant* album centroids are **Rubber Soul ↔ Sgt. Pepper's (0.861)**, while **Revolver ↔ Abbey Road (0.903)** are among the *closest* — chronology is not the dominant axis. The drift is real but it is a slow, even diffusion.

---

## 3 · Graph Theory — The Song-Similarity Network (Centrepiece)

We build a graph: nodes = 58 songs, edges = mutual-kNN (k=5) semantic similarity, edge weight = cosine. The graph is connected (one component, density 0.14, 231 edges). We then run **Louvain community detection** and compare its modularity to the album partition.

```python
G = build_knn_graph(cosine_similarity(song_emb), songs, k=5)
comms = nx.community.louvain_communities(G, weight="weight", seed=42)
Q_comm  = nx.community.modularity(G, comms, weight="weight")              # 0.348
Q_album = nx.community.modularity(G, album_partition, weight="weight")    # 0.060
ari = adjusted_rand_score(album_labels, community_labels)                 # 0.090
```

| Partition | Modularity Q | Interpretation |
|---|---|---|
| **By album** (the record) | **0.060** | almost no community structure along album lines |
| **Louvain** (semantic) | **0.348** | real, strong community structure — but not by album |

**Alignment of communities with albums:** ARI = 0.090, NMI = 0.208 — near-independent. And **69% of all edges connect songs from different albums** (significantly *below* the chance rate of 76%, see §4 — so there is a faint within-album pull, but the graph is dominated by cross-album links).

![Communities cut across albums](/tidytuesday/2026-06-16-beatles-evolution/fig4_community_composition.png)

The community composition reveals what little structure exists: it is a **soft early/late gradient**, not album boundaries. Community 0 is mostly **Rubber Soul (8) + Revolver (5)** with **zero** Abbey Road songs; community 2 is mostly **Sgt. Pepper's (7) + Abbey Road (8)** with zero Rubber Soul. The earliest and latest material separates softly; the records themselves do not.

### The graph itself, and its bridges

![The Beatles song-similarity graph](/tidytuesday/2026-06-16-beatles-evolution/fig5_song_graph.png)

> **Interactive version (the Jay-Alammar-style centrepiece):** [`beatles_song_graph.html`](/tidytuesday/2026-06-16-beatles-evolution/beatles_song_graph.html) — every song is a node you can hover (title, album, community, betweenness); size encodes bridge role; colour encodes Louvain community.

The **bridge songs** — highest betweenness centrality, the connective tissue of the catalogue — are revealing:

| Song | Album | Betweenness | Cross-album edge ratio |
|---|---|---|---|
| **A Day in the Life** | Sgt. Pepper's | 0.262 | 0.62 |
| **Got to Get You into My Life** | Revolver | 0.253 | 0.78 |
| And Your Bird Can Sing | Revolver | 0.080 | 0.75 |
| Girl | Rubber Soul | 0.061 | 0.77 |
| With a Little Help from My Friends | Sgt. Pepper's | 0.054 | 0.64 |
| You Never Give Me Your Money | Abbey Road | 0.048 | 0.73 |

*A Day in the Life* and *Got to Get You into My Life* are the two songs that hold the four-album graph together — each routes most of its connections to *other* albums. There is a pleasing musicological reading here: *A Day in the Life* is itself a collage of two unrelated fragments (Lennon's news-report verses, McCartney's "woke up, got out of bed" middle), and *Got to Get You into My Life* is a soul/Motown pastiche that points forward to the band's later eclecticism. The graph's most central songs are its most stylistically promiscuous ones.

---

## 4 · Statistical Robustness

Four album means cannot carry inference. We test the two load-bearing claims properly.

### C1 — Is the lexical diversification real *per song*?

The album-level trend (r > 0.93) is dramatic, but TTR is length-biased and album-level pooling can manufacture monotonicity. We regress song-level lexical diversity on release order across all 58 songs, controlling for `log(tokens)`:

| Measure | β (release order) | p | 
|---|---|---|
| TTR ~ order + log(tokens) | +0.002 | **0.88** (n.s.) |
| Hapax ratio ~ order + log(tokens) | +0.034 | **0.15** (n.s.) |

**The per-song trend is not significant.** The strong album-level signal is an **aggregation effect**: later albums are not made of individually richer songs — they **reuse fewer words *across* their songs**. The album's vocabulary grows because its songs overlap less with one another, not because any single lyric got more sophisticated. This is the kind of result that disappears if you only look at the aggregate, and it is the most important honesty check in the post.

### C2 — Are albums separable? (permutation tests, 1,000 shuffles)

| Test | Observed | Null (shuffled labels) | p | Verdict |
|---|---|---|---|---|
| Silhouette of album labels | −0.011 | −0.024 ± 0.005 | **0.004** | real but ≈ 0 |
| Album-partition modularity Q | 0.060 | −0.017 ± 0.026 | **0.007** | real but ≈ 0 |
| Cross-album edge fraction | 0.688 | 0.761 ± 0.024 | (z = −3.0) | below chance |

![Permutation tests](/tidytuesday/2026-06-16-beatles-evolution/fig6_permutation.png)

The verdict is a careful one: **album identity is statistically detectable but practically negligible.** Both separability measures sit just above their shuffled nulls (p < 0.01), and the cross-album edge fraction is significantly *below* chance — so there *is* a faint gravitational pull of album membership. But on the absolute scale (silhouette ≈ 0, Q = 0.06 against a cross-cutting Q = 0.35) that pull is a rounding error next to the catalogue-wide semantic structure. **Detectable, not separable.**

---

## Discussion: The Arc

Put the pieces together and the evolution arc is not the one on the album sleeves.

1. **There is no sequence of separable lyrical worlds.** In embedding space the four albums interleave; album identity explains almost none of the geometry (silhouette ≈ 0; Q_album = 0.06). The dominant structure — Louvain Q = 0.35 — cuts straight across album boundaries.

2. **What changes is a slow, even diffusion within one register.** Centroids drift ~0.10 per step with no acceleration; the most distant pair is non-adjacent (Rubber Soul ↔ Sgt. Pepper's). Lexically, the albums diversify (r > 0.93) — but as an aggregation effect (songs sharing fewer words), not as per-song enrichment (p > 0.14).

3. **The one genuine album-level identity is *Sgt. Pepper's* — as internal cohesion, not external distinctiveness.** It has the highest intra-album cohesion and nearest-neighbour purity: the measurable geometry of a concept album whose songs reference each other. *Abbey Road*, by contrast, is the most internally fragmented record — the medley as a literal patchwork of semantic fragments, the lowest NN-purity in the catalogue.

4. **The catalogue is held together by bridge songs.** *A Day in the Life* and *Got to Get You into My Life* — the two most stylistically eclectic tracks — carry the highest betweenness, routing the graph's connections across albums.

This is consistent with the recurring lesson of the series. Distributional embeddings see **surface and structure** with great fidelity (they instantly flag Sgt. Pepper's self-similarity and Abbey Road's fragmentation), but they read the Beatles' lyrics as **one shared idiom** — the same register of love, longing, characters and English-pop diction — within which everything else is a slow drift. The famous "evolution" is overwhelmingly a matter of *production, arrangement and sonics*, dimensions a lyrics-only analysis is structurally blind to (a multimodal follow-up, à la the LAION-CLAP extension in the Aquamosh post, is the natural next step).

---

## Limitations

1. **Lyrics only.** The Beatles' evolution is most dramatic in arrangement, production and harmony — none of which a text embedding sees. The near-inseparability of albums *lyrically* says nothing about their separability *sonically*.
2. **Small corpus.** 58 songs / 1,866 lines is too small for stable per-album topic models; we mitigated with a shared topic space and lean on geometry + permutation tests for inference.
3. **One embedding family.** All geometry rests on `text-embedding-3-large`. The Aquamosh study showed effects can attenuate across architectures; a five-model replication would strengthen the separability claim.
4. **Canonical-tracklist scope.** We used the UK album tracklists; mono/stereo and deluxe-edition variants were excluded by design.
5. **Betweenness on a kNN graph** is sensitive to k; the bridge ranking is stable for k ∈ {4,5,6} but the absolute values are not.

---

## Conclusion

Asked to map how the Beatles' lyrics evolve across *Rubber Soul*, *Revolver*, *Sgt. Pepper's* and *Abbey Road*, the data declines the tidy story. The albums are **detectable but not separable**: a real, near-zero album signal (silhouette and modularity both p < 0.01, both ≈ 0) sits beneath a much stronger semantic community structure that ignores album boundaries entirely (Q = 0.35, ARI = 0.09, 69% cross-album edges). Lyrical evolution is a **single register slowly diffusing** — vocabulary spreading out across songs (album-level r > 0.93, but per-song n.s.), centroids drifting evenly, with *Sgt. Pepper's* the one album whose songs genuinely cohere and *Abbey Road* the one that deliberately fragments. The bridges — *A Day in the Life*, *Got to Get You into My Life* — are the catalogue's load-bearing eclectics.

The evolution is real. It is just not where the album covers tell us to look.

---

*Code, notebook, and reproducible pipeline: `tidytuesday/beatles-evolution-analysis/` (`collect_lyrics.py` → `embeddings.py` → `topic_modeling.py` → `evolution_metrics.py` → `graph_analysis.py` → `stats_robustness.py` → `figures.py`). Previous in series: [Attention Windows: Beatles vs. Pink Floyd](/post/2026-02-10-attention-windows-beatles-floyd) · [The Quadrilingual Probe: Aquamosh](/post/2026-05-20-aquamosh-quadrilingual-anatomy).*
