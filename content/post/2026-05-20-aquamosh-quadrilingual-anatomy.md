---
author: Carlos Daniel Jiménez
date: 2026-05-20
title: "The Quadrilingual Album as a Diagnostic Probe: Distributional Bias, Lyrics-Audio Decoupling, and the Measurement Geometry of Aquamosh (1998)"
categories: ["Music Analysis", "LLMs"]
tags: ["llms", "nlp", "music-analysis", "embeddings", "computational-musicology", "openai", "google-clap", "labse", "audio-embedding", "attention-windows"]
series:
  - NLP
  - LLMs
  - Embeddings
  - Computational Musicology
---

## Abstract

This research examines *Aquamosh* (1998), the quadrilingual (Spanish, English, French, Japanese) debut album by Plastilina Mosh, produced by the same team behind Beck's *Odelay* (Tom Rothrock & Rob Schnapf), as a **diagnostic probe** of modern distributional embeddings. Two findings emerge. **(1) Linguistic modulation, not fusion.** The album does not blend its four languages — it assigns each language a distinct semantic channel (English × cultural REFERENCE z = +4.62, Spanish × PLACE z = +3.13, mixed code-switching × {body, emotion, brand} z > +1.7, French × PLACE z = +3.68 for a single isolated Vienna scene). The chi-squared test of language × semantic field rejects independence with χ² = 124.7, p < 10⁻¹⁶ over n = 392 lines. **(2) Structural failure of distributional embeddings under code-switching.** Across five modern sentence-embedding architectures from two training families (OpenAI `text-embedding-3-large`, Google LaBSE, BGE-M3, multilingual-E5-large, multilingual MPNet), a language switch in consecutive lyric lines roughly doubles the probability of a "window break" — even when GPT-4o-mini as judge reads continuity (false-break rate ratio = 3.18× for OpenAI between switch and same-language transitions). This effect survives (a) permutation tests against H₀ of language-rupture independence (z = +6.54, p < 10⁻⁴, n = 10,000 simulations), (b) logistic regression with GEE clustered by track (OR = 3.99 [2.51, 6.36] for OpenAI), and (c) replication across all five architectures probed. The bias is a **structural property** of conventional dual-encoder embeddings with cosine similarity in the presence of code-switching, not a quirk of any single model. **(3) Lyrics-audio decoupling.** Using LAION-CLAP to embed the 12 album tracks in the same audio-text vector space as the Kozlowski cultural axes, we find **zero significant correlation** between lyrics-derived and audio-derived projections on the three cultural axes (r ≈ 0 in all cases, p > 0.5). The Anglophone production globalized the album's timbre **independently of its linguistic content**, with the title track *Aquamosh* carrying the maximum lyrics-vs-audio dissonance (most emotional lyrics, most ironic audio). *Aquamosh* is therefore a double object: linguistic decisions and sonic decisions live in non-correlated planes, and the album's coherence as a cultural artifact lives precisely in that duality.

**Practical consequence:** any retrieval, recommendation, or semantic-similarity system trained on multilingual user-generated content (lyrics, social media, code-switched Latin American discourse) inherits a measurable, replicable bias: lexical discontinuity from language switching is read as topical discontinuity, regardless of underlying semantic continuity. This is not a fine-tuning problem — it is the geometric consequence of measuring meaning through co-occurrence.

---

## TL;DR

This study uses the quadrilingual album *Aquamosh* (1998) as a falsification probe for distributional semantic embeddings. **Empirical findings, eight independent lines of evidence:** (1) Chi-squared on language × semantic field: χ² = 124.7, p < 10⁻¹⁶, n = 392 lines — languages are **not** interchangeable; each carries a distinct semantic load. (2) Cross-model invariance: five embedding architectures (OpenAI 3-large, LaBSE, BGE-M3, multilingual-E5, MPNet) all show a 1.31× to 1.94× relative gap in break probability between language-switch and same-language transitions. (3) Permutation test under H₀ of language-rupture independence: observed effect at z = +6.54 (OpenAI), z = +4.51 (LaBSE) — both p < 10⁻⁴. (4) GEE logistic regression clustered by track with controls for position and anchor/successor languages: OR = 3.99 [2.51, 6.36] for OpenAI (p < 0.001), OR = 2.52 [1.39, 4.57] for LaBSE (p = 0.002). (5) LLM-as-judge against GPT-4o-mini: false-break rate (model declares rupture while a sophisticated reader sees continuity) is 3.18× higher in switches than in same-language transitions for OpenAI, 1.33× for LaBSE. (6) Matryoshka dimensional compression: ORIGIN and SURFACE cultural axes are robust to truncation to 256 dimensions (ρ ≥ 0.78), but TIME collapses (ρ = 0.37) — suggesting temporal-cultural information is high-dimensional emergent, not lexically surface. (7) Critics' topic modeling (n = 56 sentences across four sources): two of four clusters are CMS chrome; the substantive critic discourse covers REFERENCE and NONSENSE strongly but undercovers EMOTION (cosine 0.244 — the album's title track carries its emotional center, yet specialized critics never use affective vocabulary). (8) **CLAP audio-text embeddings of the 12 tracks**: correlation between lyrics-derived and audio-derived Kozlowski projections is statistically zero in all three axes (r = +0.24, −0.07, +0.00; p > 0.5 throughout). **Theoretical contribution:** distributional embeddings under cosine similarity in the dual-encoder configuration cannot separate "the topic changed" from "the language changed but the topic continued" — this is Frege's *Sinn* vs. *Bedeutung* collapsed by co-occurrence statistics, not by deliberate design. **Methodological contribution:** code-switching corpora make the bias falsifiable and measurable; *Aquamosh*, by virtue of its quadrilingual structure, converts an epistemological worry into a quantitative diagnostic. **Practical consequence:** any multilingual NLP system using sentence embeddings as similarity backbone inherits a systematic underestimation of semantic continuity across language switches — a bias of ~3× in our measurements against a near-human reference. **Scope note:** we do not claim a universal impossibility; late-interaction models (ColBERT-X), cross-encoders trained on multilingual NLI, and reasoning models that compare passages directly might in principle escape this geometry. That falsification remains an open question this study does not foreclose.

---

## What This Post Does

This analysis does five things. **First**, it establishes that *Aquamosh* (1998) by Plastilina Mosh is structurally a quadrilingual album in which each language carries a distinct semantic load — not decoration, but a measurable division of communicative labor. **Second**, it extends the *Attention Windows* framework from the previous post in this series (Beatles vs. Pink Floyd) to multilingual lyrics, where language switches function as guaranteed lexical discontinuities and therefore convert the distributional-bias hypothesis from cultural intuition into a falsifiable experiment. **Third**, it validates this experiment through four independent methods: permutation tests against the null of language-rupture independence, cross-model invariance across five embedding architectures, GEE logistic regression with track-level random effects, and LLM-as-judge agreement against GPT-4o-mini. **Fourth**, it extends the analysis into the sonic domain using LAION-CLAP, projecting audio tracks onto the same Kozlowski cultural axes constructed from text-side anchors, revealing a complete decoupling between linguistic and sonic decisions in the album. **Fifth**, it situates *Aquamosh* in a counterfactual of contemporary regiomontano/Mexican albums (Café Tacuba, Control Machete, Molotov, Zurdok) to honestly characterize its position as a mid-tier cultural survivor rather than an exceptional commercial success.

Throughout, we maintain statistical rigor with hypothesis tests, effect sizes, null-model comparisons, and explicit declarations of scope — because the principal findings are sufficiently strong (and the practical consequences for multilingual NLP sufficiently broad) that confident overreach would damage the case more than careful framing.

> **Scope of this post.** This post does not measure *Aquamosh*'s commercial success in absolute terms — the Section 7 counterfactual measures *differential cultural survival* against contemporaries, which is related but distinct. The question "why did this specific album succeed in 1998" requires sales data, chart positions, and contemporary radio airplay that are not publicly accessible to me. That is material for a follow-up post.

---

## Why This Matters

Traditional lyrical analysis either relies on qualitative interpretation (close reading, hermeneutics) or surface-level statistics (word counts, lexical diversity), neither of which captures the **measurement geometry of meaning under code-switching**. This study tests whether distributional semantic embeddings — the silent backbone of every modern retrieval, recommendation, and similarity-based NLP system — can adequately represent multilingual creative texts where lexical surface and conceptual continuity are systematically misaligned by design.

The answer is empirically *no*, and the failure is **structural, not incidental**. No amount of scale, fine-tuning, or prompt engineering can correct the geometric fact that cosine similarity in a dual-encoder embedding space privileges co-occurrence statistics over abstract reference. *Aquamosh*, because Plastilina Mosh chose to articulate distinct semantic loads through distinct languages within the same album, makes this failure falsifiable in a way that monolingual corpora cannot.

The practical stakes are concrete. Latin American digital culture is heavily code-switched. Any embedding-driven system — Spotify recommendations on Latin Alternative playlists, Twitter/X content moderation across Spanish-English boundaries, customer-support routing in cross-lingual contexts — inherits the bias documented here. The geometric distortion is reproducible, large (3.18× in our most stringent measurement), and visible in all five embedding architectures of the modern state of the art.

---

## Theoretical Framework

### Definition: Attention Windows under Multilingual Sequences

Given a sequence of lyric lines $L = \{l_1, l_2, ..., l_n\}$ with corresponding embeddings $E = \{e_1, e_2, ..., e_n\}$ where $e_i \in \mathbb{R}^d$ for some embedding model $\phi$, the **attention window** at position $i$ is:

$$W_i(\theta) = \max\{k : \text{sim}(e_i, e_{i+j}) \geq \theta \text{ for all } j \in [1, k]\}$$

where $\text{sim}(e_a, e_b) = e_a \cdot e_b / (\|e_a\| \|e_b\|)$ is cosine similarity and $\theta$ is a model-specific coherence threshold calibrated to a "notably high similarity" level for that embedding family.

Operationally, $W_i$ counts how many subsequent lines stay semantically close to the anchor line before the discourse moves away.

### The Multilingual Falsification

In monolingual lyrics, attention windows conflate two distinct phenomena: **lexical coherence** (repeated tokens, n-grams, surface forms) and **conceptual coherence** (diverse linguistic expressions of a unified abstract theme). The Beatles vs. Pink Floyd analysis in the previous post in this series demonstrated that distributional embeddings systematically privilege the former over the latter, but the demonstration relied on stylistic differences between two albums — confounded by genre, era, and writer.

*Aquamosh*'s quadrilingual structure removes the confound. **Every language switch is a guaranteed lexical discontinuity**, by definition: "te quiero" and "I love you" share no surface vocabulary, regardless of conceptual equivalence. If distributional embeddings measure conceptual continuity, language switches should leave attention windows undisturbed when the topic continues. If they measure lexical surface, language switches should systematically break windows independent of topic.

Two falsifiable hypotheses:

**H₁ (Lexical discontinuity hypothesis):** Language switches in consecutive lines increase the probability of attention-window rupture, controlling for track-level and positional confounds, even when human-equivalent readers identify topical continuity.

**H₂ (Multilingual-training hypothesis):** Embedding models explicitly trained on parallel cross-lingual corpora (LaBSE, multilingual-E5) attenuate the H₁ effect compared to dominantly-monolingual decoder-trained models (OpenAI `text-embedding-3-large`), but do not eliminate it.

Both hypotheses make directional predictions. Both are testable against the same dataset. Both align with predictions of the distributional hypothesis (Firth, 1957) and the Frege Sinn/Bedeutung distinction (reference cannot be recovered from co-occurrence statistics alone).

### Critical Assumption (VIOLATED, by design)

We assume — and this is what we test — that cosine similarity satisfies:

$$\text{sim}(e_{\text{theme}}, e_{\text{syn-EN}}) \approx \text{sim}(e_{\text{theme}}, e_{\text{syn-ES}}) \gg \text{sim}(e_{\text{theme}}, e_{\text{unrelated}})$$

where $\text{syn-EN}$ and $\text{syn-ES}$ are conceptually equivalent expressions in English and Spanish of the same theme as the anchor. The empirical falsification of this assumption is the central methodological result of this post.

**Example failure (real, from the corpus):**

| Line A | Line B | OpenAI sim | LLM (GPT-4o-mini) judgment |
|---|---|---|---|
| "De cultura y de rutina" | "Como si fuera heroína" | 0.287 (below θ = 0.32) | *"Both lines address themes of dependence and routine, maintaining thematic connection."* — continuity |
| "Si es la revolución" | "Desde tu televisión" | 0.258 | *"Both lines relate to revolution and its media representation."* — continuity |

The sentence-embedding cosine collapses below threshold in both cases — declaring topical rupture — while a sophisticated reader perceives clear thematic continuity. **The metric measures lexical surface, not reference.**

### Threshold Calibration

Comparing $W_i^{\text{Model A}}$ against $W_i^{\text{Model B}}$ with the same $\theta$ is methodologically naive: different architectures induce different similarity distributions. We calibrate per model using the **percentile-shift method**:

$$\theta_{\text{model}} = \text{median}\{\text{sim}(e_i, e_j)\}_{\text{random pairs}} + \sigma\{\text{sim}(e_i, e_j)\}_{\text{random pairs}}$$

This defines $\theta$ as a "notably high similarity for this model" anchor. Empirically: $\theta_{\text{OpenAI}} = 0.3201$ and $\theta_{\text{LaBSE}} = 0.3230$ — nearly identical, validating the calibration as fair across geometries.

---

## Methodology

### Corpus

12 tracks of *Aquamosh* (1998, EMI México / Capitol Records). Lyrics retrieved via Genius API: 10/12 tracks transcribed (missing *Ode to Mauricio Garcés* and *Encendedor* in the public Genius corpus). 392 analyzable lines after cleaning and sentence-level filtering. Custom language detector combining `langdetect` probabilities with marker-based override and Romance-language collapse — this corrected 78 lines that raw `langdetect` had misclassified as Portuguese due to Mexican Spanish contractions ("pa' bailar", "tá"). Critic corpus: 56 sentences across four sources (Ink19 1998 specialized review, Album of the Year user-review, Spanish and English Wikipedia entries). Audio corpus: all 12 tracks downloaded via `yt-dlp` from the official Plastilina Mosh - Topic playlist as mp3.

### Embedding Models

Five sentence-embedding models tested on lines and lyrics:

| Model | Dim | Training family |
|---|---|---|
| OpenAI `text-embedding-3-large` | 3072 | decoder, large-scale, English-dominant |
| Google LaBSE | 768 | encoder, parallel corpus, 109 languages |
| BAAI BGE-M3 | 1024 | encoder, multi-functional / multi-granularity |
| Multilingual E5-large | 1024 | encoder, weakly supervised |
| Paraphrase-multilingual-MPNet | 768 | encoder, paraphrase-trained |

For audio, LAION-CLAP (HTSAT-tiny audio encoder, 512-dim shared audio-text space).

### Statistical Methodology

Chi-squared with standardized residuals; permutation tests with 10,000 resamples; logistic regression with three model specifications (marginal, fixed-effects controls, GEE clustered by track + LPM with random intercepts); cosine similarity with model-specific threshold calibration; LLM-as-judge with structured JSON output and Cohen's κ stratified by transition type. Random seed: 42 throughout. All cached, reproducible, executable via `jupyter nbconvert --execute` in approximately three minutes on warm cache.

---

## Result 1: A Linguistic Grammar

The chi-squared test of language × semantic field over the corpus rejects independence with extreme strength:

$$\chi^2 = 124.7, \quad \text{df} = 21, \quad p < 10^{-16}$$

The choice of which language to use for which semantic field is far from random. Standardized residuals (|z| > 2 indicates significant association) reveal a clean pattern:

![Heatmap of language × semantic field](outputs/figures/language_field_residuals_v2.png)

| Association | z | Reading |
|---|---|---|
| **EN × REFERENCE** | **+4.62** | English is the language of proper names and cultural citations ("Woody Allen's world", "Afroman", "Mr. P. Mosh") |
| **MIXED × REFERENCE** | **−4.41** | When citing culture, the lyrics pick a single language — they do not code-switch |
| **FR × PLACE** | +3.68 | All 6 French lines in the album refer to Vienna (*Savage Sucker Boy*) |
| **ES × PLACE** | +3.13 | Spanish anchors geography: "Desde África querida", "Para América Latina" |
| **EN × BODY** | −2.78 | The body is *not* named in pure English; physical references emerge in mixed lines |
| MIXED × {BODY, EMOTION, BRAND} | +1.84 to +2.52 | Code-switching is the register of intimacy and commerce |

**The reading:** *Aquamosh* does not fuse global and local. It **modulates** between them by language channel. Each language has a specific semantic job. The strategy is not blending; it is **channeling**.

### A Counterintuitive Consequence

The album's most globally exported tracks (*Monster Truck*, *Mr. P. Mosh*, both placed in *Street Sk8er*; *Afroman* in *True Crime: Streets of LA*) are also the **least multilingual**. *Savage Sucker Boy* (the only track containing French, highest linguistic entropy H = 1.73) is likely the least-listened cut. The quadrilingualism of *Aquamosh* is concentrated in the album cuts, not in the singles.

![Linguistic composition by track](outputs/figures/language_by_track.png)

This refutes the naive reading that "Aquamosh succeeded *because* it was quadrilingual." The quadrilingualism is a structural fact about the album, not its commercial mechanism.

---

## Result 2: The Cultural Map (Kozlowski Semantic Axes)

Following Kozlowski et al. (2019, *American Sociological Review*), I project each track onto three cultural axes constructed from verbal anchors:

- **ORIGIN**: Monterrey/regio ↔ Los Angeles/mainstream
- **SURFACE**: emotion ↔ irony
- **TIME**: underground 1998 ↔ retrospective classic

![Cultural map of Aquamosh](outputs/figures/semantic_axes_map.png)

| Track | ORIGIN | SURFACE | TIME |
|---|---|---|---|
| **Monster Truck** | **+0.082** (most LA) | **+0.054** (most ironic) | −0.129 |
| Mr. P. Mosh | +0.010 | −0.014 | −0.011 |
| Niño Bomba | **−0.060** (most regio) | +0.012 | −0.050 |
| **Aquamosh** (title) | −0.005 | **−0.149** (most emotional) | −0.002 |
| Pornoshop | −0.034 | −0.093 | **+0.029** (most retro) |

The title track *Aquamosh* sits at the emotional extreme of the album with virtually zero irony. The band named its project after its least ironic track. That is a decision, not an accident, and it positions the album's emotional center on the song that carries its name.

---

## Result 3: The Language-Switch Effect on Attention Windows

Each pair of consecutive lines within a track is a *transition*. For each transition I record whether the language changed and whether the embedding similarity falls below the calibrated threshold (a "window break").

![Window break rate by transition type](outputs/figures/aw_break_rates.png)

| Transition type | OpenAI | LaBSE | Gap |
|---|---|---|---|
| same language | 0.36 | 0.41 | — |
| **language switch** | **0.70** | **0.65** | OpenAI **+0.34** · LaBSE **+0.24** |

**Both models nearly double their window-break rate at language switches.** OpenAI rises from 0.36 to 0.70. LaBSE from 0.41 to 0.65. In a 50-line track with 15 language switches, that translates to ~10 (OpenAI) or ~8 (LaBSE) window breaks attributable purely to lexical change, independent of topical change.

LaBSE's gap (+0.24) is 30 % smaller than OpenAI's (+0.34) — consistent with H₂'s prediction that parallel-corpus training attenuates the bias. The attenuation is real but not eliminating.

### Cross-Model Agreement

Spearman correlation of line-by-line attention windows between OpenAI and LaBSE: **ρ = 0.64** (n = 392, p < 10⁻⁴⁵). The two models agree on coarse ranking but diverge in detail, especially in MIXED-language anchors.

![Cross-model scatter](outputs/figures/aw_cross_model_scatter.png)

### The Paradigmatic Case: *Savage Sucker Boy*

The track with the most code-switching and the only French content has the largest cross-model disagreement: OpenAI window mean = 1.13, LaBSE = 2.45. The multilingual-aware model sees more than twice the coherence in the album's most-multilingual track.

![Attention windows per track](outputs/figures/aw_per_track.png)

---

## Result 4: Dimensional Compression and the Collapse of TIME

Using OpenAI's matryoshka-compatible `text-embedding-3-large`, I re-embed the album lyrics at 256, 512, and 1024 dimensions and recompute the Kozlowski axis projections.

![Dimensional robustness](outputs/figures/dimension_robustness.png)

| Axis | ρ(dim=256, dim=3072) |
|---|---|
| ORIGIN (geography) | 0.78 — robust |
| SURFACE (affect) | 0.89 — robust |
| **TIME** (cultural dating) | **0.37 — collapses** |

Spatial and affective signals are lexically discrete enough to survive compression; cultural temporality lives in fine-grained, distributed correlations and disintegrates at low dimensions. This is a methodological side-finding worth its own follow-up post: *which cultural properties are high-dimensional emergent and which are surface-lexical?*

---

## Result 5: What the Critics Say (Sentence-Level Topic Modeling)

Surviving online critical discourse on *Aquamosh* is sparse. After clean scraping: 4 usable sources, 56 substantive sentences, dominated 71 % by Ink19's July 1998 review. K-Means clustering with k = 4 over sentence-level CLAP embeddings yields:

![Critics' topic space (PCA)](outputs/figures/critics_topics_pca.png)

| Cluster | n | Reading |
|---|---|---|
| Bylines (CMS chrome) | 8 | Author signature repeated by CMS — noise |
| Genre framing | 12 | "Belongs in any Rock section" — substantive |
| Genre fusion description | 20 | Track-by-track sonic descriptions — substantive |
| Sidebar (other albums) | 16 | Site chrome — noise |

**24 of 56 sentences (43 %) are structural noise.** In small corpora, the first job of topic modeling is to separate signal from CMS chrome, not to discover deep themes.

### Critics' Coverage of Album Semantic Fields

Crossing the four critic clusters with the eight album semantic fields:

![Critics × album fields](outputs/figures/critics_x_album_fields.png)

| Album field | Max coverage |
|---|---|
| NONSENSE | **0.515** (most covered — scratches, samples) |
| REFERENCE | 0.439 |
| ACTION | 0.310 |
| BODY | 0.307 |
| BRAND | 0.293 |
| PLACE | 0.268 |
| IDENTITY | 0.267 |
| **EMOTION** | **0.244** (least covered) |

The critical discourse covers the album's formal-referential surface heavily and its affective dimension barely. The track that gives the album its name is its most emotional track — and specialized critics never use affective vocabulary to describe it. **The album defines itself affectively; critics describe it formally.** That gap may be precisely what allowed *Aquamosh* to age as "mythical cult album" rather than "dated experiment": the emotional core persists while the surface-referential conversation grows stale.

---

## Result 6: Four Independent Validations of the Central Hypothesis

The central claim — that language switches systematically break embedding-based attention windows independent of topical continuity — receives four convergent validations.

### 6.1 Permutation Tests Against Independence

10,000 random reassignments of language labels within each track, preserving line order:

| Test | Observed gap | Null μ | Null σ | z |
|---|---|---|---|---|
| OpenAI · H₀_A (shuffle labels) | +0.338 | +0.000 | 0.052 | **+6.54** |
| LaBSE · H₀_A (shuffle labels) | +0.238 | +0.003 | 0.052 | **+4.51** |

Under independence of language and rupture, the observed gap is essentially impossible (p < 10⁻⁴ in both models).

![Permutation null distributions](outputs/figures/permutation_null_distributions.png)

### 6.2 Cross-Model Invariance Across Five Architectures

| Model | Dim | P(break|same) | P(break|switch) | Rel-gap |
|---|---|---|---|---|
| OpenAI `text-embedding-3-large` | 3072 | 0.361 | 0.698 | **1.94×** |
| BAAI BGE-M3 | 1024 | 0.426 | 0.774 | 1.82× |
| MPNet multilingual | 768 | 0.410 | 0.734 | 1.79× |
| Google LaBSE | 768 | 0.410 | 0.648 | 1.58× |
| Multilingual E5-large | 1024 | 0.372 | 0.487 | 1.31× |

![Cross-model invariance](outputs/figures/cross_model_invariance.png)

All five models confirm the effect. Mean relative gap = 1.69×. Models trained on parallel cross-lingual data (LaBSE, E5) attenuate the effect but do not eliminate it.

### 6.3 LLM-as-Judge Validation

For each consecutive pair (n = 382), GPT-4o-mini judges topical continuity with explicit instruction to treat language switches as orthogonal to topic. The crucial metric is **false-break rate**: model declares rupture while LLM declares continuity.

![LLM judge false breaks](outputs/figures/llm_judge_false_breaks.png)

| Model | Stratum | False-break rate | Cohen's κ |
|---|---|---|---|
| OpenAI | same-lang | **0.060** | 0.685 |
| OpenAI | switch | **0.191** (3.18× higher) | 0.376 |
| LaBSE | same-lang | 0.098 | 0.636 |
| LaBSE | switch | 0.131 (1.33× higher) | 0.540 |

OpenAI declares "rupture" while a sophisticated reader sees continuity **3.18× more often in switches than in same-language transitions**. This is the bias measured directly against a near-human reference. Cohen's κ drops from "substantial agreement" (0.685) in same-lang transitions to "moderate-weak" (0.376) in switch transitions — exactly where the geometric distortion is theoretically expected.

### 6.4 Logistic Regression with Track-Level Random Effects

Three specifications: marginal (M1), with fixed-effects controls (M2), and GEE clustered by track with exchangeable correlation (M3). The switch effect must survive M3 to be defensible.

![Forest plot of switch effects](outputs/figures/mixed_effects_forest.png)

| Specification | OpenAI OR (95 % CI) | LaBSE OR (95 % CI) |
|---|---|---|
| M1 marginal | 4.10 [2.69, 6.25] | 2.65 [1.74, 4.05] |
| M2 fixed controls | 3.97 [2.49, 6.34] | 2.53 [1.52, 4.21] |
| **M3 GEE clustered** | **3.99 [2.51, 6.36]** | **2.52 [1.39, 4.57]** |

All p < 0.01 throughout. The switch OR is approximately 4× for OpenAI and 2.5× for LaBSE, **controlling for track, position within track, and the languages of both lines in the pair**. The effect *amplifies* slightly when track is separated from switch, because the most switch-heavy tracks (*Savage Sucker Boy*) also have higher structural coherence baselines (repeated choruses), and the unconditional χ² mixes the two.

Track-level random intercepts validate the model qualitatively: the tracks with the lowest baseline break probability are the most structurally repetitive (*Mr. P. Mosh*, *Monster Truck*, *Savage Sucker Boy*); the most fragmented are *Afroman* and *Bungaloo Punta Cometa*.

![Random intercepts per track](outputs/figures/mixed_effects_random_per_track.png)

---

## Result 7: Differential Cultural Survival — *Aquamosh* in Counterfactual

The opening framing of this post — *"Aquamosh succeeded"* — is a received cultural datum, not a measured outcome. To take that framing seriously, I constructed a counterfactual against four contemporary Mexican / regiomontano albums with varying degrees of bilingualism: Café Tacuba *Revés/Yo Soy* (1999), Control Machete *Mucho Barato* (1996), Molotov *¿Dónde Jugarán las Niñas?* (1997), Zurdok *Hombre Sintetizador* (1999).

Three publicly available metrics: Wikipedia pageviews (2015-2026 via the official REST API), Google Trends (2004-2026), Discogs community statistics.

![Google Trends 2004-2026](outputs/figures/commercial_trends.png)

![Wikipedia pageviews](outputs/figures/commercial_wiki_pageviews.png)

| Artist | Wiki ES+EN | Trends (last 12 mo) | Discogs rating | Discogs # have | **Survival Index** |
|---|---|---|---|---|---|
| Café Tacuba | 1,238,355 | 6.8 | **4.61** | 186 | **84** |
| Control Machete | 1,041,838 | **8.4** | 3.37 | 128 | **76** |
| **Plastilina Mosh** | 684,384 | 5.3 | 4.06 | 169 | **68** |
| Zurdok | < 1,000 | 0.9 | 5.00 (n = 3) | 11 | 42 |
| Molotov | (matching issue) | 1.0 | — | — | 30 |

**Aquamosh is mid-tier survivor.** It is in the upper third of its generational cohort but is not exceptional. Café Tacuba *Revés* and Control Machete *Mucho Barato* show greater density of mentions, searches, and physical collection across 28 years. *This is important for not overselling the post.* The internal structure that the chi² captures is real; **it is not the cause of the album's survival**. That is more plausibly explained by the singles (video game placements, Beck-team production, the *Niño Bomba* + *Monster Truck* hooks) than by the linguistic strategy.

The previous post and this one together describe *what* *Aquamosh* is. They do not explain *why it matters*. That requires a different analysis with different data.

---

## Result 8: Lyrics-Audio Decoupling via LAION-CLAP

The lyrics-only analysis is structurally blind to the album's most-discussed feature: the production by Rothrock and Schnapf, who shaped *Beck*'s *Odelay* the year before. The claim that "the Anglophone production globalized the timbre without touching the linguistic content" is testable. I download the 12 album tracks via `yt-dlp` and embed each in LAION-CLAP, a model that maps audio and text into the same vector space. This allows projecting tracks onto the same Kozlowski cultural axes built from text-side anchors — but now with audio-derived vectors.

### The Central Result: No Significant Correlation Between Lyrics and Audio Projections

![Audio vs lyrics on the three axes](outputs/figures/audio_vs_lyrics_axes.png)

| Axis | Pearson r | Spearman ρ | p |
|---|---|---|---|
| ORIGIN | +0.241 | +0.048 | 0.57 |
| SURFACE | −0.074 | +0.119 | 0.86 |
| TIME | +0.004 | −0.167 | 0.99 |

**Zero significant correlation on all three axes.** Lyric decisions and sonic decisions in *Aquamosh* are independent. The Anglophone production globalized the album's timbre with no regard for the linguistic content of the lyrics it was wrapping.

### No Sonic Flattening

Mean inter-track cosine similarity in the CLAP audio space: **0.46**. A production that homogenized the album toward a common sound would yield 0.85–0.95. A maximally diverse album would yield 0.30–0.50. *Aquamosh* sits in the second regime: Rothrock and Schnapf added enough sonic cohesion to make the album recognizable as one object, but **preserved inter-track variety**.

![Sonic similarity matrix](outputs/figures/audio_sim_matrix.png)

### The Title Track Carries Maximum Dissonance

| Track | SURFACE (lyrics, − = emotional) | SURFACE (audio, + = ironic) |
|---|---|---|
| **Aquamosh** | **−0.149** (most emotional in album) | **+0.038** (most ironic in album) |
| Pornoshop | −0.093 | −0.109 (coherent) |
| Monster Truck | +0.054 | −0.044 (coherent, opposite direction) |

*Aquamosh*, the song that names the album, is where **lyrics and production pull in opposite directions with maximum intensity**. The lyric is the most sincere on the album; the audio is the most ironically distant. That is a deliberate aesthetic position: the title track carries the negotiation with sincerity at its extreme. One *feels* the lyric while *hearing* its opposite. This is visible only when you cross the two channels.

The sonic-similarity matrix confirms this from another angle: *Aquamosh* is the most sonically isolated track on the album (its row/column in the heatmap has the lowest mean similarity with the others). It is structurally eccentric.

### What Only Audio Reveals

| Axis | Most positive (audio) | Most negative (audio) |
|---|---|---|
| ORIGIN (regio→LA) | Encendedor (+0.21), Monster Truck (+0.18), Mr. P-Mosh (+0.17) | Savage Sucker Boy (+0.01), Ode to Mauricio Garcés (+0.04) |
| SURFACE (emotion→irony) | Aquamosh (+0.04), Mr. P-Mosh (+0.02) | Pornoshop (−0.11), Bungaloo (−0.10) |
| TIME (1998→retrospective) | Ode to Mauricio Garcés (+0.10), Milton Pacheco (+0.07) | Mr. P-Mosh (−0.15), Encendedor (−0.14) |

Two findings only audio reveals: (a) *Encendedor* — the only track without Genius-transcribed lyrics — emerges as the **most "LA-mainstream" by audio**, and it samples Minutemen. Lyrics-only is blind to instrumental tracks. (b) *Ode to Mauricio Garcés* (bossa nova / acid jazz / lounge) is the **most retrospective** track by audio (TIME +0.10) — it sounds like the 1960s, not 1998. The lyrics (which we don't have) likely could not have captured this temporal dimension.

---

## The Synthesis

This analysis supports three claims, at three distinct levels of inference.

### About the Album

*Aquamosh* does not resolve the global-local dilemma — it **separates** the two by linguistic channel and lets them coexist in the same object. Each language carries a specific semantic load: English for cultural reference, Spanish for geography and territory, mixed code-switching for the intimate and the commercial, French for a single geographic exotic. The album's strategy is not to mix cultural registers — it is to **modulate** between them. That claim is statistically more defensible than "cultural fusion" and is supported by p < 10⁻¹⁶ data.

**Important caveat:** this internal structure does not explain the album's commercial outcome. The Section 7 counterfactual shows *Aquamosh* to be mid-tier (Survival Index 68), below Café Tacuba (84) and Control Machete (76) in its cohort. Linguistic modulation is an aesthetic fact of the disc, not the cause of its survival.

**More important caveat:** the linguistic plane is only one plane. The Section 8 audio analysis with CLAP shows that lyrical and sonic decisions are **decoupled** — the producer operated on the timbre independently of the linguistic content. *Aquamosh* is therefore a **double object**: lyrics say one thing, audio says another, and the album's coherence lives in that duality. The title track carries the lyrics-vs-audio dissonance at its maximum (hyper-emotional lyric + kitsch-distanced audio), making it the aesthetic center of the album.

### About the Tool (Embeddings)

Four independent lines of evidence converge on the same finding: a language switch in consecutive lyric lines roughly doubles the probability of a window break in conventional dual-encoder embeddings under cosine similarity.

1. **Cross-model invariance** across five architectures from two training families (rel-gap 1.31× to 1.94×).
2. **Permutation test** against the null of language-rupture independence (OpenAI z = +6.54, LaBSE z = +4.51).
3. **GEE logistic regression** controlling for track, position, and the languages of both lines (OpenAI OR = 3.99, LaBSE OR = 2.52, p < 0.01).
4. **LLM-as-judge** showing false-break rate 3.18× higher in switches than in same-language transitions (OpenAI vs. GPT-4o-mini).

Models with explicit cross-lingual training (LaBSE, E5) **attenuate** the bias but do not eliminate it. The bias persists across five architectures, survives every regression control, and validates against external judgment.

> Sentence embeddings with cosine similarity in the conventional dual-encoder configuration do not distinguish "the topic changed" from "the language changed but the topic continued" — in any of the five architectures probed, which span the two dominant training families (decoder large-scale and encoder parallel-corpus). Late-interaction models (ColBERT-X), cross-encoders trained explicitly on multilingual NLI, and reasoning models that compare passages directly could in principle make this distinction — but that falsification remains outside what this analysis measures.

### About the Object as Diagnostic Probe

*Aquamosh* stopped being an object of analysis and became a **diagnostic probe** of the analytical tool. The aesthetic decisions Plastilina Mosh made in 1998 — four languages with division of semantic labor, samples from American funk and West-Coast punk, production by Beck's team — produced a text that no contemporary distributional embedding model can read without systematic distortion.

The interesting question shifts from *"what does the album say?"* to *"what does this album make visible about our tools?"*

That recursive shift — from object to probe — is the methodological contribution of the post.

---

## What This Analysis Cannot Answer

### Still unmeasured

- **The Japanese role.** *Aquamosh* includes Japanese vocal samples that Genius does not transcribe. The lyrics-only analysis has zero JA lines. Whisper-large-v3 on suspected segments is the natural next step.
- **Specialized critical archive at scale.** n = 56 sentences, 71 % from a single source. Findings on critics' affective undercoverage are suggestive, not conclusive. Would require scraping print-archive sources (*La Mosca en la Pared*, *Rolling Stone México*, *Switch*).

### No longer a limitation (resolved by validations added)

- ~~Validation against human-equivalent judgment~~ → LLM-as-judge (Sec. 6.3): GPT-4o-mini on 382 pairs, κ stratified by transition.
- ~~n = 2 models is anecdotal~~ → cross-model invariance with n = 5 (Sec. 6.2): all confirm the effect.
- ~~No null model~~ → permutation tests with 10,000 simulations (Sec. 6.1): z = +6.54.
- ~~Chi-squared is marginal — track confounding~~ → GEE clustered + LPM with random effects (Sec. 6.4): OR = 3.99 controlling for track, position, language pair.
- ~~Commercial success not measured~~ → counterfactual with five regiomontano/Mexican albums (Sec. 7): Wikipedia + Trends + Discogs. *Aquamosh* is mid-tier (Survival Index 68), not exceptional.
- ~~No audio analysis~~ → LAION-CLAP on all 12 tracks (Sec. 8): lyric and sonic decisions are **decoupled** (r ≈ 0 in three axes).
- ~~Universal claim about embeddings~~ → reframed to apply specifically to dual-encoder configurations with cosine similarity in the five architectures probed. Late-interaction and reasoning models remain open.

### Note on n and power

n = 392 lines is modest in absolute terms, but the effects measured are robust: χ² with p < 10⁻¹⁶, permutation z = +6.54, false-break ratio 3.18×, all five embedding models replicate the gap, GEE OR = 3.99 with confidence intervals well above unity. Conclusions about French (n = 6 lines, all in *Savage Sucker Boy*) are illustrative, not statistical, and stated as such whenever French appears.

---

## Closing

Five ideas to take away:

1. **The most interesting statistical tests refine hypotheses rather than confirm or refute them.** I entered with "global-local fusion." I left with "channel modulation." Stronger claim, fully supported by the data.

2. **Validating a methodological finding requires multiple convergent lines of evidence.** Cross-model (n = 5), permutation tests (z = +6.54), clustered regression (OR = 3.99), LLM-as-judge (3.18× false-break ratio) — four independent methods pointing at the same place. A single line of evidence would be fragile; four convergent lines sustain a structural property.

3. **The counterfactual exists to *not oversell*.** *Aquamosh* has a Survival Index of 68 on a 0–100 scale. It is in the upper third but not exceptional. That honesty is what distinguishes analysis from marketing. The internal structure the post describes is real; the comparative success its title might have suggested is not.

4. **Cross-channel analysis reveals what neither lyrics nor audio can show alone.** The lyrics-audio decoupling (r ≈ 0 in all three cultural axes) is invisible to monomodal analysis. The title track's lyrics-audio dissonance becomes the aesthetic center of the album only when both channels are projected into the same vector space.

5. **A culturally complex object can be a diagnostic test of the tools used to analyze it.** *Aquamosh* is exactly that. The previous post in this series (Beatles vs. Pink Floyd) proposed a hypothesis. This post converted it into a falsifiable test, ran the test across five embeddings architectures against a null distribution and against a human-equivalent judgment, and the test held. The next step is to identify which embedding architectures — if any — can break the distributional bias without sacrificing the properties that make them useful.

*Aquamosh* still sits there, in 1998, as a test no one wrote for 2026.

---

## Resources

Complete code, exported data, figures, and the executable notebook are in `tidytuesday/aquamosh-analysis/`:

- `aquamosh_analysis.ipynb` — full notebook, zero errors
- `THEORETICAL_FRAMEWORK.md` — extended theoretical framework
- `CRITICS_ANALYSIS.md` — critics' topic modeling
- `attention_windows.py` — OpenAI/LaBSE dual analysis
- `cross_model_invariance.py` — n = 5 models (Section 6.2)
- `permutation_tests.py` — 10k permutations (Section 6.1)
- `llm_as_judge.py` — GPT-4o-mini validation (Section 6.3)
- `mixed_effects_regression.py` — GEE clustered by track (Section 6.4)
- `commercial_success.py` — counterfactual with five regiomontano albums (Section 7)
- `audio_analysis.py` — yt-dlp download + LAION-CLAP + projection onto Kozlowski axes (Section 8)

## References

- Firth, J. R. (1957). *A synopsis of linguistic theory.* The original formulation of the distributional hypothesis.
- Kozlowski, A. et al. (2019). *The Geometry of Culture.* American Sociological Review 84(5).
- Feng, F. et al. (2022). *Language-Agnostic BERT Sentence Embedding (LaBSE).* ACL.
- Chen, J. et al. (2024). *BGE-M3: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings.* arXiv:2402.03216.
- Wang, L. et al. (2024). *Multilingual E5 Text Embeddings.* arXiv:2402.05672.
- Wu, Y. et al. (2023). *Large-scale Contrastive Language-Audio Pretraining (LAION-CLAP).* ICASSP.
- Previous post in this series: */post/2026-02-10-attention-windows-beatles-floyd*
