---
title: "Music & AI"
layout: "page"
description: "What happens when you apply NLP, embeddings, and LLMs to music. Rigorous analysis, genuine curiosity."
---

## The problem with measuring music computationally

Music is where NLP tools go to fail in interesting ways.

The standard playbook — embed the text, compute semantic similarity, cluster by topic — works reasonably well on news articles, product reviews, and technical documentation. Apply it to song lyrics and the failures are systematic and revealing. Distributional semantics, trained on co-occurrence statistics from web corpora, cannot distinguish "same theme, different words" from "different themes, same words."

Pink Floyd's *The Dark Side of the Moon* expresses a unified philosophical theme about mortality through varied metaphorical language — "ticking away," "shorter of breath," "closer to death." A transformer embedding model scores it as semantically incoherent because the surface tokens are different. The Beatles' verse-chorus repetition scores as highly coherent because the same phrases repeat. Both measurements are technically accurate. Both are interpretively wrong.

That failure is not a bug to be fixed with a better model. It's a structural property of distributional semantics, and understanding it tells you something important about what these tools can and cannot measure — not just in music, but in any domain where meaning is carried by metaphor, allusion, and reference rather than by surface lexical patterns.

This series documents an ongoing research program into computational approaches to musical analysis. The methods are rigorous. The questions are genuine. The results are regularly surprising.

---

## Posts in this series

### [Attention Windows: Measuring Narrative Cognitive Load in Beatles vs Pink Floyd](/post/2026-02-10-attention-windows-beatles-floyd/)
*February 2026*

The founding piece. Introduces **Attention Windows** — a framework for measuring semantic persistence in song lyrics using transformer embeddings. Tests the hypothesis that Pink Floyd requires more sustained cognitive integration than the Beatles, and finds the opposite. The counterintuitive result (Beatles exhibit 2.3× longer lexical persistence, p < 0.01) exposes a fundamental limitation of distributional semantics in poetic domains — and explains why it's fundamental, not incidental.

Four measurement methods (semantic decay, rolling coherence, entropy, network analysis). Matryoshka embeddings as internal validation. Formal null hypothesis testing throughout.

---

## What's next

**Corpus expansion.** Does the Beatles/Floyd pattern generalize? I want to test the Attention Windows framework across a broader set of artists and genres — jazz standards, hip-hop, classical lieder — to understand when the lexical-vs-conceptual coherence gap appears and why.

**LLM-based interpretation.** Using LLMs as reasoning agents over the semantic structures detected by embedding models. The hypothesis: LLMs can identify "same theme, different words" by reasoning over semantic fields, even when embedding similarity fails. Testing this rigorously is the next methodological challenge.

**Spectral + lyrical fusion.** Combining audio signal analysis with NLP to get a fuller structural picture — one that includes harmonic progression and timbre alongside lexical semantics.

---

## A note on method

All analyses use Python. Code and data are in the [GitHub repository](https://github.com/carlosjimenez88M). Statistical claims include effect sizes, p-values, and null model comparisons. If a result is surprising, I try to explain whether the surprise is informative or artifactual.

I am a vinyl collector and a serious listener. That background shapes which questions I find worth asking, and which results deserve skepticism.
