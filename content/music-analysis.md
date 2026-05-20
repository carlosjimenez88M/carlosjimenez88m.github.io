---
title: "Music, NLP & LLMs"
layout: "page"
description: "A research thread on lyrics, semantics, embeddings, and what large language models can and cannot tell us about music."
---

## Why this section exists

This is not a side hobby page. It is one of the most revealing things I work on.

Music is where NLP tools fail in interesting and instructive ways. The standard workflow for text analysis works reasonably well on news, reviews, and documentation. Once you move into lyrics, metaphor, repetition, voice, and thematic indirection, the usual assumptions start to break.

That failure is useful. It tells us something about the limits of embeddings, the gap between lexical and conceptual coherence, and the extent to which LLMs can or cannot recover higher-order structure through reasoning.

So this section sits at the intersection of three things:

- **NLP as measurement**
- **LLMs as interpreters and reasoning systems**
- **Music as a hard case for both**

---

## The core problem

Distributional semantics often confuses repeated vocabulary with thematic unity and varied language with conceptual drift.

Pink Floyd's *The Dark Side of the Moon* can articulate a unified meditation on mortality using different images in every track. A transformer embedding model may interpret that as incoherence because the surface language changes. A Beatles song with tighter lexical repetition can score as more coherent for exactly the opposite reason.

Both measurements are computationally defensible. Both can be interpretively misleading. That tension is the point.

---

## What I am doing here

### [Attention Windows: Measuring Narrative Cognitive Load in Beatles vs Pink Floyd](/post/2026-02-10-attention-windows-beatles-floyd/)
*February 2026*

This is the founding piece of the section. It introduces **Attention Windows**, a framework for measuring semantic persistence in lyrics using transformer embeddings, and then shows why the most intuitive interpretation of the results is wrong.

The result is not just a Beatles-versus-Floyd comparison. It is an argument about what embeddings are actually measuring in poetic domains.

### What comes next

**Larger corpora.** I want to test whether the lexical-versus-conceptual coherence gap holds across more artists and genres.

**LLM-based interpretive layers.** The open question is whether LLMs can reason over semantic fields in a way that compensates for the blind spots of embedding similarity.

**Lyrics plus audio.** A fuller approach should combine text, structure, harmony, timbre, and recurrence rather than pretending lyrics alone are the whole object.

---

## Method, briefly

All analyses use Python. Statistical claims include effect sizes, p-values, and null-model comparisons where appropriate. If a result is surprising, I try to say whether the surprise is informative, fragile, or merely artifactual.

I am a vinyl collector and a serious listener. That matters less as biography than as method: it forces me to be suspicious of neat computational answers to questions that are structurally hard.
