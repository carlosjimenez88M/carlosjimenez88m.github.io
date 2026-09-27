---
title: "Narrative Arcs in Music"
layout: "hub"
description: "How songs and albums develop ideas, and how computational interpretations can be examined against their evidence."
hubTitle: "Writing on music and interpretation"
hubDescription: "Essays on narrative, multilingual lyrics, and the limits of the measurements used to study them."
hubCategories: ["Music Analysis"]
hubTags: ["music-analysis", "computational-musicology"]
---

## How does an album develop an idea?

A theme can return with different words. A new voice can complicate what an earlier song appeared to say. An ending can resolve a tension, preserve it, or make us reconsider the beginning.

I study those relationships through close reading, NLP, embeddings, and LLM-based experiments. The question is whether a computational account preserves the distinctions that matter to an interpretation — and what evidence would let us disagree with it.

Music is a subject of this research in its own right. It also provides demanding examples for [AI engineering](/ai-engineering/): memory, source attribution, multilingual retrieval, and the cost of reasoning over evidence.

## Start with the album-memory studies

[What Should an Agent Remember?](/post/2026-09-11-langgraph-mlflow-album-memory/) compares four ways of retaining evidence across four albums. It asks what is lost when an interpretation depends on songs far apart in the sequence.

[Memory Is Not Context](/post/2026-09-11-memory-is-not-context/) follows that question through retrieval, verification, and resource accounting. An answer may satisfy a model verifier while still making an unsupported claim.

These studies work with model-produced paraphrases of lyrics. They do not observe the full musical performance, and they do not establish definitive narratives for the albums.

[Inspect the designs, outputs, and review materials →](/experiments/)

## Earlier explorations

[When Lyrics Change Language: Aquamosh](/post/2026-05-20-aquamosh-quadrilingual-anatomy/) examines the association between language transitions and similarity thresholds. Its revised interpretation distinguishes an automated judge from human validation.

[Attention Windows: Beatles and Pink Floyd](/post/2026-02-10-attention-windows-beatles-floyd/) now documents an unresolved numerical provenance issue and explains why a similarity statistic cannot simply be interpreted as listener attention.

Both articles have dated entries in the [correction log](/research-notes/).

## Questions still open

**Interpretation and agreement.** Which relationships can readers support consistently, and where should disagreement remain part of the result?

**Larger and more varied corpora.** How much of an observed pattern belongs to selected songs, languages, or genres?

**Lyrics and sound.** What changes when harmony, timbre, recurrence, performance, and transitions enter the evidence?

**Application.** Which failures suggested by musical examples also occur in a particular retrieval or agent system? That transfer needs its own evaluation.

I am a vinyl collector and a serious listener. The listening keeps the computational representation in perspective: a model sees what we give it, while a musical work can hold more than the experiment observes.
