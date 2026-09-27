---
author: Carlos Daniel Jiménez
date: 2026-05-20
lastmod: 2026-09-27
title: "When Lyrics Change Language: An Embedding Study of Aquamosh"
description: "An exploratory comparison of embedding thresholds and a model judge in multilingual lyrics, with explicit limits on what their disagreement establishes."
categories: ["Music Analysis", "LLMs"]
tags: ["llms", "nlp", "music-analysis", "embeddings", "computational-musicology", "code-switching", "audio-embedding", "labse", "openai", "google-clap"]
series: ["NLP", "LLMs", "Embeddings", "Computational Musicology"]
images: ["/img/social/aquamosh.png"]
socialImageAlt: "When lyrics change language. An exploratory embedding study of Aquamosh."
editorialNote:
  date: "September 27, 2026"
  text: "This revision withdraws the earlier claims of architectural impossibility and human-equivalent validation. It checks descriptive tables against saved exports and narrows their interpretation. No embedding or judge calls were rerun, and no new human ratings were collected."
readerGuide:
  summary: "A change of language need not mean a change of subject. In the saved analysis of Aquamosh, all five tested embedding configurations cross their break threshold more often at language switches. A separate LLM judge sometimes disagrees. That is a reason to investigate the measurement and its target, not proof that either model has the definitive reading."
  scope: "One album, 382 adjacent lyric pairs, model-specific thresholds, and an automated comparison judge. The descriptive association does not establish a causal language effect or general performance in other applications."
  resources:
    - label: "Cross-model table"
      url: "/examples/editorial-review-2026/cross_model_invariance.csv"
    - label: "Judge comparison"
      url: "/examples/editorial-review-2026/llm_judge_stratified.csv"
    - label: "Provenance"
      url: "/examples/editorial-review-2026/provenance.json"
---

A song can change language while continuing an image, a joke, or an argument. It can also stay in one language while changing subject. *Aquamosh*, by Plastilina Mosh, offers a setting for asking how a similarity-based representation handles those possibilities.

The question is narrower than whether a model understands the album: **how do thresholded embedding similarities vary with language transitions, and where do they disagree with another model's reading?**

## What changed in this revision

The previous title claimed a falsification of the distributional hypothesis. Its interpretation also treated GPT-4o-mini as equivalent to a human reference and generalized the results to commercial recommendation, moderation, and support systems. Those claims exceed the evidence available from this experiment and are withdrawn.

The current account retains descriptive results checked against existing aggregate exports. A file comparison is not an end-to-end reproduction: it does not validate lyric collection, labels, model calls, or all downstream statistics. The earlier article is retained in [repository history](https://github.com/carlosjimenez88M/carlosjimenez88m.github.io/blob/29082f64450113a170e77f65c3c72427872fcdba/content/post/2026-05-20-aquamosh-quadrilingual-anatomy.md) as a historical record containing superseded claims.

## Corpus and measurement

The original lyric analysis reports 392 lines from ten transcribed tracks, yielding 382 adjacent within-track pairs. The comparison tables contain 183 same-language pairs and 199 language-switch pairs. Missing lyric tracks, language detection, and the treatment of mixed-language lines constrain interpretation.

For each pair, the procedure compares cosine similarity with a model-specific threshold. A *break* means that similarity falls below that threshold. It does not independently establish a narrative rupture.

The recorded calibration rule used the median similarity of sampled random pairs plus one standard deviation. This produces different thresholds in different embedding spaces. Applying the same rule makes the procedure explicit; it does not guarantee equivalent decision quality or a fair accuracy comparison across models.

## The descriptive pattern in the saved exports

The following values are rounded from the [cross-model CSV](/examples/editorial-review-2026/cross_model_invariance.csv). The ratio divides the switch break rate by the same-language break rate.

| Configuration | Threshold | Break rate: same language | Break rate: switch | Ratio |
|---|---:|---:|---:|---:|
| OpenAI `text-embedding-3-large` | 0.320 | 0.361 | 0.698 | 1.94× |
| LaBSE | 0.323 | 0.410 | 0.648 | 1.58× |
| BGE-M3 | 0.518 | 0.426 | 0.774 | 1.82× |
| Multilingual E5-large | 0.856 | 0.372 | 0.487 | 1.31× |
| Multilingual MPNet | 0.440 | 0.410 | 0.734 | 1.79× |

Every tested configuration shows a higher break rate at language switches in these pairs. That repeated direction is an observation about this corpus, these thresholds, and these models. The sizes differ substantially.

The comparison does not isolate language as the cause. Language switches may coincide with changes of speaker, topic, song position, or other features. Models also differ in multiple ways, so a smaller gap cannot by itself identify the effect of a particular training strategy. Five configurations on one album do not establish architecture-invariant failure.

## A judge model is another measurement

The original procedure asked GPT-4o-mini whether the second line continued the theme, image, or action of the first. Its prompt explicitly instructed it not to treat a language switch alone as a topic change.

The stored field called `false_break_rate` counts pairs where the embedding rule says break and the judge says continuity, divided by all pairs in the stratum. The name assumes the judge is correct. Here I describe it as **break/continuity disagreement**.

| Configuration | Same-language disagreement | Switch disagreement | Switch / same |
|---|---:|---:|---:|
| OpenAI `text-embedding-3-large` | 0.060 | 0.191 | 3.18× |
| LaBSE | 0.098 | 0.131 | 1.33× |

Source: [judge comparison CSV](/examples/editorial-review-2026/llm_judge_stratified.csv). Ratios use the unrounded values.

This records disagreement between two automated procedures. It does not establish a human error rate, demonstrate that one reading is correct, or show how frequently a deployed retrieval system fails. The stored kappa values likewise describe agreement with that judge, not agreement between independent human readers.

Research on [LLM-as-a-judge](https://arxiv.org/abs/2306.05685) documents both useful agreement and biases in the evaluated settings. Agreement in those settings does not validate this judge on multilingual lyrics. A stronger comparison would include independent bilingual readers, a task-specific rubric, explicit ambiguous cases, and adjudication of disagreements.

## The musical question behind the numbers

Language can do more than carry a topic. It can change register, create distance, introduce a quotation, or move between cultural references. Those are interpretive possibilities to investigate through close reading and listening.

An adjacent-pair label compresses that problem into a binary decision. The compression may be useful for a narrow evaluation, but it cannot recover all the relations in a track or album. The difference between maintaining an image and developing an argument should not disappear simply because both receive a continuity label.

The relevant next question is therefore case-level: which pairs produce disagreements, what evidence supports each reading, and which distinctions does the rubric miss? A mean break rate cannot answer that on its own.

## Other analyses and their limits

The original project also explored language/semantic-field associations, track-level regressions, cultural-axis projections, critics' text, and audio embeddings. Those artifacts remain in the repository, but this revision does not certify all of their numerical or inferential claims.

Several boundaries are especially important:

- A regression with observational controls is not a randomized intervention on language. Track dependence and a small number of clusters affect inferential interpretation.
- Semantic fields assigned by a model require their own validation. They do not establish the artist's intention.
- A coordinate on an anchor-defined axis is a model-dependent projection. Describing an axis as emotional or ironic does not make it a direct measure of emotion or irony.
- The absence of a statistically significant correlation between lyric and audio projections does not prove their independence. It also does not establish a producer's causal influence or intention.
- Comparing projections from different representations requires checking whether they measure comparable constructs.

The earlier stronger assertions about these extensions have been removed from the current argument. They should remain research questions until the relevant evidence supports them.

## How to inspect the result

The public [provenance file](/examples/editorial-review-2026/provenance.json) identifies the source aggregate files and their SHA-256 hashes. The copied tables contain aggregate statistics rather than full lyrics or audio. The [inspection script](https://github.com/carlosjimenez88M/carlosjimenez88m.github.io/blob/master/scripts/audit_editorial_exports.py) packages those files without generating model judgments.

Reading those tables verifies what the saved outputs say. Reproducing the experiment would additionally require appropriate access to the source material, a documented environment, frozen preprocessing and thresholds, and repeated model stages where relevant. These steps were not performed for this editorial update.

## A next experiment worth doing

I would begin with a reviewed set of multilingual pairs that distinguishes continuation, contrast, unrelated content, and uncertain cases. Where feasible, matched versions could hold the intended relationship constant while changing language. Human review would check that the transformation preserved the relevant meaning.

A held-out evaluation could then compare the existing embedding rule with alternatives under a common task and cost account. Thresholds would be chosen before examining the test results. Sampling across more songs, artists, and genres would be needed for broader claims.

For an application team, the corresponding practical question is whether its retrieval process preserves the relationships users need. The answer must be measured on that application's cases. This album study offers a way to formulate the question; it does not supply a deployment-wide failure rate.

Continue with the [album-memory experiments](/experiments/), which shift from adjacent similarities to what evidence an agent can retrieve and support in an answer.
