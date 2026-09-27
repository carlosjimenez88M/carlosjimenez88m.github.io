---
author: Carlos Daniel Jiménez
date: 2026-02-10
lastmod: 2026-09-27
title: "Attention Windows: What Embedding Similarity Can Tell Us About Beatles and Pink Floyd"
description: "An exploratory reading of lyrical similarity, with a correction on numerical provenance and the limits of interpreting a metric as attention."
categories: ["Music Analysis", "LLMs"]
tags: ["llms", "nlp", "music-analysis", "embeddings", "computational-musicology"]
series: ["NLP", "LLMs", "Embeddings", "Computational Musicology"]
editorialNote:
  date: "September 27, 2026"
  text: "The earlier headline inference is withdrawn pending reconciliation of its numerical provenance. This revision narrows the claims and documents a check of an existing CSV; it does not report a new model run or independent validation."
readerGuide:
  summary: "Can repeated similarity between lyric lines reveal a sustained idea? This study began with that question and two albums. Its most useful lesson is about the measurement: a thresholded embedding score needs validation before it can stand for narrative continuity, let alone a listener's attention."
  scope: "An exploratory lyric-based comparison. The numerical provenance issue below remains unresolved; no human attention or commercial recommendation system was measured."
  resources:
    - label: "File inspection and hashes"
      url: "/examples/editorial-review-2026/provenance.json"
    - label: "Later memory experiments"
      url: "/experiments/"
---

An album can revisit an idea through different words, voices, and images. It can also repeat the same words while changing what those words mean. That makes musical narrative an interesting setting for a measurement question: **what does similarity between lyric embeddings actually tell us?**

The original analysis compared lyrics from *Abbey Road* and *The Dark Side of the Moon*. I expected a representation of sustained themes to reflect my reading of Pink Floyd's album. That expectation is an interpretation to test, not a ground truth against which every different model result must be declared a failure.

## Correction and current evidence status

The previous article acknowledged that an earlier draft contained invented figures and stated that they had been replaced with computed results. That acknowledgment remains part of the record. The September 27 revision does not reconstruct the entire draft history or certify all of the previous analysis.

A direct inspection of the repository's `attention_windows_results.csv` also exposes a concrete provenance problem:

| Artist | Rows in the inspected CSV | Arithmetic mean of `attention_window` | Mean in the previous headline |
|---|---:|---:|---:|
| The Beatles | 403 | 0.412 | 0.57 |
| Pink Floyd | 208 | 0.053 | 0.25 |

The CSV inspection groups existing rows by artist and averages the stored values. **It does not establish which model configuration or threshold produced them.** Those identifiers are absent from this file. Different thresholds or versions could explain a discrepancy, but that explanation has not been established here.

Consequently, the previous headline comparison and its inferential claims should not be cited as validated findings. I have removed them from the current argument pending a reconciled corpus, configuration, and analysis. The values above document the discrepancy; they are not replacement estimates of narrative coherence.

[Inspection summary and source hash](/examples/editorial-review-2026/provenance.json) · [Inspection script](https://github.com/carlosjimenez88M/carlosjimenez88m.github.io/blob/master/scripts/audit_editorial_exports.py) · [Earlier article in repository history](https://github.com/carlosjimenez88M/carlosjimenez88m.github.io/blob/29082f64450113a170e77f65c3c72427872fcdba/content/post/2026-02-10-attention-windows-beatles-floyd.md)

The historical article contains superseded claims. It is linked to preserve the correction trail.

## Theoretical framework: attention windows

An *attention window* was the name given to the number of following lyric lines whose embeddings remain above a similarity threshold relative to an anchor line. The name should not be interpreted as a measurement of human cognition or transformer attention weights.

### Mathematical formulation

For normalized line embeddings $e_i$, one operational definition is:

$$W_i(\theta) = \max\{k : \operatorname{sim}(e_i,e_{i+j}) > \theta \text{ for every }j=1,\ldots,k\}.$$

The window is zero when the next line fails the criterion or no next line remains. It stays within a song. The threshold, normalization, treatment of repeated lines, and comparison operator must be recorded because they affect the result.

This is a statistic of an embedding representation and a rule. Establishing that it measures thematic continuity requires evidence beyond its mathematical definition.

## What the original question leaves open

There are several plausible explanations for a mismatch between a close reading and a similarity score:

- Repeated lyrics can make a similarity-based sequence persist without establishing a developing argument.
- A theme may recur through images whose relationships require context outside an adjacent pair.
- The critic's reading may be disputable or depend on evidence absent from the chosen representation.
- A particular model, threshold, or segmentation may be unsuitable for the task.
- Musical continuity may be carried by harmony, production, performance, or a transition that the lyrics do not capture.

These alternatives are not resolved by finding that two albums receive different mean scores. A contrary result can challenge the interpretation, the instrument, or both. A study must distinguish them rather than assume the preferred narrative is correct.

## Why the threshold matters

A high threshold tends to shorten windows; a lower one tends to lengthen them. The resulting curve can help describe the instrument's sensitivity. Selecting a threshold because it produces an attractive comparison is not independent validation.

A stronger design would choose the decision rule on separate development examples, retain a held-out evaluation set, and compare it against explicit human judgments of the particular relationship being studied. Those judgments should distinguish recurrence, contrast, change of stance, and unrelated content. Readers can reasonably disagree, so their disagreement belongs in the analysis.

Line-level examples also share songs and albums. Hundreds of lines do not amount to hundreds of independent albums. The unit of inference must match the claim.

## What needs to be reconciled before a new numerical claim

1. Freeze the album editions, song boundaries, lyric segmentation, and counts with an identifiable manifest.
2. Record model identifiers, embedding dimensions, preprocessing, thresholds, and the code revision for each export.
3. Recompute the reported tables and verify that figures use the same configuration.
4. Define and independently annotate the intended interpretive target, including ambiguous examples.
5. Report sensitivity to thresholds and dependence within songs, while keeping claims within the sampled albums.

These are outstanding requirements. This editorial revision does not claim they have been completed, and it does not introduce new p-values or accuracy estimates.

## What this means for engineering

A retrieval score is useful for selecting candidates. The more demanding question is whether the selected material supports the relationship asserted in the final answer. An application can expose a relevant source and still attribute it incorrectly or infer a progression the source does not establish.

The later [album-memory pilot](/post/2026-09-11-langgraph-mlflow-album-memory/) makes that evidence problem explicit. [Memory Is Not Context](/post/2026-09-11-memory-is-not-context/) then examines the resource use and verifier failures behind an answer. Those studies have their own stated limitations; they do not repair or validate the unresolved estimates in this earlier analysis.

There is no basis here for a universal claim that embeddings cannot represent meaning, that a different training method cannot improve the task, or that a commercial recommendation service favors one musical style because of this metric. Those claims require different experiments.

The question remains worthwhile: how can a computational representation help us examine an interpretation while preserving the ability to disagree with it? Progress begins by making the representation, evidence, and uncertainty visible.
