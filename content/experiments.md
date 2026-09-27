---
title: "Experiments & materials"
description: "Inspect the questions, code, recorded results, and limitations behind the writing."
layout: "page"
hideMeta: true
ShowToc: false
---

A result is easier to assess when you can inspect what produced it. These entries connect each question to its study materials and explain what kind of evidence is available. Start with the essays for the argument, then use the materials to examine it.

## What should an agent remember?

**Question:** When a reading of an album depends on distant songs, what survives different memory policies?

**Status:** Executed four-album pilot. Model-produced evidence cards and automated judgments; no independent human ground truth. This is a lyric-based representation, not an acoustic or listener-attention study.

Compare full context, recent tracks, a rolling summary, and selective evidence. The public package includes source-addressed paraphrases, recorded outputs, aggregation code, and usage accounting. Offline analysis reads recorded outputs; live reproduction makes provider calls and incurs usage costs.

[Read Part I](/post/2026-09-11-langgraph-mlflow-album-memory/) · [Download materials](/examples/album-memory-study.zip) · [Design and instructions](https://github.com/carlosjimenez88M/carlosjimenez88m.github.io/tree/master/research/album-memory)

## Does adaptive memory justify its cost?

**Question:** Do additional retrieval and verification steps improve evidence use enough to justify their resource use?

**Status:** Executed comparison with a subsequent qualitative editorial review. The online verifier is fallible; acceptance is not independent correctness. A blind human-review packet is available, but no completed human ratings are reported.

The study follows complete trajectories as well as final context. It includes failures of attribution and a chronology problem in a reference question. Those limitations matter when interpreting the policy averages.

[Read Part II](/post/2026-09-11-memory-is-not-context/) · [Download materials](/examples/album-memory-v2-study.zip) · [Inspect the blind review packet](/examples/album-memory-v2/human-review.html) · [Recorded policy means](/examples/album-memory-v2/policy_means.csv) · [Instructions](https://github.com/carlosjimenez88M/carlosjimenez88m.github.io/tree/master/research/album-memory-v2)

The review packet runs locally in the browser and lets you download your ratings. It does not submit them to me. The proposed Part III routing and budget experiments have not been executed.

## What happens when lyrics change language?

**Question:** Do thresholded embedding similarities and a model judge agree about continuity in multilingual lyrics?

**Status:** Exploratory analysis of one album, with a revised interpretation dated September 27, 2026. Published aggregate exports can be inspected; this editorial revision did not rerun embedding generation or validate the judge against human readers.

[Read the Aquamosh study](/post/2026-05-20-aquamosh-quadrilingual-anatomy/) · [Cross-model aggregate table](/examples/editorial-review-2026/cross_model_invariance.csv) · [Judge comparison table](/examples/editorial-review-2026/llm_judge_stratified.csv)

## What do “attention windows” measure?

**Question:** Does persistence above an embedding-similarity threshold describe narrative continuity?

**Status:** Exploratory study with a numerical provenance issue under review. The current article withdraws the earlier headline inference and separates an inspection of an existing CSV from a reproduced experiment.

[Read the revised Attention Windows note](/post/2026-02-10-attention-windows-beatles-floyd/) · [Read the correction log](/research-notes/)

## How should a prompt change reach production?

**Question:** What evidence belongs between a changed prompt and a release?

**Status:** Educational walkthrough using synthetic support documentation. It proposes an engineering workflow; it does not report client results or measured production improvements.

[Read the MLflow walkthrough](/post/2026-09-08-mlflow-prompt-engineering/) · [Inspect the example code](/examples/mlflow_prompt_workflow.py)

## Use the same discipline on your application

A musical example can suggest a failure mode. It cannot establish how often that failure occurs in your system. A [memory and evidence diagnostic](/work-with-me/) starts from your workflow, cases, and the decision your team needs to make.
