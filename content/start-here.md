---
title: "Start Here"
layout: "page"
description: "New to the blog? This is where to start — what this blog is about, and reading paths organized by what you're building."
---

## What this is and why it exists

The gap between a model that passes a notebook cell and one that runs at 2am without breaking is wider than most people are told. Closing that gap requires a specific kind of thinking — systems design, operational discipline, and a willingness to engage with the unglamorous parts of production ML. That's what this blog is about.

I'm Carlos. My background includes the United Nations, IDB, Yale, Mercado Libre, and Globant — plus a few years teaching ML across Latin America and Spain. The through-line is the same problem in different institutional forms: making machine learning systems that actually work, for the people who need to use them.

I write here when I have something specific enough to be useful. Topics cluster around three technical pillars — LLMOps, AI software engineering, and GCP — and one parallel research thread that I find equally interesting: applying NLP and LLMs to music analysis.

---

## Reading paths

### LLMOps — MLOps for the LLM era

The MLOps discipline that emerged around 2020 was designed for classical ML: feature pipelines, model registries, batch inference, drift detection on tabular data. LLMs broke most of those assumptions. This is where I work through what the new discipline looks like.

- [Anatomy of an MLOps Pipeline — Part 1: Orchestration](/post/anatomia-pipeline-mlops-part-1-en/) — start here: Hydra + MLflow, reproducible pipelines, 7-step GitHub-triggered pipeline from scratch
- [Anatomy of an MLOps Pipeline — Part 2: Deployment](/post/anatomia-pipeline-mlops-part-2-en/) — CI/CD, containerization, rollback in 30 seconds
- [Anatomy of an MLOps Pipeline — Part 3: Production](/post/anatomia-pipeline-mlops-part-3-en/) — monitoring, drift detection, production readiness
- [MLflow for Generative AI Systems](/post/mlflow_genai/) — tracing, evaluation, and versioning when "the model" is code + prompts + tools

### AI Software Engineering

Training a model is 20% of the work. The other 80% is the system around it. These posts cover that layer.

- [AI Architecture: Training and Inference](/post/2026-04-05-ai-architecture-training-inference/) — when to use CPU, GPU, TPU, or edge hardware. Real cost data. Why inference costs 15–20× more than training over a model's lifetime.
- [Statistical Learning: Foundations, Bias-Variance and the Art of Estimation](/post/2026-03-12-statistical-learning-foundations/) — the math that underlies every model evaluation. Derived from first principles, not borrowed from a textbook summary.

### Music & AI

What happens when you apply NLP, embeddings, and LLMs to music? I've been exploring this seriously — with statistical rigor and genuine curiosity about where the tools fail.

- [Attention Windows: Narrative Cognitive Load in Beatles vs Pink Floyd](/post/2026-02-10-attention-windows-beatles-floyd/) — the founding piece of this thread. A novel framework for measuring semantic persistence in lyrics, an unexpected result, and a theoretical account of why transformer embeddings systematically fail at abstract thematic analysis.

→ [Full Music & AI section](/music-analysis/)

---

## How to follow

The best formats are [RSS](/index.xml) and the [newsletter](/follow/). I write when I have something worth saying — not on a fixed schedule.

[GitHub](https://github.com/carlosjimenez88M) · [LinkedIn](https://www.linkedin.com/in/djimenezm) · [X](https://x.com/DanielJimenezM9)
