---
title: "Start Here"
layout: "page"
description: "A guided map to the blog's new center of gravity: AI Software Engineering, LLMOps, GCP, edge systems, agentic AI, and music analysis with NLP and LLMs."
---

## What changed

This blog used to read like a broad notebook on machine learning systems. It is becoming something more focused.

The center is now **AI Software Engineering**: the discipline of building AI systems that are understandable, reproducible, observable, and production-ready once models meet infrastructure, interfaces, and real users.

Inside that center, four themes matter most here:

- **LLMOps as the evolution of MLOps**
- **GCP as an operating context for production AI**
- **Edge computing and agentic AI**
- **Music analysis with NLP and LLMs**

That is the map for everything else on the site.

---

## Reading paths

### 1. LLMOps: what changes when the model becomes part of a larger system

The MLOps discipline that emerged around 2020 was designed for classical ML: feature pipelines, model registries, batch inference, drift detection on tabular data. LLMs broke most of those assumptions. This is where I work through what the new discipline looks like.

- [Anatomy of an MLOps Pipeline — Part 1: Orchestration](/post/anatomia-pipeline-mlops-part-1-en/) — start here: Hydra + MLflow, reproducible pipelines, 7-step GitHub-triggered pipeline from scratch
- [Anatomy of an MLOps Pipeline — Part 2: Deployment](/post/anatomia-pipeline-mlops-part-2-en/) — CI/CD, containerization, rollback in 30 seconds
- [Anatomy of an MLOps Pipeline — Part 3: Production](/post/anatomia-pipeline-mlops-part-3-en/) — monitoring, drift detection, production readiness
- [MLflow for Generative AI Systems](/post/mlflow_genai/) — tracing, evaluation, and versioning when "the model" is code + prompts + tools

### 2. AI Software Engineering: the system around the model

Training a model is only one slice of the problem. The more durable work is in architecture, evaluation, testing, interfaces, cost, monitoring, and operational tradeoffs.

- [AI Architecture: Training and Inference](/post/2026-04-05-ai-architecture-training-inference/) — when to use CPU, GPU, TPU, or edge hardware. Real cost data. Why inference costs 15–20× more than training over a model's lifetime.
- [Statistical Learning: Foundations, Bias-Variance and the Art of Estimation](/post/2026-03-12-statistical-learning-foundations/) — the math that underlies every model evaluation. Derived from first principles, not borrowed from a textbook summary.

### 3. GCP, edge computing, and agentic systems

This is where the blog moves from abstract engineering language to concrete system constraints: hardware, latency, cost, deployment surface, and the problem of autonomy outside a single request-response loop.

- [GCP posts](/tags/gcp/) — architecture and infrastructure notes tagged around Google Cloud
- [Edge + Agentic AI](/edge-agentic-ai/) — the dedicated hub for this thread
- [Edge Computing and Edge Machine Learning](/post/edge-computing/) — a conceptual entry point into inference beyond centralized cloud assumptions
- [Raspberry Pi 16GB, Servers, and MLOps](/post/mlops-servers-raspberry/) — what edge infrastructure looks like when you treat small hardware seriously
- [MLops into Raspberry Pi 5](/post/mlops_raspberrypi5/) — hands-on infrastructure thinking for constrained environments

### 4. Music, NLP, and LLMs

This is the parallel line of work that keeps the technical writing honest. Music is a strong test case because language models often look smarter than they are when metaphor, repetition, symbolism, and structure are doing the real work.

- [Attention Windows: Narrative Cognitive Load in Beatles vs Pink Floyd](/post/2026-02-10-attention-windows-beatles-floyd/) — the founding piece of this thread. A novel framework for measuring semantic persistence in lyrics, an unexpected result, and a theoretical account of why transformer embeddings systematically fail at abstract thematic analysis.

→ [Full Music + NLP section](/music-analysis/)

---

## How to read the site now

If you want the clearest picture of the blog's editorial direction, go next to [AI Engineering](/ai-engineering/).

If you want the research line that is least like everyone else's AI writing, go to [Music + NLP](/music-analysis/).

If you want updates when something genuinely new is ready, the best formats are [RSS](/index.xml) and the [newsletter](/follow/). I write when I have something worth saying, not to satisfy a content calendar.

[GitHub](https://github.com/carlosjimenez88M) · [LinkedIn](https://www.linkedin.com/in/djimenezm) · [X](https://x.com/DanielJimenezM9)
