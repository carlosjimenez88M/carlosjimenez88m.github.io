---
title: "AI Engineering"
layout: "page"
description: "The core editorial line of this blog: AI Software Engineering, LLMOps, GCP, edge machine learning, and agentic systems."
---

## The center of gravity

This blog is now organized around **AI Software Engineering**.

That phrase is doing specific work. I do not mean "AI" as a synonym for prompting, nor "engineering" as a synonym for deployment scripts. I mean the full problem of building AI systems that remain legible, testable, cost-aware, and operationally reliable once they leave the notebook.

In practice, that breaks into four connected lines of work:

- **LLMOps as the successor to classical MLOps.** Once prompts, tools, traces, evaluations, and agents become part of the system, the old abstractions stop being enough.
- **Systems design on GCP.** Infrastructure choices shape latency, reliability, cost, and team velocity. I care about the engineering consequences of those choices.
- **Edge machine learning and inference.** Raspberry Pi, Jetson, constrained hardware, and the question of what happens when the cloud is not the whole story.
- **Agentic AI as a software architecture problem.** Coordination, observability, context handoffs, and failure modes between agents matter as much as model quality.

If you are specifically interested in hardware-constrained deployment and autonomous workflows, go next to [Edge + Agentic AI](/edge-agentic-ai/). If you want the cloud-platform angle, browse the [GCP-tagged posts](/tags/gcp/).

---

## Reading paths

### LLMOps and production discipline

- [Anatomy of an MLOps Pipeline - Part 1: Pipeline and Orchestration](/post/anatomia-pipeline-mlops-part-1-en/)
- [Anatomy of an MLOps Pipeline - Part 2: Deployment and Infrastructure](/post/anatomia-pipeline-mlops-part-2-en/)
- [Anatomy of an MLOps Pipeline - Part 3: Production and Best Practices](/post/anatomia-pipeline-mlops-part-3-en/)
- [MLflow for Generative AI Systems](/post/mlflow_genai/)

### Architecture, inference, and hardware tradeoffs

- [AI Architecture - Notions on Training and Inference](/post/2026-04-05-ai-architecture-training-inference/)
- [Edge Computing and Edge Machine Learning](/post/edge-computing/)
- [Raspberry Pi 16GB, Servers, and MLOps](/post/mlops-servers-raspberry/)
- [MLops into Raspberry Pi 5](/post/mlops_raspberrypi5/)

### Foundations that support the engineering work

- [Statistical Learning: Foundations, Bias-Variance and the Art of Estimation](/post/2026-03-12-statistical-learning-foundations/)

---

## What comes next

The direction from here is narrower and more deliberate:

- More on **LLMOps as an engineering discipline**, not just a tooling stack.
- More on **GCP-oriented architecture** for real production systems.
- More on **edge + agentic AI**, especially where hardware constraints reshape system design.
- More on **AI Software Engineering** as the umbrella that makes these threads cohere.

If you want the broadest orientation first, start with [Start Here](/start-here/).
