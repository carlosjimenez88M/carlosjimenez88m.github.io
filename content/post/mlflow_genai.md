---
author: Carlos Daniel Jiménez
date: 2025-10-08
title: MLflow for Generative AI Systems
categories: ["MLOps", "Agentic AI"]
tags: ["mlflow", "genai", "llms", "tracing", "evaluation"]
series:
  - mlops
  - genai
---


# MLflow for Generative AI Systems

I'll start this post by recalling what Hayen said in her book **Designing Machine Learning Systems (2022): 'Systems are meant to learn'.** This statement reflects a simple fact: today, LLMs and to a lesser extent vision language models are winning in the Data Science world. But how do we measure this learning? RLHF work is always a good indicator that perplexity will improve, but let's return to a key point: LLMs must work as a system, therefore debugging is important, and that's where the necessary tool for every Data Scientist, AI Engineer, ML Engineer, and MLOps Engineer comes in: MLflow.

## What problems can MLflow solve at the Generative AI level?

Working with GenAI is qualitatively different from traditional ML:

- **You can't debug easily.** A single query can trigger 15 different operations (LLM → retrieval → ranking → LLM again). When something fails, where?
- **Traditional metrics don't apply.** There's no single **"correct value"** — remember this always. The answer can be good in multiple ways.
- **A "model" is no longer a file.** It's code + prompts + tools + configuration. Everything must move together.

## Three new capabilities

### 1. Tracing: Seeing what's happening

The idea is simple: capture each step of your GenAI application as a hierarchical structure.

```python
import mlflow

# For LangChain or LlamaIndex: one line
mlflow.langchain.autolog()

# For your custom code: a decorator
@mlflow.trace 
def my_retrieval_function(query):
    # your logic here
    return results
```

This gives you:

- A view of all steps in order
- Latency of each operation
- Tokens consumed (costs) → important for control and evaluation of each run at the operation level
- Inputs and outputs at each point

**When to use it:** Always. The overhead is minimal and the visibility is invaluable.

### 2. Systematic evaluation with LLMs

You can't use accuracy to evaluate "is this response useful?". MLflow uses LLMs as evaluators.

There are four approaches, ordered by complexity:

1. **Predefined scorers** → Standard metrics already configured
    
    ```python
    from mlflow.metrics.genai import relevance
    ```
    
2. **Guidelines** → Rules in natural language
    
    ```python
    guidelines = "The response must be concise and not mention prices"
    ```
    
3. **make_judge()** → Your own custom evaluator
    
    ```python
    judge = mlflow.metrics.genai.make_judge(
        definition="Evaluate if the response directly answers the question"
    )
    ```
    
4. **Agent-as-a-Judge** → Evaluates the complete process, not just the output
    
    ```python
    judge = make_judge(
        definition="Evaluate if the agent used the correct tool",
        trace_aware=True
    )
    ```

You can also capture human feedback and attach it to traces. This is useful for:

- Validating your automated evaluators
- Building evaluation datasets from real cases
- Iterating on problems that users actually encounter

### 3. Versioning complete applications

The problem: your "model" is now Python code that defines an agent. Serializing the object doesn't work well.

The solution: version the code directly.

```python
mlflow.langchain.log_model(
    lc_model="agent.py",  # Path to file
    artifact_path="agent"
)
```

MLflow packages:

- The source code
- Dependencies (conda.yaml, requirements.txt)
- Everything needed to reproduce the behavior

When you load the model, you get exactly what you saved.

**Bonus:** Model registry webhooks for automation. When you mark a version as "production", you can automatically trigger your deployment pipeline. I'll talk about MLproject, which is a contract manager, in another post.

## How to get started

**Step 1:** Add tracing to your existing application

```python
mlflow.langchain.autolog()
```

**Step 2:** Run your application and look at traces in the MLflow UI. Identify bottlenecks.

**Step 3:** Set up basic evaluation with predefined scorers

```python
result = mlflow.evaluate(
    model=my_agent,
    data=test_cases,
    metrics=[relevance(), correctness()]
)
```

**Step 4:** When you have a working agent, version it with "Models from Code"

**Step 5:** Iterate. Add custom judges as you understand which metrics really matter for your use case.

## Limitations and trade-offs

Some things you should know:

- **LLM-as-a-Judge isn't perfect.** Evaluators can be biased or wrong. Always validate with human feedback on a sample.
- **Tracing adds overhead.** It's small, but it exists. For high-volume production, consider sampling and especially post-stratified representation issues.
- **Evaluation has a cost.** Each evaluation with an LLM consumes tokens. For large datasets, this adds up.
- **Integration isn't magic.** If you use unsupported libraries, you need manual tracing.

## Why this matters

I've seen teams build their own ad-hoc solutions for these problems. They usually end up with:

- Custom logging scripts that nobody else understands
- Manual evaluation that doesn't scale
- Versioning that consists of "copying the folder with a timestamp"

MLflow isn't perfect, but it provides sensible primitives for these common problems. It's better to have an imperfect system that everyone uses than ten perfect systems that nobody shares.

## Resources

- Tracing documentation: `mlflow.org/docs/latest/tracing`
- Evaluation guide: `mlflow.org/docs/latest/llm-eval`
- Examples on GitHub: `mlflow/mlflow-examples`

