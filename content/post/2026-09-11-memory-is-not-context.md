---
author: Carlos Daniel Jiménez
date: 2026-09-11T00:00:00-05:00
lastmod: 2026-09-11T11:02:07-05:00
title: "Memory Is Not Context: Token Budgets, Narrative Relations, and Agent Economics"
description: "Context optimization is not agent optimization: a critical four-album study of retrieval, verifier failures, and whole-trajectory accounting with LangGraph and MLflow."
categories: ["Engineering", "Music Analysis"]
tags: ["langgraph", "mlflow", "memory", "evaluation", "narrative", "context-engineering"]
draft: false
---

An agent answered a question about *Infest* with 2,326 tokens of memory in its final context. Getting to that answer required thirteen model calls and 19,831 input and output tokens. The verifier accepted it. One of its claims attributed evidence from the fourth song to the first.

That execution contains the central problem of this study. A smaller context can conceal a more expensive decision process, and an accepted answer can conceal an unsupported interpretation. Neither becomes visible if the experiment ends at the final prompt or the aggregate score.

In [Part I](/post/2026-09-11-langgraph-mlflow-album-memory/), I examined how much evidence about an album survives different memory policies. Here the question becomes more demanding: **does representing relations between songs, and letting an agent expand its context, improve the evidence it can use enough to justify the additional work?**

The result challenges the engineering hypothesis. It also challenges the instrument used to evaluate it.

## I. Results at a glance

Each setting answers the same 24 questions: 20 positive questions and four questions whose premises are not established by the source cards. “Acceptance” below means the positive answer passed an online model verifier and a structural gate. It does **not** mean independently established correctness. Trajectory tokens include all generation, sufficiency and verification calls, averaged over all 24 questions.

| Policy | Positive answers accepted | Mean trajectory tokens |
| --- | ---: | ---: |
| Full source cards | 15/20 · 75% | 2,890 |
| Embedding retrieval · 1,200-token cap | 16/20 · 80% | 2,249 |
| Embedding retrieval · 1,600-token cap | 18/20 · 90% | 2,713 |
| Adaptive relational memory · threshold 3 | 18/20 · 90% | 10,336 |

**Adaptive memory matched the best observed embedding setting's acceptance while consuming approximately 3.8× its trajectory tokens.** Under those two measured quantities, the simpler retriever dominates this controller.

There are two immediate qualifications. The 1,600-token setting was selected after inspecting the budget grid; it is an observed comparison point, not a validated optimum. More seriously, the adaptive controller uses the online verifier to decide whether to retry. Its score measures success at satisfying that checker through bounded search. It is not an independent evaluation of the resulting answer.

A previous deterministic attribution screen reduced adaptive acceptance to 85% while leaving embeddings at 90%. The deeper review in this revision finds semantic failures in embeddings too. Neither percentage should be promoted to an accuracy claim.

{{< figure src="/img/album-memory-v2/pareto-frontier.svg" alt="Observed online acceptance versus complete trajectory tokens. Embeddings at 1600 and adaptive both reach 18 of 20, at substantially different resource use." caption="The empirical frontier describes this evaluator and these observations. It is not a validated frontier of interpretive correctness." >}}

The useful conclusion is therefore narrower than a ranking of memory architectures: **context optimization is not agent optimization.** Context size is one resource inside a trajectory. The evidence standard must apply to the answer produced at the end of that trajectory, including the relations its prose asserts.

## II. Why use albums to investigate agent memory?

My interest in musical and argumentative arcs begins with a distinction that retrieval systems often flatten: recurrence is not development. Two songs can discuss loss without occupying successive stages of recovery. A later expression of gratitude can coexist with grief. A different voice can introduce resistance without continuing the same protagonist's story.

An argumentative arc requires more than shared vocabulary. A position encounters a qualification, an alternative, or counterevidence. The analytical task is to identify that relationship and explain why its sources support it. An album gives the problem an explicit order while leaving its interpretation open to disagreement.

The corpus contains *Californication*, *Infest*, *Abbey Road*, and *De todas las flores*: 56 analytical units, including “Tightrope” separately from “Thrown Away.” Each unit contributes three paraphrased source claims with addresses such as `T04:C2`. The experiment operates on those **168 model-produced claims**, inherited from Part I. They are silver annotations, not human ground truth.

This boundary matters. The inputs do not represent harmony, timbre, performance, production or transitions. In *Abbey Road*, continuity can be carried by the music while the lyrics change characters. English paraphrases of Natalia Lafourcade's Spanish lyrics introduce another layer of interpretation. Even full context means all the cards, not all the evidence a listener could use.

The study consequently tests an agent's handling of a bounded analytical representation. It does not discover an album's definitive narrative, validate the annotations against the lyrics, or establish four general musical topologies from four chosen examples.

The engineering distinction has a useful precedent in [LongMemEval](https://arxiv.org/abs/2410.10813), which separates indexing, retrieval and reading in conversational memory. I borrow that decomposition, not its benchmark validity or results. Here we also need to distinguish reading a source from defending a relation between sources.

## III. What was actually compared

The memory policies share the answer model, question and source-card archive. What changes is selection and, for the adaptive policy, the path taken before stopping.

| Policy | Selection and exposed representation |
| --- | --- |
| Full | All compact source cards |
| Recent | Last three source cards |
| Summary | Part I's rolling summary, retaining its provenance limitations |
| TF-IDF | Whole cards ranked by lexical relevance |
| Embedding | Whole cards ranked by semantic similarity |
| Relational | Ranked relation hypotheses plus their supporting cards |
| Adaptive relational | Relational retrieval with expansion after insufficient evidence or failed verification |

The three ranked retrieval methods use memory caps of 400, 800, 1,200, 1,600 and 2,400 tokens. Cards are packed whole; the instruction and question are additional input. Relation prose consumes the same budget as source evidence. Recent and summary are single controls, not repeated under artificial budget labels.

The relation memory contains twelve generated edges per album. Each edge has two source addresses, a type—return, contrast, transformation or counterpoint—and an interpretive explanation with a caution. Edges are constructed before the new questions and carry no numerical confidence. An edge is a hypothesis derived from the cards, not independent corroboration of them.

{{< figure src="/img/album-memory-v2/relation-memory.svg" alt="Twelve source-addressed relation hypotheses per album connect track positions, colored by interpretive relation type." caption="The tested relation memory. Arc height is layout, not confidence or evidential strength. Sparse generated edges can omit the relationship a question needs." >}}

Retrieval uses `text-embedding-3-small` through separate card and relation namespaces in LangGraph's `InMemoryStore`. Relation ranking also includes a fixed preference for types relevant to the public task. Neither retriever receives the designated answer IDs.

The adaptive controller starts at 400 tokens and can visit 800, 1,200, 1,600, 2,400 and complete relational memory. A sufficiency model assigns an ordinal score from zero to four. At the primary threshold of three, the controller generates when evidence appears sufficient, then verifies the answer. Failure can trigger another expansion. The process terminates at the final level; it has a round limit but no hard cumulative-token limit.

There are six questions per album: local adjacency, distant comparison, transformation, contradiction, an ordered three-song relation, and an unsupported premise requiring abstention. The answer contract requests at most 90 words and four atomic claims, each citing exact source addresses. Questions and expected answers are model-generated. A separate model challenged the proposed negative cases against the complete cards, which improves screening without making them human-validated negatives.

The primary grid contains 456 trajectories. Two additional adaptive thresholds add 48, for 504 total. These are **24 questions evaluated repeatedly under different settings**, not 504 independent research examples. Each question-setting has one generated outcome; this grid does not estimate variation from repeated hosted-model execution.

## IV. The evaluator became part of the result

The original article acknowledged one false positive. That was necessary, but insufficient. It left the impression that correcting one source-ID mismatch might recover a trustworthy ranking.

For this revision, a Codex AI assistant reviewed all **72 final responses** from full context, embedding retrieval at 1,600 tokens, and adaptive retrieval at threshold three. The review read answer prose, structured claims and the complete source-card claims, checking support, attribution, the requested relation, temporal scope, interpretive restraint and abstention. Its [case-by-case notes](/examples/album-memory-v2/editorial-audit.json) are published.

The separate [blind-review packet](/examples/album-memory-v2/human-review.html) contains **24 Full + 24 Embedding1600 + 24 Adaptive responses**, shuffled with policy labels and automated scores withheld. Including the economical baseline matters: reviewing only full context and adaptive memory would leave the comparison that motivates this article unexamined. The form separates evidence support, attribution, the requested relation and narrative defensibility, plus whether the answer should have abstained. No human ratings have been collected.

The completed AI audit is a **post-hoc, non-blind editorial review**, with prior access to policy results. It is not human review, an independent adjudication, or a new gold standard. I report concrete disagreements rather than turning its judgments into a replacement accuracy percentage. The findings are inspectable precisely because the source addresses and original verdicts remain available.

### A correct address can support the wrong interpretation

The expensive *Infest* response assigns societal manipulation to T01 while citing `T04:C2`. The verifier accepts that inconsistent mapping. An explicit-ID check catches it, reducing the primary adaptive result from 18 to 17 accepted positives.

But consider the embedding answer to Natalia Lafourcade's three-song question. Its prose describes T08 as nostalgia and loss. Its cited `T08:C1` says that distance does not diminish closeness. Those are different propositions. The verifier explicitly treats the latter as supporting the former. The source address is valid, so the earlier ID-alignment screen misses the error.

This changes how I interpret the embedding result. It remains the economical baseline under the recorded evaluator; it has not demonstrated 90% correctness. The failure is in the semantic binding between prose and evidence, where a syntactically valid citation provides little protection.

### Contrasting emotions do not refute a linear arc

For Natalia's contradiction question, full context and embeddings describe early solitude in T01 followed by gratitude in T10. Adaptive contrasts sorrow in T02 with celebration in T12. Either pair can be compatible with a straightforward recovery narrative. To challenge that narrative, the answer must explain a return, persistence, reversal or coexistence that the proposed linear reading cannot adequately account for.

The embedding answer's claim of counterevidence is therefore under-argued even though its two source addresses are sensible. The problem is not solved by finding more relevant text. The model must establish the requested relation rather than repeat the question's label.

The verifier adds another failure: it rejects the full-context Natalia response with a supposed lack of distinct, sufficiently separated tracks, despite the T01/T10 pair. The full and embedding responses have the same main answer and claim list, with different limitation text, yet receive different verdicts. This is a concrete consistency problem, not a controlled estimate of stochastic judge variance.

### The benchmark can supply the wrong chronology

The *Abbey Road* transformation reference compares the burden in T15 with renewal in T07, calling T07 the later song. It is earlier. All three policies reproduce language suggesting movement from burden to hope.

The question permits a contrast between non-adjacent songs, so the pair need not be rejected. The error is converting that contrast into a forward album progression. The source cards support opposing stances; the sequence does not support the implied order.

This is an upstream measurement problem. Correcting only generated answers would leave a flawed reference and an invitingly directional task framing intact. Future evaluation needs to distinguish **contrast**, **ordered change**, and **continuity of a voice** instead of letting “transformation” stand for all three.

### Individually supported claims do not establish a three-song argument

The full and embedding *Californication* multi-hop answers describe an opening, a middle and an ending, but cite four tracks. T02 supplies identity; T07 supplies societal pressure. Their prose compresses those into one middle role.

The deterministic gate checks for at least three cited positions and sufficient endpoint distance. It cannot establish which single song occupies the middle role, or whether the prose follows the cited ordering. The model verifier nevertheless accepts the answers. The citations can each be locally true while their composition fails to identify the requested structure.

A stronger contract would name each selected role explicitly and validate its sources against that role. Merely requesting more atomic claims will not prevent their recombination into an unsupported sentence.

These cases support a methodological conclusion before an architectural one: the evaluator conflates source entailment, relation adequacy and narrative plausibility. Research on [LLM-as-a-judge](https://arxiv.org/abs/2306.05685) documents the usefulness and limitations of model evaluation; agreement reported on other tasks cannot validate this album-specific verifier. Here the failures are visible in its own explanations.

## V. Which questions make the controller expensive?

The aggregate hides two relevant distinctions: a policy can retrieve useful distant evidence while failing transformation questions, and it can spend heavily on a task without improving its acceptance.

{{< figure src="/img/album-memory-v2/task-acceptance-tokens.svg" alt="Two heatmaps compare four policies across six task types: accepted responses out of four and mean whole-trajectory tokens." caption="Four questions per cell. Positive-task columns use online acceptance; the abstention column uses correct negative refusals. One changed answer moves a cell by 25 percentage points." >}}

At the displayed 2,400-token setting, relational retrieval receives 4/4 acceptance on distant and multi-hop questions, but only 1/4 on transformation and contradiction. That is a reason to investigate specialization, not evidence that a relation graph generally improves compositional reasoning. Embeddings also receive 4/4 on multi-hop, and the narrative audit identifies weaknesses within supposedly successful outputs.

Adaptive contradiction questions consume a mean **15,803 trajectory tokens**, against **2,773** for embeddings. Their accepted counts are 2/4 and 4/4 respectively. Adaptive multi-hop questions consume 11,540 tokens versus 2,850, with equal 4/4 acceptance. The extra process does not earn a measured advantage in those cells.

Unsupported-premise questions are also expensive for adaptive retrieval: 11,813 tokens on average, with 3/4 correct abstentions. More search can be reasonable before declaring that evidence is absent, but this workload shows why refusal policy belongs in resource accounting. The *Californication* negative also exposes a conceptual error: lack of explicit proof that every tension resolves is not proof that every tension remains unresolved.

The heatmap does not establish general task difficulty. Each cell contains one question per album, the settings shown were inspected after the run, and the judge has documented errors. Its role is diagnostic: it tells us which executions deserve explanation and which apparent strengths require independent review.

## VI. MLflow makes the trajectory inspectable; it does not validate it

The implementation uses `mlflow.langchain.autolog()` and `mlflow.openai.autolog()`, a root `context_allocation` span, per-case runs, versioned prompt URIs and exported provider receipts. [MLflow's LangGraph integration](https://mlflow.org/docs/latest/genai/tracing/integrations/listing/langgraph/) supports graph tracing and additional child spans. Here the frozen result also records selected addresses, budget rounds, verifier outputs, models and prompt fingerprints.

Those records let us examine the *Infest* execution that opened this article:

{{< figure src="/img/album-memory-v2/mlflow-infest-trace.png" alt="Actual MLflow trace for the Infest distant adaptive case. Thirteen chat-model spans appear on the execution timeline; the header records 19,831 tokens. The first sufficiency response identifies missing second-half evidence." caption="The original Infest execution in MLflow 3.16.0, filtered to its thirteen chat-model spans. The selected first call explains why retrieval must expand; the header counts the complete trace. Success means the execution completed, not that its interpretation was correct." >}}

The screenshot makes the stopping problem concrete: the first call identifies a real absence of evidence, but that reasonable decision opens a path whose eventual cost and correctness must still be assessed. The UI’s rounded dollar estimate is not the dated pricing analysis discussed below.

The first sufficiency call declines to generate. At subsequent caps the controller generates, verifies and expands. The final exposed memory is 2,326 tokens, but that number omits the previous contexts, repeated instructions, sufficiency decisions, generated attempts and verdicts. The complete logical trajectory is 19,831 tokens.

A contrasting distant *Californication* execution finishes at the first 400-token cap, exposing 330 memory tokens and consuming 2,305 trajectory tokens. Adaptation can stop cheaply. The problem is the distribution of paths it actually takes, together with the reliability of its stopping rule.

{{< figure src="/img/album-memory-v2/trajectory-economics.svg" alt="Mean trajectory tokens split into sufficiency, answer generation and verification for four selected policies." caption="Control and verification are part of the evaluated system. Counting only the final generation would favor policies that move work into other calls." >}}

For AI engineering, the trace should answer distinct questions: which evidence reached each call, what justified another round, which prompt version produced the verdict, and how much work preceded termination? The score alone answers none of them. Conversely, a complete trace cannot make an unsupported verdict correct.

The review suggests an equally important distinction in assessment storage. The controller's stopping verdict, a deterministic attribution check, a post-hoc AI audit and a human judgment should retain separate identities. [MLflow's assessment API](https://mlflow.org/docs/latest/genai/assessments/feedback/) records feedback with its source and rationale, distinguishing model judges, code and humans. An AI review belongs under a model source, not `HUMAN`.

In this release, the new audit is a separate public artifact keyed to the original case IDs; it has **not** been inserted into the historical traces as human feedback. Original scores remain unchanged. That preserves what the controller actually saw rather than rewriting its execution with knowledge acquired afterward.

## VII. A smaller prompt is only one part of the budget

I use input plus output tokens as the primary resource measure because receipts make that quantity auditable. It is not a currency conversion: the generator and verifier use different models, and input, cached input and output can carry different prices.

A dated dollar calculation would sum, per model, **uncached input × input price + cached input × cached price + output × output price**. Cached input must first be subtracted from total input to avoid counting it twice. The 3.8× token ratio must not be relabeled a 3.8× dollar ratio without that calculation.

The experiment also distinguishes logical trajectory use from unique provider expenditure. Completed calls can be reused during resumption or across deterministic prefixes. A policy's logical trajectory still includes the calls its execution requires, even if the research replay reused a receipt. Preparation—embeddings, relation construction and card annotation—is accounted for separately. Concurrency and reused receipts also prevent a clean production-latency comparison.

The workload's tokens per accepted positive answer divide all trajectory tokens, including failures and negative questions, by the number of accepted positive answers. At embedding 1,600 this is about 3,617; adaptive is about 13,782. The denominator remains evaluator acceptance, so this measure inherits its errors.

Part I's summary amortization illustrates another trap. Building four summaries used 27,875 input tokens; dividing by the mean answer-input saving gives approximately 9.1 total queries across a balanced four-album workload, or 2.3 per album on average. That is an input-only reuse calculation. A summary can repay its construction cost while still losing the provenance this task requires.

## VIII. Conversation state suggests a tradeoff, not a memory victory

The additional conversation diagnostic contains eight actual four-turn executions: one conversation per album under full and adaptive policies. Each conversation reuses its compiled graph and thread ID. Later turns receive up to 250 tokens of previous claim records under both policies, with earlier outputs treated as fallible.

{{< figure src="/img/album-memory-v2/conversation-tradeoff.svg" alt="Full context has 11 of 16 accepted conversation turns and about 3286 tokens per turn; adaptive has 15 of 16 and about 8970 tokens per turn." caption="Adaptive gains four online-accepted turns across this workload and consumes about 2.73× the trajectory tokens per turn. These linked turns are not independent samples." >}}

Across the 16 turns per policy, adaptive uses 90,940 additional trajectory tokens for four additional online acceptances: **22,735 extra tokens per additional accepted turn**. This is a descriptive workload difference, not a causal price for memory improvement.

There is no embedding conversation control and no matched no-history ablation. Both policies receive history, and adaptive changes retrieval, sufficiency assessment and retries together. Consequently, the observed difference cannot isolate the value of conversational memory or demonstrate long-term retention. The same evaluator dependencies apply.

[LangGraph's memory documentation](https://docs.langchain.com/oss/python/langgraph/add-memory) distinguishes thread persistence from memory shared across sessions. The experiment uses in-process storage and checkpoints; it does not demonstrate durable production memory. Persisting an archive is also separate from choosing which parts become model input.

## IX. What this study leaves unresolved

The useful limit of this experiment is now clearer. It exposes the cost of deciding what to retrieve and the weakness of treating a stopping verdict as an independent measure of quality. It does not yet tell us which routing policy should replace this controller. Adding another policy here would require a new comparison, and would make it too easy to revise the question around whichever result arrived next.

The independent review also remains unfinished. A three-policy packet is a better basis for that work, but preparing it is not equivalent to collecting judgments. Before a stronger correctness claim, reviewers need to examine the evidence and the questions themselves, including ambiguous narrative requirements and the flawed chronological reference. More generations cannot resolve those defects on their own.

I would stop this experiment at that boundary. Its contribution is the distinction between the evidence an agent exposes, the interpretation it produces, and the total process required to produce it. The next study should begin with those distinctions as design constraints.

## X. What I would carry forward

Embedding retrieval at 1,600 tokens is the baseline the next memory design must earn its way past in this workload. Its advantage is economical execution under the observed checker. The audit prevents me from treating that advantage as a validated claim about interpretive correctness.

The relation graph remains valuable as an inspectable research representation: it can preserve a proposed return, make counterevidence visible, or expose where two readings disagree. That value does not imply that injecting its prose improves an agent. A useful representation for a researcher and an effective context policy for a generator are different empirical questions.

For the musical work, the most consequential failure is the conversion of a comparison into a story: two emotions become a transformation, three citations become a coherent arc, or a later reference becomes an earlier song. Better memory should preserve the distinctions that make an interpretation contestable, including the possibility that the proposed arc is not supported.

For AI software engineering, that requirement reaches beyond prompting. The source schema, role bindings, stopping policy, evaluator and resource ledger all participate in the system's behavior. MLflow helps preserve the evidence needed to examine those decisions; LangGraph makes their control flow explicit. Neither substitutes for a defensible research question.

The next implementation should be judged on whether it recovers and attributes the required evidence more reliably at an acceptable total cost. More elaborate memory earns its complexity through that comparison.

The series can now move from **Part I — What Should an Agent Remember?**, on memory and narrative, through **Part II — Memory Is Not Context**, on exposure, evidence use and agent economics, to a planned **Part III — Context Optimization Is Not Agent Optimization**. Its working question is *Adaptive Memory Is a Routing Problem: Choosing What an Agent Should Retrieve Under a Token Budget*. Task-aware routing, relation-index retrieval and hard trajectory budgets belong to that next experiment; they are not results of this one.

## Reproduction and revision record

The [experiment package](/examples/album-memory-v2-study.zip) contains the frozen source cards, relations, questions, all 504 primary/threshold trajectories, 32 conversation turns, provider-usage exports, analysis scripts, figures and review materials. The [complete results table](/examples/album-memory-v2/policy_means.csv) retains the original acceptance and post-hoc ID screen. The [task table](/examples/album-memory-v2/revision_task_table.csv) supports the new heatmap.

The environment remains pinned to MLflow 3.16.0 and LangGraph 1.2.11. Generation uses `gpt-4o-mini`; preparation, sufficiency and verification use `gpt-4.1-mini`. Exact resolved model IDs, prompt URIs and provider receipts are retained. The package distinguishes offline analysis from new paid API executions.

The omitted controls remain informative within their contracts: recent three accepts 4/20 positives; the original summary accepts none because it lacks addressable source claims under the stricter provenance requirement. That zero does not demonstrate that provenance-preserving summarization would fail. Leave-one-album-out threshold selection reaches 18/20 online acceptance at about 10,905 tokens per query and 85% after the earlier ID screen. It selects thresholds only; prior corpus inspection and selection of the embedding budget remain limitations.

**Revision, September 11, 2026:** the argument has been reorganized around trajectory cost and evaluator validity. This version adds a qualitative AI audit of 72 responses, task-level and conversation figures, and a real screenshot of the original *Infest* trace in MLflow. The receipt reconstruction remains in the downloadable artifacts. The blank blind-review packet includes 24 responses from each of Full, Embedding1600 and Adaptive. The results overview is at the beginning, and follow-up experimental protocols have been reserved for Part III. It does not change historical model outputs, invent human judgments, or present the proposed ablations as executed. Full lyrics, credentials, private trace databases and the blind-review policy key remain excluded.
