---
author: Carlos Daniel Jiménez
date: 2026-09-11T00:00:00-05:00
title: "Memory Is Not Context: Token Budgets, Narrative Relations, and Agent Economics"
description: "Part II of a four-album experiment: budgeted retrieval, an adaptive LangGraph controller, claim-level verification, and MLflow trajectory accounting."
categories: ["Engineering", "Music Analysis"]
tags: ["langgraph", "mlflow", "memory", "evaluation", "narrative", "context-engineering"]
draft: false
---

An agent can store an entire album and still fail to answer a question about two songs. Giving it the complete archive does not tell it which relationship matters. Compressing that archive does not guarantee that the relationship survives.

In [Part I](/post/2026-09-11-langgraph-mlflow-album-memory/), selecting six song cards used 54% fewer generation input tokens than the full archive and recovered more of the designated evidence in citations. But the judge also rewarded answers that missed the question's temporal requirement. That combination changes the next experiment. We need to study the path from stored evidence to a correctly attributed answer, and count the resources consumed along that path.

**Can an agent remember more while exposing less—and can it recognize when less is insufficient?**

The measured result is a useful correction to the hypothesis: a 1,600-token embedding retriever and the adaptive controller both reached 90% online-verifier acceptance, but the adaptive controller consumed about 3.8 times as many trajectory tokens. A later attribution check also exposed a false positive in that acceptance.

The word *recognize* creates an engineering obligation. A controller that retrieves again whenever it feels uncertain can consume more resources than simply reading the archive. A verifier that accepts plausible prose can make an economical system appear reliable. Both belong inside the measurement.

## I. Separate availability, use, and attribution

Start with a distinction we can actually observe:

```text
Stored cards and relation hypotheses
                  ↓
          Retrieved candidates
                  ↓
       Evidence exposed in context
                  ↓
         Evidence cited in output
                  ↓
      Claims supported by their sources
```

The last two steps are different. A citation can name a real song while attributing another song's content to it. The first experiment included exactly that failure.

For designated target items, I define retrieval availability as the fraction exposed to the generator. Citation utilization is the fraction of exposed target items that the answer cites. Attribution is assessed at the level of the answer's individual claims and their cited source claims. These definitions do not reveal transformer attention or prove that a passage caused an output. Even an uncited passage can affect generation.

Recomputing the original pilot makes the denominator problem concrete:

| Part I policy | Designated items exposed | Exposed items cited | Citation given exposure |
| --- | ---: | ---: | ---: |
| Full archive | 72 | 39 | 54.2% |
| Selective 6 | 54 | 47 | 87.0% |
| Recent 3 | 18 | 18 | 100% |

Recent three wins the conditional ratio because it receives very little of the required evidence. It does not win the task. Selection changes the denominator. That is why utilization must be displayed beside availability, rather than interpreted as a universal measure of reading ability.

The summary requires another qualification. In Part I, an identifier could survive without its supporting claim. In this experiment, a name inside prose does not count as exposure of a source claim. A conditional ratio with no exposed source claims is undefined, not zero.

I also avoid calling uncited context “waste.” We can measure the fraction associated with citations, but cannot establish that the remainder was irrelevant to the model. Operational measures should not claim access to hidden cognitive mechanisms.

## II. An album is a useful memory environment

The corpus remains the same: *Californication*, *Infest*, *Abbey Road*, and *De todas las flores*. Their 56 analytical units include “Tightrope” separately from “Thrown Away.” This continuity lets us examine the engineering extension without silently changing the corpus.

It also limits the result. The cards were already inspected in Part I, and the albums were selected intentionally. This is an extension of a small pilot, not an independent replication or evidence about whole discographies.

Each card has a position and three paraphrased claims tied to private lyric line references. The new memory format assigns global addresses such as `T05:C2`. A relation has two supporting addresses, an interpretive type, a topic, a short explanation, and an uncertainty. Relations are constructed before the new questions.

```yaml
id: E01
# Schema illustration; not a measured edge.
type: contrast
topic: approaches to connection
evidence:
  - T02:C1
  - T11:C2
interpretation: A proposed contrast that must be checked against both claims.
uncertainty: Shared vocabulary does not establish one narrator.
```

I do not attach an invented probability to an interpretive edge. A number such as 0.81 would suggest calibration that this study does not provide. The relation is a hypothesis with addresses, not a second independent source confirming its own input cards.

The four albums offer different possible stresses. *Californication* distributes connection, vulnerability, movement, and social critique across songs. *Infest* can make distinct objects of conflict look alike if memory retains only emotional intensity. *Abbey Road* combines contrasting characters and a coda, while much of its continuity is musical and therefore outside lyric-only inputs. *De todas las flores* allows us to examine changing roles of healing, nature, and farewell without assuming that loss simply disappears.

These are reasons to examine results by album. They are not validated assignments of four narrative topologies. One album per proposed pattern cannot establish that a topology determines the best memory policy.

## III. Retrieve relations, then expand their evidence

The experiment compares seven policy families:

| Policy | What it exposes |
| --- | --- |
| Full | Every compact source card |
| Recent | The last three compact source cards |
| Summary | The original rolling summary, with its original provenance limitations |
| TF-IDF | Whole cards ranked by lexical relevance |
| Embedding | Whole cards ranked by semantic similarity |
| Relational | Retrieved relation hypotheses plus the cards supporting them |
| Adaptive relational | Relational context that can expand after sufficiency or attribution failure |

The three ranked retrieval policies use exposed-memory budgets of 400, 800, 1,200, 1,600, and 2,400 tokens. Full is the unbounded reference. Recent and summary remain single controls; repeating the same context under different budget labels would not create additional observations.

A budget applies to the serialized memory, measured with the model family's tokenizer. The instruction, question, and provider framing are additional input. Cards are packed whole. Relation text counts against the budget when it is exposed. If a complete edge and its cards do not fit, the relational policy can include a ranked source card; it cannot pretend to have included a complete relation.

The embedding policy uses `text-embedding-3-small`. LangGraph's indexed `InMemoryStore` holds separate card and relation namespaces per album. Relation retrieval ranks edge descriptions and adds a small, fixed preference for the relation types relevant to the public task: for example, contrast for a question challenging a smooth account. It then expands the referenced cards. Neither ranking stage receives the designated answer IDs.

This design tests a precise idea: a narrative question may benefit from retrieving a relationship rather than the nearest single passage. It does not assume that embeddings fail at contradiction or that graph retrieval wins. A bad edge can direct retrieval toward exactly the wrong pair.

The store is in memory for this experiment. It is reused within a process, but is not a durable production database. The [LangGraph memory documentation](https://docs.langchain.com/oss/python/langgraph/add-memory) distinguishes persistent thread state from cross-session memory and describes database-backed options. Changing the storage backend would not itself change which tokens enter a model request.

## IV. Six task types and a stricter answer contract

The new question set contains one question of each type per album:

| Task | Required evidence structure |
| --- | --- |
| Local | Two adjacent songs |
| Distant | One song from each half, at least four positions apart |
| Transformation | Two separated songs contrasting a stance on a shared concern |
| Contradiction | Two separated songs challenging a proposed smooth reading |
| Multi-hop | Three songs in order, first and last at least six positions apart |
| Abstention | A requested premise not established by the available cards |

Questions make their structural demands explicit but withhold designated track IDs. Target pairs and expected answers remain silver annotations. Some new questions revisit concerns already raised by Part I; they are not an untouched benchmark of unknown concepts.

A separate adversarial check reads the complete card archive to challenge each proposed negative question. That makes the negative cases less arbitrary, but it is still model review of model-produced cards. Lack of support in the cards does not establish that a relation is absent from the music.

The generator must produce an answer of at most 90 words and no more than four atomic claims. Each claim must cite exact addresses such as `T04:C2`. It must abstain if evidence or provenance is insufficient. This is stricter than Part I's track-ID contract, so the new scores cannot be treated as a direct continuation of the old quality scale.

The online verifier receives the question, the answer, and the cited source claims. It checks each assertion separately, whether every substantive sentence is represented among the claims, and whether the answer addresses the question. A structural gate also checks the requested track positions and whether references were actually exposed. The verifier does not see silver targets or expected answers.

An initial smoke test exposed a useful implementation failure: a free-form verifier grouped several claims into one result. The final verifier uses a structured response requiring a separate result for every claim index. This enforces coverage of the verification task, not the correctness of the verdict. Model errors remain possible.

Malformed generator outputs fail visibly. They do not become successful abstentions. An adaptive policy can spend another round trying to recover, and that round counts against its resource use.

## V. The adaptive controller has a stopping problem

The primary controller begins with 400 memory tokens. Its available levels are 400, 800, 1,200, 1,600, 2,400, and the complete relation memory.

```text
Select evidence within the current budget
                    ↓
       Is the evidence sufficient?
          no ↙              ↘ yes
    Expand budget          Generate
          ↑                    ↓
          └────── Attribution and scope check
                         fail ↙      ↘ pass
                    Expand           Finish
```

The sufficiency model sees only the question, its public structural constraint, and exposed memory. It gives an ordinal score from zero to four. The primary stopping threshold is three: apparently complete evidence, allowing interpretive ambiguity. That score is not a calibrated probability.

If the answer then fails attribution or scope verification, the graph can expand again. Expansion stops at the complete memory. A failure at that point remains a failure; the controller cannot generate indefinitely until it happens to receive a favorable verdict.

There is a subtle cost here. The complete relational representation contains edge explanations as well as source cards, so it can exceed the size of the full-card baseline. Adaptive retrieval is not guaranteed to remain cheaper after repeated inspections and retries. That possibility is part of the hypothesis, not an implementation embarrassment to hide.

## VI. Count the trajectory, not just the last answer

[MLflow's LangGraph integration](https://mlflow.org/docs/latest/genai/tracing/integrations/listing/langgraph) captures the graph execution. The experiment also records provider calls, prompt versions, chosen budgets, selected evidence, verification results, and session identifiers.

For every trajectory, the accounting separates sufficiency, generation, and verification input/output tokens. The final context size is reported separately. Embedding and relation-construction work belong to preparation. Counting them as free would favor the more elaborate memories.

The primary economic measure is total trajectory tokens per model-verified supported answer:

```text
total input and output tokens for the evaluated workload
-------------------------------------------------------
      supported, non-abstaining positive answers
```

The denominator excludes negative questions, whose correct abstention rate is reported separately. The numerator includes the workload's negative cases and failed attempts because they consume resources too. If no positive answer passes, the ratio is undefined; it is not zero and should not become an attractive point on a chart.

This is a token measure, not a dollar estimate. Input and output tokens can have different prices, and cached-token pricing complicates the conversion. The artifacts retain provider usage so a dated pricing model can be applied explicitly.

Completed calls are cached for resumption and shared deterministic prefixes. Logical trajectory token use counts every call the policy requires, even when this particular experimental replay can reuse a receipt. Unique provider-call totals are exported separately. Neither quantity should be mislabeled as the other.

Latency also needs a qualification: concurrent execution and cached receipts do not provide a clean cold-start latency comparison. Figures using the sum of recorded call latencies label it as such.

## VII. Choose a threshold without using the test album's scores

In addition to the primary threshold of three, the experiment runs fixed thresholds two and four. This adds 48 trajectories to the 456 primary settings, for 504 cases in total. The additional runs make threshold alternatives observable rather than imagining what an unexecuted branch would have produced.

Leave-one-album-out selection chooses a threshold using the other three albums. The fixed feasibility targets are supported-answer rate at least 0.8, evidence availability at least 0.5, and model attribution at least 0.8. Among feasible choices, it minimizes mean trajectory tokens. If none is feasible, it selects the highest supported-answer rate and then the lower token use, and reports the failure to meet the targets.

The held-out album contributes no scores to that choice. But all four source corpora and the Part I experiment were already known, and graph construction uses each album's cards. This is a check on threshold selection, not proof of generalization to new artists or a large unseen distribution.

## VIII. Reuse changes the accounting, not the evidence

Part I's summaries consumed 27,875 input tokens across four albums. Their mean answer-input reduction relative to full context was about 3,061 tokens. Dividing those quantities gives approximately 9.1 **total queries across a balanced four-album workload**, assuming all four summaries were built up front.

It is not 9.1 queries per album. Averaged per album, the same input-only arithmetic is about 2.3 queries, with individual break-even points depending on each album's preparation cost and request sizes. Output tokens, common annotation work, and prices are excluded from that calculation.

More importantly, amortization cannot restore lost provenance. A summary can recover its construction cost while remaining unable to support the required claims. The relevant comparison is expected reuse at an acceptable evidence standard, not compression in isolation.

## IX. The result did not favor the most elaborate memory

All 504 cases completed. The table below shows selected primary settings; the full grid is included in the downloadable results. Each setting answers 20 positive questions and four negative questions. The rate is acceptance by the **online model verifier plus structural gate**, not independently measured correctness.

| Setting | Positive answers accepted | Mean trajectory tokens | Tokens per accepted positive answer |
| --- | ---: | ---: | ---: |
| Full cards | 15/20 · 75% | 2,890 | 4,624 |
| Recent three | 4/20 · 20% | 1,418 | 8,507 |
| Original summary | 0/20 · 0% | 1,145 | Undefined |
| TF-IDF, budget 800 | 13/20 · 65% | 1,819 | 3,358 |
| Embeddings, budget 1,200 | 16/20 · 80% | 2,249 | 3,373 |
| Embeddings, budget 1,600 | 18/20 · 90% | 2,713 | 3,617 |
| Relational, budget 2,400 | 12/20 · 60% | 3,520 | 7,039 |
| Adaptive relational, threshold 3 | 18/20 · 90% | 10,336 | 13,782 |

{{< figure src="/img/album-memory-v2/pareto-frontier.svg" alt="The empirical frontier compares whole-trajectory tokens with online verifier acceptance. Embedding retrieval at a 1600-token memory cap and adaptive relational retrieval both reach 90 percent acceptance, but adaptive requires about 10336 trajectory tokens versus 2713." caption="An empirical frontier of automated acceptance, not a validated frontier of musical correctness. The dashed line joins nondominated observed settings." >}}

Under these observations, embedding retrieval at 1,600 memory tokens dominates the primary adaptive controller: equal acceptance with about **3.8 times fewer trajectory tokens**. It also dominates the full-card baseline on these two axes. The adaptive controller does not demonstrate the efficiency advantage I hoped to test.

The summary is technically nondominated at the extreme low-cost end because it is the cheapest setting. It also accepts no positive answers. This is why a Pareto frontier needs a minimum quality requirement. Membership alone does not make a policy useful. The legacy summary also lacks addressable source claims under the new contract, so it cannot pass that provenance gate. Its zero is a representation-compatibility result for this control, not evidence that every form of summarization fails.

The 1,600-token embedding point is an observed grid result, chosen after seeing the measurements. It is not a validated optimum for new questions. There is one generation per setting and question, and hosted models can vary even at temperature zero. Apparent reversals between neighboring budgets may reflect sampling, selection composition, or model behavior; the experiment does not isolate those causes.

### More available evidence, less conditional citation

For embedding retrieval, designated evidence availability rises from 59.1% at budget 400 to 100% at budget 2,400. Conditional citation falls from 92.3% to 68.2%. Relational availability rises from 38.6% to 95.5%, while its conditional citation falls from 100% to 50.0%.

{{< figure src="/img/album-memory-v2/budget-utilization.svg" alt="Across five token caps, evidence availability increases for lexical, embedding, and relational retrieval, while conditional citation generally decreases." caption="The target items in the denominator change as the context expands. These curves suggest questions about evidence use; they do not measure internal attention or establish a causal distraction effect." >}}

This is consistent with a gap between availability and use, but it does not prove context dilution. Small contexts preferentially retain the easiest or most salient evidence. Increasing the budget adds different items and changes that denominator. A causal test would hold target evidence fixed while manipulating distractors or its position in the rendered context.

The distance breakdown is similarly descriptive. Track distance is confounded with task and album, and some strata contain only one question. It is not a randomized needle-position experiment.

{{< figure src="/img/album-memory-v2/distance-breakdown.svg" alt="Accepted-answer rates by distance between designated source positions, with separate lines for full, lexical, embedding, relational, and adaptive policies." caption="Distance describes these questions; it does not isolate why a policy succeeds or fails. The retrieval curves pool budgets, unlike the single full and adaptive settings." >}}

## X. Where the adaptive tokens went

The efficient adaptive case and the expensive one have different stories.

For the distant *Californication* question, the initial 400-token cap exposes cards T04 and T15 in 330 memory tokens. One sufficiency call, one answer, and one verification call lead to acceptance. The complete trajectory consumes 2,305 input and output tokens. Its cited pair differs from the designated pair, illustrating why exact target recall and a defensible answer need separate treatment.

For the distant *Infest* question, the controller visits five caps: 400, 800, 1,200, 1,600, and 2,400. It performs five sufficiency calls, four generations, and four verifications. The final answer uses a 2,326-token context, but the whole trajectory consumes **19,831 tokens**. Looking only at that final context would hide most of the work.

{{< figure src="/img/album-memory-v2/trajectory-economics.svg" alt="Stacked bars separate sufficiency, generation, and verification tokens for full cards, TF-IDF at 800, embeddings at 1600, and adaptive relational memory." caption="The controller is part of the resource cost. A small final prompt does not imply a small trajectory." >}}

An initial budget is not a spending cap. The present controller can repeat analysis and verification at successively larger levels. A resource-constrained version needs a bound on **cumulative trajectory tokens**, a policy for jumping to a larger context when expansion is likely, and an explicit decision about when another verification attempt is worth its cost. Those changes would require another measured comparison.

The extra relation text also has a price. It can explain why two cards belong together, but it consumes capacity that could otherwise expose source claims. In this particular graph, twelve generated edges per album are sparse and incomplete. An important pair may be missing from the graph or ranked poorly. The weaker relational results are evidence against treating this implementation as automatically superior to similarity retrieval.

{{< figure src="/img/album-memory-v2/relation-memory.svg" alt="Four arc diagrams show the 12 generated relation hypotheses for each album, colored by return, contrast, transformation, and counterpoint, with endpoints at their source track positions." caption="The relation graph used by the experiment. Edge height is a layout choice, not strength or confidence. These are model-generated hypotheses with source addresses." >}}

## XI. Even the stricter verifier accepted a wrong attribution

The expensive *Infest* answer contains a particularly revealing claim. It assigns a critique to T01 while citing `T04:C2`. The verifier marks the claim supported and repeats the inconsistent mapping in its explanation. Structured output ensured a verdict for each claim; it did not ensure that the verdict was right.

I added a clearly **post-hoc** check: when a claim explicitly names a track ID, that track must be represented among the claim's source addresses. This catches one accepted positive answer for each adaptive threshold and one for relational retrieval at budget 1,200. The primary adaptive rate falls from 90% online acceptance to **85% after this additional screen**. The embedding 1,600 point remains at 90% under the same screen.

That check is only a necessary condition. It cannot detect a wrong attribution expressed without an explicit track ID, or a subtler interpretive error. Its purpose is to prevent a known mistake from disappearing inside an aggregate.

There is also a measurement dependency: the adaptive policy retries until the same online verifier accepts an answer. Its acceptance rate is therefore a measure of satisfying that checker under a bounded search, not an independent evaluation of correctness. Human review—or an independent, validated evaluation procedure—is necessary before promoting that rate to a claim about reliable interpretation.

The blind review packet contains 48 responses: full and primary adaptive outputs for every album/task combination. Policies and model scores are hidden. Reviewers receive the complete source-card claims and rate support, attribution, and arc plausibility. **No human ratings have been collected for this version**, and no agreement statistic is reported. The packet also makes clear that reviewing cards is not the same as independently annotating the lyrics.

## XII. Threshold selection and actual conversations

Leave-one-album-out selection chooses threshold four for *Californication* and three for the other albums. Its held-out aggregate is 18 of 20 positive answers accepted, with about 10,905 trajectory tokens per question. The post-hoc explicit-track screen lowers that acceptance to 85%. Threshold selection does not remove the cost disadvantage.

The conversation diagnostic adds 32 turns: four linked questions per album under full and adaptive policies. Each four-turn conversation reuses one compiled graph and thread ID, and MLflow groups its traces under one session ID. Later questions receive up to 250 tokens of previous claims, including whether they passed verification. Both policies use the same history cap. Earlier outputs are explicitly treated as fallible, and new claims still need source evidence.

The full policy receives acceptance on 11 of 16 turns, using about 3,286 trajectory tokens per turn. Adaptive receives acceptance on 15 of 16, using about 8,970. Twelve later turns per policy include prior claim records. These are observations of actual linked executions; they do not establish long-term knowledge retention, and the same verifier limitations apply.

The conversations reuse task concerns from the primary question set and can propagate earlier model mistakes. They should be read as an operational diagnostic of thread memory and history exposure, not as another independent benchmark or an unseen musical discussion.

{{< figure src="/img/album-memory-v2/amortization.svg" alt="Cumulative input-token lines from Part I cross at about 9.1 total queries after constructing summaries for all four albums." caption="An input-only reuse calculation from Part I. Financial and evidential break-even are different questions, and this figure does not estimate dollars." >}}

## What I would carry into the next implementation

The first implementation decision would be simpler than the original hypothesis suggested: keep a compact source-card retriever as a strong baseline. The observed embedding point achieves the adaptive controller's online acceptance with far less trajectory work. A relation graph should justify its extra structure through held-out retrieval or attribution gains.

For the adaptive controller, I would make cumulative expenditure a first-class state variable and separate the stopping checker from the final evaluation. I would also preserve atomic source addresses through every compression step and validate explicit source/entity consistency before paying for another model judgment.

For musical interpretation, the graph remains useful as an inspectable collection of hypotheses: a return, a changed stance, a counterpoint, a coda. Its usefulness as a representation does not imply that injecting all of its prose makes an LLM reason better. The interesting question is which relations help recover the needed evidence at the moment of a question.

The result is therefore more specific than “adaptive memory beats static context.” **Stored memory, exposed context, and accepted evidence are separate engineering decisions. More sophisticated memory can improve a checker's acceptance while making the trajectory less efficient—and the checker can still be wrong.**

## Reproduce and review

The [Part II experiment package](/examples/album-memory-v2-study.zip) contains code, frozen cards and relation hypotheses, questions, all 504 primary/threshold records, 32 conversation turns, aggregate tables, provider-usage exports, figure generators, and the blind review packet. The [full results table](/examples/album-memory-v2/policy_means.csv) includes the complete budget grid and the post-hoc screen.

The environment reuses the pinned MLflow 3.16.0 and LangGraph 1.2.11 stack from Part I. Generation uses `gpt-4o-mini`; relations, questions, sufficiency, and verification use `gpt-4.1-mini`. The resolved model identifiers and provider receipts remain attached to the records. The README distinguishes offline reproduction of the analysis from new paid API calls.

Full lyrics, credentials, raw private trace databases, and the policy key for human review are excluded. The published experiment remains a model-annotated, model-checked four-album pilot. Its detailed failure cases are part of the result, not exceptions to remove before drawing the frontier.
