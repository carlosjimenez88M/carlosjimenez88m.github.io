---
author: Carlos Daniel Jiménez
date: 2026-09-11T00:00:00-05:00
title: "What Should an Agent Remember? Album Narratives with LangGraph and MLflow"
description: "An empirical study of bounded memory, distant lyrical evidence, and token use across four albums."
categories: ["Engineering", "Music Analysis"]
tags: ["langgraph", "mlflow", "memory", "narrative", "music", "evaluation"]
draft: false
---

An album ends. An agent has read every lyric. We ask it whether the ending transforms something established near the beginning.

The agent can produce a convincing paragraph with almost no memory. That is precisely the problem. Fluency gives us very little evidence that it remembered the right songs, preserved their differences, or resisted turning a collection of voices into one convenient story.

In my [previous post on MLflow](/post/2026-09-08-mlflow-prompt-engineering/), I argued that a prompt belongs in the release process because changing an instruction changes application behavior. Here I want to bring that argument into my research on musical narratives. **Memory selection changes behavior too. It needs an evaluation contract.**

The practical question is narrow enough to test: when an interpretation depends on two distant songs, how much context can we remove before we lose the evidence needed to connect them?

## I. The album is a sequence, not a single speaker

A recurring image does not establish a plot. Two songs can share a concern and disagree about what to do with it. A later expression of acceptance may coexist with unresolved conflict elsewhere. Different songs may speak through different narrators.

I use *arc* here as a proposed relationship between positions in an album: recurrence, contrast, a shift in stance, or a tension that remains unresolved. An argumentative arc is more specific: a position encounters resistance, qualification, or an alternative. These are interpretive hypotheses to support with evidence, not properties that become true because a model produces a smooth summary.

The corpus intentionally places four different lyric sequences together:

| Artist | Album | Analytical units |
| --- | --- | ---: |
| Red Hot Chili Peppers | *Californication* | 15 |
| Papa Roach | *Infest* | 12 |
| The Beatles | *Abbey Road* | 17 |
| Natalia Lafourcade | *De todas las flores* | 12 |

For *Infest*, the twelve units are the original eleven listed tracks plus the hidden song “Tightrope,” indexed separately. The other albums follow their specified track order. This is a comparison of four selected albums, not a claim about these artists' complete bodies of work.

There is also an important boundary around the word *musical*. The inputs are lyrics. This study observes no melody, harmony, timbre, tempo, production, or performance. That matters especially when a sequence achieves continuity through a musical transition rather than a repeated lyrical theme. A lyric-only model can miss the very thing that makes two songs feel connected.

Natalia Lafourcade's [description of *De todas las flores*](https://www.natalialafourcade.com.mx/en/de-todas-las-flores) provides useful artistic context. I keep that context outside the model experiment: feeding an artist's explanation into an interpretation task would make it difficult to distinguish evidence recovered from lyrics from a framing supplied in advance.

## II. Three meanings of attention

Human attention, transformer attention, and the context selected by an application operate at different levels. This experiment measures the third.

I cannot recover a listener's concentration from an API response. I also cannot inspect a hosted model's attention weights here. What I can observe is which song records reach the generator, how many tokens the provider reports, and whether the resulting answer retains the evidence required by a question.

The [*Lost in the Middle* study](https://arxiv.org/abs/2307.03172) found that the location of relevant information affected performance in its retrieval and question-answering experiments. That motivates testing memory access; it does not establish that this album task, these models, or every long context will reproduce that pattern. Our full-context baseline is necessary precisely because it may work well.

[*LongMemEval*](https://arxiv.org/abs/2410.10813) separates memory design into indexing, retrieval, and reading, and tests capabilities including temporal reasoning and abstention. I borrow the engineering distinction, not its benchmark scores. An album interpretation is a different task, and our small set of questions is not a replacement for that benchmark.

A useful operational meaning of concentration is therefore *allocation*: how much of the available record does an application put in front of its model? A smaller allocation is only desirable when it preserves the relationships the task needs.

## III. Build evidence before compressing it

The collector retrieves lyrics through Genius, checks the returned artist and title, and records source URLs, positions, and content hashes. Full lyrics remain in a private local cache. The public experiment artifacts contain analytical paraphrases and references rather than reproduced lyrics.

Each track becomes an anonymous evidence card. The annotator receives numbered lyric lines without album or artist metadata. It records a stance, movement within the song, unresolved tension, eight thematic ratings, three paraphrased claims with line references, and a caution about interpretation.

Those cards create a common representation for all four memory policies. They also introduce a common limitation: a card can omit or misunderstand something before the memory comparison begins. The full archive contains all cards, not all possible readings of the lyrics. I treat the cards as **silver annotations**—model-produced evidence for a bounded experiment, rather than human-validated ground truth.

The Spanish lyrics receive English analytical paraphrases so that the subsequent task uses one language. This adds translation loss. Withholding metadata also does not eliminate recognition: a model may recognize a familiar lyric without being told its title.

The question generator reads the full cards before any answer runs. It proposes three questions per album: a distant return, a change in stance, and counterevidence to a single smooth narrative. Each question requires a designated pair of tracks spanning the two album halves, separated by at least four positions. A deterministic validator rejects pairs that fail those conditions. The questions describe semantic relationships without revealing the target track IDs to the answering model.

## IV. Four memory policies, one answer contract

| Policy | Context supplied to the generator | Main risk |
| --- | --- | --- |
| Full archive | Every analytical card | More input tokens; possible distraction |
| Recent 3 | Only the last three cards | Early evidence is unavailable |
| Rolling summary | A sequentially updated summary capped at 300 tokens | Compression can erase a distinction or its provenance |
| Selective 6 | Opening card, last two cards, and query-relevant cards up to six total | Fixed anchors consume capacity; lexical retrieval can miss paraphrases |

Selective retrieval uses TF-IDF over card text with unigrams and bigrams. It needs no additional model call or embedding service. After selection, cards retain their original chronological order. Six cards is a fixed experimental setting, not an optimized universal memory size. Longer cards also mean that a six-card limit is not a strict token budget.

Every policy receives the same generation prompt, model, question, and output schema. The answer must cite IDs present in its memory and abstain if the memory cannot support both sides of the question. Three stochastic repetitions per question give 144 answers across the four policies. Those repetitions measure output variation on the same questions; they do not create 144 independent musical examples.

The [LangGraph memory documentation](https://docs.langchain.com/oss/python/langgraph/add-memory) distinguishes thread state and persistence from memory shared across sessions. In this implementation, the graph ingests the archive, selects a context, answers, and validates references. Separate thread IDs isolate each case. `InMemorySaver` demonstrates checkpoints within the running process; it is not durable storage across restarts.

The distinction that matters for tokens is simpler: **stored state is not automatically model input**. The application explicitly renders the selected memory into the request. A checkpoint can preserve an entire album while the next model call sees six cards. Merely adding a checkpointer would not create that saving.

{{< figure src="/img/album-memory/memory-architecture.svg" alt="Private lyrics become anonymous evidence cards. LangGraph stores cards, selects memory, generates an interpretation, and validates it. MLflow records prompts, traces, tokens, and evaluations." caption="The implemented memory experiment. Storage and generation context are separate decisions." >}}

## V. Make the memory policy observable

[MLflow's LangGraph integration](https://mlflow.org/docs/latest/genai/tracing/integrations/listing/langgraph/) supplies graph tracing through `mlflow.langchain.autolog()`. The experiment also instruments provider calls and registers the annotation, summary, question, answer, and judge prompts. Per-case runs record the policy, models, prompt identity, selected IDs, output, token usage, and evaluation scores.

Two classes of evidence must remain separate. Deterministic checks tell us whether cited IDs were available and whether the designated evidence pair survived selection. A separately prompted judge reads the full cards and silver reference to rate support and coverage. It is blind to the policy label but still fallible, and it shares a model provider with the generator. It can also infer limited context from an answer's abstention.

Identifier survival is especially weak for a summary. A summary may retain “T02” while losing the particular claim needed from that track. That is why target availability cannot substitute for answer coverage. Conversely, an alternative pair of songs may support a defensible interpretation even if it does not match the designated pair. Exact-ID recall is a diagnostic, not the complete meaning of quality.

Token accounting separates card creation, summary updates, question generation, answers, and judges. Comparing only answer tokens would hide the work needed to construct compressed memory. Whether that initial investment pays off depends on how many subsequent questions reuse it.

## VI. What the cards suggest—and where they flatten the music

The annotations offer hypotheses worth examining at song level. They do not justify a definitive plot for any of these albums.

In *Californication*, the opening card describes outward movement and connection, while “This Velvet Glove” describes a more intimate desire for understanding. “Road Trippin'” returns to companionship and travel. A possible reading is a change in the scale of connection: broad experience, interpersonal vulnerability, shared excursion. But “Californication” introduces social critique, and “Porcelain” introduces decline. Keeping these differences matters more than forcing every song into a progression from isolation to recovery. For an agent, remembering only the ending would retain travel while losing the contrasting vulnerabilities that make a return worth discussing.

*Infest* asks for a different memory structure. The cards distinguish personal despair in “Last Resort,” family rupture in “Broken Home,” rejection of materialism in “Between Angels and Insects,” and loss of control in “Thrown Away.” These positions can interact without belonging to the same speaker or resolving into one argument. “Tightrope,” treated as its own final unit, does not receive an annotation of simple resolution. A memory that compresses this album into a single emotional label would erase the difference between inward distress and outward social criticism. The agent needs to preserve *objects of conflict*, not merely their intensity.

For *Abbey Road*, juxtaposition is an essential caution. “Maxwell's Silver Hammer” and “Polythene Pam” receive sharply different narrative annotations. “Here Comes the Sun” and “Sun King” share an optimistic register, but that alone cannot explain the album's musical construction. Near the ending, the card for “Carry That Weight” records burden, “The End” records love and connection, and “Her Majesty” records uncertain admiration. Even an apparently final statement has something after it. A memory policy should preserve that coda as evidence against treating the album's nominal ending as its only ending.

In *De todas las flores*, the cards trace solitude and attempted agency in “Vine solita,” longing in “Pasan los días,” healing in “María la curandera,” gratitude toward mortality in “Muerte,” and farewell in “Que te vaya bonito Nicolás.” A defensible hypothesis is that nature changes its interpretive role across the sequence: a medium for emotional release, a resource for healing, and imagery accompanying farewell. Calling this a smooth recovery would lose something. The later songs still hold loss, and the annotation of gratitude toward death should not be mistaken for the absence of tension.

These observations suggest a memory design based on **relations with provenance**: what returns, what changes, what contradicts a proposed reading, and which tracks support each relation. They also expose the limits of our first representation. The tension scale frequently produces the same rating for quite different situations. An ordinal label cannot tell us whether a song is confrontational, grieving, ambivalent, or accepting mortality. The textual claims are more informative than a single trajectory drawn through those numbers.

{{< figure src="/img/album-memory/theme-profiles.svg" alt="Four heatmaps display eight model-annotated themes in track order for Californication, Infest, Abbey Road, and De todas las flores. Ratings range from absent to central." caption="Model-annotated thematic profiles. These are ordinal readings of lyrics, not acoustic measurements or validated emotional trajectories." >}}

### Does the order itself carry a signal?

I compared adjacent songs using cosine similarity between their eight theme ratings, then recalculated mean adjacency for 10,000 shuffled orders per album with a fixed random seed. This is an exploratory diagnostic of the representation. The shuffle preserves the songs and changes their neighbors.

| Album | Observed adjacency | Mean shuffled adjacency | Central 95% of shuffled orders |
| --- | ---: | ---: | ---: |
| *Californication* | 0.830 | 0.795 | 0.756–0.835 |
| *Infest* | 0.851 | 0.832 | 0.795–0.870 |
| *Abbey Road* | 0.623 | 0.633 | 0.560–0.710 |
| *De todas las flores* | 0.781 | 0.756 | 0.701–0.815 |

All four observed values fall inside their respective central shuffled ranges. *Californication* sits near the upper end, while *Abbey Road* falls slightly below its shuffled mean. I would not translate either result into a verdict on narrative quality. These coarse theme profiles provide limited separation between the actual ordering and alternative orderings. Four exploratory comparisons also do not support selecting the most favorable one as a confirmed discovery.

{{< figure src="/img/album-memory/order-continuity.svg" alt="Observed adjacent theme similarity falls within the central 95 percent of shuffled-order reference values for each of the four albums." caption="A descriptive ordering diagnostic. A smooth theme sequence is only one possible kind of structure, and this representation excludes musical transitions." >}}

The more useful implication is that **an arc cannot be reduced to neighboring songs being similar**. A contrast can carry an argument. A distant return can matter more than a local transition. A musical transition can be invisible to these lyric features altogether. This is why the memory task asks for evidence across an album rather than rewarding the smoothest thematic curve.

## VII. A valid citation can still point to the wrong claim

One observed *Abbey Road* response makes the problem concrete. Under the recent-three policy, the generator saw only T15, T16, and T17. Asked to compare an earlier and later stance, it answered with T15 and T16. Both identifiers were available, so the deterministic validity check passed.

But the response attributed admiration for a woman to T16. In the actual cards, that description belongs to T17, “Her Majesty.” T16 is “The End.” The response also failed to supply a first-half song. The judge still assigned coverage 3 out of 4 and support 2 out of 4, with no overclaim flag.

This is not a successful answer rescued by a creative interpretation. It is a source-attribution error combined with failure to meet the question's temporal constraint. It demonstrates a limitation of the judge rubric as implemented: it can reward a plausible contrast without adequately checking where that contrast came from.

The lesson is practical. Validate the *claim-to-source relationship*, not only membership in an allowed list. A future release gate should require evidence spanning the requested positions and verify each cited claim against its card. The current pilot preserves the original judge outputs so that this weakness remains visible rather than silently replacing inconvenient scores.

## VIII. The code boundary that controls token use

The generator receives a serialized `memory` field, not the complete graph state. The following excerpt captures that boundary in the executable experiment:

```python
def answer(state):
    key = (
        f"{state['album']}-{state['policy']}-"
        f"{state['probe']['kind']}-{state['repetition']}"
    )
    result = call(
        "generation",
        key,
        ANSWER_SYSTEM,
        dumps({
            "memory": state["memory"],
            "question": state["probe"]["question"],
        }),
        max_tokens=400,
        temperature=0.3,
    )
    return {
        "answer": result["output"],
        "generation": {k: v for k, v in result.items() if k != "output"},
    }
```

`call` is the experiment's cached provider wrapper, supplied in the downloadable code. Its key distinguishes real repetitions without adding the repetition label to the prompt. Completed responses retain the provider's model identifier, request ID, usage, and latency. Resuming the experiment reuses those receipts instead of treating repeated execution as new evidence.

The graph connects `ingest → select → answer → validate`; ingestion loops until all cards are stored. The experiment invokes a separate thread for each album, policy, question, and repetition:

```python
graph = build_graph(cards, final_summary)
result = graph.invoke(
    {
        "album": album_id,
        "policy": "selective6",
        "probe": probe,
        "repetition": 0,
        "archive": [],
        "cursor": 0,
    },
    {
        "configurable": {"thread_id": case_id},
        "recursion_limit": 100,
    },
)
```

These are excerpts from the supplied experiment rather than a standalone application. `build_graph`, the prompt constants, and the input files are included with the code. Each comparison rebuilds its archive to make state isolation explicit. A deployed reader would normally ingest an album once and reuse its persistent archive; repeated ingestion here incurs local computation and checkpoint storage, but no additional model calls.

The pilot ran with MLflow 3.16.0, LangGraph 1.2.11, OpenAI Python 3.10.0, and Python 3.13. Generation, annotation, and summary updates use `gpt-4o-mini`; question construction and judging use `gpt-4.1-mini`. The dependency lockfile records the remaining environment. These model aliases can change server-side, so the per-call resolved model identifiers and cached outputs are also part of the evidence.

## IX. What happened when we reduced context?

All 144 answer cases completed, with 144 corresponding judge calls. MLflow recorded 144 graph traces and 144 separate judge-call traces. The following means first collapse repetitions within each question, then questions within each album, then weight the four albums equally.

| Memory | Input tokens / answer | Reduction vs. full | Designated evidence recall | Judge coverage / 4 | Judge support / 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full archive | 3,456.6 | — | 54.2% | 3.28 | 2.81 |
| Recent 3 | 886.3 | 74.4% | 25.0% | 3.22 | 2.19 |
| Selective 6 | 1,589.7 | 54.0% | 65.3% | 3.39 | 2.72 |
| Rolling summary | 395.3 | 88.6% | 5.6% | 2.86 | 2.03 |

Input tokens come from the provider's usage receipts, including the instruction and question. The 300-token summary cap uses a local tokenizer estimate; it is not a promise that the complete provider request contains 300 tokens. Output length changed much less: the policy means ranged from about 119 to 126 output tokens per answer.

{{< figure src="/img/album-memory/tokens-coverage.svg" alt="Four memory policies compared by generation input tokens against model-judged coverage and designated evidence recall. Selective six uses about 1590 input tokens and has 65 percent evidence recall; the full archive uses about 3457 and has 54 percent recall." caption="Observed pilot results. Similar judge scores conceal substantial differences in evidence retrieval. Small dots show album means, not confidence intervals." >}}

Selective six is a promising candidate here: it uses roughly half the full archive's input tokens and retrieves more of the designated evidence in the generated citations. That is not evidence of uniformly better interpretation. Its support score is slightly lower than the full archive's, and its performance varies by album. In *Infest*, designated recall reaches 88.9%; in *Californication* and *De todas las flores*, it is 50%. The chosen opening and closing anchors also consume half the slots regardless of the question.

The recent-three result is more concerning. Its coverage score remains close to the full baseline, but none of its answers cites a track from each album half. This is unavoidable from the supplied context: the last three songs all belong to the second half. The generator never abstains under this policy. The judge often rewards a comparison assembled from the available ending, even though that comparison fails the cross-album requirement.

I added a **post-hoc diagnostic**, explicitly separate from the planned primary metrics, to count answers whose citations span both halves. It reaches 77.8% for full context, 75.0% for selective six, and 0% for both recent three and summary. This check still cannot prove that the cited claims are correct. It can expose a temporal failure that the judge missed. Exact target recall is also not an exhaustive measure: some questions admit defensible alternative evidence pairs.

### The summary did not simply forget the oldest song

Inspecting the final memories reveals several different failure mechanisms:

| Album | What survived in the final summary | Why it matters |
| --- | --- | --- |
| *Californication* | Explicit references only to T14 and T15 | An evolving narrative paragraph displaced earlier evidence. |
| *Infest* | T12 plus unqualified C1/C2/C3 claim labels | Locally meaningful labels became ambiguous without their track provenance. |
| *Abbey Road* | A description focused on T17 followed by a list of every track ID | Identifier presence falsely suggested complete evidence availability. |
| *De todas las flores* | Material from T01 through T06 | Prefix truncation preserved earlier material while later updates failed to survive. |

The summary policy requests no more than 180 words and applies a deterministic prefix cap of 300 locally estimated tokens after each update. That implementation choice matters. If the model expands the opening material instead of rewriting within the budget, the cap can repeatedly discard new information. The Natalia result is a failure of this particular update-and-truncate design, not a general law that summarization forgets recent songs.

The Beatles summary illustrates a different trap. It retains every identifier, so the ID-availability proxy reports complete coverage of the designated pair. Yet its generated answers recover none of those designated IDs. An index without the corresponding claims is not sufficient memory.

Summary answers also cite unavailable identifiers in seven of 36 cases. Only the summary policy abstains at all—nine times—but that caution is inconsistent. These observations argue against releasing this rolling summary as the sole interpretive memory.

### Count the work required to build the memory

| Stage | Completed calls | Input tokens | Output tokens |
| --- | ---: | ---: | ---: |
| Track annotations | 56 | 33,886 | 16,424 |
| Rolling updates | 56 | 27,875 | 11,815 |
| Question construction and corrections | 8 | 27,522 | 3,654 |
| All policy answers | 144 | 227,805 | 17,758 |
| Blind-policy judging | 144 | 482,350 | 10,633 |

The eight question calls include regenerated sets after invalid evidence pairs. One earlier request was rejected for JSON-mode formatting and returned no usage receipt; it is not included as a completed generation. The questions retain coarse positional hints such as first or second half, despite the initial instruction to omit positions. They reveal no exact target IDs. That protocol deviation is recorded, and every policy receives the same frozen questions.

The 88.6% reduction for summary describes **answer input**, not total cost. For one answer to each of the twelve questions, full context would consume 41,479 input tokens. Summary answers would consume 4,744, but their rolling updates already consumed 27,875: 32,619 combined. That is a 21.4% input-token reduction before adding common annotation work. Output tokens, provider pricing, caching discounts, and judge costs are separate; this is not a dollar saving estimate. More reuse can amortize preparation, but it cannot recover evidence already discarded.

The judge stage consumed more input tokens than all answer policies combined. That is acceptable for a small research comparison, but a deployed system should not automatically rerun this entire evaluation workload for every user question. Keep a curated regression set and inspect production failures selectively.

## X. A proposal: remember relations, retrieve evidence, check the claim

The pilot supports a concrete direction for the next agent. Keep the complete source archive outside the generation context. Alongside each song card, maintain a compact relation record: a proposed return, a change in stance, or counterevidence; the supporting track and claim IDs; and an explicit uncertainty. Such a record is an interpretive hypothesis, not a replacement for its source cards.

For *Californication*, a relation might connect broad movement and companionship to a later, more intimate desire for connection. For *Infest*, it should distinguish social rejection from personal loss of control. For *Abbey Road*, it should preserve contrasting characters and the coda. For *De todas las flores*, it should retain the difference between healing and farewell. Those distinctions are what a generic emotional summary tends to flatten.

At question time, the proposed agent would retrieve relation records, expand their source cards within a token budget, and check whether the question's required positions are represented. If the evidence is insufficient, it would retrieve another card or abstain. After generation, it would verify each claim against its cited card. MLflow would record the selected evidence, expansion decision, token budget, and validation outcome as part of the run.

That extension has **not** been measured in this pilot. Selective six is the implemented starting point. The next comparison should test relation retrieval against that baseline on new questions, ideally written and judged by independent readers, with the token budget fixed in advance. The questions used to diagnose this system should not also become the only proof that its revision works.

The result I would carry into AI Software Engineering is specific: a memory policy can make an answer much cheaper while leaving its apparent quality nearly unchanged—and still remove the very evidence the question requires. We should release memory policies with the same discipline as prompts: explicit contracts, inspectable sources, reproducible cases, and evaluation that can reject a fluent answer.

## Reproduce and inspect

The [experiment package](/examples/album-memory-study.zip) includes the source code, dependency lockfile, design and execution notes, public provenance manifest, 56 evidence cards, final memories, questions, all 144 answer records, aggregate CSV files, figure generators, and a structural verification report. Full lyrics, credentials, the local MLflow database, and local trace artifacts are excluded.

The [policy means](/examples/album-memory/policy_means.csv) and [album means](/examples/album-memory/album_means.csv) can be inspected directly. The package's README explains how to fetch the inputs with your own credentials and rerun each stage. Offline aggregation can be repeated from the included public cards and receipts; live regeneration can differ because the provider models and source pages may change.

Verification checked 56 source hashes, all 144 planned album/policy/question/repetition combinations, provider receipts, completed MLflow runs, graph traces, evidence-line bounds, and thread isolation. Hugo also builds the article and its figures. These checks establish that the experiment ran as recorded. They do not turn its musical interpretations or model-judge scores into ground truth.
