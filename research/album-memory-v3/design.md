# Part III — Context Optimization Is Not Agent Optimization

Working subtitle: Adaptive Memory Is a Routing Problem: Choosing What an Agent Should Retrieve Under a Token Budget.

Status: research plan only. No Part III experiments have been executed, and no Part III post is being published in this revision. Keep Part II closed after its results overview, three-policy blind-review packet and real MLflow screenshot.

## Central question

Can a policy route a task to the appropriate evidence representation and stop within an enforceable trajectory budget, while improving independently assessed support relative to static embedding retrieval?

## Three contributions to test

1. Task-aware memory routing: route using the public question and observed state, never silver answer IDs or test outcomes. Include routing calls in the resource ledger and compare against a frozen static embedding baseline.
2. Relation-index retrieval: use relations to select source addresses while exposing cards alone. Separate selection effects from relation-prose effects with matched card-set ablations and fixed packing and ordering.
3. Hard trajectory budgets: track cumulative expenditure and admit the next call only if known input plus reserved maximum output fits. Record budget exhaustion and stopping decisions rather than treating them as successful answers.

## Series structure

- Part I — What Should an Agent Remember? Memory and narrative.
- Part II — Memory Is Not Context. Exposure, evidence use and agent economics.
- Part III — Context Optimization Is Not Agent Optimization. Routing, budgets and stopping policies.

## Methodological prerequisites and secondary experiments

Repair ambiguous questions and chronology before evaluating new methods. Keep online stopping judgments separate from final evaluation. The three central contributions above take priority; distraction interventions and repetition are supporting studies, not additional headline claims. Preserve paired question-level analysis and disclose the four-album sampling limit.

The following protocol notes were moved out of Part II rather than added to its measured results:

## IX. What would make the next claim credible?

The revision leaves the original executions intact. The following comparisons are **proposed follow-up experiments, not completed results**. Their purpose is to separate effects the current design combines.

**First, repair and independently evaluate the task.** The blank [72-response review packet](/examples/album-memory-v2/human-review.html) now includes full, embedding 1,600 and adaptive outputs, hiding policy and model scores. Its rubric separates support, attribution, requested relation and narrative defensibility on a 0–2 scale, plus whether abstention was warranted. Two independent human reviewers could adjudicate disagreements and estimate agreement; no human ratings or kappa statistic exist in this release. Before new benchmark runs, question wording and reference chronology also need adjudication. Repeated generation cannot repair an invalid target.

**Second, separate graph retrieval from graph prose.** The present relational policy changes both candidate selection and the text shown to the generator. A controlled ablation should use the same retrieved edge ranking to select source addresses, then expose cards alone. Compare embedding-to-cards, relation-to-cards-plus-explanations, and relation-to-cards under matched memory budgets. Also compare matched card sets with and without relation prose: otherwise a result still mixes selection with how many cards fit. Freeze deduplication, packing and ordering before evaluation. Only then can an improvement be assigned to retrieving relations or explaining them.

**Third, manipulate distraction while holding evidence fixed.** The current embedding grid increases designated evidence availability from 59.1% to 100%, while citation given exposure falls from 92.3% to 68.2%. Different items enter the denominator at different budgets. That does not identify context dilution.

{{< figure src="/img/album-memory-v2/budget-utilization.svg" alt="Evidence availability rises across budget caps while conditional citation generally declines; selected evidence changes across those caps." caption="Availability and citation given exposure answer different questions. These curves neither measure transformer attention nor establish that additional context caused distraction." >}}

A stronger test would retain identical target claims, vary distractor count, and randomize whether targets appear at the beginning, middle, end or split positions. Distractors would need checking for alternative valid answers; twelve distractors are not available for every target set in every album. The causal outcome would be support and correct attribution for the same evidence under those interventions. [Lost in the Middle](https://arxiv.org/abs/2307.03172) motivates investigating positional effects; it does not establish them for this task or these models.

**Fourth, constrain the trajectory before another call.** A cost-aware controller needs cumulative usage, retrieval rounds, verification failures and an explicit remaining budget in state. Stopping only after cumulative use exceeds a limit is not a hard cap: the next call can overshoot. Admission control must reserve its known input and maximum allowed output, with a refusal or fallback when the call cannot fit. A proposed jump from 400 to 1,600 should identify missing evidence rather than assume every intermediate inspection is useful. Static embeddings, the present controller and this revised controller would then face the same independently evaluated workload.

Finally, repeat the critical settings after freezing that protocol. Repeated runs can characterize hosted-model variation, but they do not create new albums or questions. Analysis should preserve pairing and clustering by question and album. With only four albums, even an attractive interval cannot establish broad artist-level generalization.

