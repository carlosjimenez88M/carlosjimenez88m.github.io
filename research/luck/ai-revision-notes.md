# AI evaluation revision: sources, estimands, and challenged claims

Prepared October 5, 2026. This extends Part III's methodological synthesis. It
contains no hosted-model results and no provider calls. The agent experiment is
a proposed protocol, not an experiment that has been run. Existing clustered
Gaussian calculations remain synthetic and keep their recorded parameters.

## Primary reading added

| Source | Material consulted | Editorial use and limit |
| --- | --- | --- |
| Chen et al. (2021), *Evaluating Large Language Models Trained on Code* | [Section 3 and Appendix A](https://arxiv.org/html/2107.03374v1) | Candidate-pool estimator and its sampling interpretation; no copied model-performance results |
| Gelman & Loken (2013), *The garden of forking paths* | [Author-hosted manuscript, opening and Section 1](https://stat.columbia.edu/~gelman/research/unpublished/p_hacking.pdf) | Contingent analysis choices; exploration distinguished from untouched confirmation |
| Oren et al. (2023), *Proving Test Set Contamination in Black Box Language Models* | [Version 2, problem setting, methods, and limitations](https://arxiv.org/html/2310.17623v2) | Exchangeability and likelihood-query assumptions; absence of detected verbatim exposure is not proof of clean generalization |
| Horace He / Thinking Machines Lab (2025), *Defeating Nondeterminism in LLM Inference* | [Author's engineering report, batch-invariance discussion](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) | Caller-controlled seeds do not describe all serving conditions; a primary technical report, not a universal provider guarantee |
| Shah (2026), *Causal Agent Replay* | [Preprint, Sections 2–5 and 7](https://arxiv.org/html/2606.08275v1) | Interventions and continuation distributions; scope restricted to its synthetic validation and mocked-tool demonstrations |
| Zheng et al. (2023), *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* | [Version 4, Sections 3.3–3.4](https://arxiv.org/html/2306.05685v4) | Bias checks and answer order; results belong to the studied judge configurations, not every current judge |

The original sources remain in `sources.md`. These additions were purposively
selected to connect the series to engineering decisions; this is not a systematic
literature review. No source documents or copyrighted figures are bundled here.

## pass@k: distinguish three probability objects

For one problem, let `n` candidates contain `c` passing outputs and require
`1 ≤ k ≤ n`. The finite-pool quantity

`1 − choose(n − c, k) / choose(n, k)`

is the probability that a uniformly chosen subset of `k` distinct candidates
contains a passing output. Define the numerator as zero when `n − c < k`.
This combinatorial identity needs no independence assumption about how the pool
was generated.

For `n` iid candidate outcomes with common per-problem probability `p`, the same
quantity is an unbiased estimator of the probability of at least one pass in
`k` fresh iid attempts: `1 − (1 − p)^k`. An independent check is to average the
all-failing indicator over all `k`-subsets: every product has expectation
`(1 − p)^k`, so their average does too. The essay's derivation is this direct
counting argument.

The plug-in `1 − (1 − c/n)^k` is not this unbiased estimator. For `k > 1`, the
function `1 − (1 − x)^k` is concave on [0,1], so Jensen's inequality gives a
nonpositive estimation bias under the iid binomial design. Equality holds for
`k = 1` and degenerate probabilities; an individual observed pool can still give
an estimate above or below the true target. This concerns this nonlinear
estimator, not a general prohibition on using the same observations twice.

For dependent but exchangeable candidate outcomes, the subset average can target
the corresponding exchangeable `k`-outcome success probability. It does not imply
that this target equals `1 − (1 − p)^k`. For adaptive retries, contexts, stopping,
and success recognition can change after every attempt; a uniformly chosen
subset of those recorded attempts need not represent a new adaptive run.
Evaluate the actual complete procedure instead. Duplicate output strings alone
do not prove statistical dependence; iid categorical draws can also repeat.

Compute per-problem estimates before averaging with declared task weights.
Transforming a pooled success probability generally loses difficulty
heterogeneity. The essay's two-problem example uses probabilities 0 and 1: true
mean pass@2 is 0.5, whereas the pooled-probability transform is 0.75.

`k > n` is not a supported finite-pool estimate. More candidate availability also
does not establish that a deployed selector will find the passing candidate.
Count calls, latency, and the selected answer's correctness separately.

### Independent arithmetic check

The new numerical illustration uses `n=10`, `p=0.2`, `k=3`. One observed pool
with `c=2` gives subset estimate 0.5333333333333333 and plug-in 0.488. Summing over
all possible binomial counts gives:

| Quantity | Exact finite-sum result |
| --- | ---: |
| Fresh-three-attempt probability | 0.488 |
| Expected subset estimator | 0.488 |
| Expected plug-in estimator | 0.45056 |

The following standard-library calculation was executed locally. It uses no
simulation, model call, environment variables, or external dependencies.

```python
import math
n, p, k = 10, 0.2, 3
weights = [math.comb(n, c) * p**c * (1-p)**(n-c) for c in range(n+1)]
subset = sum(
    weight * (1 - (math.comb(n-c, k) if n-c >= k else 0) / math.comb(n, k))
    for c, weight in enumerate(weights)
)
plugin = sum(weight * (1 - (1-c/n)**k) for c, weight in enumerate(weights))
assert math.isclose(subset, 1-(1-p)**k, abs_tol=1e-12)
assert math.isclose(plugin, 0.45056, abs_tol=1e-12)
```

## Proposed experiment and its identification boundary

The example intervention is memory off/on crossed with a retry budget of
one/three tool calls. Four frozen policies run on each independently sampled
initial environment. The design separates policy factors, environment sampling,
within-environment random inputs, and measurement. Reset state between policies;
randomize policy order to reduce temporal confounding. Record exclusions and
stopping rules before final evaluation. Blocks sharing a person or organization
may need a higher clustering level than individual sessions.

The policy contrasts are expected outcome differences over the declared target
population of environments, with memory effects reported at each retry budget,
retry effects at each memory setting, and their interaction. Candidate pass@k,
selected-answer quality, resource consumption, and severe-failure rates are
separate outcomes. They cannot be collapsed into a single causal luck fraction.

A paired random tape is a coupling, not a recovered historical counterfactual.
It must preserve each policy's own marginal environment law. Pairing can either
reduce or increase variance. Separate mechanisms' random streams and use
meaningful event keys when request ordering diverges; using the same global
seed alone does not ensure comparable exogenous inputs. If no defensible event
match exists, do not manufacture one by replaying the next recorded response.

Exact trace replay checks the recorded harness. Interventional continuation
requires valid new responses and downstream states for the changed actions.
Record/replay is safe for identical requests at matching relevant states;
otherwise use a stateful sandbox or a specified simulator. The simulator's
contrasts depend on its mechanisms. The protocol does not recreate how a real
user would have reacted to a response they never received, nor establish an
individual's unique historical cause from one saved seed.

The replay preprint supplies useful ideas, but its point-of-commitment rule is
not adopted here as a theorem proving a unique causal culprit in every agent.
Rerolling downstream decisions can give an early irrelevant step a contrast by
rerolling later pivotal choices. Our protocol distinguishes total effects over
fresh continuations from more conditional contrasts and requires explicit
interventions, faithful state reconstruction, and checks on simulator behavior.
Synthetic validation cannot settle transport to live users.

Freeze and blind the evaluator separately from policy development. Balance
answer order, save replicate judgments and their aggregation rule, and audit
against independent references or humans. Replicating a biased judge can make a
biased estimate precise. No advertised human-agreement percentage is imported
as a guarantee for a new rubric, domain, or current hosted model.

## Editorial decisions after self-critique

- Replaced the generic opening with an agent whose third attempt succeeds, then
  connected that event to budget, memory, environment, and evaluator decisions.
- Retained the independent-user variance derivation because it identifies which
  repetitions actually add population evidence.
- Added finite-pool versus fresh-attempt versus adaptive-procedure distinctions;
  avoided interpreting pass@k as a skill or luck attribution.
- Distinguished contamination evidence, development adaptation, and legitimate
  exploratory or posterior predictive reuse of data.
- Made a frozen factorial, blocked policy experiment concrete, with full attempt
  histories, tool fault records, state restoration, and separate judge evidence.
- Removed repeated closing caveats and ended with the actionable policy contrast
  and its deployment consequences.

Publication dates, canonical URLs, newsletter exports, assets, and scheduling
are maintained by the parent task. This revision edits only Part III and these
notes; it does not claim the proposed agent experiment was implemented or run.
