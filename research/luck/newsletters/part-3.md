<!-- buttondown-editor-mode: plaintext -->

[Read this essay on the blog](https://carlosdanieljimenez.com/post/2026-10-26-statistical-luck-ai-evaluation/)

An evaluation can contain 480 generated answers and still supply only forty independent user histories.

The distinction sounds administrative until it changes the conclusion. In the synthetic experiment below, treating all repeated measurements as independent produces an interval narrow enough to miss the known effect. Respecting the user as a sampling unit produces a wider interval that includes it. Across thousands of repetitions, the narrow interval fails much more often than its label promises.

This is one place where thinking carefully about luck changes engineering practice. We stop asking whether a run was impressive and begin asking which variation the evaluation allowed us to observe, which it concealed, and which claim the design can support.

The first essay proposed a conditional description of favorable contingency. The second showed how selection and persistent advantage complicate learning from winners. Here I assemble a framework from those findings. It is a methodological proposal, with explicit failure conditions, rather than a validated instrument for measuring luck.

**A Statistical Account of Luck**

1. [What Can Statistics Mean by Luck?](https://carlosdanieljimenez.com/post/2026-10-12-statistical-luck-reference/)

2. [Why Winners Look More Skilled Than They Are](https://carlosdanieljimenez.com/post/2026-10-19-statistical-luck-winners/)

3. [A Statistical Framework for Luck in AI Evaluation](https://carlosdanieljimenez.com/post/2026-10-26-statistical-luck-ai-evaluation/)

## Four questions that should remain separate

An unexpected outcome, a causal improvement, an opportunity advantage, and an acceptable deployment risk are different objects of inquiry.

| Question | Statistical object | Evidence needed |
| --- | --- | --- |
| Was this outcome favorable relative to our prediction? | A realized excess, rank, or tail under a frozen predictive distribution | A defensible prospective reference and a defined outcome |
| Does changing the policy improve expected outcomes? | A causal contrast such as E[Y(1) − Y(0)] | An intervention design or defensible identification assumptions |
| Who receives favorable exposure or starting conditions? | A distribution of access and outcomes across defined groups | Coverage of the opportunity process; causal evidence for mechanism claims |
| Is the policy acceptable under future variation? | Expected loss, tail risk, and specified failure probabilities | Relevant repeated evaluation, uncertainty, and an explicit decision rule |

A single “luck score” would blur these questions. A model may outperform expectation on one occasion while its policy has no average advantage. A policy may improve the average while harming a subgroup. A low average error may coexist with an unacceptable severe-failure rate.

For the causal question, potential outcomes make the comparison explicit. Yᵢ(1) and Yᵢ(0) represent what unit i would experience under two defined policies. We usually observe one policy outcome for a given event, not both counterfactuals. Randomized allocation can identify an average contrast under suitable assumptions; it does not identify the missing individual counterpart simply by making the sample large.

[Hernán and Robins' *Causal Inference: What If*](https://miguelhernan.org/whatifbook), especially Chapters 1–3, provides the relevant distinctions. Their standard adjustment approach to observational identification requires exchangeability, positivity, and consistency for adequately specified interventions; other identification strategies use different assumptions. In an AI experiment, changing the model, retriever, budget, and rubric together defines a package intervention. It cannot isolate which component helped without additional design.

Nor does a paired benchmark automatically solve the counterfactual problem for a person's future interaction. We can run both systems on a recorded prompt, but neither may reproduce the user's behavior under a live conversation. Pairing supports a bounded comparison; transport to deployment requires more assumptions.

## A framework in seven commitments

The framework is intentionally a set of commitments rather than an attribution formula. Each step produces something another person can examine.

**1. Define the outcome and the population.** State the unit, horizon, and direction of preference. For retrieval, distinguish source recall from answer correctness. For an agent, distinguish a final answer score from the cost and consequences of its complete trajectory. Define the users and tasks to which the claim applies. A convenient corpus is not automatically a representative population.

**2. Freeze a prospective reference.** Record the baseline policy, available information, predictive assumptions, and expected variability before opening the final evaluation. If the distribution is estimated, carry parameter uncertainty into the prediction. If it is only a provisional model, label it as such and test sensitivity to alternatives. Do not derive the reference from the winner and then claim the winner exceeded it.

**3. Draw the boundary around control.** List what the actor could change at the relevant time: a retrieval budget, a prompt, an abstention threshold, or an allocation policy. Separately record starting conditions and exposure. A predictable advantage can remain outside the actor's control. A random seed that a team searched extensively belongs to its selection procedure, even if an individual run's output was unpredictable.

**4. Represent the sources of repetition.** Specify which units are sampled independently, which outcomes share a user or task, and what varies between runs. A seed sweep mainly studies execution variation conditional on the sampled examples. It does not replace sampling new users. [Bouthillier and colleagues' benchmark-variance study](https://arxiv.org/html/2103.03098v1) supports examining multiple sources, including data sampling and training choices; its measured tasks do not establish a universal ordering of those sources for every AI system.

**5. Preserve the search history.** Keep unsuccessful configurations, stopping rules, exclusions, and evaluator changes. Set aside evidence that the search will not touch. Estimate the performance of the procedure that selected the policy, rather than treating its maximum score as a fresh experiment. This follows the model-selection problem examined in [Cawley and Talbot](https://www.jmlr.org/papers/v11/cawley10a.html). If repeated inspection is unavoidable, use a design suited to that inspection or obtain fresh confirmatory data.

**6. Check the instrument and the probabilities.** Review whether the metric measures what its name claims. Audit an automated judge against independent, source-grounded judgments, with unresolved disagreement reported. For probability forecasts, examine calibration and predictive performance on relevant held-out outcomes. A useful average score can conceal subgroup miscalibration; a favorable single outcome does not validate a forecast.

**7. Make the decision rule visible.** Define the smallest improvement that would matter, the failures that would prevent release, and the cost of more evidence. Examine whether the decision survives plausible changes to the reference, dependence, and target population. A broad interval is sometimes a reason to collect evidence, sometimes a reason to choose a reversible policy, and sometimes a reason to abstain. Its meaning depends on the consequences.

These commitments make an analysis inspectable. They do not guarantee that its assumptions are true. A well-documented experiment can still use the wrong population or a systematically biased evaluator.

## A worked example: repeated users and false precision

Suppose we compare two policies using a paired continuous score difference. There are n = 40 independently sampled users, k = 3 tasks per user, and r = 4 repeated generations per task. That gives 480 differences, Dᵢₜᵣ, generated as

<div class='buttondown-block-math'>[ D_{itr}=\Delta+U_i+V_{it}+E_{itr}, ]</div>

where Δ = 0.02, Uᵢ ∼ Normal(0, 0.06²), Vᵢₜ ∼ Normal(0, 0.04²), and Eᵢₜᵣ ∼ Normal(0, 0.04²). All components are independent except that a user's component is shared across their tasks and a task's component across its generations.

These are **synthetic continuous score units**, not observed accuracy differences or probabilities. The model deliberately gives us the true population mean so that we can check interval coverage. It is simple enough to derive the uncertainty exactly:

<div class='buttondown-block-math'>[ \operatorname{Var}(\bar D)=\frac{\sigma_U^2}{n}+\frac{\sigma_V^2}{nk}+\frac{\sigma_E^2}{nkr}. ]</div>

The derivation follows by averaging each independent component. A user component appears twelve times but has coefficient 1/n in the grand mean. A task component appears four times with coefficient 1/(nk). Only the generation component receives the full 1/(nkr) averaging. Substituting the stipulated variances gives a standard deviation of **0.01033** for the grand mean across new panels.

Increasing generations reduces only the final term. More tasks reduce the middle and final terms. More independent users reduce all three. This arithmetic gives a concrete reason why an evaluation can generate more text without adding much information about new-user performance.

For one fixed panel, seed 20261005, the observed mean difference is **0.03481**. Average the twelve differences within each user and construct a Student t interval from the forty independent user averages. Under this Gaussian model, those averages are independent and identically normally distributed, so the t with 39 degrees of freedom calculation has its usual justification.

| Calculation | Estimated standard error | 95% interval |
| --- | ---: | --- |
| Forty independent user averages | 0.01302 | [0.00847, 0.06116] |
| Incorrectly assume 480 independent differences | 0.00439 | [0.02621, 0.04342] |

The second row uses the ordinary row-level standard error and a normal multiplier of 1.96. Its problem is the independence assumption, not a minor difference between t and normal multipliers.

![The user-level interval includes the known synthetic mean difference 0.02; the incorrectly independent row-level interval excludes it.](https://carlosdanieljimenez.com/img/luck/evaluation-units.png)

*One synthetic panel. The same estimated mean acquires very different apparent precision depending on which observations are treated as independent.*

Across 4,000 newly generated panels, the user-level intervals contained the true mean **94.65%** of the time. The naive intervals did so **50.28%** of the time. The approximate Monte Carlo standard errors of those coverage estimates are 0.36 and 0.79 percentage points. These are simulation frequencies under our stipulated process, not coverage estimates for an actual product.

There is a subtler interpretive issue. Both intervals are above zero in this particular panel, so correcting the dependence does not reverse its directional conclusion. It changes the precision and whether the interval covers the true effect. If a meaningful deployment gain had been declared to be 0.025 score units, the naive lower bound would clear it while the user-level lower bound would not. That threshold is an illustrative decision rule, not one inferred from the data.

The user-level method also has limits. Shared sessions across users, overlapping organizations, or a common evaluator shock may invalidate user independence. With unequal task counts, equal weighting of user means and equal weighting of all responses target different quantities. With nonnormal outcomes and few clusters, the convenient exact justification disappears. And no interval corrects an unrepresentative sample or a rubric that rewards unsupported answers.

## Forecasting uncertainty honestly

If a system reports a probability q for a binary event Y, the Brier loss is (q − Y)². With a true event probability p,

<div class='buttondown-block-math'>[ \mathbb E[(q-Y)^2]=p(1-p)+(q-p)^2. ]</div>

The first term is irreducible outcome variation under this Bernoulli model. The second penalizes departure from the true probability. Expected loss is uniquely minimized at q = p.

That small derivation captures the purpose of a proper scoring rule. [Gneiting and Raftery](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf) give the broader theory: evaluate predictive distributions with rules that reward honest probabilities in expectation. A rare event occurring once does not make the earlier forecast wrong. Conversely, getting one outcome right does not make an overconfident forecast reliable.

In practice p is unknown. Repeated held-out predictions, uncertainty, and calibration checks are needed; dependence and changing populations complicate them. A language model's verbal confidence is also not automatically a calibrated event probability. We need a defined event and evidence connecting the reported probability to that event.

## What this framework can and cannot conclude

The framework cannot assign a percentage of an engineer's achievement to luck. It cannot infer a unique causal story from an outcome distribution. It cannot turn a synthetic proof of possibility into an empirical estimate. Those are substantive limits, not details to remove once the article becomes inconveniently cautious.

It can make a narrower contribution. It can expose when a reference was chosen after the outcome; show when a selected maximum exaggerates repeat performance; distinguish additional executions from additional independent evidence; and connect uncertainty to a decision with stated consequences.

Across the series, the conclusion that survives is this: **luck is a relational statistical description of contingency, anchored to information, a reference process, and a boundary around control.** Favorable surprise is measurable under a model. Causal attribution requires additional evidence. The model does not decide moral credit.

For AI engineering, that conclusion changes the question we ask of success. We ask whether a procedure can be expected to help the intended users, how fragile that expectation is, and whether our evidence includes the attempts that did not look impressive. A good run remains welcome. The work is to make our next decision depend on more than its good fortune.

The [study package](https://carlosdanieljimenez.com/examples/luck-study.zip) contains all calculations, synthetic data summaries, figures, source notes, and critical-review decisions. No new embedding service or hosted generation model was needed: the uncertainty examined here comes from the statistical design, not from a shortage of representations.
