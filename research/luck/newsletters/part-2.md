<!-- buttondown-editor-mode: plaintext -->

[Read this essay on the blog](https://carlosdanieljimenez.com/post/2026-10-19-statistical-luck-winners/)

Search among one hundred candidates and keep the winner. In the synthetic experiment below, the winning score averages 88.1 across repetitions of that process. A fresh test of each selected candidate averages 75.6.

That gap could suggest a failed replication or a dishonest report. Neither is necessary here. Every candidate was evaluated under the same rules. The problem was that we remembered only the best first result.

Success is evidence, but the process that makes success visible is part of that evidence. A leaderboard, a collection of surviving companies, and a shelf of famous records are all filtered views. Reading the visible outcome as a transparent measure of underlying quality asks it to do more work than it can support.

In the first essay, I argued that a statistical description of luck needs a declared predictive reference and a boundary around control. Here I examine what happens when the reference is built from winners, or when winning changes the process itself.

**A Statistical Account of Luck**

1. [What Can Statistics Mean by Luck?](https://carlosdanieljimenez.com/post/2026-10-12-statistical-luck-reference/)

2. [Why Winners Look More Skilled Than They Are](https://carlosdanieljimenez.com/post/2026-10-19-statistical-luck-winners/)

3. A Statistical Framework for Luck in AI Evaluation — scheduled for 2026-10-26

## A winner can be better and overrated

Consider a synthetic population of candidates. Each has a stable mean performance θⱼ, and its first measured score contains occasion-specific variation:

<div class='buttondown-block-math'>[ \theta_j\sim N(70,4^2),\qquad Y_j=\theta_j+\varepsilon_j,\qquad\varepsilon_j\sim N(0,6^2). ]</div>

All latent means and errors are independent. A fresh measurement uses the same θⱼ and a new independent error. The units are an invented continuous score; this is not a model fitted to people's talent or bounded accuracy percentages.

The joint normal model gives an exact conditional expectation:

<div class='buttondown-block-math'>[ \mathbb E[\theta_j\mid Y_j=y]=70+\lambda(y-70),\qquad\lambda=\frac{4^2}{4^2+6^2}=\frac4{13}. ]</div>

One way to derive it is to note that Cov(θ, Y) = 16 and Var(Y) = 52, then use the conditional mean of a bivariate normal. The same expression is the expected fresh score. A first observation of 88 predicts a repeat mean of approximately 75.54, not 88.

This is regression toward the population mean under a specified model. It does not require the candidate to become less capable. Nor does it imply that the first outcome was meaningless: the posterior mean remains above 70.

I simulated 20,000 independent selection experiments for each candidate-pool size. In every experiment, the candidate with the largest first score was selected, and received a fresh measurement.

| Candidates considered | Mean winning first score | Mean selected latent performance | Mean fresh score |
| --- | ---: | ---: | ---: |
| 1 | 69.99 | 70.02 | 70.07 |
| 5 | 78.36 | 72.56 | 72.54 |
| 20 | 83.45 | 74.13 | 74.12 |
| 100 | 88.11 | 75.60 | 75.64 |

![As the number of candidates increases from one to one hundred, the winning observed score rises faster than the selected candidate's latent and repeat performance.](https://carlosdanieljimenez.com/img/luck/selection-and-regression.png)

*Synthetic results, seed 20261005. Searching more candidates finds stronger candidates while also selecting more favorable measurement noise.*

The point is not that searching fails. It improves the selected latent mean. The point is that the improvement in the winning observation substantially overstates that gain.

For independent candidates, Pr(maxⱼ Yⱼ ≤ t) = F(t)ᵐ. Increasing m makes an impressive maximum more likely even without changing the candidate distribution. Under this particular Gaussian model,

<div class='buttondown-block-math'>[ \mathbb E[Y_{j^*}-\theta_{j^*}]=(1-\lambda)\mathbb E[Y_{j^*}-70]. ]</div>

That identity follows by conditioning on all first scores and using the conditional mean above; j* is the selected index. It describes average selection optimism. It does not tell us the fraction of one winner's achievement caused by luck. A different dependence structure, noise level, or candidate population changes the answer.

## The record of attempts belongs in the result

In ML, candidates may be hyperparameters, prompts, checkpoints, memory policies, or evaluation rubrics. If each is tested and the highest observed score is reported, the search procedure has selected favorable evaluation variation along with any real improvement.

[Cawley and Talbot's analysis of model selection](https://www.jmlr.org/papers/v11/cawley10a.html) shows why optimization can overfit a noisy selection criterion. Its practical implication here is to evaluate the *selection procedure*, with a test set preserved from that search or an appropriate nested design. Repeatedly returning to the supposedly independent test set makes it part of the search again.

Candidate errors are often correlated because all candidates share examples or infrastructure. Our independent Gaussian experiment therefore supplies an illustration, not a correction factor for a real leaderboard. Dependence can change the benefit of search and its optimism. Keeping failed attempts and the selection rule is necessary to investigate that change.

[Gelman and Carlin's design analysis](https://stat.columbia.edu/~gelman/research/published/retropower_final.pdf) adds another caution: when estimates are noisy relative to plausible effects, selecting statistically significant results can exaggerate magnitude and can sometimes select the wrong sign. Their Type M and Type S questions concern repeated sampling under assumed effect sizes. A large observed winner is a poor substitute for external information about those sizes.

The remedy is not to distrust every positive result equally. It is to ask what opportunity the procedure had to find one, and which evidence remained untouched when it did.

## Survivors are a selected population

Imagine two independent standard normal quantities, S and U, and an inclusion rule S + U > 2. We might call them persistent capacity and favorable circumstance, but those names are interpretations of a toy model. Before selection their correlation is zero. Among selected cases, a lower S generally needs a higher U to cross the threshold.

In 200,000 simulated cases, the full-population correlation was approximately −0.001; among the 15,684 included cases, it was −0.733. Selection created a strong association where none existed in the generating population.

This is a collider mechanism: inclusion depends jointly on two inputs. It explains how studying only successful cases can distort relationships between their characteristics. It does **not** establish that successful people with one attribute lack another. The threshold and distribution were chosen to demonstrate a possibility, not estimated from human histories.

[Jerker Denrell's work on undersampling failure](https://pubsonline.informs.org/doi/10.1287/orsc.14.2.227.15164) examines a related learning problem: practices associated with visible survivors can appear more effective than they are across all attempts. Its theoretical argument warns against inferring a strategy's value from its surviving users. It does not make every admired practice ineffective.

The absent denominator matters. A story about the founder who concentrated everything in one risky idea tells us little about all the people who did the same and disappeared from the record. We also need comparable starting conditions, exposure, and the cost of failure. Otherwise the observed biography is doing the work of an experiment that never occurred.

## Early outcomes can change later probabilities

So far, the noise disappears on repetition. Some processes preserve it by changing future opportunity.

Take two alternatives, A and B, initially assigned weights one and one. Select an alternative with probability proportional to its current weight, then add one to the selected weight. After a selections of A and b of B,

<div class='buttondown-block-math'>[ \Pr(\text{next is A}\mid a,b)=\frac{1+a}{2+a+b}. ]</div>

This Pólya urn is a minimal reinforcement model. Both alternatives begin symmetrically. There is no built-in quality difference. Yet an early success changes later probabilities.

The mathematics also reveals a limit to our interpretation. For an ordered sequence with a A selections and b B selections, its probability is

<div class='buttondown-block-math'>[ \frac{B(1+a,1+b)}{B(1,1)}. ]</div>

Exactly the same sequence probability results if we first draw a fixed P ∼ Beta(1, 1) and then make independent Bernoulli selections conditional on P. An observed sequence alone cannot distinguish this urn's feedback mechanism from that latent-propensity mixture. Their predictive distributions agree even though their causal stories differ.

In the urn interpretation, the long-run A share has a uniform distribution on [0, 1]. For independent fair selections, it converges to one half. With 500 selections across 20,000 worlds, the simulated share standard deviations were 0.2894 and 0.0224, respectively. The urn simulation uses the exact beta-binomial representation above, rather than iterating each draw.

![Distributions of the A share after five hundred selections: independent fair draws concentrate near one half, while the symmetric reinforced urn produces a broad distribution.](https://carlosdanieljimenez.com/img/luck/reinforcement.png)

*Two stipulated processes with equal starting weights. This contrast illustrates persistence of early variation; it does not identify reinforcement from an observed market distribution.*

After A wins the first draw, its expected final share at draw 500 becomes (1 + 499 × 2/3)/500 = 0.6673. That calculation is a consequence of this urn's rules. It is not a measured advantage enjoyed by an actual artist, company, or model.

## What experimental evidence adds

The [Music Lab experiment by Salganik, Dodds, and Watts](https://www.princeton.edu/~mjs3/salganik_dodds_watts06_full.pdf) is valuable because it created parallel cultural markets rather than merely describing an existing ranking. Participants encountered unfamiliar songs under different social-information conditions. Across its experiments, stronger social influence increased inequality and made success less predictable across parallel worlds. Its independent-choice condition supplies a behavioral reference for appeal, not an objective definition of artistic worth. The setting also cannot establish the magnitude of feedback in today's streaming platforms.

[Bol, de Vaan, and van de Rijt's study of science funding](https://doi.org/10.1073/pnas.1719557115) uses a funding cutoff to compare applicants near the threshold. It provides evidence of cumulative funding advantage in that local setting, including later participation differences. The causal interpretation depends on the regression-discontinuity design's assumptions. It is neither a general percentage of scientific success due to luck nor evidence that all winners and nonwinners were identical.

These studies strengthen particular mechanism claims through particular designs. They do not license an unrestricted claim that quality never matters.

The simulation [*Talent versus Luck* by Pluchino, Biondo, and Rapisarda](https://arxiv.org/html/1802.07068v3) asks a different question: what outcomes can follow from stipulated encounters and compounding rewards? It is useful as a generative thought experiment. Its talent distribution, opportunities, and capital update rules are assumptions. A simulated concentration of wealth cannot validate those assumptions or estimate luck's contribution to actual wealth.

I initially considered using that model as the organizing framework. Its dependence on chosen mechanisms made a more modest framework necessary: selection, exposure, and reinforcement must be investigated separately, with evidence appropriate to each.

## A conclusion that does not erase ability

Three claims survive the review. Selecting a noisy maximum can inflate the observed advantage. Conditioning on survival can change the relationships we learn. Early outcomes can alter later opportunities when the process contains such feedback.

None establishes that achievement is entirely accidental. Our first simulation explicitly selects better latent candidates. Nor can the final outcome alone tell us whether feedback, unequal starting conditions, or persistent quality produced an advantage. The urn's exact observational equivalence makes that limitation unusually clear.

The mature response to an impressive result is therefore neither worship nor dismissal. It is to preserve the attempt history, state the selection rule, seek independent repetition, and investigate the mechanism that shaped later exposure. Those are also the ingredients needed to evaluate an AI system without mistaking its most fortunate run for its expected behavior.

The [study package](https://carlosdanieljimenez.com/examples/luck-study.zip) supplies the code, full numerical results, and source and assumption notes. Every simulated quantity in this essay is synthetic; the cited studies retain their own populations and identification limits.
