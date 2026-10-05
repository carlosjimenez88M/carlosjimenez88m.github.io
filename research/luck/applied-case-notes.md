# An empirical reference-class example, with its limits

The public `UCBAdmissions` table distributed by the R project records **4,526 applications to six large Berkeley departments in 1973**. This is a subset of the historical admissions study, not all applications to Berkeley. The CSV preserves the historical table's `Male` and `Female` labels; it contains 12 aggregate rows, not individual applicant records.

Sources verified on 2026-10-05:

- [Official R data source](https://svn.r-project.org/R/trunk/src/library/datasets/data/UCBAdmissions.R), SHA-256 `86a5801b122ec1f172098303de35c9df9a8c4c09f2f81dc9f03bfafdde102d5e`.
- [Official R dataset documentation](https://search.r-project.org/R/refmans/datasets/html/UCBAdmissions.html).
- Bickel, P. J., Hammel, E. A., and O'Connell, J. W. (1975). [Sex Bias in Graduate Admissions: Data from Berkeley](https://doi.org/10.1126/science.187.4175.398), *Science* 187(4175), 398–404. [Full article hosted by UC Merced](https://faculty.ucmerced.edu/jvevea/classes/Spark/readings/Bickel-Berkeley.pdf).

The original paper examines how pooling autonomous departments can conceal their very different application patterns and admission rates. The pooled-weight standardization below is our own descriptive reanalysis, not a reconstruction of every statistical procedure or conclusion in that paper.

## What is computed

Write $p_{sd}=A_{sd}/N_{sd}$ for the recorded admission rate in group $s$ and department $d$. The observed aggregate rate uses each group's own department mix:

$$r_s^{\mathrm{observed}}=\sum_d \frac{N_{sd}}{N_s}p_{sd}.$$

Declare a common reference composition by pooling the application counts across both recorded groups:

$$w_d=\frac{N_{\mathrm{Male},d}+N_{\mathrm{Female},d}}{4526},\qquad r_s^{\mathrm{common}}=\sum_d w_dp_{sd}.$$

This standardizes the composition. It leaves every within-department observed rate unchanged. All calculations use exact rational arithmetic before converting to decimal displays. Run `MPLCONFIGDIR=/tmp/luck-mpl python3 research/luck/applied_case.py` to regenerate results and SVG/PNG offline. The `--refresh-data` option downloads the official R source, checks its frozen hash, parses its count vector, and recreates the CSV; it never executes downloaded source code.

| Summary | Male rate | Female rate | Female minus male |
| --- | ---: | ---: | ---: |
| Each group's observed department mix | 44.52% | 30.35% | −14.16 percentage points |
| Same pooled department weights | 38.73% | 43.00% | +4.26 percentage points |

The direction of this descriptive contrast changes with the declared population composition. Department-specific contrasts are heterogeneous: female rates are higher in A, B, D, and F, while male rates are higher in C and E. The figure preserves this heterogeneity. It must not be described as a reversal in every department.

The pooled weights are one defensible, explicit reference for an illustrative comparison. They are not a uniquely correct fairness target. Another question may require different weights or the original marginal rates.

## What the case does not establish

1. **It changes the estimand, not the facts.** The raw rate describes outcomes under the actual application mix. The standardized rate describes a common mixture of the observed department-specific rates. One does not invalidate the other; they answer different questions.
2. **Department adjustment does not establish the absence of discrimination.** Access, schooling, expectations, or institutional constraints may influence which departments people apply to. Department can mediate prior inequality. Removing its contribution from a summary can remove part of the process one wants to understand.
3. **These aggregates do not supply individual counterfactuals or qualifications.** A common-mixture contrast is not the causal effect of changing an applicant's recorded sex, nor an estimate of ability or deservingness.
4. **These are retrospective cohort frequencies, not a frozen prospective predictive model.** They show why a reference composition matters. Using them to predict a future applicant would require a separate argument about exchangeability and relevant information.
5. **They do not measure luck.** The connection to the essay is methodological: an aggregate outcome acquires meaning only after its reference population and question are specified.

No confidence intervals appear in the figure. The counts identify these descriptive proportions for this recorded cohort exactly. Intervals for future cohorts or a hypothetical admission process require additional assumptions about selection, dependence, population change, and the sampling unit. Imposing independent-binomial sampling here would be an extra model, not something guaranteed by the aggregate table.

Suggested figure alt: “Admission rates for the six-department Berkeley 1973 cohort are 44.5% for male and 30.4% for female applicants under their observed department mixes; common pooled weights give 38.7% and 43.0%. Within-department differences have both signs.”

Suggested caption: “Published aggregate counts; our own pooled-weight standardization. Changing the reference mix reverses this descriptive contrast while leaving the observed departmental rates unchanged. This is neither a causal estimate of discrimination nor an attribution of luck.”

## A study design that separates three kinds of variability

The Berkeley table contains only one admissions cycle and no repeated individual measurements. It cannot identify between-cycle variance or persistent applicant-level differences. Those questions require a different design.

For evaluation, cross independent batches $b=1,\ldots,m$ with the **same** independent units $i=1,\ldots,n$, and obtain $r$ repeats per batch–unit combination. For a continuous score, a simple explicitly stipulated model is

$$Y_{bij}=\mu+B_b+V_i+\varepsilon_{bij},$$

where batch effects $B_b$, persistent unit effects $V_i$, and occasion noise $\varepsilon_{bij}$ are independent with zero means and variances $\tau_B^2$, $\tau_V^2$, and $\sigma^2$. Under this model, the balanced grand mean has

$$\operatorname{Var}(\overline Y)=\frac{\tau_B^2}{m}+\frac{\tau_V^2}{n}+\frac{\sigma^2}{mnr}.$$

Repeating the same units reduces occasion noise; it does not eliminate the persistent-unit term. New independent batches reduce the batch contribution. New independent units reduce the unit contribution. The uncertainty of a mean is therefore distinct from the variability of one new score, $\tau_B^2+\tau_V^2+\sigma^2$.

Crossing units and batches supplies interpretable covariance contrasts:

- Different units in the same batch share covariance $\tau_B^2$.
- The same unit in different batches shares covariance $\tau_V^2$.
- Repeats of the same unit in the same batch share covariance $\tau_B^2+\tau_V^2$.

The corresponding correlations divide each covariance by the total single-score variance. These identities hold under the stated additive model. Batch–unit interactions, learning, interference, temporal drift, and serial correlation would require further terms or a different design.

If batches instead contain **new, disjoint units**, replace $V_i$ with $V_{bi}$. Then

$$\operatorname{Var}(\overline Y)=\frac{\tau_B^2}{m}+\frac{\tau_V^2}{mn}+\frac{\sigma^2}{mnr}.$$

This is why a formula must state whether units are reused across batches. Counting responses alone cannot reveal the effective information in the experiment. In either design, uncertainty in the *estimated* variance components must also be carried into inference; their population identities do not make their estimates exact. A persistent unit component is not automatically ability, and a batch component is not automatically luck: these are variance descriptions under the design, not identified accounts of control or merit.

For the essay's binary shared-environment example, suppose independent batches have $P_b\sim\operatorname{Beta}(\alpha,\beta)$ and $n$ conditionally independent trials each. Let $p=\alpha/(\alpha+\beta)$ and $\rho=1/(\alpha+\beta+1)$. Then

$$\operatorname{Var}(K_b)=np(1-p)\{1+(n-1)\rho\},$$

$$\operatorname{Var}(K_b/n)=p(1-p)\left\{\rho+\frac{1-\rho}{n}\right\},$$

$$\operatorname{Var}\left(\frac1m\sum_b K_b/n\right)=\frac{p(1-p)}m\left\{\rho+\frac{1-\rho}{n}\right\}.$$

At fixed $m$, infinitely many trials per batch cannot remove the contribution $p(1-p)\rho/m$ of varying batch conditions. Adding independent batches can reduce it. The new batch's environment remains variable even when the population mean is estimated precisely.

A fixed unknown $p$ with a Bayesian posterior also yields a beta-binomial prediction when future outcomes share that parameter. A single batch's count distribution cannot distinguish that epistemic mixture from a beta-distributed physical environment. Planned independent batches, unit reuse where relevant, and information about the mechanism provide the additional structure; resemblance between predictive distributions alone does not identify a cause.

Finally, distinguish repeated-sampling precision from a predictive calculation before additional data arrive. If $\theta=(p,\rho)$ is uncertain given historical data $D$, the new batch proportion $Q=K/n$ has

$$\operatorname{Var}(Q\mid D)=\mathbb E\left[p(1-p)\left\{\rho+\frac{1-\rho}{n}\right\}\middle|D\right]+\operatorname{Var}(p\mid D).$$

The first term averages outcome variability over the current parameter posterior. The second is uncertainty about the population mean. For the average of $m$ as-yet-unobserved batches that are independent conditional on $\theta$, only the first term divides by $m$:

$$\operatorname{Var}(\overline Q_{\mathrm{future}}\mid D)=\frac1m\mathbb E\left[p(1-p)\left\{\rho+\frac{1-\rho}{n}\right\}\middle|D\right]+\operatorname{Var}(p\mid D).$$

Averaging hypothetical future batches does not itself create information about $p$. Observing additional independent batches and updating $D$ can reduce that posterior uncertainty. Confusing these two operations would make a wider predictive distribution appear to contradict improving precision of a learned mean.
