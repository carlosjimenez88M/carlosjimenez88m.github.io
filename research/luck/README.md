# A Statistical Account of Luck

Three English essays for The Probability Engine: October 5, 12, and 19, 2026.
Research and calculations completed October 5, 2026. The dates are release dates,
not dates of data collection. Author: Carlos Daniel Jiménez.

This is a methodological synthesis using primary sources and deliberately simple
generative examples, plus a descriptive reanalysis of public admissions counts.
It is not an original study identifying luck in people's success, a systematic
literature review, or a validated attribution scale. No embedding or generation API
was needed. The code makes no provider calls.

## Reproduce the demonstrations

From the repository root, with Python 3.13.2 and the versions in `requirements.txt`:

```sh
MPLCONFIGDIR=/tmp/luck-mpl python3 research/luck/analysis.py
MPLCONFIGDIR=/tmp/luck-mpl python3 research/luck/graphics.py
MPLCONFIGDIR=/tmp/luck-mpl python3 research/luck/applied_case.py
python3 research/luck/verify.py
```

The executable uses seed 20261005. `results/summary.json` records parameters,
software versions, full-precision results, and the analysis script's SHA-256.
`results/synthetic-user-means.csv` contains the forty synthetic user averages.
The analysis writes baseline figures; run `graphics.py` afterward to replace them
with seven synthetic/conceptual SVG and PNG figures in `static/img/luck/`.
`applied_case.py` adds the eighth figure from the frozen public table. Its source
provenance and exact rational results are in `results/applied-case.json`;
`verify.py` independently reconstructs the summaries and records the reliability
and pass@k checks in `results/design-checks.json`. The synthetic figures’ exact
comparisons are recorded separately in `results/graphics.json`. Numerical results
are deterministic for the recorded environment; figure metadata can differ on rerun.

For the downloadable package, extract its contents at the root of an empty folder
and run the same commands. The archive preserves repository-relative paths.

## Design and interpretation

| Demonstration | Design | What it establishes under the model | What it does not establish |
| --- | --- | --- | --- |
| Public-data reference | 4,526 Berkeley applications in six departments, 1973; observed and common composition | A descriptive contrast changes under a different declared population mix | Discrimination, qualifications, or individual luck |
| Reference distributions | Exact Binomial(20,p), p=.5,.6,.7 | An outcome's excess and tail depend on its reference | The appropriate reference for a person |
| Unknown probability | Beta(1,1) prior, 12/20 earlier trials, new 20-trial prediction; mean-matched plug-in comparison | Parameter uncertainty changes prediction even when the mean is held fixed | The fixed-p conditional independence of changing real populations |
| Shared environment | Batch P~Beta(2.4,1.6), 20 conditional trials | Common environment increases marginal variance | Whether a wide observed distribution reflects this mechanism |
| Selection | 20,000 worlds per pool size; latent N(70,4²), error N(0,6²) | Selection finds higher means and favorable noise | A calibrated correction for dependent real leaderboards |
| Collider | 200,000 independent standard-normal pairs, retain sum>2 | Inclusion alone can create association | An empirical relationship between human attributes |
| Reinforcement | Symmetric (1,1) Pólya urn, 500 draws, 20,000 worlds | Early variation persists under the stipulated update | Feedback identified from final shares or sequences |
| Clustered evaluation | 40 users × 3 tasks × 4 runs, Gaussian differences | Dependence determines precision; check across 4,000 panels | Validity of a real rubric or representativeness of a sample |

The urn is generated through its **exact beta-binomial marginal representation**,
not by sequentially updating weights. This is mathematically equivalent for the
reported counts. It also exposes the observational equivalence between feedback
and a fixed latent propensity drawn once per world.

The Gaussian scores are unbounded synthetic units. They are not accuracies,
human ability measurements, or model-service results. The clustered example uses
paired differences, so its components represent variability of those differences.
The user-level t interval is justified here by iid normal user means. These
conditions must be re-examined for any real application.

With the full repository, `python3 research/luck/verify_release_workflow.py`
also exercises publication in a temporary checkout and a local bare remote:
due-only release, idempotent reruns, generated-file staging, stale-output removal,
recalculation after synchronization, source-collision protection,
and rejection of unrelated unfinished changes. It does not push to production.
`python3 research/luck/verify_newsletter_workflow.py` uses mocked API responses to
check rescheduling, edited bodies, public-page verification, duplicate prevention,
and recovery from ambiguous requests; it does not contact Buttondown.

See `sources.md` for the scope of primary-source reading and `critical-review.md`
for claims challenged or narrowed. `verify.py` checks results against exact finite
sums, quadrature, exchangeability identities, intervention predictions, and the
analytical sampling variance. `verify_publication.py` builds the current release
and all three temporal boundaries, checking the essays’ local links.

## Releases

The first essay has a release date of October 5; subsequent essays have future
`publishDate` values at 08:00 America/Bogota on October 12 and 19. The first
newsletter is published immediately after the live blog is verified. Subsequent
Buttondown deliveries are scheduled at 09:00, adapted for email:
absolute asset links, PNG figures, native display-math blocks, and readable Unicode
inline notation. `export_newsletters.py` generates those files from the essays.
Future installments are described by date until their release, so early emails
do not link to unavailable pages.

Hugo is a static generator. A future date alone cannot update the deployed site.
The scheduled Codex follow-up runs `scripts/publish_luck_series.py` to rebuild and
push due articles. Local scheduled work requires the machine and Codex app to be
running. Buttondown's scheduled delivery is hosted independently. The publication
manifest records the exact three releases. `canonical_url` is the public URL;
`slug` is the original source filename and stable Buttondown identity. The Hugo
slugs omit dates and preserve the original paths as aliases; newsletter API receipts are local and
ignored in `.research-cache/luck-buttondown-receipts.json`.

The October 5 expansion is logged in the first essay's editorial note and the
site correction log. Update scheduled bodies with `--sync-bodies`; use
`--sync-archives` for a reviewed revision of an already-sent public archive.
Archive updates preserve id, slug, status, and send time and make no publication
request. They cannot alter a message already received by a subscriber. The local
receipt preserves the delivered-body hash separately from the revised archive.
