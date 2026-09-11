# Memory Is Not Context — Part II

Seven memory policy families; five fixed context budgets for three retrieval
methods; a bounded adaptive controller; six question types across four albums.
The fixed primary grid contains 456 trajectories. Two extra adaptive thresholds
add 48 cases for leave-one-album-out selection. Eight four-turn conversations add
32 separately reported turns. All scores are automated; 72 blinded human-review
items are prepared but no human labels have been collected.

## Read the results without calling a model

Use Python 3.13 with the included requirements lockfile. Extract the package so
that both `research/album-memory/results` and `research/album-memory-v2` retain
their paths, then run:

```sh
python research/album-memory-v2/analyze.py
python research/album-memory-v2/figures.py
```

These commands use public experiment outputs. Figures are written under
`static/img/album-memory-v2`. `human-review/index.html` is a standalone local
review form: download the ratings JSON after filling it in. The policy key is
kept privately and is excluded from the package. Human reviewers should work
independently. Nothing in the form is transmitted.

## Live reproduction

Place your own `OPENAI_API_KEY` in the repository-root `.env` or process
environment. The original lyric collection is already documented in Part I;
this experiment reuses its frozen analytical cards, not a new transcription.

```sh
python research/album-memory-v2/prepare.py
python research/album-memory-v2/test_engine.py
python research/album-memory-v2/run.py --smoke
python research/album-memory-v2/run.py
python research/album-memory-v2/sessions.py
python research/album-memory-v2/analyze.py
python research/album-memory-v2/human_review.py
python research/album-memory-v2/verify.py
```

Live commands incur provider usage. `prepare.py` reuses complete frozen public
preparation outputs if present. Remove the relevant prepared files deliberately
if you want to generate different relations/questions. The run commands reuse
local private completed cases and request caches. A downloaded public package
does not contain private API call caches or the tracking database, so it cannot
skip live requests merely because public answer records are present.

Never commit `.env`, `.research-cache`, raw lyrics, or the private review key.
The SQLite MLflow database and trace artifacts live in the private cache. The
verification command requires those original private receipts and database;
the included report records the audit performed on the original execution.

## What the metrics mean

- Availability counts designated target tracks exposed as source cards. An ID in
  the legacy summary does not qualify as source-claim exposure.
- Utilization counts citations conditional on target exposure. Its denominator
  changes with retrieval; it does not measure causal attention.
- Quality in the CSV is online-verifier acceptance among positive questions.
  The adaptive controller retries against that same verifier. It is not an
  independent correctness estimate.
- Post-hoc screened quality additionally rejects explicit track-name/source-ID
  mismatches. This necessary check cannot establish semantic support.
- Tokens per accepted answer count the entire workload's trajectory input and
  output, including negative questions and failures, divided only by accepted
  positive answers. No accepted positives means undefined.
- Trajectory receipts can be reused across thresholds. Logical policy resource
  use and unique completed provider-call totals are separate exports. Neither is
  automatically a dollar cost estimate.
- Bubble sizes use summed receipt latencies. Concurrent and cached execution
  does not establish production latency or cold-start performance.

The primary grid has one generation per question/setting. Neither apparent
nonmonotonicity nor the observed best budget proves an optimal policy for new
questions. Leave-one-album-out selects only adaptive thresholds; it does not
validate selecting the best embedding budget after looking at the whole grid.

## Recorded deviations and diagnostics

The initial smoke test found a verifier merging claim judgments. Native structured
output now requires a separate indexed verdict per claim. Four primary cases
initially failed on malformed generator source fields; the completed run retains
those outputs as format failures, rather than converting them into correct
abstentions. Previously completed cases and requests were reused.

After inspecting an accepted Infest answer, an explicit track-ID alignment check
was added as a post-hoc sensitivity analysis. Online acceptance and replayed
controller behavior were not rewritten. The article describes the false positive.

No human agreement or kappa is reported. The negative questions and source cards
are model-produced and model-audited. Previous corpus inspection and overlapping
musical concerns limit claims of novelty and generalization.

## Editorial revision, September 11, 2026

`editorial-review/audit.json` contains 72 qualitative AI review notes for full, embedding 1600 and adaptive threshold 3. This was a post-hoc, non-blind review of answer prose, claims, questions and source cards, not human validation. It adds no accuracy estimate and changes none of the 504 original trajectories. `editorial-review/build.py` packages the written notes; it does not generate judgments. `revision_figures.py` reproduces the task, conversation and receipt figures offline. The blank human packet now includes all three policies and has no completed ratings. Proposed controlled ablations and repeated runs in the revised article remain unexecuted.
