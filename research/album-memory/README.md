# Album memory pilot

This is a lyric-based interpretive experiment, not an acoustic analysis or a
measurement of human or neural attention. Read `design.md` before interpreting
the results. Four intentionally selected albums cannot represent four artists'
discographies.

## Reproduce

The recorded environment uses Python 3.13. Create an isolated environment and
install `requirements.lock.txt`. Supply `GENIUS_API_TOKEN` and `OPENAI_API_KEY`
through the repository-root `.env` or the process environment; never commit them.
The model stages incur provider charges. Repeated commands reuse cached completed
calls; changing prompts or parameters creates new requests.

From the repository root:

```sh
python research/album-memory/collect.py
python research/album-memory/test_graph.py
python research/album-memory/study.py prepare
python research/album-memory/study.py run
python research/album-memory/analyze.py
python research/album-memory/figures.py
python research/album-memory/verify.py
```

The collector validates artist and title, records source URLs and hashes, and
keeps lyrics in ignored `.research-cache/album-memory/`. The original 11 listed
tracks of *Infest* plus its hidden song *Tightrope* become 12 analytical units.
This changes the unit count, not the claimed edition. Track positions for other
albums follow the editions in `design.md`.

Preparation makes 56 anonymous evidence cards, rolling summaries, and three
questions per album. These are model-produced silver annotations. The model may
recognize lyrics despite withheld metadata. English paraphrases of Spanish lyrics
add translation loss. The corpus contains lyrics only: arrangement, timbre,
performance, and musical transitions are outside the observation set.

The run compares four policies on the same 12 questions, with three stochastic
repetitions each (144 answers and 144 blind-policy judge calls). Model judging is
not independent human validation. Case averages precede album and overall means;
repetitions are not 144 independent musical examples.

## Memory and tracking

LangGraph ingests cards into an archive and selects only the requested context
for generation. The archive is intentionally separate from the rendered prompt.
`InMemorySaver` demonstrates thread checkpoints but is not durable across process
restarts. The JSON response cache enables experiment resumption independently of
those checkpoints. Replacing it with a persistent checkpointer would not itself
reduce model tokens.

MLflow stores its SQLite run/trace database in the ignored research directory
and local run artifacts in ignored `mlruns/`. Prompt definitions are registered separately from their concrete input
data. The results include numeric prompt URIs and per-run identifiers. The
SQLite database and trace payloads remain private. No remote tracing service is
needed. Preprocessing usage is accounted for from provider receipts separately
from the answer and judge stages.

## Reading exports

- `corpus_manifest.json`: edition positions, provenance, counts, content hashes.
- `runs.json`: per-answer outputs, judge ratings, selected IDs, provider usage.
- `case_means.csv`: repetitions collapsed for each album/question/policy.
- `album_means.csv`, `policy_means.csv`: descriptive balanced means.
- `usage_by_stage.json`: actual input/output tokens, including preprocessing.
- `arc_diagnostics.json`: exploratory theme adjacency relative to 10,000 seeded
  order shuffles. This tests the chosen representation, not author intent.

Exported evidence cards are paraphrases with source line references, never full
lyrics. Inspect them critically before making interpretive claims. Target-ID
availability in a summary means the identifier survived; it does not guarantee
the associated evidence survived. Coverage judgments provide a separate,
fallible check. A question's designated pair is useful for comparison but does
not exhaust every defensible answer.

## Offline review without API calls

After extracting the package, run `analyze.py` and `figures.py` with the locked
environment. Aggregation falls back to the included public evidence cards and
retains the exported stage usage when private provider receipts are unavailable.
`verify.py` additionally requires the original private corpus, receipt cache, and
MLflow database; the included verification report records that local audit.
