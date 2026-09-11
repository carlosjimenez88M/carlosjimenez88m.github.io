# Album memory experiment — design fixed before model results

Question: Can an explicit, bounded context-selection policy preserve evidence for album-level interpretive questions while reducing generation input tokens?

Corpus: one intentionally contrasting album each from Red Hot Chili Peppers (Californication, original 15 tracks), Papa Roach (Infest, original 11 listed tracks plus separately indexed hidden Tightrope), The Beatles (Abbey Road, UK 17 tracks), Natalia Lafourcade (De todas las flores, 12 tracks). Verify editions and treat hidden material explicitly. No population claim about the artists.

Data: retrieve lyrics via configured Genius credential; validate artist/title, preserve source URLs and hashes, keep full text and raw API content ignored locally. Reuse available Beatles corpus only with provenance recorded. Never include full lyrics in published artifacts.

Stage A: anonymized per-track analytical evidence cards (English paraphrases for all languages), bounded structured fields, line IDs for internal audit; no lyric quotations. Separate narrator from artist. The card is a lossy, model-produced interpretation, not ground truth.

Stage B: fixed memory policies, identical model/prompt/output schema. Full card archive; most recent three cards; rolling bounded summary; selective retrieval (opening anchor, two recent cards, query-relevant cards up to six total). Three independently generated album questions with separated earlier and later evidence at album end; questions generated before answer runs. Three repetitions per case. Add deterministic evidence availability and retrieval metrics, alongside a separately prompted judge reading the full cards and expected evidence (silver references).

LangGraph: ingest/store cards in ordered state, select memory, answer, validate references; distinct thread IDs/checkpoints for each album/policy/question/repetition. No cross-album memory. MLflow: prompt versions, graph traces, actual provider token usage, failures, per-case scores. Checkpoint storage alone does not save prompt tokens.

Metrics: actual input/output tokens, source-ID validity, target evidence availability/recall, model-judged support and coverage, answerability/abstention; separate preprocessing, summary-update, generation, and judge usage. Track concentration is a context-allocation proxy, never neural attention or listener cognition. Comparison on same cases; aggregate repetitions per case and albums equally; no significance claim from only four albums.

Interpretation: obtain track-level thematic/tension profiles from cards, compare canonical order to fixed-seed shuffles as exploratory continuity diagnostic; similarity never proves intentional narrative. Investigate individual distant returns with counterevidence. Describe only what data supports. Distinguish album-level observations from memory-system results and from hypotheses about future agents.

Budgets: a bounded pilot (~56 track annotations, 56 rolling updates, 12 question sets, 144 answers and 144 judge evaluations at most, plus one embedding batch if needed). Cache completed calls, record actual usage, do not repeat paid calls merely for testing. Stop on repeated auth/billing errors. No automatic publishing requested for this new post yet.
