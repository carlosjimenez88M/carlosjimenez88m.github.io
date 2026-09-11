# Part II — protocol fixed before new answer results

Question: can adaptive relational memory recover and attribute narrative evidence
with less exposed context and lower trajectory token use than static retrieval?
The original published experiment remains intact. Its cards are reused; this is
an extension of one four-album pilot, not an independent replication.

Corpus: the same 56 cards and private source lyrics. Construct a provenance-linked
relation graph BEFORE constructing new questions. Edges are model interpretations,
not demonstrated author intent or calibrated confidence. Preserve their supporting
track:claim addresses; reject unknown addresses.

Tasks: one newly generated question per album for local, distant, transformation,
contradiction, multi-hop, and abstention (24 questions). Public task constraints
specify required structure; silver target IDs and answerability labels remain
outside retrieval, sufficiency, generation, and online verification. Independently
audit negative questions against the available archive; absence from this archive
does not establish absence from the underlying music.

Policies: full archive; fixed recent-three; original rolling summary; TF-IDF;
embedding retrieval; relational retrieval; adaptive relational retrieval. Three
retrieval policies are evaluated with 400/800/1200/1600/2400 context-token caps,
plus full context as the unbounded reference. The three original controls are
single points, not repeated under fictitious budgets. This yields 19 settings
and 456 primary answer trajectories. One generation per setting/question;
variation across these fixed questions is descriptive, not a stochastic CI.

Budgets apply to the serialized exposed memory under the model tokenizer,
excluding fixed instructions and question. Provider input usage is recorded
separately. Cards are packed whole, never silently truncated. Relation retrieval
retrieves edges then expands supporting cards. Edge tokens count whenever they
are exposed. If nothing fits, the system must acknowledge insufficient evidence.

Adaptive starts at 400, can grow to 800,1200,1600,2400 and full. A sufficiency
check sees only question, public constraints, and selected evidence; it cannot
see silver targets. A verifier checks individual answer claims against cited
source claims and checks the question, then can trigger expansion. Bound retries
to the six budget levels. Distinguish exhausted evidence from verified answers.

Metrics: item-level target availability; citations conditional on availability
(NA if denominator zero); per-claim model-verified attribution; observed provider
input/output tokens across the WHOLE trajectory; total tokens per supported
non-abstaining answer (undefined if none). Correct negative-case abstention is a
separate success. Report final-generation vs controller/verification costs,
latency, iterations, and preparation amortization. Do not label model checks as
human verification or cited context as causal attention. Dollar estimates require
a dated primary price source and explicit cache-token treatment; token economics
are sufficient if pricing is unavailable.

Figures: budget/utilization curve; empirical Pareto frontier using verified-by-
model answer rate and trajectory tokens; evidence availability/use/attribution
funnel; relation graphs; distance/task breakdown; cumulative amortization per
album and across a balanced workload. Frontier dominance is descriptive over
observed means; it is not a universal ranking.

Generalization: thresholds must not be selected using held-out album outcomes.
Use leave-one-album-out to choose among predeclared adaptive stop settings using
the other three albums' recorded decisions, with honest replay limitations.
No claim of unseen-artist generalization: all four corpora were previously read.

Multi-turn: four linked turns per album under full and adaptive policies, with
persistent thread IDs, bounded exposed history, and explicit MLflow session IDs.
Report history and evidence token use separately. This is a follow-up diagnostic,
not interchangeable with independent primary questions.

Human review: create a policy-blinded sample of 48 answers, balanced over albums
and task types. Obtain actual user/reviewer labels before claiming human validity.
Do not invent annotations, agreement, or kappa. If not available, deliver a ready
review packet and clearly mark the limitation.

No automatic publication of Part II. All API calls cached and resumable; lyrics,
credentials, raw traces, and private call inputs remain ignored locally.

Execution clarification before answer calls: primary adaptive threshold is 3.
Also run fixed thresholds 2 and 4 (48 additional trajectories, 504 total) to
permit honest leave-one-album-out selection from actually observed outcomes.
Shared deterministic prefixes reuse cached receipts; report both logical
trajectory tokens and unique billed-call receipts. This is not a randomized
comparison of latency between concurrent cold requests.
