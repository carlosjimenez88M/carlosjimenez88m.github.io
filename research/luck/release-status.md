# Release verification — October 5, 2026

The author requested a critical rereading, clearer graphics, and the first
publication today, followed by a substantive critique. The cadence is October 5,
12, and 19. The expanded edition uses date-free public URLs and retains the
previous paths as redirects. Original source filenames and Buttondown slug
identities remain stable, preserving the same email ids.

| Part | Blog release (America/Bogota) | Buttondown delivery (America/Bogota) |
| --- | --- | --- |
| I | October 5, today | October 5, 10:07:43 — sent |
| II | October 12, 08:00 | October 12, 09:00 |
| III | October 19, 08:00 | October 19, 09:00 |

The initial first edition was verified over HTTPS with status 200, its
revised title and argument, and four valid PNG illustrations. Buttondown
independently reports the same existing email as `sent`, with the complete
revised body and publication time October 5, 10:07:43 America/Bogota
(15:07:43 UTC). No replacement email or second publication request was made.
Its initially returned `scheduled` status with a current-time publication date
was a brief intermediate state; later readback established the actual send.
Parts II and III were updated on their original ids and independently retrieved:
`scheduled` for October 12 and 19 at 09:00 America/Bogota, with exact matches to
that edition's subject, full body, canonical URL, and unrestricted audience.

The subsequent expansion names the reference-class problem, compares modal and
control accounts of luck, adds a descriptive analysis of 4,526 public Berkeley
application records, specifies a repeated-batch design, and distinguishes
population reliability from individual attribution. Part III derives pass@k and
proposes policy interventions, credible perturbations, and state-consistent replay.
That agent protocol is a research design; it has not been run on a live system.
The first essay now contains five figures; the complete package contains eight.
The expanded source and local browser preview have passed verification. Live
deployment and same-id newsletter/archive synchronization are pending below.

The existing Codex follow-up “Publicar serie estadística sobre la suerte” was
updated for October 12 and 19, ending October 19. Local blog publication requires
the machine and app to be available; Buttondown delivery is hosted independently.
No fourth installment is scheduled.

Publication-boundary builds passed for the current time, immediately before the
first release, and just after each of the three releases. The results are in
`results/publication-checks.json`. Rendered essays have no missing local links;
unreleased installments appear as dated upcoming text. Browser inspection of
all three expanded essays found no KaTeX error nodes. The new real-data figure
was inspected within the article; all eight figures were inspected at desktop
and mobile figure widths.

Numerical checks in `verify.py` include exact tail sums, comparisons at equal
means, shared-environment variance, selected Gaussian maxima, ordered-path urn
equivalence, the forced-first-win intervention contrast, sampling variance,
recorded interval inputs, Monte Carlo coverage, the Brier identity, exact pass@k
expectations, ICC, repeated-measurement reliability, and an independent
reconstruction of the Berkeley counts and standardization. Model calculations
validate their stated assumptions; the public-data case is descriptive.

`verify_release_workflow.py` passed using a temporary checkout and local bare
remote: due-only release, idempotent reruns, stale-output cleanup, dirty-checkout
rejection, boundaries recalculated after pull, and source-collision protection.
`verify_newsletter_workflow.py` passed 17 simulated API tests for same-id content
and date updates, immediate publication, live-page and image checks, credential
isolation, archive-only revision with preserved send state, and recovery without
repeated POSTs after ambiguous requests. These
tests make no production push or real API call. Results are recorded in
`results/workflow-check.json` and `results/newsletter-workflow-check.json`.

Actual API receipts with email ids and body hashes remain in the ignored
`.research-cache/luck-buttondown-receipts.json`; credentials remain in `.env`.
