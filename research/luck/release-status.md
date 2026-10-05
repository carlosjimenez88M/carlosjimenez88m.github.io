# Release verification — October 5, 2026

The author requested a critical rereading, clearer graphics, and the first
publication today. The revised cadence is October 5, 12, and 19. Stable slugs
retain their original date prefixes so canonical links and Buttondown ids do not
change when a release is advanced.

| Part | Blog release (America/Bogota) | Buttondown delivery (America/Bogota) |
| --- | --- | --- |
| I | October 5, today | October 5, 10:07:43 — sent |
| II | October 12, 08:00 | October 12, 09:00 |
| III | October 19, 08:00 | October 19, 09:00 |

The first essay is public and was verified over HTTPS with status 200, its
revised title and argument, and four valid PNG illustrations. Buttondown
independently reports the same existing email as `sent`, with the complete
revised body and publication time October 5, 10:07:43 America/Bogota
(15:07:43 UTC). No replacement email or second publication request was made.
Its initially returned `scheduled` status with a current-time publication date
was a brief intermediate state; later readback established the actual send.
Parts II and III were updated on their original ids and independently retrieved:
`scheduled` for October 12 and 19 at 09:00 America/Bogota, with exact matches to
the revised subject, full body, canonical URL, and unrestricted audience.

The existing Codex follow-up “Publicar serie estadística sobre la suerte” was
updated for October 12 and 19, ending October 19. Local blog publication requires
the machine and app to be available; Buttondown delivery is hosted independently.
No fourth installment is scheduled.

Publication-boundary builds passed for the current time, immediately before the
first release, and just after each of the three releases. The results are in
`results/publication-checks.json`. Rendered essays have no missing local links;
unreleased installments appear as dated upcoming text. Browser inspection of
all three revised essays, and of the live first essay, found no KaTeX error nodes. Its four graphics and the
other three series figures were inspected at desktop and mobile figure widths.

Numerical checks in `verify.py` include exact tail sums, comparisons at equal
means, shared-environment variance, selected Gaussian maxima, ordered-path urn
equivalence, the forced-first-win intervention contrast, sampling variance,
recorded interval inputs, Monte Carlo coverage, and the Brier identity. They
validate calculations under stipulated models, not their empirical relevance.

`verify_release_workflow.py` passed using a temporary checkout and local bare
remote: due-only release, idempotent reruns, stale-output cleanup, dirty-checkout
rejection, boundaries recalculated after pull, and source-collision protection.
`verify_newsletter_workflow.py` passed 16 simulated API tests for same-id content
and date updates, immediate publication, live-page and image checks, credential
isolation, and recovery without repeated POSTs after ambiguous requests. These
tests make no production push or real API call. Results are recorded in
`results/workflow-check.json` and `results/newsletter-workflow-check.json`.

Actual API receipts with email ids and body hashes remain in the ignored
`.research-cache/luck-buttondown-receipts.json`; credentials remain in `.env`.
