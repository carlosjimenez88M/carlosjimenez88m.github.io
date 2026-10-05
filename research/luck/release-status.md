# Release preparation — October 5, 2026

All three complete newsletter bodies were created with status `scheduled` in
Buttondown and retrieved independently through its API. Persisted subject,
body, canonical URL, unrestricted audience, status, and timestamp matched the
local release manifest. No email was sent during preparation.

| Part | Blog rebuild (America/Bogota) | Buttondown hosted send (America/Bogota) |
| --- | --- | --- |
| I | October 12, 08:00 | October 12, 09:00 |
| II | October 19, 08:00 | October 19, 09:00 |
| III | October 26, 08:00 | October 26, 09:00 |

A Codex thread follow-up named “Publicar serie estadística sobre la suerte” is
active for those Mondays, ending October 26. Local publication depends on the
machine and app being available. There is no fourth installment. The follow-up
checks the deployed URLs and must not create duplicate newsletter deliveries.

Publication-boundary builds passed at five clock values: today; immediately
before the first release; and just after each of the three release times.
The results are in `results/publication-checks.json`. Published-series pages
had no missing local links in these builds. Later installments appear as dated
upcoming text until available. Browser inspection found no KaTeX error nodes or
unrendered display delimiters in the final previews of the three essays.

Numerical checks passed in `verify.py`: exact finite sums, maximum-score
quadrature, ordered-path urn equivalence, sampling variance, recorded interval
inputs, Monte Carlo coverage, and the Brier identity. The four figures were also
visually inspected. This verification does not validate the synthetic models
against any real-world population.

The publisher also passed an isolated end-to-end check using a temporary checkout
and local bare remote. It published only the first due essay, pushed the local
remote, preserved source files, removed stale generated output, avoided a new
commit on rerun, and rejected a dirty checkout without staging unrelated edits.
See `results/workflow-check.json`. The check did not push to production.

API receipts with email ids and body hashes are kept in the ignored local file
`.research-cache/luck-buttondown-receipts.json`; credentials remain in `.env`.
