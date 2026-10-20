# hello-c03b — the package behind the admin review queue (PRD-261 C-03b)

The fixture behind `specs/admin-theme-review.spec.ts`. The spec submits it to the local backend's
Marketplace (`POST /api/marketplace/themes/submit`, no route interception), then approves it in
`/admin/themes` as the local operator, who is the instance's super admin (`hybrid.py:642`).

| File | What it is |
|---|---|
| `build.py` | Builds A-06a's `hello` package under the id `automatos-e2e/hello-c03b`, so it never meets another spec's upload or submission (the Marketplace refuses the same id and version with other files, and a version not higher than one it holds). The spec runs it with `python3 build.py --out <file>`. |

The lane's backend has no LLM key, so the review stage's LLM check is a warning, "LLM stage did not
run", and the verdict is `review_required`: approve is allowed (only `blocked` and `not_run` are
refused), and on the local edition no isolated-job note is needed.
