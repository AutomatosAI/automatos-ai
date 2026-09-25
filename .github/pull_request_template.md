## What and why

<!-- What does this change do, and why is it needed? Link the issue if there is one. -->

## How it was verified

<!-- What you ran or checked and what you saw: tests added or updated, CI, a manual check. Only claim what you saw. -->

- [ ] Tests added or updated, and they fail without this change
- [ ] CI green (a lane marked non-required counts too)
- [ ] Both editions considered: local (`AUTH_EDITION=local`) and hosted
- [ ] UI change: screenshots below

## AI assistance

<!-- Pick one, and name the tool if you used one. -->

- [ ] None
- [ ] Assisted: tool(s): ______; it wrote or changed: ______
- [ ] I have read and understood every line of this change, and can explain it in review

## Checklist ([AGENTS.md](../AGENTS.md))

- [ ] Every commit signed off (`git commit -s`)
- [ ] No secrets, customer data or credentials anywhere in the change
- [ ] Settings through `orchestrator/config.py`; no hardcoded values
- [ ] Replaced paths deleted in this pull request (no shims or `V2` copies)
- [ ] Migration (if any): one head, chained onto the current head, safe on a `create_all`-built database, head pins moved
- [ ] New routes the frontend calls are in `orchestrator/reports/route-manifest.json`
- [ ] Docs updated where behaviour changed
