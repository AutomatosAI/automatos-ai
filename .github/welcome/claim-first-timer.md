Welcome @{login}, and thanks for jumping in! 🎉 {assigned_line}

Before you start, two short reads:

- **[CONTRIBUTING.md]({repo_url}/blob/main/CONTRIBUTING.md)**: how we work. Sign off every commit (`git commit -s`, the DCO check fails without it), and open a pull request within 7 days of claiming (a draft counts). There's also the PR template to fill in honestly, including what you actually ran.
- **[AGENTS.md]({repo_url}/blob/main/AGENTS.md)**: the code rules CI checks: functions of 50 lines or fewer on the lines you change, every fix shipped with a test that fails without it, explicit `git add` paths. If you use an AI coding agent (Claude Code, Cursor, Copilot…), point it at this file first; it's written for them.

One issue at a time until your first pull request is merged. After that, no need to ask: comment on any open, unassigned issue to claim it and go.

Tests run in CI on the pull request. Locally, backend tests are `docker compose exec backend pytest tests/<file>` and frontend tests `cd frontend && npm run test`. Ask here if anything's unclear.

And if Automatos is useful to you, a ⭐ on the repo would mean a lot. It helps more people find the project. Looking forward to the PR!

---
_This reply is automated. A maintainer reads every thread and will add any pointers specific to this issue._
