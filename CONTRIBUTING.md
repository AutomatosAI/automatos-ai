# Contributing to Automatos AI

Welcome! We’re excited you’d like to contribute. This is a concise guide. For the full, living guide see `docs/CONTRIBUTING.md`.

## How to Contribute
- Open an issue describing the problem or proposal before large changes.
- Fork the repo and create a feature branch from `main`.
- Write clear commits and include tests where applicable.
- Open a Pull Request with a descriptive title and context.
- Sign off every commit (`git commit -s`). The `Signed-off-by:` trailer is your
  [Developer Certificate of Origin](https://developercertificate.org) attestation;
  the `dco` check on each pull request verifies it. There is no CLA.

## Where to contribute capability
The platform is one codebase that ships as two editions: the **local edition**
you run yourself (`docker compose up`, no accounts) and the **hosted edition**.
Capability contributions land best where they never conflict with core:
**skills, tools, MCP integrations, Playbooks and agent packages first, core
second.** A new skill or tool needs no knowledge of the orchestrator's
internals and reaches both editions unchanged; a core change needs the
schema-drift, route-contract and fresh-clone smoke lanes to stay green in
both. Open an issue first for anything that touches auth, storage, the
tool router or a migration.

## Development Setup
1. Clone via SSH: `git clone git@github.com:AutomatosAI/automatos-ai.git`
2. Install dependencies per README in each package (`orchestrator/`, `frontend/`, etc.).
3. Use `pre-commit`/linters if configured; ensure no linter errors and tests are green.

## Coding Standards
- Follow the existing style per component (Python: PEP8; TS/JS: project ESLint/Prettier rules).
- Keep functions small and well-named; add docstrings for complex logic.
- Avoid breaking changes without discussion; maintain backward compatibility where possible.

## Pull Request Checklist
- [ ] Linked issue (if applicable)
- [ ] Clear description of change and rationale
- [ ] Tests and docs updated (README or `docs/` as needed)
- [ ] No linter/build errors

## Code of Conduct
Participation is governed by our [Code of Conduct](./CODE_OF_CONDUCT.md). Report unacceptable behavior to support@automatos.ai.

## License
By contributing, you agree your contributions are licensed under the repository’s
license (Apache-2.0). Under §5 of that license your contribution may be
distributed in every edition of the platform — including the hosted and
commercial editions run by the maintainers — under the same terms, with no
separate agreement. That is the whole deal: what you contribute stays
Apache-2.0 for everyone, and it ships everywhere the code ships.

For the detailed guidelines, templates, and examples, please see: `docs/CONTRIBUTING.md`.


