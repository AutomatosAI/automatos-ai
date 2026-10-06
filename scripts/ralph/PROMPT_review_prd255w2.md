# Ralph Review Prompt — PRD-255 Brand Kit v2, Wave 2 (the brand board, the Brand designer, Auto's brand routing)

You are a fresh-context **adversarial reviewer**. The build claims PRD-255 Wave 2 is complete. Find where:
- the kit can change without the owner's Approve, or Auto changes it itself on a brand ask;
- an agent tool reads or writes another workspace's template, Deliverable, kit or folder;
- a starter can be changed or deleted, or a template bypasses the studio's validation;
- a write action escapes the hierarchy gate, or a tool is built outside `registry.register`;
- the board shows something that is not the kit (an invented variant, a generated logo, data fields);
- the designer is duplicated, read from a file at runtime, or seeded on a runtime its edition cannot run;
- a story's evidence does not exist.

You fix NOTHING in the code yourself.

## Scope

```
BASE=$(bash scripts/ralph/acceptance-prd255w2.sh --print-base)
git diff --stat $BASE..HEAD
git diff $BASE..HEAD
```

The base is Wave 1's fork point while this branch is stacked on `feat/prd-255-w1-brand-kit-v2`, and the `origin/main` merge-base once Wave 1 is on `main`. Wave 1's code is the base: never file a finding against a line this diff does not touch, unless this wave's change makes it reachable. Read:
- `scripts/ralph/prd-255w2.json` (binding, with its `decisions` and each story's `notes`: the loop's choices for the PRD's ambiguities are the contract, not findings);
- `docs/PRDS/PRD-255-BRAND-KIT-V2.md` (FR-9..FR-12, US-009..US-014);
- the `ownerTest` list in the JSON (the owner's checks; the loop must not claim them).

## Hunt list: every item is a confirmed-risk class

1. **Approval before a kit change (FR-11).**
   - The designer's flow saves through `platform_update_brand_kit` only after a granted Approve on its proposal card; the proposal is rendered without saving (assert nothing is written to `workspace.settings` by the proposal path).
   - A non-Approve answer ("less orange") is a revision, never a save.
   - Auto's brand note files a ticket for the designer; nothing in the note's path calls the kit writer.
   - A save reachable without an Approve = CRITICAL.
2. **Tenant isolation.**
   - `render_preview`, the board route, `create_template` / `update_template` and the proposal tool resolve every id inside the caller's (or the ticket's) workspace; another workspace's id answers not-found; the PNG lands only in the ticket's own `sessions/<ticket>/` folder with a bare file name.
   - A cross-workspace read or write, or a path outside the ticket's folder = CRITICAL.
3. **The tools (3-file pattern, the hierarchy gate).**
   - Every new action is `registry.register(ActionDefinition(...))` in an `actions_*.py`, routed in `platform_executor.py`, its session name in `SESSION_TOOL_GROUPS`; every `write` action is gated or on `ALLOW_LIST` with a true reason; `python3 orchestrator/scripts/check_hierarchy_gate.py` exits 0.
   - `render_preview` is `read`; outside a session's ticket it refuses and writes nothing.
   - An ungated write, or a gate entry whose reason is false = HIGH.
4. **Templates (US-013, Decision Q3).**
   - Both tools and the REST routes share ONE validator (blocks schema, chips); validation errors come back field by field.
   - A starter (`created_by == STARTER_CREATOR`) is refused; there is no delete tool; social formats are refused; `made-by:<agent>` comes from the server-built context, never a parameter; created templates list in the studio.
   - A second validator, or a starter edit reachable = HIGH.
5. **The board (US-009, US-010, FR-9, FR-10).**
   - Laid out from the effective kit only (no data fields); every role's swatch prints its hex; the type scale with samples; spacing and clear space; tone words with meanings; the variants on light and dark with the FR-9 fallback when unset; three miniatures from the starters; one A4 page (the test reads the PDF back).
   - The social "Brand board" (4:5, 9:16) reads only `--brand-*` tokens and passed the media-render lane's `hyperframes check` (read the job log).
   - The board route renders through `render_png_isolated` (never on the API process's event loop), is workspace-scoped, plain `def`, in the committed manifest; the page refreshes the preview after a save; downloads go through `apiClient`.
   - Anything on the board that is not from the kit = HIGH.
6. **The designer (US-011).**
   - One persona source shared with the Socials package's Brand Designer; seeded insert-if-absent per workspace (idempotent; installing the package does not create a second one); the persona lives in the database.
   - The runtime follows the US-011 note: `cli` + `claude` only where `CLI_RUNTIME_ENABLED`; never a configuration `validate_runtime_configuration` would refuse on that instance.
   - The instructions carry every rule of the AC and name real tools.
   - A duplicate agent, or a seeded runtime its edition cannot run = HIGH.
7. **Auto's routing (US-014).** The note fires on a brand ask, not on a question about the kit ("what's our accent?"), and not when the owner says "yourself"; only the owner's latest message is read; the proposal card carries the proposal and the board's path (tested). A routing claim without the routing test = HIGH.
8. **Scope and conventions.**
   - No migration (or exactly one justified, create_all-first safe revision with both head pins moved). No hosted re-seed job, no onboarding change, no type-scale presets (the decisions).
   - No `os.getenv` outside `config.py`; no hardcoded values; every commit DCO-signed; no `node_modules`; no new dependency; functions ≤ 50 code lines and nesting ≤ 4 on touched code; no file over 800 lines grown; no skill content or generated-seed edits; the deny list untouched; pushes only to this branch.
9. **Claims.** Nothing in the diff, the commit messages or the DONE marks may claim a browser check, a live session, a real designer run or a night result. A claimed check = CRITICAL.

## Verification

- Run the **code-review** skill (or the code-reviewer agent) on the diff **in the foreground**. Any CRITICAL or HIGH it reports is a finding.
- Run `bash scripts/ralph/acceptance-prd255w2.sh` yourself. A green build with a red gate is a finding.
- Check CI: `gh run list --branch feat/prd-255-w2-brand-board-designer --workflow test.yml --limit 5`, then the jobs of HEAD's run.
- Spot-check three `DONE` acceptance criteria at random against the code and the CI log. Evidence that does not exist = CRITICAL.
- **Nothing runs on the owner's machine:** no server, docker, browser, pytest or database. CI is the evidence. Never end your turn with anything running in the background.

## Verdict

- **No CRITICAL/HIGH/MEDIUM:** a 5-line summary noting:
  - (a) the approval trace: proposal → card → Approve → `platform_update_brand_kit`, and the test that proves a proposal saves nothing;
  - (b) the tenant-isolation tests of each new tool and route;
  - (c) the hierarchy-gate result and each new action's permission;
  - (d) the board's one-page proof and the media-render run of the social board;
  - (e) the owner's next step: the `ownerTest` list in `scripts/ralph/prd-255w2.json`.

  Final line: `REVIEW_PASS`
- **Findings:**
  1. Append `P255-RVW-n` stories to `scripts/ralph/prd-255w2.json` (n continues from the highest existing `P255-RVW-` number), each with a title, `file:line` evidence, mechanical ACs (each ending with "CI green on HEAD"), `priority` after the last story, `passes: false`.
  2. Commit with `git commit -s -m 'chore(prd-255): Wave 2 review findings → fix stories'`, then push to `origin feat/prd-255-w2-brand-board-designer`.

  Final line: `REVIEW_FINDINGS`
