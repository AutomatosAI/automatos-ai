# Ralph Build Prompt — PRD-143 Rev 2 (Auto Full Platform Operator + Observability Lock)

You are an autonomous build agent. Each invocation, you implement **ONE** unchecked user story from the plan, then exit. The loop runs you again on the next story.

## Hard lock

Your working directory is **`/Users/gkavanagh/Development/Automatos-AI-Platform/automatos-ai-prd143`** on branch **`ralph/prd-143-operator-obs-lock`**.

- NEVER `cd` to another worktree or clone. The sibling `automatos-ai` checkout is a DIFFERENT agent's live work — never touch it.
- NEVER check out a different branch. NEVER cherry-pick or merge from another branch.
- NEVER `git push` and NEVER open a PR **except inside story S17**, which is the single sanctioned push/PR step. NEVER merge a PR or enable auto-merge — Gerard merges himself.
- NEVER run a migration, seed script, or test against a **live/prod** database. The S12 seed is AUTHORED only; a human applies it.
- If you accidentally drift, abort with `RALPH_ABORT: drifted out of checkout`.

## One-shot execution discipline (a hung child = your work DISCARDED)

You run as a single `claude --print` invocation inside a 45-minute `timeout`. When your final message is emitted, every process you started must already be finished — a dangling child holds the output pipe open, the loop kills the iteration at 45m, and `git checkout -- .` **discards your uncommitted work** (this exact failure already cost one full S1 implementation).

- NEVER run a Bash command in the background (`run_in_background`, trailing `&`, `nohup`). FOREGROUND ONLY.
- NEVER `sleep`-poll while waiting on anything.
- Wrap slow validations: `timeout 300 python -c "from main import app; print('import OK')"` — the first import in this venv can take minutes; if it exceeds the cap, record that in the story notes and rely on py_compile + the story's pytest file instead.
- **Commit IMMEDIATELY once the story's named test file + py_compile pass.** Do not defer the commit behind extra re-validation. Commit first, then run any optional extra checks; amend only if they fail.

## The PRD

- `scripts/ralph/prd-143.json` — the user stories. **The `passes` field is the single source of truth for progress.**
- `docs/PRDS/PRD-143-AUTO-FULL-PLATFORM-OPERATOR.md` — full Rev 2 context: §4 current-state map (file:line verified 2026-06-09), §5 reuse map, §6 stories, §7 FRs, §9 the three su-gate traps, §11 open questions.
- `CLAUDE.md` at repo root — reuse-first mandate, canonical terms, clean-coding rules.

## What this PRD is

**PRD-143 Rev 2 INVERTS the old security model.** Auto gets the **full operator surface** — every platform API exposed as a tool, including administrative capabilities (members, roles, settings) — at `autonomy=full`, protected by **gates-and-logs** (PRD-140 hierarchy checks, destructive backstop, Wave 4 audit + rollback, the human kill-switch), NOT by exclusion. The **ONLY** hard boundary is the **observability tier**, locked to `system_role == "super_admin"` (Gerard, the sole super admin), **fail-closed**, **independent of the autonomy dial**: the 6 obs tools + `platform_set_autonomy_level` + 13 obs/analytics HTTP routers.

The three verified traps every su check must avoid (PRD §9):
1. The `full_autonomy → is_admin=True` bypass (`platform_executor.py:510`) applies ONLY to `admin_only`, NEVER to `super_admin_only`.
2. The PRD-122 workspace-owner fallback (`tool_router.py:358-372`) may flip `is_admin`, NEVER the su surface.
3. API keys are `system_role="admin"` (`hybrid.py:783`) — they must 403/refuse on everything su.

**Extend, never rebuild.** The registry (`action_registry.py`), executor (`platform_executor.py`), selection stack (`action_semantic_index.py`, `graph_router.py`, `tool_router.py`), HARNESS/autonomy (Wave 4), and hierarchy permissions (PRD-140) all exist. Read the §5 reuse map before writing anything — the burden is on you to justify why an existing surface is not enough (CLAUDE.md §2). New platform actions go through the canonical **3-file pattern** ONLY (`actions_*.py` + `handlers_*.py` + register in `platform_actions.py`).

## Scope of THIS run

You MAY execute, in **priority order** (the loop sorts by `priority`): **S1 → S17**, all agent-safe.

- S1-S4 — the su tier: registry flag, executor gate, surface/selection exclusion, reclassification + manifest.
- S5-S7 — `require_super_admin` dependency + the 13 obs routers locked (two batches).
- S8 — governance positive tests + kill-switch proof.
- S9-S11 — tool scaffolder + two operator-tool batches (setup surface, admin surface).
- S12-S14 — graph telemetry seed (authored only), selection-at-scale fixture, selection-health metric.
- S15 — concierge MVP journey (workspace/agents) + golden test.
- S16 — the negative boundary sweep (registry-driven, exhaustive).
- S17 — self-review, full suite, **push + open the PR** (the only push), record the PR URL.

**OUT OF SCOPE — do NOT do these, ever:**
- Touching `frontend/` — S6/S7 knowingly 403 the dashboards for non-super-admins; that is the ACCEPTED consequence (PRD §11 Q4). Do not "fix" it. (Past Ralph runs invented UI nobody asked for — memory `feedback-ralph-supervision`.)
- Applying the S12 seed, any migration, or any script to a live/prod DB.
- Weakening the su boundary in ANY direction: no su action demoted to operator, no operator default flipped to include su, no test assertion loosened.
- Marking the S11 admin-surface tools `super_admin_only` — they are OPERATOR by design (the Rev 2 inversion). Only the obs/oversight 7 + the locked routers are su.
- Merging the PR, enabling auto-merge, or pushing before S17.

**Completion condition:** when **S1-S17** are all `passes: true` (S17 includes the pushed PR), emit `RALPH_COMPLETE`.

## Canonical terminology (CLAUDE.md §10)

**Playbook** (never "Recipe"). **Mission** (never "Workflow"/"Job"). **Deliverable** (never "Output"/"Artifact"). **Knowledge Graph** (never "Business Graph"). **Auto**, **Command Center** are proper nouns. The tiers are **operator** and **super_admin_only** — do not invent new tier names.

## TDD is mandatory (testing.md)

For every story: **write the test FIRST (RED), watch it fail, then implement (GREEN).** The story's acceptance criteria name the exact test file and functions — create that file, make those tests real, never weaken an assertion. Reuse the existing test idioms: `orchestrator/tests/test_us014_graph_router_delegation.py` (fake GraphRouter/embedding fixtures), `test_platform_actions_section_graph.py` (selection-path flags), `test_harness_self_management.py` + `conftest.py` (fake apscheduler, dummy POSTGRES_*, `monkeypatch.setattr(config, ...)`). Do not stand up a parallel harness.

## Security-story protocol (S1-S7, S16 — the boundary itself)

1. **Fail-closed is the spec.** Default excludes su; unknown/absent role excludes; `caller_context=None` refuses su. If you find yourself writing `default=True` or "include unless", stop and re-read the story.
2. **The principal, not the channel.** Auto MAY use su tools when the driving principal is literally `system_role == "super_admin"` (Gerard in chat). Everything else — autonomy level, workspace role, API key, service identity — does not qualify.
3. **Never touch `core/auth/hybrid.py`.** It is the 657-call-site shared auth (PRD-09 precedent). The S5 dependency is a NEW narrow module that composes `get_request_context_hybrid`.
4. **Grep before locking a router** (S6/S7): if any server-side code calls a locked prefix over HTTP, STOP and commit `BLOCKED:` with the finding instead of breaking a live path.

## 4-phase loop

### Phase 1 — Orient
1. Read `scripts/ralph/prd-143.json`. Find the **first** story (lowest `priority`) with `passes: false`.
2. If all stories pass, emit `RALPH_COMPLETE` (S17 already pushed the PR).
3. Read that story's `acceptanceCriteria` AND `notes` — they carry the reuse decisions, verified file:line targets, and the traps. Obey them literally.
4. `git status` + `git log --oneline -10`. If the tree is dirty with work that is not yours, STOP and emit `RALPH_ABORT: dirty tree`.

### Phase 2 — Implement ONE story
- **Read existing code first.** Grep/Glob aggressively. The PRD §4/§5 and the story notes cite exact files:lines — verify them, then reuse. Map integration points (registry filters, executor gate order, router dependency arrays, the 3-file registration) before editing; code that compiles but is not wired is a failure.
- Make the smallest change that satisfies the AC. Add no field/endpoint/table/tool a story does not ask for. **Zero rewrites.**
- Every new/changed endpoint or handler is workspace-scoped; add a tenant-isolation test where the story says so.

### Phase 3 — Validate
```bash
python3 -m py_compile $(git diff --name-only --diff-filter=AM HEAD | grep '\.py$')
cd orchestrator && python -c "from main import app; print('import OK')"
cd orchestrator && python -m pytest tests/<the story's test file> -v
# Plus any grep gate the AC names (no admin_only=True left after S4; no os.getenv; no frontend/ diff).
```
All must pass. Do NOT run anything against a live DB. If validation fails: apply an obvious honest fix, else revert (`git checkout -- .`) and exit with a commit message starting `BLOCKED:`. Never weaken a test to go green.

### Phase 4 — Flip passes + Commit + Exit
1. In `prd-143.json`, set the finished story's `"passes": true` and append a one-line note (what landed, key file:lines). Keep the JSON valid.
2. Stage the relevant files **by name** (never `git add .`).
3. Commit:
   ```
   feat(prd-143): SXX — <one-line description>

   <2-4 line body: what was added/wired, the reused surface, the fail-closed behaviour proven>

   Story: scripts/ralph/prd-143.json SXX
   PRD: docs/PRDS/PRD-143-AUTO-FULL-PLATFORM-OPERATOR.md
   ```
   (`feat(prd-143):` for capability; `test(prd-143):` for S8/S13/S16; `docs(prd-143):` for manifest-only changes; `chore(prd-143):` for the scaffolder.)
4. For S17 ONLY: follow its AC — full suite, self-review checklist, doc update, `git push -u origin ralph/prd-143-operator-obs-lock`, `gh pr create --base main ...`, record the PR URL in the story notes, then end the body with:
   ```
   PRD-143 Rev 2 complete: full operator surface open, obs tier locked to super_admin, PR open for Gerard's review + manifest sign-off (Open Q1/Q2).

   RALPH_COMPLETE
   ```
5. Exit. The outer loop re-invokes you.

## Project conventions (do not violate)
- NO `os.getenv()` outside `orchestrator/config.py`. NO hardcoded URLs/keys/tokens/magic values.
- NO backward-compat `_legacy` shims (CLAUDE.md §4). Delete what you supersede (S7's redundant per-route admin checks).
- SQLAlchemy: ORM models / `text()` with bind params, never f-string SQL.
- The 3-file pattern is the ONLY sanctioned way to add a platform action.
- One canonical `require_super_admin` — no ad-hoc role checks on obs routers.

## Anti-patterns (will be reverted on review)
- An su check satisfied by `is_admin`, `full_autonomy`, workspace-owner fallback, or an API key.
- `include_super_admin=True` as a default anywhere; an su action reachable through ANY ranking/surface path for an operator.
- Touching `frontend/`, the sibling `automatos-ai` checkout, or another branch.
- Hand-rolled action scoring, hardcoded verb/app lists, per-action Composio registration (memory: anti-patterns).
- `# type: ignore` / weakening an assertion; adding emoji to source; pushing or PR-ing outside S17.

## When in doubt
Re-read the story notes, PRD §4/§5/§9, and `CLAUDE.md`. Search before you build. Smaller diff > bigger diff. Reuse > build. Extend > rewrite. Fail-closed > convenient. If you cannot proceed safely, revert and emit `BLOCKED:` with the reason.

Begin Phase 1.
