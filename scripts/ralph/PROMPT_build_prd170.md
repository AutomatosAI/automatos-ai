# Ralph Build Prompt — PRD-170 Code Canvas: Claude Agent SDK Embed

You are executing **PRD-170**, one story per iteration, unattended overnight. This branch is **`ralph/prd-170-code-canvas-sdk`, cut from `main`** (independent net-new track — NOT stacked). The tip must be green after every commit.

What you are building: a Claude-Code-quality coding surface inside Automatos. A **headless Claude Agent SDK session runs per workspace in the worker container**; the platform UI renders it — file tree, streaming agent turns, diff approvals, commit/push. The user chats; the agent sees and changes the workspace; **nothing applies without approval; git is the audit trail.** The substrate already exists — reuse it, do not rebuild it.

## ⚠️ Read this about verification before anything else

This loop is **headless: no Docker, no worker containers, no running app, no browser** on this machine. PRD-170 is container + SDK + frontend heavy, so **most of its acceptance is CI-with-Docker or a morning human** — that is expected and fine. Per story you will: write the code + a **DB-free / container-free deterministic proxy** (a unit test of pure logic, a schema/contract vitest, a grep/deletion gate, a typecheck), then mark the container/browser ACs `DEFERRED — morning check: <what to verify>` in `prd-170.json`. Implement the real thing fully; only the *live verification* is deferred. If a story's core decision genuinely cannot be made without a running container (e.g. proving session-state survives a restart), build the code to the contract and reply `RALPH_BLOCKED` rather than fake a green — **the tip stays green by construction and the chain continues.** Never start a dev server, never call dev-browser, never `docker run`.

## Read first, every iteration

1. `scripts/ralph/prd-170.json` — the story list (`description` = BINDING contract + amendments). Pick the **first story whose ACs are not all marked DONE**.
2. `docs/PRDS/PRD-170-CODE-CANVAS-AGENT-SDK.md` — full spec + binding amendments **D3, D11, Q36, Q38, Q41, Q82, Q85**.
3. `CLAUDE.md` — reuse over build; **delete what you replace**; no shims; no `os.getenv` outside `config.py`; no new heavy dependency without an explicit memo in the commit body.

## Ground truth (verified 2026-06-12 — re-grep before every edit; lines drift)

- **Workspace tool substrate (reuse)**: `orchestrator/modules/tools/discovery/workspace_actions.py` — `workspace_read_file` (:19), `workspace_write_file` (:54), `workspace_exec` (:164), plus list_dir/grep/git in the same file. The SDK session drives the workspace through these semantics — do not invent a parallel file API.
- **Worker-container file/exec API (reuse)**: `orchestrator/api/workspace_files.py` — `GET /files`, `POST /exec`; mounted at `orchestrator/main.py` (~:1023). This is the **one exec surface** (Q85).
- **DELETE in S7 (Q85)**: `orchestrator/api/workspace_exec.py` — the never-mounted duplicate exec router. Gone this PRD; the session shell + `workspace_files` POST /exec cover it.
- **Canvas frontend shell (EXTEND, do not rebuild)**: `frontend/components/widgets/CodingCanvasWidget/` — `index.tsx`, `CodeEditor.tsx`, `EditorTabs.tsx`, `FileExplorer.tsx`, `RepoSelector.tsx`, `useWorkspaceFiles.ts`. S3/S4/S7 extend these; S3 live-refreshes the existing `useWorkspaceFiles` tree.
- **Git push path (reuse, S5)**: the existing `workspace_git` tool in `workspace_actions.py` + GitHub App installation tokens (PRD-165 base) + platform actor identity (PRD-168 pattern).
- **Codegraph family (reuse, S6)**: `orchestrator/modules/codegraph/` — point its tools at the workspace clone index (local-path scope, Q38).
- **SDK session manager (S1) is genuinely net-new** — there is no existing canvas/SDK session backend. Build it in the worker-container service layer; the substrate above is what it calls.

## The execution contract

- **TDD where a DB-free/container-free test is possible**: failing test first, then implement, then green. For container-only behaviour, write the contract test that CI/Docker will run and mark it DEFERRED locally.
- **Story scope**: the story's `files` list is your scope. A structural surprise → `RALPH_BLOCKED`, do not improvise.
- **Testing model — CI/Docker validates the heavy stuff, NOT this machine.** No local DB/containers. Do **NOT** run `cd orchestrator && python3 -m pytest -q` (wedges on `test_82c_wiring`). Locally you may run ONLY:
  - `python3 -m py_compile` on changed backend files, and a **pure-logic** isolated unit test (`python3 -m pytest tests/<pure_test>.py -q`) for things with no session/container — e.g. the SSE event-schema serializer (S3), the diff/patch model (S4), the commit-message generator (S5).
  - **Frontend** (no DB/container) → `cd frontend && npx tsc --noEmit` AND `npm run test` (vitest) green; `npm run lint` when touched. These gate the commit locally. The S3 event-schema vitest and S4 diff-card render test live here.
  - Everything container-backed (S1 lifecycle/resume/path-escape, S2 provisioning e2e, S5 push e2e, S6 index-on-commit) is **CI-with-Docker** (mirrors PRD-153 compose-smoke) or morning-human — DEFER locally.
- **New backend test files importing `modules.*`/`consumers.*` at module level MUST start with the collection-order guard** (copy `_sys_guard` from `orchestrator/tests/test_prd143_selection_at_scale.py`).
- **Never weaken a test to pass.** **Clean tree after every commit** (`git status --porcelain` empty).
- **Tenancy is the highest-stakes property here**: a session must never touch anything outside its workspace mount, and **no token material may ever reach logs/errors** (re-apply the PRD-154 S12 token-leak test class in S5). Treat any path-escape or token-leak gap as a build bug, not a deferral.

## Story-specific guardrails (full ACs live in `prd-170.json`)

- **S1 — Session service**: start/resume/stop a headless `claude` Agent SDK session per workspace in the worker container; transcript + state on the persistent volume (survives orchestrator restarts); platform proxy `POST /api/workspaces/{id}/canvas/sessions` with RequestContext auth; **one active session per workspace v1**; the session cannot escape its workspace mount. Lifecycle/resume/path-escape ACs are CI/Docker — DEFER with the contract test written.
- **S2 — Provisioning gap**: wizard-created workspaces have no worker container. Provision-on-demand at first canvas open; **default = per-workspace container for isolation** (a shared sandboxed runner instead requires a decision memo in the PR body); honest provisioning progress + failure/retry states in the UI (vitest the states; e2e is morning-human).
- **S3 — Event stream → UI**: bridge SDK session events (assistant text, tool calls, file edits, permission requests) over the **existing SSE channel shape**; render streaming turns in a session panel beside the file tree; `useWorkspaceFiles` live-refreshes on agent edits; **event schema versioned + vitest-validated** (a drifted name fails — that's the local gate).
- **S4 — Diff approval loop**: map SDK permission prompts to UI approval cards; file edits render as diffs using an **in-bundle viewer** (Monaco diff if already bundled — any new heavy dep needs a memo in the commit); approve applies / deny reverts + informs the session; per-turn auto-accept toggle, **session-scoped, default OFF**, visibly indicated. Render/behaviour is vitest; the apply/deny e2e is morning-human.
- **S5 — Git integration**: branch-per-session `canvas/<session-id>`; commit from UI with a generated **editable** message; push via the existing `workspace_git` with GitHub App installation token + platform actor identity (168 pattern); PR-open link-out v1. **Re-apply the PRD-154 S12 token-leak test class** — no token material in logs/errors (this gate runs locally and DOES block).
- **S6 — Codegraph on the workspace**: index the workspace clone (local-path scope, Q38) on session start + on commit; give the session the 165 codegraph tools pointed at that index so the agent navigates by call graph. "What calls X?" and index-refresh-on-commit are CI integration — DEFER with contract tests.
- **S7 — Cleanup (Q85)**: **DELETE** `orchestrator/api/workspace_exec.py`; remove read-only-era dead affordances in `CodingCanvasWidget`; the terminal panel becomes the SDK bash tool with approval, not a separate surface. One exec surface; contract + reachability green (local gate). Delete-what-you-replace.

## Hard NOs (human-gated — violating any is RALPH_ABORT territory)

- **NO second exec surface** — `workspace_exec.py` is DELETED, not mounted (Q85). **NO parallel file API** — reuse `workspace_files.py` / `workspace_actions.py`.
- **NO new heavy frontend dependency without a memo** in the commit body (prefer the in-bundle Monaco diff viewer).
- **NO multi-user concurrent sessions, NO non-Claude canvas model, NO arbitrary-repo browsing outside the workspace** (explicit non-goals — `RALPH_BLOCKED` if a story drifts toward them).
- NO `os.getenv` outside `config.py`. NO secrets/tokens in code, fixtures, logs, or errors.
- **PUSH after each story commit to `origin ralph/prd-170-code-canvas-sdk` ONLY** — never force-push, never another ref, never `main`. **NO PRs mid-run** (the runner opens a draft PR at the end). **NO merges.**
- Do NOT fake a container/browser AC green — DEFER it honestly or `RALPH_BLOCKED`.

## Per-iteration protocol

1. Pick the first story with un-DONE ACs; re-verify its ground truth fresh (grep — don't trust line numbers).
2. Write the failing DB-free/container-free proxy test. Implement minimally. Run the story's local gates.
3. Commit `feat(prd-170): <story-id> — <title>`, AC evidence in the body. Mark that story's AC lines `DONE — <evidence>` or `DEFERRED — morning/CI check: <what>` in `scripts/ralph/prd-170.json` **in the same commit**. Then push.

## Completion

- All ACs DONE/DEFERRED → run `bash scripts/ralph/acceptance-prd170.sh` (DB-free/container-free local gates only). Exit 0 → reply `RALPH_COMPLETE`.
- **The container + e2e suites are NOT in the local gate — they run on CI-with-Docker or are morning-human.** Make sure your final commit is **pushed**; the runner records the CI result and the morning human runs the live demo (open canvas → "add validation and push") before merge.
- Local gate red, or a token-leak/path-escape you can prove → fix it in the owning story (these block — they are security). Out-of-scope cause, or something only a container can confirm → reply `RALPH_BLOCKED` with one line of why.
