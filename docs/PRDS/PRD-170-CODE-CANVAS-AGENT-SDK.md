# PRD-170 — Code Canvas: Claude Agent SDK Embed

**Chain:** Net-new track, branch `ralph/prd-170-code-canvas-sdk` from main after PRD-156. Size **L**.
**Source:** report §2.14; D3 BINDING. The substrate exists: `workspace_read_file/write_file/list_dir/grep/exec/git` tools (`modules/tools/discovery/workspace_actions.py:19-341`) + worker-container file/exec APIs (`api/workspace_files.py`, mounted `main.py:1023`).

## Overview

A Claude Code-quality coding surface inside Automatos: a headless Claude Agent SDK session runs per workspace in the worker container; the platform UI renders the session — file tree, editor, streaming agent turns, diff approvals, commit/push. The user chats; the agent sees and changes the workspace; nothing applies without approval; git is the audit trail.

## Binding amendments

D3, D11 (session token budgets surfaced in UI), Q38/Q41: codegraph indexes the workspace clone as part of this PRD (local-path scope), Q82: codegraph = dev-tooling first, Q85: one exec surface — `workspace_files.py` POST /exec stays; the unmounted `workspace_exec.py` router is deleted (supersedes the mount-it option; terminal uses the session), Q36: GitHub App installation tokens for push (from 165).

## User Stories

### S1: Session service
SDK session manager in the worker container: start/resume/stop a headless `claude` (Agent SDK) session per workspace; session transcript + state on the persistent volume; platform-side `POST /api/workspaces/{id}/canvas/sessions` proxy with RequestContext auth; concurrency: one active session per workspace v1.
**Acceptance:**
- [ ] Session lifecycle integration test against a seeded workspace container
- [ ] Sessions survive orchestrator restarts (state on volume; resume test)
- [ ] Tenancy: session can only touch its workspace mount (path-escape test)

### S2: Provisioning gap
Wizard-created workspaces have no worker container (`workspace_files.py:33-47`). Provision-on-demand: first canvas open provisions the container (or attaches a shared sandboxed runner — decision memo in PR; default: per-workspace container for isolation); clear UI state while provisioning.
**Acceptance:**
- [ ] Wizard workspace → open canvas → working session (e2e)
- [ ] Provisioning failure shows honest error + retry — dev-browser verify

### S3: Event stream → UI
Bridge SDK session events (assistant text, tool calls, file edits, permission requests) over the existing SSE channel shape; render streaming turns in a session panel beside the file tree; file tree (existing `useWorkspaceFiles`) gains live-refresh on agent edits.
**Acceptance:**
- [ ] Ask "create a README" → see streamed turns + tree refresh — dev-browser verify
- [ ] Event schema versioned + vitest-validated (no stale-name drift class — reachability-style test)

### S4: Diff approval loop
Map SDK permission prompts to UI approval cards: file edits render as diffs (Monaco diff or equivalent already-in-bundle viewer — no new heavy dep without S1-style memo); approve/deny per edit or per-turn auto-accept toggle (session-scoped, default off); denied edits return feedback to the session.
**Acceptance:**
- [ ] Edit proposal → diff card → approve applies / deny reverts+informs (e2e both paths)
- [ ] Auto-accept toggle honored + visibly indicated — dev-browser verify

### S5: Git integration
Branch-per-session default (`canvas/<session-id>`); commit from UI with generated message (editable); push via `workspace_git` with GitHub App installation token + platform actor identity (168 S4 pattern); PR-open link-out v1.
**Acceptance:**
- [ ] Commit+push e2e against a test repo; author/committer correctly attributed
- [ ] No token material in logs/errors (re-run the PRD-154 S12 class of test here)

### S6: Codegraph on the workspace
Index the workspace clone (local-path scope per Q38) on session start + on commit; the SDK session gets codegraph tools (165 S4 family) pointed at the workspace index so the agent navigates by call-graph, not just grep.
**Acceptance:**
- [ ] Agent answers "what calls X?" about workspace code inside a canvas session (integration test)
- [ ] Index refresh on commit (test)

### S7: Cleanup
Delete the unmounted `workspace_exec.py` router (Q85 resolution) and the read-only-era dead affordances in `CodingCanvasWidget`; terminal panel = session shell (SDK bash tool with approval), not a separate surface.
**Acceptance:**
- [ ] One exec surface; contract + reachability green; dead code gone

## Non-Goals

Multi-user concurrent sessions, non-Claude model support in canvas v1 (the orchestrator's own agents remain model-agnostic elsewhere), arbitrary-repo browsing outside the workspace, replacing the chat surface.

## Success Metrics

- The demo: open canvas on a workspace repo → "add input validation to X and push" → streamed work, diff approvals, branch pushed — no terminal needed.
- Zero unapproved writes in audit across the pilot.
- Session start (warm container) < 10s; provisioning (cold) < 2min with honest progress.

## Testing

Session lifecycle + path-escape + provisioning integration suites (DB+container-backed, gated to a CI job with docker, mirroring compose-smoke patterns from PRD-153); event-schema vitest; e2e dev-browser flows for S3/S4/S5. Full suite + contract green.
