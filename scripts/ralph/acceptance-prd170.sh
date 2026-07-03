#!/bin/bash
# Acceptance gate — PRD-170 Code Canvas: Claude Agent SDK Embed.
# Run from the worktree repo root. Exit 0 = local DB-free/container-free gates pass.
#
# THIS MACHINE HAS NO DOCKER / NO WORKER CONTAINERS / NO RUNNING APP / NO BROWSER.
# PRD-170 is container + SDK + frontend heavy, so MOST acceptance is CI-with-Docker
# (session lifecycle, resume, path-escape, provisioning e2e, push e2e, index-on-commit)
# or a morning-human live demo. This gate is the DB-free / container-free subset only:
# py_compile on touched backend + frontend tsc/vitest/lint + the S7 deletion gate +
# the S5 token-leak proxy (security — this one is NOT deferrable).
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
FAIL=0
check() {
  local name="$1"; shift
  echo ""
  echo "── $name"
  if bash -c "$1"; then echo "   ✅ PASS: $name"; else echo "   ❌ FAIL: $name"; FAIL=1; fi
}

# --- Backend: compile-only (no DB/container). Container suites run on CI-with-Docker. ---
check "backend compiles (py_compile — known reuse surfaces + any new canvas/session module)" \
  'cd orchestrator && python3 -m compileall -q api/workspace_files.py modules/tools/discovery/workspace_actions.py && { files=$(git ls-files "*canvas*.py" "*session*service*.py" 2>/dev/null); [ -z "$files" ] || python3 -m compileall -q $files; }'

# --- S7: the never-mounted duplicate exec router is DELETED (Q85, delete-what-you-replace) ---
check "S7 workspace_exec.py deleted (Q85 — one exec surface)" \
  '[ ! -f orchestrator/api/workspace_exec.py ]'

# --- S5: token-leak proxy (security — NOT deferrable). No GitHub token material logged. ---
check "S5 no token material printed/logged in the git/push path" \
  '! grep -rnE "(print|logger\.[a-z]+)\(.*\b(installation_token|access_token|gh[ps]_[A-Za-z0-9])" orchestrator/modules/tools/discovery/workspace_actions.py'

# NOTE: session confinement (path-escape), resume-after-restart, provisioning e2e, the
# "what calls X?" codegraph query and index-on-commit are CI-with-Docker / morning-human —
# their contract tests must exist and be DEFERRED in prd-170.json, not faked green.

check "frontend-typecheck (no DB)" \
  'cd frontend && npx tsc --noEmit'

check "frontend-units (vitest — event-schema validation + diff-card render proxies for the deferred browser ACs)" \
  'cd frontend && npm run test'

check "frontend-lint (no DB)" \
  'cd frontend && npm run lint'

echo ""
if [ $FAIL -eq 0 ]; then
  echo "ACCEPTANCE: PRD-170 local gates PASS"
  echo "NOTE: container ACs (session lifecycle/resume/path-escape, provisioning e2e, push e2e, codegraph index-on-commit) are CI-with-Docker — confirm the Docker-gated job is green before merge."
  echo "NOTE: MORNING LIVE DEMO before merge — open canvas on a workspace repo → 'add input validation to X and push' → streamed turns, diff approvals, branch pushed, zero unapproved writes in audit."
  echo "NOTE: provisioning a per-workspace container is the default isolation choice; if a shared runner was chosen instead, a decision memo must be in the PR body."
else
  echo "ACCEPTANCE: PRD-170 local gates FAIL"
fi
exit $FAIL
