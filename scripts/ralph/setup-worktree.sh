#!/bin/bash
# setup-worktree.sh — per-worktree provisioning hook, invoked by the overnight
# runners' ensure_worktree() right after a worktree is created.
# Recreated 2026-08-27 (the wave's first acceptance gate died on
# "vitest: command not found" — fresh worktrees have no node_modules because
# node_modules is untracked; this hook was the missing piece).
# APFS clone (cp -c) makes the copy near-instant and space-cheap.
set -uo pipefail
WT="${1:?usage: setup-worktree.sh <worktree-path>}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# Disable the graphify per-commit hook INSIDE ralph worktrees only (each story
# commit was paying a minutes-long 100%-CPU graph rebuild — the "hanging").
# Main checkout keeps its hooks; the worktree keeps the graph built at cut time.
git -C "$ROOT" config extensions.worktreeConfig true 2>/dev/null || true
git -C "$WT" config --worktree core.hooksPath /var/empty 2>/dev/null \
  && echo "  per-commit hooks disabled in worktree (graphify tax removed)"

if [ -d "$ROOT/frontend/node_modules" ] && [ ! -d "$WT/frontend/node_modules" ]; then
  if cp -Rc "$ROOT/frontend/node_modules" "$WT/frontend/node_modules" 2>/dev/null; then
    echo "  cloned frontend/node_modules (APFS COW)"
  else
    cp -R "$ROOT/frontend/node_modules" "$WT/frontend/node_modules"
    echo "  copied frontend/node_modules (plain copy)"
  fi
fi
exit 0
