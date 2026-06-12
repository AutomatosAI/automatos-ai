#!/bin/bash
# Per-worktree provisioning hook — invoked by overnight-*.sh `ensure_worktree`.
#   $1 = worktree path. Fast, idempotent, best-effort.
#
# Why this exists: git worktrees do NOT carry node_modules (gitignored, ~947MB).
# Without it every frontend gate (npx tsc / vitest / next lint / next build)
# dies on missing deps — and PRD-154 is heavily frontend, so the whole night
# would fail acceptance. Fix: APFS clonefile (cp -c) from the MAIN checkout —
# near-instant, copy-on-write, and INDEPENDENT + mutable, so a story's
# `npm install <pkg>` (e.g. P154-S7 adds react-force-graph-3d) only diverges
# this worktree's copy and never touches main or sibling worktrees.
#
# Python needs no venv here: the orchestrator runs on system python3 and the
# acceptance scripts `pip install -e` whatever they require.
set -uo pipefail
WT="${1:?usage: setup-worktree.sh <worktree-path>}"
MAIN="$(cd "$(dirname "$0")/../.." && pwd)"   # main checkout root (.../automatos-ai)

if [[ -d "$MAIN/frontend/node_modules" && ! -d "$WT/frontend/node_modules" ]]; then
  echo "  [setup-worktree] cloning frontend/node_modules (APFS COW)…"
  # clonefile is instant on APFS; fall back to a plain recursive copy elsewhere.
  if cp -Rc "$MAIN/frontend/node_modules" "$WT/frontend/node_modules" 2>/dev/null; then
    echo "  [setup-worktree] frontend deps ready (cloned)"
  elif cp -R "$MAIN/frontend/node_modules" "$WT/frontend/node_modules"; then
    echo "  [setup-worktree] frontend deps ready (copied)"
  else
    echo "  [setup-worktree] WARNING: could not provision frontend/node_modules — frontend gates will fail"
  fi
fi
exit 0
