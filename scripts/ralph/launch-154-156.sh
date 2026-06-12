#!/bin/bash
# Night-1 launcher — Remediation Chain PRD-154 → 155 → 156 (stacked).
#
# Run THIS from YOUR terminal (the AI harness blocks launching a
# --dangerously-skip-permissions loop; you own that authority):
#
#   ./scripts/ralph/launch-154-156.sh            # all three stacked (default)
#   ./scripts/ralph/launch-154-156.sh 154        # just 154 (validate the kit first)
#   ./scripts/ralph/launch-154-156.sh 155 156    # subset, in order
#
# Billing: SUBSCRIPTION (Max 20). ANTHROPIC_API_KEY is intentionally NOT
#   exported, so child `claude` runs inherit your subscription auth. To bill
#   the API instead: `export ANTHROPIC_API_KEY=...` before running this.
# Model: Opus 4.8 (override with RALPH_MODEL=… ; e.g. claude-sonnet-4-6 to cut burn).
# Detached + logged; caffeinate keeps the Mac awake — safe to close the terminal.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
mkdir -p scripts/ralph/logs
LOG="scripts/ralph/logs/chain-154-156-$(date +%F-%H%M).log"

echo "[launch] PRD-154→156 chain (subscription, Opus 4.8, no docker gate)"
echo "[launch] live log : tail -f $LOG"
echo "[launch] per-PRD  : scripts/ralph/logs/prdNNN-{build,review,acceptance}-$(date +%F).log"
echo "[launch] report   : scripts/ralph/night-report-$(date +%F).md"
echo "[launch] state    : scripts/ralph/state/prdNNN.status  (delete to force a re-run)"

RALPH_SKIP_DOCKER_CHECK=1 nohup ./scripts/ralph/overnight-prd154-156.sh "$@" > "$LOG" 2>&1 &
echo "[launch] detached PID $! — running. Close this terminal anytime."
