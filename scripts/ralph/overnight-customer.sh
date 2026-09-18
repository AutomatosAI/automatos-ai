#!/bin/bash

# Customer night — Claude plays the operator overnight (PRD-247, the night Gerard asked for).
# A headless Claude Code session uses the LOCAL product like a customer, in the
# operator's own workspace with the real tools: talks to Auto, builds and edits
# agents, hands out work that fires Claude sessions, answers questions, reviews
# what comes back, and writes a diary and a morning report. It never touches the
# code. Iterations continue from the diary until the stop time.
#
# Usage: ./scripts/ralph/overnight-customer.sh            (launch is human-only)
# Env:   RALPH_MODEL=claude-opus-5 · RALPH_MAX_ITERS=10 · RALPH_STOP_AT=06:30
#        CUSTOMER_PERSONA=harbourline · SIM_WORKSPACE_ID=<local workspace>
# Night: ~/.automatos-sim/customer-night/<date>/{DIARY.md,MORNING-REPORT.md,logs/}

set -uo pipefail

if [[ "$(uname)" == "Darwin" ]] && command -v caffeinate >/dev/null 2>&1 && [[ -z "${RALPH_CAFFEINATED:-}" ]]; then
  export RALPH_CAFFEINATED=1
  exec caffeinate -dimsu "$0" "$@"
fi
unset CLAUDECODE   # nested Claude Code sessions are the point

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
KIT="$REPO_ROOT/scripts/ralph/customer-night"
DATE="$(date +%F)"
NIGHT_DIR="${CUSTOMER_NIGHT_DIR:-$HOME/.automatos-sim/customer-night/$DATE}"
LOG_DIR="$NIGHT_DIR/logs"
STATUS="$NIGHT_DIR/night.status"
mkdir -p "$LOG_DIR"

RALPH_MODEL="${RALPH_MODEL:-claude-opus-5}"
MAX_ITERS="${RALPH_MAX_ITERS:-10}"
STOP_AT="${RALPH_STOP_AT:-06:30}"
ITER_TIMEOUT="${RALPH_ITER_TIMEOUT:-50m}"
PERSONA="$KIT/personas/${CUSTOMER_PERSONA:-harbourline}.md"
export SIM_WORKSPACE_ID="${SIM_WORKSPACE_ID:-00000000-0000-0000-0000-0000000000c1}"
export CUSTOMER_NIGHT_DIR="$NIGHT_DIR"
TIMEOUT_BIN=$(command -v timeout || command -v gtimeout)

RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
say() { echo -e "$(date '+%H:%M:%S') $*"; echo "$(date '+%F %T') $*" | sed 's/\x1b\[[0-9;]*m//g' >> "$LOG_DIR/runner.log"; }
set_status() { echo "$1=$2" >> "$STATUS"; }

[[ -f "$PERSONA" ]] || { echo -e "${RED}no persona at $PERSONA${NC}"; exit 2; }
[[ -n "$TIMEOUT_BIN" ]] || { echo -e "${RED}need timeout/gtimeout (brew install coreutils)${NC}"; exit 2; }

# --- stop time ----------------------------------------------------------------
stop_epoch() {
  local h=${STOP_AT%%:*} m=${STOP_AT##*:} today
  today=$(date -j -f '%H:%M' "$STOP_AT" +%s 2>/dev/null || date -d "today $STOP_AT" +%s)
  [[ $today -le $(date +%s) ]] && today=$((today + 86400))   # a time already past today means tomorrow
  echo "$today"
}
STOP_EPOCH=$(stop_epoch)
past_stop() { [[ $(date +%s) -ge $STOP_EPOCH ]]; }

# --- usage-limit handling (the Ralph kit's) -------------------------------------
seconds_until_next_hour() { local s=$((10#$(date +%M) * 60 + 10#$(date +%S))); echo $((3600 - s)); }
countdown() {
  local seconds=$1 message=$2
  while [[ $seconds -gt 0 ]]; do
    printf "\r${CYAN}%s${NC} %02d:%02d:%02d " "$message" $((seconds / 3600)) $(((seconds % 3600) / 60)) $((seconds % 60))
    sleep 1; seconds=$((seconds - 1))
  done
  printf "\r%-80s\r" " "
}
is_usage_limit_error() {
  local output="$1" exit_code="$2"
  [[ "$exit_code" -eq 0 ]] && return 1
  echo "$output" | grep '^{' | jq -e 'select(.type == "result") | select(.subtype | test("error.*limit|rate_limit"))' &>/dev/null && return 0
  local text; text=$(echo "$output" | grep -v '^{' || true)
  text+=$(echo "$output" | grep '^{' | jq -r 'select(.type == "result" and .is_error == true) | .result // empty' 2>/dev/null || true)
  [[ "$text" =~ "hit your limit" || "$text" =~ Error:\ 429 || "$text" =~ Error:\ 529 || "$text" =~ rate.?limit || "$text" =~ usage.?limit ]]
}
get_sleep_duration() {
  local output="$1"
  [[ "$output" =~ "try again in "([0-9]+)" minute" ]] && { echo $(( ${BASH_REMATCH[1]} * 60 + 60 )); return; }
  [[ "$output" =~ "try again in "([0-9]+)" hour" ]] && { echo $(( ${BASH_REMATCH[1]} * 3600 + 60 )); return; }
  local wait_time; wait_time=$(seconds_until_next_hour); [[ $wait_time -lt 300 ]] && wait_time=300; echo $wait_time
}

# --- one iteration of the persona --------------------------------------------------
run_claude() {
  local prompt_file="$1" logfile="$2" tmp; tmp=$(mktemp)
  cd "$REPO_ROOT"
  "$TIMEOUT_BIN" --kill-after=30s "$ITER_TIMEOUT" claude --print \
    --model "$RALPH_MODEL" --verbose --output-format stream-json --dangerously-skip-permissions \
    < "$prompt_file" 2>&1 | tee "$tmp" | tee -a "$logfile" | sed 's/\x1b\[[0-9;]*m//g' | grep --line-buffered '^{' | jq --unbuffered -r '
      if .type == "assistant" then
        .message.content[] |
        if .type == "text" then (if (.text | split("\n") | length) <= 3 then .text else (.text | split("\n") | .[0:2] | join(" ")) end)
        elif .type == "tool_use" then
          "    [" + .name + "] " + ((.input.command // .input.file_path // .input.path // "") | tostring | split("\n") | first | .[0:100])
        else empty end
      elif .type == "result" then "--- " + ((.duration_ms / 1000 | floor) | tostring) + "s, " + (.num_turns | tostring) + " turns ---"
      else empty end' 2>/dev/null
  CLAUDE_EXIT=${PIPESTATUS[0]}
  CLAUDE_OUTPUT=$(cat "$tmp"); rm -f "$tmp"
}

# --- preflight --------------------------------------------------------------------
say "${CYAN}customer night $DATE · model $RALPH_MODEL · persona $(basename "$PERSONA" .md) · stop $STOP_AT · $NIGHT_DIR${NC}"
cd "$REPO_ROOT"
if ! python3 -m tests.sim.customer preflight | tee -a "$LOG_DIR/runner.log"; then
  say "${RED}preflight failed — fix the stack (backend, PRD-245 bridge, host) and relaunch${NC}"; exit 2
fi
# A relaunch on the same night keeps the original start: the cost window and the
# persona's NIGHT_START must cover the whole night, not the last runner process.
NIGHT_START="$(grep '^STARTED=' "$STATUS" 2>/dev/null | head -1 | cut -d= -f2)"
[[ -n "$NIGHT_START" ]] || { NIGHT_START="$(date -u +%Y-%m-%dT%H:%M:%S+00:00)"; set_status STARTED "$NIGHT_START"; }
[[ -f "$NIGHT_DIR/DIARY.md" ]] || printf '# Customer night %s — diary\n\nNight started %s (UTC). Workspace %s.\n\n' "$DATE" "$NIGHT_START" "$SIM_WORKSPACE_ID" > "$NIGHT_DIR/DIARY.md"

# --- the night ----------------------------------------------------------------------
consecutive_failures=0
for ((iter = 1; iter <= MAX_ITERS; iter++)); do
  if past_stop; then say "${YELLOW}stop time $STOP_AT reached before iteration $iter${NC}"; break; fi
  prompt="$NIGHT_DIR/prompt-iter$iter.md"; logfile="$LOG_DIR/iter$iter.log"
  python3 -m tests.sim.customer render-prompt --template "$KIT/PROMPT_customer.md" --persona "$PERSONA" --out "$prompt" \
    --night-dir "$NIGHT_DIR" --iter "$iter" --max-iters "$MAX_ITERS" --stop-at "$STOP_AT" --date "$DATE" \
    --set "NIGHT_START=$NIGHT_START" >/dev/null || { say "${RED}could not render the prompt (is the backend up?)${NC}"; sleep 120; continue; }
  say "${GREEN}▶ iteration $iter${NC}"
  set_status "ITER$iter" STARTED
  run_claude "$prompt" "$logfile"
  if is_usage_limit_error "$CLAUDE_OUTPUT" "$CLAUDE_EXIT"; then
    set_status "ITER$iter" LIMIT
    say "${YELLOW}usage limit — waiting${NC}"; countdown "$(get_sleep_duration "$CLAUDE_OUTPUT")" "Limit wait..."; iter=$((iter - 1)); continue
  fi
  if [[ $CLAUDE_EXIT -eq 124 || $CLAUDE_EXIT -eq 137 ]]; then
    # the iteration used its whole window — that is a full iteration, not a failure
    consecutive_failures=0; set_status "ITER$iter" TIMEOUT
    say "${YELLOW}iteration $iter ran to the ${ITER_TIMEOUT} limit; continuing from the diary${NC}"
    if echo "$CLAUDE_OUTPUT" | grep -q "NIGHT_COMPLETE"; then say "${GREEN}NIGHT_COMPLETE${NC}"; break; fi
    continue
  fi
  if [[ $CLAUDE_EXIT -ne 0 ]]; then
    consecutive_failures=$((consecutive_failures + 1)); set_status "ITER$iter" "EXIT$CLAUDE_EXIT"
    say "${RED}iteration $iter exited $CLAUDE_EXIT (see $logfile)${NC}"
    [[ $consecutive_failures -ge 3 ]] && { say "${RED}three failures in a row — stopping${NC}"; break; }
    countdown $((60 * consecutive_failures)) "Retrying..."; continue
  fi
  consecutive_failures=0; set_status "ITER$iter" DONE
  if echo "$CLAUDE_OUTPUT" | grep -q "NIGHT_COMPLETE"; then say "${GREEN}NIGHT_COMPLETE${NC}"; break; fi
done

# --- morning ------------------------------------------------------------------------
{
  echo; echo "## Cost (runner-appended at $(date '+%F %T'))"; echo
  python3 -m tests.sim.customer cost --since "$NIGHT_START" 2>&1
  echo; echo "## Inventory (runner-appended)"; echo
  python3 -m tests.sim.customer inventory --tag "sim-night-$DATE" 2>&1
} >> "$NIGHT_DIR/MORNING-REPORT.md"
set_status FINISHED "$(date -u +%Y-%m-%dT%H:%M:%S+00:00)"
say "${GREEN}morning report: $NIGHT_DIR/MORNING-REPORT.md · diary: $NIGHT_DIR/DIARY.md${NC}"
say "clean up when you have read it:  python3 -m tests.sim.customer purge --tag sim-night-$DATE --yes"
