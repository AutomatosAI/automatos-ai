#!/bin/bash

# Ralph Wiggum Build Loop (Claude) — PRD-142 Wave 5 (Cut List: WS-Z routes + WS-AA prep)
# Usage:
#   ./loop-prd142-wave5.sh           # Build mode (default, runs until RALPH_COMPLETE)
#   ./loop-prd142-wave5.sh 12        # Max 12 iterations (safety cap)
#
# Branch lock: ralph/prd-142-wave5-cut-list (run from the
#   automatos-ai-wave5 worktree; cut the branch from main AFTER the WS-W/WS-X
#   PR merges). Agent scope: W5-S1..S6 (route-cluster verify-then-cut) +
#   W5-S7 (reversible DROP-migration prep, never applied). Human-only:
#   applying migrations on prod; any route whose external exposure stays
#   ambiguous after the checks. See PROMPT_build_prd142-wave5.md.

set -e

# Allow nested Claude Code sessions (Ralph spawns child claude processes)
unset CLAUDECODE

MODE="build"
MAX_ITERATIONS=0
ITERATION=0
CONSECUTIVE_FAILURES=0

# Per-iteration wall-clock cap. A single story (TDD + validate + commit) should
# finish well inside this; anything longer is a hung child, not real work.
# timeout SIGTERMs at this mark, then SIGKILLs 30s later if it ignores TERM.
ITER_TIMEOUT="45m"
TIMEOUT_BIN=$(command -v timeout || command -v gtimeout)

# Colors
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

for arg in "$@"; do
  if [[ "$arg" =~ ^[0-9]+$ ]]; then
    MAX_ITERATIONS=$arg
  fi
done

PROMPT_FILE="scripts/ralph/PROMPT_build_prd142-wave5.md"

if [[ ! -f "$PROMPT_FILE" ]]; then
  echo -e "${RED}Error: $PROMPT_FILE not found${NC}"
  exit 1
fi

# Safety: refuse to run off the wrong branch (Ralph must stay on its own branch).
EXPECTED_BRANCH="ralph/prd-142-wave5-cut-list"
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
if [[ "$CURRENT_BRANCH" != "$EXPECTED_BRANCH" ]]; then
  echo -e "${RED}Error: on branch '$CURRENT_BRANCH', expected '$EXPECTED_BRANCH'.${NC}"
  echo -e "${YELLOW}Run from the automatos-ai-wave5 worktree (git worktree add ... -b $EXPECTED_BRANCH).${NC}"
  exit 1
fi

seconds_until_next_hour() {
  local now=$(date +%s)
  local current_minute=$(date +%M)
  local current_second=$(date +%S)
  local seconds_past_hour=$((10#$current_minute * 60 + 10#$current_second))
  local seconds_until=$((3600 - seconds_past_hour))
  echo $seconds_until
}

seconds_until_daily_reset() {
  local reset_hour=5
  local now=$(date +%s)
  local today_reset=$(date -v${reset_hour}H -v0M -v0S +%s 2>/dev/null || date -d "today ${reset_hour}:00:00" +%s)

  if [[ $now -ge $today_reset ]]; then
    local tomorrow_reset=$((today_reset + 86400))
    echo $((tomorrow_reset - now))
  else
    echo $((today_reset - now))
  fi
}

countdown() {
  local seconds=$1
  local message=$2

  while [[ $seconds -gt 0 ]]; do
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))
    printf "\r${CYAN}%s${NC} Time remaining: %02d:%02d:%02d " "$message" $hours $minutes $secs
    sleep 1
    ((seconds--))
  done
  printf "\r%-80s\r" " "
}

is_usage_limit_error() {
  local output="$1"
  local exit_code="$2"

  [[ "$exit_code" -eq 0 ]] && return 1

  if echo "$output" | grep '^{' | jq -e 'select(.type == "result") | select(.subtype | test("error.*limit|rate_limit"))' &>/dev/null; then
    return 0
  fi

  local error_text
  error_text=$(echo "$output" | grep -v '^{' || true)
  error_text+=$(echo "$output" | grep '^{' | jq -r 'select(.type == "result" and .is_error == true) | .result // empty' 2>/dev/null || true)

  if [[ "$error_text" =~ "You've hit your limit" ]] || [[ "$error_text" =~ "You have hit your limit" ]]; then
    return 0
  fi
  if [[ "$error_text" =~ Error:\ 429 ]] || [[ "$error_text" =~ Error:\ 529 ]]; then
    return 0
  fi
  if [[ "$error_text" =~ rate.?limit ]] || [[ "$error_text" =~ usage.?limit ]]; then
    return 0
  fi
  return 1
}

get_sleep_duration() {
  local output="$1"

  if [[ "$output" =~ "try again in "([0-9]+)" minute" ]]; then
    echo $(( ${BASH_REMATCH[1]} * 60 + 60 ))
    return
  fi

  if [[ "$output" =~ "try again in "([0-9]+)" hour" ]]; then
    echo $(( ${BASH_REMATCH[1]} * 3600 + 60 ))
    return
  fi

  if [[ "$output" =~ (daily|day|24.?hour) ]]; then
    seconds_until_daily_reset
    return
  fi

  local wait_time=$(seconds_until_next_hour)
  [[ $wait_time -lt 300 ]] && wait_time=300
  echo $wait_time
}

handle_usage_limit() {
  local output="$1"
  local sleep_duration=$(get_sleep_duration "$output")

  echo ""
  echo -e "${YELLOW}=== Usage Limit Detected ===${NC}"
  echo -e "${YELLOW}Waiting for reset...${NC}"
  echo ""

  local resume_time=$(date -v+${sleep_duration}S "+%Y-%m-%d %H:%M:%S" 2>/dev/null || date -d "+${sleep_duration} seconds" "+%Y-%m-%d %H:%M:%S")
  echo -e "Expected resume: ${CYAN}${resume_time}${NC}"
  echo ""

  countdown $sleep_duration "Waiting..."

  echo ""
  echo -e "${GREEN}Resuming...${NC}"
  echo ""

  CONSECUTIVE_FAILURES=0
}

echo -e "${GREEN}Ralph loop: BUILD mode — PRD-142 Wave 5 Cut List (W5-S1..S7 agent-safe; prod migration apply + ambiguous-exposure routes are human)${NC}"
[[ $MAX_ITERATIONS -gt 0 ]] && echo "Max iterations: $MAX_ITERATIONS"
echo "Branch: $CURRENT_BRANCH"
echo "Press Ctrl+C to stop"
echo "---"

while true; do
  ITERATION=$((ITERATION + 1))
  echo ""
  echo -e "${GREEN}=== Build Iteration $ITERATION ===${NC}"
  echo ""

  TEMP_OUTPUT=$(mktemp)
  set +e

  "$TIMEOUT_BIN" --kill-after=30s "$ITER_TIMEOUT" claude --print \
    --verbose \
    --output-format stream-json \
    --dangerously-skip-permissions \
    < "$PROMPT_FILE" 2>&1 | tee "$TEMP_OUTPUT" | sed 's/\x1b\[[0-9;]*m//g' | grep --line-buffered '^{' | jq --unbuffered -r '
      def tool_info:
        if .name == "Edit" or .name == "Write" or .name == "Read" then
          (.input.file_path // .input.path | split("/") | last | .[0:60])
        elif .name == "Bash" then
          (.input.command // .input.cmd | if contains("\n") then split("\n") | first | .[0:50] else .[0:80] end)
        elif .name == "Grep" then
          (.input.pattern | .[0:40])
        elif .name == "Glob" then
          (.input.pattern // .input.filePattern | .[0:40])
        else null end;
      if .type == "assistant" then
        .message.content[] |
        if .type == "text" then
          if (.text | split("\n") | length) <= 3 then .text else empty end
        elif .type == "tool_use" then
          "    [" + .name + "]" + (tool_info | if . then " " + . else "" end)
        else empty end
      elif .type == "result" then
        "--- " + ((.duration_ms / 1000 * 10 | floor / 10) | tostring) + "s, " + (.num_turns | tostring) + " turns ---"
      else empty end
    ' 2>/dev/null

  EXIT_CODE=${PIPESTATUS[0]}
  OUTPUT=$(cat "$TEMP_OUTPUT")
  RESULT_MSG=$(sed 's/\x1b\[[0-9;]*m//g' "$TEMP_OUTPUT" | grep '^{' | jq -r 'select(.type == "result") | .result // empty' 2>/dev/null | tail -1)
  rm -f "$TEMP_OUTPUT"
  set -e

  if is_usage_limit_error "$OUTPUT" "$EXIT_CODE"; then
    handle_usage_limit "$OUTPUT"
    ITERATION=$((ITERATION - 1))
    continue
  fi

  # Timeout: GNU timeout exits 124 (TERM at the mark) or 137 (SIGKILL after
  # --kill-after). A story that blew the cap left a half-edited tree — discard
  # it so the next iteration starts from the last clean commit, then retry.
  if [[ $EXIT_CODE -eq 124 || $EXIT_CODE -eq 137 || $EXIT_CODE -eq 143 ]]; then
    CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
    echo ""
    echo -e "${RED}=== Iteration timed out after ${ITER_TIMEOUT} (exit $EXIT_CODE) — killed hung child ===${NC}"
    echo -e "${YELLOW}Discarding partial work so the next iteration starts clean...${NC}"
    git checkout -- . 2>/dev/null || true
    git clean -fdq 2>/dev/null || true
    echo -e "${YELLOW}Retrying in 15s... (consecutive failures: $CONSECUTIVE_FAILURES)${NC}"
    countdown 15 "Waiting..."
    ITERATION=$((ITERATION - 1))
    continue
  fi

  if [[ $EXIT_CODE -ne 0 ]]; then
    CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
    echo ""
    echo -e "${RED}=== Error (exit code: $EXIT_CODE) ===${NC}"
    echo -e "${RED}Output:${NC}"
    echo "$OUTPUT" | tail -20
    echo ""

    BACKOFF=$((30 * (2 ** (CONSECUTIVE_FAILURES - 1))))
    [[ $BACKOFF -gt 300 ]] && BACKOFF=300

    echo -e "${YELLOW}Retrying in ${BACKOFF}s... (consecutive failures: $CONSECUTIVE_FAILURES)${NC}"
    countdown $BACKOFF "Waiting..."
    ITERATION=$((ITERATION - 1))
    continue
  fi

  CONSECUTIVE_FAILURES=0

  if [[ "$RESULT_MSG" =~ RALPH_COMPLETE ]]; then
    echo ""
    echo -e "${GREEN}=== Ralph Complete ===${NC}"
    echo -e "${GREEN}PRD-142 Wave 5 agent scope done. Human follow-through: apply the WS-AA DROP migrations on prod (Gerard), review any routes flagged ambiguous-external.${NC}"
    break
  fi

  if [[ "$RESULT_MSG" =~ RALPH_BLOCKED ]] || [[ "$RESULT_MSG" =~ RALPH_ABORT ]]; then
    echo ""
    echo -e "${YELLOW}=== Ralph Halted (BLOCKED/ABORT) ===${NC}"
    echo -e "${YELLOW}A story could not proceed safely. Loop halted — see commit log / last output for the reason.${NC}"
    break
  fi

  if [[ $MAX_ITERATIONS -gt 0 && $ITERATION -ge $MAX_ITERATIONS ]]; then
    echo ""
    echo -e "${GREEN}Reached max iterations ($MAX_ITERATIONS).${NC}"
    break
  fi

  sleep 2
done

echo ""
echo -e "${GREEN}Ralph loop complete. Iterations: $ITERATION${NC}"
