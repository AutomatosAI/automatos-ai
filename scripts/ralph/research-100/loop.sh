#!/bin/bash

# Super Ralph Loop — PRD-100 Research Factory
# Runs 9 loops: Loop 0 designs 8 PRD outlines, Loops 1-8 each write a full PRD
#
# Usage:
#   ./loop.sh              # Run all loops from current position
#   ./loop.sh 5            # Max 5 iterations per loop
#   ./loop.sh --loop 3     # Start from loop 3 (skip to PRD-103)
#   ./loop.sh --loop 0 3   # Run loop 0 with max 3 iterations

set -e

# Allow nested Claude Code sessions
unset CLAUDECODE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
META_FILE="$SCRIPT_DIR/meta.json"
PROMPT_FILE="$SCRIPT_DIR/PROMPT_research.md"

MAX_ITERATIONS=0
START_LOOP=-1
CONSECUTIVE_FAILURES=0

# Colors
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Parse args
for arg in "$@"; do
  if [[ "$arg" == "--loop" ]]; then
    START_LOOP="next"
  elif [[ "$START_LOOP" == "next" ]]; then
    START_LOOP=$arg
  elif [[ "$arg" =~ ^[0-9]+$ && "$START_LOOP" != "next" ]]; then
    MAX_ITERATIONS=$arg
  fi
done

if [[ ! -f "$META_FILE" ]]; then
  echo -e "${RED}Error: $META_FILE not found${NC}"
  exit 1
fi

if [[ ! -f "$PROMPT_FILE" ]]; then
  echo -e "${RED}Error: $PROMPT_FILE not found${NC}"
  exit 1
fi

# Get current loop from meta.json
get_current_loop() {
  python3 -c "import json; print(json.load(open('$META_FILE'))['currentLoop'])"
}

get_loop_name() {
  local loop_id=$1
  python3 -c "import json; loops=json.load(open('$META_FILE'))['loops']; print([l['name'] for l in loops if l['id']==$loop_id][0])"
}

get_loop_description() {
  local loop_id=$1
  python3 -c "import json; loops=json.load(open('$META_FILE'))['loops']; print([l['description'] for l in loops if l['id']==$loop_id][0])"
}

get_total_loops() {
  python3 -c "import json; print(json.load(open('$META_FILE'))['totalLoops'])"
}

update_meta_loop() {
  local loop_id=$1
  local status=$2
  python3 -c "
import json
with open('$META_FILE', 'r') as f:
    meta = json.load(f)
for loop in meta['loops']:
    if loop['id'] == $loop_id:
        loop['status'] = '$status'
if '$status' == 'complete':
    meta['currentLoop'] = $loop_id + 1
with open('$META_FILE', 'w') as f:
    json.dump(meta, f, indent=2)
"
}

seconds_until_next_hour() {
  local current_minute=$(date +%M)
  local current_second=$(date +%S)
  local seconds_past_hour=$((10#$current_minute * 60 + 10#$current_second))
  echo $((3600 - seconds_past_hour))
}

seconds_until_daily_reset() {
  local reset_hour=5
  local now=$(date +%s)
  local today_reset=$(date -v${reset_hour}H -v0M -v0S +%s 2>/dev/null || date -d "today ${reset_hour}:00:00" +%s)
  if [[ $now -ge $today_reset ]]; then
    echo $(( (today_reset + 86400) - now ))
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
  local resume_time=$(date -v+${sleep_duration}S "+%Y-%m-%d %H:%M:%S" 2>/dev/null || date -d "+${sleep_duration} seconds" "+%Y-%m-%d %H:%M:%S")
  echo -e "Expected resume: ${CYAN}${resume_time}${NC}"
  countdown $sleep_duration "Waiting..."
  echo -e "${GREEN}Resuming...${NC}"
  CONSECUTIVE_FAILURES=0
}

# Override start loop if specified
if [[ "$START_LOOP" != "-1" && "$START_LOOP" != "next" ]]; then
  python3 -c "
import json
with open('$META_FILE', 'r') as f:
    meta = json.load(f)
meta['currentLoop'] = $START_LOOP
with open('$META_FILE', 'w') as f:
    json.dump(meta, f, indent=2)
"
  echo -e "${CYAN}Starting from loop $START_LOOP${NC}"
fi

TOTAL_LOOPS=$(get_total_loops)

echo ""
echo -e "${MAGENTA}╔════════════════════════════════════════════════╗${NC}"
echo -e "${MAGENTA}║   Super Ralph Loop — PRD-100 Research Factory  ║${NC}"
echo -e "${MAGENTA}║   9 Loops: Design 8 PRDs, then Write 8 PRDs   ║${NC}"
echo -e "${MAGENTA}╚════════════════════════════════════════════════╝${NC}"
echo ""
[[ $MAX_ITERATIONS -gt 0 ]] && echo -e "Max iterations per loop: ${CYAN}$MAX_ITERATIONS${NC}"
echo -e "Press ${YELLOW}Ctrl+C${NC} to stop"
echo "---"

# Main loop — iterate through all loops
while true; do
  CURRENT_LOOP=$(get_current_loop)

  if [[ $CURRENT_LOOP -ge $TOTAL_LOOPS ]]; then
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║   ALL 9 LOOPS COMPLETE — PRDs 101-108 DONE    ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════╝${NC}"
    break
  fi

  LOOP_NAME=$(get_loop_name $CURRENT_LOOP)
  LOOP_DESC=$(get_loop_description $CURRENT_LOOP)

  echo ""
  echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo -e "${MAGENTA}  Loop $CURRENT_LOOP/$((TOTAL_LOOPS - 1)): $LOOP_DESC${NC}"
  echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

  update_meta_loop $CURRENT_LOOP "in_progress"
  ITERATION=0

  # Inner loop — iterate within a single loop
  while true; do
    ITERATION=$((ITERATION + 1))
    echo ""
    echo -e "${GREEN}=== Loop $CURRENT_LOOP · Iteration $ITERATION ===${NC}"

    TEMP_OUTPUT=$(mktemp)
    set +e

    cd "$REPO_ROOT"
    claude --print \
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

    if [[ $EXIT_CODE -ne 0 ]]; then
      CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
      echo -e "${RED}=== Error (exit code: $EXIT_CODE) ===${NC}"
      echo "$OUTPUT" | tail -20

      BACKOFF=$((30 * (2 ** (CONSECUTIVE_FAILURES - 1))))
      [[ $BACKOFF -gt 300 ]] && BACKOFF=300

      echo -e "${YELLOW}Retrying in ${BACKOFF}s... (failures: $CONSECUTIVE_FAILURES)${NC}"
      countdown $BACKOFF "Waiting..."
      ITERATION=$((ITERATION - 1))
      continue
    fi

    CONSECUTIVE_FAILURES=0

    if [[ "$RESULT_MSG" =~ RALPH_COMPLETE ]]; then
      echo ""
      echo -e "${GREEN}=== Loop $CURRENT_LOOP Complete ===${NC}"
      update_meta_loop $CURRENT_LOOP "complete"
      break
    fi

    if [[ $MAX_ITERATIONS -gt 0 && $ITERATION -ge $MAX_ITERATIONS ]]; then
      echo ""
      echo -e "${YELLOW}Reached max iterations ($MAX_ITERATIONS) for loop $CURRENT_LOOP. Moving to next loop.${NC}"
      update_meta_loop $CURRENT_LOOP "paused"
      break
    fi

    sleep 2
  done
done

echo ""
echo -e "${GREEN}Super Ralph Loop complete.${NC}"
