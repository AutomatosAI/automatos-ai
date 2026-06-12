#!/bin/bash

# Overnight Ralph chain — PRD-150 → 151 → 152 → 153 (stacked worktrees)
#
# Topology: each PRD gets its own worktree + branch, cut from the PREVIOUS
# PRD's final tip (150 cuts from main). Stories commit only behind green
# gates, so a branch tip is always stackable even if a PRD ends BLOCKED.
#
# Per PRD: build loop (Ralph) → deterministic acceptance gate → headless
# review cycle (max 2) → night report section. Chain CONTINUES on BLOCKED
# (tip is green by construction); it ABORTS only on environmental failure
# (repeated claude crashes — auth expired, CLI broken, etc).
#
# Usage:
#   ./scripts/ralph/overnight-prd150-153.sh             # full chain, resumes where it left off
#   ./scripts/ralph/overnight-prd150-153.sh 151 152     # subset, in the given order
#
# Env knobs:
#   RALPH_MAX_ITERS=25       per-PRD build-iteration cap
#   RALPH_STOP_AT=07:30      wind down gracefully at this wall-clock time
#   RALPH_SKIP_DOCKER_CHECK=1  skip the dev-stack pre-flight
#
# State:  scripts/ralph/state/prdNNN.status   (delete a file to force that PRD to re-run)
# Report: scripts/ralph/night-report-YYYY-MM-DD.md
# Logs:   scripts/ralph/logs/prdNNN-*.log
#
# caffeinate is self-applied on macOS — just run the script.

set -uo pipefail

# --- keep the Mac awake for the whole run -----------------------------------
if [[ "$(uname)" == "Darwin" ]] && command -v caffeinate >/dev/null 2>&1 && [[ -z "${RALPH_CAFFEINATED:-}" ]]; then
  export RALPH_CAFFEINATED=1
  exec caffeinate -dimsu "$0" "$@"
fi

# Allow nested Claude Code sessions (Ralph spawns child claude processes)
unset CLAUDECODE

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"     # .../automatos-ai
PLATFORM_ROOT="$(dirname "$REPO_ROOT")"              # .../Automatos-AI-Platform
RALPH_DIR="$REPO_ROOT/scripts/ralph"
STATE_DIR="$RALPH_DIR/state"
LOG_DIR="$RALPH_DIR/logs"
REPORT="$RALPH_DIR/night-report-prd150-153-$(date +%F).md"
mkdir -p "$STATE_DIR" "$LOG_DIR"

MAX_ITERS="${RALPH_MAX_ITERS:-25}"
ITER_TIMEOUT="45m"
REVIEW_TIMEOUT="30m"
MAX_REVIEW_CYCLES=2
TIMEOUT_BIN=$(command -v timeout || command -v gtimeout)

RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'

# --- per-PRD config (bash 3.2 — no assoc arrays) -----------------------------
branch_for() {
  case "$1" in
    150) echo "ralph/prd-150-auth-decoupling" ;;
    151) echo "ralph/prd-151-storage-minio" ;;
    152) echo "ralph/prd-152-mem0-decoupling" ;;
    153) echo "ralph/prd-153-one-command-run" ;;
  esac
}

base_for() {
  case "$1" in
    150) echo "main" ;;
    151) echo "$(branch_for 150)" ;;
    152) echo "$(branch_for 151)" ;;
    153) echo "$(branch_for 152)" ;;
  esac
}

wt_for() { echo "$PLATFORM_ROOT/automatos-ai-prd$1"; }

# --- stop-time handling ------------------------------------------------------
STOP_EPOCH=0
if [[ -n "${RALPH_STOP_AT:-}" ]]; then
  H="${RALPH_STOP_AT%%:*}"; M="${RALPH_STOP_AT##*:}"
  STOP_EPOCH=$(date -v"${H}"H -v"${M}"M -v0S +%s 2>/dev/null || date -d "today ${H}:${M}:00" +%s)
  [[ $STOP_EPOCH -le $(date +%s) ]] && STOP_EPOCH=$((STOP_EPOCH + 86400))
fi

past_stop_time() {
  [[ $STOP_EPOCH -gt 0 ]] && [[ $(date +%s) -ge $STOP_EPOCH ]]
}

# --- report helpers ----------------------------------------------------------
rep() { echo "$@" >> "$REPORT"; }

report_init() {
  if [[ ! -f "$REPORT" ]]; then
    rep "# Overnight Ralph chain — $(date '+%F %H:%M')"
    rep ""
    rep "Chain: ${PRD_LIST[*]} (stacked). Iter cap: $MAX_ITERS/PRD. Stop-at: ${RALPH_STOP_AT:-none}."
    rep ""
  else
    rep ""
    rep "---"
    rep "## Re-entered $(date '+%H:%M')"
  fi
}

# --- status file helpers -----------------------------------------------------
set_status() { echo "$2=$3" >> "$STATE_DIR/prd$1.status"; }
get_status() { grep "^$2=" "$STATE_DIR/prd$1.status" 2>/dev/null | tail -1 | cut -d= -f2; }

prd_already_done() {
  [[ "$(get_status "$1" BUILD)" == "COMPLETE" ]] \
    && [[ "$(get_status "$1" ACCEPT)" == "PASS" ]] \
    && [[ "$(get_status "$1" REVIEW)" == "PASS" ]]
}

# --- usage-limit handling (ported from loop-prd142-wave5.sh) ------------------
seconds_until_next_hour() {
  local seconds_past_hour=$((10#$(date +%M) * 60 + 10#$(date +%S)))
  echo $((3600 - seconds_past_hour))
}

seconds_until_daily_reset() {
  local reset_hour=5
  local now=$(date +%s)
  local today_reset=$(date -v${reset_hour}H -v0M -v0S +%s 2>/dev/null || date -d "today ${reset_hour}:00:00" +%s)
  if [[ $now -ge $today_reset ]]; then
    echo $((today_reset + 86400 - now))
  else
    echo $((today_reset - now))
  fi
}

countdown() {
  local seconds=$1 message=$2
  while [[ $seconds -gt 0 ]]; do
    printf "\r${CYAN}%s${NC} %02d:%02d:%02d " "$message" $((seconds / 3600)) $(((seconds % 3600) / 60)) $((seconds % 60))
    sleep 1
    seconds=$((seconds - 1))
  done
  printf "\r%-80s\r" " "
}

is_usage_limit_error() {
  local output="$1" exit_code="$2"
  [[ "$exit_code" -eq 0 ]] && return 1
  if echo "$output" | grep '^{' | jq -e 'select(.type == "result") | select(.subtype | test("error.*limit|rate_limit"))' &>/dev/null; then
    return 0
  fi
  local error_text
  error_text=$(echo "$output" | grep -v '^{' || true)
  error_text+=$(echo "$output" | grep '^{' | jq -r 'select(.type == "result" and .is_error == true) | .result // empty' 2>/dev/null || true)
  [[ "$error_text" =~ "You've hit your limit" ]] && return 0
  [[ "$error_text" =~ "You have hit your limit" ]] && return 0
  [[ "$error_text" =~ Error:\ 429 ]] && return 0
  [[ "$error_text" =~ Error:\ 529 ]] && return 0
  [[ "$error_text" =~ rate.?limit ]] && return 0
  [[ "$error_text" =~ usage.?limit ]] && return 0
  return 1
}

get_sleep_duration() {
  local output="$1"
  if [[ "$output" =~ "try again in "([0-9]+)" minute" ]]; then
    echo $(( ${BASH_REMATCH[1]} * 60 + 60 )); return
  fi
  if [[ "$output" =~ "try again in "([0-9]+)" hour" ]]; then
    echo $(( ${BASH_REMATCH[1]} * 3600 + 60 )); return
  fi
  if [[ "$output" =~ (daily|day|24.?hour) ]]; then
    seconds_until_daily_reset; return
  fi
  local wait_time=$(seconds_until_next_hour)
  [[ $wait_time -lt 300 ]] && wait_time=300
  echo $wait_time
}

handle_usage_limit() {
  local sleep_duration=$(get_sleep_duration "$1")
  echo -e "\n${YELLOW}=== Usage limit — waiting for reset ===${NC}"
  countdown "$sleep_duration" "Limit wait..."
  echo -e "${GREEN}Resuming...${NC}"
}

# --- one headless claude invocation ------------------------------------------
# Sets: CLAUDE_EXIT, CLAUDE_OUTPUT, CLAUDE_RESULT
run_claude() {
  local wt="$1" prompt_file="$2" tmo="$3" logfile="$4"
  local tmp; tmp=$(mktemp)

  cd "$wt"
  "$TIMEOUT_BIN" --kill-after=30s "$tmo" claude --print \
    --verbose \
    --output-format stream-json \
    --dangerously-skip-permissions \
    < "$prompt_file" 2>&1 | tee "$tmp" | tee -a "$logfile" | sed 's/\x1b\[[0-9;]*m//g' | grep --line-buffered '^{' | jq --unbuffered -r '
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

  CLAUDE_EXIT=${PIPESTATUS[0]}
  CLAUDE_OUTPUT=$(cat "$tmp")
  CLAUDE_RESULT=$(sed 's/\x1b\[[0-9;]*m//g' "$tmp" | grep '^{' | jq -r 'select(.type == "result") | .result // empty' 2>/dev/null | tail -1)
  rm -f "$tmp"
  cd "$REPO_ROOT"
}

# --- worktree management ------------------------------------------------------
ensure_worktree() {
  local num="$1" wt br base
  wt=$(wt_for "$num"); br=$(branch_for "$num"); base=$(base_for "$num")

  if [[ -d "$wt" ]]; then
    local cur
    cur=$(git -C "$wt" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
    if [[ "$cur" != "$br" ]]; then
      echo -e "${RED}Worktree $wt exists but is on '$cur', expected '$br'. Fix manually.${NC}"
      return 1
    fi
    echo -e "${CYAN}Reusing worktree $wt ($br)${NC}"
    return 0
  fi

  if git -C "$REPO_ROOT" show-ref --verify --quiet "refs/heads/$br"; then
    echo -e "${CYAN}Branch $br exists — attaching worktree${NC}"
    git -C "$REPO_ROOT" worktree add "$wt" "$br" || return 1
  else
    if ! git -C "$REPO_ROOT" show-ref --verify --quiet "refs/heads/${base#refs/heads/}" && [[ "$base" != "main" ]]; then
      echo -e "${RED}Base branch '$base' for PRD-$num does not exist — chain order broken.${NC}"
      return 1
    fi
    echo -e "${CYAN}Creating worktree $wt: $br ← $base${NC}"
    git -C "$REPO_ROOT" worktree add "$wt" -b "$br" "$base" || return 1
  fi

  # Gitignored env files don't travel with worktrees — copy from the main checkout.
  local f
  for f in .env orchestrator/.env frontend/.env frontend/.env.local; do
    if [[ -f "$REPO_ROOT/$f" && ! -f "$wt/$f" ]]; then
      cp "$REPO_ROOT/$f" "$wt/$f"
      echo "  copied $f"
    fi
  done

  # envs/*.defaults are UNTRACKED on main but load-bearing for PRD-150/153
  # (P153-S1 commits them onto its branch). Copy the whole dir.
  if [[ -d "$REPO_ROOT/envs" && ! -d "$wt/envs" ]]; then
    cp -R "$REPO_ROOT/envs" "$wt/envs"
    echo "  copied envs/"
  fi

  # Optional per-machine hook (venv, node_modules, etc).
  [[ -x "$RALPH_DIR/setup-worktree.sh" ]] && "$RALPH_DIR/setup-worktree.sh" "$wt"
  return 0
}

# Seed the loop scaffolding onto the branch (prd.json must be TRACKED — the
# loop marks stories DONE and commits it). Idempotent: skips if identical.
seed_scaffolding() {
  local num="$1" wt changed=0
  wt=$(wt_for "$num")
  mkdir -p "$wt/scripts/ralph"
  local f
  for f in "prd-$num.json" "PROMPT_build_prd$num.md" "PROMPT_review_prd$num.md" "acceptance-prd$num.sh"; do
    if [[ ! -f "$RALPH_DIR/$f" ]]; then
      echo -e "${RED}Missing $RALPH_DIR/$f — run the scaffolding step first.${NC}"
      return 1
    fi
    if ! cmp -s "$RALPH_DIR/$f" "$wt/scripts/ralph/$f" 2>/dev/null; then
      cp "$RALPH_DIR/$f" "$wt/scripts/ralph/$f"
      git -C "$wt" add "scripts/ralph/$f"
      changed=1
    fi
  done
  if [[ $changed -eq 1 ]]; then
    git -C "$wt" commit -q -m "chore(ralph): seed PRD-$num overnight loop scaffolding" || true
  fi
  return 0
}

# --- the Ralph build loop for one PRD -----------------------------------------
# Returns: 0=COMPLETE  2=BLOCKED  3=MAX_ITERS  4=ENVIRONMENTAL  5=TIME_UP
run_build_loop() {
  local num="$1" wt br prompt logfile
  wt=$(wt_for "$num"); br=$(branch_for "$num")
  prompt="$wt/scripts/ralph/PROMPT_build_prd$num.md"
  logfile="$LOG_DIR/prd$num-build-$(date +%F).log"

  local iteration=0 consecutive_failures=0

  while true; do
    past_stop_time && return 5
    iteration=$((iteration + 1))
    if [[ $iteration -gt $MAX_ITERS ]]; then return 3; fi

    echo -e "\n${GREEN}=== PRD-$num build iteration $iteration/$MAX_ITERS ($(date +%H:%M)) ===${NC}\n"

    # Branch lock — never build on the wrong branch.
    local cur
    cur=$(git -C "$wt" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
    if [[ "$cur" != "$br" ]]; then
      echo -e "${RED}PRD-$num worktree drifted to '$cur' (expected '$br') — halting PRD.${NC}"
      return 4
    fi

    run_claude "$wt" "$prompt" "$ITER_TIMEOUT" "$logfile"

    if is_usage_limit_error "$CLAUDE_OUTPUT" "$CLAUDE_EXIT"; then
      handle_usage_limit "$CLAUDE_OUTPUT"
      iteration=$((iteration - 1))
      continue
    fi

    if [[ $CLAUDE_EXIT -eq 124 || $CLAUDE_EXIT -eq 137 || $CLAUDE_EXIT -eq 143 ]]; then
      consecutive_failures=$((consecutive_failures + 1))
      echo -e "${RED}Iteration timed out (${ITER_TIMEOUT}) — discarding partial work${NC}"
      git -C "$wt" checkout -- . 2>/dev/null || true
      git -C "$wt" clean -fdq 2>/dev/null || true
      [[ $consecutive_failures -ge 4 ]] && return 4
      countdown 15 "Retrying..."
      iteration=$((iteration - 1))
      continue
    fi

    if [[ $CLAUDE_EXIT -ne 0 ]]; then
      consecutive_failures=$((consecutive_failures + 1))
      echo -e "${RED}Error (exit $CLAUDE_EXIT)${NC}"
      echo "$CLAUDE_OUTPUT" | tail -20
      [[ $consecutive_failures -ge 4 ]] && return 4
      local backoff=$((30 * (2 ** (consecutive_failures - 1))))
      [[ $backoff -gt 300 ]] && backoff=300
      countdown $backoff "Retrying..."
      iteration=$((iteration - 1))
      continue
    fi

    consecutive_failures=0

    # Per-story push: offsite backup + CI (real-Postgres suite, test.yml runs
    # on push to any branch, concurrency cancels superseded runs). Fire-and-
    # forget — CI is reconciled at the PRD boundary, never per story.
    if git -C "$wt" push -q -u origin "$br" 2>/dev/null; then
      echo -e "${CYAN}pushed → origin/$br (CI triggered)${NC}"
    else
      echo -e "${YELLOW}push failed (offline/auth?) — continuing; retry next iteration${NC}"
    fi

    if [[ "$CLAUDE_RESULT" =~ RALPH_COMPLETE ]]; then return 0; fi
    if [[ "$CLAUDE_RESULT" =~ RALPH_BLOCKED ]] || [[ "$CLAUDE_RESULT" =~ RALPH_ABORT ]]; then return 2; fi

    sleep 2
  done
}

# --- CI reconciliation (non-gating, mirrors test.yml's NON-REQUIRED status) -----
# Waits (bounded) for the branch head's CI run and RECORDS the outcome. It does
# not gate the chain: the workflow is deliberately non-required and may carry
# pre-existing gaps — the reviewer + morning human arbitrate new-vs-old reds.
# Echoes one of: SUCCESS / FAILURE / PENDING_TIMEOUT / NO_RUN / NO_GH
wait_for_ci() {
  local num="$1" wt br sha waited=0 line status conclusion
  wt=$(wt_for "$num"); br=$(branch_for "$num")
  command -v gh >/dev/null 2>&1 || { echo "NO_GH"; return 0; }
  git -C "$wt" push -q -u origin "$br" 2>/dev/null || true
  sha=$(git -C "$wt" rev-parse HEAD)
  while [[ $waited -lt 1200 ]]; do
    line=$(gh run list --branch "$br" --workflow test.yml --limit 5 \
      --json headSha,status,conclusion \
      --jq ".[] | select(.headSha == \"$sha\") | \"\(.status) \(.conclusion)\"" 2>/dev/null | head -1)
    status="${line%% *}"; conclusion="${line##* }"
    if [[ "$status" == "completed" ]]; then
      [[ "$conclusion" == "success" ]] && echo "SUCCESS" || echo "FAILURE"
      return 0
    fi
    [[ -z "$line" && $waited -ge 180 ]] && { echo "NO_RUN"; return 0; }
    sleep 60; waited=$((waited + 60))
  done
  echo "PENDING_TIMEOUT"
}

# --- draft PR at PRD completion ---------------------------------------------------
open_draft_pr() {
  local num="$1" wt br base
  wt=$(wt_for "$num"); br=$(branch_for "$num"); base=$(base_for "$num")
  command -v gh >/dev/null 2>&1 || return 0
  # Idempotent: skip if a PR for this head already exists.
  if [[ -n "$(gh pr list --head "$br" --json number --jq '.[0].number' 2>/dev/null)" ]]; then
    return 0
  fi
  (cd "$wt" && gh pr create --draft --base "$base" --head "$br" \
    --title "PRD-$num: $(branch_for "$num" | sed 's|ralph/prd-[0-9]*-||; s|-| |g') — Ralph overnight chain" \
    --body "Overnight Ralph chain output. Build COMPLETE, acceptance PASS, review PASS. See \`$(basename "$REPORT")\` and per-story commits (AC evidence in bodies). Stacked base: \`$base\` — merge in chain order.") \
    >/dev/null 2>&1 || true
}

# --- deterministic acceptance gate ---------------------------------------------
run_acceptance() {
  local num="$1" wt logfile rc
  wt=$(wt_for "$num")
  logfile="$LOG_DIR/prd$num-acceptance-$(date +%F).log"
  echo -e "\n${CYAN}=== PRD-$num acceptance gate (90m cap) ===${NC}"
  # Bounded: a hung `compose up --wait` or wedged test must not stall the chain.
  (cd "$wt" && "$TIMEOUT_BIN" --kill-after=60s 90m bash "scripts/ralph/acceptance-prd$num.sh") >> "$logfile" 2>&1
  rc=$?
  [[ $rc -eq 124 || $rc -eq 137 ]] && echo "ACCEPTANCE TIMED OUT (90m)" >> "$logfile"
  tail -15 "$logfile"
  return $rc
}

# --- headless review cycle ------------------------------------------------------
# Returns: 0=REVIEW_PASS  2=findings unresolved  3=review errored
run_review_cycle() {
  local num="$1" wt prompt logfile cycle
  wt=$(wt_for "$num")
  prompt="$wt/scripts/ralph/PROMPT_review_prd$num.md"
  logfile="$LOG_DIR/prd$num-review-$(date +%F).log"

  for cycle in $(seq 1 $MAX_REVIEW_CYCLES); do
    past_stop_time && return 2
    echo -e "\n${CYAN}=== PRD-$num review cycle $cycle/$MAX_REVIEW_CYCLES ===${NC}\n"
    run_claude "$wt" "$prompt" "$REVIEW_TIMEOUT" "$logfile"

    if is_usage_limit_error "$CLAUDE_OUTPUT" "$CLAUDE_EXIT"; then
      handle_usage_limit "$CLAUDE_OUTPUT"
      cycle=$((cycle - 1))
      continue
    fi
    [[ $CLAUDE_EXIT -ne 0 ]] && return 3

    if [[ "$CLAUDE_RESULT" =~ REVIEW_PASS ]]; then return 0; fi

    if [[ "$CLAUDE_RESULT" =~ REVIEW_FINDINGS ]]; then
      echo -e "${YELLOW}Reviewer filed fix stories — re-entering build loop${NC}"
      run_build_loop "$num"
      local rc=$?
      [[ $rc -eq 4 ]] && return 3
      run_acceptance "$num" || return 2
      continue
    fi

    # No sentinel — treat as unresolved, leave for the human.
    return 2
  done
  return 2
}

# --- pre-flight ------------------------------------------------------------------
preflight() {
  local missing=0 num
  for cmd in claude jq git; do
    command -v "$cmd" >/dev/null || { echo -e "${RED}Missing: $cmd${NC}"; missing=1; }
  done
  [[ -z "$TIMEOUT_BIN" ]] && { echo -e "${RED}Missing: timeout/gtimeout (brew install coreutils)${NC}"; missing=1; }

  for num in "${PRD_LIST[@]}"; do
    for f in "prd-$num.json" "PROMPT_build_prd$num.md" "PROMPT_review_prd$num.md" "acceptance-prd$num.sh"; do
      [[ -f "$RALPH_DIR/$f" ]] || { echo -e "${RED}Missing scaffolding: scripts/ralph/$f${NC}"; missing=1; }
    done
  done

  if [[ -z "${RALPH_SKIP_DOCKER_CHECK:-}" ]]; then
    if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qiE 'postgres|db'; then
      echo -e "${RED}Dev Postgres container not running — tests will fail all night.${NC}"
      echo -e "${YELLOW}Start the stack, or RALPH_SKIP_DOCKER_CHECK=1 to override.${NC}"
      missing=1
    fi
  fi

  [[ $missing -eq 1 ]] && exit 1
  return 0
}

# --- main --------------------------------------------------------------------------
PRD_LIST=("$@")
[[ ${#PRD_LIST[@]} -eq 0 ]] && PRD_LIST=(150 151 152 153)

preflight
report_init

trap 'rep ""; rep "**INTERRUPTED** at $(date +%H:%M)"; echo -e "\n${YELLOW}Interrupted — night report: $REPORT${NC}"; exit 130' INT TERM

echo -e "${GREEN}Overnight Ralph chain: PRD-${PRD_LIST[*]} (stacked) — report: $REPORT${NC}"

CHAIN_ABORTED=0
for NUM in "${PRD_LIST[@]}"; do
  if [[ $CHAIN_ABORTED -eq 1 ]]; then
    rep "## PRD-$NUM — SKIPPED (chain aborted)"
    continue
  fi
  if past_stop_time; then
    rep "## PRD-$NUM — SKIPPED (stop time ${RALPH_STOP_AT:-} reached)"
    continue
  fi
  if prd_already_done "$NUM"; then
    echo -e "${GREEN}PRD-$NUM already COMPLETE+PASS+PASS — skipping${NC}"
    rep "## PRD-$NUM — already done (state file), skipped"
    continue
  fi

  WT=$(wt_for "$NUM"); BR=$(branch_for "$NUM"); BASE=$(base_for "$NUM")
  rep ""
  rep "## PRD-$NUM — started $(date '+%H:%M')"
  rep "- Branch: \`$BR\` ← \`$BASE\` (worktree \`$WT\`)"

  if ! ensure_worktree "$NUM" || ! seed_scaffolding "$NUM"; then
    rep "- **SETUP FAILED** — see console; chain aborted"
    CHAIN_ABORTED=1
    continue
  fi

  BASE_SHA=$(git -C "$WT" merge-base "$BR" "$BASE" 2>/dev/null || git -C "$REPO_ROOT" rev-parse "$BASE")

  run_build_loop "$NUM"; BUILD_RC=$?
  case $BUILD_RC in
    0) set_status "$NUM" BUILD COMPLETE; rep "- Build: **RALPH_COMPLETE**" ;;
    2) set_status "$NUM" BUILD BLOCKED;  rep "- Build: **RALPH_BLOCKED** — tip is green, see last commit/log for why" ;;
    3) set_status "$NUM" BUILD MAX_ITERS; rep "- Build: hit iteration cap ($MAX_ITERS)" ;;
    5) set_status "$NUM" BUILD TIME_UP;  rep "- Build: stop time reached mid-PRD" ;;
    4) set_status "$NUM" BUILD ENV_FAIL
       rep "- Build: **ENVIRONMENTAL FAILURE** (repeated crashes) — **chain aborted**"
       CHAIN_ABORTED=1
       continue ;;
  esac

  CI_RESULT=$(wait_for_ci "$NUM")
  set_status "$NUM" CI "$CI_RESULT"
  rep "- CI (test.yml, real-Postgres suite): **$CI_RESULT** (non-gating; reviewer arbitrates new-vs-pre-existing reds)"

  if run_acceptance "$NUM"; then
    set_status "$NUM" ACCEPT PASS; rep "- Acceptance: **PASS**"
    ACCEPT_OK=1
  else
    set_status "$NUM" ACCEPT FAIL; rep "- Acceptance: **FAIL** (\`logs/prd$NUM-acceptance-$(date +%F).log\`)"
    ACCEPT_OK=0
  fi

  if [[ $BUILD_RC -eq 0 && $ACCEPT_OK -eq 1 ]]; then
    run_review_cycle "$NUM"; REVIEW_RC=$?
    case $REVIEW_RC in
      0) set_status "$NUM" REVIEW PASS;    rep "- Review: **REVIEW_PASS**"
         git -C "$WT" push -q -u origin "$BR" 2>/dev/null || true
         open_draft_pr "$NUM"
         PR_URL=$(gh pr list --head "$BR" --json url --jq '.[0].url' 2>/dev/null || true)
         [[ -n "$PR_URL" ]] && rep "- Draft PR: $PR_URL" ;;
      2) set_status "$NUM" REVIEW FINDINGS; rep "- Review: findings unresolved — human review needed" ;;
      3) set_status "$NUM" REVIEW ERROR;   rep "- Review: errored — human review needed" ;;
    esac
  else
    set_status "$NUM" REVIEW SKIPPED
    rep "- Review: skipped (build/acceptance incomplete)"
  fi

  COMMITS=$(git -C "$WT" rev-list --count "$BASE_SHA..HEAD" 2>/dev/null || echo "?")
  DIFFSTAT=$(git -C "$WT" diff --stat "$BASE_SHA..HEAD" 2>/dev/null | tail -1)
  rep "- Delta vs base: $COMMITS commits — ${DIFFSTAT:-no changes}"
  rep "- Finished $(date '+%H:%M')"
done

rep ""
rep "---"
rep "Chain finished $(date '+%F %H:%M'). Morning protocol: review + merge IN ORDER (150 first); each later branch then rebases trivially. Test one worktree at a time."

echo -e "\n${GREEN}=== Chain finished — night report: $REPORT ===${NC}"
if command -v osascript >/dev/null 2>&1; then
  osascript -e 'display notification "Night report ready — PRD-150…153" with title "Ralph overnight chain finished"' 2>/dev/null || true
fi
