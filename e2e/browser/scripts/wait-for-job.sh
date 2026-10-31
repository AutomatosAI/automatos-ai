#!/usr/bin/env bash
# PRD-261 A-17a: wait until another job of this workflow run finishes, and print how it ended.
#
# Usage: wait-for-job.sh <job name> <seconds>
#
# The browser lane runs beside the job that packs a theme (Play) from the Design Kit instead of after it
# (no `needs:`), so the kit's install and build overlap the lane's docker and frontend builds; the
# lane waits here, just before it downloads the package. Reads GITHUB_REPOSITORY and GITHUB_RUN_ID;
# GH_TOKEN needs actions: read. Polls the run's jobs (the API lists the latest attempt of each, so a
# re-run of the lane alone sees the package job's earlier result) every 10 seconds.
#
# Prints the job's conclusion (success, failure, cancelled, ...) on stdout and exits 0 once the job
# has completed. Exits 1 when <seconds> pass first. Progress goes to stderr.
set -uo pipefail

POLL_INTERVAL_SECONDS=10
JOBS_PER_PAGE=100
COMPLETED=completed

if [ "$#" -ne 2 ]; then
  echo "usage: wait-for-job.sh <job name> <seconds>" >&2
  exit 2
fi

job_name="$1"
max_wait="$2"
case "$max_wait" in
  '' | *[!0-9]*)
    echo "wait-for-job.sh: <seconds> must be a whole number, got '${max_wait}'" >&2
    exit 2
    ;;
esac

# "<status> <conclusion>" of the named job, or nothing when the run does not list it (yet).
job_state() {
  gh api --paginate "repos/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}/jobs?per_page=${JOBS_PER_PAGE}" \
    | jq -r --arg name "$job_name" '.jobs[] | select(.name == $name) | "\(.status) \(.conclusion // "")"' \
    | tail -n 1
}

elapsed=0
state=""
while [ "$elapsed" -lt "$max_wait" ]; do
  state="$(job_state || true)"
  status="${state%% *}"
  conclusion="${state#* }"
  if [ "$status" = "$COMPLETED" ]; then
    echo "OK: '${job_name}' completed after ${elapsed}s: ${conclusion}" >&2
    echo "$conclusion"
    exit 0
  fi
  echo "  ...${elapsed}s: '${job_name}' is ${status:-not listed yet}" >&2
  sleep "$POLL_INTERVAL_SECONDS"
  elapsed=$((elapsed + POLL_INTERVAL_SECONDS))
done

echo "FAIL: '${job_name}' did not complete within ${max_wait}s (last state: ${state:-not listed})" >&2
exit 1
