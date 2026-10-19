#!/usr/bin/env bash
# PRD-261 A-00: wait until a URL answers HTTP 200.
#
# Usage: wait-for-url.sh <url> <seconds>
#
# Polls every 5 seconds. Exits 0 on the first 200; exits 1 with the last status code
# when <seconds> pass without one. The browser lane uses it for the backend's
# /health/ready and the frontend's first page. A copy of the loop in
# scripts/ci/smoke-fresh-clone.sh, kept under e2e/ so the CI path planner stays narrow.
set -uo pipefail

POLL_INTERVAL_SECONDS=5
REQUEST_TIMEOUT_SECONDS=10

if [ "$#" -ne 2 ]; then
  echo "usage: wait-for-url.sh <url> <seconds>" >&2
  exit 2
fi

url="$1"
max_wait="$2"
case "$max_wait" in
  '' | *[!0-9]*)
    echo "wait-for-url.sh: <seconds> must be a whole number, got '${max_wait}'" >&2
    exit 2
    ;;
esac

elapsed=0
status="000"
while [ "$elapsed" -lt "$max_wait" ]; do
  # curl prints 000 itself when it cannot connect; `|| true` keeps that single value.
  status="$(curl -s -o /dev/null --max-time "$REQUEST_TIMEOUT_SECONDS" -w '%{http_code}' "$url" || true)"
  if [ "$status" = "200" ]; then
    echo "OK: ${url} returned 200 after ${elapsed}s"
    exit 0
  fi
  echo "  ...${elapsed}s: ${url} -> ${status:-000}"
  sleep "$POLL_INTERVAL_SECONDS"
  elapsed=$((elapsed + POLL_INTERVAL_SECONDS))
done

echo "FAIL: ${url} did not return 200 within ${max_wait}s (last status ${status:-000})" >&2
exit 1
