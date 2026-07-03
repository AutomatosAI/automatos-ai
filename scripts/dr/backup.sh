#!/usr/bin/env bash
# PRD-176 F050 — pg_dump backup for the Automatos primary database (and,
# optionally, the separate mem0 database).
#
# Produces a custom-format dump (`pg_dump -Fc`) — compressed, and restorable
# selectively with `pg_restore`. Custom format is required so the matching
# restore.sh can rebuild into a fresh volume.
#
# Connection comes from the canonical DATABASE_URL (same source the app uses).
# No credentials are hardcoded (CLAUDE.md §4 / security §Secret Management):
# pass them via DATABASE_URL in the environment. The optional mem0 target is
# MEM0_DATABASE_URL — losing mem0 loses durable memory, so it is a first-class
# backup target, not an afterthought.
#
# Usage:
#   DATABASE_URL=postgresql://user:pw@host:5432/orchestrator_db \
#     scripts/dr/backup.sh [OUTPUT_DIR]
#
#   # include the mem0 instance:
#   DATABASE_URL=... MEM0_DATABASE_URL=postgresql://.../mem0 \
#     scripts/dr/backup.sh /backups
#
# Exit 0 = all requested dumps written. Non-zero = a dump failed.
set -euo pipefail

OUTPUT_DIR="${1:-${DR_BACKUP_DIR:-./backups}}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

if [ -z "${DATABASE_URL:-}" ]; then
  echo "FAIL: DATABASE_URL is not set (the primary DB connection string)." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

_dump() {
  local url="$1"
  local label="$2"
  local out="${OUTPUT_DIR}/${label}-${TIMESTAMP}.dump"
  echo "Dumping ${label} -> ${out}"
  # -Fc custom format, --no-owner/--no-privileges so the dump restores cleanly
  # into a fresh instance with a possibly-different role.
  pg_dump --format=custom --no-owner --no-privileges --file="$out" "$url"
  local bytes
  bytes="$(wc -c < "$out" | tr -d ' ')"
  if [ "$bytes" -le 0 ]; then
    echo "FAIL: ${label} dump is empty (${out})." >&2
    return 1
  fi
  echo "OK: ${label} dump written (${bytes} bytes)"
  echo "$out"
}

echo "========================================="
echo "PRD-176 DR backup  (${TIMESTAMP})"
echo "  output dir: ${OUTPUT_DIR}"
echo "========================================="

_dump "$DATABASE_URL" "primary"

if [ -n "${MEM0_DATABASE_URL:-}" ]; then
  _dump "$MEM0_DATABASE_URL" "mem0"
else
  echo "NOTE: MEM0_DATABASE_URL unset — skipping mem0 backup."
  echo "      Set it to back up the separate mem0 instance (durable memory)."
fi

echo "Backup complete."
