#!/usr/bin/env bash
# PRD-176 F050 — pg_restore restore for an Automatos custom-format dump.
#
# Restores a `pg_dump -Fc` dump (from backup.sh) into a TARGET database. The
# target should be a fresh/empty database — this is the disaster-recovery path,
# not an in-place merge.
#
# Handles the two things a naive restore misses (PRD-176 §4.6):
#   1. pgvector — the schema declares `vector` columns, so the extension must
#      exist in the target BEFORE the restore runs. We create it first.
#   2. raw-DDL tables (e.g. document_chunks) — these ARE captured in a -Fc dump,
#      so pg_restore rebuilds them; the risk is only the extension above.
#
# Connection comes from RESTORE_DATABASE_URL (the target). No hardcoded creds.
#
# Usage:
#   RESTORE_DATABASE_URL=postgresql://user:pw@host:5432/restore_db \
#     scripts/dr/restore.sh path/to/primary-YYYYMMDDT...Z.dump
#
# Exit 0 = restore completed. Non-zero = restore failed.
set -euo pipefail

DUMP_FILE="${1:-}"

if [ -z "$DUMP_FILE" ]; then
  echo "FAIL: no dump file given. Usage: restore.sh <dump-file>" >&2
  exit 1
fi
if [ ! -f "$DUMP_FILE" ]; then
  echo "FAIL: dump file not found: ${DUMP_FILE}" >&2
  exit 1
fi
if [ -z "${RESTORE_DATABASE_URL:-}" ]; then
  echo "FAIL: RESTORE_DATABASE_URL is not set (the target DB connection string)." >&2
  exit 1
fi

echo "========================================="
echo "PRD-176 DR restore"
echo "  dump  : ${DUMP_FILE}"
echo "  target: (RESTORE_DATABASE_URL)"
echo "========================================="

# 1. Ensure pgvector exists in the target before restoring vector columns.
#    Best-effort: on a pgvector image this succeeds; on a stock Postgres the
#    extension control file is absent. Don't hard-abort the whole restore here —
#    warn, and let pg_restore surface any genuine vector-column failure at the
#    point it actually matters. (The target for a real recovery is a pgvector
#    image; the runbook says so.)
echo "Ensuring pgvector extension exists in target..."
if ! psql "$RESTORE_DATABASE_URL" -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>/dev/null; then
  echo "WARN: could not create the 'vector' extension (not a pgvector image?)." >&2
  echo "      Restore will continue; vector columns require a pgvector target." >&2
fi

# 2. Restore. --no-owner/--no-privileges to tolerate a different target role;
#    --exit-on-error so a genuine restore failure is loud (a silent partial
#    restore is worse than a failed one).
echo "Restoring dump with pg_restore..."
pg_restore \
  --no-owner \
  --no-privileges \
  --exit-on-error \
  --dbname="$RESTORE_DATABASE_URL" \
  "$DUMP_FILE"

echo "OK: restore completed."
