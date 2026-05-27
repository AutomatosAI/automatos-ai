#!/usr/bin/env bash
# PostToolUse hook: warn when os.getenv() lands in a .py file that isn't config.py.
# Enforces the Automatos rule: all env reads go through config.py.
set -euo pipefail

input="$(cat)"
file_path="$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty' 2>/dev/null || true)"

if [ -z "$file_path" ] || [ ! -f "$file_path" ]; then
  exit 0
fi

# Only check Python files
case "$file_path" in
  *.py) ;;
  *) exit 0 ;;
esac

# Allow these paths to use os.getenv directly
case "$file_path" in
  */config.py|*/conftest.py|*/tests/*|*/scripts/*|*/alembic/env.py)
    exit 0
    ;;
esac

if grep -qE '\bos\.getenv\(' "$file_path"; then
  echo "WARN: os.getenv() detected in $file_path" >&2
  echo "      Automatos rule: env reads must go through config.py (memory: configuration-rules.md)" >&2
fi

exit 0
