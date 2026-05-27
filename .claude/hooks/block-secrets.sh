#!/usr/bin/env bash
# PreToolUse hook: refuse to edit secret/env/credential files.
# Reads the standard Claude Code hook JSON from stdin.
set -euo pipefail

input="$(cat)"
file_path="$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty' 2>/dev/null || true)"

if [ -z "$file_path" ]; then
  exit 0
fi

case "$file_path" in
  *.env|*.env.*|*/.env|*/.env.*|\
  */credentials*|*/secrets/*|*/secret/*|\
  *.pem|*.key|*_rsa|*_rsa.pub|\
  */service-account*.json|*/sa-key*.json)
    echo "BLOCKED: refusing to edit secret/env file: $file_path" >&2
    echo "If this edit is genuinely needed, do it outside Claude Code." >&2
    exit 2
    ;;
esac

exit 0
