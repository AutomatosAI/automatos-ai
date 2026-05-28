#!/usr/bin/env bash
# PostToolUse hook: warn when model IDs or LLM defaults are hardcoded outside the central defaults files.
# Enforces: LLM defaults live in lib/llm-defaults.ts (frontend) + core/llm/defaults.py (backend).
set -euo pipefail

input="$(cat)"
file_path="$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty' 2>/dev/null || true)"

if [ -z "$file_path" ] || [ ! -f "$file_path" ]; then
  exit 0
fi

# Allow the canonical defaults files
case "$file_path" in
  */lib/llm-defaults.ts|*/core/llm/defaults.py|*/tests/*|*/scripts/*|*.md|*.json|*.yaml|*.yml)
    exit 0
    ;;
esac

# Only check source files
case "$file_path" in
  *.py|*.ts|*.tsx) ;;
  *) exit 0 ;;
esac

# Match common model identifiers (openrouter/anthropic/openai naming)
if grep -qE '"(anthropic|openai|deepseek|qwen|google|meta)/[a-z0-9._-]+"' "$file_path"; then
  echo "WARN: hardcoded model id in $file_path" >&2
  echo "      Automatos rule: model ids belong in lib/llm-defaults.ts or core/llm/defaults.py" >&2
fi

exit 0
