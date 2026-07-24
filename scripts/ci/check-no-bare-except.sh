#!/usr/bin/env bash
# CI gate (PRD-141 Phase 0): fail if any bare `except:` exists under orchestrator/.
#
# A bare `except:` catches BaseException and therefore swallows SystemExit and
# KeyboardInterrupt, masking shutdown signals and hiding real failures. All
# handlers must name `except Exception:` (or a narrower type).
#
# Exit 0 = clean (gate passes). Exit 1 = bare except found (gate fails).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# Anchor on statement position: (indent)(except)(optional ws)(colon). This
# matches a real bare `except:` / `except :` while ignoring `except Exception:`
# and — unlike a plain `grep 'except:'` — the literal text "except:" inside
# comments, docstrings, or string literals. `|| true` stops `set -o pipefail`
# from tripping when grep finds nothing (exit 1).
matches="$(grep -rnE '^[[:space:]]*except[[:space:]]*:' orchestrator/ --include='*.py' | grep -v __pycache__ || true)"

if [ -n "$matches" ]; then
  echo "FAIL: bare 'except:' found — use 'except Exception:' instead:" >&2
  echo "$matches" >&2
  exit 1
fi

echo "OK: no bare 'except:' under orchestrator/"
exit 0
