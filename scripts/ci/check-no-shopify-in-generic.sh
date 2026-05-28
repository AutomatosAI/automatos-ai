#!/usr/bin/env bash
# scripts/ci/check-no-shopify-in-generic.sh
#
# PRD-141 §12 Integration Coupling Rules — CI gate.
#
# Generic widget / context / chatbot surfaces MUST NOT contain Shopify-
# specific identifiers. Vertical-specific code lives under
# orchestrator/integrations/<vertical>/ and is reachable only via the
# PLUGIN_REGISTRY exposed by orchestrator/integrations/__init__.py.
#
# Two passes over the gated paths (POSIX ERE — no PCRE lookahead so the
# gate runs on macOS bsd-grep and GNU grep identically):
#
#   Pass A — forbid Shopify field keys
#            (productHandle, productTitle, cartItems, cartItemCount,
#             cartTotalPrice, onlineStoreUrl)
#
#   Pass B — forbid the 'shopify_' prefix, allowing only the three
#            legitimate platform forms used for integration wiring:
#              shopify_plugin
#              shopify_integration
#              shopify_sync
#
# Either pass producing a hit fails CI with the offending file:line so
# the reviewer can see exactly what needs to move into integrations/.
#
# Excluded by design (not in the gated path list — kept here for the
# next engineer who wonders why these directories aren't scanned):
#   - orchestrator/integrations/        — vertical-specific code lives here
#   - orchestrator/api/shopify.py       — catalog sync, pre-PRD-141 surface
#   - graph_extraction.py               — map_shopify_catalog, outside scope
#
# Reference: docs/PRDS/141-WIDGET-VERTICAL-AGNOSTIC-REFACTOR.md §12

set -u

GATED_PATHS=(
  "orchestrator/api/widgets/"
  "orchestrator/modules/context/"
  "orchestrator/modules/knowledge/graph_service.py"
  "orchestrator/consumers/chatbot/"
)

PASS_A_PATTERN='productHandle|productTitle|cartItems|cartItemCount|cartTotalPrice|onlineStoreUrl'
PASS_B_PATTERN='shopify_'
PASS_B_ALLOWLIST='shopify_(plugin|integration|sync)'

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

EXISTING_PATHS=()
for p in "${GATED_PATHS[@]}"; do
  if [ -e "$p" ]; then
    EXISTING_PATHS+=("$p")
  else
    echo "WARN: gated path missing, skipping: $p" >&2
  fi
done

if [ ${#EXISTING_PATHS[@]} -eq 0 ]; then
  echo "ERROR: no gated paths found — check repo layout" >&2
  exit 2
fi

EXIT=0

PASS_A=$(grep -rnE "$PASS_A_PATTERN" "${EXISTING_PATHS[@]}" 2>/dev/null || true)
if [ -n "$PASS_A" ]; then
  echo "FAIL (Pass A) — forbidden Shopify field key in generic surface:"
  echo "$PASS_A"
  echo
  EXIT=1
fi

PASS_B=$(grep -rnE "$PASS_B_PATTERN" "${EXISTING_PATHS[@]}" 2>/dev/null \
  | grep -vE "$PASS_B_ALLOWLIST" || true)
if [ -n "$PASS_B" ]; then
  echo "FAIL (Pass B) — forbidden 'shopify_' identifier in generic surface:"
  echo "(allowed forms: shopify_plugin, shopify_integration, shopify_sync)"
  echo "$PASS_B"
  echo
  EXIT=1
fi

if [ $EXIT -eq 0 ]; then
  echo "OK — no Shopify identifiers in generic surfaces."
fi

exit $EXIT
