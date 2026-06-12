#!/bin/bash
# Deploy-parity check — proves a branch can run BOTH ways from one codebase:
#   (1) Railway/SaaS path: the production Docker images Railway builds actually build.
#   (2) Local/OSS path (--boot): the root compose stack boots and serves /health.
#
# Railway builds each service from its own dir via `dockerfileTarget: production`
# (railway.json). This runs the SAME build locally, so a branch that would fail
# Railway's build is caught here before you waste a Railway deploy.
#
# Usage:
#   ./scripts/ralph/deploy-parity-check.sh            # build-only (Railway-build proof) — run on any branch
#   ./scripts/ralph/deploy-parity-check.sh --boot     # + boot root compose & smoke /health (local-run proof)
#
# Use --boot on main for the known-good BASELINE; build-only per stacked branch.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
BOOT=0; [[ "${1:-}" == "--boot" ]] && BOOT=1
FAIL=0
BR=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "?")

if ! docker info >/dev/null 2>&1; then
  echo -e "${RED}Docker daemon not reachable — start Docker Desktop first.${NC}"
  exit 1
fi

echo -e "${CYAN}=== Deploy-parity check on branch '$BR' ===${NC}"

# --- (1) Railway-build proof: build the production targets Railway uses --------
# orchestrator/Dockerfile:95 `as production`, frontend/Dockerfile:92 `as production`.
build_prod() {
  local svc="$1" dir="$2"
  echo -e "\n${CYAN}── building $svc production image (railway target)${NC}"
  if docker build --target production -t "automatos-$svc:parity-$BR" -f "$dir/Dockerfile" "$dir"; then
    echo -e "${GREEN}✓ $svc production image builds${NC}"
  else
    echo -e "${RED}✗ $svc production build FAILED — Railway deploy of this branch would fail${NC}"
    FAIL=1
  fi
}
build_prod orchestrator orchestrator
build_prod frontend frontend

# --- (2) Local-run proof (--boot): root compose boots and serves --------------
if [[ $BOOT -eq 1 ]]; then
  echo -e "\n${CYAN}── booting root compose (isolated project 'parity-smoke')${NC}"
  PROJECT="parity-smoke"
  cleanup() { docker compose -p "$PROJECT" down -v --remove-orphans >/dev/null 2>&1 || true; }
  trap cleanup EXIT
  if docker compose -p "$PROJECT" up -d --build 2>&1 | tail -8; then
    echo "waiting for backend /health..."
    OK=0
    for i in $(seq 1 36); do
      # backend health on its mapped port — adjust if root compose differs
      if curl -fsS http://localhost:8000/health >/dev/null 2>&1 || curl -fsS http://localhost:8000/api/v1/health >/dev/null 2>&1; then OK=1; break; fi
      sleep 5
    done
    if [[ $OK -eq 1 ]]; then echo -e "${GREEN}✓ local compose boots and serves /health${NC}"; else echo -e "${RED}✗ backend never became healthy${NC}"; FAIL=1; fi
  else
    echo -e "${RED}✗ compose up failed${NC}"; FAIL=1
  fi
fi

echo ""
if [[ $FAIL -eq 0 ]]; then
  echo -e "${GREEN}=== PARITY OK on '$BR' — Railway images build$([[ $BOOT -eq 1 ]] && echo ' + local compose serves') ===${NC}"
else
  echo -e "${RED}=== PARITY FAILED on '$BR' — see above ===${NC}"
fi
exit $FAIL
