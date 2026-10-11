#!/usr/bin/env bash
# PRD-258 US-003 (D3): open the ngrok tunnel Retell reaches the local backend
# through. Steps and context: scripts/voice/README.md.
#
#   NGROK_DOMAIN=<your static domain> scripts/voice/tunnel.sh
#
# The domain is yours (one free ngrok account, one static domain); it is read
# from the environment and never committed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
COMPOSE_FILE="$REPO_ROOT/docker-compose.yml"
# Compose reads API_PORT from the shell, then from the repo's .env.
COMPOSE_ENV_FILE="$REPO_ROOT/.env"
# Only Retell's routes pass the tunnel; everything else is refused at the edge.
POLICY_FILE="$SCRIPT_DIR/ngrok-policy.yml"
# The backend's published port: docker-compose.yml, service `backend`, under
# `ports:` — the line `- "${BIND_ADDRESS:-127.0.0.1}:${API_PORT:-8000}:8000"`.
# API_PORT overrides it there (shell, then .env), so it overrides it here;
# otherwise the default is read from that line.
PORT_MAPPING_PATTERN='\$\{API_PORT:-[0-9]+\}:8000'

die() {
  echo "tunnel.sh: $*" >&2
  exit 1
}

command -v ngrok >/dev/null 2>&1 \
  || die "ngrok is not installed (https://ngrok.com/download), then: ngrok config add-authtoken <token>"
[ -n "${NGROK_DOMAIN:-}" ] \
  || die "set NGROK_DOMAIN to your ngrok static domain (bare host, no https://)"
case "$NGROK_DOMAIN" in
  *://* | */*) die "NGROK_DOMAIN must be a bare host name, without a scheme or a path" ;;
esac
[ -f "$COMPOSE_FILE" ] || die "cannot find $COMPOSE_FILE"
[ -f "$POLICY_FILE" ] || die "cannot find $POLICY_FILE"

mapping="$(grep -Eo "$PORT_MAPPING_PATTERN" "$COMPOSE_FILE" | head -n 1 || true)"
[ -n "$mapping" ] || die "no backend port mapping matching $PORT_MAPPING_PATTERN in $COMPOSE_FILE"
default_port="$(printf '%s' "$mapping" | sed -E 's/^\$\{API_PORT:-([0-9]+)\}:8000$/\1/')"
env_port=""
if [ -z "${API_PORT:-}" ] && [ -f "$COMPOSE_ENV_FILE" ]; then
  # Read the one API_PORT line only; .env is never sourced.
  env_port="$(sed -nE 's/^[[:space:]]*API_PORT[[:space:]]*=[[:space:]]*"?([0-9]+)"?[[:space:]]*$/\1/p' "$COMPOSE_ENV_FILE" | tail -n 1)"
fi
port="${API_PORT:-${env_port:-$default_port}}"

cat <<MSG
Put this in the backend's environment: in your local compose override
(docker-compose.override.yml — never .env, never committed), then restart
the backend and arm Auto Live in Settings:

    PUBLIC_API_HOST: ${NGROK_DOMAIN}

Tunnelling https://${NGROK_DOMAIN}/api/voice/retell/ -> localhost:${port}
(only Retell's routes pass; Ctrl-C stops it).
MSG

exec ngrok http --url="$NGROK_DOMAIN" --traffic-policy-file="$POLICY_FILE" "$port"
