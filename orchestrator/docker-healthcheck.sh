#!/bin/sh
# =============================================================================
# Automatos AI - Backend Healthcheck
# =============================================================================
# One definition of "the backend serves": GET /health answers 2xx within the
# timeout. Exit 0 when it does, non-zero when it does not (refused, timed out,
# or an error status).
#
# Two callers, so they never disagree:
#   - Docker's healthcheck (the Dockerfile HEALTHCHECK and compose's backend
#     healthcheck), which marks the container healthy or unhealthy.
#   - The entrypoint's liveness watchdog (docker-entrypoint.sh), which ends the
#     server once this fails AUTOMATOS_LIVENESS_RETRIES times in a row, so the
#     container exits and its restart policy brings it back (F416: Docker never
#     restarts a container that is only unhealthy).
#
# PORT is the server's port (Railway sets it; 8000 otherwise).
# AUTOMATOS_HEALTHCHECK_TIMEOUT caps one probe, in seconds.
# =============================================================================

HEALTHCHECK_DEFAULT_PORT=8000
HEALTHCHECK_DEFAULT_TIMEOUT_SECONDS=10

exec curl -fsS -o /dev/null \
    --max-time "${AUTOMATOS_HEALTHCHECK_TIMEOUT:-$HEALTHCHECK_DEFAULT_TIMEOUT_SECONDS}" \
    "http://localhost:${PORT:-$HEALTHCHECK_DEFAULT_PORT}/health"
