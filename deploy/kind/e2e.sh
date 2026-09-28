#!/usr/bin/env bash
# =============================================================================
# End-to-end test of charts/automatos on a throwaway kind cluster.
#
#   deploy/kind/e2e.sh cycle [N]   create -> install -> test -> upgrade -> test
#                                  -> delete, N times (default 1)
#   deploy/kind/e2e.sh up          create the cluster and install (keeps it)
#   deploy/kind/e2e.sh test        run the checks against a running install
#   deploy/kind/e2e.sh down        delete the cluster
#
# The API image is built from this checkout (so it carries the entrypoint and
# seed-loader changes under test); the web app and worker are the published
# images. Everything runs in its own kind cluster with its own kubeconfig, so
# no other cluster or context is touched.
#
# Needs: docker, kind, kubectl, helm, curl, openssl, python3.
# Env:   E2E_SKIP_BUILD=1  reuse an existing automatos-api:e2e image
#        E2E_IMAGE_TAG     published frontend/worker tag (default: edge)
# Not for CI merge gates: it needs Docker and several GB of images.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CLUSTER=automatos-e2e
NS=automatos
RELEASE=automatos
KUBECONFIG="${TMPDIR:-/tmp}/automatos-e2e.kubeconfig"
export KUBECONFIG
TAG="${E2E_IMAGE_TAG:-edge}"
API_IMAGE=automatos-api:e2e
FRONTEND_IMAGE="ghcr.io/automatosai/automatos-frontend:$TAG"
WORKER_IMAGE="ghcr.io/automatosai/automatos-workspace-worker:$TAG"
DATASTORE_IMAGES=(pgvector/pgvector:pg16 redis:7-alpine)
WORKSPACE_ID=00000000-0000-0000-0000-0000000000c1
API_PORT=18000
FRONTEND_PORT=13000
FAILURES=0
FORWARDS=()

log() { printf '\n==> %s\n' "$*"; }

cleanup_forwards() {
    for pid in "${FORWARDS[@]:-}"; do
        [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
    done
    FORWARDS=()
}
trap cleanup_forwards EXIT

check() {
    local description="$1"; shift
    if "$@" >/dev/null 2>&1; then
        printf '  ✓ %s\n' "$description"
    else
        printf '  ✗ %s\n' "$description"
        FAILURES=$((FAILURES + 1))
    fi
}

create_cluster() {
    if kind get clusters 2>/dev/null | grep -qx "$CLUSTER"; then
        log "Cluster $CLUSTER already exists; reusing it"
        kind export kubeconfig --name "$CLUSTER" --kubeconfig "$KUBECONFIG" >/dev/null
    else
        log "Creating kind cluster $CLUSTER"
        kind create cluster --name "$CLUSTER" --config "$ROOT/deploy/kind/cluster.yaml" --kubeconfig "$KUBECONFIG" --wait 120s
    fi
}

prepare_images() {
    if [ "${E2E_SKIP_BUILD:-}" != "1" ] || ! docker image inspect "$API_IMAGE" >/dev/null 2>&1; then
        log "Building $API_IMAGE from this checkout"
        docker build --target production --build-arg INSTALL_GRAPH_EXTRAS=false -t "$API_IMAGE" "$ROOT/orchestrator"
    fi
    for image in "$FRONTEND_IMAGE" "$WORKER_IMAGE" "${DATASTORE_IMAGES[@]}"; do
        docker image inspect "$image" >/dev/null 2>&1 || docker pull -q "$image"
    done
    log "Loading images into the cluster"
    # `kind load docker-image` imports every platform an image index lists, which
    # fails under Docker's containerd image store when attestations or other
    # architectures aren't stored locally. Save just the node's platform instead.
    local arch archive
    arch="$(docker exec "$CLUSTER-control-plane" uname -m | sed 's/aarch64/arm64/;s/x86_64/amd64/')"
    archive="$(mktemp "${TMPDIR:-/tmp}/automatos-e2e-image.XXXXXX")"
    for image in "$API_IMAGE" "$FRONTEND_IMAGE" "$WORKER_IMAGE" "${DATASTORE_IMAGES[@]}"; do
        docker save --platform "linux/$arch" "$image" -o "$archive"
        kind load image-archive "$archive" --name "$CLUSTER" >/dev/null
    done
    rm -f "$archive"
}

create_secrets() {
    log "Creating dev datastores and the chart's Secret (random values)"
    kubectl create namespace "$NS" --dry-run=client -o yaml | kubectl apply -f - >/dev/null
    local pg_password redis_password fernet
    pg_password="$(openssl rand -hex 16)"
    redis_password="$(openssl rand -hex 16)"
    fernet="$(python3 -c 'import base64, os; print(base64.urlsafe_b64encode(os.urandom(32)).decode())')"
    kubectl -n "$NS" create secret generic e2e-datastores \
        --from-literal=POSTGRES_PASSWORD="$pg_password" \
        --from-literal=REDIS_PASSWORD="$redis_password" \
        --dry-run=client -o yaml | kubectl apply -f - >/dev/null
    kubectl -n "$NS" create secret generic automatos-secrets \
        --from-literal=DATABASE_URL="postgresql://postgres:$pg_password@postgres:5432/orchestrator_db" \
        --from-literal=REDIS_URL="redis://:$redis_password@redis:6379/0" \
        --from-literal=API_KEY="$(openssl rand -hex 24)" \
        --from-literal=CREDENTIAL_ENCRYPTION_KEY="$fernet" \
        --from-literal=WORKER_INTERNAL_TOKEN="$(openssl rand -hex 24)" \
        --dry-run=client -o yaml | kubectl apply -f - >/dev/null
    kubectl -n "$NS" apply -f "$ROOT/deploy/kind/datastores.yaml" >/dev/null
    kubectl -n "$NS" rollout status deploy/postgres --timeout=180s
    kubectl -n "$NS" rollout status deploy/redis --timeout=120s
}

helm_release() {
    local action="$1"
    log "helm $action (the migration hook runs first)"
    if ! helm "$action" "$RELEASE" "$ROOT/charts/automatos" -n "$NS" \
        -f "$ROOT/deploy/kind/values.yaml" \
        --set frontend.image.tag="$TAG" --set worker.image.tag="$TAG" \
        --wait --timeout 20m; then
        dump_diagnostics
        return 1
    fi
}

dump_diagnostics() {
    log "Diagnostics"
    kubectl -n "$NS" get pods,jobs -o wide || true
    kubectl -n "$NS" logs job/"$RELEASE"-migrate --tail=80 2>/dev/null || true
    kubectl -n "$NS" logs deploy/"$RELEASE"-api --tail=80 2>/dev/null || true
}

forward() {
    local service="$1" local_port="$2" remote_port="$3"
    kubectl -n "$NS" port-forward "svc/$service" "$local_port:$remote_port" >/dev/null 2>&1 &
    FORWARDS+=("$!")
    for _ in $(seq 1 30); do
        curl -s -o /dev/null "http://127.0.0.1:$local_port/" && return 0
        sleep 1
    done
    return 1
}

http_ok() { [ "$(curl -s -o /dev/null -w '%{http_code}' "$1")" = "200" ]; }
# Follows redirects: the web app's / redirects to /chat.
page_ok() { [ "$(curl -sL -o /dev/null -w '%{http_code}' "$1")" = "200" ]; }
body_has() { curl -sf "$1" | grep -q "$2"; }
sql() { kubectl -n "$NS" exec deploy/postgres -- psql -U postgres -d orchestrator_db -tAc "$1"; }
sql_positive() { [ "$(sql "$1")" -gt 0 ]; }

run_checks() {
    log "Checks"
    check "the release is deployed" sh -c "helm status $RELEASE -n $NS | grep -q 'STATUS: deployed'"
    check "the database is at an Alembic head" sql_positive "SELECT count(*) FROM alembic_version"
    check "credential types were seeded (through DATABASE_URL)" sql_positive "SELECT count(*) FROM credential_types"
    check "the local workspace exists" sql_positive "SELECT count(*) FROM workspaces WHERE id = '$WORKSPACE_ID'"
    check "the local operator exists" sql_positive "SELECT count(*) FROM users WHERE id = 1"

    forward "$RELEASE-api" "$API_PORT" 8000 || true
    check "API /health is 200" http_ok "http://127.0.0.1:$API_PORT/health"
    check "API /health/ready is 200" http_ok "http://127.0.0.1:$API_PORT/health/ready"
    check "API serves the local workspace anonymously" body_has "http://127.0.0.1:$API_PORT/api/workspaces/current" "$WORKSPACE_ID"
    check "API lists the seeded credential types" http_ok "http://127.0.0.1:$API_PORT/api/credentials/types"
    check "API lists agents" http_ok "http://127.0.0.1:$API_PORT/api/agents/"

    forward "$RELEASE-frontend" "$FRONTEND_PORT" 3000 || true
    check "the web app serves its home page (/ redirects to /chat)" page_ok "http://127.0.0.1:$FRONTEND_PORT/"

    check "the worker is healthy" kubectl -n "$NS" exec deploy/"$RELEASE"-worker -- curl -sf http://127.0.0.1:8081/health
    check "a file the worker writes is visible to the API" sh -c \
        "kubectl -n $NS exec deploy/$RELEASE-worker -- sh -c 'echo e2e > /workspaces/.e2e-probe' \
         && kubectl -n $NS exec deploy/$RELEASE-api -- cat /workspaces/.e2e-probe | grep -q e2e"
    check "the API cannot write the workspaces (read-only mount)" sh -c \
        "! kubectl -n $NS exec deploy/$RELEASE-api -- sh -c 'echo x > /workspaces/.api-probe' 2>/dev/null"
    check "API pods did not run migrations" sh -c \
        "! kubectl -n $NS logs deploy/$RELEASE-api | grep -q 'alembic.runtime.migration'"
    cleanup_forwards
}

cmd_up() {
    create_cluster
    prepare_images
    create_secrets
    helm_release install
}

cmd_test() {
    run_checks
    [ "$FAILURES" -eq 0 ]
}

cmd_down() {
    log "Deleting kind cluster $CLUSTER"
    kind delete cluster --name "$CLUSTER" --kubeconfig "$KUBECONFIG"
    rm -f "$KUBECONFIG"
}

cmd_cycle() {
    local runs="${1:-1}" run
    for run in $(seq 1 "$runs"); do
        log "Cycle $run of $runs"
        FAILURES=0
        cmd_up
        run_checks
        helm_release upgrade
        run_checks
        cmd_down
        if [ "$FAILURES" -ne 0 ]; then
            log "Cycle $run: $FAILURES check(s) failed"
            return 1
        fi
        log "Cycle $run: all checks passed"
    done
}

case "${1:-}" in
    cycle) shift; cmd_cycle "$@" ;;
    up) cmd_up ;;
    test) cmd_test ;;
    down) cmd_down ;;
    *) sed -n '2,20p' "$0"; exit 2 ;;
esac
