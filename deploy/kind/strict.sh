#!/usr/bin/env bash
# =============================================================================
# The chart under a strict namespace (PRD-280 US-100), run by chart.yml's install lane
# on a kind cluster it has just created, with the images it has just built loaded.
#
#   deploy/kind/strict.sh prepare      datastores and the collector in their own namespace,
#                                      the chart's namespace, labelled, and its Secret
#   deploy/kind/strict.sh install      helm install with every optional component on
#   deploy/kind/strict.sh check        every pod the release makes against "restricted";
#                                      every container's root file system read-only
#   deploy/kind/strict.sh diagnostics  pods, events and logs, after a failure
#
# The chart's namespace holds the chart's pods and nothing else. It enforces the
# "baseline" level and warns and audits at "restricted": the worker starts as root until
# PRD-280 Wave 2 (US-201), so it is the one pod let run at "baseline". `check` creates
# every pod the release makes, the migration Job's included, as a server-side dry run in
# a second namespace that ENFORCES "restricted", and fails on any refusal but the worker's.
#
# Needs: kubectl and helm on a cluster kubectl already reaches, openssl, python3.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NS=automatos
STORES_NS=automatos-datastores
PROBE_NS=automatos-restricted-probe
RELEASE=automatos
# The pod the probe namespace may refuse until Wave 2, and the pods it must see.
BASELINE_POD="$RELEASE-worker"
EXPECTED_PODS=("$RELEASE-api" "$RELEASE-frontend" "$RELEASE-worker" "$RELEASE-migrate")
# The containers whose root file system must refuse a write, each with a file of the image
# that its process may write on a writable root (it owns it, or it is root), so a refusal
# can only be the read-only root's. Opened for appending, nothing written. Then those
# with a scratch /tmp.
declare -A WRITABLE_ON_A_WRITABLE_ROOT=(
    [api]=/app/main.py
    [frontend]=/app/server.js
    [worker]=/app/main.py
)
SCRATCH_TMP=(api worker)

log() { printf '\n==> %s\n' "$*"; }

label_namespace() {  # NAMESPACE ENFORCE-LEVEL
    kubectl create namespace "$1" --dry-run=client -o yaml | kubectl apply -f - >/dev/null
    kubectl label --overwrite namespace "$1" \
        "pod-security.kubernetes.io/enforce=$2" pod-security.kubernetes.io/enforce-version=latest \
        pod-security.kubernetes.io/warn=restricted pod-security.kubernetes.io/warn-version=latest \
        pod-security.kubernetes.io/audit=restricted pod-security.kubernetes.io/audit-version=latest >/dev/null
}

cmd_prepare() {
    log "Datastores and the collector in $STORES_NS (random passwords)"
    kubectl create namespace "$STORES_NS" --dry-run=client -o yaml | kubectl apply -f - >/dev/null
    local pg_password redis_password fernet
    pg_password="$(openssl rand -hex 16)"
    redis_password="$(openssl rand -hex 16)"
    fernet="$(python3 -c 'import base64, os; print(base64.urlsafe_b64encode(os.urandom(32)).decode())')"
    kubectl -n "$STORES_NS" create secret generic e2e-datastores \
        --from-literal=POSTGRES_PASSWORD="$pg_password" --from-literal=REDIS_PASSWORD="$redis_password" \
        --dry-run=client -o yaml | kubectl apply -f - >/dev/null
    kubectl -n "$STORES_NS" apply -f "$ROOT/deploy/kind/datastores.yaml" >/dev/null
    kubectl -n "$STORES_NS" create configmap otel-collector \
        --from-file=collector.yaml="$ROOT/deploy/otel/collector.yaml" \
        --from-file=otel-debug.yaml="$ROOT/deploy/kind/otel-debug.yaml" \
        --dry-run=client -o yaml | kubectl apply -f - >/dev/null
    kubectl -n "$STORES_NS" apply -f "$ROOT/deploy/kind/otel-collector.yaml" >/dev/null

    log "The chart's namespace $NS: enforces baseline, warns and audits restricted"
    label_namespace "$NS" baseline
    kubectl -n "$NS" create secret generic automatos-secrets \
        --from-literal=DATABASE_URL="postgresql://postgres:$pg_password@postgres.$STORES_NS:5432/orchestrator_db" \
        --from-literal=REDIS_URL="redis://:$redis_password@redis.$STORES_NS:6379/0" \
        --from-literal=API_KEY="$(openssl rand -hex 24)" \
        --from-literal=CREDENTIAL_ENCRYPTION_KEY="$fernet" \
        --from-literal=WORKER_INTERNAL_TOKEN="$(openssl rand -hex 24)" \
        --dry-run=client -o yaml | kubectl apply -f - >/dev/null

    # The kubelet pulls the digest-pinned datastores itself, so allow for the pull.
    kubectl -n "$STORES_NS" rollout status deploy/postgres --timeout=300s
    kubectl -n "$STORES_NS" rollout status deploy/redis --timeout=180s
    kubectl -n "$STORES_NS" rollout status deploy/otel-collector --timeout=180s
}

cmd_install() {
    log "helm install, every optional component on (the migration hook runs first)"
    helm install "$RELEASE" "$ROOT/charts/automatos" -n "$NS" \
        -f "$ROOT/deploy/kind/strict-values.yaml" --wait --timeout 20m
}

# Waits for the probe namespace's default account: the dry run's Pods run under it.
prepare_probe_namespace() {
    label_namespace "$PROBE_NS" restricted
    for _ in $(seq 1 30); do
        kubectl -n "$PROBE_NS" get serviceaccount default >/dev/null 2>&1 && return 0
        sleep 1
    done
    echo "the namespace $PROBE_NS has no default account after 30 s" >&2
    return 1
}

# One line per pod the release makes, the migration Job's (a hook) included.
release_pods() {
    { helm get manifest "$RELEASE" -n "$NS"; printf '\n---\n'; helm get hooks "$RELEASE" -n "$NS"; } \
        | kubectl create --dry-run=client -o json -f - \
        | python3 "$ROOT/scripts/ci/chart_pod_specs.py"
}

# A refused pod fails the check, except the worker's, refused by Pod Security, until Wave 2.
probe_one() {  # NAME POD-JSON
    local name="$1" answer
    if answer="$(printf '%s' "$2" | kubectl -n "$PROBE_NS" create --dry-run=server -f - 2>&1)"; then
        echo "  ✓ $name: admitted under restricted"
        [ "$name" != "$BASELINE_POD" ] || echo "::notice title=PRD-280::$name now passes restricted: remove its exception from deploy/kind/strict.sh"
        return 0
    fi
    if [ "$name" = "$BASELINE_POD" ] && grep -q 'violates PodSecurity' <<<"$answer"; then
        echo "  ~ $name: refused under restricted, allowed at baseline until PRD-280 Wave 2 (US-201): $answer"
        echo "::notice title=PRD-280 Wave 1::The worker runs at the baseline pod security level until Wave 2 (US-201). Every other pod passes restricted."
        return 0
    fi
    echo "  ✗ $name: refused under restricted: $answer"
    return 1
}

check_restricted() {
    log "Every pod the release makes, created as a dry run in $PROBE_NS (enforces restricted)"
    prepare_probe_namespace
    local failures=0 seen=() name pod expected
    while IFS=$'\t' read -r name pod; do
        seen+=("$name")
        probe_one "$name" "$pod" || failures=$((failures + 1))
    done < <(release_pods)
    for expected in "${EXPECTED_PODS[@]}"; do
        printf '%s\n' "${seen[@]}" | grep -qx "$expected" || { echo "  ✗ $expected: not in the release"; failures=$((failures + 1)); }
    done
    return "$failures"
}

check_read_only_root() {
    log "Every container's root file system refuses a write; its scratch volumes take one"
    local failures=0 component file answer
    for component in "${!WRITABLE_ON_A_WRITABLE_ROOT[@]}"; do
        file="${WRITABLE_ON_A_WRITABLE_ROOT[$component]}"
        if answer="$(kubectl -n "$NS" exec "deploy/$RELEASE-$component" -- sh -c ": >> '$file'" 2>&1)"; then
            echo "  ✗ $component: opened $file for writing"; failures=$((failures + 1))
        elif grep -qi 'read-only file system' <<<"$answer"; then
            echo "  ✓ $component: its root file system is read-only ($file)"
        else
            echo "  ✗ $component: $file refused, but not as read-only: $answer"; failures=$((failures + 1))
        fi
    done
    for component in "${SCRATCH_TMP[@]}"; do
        if kubectl -n "$NS" exec "deploy/$RELEASE-$component" -- sh -c 'touch /tmp/.scratch-probe && rm /tmp/.scratch-probe' >/dev/null 2>&1; then
            echo "  ✓ $component: /tmp takes a write"
        else
            echo "  ✗ $component: cannot write /tmp"; failures=$((failures + 1))
        fi
    done
    return "$failures"
}

cmd_check() {
    local failures=0
    check_restricted || failures=$((failures + $?))
    check_read_only_root || failures=$((failures + $?))
    [ "$failures" -eq 0 ] || { log "$failures check(s) failed"; return 1; }
    log "Every check passed"
}

cmd_diagnostics() {
    log "Diagnostics"
    kubectl get pods -A -o wide || true
    kubectl -n "$NS" get events --sort-by=.lastTimestamp | tail -n 60 || true
    kubectl -n "$NS" describe pods || true
    local component
    for component in migrate api frontend worker; do
        kubectl -n "$NS" logs "job/$RELEASE-$component" --tail=80 2>/dev/null \
            || kubectl -n "$NS" logs "deploy/$RELEASE-$component" --tail=80 2>/dev/null || true
    done
}

case "${1:-}" in
    prepare) cmd_prepare ;;
    install) cmd_install ;;
    check) cmd_check ;;
    diagnostics) cmd_diagnostics ;;
    *) sed -n '2,20p' "$0"; exit 2 ;;
esac
