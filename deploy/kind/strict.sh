#!/usr/bin/env bash
# =============================================================================
# The chart under a strict namespace (PRD-280 US-100), run by chart.yml's install lane
# on a kind cluster it has just created, with the images it has just built loaded.
#
#   deploy/kind/strict.sh prepare      datastores and the collector in their own namespace;
#                                      the chart's namespace and the worker's, labelled, each
#                                      with its Secret; the agent-files volume's two claims
#   deploy/kind/strict.sh install      helm install with strict-values.yaml (its header
#                                      lists what is on and what is not exercised)
#   deploy/kind/strict.sh check        every pod the release makes against "restricted";
#                                      every container's root file system read-only; the
#                                      worker as uid 1000, sharing its files with the API
#   deploy/kind/strict.sh enterprise   the enterprise edition rendered with the lane's values
#                                      and strict-enterprise-values.yaml: every pod of it
#                                      against "restricted" and its read-only roots, as `check`
#                                      does; then its web pod, the enterprise web image built in
#                                      this run, started in the probe namespace: it must become
#                                      ready, send a page to sign in, and answer the sign-in page
#                                      with the media origin it was given in its policy
#   deploy/kind/strict.sh diagnostics  pods, events and logs (and each container's last
#                                      crash), after a failure
#
# `enterprise` proves no sign-in: no identity provider runs (none is faked), and the
# enterprise API is rendered and dry-run, not started.
#
# The chart's namespace holds the chart's pods but the worker's; the worker has a namespace
# of its own (PRD-280 US-202), with its own Secret. Both enforce, warn and audit at
# "restricted": every pod, the worker's included (US-201), must pass it. The operator's part
# is done here as an operator would: the namespaces and their labels, both Secrets, and two
# claims, one in each namespace, bound to one folder on the kind node. `check` also creates
# every pod the release makes, the migration Job's included (its pod is gone once the hook
# succeeds), as a server-side dry run in a third namespace that enforces "restricted", and
# fails on any refusal, naming the pod and each violation. From the same pod specs it
# requires a read-only root on every container and init container, then tries a write in
# each running one.
#
# Needs: kubectl and helm on a cluster kubectl already reaches, docker (to make the shared
# folder on the kind node, "$CLUSTER-control-plane"), openssl, python3.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NS=automatos
WORKER_NS=automatos-worker
STORES_NS=automatos-datastores
PROBE_NS=automatos-restricted-probe
RELEASE=automatos
POD_SPECS="$ROOT/scripts/ci/chart_pod_specs.py"
# The kind node, and the folder on it both claims of the agent-files volume are bound to
# (deploy/kind/strict-workspaces.yaml). Group 1000, the pods' group, may write it.
KIND_NODE="${CLUSTER:-automatos-ci}-control-plane"
SHARED_FOLDER=/var/lib/automatos-ci/workspaces
# The pods the probe namespace must see, and admit.
EXPECTED_PODS=("$RELEASE-api" "$RELEASE-frontend" "$RELEASE-worker" "$RELEASE-migrate")
# Each running component's namespace.
declare -A COMPONENT_NS=([api]="$NS" [frontend]="$NS" [worker]="$WORKER_NS")
# The running containers whose root file system must refuse a write, each with a file of the image
# that its process may write on a writable root (it owns it, or it is root), so a refusal
# can only be the read-only root's. Opened for appending, nothing written. Then those
# with a scratch /tmp.
declare -A WRITABLE_ON_A_WRITABLE_ROOT=(
    [api]=/app/main.py
    [frontend]=/app/server.js
    [worker]=/app/main.py
)
SCRATCH_TMP=(api worker)
# The worker's user (services/workspace-worker/Dockerfile's USER, the chart's runAsUser), and
# the file it writes on its volume, which the API must then read on its own claim.
WORKER_UID=1000
WORKSPACE_PROBE=/workspaces/.strict-probe
# The enterprise edition (PRD-277): its values over the lane's, the web pod started from its
# render, and the media origin those values give the web server (frontend.mediaOrigins).
ENTERPRISE_VALUES="$ROOT/deploy/kind/strict-enterprise-values.yaml"
ENTERPRISE_WEB="$RELEASE-frontend"
ENTERPRISE_MEDIA_ORIGIN=https://media.ci.invalid
ENTERPRISE_READY_TIMEOUT=180s
# "<status> <Content-Security-Policy>" of one path, asked from inside the web pod with the
# image's own node, redirects not followed (as frontend/scripts/check-enterprise-image.sh asks).
ANSWER_JS='fetch(`http://127.0.0.1:${process.env.PORT || 3000}${process.argv[1]}`, { redirect: "manual" }).then((r) => console.log(`${r.status} ${r.headers.get("content-security-policy") || ""}`), () => console.log("0"))'

log() { printf '\n==> %s\n' "$*"; }

label_namespace() {  # NAMESPACE ENFORCE-LEVEL
    kubectl create namespace "$1" --dry-run=client -o yaml | kubectl apply -f - >/dev/null &&
    kubectl label --overwrite namespace "$1" \
        "pod-security.kubernetes.io/enforce=$2" pod-security.kubernetes.io/enforce-version=latest \
        pod-security.kubernetes.io/warn=restricted pod-security.kubernetes.io/warn-version=latest \
        pod-security.kubernetes.io/audit=restricted pod-security.kubernetes.io/audit-version=latest >/dev/null
}

# A Secret from KEY=VALUE pairs, created or replaced: NAMESPACE NAME KEY=VALUE...
apply_secret() {
    local namespace="$1" name="$2" pair literals=()
    shift 2
    for pair in "$@"; do literals+=("--from-literal=$pair"); done
    kubectl -n "$namespace" create secret generic "$name" "${literals[@]}" \
        --dry-run=client -o yaml | kubectl apply -f - >/dev/null
}

prepare_datastores() {  # PG-PASSWORD REDIS-PASSWORD
    log "Datastores and the collector in $STORES_NS (random passwords)"
    kubectl create namespace "$STORES_NS" --dry-run=client -o yaml | kubectl apply -f - >/dev/null
    apply_secret "$STORES_NS" e2e-datastores "POSTGRES_PASSWORD=$1" "REDIS_PASSWORD=$2"
    kubectl -n "$STORES_NS" apply -f "$ROOT/deploy/kind/datastores.yaml" >/dev/null
    kubectl -n "$STORES_NS" create configmap otel-collector \
        --from-file=collector.yaml="$ROOT/deploy/otel/collector.yaml" \
        --from-file=otel-debug.yaml="$ROOT/deploy/kind/otel-debug.yaml" \
        --dry-run=client -o yaml | kubectl apply -f - >/dev/null
    kubectl -n "$STORES_NS" apply -f "$ROOT/deploy/kind/otel-collector.yaml" >/dev/null
}

# The two namespaces and their Secrets: the API's whole one, and the worker's with only what
# the worker reads, the internal token the same in both (PRD-280 US-202).
prepare_namespaces() {  # PG-PASSWORD REDIS-PASSWORD
    local database="postgresql://postgres:$1@postgres.$STORES_NS:5432/orchestrator_db"
    local redis="redis://:$2@redis.$STORES_NS:6379/0" token fernet
    token="$(openssl rand -hex 24)"
    fernet="$(python3 -c 'import base64, os; print(base64.urlsafe_b64encode(os.urandom(32)).decode())')"
    log "The chart's namespace $NS and the worker's $WORKER_NS: each enforces, warns and audits restricted"
    label_namespace "$NS" restricted
    label_namespace "$WORKER_NS" restricted
    apply_secret "$NS" automatos-secrets "DATABASE_URL=$database" "REDIS_URL=$redis" \
        "API_KEY=$(openssl rand -hex 24)" "CREDENTIAL_ENCRYPTION_KEY=$fernet" \
        "WORKER_INTERNAL_TOKEN=$token" "OTEL_EXPORTER_OTLP_HEADERS=x-automatos-ci=1"
    apply_secret "$WORKER_NS" automatos-worker-secrets "DATABASE_URL=$database" "REDIS_URL=$redis" \
        "WORKER_INTERNAL_TOKEN=$token" "OTEL_EXPORTER_OTLP_HEADERS=x-automatos-ci=1"
}

# The agent-files volume across the two namespaces: one folder on the node, group 1000's, and
# a claim in each namespace bound to a volume on it, as an operator binds two claims to one
# file system.
prepare_workspaces() {
    log "The agent-files volume: $SHARED_FOLDER on $KIND_NODE, a claim in $NS and one in $WORKER_NS"
    docker exec "$KIND_NODE" sh -c "mkdir -p '$SHARED_FOLDER' && chown 0:1000 '$SHARED_FOLDER' && chmod 2770 '$SHARED_FOLDER'"
    kubectl apply -f "$ROOT/deploy/kind/strict-workspaces.yaml" >/dev/null
}

cmd_prepare() {
    local pg_password redis_password
    pg_password="$(openssl rand -hex 16)"
    redis_password="$(openssl rand -hex 16)"
    prepare_datastores "$pg_password" "$redis_password"
    prepare_namespaces "$pg_password" "$redis_password"
    prepare_workspaces
    # The kubelet pulls the digest-pinned datastores itself, so allow for the pull.
    kubectl -n "$STORES_NS" rollout status deploy/postgres --timeout=300s
    kubectl -n "$STORES_NS" rollout status deploy/redis --timeout=180s
    kubectl -n "$STORES_NS" rollout status deploy/otel-collector --timeout=180s
}

cmd_install() {
    log "helm install with strict-values.yaml (the migration hook runs first)"
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

# Every object the release made, the migration Job (a hook) included, as kubectl's JSON.
release_objects() {
    { helm get manifest "$RELEASE" -n "$NS"; printf '\n---\n'; helm get hooks "$RELEASE" -n "$NS"; } \
        | kubectl create --dry-run=client -o json -f -
}

# A refused pod fails the check; the refusal's violations are listed one per line.
probe_one() {  # NAME POD-JSON
    local name="$1" answer
    if answer="$(printf '%s' "$2" | kubectl -n "$PROBE_NS" create --dry-run=server -f - 2>&1)"; then
        echo "  ✓ $name: admitted under restricted"
        return 0
    fi
    echo "  ✗ $name: refused under restricted: $answer"
    python3 "$POD_SPECS" violations <<<"$answer" | sed 's/^/      /' || true
    return 1
}

check_restricted() {  # OBJECTS-JSON
    log "Every pod the release makes, created as a dry run in $PROBE_NS (enforces restricted)"
    prepare_probe_namespace || return 1
    local failures=0 seen=() name pod expected pods
    pods="$(python3 "$POD_SPECS" <<<"$1")" || { echo "  ✗ the release's pods could not be read (above)"; return 1; }
    while IFS=$'\t' read -r name pod; do
        seen+=("$name")
        probe_one "$name" "$pod" || failures=$((failures + 1))
    done <<<"$pods"
    for expected in "${EXPECTED_PODS[@]}"; do
        printf '%s\n' "${seen[@]}" | grep -qx "$expected" || { echo "  ✗ $expected: not in the release"; failures=$((failures + 1)); }
    done
    return "$failures"
}

# From the specs: every container and init container of every pod, the migration Job's
# included (its pod is gone once the hook succeeds), sets readOnlyRootFilesystem: true.
check_read_only_spec() {  # OBJECTS-JSON
    log "Every container and init container of every pod the release makes has a read-only root (its spec)"
    local writable line
    if writable="$(python3 "$POD_SPECS" read-only-roots <<<"$1")"; then
        echo "  ✓ every one sets readOnlyRootFilesystem: true"
        return 0
    fi
    [ -n "$writable" ] || { echo "  ✗ the release's pods could not be read (above)"; return 1; }
    while read -r line; do
        echo "  ✗ $line: readOnlyRootFilesystem is not true"
    done <<<"$writable"
    return 1
}

# Runs a command in a running component, in that component's namespace: COMPONENT ARGS...
in_component() {
    local component="$1"
    shift
    kubectl -n "${COMPONENT_NS[$component]}" exec "deploy/$RELEASE-$component" -- "$@"
}

# In each running pod: the root refuses a write, and the scratch /tmp takes one.
check_read_only_root() {
    log "Every running container's root file system refuses a write; its scratch volumes take one"
    local failures=0 component file answer
    for component in "${!WRITABLE_ON_A_WRITABLE_ROOT[@]}"; do
        file="${WRITABLE_ON_A_WRITABLE_ROOT[$component]}"
        if answer="$(in_component "$component" sh -c ": >> '$file'" 2>&1)"; then
            echo "  ✗ $component: opened $file for writing"; failures=$((failures + 1))
        elif grep -qi 'read-only file system' <<<"$answer"; then
            echo "  ✓ $component: its root file system is read-only ($file)"
        else
            echo "  ✗ $component: $file refused, but not as read-only: $answer"; failures=$((failures + 1))
        fi
    done
    for component in "${SCRATCH_TMP[@]}"; do
        if in_component "$component" sh -c 'touch /tmp/.scratch-probe && rm /tmp/.scratch-probe' >/dev/null 2>&1; then
            echo "  ✓ $component: /tmp takes a write"
        else
            echo "  ✗ $component: cannot write /tmp"; failures=$((failures + 1))
        fi
    done
    return "$failures"
}

# PRD-280 US-201 and US-202: the worker runs as uid 1000 in its own namespace, writes its
# volume without having re-owned it, and the API reads what it wrote through its own claim.
check_worker_start() {
    log "The worker runs as uid 1000 in $WORKER_NS, and the API reads the files it writes"
    local failures=0 uid stamp seen
    uid="$(in_component worker id -u 2>&1)" || uid="(exec failed: $uid)"
    if [ "$uid" = "$WORKER_UID" ]; then
        echo "  ✓ worker: runs as uid $uid"
    else
        echo "  ✗ worker: runs as uid $uid, not $WORKER_UID"; failures=$((failures + 1))
    fi
    stamp="strict-$(openssl rand -hex 8)"
    if ! in_component worker sh -c "echo '$stamp' > '$WORKSPACE_PROBE'" >/dev/null 2>&1; then
        echo "  ✗ worker: cannot write /workspaces"; return $((failures + 1))
    fi
    echo "  ✓ worker: writes /workspaces"
    seen="$(in_component api cat "$WORKSPACE_PROBE" 2>&1)" || true
    if [ "$seen" = "$stamp" ]; then
        echo "  ✓ api: reads the worker's file through its own claim, in $NS"
    else
        echo "  ✗ api: reads '$seen' at $WORKSPACE_PROBE, not the worker's '$stamp'"; failures=$((failures + 1))
    fi
    in_component worker rm -f "$WORKSPACE_PROBE" >/dev/null 2>&1 || true
    return "$failures"
}

cmd_check() {
    local failures=0 objects
    # Read first, so a failure to read the release stops here with helm's or kubectl's error.
    objects="$(release_objects)" || { log "Could not read the release's objects (error above)"; return 1; }
    check_restricted "$objects" || failures=$((failures + $?))
    check_read_only_spec "$objects" || failures=$((failures + $?))
    check_read_only_root || failures=$((failures + $?))
    check_worker_start || failures=$((failures + $?))
    [ "$failures" -eq 0 ] || { log "$failures check(s) failed"; return 1; }
    log "Every check passed"
}

# The enterprise edition's objects, rendered (hooks included) with the lane's values and the
# enterprise values over them, as kubectl's JSON. Nothing is installed.
enterprise_objects() {
    helm template "$RELEASE" "$ROOT/charts/automatos" -n "$NS" \
        -f "$ROOT/deploy/kind/strict-values.yaml" -f "$ENTERPRISE_VALUES" \
        | kubectl create --dry-run=client -o json -f -
}

# Creates the enterprise web pod for real in the probe namespace, from its render's spec, and
# waits for its readiness probe (the chart's own) to pass: OBJECTS-JSON.
start_enterprise_web() {
    log "The enterprise web image built in this run, started in $PROBE_NS as the chart renders it"
    local pod
    pod="$(python3 "$POD_SPECS" <<<"$1" | awk -F '\t' -v name="$ENTERPRISE_WEB" '$1 == name { print $2 }')"
    [ -n "$pod" ] || { echo "  ✗ the enterprise render has no $ENTERPRISE_WEB"; return 1; }
    kubectl -n "$PROBE_NS" delete pod "$ENTERPRISE_WEB" --ignore-not-found --wait >/dev/null
    if ! printf '%s' "$pod" | kubectl -n "$PROBE_NS" create -f - >/dev/null; then
        echo "  ✗ $ENTERPRISE_WEB: refused (above)"; return 1
    fi
    if ! kubectl -n "$PROBE_NS" wait --for=condition=Ready "pod/$ENTERPRISE_WEB" --timeout="$ENTERPRISE_READY_TIMEOUT" >/dev/null; then
        echo "  ✗ $ENTERPRISE_WEB: not ready after $ENTERPRISE_READY_TIMEOUT"
        kubectl -n "$PROBE_NS" logs "pod/$ENTERPRISE_WEB" --tail=80 || true
        return 1
    fi
    echo "  ✓ $ENTERPRISE_WEB: admitted under restricted, started, and ready (the chart's readiness probe on /)"
}

# The running enterprise web server: a page with no session goes to sign in, and the sign-in
# page's policy names the media origin the pod was started with.
check_enterprise_web_answers() {
    local failures=0 page sign_in
    page="$(kubectl -n "$PROBE_NS" exec "$ENTERPRISE_WEB" -- node -e "$ANSWER_JS" / 2>&1)" || page="0"
    sign_in="$(kubectl -n "$PROBE_NS" exec "$ENTERPRISE_WEB" -- node -e "$ANSWER_JS" /sign-in 2>&1)" || sign_in="0"
    if [ "${page%% *}" = 307 ]; then
        echo "  ✓ $ENTERPRISE_WEB: / with no session is sent to sign in (307)"
    else
        echo "  ✗ $ENTERPRISE_WEB: / answered ${page%% *}, not 307"; failures=$((failures + 1))
    fi
    if [ "${sign_in%% *}" != 200 ]; then
        echo "  ✗ $ENTERPRISE_WEB: /sign-in answered ${sign_in%% *}, not 200"; failures=$((failures + 1))
    elif grep -qF "$ENTERPRISE_MEDIA_ORIGIN" <<<"$sign_in"; then
        echo "  ✓ $ENTERPRISE_WEB: /sign-in answers 200, its policy naming $ENTERPRISE_MEDIA_ORIGIN (MEDIA_ORIGINS)"
    else
        echo "  ✗ $ENTERPRISE_WEB: /sign-in's policy does not name $ENTERPRISE_MEDIA_ORIGIN"; failures=$((failures + 1))
    fi
    return "$failures"
}

cmd_enterprise() {
    local failures=0 objects
    log "The enterprise edition, rendered with the lane's values and $(basename "$ENTERPRISE_VALUES")"
    objects="$(enterprise_objects)" || { log "Could not render the enterprise edition (error above)"; return 1; }
    check_restricted "$objects" || failures=$((failures + $?))
    check_read_only_spec "$objects" || failures=$((failures + $?))
    if start_enterprise_web "$objects"; then
        check_enterprise_web_answers || failures=$((failures + $?))
    else
        failures=$((failures + 1))
    fi
    [ "$failures" -eq 0 ] || { log "$failures enterprise check(s) failed"; return 1; }
    log "Every enterprise check passed (no sign-in: no identity provider runs here)"
}

diagnose_namespace() {  # NAMESPACE COMPONENT...
    local namespace="$1" component target
    shift
    kubectl -n "$namespace" get events --sort-by=.lastTimestamp | tail -n 60 || true
    kubectl -n "$namespace" describe pods || true
    kubectl -n "$namespace" get pvc || true
    for component in "$@"; do
        for target in "job/$RELEASE-$component" "deploy/$RELEASE-$component"; do
            kubectl -n "$namespace" get "$target" >/dev/null 2>&1 || continue
            log "Logs: $namespace/$target"
            kubectl -n "$namespace" logs "$target" --all-containers --tail=80 || true
            log "Logs: $namespace/$target, the run before its last restart (a crash loop's last crash)"
            kubectl -n "$namespace" logs "$target" --all-containers --previous --tail=80 || true
        done
    done
}

cmd_diagnostics() {
    log "Diagnostics"
    kubectl get pods -A -o wide || true
    kubectl get pv || true
    diagnose_namespace "$NS" migrate api frontend
    diagnose_namespace "$WORKER_NS" worker
    if kubectl -n "$PROBE_NS" get pod "$ENTERPRISE_WEB" >/dev/null 2>&1; then
        log "The enterprise web pod in $PROBE_NS"
        kubectl -n "$PROBE_NS" describe pod "$ENTERPRISE_WEB" || true
        kubectl -n "$PROBE_NS" logs "pod/$ENTERPRISE_WEB" --tail=80 || true
    fi
}

case "${1:-}" in
    prepare) cmd_prepare ;;
    install) cmd_install ;;
    check) cmd_check ;;
    enterprise) cmd_enterprise ;;
    diagnostics) cmd_diagnostics ;;
    *) sed -n '2,41p' "$0"; exit 2 ;;
esac
