# shellcheck shell=bash
# =============================================================================
# OpenTelemetry on the kind cluster (PRD-256 O6, #847), sourced by e2e.sh.
#
# A collector runs the reference config (deploy/otel/collector.yaml) in the
# namespace, with its exporter swapped for the debug exporter
# (deploy/kind/otel-debug.yaml), and the chart points the API and the worker at it
# (deploy/kind/values.yaml: otel.*). The checks send one request that crosses API ->
# worker, then read the collector's log: spans from both services, in ONE trace,
# carrying the pod name the chart adds, and never a query value from the URL.
# =============================================================================

OTEL_REQUEST_ID=e2e-otel-probe

install_otel_collector() {
    log "Starting the OpenTelemetry Collector (the reference config, debug exporter)"
    kubectl -n "$NS" create configmap otel-collector \
        --from-file=collector.yaml="$ROOT/deploy/otel/collector.yaml" \
        --from-file=otel-debug.yaml="$ROOT/deploy/kind/otel-debug.yaml" \
        --dry-run=client -o yaml | kubectl apply -f - >/dev/null
    kubectl -n "$NS" apply -f "$ROOT/deploy/kind/otel-collector.yaml" >/dev/null
    kubectl -n "$NS" rollout status deploy/otel-collector --timeout=120s
}

# The collector's log since it started (the debug exporter's output).
collector_log() { kubectl -n "$NS" logs deploy/otel-collector --tail=-1; }
collector_log_has() { collector_log | grep -qF "$1"; }
collector_log_lacks() { ! collector_log | grep -qF "$1"; }

# Succeeds when one trace ID in the collector's log has spans from both services,
# and the API's resource carries the pod name the chart adds.
one_trace_across_api_and_worker() {
    collector_log | python3 -c '
import re, sys
traces, service, pod_named = {}, None, False
for line in sys.stdin:
    found = re.search(r"service\.name: Str\(([^)]+)\)", line)
    if found:
        service = found.group(1)
    if service == "automatos-api" and "k8s.pod.name: Str(" in line:
        pod_named = True
    found = re.search(r"Trace ID\s*:\s*([0-9a-f]{32})", line)
    if found and service:
        traces.setdefault(service, set()).add(found.group(1))
shared = traces.get("automatos-api", set()) & traces.get("automatos-workspace-worker", set())
sys.exit(0 if shared and pod_named else 1)
'
}

run_otel_checks() {
    log "OpenTelemetry checks"
    # Lists the workspace's files: the API calls the worker, which continues the trace.
    curl -s -o /dev/null -H "X-Request-ID: $OTEL_REQUEST_ID" \
        "http://127.0.0.1:$API_PORT/api/workspaces/$WORKSPACE_ID/files?path=.&probe=SECRET-QUERY-VALUE" || true
    sleep 8   # the SDK's batch delay (5 s), then the collector's batch
    check "the collector is up (its health check answers in the cluster)" \
        kubectl -n "$NS" exec deploy/"$RELEASE"-worker -- curl -sf http://otel-collector:13133/
    check "the API's spans reach the collector" collector_log_has "service.name: Str(automatos-api)"
    check "the worker's spans reach the collector" collector_log_has "service.name: Str(automatos-workspace-worker)"
    check "one request is one trace across the API and the worker, named by pod" one_trace_across_api_and_worker
    check "the request's own ID is on its span" collector_log_has "automatos.request_id: Str($OTEL_REQUEST_ID)"
    check "a query value never reaches the collector" collector_log_lacks "SECRET-QUERY-VALUE"
}
