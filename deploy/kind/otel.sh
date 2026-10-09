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

OTEL_REQUEST_ID=""   # a fresh one each round (run_otel_checks): the log keeps earlier rounds
OTEL_PROBE_STATUS=""

install_otel_collector() {
    log "Starting the OpenTelemetry Collector (the reference config, debug exporter)"
    kubectl -n "$NS" create configmap otel-collector \
        --from-file=collector.yaml="$ROOT/deploy/otel/collector.yaml" \
        --from-file=otel-debug.yaml="$ROOT/deploy/kind/otel-debug.yaml" \
        --dry-run=client -o yaml | kubectl apply -f - >/dev/null
    kubectl -n "$NS" apply -f "$ROOT/deploy/kind/otel-collector.yaml" >/dev/null
    kubectl -n "$NS" rollout status deploy/otel-collector --timeout=120s
}

# The collector's log since it started (the debug exporter's output). Read whole,
# then searched: `kubectl logs | grep -q` stops reading at the first match, and
# under e2e.sh's pipefail kubectl's broken pipe would fail the check.
collector_log() { kubectl -n "$NS" logs deploy/otel-collector --tail=-1; }
collector_log_has() { local log; log="$(collector_log)" && grep -qF "$1" <<<"$log"; }
collector_log_lacks() { local log; log="$(collector_log)" && ! grep -qF "$1" <<<"$log"; }

# Succeeds when the span carrying the probe's request ID ($1) is in a trace that
# has spans from both services, each from a resource carrying the pod name the
# chart adds. Tied to this probe, so an earlier round's trace never passes it.
probe_is_one_trace_across_api_and_worker() {
    collector_log | python3 -c '
import re, sys
probe, services = sys.argv[1], {"automatos-api", "automatos-workspace-worker"}
spans, probe_traces = set(), set()   # (service, trace ID, resource named by pod)
service, pod_named, trace = None, False, None
for line in sys.stdin:
    if line.lstrip().startswith("ResourceSpans #"):
        service, pod_named = None, False
    found = re.search(r"service\.name: Str\(([^)]+)\)", line)
    if found:
        service = found.group(1)
    if "k8s.pod.name: Str(" in line:
        pod_named = True
    found = re.search(r"Trace ID\s*:\s*([0-9a-f]{32})", line)
    if found:
        trace = found.group(1)
        spans.add((service, trace, pod_named))
    if trace and f"automatos.request_id: Str({probe})" in line:
        probe_traces.add(trace)
named = {(svc, trace) for svc, trace, pod in spans if pod}
sys.exit(0 if any(all((svc, trace) in named for svc in services) for trace in probe_traces) else 1)
' "$1"
}

run_otel_checks() {
    log "OpenTelemetry checks"
    OTEL_REQUEST_ID="e2e-otel-$(openssl rand -hex 4)"
    # Lists the workspace's files: the API calls the worker, which continues the trace.
    OTEL_PROBE_STATUS="$(curl -s -o /dev/null -w '%{http_code}' -H "X-Request-ID: $OTEL_REQUEST_ID" \
        "http://127.0.0.1:$API_PORT/api/workspaces/$WORKSPACE_ID/files?path=.&probe=SECRET-QUERY-VALUE" || true)"
    # Both services export in batches (the SDK's 5 s, then the collector's): wait
    # for this request's whole trace, not a fixed time (review on #1053).
    for _ in $(seq 1 60); do
        probe_is_one_trace_across_api_and_worker "$OTEL_REQUEST_ID" && break
        sleep 1
    done
    check "the probe request (API -> worker) is answered" test "$OTEL_PROBE_STATUS" = 200
    check "the collector is up (its health check answers in the cluster)" \
        kubectl -n "$NS" exec deploy/"$RELEASE"-worker -- curl -sf http://otel-collector:13133/
    check "the API's spans reach the collector" collector_log_has "service.name: Str(automatos-api)"
    check "the worker's spans reach the collector" collector_log_has "service.name: Str(automatos-workspace-worker)"
    check "the probe is one trace across the API and the worker, named by pod" \
        probe_is_one_trace_across_api_and_worker "$OTEL_REQUEST_ID"
    check "a query value never reaches the collector" collector_log_lacks "SECRET-QUERY-VALUE"
}
