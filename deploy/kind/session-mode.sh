# shellcheck shell=bash
# =============================================================================
# Session mode on the kind cluster (#848), sourced by e2e.sh.
#
# The operator's CLI host runs HERE, on this machine, against the cluster's API,
# with the repo's fake `claude` (services/cli-host/tests/fake_claude.py): no real
# Claude login, no spend. The host pairs, claims a session ticket, runs it, and
# uploads what the session left in the ticket's deliverables folder; the checks
# read the file on the worker's volume and the Deliverable the API registered.
#
# The host gets its own HOME, state folder and deliverables root under a temp
# folder, so the operator's ~/.claude.json and paired host are never touched.
#
# E2E_INGRESS=1 also installs ingress-nginx and sends the host through the
# chart's Ingress, then proves the session-mode Ingress takes a request body
# past ingress-nginx's 1 MB default: a 2 MB upload reaches the API (409, the
# ticket is finished) instead of being refused by the controller (413).
# =============================================================================

INGRESS_NAMESPACE=ingress-nginx
INGRESS_NGINX_CHART_VERSION=4.15.1
INGRESS_PORT=18080
SESSION_HOST_TIMEOUT_SECONDS=180
SESSION_NOTE_TEXT="written in the session folder"
# Past ingress-nginx's 1 MB default request-body limit, under the 50 MB file limit.
INGRESS_PROBE_BYTES=$((2 * 1024 * 1024))

ingress_on() { [ "${E2E_INGRESS:-}" = "1" ]; }

install_ingress_controller() {
    ingress_on || return 0
    log "Installing ingress-nginx $INGRESS_NGINX_CHART_VERSION"
    helm upgrade --install ingress-nginx ingress-nginx \
        --repo https://kubernetes.github.io/ingress-nginx --version "$INGRESS_NGINX_CHART_VERSION" \
        -n "$INGRESS_NAMESPACE" --create-namespace \
        --set controller.service.type=ClusterIP \
        --set controller.admissionWebhooks.enabled=false \
        --set controller.ingressClassResource.name=nginx \
        --wait --timeout 5m
}

# Sets SESSION_API_URL, the address the host dials: the ingress controller, or the API
# service. Call it in this shell, never as $(...): the port-forward's PID must reach
# FORWARDS here, or cleanup_forwards leaves it running into the next pass.
start_session_api() {
    SESSION_API_URL="http://127.0.0.1:$API_PORT"
    ingress_on || return 0
    kubectl -n "$INGRESS_NAMESPACE" port-forward svc/ingress-nginx-controller "$INGRESS_PORT:80" >/dev/null 2>&1 &
    FORWARDS+=("$!")
    for _ in $(seq 1 30); do
        curl -s -o /dev/null "http://127.0.0.1:$INGRESS_PORT/health" && break
        sleep 1
    done
    SESSION_API_URL="http://127.0.0.1:$INGRESS_PORT"
}

# The agent through the API, the way the app creates one; the ticket straight into the board.
seed_session_ticket() {
    local suffix agent
    suffix="$(openssl rand -hex 4)"
    agent="$(curl -sf -X POST -H 'Content-Type: application/json' \
        -d "{\"name\": \"e2e-session-$suffix\", \"agent_type\": \"custom\",
             \"configuration\": {\"runtime\": \"cli\", \"provider\": \"claude\"}}" \
        "http://127.0.0.1:$API_PORT/api/agents/" \
        | python3 -c 'import json, sys; print(json.load(sys.stdin)["id"])')"
    sql "INSERT INTO board_tasks (workspace_id, title, status, assigned_agent_id, priority, source_type, attempts)
         VALUES ('$WORKSPACE_ID', 'e2e session $suffix', 'assigned', $agent, 'medium', 'user', 0)
         RETURNING id" | head -1
}

pairing_code() {
    curl -sf -X POST -H 'Content-Type: application/json' -d '{"name": "e2e-host"}' \
        "http://127.0.0.1:$API_PORT/api/v1/cli-hosts/pairing-codes" \
        | python3 -c 'import json, sys; print(json.load(sys.stdin)["code"])'
}

# One claim cycle of the real host, then it exits: pairs first when given a code.
run_session_host() {
    local url="$1" code="$2" tmp="$3" pair=()
    [ -n "$code" ] && pair=(--pair "$code")
    mkdir -p "$tmp/home" "$tmp/ws"
    printf '%s\n' '{"hasCompletedOnboarding": true, "projects": {}}' > "$tmp/home/.claude.json"
    cp "$ROOT/services/cli-host/tests/fake_claude.py" "$tmp/claude"
    chmod +x "$tmp/claude"
    (
        cd "$ROOT/services/cli-host"
        env -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN -u ANTHROPIC_BASE_URL -u CLAUDE_CONFIG_DIR \
            -u CLAUDECODE -u CLAUDE_CODE_ENTRYPOINT -u CLAUDE_CODE_CHILD_SESSION \
            HOME="$tmp/home" FAKE_CLAUDE_SESSION_NOTE=1 \
            python3 -m automatos_cli_host --url "$url" --dir "$tmp/state" ${pair[@]+"${pair[@]}"} --name e2e-host \
                --allow "$tmp/ws" --default-root "$tmp/ws" --cli-binary "claude=$tmp/claude" \
                --once --no-worktrees --no-terminal --no-session-sandbox \
                --session-timeout "$SESSION_HOST_TIMEOUT_SECONDS"
    ) >> "$tmp/host.log" 2>&1
}

# The workspace's default policy holds a new ticket for the operator ('always_ask'):
# the first claim parks it on a grant. Approve it the way the Command Center does.
ticket_waits_for_approval() {
    [ "$(sql "SELECT status FROM board_tasks WHERE id = $1")" = "blocked" ]
}

approve_ticket() {
    local grant
    grant="$(sql "SELECT id FROM approval_grants WHERE subject_type = 'board_task' AND subject_id = '$1'
                  AND status = 'pending' ORDER BY id DESC LIMIT 1")"
    [ -n "$grant" ] && curl -sf -o /dev/null -X POST "http://127.0.0.1:$API_PORT/api/v1/approval-grants/$grant/grant"
}

ticket_finished() {
    case "$(sql "SELECT status FROM board_tasks WHERE id = $1")" in
        done|review) return 0 ;;
        *) return 1 ;;
    esac
}

file_on_worker_volume() {
    kubectl -n "$NS" exec deploy/"$RELEASE"-worker -- \
        cat "/workspaces/$WORKSPACE_ID/sessions/$1/note.md" | grep -q "$SESSION_NOTE_TEXT"
}

deliverable_registered() {
    sql_positive "SELECT count(*) FROM deliverables
                   WHERE workspace_id = '$WORKSPACE_ID' AND file_path = 'sessions/$1/note.md'
                     AND deleted_at IS NULL"
}

# A 2 MB upload through the session-mode Ingress, with the paired host's token, to
# the ticket that just finished: the API answers 409; ingress-nginx would say 413.
large_upload_reaches_the_api() {
    local url="$1" tmp="$2" task="$3" host_id token status
    host_id="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["host_id"])' "$tmp/state/host.json")"
    token="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["token"])' "$tmp/state/host.json")"
    head -c "$INGRESS_PROBE_BYTES" /dev/zero > "$tmp/probe.bin"
    status="$(curl -s -o /dev/null -w '%{http_code}' -X PUT --data-binary @"$tmp/probe.bin" \
        -H 'Content-Type: application/octet-stream' -H "X-CLI-Host-Token: $token" \
        "$url/api/v1/cli-hosts/$host_id/tasks/$task/files?path=probe.bin")"
    [ "$status" = "409" ]
}

run_session_mode_checks() {
    log "Session mode (#848)"
    local tmp url code task
    tmp="$(mktemp -d /tmp/ae2e.XXXXXX)"   # short: the host's hook socket path is capped
    start_session_api
    url="$SESSION_API_URL"
    check "the API has session mode on" body_has "http://127.0.0.1:$API_PORT/health" '"cli_runtime_enabled": *true'
    task="$(seed_session_ticket)"
    code="$(pairing_code)"
    check "the CLI host on this machine pairs and claims ticket #$task" run_session_host "$url" "$code" "$tmp"
    check "ticket #$task waits for the operator's approval" ticket_waits_for_approval "$task"
    check "the operator approves ticket #$task" approve_ticket "$task"
    check "the host runs ticket #$task, uploads its file and reports" run_session_host "$url" "" "$tmp"
    check "ticket #$task finished" ticket_finished "$task"
    check "the session's file was uploaded into the workspace volume" file_on_worker_volume "$task"
    check "the file is the ticket's Deliverable" deliverable_registered "$task"
    if ingress_on; then
        check "a 2 MB upload passes the session-mode Ingress" large_upload_reaches_the_api "$url" "$tmp" "$task"
    fi
    [ "$FAILURES" -eq 0 ] || sed -n '1,200p' "$tmp/host.log"
    rm -rf "$tmp"
}
