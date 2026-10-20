#!/bin/bash
# Acceptance gate — PRD-266 W3 Store duties, the Automatos Go backend (10 stories: US-016a … US-018d).
#   bash scripts/ralph/acceptance-prd266w3.sh | --print-base | --no-ci | --negative-only ; GATE_BASE=<ref> (smoke runs only).
# Base: the merge-base with origin/main (the branch is cut from origin/main, which carries the PRD since #1257, and
# the wave's PR targets main). Nothing runs on the owner's machine: greps, git and stdlib python only; CI once per
# wave on the draft PR. No pipefail (a negative `! producer | grep -q` would invert on SIGPIPE); every check runs in
# its own `( eval … )`; stdin is closed so a grep with a missing file argument fails instead of hanging.
#
# Negative checks and the base (20 Oct 2026, origin/main 7ae677b52):
#   - every "added lines" check reads `git diff BASE..HEAD`, EMPTY when HEAD is the base, so it passes there by
#     construction; each is scoped to code dirs and excludes tests, the kit (scripts/ralph) and docs, so a check's own
#     pattern line or a test asserting absence never trips it;
#   - the whole Scope block was smoke-run on the base (GATE_BASE=7ae677b52 --negative-only, from a throwaway shared
#     clone at 7ae677b52): green; every story check was then proven both ways on throwaway commits in that clone
#     (green on stand-in files, red on a decrypting resolver, a wide public path, a removed bell entry, a fetch in
#     /go/pair), and comment-only mentions (docstrings, // lines) were shown not to trip the code checks;
#   - the story checks are POSITIVE (the wave's work): they fail on the base by design (no services/content_reports,
#     no services/ai_providers.py, no /api/v1/reports, no app/.well-known, no app/go/pair there; read 20 Oct).
set -u
exec </dev/null
cd "$(dirname "$0")/../.." || exit 1
export DATABASE_URL="postgresql://ralph:ralph@127.0.0.1:1/ralph_no_db"
export POSTGRES_HOST=127.0.0.1 POSTGRES_PORT=1 POSTGRES_USER=ralph POSTGRES_PASSWORD=ralph POSTGRES_DB=ralph_no_db
export REDIS_URL="redis://127.0.0.1:1/0" REDIS_HOST=127.0.0.1 REDIS_PORT=1
export ENVIRONMENT=development
# Homebrew's python (3.14) when present: CI's code-shape script needs 3.10+ (ast.Match); macOS's /usr/bin is 3.9.
PY=$(command -v /opt/homebrew/bin/python3 || command -v python3)

if [ -n "${GATE_BASE:-}" ]; then
  BASE=$(git rev-parse --verify -q "$GATE_BASE^{commit}") || { echo "GATE_BASE $GATE_BASE does not resolve"; exit 1; }
else
  BASE=$(git merge-base HEAD origin/main) || { echo "cannot resolve merge-base with origin/main"; exit 1; }
fi
MODE=full
case "${1:-}" in --print-base) echo "$BASE"; exit 0 ;; --no-ci) MODE=no-ci ;; --negative-only) MODE=negative ;; "") ;; *) echo "unknown argument: $1"; exit 2 ;; esac
echo "base: $(git log -1 --format='%h %s' "$BASE") · HEAD: $(git log -1 --format='%h %s' HEAD) · mode: $MODE · python: $("$PY" -V 2>&1)"
FAIL=0
check() { local name="$1" body="$2"; echo ""; echo "── $name"; if ( eval "$body" ); then echo "   ✅ PASS: $name"; else echo "   ❌ FAIL: $name"; FAIL=1; fi; }
warn() { local name="$1" body="$2"; echo ""; echo "── $name"; if ( eval "$body" ); then echo "   ✅ ok: $name"; else echo "   ⚠️  WARN (not a failure): $name"; fi; }

KIT=scripts/ralph/prd-266w3.json
PRD_DOC=docs/PRDS/PRD-266-AUTOMATOS-GO-BACKEND.md
O=orchestrator
TESTS=$O/tests
VERSIONS=$O/alembic/versions
MANIFEST=$O/reports/route-manifest.json
SURFACE=$O/reports/config-surface.json
CR=$O/services/content_reports
CR_ROUTER=$O/api/content_reports.py
DISPATCHER=$O/core/services/notification_dispatcher.py
PROVIDERS=$O/services/ai_providers.py
TASK_ITEMS=$O/services/activity_scheduled_task_items.py
ENV_DOC=docs/deployment-infrastructure/environment-variables.md
F=frontend
BELL=$F/components/notifications/notification-bell.tsx
BELL_GUARD=$F/components/notifications/__tests__/notification-bell-links.test.ts
AUDIT_PANE=$F/components/command-center/governance/audit-pane.tsx
CC=$F/components/command-center
AASA=$F/app/.well-known/apple-app-site-association/route.ts
ASSETLINKS=$F/app/.well-known/assetlinks.json/route.ts
GO_PAIR=$F/app/go/pair/page.tsx
GO_PAIR_DIR=$F/components/go-pair
MIDDLEWARE=$F/middleware.ts
SETTINGS="MODERATION_WEBHOOK_URL"
IOS_APP_ID="KGVA9933BG.com.automatos.app"
# §4.10: files over 800 lines that may not grow (the two this wave touches, and the five it never touches).
NO_GROW="$O/api/workspaces.py $O/services/activity_service.py $O/main.py $O/core/auth/hybrid.py $O/api/chat.py $O/consumers/chatbot/service.py $F/lib/api-client.ts"
# Over 800 lines and allowed to grow: settings live only in config.py (AGENTS.md); the two reports are generated lists.
GROW_OK="$O/config.py $MANIFEST $SURFACE"
# K5: W1's files (built in parallel on feat/prd-266-w1-connect) and W1's chat and voice files: never created or edited here.
W1_PATHS="$O/services/mobile $O/api/mobile_devices.py $O/api/well_known.py $O/core/auth/pairing.py $O/core/auth/bearer_registry.py $O/core/auth/credentials.py $O/core/auth/device.py $O/core/models/mobile_devices.py $O/core/security/rate_limiter.py $O/core/security/ip_bucket.py $O/api/chat.py $O/consumers/chatbot/service.py $O/api/voice_retell.py $O/modules/voice $F/components/settings/phones-devices $F/components/local-auth-provider.tsx"
# PRD-261's theme contract (HOST_OWNED_ROUTES, TS + Python with a parity test) and the Next.js config: untouched (K14, K15).
UNTOUCHED_FRONT="$F/lib/theme-host/package-manifest.ts $O/services/themes/contract.py $F/next.config.js"

# The wave's ADDED lines under the given paths: tests, the kit and docs excluded (negative greps only).
added_code() { git diff "$BASE"..HEAD -- "$@" ':!orchestrator/tests' ':!*__tests__*' ':!*.test.ts' ':!*.test.tsx' ':!scripts/ralph' ':!docs' | grep '^+' | grep -v '^+++'; }
added_files() { git diff --name-only --diff-filter=A "$BASE"..HEAD -- "$@"; }
lines_at() { git show "$1:$2" 2>/dev/null | wc -l | tr -d ' '; }
has() { [ -f "$2" ] && grep -qE -e "$1" "$2"; }      # has <ERE> <file>: false (never a stdin read) when the file is missing
all_exist() { local f; for f in "$@"; do [ -f "$f" ] || { echo "   missing: $f"; return 1; }; done; }
# Branch-scoped: each file exists AND was added by this wave (a test that predates it proves nothing about it).
new_files() { local f bad=0; for f in "$@"; do
    if [ ! -f "$f" ]; then echo "   missing: $f"; bad=1
    elif [ -z "$(added_files "$f")" ]; then echo "   not new on this branch: $f"; bad=1; fi
  done; [ $bad -eq 0 ]; }

env_read_added()     { added_code orchestrator ':!orchestrator/config.py' | grep -E 'os\.(getenv|environ)' | grep -q .; }
print_added()        { added_code orchestrator ':!orchestrator/scripts' | grep -E '^\+\s*print\(' | grep -q .; }
raw_fetch_added()    { added_code frontend | grep -E "fetch\(\s*['\"\`]/api" | grep -q .; }
inner_html_added()   { added_code frontend | grep -E 'dangerouslySetInnerHTML|\.innerHTML\s*=|\.outerHTML\s*=|insertAdjacentHTML|document\.write\(' | grep -q .; }
orange_added()       { added_code frontend | grep -E '\b(bg|text|border|ring|from|to|via|fill|stroke)-orange-[0-9]' | grep -q .; }
dark_added()         { added_code frontend | grep -E '\bdark:[a-z]' | grep -q .; }
custom_scheme_added() { added_code orchestrator frontend | grep -iE 'automatos[a-z_-]*://' | grep -q .; }
client_added()       { added_code "$CR" "$CR_ROUTER" "$PROVIDERS" "$TASK_ITEMS" | grep -E '^\+\s*(import|from)\s+(httpx|requests|aiohttp|urllib3)\b' | grep -q .; }
# K19: gitleaks' slack-webhook rule; tests and docs included (a fixture would trip the push scan).
slack_url_added()    { git diff "$BASE"..HEAD -- . ':!scripts/ralph' | grep '^+' | grep -v '^+++' | grep -qE 'hooks\.slack\.com/(services|workflows)/'; }
# Mirrors .github/workflows/dco.yml, which checks `git rev-list --no-merges`: a merge commit is exempt.
every_commit_signed() { [ -z "$(git log --no-merges --format='%H %(trailers:key=Signed-off-by,valueonly)' "$BASE"..HEAD | awk 'NF<2{print $1}')" ]; }
file_sizes_ok() { local f n bad=0; for f in $(added_files orchestrator frontend docs services | grep -E '\.(py|ts|tsx|js|jsx|mjs|css|md|json|ya?ml|sh)$'); do [ -f "$f" ] || continue; n=$(wc -l < "$f" | tr -d ' '); [ "$n" -le 800 ] || { echo "   $f: $n lines"; bad=1; }; done; [ $bad -eq 0 ]; }
not_grown() { local f="$1" b h; b=$(lines_at "$BASE" "$f"); h=$(wc -l < "$f" | tr -d ' '); echo "   $f: base $b, HEAD $h"; [ "$h" -le "$b" ]; }
shrank() { local f="$1" b h; b=$(lines_at "$BASE" "$f"); h=$(wc -l < "$f" | tr -d ' '); echo "   $f: base $b, HEAD $h"; [ "$h" -lt "$b" ]; }
named_files_not_grown() { local f bad=0; for f in $NO_GROW; do [ -f "$f" ] || continue; not_grown "$f" || bad=1; done; [ $bad -eq 0 ]; }
# Every modified file that was over 800 lines on the base, outside GROW_OK, has not grown.
no_big_file_grew() { local f b h bad=0; for f in $(git diff --name-only --diff-filter=M "$BASE"..HEAD); do
    [ -f "$f" ] || continue
    case " $GROW_OK " in *" $f "*) continue ;; esac
    b=$(lines_at "$BASE" "$f"); h=$(wc -l < "$f" | tr -d ' ')
    [ "$b" -gt 800 ] && [ "$h" -gt "$b" ] && { echo "   $f grew: base $b, HEAD $h"; bad=1; }
  done; [ $bad -eq 0 ]; }
config_growth() { local b h; b=$(lines_at "$BASE" $O/config.py); h=$(wc -l < $O/config.py | tr -d ' '); echo "   config.py: base $b, HEAD $h (one setting and its comment: at most 10 lines)"; [ "$h" -le $((b + 10)) ]; }
tests_not_weakened() { local d a; d=$(git diff "$BASE"..HEAD -- $TESTS frontend | grep -cE '^-\s*(async )?def test_|^-\s*(it|test)\('); a=$(git diff "$BASE"..HEAD -- $TESTS frontend | grep -cE '^\+\s*(async )?def test_|^\+\s*(it|test)\('); echo "   test functions removed $d, added $a"; [ "$a" -ge "$d" ]; }
old_migrations_untouched() { [ -z "$(git diff --name-only --diff-filter=MDR "$BASE"..HEAD -- "$VERSIONS")" ]; }
no_new_revision() { local n; n=$(added_files "$VERSIONS" | grep -c .); echo "   new revisions: $n"; [ "$n" -eq 0 ]; }
paths_untouched() { local p bad=0; for p in "$@"; do git diff --quiet "$BASE"..HEAD -- "$p" || { echo "   touched: $p"; bad=1; }; done; [ $bad -eq 0 ]; }
# K20: no path this wave adds or changes matches the global secrets hook (block-secrets.sh case patterns, absolute form).
hook_safe_paths() { local f bad=0; for f in $(git diff --name-only "$BASE"..HEAD); do
    case "/$f" in
      *.env|*.env.*|*/.env|*/.env.*|*/credentials*|*/secrets/*|*/secret/*|*.pem|*.key|*_rsa|*_rsa.pub|*/service-account*.json|*/sa-key*.json)
        echo "   matches the secrets hook: $f"; bad=1 ;;
    esac
  done; [ $bad -eq 0 ]; }
code_shape_ok() {
  "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || { echo "   $PY is older than 3.10: put /opt/homebrew/bin first on PATH"; return 1; }
  "$PY" scripts/ci/check_changed_code_shape.py --base "$BASE" | tail -15
  [ "${PIPESTATUS[0]}" -eq 0 ]
}

# The source of one function (from its `def` line: decorators excluded) is byte-identical to the base's.
fn_unchanged() { "$PY" - "$1" "$2" "$BASE" <<'PY'
import ast, subprocess, sys
path, name, base = sys.argv[1:4]
def segment(src):
    lines = src.splitlines()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return "\n".join(lines[node.lineno - 1:node.end_lineno])
    return None
old = segment(subprocess.run(["git", "show", f"{base}:{path}"], capture_output=True, text=True, check=True).stdout)
new = segment(open(path, encoding="utf-8").read())
if old is None or new is None:
    print(f"   {name} not found ({'base' if old is None else 'HEAD'})"); sys.exit(1)
if old != new:
    print(f"   {name} in {path} differs from the base"); sys.exit(1)
print(f"   {name}: byte-identical to the base ({len(old.splitlines())} lines)")
PY
}
# Code lines of one function (the checker's count: no blank, comment-only or docstring lines) <= the limit.
fn_short() { "$PY" - "$1" "$2" "$3" <<'PY'
import ast, sys
path, name, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
src = open(path, encoding="utf-8").read(); lines = src.splitlines()
for node in ast.walk(ast.parse(src)):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
        doc = set()
        first = node.body[0] if node.body else None
        if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) and isinstance(first.value.value, str):
            doc = set(range(first.lineno, first.end_lineno + 1))
        n = sum(1 for i in range(node.lineno, node.end_lineno + 1)
                if lines[i - 1].strip() and not lines[i - 1].strip().startswith("#") and i not in doc)
        print(f"   {name}: {n} code lines (limit {limit})"); sys.exit(0 if n <= limit else 1)
print(f"   {name} not found in {path}"); sys.exit(1)
PY
}
# One name is in a module-level frozenset/set/tuple constant (read with ast, never imported).
in_constant() { "$PY" - "$1" "$2" "$3" <<'PY'
import ast, sys
path, const, want = sys.argv[1:4]
tree = ast.parse(open(path, encoding="utf-8").read())
for node in tree.body:
    targets = node.targets if isinstance(node, ast.Assign) else ([node.target] if isinstance(node, ast.AnnAssign) else [])
    if not any(isinstance(t, ast.Name) and t.id == const for t in targets) or node.value is None:
        continue
    names = {n.value for n in ast.walk(node.value) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    ok = want in names
    print(f"   {const} {'holds' if ok else 'lacks'} {want!r} ({len(names)} names)"); sys.exit(0 if ok else 1)
print(f"   {const} not found in {path}"); sys.exit(1)
PY
}

one_head() { "$PY" - "$VERSIONS" <<'PY'
import ast, pathlib, sys
revs, downs = {}, set()
for p in sorted(pathlib.Path(sys.argv[1]).glob("*.py")):
    try:
        tree = ast.parse(p.read_text())
    except SyntaxError as e:
        print(f"   {p.name}: {e}"); sys.exit(1)
    rev = down = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names, val = [t.id for t in node.targets if isinstance(t, ast.Name)], node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            names, val = [node.target.id], node.value
        else:
            continue
        try:
            lit = ast.literal_eval(val)
        except ValueError:
            continue
        if "revision" in names: rev = lit
        if "down_revision" in names: down = lit
    if rev is None: continue
    revs[rev] = p.name
    downs.update(down if isinstance(down, (list, tuple)) else ([down] if down else []))
heads = sorted(set(revs) - downs)
print(f"   {len(revs)} revisions, heads: {heads}")
sys.exit(0 if len(heads) == 1 else 1)
PY
}
# US-017a: services/ai_providers.py never calls a resolver that decrypts or logs a key (read with ast: a docstring
# that names them never counts).
no_key_resolvers() { "$PY" - "$PROVIDERS" <<'PY'
import ast, sys
banned = {"resolve_provider_key", "get_platform_workspace_key", "decrypt", "get_encryption_service"}
tree = ast.parse(open(sys.argv[1], encoding="utf-8").read())
hits = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Name) and node.id in banned: hits.add(node.id)
    elif isinstance(node, ast.Attribute) and node.attr in banned: hits.add(node.attr)
    elif isinstance(node, ast.alias) and node.name.split(".")[-1] in banned: hits.add(node.name)
print(f"   uses {sorted(hits)}" if hits else "   reads key presence only (no decrypting resolver)"); sys.exit(1 if hits else 0)
PY
}
# US-018d: /go/pair calls nothing and stores nothing (code only: a // or block comment that names fetch never counts).
go_pair_calls_nothing() { local pat="(from ['\"]@/lib/api-client['\"]|apiClient\.|\bfetch\(|axios|XMLHttpRequest|sendBeacon|localStorage\.|sessionStorage\.|document\.cookie)"
  ! grep -rhE "$pat" "$F/app/go" "$GO_PAIR_DIR" 2>/dev/null | grep -vE '^\s*(\*|/\*)' | sed -E 's#//.*$##' | grep -qE "$pat"; }

# --- routes, settings ---------------------------------------------------------------------------
# routes_in_manifest "METHOD path-regex" … : each pair is in the manifest, and route_count matches.
routes_in_manifest() { "$PY" - "$MANIFEST" "$@" <<'PY'
import json, re, sys
data = json.load(open(sys.argv[1]))
pairs = {(r.get("method", ""), r.get("path", "")) for r in data["routes"]}
missing = []
for want in sys.argv[2:]:
    method, pattern = want.split(" ", 1)
    if not any(m == method and re.fullmatch(pattern, p) for m, p in pairs):
        missing.append(want)
if data.get("route_count") != len(data["routes"]):
    missing.append(f"route_count {data.get('route_count')} != {len(data['routes'])} routes")
order = [(r["path"], r["method"]) for r in data["routes"]]
if order != sorted(order):
    missing.append("routes are not sorted by (path, method)")
print("\n".join(f"   missing: {x}" for x in missing) or "   present, counted and sorted"); sys.exit(1 if missing else 0)
PY
}
settings_declared() { local s bad=0; for s in "$@"; do
    grep -qE "^\s+$s\b" $O/config.py || { echo "   config.py lacks $s"; bad=1; }
    grep -q "\"$s\"" "$SURFACE" || { echo "   config-surface.json lacks $s"; bad=1; }
  done; [ $bad -eq 0 ]; }
surface_sorted() { "$PY" - "$SURFACE" <<'PY'
import json, sys
names = json.load(open(sys.argv[1]))["settings"]
ok = names == sorted(names) and len(names) == len(set(names))
print(f"   {len(names)} names, {'sorted and unique' if ok else 'NOT sorted/unique'}"); sys.exit(0 if ok else 1)
PY
}
router_spec() { grep -qE "RouterSpec\(\"$1\"" $O/router_manifest.py; }

# --- the saas middleware's public list: exact paths only (K14, K15) -----------------------------
public_list() { [ -f "$MIDDLEWARE" ] && "$PY" -c 'import sys; s = open(sys.argv[1], encoding="utf-8").read(); i = s.find("createRouteMatcher(["); print(s[i:s.find("]);", i)] if i >= 0 else "")' "$MIDDLEWARE"; }
public_has() { local p; for p in "$@"; do public_list | grep -qF "\"$p\"" || { echo "   the public list lacks \"$p\""; return 1; }; done; }
public_not_wide() { ! public_list | grep -qE '"/(\.well-known|go)(/\(\.\*\)|\(\.\*\)|/\*)?"'; }

# --- the bell's drift guard grew, nothing removed (US-016c) ---------------------------------------
fixture_grew() { local b h; b=$(git show "$BASE:$BELL_GUARD" 2>/dev/null | grep -cE "^  '[a-z_]+',"); h=$(grep -cE "^  '[a-z_]+'," "$BELL_GUARD"); echo "   BACKEND_NOTIFICATION_LINK_TYPES: base $b, HEAD $h"; [ "$h" -gt "$b" ] && [ -z "$(git diff "$BASE"..HEAD -- "$BELL_GUARD" | grep -E "^-  '[a-z_]+',")" ]; }

all_acs_done() { "$PY" - "$KIT" <<'PY'
import json, sys, re
d = json.load(open(sys.argv[1])); bad = []
strip = lambda x: re.sub(r"`[^`]*`", "", x)   # the marks are plain text; the instructions quote them in backticks
for s in d["userStories"]:
    for raw in s["acceptanceCriteria"]:
        ac = strip(raw)
        if ac.startswith("→ OWNER:") or "→ DONE" in ac: continue
        bad.append(f"{s['id']}: {raw[:80]}")
print("\n".join(bad)); sys.exit(1 if bad else 0)
PY
}

# The commit CI tested: HEAD, or, when HEAD and its parents are kit-only '[skip ci]' commits (scripts/ralph/
# marks, gate or prompt fixes after the wave's CI commit), the nearest ancestor that is not. A kit-only commit
# cannot change a test outcome, so that ancestor's run is the wave's run.
ci_commit_for_head() {
  local sha; sha=$(git rev-parse HEAD)
  while git log -1 --format=%s "$sha" | grep -q 'skip ci' && [ -z "$(git show --format= --name-only "$sha" | grep -v '^scripts/ralph/')" ]; do
    sha=$(git rev-parse "$sha^")
  done
  echo "$sha"
}
# Required jobs = test.yml's CURRENT jobs (read on origin/main 7ae677b52, 20 Oct). This wave touches orchestrator/
# (backend_mode full → the shards, orchestrator-tests, Alembic from-zero, Schema-drift, the module + integration
# tests F056) and frontend/ + route-manifest.json (Frontend CI). Code standards runs on every PR. Path-gated, green
# or skipped: "Prod images built" (Dockerfiles, package manifests, next.config.*, lib/auth-edition.ts, test.yml:
# this wave touches none) and the "CLI host (PRD-234)" lane (services/cli-host/: untouched).
ci_green_on_head() {
  local br sha waited=0 run id jobs job hits
  br=$(git rev-parse --abbrev-ref HEAD); sha=$(ci_commit_for_head)
  [ "$sha" = "$(git rev-parse HEAD)" ] || echo "   HEAD is kit-only [skip ci] commits over $(git log -1 --format=%h "$sha"): reading that commit's run"
  command -v gh >/dev/null || { echo "   gh is required"; return 1; }
  [ -n "$(gh pr list --head "$br" --state open --json number --jq '.[0].number' 2>/dev/null)" ] || { echo "   no open PR for $br: test.yml runs on pull_request only (#993)"; return 1; }
  while :; do
    run=$(gh run list --branch "$br" --workflow test.yml --limit 20 --json databaseId,headSha,status --jq ".[] | select(.headSha == \"$sha\") | \"\(.databaseId) \(.status)\"" 2>/dev/null | head -1)
    id="${run%% *}"
    [ -n "$run" ] && [ "${run##* }" = "completed" ] && break
    [ "$waited" -ge 3600 ] && { echo "   no completed test.yml run for $sha after 60 min (${run:-no run})"; return 1; }
    sleep 60; waited=$((waited + 60))
  done
  jobs=$(gh run view "$id" --json jobs --jq '.jobs[] | "\(.conclusion)\t\(.name)"')
  local REQUIRED=("Which lanes this change needs" "orchestrator-tests" "Alembic from-zero" "Schema-drift check" "Frontend CI" "Code standards on changed lines" "Orchestrator module + integration tests" "Prod images built" "CLI host (PRD-234)")
  local PATH_GATED=("Prod images built" "CLI host (PRD-234)")
  for job in "${REQUIRED[@]}"; do
    hits=$(echo "$jobs" | grep -F "$job" | cut -f1)
    [ -n "$hits" ] || { echo "   missing job: $job"; return 1; }
    if printf '%s\n' "${PATH_GATED[@]}" | grep -qxF "$job"; then echo "$hits" | grep -qvE '^(success|skipped)$' && { echo "   red: $job"; return 1; }
    else echo "$hits" | grep -qvE '^success$' && { echo "   red: $job"; return 1; }; fi
  done
  echo "   test.yml run $id is green on $sha for all ${#REQUIRED[@]} required jobs (Prod images, CLI host: green or skipped)"
}
# The wave's PR body records the tracking issue, the PRD PR, where US-019 and the history paging went, the setting,
# the public frontend paths, that there is no migration, and the AI disclosure.
pr_body_records() {
  local br n body s miss=0
  br=$(git rev-parse --abbrev-ref HEAD)
  n=$(gh pr list --head "$br" --state open --json number --jq '.[0].number' 2>/dev/null)
  [ -n "$n" ] || { echo "   no open PR for $br"; return 1; }
  body=$(gh pr view "$n" --json body --jq .body 2>/dev/null)
  for s in "#1255" "#1257" "US-019" "US-007b" "MODERATION_WEBHOOK_URL" "/go/pair" "apple-app-site-association" "No migration" "AI assistance"; do
    printf '%s' "$body" | grep -qF -- "$s" || { echo "   the PR body lacks: $s"; miss=1; }
  done
  [ $miss -eq 0 ] && echo "   PR #$n records #1255/#1257, US-019's move, US-007b, the setting, the public paths, no migration and the AI disclosure"
}

if [ "$MODE" != "negative" ]; then
  echo ""; echo "=== Seeded"
  check "the PRD is tracked at $PRD_DOC" "git ls-files --error-unmatch $PRD_DOC >/dev/null 2>&1"

  echo ""; echo "=== Stories (priority order)"
  check "US-016a the content_reports package, the thin router, RouterSpec, POST /api/v1/reports in route-manifest.json" "all_exist $CR/__init__.py $CR/targets.py $CR/service.py $CR_ROUTER && router_spec api.content_reports && has '/api/v1/reports' $CR_ROUTER && routes_in_manifest 'POST /api/v1/reports'"
  check "US-016a the members:read gate; K8's reasons and target types; the audit row, the owner and the limiter through the existing seams" "has \"require_workspace_permission\\\\(['\\\"]members:read['\\\"]\\\\)\" $CR_ROUTER && grep -rqE 'REPORT_REASONS' $CR && for r in harmful offensive inaccurate privacy other message deliverable; do grep -rqE \"['\\\"]\$r['\\\"]\" $CR || { echo \"   no '\$r' in $CR\"; exit 1; }; done && grep -rq 'AuditService' $CR && grep -rq 'workspace_owner' $CR && grep -rq 'check_rate_limit' $CR $CR_ROUTER && grep -rq 'rate_limit_for' $CR && grep -rqE \"['\\\"]content_reported['\\\"]\" $CR"
  check "US-016a content_reported is in VALID_EVENT_TYPES (the module constant; dispatch is compared in N14)" "in_constant $DISPATCHER VALID_EVENT_TYPES content_reported"
  check "US-016a the service and route tests are new" "new_files $TESTS/test_prd266_content_reports.py $TESTS/test_prd266_content_reports_route.py"
  check "US-016b MODERATION_WEBHOOK_URL in config.py and config-surface.json (sorted)" "settings_declared $SETTINGS && surface_sorted"
  check "US-016b moderation.py: the existing webhook driver, SaaS and https only, posted after the response" "all_exist $CR/moderation.py && has \"get_driver\\\\(['\\\"]webhook['\\\"]\\\\)\" $CR/moderation.py && has 'AUTH_EDITION|IS_SAAS_EDITION' $CR/moderation.py && has 'https' $CR/moderation.py && has 'MODERATION_WEBHOOK_URL' $CR/moderation.py && grep -rqE 'BackgroundTasks' $CR_ROUTER"
  check "US-016b the env doc names the setting; the moderation test is new" "has 'MODERATION_WEBHOOK_URL' $ENV_DOC && new_files $TESTS/test_prd266_content_reports_moderation.py"
  check "US-016c the bell routes content_report; the drift guard gained it and lost nothing; the audit pane filters content_reported" "has \"'content_report'\" $BELL && has \"'content_report'\" $BELL_GUARD && fixture_grew && has 'content_reported' $AUDIT_PANE && new_files $CC/governance/__tests__/prd266-content-reports.test.tsx"
  check "US-017a services/ai_providers.py from key presence only (no resolver that decrypts or logs a key)" "all_exist $PROVIDERS && has 'def workspace_ai_providers' $PROVIDERS && has 'byok_overrides' $PROVIDERS && has 'platform_key_allowed' $PROVIDERS && has 'PLATFORM_KEY_WORKSPACE_ID' $PROVIDERS && no_key_resolvers && new_files $TESTS/test_prd266_ai_providers.py"
  check "US-017b the other AI services: embeddings, reranker, session runtimes, and Retell by the voice gates (switch, armed, workspace, plan)" "has 'EMBEDDING_PROVIDER' $PROVIDERS && has 'rerank' $PROVIDERS && has 'is_cli_agent|session_cli' $PROVIDERS && has \"['\\\"]retell['\\\"]\" $PROVIDERS && has 'Retell AI' $PROVIDERS && has 'voice_live_enabled' $PROVIDERS && has 'armed' $PROVIDERS && has 'parse_workspace_voice_live' $PROVIDERS && has 'exposure_for_plan' $PROVIDERS && new_files $TESTS/test_prd266_ai_providers_services.py"
  check "US-017c GET /current returns ai_providers; get_current_workspace ≤ 50 code lines; api/workspaces.py did not grow; the route's existing tests untouched" "has 'ai_providers' $O/api/workspaces.py && fn_short $O/api/workspaces.py get_current_workspace 50 && not_grown $O/api/workspaces.py && paths_untouched $TESTS/test_prd251_settings.py && new_files $TESTS/test_prd266_ai_providers_route.py"
  check "US-018a the schedule source moved out with paused; activity_service.py shrank; the existing schedule tests untouched" "all_exist $TASK_ITEMS && has 'def scheduled_task_items' $TASK_ITEMS && has 'paused' $TASK_ITEMS && fn_short $O/services/activity_service.py _scheduled_task_items 5 && shrank $O/services/activity_service.py && paths_untouched $TESTS/test_schedule_endpoint.py $TESTS/test_prd251w3_calendar.py && new_files $TESTS/test_prd266_schedule_paused.py"
  check "US-018b the web calendar: paused on ScheduleItem, Resume, the word Paused, Next up skips it" "has 'paused' $F/hooks/use-activity-api.ts && has 'Resume' $CC/calendar-actions.ts && grep -qs 'Paused' $CC/calendar-*.ts* && has 'paused' $CC/calendar-view.ts && new_files $CC/__tests__/prd266-calendar-paused.test.tsx"
  check "US-018c the two association files: route handlers, the app id, the Android variable, exact public paths, the env doc" "all_exist $AASA $ASSETLINKS && grep -rqF '$IOS_APP_ID' $F/app/.well-known $F/lib && grep -rqF 'com.automatos.app' $F/app/.well-known $F/lib && grep -rqF 'ANDROID_APP_SHA256_CERT_FINGERPRINTS' $F/app/.well-known $F/lib && public_has /.well-known/apple-app-site-association /.well-known/assetlinks.json && public_not_wide && has 'ANDROID_APP_SHA256_CERT_FINGERPRINTS' $ENV_DOC && new_files $F/app/__tests__/prd266-universal-links.test.ts"
  check "US-018d /go/pair: the page, its parts, the shared frame, the exact public path, no API call of any kind" "all_exist $GO_PAIR && [ -d $GO_PAIR_DIR ] && grep -rqE 'PageHeader' $GO_PAIR $GO_PAIR_DIR && public_has /go/pair && public_not_wide && go_pair_calls_nothing && new_files $F/app/__tests__/prd266-go-pair-page.test.tsx"
  check "every acceptance criterion in $KIT is DONE" "all_acs_done"
fi

echo ""; echo "=== Scope and conventions"
check "N1 NO migration: no new revision, the two head pins untouched (K4)" "no_new_revision && paths_untouched $TESTS/test_prd209_alembic_single_head.py $TESTS/test_prd236_w1_routes.py"
check "N2 exactly one Alembic head in the tree" "one_head"
check "N3 no env reads added outside config.py" "! env_read_added"
check "N4 no print() added in orchestrator application code (scripts may print)" "! print_added"
check "N5 no raw fetch('/api…') added in the frontend" "! raw_fetch_added"
check "N6 no new Python dependency" "git diff --quiet $BASE..HEAD -- orchestrator/requirements.txt orchestrator/requirements-dev.txt orchestrator/pyproject.toml"
check "N7 no new npm dependency (package.json and package-lock.json untouched)" "git diff --quiet $BASE..HEAD -- frontend/package.json frontend/package-lock.json"
check "N8 the Composio deny list and the generated Auto skill seed are untouched" "git diff --quiet $BASE..HEAD -- orchestrator/core/composio/deny_list.py orchestrator/core/seeds/platform-management-skill.md"
check "N9 no node_modules in the diff" "! git diff --name-only $BASE..HEAD | grep -q node_modules"
check "N10 every commit is DCO-signed" "every_commit_signed"
check "N11 a new file is at most 800 lines" "file_sizes_ok"
check "N12a the files over 800 lines this wave may touch did not grow (and the five it never touches)" "named_files_not_grown"
check "N12b no other file over 800 lines grew (config.py and the two reports excepted)" "no_big_file_grew"
warn "N12c config.py grew by its one setting only (≤ 10 lines)" "config_growth"
check "N13 get_request_context_hybrid is byte-identical to the base" "fn_unchanged $O/core/auth/hybrid.py get_request_context_hybrid"
check "N14 NotificationDispatcher.dispatch is byte-identical to the base" "fn_unchanged $DISPATCHER dispatch"
check "N14b EmbeddingManager._load_provider is byte-identical to the base" "fn_unchanged $O/core/llm/embedding_manager.py _load_provider"
check "N15 no innerHTML / dangerouslySetInnerHTML added in the frontend" "! inner_html_added"
check "N16 no W1 file created or edited, nor W1's chat and voice files (K5, K3)" "paths_untouched $W1_PATHS"
check "N17 no third-party HTTP client imported by the wave's new backend modules (the post rides channels/drivers/webhook.py)" "! client_added"
check "N18 no custom URL scheme added (universal links only)" "! custom_scheme_added"
check "N19 no test weakened: added test functions >= removed" "tests_not_weakened"
check "N20 the hierarchy gate passes (stdlib ast)" "\"$PY\" orchestrator/scripts/check_hierarchy_gate.py >/dev/null"
check "N21 no existing migration modified, deleted or renamed" "old_migrations_untouched"
check "N22 HOST_OWNED_ROUTES (TS and Python) and next.config.js untouched (K14, K15)" "paths_untouched $UNTOUCHED_FRONT"
check "N23 no changed path matches the global secrets hook; no .env file (K20)" "hook_safe_paths"
check "N24 CI's changed-lines code shape passes on the committed wave (functions ≤ 50, nesting ≤ 4, new files ≤ 800)" "code_shape_ok"
check "N25 no orange-* palette class added (.eslintrc no-restricted-syntax)" "! orange_added"
check "N26 no dark: palette class added (lib/__tests__/dark-variant-tokens.test.ts)" "! dark_added"
check "N27 no Slack-webhook-shaped URL added anywhere, tests and docs included (gitleaks, K19)" "! slack_url_added"
check "N28 the saas middleware's public list never opens a prefix (/.well-known/(.*), /go/(.*))" "public_not_wide"
warn "N29 story commits carry [skip ci] (all but the final CI commit)" "[ -z \"\$(git log --format='%s' $BASE..HEAD^ | grep -v 'skip ci')\" ]"

if [ "$MODE" = "full" ]; then
  echo ""; echo "=== CI and the PR (once per wave)"
  check "CI: the required jobs present and green on the tested commit" "ci_green_on_head"
  check "the PR body records #1255, #1257, US-019's move, US-007b, the setting, the public paths, no migration and the AI disclosure" "pr_body_records"
fi
echo ""
if [ "$FAIL" -eq 0 ]; then echo "✅ PRD-266 W3 Store duties acceptance ($MODE): PASS"; else echo "❌ PRD-266 W3 Store duties acceptance ($MODE): FAIL"; fi
exit "$FAIL"
