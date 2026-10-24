#!/bin/bash
# Acceptance gate — PRD-266 W2 Push, the Automatos Go backend (16 stories: US-010a … US-019c).
#   bash scripts/ralph/acceptance-prd266w2.sh | --print-base | --no-ci | --negative-only ; GATE_BASE=<ref> (smoke runs only).
# Base (K1): the wave is STACKED on W1 (feat/prd-266-w1-connect). While HEAD sits on W1's work, the base is the W1 commit
# the branch grew from (merge-base with origin/feat/prd-266-w1-connect), so W1's own diff is never read as this wave's;
# once W1 has landed on main and main was merged in (or W1's branch is gone), the base is the merge-base with origin/main.
# Nothing runs on the owner's machine: greps, git and stdlib python only; CI once per wave on the draft PR. No pipefail
# (a negative `! producer | grep -q` would invert on SIGPIPE); every check runs in its own `( eval … )`; stdin is closed
# so a grep with a missing file argument fails instead of hanging.
#
# Negative checks and the base (20 Oct 2026):
#   - every "added lines" check reads `git diff BASE..HEAD`, EMPTY when HEAD is the base, so it passes there by
#     construction; each is scoped to code dirs and excludes tests, the kit (scripts/ralph) and docs, so a check's own
#     pattern line or a test asserting absence never trips it;
#   - N2 (one Alembic head), the byte-identical function checks, N20 (the hierarchy gate) and N24 (CI's changed-lines
#     code-shape script) hold on the base by construction (HEAD = BASE: no diff, the same source);
#   - the story checks are POSITIVE (the wave's work): they fail on the base by design (no services/push/, no
#     prd266_w2_push, no api/mobile_push.py, no review_pending or auto_message event there).
set -u
exec </dev/null
cd "$(dirname "$0")/../.." || exit 1
export DATABASE_URL="postgresql://ralph:ralph@127.0.0.1:1/ralph_no_db"
export POSTGRES_HOST=127.0.0.1 POSTGRES_PORT=1 POSTGRES_USER=ralph POSTGRES_PASSWORD=ralph POSTGRES_DB=ralph_no_db
export REDIS_URL="redis://127.0.0.1:1/0" REDIS_HOST=127.0.0.1 REDIS_PORT=1
export ENVIRONMENT=development
# Homebrew's python (3.14) when present: CI's code-shape script needs 3.10+ (ast.Match); macOS's /usr/bin is 3.9.
PY=$(command -v /opt/homebrew/bin/python3 || command -v python3)

STACK_REF=origin/feat/prd-266-w1-connect
W1_BRANCH=feat/prd-266-w1-connect
if [ -n "${GATE_BASE:-}" ]; then
  BASE=$(git rev-parse --verify -q "$GATE_BASE^{commit}") || { echo "GATE_BASE $GATE_BASE does not resolve"; exit 1; }
else
  MB_MAIN=$(git merge-base HEAD origin/main) || { echo "cannot resolve merge-base with origin/main"; exit 1; }
  BASE=$MB_MAIN
  if git rev-parse --verify -q "$STACK_REF^{commit}" >/dev/null; then
    MB_W1=$(git merge-base HEAD "$STACK_REF" 2>/dev/null || true)
    # Stacked: W1's work sits between main and this wave (main's merge-base is behind W1's).
    if [ -n "$MB_W1" ] && [ "$MB_W1" != "$MB_MAIN" ] && git merge-base --is-ancestor "$MB_MAIN" "$MB_W1"; then BASE=$MB_W1; fi
  fi
fi
MODE=full
case "${1:-}" in --print-base) echo "$BASE"; exit 0 ;; --no-ci) MODE=no-ci ;; --negative-only) MODE=negative ;; "") ;; *) echo "unknown argument: $1"; exit 2 ;; esac
echo "base: $(git log -1 --format='%h %s' "$BASE") · HEAD: $(git log -1 --format='%h %s' HEAD) · mode: $MODE · python: $("$PY" -V 2>&1)"
FAIL=0
check() { local name="$1" body="$2"; echo ""; echo "── $name"; if ( eval "$body" ); then echo "   ✅ PASS: $name"; else echo "   ❌ FAIL: $name"; FAIL=1; fi; }
warn() { local name="$1" body="$2"; echo ""; echo "── $name"; if ( eval "$body" ); then echo "   ✅ ok: $name"; else echo "   ⚠️  WARN (not a failure): $name"; fi; }

KIT=scripts/ralph/prd-266w2.json
PRD_DOC=docs/PRDS/PRD-266-AUTOMATOS-GO-BACKEND.md
O=orchestrator
TESTS=$O/tests
VERSIONS=$O/alembic/versions
MANIFEST=$O/reports/route-manifest.json
SURFACE=$O/reports/config-surface.json
PUSH=$O/services/push
EVENTS=$O/services/notification_events
MOBILE=$O/services/mobile
ROUTER=$O/api/mobile_push.py
DISPATCHER=$O/core/services/notification_dispatcher.py
NOTIFY_API=$O/api/notifications.py
PD=frontend/components/settings/phones-devices
NEW_REV=prd266_w2_push
SETTINGS_SENDER="EXPO_PUSH_ACCESS_TOKEN EXPO_PUSH_TIMEOUT_SECONDS"
# §4.10: files over 800 lines that may not grow (the PRD's list, and the others this wave must not grow).
NO_GROW="$O/core/auth/hybrid.py $O/api/chat.py $O/api/approval_grants.py $O/main.py $O/services/cli_host_service.py $O/consumers/chatbot/service.py frontend/lib/api-client.ts $O/api/workspaces.py $O/api/board_tasks.py $O/services/coordinator_service.py"
# Over 800 lines and allowed to grow: settings live only in config.py (AGENTS.md); the two reports are generated lists.
GROW_OK="$O/config.py $MANIFEST $SURFACE"
# The dispatcher's existing tests patch its helpers: never edited (K3, US-010d); the defaults test pins nine rows.
FROZEN_TESTS="$TESTS/test_prd128_notification_dispatcher.py $TESTS/test_f197_a_running_out_of_credit_is_said_once.py $TESTS/test_f197_b_a_failed_tasks_report_is_held_in_an_outage.py $TESTS/test_f209_b_a_notice_is_dated_when_it_is_written.py $TESTS/test_prd128_default_prefs.py"
FIVE_EVENTS="approval_pending auto_message mission_plan_ready question_pending review_pending"

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
custom_scheme_added() { added_code orchestrator frontend | grep -iE 'automatos[a-z_-]*://' | grep -q .; }
# AGENTS.md exception 2: services/push/expo.py is the ONE module of the wave that may speak HTTP.
client_added() { added_code orchestrator ':!orchestrator/services/push/expo.py' | grep -E '^\+\s*(import|from)\s+(httpx|requests|aiohttp|urllib3)\b' | grep -q .; }
# The push package talks to Expo's host only.
foreign_host_added() { added_code "$PUSH" | grep -oE 'https?://[A-Za-z0-9.-]+' | grep -vE '^https://exp\.host$' | grep -q .; }
# Never log a push token, a push key, the Expo access token or a question's text.
secret_logged() { added_code "$PUSH" "$EVENTS" "$MOBILE" "$ROUTER" "$DISPATCHER" | grep -E 'logger\.[a-z]+\(' | grep -iE 'push_key|expo_push_token|access_token|EXPO_PUSH_ACCESS_TOKEN|question_md|\.body\b' | grep -q .; }
# W3 and later stay out (K21): content reports, ai_providers, the moderation webhook, universal links, /go/pair, deletion.
w3_added() {
  [ -n "$(added_files "$O/api/content_reports.py" "$O/services/reports" "$O/api/account.py" "$O/services/account" frontend/public/.well-known frontend/app/go)" ] && { echo "   a W3 file was added"; return 0; }
  added_code orchestrator frontend | grep -E 'ai_providers|MODERATION_WEBHOOK_URL|apple-app-site-association|assetlinks\.json' | grep -q . && { echo "   a W3 name was added"; return 0; }
  return 1
}
# Mirrors .github/workflows/dco.yml, which checks `git rev-list --no-merges`: a merge commit is exempt.
every_commit_signed() { [ -z "$(git log --no-merges --format='%H %(trailers:key=Signed-off-by,valueonly)' "$BASE"..HEAD | awk 'NF<2{print $1}')" ]; }
file_sizes_ok() { local f n bad=0; for f in $(added_files orchestrator frontend docs services | grep -E '\.(py|ts|tsx|js|jsx|mjs|css|md|json|ya?ml|sh)$'); do [ -f "$f" ] || continue; case "$f" in frontend/lib/vendor/*) continue ;; esac; n=$(wc -l < "$f" | tr -d ' '); [ "$n" -le 800 ] || { echo "   $f: $n lines"; bad=1; }; done; [ $bad -eq 0 ]; }
not_grown() { local f="$1" b h; b=$(lines_at "$BASE" "$f"); h=$(wc -l < "$f" | tr -d ' '); echo "   $f: base $b, HEAD $h"; [ "$h" -le "$b" ]; }
named_files_not_grown() { local f bad=0; for f in $NO_GROW; do [ -f "$f" ] || continue; not_grown "$f" || bad=1; done; [ $bad -eq 0 ]; }
# Every modified file that was over 800 lines on the base, outside GROW_OK, has not grown.
no_big_file_grew() { local f b h bad=0; for f in $(git diff --name-only --diff-filter=M "$BASE"..HEAD); do
    [ -f "$f" ] || continue
    case " $GROW_OK " in *" $f "*) continue ;; esac
    b=$(lines_at "$BASE" "$f"); h=$(wc -l < "$f" | tr -d ' ')
    [ "$b" -gt 800 ] && [ "$h" -gt "$b" ] && { echo "   $f grew: base $b, HEAD $h"; bad=1; }
  done; [ $bad -eq 0 ]; }
config_growth() { local b h; b=$(lines_at "$BASE" $O/config.py); h=$(wc -l < $O/config.py | tr -d ' '); echo "   config.py: base $b, HEAD $h (settings only: at most ~5 lines per setting)"; [ "$h" -le $((b + 40)) ]; }
tests_not_weakened() { local d a; d=$(git diff "$BASE"..HEAD -- $TESTS frontend | grep -cE '^-\s*(async )?def test_|^-\s*(it|test)\('); a=$(git diff "$BASE"..HEAD -- $TESTS frontend | grep -cE '^\+\s*(async )?def test_|^\+\s*(it|test)\('); echo "   test functions removed $d, added $a"; [ "$a" -ge "$d" ]; }
frozen_tests_unchanged() { local f bad=0; for f in $FROZEN_TESTS; do git diff --quiet "$BASE"..HEAD -- "$f" || { echo "   edited: $f"; bad=1; }; done; [ $bad -eq 0 ]; }
old_migrations_untouched() { [ -z "$(git diff --name-only --diff-filter=MDR "$BASE"..HEAD -- "$VERSIONS")" ]; }
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
try:
    src = open(path, encoding="utf-8").read()
except OSError:
    print(f"   {path} is missing"); sys.exit(1)
lines = src.splitlines()
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
# A module-level frozenset/tuple/set/dict literal's string members (in its whole value, dict keys and values included).
literal_strings() { "$PY" - "$1" "$2" <<'PY'
import ast, sys
path, name = sys.argv[1:3]
try:
    tree = ast.parse(open(path, encoding="utf-8").read())
except OSError:
    sys.exit(1)
for node in tree.body:
    targets = node.targets if isinstance(node, ast.Assign) else ([node.target] if isinstance(node, ast.AnnAssign) else [])
    if any(isinstance(t, ast.Name) and t.id == name for t in targets) and node.value is not None:
        found = sorted({n.value for n in ast.walk(node.value) if isinstance(n, ast.Constant) and isinstance(n.value, str)})
        print(" ".join(found)); sys.exit(0)
sys.exit(1)
PY
}
set_has() { local file="$1" name="$2" want="$3" got; got=$(literal_strings "$file" "$name") || { echo "   $name not found in $file"; return 1; }
  case " $got " in *" $want "*) return 0 ;; esac; echo "   $name in $file lacks '$want'"; return 1; }
set_lacks() { local got; got=$(literal_strings "$1" "$2") || return 0; case " $got " in *" $3 "*) echo "   $2 in $1 holds '$3'"; return 1 ;; esac; return 0; }
# The event types PUSH_TYPE_EVENTS maps the three push types to (the dict's VALUES), exactly the five (FR-7, K8).
# Signature v2 (K9, the lead 21 Oct): the ONE shared vector file carries the v2 schema, the window and K9's five sigs.
V2_SIGS="ebbf977c0e83aefa57f52a0559c6907b31be5ebb44112450b8b88c248f4080fa 800c8568838c16721faf7e738b7d715f0730c3c7d5cefb9941ab76ba69e293d3 5f11674af935f0ab0a912494ca6502750354541c774c667934922177df07dc88 3e947349177d6ee98138bb05cc14112069522d3ba2c042d8f67ac18b22b02aa5 e0e3a50468bb5e8de2a97706037d82110e07d92d8f8dbb0cdca0de8e01442136"
v2_vectors_ok() {
  local f=$TESTS/fixtures/prd266_push_signature_vectors.json s
  [ -f "$f" ] || { echo "   missing $f"; return 1; }
  grep -q '"automatos-push-signature-vectors/v2"' "$f" || { echo "   $f: schema is not automatos-push-signature-vectors/v2"; return 1; }
  grep -q '604800' "$f" && grep -q '"max_future_seconds": *300' "$f" || { echo "   $f: the window is not 604800 / 300"; return 1; }
  for s in $V2_SIGS; do grep -q "$s" "$f" || { echo "   $f lacks K9's case $s"; return 1; }; done
  grep -q 'tampered' "$f" || { echo "   $f has no tampered cases"; return 1; }
}
# The device name (US-009, the lead 21 Oct): RegisterDeviceRequest takes it, REGISTER_COLUMNS writes it, the router stays < 400 lines.
device_name_ok() {
  sed -n '/^class RegisterDeviceRequest/,/^class UpdateDeviceRequest/p' $O/api/mobile_devices.py | grep -qE '^ +name:' || { echo "   RegisterDeviceRequest has no name field"; return 1; }
  grep -A3 'REGISTER_COLUMNS =' $MOBILE/devices.py | grep -q '"name"' || { echo "   REGISTER_COLUMNS does not write name"; return 1; }
  [ "$(wc -l < $O/api/mobile_devices.py | tr -d ' ')" -lt 400 ] || { echo "   api/mobile_devices.py reached 400 lines"; return 1; }
}
five_push_events() { "$PY" - "$PUSH/payload.py" "$FIVE_EVENTS" <<'PY'
import ast, sys
path, want = sys.argv[1], sorted(sys.argv[2].split())
try:
    tree = ast.parse(open(path, encoding="utf-8").read())
except OSError:
    print(f"   {path} is missing"); sys.exit(1)
for node in tree.body:
    targets = node.targets if isinstance(node, ast.Assign) else ([node.target] if isinstance(node, ast.AnnAssign) else [])
    if not any(isinstance(t, ast.Name) and t.id == "PUSH_TYPE_EVENTS" for t in targets) or node.value is None:
        continue
    table = next((n for n in ast.walk(node.value) if isinstance(n, ast.Dict)), None)
    if table is None:
        print("   PUSH_TYPE_EVENTS is not a dict literal"); sys.exit(1)
    keys = sorted(k.value for k in table.keys if isinstance(k, ast.Constant))
    events = sorted({n.value for v in table.values for n in ast.walk(v) if isinstance(n, ast.Constant) and isinstance(n.value, str)})
    print(f"   push types {keys}; events {events}")
    sys.exit(0 if keys == ["auto_message", "needs_answer", "review"] and events == want else 1)
print("   PUSH_TYPE_EVENTS not found"); sys.exit(1)
PY
}

# --- US-010b: the migration ----------------------------------------------------------------------
base_head() { git show "$BASE:$TESTS/test_prd209_alembic_single_head.py" 2>/dev/null | grep -E '^EXPECTED_HEAD *= *"' | head -1 | sed -E 's/.*"([^"]+)".*/\1/'; }
new_revisions() { added_files "$VERSIONS" | grep '\.py$'; }
one_revision() { local n; n=$(new_revisions | grep -c .); echo "   new revisions: $n ($(new_revisions | tr '\n' ' '))"; [ "$n" -eq 1 ]; }
revision_ok() {
  local f bh c; f="$VERSIONS/$NEW_REV.py"; bh=$(base_head)
  [ -f "$f" ] || { echo "   $f is missing"; return 1; }
  [ -n "$bh" ] || { echo "   cannot read the base's EXPECTED_HEAD"; return 1; }
  grep -qE "^revision[^=]*= *['\"]$NEW_REV['\"]" "$f" || { echo "   $f: revision is not $NEW_REV"; return 1; }
  grep -qE "^down_revision[^=]*= *['\"]$bh['\"]" "$f" || { echo "   $f: down_revision is not the base head $bh"; return 1; }
  for c in 'CREATE TABLE IF NOT EXISTS mobile_push_tickets' 'CREATE TABLE IF NOT EXISTS mobile_digest_posts'; do
    grep -qiE "$c" "$f" || { echo "   $f: no $c"; return 1; }
  done
  [ "$(grep -ciE 'CREATE (TABLE|(UNIQUE )?INDEX)' "$f")" -le "$(grep -ciE 'IF NOT EXISTS' "$f")" ] || { echo "   $f: a CREATE without IF NOT EXISTS"; return 1; }
  for c in ticket_id device_id push_type local_date posted_at; do
    grep -q "$c" "$f" || { echo "   $f lacks the column $c"; return 1; }
  done
  grep -q "EXPECTED_HEAD = \"$NEW_REV\"" $TESTS/test_prd209_alembic_single_head.py || { echo "   test_prd209 EXPECTED_HEAD is not $NEW_REV"; return 1; }
  grep -qF "$NEW_REV.py" $TESTS/test_prd236_w1_routes.py || { echo "   test_prd236_w1_routes.py does not read $NEW_REV.py"; return 1; }
  grep -qF "EXPECTED_HEAD = \"$NEW_REV\"" $TESTS/test_prd236_w1_routes.py || { echo "   test_prd236_w1_routes.py does not assert EXPECTED_HEAD = $NEW_REV"; return 1; }
  echo "   $NEW_REV chains onto $bh; both pins moved"
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

# --- routes and settings -------------------------------------------------------------------------
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
# added_in <ERE> <file>: a line this wave added to <file> matches (a positive smoke signal; the tests are the proof).
added_in() { git diff "$BASE"..HEAD -- "$2" | grep '^+' | grep -v '^+++' | grep -qE -e "$1"; }
setting_not_in_config() { ! grep -qE "^\s+$1\b" $O/config.py; }
router_spec() { grep -qE "RouterSpec\(\"$1\"" $O/router_manifest.py; }
# The new router's handlers are plain def (sync session; AGENTS.md).
router_plain_def() { "$PY" - "$ROUTER" <<'PY'
import ast, sys
bad = []
try:
    tree = ast.parse(open(sys.argv[1], encoding="utf-8").read())
except OSError:
    print(f"   {sys.argv[1]} is missing"); sys.exit(1)
for node in ast.walk(tree):
    if isinstance(node, ast.AsyncFunctionDef):
        for d in node.decorator_list:
            f = d.func if isinstance(d, ast.Call) else d
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "router":
                bad.append(f"{sys.argv[1]}:{node.lineno} {node.name} is async def")
print("\n".join(f"   {b}" for b in bad) or "   every route handler is a plain def"); sys.exit(1 if bad else 0)
PY
}
# A mark counts when it sits outside backticks (the instructions quote the marks in backticks).
all_acs_done() { "$PY" - "$KIT" <<'PY'
import json, sys, re
d = json.load(open(sys.argv[1])); bad = []
strip = lambda x: re.sub(r"`[^`]*`", "", x)
for s in d["userStories"]:
    for raw in s["acceptanceCriteria"]:
        ac = strip(raw)
        if "→ DONE" in ac or "→ OWNER:" in ac: continue
        bad.append(f"{s['id']}: {raw[:80]}")
print("\n".join(bad)); sys.exit(1 if bad else 0)
PY
}
# US-019c: while K17 is not approved every AC carries the OWNER mark and no encoder exists; once approved, it is built.
qr_story_ok() { "$PY" - "$KIT" "$PD" <<'PY'
import json, os, re, sys
kit, pd = sys.argv[1:3]
d = json.load(open(kit))
k17 = next((x for x in d["decisions"] if x.startswith("K17")), "")
story = next((s for s in d["userStories"] if s["id"] == "US-019c"), None)
if story is None:
    print("   US-019c missing"); sys.exit(1)
strip = lambda x: re.sub(r"`[^`]*`", "", x)
vendored = os.path.isdir("frontend/lib/vendor/qrcodegen")
if "STATUS: NOT APPROVED" in k17:
    marked = all("→ OWNER:" in strip(ac) for ac in story["acceptanceCriteria"])
    print(f"   K17 not approved: OWNER marks {'on every AC' if marked else 'MISSING'}; vendored encoder {'PRESENT' if vendored else 'absent'}")
    sys.exit(0 if marked and not vendored else 1)
test = os.path.join(pd, "__tests__", "pair-phone-qr.test.tsx")
print(f"   K17 approved: encoder {'vendored' if vendored else 'MISSING'}; test {'present' if os.path.isfile(test) else 'MISSING'}")
sys.exit(0 if vendored and os.path.isfile(test) else 1)
PY
}

# The commit CI tested: HEAD, or, when HEAD and its parents are kit-only '[skip ci]' commits (scripts/ralph/
# marks, gate or prompt fixes after the wave's CI commit), the nearest ancestor that is not.
ci_commit_for_head() {
  local sha; sha=$(git rev-parse HEAD)
  while git log -1 --format=%s "$sha" | grep -q 'skip ci' && [ -z "$(git show --format= --name-only "$sha" | grep -v '^scripts/ralph/')" ]; do
    sha=$(git rev-parse "$sha^")
  done
  echo "$sha"
}
# Required jobs = test.yml's CURRENT jobs (read on origin/main 7ae677b52, 20 Oct). This wave touches orchestrator/
# (backend_mode full → the shards, orchestrator-tests, Alembic from-zero, Schema-drift, the module + integration tests)
# and frontend/ + route-manifest.json (Frontend CI). Code standards runs on every PR. Path-gated, green or skipped:
# "Prod images built" and "CLI host (PRD-234)" (services/cli-host/: untouched by this wave).
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
# The wave's PR body records the exception it uses, the approvals, the stack and the PRD's PR-body ACs.
pr_body_records() {
  local br n body s miss=0
  br=$(git rev-parse --abbrev-ref HEAD)
  n=$(gh pr list --head "$br" --state open --json number --jq '.[0].number' 2>/dev/null)
  [ -n "$n" ] || { echo "   no open PR for $br"; return 1; }
  body=$(gh pr view "$n" --json body --jq .body 2>/dev/null)
  for s in "AGENTS.md exception 2" "services/push/expo.py" "#1254" "#1255" "#1253" "Stacked on" "mobile_push_tickets" "mobile_digest_posts" "daily_digest" "K12" "AI assistance"; do
    printf '%s' "$body" | grep -qF -- "$s" || { echo "   the PR body lacks: $s"; miss=1; }
  done
  [ $miss -eq 0 ] && echo "   PR #$n records exception 2, the approvals (#1254, #1255), the stack on #1253's wave, the two tables, the digest default and the AI disclosure"
}
# The stacked-PR trap (K1): while W1's PR is open this PR targets W1's branch; once W1 merged, main.
pr_base_ok() {
  local br n base w1
  br=$(git rev-parse --abbrev-ref HEAD)
  n=$(gh pr list --head "$br" --state open --json number --jq '.[0].number' 2>/dev/null)
  [ -n "$n" ] || { echo "   no open PR for $br"; return 1; }
  base=$(gh pr view "$n" --json baseRefName --jq .baseRefName 2>/dev/null)
  w1=$(gh pr list --head "$W1_BRANCH" --state all --json state --jq '.[0].state' 2>/dev/null)
  echo "   PR #$n base: $base; W1's PR: ${w1:-none}"
  case "$w1" in
    MERGED) [ "$base" = "main" ] ;;
    *) [ "$base" = "$W1_BRANCH" ] || [ "$base" = "main" ] ;;
  esac
}

if [ "$MODE" != "negative" ]; then
  echo ""; echo "=== Seeded and stacked"
  check "the PRD is tracked at $PRD_DOC" "git ls-files --error-unmatch $PRD_DOC >/dev/null 2>&1"
  check "W1's work is under this wave (mobile_devices, the device lane, services/mobile/devices.py)" "all_exist $O/core/models/mobile_devices.py $O/core/auth/device.py $MOBILE/devices.py $O/api/mobile_devices.py"

  echo ""; echo "=== Stories (priority order)"
  check "US-010a the PushSender protocol and ExpoPushSender (Expo's send URL, batches of 100, exception 2 named)" "has 'class PushSender' $PUSH/sender.py && has 'exp\.host/--/api/v2/push/send' $PUSH/expo.py && has 'getReceipts' $PUSH/expo.py && has 'EXPO_SEND_BATCH *= *100' $PUSH/expo.py && has 'exception 2' $PUSH/expo.py"
  check "US-010a the two sender settings are in config.py and config-surface.json; the sender test is new" "settings_declared $SETTINGS_SENDER && new_files $TESTS/test_prd266_push_expo_sender.py"
  check "US-010b exactly one new Alembic revision" "one_revision"
  check "US-010b $NEW_REV chains onto the base head with both tables; both head pins moved" "revision_ok"
  check "US-010b the models are declared and registered; the test is new" "has '__tablename__ = .mobile_push_tickets.' $O/core/models/mobile_push.py && has '__tablename__ = .mobile_digest_posts.' $O/core/models/mobile_push.py && has 'mobile_push' $O/core/models/__init__.py && new_files $TESTS/test_prd266_push_tables_model.py"
  check "US-011a PUSH_TYPE_EVENTS names exactly the five push events (FR-7)" "five_push_events"
  check "US-011a the payload signs v2 with HMAC-SHA256 (the canonical array, the freshness window) and the AI badge; the ONE shared vector file and the tests are new" "has 'hmac' $PUSH/payload.py && has 'sha256' $PUSH/payload.py && has 'badge_for' $PUSH/payload.py && has 'compare_digest' $PUSH/payload.py && has 'PUSH_SIG_MAX_AGE_SECONDS' $PUSH/payload.py && has 'PUSH_SIG_MAX_FUTURE_SECONDS' $PUSH/payload.py && has \"'v2'|\\\"v2\\\"\" $PUSH/payload.py && new_files $TESTS/fixtures/prd266_push_signature_vectors.json $TESTS/test_prd266_push_payload.py && v2_vectors_ok"
  check "US-011a the GDPR export drops the device secrets (SECRET_COLUMNS and the Expo token); its test is new" "has 'EXPORT_REDACTED_COLUMNS' $O/services/gdpr_service.py && has 'SECRET_COLUMNS' $O/services/gdpr_service.py && has 'expo_push_token' $O/services/gdpr_service.py && new_files $TESTS/test_prd266_gdpr_export_device_secrets.py"
  check "US-010c the consent lives in system settings (not config.py); the router, RouterSpec, discovery follows it" "has 'push_via_expo' $PUSH/consent.py && has 'get_system_setting' $PUSH/consent.py && setting_not_in_config PUSH_VIA_EXPO && all_exist $ROUTER && router_spec api.mobile_push && added_in expo_push_allowed $MOBILE/discovery.py"
  check "US-010c GET and PUT /api/v1/mobile/push-consent are in route-manifest.json; the docs; the test is new" "routes_in_manifest 'GET /api/v1/mobile/push-consent' 'PUT /api/v1/mobile/push-consent' && for d in QUICKSTART.md SETUP.md; do has \"Expo's push service\" \$d || { echo \"   \$d lacks the notifications paragraph\"; exit 1; }; done && new_files $TESTS/test_prd266_push_consent.py"
  check "US-011b recipients and fan-out: hand_over through core.best_effort; the test is new" "has 'def recipients' $PUSH/recipients.py && has 'def hand_over' $PUSH/fanout.py && has 'off_loop' $PUSH/fanout.py && has 'expo_push_allowed' $PUSH/fanout.py && has 'DeviceNotRegistered' $PUSH/fanout.py && new_files $TESTS/test_prd266_push_fanout.py"
  check "US-010d dispatch is at most 50 code lines; _insert_in_app returns the row id; the push leg calls hand_over" "fn_short $DISPATCHER dispatch 50 && grep -q 'RETURNING id' $DISPATCHER && has 'hand_over' $DISPATCHER"
  check "US-010d push is a valid destination in preferences and auto-reporting, never an _EXTERNAL_DESTINATIONS channel" "set_has $NOTIFY_API _VALID_DESTINATIONS push && set_has $O/core/services/auto_reporting.py _VALID_DESTINATIONS push && set_lacks $DISPATCHER _EXTERNAL_DESTINATIONS push && set_lacks $O/core/services/auto_reporting.py _VALID_CHANNELS push"
  check "US-010d the dispatcher's existing tests are untouched; the push test is new" "frozen_tests_unchanged && new_files $TESTS/test_prd266_dispatcher_push.py"
  check "US-010e the receipts job registers on the leader's scheduler; its setting; the test is new" "has 'def register' $PUSH/receipts.py && has 'def run_receipts' $PUSH/receipts.py && grep -qE 'services\.push\.receipts|from services\.push import receipts' $O/services/schedule_reconcile.py && settings_declared PUSH_RECEIPTS_INTERVAL_SECONDS && new_files $TESTS/test_prd266_push_receipts.py"
  check "US-009 registration: the token handover and push_token_missing; the optional device name; discovery lists push; the test is new" "has 'token_moved' $MOBILE/devices.py && has 'push_token_missing' $MOBILE/devices.py && device_name_ok && added_in push $MOBILE/discovery.py && new_files $TESTS/test_prd266_push_registration.py"
  check "US-012a review_pending: the event, the ORM listener, its registration, the sweeper hook, the dispatch; the test is new" "set_has $DISPATCHER VALID_EVENT_TYPES review_pending && has 'BoardTask\.status' $O/core/models/review_entries.py && has 'after_commit' $O/core/models/review_entries.py && has 'review_entries' $O/core/models/__init__.py && has 'review_pending' $O/services/board_dispatcher.py && has 'def send_soon' $EVENTS/soon.py && has 'def announce' $EVENTS/review_pending.py && new_files $TESTS/test_prd266_review_pending.py"
  check "US-012a requeue_expired_leases is byte-identical to the base" "fn_unchanged $O/services/board_dispatcher.py requeue_expired_leases"
  check "US-012b auto_message: the event, the allow-listed origins, the hook in deliver_background_message; the test is new" "set_has $DISPATCHER VALID_EVENT_TYPES auto_message && has 'ANNOUNCED_ORIGINS' $EVENTS/auto_message.py && has 'AUTO_MESSAGE_COALESCE_SECONDS' $EVENTS/auto_message.py && grep -qE 'auto_message' $O/services/chat_messenger.py && new_files $TESTS/test_prd266_auto_message.py"
  check "US-012b post_background_message is byte-identical to the base" "fn_unchanged $O/services/chat_messenger.py post_background_message"
  check "US-012b the bell links a chat; the drift-guard fixture names it" "grep -qE \"case 'chat'\" frontend/components/notifications/notification-bell.tsx && grep -qE \"'chat'\" frontend/components/notifications/__tests__/notification-bell-links.test.ts && ! git diff --quiet $BASE..HEAD -- frontend/components/notifications/__tests__/notification-bell-links.test.ts"
  check "US-014a push preferences: the daily_digest switch, the defaults, push_types; the test is new" "set_has $DISPATCHER VALID_EVENT_TYPES daily_digest && has 'def with_push_defaults' $PUSH/preferences.py && has 'push_types' $NOTIFY_API && new_files $TESTS/test_prd266_push_preferences.py"
  check "US-014a bulk_update_preferences is byte-identical to the base" "fn_unchanged $NOTIFY_API bulk_update_preferences"
  check "US-013 the digest job: the claim table, the scheduler registration, its settings, discovery lists digest; the test is new" "has 'mobile_digest_posts|MobileDigestPost' $MOBILE/digest.py && has 'generate_digest' $MOBILE/digest.py && has 'daily_digest' $MOBILE/digest.py && grep -qE 'services\.mobile\.digest|from services\.mobile import digest' $O/services/schedule_reconcile.py && settings_declared DIGEST_JOB_INTERVAL_SECONDS DIGEST_DEFAULT_TIME && added_in digest $MOBILE/discovery.py && new_files $TESTS/test_prd266_digest_job.py"
  check "US-014b PUT /api/v1/mobile/push-settings (workspace:manage) is in route-manifest.json; the test is new" "has 'redact_questions' $ROUTER && has 'workspace:manage' $ROUTER && routes_in_manifest 'PUT /api/v1/mobile/push-settings' && new_files $TESTS/test_prd266_push_redaction.py"
  check "US-014b get_current_workspace is byte-identical to the base (api/workspaces.py untouched)" "git diff --quiet $BASE..HEAD -- $O/api/workspaces.py && fn_unchanged $O/api/workspaces.py get_current_workspace"
  check "US-019a Studio asks once and keeps a switch: the hook and the test are new" "all_exist frontend/hooks/use-push-consent.ts && grep -qF '/api/v1/mobile/push-consent' frontend/hooks/use-push-consent.ts && new_files $PD/__tests__/push-consent.test.tsx"
  check "US-019b owner_label and push states: the tests are new" "has 'owner_label' $MOBILE/devices.py && new_files $TESTS/test_prd266_devices_owner_label.py $PD/__tests__/phones-devices-push.test.tsx"
  check "US-019c the QR image follows K17 (OWNER marks and no encoder while not approved)" "qr_story_ok"
  check "US-019d the review pairing code: --review/--ttl-hours bounded by REVIEW_CODE_MAX_HOURS, audited; the test is new" "has 'REVIEW_CODE_MAX_HOURS' $O/scripts/mint_pairing_code.py && has 'ttl.hours' $O/scripts/mint_pairing_code.py && has 'review' $O/scripts/mint_pairing_code.py && new_files $TESTS/test_prd266_review_pairing_code.py"
  check "every acceptance criterion in $KIT is marked (DONE, or OWNER)" "all_acs_done"
fi

echo ""; echo "=== Scope and conventions"
check "N2 exactly one Alembic head in the tree" "one_head"
check "N3 no env reads added outside config.py" "! env_read_added"
check "N4 no print() added in orchestrator application code (scripts may print)" "! print_added"
check "N5 no raw fetch('/api…') added in the frontend" "! raw_fetch_added"
check "N6 no new Python dependency" "git diff --quiet $BASE..HEAD -- orchestrator/requirements.in orchestrator/requirements.lock orchestrator/requirements-dev.txt orchestrator/requirements-dev.lock"
check "N7 no new npm dependency (package.json and package-lock.json untouched)" "git diff --quiet $BASE..HEAD -- frontend/package.json frontend/package-lock.json"
check "N8 the Composio deny list and the generated Auto skill seed are untouched" "git diff --quiet $BASE..HEAD -- orchestrator/core/composio/deny_list.py orchestrator/core/seeds/platform-management-skill.md"
check "N9 no node_modules in the diff" "! git diff --name-only $BASE..HEAD | grep -q node_modules"
check "N10 every commit is DCO-signed" "every_commit_signed"
check "N11 a new file is at most 800 lines (a vendored encoder under frontend/lib/vendor excepted)" "file_sizes_ok"
check "N12a the §4.10 files over 800 lines did not grow" "named_files_not_grown"
check "N12b no other file over 800 lines grew (config.py and the two reports excepted)" "no_big_file_grew"
warn "N12c config.py grew by settings only (≤ 40 lines)" "config_growth"
check "N13 get_request_context_hybrid is byte-identical to the base (hybrid.py untouched by this wave)" "git diff --quiet $BASE..HEAD -- $O/core/auth/hybrid.py && fn_unchanged $O/core/auth/hybrid.py get_request_context_hybrid"
check "N14 _stream_response_with_agent_scoped is byte-identical to the base" "fn_unchanged $O/consumers/chatbot/service.py _stream_response_with_agent_scoped"
check "N15 no innerHTML / dangerouslySetInnerHTML added in the frontend" "! inner_html_added"
check "N16 the new router's handlers are plain def (sync session)" "[ ! -f $ROUTER ] || router_plain_def"
check "N17 services/push/expo.py is the wave's only third-party HTTP client (AGENTS.md exception 2)" "! client_added"
check "N18 no custom URL scheme added" "! custom_scheme_added"
check "N19 no test weakened: added test functions >= removed" "tests_not_weakened"
check "N20 the hierarchy gate passes (stdlib ast)" "\"$PY\" orchestrator/scripts/check_hierarchy_gate.py >/dev/null"
check "N21 no existing migration modified, deleted or renamed (W1's included)" "old_migrations_untouched"
check "N22 the push package talks to exp.host only" "! foreign_host_added"
check "N23 docker-compose.yml is untouched by this wave" "git diff --quiet $BASE..HEAD -- docker-compose.yml"
check "N24 CI's changed-lines code shape passes on the committed wave (functions ≤ 50, nesting ≤ 4, new files ≤ 800)" "code_shape_ok"
check "N25 no orange-* palette class added (.eslintrc no-restricted-syntax)" "! orange_added"
warn "N26 story commits carry [skip ci] (all but the final CI commit and a merge of main)" "[ -z \"\$(git log --no-merges --format='%s' $BASE..HEAD^ | grep -v 'skip ci')\" ]"
check "N27 no push token, push key, Expo access token or question text in a log call" "! secret_logged"
check "N28 W1's kit files are untouched on this branch" "git diff --quiet $BASE..HEAD -- scripts/ralph/prd-266w1.json scripts/ralph/PROMPT_build_prd266w1.md scripts/ralph/PROMPT_review_prd266w1.md scripts/ralph/acceptance-prd266w1.sh"
check "N29 no W3 surface added (content reports, ai_providers, the moderation webhook, universal links, /go/pair, deletion)" "! w3_added"
check "N30 DEFAULT_NOTIFICATION_PREFERENCES keeps its nine rows (test_prd128_default_prefs untouched)" "git diff --quiet $BASE..HEAD -- $TESTS/test_prd128_default_prefs.py"
check "N31 the dispatcher's existing tests are untouched (they patch its helpers: K3)" "frozen_tests_unchanged"

if [ "$MODE" = "full" ]; then
  echo ""; echo "=== CI and the PR (once per wave)"
  check "CI: the required jobs present and green on the tested commit" "ci_green_on_head"
  check "the PR targets W1's branch while W1's PR is open, main once W1 merged (the stacked-PR trap, K1)" "pr_base_ok"
  check "the PR body records exception 2, the approvals (#1254, #1255), the stack, the two tables, the digest default and the AI disclosure" "pr_body_records"
fi
echo ""
if [ "$FAIL" -eq 0 ]; then echo "✅ PRD-266 W2 Push acceptance ($MODE): PASS"; else echo "❌ PRD-266 W2 Push acceptance ($MODE): FAIL"; fi
exit "$FAIL"
