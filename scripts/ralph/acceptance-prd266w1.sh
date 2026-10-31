#!/bin/bash
# Acceptance gate — PRD-266 W1 Connect, the Automatos Go backend (16 stories: US-003a … US-008b).
#   bash scripts/ralph/acceptance-prd266w1.sh | --print-base | --no-ci | --negative-only ; GATE_BASE=<ref> (smoke runs only).
# Base: the merge-base with origin/main (the wave's PR targets main; the branch is cut from the docs branch of PR #1257,
# whose one commit is the PRD). Nothing runs on the owner's machine: greps, git and stdlib python only; CI once per
# wave on the draft PR. No pipefail (a negative `! producer | grep -q` would invert on SIGPIPE); every check runs in
# its own `( eval … )`; stdin is closed so a grep with a missing file argument fails instead of hanging.
#
# Negative checks and the base (20 Oct 2026, b5628af98 = origin/main 630fc1c85 + the PRD commit):
#   - every "added lines" check reads `git diff BASE..HEAD`, EMPTY when HEAD is the base, so it passes there by
#     construction; each is scoped to code dirs and excludes tests, the kit (scripts/ralph) and docs, so a check's own
#     pattern line or a test asserting absence never trips it;
#   - N2 (exactly one Alembic head), N13/N14 (the two long functions byte-identical to the base), N20 (the hierarchy
#     gate) and N24 (CI's changed-lines code-shape script) were smoke-run on the base: green;
#   - the story checks are POSITIVE (the wave's work): they fail on the base by design (no core/auth/pairing.py, no
#     mobile_devices, no /api/v1/mobile routes, no discovery, no publish variables there; read 20 Oct).
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

KIT=scripts/ralph/prd-266w1.json
PRD_DOC=docs/PRDS/PRD-266-AUTOMATOS-GO-BACKEND.md
O=orchestrator
TESTS=$O/tests
VERSIONS=$O/alembic/versions
MANIFEST=$O/reports/route-manifest.json
SURFACE=$O/reports/config-surface.json
AUTH=$O/core/auth
MOBILE=$O/services/mobile
ROUTER=$O/api/mobile_devices.py
WELL_KNOWN=$O/api/well_known.py
PD=frontend/components/settings/phones-devices
NEW_REV=prd266_mobile_devices
SETTINGS="LOCAL_REMOTE_ACCESS API_PUBLISH_ADDRESS WEB_PUBLISH_ADDRESS MOBILE_TOKEN_ROTATE_DAYS MOBILE_TOKEN_ROTATION_OVERLAP_SECONDS"
# §4.10: files over 800 lines that may not grow (the PRD's five, and the two others this wave may touch).
NO_GROW="$O/core/auth/hybrid.py $O/api/chat.py $O/api/approval_grants.py $O/main.py $O/services/cli_host_service.py $O/consumers/chatbot/service.py frontend/lib/api-client.ts"
# Over 800 lines and allowed to grow: settings live only in config.py (AGENTS.md); the two reports are generated lists.
GROW_OK="$O/config.py $MANIFEST $SURFACE"
# US-003a: the CLI-host tests stay byte-identical (the compose bind test changes in US-005a only, K10).
CLI_HOST_TESTS="$TESTS/test_prd234_s1a_cli_hosts_realdb.py $TESTS/test_prd234_s1a_cli_hosts_api.py $TESTS/test_prd245_session_token_realdb.py $TESTS/test_848_session_files_reach_the_cluster.py $TESTS/test_848_session_files_reach_the_cluster_realdb.py $TESTS/test_prd235_host_always_on.py $TESTS/helpers_mission_lane.py"
COMPOSE_BIND_TEST=$TESTS/test_prd234_s1a_compose_bind.py

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
client_added()       { added_code "$AUTH" "$MOBILE" "$ROUTER" "$WELL_KNOWN" | grep -E '^\+\s*(import|from)\s+(httpx|requests|aiohttp|urllib3)\b' | grep -q .; }
# Identity is a credential, never an address (§4.3): the auth code the wave adds never reads a source address.
address_trust_added() { added_code "$AUTH" | grep -E 'request\.client|[Xx]-[Ff]orwarded-[Ff]or|[Xx]-[Rr]eal-[Ii][Pp]|get_client_ip' | grep -q .; }
# Mirrors .github/workflows/dco.yml, which checks `git rev-list --no-merges`: a merge commit is exempt.
every_commit_signed() { [ -z "$(git log --no-merges --format='%H %(trailers:key=Signed-off-by,valueonly)' "$BASE"..HEAD | awk 'NF<2{print $1}')" ]; }
file_sizes_ok() { local f n bad=0; for f in $(added_files orchestrator frontend docs services | grep -E '\.(py|ts|tsx|js|jsx|mjs|css|md|json|ya?ml|sh)$'); do [ -f "$f" ] || continue; n=$(wc -l < "$f" | tr -d ' '); [ "$n" -le 800 ] || { echo "   $f: $n lines"; bad=1; }; done; [ $bad -eq 0 ]; }
not_grown() { local f="$1" b h; b=$(lines_at "$BASE" "$f"); h=$(wc -l < "$f" | tr -d ' '); echo "   $f: base $b, HEAD $h"; [ "$h" -le "$b" ]; }
named_files_not_grown() { local f bad=0; for f in $NO_GROW; do [ -f "$f" ] || continue; not_grown "$f" || bad=1; done; [ $bad -eq 0 ]; }
# Every modified file that was over 800 lines on the base, outside GROW_OK, has not grown.
no_big_file_grew() { local f b h bad=0; for f in $(git diff --name-only --diff-filter=M "$BASE"..HEAD); do
    [ -f "$f" ] || continue
    case " $GROW_OK " in *" $f "*) continue ;; esac
    b=$(lines_at "$BASE" "$f"); h=$(wc -l < "$f" | tr -d ' ')
    [ "$b" -gt 800 ] && [ "$h" -gt "$b" ] && { echo "   $f grew: base $b, HEAD $h"; bad=1; }
  done; [ $bad -eq 0 ]; }
config_growth() { local b h; b=$(lines_at "$BASE" $O/config.py); h=$(wc -l < $O/config.py | tr -d ' '); echo "   config.py: base $b, HEAD $h (settings only: at most ~4 lines per setting)"; [ "$h" -le $((b + 40)) ]; }
tests_not_weakened() { local d a; d=$(git diff "$BASE"..HEAD -- $TESTS frontend | grep -cE '^-\s*(async )?def test_|^-\s*(it|test)\('); a=$(git diff "$BASE"..HEAD -- $TESTS frontend | grep -cE '^\+\s*(async )?def test_|^\+\s*(it|test)\('); echo "   test functions removed $d, added $a"; [ "$a" -ge "$d" ]; }
test_count_kept() { local f="$1" b h; b=$(git show "$BASE:$f" 2>/dev/null | grep -cE '^def test_'); h=$(grep -cE '^def test_' "$f"); echo "   $f: test functions base $b, HEAD $h"; [ "$h" -ge "$b" ]; }
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

# --- US-003a: the kernel ------------------------------------------------------------------------
kernel_ok() { "$PY" - "$AUTH/pairing.py" "$O/services/cli_host_service.py" <<'PY'
import ast, sys
kernel_path, cli_path = sys.argv[1:3]
def names(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    defs = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assigns = {t.id for n in tree.body if isinstance(n, (ast.Assign, ast.AnnAssign))
               for t in (n.targets if isinstance(n, ast.Assign) else [n.target]) if isinstance(t, ast.Name)}
    imports = {n.module for n in tree.body if isinstance(n, ast.ImportFrom)}
    return defs, assigns, imports
try:
    kdefs, kassigns, _ = names(kernel_path)
except OSError:
    print(f"   {kernel_path} is missing"); sys.exit(1)
bad = [f"kernel lacks {n}" for n in ("hash_secret", "normalize_pairing_code", "new_pairing_code", "mint_token", "secrets_match") if n not in kdefs]
bad += [f"kernel lacks {n}" for n in ("PAIRING_ALPHABET", "PAIRING_CODE_TTL_SECONDS", "PAIRING_CODE_LENGTH") if n not in kassigns]
cdefs, cassigns, cimports = names(cli_path)
bad += [f"cli_host_service.py still defines {n}" for n in ("hash_secret", "normalize_pairing_code", "_new_pairing_code", "new_pairing_code") if n in cdefs]
bad += [f"cli_host_service.py still assigns {n}" for n in ("_PAIRING_ALPHABET", "PAIRING_ALPHABET", "PAIRING_CODE_TTL_SECONDS") if n in cassigns]
if "core.auth.pairing" not in cimports:
    bad.append("cli_host_service.py does not import from core.auth.pairing")
print("\n".join(f"   {b}" for b in bad) or "   the kernel holds the primitives; the CLI-host copies are gone"); sys.exit(1 if bad else 0)
PY
}
cli_host_tests_unchanged() { local f bad=0; for f in $CLI_HOST_TESTS; do git diff --quiet "$BASE"..HEAD -- "$f" || { echo "   edited: $f"; bad=1; }; done; [ $bad -eq 0 ]; }

# --- US-002: the migration ----------------------------------------------------------------------
base_head() { git show "$BASE:$TESTS/test_prd209_alembic_single_head.py" 2>/dev/null | grep -E '^EXPECTED_HEAD *= *"' | head -1 | sed -E 's/.*"([^"]+)".*/\1/'; }
new_revisions() { added_files "$VERSIONS" | grep '\.py$'; }
one_revision() { local n; n=$(new_revisions | grep -c .); echo "   new revisions: $n ($(new_revisions | tr '\n' ' '))"; [ "$n" -eq 1 ]; }
revision_ok() {
  local f bh c; f="$VERSIONS/$NEW_REV.py"; bh=$(base_head)
  [ -f "$f" ] || { echo "   $f is missing"; return 1; }
  [ -n "$bh" ] || { echo "   cannot read the base's EXPECTED_HEAD"; return 1; }
  grep -qE "^revision[^=]*= *['\"]$NEW_REV['\"]" "$f" || { echo "   $f: revision is not $NEW_REV"; return 1; }
  grep -qE "^down_revision[^=]*= *['\"]$bh['\"]" "$f" || { echo "   $f: down_revision is not the base head $bh"; return 1; }
  grep -qiE 'CREATE TABLE IF NOT EXISTS mobile_devices' "$f" || { echo "   $f: no CREATE TABLE IF NOT EXISTS mobile_devices"; return 1; }
  [ "$(grep -ciE 'CREATE (TABLE|(UNIQUE )?INDEX)' "$f")" -le "$(grep -ciE 'IF NOT EXISTS' "$f")" ] || { echo "   $f: a CREATE without IF NOT EXISTS"; return 1; }
  for c in workspace_id user_id kind install_id platform status pairing_code_hash pairing_expires_at token_hash prev_token_hash prev_token_valid_until expo_push_token push_enabled push_key last_seen_at revoked_at events digest_time timezone; do
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

# --- routes, settings, compose ------------------------------------------------------------------
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
router_spec() { grep -qE "RouterSpec\(\"$1\"" $O/router_manifest.py; }
compose_ok() { "$PY" - docker-compose.yml <<'PY'
import re, sys
BIND = "${BIND_ADDRESS:-127.0.0.1}"
API = "${API_PUBLISH_ADDRESS:-" + BIND + "}"
WEB = "${WEB_PUBLISH_ADDRESS:-" + BIND + "}"
text = open(sys.argv[1], encoding="utf-8").read()
ports, svc, in_ports = {}, None, False
for line in text.splitlines():
    if re.match(r"^\S", line):
        svc, in_ports = None, False; continue
    m = re.match(r"^  ([A-Za-z0-9_.-]+):\s*$", line)
    if m:
        svc, in_ports = m.group(1), False; continue
    if svc is None:
        continue
    if re.match(r"^    ports:\s*$", line):
        in_ports = True; continue
    if in_ports:
        pm = re.match(r'^\s+-\s+"?([^"#]+?)"?\s*(#.*)?$', line)
        if pm and line.startswith("      "):
            ports.setdefault(svc, []).append(pm.group(1)); continue
        if re.match(r"^    \S", line):
            in_ports = False
want = {"backend": API, "frontend": WEB}
bad = []
for name in ("backend", "frontend", "postgres", "redis", "minio", "qdrant"):
    if name not in ports:
        bad.append(f"no ports read for {name}")
for name, entries in ports.items():
    prefix = want.get(name, BIND)
    for e in entries:
        if not e.startswith(prefix + ":"):
            bad.append(f"{name}: {e!r} does not publish on {prefix}")
        if name not in want and "PUBLISH_ADDRESS" in e:
            bad.append(f"{name}: a datastore follows a publish variable ({e!r})")
env = {
    "API_PUBLISH_ADDRESS": "API_PUBLISH_ADDRESS: " + API,
    "WEB_PUBLISH_ADDRESS": "WEB_PUBLISH_ADDRESS: " + WEB,
    "LOCAL_REMOTE_ACCESS": "LOCAL_REMOTE_ACCESS: ${LOCAL_REMOTE_ACCESS:-off}",
}
bad += [f"the backend's environment lacks {line!r}" for line in env.values() if line not in text]
print("\n".join(f"   {b}" for b in bad) or f"   {sum(len(v) for v in ports.values())} ports read: API and web on their variables, every other port on BIND"); sys.exit(1 if bad else 0)
PY
}
# The datastores never follow a publish variable (also holds on the base: N23).
datastores_on_bind() { ! grep -nE '(API|WEB)_PUBLISH_ADDRESS' docker-compose.yml | grep -vE '^\s*[0-9]+:\s*#' | grep -E ':(5432|6379|9000|9001|6333|8080)"' | grep -q .; }

# The theme membership resolver names the device lane (PERSON_AUTH_TYPES or its own branch).
membership_knows_device() { grep -qE "[\"']device[\"']" "$O/services/themes/membership.py"; }

# --- US-006a: the approval-grant dependencies ---------------------------------------------------
grant_deps_ok() { "$PY" - "$O/api/approval_grants.py" <<'PY'
import ast, sys
tree = ast.parse(open(sys.argv[1], encoding="utf-8").read())
want = {"list_grants": "require_workspace_answerer", "grant_approval": "require_workspace_answerer",
        "deny_approval": "require_workspace_answerer", "answer_question": "require_workspace_answerer",
        "revoke_approval": "require_workspace_admin"}
found = {}
for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in want:
        deps = set()
        for d in list(node.args.defaults) + [k for k in node.args.kw_defaults if k is not None]:
            if isinstance(d, ast.Call) and getattr(d.func, "id", None) == "Depends" and d.args and isinstance(d.args[0], ast.Name):
                deps.add(d.args[0].id)
        found[node.name] = deps
bad = [f"{k}: {sorted(found.get(k, set()))}, want {v}" for k, v in want.items() if v not in found.get(k, set())]
if "require_workspace_answerer" in found.get("revoke_approval", set()):
    bad.append("revoke_approval uses the answerer gate")
print("\n".join(f"   {b}" for b in bad) or "   list/grant/deny/answer: answerer; revoke: admin"); sys.exit(1 if bad else 0)
PY
}
# The two new routers' handlers are plain def (sync session; AGENTS.md).
routers_plain_def() { "$PY" - "$ROUTER" "$WELL_KNOWN" <<'PY'
import ast, sys
bad = []
for path in sys.argv[1:]:
    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
    except OSError:
        continue
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef):
            for d in node.decorator_list:
                f = d.func if isinstance(d, ast.Call) else d
                if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "router":
                    bad.append(f"{path}:{node.lineno} {node.name} is async def")
print("\n".join(f"   {b}" for b in bad) or "   every route handler is a plain def"); sys.exit(1 if bad else 0)
PY
}
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
# Required jobs = test.yml's CURRENT jobs (read on origin/main 630fc1c85, 20 Oct, all green there). This wave touches
# orchestrator/ (backend_mode full → the four shards, orchestrator-tests, Alembic from-zero, Schema-drift, the module +
# integration tests F056) and frontend/ + route-manifest.json (Frontend CI). Code standards runs on every PR. Path-gated,
# green or skipped: "Prod images built" (Dockerfiles, manifests, next.config.*, lib/auth-edition.ts, test.yml) and the
# "CLI host (PRD-234)" lane (services/cli-host/; the wave changes the orchestrator side of pairing only).
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
# The wave's PR body records the AGENTS.md exception it uses, the stop-and-ask approvals and the PRD's PR-body ACs.
pr_body_records() {
  local br n body s miss=0
  br=$(git rev-parse --abbrev-ref HEAD)
  n=$(gh pr list --head "$br" --state open --json number --jq '.[0].number' 2>/dev/null)
  [ -n "$n" ] || { echo "   no open PR for $br"; return 1; }
  body=$(gh pr view "$n" --json body --jq .body 2>/dev/null)
  for s in "AGENTS.md exception" "/.well-known/automatos.json" "/api/v1/mobile/pair" "#1253" "#1257" "PRD-196" "cli_hosts" "AI assistance"; do
    printf '%s' "$body" | grep -qF -- "$s" || { echo "   the PR body lacks: $s"; miss=1; }
  done
  [ $miss -eq 0 ] && echo "   PR #$n records the public-route exception, the approvals (#1253), PRD-196, cli_hosts and the AI disclosure"
}

if [ "$MODE" != "negative" ]; then
  echo ""; echo "=== Seeded"
  check "the PRD is tracked at $PRD_DOC" "git ls-files --error-unmatch $PRD_DOC >/dev/null 2>&1"

  echo ""; echo "=== Stories (priority order)"
  check "US-003a the pairing kernel holds the primitives; the CLI-host copies are deleted" "kernel_ok"
  check "US-003a the CLI-host tests are unchanged; the kernel's test is new" "cli_host_tests_unchanged && new_files $TESTS/test_prd266_pairing_kernel.py"
  check "US-002 exactly one new Alembic revision" "one_revision"
  check "US-002 $NEW_REV chains onto the base head with every §4.2 column; both head pins moved" "revision_ok"
  check "US-002 the model is declared and registered; its test is new" "has '__tablename__ = .mobile_devices.' $O/core/models/mobile_devices.py && has 'mobile_devices' $O/core/models/__init__.py && new_files $TESTS/test_prd266_mobile_devices_model.py"
  check "US-003b pairing service + router, RouterSpec, the universal-link payload, the rate-limit operations" "has 'core\.auth\.pairing|from core\.auth import pairing' $MOBILE/pairing.py && has '/go/pair\?' $MOBILE/pairing.py && all_exist $ROUTER && router_spec api.mobile_devices && grep -qE 'mobile_pair' $O/core/security/rate_limiter.py $ROUTER $MOBILE/*.py"
  check "US-003b POST pairing-codes and POST pair are in route-manifest.json" "routes_in_manifest 'POST /api/v1/mobile/pairing-codes' 'POST /api/v1/mobile/pair'"
  check "US-003b/US-005a the five W1 settings are in config.py and config-surface.json" "settings_declared $SETTINGS"
  check "US-003b the pairing tests are new" "new_files $TESTS/test_prd266_mobile_pairing.py $TESTS/test_prd266_mobile_pairing_routes.py"
  check "US-004a the registry, the lane, the decorator on hybrid; the lane is local-only" "has 'amd_' $AUTH/bearer_registry.py && has 'AUTH_EDITION' $AUTH/device.py && has 'device_revoked' $AUTH/device.py && has 'core\\.auth\\.bearer_registry|from core\\.auth import bearer_registry' $AUTH/hybrid.py && has 'mobile_device' $AUTH/principal.py && grep -qE 'device' $AUTH/dependencies.py"
  check "US-004a the device-lane test is new" "new_files $TESTS/test_prd266_device_lane.py"
  check "US-003c the device routes are in route-manifest.json" "routes_in_manifest 'POST /api/v1/mobile/devices' 'GET /api/v1/mobile/devices' 'PATCH /api/v1/mobile/devices/\{[a-z_]+\}' 'DELETE /api/v1/mobile/devices/\{[a-z_]+\}' 'POST /api/v1/mobile/devices/\{[a-z_]+\}/rotate'"
  check "US-003c the devices service and its test are new" "all_exist $MOBILE/devices.py && new_files $TESTS/test_prd266_mobile_devices_routes.py"
  check "US-004b every resolver knows device" "membership_knows_device && has 'resolve_user_pk' $O/api/notifications.py && has 'resolve_user_pk|raw_claims' $O/modules/voice/caller.py && has 'resolve_user_pk' $AUTH/workspace_admin.py && has 'device' $AUTH/workspace_permission.py"
  check "US-004b the resolver and integration tests are new" "new_files $TESTS/test_prd266_device_resolvers.py $TESTS/test_prd266_device_integration.py"
  check "US-005a compose: API and web on their publish variables, every other port on BIND, the three variables reach the backend" "compose_ok"
  check "US-005a remote_access, the boot check in validate_auth_edition, the paired-mode closure, the console script" "all_exist $MOBILE/remote_access.py $O/scripts/mint_pairing_code.py && has 'remote_access' $O/config.py && grep -rqE 'device_pairing_required' $AUTH $MOBILE"
  check "US-005a the compose bind test kept every test function; the remote-access tests are new" "test_count_kept $COMPOSE_BIND_TEST && new_files $TESTS/test_prd266_remote_access.py $TESTS/test_prd266_remote_access_compose.py $TESTS/test_prd266_mint_pairing_code.py"
  check "US-001 discovery: router, RouterSpec, service, the v1 schema, the manifest" "all_exist $WELL_KNOWN && router_spec api.well_known && has 'automatos-discovery/v1' $MOBILE/discovery.py && routes_in_manifest 'GET /\.well-known/automatos\.json'"
  check "US-001 the discovery test is new" "new_files $TESTS/test_prd266_discovery.py"
  check "US-005b pair a browser: the local auth provider holds the device token; the pair and discovery calls go through apiClient" "! git diff --quiet $BASE..HEAD -- frontend/components/local-auth-provider.tsx && grep -rqF '/api/v1/mobile/pair' frontend/components frontend/hooks frontend/lib && grep -rqF '/.well-known/automatos.json' frontend/components frontend/hooks frontend/lib && new_files $PD/__tests__/pair-browser.test.tsx"
  check "US-005c Settings → Phones & devices in both editions, the hook, the test" "has 'phones-devices' frontend/components/settings/SettingsPanel.tsx && all_exist frontend/hooks/use-mobile-devices.ts && new_files $PD/__tests__/phones-devices-tab.test.tsx"
  check "US-005c the self-hosting docs section in QUICKSTART.md and SETUP.md" "for d in QUICKSTART.md SETUP.md; do has 'Reach your install from your phone' \$d && has 'LOCAL_REMOTE_ACCESS=paired' \$d && has 'API_PUBLISH_ADDRESS' \$d && has 'mint_pairing_code' \$d || { echo \"   \$d lacks the section\"; exit 1; }; done"
  check "US-006a list/grant/deny/answer use require_workspace_answerer; revoke stays admin; needs-you follows" "has 'def require_workspace_answerer|async def require_workspace_answerer' $AUTH/workspace_admin.py && grant_deps_ok && grep -qE 'may_answer_workspace_questions|require_workspace_answerer' $O/api/activity.py && ! grep -q 'may_answer=may_see_own_workspace_health' $O/api/activity.py"
  check "US-006a the editors-answer test is new" "new_files $TESTS/test_prd266_editors_answer.py"
  check "US-006b the web: no admin-only copy for editors; the test is new" "! grep -q 'You may not be a workspace admin' frontend/components/command-center/questions-tab.tsx && new_files frontend/components/command-center/__tests__/prd266-editors-answer.test.tsx"
  check "US-007a ChatRequest lives in api/chat_models.py with spokenMode; api/chat.py has no copy and shrank" "has 'class ChatRequest' $O/api/chat_models.py && has 'spokenMode' $O/api/chat_models.py && ! grep -qE '^class (ChatRequest|MessagePart|ChatMessageRequest)\b' $O/api/chat.py && [ \$(wc -l < $O/api/chat.py) -lt \$(git show $BASE:$O/api/chat.py | wc -l) ]"
  check "US-007a discovery lists spoken_mode; the spoken-mode test is new" "has 'spoken_mode' $MOBILE/discovery.py && new_files $TESTS/test_prd266_chat_spoken_mode.py"
  check "US-007b _stream_response_with_agent_scoped is byte-identical to the base; history pages with starting_after" "fn_unchanged $O/consumers/chatbot/service.py _stream_response_with_agent_scoped && has 'starting_after' $O/api/chat.py"
  check "US-007b the backend and frontend tests are new" "new_files $TESTS/test_prd266_chat_agent_info_history.py frontend/lib/chat/__tests__/prd266-agent-info-frame.test.ts"
  check "US-008a the Business gate, the transport fields, mint_web_call ≤ 50 code lines, discovery lists voice_live" "grep -rqF 'Auto Live is on the Business plan' $O/api/voice_retell.py $O/modules/voice && has 'ice_servers' $O/modules/voice/retell_api.py && fn_short $O/api/voice_retell.py mint_web_call 50 && has 'voice_live' $MOBILE/discovery.py"
  check "US-008a the voice-mint test is new" "new_files $TESTS/test_prd266_voice_mint.py"
  check "US-008b the live statuses are one constant; the socket refuses the rest; the test is new" "has 'LIVE_CALL_STATUSES' $O/modules/voice/live_socket.py && grep -qE 'LIVE_CALL_STATUSES|is_live|call_is_live' $O/api/voice_retell.py && new_files $TESTS/test_prd266_voice_live_only.py"
  check "every acceptance criterion in $KIT is DONE" "all_acs_done"
fi

echo ""; echo "=== Scope and conventions"
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
check "N12a the §4.10 files over 800 lines did not grow" "named_files_not_grown"
check "N12b no other file over 800 lines grew (config.py and the two reports excepted)" "no_big_file_grew"
warn "N12c config.py grew by settings only (≤ 40 lines)" "config_growth"
check "N13 get_request_context_hybrid is byte-identical to the base" "fn_unchanged $AUTH/hybrid.py get_request_context_hybrid"
check "N14 _stream_response_with_agent_scoped is byte-identical to the base" "fn_unchanged $O/consumers/chatbot/service.py _stream_response_with_agent_scoped"
check "N15 no innerHTML / dangerouslySetInnerHTML added in the frontend" "! inner_html_added"
check "N16 the new routers' handlers are plain def (sync session)" "routers_plain_def"
check "N17 no third-party HTTP client added in the auth, mobile and new router code (exception 2 is W2's)" "! client_added"
check "N18 no custom URL scheme added (the QR payload is a universal link)" "! custom_scheme_added"
check "N19 no test weakened: added test functions >= removed" "tests_not_weakened"
check "N20 the hierarchy gate passes (stdlib ast)" "\"$PY\" orchestrator/scripts/check_hierarchy_gate.py >/dev/null"
check "N21 no existing migration modified, deleted or renamed" "old_migrations_untouched"
check "N22 identity never reads a source address (core/auth added lines)" "! address_trust_added"
check "N23 no datastore port follows a publish variable" "datastores_on_bind"
check "N24 CI's changed-lines code shape passes on the committed wave (functions ≤ 50, nesting ≤ 4, new files ≤ 800)" "code_shape_ok"
check "N25 no orange-* palette class added (.eslintrc no-restricted-syntax)" "! orange_added"
warn "N26 story commits carry [skip ci] (all but the final CI commit)" "[ -z \"\$(git log --format='%s' $BASE..HEAD^ | grep -v 'skip ci' | grep -v '^docs(prd-266): Automatos Go backend')\" ]"

if [ "$MODE" = "full" ]; then
  echo ""; echo "=== CI and the PR (once per wave)"
  check "CI: the required jobs present and green on the tested commit" "ci_green_on_head"
  check "the PR body records the AGENTS.md exception, the approvals (#1253, #1257), PRD-196, cli_hosts and the AI disclosure" "pr_body_records"
fi
echo ""
if [ "$FAIL" -eq 0 ]; then echo "✅ PRD-266 W1 Connect acceptance ($MODE): PASS"; else echo "❌ PRD-266 W1 Connect acceptance ($MODE): FAIL"; fi
exit "$FAIL"
