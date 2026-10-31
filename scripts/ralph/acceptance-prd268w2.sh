#!/bin/bash
# Acceptance gate — PRD-268 W2 Delivery, Proactive Auto reaches people (24 stories: US-C19a … US-W05).
#   bash scripts/ralph/acceptance-prd268w2.sh | --print-base | --no-ci | --negative-only ; GATE_BASE=<ref> (smoke runs only).
# Base (K1): the wave is STACKED on W1 (feat/prd-268-w1-check-ins). While HEAD sits on W1's work, the base is the W1
# commit the branch grew from (merge-base with origin/feat/prd-268-w1-check-ins), so W1's own diff is never read as
# this wave's; once W1 has landed on main and main was merged in (or W1's branch is gone), the base is the merge-base
# with origin/main. Nothing runs on the owner's machine: greps, git and stdlib python only; CI once per wave on the
# draft PR. No pipefail (a negative `! producer | grep -q` would invert on SIGPIPE); every check runs in its own
# `( eval … )`; stdin is closed so a grep with a missing file argument fails instead of hanging.
# Derived from acceptance-prd268w1.sh and acceptance-prd266w2.sh (28 Oct 2026): the generic helpers are unchanged;
# the stacked base is 266w2's; the story and scope checks are PRD-268 W2's.
#   - every "added lines" check reads `git diff BASE..HEAD`, EMPTY when HEAD is the base, so it passes there by
#     construction; each is scoped to code dirs and excludes tests, the kit (scripts/ralph) and docs;
#   - N2, the byte-identical function checks, N20 and N24 hold on the base by construction;
#   - the story checks are POSITIVE (the wave's work): they fail on the base by design.
set -u
exec </dev/null
cd "$(dirname "$0")/../.." || exit 1
export DATABASE_URL="postgresql://ralph:ralph@127.0.0.1:1/ralph_no_db"
export POSTGRES_HOST=127.0.0.1 POSTGRES_PORT=1 POSTGRES_USER=ralph POSTGRES_PASSWORD=ralph POSTGRES_DB=ralph_no_db
export REDIS_URL="redis://127.0.0.1:1/0" REDIS_HOST=127.0.0.1 REDIS_PORT=1
export ENVIRONMENT=development
# Homebrew's python (3.14) when present: CI's code-shape script needs 3.10+ (ast.Match); macOS's /usr/bin is 3.9.
PY=$(command -v /opt/homebrew/bin/python3 || command -v python3)

STACK_REF=origin/feat/prd-268-w1-check-ins
W1_BRANCH=feat/prd-268-w1-check-ins
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

KIT=scripts/ralph/prd-268w2.json
PRD_DOC=docs/PRDS/PRD-268-PROACTIVE-AUTO.md
O=orchestrator
TESTS=$O/tests
VERSIONS=$O/alembic/versions
MANIFEST=$O/reports/route-manifest.json
SURFACE=$O/reports/config-surface.json
PRO=$O/services/proactive
BRIEF=$O/services/brief
PUSH=$O/services/push
DISC=$O/modules/tools/discovery
FMT=$O/modules/tools/formatting
HB=$O/services/heartbeat_service.py
DISPATCHER=$O/core/services/notification_dispatcher.py
DISCOVERY=$O/services/mobile/discovery.py
MESSENGER=$O/services/chat_messenger.py
BRIEF_ROUTER=$O/api/proactive_brief.py
VIEW_ROUTER=$O/api/approval_views.py
FE=frontend
PROC=$FE/components/proactive
GLOBALS=$FE/app/globals.css
NEW_REV=prd268_w2_delivery
W1_HEAD=prd268_w1_proactive
# K3: files over 800 lines this wave may touch; none may grow.
NO_GROW="$FMT/result_formatter.py $DISC/platform_executor.py $O/api/board_tasks.py $O/api/approval_grants.py $O/api/chat.py $O/services/activity_service.py $HB $O/consumers/chatbot/service.py $O/main.py $O/core/llm/manager.py $FE/components/chatbot/chat.tsx $FE/lib/api-client.ts"
# Over 800 lines and allowed to grow: settings live only in config.py; the two reports are generated lists; globals.css
# gains only the --running lines (bounded by N12d).
GROW_OK="$O/config.py $MANIFEST $SURFACE $GLOBALS"
# K19: the pinned tests that move with approved changes; each keeps at least its test count.
PIN_TESTS="$TESTS/test_prd266_push_payload.py $TESTS/test_prd266_push_preferences.py $TESTS/test_prd266_push_tables_model.py $TESTS/test_prd266_mobile_devices_model.py $TESTS/test_prd266_discovery.py $TESTS/test_prd263_harness_review_cadence.py $TESTS/test_prd209_alembic_single_head.py $TESTS/test_prd236_w1_routes.py"
W1_KIT_FILES="scripts/ralph/prd-268w1.json scripts/ralph/PROMPT_build_prd268w1.md scripts/ralph/PROMPT_review_prd268w1.md scripts/ralph/acceptance-prd268w1.sh"

# The wave's ADDED lines under the given paths: tests, the kit and docs excluded (negative greps only).
added_code() { git diff "$BASE"..HEAD -- "$@" ':!orchestrator/tests' ':!*__tests__*' ':!*.test.ts' ':!*.test.tsx' ':!e2e' ':!scripts/ralph' ':!docs' | grep '^+' | grep -v '^+++'; }
added_files() { git diff --name-only --diff-filter=A "$BASE"..HEAD -- "$@"; }
lines_at() { git show "$1:$2" 2>/dev/null | wc -l | tr -d ' '; }
has() { [ -f "$2" ] && grep -qE -e "$1" "$2"; }      # has <ERE> <file>: false (never a stdin read) when the file is missing
all_exist() { local f; for f in "$@"; do [ -f "$f" ] || { echo "   missing: $f"; return 1; }; done; }
# Branch-scoped: each file exists AND was added by this wave (a test that predates it proves nothing about it).
new_files() { local f bad=0; for f in "$@"; do
    if [ ! -f "$f" ]; then echo "   missing: $f"; bad=1
    elif [ -z "$(added_files "$f")" ]; then echo "   not new on this branch: $f"; bad=1; fi
  done; [ $bad -eq 0 ]; }
changed() { ! git diff --quiet "$BASE"..HEAD -- "$@"; }
# added_in <ERE> <file>: a line this wave added to <file> matches (a positive smoke signal; the tests are the proof).
added_in() { git diff "$BASE"..HEAD -- "$2" | grep '^+' | grep -v '^+++' | grep -qE -e "$1"; }

env_read_added()   { added_code orchestrator ':!orchestrator/config.py' | grep -E 'os\.(getenv|environ)' | grep -q .; }
print_added()      { added_code orchestrator ':!orchestrator/scripts' | grep -E '^\+\s*print\(' | grep -q .; }
raw_fetch_added()  { added_code frontend | grep -E "fetch\(\s*['\"\`]/api" | grep -q .; }
inner_html_added() { added_code frontend | grep -E 'dangerouslySetInnerHTML|\.innerHTML\s*=|\.outerHTML\s*=|insertAdjacentHTML|document\.write\(' | grep -q .; }
orange_added()     { added_code frontend | grep -E '\b(bg|text|border|ring|from|to|via|fill|stroke)-orange-[0-9]' | grep -q .; }
hex_added()        { added_code "$PROC" frontend/components/settings frontend/components/voice | grep -E "#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b['\" ;)]" | grep -q .; }
rounded3_added()   { added_code "$PROC" | grep -E '\brounded-3xl\b' | grep -q .; }
autoplay_added()   { added_code frontend | grep -iE '\bautoplay\b|autoPlay\s*[=:]' | grep -q .; }
client_added()     { added_code orchestrator | grep -E '^\+\s*(import|from)\s+(httpx|requests|aiohttp|urllib3)\b' | grep -q .; }
factory_in_lane()  { added_code "$PRO" "$BRIEF" $O/modules/proactive | grep -E 'AgentFactory|agent_factory|execute_with_prompt' | grep -q .; }
# K10 (§12-Q17, Q28): no Always allow control anywhere (the contract's answers type may name it; the tests prove the
# builder never emits it), and the approval buttons keep today's words.
always_allow_added() { added_code frontend/components frontend/lib/theme-sdk/host-cards orchestrator/api | grep -E "Always allow|always[-_]allow['\"]? *[:=(]|/always" | grep -vE '^\+\s*(//|#|\*|/\*)' | grep -q .; }
new_words_added()  { added_code frontend/components/widgets frontend/lib/theme-sdk/host-cards | grep -E "['\">]Allow once['\"<]|['\">]Refuse['\"<]" | grep -q .; }
# K23: no W3/W4 surface.
later_wave_added() {
  [ -n "$(added_files "$O/evals/proactive")" ] && { echo "   an evals runner was added (US-C13)"; return 0; }
  added_code orchestrator frontend | grep -E 'platform_schedule_follow_up|platform_list_follow_ups|platform_cancel_follow_up|platform_offer_replies|/api/auto/calibration' | grep -vE '^\+\s*(//|#|\*)' | grep -q . && { echo "   a W3/W4 name was added"; return 0; }
  return 1
}
# Mirrors .github/workflows/dco.yml, which checks `git rev-list --no-merges`: a merge commit is exempt.
every_commit_signed() { [ -z "$(git log --no-merges --format='%H %(trailers:key=Signed-off-by,valueonly)' "$BASE"..HEAD | awk 'NF<2{print $1}')" ]; }
file_sizes_ok() { local f n bad=0; for f in $(added_files orchestrator frontend docs services e2e | grep -E '\.(py|ts|tsx|js|jsx|mjs|css|md|json|ya?ml|sh)$'); do [ -f "$f" ] || continue; n=$(wc -l < "$f" | tr -d ' '); [ "$n" -le 800 ] || { echo "   $f: $n lines"; bad=1; }; done; [ $bad -eq 0 ]; }
not_grown() { local f="$1" b h; b=$(lines_at "$BASE" "$f"); h=$(wc -l < "$f" | tr -d ' '); echo "   $f: base $b, HEAD $h"; [ "$h" -le "$b" ]; }
named_files_not_grown() { local f bad=0; for f in $NO_GROW; do [ -f "$f" ] || continue; not_grown "$f" || bad=1; done; [ $bad -eq 0 ]; }
# Every modified file that was over 800 lines on the base, outside GROW_OK, has not grown.
no_big_file_grew() { local f b h bad=0; for f in $(git diff --name-only --diff-filter=M "$BASE"..HEAD); do
    [ -f "$f" ] || continue
    case " $GROW_OK " in *" $f "*) continue ;; esac
    b=$(lines_at "$BASE" "$f"); h=$(wc -l < "$f" | tr -d ' ')
    [ "$b" -gt 800 ] && [ "$h" -gt "$b" ] && { echo "   $f grew: base $b, HEAD $h"; bad=1; }
  done; [ $bad -eq 0 ]; }
config_growth() { local b h; b=$(lines_at "$BASE" $O/config.py); h=$(wc -l < $O/config.py | tr -d ' '); echo "   config.py: base $b, HEAD $h (settings only: at most ~5 lines per setting)"; [ "$h" -le $((b + 20)) ]; }
globals_growth() { local b h; b=$(lines_at "$BASE" "$GLOBALS"); h=$(wc -l < "$GLOBALS" | tr -d ' '); echo "   globals.css: base $b, HEAD $h (the --running lines only, at most 8)"; [ "$h" -le $((b + 8)) ]; }
tests_not_weakened() { local d a; d=$(git diff "$BASE"..HEAD -- $TESTS frontend e2e | grep -cE '^-\s*(async )?def test_|^-\s*(it|test)(\.each\([^)]*\))?\('); a=$(git diff "$BASE"..HEAD -- $TESTS frontend e2e | grep -cE '^\+\s*(async )?def test_|^\+\s*(it|test)(\.each\([^)]*\))?\('); echo "   test functions removed $d, added $a"; [ "$a" -ge "$d" ]; }
test_count_kept() { local f="$1" b h; [ -f "$f" ] || { echo "   $f is missing"; return 1; }; b=$(git show "$BASE:$f" 2>/dev/null | grep -cE '^(async )?def test_'); h=$(grep -cE '^(async )?def test_' "$f"); echo "   $f: test functions base $b, HEAD $h"; [ "$h" -ge "$b" ]; }
pin_tests_kept() { local f bad=0; for f in $PIN_TESTS; do test_count_kept "$f" >/dev/null || { test_count_kept "$f"; bad=1; }; done; [ $bad -eq 0 ] && echo "   every K19 pin file keeps its test count"; [ $bad -eq 0 ]; }
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
# fn_has_param <file> <function> <param>: the function takes a parameter of that name (any kind).
fn_has_param() { "$PY" - "$1" "$2" "$3" <<'PY'
import ast, sys
path, name, param = sys.argv[1:4]
try:
    tree = ast.parse(open(path, encoding="utf-8").read())
except OSError:
    print(f"   {path} is missing"); sys.exit(1)
for node in ast.walk(tree):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
        a = node.args
        names = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
        print(f"   {name}({', '.join(names)})"); sys.exit(0 if param in names else 1)
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
# The signed field set and order of the push payload are the base's (K9: no new signed field).
signed_fields_kept() { "$PY" - "$PUSH/payload.py" "$BASE" <<'PY'
import ast, subprocess, sys
path, base = sys.argv[1:3]
def fields(src):
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "SIGNED_TEXT_FIELDS" for t in node.targets):
            return ast.literal_eval(node.value)
    return None
old = fields(subprocess.run(["git", "show", f"{base}:{path}"], capture_output=True, text=True, check=True).stdout)
new = fields(open(path, encoding="utf-8").read())
print(f"   SIGNED_TEXT_FIELDS base {old}, HEAD {new}")
sys.exit(0 if old is not None and old == new else 1)
PY
}
# The push types are at most the four (§6.5.3): never a fifth.
push_types_bounded() { "$PY" - "$PUSH/payload.py" <<'PY'
import ast, sys
allowed = {"needs_answer", "review", "auto_message", "suggestion"}
for node in ast.parse(open(sys.argv[1], encoding="utf-8").read()).body:
    targets = node.targets if isinstance(node, ast.Assign) else ([node.target] if isinstance(node, ast.AnnAssign) else [])
    if any(isinstance(t, ast.Name) and t.id == "PUSH_TYPE_EVENTS" for t in targets):
        table = next((n for n in ast.walk(node.value) if isinstance(n, ast.Dict)), None)
        keys = sorted(k.value for k in table.keys if isinstance(k, ast.Constant))
        print(f"   push types: {keys}")
        sys.exit(0 if set(keys) <= allowed else 1)
print("   PUSH_TYPE_EVENTS not found"); sys.exit(1)
PY
}
# K9: if the switch exists, its default is off (config.py's convention: os.getenv("NAME", "<default>")).
push_switch_off() { local line d; line=$(grep -E '^\s+SUGGESTION_PUSH_ENABLED\b' $O/config.py | head -1)
  [ -z "$line" ] && { echo "   SUGGESTION_PUSH_ENABLED not declared (base)"; return 0; }
  echo "   $line"
  echo "$line" | grep -qE ':\s*bool\s*=\s*False\b' && return 0
  d=$(echo "$line" | sed -nE 's/.*getenv\("SUGGESTION_PUSH_ENABLED", *"([^"]*)".*/\1/p' | tr 'A-Z' 'a-z')
  case "$d" in false|0|no|off) return 0 ;; *) echo "   the default is '${d}', not off"; return 1 ;; esac; }
# chat.tsx changes are import lines only (K4: the Chat component is never touched).
chat_imports_only() { local f=$FE/components/chatbot/chat.tsx; git diff -U0 "$BASE"..HEAD -- "$f" | grep -E '^[+-]' | grep -vE '^(\+\+\+|---)' | grep -vE '^[+-]import ' | grep -q . && { echo "   chat.tsx has a non-import change"; return 1; }; return 0; }

# --- the migration ------------------------------------------------------------------------------
base_head() { git show "$BASE:$TESTS/test_prd209_alembic_single_head.py" 2>/dev/null | grep -E '^EXPECTED_HEAD *= *"' | head -1 | sed -E 's/.*"([^"]+)".*/\1/'; }
new_revisions() { added_files "$VERSIONS" | grep '\.py$'; }
one_revision() { local n; n=$(new_revisions | awk 'NF' | wc -l | tr -d ' '); echo "   new revisions: $n ($(new_revisions | tr '\n' ' '))"; [ "$n" -eq 1 ]; }
revision_ok() {
  local f bh; f="$VERSIONS/$NEW_REV.py"; bh=$(base_head)
  [ -f "$f" ] || { echo "   $f is missing"; return 1; }
  [ -n "$bh" ] || { echo "   cannot read the base's EXPECTED_HEAD"; return 1; }
  grep -qE "^revision[^=]*= *['\"]$NEW_REV['\"]" "$f" || { echo "   $f: revision is not $NEW_REV"; return 1; }
  grep -qE "^down_revision[^=]*= *['\"]$bh['\"]" "$f" || { echo "   $f: down_revision is not the base head $bh"; return 1; }
  grep -qiE "DROP CONSTRAINT IF EXISTS ck_mobile_push_tickets_push_type" "$f" || { echo "   $f: the push-type CHECK is not dropped IF EXISTS"; return 1; }
  grep -qE "'auto_message', *'suggestion'" "$f" || { echo "   $f: the widened CHECK lacks 'suggestion'"; return 1; }
  grep -qiE "ADD COLUMN IF NOT EXISTS push_categories JSONB NOT NULL DEFAULT '\[\]'" "$f" || { echo "   $f: no ADD COLUMN IF NOT EXISTS push_categories JSONB NOT NULL DEFAULT '[]'"; return 1; }
  ! grep -qiE "mobile_digest_posts|proactive_person_settings" "$f" || { echo "   $f touches a table K6 leaves alone"; return 1; }
  grep -q "EXPECTED_HEAD = \"$NEW_REV\"" $TESTS/test_prd209_alembic_single_head.py || { echo "   test_prd209 EXPECTED_HEAD is not $NEW_REV"; return 1; }
  grep -qF "$NEW_REV.py" $TESTS/test_prd236_w1_routes.py || { echo "   test_prd236_w1_routes.py does not read $NEW_REV.py"; return 1; }
  grep -qF "EXPECTED_HEAD = \"$NEW_REV\"" $TESTS/test_prd236_w1_routes.py || { echo "   test_prd236_w1_routes.py does not assert EXPECTED_HEAD = $NEW_REV"; return 1; }
  echo "   $NEW_REV chains onto $bh; the CHECK widened; push_categories added; both pins moved"
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
capability() { grep -qE "\"$1\": *\(" "$DISCOVERY"; }
# The new routers' handlers are plain def (sync session; AGENTS.md). A router not built yet is skipped.
routers_plain_def() { "$PY" - "$BRIEF_ROUTER" "$VIEW_ROUTER" <<'PY'
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
print("\n".join(f"   {b}" for b in bad) or "   every new route handler is a plain def"); sys.exit(1 if bad else 0)
PY
}
# The new mutating route carries require_workspace_permission (K21, the authz sweep's class (a)).
brief_route_gated() { grep -B3 -A3 'brief/open' "$BRIEF_ROUTER" | grep -q 'require_workspace_permission'; }
# The approval view's answers never hold always_allow (K10): the builder names only refuse and allow_once.
view_answers_ok() { has "'refuse'|\"refuse\"" "$FMT/approval_view.py" && has "'allow_once'|\"allow_once\"" "$FMT/approval_view.py"; }
decorated() { "$PY" - "$1" "$2" "$3" <<'PY'
import ast, sys
path, fn, deco = sys.argv[1:4]
for node in ast.walk(ast.parse(open(path, encoding="utf-8").read())):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn:
        names = [ast.unparse(d) for d in node.decorator_list]
        ok = any(deco in n for n in names)
        print(f"   {fn} decorators: {names}"); sys.exit(0 if ok else 1)
print(f"   {fn} not found in {path}"); sys.exit(1)
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

# The commit CI tested: HEAD, or, when HEAD and its parents are kit-only '[skip ci]' commits (scripts/ralph/
# marks, gate or prompt fixes after the wave's CI commit), the nearest ancestor that is not.
ci_commit_for_head() {
  local sha; sha=$(git rev-parse HEAD)
  while git log -1 --format=%s "$sha" | grep -q 'skip ci' && [ -z "$(git show --format= --name-only "$sha" | grep -v '^scripts/ralph/')" ]; do
    sha=$(git rev-parse "$sha^")
  done
  echo "$sha"
}
# wait_for_run <workflow> <sha>: echoes "<id> <conclusion>" of the completed run (60 min cap), or nothing.
wait_for_run() {
  local wf="$1" sha="$2" br waited=0 run
  br=$(git rev-parse --abbrev-ref HEAD)
  while :; do
    run=$(gh run list --branch "$br" --workflow "$wf" --limit 20 --json databaseId,headSha,status,conclusion --jq ".[] | select(.headSha == \"$sha\") | \"\(.databaseId) \(.status) \(.conclusion)\"" 2>/dev/null | head -1)
    if [ -n "$run" ] && [ "$(echo "$run" | cut -d' ' -f2)" = "completed" ]; then echo "$run" | cut -d' ' -f1,3; return 0; fi
    [ "$waited" -ge 3600 ] && return 1
    sleep 60; waited=$((waited + 60))
  done
}
# Required jobs = test.yml's CURRENT jobs. This wave touches orchestrator/ and frontend/, so Frontend CI must be
# green (not skipped). Path-gated, green or skipped: "Prod images built" and "CLI host (PRD-234)".
ci_green_on_head() {
  local br sha got id jobs job hits
  br=$(git rev-parse --abbrev-ref HEAD); sha=$(ci_commit_for_head)
  [ "$sha" = "$(git rev-parse HEAD)" ] || echo "   HEAD is kit-only [skip ci] commits over $(git log -1 --format=%h "$sha"): reading that commit's run"
  command -v gh >/dev/null || { echo "   gh is required"; return 1; }
  [ -n "$(gh pr list --head "$br" --state open --json number --jq '.[0].number' 2>/dev/null)" ] || { echo "   no open PR for $br: test.yml runs on pull_request only (#993)"; return 1; }
  got=$(wait_for_run test.yml "$sha") || { echo "   no completed test.yml run for $sha after 60 min"; return 1; }
  id="${got%% *}"
  jobs=$(gh run view "$id" --json jobs --jq '.jobs[] | "\(.conclusion)\t\(.name)"')
  local REQUIRED=("Which lanes this change needs" "orchestrator-tests" "Alembic from-zero" "Schema-drift check" "Frontend CI" "Code standards on changed lines" "Orchestrator module + integration tests" "Prod images built" "CLI host (PRD-234)")
  local PATH_GATED=("Prod images built" "CLI host (PRD-234)")
  # Counted with awk, never `grep -vq` (the ugrep wrapper trap, 26 Oct: it read GREEN on a failure).
  for job in "${REQUIRED[@]}"; do
    hits=$(printf '%s\n' "$jobs" | awk -F'\t' -v j="$job" 'index($2, j) {print $1}')
    [ -n "$hits" ] || { echo "   missing job: $job"; return 1; }
    if printf '%s\n' "${PATH_GATED[@]}" | awk -v j="$job" '$0 == j {f=1} END {exit !f}'; then
      [ "$(printf '%s\n' "$hits" | awk 'NF && $1 != "success" && $1 != "skipped"' | wc -l | tr -d ' ')" -eq 0 ] || { echo "   red: $job"; return 1; }
    else
      [ "$(printf '%s\n' "$hits" | awk 'NF && $1 != "success"' | wc -l | tr -d ' ')" -eq 0 ] || { echo "   red: $job"; return 1; }
    fi
  done
  echo "   test.yml run $id is green on $sha for all ${#REQUIRED[@]} required jobs (Frontend CI green; Prod images, CLI host: green or skipped)"
}
# The browser lane (US-C21f, US-W04's count spec) runs in browser-e2e.yml on the PR. A WebKit 'internal error' on
# reload is a known Playwright 1.57 flake: rerun the failed job before calling it red.
browser_lane_green() {
  local sha got; sha=$(ci_commit_for_head)
  got=$(wait_for_run browser-e2e.yml "$sha") || { echo "   no completed browser-e2e.yml run for $sha after 60 min"; return 1; }
  echo "   browser-e2e.yml run ${got%% *}: ${got##* }"; [ "${got##* }" = "success" ]
}
pr_body_records() {
  local br n body s miss=0
  br=$(git rev-parse --abbrev-ref HEAD)
  n=$(gh pr list --head "$br" --state open --json number --jq '.[0].number' 2>/dev/null)
  [ -n "$n" ] || { echo "   no open PR for $br"; return 1; }
  body=$(gh pr view "$n" --json body --jq .body 2>/dev/null)
  for s in "#1359" "#1358" "Stacked on" "fourth push type" "SUGGESTION_PUSH_ENABLED" "rule sentence" "SDK 1.3.0" "prd268_w2_delivery" "Always allow" "AI assistance"; do
    printf '%s' "$body" | grep -qF -- "$s" || { echo "   the PR body lacks: $s"; miss=1; }
  done
  [ $miss -eq 0 ] && echo "   PR #$n records #1359, #1358, the stack, the push type and its switch, the policy text, the SDK bump, the migration, Always allow left out and the AI disclosure"
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
  check "the kit is tracked and valid JSON" "git ls-files --error-unmatch $KIT >/dev/null 2>&1 && \"$PY\" -m json.tool $KIT >/dev/null"
  check "W1's work is under this wave (the proactive models, the switches, the runner, the suggestions router)" "all_exist $O/core/models/proactive.py $PRO/switches.py $PRO/runner.py $O/api/proactive.py"

  echo ""; echo "=== Stories (priority order; positive: red on the base by design)"
  check "US-C19a one revision $NEW_REV on the base head: the CHECK widened, push_categories, both pins" "one_revision && revision_ok"
  check "US-C19a the models follow (PUSH_TYPES, push_categories); the K19 pins moved; the test is new" "grep -qE \"'suggestion'\" $O/core/models/mobile_push.py && has 'push_categories' $O/core/models/mobile_devices.py && changed $TESTS/test_prd266_push_tables_model.py $TESTS/test_prd266_mobile_devices_model.py && new_files $TESTS/test_prd268w2_delivery_migration.py"
  check "US-C19b post_background_message is short and takes parts; services/chat_parts.py; the test is new" "fn_short $MESSENGER post_background_message 50 && fn_has_param $MESSENGER post_background_message parts && has 'brief@1' $O/services/chat_parts.py && has 'suggestion@1' $O/services/chat_parts.py && new_files $TESTS/test_prd268w2_message_parts.py"
  check "US-C19c services/brief: the model, the builder, the strings in en and pt-BR; the test is new" "all_exist $BRIEF/model.py $BRIEF/build.py $BRIEF/strings.py && has 'class Brief\\b' $BRIEF/model.py && has 'spoken' $BRIEF/model.py && has 'def build_brief' $BRIEF/build.py && has 'pt-BR' $BRIEF/strings.py && new_files $TESTS/test_prd268w2_brief_build.py"
  check "US-C19d platform_get_brief: registered as read, handled, wired; the test is new" "all_exist $DISC/actions_brief.py $DISC/handlers_brief.py && grep -A12 'platform_get_brief' $DISC/actions_brief.py | grep -q 'permission_level=\"read\"' && grep -q 'platform_get_brief' $DISC/platform_executor.py && grep -qE 'register_brief_actions' $DISC/platform_actions.py && new_files $TESTS/test_prd268w2_get_brief_tool.py"
  check "US-C10a the shared claim; the scheduled digest posts the brief; an open brief is not announced; the test is new" "all_exist $BRIEF/claim.py && has 'def claim_brief' $BRIEF/claim.py && has 'def todays_brief_message' $BRIEF/claim.py && added_in 'brief' $O/services/mobile/digest.py && added_in 'locale' $O/services/mobile/digest.py && added_in 'via' $O/services/notification_events/auto_message.py && new_files $TESTS/test_prd268w2_brief_claim.py"
  check "US-C10b POST /api/proactive/brief/open: RouterSpec, gated, in the manifest; discovery brief; the test is new" "all_exist $BRIEF_ROUTER && router_spec api.proactive_brief && brief_route_gated && routes_in_manifest 'POST /api/proactive/brief/open' && capability brief && new_files $TESTS/test_prd268w2_brief_open_route.py"
  check "US-C10b the cadence Daily Brief is retired; its test asserts the retirement" "! grep -qE 'daily_brief|_DAILY_BRIEF' $O/core/services/auto_cadence.py && changed $TESTS/test_prd263_harness_review_cadence.py"
  check "US-C08a delivery: deliver_run, suggestion_ready, not announced; the test is new" "all_exist $PRO/delivery.py && has 'def deliver_run' $PRO/delivery.py && has 'suggestion@1' $PRO/delivery.py && set_has $DISPATCHER VALID_EVENT_TYPES suggestion_ready && set_lacks $O/services/notification_events/auto_message.py ANNOUNCED_ORIGINS suggestion && has 'deliver_run' $PRO/runner.py && settings_declared PROACTIVE_MAX_SUGGESTION_PUSHES_PER_DAY && new_files $TESTS/test_prd268w2_suggestion_delivery.py"
  check "US-C08b suggestion_changed on the live lane; discovery suggestions and readiness; the test is new" "has 'def notify_suggestion_event' $O/services/board_events.py && has 'suggestion_changed' $O/services/board_events.py && capability suggestions && capability readiness && new_files $TESTS/test_prd268w2_suggestion_events.py"
  check "US-C09a the suggestion push type: events, channel, line, category, model, the switch (off), the vector; the test is new" "set_has $PUSH/payload.py PUSH_TYPE_EVENTS suggestion_ready && set_has $PUSH/payload.py CHANNEL_BY_TYPE suggestions && has 'Auto has a suggestion' $PUSH/payload.py && has 'automatos\\.suggestion' $PUSH/payload.py && settings_declared SUGGESTION_PUSH_ENABLED && push_switch_off && changed $TESTS/fixtures/prd266_push_signature_vectors.json && has '\"suggestion\"' $TESTS/fixtures/prd266_push_signature_vectors.json && changed $TESTS/test_prd266_push_payload.py $TESTS/test_prd266_push_preferences.py && new_files $TESTS/test_prd268w2_suggestion_push_type.py"
  check "US-C09b the rules: registered categories, the cap, quiet hours; discovery suggestion_push behind the switch; the test is new" "all_exist $PUSH/suggestion_rules.py && has 'push_categories' $O/api/mobile_devices.py && has 'push_categories' $O/services/mobile/devices.py && added_in 'suggestion' $PUSH/fanout.py && capability suggestion_push && grep -A4 '^CONFIGURED_CAPABILITIES' $DISCOVERY | grep -q 'suggestion_push' && new_files $TESTS/test_prd268w2_suggestion_push_rules.py"
  check "US-C20a rule sentences, ApprovalView@1 by decorator, the view route; the test is new" "all_exist $O/modules/policy/rule_sentences.py $FMT/approval_view.py $VIEW_ROUTER && has 'pt-BR' $O/modules/policy/rule_sentences.py && decorated $FMT/result_formatter.py format_for_frontend carries_the_approval_view && view_answers_ok && router_spec api.approval_views && routes_in_manifest 'GET /api/v1/approval-grants/\\{[a-z_]+\\}/view' && new_files $TESTS/test_prd268w2_approval_view.py"
  check "US-C20b board approvals write the sentence; evaluate_board_task_approval is short; blocked-reason shows it; the tests are new" "fn_short $O/services/board_approval.py evaluate_board_task_approval 50 && ! grep -q 'requires approval under' $O/services/board_approval.py && has 'rule_sentence' $O/services/board_approval.py && has 'sentence' $FE/components/activity/board/blocked-reason.ts && new_files $TESTS/test_prd268w2_board_approval_sentence.py $FE/components/activity/board/__tests__/blocked-reason-sentence.test.ts"
  check "US-C20c the host cards read approval_view; the protected-surface test is extended; the test is new" "has 'approval_view' $FE/lib/chat/approvals.ts && has 'approval_view' $FE/components/widgets/types.ts && has 'approval_view' $FE/lib/theme-sdk/host-cards/approvals.tsx && all_exist $FE/components/widgets/approval-view-parts.tsx && changed $FE/components/theme-host/__tests__/protected-surface.test.tsx && new_files $FE/components/widgets/__tests__/approval-view.test.tsx"
  check "US-C21a --running in globals.css and Tailwind; STUDIO_STATUS_TOKEN_NAMES; the test is new" "[ \"\$(grep -c -- '--running:' $GLOBALS)\" -ge 4 ] && grep -q 'hsl(var(--running))' $GLOBALS && has 'running' $FE/tailwind.config.ts && has 'STUDIO_STATUS_TOKEN_NAMES' $FE/lib/look-kit/studio-tokens.ts && has '--running' $FE/lib/look-kit/studio-tokens.ts && new_files $FE/lib/look-kit/__tests__/status-tokens.test.ts"
  check "US-C21b the types, DISPLAY_STATE_LABELS and the hooks; the tests are new" "all_exist $FE/lib/proactive/types.ts $FE/lib/proactive/display-state.ts $FE/hooks/use-suggestions.ts $FE/hooks/use-brief.ts $FE/hooks/use-readiness.ts $FE/hooks/use-calibration.ts && has 'DISPLAY_STATE_LABELS' $FE/lib/proactive/display-state.ts && new_files $FE/lib/proactive/__tests__/display-state.test.ts $FE/hooks/__tests__/use-suggestions.test.tsx"
  check "US-C21c StatusLabel, SuggestionCard, BriefCard with data-ai-generated; the tests are new" "all_exist $PROC/status-label.tsx $PROC/suggestion-card.tsx $PROC/brief-card.tsx && has 'data-ai-generated' $PROC/suggestion-card.tsx && has 'data-ai-generated' $PROC/brief-card.tsx && new_files $PROC/__tests__/status-label.test.tsx $PROC/__tests__/suggestion-card.test.tsx $PROC/__tests__/brief-card.test.tsx"
  check "US-C21d ReadinessList, CalibrationCard, ReplyChips, BriefVoice in the brief-voice surface; the tests are new" "all_exist $PROC/readiness-list.tsx $PROC/calibration-card.tsx $PROC/reply-chips.tsx $FE/components/voice/brief-voice.tsx && has \"'brief-voice'\" $FE/components/theme-host/protected-surface.tsx && new_files $PROC/__tests__/readiness-list.test.tsx $PROC/__tests__/calibration-card.test.tsx $PROC/__tests__/reply-chips.test.tsx $FE/components/voice/__tests__/brief-voice.test.tsx"
  check "US-C21e SDK 1.3.0: the proactive area, its surface and d.ts, the versions (host stays 1.1.0); the test is new" "all_exist $FE/lib/theme-sdk/proactive.ts $FE/lib/theme-sdk/surface/proactive.json $FE/packages/theme-sdk/areas/proactive.d.ts && has 'SDK_VERSION = \"1\\.3\\.0\"' $O/services/themes/contract.py && has 'THEME_HOST_VERSION = \"1\\.1\\.0\"' $O/services/themes/contract.py && has '\"sdkVersion\": \"1\\.3\\.0\"' $FE/lib/theme-host/shared-modules.json && has '\"version\": \"1\\.3\\.0\"' $FE/packages/theme-sdk/package.json && has '\"proactive\"' $O/services/themes/data/sdk-surface.json && has 'useBrief' docs/theme-host/sdk-surface-map.md && changed $FE/lib/theme-sdk/__tests__/surface.test.ts && new_files $FE/lib/theme-sdk/__tests__/proactive.test.tsx"
  check "US-C21f the browser lane: the hello-proactive fixture and the defaults spec; the tests are new" "new_files e2e/browser/fixtures/hello-proactive/theme.json e2e/browser/fixtures/hello-proactive/entry.js e2e/browser/specs/proactive-defaults.spec.ts $FE/lib/theme-sdk/__tests__/hello-proactive-fixture.test.ts"
  check "US-W01a Suggested on the Command Center (both shells), the bell's suggestion link, 'From suggestions'; the tests are new" "all_exist $PROC/suggested-list.tsx $PROC/command-center-read.tsx && has 'CommandCenterRead' $FE/components/command-center/summary-tab.tsx && grep -qE '^import \\{ CommandCenterRead as AutosRead \\}' $FE/components/activity/activity-page.tsx && grep -qE \"case 'suggestion'\" $FE/components/notifications/notification-bell.tsx && changed $FE/components/notifications/__tests__/notification-bell-links.test.ts && has 'from_suggestions' $FE/hooks/use-board-tasks.ts && new_files $PROC/__tests__/suggested-list.test.tsx $FE/components/activity/board/__tests__/from-suggestions.test.tsx"
  check "US-W01b thread cards through message-with-receipts, the pinned card by import alias, the live frame; the tests are new" "all_exist $PROC/thread-cards.tsx $PROC/composer-with-suggestion.tsx $PROC/mission-suggestion-presence.ts $FE/lib/live/route-frame.ts && has 'thread-cards|ThreadCards' $FE/components/chatbot/message-with-receipts.tsx && changed $FE/components/chatbot/chat.tsx && has 'suggestion' $FE/hooks/use-board-event-stream.ts && new_files $PROC/__tests__/thread-cards.test.tsx $PROC/__tests__/pinned-suggestion.test.tsx $FE/lib/live/__tests__/route-frame.test.ts"
  check "US-W02 BriefOnOpen in the providers, AutosRead's rows; the test is new" "all_exist $PROC/brief-on-open.tsx && has 'BriefOnOpen' $FE/components/providers.tsx && has 'useBrief' $FE/components/activity/autos-read.tsx && new_files $PROC/__tests__/brief-on-open.test.tsx"
  check "US-W04 the status vocabulary: the tests and the browser-lane count spec are new" "new_files $PROC/__tests__/status-vocabulary.test.tsx $PROC/__tests__/no-colour-alone.test.ts e2e/browser/specs/proactive-needs-you-count.spec.ts"
  check "US-W05 'What Auto depends on' in Settings → Orchestrator; the test is new" "all_exist $FE/components/settings/what-auto-depends-on.tsx && has 'WhatAutoDependsOn|what-auto-depends-on' $FE/components/settings/SettingsPanel.tsx && new_files $FE/components/settings/__tests__/what-auto-depends-on.test.tsx"
  check "every acceptance criterion in $KIT is marked (DONE, or OWNER)" "all_acs_done"
fi

echo ""; echo "=== Scope and conventions"
check "N2 exactly one Alembic head in the tree" "one_head"
check "N3 no env reads added outside config.py" "! env_read_added"
check "N4 no print() added in orchestrator application code (scripts may print)" "! print_added"
check "N5 no raw fetch('/api…') added in the frontend" "! raw_fetch_added"
check "N6 no new Python dependency" "git diff --quiet $BASE..HEAD -- orchestrator/requirements.in orchestrator/requirements.lock orchestrator/requirements-dev.txt orchestrator/requirements-dev.lock"
check "N7 no new npm dependency (frontend and the browser lane; K18)" "git diff --quiet $BASE..HEAD -- frontend/package.json frontend/package-lock.json e2e/browser/package.json e2e/browser/package-lock.json"
check "N8 the Composio deny list and the generated Auto skill seed are untouched" "git diff --quiet $BASE..HEAD -- orchestrator/core/composio/deny_list.py orchestrator/core/seeds/platform-management-skill.md"
check "N9 no node_modules in the diff" "! git diff --name-only $BASE..HEAD | grep -q node_modules"
check "N10 every commit is DCO-signed" "every_commit_signed"
check "N11 a new file is at most 800 lines" "file_sizes_ok"
check "N12a the K3 files over 800 lines did not grow" "named_files_not_grown"
check "N12b no other file over 800 lines grew (config.py, the two reports and globals.css excepted)" "no_big_file_grew"
warn "N12c config.py grew by settings only (≤ 20 lines)" "config_growth"
check "N12d globals.css grew by the --running lines only (≤ 8)" "globals_growth"
for fn in _agent_tick _orchestrator_tick _orchestrator_tick_llm _dispatch_heartbeat_notification _route_heartbeat_to_auto; do
  check "N13 $fn is byte-identical to the base (W1's K4)" "fn_unchanged $HB $fn"
done
check "N14a PlatformActionExecutor.clear is byte-identical to the base" "fn_unchanged $DISC/platform_executor.py clear"
check "N14b PlatformActionExecutor._run_cleared is byte-identical to the base" "fn_unchanged $DISC/platform_executor.py _run_cleared"
check "N14c ToolResultFormatter.format_for_frontend is byte-identical (the view is a decorator, K4)" "fn_unchanged $FMT/result_formatter.py format_for_frontend"
check "N14d bulk_update_preferences is byte-identical to the base" "fn_unchanged $O/api/notifications.py bulk_update_preferences"
check "N14e _board_task_blocked_pending_approval is byte-identical to the base" "fn_unchanged $O/api/board_tasks.py _board_task_blocked_pending_approval"
check "N15 frontend/components/chatbot/message.tsx is untouched (the Message component, K4)" "git diff --quiet $BASE..HEAD -- frontend/components/chatbot/message.tsx"
check "N16 chat.tsx changes are import lines only (the Chat component, K4)" "chat_imports_only"
check "N17 api/approval_grants.py is untouched (806 lines; the view route lives in its own router)" "git diff --quiet $BASE..HEAD -- orchestrator/api/approval_grants.py"
check "N18 the new routers' handlers are plain def (sync session)" "routers_plain_def"
check "N19 no third-party HTTP client added" "! client_added"
check "N20 the hierarchy gate passes (stdlib ast)" "\"$PY\" orchestrator/scripts/check_hierarchy_gate.py >/dev/null"
check "N21 no existing migration modified, deleted or renamed (W1's included)" "old_migrations_untouched"
check "N22 no W3/W4 surface added (follow-ups, the reply-chip tool, calibration routes, evals; K23)" "! later_wave_added"
check "N23 no Always allow anywhere, and the approval buttons keep today's words (K10)" "! always_allow_added && ! new_words_added"
check "N24 CI's changed-lines code shape passes on the committed wave (functions ≤ 50, nesting ≤ 4, new files ≤ 800)" "code_shape_ok"
check "N25 W1's kit files are untouched on this branch" "git diff --quiet $BASE..HEAD -- $W1_KIT_FILES"
warn "N26 story commits carry [skip ci] (all but the final CI commit and a merge of main)" "[ -z \"\$(git log --no-merges --format='%s' $BASE..HEAD^ | grep -v 'skip ci')\" ]"
check "N27 the suggestion push ships off: SUGGESTION_PUSH_ENABLED defaults to false (K9)" "push_switch_off"
check "N28 the push's signed field set and order are the base's (K9)" "signed_fields_kept"
check "N29 at most the four push types (§6.5.3)" "push_types_bounded"
check "N30 no innerHTML / dangerouslySetInnerHTML added in the frontend" "! inner_html_added"
check "N31 no orange-* class, no hex colour and no rounded-3xl in the new web surfaces (K17)" "! orange_added && ! hex_added && ! rounded3_added"
check "N32 nothing auto-plays (§12-Q12)" "! autoplay_added"
check "N33 the K19 pin files keep their test counts" "pin_tests_kept"
check "N34 no test weakened: added test functions >= removed" "tests_not_weakened"
check "N35 the check-in lane and the brief never reach AgentFactory (W1 K8)" "! factory_in_lane"
check "N36 THEME_HOST_VERSION stays 1.1.0 (K12)" "grep -qE '^THEME_HOST_VERSION = \"1\\.1\\.0\"' $O/services/themes/contract.py"
check "N37 docker-compose.yml is untouched by this wave" "git diff --quiet $BASE..HEAD -- docker-compose.yml"

if [ "$MODE" = "full" ]; then
  echo ""; echo "=== CI and the PR (once per wave)"
  check "CI: the required jobs present and green on the tested commit" "ci_green_on_head"
  check "the browser lane (browser-e2e.yml) is green on the tested commit" "browser_lane_green"
  check "the PR targets W1's branch while W1's PR is open, main once W1 merged (the stacked-PR trap, K1)" "pr_base_ok"
  check "the PR body records #1359, #1358, the stack, the push type and its switch, the policy text, the SDK bump, the migration and the AI disclosure" "pr_body_records"
fi
echo ""
if [ "$FAIL" -eq 0 ]; then echo "✅ PRD-268 W2 Delivery acceptance ($MODE): PASS"; else echo "❌ PRD-268 W2 Delivery acceptance ($MODE): FAIL"; fi
exit "$FAIL"
