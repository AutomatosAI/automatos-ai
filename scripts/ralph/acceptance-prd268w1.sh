#!/bin/bash
# Acceptance gate — PRD-268 W1 Check-ins, Proactive Auto's engine (19 stories: US-C01 … US-C18).
#   bash scripts/ralph/acceptance-prd268w1.sh | --print-base | --no-ci | --negative-only ; GATE_BASE=<ref> (smoke runs only).
# Base: the merge-base with origin/main (the wave's PR targets main). Nothing runs on the owner's machine: greps, git
# and stdlib python only; CI once per wave on the draft PR. No pipefail (a negative `! producer | grep -q` would invert
# on SIGPIPE); every check runs in its own `( eval … )`; stdin is closed so a grep with a missing file argument fails.
# Derived from acceptance-prd266w1.sh (28 Oct 2026): the generic helpers are unchanged; the story and scope checks are
# PRD-268's. Story checks are POSITIVE (they fail on the base by design: no proactive tables, no lane, no runner).
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

KIT=scripts/ralph/prd-268w1.json
PRD_DOC=docs/PRDS/PRD-268-PROACTIVE-AUTO.md
O=orchestrator
TESTS=$O/tests
VERSIONS=$O/alembic/versions
MANIFEST=$O/reports/route-manifest.json
SURFACE=$O/reports/config-surface.json
PRO=$O/services/proactive
PURE=$O/modules/proactive
DISC=$O/modules/tools/discovery
HB=$O/services/heartbeat_service.py
ROUTER=$O/api/proactive.py
READY_ROUTER=$O/api/readiness.py
NEW_REV=prd268_w1_proactive
SETTINGS="PROACTIVE_MODE_DEFAULT PROACTIVE_MAX_MODEL_RUNS_PER_DAY PROACTIVE_MAX_SUGGESTIONS_PER_RUN PROACTIVE_MAX_NEW_PER_PERSON_PER_DAY PROACTIVE_MAX_OPEN_PER_PERSON PROACTIVE_MAX_OPEN_PER_AGENT PROACTIVE_MIN_INTERVAL_MINUTES PROACTIVE_MAX_INTERVAL_MINUTES READINESS_PROBE_TIMEOUT_SECONDS"
# K3: files over 800 lines this wave may touch; none may grow.
NO_GROW="$HB $DISC/platform_executor.py $O/modules/tools/execution/unified_executor.py $O/main.py $O/modules/agents/factory/agent_factory.py $O/api/missions.py $O/api/board_tasks.py $O/services/activity_service.py $O/core/llm/manager.py"
GROW_OK="$O/config.py $MANIFEST $SURFACE"

added_code() { git diff "$BASE"..HEAD -- "$@" ':!orchestrator/tests' ':!*__tests__*' ':!*.test.ts' ':!*.test.tsx' ':!scripts/ralph' ':!docs' | grep '^+' | grep -v '^+++'; }
added_files() { git diff --name-only --diff-filter=A "$BASE"..HEAD -- "$@"; }
lines_at() { git show "$1:$2" 2>/dev/null | wc -l | tr -d ' '; }
has() { [ -f "$2" ] && grep -qE -e "$1" "$2"; }
all_exist() { local f; for f in "$@"; do [ -f "$f" ] || { echo "   missing: $f"; return 1; }; done; }
new_files() { local f bad=0; for f in "$@"; do
    if [ ! -f "$f" ]; then echo "   missing: $f"; bad=1
    elif [ -z "$(added_files "$f")" ]; then echo "   not new on this branch: $f"; bad=1; fi
  done; [ $bad -eq 0 ]; }

env_read_added()  { added_code orchestrator ':!orchestrator/config.py' | grep -E 'os\.(getenv|environ)' | grep -q .; }
print_added()     { added_code orchestrator ':!orchestrator/scripts' | grep -E '^\+\s*print\(' | grep -q .; }
client_added()    { added_code "$PRO" "$PURE" "$DISC" $O/api | grep -E '^\+\s*(import|from)\s+(httpx|requests|aiohttp|urllib3)\b' | grep -q .; }
# K8: the check-in lane never reaches AgentFactory.
factory_in_lane() { added_code "$PRO" "$PURE" | grep -E 'AgentFactory|agent_factory|execute_with_prompt' | grep -q .; }
# No delivery in W1 (W2): no new push type, no Auto-thread post from the proactive code.
delivery_added()  { added_code "$PRO" $O/services/push | grep -E 'deliver_background_message|post_background_message|suggestion_ready|"suggestion":' | grep -q .; }
frontend_touched() { [ -n "$(git diff --name-only "$BASE"..HEAD -- frontend)" ]; }
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
base_head() { git show "$BASE:$TESTS/test_prd209_alembic_single_head.py" 2>/dev/null | grep -E '^EXPECTED_HEAD *= *"' | head -1 | sed -E 's/.*"([^"]+)".*/\1/'; }
new_revisions() { added_files "$VERSIONS" | grep '\.py$'; }
one_revision() { local n; n=$(new_revisions | grep -c .); echo "   new revisions: $n ($(new_revisions | tr '\n' ' '))"; [ "$n" -eq 1 ]; }
revision_ok() {
  local f bh t; f="$VERSIONS/$NEW_REV.py"; bh=$(base_head)
  [ -f "$f" ] || { echo "   $f is missing"; return 1; }
  [ -n "$bh" ] || { echo "   cannot read the base's EXPECTED_HEAD"; return 1; }
  grep -qE "^revision[^=]*= *['\"]$NEW_REV['\"]" "$f" || { echo "   $f: revision is not $NEW_REV"; return 1; }
  grep -qE "^down_revision[^=]*= *['\"]$bh['\"]" "$f" || { echo "   $f: down_revision is not the base head $bh"; return 1; }
  for t in proactive_suggestions proactive_runs proactive_person_settings; do
    grep -qiE "CREATE TABLE IF NOT EXISTS $t" "$f" || { echo "   $f: no CREATE TABLE IF NOT EXISTS $t"; return 1; }
  done
  [ "$(grep -ciE 'CREATE (TABLE|(UNIQUE )?INDEX)' "$f")" -le "$(grep -ciE 'IF NOT EXISTS' "$f")" ] || { echo "   $f: a CREATE without IF NOT EXISTS"; return 1; }
  grep -qiE "WHERE +status *= *'open'" "$f" || { echo "   $f: no partial unique index WHERE status = 'open'"; return 1; }
  grep -q "EXPECTED_HEAD = \"$NEW_REV\"" $TESTS/test_prd209_alembic_single_head.py || { echo "   test_prd209 EXPECTED_HEAD is not $NEW_REV"; return 1; }
  grep -qF "$NEW_REV.py" $TESTS/test_prd236_w1_routes.py || { echo "   test_prd236_w1_routes.py does not read $NEW_REV.py"; return 1; }
  grep -qF "EXPECTED_HEAD = \"$NEW_REV\"" $TESTS/test_prd236_w1_routes.py || { echo "   test_prd236_w1_routes.py does not assert EXPECTED_HEAD = $NEW_REV"; return 1; }
  echo "   $NEW_REV chains onto $bh; three tables; both pins moved"
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
# Every ActionDefinition(...) in the discovery package states permission_level (AST), and the dataclass has no default.
levels_stated() { "$PY" - "$DISC" <<'PY'
import ast, pathlib, sys
bad = []
for p in sorted(pathlib.Path(sys.argv[1]).glob("*.py")):
    tree = ast.parse(p.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) == "ActionDefinition":
            if not any(k.arg == "permission_level" for k in node.keywords):
                bad.append(f"{p.name}:{node.lineno}")
src = (pathlib.Path(sys.argv[1]) / "action_registry.py").read_text(encoding="utf-8")
if 'permission_level: str = "read"' in src:
    bad.append("action_registry.py still defaults permission_level to read")
print("\n".join(f"   {b}" for b in bad) or "   every ActionDefinition states its level; no default"); sys.exit(1 if bad else 0)
PY
}
# The lane decorator sits on PlatformActionExecutor.clear and UnifiedToolExecutor.execute_tool.
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
routers_plain_def() { "$PY" - "$ROUTER" "$READY_ROUTER" <<'PY'
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
ci_commit_for_head() {
  local sha; sha=$(git rev-parse HEAD)
  while git log -1 --format=%s "$sha" | grep -q 'skip ci' && [ -z "$(git show --format= --name-only "$sha" | grep -v '^scripts/ralph/')" ]; do
    sha=$(git rev-parse "$sha^")
  done
  echo "$sha"
}
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
  local PATH_GATED=("Prod images built" "CLI host (PRD-234)" "Frontend CI")
  for job in "${REQUIRED[@]}"; do
    hits=$(echo "$jobs" | grep -F "$job" | cut -f1)
    [ -n "$hits" ] || { echo "   missing job: $job"; return 1; }
    if printf '%s\n' "${PATH_GATED[@]}" | grep -qxF "$job"; then echo "$hits" | grep -qvE '^(success|skipped)$' && { echo "   red: $job"; return 1; }
    else echo "$hits" | grep -qvE '^success$' && { echo "   red: $job"; return 1; }; fi
  done
  echo "   test.yml run $id is green on $sha for all ${#REQUIRED[@]} required jobs (Prod images, CLI host: green or skipped)"
}
pr_body_records() {
  local br n body s miss=0
  br=$(git rev-parse --abbrev-ref HEAD)
  n=$(gh pr list --head "$br" --state open --json number --jq '.[0].number' 2>/dev/null)
  [ -n "$n" ] || { echo "   no open PR for $br"; return 1; }
  body=$(gh pr view "$n" --json body --jq .body 2>/dev/null)
  for s in "#1358" "#1362" "check-in lane" "shadow" "AI assistance"; do
    printf '%s' "$body" | grep -qF -- "$s" || { echo "   the PR body lacks: $s"; miss=1; }
  done
  [ $miss -eq 0 ] && echo "   PR #$n records the approval (#1358), the PRD (#1362), the lane, the shadow rollout and the AI disclosure"
}

if [ "$MODE" != "negative" ]; then
  echo ""; echo "=== Seeded"
  check "the PRD is tracked at $PRD_DOC" "git ls-files --error-unmatch $PRD_DOC >/dev/null 2>&1"
  check "the kit is tracked and valid JSON" "git ls-files --error-unmatch $KIT >/dev/null 2>&1 && \"$PY\" -m json.tool $KIT >/dev/null"
  echo ""; echo "=== The wave's work (positive: red on the base by design)"
  check "US-C01 one revision $NEW_REV: three tables, the partial unique index, both pins" "one_revision && revision_ok"
  check "US-C01 the models and their test" "all_exist $O/core/models/proactive.py && has 'proactive' $O/core/models/__init__.py && new_files $TESTS/test_prd268_proactive_models.py"
  check "US-C02a every action states its level; platform_field_inject is write; the test is new" "levels_stated && grep -A12 'platform_field_inject' $DISC/actions_field.py | grep -q 'permission_level=\"write\"' && new_files $TESTS/test_prd268_action_levels.py"
  check "US-C02a the hierarchy gate no longer defaults a missing level to read" "! grep -qE '^DEFAULT_PERMISSION_LEVEL *= *\"read\"' $O/scripts/check_hierarchy_gate.py"
  check "US-C02b the lane decorator on clear and execute_tool; the refusal text; the test is new" "all_exist $DISC/check_in_lane.py && decorated $DISC/platform_executor.py clear reads_only_in_a_check_in && decorated $O/modules/tools/execution/unified_executor.py execute_tool check_in && has 'Not available in a check-in, which can only read' $DISC/check_in_lane.py && new_files $TESTS/test_prd268_check_in_lane.py"
  check "US-C02c ContextMode.CHECK_IN, the read_only strategy, its budget; the test is new" "has 'CHECK_IN' $O/modules/context/modes.py && grep -qiE 'read_only' $O/modules/context/sections/tools.py && grep -qiE 'check_in' $O/modules/context/budget.py && new_files $TESTS/test_prd268_check_in_mode.py"
  check "US-C02d the read-actions sweep test is new" "new_files $TESTS/test_prd268_read_actions_write_nothing.py"
  check "US-C03a the pure proactive module and its test" "[ -d $PURE ] && grep -rqE 'def check_in_window' $PURE && grep -rqE 'def provenance_ok' $PURE && new_files $TESTS/test_prd268_proactive_pure.py"
  check "US-C03b platform_suggest_next_step: registered, handled, gated; the test is new" "all_exist $DISC/actions_proactive.py $DISC/handlers_proactive.py && grep -qE 'register_proactive_actions' $DISC/platform_actions.py && grep -q 'platform_suggest_next_step' $DISC/platform_executor.py && grep -q 'platform_suggest_next_step' $O/scripts/check_hierarchy_gate.py && new_files $TESTS/test_prd268_suggest_tool.py"
  check "US-C03c the signal readers and their test" "all_exist $PRO/signals.py && new_files $TESTS/test_prd268_signals.py"
  check "US-C03d the runner; check_in joins BACKGROUND_LANES; the test is new" "all_exist $PRO/runner.py && grep -qE 'run_workspace_check_in' $PRO/runner.py && grep -qE 'run_agent_check_in' $PRO/runner.py && grep -qE 'BACKGROUND_LANES.*check_in|\"check_in\"' $O/services/credits/asks.py && new_files $TESTS/test_prd268_runner.py"
  check "US-C04a/b the bridge; the scheduler passes its jobs; the split and bridge tests are new" "all_exist $PRO/heartbeat_bridge.py && grep -qE 'heartbeat_bridge|agent_job' $HB && new_files $TESTS/test_prd268_heartbeat_split.py $TESTS/test_prd268_heartbeat_bridge.py"
  check "US-C04b heartbeat.mode accepted by the payload" "grep -qE '\bmode\b' $O/api/heartbeat.py"
  check "US-C05 the streaks test is new" "new_files $TESTS/test_prd268_streaks.py"
  check "US-C06a the router, RouterSpec, the manifest routes; the test is new" "all_exist $ROUTER && router_spec api.proactive && routes_in_manifest 'GET /api/proactive/suggestions' 'POST /api/proactive/suggestions/\{[a-z_]+\}/start' 'POST /api/proactive/suggestions/\{[a-z_]+\}/hide' 'GET /api/proactive/runs' 'GET /api/proactive/settings' 'PUT /api/proactive/settings' 'GET /api/proactive/me' 'PUT /api/proactive/me' && new_files $TESTS/test_prd268_suggestions_api.py"
  check "US-C06b the started-watch test is new" "new_files $TESTS/test_prd268_started_watch.py"
  check "US-C07a the switches, the settings in config.py and config-surface.json; the test is new" "all_exist $PRO/switches.py && settings_declared $SETTINGS && new_files $TESTS/test_prd268_switches.py"
  check "US-C07b per-person quiet hours; the test is new" "all_exist $PRO/quiet.py && new_files $TESTS/test_prd268_quiet_hours.py"
  check "US-C17a display_state, the strings table in en and pt-BR; the test is new" "all_exist $PRO/display_state.py $PRO/strings.py && grep -q 'pt-BR' $PRO/strings.py && new_files $TESTS/test_prd268_needs_you_status.py"
  check "US-C17b the Missions and board status test is new" "new_files $TESTS/test_prd268_missions_board_status.py"
  check "US-C18 readiness: probes, the routes in the manifest; the test is new" "[ -d $O/services/readiness ] && routes_in_manifest 'GET /api/readiness' 'POST /api/readiness/check' && new_files $TESTS/test_prd268_readiness.py"
  check "every acceptance criterion in $KIT is DONE" "all_acs_done"
fi

echo ""; echo "=== Scope and conventions"
check "N2 exactly one Alembic head in the tree" "one_head"
check "N3 no env reads added outside config.py" "! env_read_added"
check "N4 no print() added in orchestrator application code" "! print_added"
check "N5 no frontend change in this wave (core only)" "! frontend_touched"
check "N6 no new Python dependency" "git diff --quiet $BASE..HEAD -- orchestrator/requirements.txt orchestrator/requirements-dev.txt orchestrator/pyproject.toml"
check "N7 no new npm dependency" "git diff --quiet $BASE..HEAD -- frontend/package.json frontend/package-lock.json"
check "N8 the Composio deny list and the generated Auto skill seed are untouched" "git diff --quiet $BASE..HEAD -- orchestrator/core/composio/deny_list.py orchestrator/core/seeds/platform-management-skill.md"
check "N9 no node_modules in the diff" "! git diff --name-only $BASE..HEAD | grep -q node_modules"
check "N10 every commit is DCO-signed" "every_commit_signed"
check "N11 a new file is at most 800 lines" "file_sizes_ok"
check "N12a the K3 files over 800 lines did not grow" "named_files_not_grown"
check "N12b no other file over 800 lines grew (config.py and the two reports excepted)" "no_big_file_grew"
warn "N12c config.py grew by settings only (≤ 40 lines)" "config_growth"
for fn in _agent_tick _orchestrator_tick _orchestrator_tick_llm _dispatch_heartbeat_notification _route_heartbeat_to_auto; do
  check "N13 $fn is byte-identical to the base (K4)" "fn_unchanged $HB $fn"
done
check "N14 PlatformActionExecutor.clear is byte-identical to the base (the lane is a decorator)" "fn_unchanged $DISC/platform_executor.py clear"
check "N15 PlatformActionExecutor._run_cleared is byte-identical to the base" "fn_unchanged $DISC/platform_executor.py _run_cleared"
check "N16 the new routers' handlers are plain def (sync session)" "routers_plain_def"
check "N17 no third-party HTTP client added in the proactive, discovery or api code" "! client_added"
check "N18 the check-in lane never reaches AgentFactory (K8)" "! factory_in_lane"
check "N19 no test weakened: added test functions >= removed" "tests_not_weakened"
check "N20 the hierarchy gate passes (stdlib ast)" "\"$PY\" orchestrator/scripts/check_hierarchy_gate.py >/dev/null"
check "N21 no existing migration modified, deleted or renamed" "old_migrations_untouched"
check "N22 no delivery built (W2): no Auto-thread post, no suggestion push" "! delivery_added"
check "N24 CI's changed-lines code shape passes on the committed wave" "code_shape_ok"
warn "N26 story commits carry [skip ci] (all but the final CI commit)" "[ -z \"\$(git log --format='%s' $BASE..HEAD^ | grep -v 'skip ci' | grep -v '^docs(prd-268)')\" ]"

if [ "$MODE" = "full" ]; then
  echo ""; echo "=== CI and the PR (once per wave)"
  check "CI: the required jobs present and green on the tested commit" "ci_green_on_head"
  check "the PR body records the approval (#1358), the PRD (#1362), the lane, the shadow rollout and the AI disclosure" "pr_body_records"
fi
echo ""
if [ "$FAIL" -eq 0 ]; then echo "✅ PRD-268 W1 Check-ins acceptance ($MODE): PASS"; else echo "❌ PRD-268 W1 Check-ins acceptance ($MODE): FAIL"; fi
exit "$FAIL"
