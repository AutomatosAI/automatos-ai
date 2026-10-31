#!/bin/bash
# Acceptance gate — PRD-268 W3 Follow-ups, Auto keeps its word (17 stories: US-C11a … US-W03c).
#   bash scripts/ralph/acceptance-prd268w3.sh | --print-base | --no-ci | --negative-only
#   GATE_BASE=<ref> and GATE_HEAD=<ref> (smoke runs only): the base, and the commit read as HEAD.
# Base (K1): the wave is STACKED on W2 (feat/prd-268-w2-delivery). While HEAD sits on W2's work, the base is the W2
# commit the branch grew from (merge-base with origin/feat/prd-268-w2-delivery), so W2's own diff is never read as
# this wave's; once W2 has landed on main and main was merged in (or W2's branch is gone), the base is the merge-base
# with origin/main.
# The gate reads COMMITTED work: every file read, diff and count is taken from the HEAD commit (or GATE_HEAD), never
# the working tree, so a smoke can read origin/main from a checkout that is behind it. The two checks that can only
# read a working tree (CI's changed-lines code shape and the hierarchy gate) run when HEAD is the commit read, and
# are reported skipped in a GATE_HEAD smoke of another commit.
# Nothing runs on the owner's machine: greps, git and stdlib python only; CI once per wave on the draft PR. No
# pipefail (a negative `! producer | grep -q` would invert on SIGPIPE); every check runs in its own `( eval … )`;
# stdin is closed so a grep with a missing file argument fails instead of hanging. Counts use awk, never `grep -vq`
# (the ugrep wrapper trap, 26 Oct).
# Derived from acceptance-prd268w2.sh (28 Oct 2026): the generic helpers are W2's, moved onto the commit; the
# stacked base is W2's own pattern one wave up; the story and scope checks are PRD-268 W3's.
#   - every "added lines" check reads the diff BASE → HEAD, EMPTY when HEAD is the base, so it passes there by
#     construction; each is scoped to code dirs and excludes tests, the kit (scripts/ralph) and docs;
#   - the byte-identical function checks and the pin counts hold on the base by construction;
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

STACK_REF=origin/feat/prd-268-w2-delivery
W2_BRANCH=feat/prd-268-w2-delivery
HEAD_SHA=$(git rev-parse --verify -q "${GATE_HEAD:-HEAD}^{commit}") || { echo "GATE_HEAD ${GATE_HEAD:-HEAD} does not resolve"; exit 1; }
# SMOKE: the commit read is not the checkout's HEAD, so the working-tree checks cannot read it.
SMOKE=0
[ "$HEAD_SHA" = "$(git rev-parse HEAD)" ] || SMOKE=1
if [ -n "${GATE_BASE:-}" ]; then
  BASE=$(git rev-parse --verify -q "$GATE_BASE^{commit}") || { echo "GATE_BASE $GATE_BASE does not resolve"; exit 1; }
else
  MB_MAIN=$(git merge-base "$HEAD_SHA" origin/main) || { echo "cannot resolve merge-base with origin/main"; exit 1; }
  BASE=$MB_MAIN
  if git rev-parse --verify -q "$STACK_REF^{commit}" >/dev/null; then
    MB_W2=$(git merge-base "$HEAD_SHA" "$STACK_REF" 2>/dev/null || true)
    # Stacked: W2's work sits between main and this wave (main's merge-base is behind W2's).
    if [ -n "$MB_W2" ] && [ "$MB_W2" != "$MB_MAIN" ] && git merge-base --is-ancestor "$MB_MAIN" "$MB_W2"; then BASE=$MB_W2; fi
  fi
fi
MODE=full
case "${1:-}" in --print-base) echo "$BASE"; exit 0 ;; --no-ci) MODE=no-ci ;; --negative-only) MODE=negative ;; "") ;; *) echo "unknown argument: $1"; exit 2 ;; esac
echo "base: $(git log -1 --format='%h %s' "$BASE") · HEAD read: $(git log -1 --format='%h %s' "$HEAD_SHA") · mode: $MODE · smoke: $SMOKE · python: $("$PY" -V 2>&1)"
FAIL=0
check() { local name="$1" body="$2"; echo ""; echo "── $name"; if ( eval "$body" ); then echo "   ✅ PASS: $name"; else echo "   ❌ FAIL: $name"; FAIL=1; fi; }
warn() { local name="$1" body="$2"; echo ""; echo "── $name"; if ( eval "$body" ); then echo "   ✅ ok: $name"; else echo "   ⚠️  WARN (not a failure): $name"; fi; }

KIT=scripts/ralph/prd-268w3.json
PRD_DOC=docs/PRDS/PRD-268-PROACTIVE-AUTO.md
O=orchestrator
TESTS=$O/tests
VERSIONS=$O/alembic/versions
MANIFEST=$O/reports/route-manifest.json
SURFACE=$O/reports/config-surface.json
PRO=$O/services/proactive
CAL=$O/services/calibration
EVALS=$O/evals/proactive
DISC=$O/modules/tools/discovery
EXEC=$O/modules/tools/execution
CHATBOT=$O/consumers/chatbot
HB=$O/services/heartbeat_service.py
STS=$O/services/scheduled_task_service.py
STEPS=$O/services/scheduled_task_steps.py
LOOP=$EXEC/tool_loop.py
LANE=$DISC/check_in_lane.py
EXECUTOR=$DISC/platform_executor.py
DISCOVERY=$O/services/mobile/discovery.py
AUTO_MESSAGE=$O/services/notification_events/auto_message.py
FU_ROUTER=$O/api/proactive_follow_ups.py
CAL_ROUTER=$O/api/auto_calibration.py
FE=frontend
PROC=$FE/components/proactive
NEW_REV=prd268_w3_follow_ups
W2_HEAD=prd268_w2_delivery
# K3: files over 800 lines this wave may touch; none may grow.
NO_GROW="$CHATBOT/service.py $LOOP $EXECUTOR $EXEC/unified_executor.py $O/modules/memory/unified_memory_service.py $HB $O/api/chat.py $O/modules/tools/formatting/result_formatter.py $O/main.py $O/core/llm/manager.py $FE/components/chatbot/chat.tsx $FE/lib/api-client.ts $FE/app/globals.css"
# Over 800 lines and allowed to grow: settings live only in config.py; the two reports are generated lists.
GROW_OK="$O/config.py $MANIFEST $SURFACE"
# K20: the pinned tests that move with approved changes; each keeps at least its test count.
PIN_TESTS="$TESTS/test_prd209_alembic_single_head.py $TESTS/test_prd236_w1_routes.py $TESTS/test_prd268_proactive_models.py $TESTS/test_prd266_auto_message.py $TESTS/test_prd266_discovery.py"
EARLIER_KIT_FILES="scripts/ralph/prd-268w1.json scripts/ralph/PROMPT_build_prd268w1.md scripts/ralph/PROMPT_review_prd268w1.md scripts/ralph/acceptance-prd268w1.sh scripts/ralph/prd-268w2.json scripts/ralph/PROMPT_build_prd268w2.md scripts/ralph/PROMPT_review_prd268w2.md scripts/ralph/acceptance-prd268w2.sh"
# K17: the SDK's surface and versions are W2's.
SDK_PATHS="$FE/packages/theme-sdk $FE/lib/theme-sdk/surface $O/services/themes/data/sdk-surface.json $FE/lib/theme-host/shared-modules.json"
# K4: never touched.
UNTOUCHED="$CHATBOT/service.py $FE/components/chatbot/message.tsx $FE/components/chatbot/chat.tsx $FE/components/chatbot/multimodal-input.tsx"

# --- reading the commit -----------------------------------------------------------------------------
src() { git show "$HEAD_SHA:$1" 2>/dev/null; }
exists() { git cat-file -e "$HEAD_SHA:$1" 2>/dev/null; }
lines_at() { git show "$1:$2" 2>/dev/null | wc -l | tr -d ' '; }
has() { exists "$2" && src "$2" | grep -qE -e "$1"; }      # has <ERE> <file>: false when the file is missing
lacks() { ! has "$1" "$2"; }
all_exist() { local f; for f in "$@"; do exists "$f" || { echo "   missing: $f"; return 1; }; done; }
# The wave's ADDED lines under the given paths: tests, the kit and docs excluded (negative greps only).
added_code() { git diff "$BASE" "$HEAD_SHA" -- "$@" ':!orchestrator/tests' ':!*__tests__*' ':!*.test.ts' ':!*.test.tsx' ':!e2e' ':!scripts/ralph' ':!docs' | grep '^+' | grep -v '^+++'; }
added_files() { git diff --name-only --diff-filter=A "$BASE" "$HEAD_SHA" -- "$@"; }
# Branch-scoped: each file exists AND was added by this wave (a test that predates it proves nothing about it).
new_files() { local f bad=0; for f in "$@"; do
    if ! exists "$f"; then echo "   missing: $f"; bad=1
    elif [ -z "$(added_files "$f")" ]; then echo "   not new on this branch: $f"; bad=1; fi
  done; [ $bad -eq 0 ]; }
changed() { ! git diff --quiet "$BASE" "$HEAD_SHA" -- "$@"; }
untouched() { git diff --quiet "$BASE" "$HEAD_SHA" -- "$@"; }
# added_in <ERE> <file>: a line this wave added to <file> matches (a positive smoke signal; the tests are the proof).
added_in() { git diff "$BASE" "$HEAD_SHA" -- "$2" | grep '^+' | grep -v '^+++' | grep -qE -e "$1"; }

env_read_added()   { added_code orchestrator ':!orchestrator/config.py' | grep -E 'os\.(getenv|environ)' | grep -q .; }
# evals/proactive/__main__.py prints its bars (ruff.toml exempts **/__main__.py from T20); scripts may print.
print_added()      { added_code orchestrator ':!orchestrator/scripts' ':(exclude,glob)orchestrator/**/__main__.py' | grep -E '^\+\s*print\(' | grep -q .; }
raw_fetch_added()  { added_code frontend | grep -E "fetch\(\s*['\"\`]/api" | grep -q .; }
inner_html_added() { added_code frontend | grep -E 'dangerouslySetInnerHTML|\.innerHTML\s*=|\.outerHTML\s*=|insertAdjacentHTML|document\.write\(' | grep -q .; }
orange_added()     { added_code frontend | grep -E '\b(bg|text|border|ring|from|to|via|fill|stroke)-orange-[0-9]' | grep -q .; }
hex_added()        { added_code "$PROC" frontend/lib/chat frontend/hooks | grep -E "#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b['\" ;)]" | grep -q .; }
rounded3_added()   { added_code "$PROC" | grep -E '\brounded-3xl\b' | grep -q .; }
autoplay_added()   { added_code frontend | grep -iE '\bautoplay\b|autoPlay\s*[=:]' | grep -q .; }
client_added()     { added_code orchestrator | grep -E '^\+\s*(import|from)\s+(httpx|requests|aiohttp|urllib3)\b' | grep -q .; }
factory_in_lane()  { added_code "$PRO" "$CAL" "$EVALS" $O/modules/proactive | grep -E 'AgentFactory|agent_factory|execute_with_prompt' | grep -q .; }
# W2's K10 holds: no Always allow control anywhere.
always_allow_added() { added_code frontend/components frontend/lib/theme-sdk/host-cards orchestrator/api | grep -E "Always allow|always[-_]allow['\"]? *[:=(]|/always" | grep -vE '^\+\s*(//|#|\*|/\*)' | grep -q .; }
# K26: no W4 surface (Mission or Playbook proposals, Telegram/Slack start-or-hide, Hide learning into field memory).
later_wave_added() {
  added_code orchestrator frontend | grep -E "Reply start or hide|platform_propose_(mission|playbook)|hide_reason.*field_memory|field_memory.*hide_reason" | grep -vE '^\+\s*(//|#|\*)' | grep -q . && { echo "   a W4 surface was added"; return 0; }
  return 1
}
# Mirrors .github/workflows/dco.yml, which checks `git rev-list --no-merges`: a merge commit is exempt.
every_commit_signed() { [ -z "$(git log --no-merges --format='%H %(trailers:key=Signed-off-by,valueonly)' "$BASE".."$HEAD_SHA" | awk 'NF<2{print $1}')" ]; }
file_sizes_ok() { local f n bad=0; for f in $(added_files orchestrator frontend docs services e2e | grep -E '\.(py|ts|tsx|js|jsx|mjs|css|md|json|ya?ml|sh)$'); do exists "$f" || continue; n=$(lines_at "$HEAD_SHA" "$f"); [ "$n" -le 800 ] || { echo "   $f: $n lines"; bad=1; }; done; [ $bad -eq 0 ]; }
not_grown() { local f="$1" b h; b=$(lines_at "$BASE" "$f"); h=$(lines_at "$HEAD_SHA" "$f"); echo "   $f: base $b, HEAD $h"; [ "$h" -le "$b" ]; }
named_files_not_grown() { local f bad=0; for f in $NO_GROW; do exists "$f" || continue; not_grown "$f" || bad=1; done; [ $bad -eq 0 ]; }
# Every modified file that was over 800 lines on the base, outside GROW_OK, has not grown.
no_big_file_grew() { local f b h bad=0; for f in $(git diff --name-only --diff-filter=M "$BASE" "$HEAD_SHA"); do
    exists "$f" || continue
    case " $GROW_OK " in *" $f "*) continue ;; esac
    b=$(lines_at "$BASE" "$f"); h=$(lines_at "$HEAD_SHA" "$f")
    [ "$b" -gt 800 ] && [ "$h" -gt "$b" ] && { echo "   $f grew: base $b, HEAD $h"; bad=1; }
  done; [ $bad -eq 0 ]; }
at_most() { local n; n=$(lines_at "$HEAD_SHA" "$1"); echo "   $1: $n lines (at most $2)"; [ "$n" -le "$2" ]; }
config_growth() { local b h; b=$(lines_at "$BASE" $O/config.py); h=$(lines_at "$HEAD_SHA" $O/config.py); echo "   config.py: base $b, HEAD $h (settings only: about 4 lines per setting, 10 settings)"; [ "$h" -le $((b + 45)) ]; }
test_fn_count() { git show "$1:$2" 2>/dev/null | awk '/^(async )?def test_/{n++} END{print n+0}'; }
tests_not_weakened() { local d a; d=$(git diff "$BASE" "$HEAD_SHA" -- $TESTS frontend e2e | awk '/^-[ \t]*(async )?def test_|^-[ \t]*(it|test)(\.each\([^)]*\))?\(/{n++} END{print n+0}'); a=$(git diff "$BASE" "$HEAD_SHA" -- $TESTS frontend e2e | awk '/^\+[ \t]*(async )?def test_|^\+[ \t]*(it|test)(\.each\([^)]*\))?\(/{n++} END{print n+0}'); echo "   test functions removed $d, added $a"; [ "$a" -ge "$d" ]; }
test_count_kept() { local f="$1" b h; exists "$f" || { echo "   $f is missing"; return 1; }; b=$(test_fn_count "$BASE" "$f"); h=$(test_fn_count "$HEAD_SHA" "$f"); echo "   $f: test functions base $b, HEAD $h"; [ "$h" -ge "$b" ]; }
pin_tests_kept() { local f bad=0; for f in $PIN_TESTS; do test_count_kept "$f" >/dev/null || { test_count_kept "$f"; bad=1; }; done; [ $bad -eq 0 ] && echo "   every K20 pin file keeps its test count"; [ $bad -eq 0 ]; }
old_migrations_untouched() { [ -z "$(git diff --name-only --diff-filter=MDR "$BASE" "$HEAD_SHA" -- "$VERSIONS")" ]; }
# The two checks that read a working tree: run when HEAD is the commit read; skipped in a GATE_HEAD smoke.
code_shape_ok() {
  [ "$SMOKE" -eq 0 ] || { echo "   skipped: a GATE_HEAD smoke of another commit (the script reads the working tree)"; return 0; }
  "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || { echo "   $PY is older than 3.10: put /opt/homebrew/bin first on PATH"; return 1; }
  "$PY" scripts/ci/check_changed_code_shape.py --base "$BASE" | tail -15
  [ "${PIPESTATUS[0]}" -eq 0 ]
}
hierarchy_gate_ok() {
  [ "$SMOKE" -eq 0 ] || { echo "   skipped: a GATE_HEAD smoke of another commit (the script reads the working tree)"; return 0; }
  "$PY" orchestrator/scripts/check_hierarchy_gate.py >/dev/null
}

# A function by its name ('Class.method', or a plain name anywhere), read from a commit with git. One python
# helper for the byte-identical, length, content and decorator checks.
fn_tool() { "$PY" - "$@" <<'PY'
import ast, re, subprocess, sys
op, ref, path, qual = sys.argv[1:5]
extra = sys.argv[5:]
def source(rev):
    out = subprocess.run(["git", "show", f"{rev}:{path}"], capture_output=True, text=True)
    return out.stdout if out.returncode == 0 else None
def find(src):
    if src is None:
        return None
    tree = ast.parse(src)
    owner, _, name = qual.rpartition(".")
    funcs = (ast.FunctionDef, ast.AsyncFunctionDef)
    if owner:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == owner:
                for item in node.body:
                    if isinstance(item, funcs) and item.name == name:
                        return item
        return None
    return next((n for n in ast.walk(tree) if isinstance(n, funcs) and n.name == name), None)
def segment(src, node):
    return "\n".join(src.splitlines()[node.lineno - 1:node.end_lineno])
head = source(ref)
node = find(head)
if node is None:
    print(f"   {qual} not found in {path}"); sys.exit(1)
if op == "same":
    base = source(extra[0]); old = find(base)
    if old is None:
        print(f"   {qual} not found in {path} on the base"); sys.exit(1)
    if segment(base, old) != segment(head, node):
        print(f"   {qual} in {path} differs from the base"); sys.exit(1)
    print(f"   {qual}: byte-identical to the base ({node.end_lineno - node.lineno + 1} lines)"); sys.exit(0)
if op == "short":
    lines = head.splitlines(); doc = set()
    first = node.body[0] if node.body else None
    if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) and isinstance(first.value.value, str):
        doc = set(range(first.lineno, first.end_lineno + 1))
    n = sum(1 for i in range(node.lineno, node.end_lineno + 1)
            if lines[i - 1].strip() and not lines[i - 1].strip().startswith("#") and i not in doc)
    print(f"   {qual}: {n} code lines (limit {extra[0]})"); sys.exit(0 if n <= int(extra[0]) else 1)
if op == "contains":
    ok = re.search(extra[0], segment(head, node)) is not None
    print(f"   {qual} {'mentions' if ok else 'does not mention'} /{extra[0]}/"); sys.exit(0 if ok else 1)
if op == "decorated":
    names = [ast.unparse(d) for d in node.decorator_list]
    print(f"   {qual} decorators: {names}"); sys.exit(0 if any(extra[0] in n for n in names) else 1)
print(f"   unknown op {op}"); sys.exit(2)
PY
}
fn_unchanged() { fn_tool same "$HEAD_SHA" "$1" "$2" "$BASE"; }
fn_short() { fn_tool short "$HEAD_SHA" "$1" "$2" "$3"; }
fn_contains() { fn_tool contains "$HEAD_SHA" "$1" "$2" "$3"; }
decorated() { fn_tool decorated "$HEAD_SHA" "$1" "$2" "$3"; }
# A module-level literal's string members (its whole value: dict keys and values included), read from the commit.
literal_strings() { "$PY" - "$HEAD_SHA" "$1" "$2" <<'PY'
import ast, subprocess, sys
ref, path, name = sys.argv[1:4]
out = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, text=True)
if out.returncode != 0:
    sys.exit(1)
for node in ast.parse(out.stdout).body:
    targets = node.targets if isinstance(node, ast.Assign) else ([node.target] if isinstance(node, ast.AnnAssign) else [])
    if any(isinstance(t, ast.Name) and t.id == name for t in targets) and node.value is not None:
        found = sorted({n.value for n in ast.walk(node.value) if isinstance(n, ast.Constant) and isinstance(n.value, str)})
        print(" ".join(found)); sys.exit(0)
sys.exit(1)
PY
}
set_has() { local got; got=$(literal_strings "$1" "$2") || { echo "   $2 not found in $1"; return 1; }
  case " $got " in *" $3 "*) return 0 ;; esac; echo "   $2 in $1 lacks '$3'"; return 1; }
# A boolean switch, if declared, defaults to off (config.py's convention: os.getenv("NAME", "<default>")).
switch_off() { local name="$1" line d; line=$(src $O/config.py | grep -E "^\s+$name\b" | head -1)
  [ -z "$line" ] && { echo "   $name not declared"; return 0; }
  echo "   $line"
  echo "$line" | grep -qE ':\s*bool\s*=\s*False\b' && return 0
  d=$(echo "$line" | sed -nE "s/.*getenv\(\"$name\", *\"([^\"]*)\".*/\1/p; s/.*getenv\(\"$name\"\) *or *\"([^\"]*)\".*/\1/p" | head -1 | tr 'A-Z' 'a-z')
  case "$d" in false|0|no|off) return 0 ;; *) echo "   the default is '${d}', not off"; return 1 ;; esac; }
# K17: the SDK and host contract versions are the base's.
versions_kept() { local f=$O/services/themes/contract.py b h
  b=$(git show "$BASE:$f" 2>/dev/null | grep -E '^(SDK_VERSION|THEME_HOST_VERSION) *=')
  h=$(src "$f" | grep -E '^(SDK_VERSION|THEME_HOST_VERSION) *=')
  echo "   base: $(echo $b) · HEAD: $(echo $h)"; [ -n "$b" ] && [ "$b" = "$h" ]; }
# K14: no gold data is tracked under the evals folder.
no_gold_tracked() { local tracked; tracked=$(git ls-tree -r --name-only "$HEAD_SHA" -- "$EVALS" | awk '/\.jsonl$/'); [ -z "$tracked" ] || { echo "$tracked" | sed 's/^/   tracked: /'; return 1; }; }

# --- the migration ------------------------------------------------------------------------------
base_head() { git show "$BASE:$TESTS/test_prd209_alembic_single_head.py" 2>/dev/null | grep -E '^EXPECTED_HEAD *= *"' | head -1 | sed -E 's/.*"([^"]+)".*/\1/'; }
new_revisions() { added_files "$VERSIONS" | grep '\.py$'; }
one_revision() { local n; n=$(new_revisions | awk 'NF' | wc -l | tr -d ' '); echo "   new revisions: $n ($(new_revisions | tr '\n' ' '))"; [ "$n" -eq 1 ]; }
revision_ok() {
  local f bh; f="$VERSIONS/$NEW_REV.py"; bh=$(base_head)
  exists "$f" || { echo "   $f is missing"; return 1; }
  [ -n "$bh" ] || { echo "   cannot read the base's EXPECTED_HEAD"; return 1; }
  has "^revision[^=]*= *['\"]$NEW_REV['\"]" "$f" || { echo "   $f: revision is not $NEW_REV"; return 1; }
  has "^down_revision[^=]*= *['\"]$bh['\"]" "$f" || { echo "   $f: down_revision is not the base head $bh"; return 1; }
  src "$f" | grep -qiE "DROP CONSTRAINT IF EXISTS ck_scheduled_task_deliver_as" || { echo "   $f: the deliver_as CHECK is not dropped IF EXISTS"; return 1; }
  has "'board_task', *'follow_up'" "$f" || { echo "   $f: the widened deliver_as CHECK lacks 'follow_up'"; return 1; }
  src "$f" | grep -qiE "ADD COLUMN IF NOT EXISTS for_user_id INTEGER" || { echo "   $f: no ADD COLUMN IF NOT EXISTS for_user_id INTEGER"; return 1; }
  src "$f" | grep -qiE "CREATE INDEX IF NOT EXISTS ix_scheduled_tasks_follow_ups" || { echo "   $f: no ix_scheduled_tasks_follow_ups"; return 1; }
  src "$f" | grep -qiE "DROP CONSTRAINT IF EXISTS ck_proactive_runs_outcome" || { echo "   $f: the outcome CHECK is not dropped IF EXISTS"; return 1; }
  has "'posted'" "$f" || { echo "   $f: the outcome CHECK lacks 'posted'"; return 1; }
  src "$f" | grep -qiE "CREATE TABLE IF NOT EXISTS proactive_calibrations" || { echo "   $f: no CREATE TABLE IF NOT EXISTS proactive_calibrations"; return 1; }
  ! src "$f" | grep -qiE "mobile_digest_posts|mobile_devices|mobile_push_tickets|proactive_person_settings|ALTER TABLE proactive_suggestions" || { echo "   $f touches a table K6 leaves alone"; return 1; }
  has "EXPECTED_HEAD = \"$NEW_REV\"" $TESTS/test_prd209_alembic_single_head.py || { echo "   test_prd209 EXPECTED_HEAD is not $NEW_REV"; return 1; }
  has "$NEW_REV\\.py" $TESTS/test_prd236_w1_routes.py || { echo "   test_prd236_w1_routes.py does not read $NEW_REV.py"; return 1; }
  has "EXPECTED_HEAD = \"$NEW_REV\"" $TESTS/test_prd236_w1_routes.py || { echo "   test_prd236_w1_routes.py does not assert EXPECTED_HEAD = $NEW_REV"; return 1; }
  echo "   $NEW_REV chains onto $bh; deliver_as widened; for_user_id and its index; posted; proactive_calibrations; both pins moved"
}
one_head() { "$PY" - "$HEAD_SHA" "$VERSIONS" <<'PY'
import ast, subprocess, sys
ref, folder = sys.argv[1:3]
listing = subprocess.run(["git", "ls-tree", "-r", ref, "--", folder], capture_output=True, text=True, check=True).stdout
blobs = [(line.split()[2], line.split("\t", 1)[1]) for line in listing.splitlines() if line.endswith(".py")]
# Bytes, never text: --batch sizes are byte counts, and the revisions' docstrings hold non-ASCII dashes.
batch = subprocess.run(["git", "cat-file", "--batch"], input=("\n".join(sha for sha, _ in blobs) + "\n").encode(),
                       capture_output=True, check=True).stdout
revs, downs, pos = {}, set(), 0
for sha, path in blobs:
    header_end = batch.index(b"\n", pos)
    size = int(batch[pos:header_end].split()[2])
    body = batch[header_end + 1:header_end + 1 + size].decode("utf-8")
    pos = header_end + 1 + size + 1
    try:
        tree = ast.parse(body)
    except SyntaxError as e:
        print(f"   {path}: {e}"); sys.exit(1)
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
    revs[rev] = path
    downs.update(down if isinstance(down, (list, tuple)) else ([down] if down else []))
heads = sorted(set(revs) - downs)
print(f"   {len(revs)} revisions, heads: {heads}")
sys.exit(0 if len(heads) == 1 else 1)
PY
}

# --- routes and settings -------------------------------------------------------------------------
routes_in_manifest() { "$PY" - "$HEAD_SHA" "$MANIFEST" "$@" <<'PY'
import json, re, subprocess, sys
ref, path = sys.argv[1:3]
data = json.loads(subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, text=True, check=True).stdout)
pairs = {(r.get("method", ""), r.get("path", "")) for r in data["routes"]}
missing = []
for want in sys.argv[3:]:
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
    has "^\s+$s\b" $O/config.py || { echo "   config.py lacks $s"; bad=1; }
    has "\"$s\"" "$SURFACE" || { echo "   config-surface.json lacks $s"; bad=1; }
  done; [ $bad -eq 0 ]; }
router_spec() { has "RouterSpec\(\"$1\"" $O/router_manifest.py; }
capability() { has "\"$1\": *\(" "$DISCOVERY"; }
configured_capability() { src "$DISCOVERY" | grep -A6 '^CONFIGURED_CAPABILITIES' | grep -q "\"$1\""; }
# The new routers' handlers are plain def (sync session; AGENTS.md). A router not built yet is skipped.
routers_plain_def() { "$PY" - "$HEAD_SHA" "$FU_ROUTER" "$CAL_ROUTER" <<'PY'
import ast, subprocess, sys
ref, bad = sys.argv[1], []
for path in sys.argv[2:]:
    out = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, text=True)
    if out.returncode != 0:
        continue
    for node in ast.walk(ast.parse(out.stdout)):
        if isinstance(node, ast.AsyncFunctionDef):
            for d in node.decorator_list:
                f = d.func if isinstance(d, ast.Call) else d
                if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "router":
                    bad.append(f"{path}:{node.lineno} {node.name} is async def")
print("\n".join(f"   {b}" for b in bad) or "   every new route handler is a plain def"); sys.exit(1 if bad else 0)
PY
}
# A mutating route carries require_workspace_permission (K22, the authz sweep's class (a)).
route_gated() { src "$1" | grep -B4 -A4 -E "$2" | grep -q 'require_workspace_permission'; }
# The four chat-only tools sit in check_in_lane's CHAT_ONLY_ACTIONS (K10): the set is declared, and the lane
# module names each tool (as a literal in the set, or as the constant the set holds).
chat_only_ok() { local t; has '^CHAT_ONLY_ACTIONS\b' "$LANE" || { echo "   CHAT_ONLY_ACTIONS is not declared in $LANE"; return 1; }
  for t in platform_schedule_follow_up platform_list_follow_ups platform_cancel_follow_up platform_offer_replies; do
    has "[\"']$t[\"']" "$LANE" || { echo "   $LANE does not name $t"; return 1; }; done
  echo "   CHAT_ONLY_ACTIONS is declared and the lane names the three follow-up tools and platform_offer_replies"; }
# An action's registered definition (its ActionDefinition(...) call in the actions file, read with ast):
# action_def <file> <action> <keyword> <expected>, e.g. permission_level read, promoted True.
action_def() { "$PY" - "$HEAD_SHA" "$@" <<'PY'
import ast, subprocess, sys
ref, path, action, key, want = sys.argv[1:6]
out = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, text=True)
if out.returncode != 0:
    print(f"   {path} is missing"); sys.exit(1)
for node in ast.walk(ast.parse(out.stdout)):
    func = getattr(node, "func", None)
    if not isinstance(node, ast.Call) or getattr(func, "id", getattr(func, "attr", "")) != "ActionDefinition":
        continue
    kw = {k.arg: k.value for k in node.keywords if k.arg}
    if not (isinstance(kw.get("name"), ast.Constant) and kw["name"].value == action):
        continue
    got = kw.get(key)
    value = repr(got.value) if isinstance(got, ast.Constant) else (ast.unparse(got) if got is not None else "unset")
    print(f"   {action}: {key} = {value}"); sys.exit(0 if value.strip("'\"") == want else 1)
print(f"   {action} is not registered in {path}"); sys.exit(1)
PY
}
action_level() { action_def "$1" "$2" permission_level "$3"; }
action_promoted() { action_def "$1" "$2" promoted True; }
wired() { has "\"$1\"" "$EXECUTOR"; }
allow_listed() { has "\"$1\"" $O/scripts/check_hierarchy_gate.py; }
# same_line <file> <ERE>: the lines matching ERE are the base's (a constant this wave must not widen).
same_line() { local b h; b=$(git show "$BASE:$1" 2>/dev/null | grep -E "$2"); h=$(src "$1" | grep -E "$2"); echo "   base: ${b:-none} · HEAD: ${h:-none}"; [ -n "$b" ] && [ "$b" = "$h" ]; }
# A mark counts when it sits outside backticks (the instructions quote the marks in backticks).
all_acs_done() { "$PY" - "$HEAD_SHA" "$KIT" <<'PY'
import json, re, subprocess, sys
ref, path = sys.argv[1:3]
out = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, text=True)
if out.returncode != 0:
    print(f"   {path} is not committed"); sys.exit(1)
d = json.loads(out.stdout); bad = []
strip = lambda x: re.sub(r"`[^`]*`", "", x)
for s in d["userStories"]:
    for raw in s["acceptanceCriteria"]:
        ac = strip(raw)
        if "→ DONE" in ac or "→ OWNER:" in ac: continue
        bad.append(f"{s['id']}: {raw[:80]}")
print("\n".join(bad)); sys.exit(1 if bad else 0)
PY
}

# --- CI and the PR (full mode) -------------------------------------------------------------------------
# The commit CI tested: HEAD, or, when HEAD and its parents are kit-only '[skip ci]' commits (scripts/ralph/
# marks, gate or prompt fixes after the wave's CI commit), the nearest ancestor that is not.
ci_commit_for_head() {
  local sha="$HEAD_SHA"
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
  [ "$sha" = "$HEAD_SHA" ] || echo "   HEAD is kit-only [skip ci] commits over $(git log -1 --format=%h "$sha"): reading that commit's run"
  command -v gh >/dev/null || { echo "   gh is required"; return 1; }
  [ -n "$(gh pr list --head "$br" --state open --json number --jq '.[0].number' 2>/dev/null)" ] || { echo "   no open PR for $br: test.yml runs on pull_request only (#993)"; return 1; }
  got=$(wait_for_run test.yml "$sha") || { echo "   no completed test.yml run for $sha after 60 min"; return 1; }
  id="${got%% *}"
  jobs=$(gh run view "$id" --json jobs --jq '.jobs[] | "\(.conclusion)\t\(.name)"')
  local REQUIRED=("Which lanes this change needs" "orchestrator-tests" "Alembic from-zero" "Schema-drift check" "Frontend CI" "Code standards on changed lines" "Orchestrator module + integration tests" "Prod images built" "CLI host (PRD-234)")
  local PATH_GATED=("Prod images built" "CLI host (PRD-234)")
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
# The browser lane (US-W03c) runs in browser-e2e.yml on the PR. A WebKit 'internal error' on reload is a known
# Playwright 1.57 flake: rerun the failed job before calling it red.
browser_lane_green() {
  local sha got; sha=$(ci_commit_for_head)
  got=$(wait_for_run browser-e2e.yml "$sha") || { echo "   no completed browser-e2e.yml run for $sha after 60 min"; return 1; }
  echo "   browser-e2e.yml run ${got%% *}: ${got##* }"; [ "${got##* }" = "success" ]
}
open_pr() { gh pr list --head "$(git rev-parse --abbrev-ref HEAD)" --state open --json number --jq '.[0].number' 2>/dev/null; }
pr_body_records() {
  local n body s miss=0
  n=$(open_pr); [ -n "$n" ] || { echo "   no open PR"; return 1; }
  body=$(gh pr view "$n" --json body --jq .body 2>/dev/null)
  for s in "#1360" "#1359" "Stacked on" "DRAFT" "REVIEW_PASS" "prd268_w3_follow_ups" "CALIBRATION_ENABLED" "gold sets" "follow-up" "reply chips" "--subject" "AI assistance"; do
    printf '%s' "$body" | grep -qF -- "$s" || { echo "   the PR body lacks: $s"; miss=1; }
  done
  [ $miss -eq 0 ] && echo "   PR #$n records #1360, #1359, the stack, the draft rule, the migration, calibration off, gold sets local, follow-ups, reply chips, the squash rule and the AI disclosure"
}
# The stacked-PR trap (K1): while W2's PR is open this PR targets W2's branch; once W2 merged, main.
pr_base_ok() {
  local n base w2
  n=$(open_pr); [ -n "$n" ] || { echo "   no open PR"; return 1; }
  base=$(gh pr view "$n" --json baseRefName --jq .baseRefName 2>/dev/null)
  w2=$(gh pr list --head "$W2_BRANCH" --state all --json state --jq '.[0].state' 2>/dev/null)
  echo "   PR #$n base: $base; W2's PR: ${w2:-none}"
  case "$w2" in
    MERGED) [ "$base" = "main" ] ;;
    *) [ "$base" = "$W2_BRANCH" ] || [ "$base" = "main" ] ;;
  esac
}
# K1: the PR stays a draft until REVIEW_PASS; the owner undrafts it.
pr_is_draft() { local n d; n=$(open_pr); [ -n "$n" ] || { echo "   no open PR"; return 1; }; d=$(gh pr view "$n" --json isDraft --jq .isDraft 2>/dev/null); echo "   PR #$n isDraft: $d"; [ "$d" = "true" ]; }

if [ "$MODE" != "negative" ]; then
  echo ""; echo "=== Seeded and stacked"
  check "the PRD is tracked at $PRD_DOC" "exists $PRD_DOC"
  check "the kit is committed and valid JSON" "exists $KIT && src $KIT | \"$PY\" -m json.tool >/dev/null"
  check "W1's and W2's work is under this wave (the proactive models, the runner, the message parts, the brief, the defaults)" "all_exist $O/core/models/proactive.py $PRO/runner.py $O/services/chat_parts.py $O/services/brief/build.py $PROC/thread-cards.tsx $PROC/composer-with-suggestion.tsx $FE/hooks/use-calibration.ts"

  echo ""; echo "=== Stories (priority order; positive: red on the base by design)"
  check "US-C11a one revision $NEW_REV on the base head: deliver_as, for_user_id, posted, proactive_calibrations, both pins" "one_revision && revision_ok"
  check "US-C11a the models follow (RUN_POSTED, ProactiveCalibration, DELIVER_FOLLOW_UP; DELIVERY_MODES unchanged); W1's model pin moved; the test is new" "has 'RUN_POSTED' $O/core/models/proactive.py && has 'class ProactiveCalibration' $O/core/models/proactive.py && has 'DELIVER_FOLLOW_UP' $STEPS && same_line $STEPS '^DELIVERY_MODES *=' && changed $TESTS/test_prd268_proactive_models.py && new_files $TESTS/test_prd268w3_migration.py"
  check "US-C11b list_tasks is short and every generic reader leaves follow-ups out; the test is new" "fn_short $STS ScheduledTaskService.list_tasks 50 && fn_contains $STS ScheduledTaskService.list_tasks '(?i)follow_?up' && fn_contains $STS ScheduledTaskService._limit_error '(?i)follow_?up' && fn_contains $STS ScheduledTaskService.update_task_status '(?i)follow_?up' && added_in '(follow_up|FOLLOW_UP)' $O/services/activity_scheduled_task_items.py && added_in '(follow_up|FOLLOW_UP)' $PRO/signals.py && added_in '(follow_up|FOLLOW_UP)' $DISC/handlers_analytics.py && new_files $TESTS/test_prd268w3_follow_ups_kept_private.py"
  check "US-C11c the caps, the service under the advisory lock, the three tools, the context decorator, chat-only; the tests are new" "all_exist $O/modules/proactive/follow_up_caps.py $PRO/follow_ups.py $DISC/actions_follow_ups.py $DISC/handlers_follow_ups.py $DISC/follow_up_context.py && has 'pg_advisory_xact_lock' $PRO/follow_ups.py && decorated $EXECUTOR PlatformActionExecutor.execute carries_the_follow_up && action_level $DISC/actions_follow_ups.py platform_schedule_follow_up write && action_level $DISC/actions_follow_ups.py platform_list_follow_ups read && action_level $DISC/actions_follow_ups.py platform_cancel_follow_up write && wired platform_schedule_follow_up && wired platform_list_follow_ups && wired platform_cancel_follow_up && allow_listed platform_schedule_follow_up && allow_listed platform_cancel_follow_up && chat_only_ok && settings_declared PROACTIVE_FOLLOW_UP_MIN_MINUTES PROACTIVE_FOLLOW_UP_MAX_DAYS PROACTIVE_FOLLOW_UP_MAX_PENDING PROACTIVE_FOLLOW_UP_MAX_PER_DAY PROACTIVE_FOLLOW_UP_MAX_PER_WORKSPACE_PER_DAY PROACTIVE_FOLLOW_UP_CONTEXT_MESSAGES PROACTIVE_FOLLOW_UP_REASON_CHARS && new_files $TESTS/test_prd268w3_follow_up_caps.py $TESTS/test_prd268w3_follow_up_tools.py"
  check "US-C11d the firing: one branch in _deliver, the fire module, the follow_up kind, the loop's text, announced; the test is new" "fn_contains $STS ScheduledTaskService._deliver '(?i)follow_?up' && all_exist $PRO/follow_up_fire.py $PRO/follow_up_prompts.py && has 'def fire_follow_up' $PRO/follow_up_fire.py && has 'KIND_FOLLOW_UP' $PRO/runs.py && has 'final_text' $PRO/loop.py && set_has $AUTO_MESSAGE ANNOUNCED_ORIGINS follow_up && changed $TESTS/test_prd266_auto_message.py && at_most $STS 800 && new_files $TESTS/test_prd268w3_follow_up_fire.py"
  check "US-C12a platform_offer_replies: read, promoted, wired; the test is new" "all_exist $DISC/actions_replies.py $DISC/handlers_replies.py && action_level $DISC/actions_replies.py platform_offer_replies read && action_promoted $DISC/actions_replies.py platform_offer_replies && wired platform_offer_replies && has 'register_replies_actions' $DISC/platform_actions.py && new_files $TESTS/test_prd268w3_offer_replies_tool.py"
  check "US-C12b the terminal turn: a decorator on ToolLoopExecutor.run; the test is new" "all_exist $EXEC/reply_chips_turn.py && decorated $LOOP ToolLoopExecutor.run ends_on_reply_chips && has 'reply_chips' $EXEC/reply_chips_turn.py && new_files $TESTS/test_prd268w3_reply_chips_terminal.py"
  check "US-C12c the turn's parts: reply_chips@1 and follow_up@1 through the receipts decorators; the test is new" "all_exist $CHATBOT/turn_parts.py && has 'reply_chips@1' $CHATBOT/turn_parts.py && has 'follow_up@1' $CHATBOT/turn_parts.py && has 'turn_parts' $CHATBOT/receipts.py && new_files $TESTS/test_prd268w3_turn_parts.py"
  check "US-C11e the follow-up routes: RouterSpec, gated cancel, in the manifest; the test is new" "all_exist $FU_ROUTER && router_spec api.proactive_follow_ups && route_gated $FU_ROUTER 'cancel' && routes_in_manifest 'GET /api/proactive/follow-ups' 'POST /api/proactive/follow-ups/\\{[a-z_]+\\}/cancel' && new_files $TESTS/test_prd268w3_follow_up_routes.py"
  check "US-C13a the scorers, the bars, the gold reader; .gitignore covers the gold sets; the settings; the tests are new" "all_exist $EVALS/__init__.py $EVALS/scoring.py $EVALS/bars.py $EVALS/gold.py && has 'PROACTIVE_EVAL_GOLD_DIR' $EVALS/gold.py && has 'orchestrator/evals/proactive/.*jsonl' .gitignore && settings_declared PROACTIVE_EVAL_GOLD_DIR PROACTIVE_EVAL_WORKSPACE_ID && new_files $TESTS/test_prd268w3_eval_scoring.py $TESTS/test_prd268w3_no_gold_sets.py"
  check "US-C13b the triage, silence and chips runners and the command line; the test is new" "all_exist $EVALS/triage.py $EVALS/silence.py $EVALS/reply_chips.py $EVALS/__main__.py && has 'one_brief' $EVALS/__main__.py && new_files $TESTS/test_prd268w3_eval_runners.py"
  check "US-C13c the spoken, precision and one-brief runners; the test is new" "all_exist $EVALS/spoken.py $EVALS/precision.py $EVALS/one_brief.py && has 'speechify' $EVALS/spoken.py && new_files $TESTS/test_prd268w3_eval_runners_live.py"
  check "US-C22a Calibration@1 from titles and kinds: sources, model, build, strings, the classified purpose; the test is new" "all_exist $CAL/sources.py $CAL/model.py $CAL/build.py $CAL/strings.py && has 'pt-BR' $CAL/strings.py && set_has $O/services/credits/classification.py CUSTOMER_PURPOSES calibration && lacks 'read_document|get_deliverable|search_documents' $CAL/sources.py && new_files $TESTS/test_prd268w3_calibration_build.py"
  check "US-C22b the calibration routes (off by default), the claim, the memory writes, the part, discovery; the tests are new" "all_exist $CAL_ROUTER $CAL/claim.py $CAL/memory.py && router_spec api.auto_calibration && routes_in_manifest 'GET /api/auto/calibration' 'POST /api/auto/calibration' 'POST /api/auto/calibration/\\{[a-z_]+\\}/lines/\\{[a-z_]+\\}/reject' 'POST /api/auto/calibration/\\{[a-z_]+\\}/accept' && route_gated $CAL_ROUTER 'accept' && has 'store_long_term' $CAL/memory.py && has 'violates_exclusions' $CAL/memory.py && set_has $O/services/chat_parts.py STRUCTURED_PART_TYPES calibration@1 && capability calibration && configured_capability calibration && settings_declared CALIBRATION_ENABLED && changed $TESTS/test_prd266_discovery.py && new_files $TESTS/test_prd268w3_calibration_routes.py $TESTS/test_prd268w3_calibration_memory.py"
  check "US-W03a chips on the web: the turn parts reader, the newest-message store, the thread card, the composer's send; the tests are new" "all_exist $FE/lib/chat/turn-parts.ts $PROC/newest-message.ts && has 'replyChips' $FE/types/chat.ts && has 'reply_chips@1' $FE/lib/chat/turn-parts.ts && has 'ReplyChips' $PROC/thread-cards.tsx && has 'automatos:reply-chip' $PROC/composer-with-suggestion.tsx && has 'turn-parts' $FE/lib/chat/use-chat-with-receipts.ts && new_files $FE/lib/chat/__tests__/turn-parts.test.ts $PROC/__tests__/thread-reply-chips.test.tsx"
  check "US-W03b the follow-up line and its hook; the test is new" "all_exist $FE/hooks/use-follow-ups.ts $PROC/follow-up-line.tsx && has 'FollowUpLine|follow-up-line' $PROC/thread-cards.tsx && has '/api/proactive/follow-ups' $FE/hooks/use-follow-ups.ts && new_files $PROC/__tests__/follow-up-line.test.tsx"
  check "US-W06 calibration on the web: the hooks live, the consent line, the card; the tests are new" "all_exist $PROC/calibration-offer.tsx && has '/api/auto/calibration' $FE/hooks/use-calibration.ts && has 'calibration@1' $PROC/thread-cards.tsx && has 'CalibrationOffer|calibration-offer' $PROC/composer-with-suggestion.tsx && new_files $PROC/__tests__/calibration-offer.test.tsx $PROC/__tests__/thread-calibration.test.tsx $FE/hooks/__tests__/use-calibration-live.test.tsx"
  check "US-W03c the browser lane's W3 spec is new" "new_files e2e/browser/specs/proactive-w3.spec.ts"
  check "every acceptance criterion in $KIT is marked (DONE, or OWNER)" "all_acs_done"
fi

echo ""; echo "=== Scope and conventions"
check "N2 exactly one Alembic head in the tree" "one_head"
check "N3 no env reads added outside config.py" "! env_read_added"
check "N4 no print() added in orchestrator application code (scripts and __main__.py may print)" "! print_added"
check "N5 no raw fetch('/api…') added in the frontend" "! raw_fetch_added"
check "N6 no new Python dependency" "untouched orchestrator/requirements.in orchestrator/requirements.lock orchestrator/requirements-dev.txt orchestrator/requirements-dev.lock"
check "N7 no new npm dependency (frontend and the browser lane)" "untouched frontend/package.json frontend/package-lock.json e2e/browser/package.json e2e/browser/package-lock.json"
check "N8 the Composio deny list and the generated Auto skill seed are untouched" "untouched orchestrator/core/composio/deny_list.py orchestrator/core/seeds/platform-management-skill.md"
check "N9 no node_modules in the diff" "! git diff --name-only $BASE $HEAD_SHA | grep -q node_modules"
check "N10 every commit is DCO-signed" "every_commit_signed"
check "N11 a new file is at most 800 lines" "file_sizes_ok"
check "N12a the K3 files over 800 lines did not grow" "named_files_not_grown"
check "N12b no other file over 800 lines grew (config.py and the two reports excepted)" "no_big_file_grew"
warn "N12c config.py grew by settings only (≤ 45 lines)" "config_growth"
check "N12d services/scheduled_task_service.py stays at most 800 lines" "at_most $STS 800"
for fn in _agent_tick _orchestrator_tick _orchestrator_tick_llm _dispatch_heartbeat_notification _route_heartbeat_to_auto; do
  check "N13 $fn is byte-identical to the base (W1's K4)" "fn_unchanged $HB $fn"
done
check "N14a PlatformActionExecutor.clear is byte-identical to the base" "fn_unchanged $EXECUTOR PlatformActionExecutor.clear"
check "N14b PlatformActionExecutor._run_cleared is byte-identical to the base" "fn_unchanged $EXECUTOR PlatformActionExecutor._run_cleared"
check "N14c ToolResultFormatter.format_for_frontend is byte-identical to the base (W2's K4)" "fn_unchanged $O/modules/tools/formatting/result_formatter.py ToolResultFormatter.format_for_frontend"
check "N14d bulk_update_preferences is byte-identical to the base" "fn_unchanged $O/api/notifications.py bulk_update_preferences"
check "N14e _board_task_blocked_pending_approval is byte-identical to the base" "fn_unchanged $O/api/board_tasks.py _board_task_blocked_pending_approval"
check "N15a ToolLoopExecutor.run is byte-identical (the terminal turn is a decorator, K4)" "fn_unchanged $LOOP ToolLoopExecutor.run"
check "N15b ToolLoopExecutor._execute_round is byte-identical to the base" "fn_unchanged $LOOP ToolLoopExecutor._execute_round"
check "N15c ScheduledTaskService.create_task is byte-identical (a follow-up never goes through it, K4)" "fn_unchanged $STS ScheduledTaskService.create_task"
check "N15d ScheduledTaskService.execute_task is byte-identical (the firing is a branch in _deliver)" "fn_unchanged $STS ScheduledTaskService.execute_task"
check "N15e ScheduledTaskService._trigger_agent_chat is byte-identical to the base" "fn_unchanged $STS ScheduledTaskService._trigger_agent_chat"
check "N16 consumers/chatbot/service.py, message.tsx, chat.tsx and multimodal-input.tsx are untouched (K4)" "untouched $UNTOUCHED"
check "N17 the hierarchy gate passes (stdlib ast)" "hierarchy_gate_ok"
check "N18 the new routers' handlers are plain def (sync session)" "routers_plain_def"
check "N19 no third-party HTTP client added" "! client_added"
check "N20 no existing migration modified, deleted or renamed (W1's and W2's included)" "old_migrations_untouched"
check "N21 no W4 surface added (K26)" "! later_wave_added"
check "N22 no Always allow anywhere (W2's K10)" "! always_allow_added"
check "N23 CI's changed-lines code shape passes on the committed wave (functions ≤ 50, nesting ≤ 4, new files ≤ 800)" "code_shape_ok"
check "N24 W1's and W2's kit files are untouched on this branch" "untouched $EARLIER_KIT_FILES"
warn "N25 story commits carry [skip ci] (all but the final CI commit and a merge of main)" "[ -z \"\$(git log --no-merges --format='%s' $BASE..$HEAD_SHA^ 2>/dev/null | grep -v 'skip ci')\" ]"
check "N26 calibration ships off: CALIBRATION_ENABLED defaults to false (K15)" "switch_off CALIBRATION_ENABLED"
check "N27 W2's suggestion push is still off, and services/push is untouched (K26)" "switch_off SUGGESTION_PUSH_ENABLED && untouched $O/services/push"
check "N28 no SDK change: the surface, the d.ts, the shared modules and both versions are the base's (K17)" "untouched $SDK_PATHS && versions_kept"
check "N29 no gold set is tracked under $EVALS (K14)" "no_gold_tracked"
check "N30 no innerHTML / dangerouslySetInnerHTML added in the frontend" "! inner_html_added"
check "N31 no orange-* class, no hex colour and no rounded-3xl in the new web surfaces (K19)" "! orange_added && ! hex_added && ! rounded3_added"
check "N32 nothing auto-plays" "! autoplay_added"
check "N33 the K20 pin files keep their test counts" "pin_tests_kept"
check "N34 no test weakened: added test functions >= removed" "tests_not_weakened"
check "N35 the follow-ups, calibration and the evals never reach AgentFactory (W1's K8)" "! factory_in_lane"
warn "N36 no percentage written into calibration code or its card (§12-Q26; a docstring that quotes one trips this too)" "! added_code $CAL $PROC/calibration-offer.tsx | grep -E \"[0-9] ?%['\\\" ]\" | grep -q ."
check "N37 docker-compose.yml is untouched by this wave" "untouched docker-compose.yml"

if [ "$MODE" = "full" ]; then
  echo ""; echo "=== CI and the PR (once per wave)"
  check "CI: the required jobs present and green on the tested commit" "ci_green_on_head"
  check "the browser lane (browser-e2e.yml) is green on the tested commit" "browser_lane_green"
  check "the PR targets W2's branch while W2's PR is open, main once W2 merged (the stacked-PR trap, K1)" "pr_base_ok"
  check "the PR body records #1360, #1359, the stack, the draft rule, the migration, calibration off, gold sets, the squash rule and the AI disclosure" "pr_body_records"
  warn "the PR is still a draft (K1: only the owner undrafts it, after REVIEW_PASS)" "pr_is_draft"
fi
echo ""
if [ "$FAIL" -eq 0 ]; then echo "✅ PRD-268 W3 Follow-ups acceptance ($MODE): PASS"; else echo "❌ PRD-268 W3 Follow-ups acceptance ($MODE): FAIL"; fi
exit "$FAIL"
