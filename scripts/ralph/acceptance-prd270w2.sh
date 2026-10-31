#!/bin/bash
# Acceptance gate — PRD-270 W2 Cloud bundles (20 stories: US-270-11a … US-270-14f).
#   bash scripts/ralph/acceptance-prd270w2.sh | --print-base | --no-ci | --negative-only
#   GATE_BASE=<ref> and GATE_HEAD=<ref> (smoke runs only): the base, and the commit read as HEAD.
# Base (K1): the wave is STACKED on W1 (feat/prd-270-w1-open-marketplace). While HEAD sits on W1's work, the base is
# the W1 commit the branch grew from (merge-base with origin/feat/prd-270-w1-open-marketplace), so W1's own diff is
# never read as this wave's; once W1 has landed on main and main was merged in (or W1's branch is gone), the base is
# the merge-base with origin/main.
# The gate reads COMMITTED work: every file read, diff and count is taken from the HEAD commit (or GATE_HEAD), never
# the working tree, so a smoke can read origin/main from a checkout that is behind it. The two checks that can only
# read a working tree (CI's changed-lines code shape and the hierarchy gate) run when HEAD is the commit read, and
# are reported skipped in a GATE_HEAD smoke of another commit.
# Nothing runs on the owner's machine: greps, git and stdlib python only; CI once per wave on the draft PR. No
# pipefail (a negative `! producer | grep -q` would invert on SIGPIPE); every check runs in its own `( eval … )`;
# stdin is closed so a grep with a missing file argument fails instead of hanging. Counts use awk, never `grep -vq`
# (the ugrep wrapper trap, 26 Oct).
# Derived from acceptance-prd270w1.sh (30 Oct 2026): the generic helpers are W1's; the stacked base is 268w3's
# pattern one PRD over; the story and scope checks are PRD-270 W2's.
#   - every "added lines" check reads the diff BASE → HEAD, EMPTY when HEAD is the base, so it passes there by
#     construction; each is scoped to code dirs and excludes tests, the kit (scripts/ralph), e2e and docs, so no
#     check ever reads its own pattern lines;
#   - the untouched files and the pin counts hold on the base by construction;
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

STACK_REF=origin/feat/prd-270-w1-open-marketplace
W1_BRANCH=feat/prd-270-w1-open-marketplace
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
    MB_W1=$(git merge-base "$HEAD_SHA" "$STACK_REF" 2>/dev/null || true)
    # Stacked: W1's work sits between main and this wave (main's merge-base is behind W1's).
    if [ -n "$MB_W1" ] && [ "$MB_W1" != "$MB_MAIN" ] && git merge-base --is-ancestor "$MB_MAIN" "$MB_W1"; then BASE=$MB_W1; fi
  fi
fi
MODE=full
case "${1:-}" in --print-base) echo "$BASE"; exit 0 ;; --no-ci) MODE=no-ci ;; --negative-only) MODE=negative ;; "") ;; *) echo "unknown argument: $1"; exit 2 ;; esac
echo "base: $(git log -1 --format='%h %s' "$BASE") · HEAD read: $(git log -1 --format='%h %s' "$HEAD_SHA") · mode: $MODE · smoke: $SMOKE · python: $("$PY" -V 2>&1)"
FAIL=0
check() { local name="$1" body="$2"; echo ""; echo "── $name"; if ( eval "$body" ); then echo "   ✅ PASS: $name"; else echo "   ❌ FAIL: $name"; FAIL=1; fi; }
warn() { local name="$1" body="$2"; echo ""; echo "── $name"; if ( eval "$body" ); then echo "   ✅ ok: $name"; else echo "   ⚠️  WARN (not a failure): $name"; fi; }

KIT=scripts/ralph/prd-270w2.json
PRD_DOC=docs/PRDS/PRD-270-SYNC-WITH-AUTOMATOS-CLOUD.md
O=orchestrator
TESTS=$O/tests
VERSIONS=$O/alembic/versions
MANIFEST=$O/reports/route-manifest.json
SURFACE=$O/reports/config-surface.json
MB=$O/services/marketplace_bundles
MC=$O/services/marketplace_cloud
FIX=$TESTS/fixtures/marketplace_bundles
CATI_ROUTER=$O/api/marketplace_catalog_items.py
ITEMS_ROUTER=$O/api/marketplace_cloud_items.py
W1_CAT_ROUTER=$O/api/marketplace_catalog.py
W1_SYNC_ROUTER=$O/api/marketplace_cloud_sync.py
FE=frontend
MKT=$FE/components/marketplace
CLOUD_FE=$MKT/cloud
NEW_REV=prd270_w2_cloud_bundles
W1_REV=prd270_w1_cloud_sync
SYNC_DOC=docs/community-marketplace/sync-with-automatos-cloud.md
# K3: files over 800 lines this wave may touch or sits beside; none may grow.
NO_GROW="$FE/lib/api-client.ts $O/main.py $O/api/marketplace.py $O/api/workflow_recipes.py $O/api/admin_plugins.py $O/core/models/core.py docs/getting-started/self-hosting.md $MKT/marketplace-plugins-tab.tsx"
# Over 800 lines and allowed to grow: settings live only in config.py; the two reports are generated lists.
GROW_OK="$O/config.py $MANIFEST $SURFACE"
# K29: the pinned tests that move with approved changes; each keeps at least its test count.
PIN_TESTS="$TESTS/test_prd209_alembic_single_head.py $TESTS/test_prd236_w1_routes.py"
# K4: the install machinery this wave reuses as it is, never edits.
# K4 amended 31 Oct 2026 (the night manager, PLAY): cascade_installer.py may carry the two bug fixes W2's first CI
# found (646d922f6: the composio_connections insert sets created_at; 838abe7ce: entity creation flush-only inside a
# cascade). The review reads both; every other file here stays untouched.
INSTALL_MACHINERY="$O/modules/tools/discovery/handlers_marketplace.py $O/modules/tools/discovery/handlers_packages.py $O/services/package_installer.py $O/services/marketplace_packages.py $O/core/services/marketplace_s3.py $O/core/services/plugin_upload_service.py $O/core/services/plugin_security_scanner.py $O/core/services/skill_materializer.py $O/core/models/core.py $O/core/models/marketplace_packages.py $O/core/models/marketplace_plugins.py $O/core/models/personas.py $O/core/seeds"
MEMBER_ROUTERS="$O/api/marketplace.py $O/api/workflow_recipes.py $O/api/admin_plugins.py $O/api/marketplace_plugins.py $O/api/workspace_plugins.py $O/api/llm_marketplace.py $O/api/composio.py $O/api/workspace_skills.py $O/api/approval_grants.py $O/main.py"
# K4, K17: the chat install path is gated at confirmation_gate.py only; the executor, the grants and the actions stay.
CHAT_PATH_UNTOUCHED="$O/modules/tools/discovery/platform_executor.py $O/modules/tools/execution/tool_grants.py $O/modules/tools/discovery/actions_packages.py $O/modules/tools/discovery/actions_marketplace.py"
CONF_GATE=$O/modules/tools/discovery/confirmation_gate.py
CHAT_ASKS=$O/modules/tools/discovery/cloud_install_asks.py
# This wave has no theme work; W1's guarded GET is called, never edited (K4, K13).
THEME_AND_FETCH="$O/services/themes $O/services/theme_revocation_scheduler.py $O/api/themes_revocations_feed.py $O/api/admin_theme_revocation_sync.py $MC/fetch.py"
UNTOUCHED_WEB="$FE/components/settings/SystemSettingsTab.tsx $MKT/marketplace-playbooks-tab.tsx $MKT/marketplace-tools-tab.tsx $MKT/marketplace-llms-tab.tsx $MKT/marketplace-skills-tab.tsx $MKT/marketplace-agents-tab.tsx $MKT/package-detail-modal.tsx $MKT/marketplace-plugin-detail-modal.tsx $FE/components/auth/sign-in-form.tsx $FE/components/auth/sign-up-form.tsx $FE/components/auth/sign-up-parts.tsx $FE/app/sign-in $FE/app/sign-up $FE/app/sso-callback $FE/middleware.ts"
# This wave adds no SDK change: the theme SDK's surface and both versions are the base's.
SDK_PATHS="$FE/packages/theme-sdk $FE/lib/theme-sdk/surface $FE/lib/theme-host/shared-modules.json"

# --- reading the commit -----------------------------------------------------------------------------
src() { git show "$HEAD_SHA:$1" 2>/dev/null; }
exists() { git cat-file -e "$HEAD_SHA:$1" 2>/dev/null; }
lines_at() { git show "$1:$2" 2>/dev/null | wc -l | tr -d ' '; }
has() { exists "$2" && src "$2" | grep -qE -e "$1"; }      # has <ERE> <file>: false when the file is missing
lacks() { ! has "$1" "$2"; }
# grep_tree <ERE> <paths…>: a line of the committed tree under the paths matches.
grep_tree() { local pattern="$1"; shift; git grep -qE -e "$pattern" "$HEAD_SHA" -- "$@" 2>/dev/null; }
all_exist() { local f; for f in "$@"; do exists "$f" || { echo "   missing: $f"; return 1; }; done; }
json_ok() { local f; for f in "$@"; do src "$f" | "$PY" -m json.tool >/dev/null 2>&1 || { echo "   not valid JSON: $f"; return 1; }; done; }
# The wave's ADDED lines under the given paths: tests, the kit, e2e and docs excluded (negative greps only).
added_code() { git diff "$BASE" "$HEAD_SHA" -- "$@" ':!orchestrator/tests' ':!*__tests__*' ':!*.test.ts' ':!*.test.tsx' ':!e2e' ':!scripts/ralph' ':!docs' | grep '^+' | grep -v '^+++'; }
added_files() { git diff --name-only --diff-filter=A "$BASE" "$HEAD_SHA" -- "$@"; }
# Branch-scoped: each file exists AND was added by this wave (a test that predates it proves nothing about it).
new_files() { local f bad=0; for f in "$@"; do
    if ! exists "$f"; then echo "   missing: $f"; bad=1
    elif [ -z "$(added_files "$f")" ]; then echo "   not new on this branch: $f"; bad=1; fi
  done; [ $bad -eq 0 ]; }
changed() { ! git diff --quiet "$BASE" "$HEAD_SHA" -- "$@"; }
untouched() { local out; out=$(git diff --name-only "$BASE" "$HEAD_SHA" -- "$@"); [ -z "$out" ] || { echo "$out" | sed 's/^/   changed: /'; return 1; }; }
# added_in <ERE> <file>: a line this wave added to <file> matches (a positive smoke signal; the tests are the proof).
added_in() { git diff "$BASE" "$HEAD_SHA" -- "$2" | grep '^+' | grep -v '^+++' | grep -qE -e "$1"; }

env_read_added()   { added_code orchestrator ':!orchestrator/config.py' | grep -E 'os\.(getenv|environ)' | grep -q .; }
print_added()      { added_code orchestrator ':!orchestrator/scripts' ':(exclude,glob)orchestrator/**/__main__.py' | grep -E '^\+\s*print\(' | grep -q .; }
raw_fetch_added()  { added_code frontend | grep -E "fetch\(\s*['\"\`]/api" | grep -q .; }
inner_html_added() { added_code frontend | grep -E 'dangerouslySetInnerHTML|\.innerHTML\s*=|\.outerHTML\s*=|insertAdjacentHTML|document\.write\(' | grep -q .; }
orange_added()     { added_code frontend | grep -E '\b(bg|text|border|ring|from|to|via|fill|stroke)-orange-[0-9]' | grep -q .; }
NEW_WEB="$CLOUD_FE $FE/lib/marketplace $FE/hooks/use-cloud-items.ts $MKT/plugin-types.ts"
hex_added()        { added_code $NEW_WEB | grep -E "#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b['\" ;)]" | grep -q .; }
rounded3_added()   { added_code $NEW_WEB | grep -E '\brounded-3xl\b' | grep -q .; }
autoplay_added()   { added_code frontend | grep -iE '\bautoplay\b|autoPlay\s*[=:]' | grep -q .; }
# K13: W1's fetch.py is the one httpx user (untouched); no other HTTP client anywhere.
other_client_added() { added_code orchestrator | grep -E '^\+\s*(import|from)\s+(requests|aiohttp|urllib3|httpx)\b' | grep -q .; }
# K26: no model call in this wave; K15: never the plugin LLM scan (its call and import shapes).
model_call_added() { added_code orchestrator | grep -E 'LLMManager|get_llm_manager|orchestrator_llm|chat_completion|\.complete\(|anthropic\.' | grep -q .; }
llm_scan_used() { git grep -nE 'PluginScanService\(|await llm_security_scan\(|import .*\bllm_security_scan\b|import .*\bPluginScanService\b' "$HEAD_SHA" -- "$MB" 2>/dev/null | grep -q .; }
# K9 / §8-Q3a: no private key and no signing in core.
signing_added() { added_code orchestrator | grep -E 'Ed25519PrivateKey|generate_private_key|private_bytes\(|\.sign\(' | grep -q .; }
# K9: nothing in the bundle package claims Official.
official_claimed() { git grep -nE '"official": *True|official *= *True' "$HEAD_SHA" -- "$MB" 2>/dev/null | grep -q .; }
# K28 / §8-Q2: no scheduled Sync.
scheduled_sync_added() { added_code orchestrator | grep -E 'add_job\(|IntervalTrigger|CronTrigger' | grep -q .; }
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
config_growth() { local b h; b=$(lines_at "$BASE" $O/config.py); h=$(lines_at "$HEAD_SHA" $O/config.py); echo "   config.py: base $b, HEAD $h (settings only: 3 settings with their comments)"; [ "$h" -le $((b + 30)) ]; }
test_fn_count() { git show "$1:$2" 2>/dev/null | awk '/^(async )?def test_/{n++} END{print n+0}'; }
tests_not_weakened() { local d a; d=$(git diff "$BASE" "$HEAD_SHA" -- $TESTS frontend e2e | awk '/^-[ \t]*(async )?def test_|^-[ \t]*(it|test)(\.each\([^)]*\))?\(/{n++} END{print n+0}'); a=$(git diff "$BASE" "$HEAD_SHA" -- $TESTS frontend e2e | awk '/^\+[ \t]*(async )?def test_|^\+[ \t]*(it|test)(\.each\([^)]*\))?\(/{n++} END{print n+0}'); echo "   test functions removed $d, added $a"; [ "$a" -ge "$d" ]; }
pin_tests_kept() { local f b h bad=0
  for f in $PIN_TESTS; do exists "$f" || { echo "   $f is missing"; bad=1; continue; }
    b=$(test_fn_count "$BASE" "$f"); h=$(test_fn_count "$HEAD_SHA" "$f"); [ "$h" -ge "$b" ] || { echo "   $f: base $b, HEAD $h"; bad=1; }; done
  [ $bad -eq 0 ] && echo "   every K29 pin file keeps its test count"; [ $bad -eq 0 ]; }
old_migrations_untouched() { [ -z "$(git diff --name-only --diff-filter=MDR "$BASE" "$HEAD_SHA" -- "$VERSIONS")" ]; }
# Only this kit's own files change under scripts/ralph (the runner seeds the four; the marks commit the json).
other_kits_untouched() { local out; out=$(git diff --name-only "$BASE" "$HEAD_SHA" -- scripts/ralph | awk '!/scripts\/ralph\/(prd-270w2\.json|PROMPT_build_prd270w2\.md|PROMPT_review_prd270w2\.md|acceptance-prd270w2\.sh)$/'); [ -z "$out" ] || { echo "$out" | sed 's/^/   another kit file changed: /'; return 1; }; }
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
  src "$f" | grep -qiE "CREATE TABLE IF NOT EXISTS marketplace_cloud_items" || { echo "   $f: no CREATE TABLE IF NOT EXISTS marketplace_cloud_items"; return 1; }
  local name; for name in ck_marketplace_cloud_items_type ck_marketplace_cloud_items_one_row uq_marketplace_cloud_items_external uq_marketplace_cloud_items_agent uq_marketplace_cloud_items_recipe uq_marketplace_cloud_items_package uq_marketplace_cloud_items_plugin num_nonnulls; do
    has "$name" "$f" || { echo "   $f: no $name (K6)"; return 1; }
  done
  # SQL statement shapes only, so a docstring that says what the revision does not do never trips it.
  ! src "$f" | grep -qE "ALTER TABLE [a-z_]+ (ADD|DROP|ALTER|RENAME)|DELETE FROM [a-z_]+|UPDATE [a-z_]+ SET|TRUNCATE( TABLE)? [a-z_]+" || { echo "   $f alters a table or changes rows (K6: one new table, nothing else)"; return 1; }
  has "EXPECTED_HEAD = \"$NEW_REV\"" $TESTS/test_prd209_alembic_single_head.py || { echo "   test_prd209 EXPECTED_HEAD is not $NEW_REV"; return 1; }
  has "$NEW_REV\\.py" $TESTS/test_prd236_w1_routes.py || { echo "   test_prd236_w1_routes.py does not read $NEW_REV.py"; return 1; }
  has "EXPECTED_HEAD = \"$NEW_REV\"" $TESTS/test_prd236_w1_routes.py || { echo "   test_prd236_w1_routes.py does not assert EXPECTED_HEAD = $NEW_REV"; return 1; }
  echo "   $NEW_REV chains onto $bh; marketplace_cloud_items with its CHECKs and keys; no table altered, no row changed; both pins moved"
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

# --- routes, settings ---------------------------------------------------------------------------
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
# K22: every route this wave adds to the manifest is a GET (no new mutating route).
added_routes_are_get() { "$PY" - "$BASE" "$HEAD_SHA" "$MANIFEST" <<'PY'
import json, subprocess, sys
base, head, path = sys.argv[1:4]
def pairs(ref):
    out = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, text=True)
    return set() if out.returncode else {(r["method"], r["path"]) for r in json.loads(out.stdout)["routes"]}
added = sorted(pairs(head) - pairs(base))
bad = [f"{m} {p}" for m, p in added if m != "GET"]
print(f"   routes added: {len(added)}" + "".join(f"\n   not a GET: {b}" for b in bad)); sys.exit(1 if bad else 0)
PY
}
settings_declared() { local s bad=0; for s in "$@"; do
    has "^\s+$s\b" $O/config.py || { echo "   config.py lacks $s"; bad=1; }
    has "\"$s\"" "$SURFACE" || { echo "   config-surface.json lacks $s"; bad=1; }
  done; [ $bad -eq 0 ]; }
router_spec() { has "RouterSpec\(\"$1\"" $O/router_manifest.py; }
# The new routers' handlers are plain def (sync session; AGENTS.md). A router not built yet is skipped.
routers_plain_def() { "$PY" - "$HEAD_SHA" "$CATI_ROUTER" "$ITEMS_ROUTER" <<'PY'
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
# K12: the hosted router carries no hybrid dependency (not built yet passes: its story check fails instead).
hosted_router_unauthenticated() { ! has 'get_request_context_hybrid' "$CATI_ROUTER"; }
# K19: the gate module imports nothing from api-client (request() passes itself: no import cycle).
gate_has_no_cycle() { ! has "from ['\"](\\.\\./api-client|@/lib/api-client)['\"]" $FE/lib/marketplace/cloud-install-gate.ts; }
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
# The theme SDK's two versions are the base's (contract.py).
versions_kept() { local f=$O/services/themes/contract.py b h
  b=$(git show "$BASE:$f" 2>/dev/null | grep -E '^(SDK_VERSION|THEME_HOST_VERSION) *=')
  h=$(src "$f" | grep -E '^(SDK_VERSION|THEME_HOST_VERSION) *=')
  echo "   base: $(echo $b) · HEAD: $(echo $h)"; [ -n "$b" ] && [ "$b" = "$h" ]; }

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
# The browser lane (US-270-14f) runs in browser-e2e.yml on the PR. A WebKit 'internal error' on reload is a known
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
  for s in "#1377" "Stacked on" "DRAFT" "REVIEW_PASS" "prd270_w2_cloud_bundles" "marketplace_cloud_items" "Reviewed on Automatos Cloud" "--subject" "AI assistance"; do
    printf '%s' "$body" | grep -qF -- "$s" || { echo "   the PR body lacks: $s"; miss=1; }
  done
  [ $miss -eq 0 ] && echo "   PR #$n records #1377, the stack on W1, the draft rule, the migration and its table, the trust label, the squash rule and the AI disclosure"
}
# The stacked-PR trap (K1): while W1's PR is open this PR targets W1's branch; once W1 merged, main.
pr_base_ok() {
  local n base w1
  n=$(open_pr); [ -n "$n" ] || { echo "   no open PR"; return 1; }
  base=$(gh pr view "$n" --json baseRefName --jq .baseRefName 2>/dev/null)
  w1=$(gh pr list --head "$W1_BRANCH" --state all --json state --jq '.[0].state' 2>/dev/null)
  echo "   PR #$n base: $base; W1's PR: ${w1:-none}"
  case "$w1" in
    MERGED) [ "$base" = "main" ] ;;
    *) [ "$base" = "$W1_BRANCH" ] || [ "$base" = "main" ] ;;
  esac
}
# K1: the PR stays a draft until REVIEW_PASS; the owner undrafts it.
pr_is_draft() { local n d; n=$(open_pr); [ -n "$n" ] || { echo "   no open PR"; return 1; }; d=$(gh pr view "$n" --json isDraft --jq .isDraft 2>/dev/null); echo "   PR #$n isDraft: $d"; [ "$d" = "true" ]; }

if [ "$MODE" != "negative" ]; then
  echo ""; echo "=== Seeded and stacked"
  check "the PRD is tracked at $PRD_DOC" "exists $PRD_DOC"
  check "the kit is committed and valid JSON" "exists $KIT && json_ok $KIT"
  check "W1's work is under this wave (its migration, model, pull, feed, fetch guard, catalogue and Sync routes)" "all_exist $VERSIONS/$W1_REV.py $O/core/models/marketplace_cloud.py $MC/pull.py $MC/feed.py $MC/fetch.py $W1_CAT_ROUTER $W1_SYNC_ROUTER"

  echo ""; echo "=== Stories (priority order; positive: red on the base by design)"
  check "US-270-11a one revision $NEW_REV on the base head: marketplace_cloud_items, both pins" "one_revision && revision_ok"
  check "US-270-11a the model, the three settings; the tests are new" "has 'class MarketplaceCloudItem' $O/core/models/marketplace_cloud.py && has 'CLOUD_ITEM_TYPES' $O/core/models/marketplace_cloud.py && has 'MarketplaceCloudItem' $O/core/models/__init__.py && settings_declared MARKETPLACE_CATALOG_BUNDLE_MAX_KB AUTOMATOS_CLOUD_BUNDLE_MAX_KB AUTOMATOS_CLOUD_PLUGINS_PER_SYNC && new_files $TESTS/test_prd270w2_migration.py $TESTS/test_prd270w2_settings.py"
  check "US-270-11b the bundle format (envelope, agent, Playbook): strict validator, refs, canonical bytes; the corpus; the test is new" "all_exist $MB/__init__.py $MB/schema.py $MB/validate.py $MB/refs.py $MB/canonical.py && lacks '^(from|import) ' $MB/__init__.py && has 'def validate_bundle' $MB/validate.py && has 'BundleError' $MB/validate.py && has 'def canonical_bytes' $MB/canonical.py && has 'sort_keys=True' $MB/canonical.py && has 'def revision_of' $MB/canonical.py && new_files $FIX/agent.json $FIX/playbook.json $FIX/cases.json && json_ok $FIX/agent.json $FIX/playbook.json $FIX/cases.json && new_files $TESTS/test_prd270w2_bundle_format.py"
  check "US-270-11c the package and plugin bundles, local_version; the corpus completed; the test is new" "has 'def local_version' $MB/canonical.py && grep_tree 'plugin' $MB/schema.py && grep_tree 'package' $MB/schema.py && new_files $FIX/package.json $FIX/plugin.json && json_ok $FIX/package.json $FIX/plugin.json && new_files $TESTS/test_prd270w2_bundle_format_more.py"
  check "US-270-12a SaaS export of agents and Playbooks (lineage, seeded rows out); the test is new" "all_exist $MB/seeded.py $MB/lineage.py $MB/export_agents.py $MB/export_playbooks.py && has 'def exported_agents' $MB/export_agents.py && has 'def exported_playbooks' $MB/export_playbooks.py && has 'automatos-cloud:' $MB/lineage.py && new_files $TESTS/test_prd270w2_export.py"
  check "US-270-12b SaaS export of packages and plugins (approved, verdict safe); the test is new" "all_exist $MB/export_packages.py $MB/export_plugins.py && has 'def exported_packages' $MB/export_packages.py && has 'def exported_plugins' $MB/export_plugins.py && has 'overall_verdict' $MB/export_plugins.py && grep_tree 'seed_packages' $MB && new_files $TESTS/test_prd270w2_export_more.py"
  check "US-270-12c item entries in W1's catalogue and the bundle route (SaaS only); the test is new" "all_exist $MB/catalog_items.py $CATI_ROUTER && has 'def with_items' $MB/catalog_items.py && { added_in 'with_items' $W1_CAT_ROUTER || added_in 'with_items' $MC/catalog.py; } && router_spec api.marketplace_catalog_items && routes_in_manifest 'GET /api/marketplace/catalog/bundles/\\{revision\\}' && has 'IS_LOCAL_EDITION' $CATI_ROUTER && new_files $TESTS/test_prd270w2_catalog_items.py"
  check "US-270-12d the plugin package route (a deterministic zip of an exported plugin); the test is new" "all_exist $MB/plugin_files.py && has 'def plugin_zip' $MB/plugin_files.py && has '1980' $MB/plugin_files.py && routes_in_manifest 'GET /api/marketplace/catalog/plugins/\\{slug\\}/\\{version\\}/package\\.zip' && new_files $TESTS/test_prd270w2_plugin_package.py"
  check "US-270-13a local: item entries dispatched by type, bundles fetched from the derived URL; the test is new" "all_exist $MB/items_feed.py && has 'def parse_item_entry' $MB/items_feed.py && has 'def fetch_bundle' $MB/items_feed.py && has 'def changed_entries' $MB/items_feed.py && added_in 'BUNDLE_TYPES|marketplace_bundles' $MC/feed.py && new_files $TESTS/test_prd270w2_items_feed.py"
  check "US-270-15a local plugins: checked like an upload, static scan, stored, approved on arrival, materialised; the test is new" "all_exist $MB/plugin_import.py && has 'def import_plugins' $MB/plugin_import.py && has 'static_scan' $MB/plugin_import.py && has 'extract_plugin' $MB/plugin_import.py && has 'materialize_plugin' $MB/plugin_import.py && has 'AUTOMATOS_CLOUD_PLUGINS_PER_SYNC' $MB/plugin_import.py && ! llm_scan_used && new_files $TESTS/test_prd270w2_plugin_import.py"
  check "US-270-13b local agents upserted by their mapping; the test is new" "all_exist $MB/resolve.py $MB/import_agents.py && has 'def upsert_agent' $MB/import_agents.py && new_files $TESTS/test_prd270w2_import_agents.py"
  check "US-270-13c local Playbooks and packages, the item pass in one transaction, the round trip; the tests are new" "all_exist $MB/import_playbooks.py $MB/import_packages.py $MB/import_items.py && has 'def upsert_playbook' $MB/import_playbooks.py && has 'def upsert_package' $MB/import_packages.py && has 'def import_items' $MB/import_items.py && has 'type.: *.manual' $MB/import_playbooks.py && new_files $TESTS/test_prd270w2_import_items.py $TESTS/test_prd270w2_round_trip.py"
  check "US-270-13d the Sync runs every pass (W1's POST); the tests are new" "all_exist $MB/sync_items.py && has 'def run_item_passes' $MB/sync_items.py && added_in 'run_item_passes' $W1_SYNC_ROUTER && new_files $TESTS/test_prd270w2_sync_routes.py $TESTS/test_prd270w2_wire.py"
  check "US-270-14a local reads: Cloud items, disclosure, updates (local only); the test is new" "all_exist $MB/local_reads.py $MB/disclosure.py $MB/updates.py $ITEMS_ROUTER && router_spec api.marketplace_cloud_items && routes_in_manifest 'GET /api/marketplace/cloud-sync/items' 'GET /api/marketplace/cloud-sync/items/\\{kind\\}/\\{key\\}/disclosure' 'GET /api/marketplace/cloud-sync/updates' && has 'IS_LOCAL_EDITION' $ITEMS_ROUTER && new_files $TESTS/test_prd270w2_local_reads.py"
  check "US-270-14g Auto's chat installs of a Cloud item ask first (the existing approval carrying the disclosure); the test is new" "all_exist $MB/chat_disclosure.py $CHAT_ASKS && has 'def cloud_target' $MB/chat_disclosure.py && has 'def disclosure_markdown' $MB/chat_disclosure.py && has 'Reviewed on Automatos Cloud' $MB/chat_disclosure.py && has 'def cloud_install_asks' $CHAT_ASKS && has 'single_use=True' $CHAT_ASKS && has 'question_md' $CHAT_ASKS && added_in 'cloud_install_asks' $CONF_GATE && new_files $TESTS/test_prd270w2_chat_installs.py"
  check "US-270-14b web: Cloud items data and the two badges on every tab's seam; the tests are new" "all_exist $FE/lib/marketplace/cloud-items.ts $FE/hooks/use-cloud-items.ts $CLOUD_FE/cloud-item-badges.tsx $MKT/plugin-types.ts && has 'Reviewed on Automatos Cloud' $CLOUD_FE/cloud-item-badges.tsx && has 'CloudItemBadges' $MKT/marketplace-agent-cards.tsx && has 'CloudItemBadges' $MKT/marketplace-packages-tab.tsx && has 'CloudItemBadges' $MKT/marketplace-plugins-tab.tsx && has 'CloudItemBadges' $MKT/marketplace-item-sections.tsx && not_grown $MKT/marketplace-plugins-tab.tsx && new_files $CLOUD_FE/__tests__/cloud-item-badges.test.tsx $FE/lib/marketplace/__tests__/cloud-items.test.ts"
  check "US-270-14c web: the disclosure dialog and its sections; the test is new" "all_exist $CLOUD_FE/disclosure-dialog.tsx $CLOUD_FE/disclosure-sections.tsx && has 'Add to my workspace' $CLOUD_FE/disclosure-dialog.tsx && new_files $CLOUD_FE/__tests__/disclosure-dialog.test.tsx"
  check "US-270-14d web: the install gate at apiClient.request (api-client.ts not grown), the layer in the Marketplace; the tests are new" "all_exist $FE/lib/marketplace/cloud-install-gate.ts $CLOUD_FE/cloud-items-layer.tsx && has 'export (async )?function cloudInstallGate' $FE/lib/marketplace/cloud-install-gate.ts && has 'cloudInstallGate' $FE/lib/api-client.ts && not_grown $FE/lib/api-client.ts && gate_has_no_cycle && has 'CloudItemsLayer' $MKT/marketplace-homepage.tsx && new_files $FE/lib/marketplace/__tests__/cloud-install-gate.test.ts $FE/lib/__tests__/api-client-cloud-gate.test.ts $CLOUD_FE/__tests__/cloud-items-layer.test.tsx"
  check "US-270-14e web: update notices in the layer; the test is new" "all_exist $CLOUD_FE/update-notices.tsx && has 'CloudUpdateNotices' $CLOUD_FE/cloud-items-layer.tsx && new_files $CLOUD_FE/__tests__/update-notices.test.tsx"
  check "US-270-16 docs: the hub row in place, the Sync page's Cloud items section; the test is new" "added_in 'Reviewed on Automatos Cloud' $SYNC_DOC && has 'AUTOMATOS_CLOUD_PLUGINS_PER_SYNC' $SYNC_DOC && changed docs/getting-started/self-hosting.md && not_grown docs/getting-started/self-hosting.md && new_files $TESTS/test_prd270w2_docs.py"
  check "US-270-14f the browser lane's spec is new and tagged @host; the lane and its dependencies unchanged" "new_files e2e/browser/specs/cloud-items.spec.ts && has '@host' e2e/browser/specs/cloud-items.spec.ts && untouched .github/workflows/browser-e2e.yml e2e/browser/package.json e2e/browser/package-lock.json"
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
check "N12a the K3 files over 800 lines did not grow (api-client.ts, main.py, marketplace.py, workflow_recipes.py, admin_plugins.py, core.py, self-hosting.md, the plugins tab)" "named_files_not_grown"
check "N12b no other file over 800 lines grew (config.py and the two reports excepted)" "no_big_file_grew"
warn "N12c config.py grew by settings only (≤ 30 lines)" "config_growth"
check "N13 the install machinery is untouched (cascade, package installer, plugin and skill services, core models, seeds; K4)" "untouched $INSTALL_MACHINERY"
check "N14a the member Marketplace routers, the approvals route and main.py are untouched (K4)" "untouched $MEMBER_ROUTERS"
check "N14d the chat install path is gated only in confirmation_gate.py: the executor, tool_grants and the install actions untouched (K4, K17)" "untouched $CHAT_PATH_UNTOUCHED"
check "N14b no theme work and W1's guarded GET untouched (services/themes, the revocation modules, fetch.py; K4, K13)" "untouched $THEME_AND_FETCH"
check "N14c the untouchable web files are untouched (the big tabs and modals, SystemSettingsTab, the auth surfaces, the middleware; K4)" "untouched $UNTOUCHED_WEB"
check "N15 the hierarchy gate passes (stdlib ast)" "hierarchy_gate_ok"
check "N16 the new routers' handlers are plain def (sync session)" "routers_plain_def"
check "N17 no new HTTP client: requests, aiohttp, urllib3 and httpx never imported by this wave (W1's fetch.py is the one; K13)" "! other_client_added"
check "N18 no model call added, and the plugin LLM scan never used locally (K15, K26)" "! model_call_added && ! llm_scan_used"
check "N19 no existing migration modified, deleted or renamed" "old_migrations_untouched"
check "N20 no private key and no signing in core (§8-Q3a; K9)" "! signing_added"
check "N21 the hosted router carries no get_request_context_hybrid (K12)" "hosted_router_unauthenticated"
check "N22 no scheduled Sync (§8-Q2)" "! scheduled_sync_added"
check "N23 CI's changed-lines code shape passes on the committed wave (functions ≤ 50, nesting ≤ 4, new files ≤ 800)" "code_shape_ok"
check "N24 no other kit's file changed on this branch" "other_kits_untouched"
warn "N25 story commits carry [skip ci] (all but the final CI commit and a merge of main)" "[ -z \"\$(git log --no-merges --format='%s' $BASE..$HEAD_SHA^ 2>/dev/null | grep -v 'skip ci')\" ]"
check "N26 nothing in the bundle package claims Official (§8-Q3a)" "! official_claimed"
check "N27 no SDK change: the theme SDK's surface and both versions are the base's" "untouched $SDK_PATHS && versions_kept"
check "N28 no innerHTML / dangerouslySetInnerHTML added in the frontend" "! inner_html_added"
check "N29 no orange-* class, no hex colour and no rounded-3xl in the new web surfaces (K30)" "! orange_added && ! hex_added && ! rounded3_added"
check "N30 nothing auto-plays" "! autoplay_added"
check "N31 the K29 pin files keep their test counts" "pin_tests_kept"
check "N32 no test weakened: added test functions >= removed" "tests_not_weakened"
check "N33 every route this wave adds is a GET (no new mutating route; K22)" "added_routes_are_get"
check "N34 docker-compose.yml and the browser lane's workflow are untouched by this wave" "untouched docker-compose.yml .github/workflows/browser-e2e.yml"

if [ "$MODE" = "full" ]; then
  echo ""; echo "=== CI and the PR (once per wave)"
  check "CI: the required jobs present and green on the tested commit" "ci_green_on_head"
  check "the browser lane (browser-e2e.yml) is green on the tested commit" "browser_lane_green"
  check "the PR targets W1's branch while W1's PR is open, main once W1 merged (the stacked-PR trap, K1)" "pr_base_ok"
  check "the PR body records #1377, the stack, the draft rule, the migration, the trust label, the squash rule and the AI disclosure" "pr_body_records"
  warn "the PR is still a draft (K1: only the owner undrafts it, after REVIEW_PASS)" "pr_is_draft"
fi
echo ""
if [ "$FAIL" -eq 0 ]; then echo "✅ PRD-270 W2 Cloud bundles acceptance ($MODE): PASS"; else echo "❌ PRD-270 W2 Cloud bundles acceptance ($MODE): FAIL"; fi
exit "$FAIL"
