#!/bin/bash
# Acceptance gate — PRD-270 W1 The open Marketplace (19 stories: US-270-00 … US-270-05c).
#   bash scripts/ralph/acceptance-prd270w1.sh | --print-base | --no-ci | --negative-only
#   GATE_BASE=<ref> and GATE_HEAD=<ref> (smoke runs only): the base, and the commit read as HEAD.
# Base (K1): the wave is NOT stacked: the base is the merge-base of HEAD with origin/main.
# The gate reads COMMITTED work: every file read, diff and count is taken from the HEAD commit (or GATE_HEAD), never
# the working tree, so a smoke can read origin/main from a checkout that is behind it. The two checks that can only
# read a working tree (CI's changed-lines code shape and the hierarchy gate) run when HEAD is the commit read, and
# are reported skipped in a GATE_HEAD smoke of another commit.
# Nothing runs on the owner's machine: greps, git and stdlib python only; CI once per wave on the draft PR. No
# pipefail (a negative `! producer | grep -q` would invert on SIGPIPE); every check runs in its own `( eval … )`;
# stdin is closed so a grep with a missing file argument fails instead of hanging. Counts use awk, never `grep -vq`
# (the ugrep wrapper trap, 26 Oct).
# Derived from acceptance-prd268w3.sh (30 Oct 2026): the generic helpers are W3's; the base is the non-stacked one
# (268 W1's); the story and scope checks are PRD-270 W1's.
#   - every "added lines" check reads the diff BASE → HEAD, EMPTY when HEAD is the base, so it passes there by
#     construction; each is scoped to code dirs and excludes tests, the kit (scripts/ralph), e2e and docs, so no
#     check ever reads its own pattern lines;
#   - the byte-identical function checks, the untouched files and the pin counts hold on the base by construction;
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

HEAD_SHA=$(git rev-parse --verify -q "${GATE_HEAD:-HEAD}^{commit}") || { echo "GATE_HEAD ${GATE_HEAD:-HEAD} does not resolve"; exit 1; }
# SMOKE: the commit read is not the checkout's HEAD, so the working-tree checks cannot read it.
SMOKE=0
[ "$HEAD_SHA" = "$(git rev-parse HEAD)" ] || SMOKE=1
if [ -n "${GATE_BASE:-}" ]; then
  BASE=$(git rev-parse --verify -q "$GATE_BASE^{commit}") || { echo "GATE_BASE $GATE_BASE does not resolve"; exit 1; }
else
  BASE=$(git merge-base "$HEAD_SHA" origin/main) || { echo "cannot resolve merge-base with origin/main"; exit 1; }
fi
MODE=full
case "${1:-}" in --print-base) echo "$BASE"; exit 0 ;; --no-ci) MODE=no-ci ;; --negative-only) MODE=negative ;; "") ;; *) echo "unknown argument: $1"; exit 2 ;; esac
echo "base: $(git log -1 --format='%h %s' "$BASE") · HEAD read: $(git log -1 --format='%h %s' "$HEAD_SHA") · mode: $MODE · smoke: $SMOKE · python: $("$PY" -V 2>&1)"
FAIL=0
check() { local name="$1" body="$2"; echo ""; echo "── $name"; if ( eval "$body" ); then echo "   ✅ PASS: $name"; else echo "   ❌ FAIL: $name"; FAIL=1; fi; }
warn() { local name="$1" body="$2"; echo ""; echo "── $name"; if ( eval "$body" ); then echo "   ✅ ok: $name"; else echo "   ⚠️  WARN (not a failure): $name"; fi; }

KIT=scripts/ralph/prd-270w1.json
PRD_DOC=docs/PRDS/PRD-270-SYNC-WITH-AUTOMATOS-CLOUD.md
O=orchestrator
TESTS=$O/tests
VERSIONS=$O/alembic/versions
MANIFEST=$O/reports/route-manifest.json
SURFACE=$O/reports/config-surface.json
THEMES_SVC=$O/services/themes
MC=$O/services/marketplace_cloud
MP=$O/services/marketplace_public
CAT_ROUTER=$O/api/marketplace_catalog.py
PUB_ROUTER=$O/api/public_marketplace.py
SYNC_ROUTER=$O/api/marketplace_cloud_sync.py
FE=frontend
MKT=$FE/components/marketplace
PUBC=$MKT/public
THEMES_FE=$MKT/themes
NEW_REV=prd270_w1_cloud_sync
# K3: files over 800 lines this wave may touch; none may grow.
NO_GROW="$FE/lib/api-client.ts $O/main.py $O/api/marketplace.py docs/getting-started/self-hosting.md $TESTS/test_prd261_theme_models.py"
# Over 800 lines and allowed to grow: settings live only in config.py; the two reports are generated lists.
GROW_OK="$O/config.py $MANIFEST $SURFACE"
# K31: the pinned tests that move with approved changes; each keeps at least its test count.
PIN_TESTS="$TESTS/test_prd209_alembic_single_head.py $TESTS/test_prd236_w1_routes.py $TESTS/test_prd261_theme_models.py"
PIN_TS_TESTS="$FE/lib/__tests__/csp-media-sources.test.ts"
# K4: never touched (the pattern copied from them, the member routes, the untouchable components and pages).
REVOCATION_FILES="$THEMES_SVC/revocation_pull.py $THEMES_SVC/revocation_feed.py $O/services/theme_revocation_scheduler.py $O/api/themes_revocations_feed.py $O/api/admin_theme_revocation_sync.py"
MEMBER_ROUTERS="$O/api/marketplace.py $O/api/marketplace_plugins.py $O/api/llm_marketplace.py $O/api/composio.py $O/api/workspace_skills.py $O/main.py"
BIG_TABS="$MKT/marketplace-tools-tab.tsx $MKT/marketplace-plugins-tab.tsx $MKT/marketplace-llms-tab.tsx $MKT/marketplace-skills-tab.tsx $MKT/marketplace-playbooks-tab.tsx $MKT/marketplace-agents-tab.tsx $MKT/marketplace-packages-tab.tsx"
AUTH_SURFACES="$FE/components/auth/sign-in-form.tsx $FE/components/auth/sign-up-form.tsx $FE/components/auth/sign-up-parts.tsx $FE/app/sign-in $FE/app/sign-up $FE/app/sso-callback"
# K4: byte-identical with the base, as 'file function' pairs.
SAME_FUNCTIONS="$THEMES_SVC/review.py:approve $THEMES_SVC/listings.py:install_listing $THEMES_SVC/listing_updates.py:offer_listed_update $THEMES_SVC/auto_update.py:apply_listed_update $THEMES_SVC/updates.py:apply_update $THEMES_SVC/serve.py:authorised_package $THEMES_SVC/serve.py:read_stored_file $THEMES_SVC/serve.py:verified_file $THEMES_SVC/review_package.py:rebuilt_package $THEMES_SVC/signed_upload.py:check_upload_signature $THEMES_SVC/signature_verify.py:verify $THEMES_SVC/package.py:read_theme_package"
# This wave adds no SDK change: the theme SDK's surface and both versions are the base's.
SDK_PATHS="$FE/packages/theme-sdk $FE/lib/theme-sdk/surface $THEMES_SVC/data/sdk-surface.json $FE/lib/theme-host/shared-modules.json"

# --- reading the commit -----------------------------------------------------------------------------
src() { git show "$HEAD_SHA:$1" 2>/dev/null; }
exists() { git cat-file -e "$HEAD_SHA:$1" 2>/dev/null; }
lines_at() { git show "$1:$2" 2>/dev/null | wc -l | tr -d ' '; }
has() { exists "$2" && src "$2" | grep -qE -e "$1"; }      # has <ERE> <file>: false when the file is missing
lacks() { ! has "$1" "$2"; }
# grep_tree <ERE> <paths…>: a line of the committed tree under the paths matches.
grep_tree() { local pattern="$1"; shift; git grep -qE -e "$pattern" "$HEAD_SHA" -- "$@" 2>/dev/null; }
all_exist() { local f; for f in "$@"; do exists "$f" || { echo "   missing: $f"; return 1; }; done; }
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
NEW_WEB="$PUBC $MKT/cloud $MKT/deep-link $THEMES_FE $FE/components/settings/automatos-cloud-panel.tsx $FE/components/auth/resume-after-sign-in.tsx $FE/lib/marketplace $FE/hooks"
hex_added()        { added_code $NEW_WEB | grep -E "#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b['\" ;)]" | grep -q .; }
rounded3_added()   { added_code $NEW_WEB | grep -E '\brounded-3xl\b' | grep -q .; }
autoplay_added()   { added_code frontend | grep -iE '\bautoplay\b|autoPlay\s*[=:]' | grep -q .; }
# K13: httpx (already a dependency) only in the one guarded GET; no other HTTP client anywhere.
other_client_added() { added_code orchestrator | grep -E '^\+\s*(import|from)\s+(requests|aiohttp|urllib3)\b' | grep -q .; }
httpx_outside_fetch() { added_code orchestrator ":!$MC/fetch.py" | grep -E '^\+\s*(import|from)\s+httpx\b' | grep -q .; }
# K27: no model call in this wave.
model_call_added() { added_code orchestrator | grep -E 'LLMManager|get_llm_manager|orchestrator_llm|chat_completion|\.complete\(|anthropic\.' | grep -q .; }
# K28 / §8-Q2: no Phase 2 surface, no scheduled Sync.
phase2_added() { added_code orchestrator frontend | grep -E 'item_bundle|ItemBundle|bundle_format|export_bundle|import_bundle|Reviewed on Automatos Cloud' | grep -q .; }
scheduled_sync_added() { added_code orchestrator | grep -E 'add_job\(|IntervalTrigger|CronTrigger' | grep -q .; }
# K16: no module of the Cloud pull imports the review queue (review.approve is never called for a Cloud row).
no_review_import() { local f bad=0; for f in $(git ls-tree -r --name-only "$HEAD_SHA" -- "$MC" | awk '/\.py$/'); do
    src "$f" | grep -qE '^\s*from services\.themes\.review import|^\s*from services\.themes import .*\breview\b|^\s*import services\.themes\.review\b' && { echo "   $f imports services.themes.review"; bad=1; }
  done; [ $bad -eq 0 ]; }
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
config_growth() { local b h; b=$(lines_at "$BASE" $O/config.py); h=$(lines_at "$HEAD_SHA" $O/config.py); echo "   config.py: base $b, HEAD $h (settings only: 5 settings with their comments)"; [ "$h" -le $((b + 40)) ]; }
test_fn_count() { git show "$1:$2" 2>/dev/null | awk '/^(async )?def test_/{n++} END{print n+0}'; }
ts_test_count() { git show "$1:$2" 2>/dev/null | awk '/^[ \t]*(it|test)(\.each\([^)]*\))?\(/{n++} END{print n+0}'; }
tests_not_weakened() { local d a; d=$(git diff "$BASE" "$HEAD_SHA" -- $TESTS frontend e2e | awk '/^-[ \t]*(async )?def test_|^-[ \t]*(it|test)(\.each\([^)]*\))?\(/{n++} END{print n+0}'); a=$(git diff "$BASE" "$HEAD_SHA" -- $TESTS frontend e2e | awk '/^\+[ \t]*(async )?def test_|^\+[ \t]*(it|test)(\.each\([^)]*\))?\(/{n++} END{print n+0}'); echo "   test functions removed $d, added $a"; [ "$a" -ge "$d" ]; }
pin_tests_kept() { local f b h bad=0
  for f in $PIN_TESTS; do exists "$f" || { echo "   $f is missing"; bad=1; continue; }
    b=$(test_fn_count "$BASE" "$f"); h=$(test_fn_count "$HEAD_SHA" "$f"); [ "$h" -ge "$b" ] || { echo "   $f: base $b, HEAD $h"; bad=1; }; done
  for f in $PIN_TS_TESTS; do exists "$f" || { echo "   $f is missing"; bad=1; continue; }
    b=$(ts_test_count "$BASE" "$f"); h=$(ts_test_count "$HEAD_SHA" "$f"); [ "$h" -ge "$b" ] || { echo "   $f: base $b, HEAD $h"; bad=1; }; done
  [ $bad -eq 0 ] && echo "   every K31 pin file keeps its test count"; [ $bad -eq 0 ]; }
old_migrations_untouched() { [ -z "$(git diff --name-only --diff-filter=MDR "$BASE" "$HEAD_SHA" -- "$VERSIONS")" ]; }
# Only this kit's own files change under scripts/ralph (the runner seeds the four; the marks commit the json).
other_kits_untouched() { local out; out=$(git diff --name-only "$BASE" "$HEAD_SHA" -- scripts/ralph | awk '!/scripts\/ralph\/(prd-270w1\.json|PROMPT_build_prd270w1\.md|PROMPT_review_prd270w1\.md|acceptance-prd270w1\.sh)$/'); [ -z "$out" ] || { echo "$out" | sed 's/^/   another kit file changed: /'; return 1; }; }
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

# A function by its name ('Class.method', or a plain name anywhere), read from a commit with git.
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
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    return "\n".join(src.splitlines()[start - 1:node.end_lineno])
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
if op == "contains":
    ok = re.search(extra[0], segment(head, node)) is not None
    print(f"   {qual} {'mentions' if ok else 'does not mention'} /{extra[0]}/"); sys.exit(0 if ok else 1)
print(f"   unknown op {op}"); sys.exit(2)
PY
}
fn_unchanged() { fn_tool same "$HEAD_SHA" "$1" "$2" "$BASE"; }
same_functions_ok() { local pair bad=0; for pair in $SAME_FUNCTIONS; do fn_unchanged "${pair%%:*}" "${pair##*:}" || bad=1; done; [ $bad -eq 0 ]; }

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
  has '^_ADD_SOURCE *=' "$f" || { echo "   $f: no _ADD_SOURCE guarded block (K6)"; return 1; }
  src "$f" | grep -qiE "ALTER TABLE theme_packages ADD COLUMN source VARCHAR\(16\);" || { echo "   $f: no ADD COLUMN source VARCHAR(16)"; return 1; }
  src "$f" | grep -qF "ADD CONSTRAINT ck_theme_packages_source CHECK (source IS NULL OR (source = 'cloud' AND owner_type = 'marketplace'));" || { echo "   $f: ck_theme_packages_source is not K6's CHECK"; return 1; }
  src "$f" | grep -qiE "CREATE TABLE IF NOT EXISTS marketplace_cloud_sync" || { echo "   $f: no CREATE TABLE IF NOT EXISTS marketplace_cloud_sync"; return 1; }
  has 'ck_marketplace_cloud_sync_single_row' "$f" || { echo "   $f: no single-row CHECK"; return 1; }
  ! src "$f" | grep -qiE "DELETE FROM|TRUNCATE|UPDATE theme_packages" || { echo "   $f deletes or rewrites rows (K6: none)"; return 1; }
  ! src "$f" | grep -qiE "ALTER TABLE (theme_installations|theme_revocations|theme_revocation_sync|agents|workflow_recipes|users|workspaces|marketplace_plugins)" || { echo "   $f touches a table K6 leaves alone"; return 1; }
  has "EXPECTED_HEAD = \"$NEW_REV\"" $TESTS/test_prd209_alembic_single_head.py || { echo "   test_prd209 EXPECTED_HEAD is not $NEW_REV"; return 1; }
  has "$NEW_REV\\.py" $TESTS/test_prd236_w1_routes.py || { echo "   test_prd236_w1_routes.py does not read $NEW_REV.py"; return 1; }
  has "EXPECTED_HEAD = \"$NEW_REV\"" $TESTS/test_prd236_w1_routes.py || { echo "   test_prd236_w1_routes.py does not assert EXPECTED_HEAD = $NEW_REV"; return 1; }
  has "$NEW_REV" $TESTS/test_prd261_theme_models.py || { echo "   test_prd261_theme_models.py does not read $NEW_REV (K31)"; return 1; }
  echo "   $NEW_REV chains onto $bh; theme_packages.source and its CHECK; marketplace_cloud_sync; no row changed; the three pins moved"
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

# --- routes, settings, the middleware -----------------------------------------------------------
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
# K7: AUTOMATOS_CLOUD_URL defaults to Automatos Cloud (on for every local install, §8-Q6), empty turns it off.
# Not declared yet (the base) passes: US-270-00's story check requires the setting.
cloud_url_default() { local block; block=$(src $O/config.py | grep -A3 -E '^\s+AUTOMATOS_CLOUD_URL\b')
  [ -z "$block" ] && { echo "   AUTOMATOS_CLOUD_URL not declared"; return 0; }
  echo "$block" | sed 's/^/   /' | head -4; echo "$block" | grep -q '"https://api.automatos.app"'; }
# The new routers' handlers are plain def (sync session; AGENTS.md). A router not built yet is skipped.
routers_plain_def() { "$PY" - "$HEAD_SHA" "$CAT_ROUTER" "$PUB_ROUTER" "$SYNC_ROUTER" <<'PY'
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
# K11: the public routers carry no hybrid dependency (a router not built yet passes: its story check fails instead).
public_routers_unauthenticated() { local f bad=0; for f in "$CAT_ROUTER" "$PUB_ROUTER"; do
    has 'get_request_context_hybrid' "$f" && { echo "   $f imports get_request_context_hybrid"; bad=1; }
  done; [ $bad -eq 0 ]; }
# K19: frontend/middleware.ts's public list is the base's plus exactly '/marketplace', and no line of the file was removed.
# middleware_one_path strict: '/marketplace' must have been added (the story); lenient: nothing, or only it (the scope).
middleware_one_path() {
  local removed mode="${1:-lenient}"
  removed=$(git diff "$BASE" "$HEAD_SHA" -- frontend/middleware.ts | awk '/^-/ && !/^---/' | wc -l | tr -d ' ')
  [ "$removed" -eq 0 ] || { echo "   frontend/middleware.ts lost $removed line(s)"; return 1; }
  "$PY" - "$BASE" "$HEAD_SHA" "$mode" <<'PY'
import re, subprocess, sys
def routes(ref):
    out = subprocess.run(["git", "show", f"{ref}:frontend/middleware.ts"], capture_output=True, text=True)
    if out.returncode != 0:
        return None
    text = out.stdout
    start = text.find("createRouteMatcher([")
    end = text.find("]);", start)
    if start < 0 or end < 0:
        return None
    block = re.sub(r"//[^\n]*", "", text[start:end])
    return sorted(a or b for a, b in re.findall(r'"([^"]*)"|\'([^\']*)\'', block))
base, head = routes(sys.argv[1]), routes(sys.argv[2])
if base is None or head is None:
    print("   the public route list can't be read"); sys.exit(1)
added = list(head)
for item in base:
    if item in added:
        added.remove(item)
    else:
        print(f"   {item} left the public list"); sys.exit(1)
print(f"   added to isPublicRoute: {added}")
allowed = ([["/marketplace"]] if sys.argv[3] == "strict" else [[], ["/marketplace"]])
sys.exit(0 if added in allowed else 1)
PY
}
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
versions_kept() { local f=$THEMES_SVC/contract.py b h
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
# The browser lane (US-270-05c) runs in browser-e2e.yml on the PR. A WebKit 'internal error' on reload is a known
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
  for s in "#1377" "#1373" "DRAFT" "REVIEW_PASS" "prd270_w1_cloud_sync" "AUTOMATOS_CLOUD_URL" "/marketplace" "Phase 2" "--subject" "AI assistance"; do
    printf '%s' "$body" | grep -qF -- "$s" || { echo "   the PR body lacks: $s"; miss=1; }
  done
  [ $miss -eq 0 ] && echo "   PR #$n records #1377, #1373, the draft rule, the migration, the Cloud setting, the public page, Phase 2's own wave, the squash rule and the AI disclosure"
}
pr_base_ok() { local n base; n=$(open_pr); [ -n "$n" ] || { echo "   no open PR"; return 1; }; base=$(gh pr view "$n" --json baseRefName --jq .baseRefName 2>/dev/null); echo "   PR #$n base: $base"; [ "$base" = "main" ]; }
# K1: the PR stays a draft until REVIEW_PASS; the owner undrafts it.
pr_is_draft() { local n d; n=$(open_pr); [ -n "$n" ] || { echo "   no open PR"; return 1; }; d=$(gh pr view "$n" --json isDraft --jq .isDraft 2>/dev/null); echo "   PR #$n isDraft: $d"; [ "$d" = "true" ]; }

if [ "$MODE" != "negative" ]; then
  echo ""; echo "=== Seeded"
  check "the PRD is tracked at $PRD_DOC" "exists $PRD_DOC"
  check "the kit is committed and valid JSON" "exists $KIT && src $KIT | \"$PY\" -m json.tool >/dev/null"
  check "the base carries #1373's phone previews (phoneCover in listings.py, a 32 MB package cap)" "has 'phoneCover' $THEMES_SVC/listings.py && has 'THEME_MAX_PACKAGE_MB.*32' $O/config.py"

  echo ""; echo "=== Stories (priority order; positive: red on the base by design)"
  check "US-270-00 one revision $NEW_REV on the base head: theme_packages.source, marketplace_cloud_sync, the pins" "one_revision && revision_ok"
  check "US-270-00 the models, the five settings, envs/api.defaults; the tests are new" "has 'source = Column' $O/core/models/themes.py && has 'ck_theme_packages_source' $O/core/models/themes.py && has 'class MarketplaceCloudSync' $O/core/models/marketplace_cloud.py && has 'MarketplaceCloudSync' $O/core/models/__init__.py && settings_declared AUTOMATOS_CLOUD_URL AUTOMATOS_CLOUD_FEED_MAX_KB AUTOMATOS_CLOUD_TIMEOUT_SECONDS MARKETPLACE_CATALOG_FEED_MAX_KB MARKETPLACE_PUBLIC_RATE_PER_MINUTE && has '^AUTOMATOS_CLOUD_URL=' envs/api.defaults && new_files $TESTS/test_prd270w1_migration.py $TESTS/test_prd270w1_settings.py"
  check "US-270-01 the hosted catalogue: build_catalog from listings, the public route (404 local, cached); the test is new" "all_exist $MC/__init__.py $MC/catalog.py $CAT_ROUTER && has 'def build_catalog' $MC/catalog.py && has 'all_listings' $MC/catalog.py && has 'themeJson' $MC/catalog.py && router_spec api.marketplace_catalog && routes_in_manifest 'GET /api/marketplace/catalog' && has 'IS_LOCAL_EDITION' $CAT_ROUTER && grep_tree 'max-age=300|max-age=\\{' $CAT_ROUTER $MC && new_files $TESTS/test_prd270w1_catalog_feed.py"
  check "US-270-02 the catalogue files: the rebuilt package and listed screenshots, the serving headers; the test is new" "all_exist $MC/catalog_files.py && has 'rebuilt_package' $MC/catalog_files.py && has 'verified_file' $MC/catalog_files.py && grep_tree 'SERVING_HEADERS' $CAT_ROUTER $MC/catalog_files.py && routes_in_manifest 'GET /api/marketplace/catalog/files/\\{digest\\}/\\{path:path\\}' && new_files $TESTS/test_prd270w1_catalog_files.py"
  check "US-270-07a the public reads for agents, Playbooks and packages: allowlists, rate limit, SaaS only; the test is new" "all_exist $MP/__init__.py $MP/fields.py $MP/reads.py $MP/agents.py $MP/playbooks.py $MP/packages.py $PUB_ROUTER && router_spec api.public_marketplace && routes_in_manifest 'GET /api/public/marketplace/\\{[a-z_]+\\}' && has 'RateLimitStore' $PUB_ROUTER && has 'IS_LOCAL_EDITION' $PUB_ROUTER && new_files $TESTS/test_prd270w1_public_reads.py"
  check "US-270-07b the public reads for applications, plugins, skills and LLMs; the test is new" "all_exist $MP/applications.py $MP/plugins.py $MP/skills.py $MP/llms.py && has 'applications' $MP/reads.py && has 'llms' $MP/reads.py && new_files $TESTS/test_prd270w1_public_reads_more.py"
  check "US-270-03a the local pull: the guarded GET, the per-item checks, the one-transaction Sync and its status; the test is new" "all_exist $MC/fetch.py $MC/feed.py $MC/pull.py $MC/urls.py && has 'follow_redirects=False' $MC/fetch.py && has 'Automatos/' $MC/fetch.py && has 'check_upload_signature' $MC/feed.py && has 'digest_of_manifest' $MC/feed.py && has 'IS_LOCAL_EDITION' $MC/pull.py && has 'MarketplaceCloudSync' $MC/pull.py && new_files $TESTS/test_prd270w1_cloud_pull.py"
  check "US-270-03b the sync routes (local edition, workspace:manage) and the wave's wire test; the tests are new" "all_exist $SYNC_ROUTER && router_spec api.marketplace_cloud_sync && routes_in_manifest 'POST /api/marketplace/cloud-sync' 'GET /api/marketplace/cloud-sync/status' && has 'require_workspace_permission' $SYNC_ROUTER && has 'workspace:manage' $SYNC_ROUTER && has 'IS_LOCAL_EDITION' $SYNC_ROUTER && new_files $TESTS/test_prd270w1_cloud_sync_routes.py $TESTS/test_prd270w1_wire.py"
  check "US-270-04a ensure_stored: download, read, digest, signature, store; wired into install and update; the test is new" "all_exist $MC/package.py && has 'def ensure_stored' $MC/package.py && has 'read_theme_package' $MC/package.py && has 'check_upload_signature' $MC/package.py && has 'write_theme_files' $MC/package.py && added_in 'ensure_stored' $O/api/marketplace_themes.py && added_in 'ensure_stored' $O/api/themes_install.py && new_files $TESTS/test_prd270w1_cloud_install.py"
  check "US-270-04b the auto-update after a Sync (offer + apply, never review.approve); the test is new" "all_exist $MC/updates.py && has 'offer_listed_update' $MC/updates.py && has 'apply_listed_update' $MC/updates.py && no_review_import && grep_tree 'updatesApplied' $SYNC_ROUTER $MC && new_files $TESTS/test_prd270w1_cloud_auto_update.py"
  check "US-270-03c Cloud rows in the local storefront: source on the card, screenshots from Cloud; the test is new" "added_in 'source' $THEMES_SVC/listings.py && has 'cloud_file_url' $THEMES_SVC/listings.py && has 'def cloud_file_url' $MC/urls.py && new_files $TESTS/test_prd270w1_cloud_listings.py"
  check "US-270-05a the Sync button, the badge, Cloud screenshots allowed (themeFileUrl, CSP); the tests are new" "all_exist $FE/lib/marketplace/cloud-sync.ts $FE/hooks/use-cloud-sync.ts $MKT/cloud/sync-button.tsx && has 'SyncButton|sync-button' $MKT/marketplace-homepage.tsx && grep_tree 'From Automatos Cloud' $MKT $FE/lib/marketplace && has 'source' $THEMES_FE/listing-badges.tsx && has 'catalog/files' $THEMES_FE/listing-format.ts && has 'https://api\\.automatos\\.app' $FE/csp-sources.js && changed $FE/next.config.js && changed $FE/lib/__tests__/csp-media-sources.test.ts && new_files $MKT/cloud/__tests__/sync-button.test.tsx $THEMES_FE/__tests__/cloud-badge.test.tsx $FE/lib/__tests__/csp-cloud-images.test.ts"
  check "US-270-05b Settings › Automatos Cloud (a SettingsPanel tab; SystemSettingsTab untouched); the test is new" "all_exist $FE/components/settings/automatos-cloud-panel.tsx && has 'automatos-cloud' $FE/components/settings/SettingsPanel.tsx && untouched $FE/components/settings/SystemSettingsTab.tsx && new_files $FE/components/settings/__tests__/automatos-cloud-panel.test.tsx"
  check "US-270-08a /marketplace public (one path), the page split, the visitor shell; the tests are new" "middleware_one_path strict && all_exist $PUBC/marketplace-page.tsx $PUBC/visitor-marketplace.tsx $PUBC/visitor-header.tsx && lacks '^.use client.' $FE/app/marketplace/page.tsx && has 'marketplace-page|MarketplacePage' $FE/app/marketplace/page.tsx && new_files $FE/lib/__tests__/marketplace-public-middleware.test.ts $PUBC/__tests__/marketplace-page.test.tsx"
  check "US-270-08b the visitor's tabs: the catalogue storefront through the listing source, the read-only tabs; the tests are new" "all_exist $FE/lib/marketplace/public-catalog.ts $FE/hooks/use-public-marketplace.ts $THEMES_FE/listing-source.tsx $PUBC/visitor-tabs.tsx && has '/api/marketplace/catalog' $FE/lib/marketplace/public-catalog.ts && has '/api/public/marketplace' $FE/lib/marketplace/public-catalog.ts && has 'listing-source|ListingSource' $THEMES_FE/storefront/hero.tsx && new_files $FE/lib/marketplace/__tests__/public-catalog.test.ts $PUBC/__tests__/visitor-tabs.test.tsx"
  check "US-270-09a Add while signed out: the remembered return path, ResumeAfterSignIn; the auth pages untouched; the tests are new" "all_exist $FE/lib/marketplace/return-after-sign-in.ts $FE/components/auth/resume-after-sign-in.tsx && has 'ResumeAfterSignIn' $FE/components/providers.tsx && has 'redirect_url' $FE/lib/marketplace/return-after-sign-in.ts && has 'sessionStorage' $FE/lib/marketplace/return-after-sign-in.ts && untouched $AUTH_SURFACES && new_files $FE/lib/marketplace/__tests__/return-after-sign-in.test.ts $FE/components/auth/__tests__/resume-after-sign-in.test.tsx $PUBC/__tests__/visitor-add.test.tsx"
  check "US-270-09b ?item=&add=1: the themes dialog and the one deep-link component; the big tabs untouched; the tests are new" "all_exist $MKT/deep-link/marketplace-deep-link.tsx && has 'MarketplaceDeepLink' $MKT/marketplace-homepage.tsx && added_in 'add' $THEMES_FE/tab.tsx && untouched $BIG_TABS && new_files $MKT/__tests__/marketplace-deep-link.test.tsx $THEMES_FE/__tests__/themes-tab-add-link.test.tsx"
  check "US-270-10 share cards (generateMetadata, og:image, canonical) and robots; the tests are new" "has 'generateMetadata' $FE/app/marketplace/page.tsx && grep_tree 'openGraph' $FE/app/marketplace $FE/lib/marketplace && all_exist $FE/app/robots.ts && has '/marketplace' $FE/app/robots.ts && new_files $FE/app/marketplace/__tests__/marketplace-metadata.test.ts $FE/app/__tests__/robots.test.ts"
  check "US-270-06 the docs: the hub row in place, the Sync page, linked; the test is new" "all_exist docs/community-marketplace/sync-with-automatos-cloud.md && has 'AUTOMATOS_CLOUD_URL' docs/community-marketplace/sync-with-automatos-cloud.md && has 'sync-with-automatos-cloud' docs/community-marketplace/_index.md && added_in 'Automatos Cloud' docs/getting-started/self-hosting.md && not_grown docs/getting-started/self-hosting.md && new_files $TESTS/test_prd270w1_docs.py"
  check "US-270-05c the browser lane's spec is new and tagged @host; the lane and its dependencies unchanged" "new_files e2e/browser/specs/cloud-sync.spec.ts && has '@host' e2e/browser/specs/cloud-sync.spec.ts && untouched .github/workflows/browser-e2e.yml e2e/browser/package.json e2e/browser/package-lock.json"
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
check "N12a the K3 files over 800 lines did not grow (api-client.ts, main.py, api/marketplace.py, self-hosting.md, test_prd261_theme_models.py)" "named_files_not_grown"
check "N12b no other file over 800 lines grew (config.py and the two reports excepted)" "no_big_file_grew"
warn "N12c config.py grew by settings only (≤ 40 lines)" "config_growth"
check "N13 the K4 functions are byte-identical to the base (review.approve, install_listing, the update moves, serve, the re-pack, the checks)" "same_functions_ok"
check "N14a the revocation modules are untouched (K13: the pattern is copied, never edited)" "untouched $REVOCATION_FILES"
check "N14b the member Marketplace routers and main.py are untouched (K10, K12)" "untouched $MEMBER_ROUTERS"
check "N14c the untouchable web components are untouched (SystemSettingsTab, the seven big tabs, the auth forms and pages; K4)" "untouched $FE/components/settings/SystemSettingsTab.tsx $BIG_TABS $AUTH_SURFACES"
check "N15 the hierarchy gate passes (stdlib ast)" "hierarchy_gate_ok"
check "N16 the new routers' handlers are plain def (sync session)" "routers_plain_def"
check "N17 no new HTTP client: requests/aiohttp/urllib3 never, httpx only in $MC/fetch.py (K13)" "! other_client_added && ! httpx_outside_fetch"
check "N18 no model call added (K27)" "! model_call_added"
check "N19 no existing migration modified, deleted or renamed" "old_migrations_untouched"
check "N20 isPublicRoute gains nothing but '/marketplace' and loses nothing (K19)" "middleware_one_path lenient"
check "N21 the public routers carry no get_request_context_hybrid (K11)" "public_routers_unauthenticated"
check "N22 no Phase 2 surface and no scheduled Sync (K28, §8-Q2)" "! phase2_added && ! scheduled_sync_added"
check "N23 CI's changed-lines code shape passes on the committed wave (functions ≤ 50, nesting ≤ 4, new files ≤ 800)" "code_shape_ok"
check "N24 no other kit's file changed on this branch" "other_kits_untouched"
warn "N25 story commits carry [skip ci] (all but the final CI commit and a merge of main)" "[ -z \"\$(git log --no-merges --format='%s' $BASE..$HEAD_SHA^ 2>/dev/null | grep -v 'skip ci')\" ]"
check "N26 AUTOMATOS_CLOUD_URL defaults to https://api.automatos.app (on by default, §8-Q6; empty = off)" "cloud_url_default"
check "N27 no SDK change: the theme SDK's surface and both versions are the base's" "untouched $SDK_PATHS && versions_kept"
check "N28 no innerHTML / dangerouslySetInnerHTML added in the frontend" "! inner_html_added"
check "N29 no orange-* class, no hex colour and no rounded-3xl in the new web surfaces (K29)" "! orange_added && ! hex_added && ! rounded3_added"
check "N30 nothing auto-plays" "! autoplay_added"
check "N31 the K31 pin files keep their test counts" "pin_tests_kept"
check "N32 no test weakened: added test functions >= removed" "tests_not_weakened"
check "N33 no module of the Cloud pull imports the review queue (K16)" "no_review_import"
check "N34 docker-compose.yml and the browser lane's workflow are untouched by this wave" "untouched docker-compose.yml .github/workflows/browser-e2e.yml"

if [ "$MODE" = "full" ]; then
  echo ""; echo "=== CI and the PR (once per wave)"
  check "CI: the required jobs present and green on the tested commit" "ci_green_on_head"
  check "the browser lane (browser-e2e.yml) is green on the tested commit" "browser_lane_green"
  check "the PR targets main (not stacked, K1)" "pr_base_ok"
  check "the PR body records #1377, #1373, the draft rule, the migration, the Cloud setting, /marketplace, Phase 2, the squash rule and the AI disclosure" "pr_body_records"
  warn "the PR is still a draft (K1: only the owner undrafts it, after REVIEW_PASS)" "pr_is_draft"
fi
echo ""
if [ "$FAIL" -eq 0 ]; then echo "✅ PRD-270 W1 open Marketplace acceptance ($MODE): PASS"; else echo "❌ PRD-270 W1 open Marketplace acceptance ($MODE): FAIL"; fi
exit "$FAIL"
