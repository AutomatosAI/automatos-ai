#!/bin/bash
# Acceptance gate — PRD-259 Skins in Studio, Wave 1 (S-001, S-002, S-003, S-009, S-004a, S-004b, S-010a, S-010b).
#   bash scripts/ralph/acceptance-prd259w1.sh | --print-base | --no-ci | --negative-only ; GATE_BASE=<ref> (smoke runs only).
# Base: the merge-base with origin/main (Wave 1 is cut from main). Nothing runs on the owner's machine: greps, git
# and stdlib python only; CI once per wave on the draft PR. No pipefail (a negative `! producer | grep -q` would
# invert on SIGPIPE); every check runs in its own `( eval … )`; stdin is closed so a grep with a missing file
# argument fails instead of hanging.
#
# Negative checks and the base (14 Oct 2026, origin/main f2be2abb0):
#   - every "added lines" check (N3–N10, N14–N17) reads `git diff BASE..HEAD`, which is EMPTY when HEAD is the
#     base, so it passes there by construction; each is scoped to code dirs and excludes tests, fixtures, the kit
#     (scripts/ralph) and docs, so a check's own pattern line or a test asserting absence never trips it;
#   - N2 (exactly one Alembic head) was smoke-checked by running its stdlib-ast parser over the base tree's
#     orchestrator/alembic/versions: 201 revisions, one head, prd258_voice_timings;
#   - N12 (new files <= 800 lines), N13 (api-client.ts not grown: 3,429 lines on the base), N11 (DCO) and N18
#     (tests not weakened) compare against the base and pass on it; N19 (hierarchy gate) is the script CI and the
#     256 gates already run — smoke-checked by reading only, not executed here;
#   - the darkMode, --radius, manifest and config-surface checks are POSITIVE (the wave's work): they fail on the
#     base by design (darkMode is ['variant', ['&:is(.dark *)']] and --radius is undefined there, read 14 Oct).
set -u
exec </dev/null
cd "$(dirname "$0")/../.." || exit 1
export DATABASE_URL="postgresql://ralph:ralph@127.0.0.1:1/ralph_no_db"
export POSTGRES_HOST=127.0.0.1 POSTGRES_PORT=1 POSTGRES_USER=ralph POSTGRES_PASSWORD=ralph POSTGRES_DB=ralph_no_db
export REDIS_URL="redis://127.0.0.1:1/0" REDIS_HOST=127.0.0.1 REDIS_PORT=1
export ENVIRONMENT=development

if [ -n "${GATE_BASE:-}" ]; then
  BASE=$(git rev-parse --verify -q "$GATE_BASE^{commit}") || { echo "GATE_BASE $GATE_BASE does not resolve"; exit 1; }
else
  BASE=$(git merge-base HEAD origin/main) || { echo "cannot resolve merge-base with origin/main"; exit 1; }
fi
MODE=full
case "${1:-}" in --print-base) echo "$BASE"; exit 0 ;; --no-ci) MODE=no-ci ;; --negative-only) MODE=negative ;; "") ;; *) echo "unknown argument: $1"; exit 2 ;; esac
echo "base: $(git log -1 --format='%h %s' "$BASE") · HEAD: $(git log -1 --format='%h %s' HEAD) · mode: $MODE"
FAIL=0
check() { local name="$1" body="$2"; echo ""; echo "── $name"; if ( eval "$body" ); then echo "   ✅ PASS: $name"; else echo "   ❌ FAIL: $name"; FAIL=1; fi; }
warn() { local name="$1" body="$2"; echo ""; echo "── $name"; if ( eval "$body" ); then echo "   ✅ ok: $name"; else echo "   ⚠️  WARN (not a failure): $name"; fi; }

KIT=scripts/ralph/prd-259w1.json
TESTS=orchestrator/tests
VERSIONS=orchestrator/alembic/versions
SVC=orchestrator/services/skins
API=orchestrator/api/marketplace_skins.py
LIB=frontend/lib/skins
MANIFEST=orchestrator/reports/route-manifest.json
SURFACE=orchestrator/reports/config-surface.json
AUDIT=docs/PRDS/PRD-259-DARK-VARIANT-AUDIT.md
NEW_REV=prd259_skins
SETTINGS="SKIN_MAX_PACKAGE_MB SKIN_MAX_FONT_KB SKIN_MAX_MASCOT_MB SKIN_MAX_PREVIEW_KB SKIN_ASSET_CACHE_SECONDS"

# The wave's ADDED lines under the given paths: tests, fixtures, the kit and docs excluded (negative greps only).
added_code() { git diff "$BASE"..HEAD -- "$@" ':!orchestrator/tests' ':!*__tests__*' ':!*.test.ts' ':!*.test.tsx' ':!scripts/ralph' ':!docs' | grep '^+' | grep -v '^+++'; }
added_files() { git diff --name-only --diff-filter=A "$BASE"..HEAD -- "$@"; }
lines_at() { git show "$1:$2" 2>/dev/null | wc -l | tr -d ' '; }
has() { [ -f "$2" ] && grep -qE "$1" "$2"; }   # has <ERE> <file>: false (never a stdin read) when the file is missing

env_read_added()     { added_code orchestrator ':!orchestrator/config.py' | grep -E 'os\.(getenv|environ)' | grep -q .; }
print_added()        { added_code orchestrator ':!orchestrator/scripts' | grep -E '^\+\s*print\(' | grep -q .; }
raw_fetch_added()    { added_code frontend | grep -E "fetch\(\s*['\"\`]/api" | grep -q .; }
s3_client_added()    { added_code orchestrator | grep -E 'boto3\.(client|resource|session\.Session)\(|Minio\(|aioboto3' | grep -q .; }
inner_html_added()   { added_code frontend ':!frontend/app/layout.tsx' | grep -E 'dangerouslySetInnerHTML|\.innerHTML\s*=' | grep -q .; }
font_face_added()    { added_code "$LIB" | grep -E '@font-face' | grep -v -E '^\+\s*(//|\*|/\*)' | grep -q .; }
play_import_added()  { added_code frontend orchestrator | grep -E "from ['\"][^'\"]*automatos-play|require\(['\"][^'\"]*automatos-play" | grep -q .; }
orange_added()       { added_code frontend | grep -E '\b(bg|text|border|ring|from|to|via)-orange-[0-9]' | grep -q .; }
every_commit_signed() { [ -z "$(git log --format='%H %(trailers:key=Signed-off-by,valueonly)' "$BASE"..HEAD | awk 'NF<2{print $1}')" ]; }
file_sizes_ok() { local f n bad=0; for f in $(added_files orchestrator frontend); do [ -f "$f" ] || continue; n=$(wc -l < "$f" | tr -d ' '); [ "$n" -le 800 ] || { echo "   $f: $n lines"; bad=1; }; done; [ $bad -eq 0 ]; }
not_grown() { local f="$1" b h; b=$(lines_at "$BASE" "$f"); h=$(wc -l < "$f" | tr -d ' '); echo "   $f: base $b, HEAD $h"; [ "$h" -le "$b" ]; }
tests_not_weakened() { local d a; d=$(git diff "$BASE"..HEAD -- $TESTS frontend | grep -cE '^-\s*(async )?def test_|^-\s*(it|test)\('); a=$(git diff "$BASE"..HEAD -- $TESTS frontend | grep -cE '^\+\s*(async )?def test_|^\+\s*(it|test)\('); echo "   test functions removed $d, added $a"; [ "$a" -ge "$d" ]; }
old_migrations_untouched() { [ -z "$(git diff --name-only --diff-filter=MDR "$BASE"..HEAD -- "$VERSIONS")" ]; }

# --- the migration (S-002) ----------------------------------------------------------
base_head() { git show "$BASE:$TESTS/test_prd209_alembic_single_head.py" 2>/dev/null | grep -E '^EXPECTED_HEAD *= *"' | head -1 | sed -E 's/.*"([^"]+)".*/\1/'; }
new_revisions() { added_files "$VERSIONS" | grep '\.py$'; }
one_revision() { local n; n=$(new_revisions | grep -c .); echo "   new revisions: $n ($(new_revisions | tr '\n' ' '))"; [ "$n" -eq 1 ]; }
revision_ok() {
  local f bh; f="$VERSIONS/$NEW_REV.py"; bh=$(base_head)
  [ -f "$f" ] || { echo "   $f is missing"; return 1; }
  [ -n "$bh" ] || { echo "   cannot read the base's EXPECTED_HEAD"; return 1; }
  grep -qE "^revision[^=]*= *['\"]$NEW_REV['\"]" "$f" || { echo "   $f: revision is not $NEW_REV"; return 1; }
  grep -qE "^down_revision[^=]*= *['\"]$bh['\"]" "$f" || { echo "   $f: down_revision is not the base head $bh"; return 1; }
  grep -q 'marketplace_skins' "$f" && grep -q 'skin_installations' "$f" || { echo "   $f lacks a table"; return 1; }
  [ "$(grep -ciE 'CREATE (TABLE|(UNIQUE )?INDEX)' "$f")" -le "$(grep -ciE 'IF NOT EXISTS' "$f")" ] || { echo "   $f: a CREATE without IF NOT EXISTS"; return 1; }
  [ "$(grep -ciE 'CREATE TABLE IF NOT EXISTS' "$f")" -ge 2 ] || { echo "   $f: the two tables are not CREATE TABLE IF NOT EXISTS (AGENTS.md → Migrations)"; return 1; }
  grep -q "EXPECTED_HEAD = \"$NEW_REV\"" $TESTS/test_prd209_alembic_single_head.py || { echo "   test_prd209 EXPECTED_HEAD is not $NEW_REV"; return 1; }
  grep -qF "$NEW_REV.py" $TESTS/test_prd236_w1_routes.py && grep -qF "EXPECTED_HEAD = \"$NEW_REV\"" $TESTS/test_prd236_w1_routes.py \
    || { echo "   test_prd236_w1_routes.py does not chain $NEW_REV or its EXPECTED_HEAD assert"; return 1; }
  echo "   $NEW_REV chains onto $bh; both pins moved"
}
one_head() { python3 - "$VERSIONS" <<'PY'
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

# --- manifests, settings, CSS facts ----------------------------------------------------
routes_in_manifest() { python3 - "$MANIFEST" <<'PY'
import json, re, sys
pairs = {(r.get("method", ""), r.get("path", "")) for r in json.load(open(sys.argv[1]))["routes"]}
want = [("GET", r"/api/skins/schema"), ("POST", r"/api/skins/import"), ("GET", r"/api/skins/installed"),
        ("GET", r"/api/skins/\{[a-z_]+\}/assets/\{[a-z_]+(:path)?\}")]
missing = [f"{m} {p}" for m, p in want if not any(m == pm and re.fullmatch(p, pp) for pm, pp in pairs)]
data = json.load(open(sys.argv[1]))
if data.get("route_count") != len(data["routes"]): missing.append(f"route_count {data.get('route_count')} != {len(data['routes'])} routes")
print("\n".join(f"   missing: {x}" for x in missing)); sys.exit(1 if missing else 0)
PY
}
settings_declared() { local s bad=0; for s in $SETTINGS; do
    grep -qE "^\s+$s\b" orchestrator/config.py || { echo "   config.py lacks $s"; bad=1; }
    grep -q "\"$s\"" "$SURFACE" || { echo "   config-surface.json lacks $s"; bad=1; }
  done; [ $bad -eq 0 ]; }
dark_mode_class() { grep -qE "darkMode:\s*\[\s*['\"]class['\"]\s*,\s*['\"]\.dark['\"]\s*\]" frontend/tailwind.config.ts && ! grep -qE "darkMode:\s*\[\s*['\"]variant" frontend/tailwind.config.ts; }
radius_defined() { python3 - frontend/app/globals.css <<'PY'
import re, sys
css = open(sys.argv[1]).read()
def block(sel):
    m = re.search(r"(^|\n)\s*" + re.escape(sel) + r"\s*\{", css)
    if not m: return ""
    return css[m.end():css.index("}", m.end())]
bad = [s for s in (":root", ".studio") if not re.search(r"--radius\s*:\s*[0-9.]+(rem|px)\s*;", block(s))]
print("\n".join(f"   --radius missing in the first {s} block" for s in bad)); sys.exit(1 if bad else 0)
PY
}
# main.py's PATH_BODY_LIMITS must carry the import route (the 10 MB transport default would 413 a 15 MB package).
body_limit_for_import() { awk '/^PATH_BODY_LIMITS *= *\{/{f=1} f{print} f&&/^\}/{exit}' orchestrator/main.py | grep -qiE 'skin'; }
dark_base_count() { git grep -noE "dark:[a-z!\[-][^ \"']*" "$BASE" -- 'frontend/*.ts' 'frontend/*.tsx' | grep -c .; }
audit_complete() {
  [ -f "$AUDIT" ] || { echo "   $AUDIT is missing"; return 1; }
  local want rows pending; want=$(dark_base_count); rows=$(grep -cE '^\|.*dark:' "$AUDIT"); pending=$(grep -ciE '^\|.*pending' "$AUDIT")
  echo "   audit rows $rows (base uses $want), pending $pending"
  [ "$rows" -ge "$want" ] && [ "$pending" -eq 0 ]
}
corpus_ok() { python3 - "$TESTS/fixtures/skins/cases.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
cases, toks = d.get("cases", []), d.get("allowedTokens", [])
print(f"   {len(cases)} cases, {len(toks)} allowed tokens, {len(d.get('contrastPairs', []))} pairs")
sys.exit(0 if len(cases) >= 25 and len(toks) == 47 and d.get("contrastPairs") else 1)
PY
}
all_acs_done() { python3 - "$KIT" <<'PY'
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
# Required jobs = test.yml's CURRENT jobs (read on origin/main 14 Oct): this wave touches orchestrator/ (backend_mode
# full → the shards, orchestrator-tests, Alembic from-zero, Schema-drift run) and frontend/ (Frontend CI runs).
# "Prod images built" runs only when a Dockerfile, a manifest, next.config.* or test.yml changes: green or skipped.
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
  local REQUIRED=("Which lanes this change needs" "orchestrator-tests" "Alembic from-zero" "Schema-drift check" "Frontend CI" "Code standards on changed lines" "Prod images built")
  local PATH_GATED=("Prod images built")
  for job in "${REQUIRED[@]}"; do
    hits=$(echo "$jobs" | grep -F "$job" | cut -f1)
    [ -n "$hits" ] || { echo "   missing job: $job"; return 1; }
    if printf '%s\n' "${PATH_GATED[@]}" | grep -qxF "$job"; then echo "$hits" | grep -qvE '^(success|skipped)$' && { echo "   red: $job"; return 1; }
    else echo "$hits" | grep -qvE '^success$' && { echo "   red: $job"; return 1; }; fi
  done
  echo "   test.yml run $id is green on $sha for all ${#REQUIRED[@]} required jobs (Prod images: green or skipped)"
}

if [ "$MODE" != "negative" ]; then
  echo ""; echo "=== Wave 1 stories"
  check "S-001 the rule set on both sides" "test -f $LIB/schema.ts && test -f $LIB/validate.ts && test -f $LIB/contrast.ts && test -f $LIB/color.ts && test -f $SVC/schema.py && test -f $SVC/validate.py && test -f $SVC/contrast.py"
  check "S-001 the shared corpus (>= 25 cases, 47 tokens) read by both suites" "corpus_ok && test -f $TESTS/test_prd259_skin_validator.py && has 'cases\.json' $LIB/__tests__/validator-parity.test.ts"
  check "S-001 GET /api/skins/schema routed and tested" "test -f $API && has 'marketplace_skins' orchestrator/main.py && grep -qsE 'schema' orchestrator/api/marketplace_skins*.py && test -f $TESTS/test_prd259_skin_schema_route.py"
  check "S-002 exactly one new Alembic revision" "one_revision"
  check "S-002 $NEW_REV chains onto the base head; both head pins moved" "revision_ok"
  check "S-002 the models are declared and registered" "has '__tablename__ = .marketplace_skins.' orchestrator/core/models/marketplace_skins.py && has '__tablename__ = .skin_installations.' orchestrator/core/models/marketplace_skins.py && has 'marketplace_skins' orchestrator/core/models/__init__.py"
  check "S-002 storage through core.storage; tests" "has 'core\.storage' $SVC/storage.py && has '_workspace' $SVC/storage.py && test -f $TESTS/test_prd259_skins_models.py && test -f $TESTS/test_prd259_skin_storage.py"
  check "S-002 the SKIN_* settings in config.py and config-surface.json" "settings_declared"
  check "S-003 the package reader, the import body limit, tests" "test -f $SVC/package.py && body_limit_for_import && test -f $TESTS/test_prd259_skin_package.py && test -f $TESTS/test_prd259_skin_routes.py"
  check "S-003 the assets route sets nosniff and the cache header" "grep -qsE 'nosniff' orchestrator/api/marketplace_skins*.py $SVC/*.py && grep -qsE 'SKIN_ASSET_CACHE_SECONDS' orchestrator/api/marketplace_skins*.py $SVC/*.py"
  check "S-001/S-003 the four routes are in route-manifest.json (route_count matches)" "routes_in_manifest"
  check "S-009 --radius in the first :root and .studio blocks of globals.css" "radius_defined && test -f frontend/components/__tests__/prd259-radius.test.ts"
  check "S-004a the generator: skinToCss, formatTriplet, SKIN_CSS_GRAMMAR" "has 'export function skinToCss|export const skinToCss' $LIB/css.ts && has 'formatTriplet' $LIB/css.ts && has 'SKIN_CSS_GRAMMAR' $LIB/css.ts"
  check "S-004a the raw-string test exists" "has 'never interpolates a raw string' $LIB/__tests__/css.test.ts"
  check "S-004a the pre-paint script, its cache and tests; fonts through apiClient's seam" "has 'SKIN_PREPAINT_SCRIPT' $LIB/prepaint.ts && test -f $LIB/cache.ts && test -f $LIB/__tests__/prepaint.test.ts && has 'getAuthHeaders|apiFileUrl|fetchBrandFileObjectUrl|ObjectUrl' $LIB/fonts.ts"
  check "S-004b the base cookie, data-skin, the pre-paint script in the layout" "has 'automatos-skin-base' frontend/lib/ui-style.ts && has 'data-skin' frontend/app/layout.tsx && has 'SKIN_PREPAINT_SCRIPT' frontend/app/layout.tsx"
  check "S-004b the dropdown's Skins group and Add a skin…; tests" "grep -rqE 'Add a skin' frontend/components/ui && test -f frontend/components/__tests__/prd259-style-dropdown.test.tsx && test -f frontend/lib/__tests__/ui-style-skins.test.ts"
  check "S-010a darkMode is ['class', '.dark'] (Tailwind 3.3.3 ignores 'variant')" "dark_mode_class && test -f frontend/lib/__tests__/tailwind-dark-variant.test.ts"
  check "S-010a/b the dark: audit covers every base use, none pending; the guard test" "audit_complete && test -f frontend/lib/__tests__/dark-variant-tokens.test.ts"
  check "every acceptance criterion in $KIT is DONE (the owner's are left)" "all_acs_done"
fi

echo ""; echo "=== Scope and conventions"
check "N2 exactly one Alembic head in the tree" "one_head"
check "N3 no env reads added outside config.py" "! env_read_added"
check "N4 no print() added in orchestrator code" "! print_added"
check "N5 no raw fetch('/api…') added in the frontend" "! raw_fetch_added"
check "N6 no new S3/MinIO client added (core.storage only)" "! s3_client_added"
check "N7 no new Python dependency" "git diff --quiet $BASE..HEAD -- orchestrator/requirements.txt orchestrator/requirements-dev.txt orchestrator/pyproject.toml"
check "N8 no new npm dependency" "git diff --quiet $BASE..HEAD -- frontend/package.json"
check "N9 the Composio deny list and the generated Auto skill seed are untouched" "git diff --quiet $BASE..HEAD -- orchestrator/core/composio/deny_list.py orchestrator/core/seeds/platform-management-skill.md"
check "N10 no node_modules in the diff" "! git diff --name-only $BASE..HEAD | grep -q node_modules"
check "N11 every commit is DCO-signed" "every_commit_signed"
check "N12 a new file is at most 800 lines" "file_sizes_ok"
check "N13 frontend/lib/api-client.ts did not grow (use its getBaseUrl/getAuthHeaders seam)" "not_grown frontend/lib/api-client.ts"
check "N14 no innerHTML / dangerouslySetInnerHTML added outside app/layout.tsx's pre-paint constant" "! inner_html_added"
check "N15 the skin engine writes no @font-face (fonts go through FontFace: K2)" "! font_face_added"
check "N16 nothing imports from automatos-play (port, don't import: D5)" "! play_import_added"
check "N17 no orange-* palette class added (.eslintrc no-restricted-syntax)" "! orange_added"
check "N18 no test weakened: added test functions >= removed" "tests_not_weakened"
check "N19 the hierarchy gate passes (stdlib ast)" "python3 orchestrator/scripts/check_hierarchy_gate.py >/dev/null"
check "N20 no existing migration modified, deleted or renamed" "old_migrations_untouched"
warn "N21 story commits carry [skip ci] (all but the final CI commit)" "[ -z \"\$(git log --format='%s' $BASE..HEAD^ | grep -v 'skip ci' | grep -v '^chore(prd-259): seed')\" ]"

if [ "$MODE" = "full" ]; then
  echo ""; echo "=== CI (once per wave)"
  check "CI: the required jobs present and green on the tested commit" "ci_green_on_head"
fi
echo ""
if [ "$FAIL" -eq 0 ]; then echo "✅ PRD-259 Wave 1 acceptance ($MODE): PASS"; else echo "❌ PRD-259 Wave 1 acceptance ($MODE): FAIL"; fi
exit "$FAIL"
