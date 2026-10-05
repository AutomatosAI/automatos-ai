#!/bin/bash
# Acceptance gate — PRD-255 Brand Kit v2, Wave 1: the v2 kit and every renderer reading it (US-001..US-008).
#   bash scripts/ralph/acceptance-prd255w1.sh                  run the gate
#   bash scripts/ralph/acceptance-prd255w1.sh --print-base     print the diff base
#   bash scripts/ralph/acceptance-prd255w1.sh --no-ci          every check but CI (a smoke run, never the loop's gate)
#   bash scripts/ralph/acceptance-prd255w1.sh --negative-only  only the scope/convention (negative) checks: each
#                                                              must pass on the base itself (smoke-run them there)
#   GATE_BASE=<ref> overrides the diff base (smoke runs only).
# Base: the origin/main merge-base. The loop never merges main in; after an integration
# merge the base is origin/main again (the merge-base moves with it).
# Nothing runs on the owner's machine (owner rule: CI is the only gate): no pytest, no npm,
# no docker. The checks are greps, git and stdlib python; the tests are proven by test.yml on
# the pushed HEAD, whose required jobs must be present and green.
# No `pipefail` on purpose: with it, `! producer | grep -q X` can PASS when X matched (grep -q
# exits on the first match, the producer takes SIGPIPE, pipefail reports 141 and the `!`
# turns that into success). Negative greps read only the wave's ADDED code lines (tests and
# generated seeds excluded), so a pattern that exists elsewhere on the base never trips them.
set -u
exec </dev/null   # no check may ever wait on stdin (a grep with no file argument would)
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
case "${1:-}" in
  --print-base) echo "$BASE"; exit 0 ;;
  --no-ci) MODE=no-ci ;;
  --negative-only) MODE=negative ;;
  "") ;;
  *) echo "unknown argument: $1"; exit 2 ;;
esac
echo "base: $(git log -1 --format='%h %s' "$BASE") · HEAD: $(git log -1 --format='%h %s' HEAD) · mode: $MODE"
FAIL=0

# Each check body runs in a subshell: a `cd` or a variable inside one never leaks into the next.
check() {
  local name="$1" body="$2"
  echo ""
  echo "── $name"
  if ( eval "$body" ); then echo "   ✅ PASS: $name"; else echo "   ❌ FAIL: $name"; FAIL=1; fi
}

KIT=scripts/ralph/prd-255w1.json
MAN=orchestrator/reports/route-manifest.json
DOCS=orchestrator/modules/documents
BLOCKS=$DOCS/blocks
PALETTE=orchestrator/core/brand_palette.py
BUNDLE=orchestrator/core/media_render_bundle.py
BRAND_KIT=$DOCS/brand_kit.py
BRAND_API=orchestrator/api/document_brand_kit.py
RULES=orchestrator/services/brand_rules.py
TOOL=orchestrator/modules/tools/discovery/actions_brand_kit_update.py
XLSX=$DOCS/xlsx_render.py
FE_BRAND=frontend/components/deliverables/brand
LEGACY="$DOCS/templates/basic_report.html $DOCS/templates/invoice.html $DOCS/templates/executive_summary.html"
# The Python renderers whose added lines may not carry a raw colour or read primary_color.
RENDERER_PATHS="$BLOCKS $XLSX $DOCS/xlsx_letterhead.py $DOCS/letterhead.py $DOCS/legacy_jinja.py"

# ── helpers ──────────────────────────────────────────────────────────────────
# Added lines only, code only (tests, the generated Auto skill seed and docs excluded).
added_code_lines() {
  git diff -U0 "$BASE"..HEAD -- orchestrator frontend services \
      ':(exclude)orchestrator/tests' ':(exclude)frontend/**/__tests__/**' ':(exclude)frontend/**/*.test.ts' \
      ':(exclude)frontend/**/*.test.tsx' ':(exclude)orchestrator/core/seeds/platform-management-skill.md' \
    | grep -E '^\+[^+]'
}
# Added lines of the given paths only (code; tests excluded).
added_lines_in() {
  git diff -U0 "$BASE"..HEAD -- "$@" ':(exclude)orchestrator/tests' | grep -E '^\+[^+]'
}
added_files() { git diff --name-only --diff-filter=A "$BASE"..HEAD -- "$@"; }
# A file at HEAD has a regex (python `re`, MULTILINE); a missing file fails with a message.
file_has() {
  local f="$1" rx="$2"
  [ -f "$f" ] || { echo "   missing file: $f"; return 1; }
  python3 - "$f" "$rx" <<'PY' || { echo "   $f lacks /$rx/"; return 1; }
import re, sys
sys.exit(0 if re.search(sys.argv[2], open(sys.argv[1], encoding="utf-8").read(), re.M) else 1)
PY
}
# Some file under a directory (HEAD, tracked) has a regex.
tree_has() {
  local dir="$1" rx="$2"
  git grep -qE "$rx" -- "$dir" ':(exclude)orchestrator/tests' || { echo "   nothing under $dir matches /$rx/"; return 1; }
}
# A test file matching a glob exists and mentions each extra regex.
test_file_with() {
  local glob="$1"; shift
  local files; files=$(ls $glob 2>/dev/null | tr '\n' ' ')
  [ -n "$files" ] || { echo "   no test file $glob"; return 1; }
  local rx
  for rx in "$@"; do
    grep -qiE "$rx" $files || { echo "   $glob never mentions /$rx/"; return 1; }
  done
}
# A route (method + path) is in the COMMITTED manifest ({param} names vary, so match any).
manifest_has() {
  python3 - "$1" "$2" <<'PY2'
import json, re, sys
method, path = sys.argv[1], sys.argv[2]
rx = re.compile("^" + re.sub(r"\\\{[a-z_]+\\\}", r"\\{[a-z_]+\\}", re.escape(path)) + "$")
routes = json.load(open("orchestrator/reports/route-manifest.json"))["routes"]
if not any(r.get("method") == method and rx.match(r.get("path", "")) for r in routes):
    print(f"   route not in the committed manifest: {method} {path}")
    sys.exit(1)
PY2
}
# Every name is a field (``name:`` annotation) of some class in the given Python files.
fields_declared() {
  local files="$1"; shift
  python3 - "$files" "$@" <<'PY'
import ast, sys, glob
paths = [p for pat in sys.argv[1].split() for p in glob.glob(pat)]
fields = set()
for p in paths:
    for node in ast.walk(ast.parse(open(p, encoding="utf-8").read())):
        if isinstance(node, ast.ClassDef):
            for st in node.body:
                if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                    fields.add(st.target.id)
missing = [n for n in sys.argv[2:] if n not in fields]
for n in missing:
    print(f"   no model declares the field {n}")
sys.exit(1 if missing else 0)
PY
}
# A class (in one file) declares each field, itself or through a base class defined in
# orchestrator/modules/documents/*.py (a mixin of the new models module counts).
class_has_fields() {
  local f="$1" cls="$2"; shift 2
  python3 - "$f" "$cls" "$@" <<'PY'
import ast, glob, sys
classes = {}
for p in [sys.argv[1]] + glob.glob("orchestrator/modules/documents/*.py"):
    for n in ast.walk(ast.parse(open(p, encoding="utf-8").read())):
        if isinstance(n, ast.ClassDef):
            classes.setdefault(n.name, n)
if sys.argv[2] not in classes:
    print(f"   no class {sys.argv[2]} in {sys.argv[1]}"); sys.exit(1)
def fields(name, seen=()):
    node = classes.get(name)
    if node is None or name in seen:
        return set()
    own = {st.target.id for st in node.body if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name)}
    for b in node.bases:
        own |= fields(getattr(b, "id", getattr(b, "attr", "")), seen + (name,))
    return own
have = fields(sys.argv[2])
missing = [n for n in sys.argv[3:] if n not in have]
for n in missing:
    print(f"   {sys.argv[2]} lacks {n}")
sys.exit(1 if missing else 0)
PY
}
# A function's body (in one file) calls one of the given names.
function_uses() {
  local f="$1" fn="$2"; shift 2
  python3 - "$f" "$fn" "$@" <<'PY'
import ast, sys
tree = ast.parse(open(sys.argv[1], encoding="utf-8").read())
fn = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == sys.argv[2]), None)
if fn is None:
    print(f"   no function {sys.argv[2]} in {sys.argv[1]}"); sys.exit(1)
called = set()
for n in ast.walk(fn):
    if isinstance(n, ast.Call):
        f = n.func
        called.add(f.id if isinstance(f, ast.Name) else getattr(f, "attr", ""))
ok = any(name in called for name in sys.argv[3:])
if not ok:
    print(f"   {sys.argv[2]} calls none of {sys.argv[3:]}")
sys.exit(0 if ok else 1)
PY
}
# Every token-name constant the base's bundle and palette emitted is still defined at HEAD.
token_names_kept() {
  python3 - "$BASE" <<'PY'
import ast, subprocess, sys
base = sys.argv[1]
def literals(src, names):
    out = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
            for c in ast.walk(node.value):
                if isinstance(c, ast.Constant) and isinstance(c.value, str):
                    out.add(c.value)
    return out
def show(path, ref=None):
    if ref:
        return subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, text=True, check=True).stdout
    return open(path, encoding="utf-8").read()
bundle_names = {"COLOUR_TOKENS", "BODY_FONT_TOKEN", "HEADING_FONT_TOKEN"}
want = literals(show("orchestrator/core/media_render_bundle.py", base), bundle_names)
want = {w for w in want if not w.endswith("_color")}  # the kit field names, not token names
pal = show("orchestrator/core/brand_palette.py", base)
tree = ast.parse(pal)
for node in tree.body:
    if isinstance(node, ast.Assign) and isinstance(node.value, (ast.Constant, ast.Tuple)):
        for c in ast.walk(node.value):
            if isinstance(c, ast.Constant) and isinstance(c.value, str) and c.value.replace("-", "").isalpha() and c.value.islower():
                want.add(c.value)
head = show("orchestrator/core/media_render_bundle.py") + show("orchestrator/core/brand_palette.py")
missing = sorted(w for w in want if f'"{w}"' not in head and f"'{w}'" not in head)
for m in missing:
    print(f"   token name no longer defined: {m}")
sys.exit(1 if missing else 0)
PY
}
every_commit_signed() {
  local c
  for c in $(git rev-list "$BASE"..HEAD); do
    git show -s --format=%B "$c" | grep -q '^Signed-off-by:' || { echo "   unsigned: $c"; return 1; }
  done
}
# Every AC is DONE, or is the owner's (→ OWNER:) and does NOT claim DONE.
all_acs_done() {
  python3 - "$KIT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
bad = []
for s in d["userStories"]:
    for i, ac in enumerate(s["acceptanceCriteria"], 1):
        owner = "→ OWNER:" in ac
        if owner and "→ DONE" in ac:
            bad.append(f"{s['id']} AC{i} is the owner's check but is marked DONE")
        elif not owner and "DONE" not in ac:
            bad.append(f"{s['id']} AC{i} not DONE")
for b in bad[:30]:
    print("   " + b)
sys.exit(1 if bad else 0)
PY
}
# A new file is at most 800 lines (an error, as scripts/ci/check_changed_code_shape.py says);
# a file already over 800 that grows is a WARNING there and here (AGENTS.md: split it instead).
file_sizes_ok() {
  local f n b bad="" warn=""
  for f in $(git diff --name-only --diff-filter=AM "$BASE"..HEAD -- orchestrator frontend services | grep -E '\.(py|ts|tsx)$'); do
    [ -f "$f" ] || continue
    case "$f" in orchestrator/tests/*|*/__tests__/*|*.test.ts|*.test.tsx|orchestrator/alembic/*) continue ;; esac
    n=$(wc -l < "$f" | tr -d ' ')
    b=$(git show "$BASE:$f" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$b" -eq 0 ]; then
      [ "$n" -le 800 ] || bad="$bad $f(new, $n lines)"
    elif [ "$b" -gt 800 ] && [ "$n" -gt "$b" ]; then
      warn="$warn $f($b→$n)"
    fi
  done
  [ -z "$warn" ] || echo "   WARNING, grew past 800 (split instead where you can):$warn"
  [ -z "$bad" ] || { echo "   new file over 800 lines:$bad"; return 1; }
}
# F105: a route function added in orchestrator/api that is `async def` must await something.
no_async_route_without_await() {
  python3 - "$BASE" <<'PY'
import ast, re, subprocess, sys
base = sys.argv[1]
diff = subprocess.run(["git", "diff", "-U0", f"{base}..HEAD", "--", "orchestrator/api"], capture_output=True, text=True).stdout
added = {}
cur = None
for line in diff.splitlines():
    if line.startswith("+++ b/"):
        cur = line[6:]
    elif cur and line.startswith("+") and not line.startswith("+++"):
        m = re.match(r"\+\s*async def (\w+)", line)
        if m:
            added.setdefault(cur, set()).add(m.group(1))
bad = []
for path, names in added.items():
    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
    except FileNotFoundError:
        continue
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name in names:
            if not any(isinstance(n, (ast.Await, ast.AsyncFor, ast.AsyncWith)) for n in ast.walk(node)):
                bad.append(f"{path}:{node.lineno} async def {node.name} awaits nothing (F105: make it a plain def)")
for b in bad:
    print("   " + b)
sys.exit(1 if bad else 0)
PY
}
# The pushed HEAD's test.yml run must carry every required job, each green.
# Polls up to 40 min. A newer push cancels an older run, so only HEAD's own run counts.
# "Code standards on changed lines" is marked non-required on pull requests; this run holds
# its own changes to it (AGENTS.md → Code shape). On a PR run the two image lanes are
# path-gated by test.yml's `changes` job: skipped means the branch touches none of their
# paths. The gate job itself must be green, or a failed gate would pass as two skips.
REQUIRED_JOBS=("orchestrator-tests" "Alembic from-zero" "Schema-drift check" "Frontend CI" "Prod images built" "media-render" "Code standards on changed lines" "Which image lanes this change needs")
PATH_GATED_JOBS=("Prod images built" "media-render")
path_gated() { local g; for g in "${PATH_GATED_JOBS[@]}"; do [ "$g" = "$1" ] && return 0; done; return 1; }
ci_green_on_head() {
  command -v gh >/dev/null || { echo "   gh is required"; return 1; }
  local br sha remote waited=0 run="" status="" id jobs job hits bad=0
  br=$(git rev-parse --abbrev-ref HEAD); sha=$(git rev-parse HEAD)
  remote=$(git ls-remote origin "refs/heads/$br" | cut -f1)
  [ "$remote" = "$sha" ] || { echo "   HEAD $sha is not pushed to origin/$br (remote: ${remote:-none})"; return 1; }
  while :; do
    run=$(gh run list --branch "$br" --workflow test.yml --limit 20 --json databaseId,headSha,status \
          --jq ".[] | select(.headSha == \"$sha\") | \"\(.databaseId) \(.status)\"" 2>/dev/null | head -1)
    status="${run#* }"
    [ -n "$run" ] && [ "$status" = "completed" ] && break
    [ "$waited" -ge 2400 ] && { echo "   no completed test.yml run for $sha after 40 min (${run:-no run})"; return 1; }
    sleep 60; waited=$((waited + 60))
  done
  id="${run%% *}"
  jobs=$(gh run view "$id" --json jobs --jq '.jobs[] | "\(.conclusion)\t\(.name)"')
  echo "$jobs" | sed 's/^/   /'
  for job in "${REQUIRED_JOBS[@]}"; do
    hits=$(echo "$jobs" | grep -F "$(printf '\t')$job" || true)
    if [ -z "$hits" ]; then
      echo "   required job missing from run $id: $job"; bad=1
    elif path_gated "$job" && ! echo "$hits" | grep -qvE '^(success|skipped)'; then
      echo "$hits" | grep -q '^skipped' && echo "   $job: skipped (the branch touches none of its paths)"
    elif echo "$hits" | grep -qv '^success'; then
      echo "   required job not green: $job"; bad=1
    fi
  done
  [ "$bad" -eq 0 ] || return 1
  echo "   test.yml run $id is green on $sha for all ${#REQUIRED_JOBS[@]} required jobs (path-gated lanes: green or skipped)"
}

# ── regex-heavy checks as functions (no quoting through eval) ────────────────
reads_primary_color() { grep -E "get\(['\"]primary_color|\[['\"]primary_color['\"]\]|\.primary_color\b|brand\.primary_color"; }
n3_no_primary_in_renderers() { ! added_lines_in $RENDERER_PATHS | reads_primary_color | grep -q .; }
n4_no_raw_colour_in_renderers() {
  ! added_lines_in $RENDERER_PATHS | grep -vE '^\+\s*#' | grep -vE '^\+\s*[A-Z][A-Z0-9_]*\s*(:[^=]+)?=' \
    | grep -qE "([=(,:{[]|\b(or|return))\s*['\"]#[0-9a-fA-F]{3}([0-9a-fA-F]{3})?['\"]"
}
n2_no_env_reads() {
  ! git diff -U0 "$BASE"..HEAD -- orchestrator ':(exclude)orchestrator/config.py' ':(exclude)orchestrator/tests' \
      ':(exclude)orchestrator/conftest.py' ':(exclude)**/conftest.py' \
    | grep -E '^\+[^+]' | grep -vE '^\+\s*#' | grep -qE 'os\.(getenv|environ)'
}
n7_no_raw_fetch() { ! added_code_lines | grep -qE "fetch\(\s*['\"\`]/api"; }
n14_no_type_presets() {
  ! added_lines_in orchestrator/modules/documents orchestrator/api orchestrator/services/brand_rules.py \
    | grep -qiE "['\"](editorial|compact)['\"]"
}
n15_no_reseed_job() {
  [ -z "$(git diff --name-only --diff-filter=A "$BASE"..HEAD | grep -iE 're_?seed')" ] \
    && ! added_lines_in orchestrator/modules/documents orchestrator/core/seeds orchestrator/services | grep -qE 'add_job\('
}
n17_accent_never_bold_by_default() {
  ! added_code_lines | grep -qE "accent_use.{0,40}[^=!<>]=\s*['\"]bold['\"]|['\"]accent_use['\"]\s*:\s*['\"]bold['\"]|accent_use=['\"]bold['\"]"
}
n18_no_generated_logo() {
  ! added_lines_in orchestrator/modules/documents orchestrator/api/document_brand_kit.py \
    | grep -qiE 'generate_image|images\.generate|dall-?e|stable.?diffusion|higgsfield'
}
accent_defaults_to_sparing() { git grep -qE "accent_use[[:space:]]*:.*(=[[:space:]]*|default=)['\"]sparing['\"]" -- "$DOCS"; }
blocks_never_read_primary() { ! git grep -nE "get\(['\"]primary_color|\[['\"]primary_color['\"]\]" -- "$BLOCKS" | grep -q .; }
xlsx_on_tokens() {
  file_has "$XLSX" 'surface_2' && file_has "$XLSX" 'type_scale|body_size|body_pt' && file_has "$XLSX" 'currency' \
    && ! grep -qE "get\(['\"]primary['\"]\)|\[['\"]primary['\"]\]" "$XLSX"
}
parity_test_extended() {
  ! git diff --quiet "$BASE"..HEAD -- orchestrator/tests/test_prd251w1_brand_kit_tools.py \
    || test_file_with 'orchestrator/tests/test_prd255w1_*tools*.py' 'PATCH_FIELDS|model_fields'
}
tool_schema_has_v2() {
  local n
  for n in palette accent_use type_scale spacing_unit_pt page_margin_mm logo_rules currency date_style; do
    grep -q "\"$n\"" "$TOOL" || { echo "   $TOOL lacks $n"; return 1; }
  done
}
legacy_on_tokens() {
  local f
  for f in $LEGACY; do
    [ -f "$f" ] || { echo "   missing $f"; return 1; }
    grep -q 'brand\.palette\.' "$f" && grep -q 'brand\.type\.' "$f" || { echo "   $f lacks brand.palette / brand.type"; return 1; }
    ! grep -q 'brand\.primary_color' "$f" || { echo "   $f still reads brand.primary_color"; return 1; }
  done
}
variant_routes_in_manifest() {
  local v m
  for v in logo-dark logo-mono; do for m in POST GET DELETE; do manifest_has "$m" "/api/documents/brand-kit/$v" || return 1; done; done
}
page_sections() {
  local t
  for t in 'Colours' 'Type' 'Spacing' 'Logo variants' 'Locale' 'Voice' 'Reset to derived'; do
    git grep -qF "$t" -- "$FE_BRAND" ":(exclude)$FE_BRAND/__tests__" || { echo "   no '$t' in $FE_BRAND"; return 1; }
  done
}

# ═════ NEGATIVE checks: scope and conventions (each passes on the base itself) ═════
check "N1 no Alembic revision (the kit is JSON; FR-2)" \
  "[ -z \"\$(added_files orchestrator/alembic/versions)\" ] || { echo \"   added: \$(added_files orchestrator/alembic/versions | tr '\n' ' ')\"; exit 1; }"
check "N2 no env reads added outside config.py (tests and conftest excluded)" "n2_no_env_reads"
check "N3 no renderer reads primary_color in an added line (FR-3)" "n3_no_primary_in_renderers"
check "N4 no raw colour literal added to a Python renderer outside a named constant" "n4_no_raw_colour_in_renderers"
check "N5 one contrast implementation: no new luminance/contrast function outside core/brand_palette.py" \
  "! git diff -U0 $BASE..HEAD -- orchestrator frontend ':(exclude)orchestrator/core/brand_palette.py' ':(exclude)orchestrator/tests' ':(exclude)frontend/**/__tests__/**' | grep -E '^\+[^+]' | grep -qE 'def (_?relative_luminance|_?luminance|_?contrast_ratio|_?wcag_contrast)\b'"
check "N6 no print() added in orchestrator code" \
  "! git diff -U0 $BASE..HEAD -- orchestrator ':(exclude)orchestrator/tests' ':(exclude)orchestrator/scripts' | grep -qE '^\+\s*print\('"
check "N7 no raw fetch('/api…') added in the frontend" "n7_no_raw_fetch"
check "N8 no new Python dependency (WeasyPrint stays <70)" "git diff --quiet $BASE..HEAD -- orchestrator/requirements.txt"
check "N9 no new npm dependency" "git diff --quiet $BASE..HEAD -- frontend/package.json"
check "N10 the Composio deny list is untouched" "git diff --quiet $BASE..HEAD -- orchestrator/core/composio/deny_list.py"
check "N11 the generated Auto skill seed is untouched" \
  "git diff --quiet $BASE..HEAD -- orchestrator/core/seeds/platform-management-skill.md"
check "N12 no node_modules in the diff" "! git diff --name-only $BASE..HEAD | grep -q node_modules"
check "N13 every commit is DCO-signed" "every_commit_signed"
check "N14 one default type scale: no presets (Decision Q5)" "n14_no_type_presets"
check "N15 no hosted re-seed job (Decision Q2)" "n15_no_reseed_job"
check "N16 no onboarding change (Decision Q4)" \
  "[ -z \"\$(git diff --name-only $BASE..HEAD | grep -i onboarding | grep -vE '^orchestrator/tests/|__tests__')\" ]"
check "N17 accent_use is never defaulted or backfilled to 'bold' (Decision Q1)" "n17_accent_never_bold_by_default"
check "N18 no generated or altered logo (FR-9, non-goals)" "n18_no_generated_logo"
check "N19 the logo-variant files stay server-managed: the tool never sets them" \
  "! grep -qE 'logo_dark_path|logo_mono_path' $TOOL"
check "N20 file sizes: a new file ≤ 800 lines (growth past 800 warns)" "file_sizes_ok"
check "N21 F105: no added async route that awaits nothing" "no_async_route_without_await"
check "N22 the hierarchy gate passes (stdlib ast)" "python3 orchestrator/scripts/check_hierarchy_gate.py >/dev/null"

if [ "$MODE" != "negative" ]; then
# ═════ US-003 every kit derived ═══════════════════════════════════════════════
check "US-003 derive_palette and effective_palette in core/brand_palette.py, exported" \
  "file_has $PALETTE '^def derive_palette\\(' && file_has $PALETTE '^def effective_palette\\(' && file_has $PALETTE '\"derive_palette\"'"
check "US-003 derive_palette reuses the contrast search (_least/_most)" "function_uses $PALETTE derive_palette _least _most"
check "US-003 tests: the Automatos, Harbourline, dark-only and light-only kits" \
  "test_file_with 'orchestrator/tests/test_prd255w1_*palette*.py' 'c44a1a' '1e3a5f' 'c26a2e' 'dark' 'light'"

# ═════ US-001 colour roles ════════════════════════════════════════════════════
check "US-001 the nine roles are model fields" \
  "fields_declared '$DOCS/*.py' ink heading paper surface surface_2 accent accent_2 muted rule palette accent_use"
check "US-001 accent_use defaults to sparing" "accent_defaults_to_sparing"
check "US-001 BrandKit and BrandKitPatch carry palette and accent_use" \
  "class_has_fields $BRAND_KIT BrandKit palette accent_use && class_has_fields $BRAND_KIT BrandKitPatch palette accent_use"
check "US-001 GET says set or derived (palette_source)" "git grep -q 'palette_source' -- $BRAND_API $DOCS"
check "US-001 tests for validation and contrast" "test_file_with 'orchestrator/tests/test_prd255w1_*.py' 'palette_source' '422'"

# ═════ US-002 type, spacing, logo rules, variants, locale, tone ══════════════
check "US-002 type_scale and its seven steps of {size_pt, line_pt, weight}" \
  "fields_declared '$DOCS/*.py' type_scale display h1 h2 h3 body small caption size_pt line_pt weight"
check "US-002 spacing, margins, logo rules, variants, locale, tone meanings" \
  "fields_declared '$DOCS/*.py' spacing_unit_pt page_margin_mm logo_rules letterhead_mm clear_space min_mm logo_dark_path logo_mono_path currency date_style meaning"
check "US-002 both date styles" "git grep -qF 'd MMMM yyyy' -- $DOCS && git grep -qF 'MMMM d, yyyy' -- $DOCS"
check "US-002 the variants are server-managed" \
  "file_has $BRAND_KIT 'SERVER_MANAGED_FIELDS[^\\n]*logo_dark_path' && file_has $BRAND_KIT 'SERVER_MANAGED_FIELDS[^\\n]*logo_mono_path'"
check "US-002 routes: POST/GET/DELETE /brand-kit/logo-dark and /brand-kit/logo-mono in the committed manifest" "variant_routes_in_manifest"
check "US-002 tests for each field's validation and defaults" \
  "test_file_with 'orchestrator/tests/test_prd255w1_*.py' 'type_scale' 'spacing_unit_pt' 'logo_rules' 'currency' 'date_style' 'meaning'"

# ═════ US-004 documents ══════════════════════════════════════════════════════
check "US-004 no block renderer reads primary_color any more (FR-3)" "blocks_never_read_primary"
check "US-004 the block renderers read the roles and the type scale" \
  "tree_has $BLOCKS 'surface_2' && tree_has $BLOCKS 'type_scale' && tree_has $BLOCKS 'page_margin_mm|spacing_unit_pt' && tree_has $BLOCKS 'letterhead_mm'"
check "US-004 DOCX header and footer with page numbers" \
  "git grep -qE 'NUMPAGES|instrText|fldSimple|PAGE' -- $BLOCKS/docx_*.py && git grep -qE '\\.footer|footer' -- $BLOCKS/docx_*.py"
check "US-004 the legacy templates read brand.palette and brand.type, not brand.primary_color" "legacy_on_tokens"
check "US-004 currency and date_style reach the page" \
  "git grep -q 'currency' -- $DOCS/amounts.py && git grep -q 'date_style' -- $DOCS/variables"
check "US-004 a brand.sign_off chip, used by the Branded Letter" \
  "git grep -q 'brand.sign_off' -- $DOCS/variables/catalog.py && git grep -q 'brand.sign_off' -- $DOCS/presets.py"
check "US-004 render tests read real pages back (accent share ≤ 15%)" \
  "test_file_with 'orchestrator/tests/test_prd255w1_*document*.py' 'pdf_first_page_png|pypdfium2' 'ACCENT_SHARE_MAX|0\\.15' 'heading'"

# ═════ US-005 spreadsheets ═══════════════════════════════════════════════════
check "US-005 the sheet reads surface_2, the body type and the currency, never the raw primary" "xlsx_on_tokens"
check "US-005 test: header fill, font, currency format" \
  "test_file_with 'orchestrator/tests/test_prd255w1_*xlsx*.py' 'fill' 'font' 'number_format|currency'"

# ═════ US-006 socials ════════════════════════════════════════════════════════
check "US-006 brand_tokens reads the v2 roles and the type scale, and the dark logo" \
  "file_has $BUNDLE 'effective_palette|palette' && file_has $BUNDLE 'type_scale' && file_has $BUNDLE 'logo_dark'"
check "US-006 every token name of the base is still emitted" "token_names_kept"
check "US-006 the previews render with both kits" \
  "grep -qi '1e3a5f' scripts/ci/social_template_previews.py && grep -qi 'c44a1a' scripts/ci/social_template_previews.py"
check "US-006 tests: both kits, sparing keeps the accent off text-heavy backgrounds" \
  "test_file_with 'orchestrator/tests/test_prd255w1_*social*.py' 'sparing' '1e3a5f|harbourline' 'c44a1a|automatos'"

# ═════ US-008 agents ═════════════════════════════════════════════════════════
check "US-008 the update tool's schema carries every v2 field" "tool_schema_has_v2"
check "US-008 the rules block carries roles, type scale, currency, date style, logo variants" \
  "file_has $RULES 'surface_2|heading' && file_has $RULES 'type_scale' && file_has $RULES 'currency' && file_has $RULES 'date_style' && file_has $RULES 'logo_dark'"
check "US-008 the schema-parity test is extended" "parity_test_extended"
check "US-008 tests: the action round-trips every field; the rules block lists the roles" \
  "test_file_with 'orchestrator/tests/test_prd255w1_*.py' 'platform_update_brand_kit' 'rules_for_kit'"

# ═════ US-007 the page ═══════════════════════════════════════════════════════
check "US-007 the Brand kit tab's sections and Reset to derived" "page_sections"
check "US-007 the page uploads the dark and mono variants" \
  "git grep -q 'brand-kit/logo-dark' -- frontend ':(exclude)**/__tests__/**' && git grep -q 'brand-kit/logo-mono' -- frontend ':(exclude)**/__tests__/**'"
check "US-007 vitest for the sections (at least two new brand test files)" \
  "[ \$(added_files $FE_BRAND | grep -cE '\\.test\\.tsx?\$') -ge 2 ]"

# ═════ the contract ═══════════════════════════════════════════════════════════
check "every acceptance criterion in $KIT is DONE (the owner's are left to the owner)" "all_acs_done"
fi

# ── tests: proven by CI on the pushed HEAD, never run here ───────────────────
if [ "$MODE" = "full" ]; then
  check "CI: the required jobs present and green on HEAD" "ci_green_on_head"
fi

echo ""
if [ "$FAIL" -eq 0 ]; then echo "✅ PRD-255 Wave 1 acceptance ($MODE): PASS"; else echo "❌ PRD-255 Wave 1 acceptance ($MODE): FAIL"; fi
exit "$FAIL"
