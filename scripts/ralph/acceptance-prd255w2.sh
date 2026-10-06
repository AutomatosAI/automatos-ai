#!/bin/bash
# Acceptance gate — PRD-255 Brand Kit v2, Wave 2: the brand board, render_preview, the template tools, the Brand
# designer, Auto's brand routing (US-009..US-014).
#   bash scripts/ralph/acceptance-prd255w2.sh                  run the gate
#   bash scripts/ralph/acceptance-prd255w2.sh --print-base     print the diff base
#   bash scripts/ralph/acceptance-prd255w2.sh --no-ci          every check but CI (a smoke run, never the loop's gate)
#   bash scripts/ralph/acceptance-prd255w2.sh --negative-only  only the scope/convention (negative) checks: each
#                                                              must pass on the base itself (smoke-run them there)
#   GATE_BASE=<ref> overrides the diff base (smoke runs only).
# Base: stacked on Wave 1 (feat/prd-255-w1-brand-kit-v2) until Wave 1 is on main, then main.
# The base is the NEWER of the two fork points, so neither side's changes read as this
# run's. The loop never merges either branch in.
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
  PREV_REF=origin/feat/prd-255-w1-brand-kit-v2
  if git rev-parse -q --verify "$PREV_REF" >/dev/null; then
    PREV_BASE=$(git merge-base HEAD "$PREV_REF" 2>/dev/null || true)
    if [ -n "$PREV_BASE" ] && [ "$PREV_BASE" != "$BASE" ] && git merge-base --is-ancestor "$BASE" "$PREV_BASE"; then
      BASE="$PREV_BASE"
    fi
  fi
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

KIT=scripts/ralph/prd-255w2.json
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
DISCOVERY=orchestrator/modules/tools/discovery
GROUPS=orchestrator/services/session_tool_groups.py
GATE_PY=orchestrator/scripts/check_hierarchy_gate.py
SEEDS=orchestrator/core/seeds
CHAT=orchestrator/consumers/chatbot
SOCIAL=$DOCS/templates/social
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

n24_chat_never_writes_kit() {
  ! added_lines_in "$CHAT" | grep -vE '^\+\s*#' | grep -qE '(update_brand_kit|save_brand_kit|validate_brand_kit)\('
}
one_persona_source() {
  local n; n=$(git grep -cE '^_?BRAND_DESIGNER_PERSONA[[:space:]]*=' -- orchestrator ':(exclude)orchestrator/tests' | awk -F: '{s+=$NF} END {print s+0}')
  [ "$n" -eq 1 ] || { echo "   BRAND_DESIGNER_PERSONA is defined $n times (one source)"; return 1; }
}
# An action is registered as registry.register(ActionDefinition(name=..., permission_level=...)) under discovery/.
action_registered() {
  python3 - "$1" "$2" <<'PY'
import ast, glob, sys
want, level = sys.argv[1], sys.argv[2]
for p in glob.glob("orchestrator/modules/tools/discovery/**/*.py", recursive=True):
    for node in ast.walk(ast.parse(open(p, encoding="utf-8").read())):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "register" and node.args):
            continue
        arg = node.args[0]
        if not (isinstance(arg, ast.Call) and getattr(arg.func, "id", getattr(arg.func, "attr", "")) == "ActionDefinition"):
            continue
        kw = {k.arg: k.value for k in arg.keywords}
        name = kw.get("name")
        if isinstance(name, ast.Constant) and name.value == want:
            lv = kw.get("permission_level")
            got = lv.value if isinstance(lv, ast.Constant) else None
            if got != level:
                print(f"   {want} is registered with permission_level {got!r}, not {level!r} ({p})"); sys.exit(1)
            sys.exit(0)
print(f"   {want} is not registered via registry.register(ActionDefinition(...)) under discovery/")
sys.exit(1)
PY
}
social_board_sizes() {
  local j; j=$(ls $SOCIAL/brand-board*.json 2>/dev/null | head -1)
  [ -n "$j" ] || { echo "   no $SOCIAL/brand-board*.json"; return 1; }
  grep -q '"social_image"' "$j" && grep -q '1080x1350' "$j" && grep -q '1080x1920' "$j" \
    || { echo "   $j is not a social_image at 1080x1350 and 1080x1920"; return 1; }
  ls $SOCIAL/brand-board*.html >/dev/null 2>&1 || { echo "   no $SOCIAL/brand-board*.html"; return 1; }
}
designer_seeded() {
  local files; files=$(git grep -liE 'brand designer' -- "$SEEDS" ':(exclude)orchestrator/tests' | tr '\n' ' ')
  [ -n "$files" ] || { echo "   no seed under $SEEDS names the Brand designer"; return 1; }
  grep -qE "['\"]cli['\"]" $files && grep -qE "['\"]claude['\"]" $files && grep -q 'CLI_RUNTIME_ENABLED' $files \
    || { echo "   the designer's seed does not set runtime cli / provider claude by CLI_RUNTIME_ENABLED"; return 1; }
}
designer_instructions() {
  local rx files; files=$(git grep -liE 'brand designer' -- "$SEEDS" ':(exclude)orchestrator/tests' | tr '\n' ' ')
  [ -n "$files" ] || { echo "   no designer seed"; return 1; }
  for rx in 'logo' 'sparing' 'platform_update_brand_kit' 'approv' 'render_preview'; do
    grep -qiE "$rx" $files || { echo "   the designer's instructions never say /$rx/"; return 1; }
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

check "N23 no template delete tool" "! git grep -qE 'name=\"platform_delete_template\"' -- $DISCOVERY"
check "N24 Auto's chat path never writes the kit (Auto delegates; FR-11)" "n24_chat_never_writes_kit"
check "N25 one Brand Designer persona source" "one_persona_source"

if [ "$MODE" != "negative" ]; then
# ═════ US-009 the brand board ════════════════════════════════════════════════
check "US-009 a 'Brand Board' block starter in the brand category" \
  "git grep -qF 'Brand Board' -- $DOCS/presets.py $DOCS/seed_templates.py $DOCS/blocks && git grep -qE \"['\\\"]brand['\\\"]\" -- $DOCS/presets.py $DOCS/seed_templates.py"
check "US-009 the social 'Brand board' at 4:5 and 9:16" "social_board_sizes"
check "US-009 render test: the Automatos kit, one A4 page" \
  "test_file_with 'orchestrator/tests/test_prd255w2_*board*.py' 'c44a1a|automatos' 'page|pages' 'a4|595|842|210'"

# ═════ US-010 the board on the page ══════════════════════════════════════════
check "US-010 the board route is in the committed manifest" "manifest_has GET /api/documents/brand-kit/board"
check "US-010 the page shows the board and both downloads" \
  "for t in 'Brand board' 'Download PDF' 'Download PNG'; do git grep -qiF \"\$t\" -- $FE_BRAND ':(exclude)$FE_BRAND/__tests__' || { echo \"   no '\$t' in $FE_BRAND\"; exit 1; }; done && git grep -q 'brand-kit/board' -- frontend ':(exclude)**/__tests__/**'"
check "US-010 vitest for the board" "[ \$(added_files $FE_BRAND | grep -ciE 'board.*\\.test\\.tsx?\$') -ge 1 ]"

# ═════ US-012 render_preview ═════════════════════════════════════════════════
check "US-012 platform_render_preview is registered, read-only" "action_registered platform_render_preview read"
check "US-012 render_preview is a session tool" "grep -q 'render_preview' $GROUPS"
check "US-012 tests: the PNG in the ticket's folder, another workspace refused" \
  "test_file_with 'orchestrator/tests/test_prd255w2_*render_preview*.py' 'sessions/' 'workspace' 'png'"

# ═════ US-013 the template tools ═════════════════════════════════════════════
check "US-013 platform_create_template and platform_update_template are registered writes" \
  "action_registered platform_create_template write && action_registered platform_update_template write"
check "US-013 both are session tools" "grep -q 'create_template' $GROUPS && grep -q 'update_template' $GROUPS"
check "US-013 made-by tags, starters refused, social formats refused" \
  "git grep -q 'made-by:' -- $DISCOVERY $DOCS ':(exclude)orchestrator/tests' && git grep -q 'STARTER_CREATOR' -- $DISCOVERY && git grep -qE 'is_social_format|social_image|SOCIAL_FORMATS' -- $DISCOVERY"
check "US-013 tests: create, edit, a starter refused, validation errors passed back" \
  "test_file_with 'orchestrator/tests/test_prd255w2_*template*.py' 'create' 'update' 'starter' 'valid'"

# ═════ US-011 the Brand designer ═════════════════════════════════════════════
check "US-011 the designer is seeded per workspace, cli + claude where sessions run" "designer_seeded"
check "US-011 its instructions carry every rule" "designer_instructions"
check "US-011 tests: seeded once, the runtime by edition, no duplicate" \
  "test_file_with 'orchestrator/tests/test_prd255w2_*designer*.py' 'cli' 'claude' 'idempot|once|duplicate'"

# ═════ US-014 Auto's routing and the flow ════════════════════════════════════
check "US-014 a brand ask goes to the Brand designer (a chat note)" \
  "git grep -qiE 'brand designer' -- $CHAT ':(exclude)orchestrator/tests'"
check "US-014 platform_propose_brand_kit files the card (registered, a session tool)" \
  "action_registered platform_propose_brand_kit write && grep -q 'propose_brand_kit' $GROUPS"
check "US-014 tests: the routing test and the card carries the proposal and the board" \
  "test_file_with 'orchestrator/tests/test_prd255w2_*routing*.py' 'brand' 'yourself' && test_file_with 'orchestrator/tests/test_prd255w2_*card*.py' 'proposal' 'board' 'Approve'"

# ═════ the contract ═══════════════════════════════════════════════════════════
check "every acceptance criterion in $KIT is DONE (the owner's are left to the owner)" "all_acs_done"
fi

# ── tests: proven by CI on the pushed HEAD, never run here ───────────────────
if [ "$MODE" = "full" ]; then
  check "CI: the required jobs present and green on HEAD" "ci_green_on_head"
fi

echo ""
if [ "$FAIL" -eq 0 ]; then echo "✅ PRD-255 Wave 2 acceptance ($MODE): PASS"; else echo "❌ PRD-255 Wave 2 acceptance ($MODE): FAIL"; fi
exit "$FAIL"
