#!/bin/bash
# Acceptance gate — PRD-260 Play, a mascot theme in Automatos (one wave, T-001..T-010; frontend only).
#   bash scripts/ralph/acceptance-prd260.sh | --print-base | --no-ci | --negative-only ; GATE_BASE=<ref> (smoke runs only).
# Base: the merge-base with origin/main (the wave is cut from main). Nothing runs on the owner's machine: greps, git
# and stdlib python only; CI once per wave on the draft PR. No pipefail (a negative `! producer | grep -q` would
# invert on SIGPIPE); every check runs in its own `( eval … )`; stdin is closed so a grep with a missing file
# argument fails instead of hanging.
#
# Import checks read import SYNTAX, never raw text: a comment-stripping scanner pulls the module specifiers out of
# each changed .ts/.tsx/.js file (14 Oct: a docstring saying 'MinIO' failed PRD-259's text-grep guard).
#
# Negative checks on the base (14 Oct 2026, origin/main d7affad60):
#   - every "changed/added" check (N3–N9, N11–N20) reads `git diff BASE..HEAD`, EMPTY when HEAD is the base, so it
#     passes there by construction; the added-line greps are scoped to frontend code and exclude tests, the kit
#     (scripts/ralph) and docs, so a check's own pattern line or a test asserting absence never trips it;
#   - N2 (exactly one Alembic head) was smoke-run: its stdlib-ast parser over the base tree's
#     orchestrator/alembic/versions finds one head; N10 (api-client.ts not grown: 3,429 lines on the base),
#     N16 (globals.css grows by at most the two cherry-picked --radius lines) and N17 (the appearance defaults
#     version unchanged) compare HEAD with the base and pass on it — smoke-checked by reading;
#   - the story checks and the cherry-pick and clip checks are POSITIVE (the wave's work): they fail on the base by
#     design (no 'play' style, no frontend/public/auto, no cherry-picked commit there, read 14 Oct).
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

KIT=scripts/ralph/prd-260.json
FE=frontend
LOOKS=frontend/lib/looks
MASCOT=frontend/lib/mascot
PLAY=frontend/components/play
VERSIONS=orchestrator/alembic/versions
CLIPS=frontend/public/auto
STATES="hello idle listening thinking working searching reading celebrate oops sleeping lost loading"
PICKS="cc08db61c1c883521a4ee6528af0bc0757a5aea5:frontend/components/__tests__/prd259-radius.test.ts
ad2d3fee7bd02ac7f25d967089ddc2d73057ef0f:frontend/lib/__tests__/tailwind-dark-variant.test.ts
15c70c94dc096f99cd0a32be8d2faecd6b799898:frontend/lib/__tests__/dark-variant-tokens.test.ts"

# The wave's ADDED lines under the given paths: tests, the kit and docs excluded (negative greps only).
added_code() { git diff "$BASE"..HEAD -- "$@" ':!*__tests__*' ':!*.test.ts' ':!*.test.tsx' ':!scripts/ralph' ':!docs' | grep '^+' | grep -v '^+++'; }
added_files() { git diff --name-only --diff-filter=A "$BASE"..HEAD -- "$@"; }
changed_files() { git diff --name-only --diff-filter=AM "$BASE"..HEAD -- "$@"; }
lines_at() { git show "$1:$2" 2>/dev/null | wc -l | tr -d ' '; }
has() { [ -f "$2" ] && grep -qE -e "$1" "$2"; }      # has <ERE> <file>: false (never a stdin read) when the file is missing
all_exist() { local f; for f in "$@"; do [ -f "$f" ] || { echo "   missing: $f"; return 1; }; done; }

raw_fetch_added()   { added_code frontend | grep -E "fetch\(\s*['\"\`]/api" | grep -q .; }
orange_added()      { added_code frontend | grep -E '\b(bg|text|border|ring|from|to|via|fill|stroke)-orange-[0-9]' | grep -q .; }
inner_html_added()  { added_code frontend ':!frontend/app/layout.tsx' | grep -E 'dangerouslySetInnerHTML|\.innerHTML\s*=|\.outerHTML\s*=|insertAdjacentHTML|document\.write\(' | grep -q .; }
# app/layout.tsx may add dangerouslySetInnerHTML only as the pre-paint constant.
layout_html_ok() { ! git diff "$BASE"..HEAD -- frontend/app/layout.tsx | grep '^+' | grep 'dangerouslySetInnerHTML' | grep -vqE 'dangerouslySetInnerHTML=\{\{\s*__html:\s*LOOK_PREPAINT_SCRIPT\s*\}\}'; }
every_commit_signed() { [ -z "$(git log --format='%H %(trailers:key=Signed-off-by,valueonly)' "$BASE"..HEAD | awk 'NF<2{print $1}')" ]; }
file_sizes_ok() { local f n bad=0; for f in $(added_files frontend docs | grep -E '\.(ts|tsx|js|jsx|mjs|css|md|json)$'); do [ -f "$f" ] || continue; n=$(wc -l < "$f" | tr -d ' '); [ "$n" -le 800 ] || { echo "   $f: $n lines"; bad=1; }; done; [ $bad -eq 0 ]; }
not_grown() { local f="$1" slack="${2:-0}" b h; b=$(lines_at "$BASE" "$f"); h=$(wc -l < "$f" | tr -d ' '); echo "   $f: base $b, HEAD $h (allowed +$slack)"; [ "$h" -le $((b + slack)) ]; }
tests_not_weakened() { local d a; d=$(git diff "$BASE"..HEAD -- frontend orchestrator | grep -cE '^-\s*(async )?def test_|^-\s*(it|test)\('); a=$(git diff "$BASE"..HEAD -- frontend orchestrator | grep -cE '^\+\s*(async )?def test_|^\+\s*(it|test)\('); echo "   test functions removed $d, added $a"; [ "$a" -ge "$d" ]; }
defaults_version_same() { local re="APPEARANCE_DEFAULTS_VERSION *= *'[^']*'"; [ "$(git show "$BASE:frontend/lib/ui-style.ts" | grep -oE "$re")" = "$(grep -oE "$re" frontend/lib/ui-style.ts)" ]; }
radius_slack() { if git cat-file -e "$BASE:frontend/components/__tests__/prd259-radius.test.ts" 2>/dev/null; then echo 0; else echo 2; fi; }

# The module specifiers of every changed JS/TS file in the frontend, comments and string contents outside
# import syntax ignored. Prints "<file>\t<specifier>" lines.
import_specifiers() { python3 - "$BASE" <<'PY'
import re, subprocess, sys
base = sys.argv[1]
files = subprocess.run(["git", "diff", "--name-only", "--diff-filter=AM", f"{base}..HEAD", "--", "frontend"],
                       capture_output=True, text=True, check=True).stdout.split()
def strip_comments(src: str) -> str:
    out, i, n, quote = [], 0, len(src), None
    while i < n:
        c = src[i]
        if quote:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(src[i + 1]); i += 2; continue
            if c == quote:
                quote = None
            i += 1; continue
        if c in "'\"`":
            quote = c; out.append(c); i += 1; continue
        if src.startswith("//", i):
            j = src.find("\n", i); i = n if j < 0 else j; continue
        if src.startswith("/*", i):
            j = src.find("*/", i + 2); i = n if j < 0 else j + 2; out.append(" "); continue
        out.append(c); i += 1
    return "".join(out)
SPEC = re.compile(r"""(?:\bimport\s*\(\s*|\brequire\s*\(\s*|\bfrom\s+|^\s*import\s+|\bexport\s+\*\s+from\s+)(['"])([^'"\n]+)\1""", re.M)
for f in files:
    if not re.search(r"\.(ts|tsx|js|jsx|mjs|cjs)$", f):
        continue
    try:
        src = open(f, encoding="utf-8").read()
    except OSError:
        continue
    for m in SPEC.finditer(strip_comments(src)):
        print(f"{f}\t{m.group(2)}")
PY
}
forbidden_import() { import_specifiers | grep -E $'\t''.*(prd260-spec|play-ref|automatos-play|scripts/ralph)' ; }
# A file that IMPORTS next/font/google (import syntax, via the scanner) may keep the import lines it had on the
# base, never add or change one: Play's type is vendored woff2 through next/font/local (the Turbopack trap).
google_font_added() {
  local f l bad=0
  for f in $(import_specifiers | grep -F "$(printf '\tnext/font/google')" | cut -f1 | sort -u); do
    while IFS= read -r l; do
      git show "$BASE:$f" 2>/dev/null | grep -qxF -- "$l" || { echo "   $f: $l"; bad=1; }
    done < <(grep -E "^[[:space:]]*(import[[:space:]].*|[}].*)from[[:space:]]+['\"]next/font/google['\"]" "$f")
  done
  [ $bad -eq 1 ]
}
# A Python star import added anywhere (ruff F403), read with ast.
star_import_added() { python3 - "$BASE" <<'PY'
import ast, subprocess, sys
base = sys.argv[1]
files = subprocess.run(["git", "diff", "--name-only", "--diff-filter=AM", f"{base}..HEAD", "--", "*.py"],
                       capture_output=True, text=True, check=True).stdout.split()
hits = []
for f in files:
    try:
        tree = ast.parse(open(f, encoding="utf-8").read())
    except (OSError, SyntaxError):
        continue
    hits += [f"{f}:{n.lineno}" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names)]
print("\n".join(f"   star import: {h}" for h in hits)); sys.exit(0 if hits else 1)
PY
}
# Play's CSS defines and reads only --pl-* variables (decision K2: Play's --accent and --ring would clobber
# Studio's HSL tokens of the same names), plus the core's font and radius variables.
play_css_vars_ok() { python3 - "$BASE" <<'PY'
import re, subprocess, sys
base = sys.argv[1]
files = [f for f in subprocess.run(["git", "diff", "--name-only", "--diff-filter=AM", f"{base}..HEAD", "--", "frontend/*.css"],
         capture_output=True, text=True, check=True).stdout.split() if f != "frontend/app/globals.css"]
ok = re.compile(r"^--(pl-[\w-]+|font-geist-sans|font-geist-mono|font-newsreader|radius)$")
bad = []
for f in files:
    css = re.sub(r"/\*.*?\*/", " ", open(f, encoding="utf-8").read(), flags=re.S)
    for name in re.findall(r"(?<![\w-])(--[\w-]+)\s*:", css) + re.findall(r"var\(\s*(--[\w-]+)", css):
        if not ok.match(name):
            bad.append(f"{f}: {name}")
print("\n".join(f"   {b}" for b in sorted(set(bad))[:40])); sys.exit(1 if bad else 0)
PY
}

# --- PRD-260 §5: the three cherry-picks; §3: the clips --------------------------------------
picks_present() {
  local line sha marker bad=0
  while IFS= read -r line; do
    sha="${line%%:*}"; marker="${line#*:}"
    if git cat-file -e "$BASE:$marker" 2>/dev/null; then echo "   ${sha:0:9}: already on the base ($marker)"; continue; fi
    if git log --format=%B "$BASE"..HEAD | grep -qF "cherry picked from commit $sha"; then
      [ -f "$marker" ] && echo "   ${sha:0:9}: cherry-picked (-x), $marker present" || { echo "   ${sha:0:9}: picked but $marker is gone"; bad=1; }
    else echo "   ${sha:0:9}: missing"; bad=1; fi
  done <<< "$PICKS"
  [ $bad -eq 0 ]
}
clips_ok() { python3 - "$CLIPS" $STATES <<'PY'
import subprocess, sys
d, states = sys.argv[1], sys.argv[2:]
want = {f"{s}.{e}" for s in states for e in ("webm", "mp4", "png")}
have = set(subprocess.run(["git", "ls-tree", "--name-only", "HEAD", f"{d}/"], capture_output=True, text=True).stdout.split())
have = {h.rsplit("/", 1)[-1] for h in have}
bad = [f"missing {w}" for w in sorted(want - have)] + [f"unexpected {h}" for h in sorted(have - want)]
def head(name):
    return subprocess.run(["git", "cat-file", "blob", f"HEAD:{d}/{name}"], capture_output=True).stdout[:12]
for w in sorted(want & have):
    b = head(w)
    ok = (w.endswith(".webm") and b[:4] == b"\x1a\x45\xdf\xa3") or (w.endswith(".mp4") and b[4:8] == b"ftyp") \
        or (w.endswith(".png") and b[:4] == b"\x89PNG")
    if not ok: bad.append(f"{w}: wrong file signature")
print(f"   {len(have)} files tracked under {d}" + "".join(f"\n   {x}" for x in bad)); sys.exit(1 if bad or len(have) != 36 else 0)
PY
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
# Required jobs = test.yml's CURRENT jobs (read on origin/main d7affad60, 14 Oct). The PR carries the kit's
# scripts/ralph/ files (seeded with add -f), which scripts/ci/backend_test_plan.py cannot select, so backend_mode
# is full: the shards, orchestrator-tests, Alembic from-zero and Schema-drift all run. frontend/ changes → Frontend
# CI (vitest, tsc baselined, route-contract, changed-lines code shape) runs. Code standards runs on every PR.
# "Prod images built" runs only when a Dockerfile, a dependency manifest, next.config.*, lib/auth-edition.ts or
# test.yml changes: green or skipped (so NO lane runs `next build` here — an ownerTest item).
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
  echo ""; echo "=== Seeded (PRD-260 §3, §5)"
  check "the three PRD-259 W1 fixes are cherry-picked (-x) or already on the base" "picks_present"
  check "Auto's 36 clips are tracked under $CLIPS (12 states x webm/mp4/png, checked signatures)" "clips_ok"
  check "the PRD is tracked at docs/PRDS/PRD-260-PLAY-THEME.md" "git ls-files --error-unmatch docs/PRDS/PRD-260-PLAY-THEME.md >/dev/null 2>&1"

  echo ""; echo "=== Stories"
  check "T-001 the Style axis gains play; useIsPlay; the menu; tests" "has \"'play'\" $FE/lib/ui-style.ts && has 'PLAY_HTML_CLASS' $FE/lib/ui-style.ts && has 'export function useIsPlay' $FE/hooks/use-studio-theme.ts && grep -rqF 'Customise look' $FE/components/ui && all_exist $FE/lib/__tests__/ui-style-play.test.ts $FE/components/__tests__/prd260-theme-menu.test.tsx"
  check "T-002a the look engine in lib/looks with Studio's tokens; tests" "all_exist $LOOKS/catalogue.ts $LOOKS/types.ts $LOOKS/presets.ts $LOOKS/validate.ts $LOOKS/contrast.ts $LOOKS/css.ts $LOOKS/studio-tokens.ts $LOOKS/storage.ts $LOOKS/store.ts $LOOKS/transfer.ts $LOOKS/editor.ts $LOOKS/__tests__/contrast.test.ts $LOOKS/__tests__/studio-tokens.test.ts $LOOKS/__tests__/css.test.ts $LOOKS/__tests__/validate.test.ts $LOOKS/__tests__/transfer.test.ts $FE/lib/__tests__/prd260-play-guards.test.ts && has \"'--background'\" $LOOKS/studio-tokens.ts && has \"--pl-\" $LOOKS/css.ts"
  check "T-002b the provider, the pre-paint script under play, the tone, the fonts; tests" "has 'LOOK_PREPAINT_SCRIPT' $LOOKS/prepaint.ts && has 'LOOK_PREPAINT_SCRIPT' $FE/app/layout.tsx && all_exist $FE/components/looks/look-provider.tsx $FE/app/play-fonts.ts $LOOKS/__tests__/prepaint.test.ts $FE/components/looks/__tests__/look-provider.test.tsx && has 'next/font/local' $FE/app/play-fonts.ts && [ \"\$(find $FE/app/fonts/play -name '*.woff2' 2>/dev/null | grep -c .)\" -eq 4 ]"
  check "T-003 the mascot, the clip rule and the chat-event mapping; tests" "all_exist $MASCOT/types.ts $MASCOT/reducer.ts $MASCOT/player.ts $MASCOT/ring.ts $MASCOT/tools.ts $MASCOT/media.ts $MASCOT/events.ts $FE/hooks/use-mascot-player.ts $FE/hooks/use-auto-mascot.ts $FE/components/mascot/auto-mascot.tsx $MASCOT/__tests__/reducer.test.ts $MASCOT/__tests__/events.test.ts $MASCOT/__tests__/media.test.ts $FE/components/mascot/__tests__/auto-mascot.test.tsx"
  check "T-004 Play's frame inside main-layout; tests" "all_exist $PLAY/shell/play-frame.tsx $PLAY/shell/play-sidebar.tsx $PLAY/shell/connection-pill.tsx $PLAY/shell/play-nav.ts $PLAY/shell/__tests__/play-sidebar.test.tsx $PLAY/shell/__tests__/play-frame.test.tsx && has 'PlayFrame' $FE/components/layout/main-layout.tsx && has 'STUDIO_MENU_PRIMARY' $PLAY/shell/play-nav.ts"
  check "T-005a Play's chat on the core's chat hook, through the chat layout's switch; tests" "all_exist $PLAY/play-route-switch.tsx $PLAY/chat/play-chat.tsx $PLAY/chat/__tests__/play-chat.test.tsx && has 'PlayRouteSwitch' $FE/app/chat/layout.tsx && grep -rqE 'useChatWithReceipts' $PLAY/chat && grep -rqE 'useChatSessionStore' $PLAY/chat"
  check "T-005b tool lines, receipts, the Deliverable card, the snag box, mission suggestions; tests" "all_exist $PLAY/chat/tool-lines.tsx $PLAY/chat/receipts.tsx $PLAY/chat/deliverable-card.tsx $PLAY/chat/snag-box.tsx $PLAY/chat/__tests__/play-chat-turn.test.tsx && grep -rqE 'MissionSuggestionCard' $PLAY/chat && grep -rqE 'receiptsOf|receiptLine' $PLAY/chat"
  check "T-006 /conversations on getChatHistory; tests" "all_exist $FE/app/conversations/page.tsx $PLAY/conversations/play-conversations.tsx $PLAY/conversations/__tests__/play-conversations.test.tsx && grep -rqE 'getChatHistory' $PLAY/conversations && grep -qE \"redirect\\(['\\\"]/chat\" $FE/app/conversations/page.tsx\""
  check "T-007a Play's shelf on use-deliverables-api, through the deliverables layout's switch; tests" "all_exist $PLAY/deliverables/play-shelf.tsx $PLAY/deliverables/__tests__/play-shelf.test.tsx && has 'PlayRouteSwitch' $FE/app/deliverables/layout.tsx && grep -rqE 'useDeliverables\\b' $PLAY/deliverables"
  check "T-007b Play's Deliverable detail; tests" "all_exist $PLAY/deliverables/play-deliverable.tsx $PLAY/deliverables/__tests__/play-deliverable.test.tsx && grep -rqE 'useDeliverable\\(' $PLAY/deliverables"
  check "T-008a Make it yours at /looks; tests" "all_exist $FE/app/looks/page.tsx $FE/components/looks/looks-screen.tsx $FE/components/looks/__tests__/looks-screen.test.tsx"
  check "T-008b the live style sheet at /looks/style; tests" "all_exist $FE/app/looks/style/page.tsx $LOOKS/style.ts $FE/components/looks/style/style-screen.tsx $FE/components/looks/__tests__/style-screen.test.tsx"
  check "T-009 Play's 404 and Automatos is asleep; tests" "all_exist $FE/app/not-found.tsx $PLAY/edges/lost-screen.tsx $PLAY/edges/asleep-screen.tsx $PLAY/edges/__tests__/edges.test.tsx"
  check "T-010 the docs and the RUN checklist" "all_exist docs/frontend-architecture/play-theme.md docs/testing/PRD-260-PLAY-RUN-CHECKLIST.md && has 'play-theme' docs/frontend-architecture/_index.md"
  check "every acceptance criterion in $KIT is DONE (the owner's are left)" "all_acs_done"
fi

echo ""; echo "=== Scope and conventions"
check "N1 no new Alembic revision (this wave needs none)" "[ -z \"\$(added_files $VERSIONS)\" ]"
check "N2 exactly one Alembic head in the tree" "one_head"
check "N3 no backend change: orchestrator/ and services/ untouched (Play runs on the existing API)" "git diff --quiet $BASE..HEAD -- orchestrator services"
check "N4 no raw fetch('/api…') added in the frontend" "! raw_fetch_added"
check "N5 no new npm dependency (package.json and package-lock.json untouched)" "git diff --quiet $BASE..HEAD -- frontend/package.json frontend/package-lock.json"
check "N6 no new Python dependency" "git diff --quiet $BASE..HEAD -- orchestrator/requirements.txt orchestrator/requirements-dev.txt orchestrator/pyproject.toml"
check "N7 no innerHTML / dangerouslySetInnerHTML added outside app/layout.tsx" "! inner_html_added"
check "N8 app/layout.tsx adds dangerouslySetInnerHTML only as {{ __html: LOOK_PREPAINT_SCRIPT }}" "layout_html_ok"
check "N9 nothing imports from scripts/ralph/prd260-spec, play-ref or automatos-play (import syntax, comments ignored)" "! forbidden_import"
check "N10 frontend/lib/api-client.ts did not grow" "not_grown frontend/lib/api-client.ts"
check "N11 the Play reference is never committed (scripts/ralph/prd260-spec untracked)" "[ -z \"\$(git diff --name-only $BASE..HEAD -- scripts/ralph/prd260-spec)\" ] && [ -z \"\$(git ls-files scripts/ralph/prd260-spec)\" ]"
check "N12 every commit is DCO-signed" "every_commit_signed"
check "N13 a new text file is at most 800 lines" "file_sizes_ok"
check "N14 no node_modules in the diff" "! git diff --name-only $BASE..HEAD | grep -q node_modules"
check "N15 no orange-* palette class added (.eslintrc no-restricted-syntax)" "! orange_added"
check "N16 app/globals.css grows by at most the two cherry-picked --radius lines (Play's CSS lives in its own files)" "not_grown frontend/app/globals.css \$(radius_slack)"
check "N17 the appearance defaults version is unchanged (no browser is moved off its chosen style)" "defaults_version_same"
check "N18 no next/font/google import added (Play's type is vendored woff2 through next/font/local)" "! google_font_added"
check "N19 Play's CSS defines and reads only --pl-* variables (plus the core's font and radius ones)" "play_css_vars_ok"
check "N20 no Python star import added (ruff F403), read with ast" "! star_import_added"
check "N21 no test weakened: added test functions >= removed" "tests_not_weakened"
warn "N22 story commits carry [skip ci] (all but the final CI commit)" "[ -z \"\$(git log --format='%s' $BASE..HEAD^ | grep -v 'skip ci' | grep -v '^chore(prd-260): seed')\" ]"

if [ "$MODE" = "full" ]; then
  echo ""; echo "=== CI (once per wave)"
  check "CI: the required jobs present and green on the tested commit" "ci_green_on_head"
fi
echo ""
if [ "$FAIL" -eq 0 ]; then echo "✅ PRD-260 Play acceptance ($MODE): PASS"; else echo "❌ PRD-260 Play acceptance ($MODE): FAIL"; fi
exit "$FAIL"
