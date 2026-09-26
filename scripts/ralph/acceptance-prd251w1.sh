#!/bin/bash
# Acceptance gate — PRD-251 Socials, Wave 1: the video engine (US-101..US-114)
# and the agent layer (US-115..US-120).
#   bash scripts/ralph/acceptance-prd251w1.sh               run the gate
#   bash scripts/ralph/acceptance-prd251w1.sh --print-base  print the diff base
# Base: Wave 1 is stacked on Wave 0 (feat/prd-251-socials, #782) until #782
# merges, then moves onto main (owner, 2026-09-23: "push forward"). The base is
# the NEWER of the two fork points: the Wave 0 fork point while stacked, the
# origin/main merge-base after the move. The loop never merges either branch in,
# so neither side's changes can read as this wave's.
# Nothing runs on this machine (owner rule: CI is the only gate): no pytest, no
# npm, no docker, no render. The code checks are greps and git; the tests and the
# fixture render are proven by test.yml on the pushed HEAD, whose six required
# jobs (the five PRD-251 jobs plus media-render) must be present and green.
# No `pipefail` on purpose: with it, `! producer | grep -q X` can PASS when X
# matched — grep -q exits on the first match, the producer takes SIGPIPE (141),
# pipefail reports 141 and the `!` turns that into success. Every negative check
# below would silently invert. BASE is validated before any check runs.
set -u
cd "$(dirname "$0")/../.." || exit 1

export DATABASE_URL="postgresql://ralph:ralph@127.0.0.1:1/ralph_no_db"
export POSTGRES_HOST=127.0.0.1 POSTGRES_PORT=1 POSTGRES_USER=ralph POSTGRES_PASSWORD=ralph POSTGRES_DB=ralph_no_db
export REDIS_URL="redis://127.0.0.1:1/0" REDIS_HOST=127.0.0.1 REDIS_PORT=1
export ENVIRONMENT=development

W0_REF=origin/feat/prd-251-socials
BASE=$(git merge-base HEAD origin/main) || { echo "cannot resolve merge-base with origin/main"; exit 1; }
if git rev-parse -q --verify "$W0_REF" >/dev/null; then
  W0_BASE=$(git merge-base HEAD "$W0_REF" 2>/dev/null || true)
  if [ -n "$W0_BASE" ] && [ "$W0_BASE" != "$BASE" ] && git merge-base --is-ancestor "$BASE" "$W0_BASE"; then
    BASE="$W0_BASE"
  fi
fi
if [ "${1:-}" = "--print-base" ]; then echo "$BASE"; exit 0; fi
echo "base: $(git log -1 --format='%h %s' "$BASE")"
FAIL=0

# Each check body runs in a subshell: a `cd` inside one never leaks into the next.
check() {
  local name="$1" body="$2"
  echo ""
  echo "── $name"
  if ( eval "$body" ); then echo "   ✅ PASS: $name"; else echo "   ❌ FAIL: $name"; FAIL=1; fi
}

MR=services/media-render
CFG=orchestrator/config.py
MAN=orchestrator/reports/route-manifest.json
SURFACE=orchestrator/reports/config-surface.json
BK=orchestrator/modules/documents/brand_kit.py
KIT=scripts/ralph/prd-251w1.json

# ── helpers for the checks that need more than one line ─────────────────────
new_revision_file() {
  git diff --name-only --diff-filter=A "$BASE"..HEAD -- orchestrator/alembic/versions | grep '\.py$'
}
one_new_revision() {
  [ "$(new_revision_file | grep -c .)" -eq 1 ]
}
new_revision_id() {
  sed -nE "s/^revision(: *str)? *= *['\"]([^'\"]+)['\"].*/\2/p" "$(new_revision_file | head -1)" | head -1
}
# The base's single alembic head: prd251_socials while stacked on Wave 0,
# f049_prd251_merge_heads once Wave 0 reached main through the customer-night PR.
base_head() {
  git show "$BASE:orchestrator/tests/test_prd209_alembic_single_head.py" 2>/dev/null \
    | sed -nE 's/^EXPECTED_HEAD = "([^"]+)".*/\1/p' | head -1
}
# 89d89c250: a create_all-first boot crash-looped on a migration that assumed empty tables.
migration_is_create_all_safe() {
  local f; f=$(new_revision_file | head -1)
  [ -n "$f" ] && grep -qE 'has_table|IF NOT EXISTS|IF EXISTS|_create_missing_indexes|get_indexes' "$f"
}
migration_chains_onto_base_head() {
  local f head; f=$(new_revision_file | head -1); head=$(base_head)
  if [ -z "$f" ] || [ -z "$head" ]; then echo "   new revision: ${f:-none}; base head: ${head:-unknown}"; return 1; fi
  grep -qE "^down_revision.*['\"]$head['\"]" "$f" || { echo "   $f does not chain onto the base head $head"; return 1; }
}
head_pins_moved() {
  local rev; rev=$(new_revision_id)
  [ -n "$rev" ] \
    && grep -q "EXPECTED_HEAD = \"$rev\"" orchestrator/tests/test_prd209_alembic_single_head.py \
    && grep -q "$rev" orchestrator/tests/test_prd236_w1_routes.py
}
manifest_has_action() {
  grep -qE "\"/api/socials/posts/\\{[a-z_]+\\}/$1\"" "$MAN"
}
hyperframes_pinned() {
  git grep -qE 'hyperframes@0\.8\.62|"hyperframes"[[:space:]]*:[[:space:]]*"0\.8\.62"' -- "$MR"
}
media_render_env_off() {
  local v
  for v in HYPERFRAMES_NO_TELEMETRY DO_NOT_TRACK HYPERFRAMES_SKIP_SKILLS HYPERFRAMES_NO_UPDATE_CHECK; do
    grep -q "$v" "$MR/Dockerfile" || { echo "   Dockerfile does not set $v"; return 1; }
  done
}
# GPL boundary: phonemizer / espeak-ng (and kokoro-onnx, which pulls them in)
# are never a dependency of, or imported by, the orchestrator.
# No `\b` here: git grep -E on macOS does not support it, and a pattern that can
# never match would turn these negative checks into silent passes.
gpl_boundary_ok() {
  ! git grep -qE '^[[:space:]]*(import|from)[[:space:]]+(phonemizer|espeakng_loader|kokoro_onnx|misaki)([^A-Za-z0-9_]|$)' -- orchestrator \
    && ! git grep -qiE '^[[:space:]]*"?(phonemizer|espeakng|kokoro[-_]onnx|misaki)([^A-Za-z0-9_]|$)' \
         -- 'orchestrator/requirements*.txt' orchestrator/pyproject.toml
}
render_routes_exist() {
  git grep -qE "[\"']/render" -- "$MR" && git grep -qE "[\"']/tts[\"']" -- "$MR"
}
docs_cover_media_profile() {
  grep -q 'media-render' docs/deployment-infrastructure/docker-containerization.md \
    && grep -qE 'profile media|COMPOSE_PROFILES=[^[:space:]]*media' docs/getting-started/self-hosting.md
}
font_route_exists() {
  grep -qE '@router\.(post|put)\("/brand-kit/fonts?' orchestrator/api/document_brand_kit.py
}
# The Auto skill seed is GENERATED from automatos-skills by the sync script;
# the owner's rule is that skills change there first, never here.
no_clone_refs_outside_seed() {
  # Code only: the tests that pin the clone's absence name it, and so may the
  # generated skill seeds (synced from automatos-skills, US-119).
  ! git grep -q 'repos/automatos-social' -- orchestrator frontend services \
      ':(exclude)orchestrator/tests' ':(exclude)orchestrator/core/seeds/platform-management-skill.md' \
      ':(exclude)orchestrator/core/seeds/skills'
}
# D15: paid tools only through the workspace's Composio connection.
no_provider_clients() {
  ! git diff -U0 "$BASE"..HEAD -- orchestrator services frontend ':(exclude)orchestrator/tests' \
      | grep -E '^\+[^+]' \
      | grep -qiE 'api\.fish\.audio|api\.elevenlabs\.io|fal\.run|higgsfield\.ai|api\.kie\.ai|kie\.ai/api'
}
compose_media_profile() {
  python3 - <<'PY'
import re, sys
s = open("docker-compose.yml").read()
m = re.search(r"^  media-render:\n(.*?)(?=^ {0,2}\S|\Z)", s, re.S | re.M)
ok = bool(m) and re.search(r"profiles:\s*(\[[^\]]*\bmedia\b[^\]]*\]|-\s*['\"]?media\b)", m.group(1))
sys.exit(0 if ok else 1)
PY
}
quota_values_ok() {
  python3 - <<'PY'
import re, sys
s = open("orchestrator/config.py").read()
vals = {int(v) for v in re.findall(r"render_minutes_month['\"]?\s*[:=]\s*(\d+)", s)}
if not {10, 60, 240} <= vals:
    print(f"   render_minutes_month values found: {sorted(vals)} (want 10, 60, 240)")
    sys.exit(1)
PY
}
music_manifest_ok() {
  python3 - <<'PY'
import json, sys
want = {
    "2feeff3d7bfffc24712bf0b9df1efc76a1777973fde3143b061579c4541d6c83",  # Where the Night Begins
    "5e71ada9dc33cd4fc21e69a38008b4b263d45de8aca2b2b5ed466816f66e9792",  # Da Da Da Da Da De De De De
    "ab1805743e5c3a728013ad45e06a80fe6cd22384ea9f2602e4d9209cf585016d",  # Spring of 2026
    "ddb3174cb4f2337a04c918902e4a766a427e947c6b04ae8d406317db07b32109",  # Deep House 003
}
try:
    d = json.load(open("services/media-render/music/manifest.json"))
except Exception as e:
    print(f"   manifest unreadable: {e}")
    sys.exit(1)
tracks = d.get("tracks", []) if isinstance(d, dict) else d
ok = bool(tracks)
for t in tracks:
    for k in ("title", "url", "sha256", "attribution"):
        if not t.get(k):
            print(f"   {t.get('title', '?')}: missing {k}"); ok = False
    if not (t.get("licence") or t.get("license")):
        print(f"   {t.get('title', '?')}: missing licence"); ok = False
missing = want - {t.get("sha256") for t in tracks}
if missing:
    print(f"   reference tracks missing: {sorted(missing)}"); ok = False
sys.exit(0 if ok else 1)
PY
}
all_acs_done() {
  python3 - <<'PY'
import json, sys
d = json.load(open("scripts/ralph/prd-251w1.json"))
todo = [(s["id"], i + 1) for s in d["userStories"] for i, ac in enumerate(s["acceptanceCriteria"]) if "DONE" not in ac]
for sid, n in todo[:25]:
    print(f"   not DONE: {sid} AC{n}")
sys.exit(1 if todo else 0)
PY
}
# No generated media in git; fixtures only, and tiny.
no_generated_media() {
  local bad f big=""
  bad=$(git diff --name-only --diff-filter=A "$BASE"..HEAD \
        | grep -iE '\.(mp4|mov|webm|mkv|mp3|wav|m4a|aac|flac|ogg)$' | grep -v "^$MR/fixtures/" || true)
  [ -z "$bad" ] || { echo "$bad" | sed 's/^/   media outside fixtures: /'; return 1; }
  for f in $(git diff --name-only --diff-filter=A "$BASE"..HEAD -- "$MR/fixtures"); do
    [ -f "$f" ] && [ "$(wc -c < "$f")" -gt 1048576 ] && big="$big $f"
  done
  [ -z "$big" ] || { echo "   fixture over 1 MB:$big"; return 1; }
}
no_env_reads_outside_config() {
  ! git diff -U0 "$BASE"..HEAD -- orchestrator ':(exclude)orchestrator/config.py' ':(exclude)orchestrator/tests' \
      | grep -E '^\+[^+].*os\.(getenv|environ)' | grep -q .
}
every_commit_signed() {
  local c
  for c in $(git rev-list "$BASE"..HEAD); do
    git show -s --format=%B "$c" | grep -q '^Signed-off-by:' || { echo "   unsigned: $c"; return 1; }
  done
}
no_wave2_code() {
  # Code only: this script's own pattern line is in the diff too.
  git diff --quiet "$BASE"..HEAD -- orchestrator/modules/socials/publisher.py \
    && ! git diff "$BASE"..HEAD -- orchestrator/ frontend/ services/ | grep -qE '^\+.*social-publish-'
}
# The pushed HEAD's test.yml run must carry every required job, each green.
# Polls up to 40 min (a full run takes ~10-20 with the media-render image build).
# A newer push cancels an older run, so only HEAD's own run counts.
REQUIRED_JOBS=("orchestrator-tests" "Alembic from-zero" "Schema-drift check" "Frontend CI" "Prod images built" "media-render")
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
    elif echo "$hits" | grep -qv '^success'; then
      echo "   required job not green: $job"; bad=1
    fi
  done
  [ "$bad" -eq 0 ] || return 1
  echo "   test.yml run $id is green on $sha for all ${#REQUIRED_JOBS[@]} required jobs"
}

# ── agent-layer helpers (US-115..US-120) ─────────────────────────────────────
DISC=orchestrator/modules/tools/discovery
cover_image_post_id_optional() {
  python3 - <<'PY'
import re, sys
s = open("orchestrator/modules/tools/discovery/actions_blog.py").read()
i = s.find('name="platform_generate_cover_image"')
if i < 0:
    print("   platform_generate_cover_image not found"); sys.exit(1)
m = re.search(r'"required"\s*:\s*\[([^\]]*)\]', s[i:i + 4000])
sys.exit(0 if (m and "post_id" not in m.group(1)) else 1)
PY
}
no_second_image_tool() {
  ! git diff -U0 "$BASE"..HEAD -- "$DISC" | grep -E '^\+[^+]' \
      | grep -qE 'name="platform_generate_(image|social_image|still|picture)'
}
socials_tools_cannot_publish() {
  [ -f "$DISC/actions_socials.py" ] || return 1
  # No tool this wave adds is named approve/schedule/publish. The tree already has
  # platform_publish_blog_post, platform_approve_mission, platform_schedule_playbook
  # and platform_schedule_task (on the base), none of them a post's.
  ! grep -qE '"(approve|approved|approval|schedule|scheduled_at|publish|publish_now)"[[:space:]]*:' "$DISC/actions_socials.py" \
    && ! git diff -U0 "$BASE"..HEAD -- "$DISC" | grep -E '^\+[^+]' | grep -qE 'name="platform_(approve|schedule|publish)[a-z_]*"'
}
# US-117: generate_document's format enum is DOCUMENT_TEMPLATE_FORMATS (core/models/core.py),
# which carries core/social_templates.py's two social formats.
generate_document_takes_social_formats() {
  grep -q 'list(DOCUMENT_TEMPLATE_FORMATS)' orchestrator/modules/agents/services/agent_platform_tools.py \
    && grep -qE '^DOCUMENT_TEMPLATE_FORMATS = .*SOCIAL_TEMPLATE_FORMATS' orchestrator/core/models/core.py \
    && grep -qF 'SOCIAL_IMAGE, SOCIAL_VIDEO = "social_image", "social_video"' orchestrator/core/social_templates.py
}
post_gate_seeded() {
  local added
  added=$(git diff "$BASE"..HEAD -- orchestrator/alembic/versions | grep -E '^\+[^+]')
  echo "$added" | grep -q 'post_actions' && echo "$added" | grep -q 'INSTAGRAM_POST_IG_USER_MEDIA_PUBLISH'
}
post_gate_points_to_drafts() {
  git grep -q 'post_actions' -- orchestrator/core/composio orchestrator/modules/socials \
    && git grep -q 'platform_create_social_post' -- orchestrator/core/composio orchestrator/modules/socials
}
builtin_skill_files_generated() {
  local f bad=""
  for f in $(git diff --name-only --diff-filter=AM "$BASE"..HEAD -- orchestrator/core/seeds/skills | grep -v 'manifest\.json$'); do
    [ -f "$f" ] && ! grep -q 'GENERATED FILE' "$f" && bad="$bad $f"
  done
  [ -z "$bad" ] || { echo "   hand-written skill file(s):$bad"; return 1; }
}
package_seeded() {
  git grep -q 'social-media-director' -- orchestrator/core/seeds orchestrator/scripts \
    && git grep -q 'brand-designer' -- orchestrator/core/seeds orchestrator/scripts \
    && git grep -qE "[\"']slug[\"'][[:space:]]*:[[:space:]]*[\"']socials[\"']" -- orchestrator/core/seeds
}
no_publisher_skills_seeded() {
  ! git diff "$BASE"..HEAD -- 'orchestrator/core/seeds/*.py' 'orchestrator/scripts/*.py' | grep -E '^\+[^+]' \
      | grep -qE 'instagram-curator|twitter-engager|linkedin-content-creator|html-to-png'
}
no_marketplace_items_writes() {
  ! git diff -U0 "$BASE"..HEAD -- orchestrator ':(exclude)orchestrator/tests' | grep -E '^\+[^+]' | grep -q 'marketplace_items'
}

# ── S1.1a the media-render service and its CI job (US-101) ──────────────────
check "media-render has a Dockerfile" "[ -f $MR/Dockerfile ]"
check "hyperframes pinned at exactly 0.8.62" "hyperframes_pinned"
check "Node 22 in the image" "grep -qE 'setup_22|node:22|NODE_MAJOR=22' $MR/Dockerfile"
check "ffmpeg in the image" "grep -q 'ffmpeg' $MR/Dockerfile"
check "telemetry, skills and update checks off in the image" "media_render_env_off"
check "a fixture composition is committed" "git ls-files $MR/fixtures | grep -q '\.html$'"
check "test.yml has the media-render job" "grep -q 'media-render' .github/workflows/test.yml"
check "media-render carries its own tests" "git ls-files $MR | grep -qE '(^|/)tests?/|/test_[^/]+\.py$'"
check "GPL boundary: phonemizer/espeak-ng/kokoro-onnx never reach the orchestrator" "gpl_boundary_ok"

# ── S1.1b the media-render API (US-102) ─────────────────────────────────────
check "media-render exposes /render and /tts" "render_routes_exist"
check "media-render checks X-Internal-Token" "git grep -q 'X-Internal-Token' -- $MR"
check "a failed hyperframes check returns 422" "git grep -q '422' -- $MR"
check "the mix ducks music and normalises to -14 LUFS" \
  "git grep -q 'sidechaincompress' -- $MR && git grep -q 'loudnorm' -- $MR && git grep -q 'I=-14' -- $MR"

# ── S4.4 the media usage lane (US-103) ──────────────────────────────────────
check "LANE_MEDIA exists" "grep -q 'LANE_MEDIA' orchestrator/core/llm/usage_context.py"
check "track_media exists" "grep -q 'def track_media' orchestrator/core/llm/usage_tracker.py"

# ── S1.1c client, lifecycle, quotas, compose, Railway (US-104) ──────────────
check "orchestrator render client exists and sends the internal token" \
  "[ -f orchestrator/core/media_render_client.py ] && grep -q 'X-Internal-Token' orchestrator/core/media_render_client.py"
check "config.py defines SOCIALS_RENDER_TOKEN and SOCIALS_RENDER_TIMEOUT_SECONDS" \
  "grep -q 'SOCIALS_RENDER_TOKEN' $CFG && grep -q 'SOCIALS_RENDER_TIMEOUT_SECONDS' $CFG"
check "config-surface.json lists the new settings and Wave 0's" \
  "grep -q '\"SOCIALS_RENDER_TOKEN\"' $SURFACE && grep -q '\"SOCIALS_RENDER_TIMEOUT_SECONDS\"' $SURFACE && grep -q '\"SOCIALS_ENABLED_DEFAULT\"' $SURFACE"
check "plan tiers carry render_minutes_month = 10 / 60 / 240" "quota_values_ok"
check "plan_limits_for_tier surfaces the render quota" "grep -q 'render_minutes_month' orchestrator/services/plan_tiers.py"
check "route manifest lists the render action" "manifest_has_action render"
check "compose runs media-render under the media profile" "compose_media_profile"
check "Railway manifest carries media-render" "grep -q 'media-render' infrastructure/railway-manifest.json"
check "deployment docs cover media-render and the media profile" "docs_cover_media_profile"
check "the Socials tab shows render minutes" \
  "git grep -qiE 'render.?minutes|renderMinutes' -- frontend/components/deliverables/socials frontend/lib"

# ── S1.2a templates as data: ONE migration (US-105) ─────────────────────────
check "exactly one new alembic revision on the branch" "one_new_revision"
check "it chains onto the base's single alembic head" "migration_chains_onto_base_head"
check "the migration is create_all-first safe (guards present)" "migration_is_create_all_safe"
check "both head pins moved to the new revision" "head_pins_moved"
check "document_templates accepts social_image and social_video" \
  "grep -q 'social_image' orchestrator/core/models/core.py && grep -q 'social_video' orchestrator/core/models/core.py"
check "social templates are validated on save" "grep -q 'variables_schema' orchestrator/modules/documents/template_service.py"
check "generate() dispatches the social formats" "grep -qE 'social_(video|image)' orchestrator/modules/documents/generation_service.py"

# ── S1.2b/S1.2c the seeded templates (US-106, US-107) ───────────────────────
for name in "UI story" "cinematic product" "app promo" "data story"; do
  check "seeded video template: $name" "git grep -qi '$name' -- orchestrator/modules ':!orchestrator/tests'"
done
check "no code reference to the automatos-social clone remains (generated skill seed excluded)" "no_clone_refs_outside_seed"
check "the generated Auto skill seed is untouched (skills change in automatos-skills first)" \
  "git diff --quiet $BASE..HEAD -- orchestrator/core/seeds/platform-management-skill.md"

# ── S1.3 brand kit (US-108) ─────────────────────────────────────────────────
for field in heading_font font_files logo_mark_url social_handles; do
  check "brand kit field: $field" "grep -q '$field' $BK"
done
check "font_family stays the body font (no second body-font field)" \
  "grep -q 'font_family' $BK && ! grep -qE '^[[:space:]]+body_font[[:space:]]*:' $BK"
check "brand kit carries the voice (tone words, banned phrases)" "grep -qw 'voice' $BK"
check "font upload route" "font_route_exists"
check "the dialog edits the new fields" \
  "grep -qE 'heading_font|headingFont' frontend/components/documents/blocks/BrandKitDialog.tsx"

# ── S1.4 sources, D16 registry, S1.5 voice (US-109..US-111) ─────────────────
check "route manifest lists /api/socials/sources" "grep -q '\"/api/socials/sources\"' $MAN"
check "report sources read agent_reports" "git grep -qE 'agent_reports|ReportService|report_service' -- orchestrator/modules/socials"
check "the media capability registry exists" "[ -f orchestrator/modules/socials/capabilities.py ]"
check "the Wave 0 deny list is untouched" "git diff --quiet $BASE..HEAD -- orchestrator/core/composio/deny_list.py"
check "voice recipes exist and use Composio" \
  "[ -f orchestrator/modules/socials/recipes/voice.py ] && git grep -q 'FISH_AUDIO_SYNTHESIZE_SPEECH' -- orchestrator"
check "no direct provider clients (D15)" "no_provider_clients"

# ── S1.6 music, S1.7 infographic, S1.8 footage (US-112..US-114) ─────────────
check "music manifest: the four reference tracks with licence, attribution, URL and sha256" "music_manifest_ok"
check "the infographic template is seeded" "git grep -qi 'infographic' -- orchestrator/modules ':!orchestrator/tests'"
check "footage recipes exist" "[ -f orchestrator/modules/socials/recipes/footage.py ]"
check "the fal estimate action is allowlisted" "git grep -q 'FAL_AI_ESTIMATE_PRICING' -- orchestrator ':(exclude)orchestrator/tests'"
check "the workspace monthly media cap exists" "git grep -q 'media_monthly_cap_usd' -- orchestrator"
check "footage spend is booked on the media lane" "git grep -q 'track_media' -- orchestrator/modules/socials"

# ── the agent layer (US-115..US-120) ────────────────────────────────────────
check "brand-kit tools are defined in the documents domain and routed" \
  "grep -q 'platform_get_brand_kit' $DISC/actions_documents.py && grep -q 'platform_update_brand_kit' $DISC/actions_documents.py && grep -q 'platform_get_brand_kit' $DISC/platform_executor.py && grep -q 'platform_update_brand_kit' $DISC/platform_executor.py"
for t in platform_create_social_post platform_update_social_post platform_submit_social_post platform_get_social_post platform_list_social_posts; do
  check "socials tool $t is defined and routed" "grep -q '$t' $DISC/actions_socials.py && grep -q '$t' $DISC/platform_executor.py"
done
check "register_all_actions wires the socials tools" "grep -q 'actions_socials' $DISC/platform_actions.py"
check "no tool approves, schedules or publishes a post" "socials_tools_cannot_publish"
check "generate_document accepts social_image and social_video" "generate_document_takes_social_formats"
check "the image tool no longer requires a blog post" "cover_image_post_id_optional"
check "no second image tool" "no_second_image_tool"
check "the post gate's action list is seeded in the wave's migration" "post_gate_seeded"
check "the post gate points agents at the draft tool" "post_gate_points_to_drafts"
check "the built-in skills manifest exists and the loader reads it" \
  "[ -f orchestrator/core/seeds/skills/manifest.json ] && git grep -q 'manifest.json' -- orchestrator/modules/agents"
check "sync-skills.py replaces sync-auto-skill.py" "[ -f scripts/sync-skills.py ] && [ ! -f scripts/sync-auto-skill.py ]"
check "skill files under seeds/skills are generated, never hand-written" "builtin_skill_files_generated"
check "the Socials package and its two agents are seeded" "package_seeded"
for pb in "Brand kit from your website" "Launch video" "Weekly social posts" "Image carousel"; do
  check "marketplace playbook seeded: $pb" "git grep -q '$pb' -- orchestrator/core/seeds orchestrator/scripts"
done
check "no publisher skill or html-to-png in the new seeds" "no_publisher_skills_seeded"
check "nothing new is written to marketplace_items" "no_marketplace_items_writes"

# ── scope and conventions ────────────────────────────────────────────────────
check "no env reads added outside config.py (tests excluded)" "no_env_reads_outside_config"
check "every commit is DCO-signed" "every_commit_signed"
check "no node_modules in the diff" "! git diff --name-only $BASE..HEAD | grep -q node_modules"
check "no generated video or audio in git (fixtures only, under 1 MB)" "no_generated_media"
check "no Wave 2+ code (publisher untouched, no publish jobs)" "no_wave2_code"
check "at least eight Wave 1 backend test files" \
  "[ \$(ls orchestrator/tests/test_prd251w1_*.py 2>/dev/null | wc -l) -ge 8 ]"
check "at least one new Socials frontend test file" \
  "git diff --name-only --diff-filter=A $BASE..HEAD -- frontend | grep -qiE 'socials.*\.test\.tsx?$'"
check "every acceptance criterion in $KIT is marked DONE" "all_acs_done"

# ── tests: proven by CI on the pushed HEAD, never run here ───────────────────
check "CI: the five PRD-251 jobs and media-render present and green on HEAD" "ci_green_on_head"

echo ""
if [ "$FAIL" -eq 0 ]; then echo "✅ PRD-251 Wave 1 acceptance: PASS"; else echo "❌ PRD-251 Wave 1 acceptance: FAIL"; fi
exit "$FAIL"
