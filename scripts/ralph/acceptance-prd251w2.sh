#!/bin/bash
# Acceptance gate — PRD-251 Socials, Wave 2: the Socials tab (US-205..US-210) and what the
# composer needs (US-201..US-204: the migration, media hosting, the registry, channels as
# approved content), plus Wave 1's review fixes (P251W1-RVW-2..7). Nothing publishes yet.
#   bash scripts/ralph/acceptance-prd251w2.sh               run the gate
#   bash scripts/ralph/acceptance-prd251w2.sh --print-base  print the diff base
# Base: stacked on Wave 1 (feat/prd-251-w1-video-engine) until Wave 1 is on main,
# then main. The base is the NEWER of the two fork points: Wave 1's fork point while
# stacked, the origin/main merge-base after the move. The loop never merges either
# branch in, so neither side's changes can read as this run's.
# Nothing runs on this machine (owner rule: CI is the only gate): no pytest, no npm,
# no docker. The code checks are greps and git; the tests are proven by test.yml on
# the pushed HEAD, whose seven required jobs must be present and green.
# No `pipefail` on purpose: with it, `! producer | grep -q X` can PASS when X matched
# — grep -q exits on the first match, the producer takes SIGPIPE (141), pipefail
# reports 141 and the `!` turns that into success. Every negative check below would
# silently invert. BASE is validated before any check runs.
set -u
cd "$(dirname "$0")/../.." || exit 1

export DATABASE_URL="postgresql://ralph:ralph@127.0.0.1:1/ralph_no_db"
export POSTGRES_HOST=127.0.0.1 POSTGRES_PORT=1 POSTGRES_USER=ralph POSTGRES_PASSWORD=ralph POSTGRES_DB=ralph_no_db
export REDIS_URL="redis://127.0.0.1:1/0" REDIS_HOST=127.0.0.1 REDIS_PORT=1
export ENVIRONMENT=development

W1_REF=origin/feat/prd-251-w1-video-engine
BASE=$(git merge-base HEAD origin/main) || { echo "cannot resolve merge-base with origin/main"; exit 1; }
if git rev-parse -q --verify "$W1_REF" >/dev/null; then
  W1_BASE=$(git merge-base HEAD "$W1_REF" 2>/dev/null || true)
  if [ -n "$W1_BASE" ] && [ "$W1_BASE" != "$BASE" ] && git merge-base --is-ancestor "$BASE" "$W1_BASE"; then
    BASE="$W1_BASE"
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

CFG=orchestrator/config.py
MAN=orchestrator/reports/route-manifest.json
SURFACE=orchestrator/reports/config-surface.json
SOC=orchestrator/modules/socials
API=orchestrator/api/socials.py
FE=frontend
KIT=scripts/ralph/prd-251w2.json

# ── helpers ──────────────────────────────────────────────────────────────────
new_revision_file() {
  git diff --name-only --diff-filter=A "$BASE"..HEAD -- orchestrator/alembic/versions | grep '\.py$'
}
one_new_revision() {
  [ "$(new_revision_file | grep -c .)" -eq 1 ]
}
new_revision_id() {
  sed -nE "s/^revision(: *str)? *= *['\"]([^'\"]+)['\"].*/\2/p" "$(new_revision_file | head -1)" | head -1
}
# The base's single alembic head: Wave 1's revision (stacked or on main).
base_head() {
  git show "$BASE:orchestrator/tests/test_prd209_alembic_single_head.py" 2>/dev/null \
    | sed -nE 's/^EXPECTED_HEAD = "([^"]+)".*/\1/p' | head -1
}
# 89d89c250: a create_all-first boot crash-looped on a migration that assumed empty tables.
migration_is_create_all_safe() {
  local f; f=$(new_revision_file | head -1)
  [ -n "$f" ] && grep -qE 'has_table|IF NOT EXISTS|IF EXISTS|_create_missing_indexes|get_indexes|get_foreign_keys' "$f"
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
# The frontend calls each new route with the right verb (route-contract checks paths only).
api_client_verbs() {
  python3 - <<'PY2'
import re, sys
src = open("frontend/lib/api-client.ts").read()
calls = []
for m in re.finditer(r"request(?:<[^>]*>)?\(\s*[`'\"]([^`'\"]+)[`'\"]", src):
    path = re.sub(r"\$\{[^}]*\}", "*", m.group(1)).split("?")[0]
    tail = src[m.end():m.end() + 400].split("request(")[0].split("\n  async ")[0]
    mm = re.search(r"method:\s*['\"](\w+)['\"]", tail)
    calls.append(((mm.group(1) if mm else "GET"), path))
want = [("PUT", r"/api/socials/posts/\*/targets"), ("POST", r"/api/socials/compose"),
        ("GET", r"/api/socials/channels"), ("GET", r"/api/socials/posts/\*/media"),
        ("POST", r"/api/socials/campaigns/\*/approve")]
bad = []
for verb, rx in want:
    hits = [v for v, p in calls if re.fullmatch(rx, p)]
    if not hits:
        bad.append(f"no apiClient call to {rx}")
    elif verb not in hits:
        bad.append(f"{rx} is called with {sorted(set(hits))}, not {verb}")
for b in bad:
    print("   " + b)
sys.exit(1 if bad else 0)
PY2
}
config_and_surface() {
  local n
  for n in "$@"; do
    grep -q "$n" "$CFG" || { echo "   $n missing from config.py"; return 1; }
    grep -q "\"$n\"" "$SURFACE" || { echo "   $n missing from config-surface.json"; return 1; }
  done
}
# Added lines only, code only (tests, the generated Auto skill seed and docs excluded).
added_code_lines() {
  git diff -U0 "$BASE"..HEAD -- orchestrator frontend services \
      ':(exclude)orchestrator/tests' ':(exclude)frontend/**/__tests__/**' \
      ':(exclude)orchestrator/core/seeds/platform-management-skill.md' \
    | grep -E '^\+[^+]'
}
# The stale or URL-pull slugs the adapters must never CALL. A line that names one
# only to say it is not used (deprecated / never / stale / not used) is allowed.
no_forbidden_slugs() {
  ! added_code_lines \
      | grep -E 'TWITTER_CREATE_TWEET|INSTAGRAM_CREATE_MEDIA_CONTAINER|INSTAGRAM_CREATE_POST[^_]|INSTAGRAM_GET_POST_STATUS|TIKTOK_PUBLISH_VIDEO|TIKTOK_POST_PHOTO' \
      | grep -viqE 'deprecat|never|stale|not used|unused|forbid'
}
# FILE-FIRST: the global UPLOAD_ACTIONS is not widened (Instagram and TikTok stay out).
upload_actions_not_widened() {
  ! git diff -U0 "$BASE"..HEAD -- orchestrator/core/composio/tool_executor.py \
      | grep -E '^\+[^+]' | grep -qE '"(INSTAGRAM|TIKTOK|YOUTUBE)_[A-Z_]+"'
}
no_boto3_outside_factory() {
  ! git diff -U0 "$BASE"..HEAD -- orchestrator ':(exclude)orchestrator/tests' ':(exclude)orchestrator/core/storage/s3.py' \
      | grep -E '^\+[^+]' | grep -qE 'boto3\.(client|resource|session\.Session)\('
}
minio_in_ci() {
  grep -qi 'minio' .github/workflows/test.yml && grep -q 'SOCIALS_TEST_S3_ENDPOINT' .github/workflows/test.yml
}
content_hash_covers_targets() {
  python3 - <<'PY'
import re, sys
s = open("orchestrator/modules/socials/service.py").read()
m = re.search(r"def _content_of\(.*?\n(?=def )", s, re.S)
sys.exit(0 if (m and "target" in m.group(0)) else 1)
PY
}
publisher_guard_first() {
  python3 - <<'PY'
import re, sys
s = open("orchestrator/modules/socials/publisher.py").read()
ok = "assert_publishable" in s and "assert_retryable" in s
sys.exit(0 if ok else 1)
PY
}
all_adapter_slugs_present() {
  local s missing=""
  for s in LINKEDIN_CREATE_LINKED_IN_POST LINKEDIN_UPLOAD_VIDEO LINKEDIN_CREATE_VIDEO_POST \
           TWITTER_CREATION_OF_A_POST TWITTER_UPLOAD_MEDIA \
           INSTAGRAM_POST_IG_USER_MEDIA INSTAGRAM_POST_IG_USER_MEDIA_PUBLISH INSTAGRAM_CREATE_CAROUSEL_CONTAINER \
           TIKTOK_QUERY_CREATOR_INFO TIKTOK_UPLOAD_VIDEO TIKTOK_FETCH_PUBLISH_STATUS \
           YOUTUBE_UPLOAD_VIDEO YOUTUBE_UPDATE_THUMBNAIL; do
    git grep -q "$s" -- "$SOC" ':(exclude)orchestrator/tests' || missing="$missing $s"
  done
  [ -z "$missing" ] || { echo "   adapter data lacks:$missing"; return 1; }
}
x_video_path_present() {
  git grep -qE 'TWITTER_UPLOAD_LARGE_MEDIA|TWITTER_FINALIZE_MEDIA_UPLOAD' -- "$SOC" ':(exclude)orchestrator/tests'
}
event_types_added() {
  grep -q 'social_post_missed' orchestrator/core/services/notification_dispatcher.py \
    && grep -q 'social_post_failed' orchestrator/core/services/notification_dispatcher.py
}
calendar_unions_have_social() {
  python3 - <<'PY2'
import re, sys
u = open("frontend/hooks/use-activity-api.ts").read()
m = re.search(r"export type ScheduleItemType\s*=(.*?)(?:\n\s*\n|\nexport\s|\Z)", u, re.S)
ok = bool(m) and "'social'" in m.group(1)
ok = ok and "social" in open("frontend/components/command-center/calendar-kinds.ts").read()
ok = ok and re.search(r"case\s+'social'", open("frontend/components/command-center/calendar-actions.ts").read()) is not None
sys.exit(0 if ok else 1)
PY2
}
api_client_has() {
  local m missing=""
  for m in "$@"; do grep -q "$m" "$FE/lib/api-client.ts" || missing="$missing $m"; done
  [ -z "$missing" ] || { echo "   api-client lacks:$missing"; return 1; }
}
no_new_drag_dependency() {
  ! git diff -U0 "$BASE"..HEAD -- "$FE/package.json" | grep -E '^\+[^+]' | grep -qiE 'dnd|drag|sortable|draggable'
}
no_new_async_db_routes() {
  ! git diff -U0 "$BASE"..HEAD -- "$API" | grep -qE '^\+async def '
}
deny_list_untouched() {
  git diff --quiet "$BASE"..HEAD -- orchestrator/core/composio/deny_list.py
}
no_phase2_code() {
  ! added_code_lines | grep -qE 'INSTAGRAM_GET_IG_MEDIA_COMMENTS|INSTAGRAM_POST_IG_COMMENT_REPLIES|INSTAGRAM_POST_IG_MEDIA_COMMENTS|YOUTUBE_POST_COMMENT|YOUTUBE_CREATE_COMMENT_REPLY|LINKEDIN_CREATE_COMMENT_ON_POST'
}
no_generated_api_images() {
  ! git grep -q '/api/generated-images' -- "$SOC"
}
fixtures_small() {
  local f bad=""
  for f in $(git diff --name-only --diff-filter=A "$BASE"..HEAD | grep -iE '\.(mp4|mov|webm|mkv|mp3|wav|m4a|aac|flac|ogg|png|jpe?g|gif|webp)$'); do
    case "$f" in
      orchestrator/tests/fixtures/*|frontend/*/__tests__/*) [ -f "$f" ] && [ "$(wc -c < "$f")" -gt 102400 ] && bad="$bad $f(>100KB)" ;;
      frontend/public/*) ;;  # UI assets (icons) are allowed
      *) bad="$bad $f(outside fixtures)" ;;
    esac
  done
  [ -z "$bad" ] || { echo "   media:$bad"; return 1; }
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
all_acs_done() {
  python3 - <<'PY'
import json, sys
d = json.load(open("scripts/ralph/prd-251w2.json"))
todo = [(s["id"], i + 1) for s in d["userStories"] for i, ac in enumerate(s["acceptanceCriteria"]) if "DONE" not in ac]
for sid, n in todo[:25]:
    print(f"   not DONE: {sid} AC{n}")
sys.exit(1 if todo else 0)
PY
}
# The pushed HEAD's test.yml run must carry every required job, each green.
# Polls up to 40 min. A newer push cancels an older run, so only HEAD's own run counts.
# "Code standards on changed lines" is marked non-required on pull requests; this run holds
# its own changes to it (AGENTS.md → Code shape).
REQUIRED_JOBS=("orchestrator-tests" "Alembic from-zero" "Schema-drift check" "Frontend CI" "Prod images built" "media-render" "Code standards on changed lines")
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
# The MinIO presign test must have RUN (not skipped) in HEAD's orchestrator-tests job.
minio_test_ran_on_head() {
  command -v gh >/dev/null || { echo "   gh is required"; return 1; }
  local br sha id log
  br=$(git rev-parse --abbrev-ref HEAD); sha=$(git rev-parse HEAD)
  id=$(gh run list --branch "$br" --workflow test.yml --limit 20 --json databaseId,headSha \
        --jq ".[] | select(.headSha == \"$sha\") | .databaseId" 2>/dev/null | head -1)
  [ -n "$id" ] || { echo "   no test.yml run for HEAD"; return 1; }
  log=$(gh run view "$id" --log 2>/dev/null | grep 'test_prd251w2_media_hosting' || true)
  echo "$log" | grep -q 'PASSED' || { echo "   test_prd251w2_media_hosting did not PASS in run $id"; return 1; }
  ! echo "$log" | grep -q 'SKIPPED' || { echo "   test_prd251w2_media_hosting SKIPPED in run $id"; return 1; }
}

# Nothing publishes in Wave 2: the Wave 0 seam still answers 501, and no publish jobs exist.
no_wave3_code() {
  git diff --quiet "$BASE"..HEAD -- orchestrator/modules/socials/publisher.py \
    && git grep -q 'PublishingUnavailable' -- orchestrator/modules/socials/publisher.py \
    && ! git grep -qE 'social-publish-|DateTrigger' -- orchestrator/modules/socials
}
# ── US-201 Wave 2's ONE migration ───────────────────────────────────────────
check "exactly one new alembic revision on the branch" "one_new_revision"
check "it chains onto the base's single alembic head" "migration_chains_onto_base_head"
check "the migration is create_all-first safe (guards present)" "migration_is_create_all_safe"
check "both head pins moved to the new revision" "head_pins_moved"
check "SocialCampaign model exists" "grep -q 'class SocialCampaign' orchestrator/core/models/socials.py"

# ── P251W1-RVW-2..7 Wave 1's review fixes (their tests are proven by CI) ─────
check "voice checks the media caps before it speaks (P251W1-RVW-2)" \
  "grep -qE 'check_caps|media_caps' $SOC/recipes/voice.py"
check "media-render bounds each workspace's active jobs, everywhere it is configured (P251W1-RVW-4)" \
  "grep -q 'MEDIA_RENDER_MAX_ACTIVE_JOBS_PER_WORKSPACE' services/media-render/media_render/config.py && grep -q 'MEDIA_RENDER_MAX_ACTIVE_JOBS_PER_WORKSPACE' docker-compose.yml && grep -q 'MEDIA_RENDER_MAX_ACTIVE_JOBS_PER_WORKSPACE' infrastructure/railway-manifest.json && grep -q 'MEDIA_RENDER_MAX_ACTIVE_JOBS_PER_WORKSPACE' services/media-render/README.md"
check "media-render refuses a busy workspace with workspace_busy (P251W1-RVW-4)" \
  "grep -q 'workspace_busy' services/media-render/media_render/server.py && grep -q 'workspace_busy' orchestrator/core/media_render_client.py"
check "config-surface lists Wave 1's two missing settings (P251W1-RVW-7)" \
  "grep -q '\"PLAYBOOK_PROGRESS_STAMP_SECONDS\"' $SURFACE && grep -q '\"SOCIALS_POST_ACTIONS_CACHE_TTL_SECONDS\"' $SURFACE"

# ── US-202 media hosting ─────────────────────────────────────────────────────
check "media_urls presigns inline with a content type and the config TTL" \
  "[ -f $SOC/media_urls.py ] && grep -q 'ResponseContentDisposition' $SOC/media_urls.py && grep -q 'inline' $SOC/media_urls.py && grep -q 'ResponseContentType' $SOC/media_urls.py && grep -q 'SOCIALS_MEDIA_URL_TTL_SECONDS' $SOC/media_urls.py"
check "CI starts MinIO and hands its endpoint to the tests" "minio_in_ci"
check "route: GET /api/socials/posts/{post_id}/media" "manifest_has GET '/api/socials/posts/{post_id}/media'"
check "no boto3 client built outside the storage factory" "no_boto3_outside_factory"
check "Socials media never goes through /api/generated-images" "no_generated_api_images"
check "self-hosting docs name SOCIALS_PUBLIC_MEDIA_BUCKET" "grep -q 'SOCIALS_PUBLIC_MEDIA_BUCKET' docs/getting-started/self-hosting.md"

# ── US-203 the registry ──────────────────────────────────────────────────────
check "the registry carries the publish class" "grep -q 'publish' $SOC/capabilities.py"
check "route: GET /api/socials/channels" "manifest_has GET '/api/socials/channels'"
check "UPLOAD_ACTIONS is not widened (file-first through the adapters' own upload spec)" "upload_actions_not_widened"
check "no stale or URL-pull slug is called" "no_forbidden_slugs"
check "the Wave 0 deny list is untouched" "deny_list_untouched"

# ── US-203 the registry's channel data (the publishers are Wave 3's) ─────────
check "the adapter data carries every channel's documented actions" "all_adapter_slugs_present"
check "X video goes through the chunked path" "x_video_path_present"
# ── US-204 channels are approved content ─────────────────────────────────────
check "the content hash covers the targets" "content_hash_covers_targets"
check "route: PUT /api/socials/posts/{post_id}/targets" "manifest_has PUT '/api/socials/posts/{post_id}/targets'"

# ── US-205 the tab ───────────────────────────────────────────────────────────
check "global search has the Socials page" "grep -q 'nav-socials' $FE/hooks/use-global-search.ts"
check "the posts list takes q" "grep -qE '(^|[ (,])q: ' $API"

# ── US-206 the approval UI ───────────────────────────────────────────────────
check "Socials media previews use the shared FilePreview" "git grep -q 'FilePreview' -- $FE/components/deliverables/socials"
check "approvers are notified (approval_pending)" "git grep -q 'approval_pending' -- $SOC $API"
check "the frontend links social_post notifications" "git grep -q 'social_post' -- $FE ':(exclude)$FE/**/__tests__/**'"

# ── US-207..US-209 the composer ──────────────────────────────────────────────
check "route: POST /api/socials/compose" "manifest_has POST '/api/socials/compose'"
check "the composer uses the platform LLM manager with its own request type" \
  "git grep -q 'create_llm_manager' -- $SOC $API && git grep -q 'socials_compose' -- $SOC $API"
check "preview renders exist" "git grep -qE 'preview' -- $SOC $API"

check "the compose timeout is in config.py and config-surface.json" "config_and_surface SOCIALS_COMPOSE_TIMEOUT_SECONDS"

# ── US-210 series approval ───────────────────────────────────────────────────
check "route: /api/socials/campaigns" "manifest_has POST '/api/socials/campaigns' && manifest_has GET '/api/socials/campaigns'"
check "route: POST /api/socials/campaigns/{campaign_id}/approve" "manifest_has POST '/api/socials/campaigns/{campaign_id}/approve'"
check "series approval is a workspace switch" "grep -q 'series_approval' $SOC/settings.py"

# ── nothing publishes yet ────────────────────────────────────────────────────
check "no Wave 3 code: the 501 seam stands, no publish jobs" "no_wave3_code"
check "apiClient calls every new route with the right HTTP verb" "api_client_verbs"

# ── scope and conventions ────────────────────────────────────────────────────
check "no new async def route over a sync Session in api/socials.py (F105)" "no_new_async_db_routes"
check "no env reads added outside config.py (tests excluded)" "no_env_reads_outside_config"
check "every commit is DCO-signed" "every_commit_signed"
check "no node_modules in the diff" "! git diff --name-only $BASE..HEAD | grep -q node_modules"
check "media files only as tiny fixtures" "fixtures_small"
check "the generated Auto skill seed is untouched" \
  "git diff --quiet $BASE..HEAD -- orchestrator/core/seeds/platform-management-skill.md"
check "no Phase 2 engagement code" "no_phase2_code"
check "at least six backend test files for this wave" \
  "[ \$(ls orchestrator/tests/test_prd251w2_*.py 2>/dev/null | wc -l) -ge 6 ]"
check "at least three new Socials frontend test files" \
  "[ \$(git diff --name-only --diff-filter=A $BASE..HEAD -- $FE | grep -ciE 'socials.*\.test\.tsx?$') -ge 3 ]"
check "every acceptance criterion in $KIT is marked DONE" "all_acs_done"

# ── tests: proven by CI on the pushed HEAD, never run here ───────────────────
check "CI: the seven required jobs present and green on HEAD" "ci_green_on_head"
check "CI: the MinIO presign test (Wave 2) ran and passed on HEAD" "minio_test_ran_on_head"

echo ""
if [ "$FAIL" -eq 0 ]; then echo "✅ PRD-251 Wave 2 acceptance: PASS"; else echo "❌ PRD-251 Wave 2 acceptance: FAIL"; fi
exit "$FAIL"
