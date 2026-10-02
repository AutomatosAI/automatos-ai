#!/bin/bash
# Acceptance gate — PRD-251 Socials, Wave 3: scheduling and publishing (US-301..US-309).
#   bash scripts/ralph/acceptance-prd251w3.sh               run the gate
#   bash scripts/ralph/acceptance-prd251w3.sh --print-base  print the diff base
# Base: stacked on Wave 2 (feat/prd-251-w2-socials-tab) until Wave 2 is on main,
# then main. The base is the NEWER of the two fork points. The loop never merges
# either branch in, so neither side's changes can read as this run's.
# Nothing runs on the owner's machine (owner rule: CI is the only gate): no pytest,
# no npm, no docker. The code checks are greps and git; the tests are proven by
# test.yml on the pushed HEAD, whose seven required jobs must be present and green.
# No `pipefail` on purpose: with it, `! producer | grep -q X` can PASS when X matched
# (grep -q exits on the first match, the producer takes SIGPIPE, pipefail reports 141
# and the `!` turns that into success). BASE is validated before any check runs.
set -u
cd "$(dirname "$0")/../.." || exit 1

export DATABASE_URL="postgresql://ralph:ralph@127.0.0.1:1/ralph_no_db"
export POSTGRES_HOST=127.0.0.1 POSTGRES_PORT=1 POSTGRES_USER=ralph POSTGRES_PASSWORD=ralph POSTGRES_DB=ralph_no_db
export REDIS_URL="redis://127.0.0.1:1/0" REDIS_HOST=127.0.0.1 REDIS_PORT=1
export ENVIRONMENT=development

W2_REF=origin/feat/prd-251-w2-socials-tab
BASE=$(git merge-base HEAD origin/main) || { echo "cannot resolve merge-base with origin/main"; exit 1; }
if git rev-parse -q --verify "$W2_REF" >/dev/null; then
  W2_BASE=$(git merge-base HEAD "$W2_REF" 2>/dev/null || true)
  if [ -n "$W2_BASE" ] && [ "$W2_BASE" != "$BASE" ] && git merge-base --is-ancestor "$BASE" "$W2_BASE"; then
    BASE="$W2_BASE"
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
KIT=scripts/ralph/prd-251w3.json

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
want = [("POST", r"/api/socials/posts/\*/publish-now"), ("POST", r"/api/socials/posts/\*/retry"),
        ("POST", r"/api/socials/posts/\*/schedule"), ("POST", r"/api/socials/posts/\*/unschedule")]
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
d = json.load(open("scripts/ralph/prd-251w3.json"))
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
# ── Wave 3 helpers ───────────────────────────────────────────────────────────
PUB_FILES="$SOC/publisher.py $(ls $SOC/publishing*.py $SOC/publish_*.py 2>/dev/null | tr '\n' ' ')"
# The Wave 0 seam is filled: no PublishingUnavailable left, and the guard runs first.
seam_filled() {
  ! git grep -q 'PublishingUnavailable' -- orchestrator ':(exclude)orchestrator/tests' \
    && git grep -q 'assert_publishable' -- $PUB_FILES
}
publisher_guard_first() {
  git grep -q 'assert_publishable' -- $PUB_FILES && git grep -q 'assert_retryable' -- $PUB_FILES
}
# Only publishing code (and the gate that defines it) names the way through.
way_through_only_in_publisher() {
  local hits
  hits=$(git grep -l 'PLATFORM_PUBLISHER' -- orchestrator ':(exclude)orchestrator/tests' \
         | grep -vE '^orchestrator/core/composio/(post_gate|tool_executor)\.py$' \
         | grep -vE '^orchestrator/modules/socials/(publisher|publishing[a-z_]*|publish_[a-z_]+)\.py$' || true)
  [ -z "$hits" ] || { echo "   PLATFORM_PUBLISHER referenced outside the publisher:$(echo $hits)"; return 1; }
  git grep -q 'PLATFORM_PUBLISHER' -- $PUB_FILES || { echo "   the publisher never passes PLATFORM_PUBLISHER"; return 1; }
}
# One engine: no channel names branched on in the publishing code.
no_channel_branches() {
  ! git grep -nE "(==|!=|in \(|\bcase\b).{0,6}['\"](linkedin|twitter|instagram|tiktok|youtube)['\"]" -- $PUB_FILES | grep -q .
}
# Channel slugs only in the adapter data (tests and docs excluded).
slugs_only_in_adapter_data() {
  ! added_code_lines | grep -E '(LINKEDIN|TWITTER|INSTAGRAM|TIKTOK|YOUTUBE)_[A-Z_]{4,}' \
      | grep -vE 'deprecat|never|stale|not used|unused|forbid' | grep -q . \
    || { git diff -U0 "$BASE"..HEAD -- orchestrator ':(exclude)orchestrator/tests' ':(exclude)orchestrator/modules/socials/channel_adapters.py' \
         | grep -E '^\+[^+]' | grep -E '(LINKEDIN|TWITTER|INSTAGRAM|TIKTOK|YOUTUBE)_[A-Z_]{4,}' \
         | grep -viqE 'deprecat|never|stale|not used|unused|forbid' && { echo "   channel slug outside channel_adapters.py"; return 1; }; return 0; }
}
scheduling_jobs() {
  git grep -q 'social-publish-' -- "$SOC" orchestrator/services \
    && git grep -q 'DateTrigger' -- "$SOC" orchestrator/services/schedule_reconcile.py orchestrator/services/scheduled_task_service.py \
    && git grep -qE 'social' -- orchestrator/services/schedule_reconcile.py
}
missed_handled() {
  git grep -q 'SOCIALS_MISFIRE_GRACE_SECONDS' -- "$SOC" ':(exclude)orchestrator/tests' \
    && git grep -qE "MISSED|'missed'|\"missed\"" -- "$SOC/service.py"
}
calendar_source() {
  git grep -q '_social_post_items' -- orchestrator/services && git grep -q 'social-' -- orchestrator/services
}
activity_service_not_grown_much() {
  local added; added=$(git diff --numstat "$BASE"..HEAD -- orchestrator/services/activity_service.py | cut -f1)
  [ -z "$added" ] || [ "$added" -le 15 ] || { echo "   activity_service.py grew by $added lines: put the source in its own module"; return 1; }
}
no_live_composio_in_tests() {
  ! git grep -nE 'COMPOSIO_(API_)?KEY\s*=\s*["\x27][A-Za-z0-9]{12,}' -- orchestrator/tests | grep -q .
}
no_new_python_dependency() {
  git diff --quiet "$BASE"..HEAD -- orchestrator/requirements.txt
}
reaper_covers_publishing() {
  grep -qiE 'publishing' orchestrator/core/boot/reaper.py
}

# ── US-301 the publisher ─────────────────────────────────────────────────────
check "the Wave 0 seam is filled: no PublishingUnavailable, the guard runs first" "seam_filled"
check "retry checks the approval too (assert_publishable and assert_retryable)" "publisher_guard_first"
check "only the publisher passes PLATFORM_PUBLISHER" "way_through_only_in_publisher"
check "one engine: the publisher never branches on a channel's name" "no_channel_branches"
check "UPLOAD_ACTIONS is not widened (file-first through each step's own upload spec)" "upload_actions_not_widened"
check "no stale or URL-pull slug is called" "no_forbidden_slugs"
check "channel slugs live only in the adapter data" "slugs_only_in_adapter_data"
check "the deny list is untouched" "deny_list_untouched"
check "the publish settings are in config.py and config-surface.json" \
  "config_and_surface SOCIALS_PUBLISH_POLL_SECONDS SOCIALS_PUBLISH_MAX_WAIT_SECONDS SOCIALS_PUBLISH_RETRY_BACKOFF_SECONDS SOCIALS_MAX_TARGET_ATTEMPTS"
check "route: POST /api/socials/posts/{post_id}/retry" "manifest_has POST '/api/socials/posts/{post_id}/retry'"
check "route: POST /api/socials/posts/{post_id}/publish-now" "manifest_has POST '/api/socials/posts/{post_id}/publish-now'"
check "failed and missed posts notify (new event types)" \
  "grep -q 'social_post_failed' orchestrator/core/services/notification_dispatcher.py && grep -q 'social_post_missed' orchestrator/core/services/notification_dispatcher.py"
check "the boot reaper covers a post stuck in publishing" "reaper_covers_publishing"
check "Socials media never goes through /api/generated-images" "no_generated_api_images"

# ── US-302..US-305 the channels (data; the tests are proven by CI) ───────────
check "the adapter data carries every channel's documented actions" "all_adapter_slugs_present"
check "X video goes through the chunked path" "x_video_path_present"
check "a test file per channel" \
  "for c in linkedin x instagram tiktok youtube; do ls orchestrator/tests/test_prd251w3_*\$c*.py >/dev/null 2>&1 || { echo \"   no test_prd251w3_*\$c*.py\"; exit 1; }; done"

# ── US-306 scheduling ────────────────────────────────────────────────────────
check "scheduled posts get social-publish-<id> DateTrigger jobs, kept by the reconcile pass" "scheduling_jobs"
check "a slot missed beyond the grace ends missed" "missed_handled"

# ── US-307 the calendar ──────────────────────────────────────────────────────
check "the calendar has a social source (social-<post_id>)" "calendar_source"
check "activity_service.py only wires the source in" "activity_service_not_grown_much"
check "the three calendar union files carry 'social'" "calendar_unions_have_social"
check "no drag-and-drop dependency" "no_new_drag_dependency"

# ── US-308 the publish controls ──────────────────────────────────────────────
check "apiClient calls the publish, retry and schedule routes with the right verbs" "api_client_verbs"

# ── US-309 one way out, the docs ─────────────────────────────────────────────
check "self-hosting docs say what publishing needs" \
  "grep -q 'SOCIALS_PUBLIC_MEDIA_BUCKET' docs/getting-started/self-hosting.md && grep -qi 'X API\\|own X app\\|X app' docs/getting-started/self-hosting.md"
check "the owner's test exists" "[ -f docs/PRDS/prd251-w3-owner-test.md ]"

# ── scope and conventions ────────────────────────────────────────────────────
check "no new alembic revision (or exactly one, create_all-first safe, head pins moved)" \
  "[ \$(new_revision_file | grep -c .) -eq 0 ] || { one_new_revision && migration_chains_onto_base_head && migration_is_create_all_safe && head_pins_moved; }"
check "no new async def route over a sync Session in api/socials.py (F105)" "no_new_async_db_routes"
check "no env reads added outside config.py (tests excluded)" "no_env_reads_outside_config"
check "no new Python dependency" "no_new_python_dependency"
check "no live Composio credentials in tests" "no_live_composio_in_tests"
check "every commit is DCO-signed" "every_commit_signed"
check "no node_modules in the diff" "! git diff --name-only $BASE..HEAD | grep -q node_modules"
check "media files only as tiny fixtures" "fixtures_small"
check "the generated Auto skill seed is untouched" \
  "git diff --quiet $BASE..HEAD -- orchestrator/core/seeds/platform-management-skill.md"
check "no Phase 2 engagement code" "no_phase2_code"
check "at least eight backend test files for this wave" \
  "[ \$(ls orchestrator/tests/test_prd251w3_*.py 2>/dev/null | wc -l) -ge 8 ]"
check "at least two new Socials or calendar frontend test files" \
  "[ \$(git diff --name-only --diff-filter=A $BASE..HEAD -- $FE | grep -ciE '(socials|calendar).*\.test\.tsx?$') -ge 2 ]"
check "every acceptance criterion in $KIT is marked DONE" "all_acs_done"

# ── tests: proven by CI on the pushed HEAD, never run here ───────────────────
check "CI: the seven required jobs present and green on HEAD" "ci_green_on_head"

echo ""
if [ "$FAIL" -eq 0 ]; then echo "✅ PRD-251 Wave 3 acceptance: PASS"; else echo "❌ PRD-251 Wave 3 acceptance: FAIL"; fi
exit "$FAIL"
