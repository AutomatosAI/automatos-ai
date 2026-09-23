#!/bin/bash
# Acceptance gate — PRD-251 Socials, Wave 0 (US-001..US-009).
# Base: merge-base with origin/main, so an integration merge of main never makes
# main's changes read as this branch's (the PRD-232 gate-base trap).
# Nothing runs on this machine (owner rule: CI is the only gate): no pytest, no
# npm, no docker. The code checks are greps and git; the tests are proven by
# test.yml on the pushed HEAD, whose five PRD-251 jobs must be green.
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

BASE=$(git merge-base HEAD origin/main) || { echo "cannot resolve merge-base with origin/main"; exit 1; }
FAIL=0

# Each check body runs in a subshell: a `cd` inside one never leaks into the next.
check() {
  local name="$1" body="$2"
  echo ""
  echo "── $name"
  if ( eval "$body" ); then echo "   ✅ PASS: $name"; else echo "   ❌ FAIL: $name"; FAIL=1; fi
}

CFG=orchestrator/config.py
SET=orchestrator/modules/socials/settings.py
MIG=orchestrator/alembic/versions/prd251_socials.py
MODELS=orchestrator/core/models/socials.py
LI=orchestrator/core/composio/linkedin_image_workaround.py
MAN=orchestrator/reports/route-manifest.json
TABS=frontend/lib/deliverables/tabs.ts

# ── helpers for the checks that need more than one line ─────────────────────
migration_chain_ok() {
  grep -qE "^revision(: *str)? *= *['\"]prd251_socials['\"]" "$MIG" \
    && grep -qE "^down_revision.*kb_multimodal_tables" "$MIG"
}
one_new_revision() {
  local n
  n=$(git diff --name-only --diff-filter=A "$BASE"..HEAD -- orchestrator/alembic/versions | grep -c '\.py$')
  [ "$n" -eq 1 ]
}
manifest_has_action() {
  grep -qE "\"/api/socials/posts/\\{[a-z_]+\\}/$1\"" "$MAN"
}
linkedin_loader_takes_workspace() {
  grep -A4 'def _load_linkedin_credentials' "$LI" | grep -q 'workspace_id'
}
video_registrable() {
  python3 - <<'PY'
import re, sys
s = open("orchestrator/services/deliverable_service.py").read()
m = re.search(r"AGENT_REGISTERABLE_ARTIFACT_TYPES\s*=\s*frozenset\(\{(.*?)\}\)", s, re.S)
sys.exit(0 if m and re.search(r"['\"]video['\"]", m.group(1)) else 1)
PY
}
no_raw_fetch_in_socials_ui() {
  [ -d frontend/components/deliverables/socials ] || return 1
  ! grep -rnE --include='*.ts' --include='*.tsx' '(^|[^.a-zA-Z])fetch\(' frontend/components/deliverables/socials \
      | grep -v '__tests__' | grep -q .
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
# The pushed HEAD's test.yml run must be green on the five jobs PRD-251 touches.
# Polls up to 40 min (a full run takes ~10). A newer push cancels an older run,
# so only HEAD's own run counts.
ci_green_on_head() {
  command -v gh >/dev/null || { echo "   gh is required"; return 1; }
  local br sha remote waited=0 run="" status="" id jobs bad
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
  bad=$(echo "$jobs" | grep -E $'\t(orchestrator-tests|Alembic from-zero|Schema-drift check|Frontend CI|Prod images built)' | grep -v '^success' || true)
  [ -z "$bad" ] || { echo "   PRD-251 jobs not green:"; echo "$bad" | sed 's/^/     /'; return 1; }
  echo "   test.yml run $id is green on $sha for the five PRD-251 jobs"
}
no_wave1_code() {
  ! git diff --name-only "$BASE"..HEAD | grep -qE '^services/media-render/|media_render_client' \
    && ! git diff "$BASE"..HEAD | grep -qE '^\+.*social-publish-'
}

# ── S0.1 switches, config, permission ────────────────────────────────────────
for attr in SOCIALS_ENABLED_DEFAULT SOCIALS_MEDIA_URL_TTL_SECONDS SOCIALS_MISFIRE_GRACE_SECONDS \
            SOCIALS_MAX_TARGET_ATTEMPTS SOCIALS_RENDER_URL SOCIALS_PUBLIC_MEDIA_BUCKET; do
  check "config.py defines $attr" "grep -q '$attr' $CFG"
done
check "settings module: master switch, workspace parse, validation, route gate" \
  "grep -q 'def socials_master_enabled' $SET && grep -q 'def parse_workspace_socials' $SET && grep -q 'def validate_socials_update' $SET && grep -q 'require_socials_enabled' $SET"
check "socials:approve permission exists" "grep -q 'socials:approve' orchestrator/modules/policy/roles.py"
check "no plan exposure key (owner: every plan gets Socials)" \
  "git diff --quiet $BASE..HEAD -- orchestrator/services/plan_tiers.py"
check "workspace switch route exists" "grep -q '/current/socials' orchestrator/api/workspaces.py"

# ── S0.2 tables and ONE migration ────────────────────────────────────────────
check "migration prd251_socials chained onto kb_multimodal_tables" "migration_chain_ok"
check "exactly one new alembic revision on the branch" "one_new_revision"
check "no social_campaigns table (Wave 2, only if series approval ships)" \
  "! grep -q 'social_campaigns' $MIG $MODELS"
check "both head pins moved" \
  "grep -q 'EXPECTED_HEAD = \"prd251_socials\"' orchestrator/tests/test_prd209_alembic_single_head.py && grep -q 'prd251_socials' orchestrator/tests/test_prd236_w1_routes.py"

# ── S0.3 routes in the COMMITTED manifest ────────────────────────────────────
check "route manifest lists /api/socials/posts and the workspace switch" \
  "grep -q '\"/api/socials/posts\"' $MAN && grep -q '\"/api/workspaces/current/socials\"' $MAN"
for action in submit approve request-changes reject schedule unschedule publish-now; do
  check "route manifest lists the $action action" "manifest_has_action $action"
done

# ── S0.4 the three prerequisite fixes ────────────────────────────────────────
check "LinkedIn loader takes the workspace" "linkedin_loader_takes_workspace"
check "no process-global single LinkedIn credential remains" \
  "! grep -qE '^_cached_creds: *Optional\\[Dict' $LI"
check "api/composio.py is on the workaround's removal checklist" "grep -q 'api/composio.py' $LI"
check "video is agent-registrable" "video_registrable"
check "video is a frontend Deliverable type" "grep -qE \"['\\\"]video['\\\"]\" frontend/components/icons/deliverable-icon.tsx"
check "image lookup no longer decided by one 1000-key page" \
  "! grep -q 'MaxKeys=1000' orchestrator/core/services/image_store.py || grep -qE 'ContinuationToken|get_paginator' orchestrator/core/services/image_store.py"
check "generated-images route honours Range (206)" \
  "grep -qi 'range' orchestrator/api/generated_images.py && grep -q '206' orchestrator/api/generated_images.py"

# ── S0.5 the bare tab ────────────────────────────────────────────────────────
check "socials is a Deliverables tab" "grep -qE \"['\\\"]socials['\\\"]\" $TABS"
check "apiClient carries the socials calls" "grep -q '/api/socials/posts' frontend/lib/api-client.ts"
check "no raw fetch in the socials components" "no_raw_fetch_in_socials_ui"

# ── S0.6 the Composio deny list ──────────────────────────────────────────────
check "the deny list is seeded with the real-money Higgsfield action" \
  "grep -rq --include='*.py' 'HIGGSFIELD_MCP_CONFIRM_BILLING_PURCHASE' orchestrator"
check "the deny list is a system setting (data, not a code constant)" \
  "grep -rq --include='*.py' 'denied_actions' orchestrator/core/composio orchestrator/modules orchestrator/api"
check "the deny-list tests exist" "[ -f orchestrator/tests/test_prd251_composio_deny.py ]"

# ── scope and conventions ────────────────────────────────────────────────────
check "no env reads added outside config.py (tests excluded)" "no_env_reads_outside_config"
check "every commit is DCO-signed" "every_commit_signed"
check "no node_modules in the diff" "! git diff --name-only $BASE..HEAD | grep -q node_modules"
check "no Wave 1+ code (media-render, render client, publish jobs)" "no_wave1_code"
check "at least seven PRD-251 backend test files" \
  "[ \$(ls orchestrator/tests/test_prd251_*.py 2>/dev/null | wc -l) -ge 7 ]"

# ── tests: proven by CI on the pushed HEAD, never run here ───────────────────
check "at least one committed PRD-251 frontend test file" \
  "git ls-files frontend | grep -qiE 'socials.*\.test\.tsx?$'"
check "CI: orchestrator-tests, alembic-from-zero, schema-drift, Frontend CI and the prod image build green on HEAD" "ci_green_on_head"

echo ""
if [ "$FAIL" -eq 0 ]; then echo "✅ PRD-251 Wave 0 acceptance: PASS"; else echo "❌ PRD-251 Wave 0 acceptance: FAIL"; fi
exit "$FAIL"
