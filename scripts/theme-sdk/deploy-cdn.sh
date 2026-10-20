#!/usr/bin/env bash
# Theme SDK types: CDN deploy (PRD-261 A-10d, Gerard D2-b)
# ========================================================
#
# Packs frontend/packages/theme-sdk (npm pack) and uploads the tarball to the
# widget SDK's S3 bucket + CloudFront distribution (widgets.automatos.app),
# the same way automatos-widget-sdk/scripts/deploy-cdn.sh ships the loader:
#
#   s3://<bucket>/theme-sdk/v<version>/theme-sdk.tgz   immutable, max-age 1 year,
#                                                       never overwritten
#   s3://<bucket>/theme-sdk/v<major>/theme-sdk.tgz     major alias, max-age 3600
#
# then invalidates CloudFront on the alias only, waits for the invalidation to
# complete, checks that both URLs serve this tarball's bytes, and last writes
#   s3://<bucket>/theme-sdk/v<version>/published.json  the completion marker
# Theme authors install with
#   npm install https://widgets.automatos.app/theme-sdk/v1/theme-sdk.tgz
#
# A version counts as published only once its marker exists. A run that dies
# after the immutable upload (alias PUT, invalidation or smoke test failed)
# leaves no marker, so the next run resumes: it downloads the immutable copy
# (never re-packs over it) and redoes the alias, the invalidation and the checks.
#
# The version is package.json's "version" (= sdkVersion in
# frontend/lib/theme-host/shared-modules.json): the tarball carries it, so the
# script takes no version argument and cannot file a tarball under another one.
#
# Usage:
#   scripts/theme-sdk/deploy-cdn.sh [--dry-run]
#
# Exit codes: 0 deployed or resumed (or dry run), 1 error, 2 usage or bad version,
#             3 this version is already published, marker and all (CI: a skip).
#
# Auth: AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY in the shell, or an AWS
# profile. Unlike the widget script it never sources an .env file.
#
# Optional env (defaults match the widget SDK's):
#   AWS_SDK_DEPLOY_BUCKET           default: automatos-widget-sdk
#   AWS_SDK_DEPLOY_DISTRIBUTION_ID  default: auto-detect from CloudFront
#   AWS_SDK_DEPLOY_REGION           default: us-east-1
#   AWS_SDK_DEPLOY_DOMAIN           default: widgets.automatos.app
#
# Prerequisites: aws CLI v2, npm (Node 20+), curl, sha256sum or shasum.

set -euo pipefail

readonly EXIT_ERROR=1
readonly EXIT_USAGE=2
readonly EXIT_ALREADY_PUBLISHED=3
readonly PREFIX="theme-sdk"
readonly OBJECT_NAME="theme-sdk.tgz"
readonly MARKER_NAME="published.json"
readonly CONTENT_TYPE="application/gzip"
readonly CONTENT_TYPE_MARKER="application/json"
readonly CACHE_IMMUTABLE="public, max-age=31536000, immutable"
readonly CACHE_ALIAS="public, max-age=3600"
readonly CACHE_MARKER="no-cache"
readonly VERSION_RE='^[0-9]+\.[0-9]+\.[0-9]+(-[A-Za-z0-9.]+)?$'
readonly SMOKE_TRIES=24
readonly SMOKE_WAIT_SECONDS=5

BUCKET="${AWS_SDK_DEPLOY_BUCKET:-automatos-widget-sdk}"
DIST_ID="${AWS_SDK_DEPLOY_DISTRIBUTION_ID:-}"
REGION="${AWS_SDK_DEPLOY_REGION:-us-east-1}"
DOMAIN="${AWS_SDK_DEPLOY_DOMAIN:-widgets.automatos.app}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-$REGION}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PKG_DIR="$REPO_ROOT/frontend/packages/theme-sdk"
DRY_RUN=false
WORK_DIR=""

log() { echo "[theme-sdk-cdn] $*"; }
fail() { echo "ERROR: $2" >&2; exit "$1"; }

# Runs a write, or prints it on --dry-run. Arguments stay an array: nothing is eval'd.
run() {
  if $DRY_RUN; then
    log "DRY-RUN: $*"
  else
    log "RUN:     $*"
    "$@"
  fi
}

parse_args() {
  case "${1:-}" in
    "") ;;
    --dry-run) DRY_RUN=true ;;
    *) fail "$EXIT_USAGE" "usage: $0 [--dry-run] (the version is package.json's, got: $1)" ;;
  esac
}

preflight() {
  local tool
  for tool in aws npm curl; do
    command -v "$tool" >/dev/null 2>&1 || fail "$EXIT_ERROR" "$tool is not installed"
  done
  command -v sha256sum >/dev/null 2>&1 || command -v shasum >/dev/null 2>&1 \
    || fail "$EXIT_ERROR" "neither sha256sum nor shasum is installed"
  [[ -f "$PKG_DIR/package.json" ]] || fail "$EXIT_ERROR" "no package at $PKG_DIR"
  aws sts get-caller-identity >/dev/null 2>&1 \
    || fail "$EXIT_ERROR" "AWS credentials not configured (set AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY, or a profile)"
  log "AWS identity: $(aws sts get-caller-identity --query Arn --output text)"
}

# package.json's version; npm prints it as a JSON string ("1.0.0").
read_version() {
  local raw
  raw="$(cd "$PKG_DIR" && npm pkg get version)"
  raw="${raw//\"/}"
  [[ "$raw" =~ $VERSION_RE ]] || fail "$EXIT_USAGE" "package.json version must look like 1.2.3 or 1.2.3-beta.1, got: $raw"
  echo "$raw"
}

detect_distribution() {
  [[ -n "$DIST_ID" ]] && return 0
  log "Auto-detecting the CloudFront distribution for $DOMAIN..."
  DIST_ID="$(aws cloudfront list-distributions \
    --query "DistributionList.Items[?Aliases.Items != null] | [?contains(Aliases.Items, '$DOMAIN')].Id | [0]" \
    --output text 2>/dev/null || true)"
  if [[ -z "$DIST_ID" || "$DIST_ID" == "None" ]]; then
    fail "$EXIT_ERROR" "could not auto-detect the distribution for $DOMAIN; set AWS_SDK_DEPLOY_DISTRIBUTION_ID"
  fi
  log "Distribution: $DIST_ID"
}

# 0 when the key exists, 1 on a 404. Any other error (403 without ListBucket,
# throttling) stops the deploy: fail closed.
object_exists() {
  local key="$1" err
  if err="$(aws s3api head-object --bucket "$BUCKET" --key "$key" 2>&1 >/dev/null)"; then
    return 0
  fi
  grep -qE '\(404\)|Not Found' <<<"$err" || fail "$EXIT_ERROR" "could not check s3://$BUCKET/$key: $err"
  return 1
}

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

# npm pack into an empty folder, so the one .tgz there is this package's tarball.
pack_tarball() {
  local dest="$1" tarballs
  (cd "$PKG_DIR" && npm pack --ignore-scripts --pack-destination "$dest" >/dev/null)
  shopt -s nullglob
  tarballs=("$dest"/*.tgz)
  shopt -u nullglob
  [[ ${#tarballs[@]} -eq 1 ]] || fail "$EXIT_ERROR" "npm pack produced ${#tarballs[@]} tarballs in $dest"
  echo "${tarballs[0]}"
}

# Resume: the alias must carry the bytes already published as immutable, so the
# tarball comes from S3 (a read, also on --dry-run), not from a fresh npm pack.
fetch_immutable() {
  local dest="$1/$OBJECT_NAME" key="$2"
  aws s3api get-object --bucket "$BUCKET" --key "$key" "$dest" >/dev/null \
    || fail "$EXIT_ERROR" "could not download s3://$BUCKET/$key to resume"
  echo "$dest"
}

# The immutable copy: --if-none-match '*' makes S3 itself refuse an existing key,
# so two concurrent deploys of one version cannot both write it.
upload_immutable() {
  local file="$1" key="$2" err
  if $DRY_RUN; then
    log "DRY-RUN: aws s3api put-object --bucket $BUCKET --key $key (immutable, if-none-match *)"
    return 0
  fi
  log "Uploading immutable: s3://$BUCKET/$key"
  if ! err="$(aws s3api put-object --bucket "$BUCKET" --key "$key" --body "$file" \
    --content-type "$CONTENT_TYPE" --cache-control "$CACHE_IMMUTABLE" \
    --if-none-match '*' 2>&1 >/dev/null)"; then
    if grep -qE 'PreconditionFailed|\(412\)' <<<"$err"; then
      fail "$EXIT_ALREADY_PUBLISHED" "v$VERSION already published at s3://$BUCKET/$key (lost a concurrent upload)"
    fi
    fail "$EXIT_ERROR" "upload of s3://$BUCKET/$key failed: $err"
  fi
}

upload_alias() {
  local file="$1" key="$2"
  log "Updating the major alias: s3://$BUCKET/$key -> v$VERSION"
  run aws s3api put-object --bucket "$BUCKET" --key "$key" --body "$file" \
    --content-type "$CONTENT_TYPE" --cache-control "$CACHE_ALIAS"
}

# Only the alias changes content under the same URL, so only it is invalidated.
# create-invalidation returns at once; the waiter blocks until every edge has
# dropped the old alias (it polls every 20 s, up to 10 min, then fails the run).
invalidate_alias() {
  local path="/$PREFIX/$MAJOR_ALIAS/*" inv_id
  log "Invalidating CloudFront $DIST_ID at $path"
  if $DRY_RUN; then
    log "DRY-RUN: aws cloudfront create-invalidation --distribution-id $DIST_ID --paths $path (then wait)"
    return 0
  fi
  inv_id="$(aws cloudfront create-invalidation --distribution-id "$DIST_ID" \
    --paths "$path" --query 'Invalidation.Id' --output text)"
  log "Invalidation: $inv_id, waiting for it to complete..."
  aws cloudfront wait invalidation-completed --distribution-id "$DIST_ID" --id "$inv_id" \
    || fail "$EXIT_ERROR" "CloudFront invalidation $inv_id did not complete"
  log "Invalidation $inv_id completed"
}

# A 200 is not enough: an edge can still answer the alias with the previous
# version. The served bytes must hash to the tarball this run published.
smoke_test() {
  local url="$1" want="$2" body="$WORK_DIR/served" code="" got="" i
  if $DRY_RUN; then
    log "DRY-RUN: curl $url (expect sha256 $want)"
    return 0
  fi
  for ((i = 1; i <= SMOKE_TRIES; i++)); do
    rm -f "$body"
    code="$(curl -s -o "$body" -w '%{http_code}' "$url" || true)"
    got=""
    [[ "$code" == "200" && -f "$body" ]] && got="$(sha256_of "$body")"
    if [[ "$got" == "$want" ]]; then
      log "$url is live with v$VERSION (HTTP 200, sha256 $want)"
      return 0
    fi
    log "attempt $i: HTTP $code, sha256 ${got:-none} (want $want) for $url, retrying in ${SMOKE_WAIT_SECONDS}s"
    sleep "$SMOKE_WAIT_SECONDS"
  done
  fail "$EXIT_ERROR" "smoke test failed for $url (last: HTTP $code, sha256 ${got:-none}, want $want)"
}

# Written last: its presence is what makes the next run exit 3.
write_marker() {
  local key="$1" sha="$2" file="$WORK_DIR/$MARKER_NAME"
  printf '{"version":"%s","alias":"%s","sha256":"%s"}\n' "$VERSION" "$MAJOR_ALIAS" "$sha" >"$file"
  log "Marking v$VERSION published: s3://$BUCKET/$key"
  run aws s3api put-object --bucket "$BUCKET" --key "$key" --body "$file" \
    --content-type "$CONTENT_TYPE_MARKER" --cache-control "$CACHE_MARKER"
}

# The tarball to publish: a fresh pack (uploaded as immutable), or on resume the
# immutable copy already in S3. It runs in $(...), where bash drops `set -e`, so
# every step that can fail exits explicitly.
prepare_tarball() {
  local immutable_key="$1" tarball
  if object_exists "$immutable_key"; then
    log "Resuming: s3://$BUCKET/$immutable_key exists but v$VERSION has no marker; redoing the alias, invalidation and checks" >&2
    fetch_immutable "$WORK_DIR" "$immutable_key" || exit $?
    return 0
  fi
  tarball="$(pack_tarball "$WORK_DIR")" || exit $?
  log "Packed: $(basename "$tarball") ($(wc -c <"$tarball" | tr -d ' ') bytes)" >&2
  upload_immutable "$tarball" "$immutable_key" >&2
  echo "$tarball"
}

main() {
  parse_args "$@"
  log "Pre-flight checks..."
  preflight
  VERSION="$(read_version)"
  MAJOR_ALIAS="v${VERSION%%.*}"
  local immutable_key="$PREFIX/v$VERSION/$OBJECT_NAME"
  local alias_key="$PREFIX/$MAJOR_ALIAS/$OBJECT_NAME"
  local marker_key="$PREFIX/v$VERSION/$MARKER_NAME"
  log "Version: v$VERSION (alias $MAJOR_ALIAS)"
  detect_distribution
  if object_exists "$marker_key"; then
    fail "$EXIT_ALREADY_PUBLISHED" "v$VERSION already published (s3://$BUCKET/$marker_key; versions are immutable: bump package.json and sdkVersion together)"
  fi

  local tarball sha
  WORK_DIR="$(mktemp -d)"
  trap 'rm -rf "$WORK_DIR"' EXIT
  tarball="$(prepare_tarball "$immutable_key")"
  sha="$(sha256_of "$tarball")"

  upload_alias "$tarball" "$alias_key"
  invalidate_alias
  smoke_test "https://$DOMAIN/$immutable_key" "$sha"
  smoke_test "https://$DOMAIN/$alias_key" "$sha"
  write_marker "$marker_key" "$sha"

  log "=== Done ==="
  log "Immutable: https://$DOMAIN/$immutable_key"
  log "Alias:     https://$DOMAIN/$alias_key"
  log "Install:   npm install https://$DOMAIN/$alias_key"
}

main "$@"
