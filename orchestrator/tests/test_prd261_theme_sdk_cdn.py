"""PRD-261 A-10d (Gerard D2-b): the Theme SDK types ship the widget SDK's way.

``scripts/theme-sdk/deploy-cdn.sh`` packs ``frontend/packages/theme-sdk`` and
uploads the tarball to ``theme-sdk/v<version>/theme-sdk.tgz`` (immutable, one
year, never overwritten) and ``theme-sdk/v<major>/theme-sdk.tgz`` (alias, one
hour), invalidates CloudFront on the alias only and waits for it, checks that
both URLs serve the tarball's bytes, and last writes the completion marker
``theme-sdk/v<version>/published.json``. Only the marker means "published": a run
that died after the immutable upload is resumed, not skipped.
``.github/workflows/theme-sdk-cdn.yml`` runs it on main and by hand, and exits 0
with a notice while the five ``AWS_SDK_DEPLOY_*`` secrets are absent.

The script runs here against fake ``aws``, ``npm``, ``curl`` and ``sleep`` executables put
first on ``PATH``: each logs its arguments, so the tests assert what would be
sent to AWS without credentials or network.
"""
from __future__ import annotations

import json
import os
import pathlib
import stat
import subprocess
from typing import Dict, List

import pytest
import yaml

_REPO = pathlib.Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts" / "theme-sdk" / "deploy-cdn.sh"
_WORKFLOW = _REPO / ".github" / "workflows" / "theme-sdk-cdn.yml"
_PACKAGE = _REPO / "frontend" / "packages" / "theme-sdk"
_SECRETS = (
    "AWS_SDK_DEPLOY_ACCESS_KEY_ID",
    "AWS_SDK_DEPLOY_SECRET_ACCESS_KEY",
    "AWS_SDK_DEPLOY_REGION",
    "AWS_SDK_DEPLOY_BUCKET",
    "AWS_SDK_DEPLOY_DISTRIBUTION_ID",
)
_SEP = "\x1f"
_BUCKET = "test-bucket"
_DIST = "E2TESTDIST"
_INSTALL_LINE = "npm install https://widgets.automatos.app/theme-sdk/v1/theme-sdk.tgz"
_ALREADY_PUBLISHED = 3
_IMMUTABLE = "theme-sdk/v1.4.2/theme-sdk.tgz"
_ALIAS = "theme-sdk/v1/theme-sdk.tgz"
_MARKER = "theme-sdk/v1.4.2/published.json"
_ALIAS_URL = f"https://widgets.automatos.app/{_ALIAS}"

_FAKE_AWS = r"""#!/usr/bin/env bash
(IFS=$'\x1f'; printf 'aws\x1f%s\n' "$*") >> "$FAKE_LOG"
key=""; prev=""
for arg in "$@"; do
  if [ "$prev" = "--key" ]; then key="$arg"; fi
  prev="$arg"
done
state="$FAKE_HEAD"
if [ "$state" = "partial" ]; then
  case "$key" in *published.json) state="missing" ;; *) state="found" ;; esac
fi
case "$1 $2" in
  "sts get-caller-identity") echo "arn:aws:iam::000000000000:user/ci" ;;
  "s3api head-object")
    case "$state" in
      found) echo '{"ContentLength": 3}' ;;
      forbidden) echo "An error occurred (403) when calling the HeadObject operation: Forbidden" >&2; exit 254 ;;
      *) echo "An error occurred (404) when calling the HeadObject operation: Not Found" >&2; exit 254 ;;
    esac ;;
  "s3api get-object") printf 'tgz' > "${@: -1}"; echo '{"ContentLength": 3}' ;;
  "s3api put-object") echo '{"ETag": "\"x\""}' ;;
  "cloudfront create-invalidation") echo "I2TESTINV" ;;
  "cloudfront wait") if [ "$FAKE_WAIT" = "fail" ]; then echo "Waiter failed" >&2; exit 255; fi ;;
esac
"""

_FAKE_NPM = r"""#!/usr/bin/env bash
(IFS=$'\x1f'; printf 'npm\x1f%s\n' "$*") >> "$FAKE_LOG"
case "$1" in
  pkg) printf '"%s"\n' "$FAKE_VERSION" ;;
  pack)
    dest=""
    while [ $# -gt 0 ]; do
      if [ "$1" = "--pack-destination" ]; then dest="$2"; fi
      shift
    done
    printf 'tgz' > "$dest/automatos-theme-sdk-$FAKE_VERSION.tgz"
    echo "automatos-theme-sdk-$FAKE_VERSION.tgz" ;;
esac
"""

# Serves the fake tarball's bytes ("tgz"), or the previous version's ("old") for
# FAKE_STALE_URL: an edge still holding the old alias.
_FAKE_CURL = r"""#!/usr/bin/env bash
(IFS=$'\x1f'; printf 'curl\x1f%s\n' "$*") >> "$FAKE_LOG"
out=""; prev=""
for arg in "$@"; do
  if [ "$prev" = "-o" ]; then out="$arg"; fi
  prev="$arg"
done
body="tgz"
if [ "${@: -1}" = "$FAKE_STALE_URL" ]; then body="old"; fi
if [ -n "$out" ]; then printf '%s' "$body" > "$out"; fi
printf '200'
"""

_FAKE_SLEEP = r"""#!/usr/bin/env bash
(IFS=$'\x1f'; printf 'sleep\x1f%s\n' "$*") >> "$FAKE_LOG"
"""


def _write_tool(folder: pathlib.Path, name: str, body: str) -> None:
    tool = folder / name
    tool.write_text(body, encoding="utf-8")
    tool.chmod(tool.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _deploy(
    tmp_path: pathlib.Path,
    *args: str,
    version: str = "1.4.2",
    head: str = "missing",
    stale_url: str = "",
    wait: str = "ok",
):
    """Runs the deploy script with the fakes; returns (result, logged calls).

    ``head``: missing (nothing on S3), found (published, marker and all),
    partial (immutable copy but no marker: a failed run), forbidden.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    fakes = (("aws", _FAKE_AWS), ("npm", _FAKE_NPM), ("curl", _FAKE_CURL), ("sleep", _FAKE_SLEEP))
    for name, body in fakes:
        _write_tool(bin_dir, name, body)
    log = tmp_path / "calls.log"
    env: Dict[str, str] = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "HOME": str(tmp_path),
        "FAKE_LOG": str(log),
        "FAKE_VERSION": version,
        "FAKE_HEAD": head,
        "FAKE_STALE_URL": stale_url,
        "FAKE_WAIT": wait,
        "AWS_SDK_DEPLOY_BUCKET": _BUCKET,
        "AWS_SDK_DEPLOY_DISTRIBUTION_ID": _DIST,
    }
    result = subprocess.run(
        ["bash", str(_SCRIPT), *args], env=env, capture_output=True, text=True, timeout=60
    )
    lines = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    return result, [line.split(_SEP) for line in lines]


def _calls(calls: List[List[str]], *prefix: str) -> List[List[str]]:
    return [c for c in calls if c[: len(prefix)] == list(prefix)]


def _flag(call: List[str], name: str) -> str:
    return call[call.index(name) + 1]


def _workflow() -> dict:
    return yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))


def _triggers(workflow: dict) -> dict:
    # PyYAML (YAML 1.1) reads the bare key `on` as the boolean True.
    return workflow.get("on", workflow.get(True))


def _steps() -> List[dict]:
    return _workflow()["jobs"]["deploy"]["steps"]


def _step(name_prefix: str) -> dict:
    return next(s for s in _steps() if s.get("name", "").startswith(name_prefix))


# ── the script ──────────────────────────────────────────────────────────────


def test_script_is_executable_and_parses():
    assert _SCRIPT.stat().st_mode & stat.S_IXUSR
    check = subprocess.run(["bash", "-n", str(_SCRIPT)], capture_output=True, text=True)
    assert check.returncode == 0, check.stderr


def test_fresh_version_uploads_immutable_then_alias_and_invalidates_the_alias(tmp_path):
    result, calls = _deploy(tmp_path)
    assert result.returncode == 0, result.stderr
    puts = _calls(calls, "aws", "s3api", "put-object")
    assert [_flag(p, "--key") for p in puts] == [_IMMUTABLE, _ALIAS, _MARKER]
    immutable, alias, marker = puts
    assert _flag(immutable, "--cache-control") == "public, max-age=31536000, immutable"
    assert _flag(immutable, "--if-none-match") == "*"
    assert _flag(alias, "--cache-control") == "public, max-age=3600"
    assert "--if-none-match" not in alias
    assert _flag(marker, "--content-type") == "application/json"
    assert _flag(marker, "--cache-control") == "no-cache"
    for put in (immutable, alias):
        assert _flag(put, "--bucket") == _BUCKET
        assert _flag(put, "--content-type") == "application/gzip"
        assert _flag(put, "--body").endswith("automatos-theme-sdk-1.4.2.tgz")
    invalidations = _calls(calls, "aws", "cloudfront", "create-invalidation")
    assert len(invalidations) == 1
    assert _flag(invalidations[0], "--distribution-id") == _DIST
    assert _flag(invalidations[0], "--paths") == "/theme-sdk/v1/*"


def test_the_existence_checks_come_before_any_write(tmp_path):
    _, calls = _deploy(tmp_path)
    s3 = [c for c in calls if c[:2] == ["aws", "s3api"]]
    assert [c[2] for c in s3[:2]] == ["head-object", "head-object"]
    assert [_flag(c, "--key") for c in s3[:2]] == [_MARKER, _IMMUTABLE]


def test_the_marker_is_the_last_write_after_the_invalidation_and_both_checks(tmp_path):
    _, calls = _deploy(tmp_path)
    tools = [c[0] if c[0] != "aws" else " ".join(c[:3]) for c in calls]
    marker_at = max(i for i, c in enumerate(calls) if c[:3] == ["aws", "s3api", "put-object"])
    assert _flag(calls[marker_at], "--key") == _MARKER
    assert marker_at == len(calls) - 1
    wait_at = tools.index("aws cloudfront wait")
    assert tools.index("aws cloudfront create-invalidation") < wait_at
    curls = [i for i, t in enumerate(tools) if t == "curl"]
    assert len(curls) == 2 and wait_at < curls[0] and curls[-1] < marker_at


def test_the_invalidation_is_waited_on_by_its_id(tmp_path):
    _, calls = _deploy(tmp_path)
    (wait,) = _calls(calls, "aws", "cloudfront", "wait")
    assert wait[3] == "invalidation-completed"
    assert _flag(wait, "--distribution-id") == _DIST
    assert _flag(wait, "--id") == "I2TESTINV"


def test_an_unfinished_invalidation_fails_and_leaves_no_marker(tmp_path):
    result, calls = _deploy(tmp_path, wait="fail")
    assert result.returncode == 1
    assert "did not complete" in result.stderr
    keys = [_flag(p, "--key") for p in _calls(calls, "aws", "s3api", "put-object")]
    assert _MARKER not in keys
    assert not _calls(calls, "curl")


def test_an_alias_still_serving_the_old_version_fails_and_leaves_no_marker(tmp_path):
    result, calls = _deploy(tmp_path, stale_url=_ALIAS_URL)
    assert result.returncode == 1
    assert "smoke test failed for " + _ALIAS_URL in result.stderr
    assert "sha256" in result.stderr
    keys = [_flag(p, "--key") for p in _calls(calls, "aws", "s3api", "put-object")]
    assert _MARKER not in keys
    assert _calls(calls, "sleep"), "a stale edge is retried before the run fails"


def test_a_run_that_died_after_the_immutable_upload_is_resumed_not_skipped(tmp_path):
    result, calls = _deploy(tmp_path, head="partial")
    assert result.returncode == 0, result.stderr
    assert "Resuming" in result.stderr
    assert not _calls(calls, "npm", "pack"), "a published version is never re-packed"
    (fetch,) = _calls(calls, "aws", "s3api", "get-object")
    assert _flag(fetch, "--key") == _IMMUTABLE
    puts = _calls(calls, "aws", "s3api", "put-object")
    assert [_flag(p, "--key") for p in puts] == [_ALIAS, _MARKER]
    assert _flag(puts[0], "--body") == fetch[-1]
    assert len(_calls(calls, "aws", "cloudfront", "create-invalidation")) == 1
    assert len(_calls(calls, "aws", "cloudfront", "wait")) == 1
    assert len(_calls(calls, "curl")) == 2


def test_pack_ignores_lifecycle_scripts(tmp_path):
    _, calls = _deploy(tmp_path)
    (pack,) = _calls(calls, "npm", "pack")
    assert "--ignore-scripts" in pack


def test_smoke_tests_both_urls(tmp_path):
    _, calls = _deploy(tmp_path)
    urls = [c[-1] for c in _calls(calls, "curl")]
    assert urls == [f"https://widgets.automatos.app/{_IMMUTABLE}", _ALIAS_URL]
    for call in _calls(calls, "curl"):
        assert _flag(call, "-o") != "/dev/null", "the body is hashed, not discarded"


def test_published_version_is_refused_with_exit_3_and_nothing_written(tmp_path):
    result, calls = _deploy(tmp_path, head="found")
    assert result.returncode == _ALREADY_PUBLISHED
    assert "already published" in result.stderr
    assert [_flag(c, "--key") for c in _calls(calls, "aws", "s3api")] == [_MARKER]
    assert not _calls(calls, "aws", "s3api", "put-object")
    assert not _calls(calls, "aws", "cloudfront")


def test_an_unreadable_existence_check_fails_closed(tmp_path):
    result, calls = _deploy(tmp_path, head="forbidden")
    assert result.returncode == 1
    assert "could not check" in result.stderr
    assert not _calls(calls, "aws", "s3api", "put-object")


def test_dry_run_writes_nothing(tmp_path):
    result, calls = _deploy(tmp_path, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert not _calls(calls, "aws", "s3api", "put-object")
    assert not _calls(calls, "aws", "cloudfront")
    assert not _calls(calls, "curl")
    assert _ALIAS in result.stdout
    assert _MARKER in result.stdout


def test_prerelease_aliases_to_its_major(tmp_path):
    result, calls = _deploy(tmp_path, version="2.0.0-beta.1")
    assert result.returncode == 0, result.stderr
    keys = [_flag(p, "--key") for p in _calls(calls, "aws", "s3api", "put-object")]
    assert keys == [
        "theme-sdk/v2.0.0-beta.1/theme-sdk.tgz",
        "theme-sdk/v2/theme-sdk.tgz",
        "theme-sdk/v2.0.0-beta.1/published.json",
    ]


@pytest.mark.parametrize("bad", ["1.0", "v1.0.0", "1.0.0;rm", ""])
def test_a_malformed_package_version_is_refused_before_any_aws_write(tmp_path, bad):
    result, calls = _deploy(tmp_path, version=bad)
    assert result.returncode == 2
    assert not _calls(calls, "aws", "s3api")


def test_a_version_argument_is_refused(tmp_path):
    result, calls = _deploy(tmp_path, "v9.9.9")
    assert result.returncode == 2
    assert not calls


def test_the_version_comes_from_the_package_and_equals_sdk_version():
    package = json.loads((_PACKAGE / "package.json").read_text(encoding="utf-8"))
    shared = _REPO / "frontend" / "lib" / "theme-host" / "shared-modules.json"
    assert package["version"] == json.loads(shared.read_text(encoding="utf-8"))["sdkVersion"]
    assert package["private"] is True


# ── the workflow ────────────────────────────────────────────────────────────


def test_workflow_triggers_on_main_for_the_package_and_the_surface_and_by_hand():
    on = _triggers(_workflow())
    assert on["push"]["branches"] == ["main"]
    assert sorted(on["push"]["paths"]) == [
        "frontend/lib/theme-sdk/surface/**",
        "frontend/packages/theme-sdk/**",
    ]
    assert "workflow_dispatch" in on
    assert "pull_request" not in on


def test_workflow_never_cancels_a_deploy_and_reads_contents_only():
    workflow = _workflow()
    assert workflow["concurrency"]["cancel-in-progress"] is False
    assert workflow["permissions"] == {"contents": "read"}


def test_workflow_uses_the_widget_sdks_five_secrets_through_env_only():
    text = _WORKFLOW.read_text(encoding="utf-8")
    for name in _SECRETS:
        assert f"secrets.{name} }}}}" in text
    for step in _steps():
        assert "secrets." not in step.get("run", ""), step.get("name")


def test_missing_secrets_print_a_notice_and_exit_0():
    check = _step("Check the deploy secrets")
    assert sorted(check["env"]) == sorted(_SECRETS)
    assert "::notice::" in check["run"]
    assert "configured=false" in check["run"]
    assert "exit 0" in check["run"]
    for step in _steps():
        if step["name"].startswith(("Setup Node", "Resolve version", "Deploy via")):
            assert step["if"] == "steps.secrets.outputs.configured == 'true'", step["name"]


def test_deploy_step_runs_the_script_and_reads_exit_3_as_a_skip():
    deploy = _step("Deploy via")
    assert "bash scripts/theme-sdk/deploy-cdn.sh" in deploy["run"]
    assert '"$rc" -eq 3' in deploy["run"]
    assert "skipped=true" in deploy["run"]
    assert deploy["env"]["AWS_ACCESS_KEY_ID"] == "${{ secrets.AWS_SDK_DEPLOY_ACCESS_KEY_ID }}"


def test_readme_documents_the_install_line():
    readme = (_PACKAGE / "README.md").read_text(encoding="utf-8")
    assert _INSTALL_LINE in readme
    assert "scripts/theme-sdk/deploy-cdn.sh" in readme
