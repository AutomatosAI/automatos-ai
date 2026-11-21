"""PRD-277 US-505: images.yml publishes the enterprise web image beside the local one.

The image itself is ``enterprise-frontend`` in images.yml's matrix (its build args, inputs and
cache are pinned by test_prd262_image_matrix.py). These are the checks of its boot step and of
frontend/scripts/check-enterprise-image.sh, which starts the pushed image on each arch:

* the step runs on the enterprise image alone, after the scan, with the image and digest from
  the step's env and no expression inside ``run:``;
* the merge job publishes it, from the frontend's context, under its own package name;
* the script's refusal checks look for the line the server prints (start-up-node.ts), and each
  refusal the server can give starts with the setting's name, so a check that finds the line
  has found the reason;
* its made-up values are of the right shape, the bad one an origin the server refuses, and none
  is secret-shaped;
* the sign-in page it asks for is a public route, which a signed-out request reaches.

Pure file reads: no Docker, no network.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
IMAGES_YML = yaml.safe_load((ROOT / ".github" / "workflows" / "images.yml").read_text(encoding="utf-8"))
SCRIPT_PATH = "frontend/scripts/check-enterprise-image.sh"
SCRIPT = (ROOT / SCRIPT_PATH).read_text(encoding="utf-8")
ENTERPRISE = ROOT / "frontend" / "lib" / "enterprise"
BOOT_RUN = 'bash frontend/scripts/check-enterprise-image.sh "${IMAGE}@${DIGEST}"'


def _boot_step() -> tuple[int, dict]:
    steps = IMAGES_YML["jobs"]["build"]["steps"]
    found = [(index, step) for index, step in enumerate(steps) if BOOT_RUN in step.get("run", "")]
    assert len(found) == 1, "expected one images.yml build step running the enterprise check"
    return found[0]


def _value(name: str) -> str:
    match = re.search(rf'^{name}="([^"]*)"$', SCRIPT, re.MULTILINE)
    assert match, name
    return match.group(1)


def test_images_yml_starts_the_enterprise_image_each_build_pushed():
    position, step = _boot_step()
    assert step["if"] == (
        "${{ !cancelled() && matrix.name == 'enterprise-frontend' && steps.build.outcome == 'success' }}")
    assert step["env"] == {"IMAGE": "${{ matrix.image }}", "DIGEST": "${{ steps.build.outputs.digest }}"}
    assert step["run"].strip().splitlines() == ['docker pull --quiet "${IMAGE}@${DIGEST}"', BOOT_RUN]
    assert "${{" not in step["run"]
    steps = IMAGES_YML["jobs"]["build"]["steps"]
    policy = next(index for index, other in enumerate(steps) if other.get("name") == "Scan policy")
    assert position > policy, "the scan and its policy run first"


def test_the_merge_job_publishes_it_from_the_frontends_context():
    images = {entry["name"]: entry for entry in IMAGES_YML["jobs"]["merge"]["strategy"]["matrix"]["image"]}
    assert images["enterprise-frontend"]["context"] == images["frontend"]["context"] == "frontend"
    assert "enterprise edition" in images["enterprise-frontend"]["description"]
    assert "local edition" in images["frontend"]["description"]


def test_a_change_to_the_script_builds_and_checks_the_image():
    assert SCRIPT_PATH in IMAGES_YML[True]["pull_request"]["paths"]  # PyYAML reads `on` as True


def test_the_script_starts_the_image_with_no_network_and_says_what_it_proved():
    assert SCRIPT.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in SCRIPT
    assert '--network none "$@" "$IMAGE"' in SCRIPT
    # One for each refusal (expect_refused, run three times), one for the standalone checks, one for the policy.
    assert SCRIPT.count('echo "proved: ') == 3
    assert 'exit "$bad"' == SCRIPT.rstrip().splitlines()[-1]
    assert "Not proved here:" in SCRIPT


def test_it_runs_both_refusals_then_the_standalone_checks_then_the_policy():
    order = [
        'expect_refused BACKEND_INTERNAL_URL "" -e "MEDIA_ORIGINS=${MEDIA_ORIGIN}"',
        'expect_refused MEDIA_ORIGINS "$BAD_MEDIA_ORIGIN" -e "BACKEND_INTERNAL_URL=${BACKEND}" '
        '-e "MEDIA_ORIGINS=${BAD_MEDIA_ORIGIN}"',
        'expect_refused ENTERPRISE_CALLS_CLOUD_SYNC "$BAD_CLOUD_SYNC" -e "BACKEND_INTERNAL_URL=${BACKEND}" '
        '-e "ENTERPRISE_CALLS_CLOUD_SYNC=${BAD_CLOUD_SYNC}"',
        'bash "${HERE}/check-standalone-image.sh" "$IMAGE" /sign-in',
        "check_running_policy\n",
    ]
    positions = [SCRIPT.index(line) for line in order]
    assert positions == sorted(positions)


def test_the_refusal_check_reads_the_line_the_server_prints():
    start_up_node = (ENTERPRISE / "start-up-node.ts").read_text(encoding="utf-8")
    assert "console.error(`[start-up] The web app cannot start: ${problem}`)" in start_up_node
    assert '"The web app cannot start: ${setting}"' in SCRIPT


def test_each_refusal_the_server_gives_starts_with_its_settings_name():
    start_up = (ENTERPRISE / "start-up.ts").read_text(encoding="utf-8")
    media = (ENTERPRISE / "media-origins.ts").read_text(encoding="utf-8")
    assert "export const BACKEND_INTERNAL_URL_SETTING = 'BACKEND_INTERNAL_URL'" in start_up
    assert "`${BACKEND_INTERNAL_URL_SETTING} must be" in start_up
    assert "export const MEDIA_ORIGINS_SETTING = 'MEDIA_ORIGINS'" in media
    assert "`${MEDIA_ORIGINS_SETTING} must list" in media
    cloud_sync = (ENTERPRISE / "cloud-sync-setting.ts").read_text(encoding="utf-8")
    assert "export const CLOUD_SYNC_SETTING = 'ENTERPRISE_CALLS_CLOUD_SYNC'" in cloud_sync
    assert "`${CLOUD_SYNC_SETTING} must be true or false" in cloud_sync
    assert _value("BAD_CLOUD_SYNC") not in cloud_sync, "the refusal text never holds the bad value"


def test_the_values_are_made_up_and_of_the_right_shape():
    assert _value("BACKEND") == "http://automatos-api:8000"
    assert _value("MEDIA_ORIGIN") == "https://media.example.com"
    bad = _value("BAD_MEDIA_ORIGIN")
    assert bad == "http://media.example.com", "http and not a local address: media-origins.ts refuses it"
    media = (ENTERPRISE / "media-origins.ts").read_text(encoding="utf-8")
    assert "http only for a local development address" in media.replace("`", "")
    assert bad not in media, "the refusal text never holds the bad value, so the log check can find it only if printed"


def test_the_page_it_asks_for_is_public():
    routes = (ROOT / "frontend" / "lib" / "public-routes.ts").read_text(encoding="utf-8")
    assert '"/sign-in(.*)",' in routes
