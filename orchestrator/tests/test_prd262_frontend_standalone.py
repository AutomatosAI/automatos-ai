"""PRD-262 US-046: CI boots both frontend editions and proves the standalone diet.

test.yml's prod-image-parity job builds the SaaS frontend (the Dockerfile's default
edition) and the local one (images.yml's args), and runs
frontend/scripts/check-standalone-image.sh against each: no sharp, libvips, TypeScript
or AMP validator in /app; with no network, the page answers 200 and /_next/image 404;
the server's log never mentions sharp.

No CI build passes Clerk keys, and clerkMiddleware and ClerkProvider throw without
them, so the SaaS build gets format-valid production-instance dummies, made at run
time and never committed (gitleaks' Stripe rule matches a literal sk_live_ key).

images.yml runs the same script against the frontend image each build job pushed, on amd64
and arm64, after the scan and whenever the build pushed one, and a change to the script
builds and checks that image.

The config and the brand mark are checked by frontend/lib/__tests__/standalone-diet.test.ts.
Pure file reads: no Docker, no network.
"""
from __future__ import annotations

import json
import re
import shlex
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
TEST_YML = (WORKFLOWS / "test.yml").read_text()
CHECK_SCRIPT = ROOT / "frontend" / "scripts" / "check-standalone-image.sh"
CHECK = "bash frontend/scripts/check-standalone-image.sh"
PK = "pk_live_$(printf 'clerk.example.com$' | base64)"
SK = "sk_live_$(openssl rand -hex 16)"


def _parity_steps() -> list[dict]:
    return yaml.safe_load(TEST_YML)["jobs"]["prod-image-parity"]["steps"]


def _step_running(needle: str) -> str:
    runs = [step.get("run", "") for step in _parity_steps() if needle in step.get("run", "")]
    assert len(runs) == 1, f"expected one prod-image-parity step running {needle!r}, found {len(runs)}"
    return runs[0]


def _build_args(script: str, tag: str) -> list[str]:
    """The --build-arg values of the `docker build` that tags `tag`."""
    command = next(line for line in script.replace("\\\n", " ").splitlines() if f"-t {tag} " in line)
    words = shlex.split(command)
    return [words[i + 1] for i, word in enumerate(words) if word == "--build-arg"]


def _images_yml() -> dict:
    return yaml.safe_load((WORKFLOWS / "images.yml").read_text())


def _published_frontend() -> dict:
    pick = next(step for step in _images_yml()["jobs"]["changes"]["steps"] if step.get("id") == "pick")
    return next(image for image in json.loads(pick["env"]["IMAGES"]) if image["name"] == "frontend")


def _published_frontend_args() -> list[str]:
    return _published_frontend()["build_args"]


def test_the_saas_build_gets_a_dummy_production_publishable_key():
    script = _step_running("-t parity-fe frontend/")
    assert f'pk="{PK}"' in script
    # The only arg: the edition is still the Dockerfile's default, so the edition check keeps its meaning.
    assert _build_args(script, "parity-fe") == ["NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=$pk"]


def test_the_saas_build_serves_sign_in_with_a_dummy_secret_key():
    script = _step_running(f"{CHECK} parity-fe /sign-in")
    assert f'sk="{SK}"' in script
    assert f'{CHECK} parity-fe /sign-in -e CLERK_SECRET_KEY="$sk"' in script


def test_the_local_build_uses_the_args_images_yml_publishes_with_and_serves_chat():
    script = _step_running(f"{CHECK} parity-fe-local /chat")
    assert _build_args(script, "parity-fe-local") == _published_frontend_args()


def test_a_change_to_the_check_script_or_the_config_runs_the_parity_job():
    match = next(
        step for step in yaml.safe_load(TEST_YML)["jobs"]["changes"]["steps"] if step.get("id") == "match"
    )
    paths = match["env"]["PROD_IMAGE_PATHS"]
    for changed in ("frontend/scripts/check-standalone-image.sh", "frontend/next.config.js"):
        assert re.search(paths, changed), changed


def test_no_clerk_key_is_committed_in_the_workflow():
    assert re.findall(r"\b[ps]k_live_[A-Za-z0-9]", TEST_YML) == []


def test_the_builder_keeps_the_sharp_check_with_the_same_key():
    """US-009's check moved to the builder: the runtime image has no sharp to look at."""
    script = _step_running("-t parity-fe-builder frontend/")
    assert "--target builder" in script
    assert _build_args(script, "parity-fe-builder") == ["NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=$PARITY_CLERK_PK"]
    assert 'echo "PARITY_CLERK_PK=${pk}" >> "$GITHUB_ENV"' in _step_running("-t parity-fe frontend/")


def test_the_check_script_boots_with_no_network_and_reads_the_log():
    script = CHECK_SCRIPT.read_text()
    assert script.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in script
    assert '--network none "$@" "$IMAGE"' in script
    assert 'expect_status "$PAGE" 200' in script
    assert "OPTIMISER_PATH='/_next/image?" in script
    assert 'expect_status "$OPTIMISER_PATH" 404' in script
    assert 'grep -qi sharp <<< "$logs"' in script
    assert 'exit "$bad"' in script.rstrip().splitlines()[-1]


# ── images.yml: the published image, on both arches ─────────────────────────

IMAGES_CHECK = 'bash frontend/scripts/check-standalone-image.sh "${IMAGE}@${DIGEST}" /chat'


def test_images_yml_checks_the_frontend_image_each_build_pushed():
    steps = _images_yml()["jobs"]["build"]["steps"]
    found = [(i, step) for i, step in enumerate(steps) if IMAGES_CHECK in step.get("run", "")]
    assert len(found) == 1, "expected one images.yml build step running the standalone check"
    position, step = found[0]
    # The published frontend is the local edition, so /chat is its page.
    assert "NEXT_PUBLIC_AUTH_EDITION=local" in _published_frontend_args()
    assert step["if"] == "${{ !cancelled() && matrix.name == 'frontend' && steps.build.outcome == 'success' }}"
    assert step["env"] == {"IMAGE": "${{ matrix.image }}", "DIGEST": "${{ steps.build.outputs.digest }}"}
    lines = step["run"].strip().splitlines()
    assert lines == ['docker pull --quiet "${IMAGE}@${DIGEST}"', IMAGES_CHECK]
    policy = next(i for i, other in enumerate(steps) if other.get("name") == "Scan policy")
    assert position > policy, "the scan and its policy run first"


def test_a_change_to_the_check_script_builds_and_checks_the_frontend_image():
    script = "frontend/scripts/check-standalone-image.sh"
    assert script in _published_frontend()["inputs"]
    assert script in _images_yml()[True]["pull_request"]["paths"]  # PyYAML reads `on` as True
