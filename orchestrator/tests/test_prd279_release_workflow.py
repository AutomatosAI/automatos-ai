"""PRD-279 Waves 1 and 2: what images.yml does on a `v*` tag, and only there.

Static checks on the workflow files:

* every step and job the release adds runs on a `v*` tag only, and the merge job's
  other steps are the ones it had, in their order, so a push to main, a pull request,
  the weekly run and a manual run on a branch publish, scan and sign as before (US-101);
* the preflight runs before any build, and on a tag the order is: a staging index, its
  signature, the signature's check, and only after every image's check the version
  tags, in one job; the chart job needs that job (US-101, US-103, FR-6);
* a tag pushes no `sha-<7>` and no version from the merge job, and `edge` never moves
  on a tag (US-101, FR-9);
* a failing scan leaves nothing tagged or signed: the merge job needs every build, and
  the chart job needs the merge job (US-102);
* each image's signature is checked right after it is made (US-103);
* the chart job packages, renders, refuses an existing version, pushes, signs and
  checks, with only the permissions it needs, no secret but the built-in token, and
  every tool pinned (US-201, US-202).

Pure file reads: no Docker, no network.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
IMAGES = yaml.safe_load((WORKFLOWS / "images.yml").read_text(encoding="utf-8"))
SMOKE = yaml.safe_load((WORKFLOWS / "smoke-images.yml").read_text(encoding="utf-8"))
JOBS = IMAGES["jobs"]
ON_A_TAG = "startsWith(github.ref, 'refs/tags/v')"
# The merge job's steps before PRD-279: they run on every publishing run, as they did.
MERGE_STEPS_BEFORE = [
    "Checkout", "Image name", "Download digests", "Set up Buildx", "Log in to GHCR", "Metadata",
    "Create the multi-arch index", "Resolve the index digest", "Install cosign", "Sign (keyless)",
    "Tag the context's tree (tree-<12>, once per tree)",
]
_INTERPOLATION = re.compile(r"\$\{\{")


def _steps(job: str) -> list:
    return JOBS[job]["steps"]


def _step(job: str, name: str) -> dict:
    return next(step for step in _steps(job) if step.get("name") == name)


def _tags() -> list:
    meta = next(step for step in _steps("merge") if step.get("id") == "meta")
    return [line.strip() for line in meta["with"]["tags"].splitlines() if line.strip()]


# --------------------------------------------------------------------------- only on a tag (US-101)


def test_the_merge_job_runs_its_old_steps_as_before_and_every_new_one_only_on_a_tag():
    old = [step["name"] for step in _steps("merge") if step.get("if") != ON_A_TAG]
    assert old == MERGE_STEPS_BEFORE
    new = [step for step in _steps("merge") if step.get("if") == ON_A_TAG]
    assert [step["name"] for step in new] == [
        "Verify the signature (v* tags only)", "Record the digest for the chart (v* tags only)",
        "Upload the digest for the chart (v* tags only)"]


def _changes_index(predicate) -> int:
    return next(i for i, step in enumerate(_steps("changes")) if predicate(step))


def test_the_version_and_the_preflight_come_before_anything_builds_and_only_on_a_tag():
    steps = _steps("changes")
    release = next(step for step in steps if step.get("id") == "release")
    preflight = _step("changes", "Release preflight (v* tags only)")
    assert release["if"] == preflight["if"] == ON_A_TAG
    pick = _changes_index(lambda step: step.get("id") == "pick")
    assert steps.index(release) < steps.index(preflight) < pick
    assert JOBS["build"]["needs"] == "changes", "every build waits for the job the preflight is in"
    assert 'scripts/ci/release_version.py --ref "$GITHUB_REF"' in release["run"]
    assert "contents/charts/automatos/Chart.yaml?ref=${GITHUB_SHA}" in release["run"]
    assert JOBS["changes"]["outputs"]["version"] == "${{ steps.release.outputs.version }}"
    assert preflight["env"]["VERSION"] == "${{ steps.release.outputs.version }}"
    named = re.findall(r"--image (\S+)", preflight["run"])
    assert sorted(named) == sorted(entry["name"] for entry in JOBS["merge"]["strategy"]["matrix"]["image"])
    for part in ("scripts/ci/release_preflight.py", '--sha "$GITHUB_SHA"', '--default-branch "$DEFAULT_BRANCH"'):
        assert part in preflight["run"], part
    assert JOBS["changes"]["permissions"] == {"actions": "read", "contents": "read", "pull-requests": "read"}


def test_on_a_tag_the_merge_job_publishes_only_a_staging_tag_and_no_sha():
    tags = _tags()
    assert not [tag for tag in tags if tag.startswith("type=semver")], "the version tags come after the checks"
    assert ("type=raw,value=release-staging-${{ github.run_id }}-${{ github.run_attempt }},"
            "enable=${{ startsWith(github.ref, 'refs/tags/v') }}") in tags
    sha = next(tag for tag in tags if tag.startswith("type=sha,"))
    assert sha == "type=sha,enable=${{ github.event_name != 'schedule' && !startsWith(github.ref, 'refs/tags/v') }}"
    assert not any("edge" in tag or "latest=true" in tag for tag in tags)


def test_on_a_tag_the_order_is_stage_sign_verify_then_every_image_is_tagged():
    names = [step["name"] for step in _steps("merge")]
    order = ["Create the multi-arch index", "Resolve the index digest", "Sign (keyless)",
             "Verify the signature (v* tags only)", "Upload the digest for the chart (v* tags only)"]
    assert [names.index(name) for name in order] == sorted(names.index(name) for name in order)
    job = JOBS["release-images"]
    assert job["if"] == ON_A_TAG and set(job["needs"]) == {"changes", "merge"}, "after every merge leg"
    assert job["permissions"] == {"contents": "read", "packages": "write"}
    (tagging,) = [step for step in job["steps"] if "release_tags.py" in step.get("run", "")]
    named = re.findall(r"--image (\S+)", tagging["run"])
    assert sorted(named) == sorted(entry["name"] for entry in JOBS["merge"]["strategy"]["matrix"]["image"])
    assert '--version "$VERSION"' in tagging["run"] and '--digests "${RUNNER_TEMP}/release-digests"' in tagging["run"]
    download = next(step for step in job["steps"] if "download-artifact" in step.get("uses", ""))
    assert download["with"]["pattern"] == "release-digest-*"
    assert set(JOBS["release-chart"]["needs"]) == {"changes", "release-images"}


def test_edge_never_moves_on_a_tag():
    """smoke-images, and the promote-edge job after it, run only for a run on main."""
    triggers = SMOKE.get("on", SMOKE.get(True))
    assert triggers["workflow_run"]["branches"] == ["main"]
    assert "github.event.workflow_run.head_branch == 'main'" in " ".join(SMOKE["jobs"]["smoke-images"]["if"].split())
    assert SMOKE["jobs"]["promote-edge"]["needs"] == "smoke-images"


# --------------------------------------------------------------------------- the scan (US-102)


def test_a_failing_scan_leaves_nothing_tagged_or_signed():
    assert set(JOBS["merge"]["needs"]) == {"changes", "build"}
    assert "merge" in JOBS["release-images"]["needs"] and "release-images" in JOBS["release-chart"]["needs"]
    for job in ("merge", "release-images", "release-chart"):
        assert not re.search(r"\b(?:always|failure|cancelled)\(\)", JOBS[job]["if"]), job


# --------------------------------------------------------------------------- the signatures (US-103)


def test_each_image_signature_is_checked_right_after_it_is_made():
    names = [step["name"] for step in _steps("merge")]
    assert names.index("Verify the signature (v* tags only)") == names.index("Sign (keyless)") + 1
    verify = _step("merge", "Verify the signature (v* tags only)")
    assert verify["env"]["DIGEST"] == _step("merge", "Sign (keyless)")["env"]["DIGEST"]
    for part in ('release_signing.py verify --image "${IMAGE}@${DIGEST}"', '--repository "$GITHUB_REPOSITORY"',
                 '--ref "$GITHUB_REF"', '--sha "$GITHUB_SHA"'):
        assert part in verify["run"], part


# --------------------------------------------------------------------------- the chart job (US-201, US-202)


def test_the_chart_job_runs_on_a_tag_only_with_only_the_permissions_it_needs():
    job = JOBS["release-chart"]
    assert job["if"] == ON_A_TAG
    assert job["permissions"] == {"contents": "read", "packages": "write", "id-token": "write"}
    text = yaml.safe_dump(job)
    assert set(re.findall(r"secrets\.(\w+)", text)) == {"GITHUB_TOKEN"}
    assert job["env"]["VERSION"] == "${{ needs.changes.outputs.version }}"


def test_the_chart_job_packages_renders_refuses_pushes_signs_and_checks_in_that_order():
    names = [step["name"] for step in _steps("release-chart")]
    order = ["Package the chart with every image's digest", "Render the packaged chart; fail on any image named by a tag",
             "Refuse a version that is already published", "Push the chart", "Resolve the chart's digest",
             "Sign the chart (keyless)", "Verify the chart's signature", "Remove the chart version it could not prove"]
    assert [names.index(name) for name in order] == sorted(names.index(name) for name in order)
    package = _step("release-chart", order[0])["run"]
    assert 'helm package "${RUNNER_TEMP}/chart/automatos" --version "$VERSION" --app-version "$VERSION"' in package
    assert "release_chart.py prepare --chart charts/automatos" in package
    assert '--render-values "${RUNNER_TEMP}/render-values"' in package
    render = _step("release-chart", order[1])["run"]
    # Once per published edition (US-201 line 3): existingSecret and the edition's values file only.
    assert 'for values in "${RUNNER_TEMP}/render-values"/*.yaml; do' in render
    assert 'helm template automatos "$PACKAGE" --set existingSecret=ci --values "$values" \\' in render
    assert '> "${RUNNER_TEMP}/rendered/$(basename "$values")"' in render
    assert 'release_chart.py check-rendered --rendered "${RUNNER_TEMP}/rendered"' in render
    assert "${{" not in render


def test_a_version_that_exists_is_never_pushed_again():
    refuse = _step("release-chart", "Refuse a version that is already published")["run"]
    assert 'scripts/stack_images.py published "${CHART}:${VERSION}"' in refuse
    branches = dict(re.findall(r"^\s*(\S+)\) (.*)$", refuse, re.M))
    assert "exit 1" not in branches["1"]
    assert "::error" in branches["0"] and "::error" in branches["*"], "published, or unreadable, stops the job"
    assert "exit 1" in refuse.split("0)", 1)[1].split("*)", 1)[0]
    assert IMAGES["concurrency"]["cancel-in-progress"] == "${{ github.event_name == 'pull_request' }}"


def test_the_chart_is_signed_and_checked_by_digest_as_the_images_are():
    sign = _step("release-chart", "Sign the chart (keyless)")
    assert sign["run"] == 'cosign sign --yes "${CHART}@${DIGEST}"'
    verify = _step("release-chart", "Verify the chart's signature")
    assert sign["env"]["DIGEST"] == verify["env"]["DIGEST"] == "${{ steps.chart.outputs.digest }}"
    assert 'release_signing.py verify --image "${CHART}@${DIGEST}"' in verify["run"]
    assert '--ref "$GITHUB_REF" --sha "$GITHUB_SHA"' in verify["run"]


def test_helm_and_cosign_are_pinned_exactly():
    env = JOBS["release-chart"]["env"]
    assert re.fullmatch(r"v\d+\.\d+\.\d+", env["HELM_VERSION"]) and re.fullmatch(r"[0-9a-f]{64}", env["HELM_SHA256"])
    install = _step("release-chart", "Install Helm (pinned, checksum checked)")["run"]
    assert 'echo "${HELM_SHA256}  ${RUNNER_TEMP}/helm.tar.gz" | sha256sum -c -' in install
    cosign = _step("release-chart", "Install cosign")
    assert cosign["with"]["cosign-release"] == "${{ env.COSIGN_VERSION }}"
    assert re.fullmatch(r"v\d+\.\d+\.\d+", env["COSIGN_VERSION"])


def test_every_action_the_release_adds_is_one_this_repository_already_pins():
    def uses(job: dict) -> set:
        return {step["uses"] for step in job.get("steps", []) if "uses" in step}
    added = ("release-images", "release-chart", "release-notes")
    others = set().union(*(uses(job) for name, job in JOBS.items() if name not in added))
    for name in added:
        assert uses(JOBS[name]) <= others, name


def test_no_new_step_interpolates_an_expression_into_its_shell():
    """The tag name and every other value reach the shell through env:, quoted."""
    new = [step for step in _steps("merge") if step.get("if") == ON_A_TAG]
    new += [next(step for step in _steps("changes") if step.get("id") == "release")]
    new += [_step("changes", "Release preflight (v* tags only)"), _step("changes", "Release page check (v* tags only)")]
    new += _steps("release-images") + _steps("release-chart") + _steps("release-notes")
    for step in new:
        assert not _INTERPOLATION.search(step.get("run", "")), step["name"]


def test_a_chart_that_was_not_proven_is_removed_and_only_one_this_job_pushed():
    remove = _step("release-chart", "Remove the chart version it could not prove")
    assert remove["if"] == "failure() && steps.push.outcome == 'success'", "never a version the job did not push"
    assert _step("release-chart", "Push the chart")["id"] == "push"
    assert 'release_chart.py remove --owner "$GITHUB_REPOSITORY_OWNER"' in remove["run"]
    assert '--tag "$VERSION"' in remove["run"]
    assert remove["env"]["GH_TOKEN"] == "${{ github.token }}", "the built-in token, no broader one"
