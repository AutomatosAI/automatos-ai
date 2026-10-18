"""PRD-262 US-003: a lane that boots what we publish.

Static checks on ``.github/workflows/smoke-images.yml``, the smoke script's image
modes and ``images.yml``'s tags. The lane itself runs only in CI:

* ``smoke-images`` runs after images.yml publishes on main (workflow_run) and by hand
  with a tag, on ubuntu-24.04 and ubuntu-24.04-arm, and boots ``sha-<7>`` through
  today's published path (docker-compose.images.yml, ``up --no-build``);
* the pull-request variant watches compose, the Makefile, ``envs/`` and every
  images.yml input, and builds the three amd64 images with images.yml's context,
  target, build args, labels and GHA cache (read only), tagged as compose expects;
* images.yml's merge job no longer pushes ``edge``; ``promote-edge`` moves it after
  both arches pass, for the images the merge job publishes;
* the script empties the Docker cache before the pull, never builds in an image mode,
  and asserts the production command's worker count (orchestrator/Dockerfile's CMD);
* outside CI, the pull refuses to empty the host's image cache unless told it may.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[2]
_WORKFLOW = _REPO / ".github" / "workflows" / "smoke-images.yml"
_IMAGES = _REPO / ".github" / "workflows" / "images.yml"
_FRESH = _REPO / ".github" / "workflows" / "smoke-fresh-clone.yml"
_SCRIPT = _REPO / "scripts" / "ci" / "smoke-fresh-clone.sh"
_DOCKERFILE = _REPO / "orchestrator" / "Dockerfile"
_OVERRIDE = _REPO / "docker-compose.images.yml"
PINNED = re.compile(r"^[\w.-]+/[\w./-]+@[0-9a-f]{40}$")


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _triggers(data: dict) -> dict:
    return data.get("on", data.get(True))  # PyYAML reads the bare `on:` key as True


WORKFLOW = _yaml(_WORKFLOW)
JOBS = WORKFLOW["jobs"]
IMAGES_WORKFLOW = _yaml(_IMAGES)
PICK = next(step for step in IMAGES_WORKFLOW["jobs"]["changes"]["steps"] if step.get("id") == "pick")
PUBLISHED = [image for image in json.loads(PICK["env"]["IMAGES"]) if image["publish"]]
# docker-compose.images.yml's service for each published image.
SERVICES = {"api": "backend", "frontend": "frontend", "workspace-worker": "workspace-worker"}


def _step(job: str, predicate) -> dict:
    return next(step for step in JOBS[job]["steps"] if predicate(step))


def _boot_step(job: str) -> dict:
    return _step(job, lambda step: step.get("run") == "bash scripts/ci/smoke-fresh-clone.sh")


# --------------------------------------------------------------------------- triggers


def test_it_runs_after_images_publishes_on_main_and_by_hand_with_a_tag():
    triggers = _triggers(WORKFLOW)
    run = triggers["workflow_run"]
    assert run["workflows"] == [IMAGES_WORKFLOW["name"]]
    assert run["types"] == ["completed"] and run["branches"] == ["main"]
    tag = triggers["workflow_dispatch"]["inputs"]["tag"]
    assert tag["required"] is True and tag["type"] == "string"


def test_only_a_successful_publish_from_this_repositorys_main_is_booted():
    condition = " ".join(JOBS["smoke-images"]["if"].split())
    for clause in ("github.event.workflow_run.conclusion == 'success'",
                   "github.event.workflow_run.head_branch == 'main'",
                   "github.event.workflow_run.head_repository.full_name == github.repository",
                   "github.event.workflow_run.event == 'push'",
                   "github.event_name == 'workflow_dispatch'"):
        assert clause in condition, clause
    checkout = _step("smoke-images", lambda step: "actions/checkout" in step.get("uses", ""))
    assert checkout["with"]["ref"] == "${{ github.event.workflow_run.head_sha || github.sha }}"


def test_the_pull_request_variant_watches_compose_the_makefile_envs_and_every_image_input():
    paths = _triggers(WORKFLOW)["pull_request"]["paths"]
    assert len(paths) == len(set(paths)), "a path is listed twice"
    image_inputs = set(_triggers(IMAGES_WORKFLOW)["pull_request"]["paths"])
    missing = (image_inputs | {"docker-compose*.yml", "Makefile", "envs/**"}) - set(paths)
    assert not missing, f"smoke-images.yml does not watch {sorted(missing)}"
    own = {"scripts/ci/smoke-fresh-clone.sh", "scripts/ci/smoke_boot.py", "scripts/ci/smoke_report.py",
           "scripts/ci/smoke_images.py", "scripts/ci/smoke_images_report.py", ".github/workflows/smoke-images.yml"}
    assert own <= set(paths)


# --------------------------------------------------------------------------- the main runs


def test_main_runs_boot_on_both_arches():
    job = JOBS["smoke-images"]
    assert job["runs-on"] == "${{ matrix.runner }}"
    arches = {entry["arch"]: entry["runner"] for entry in job["strategy"]["matrix"]["include"]}
    assert arches == {"amd64": "ubuntu-24.04", "arm64": "ubuntu-24.04-arm"}
    assert job["strategy"]["fail-fast"] is False


def test_main_runs_pull_the_target_tag_through_the_published_path():
    target = _step("smoke-images", lambda step: step.get("id") == "target")
    assert "smoke_images.py target" in target["run"] and '--outputs "$GITHUB_OUTPUT"' in target["run"]
    assert target["env"]["TAG"] == "${{ inputs.tag }}", "the dispatch input goes through env, never inline"
    env = _boot_step("smoke-images")["env"]
    assert env["SMOKE_IMAGES"] == "pull"
    assert env["AUTOMATOS_IMAGE_TAG"] == "${{ steps.target.outputs.tag }}"
    assert env["SMOKE_EXPECT_REVISION"] == "${{ steps.target.outputs.revision }}"


@pytest.mark.parametrize("job", ["smoke-images", "smoke-images-pr"])
def test_both_variants_use_the_fresh_clone_lanes_throwaway_secrets_and_upload_the_records(job):
    fresh = next(step for step in _yaml(_FRESH)["jobs"]["fresh-clone-boot"]["steps"]
                 if step.get("run") == "bash scripts/ci/smoke-fresh-clone.sh")["env"]
    env = _boot_step(job)["env"]
    for key in ("POSTGRES_PASSWORD", "REDIS_PASSWORD", "API_KEY", "AUTH_EDITION", "REQUIRE_AUTH", "DEFAULT_WORKSPACE_ID"):
        assert env[key] == fresh[key], key
    upload = _step(job, lambda step: "upload-artifact" in step.get("uses", ""))
    assert upload["if"] == "always()"
    assert upload["with"]["path"].rstrip("/") == env["SMOKE_OUT_DIR"].rstrip("/")
    assert "continue-on-error" not in _boot_step(job), "a boot that fails fails the job"


# --------------------------------------------------------------------------- edge


def test_the_merge_job_no_longer_pushes_edge():
    merge = IMAGES_WORKFLOW["jobs"]["merge"]
    meta = next(step for step in merge["steps"] if step.get("id") == "meta")
    tags = [line.strip() for line in meta["with"]["tags"].splitlines() if line.strip()]
    assert "type=sha" in tags
    assert not any(tag.startswith("type=edge") for tag in tags), "edge moves only after smoke-images passes"


def test_promote_edge_waits_for_both_arches_of_a_main_publish():
    job = JOBS["promote-edge"]
    assert job["needs"] == "smoke-images"
    assert job["if"] == "github.event_name == 'workflow_run'", "a manual run never moves edge"
    assert job["permissions"] == {"contents": "read", "packages": "write"}
    assert job["concurrency"] == {"group": "promote-edge", "cancel-in-progress": False}
    assert JOBS["smoke-images"]["outputs"]["tag"] == "${{ steps.target.outputs.tag }}"


def test_promote_edge_moves_exactly_the_images_the_merge_job_publishes():
    move = _step("promote-edge", lambda step: "promote_edge.py" in step.get("run", ""))
    named = re.findall(r"--image (\S+)", move["run"])
    merged = [entry["name"] for entry in IMAGES_WORKFLOW["jobs"]["merge"]["strategy"]["matrix"]["image"]]
    assert sorted(named) == sorted(merged) == sorted(image["name"] for image in PUBLISHED)
    assert move["env"]["TAG"] == "${{ needs.smoke-images.outputs.tag }}"
    assert move["env"]["SHA"] == "${{ github.event.workflow_run.head_sha }}"
    assert '--prefix "ghcr.io/${GITHUB_REPOSITORY_OWNER,,}/automatos-"' in move["run"]


def test_only_promote_edge_can_write_packages():
    assert WORKFLOW["permissions"] == {"contents": "read"}
    writers = [name for name, job in JOBS.items() if (job.get("permissions") or {}).get("packages") == "write"]
    assert writers == ["promote-edge"]


# --------------------------------------------------------------------------- the pull-request variant


def _compose_images() -> dict:
    class Loader(yaml.SafeLoader):
        pass

    for tag in ("!reset", "!override"):
        Loader.add_constructor(tag, lambda loader, node: None)
    services = yaml.load(_OVERRIDE.read_text(encoding="utf-8"), Loader=Loader)["services"]
    return {service: spec["image"] for service, spec in services.items()}


@pytest.mark.parametrize("image", PUBLISHED, ids=lambda image: image["name"])
def test_the_pr_variant_builds_each_image_as_images_yml_does(image):
    name = image["name"]
    build = _step("smoke-images-pr", lambda step: step.get("name") == f"Build {name} (amd64)")["with"]
    assert build["context"] == image["context"]
    assert build.get("target", "") == image["target"]
    assert build["platforms"] == "linux/amd64"
    assert build["build-args"].split() == image["build_args"]
    assert build["cache-from"] == f"type=gha,scope={name}-amd64", "images.yml's cache scope"
    assert "cache-to" not in build, "PR builds read the cache and never write it (§9-Q13)"
    assert build["load"] is True and build["provenance"] is False and build["sbom"] is False
    assert build["labels"] == f"${{{{ steps.meta-{name}.outputs.labels }}}}"


@pytest.mark.parametrize("image", PUBLISHED, ids=lambda image: image["name"])
def test_the_pr_variant_tags_each_image_with_the_name_compose_runs(image):
    name = image["name"]
    build = _step("smoke-images-pr", lambda step: step.get("name") == f"Build {name} (amd64)")["with"]
    compose_image = _compose_images()[SERVICES[name]]
    assert build["tags"] == compose_image.replace("${AUTOMATOS_IMAGE_TAG:-edge}", "${{ steps.target.outputs.tag }}")


@pytest.mark.parametrize("image", PUBLISHED, ids=lambda image: image["name"])
def test_the_pr_variant_stamps_the_labels_images_yml_stamps(image):
    name = image["name"]
    meta = _step("smoke-images-pr", lambda step: step.get("id") == f"meta-{name}")
    published = next(step for step in IMAGES_WORKFLOW["jobs"]["build"]["steps"] if step.get("id") == "meta")
    assert meta["uses"] == published["uses"], "the same metadata-action, which sets the revision label"
    expected = published["with"]["labels"].replace("${{ matrix.name }}", name)
    assert meta["with"]["labels"].split() == expected.split()


def test_the_pr_variant_boots_prebuilt_images_under_the_pr_tag():
    job = JOBS["smoke-images-pr"]
    assert job["if"] == "github.event_name == 'pull_request'"
    assert job["runs-on"] == "ubuntu-24.04"
    target = _step("smoke-images-pr", lambda step: step.get("id") == "target")
    assert target["env"] == {"EVENT": "${{ github.event_name }}", "SHA": "${{ github.sha }}"}
    env = _boot_step("smoke-images-pr")["env"]
    assert env["SMOKE_IMAGES"] == "prebuilt"
    assert env["AUTOMATOS_IMAGE_TAG"] == "${{ steps.target.outputs.tag }}"
    assert env["SMOKE_EXPECT_REVISION"] == "${{ steps.target.outputs.revision }}"


def test_published_images_carry_the_revision_label_from_metadata_action():
    build = IMAGES_WORKFLOW["jobs"]["build"]
    meta = next(step for step in build["steps"] if step.get("id") == "meta")
    push = next(step for step in build["steps"] if step.get("id") == "build")
    assert meta["uses"].startswith("docker/metadata-action@")
    assert push["with"]["labels"] == "${{ steps.meta.outputs.labels }}"


def test_every_action_is_pinned_by_sha():
    uses = [step["uses"] for job in JOBS.values() for step in job["steps"] if "uses" in step]
    assert uses and all(PINNED.match(ref) for ref in uses), uses


# --------------------------------------------------------------------------- the script


def _script() -> str:
    return _SCRIPT.read_text(encoding="utf-8")


def _function(text: str, name: str) -> str:
    match = re.search(rf"^{name}\(\) \{{\n(.*?)^\}}", text, re.M | re.S)
    assert match, f"smoke-fresh-clone.sh has no {name}() function"
    return match.group(1)


def test_image_modes_run_the_published_compose_override():
    text = _script()
    assert 'IMAGES_COMPOSE_FILE="docker-compose.images.yml"' in text
    assert 'COMPOSE_ARGS+=(-f "$IMAGES_COMPOSE_FILE")' in text
    assert 'docker compose "${COMPOSE_ARGS[@]}" "$@"' in _function(text, "compose")
    modes = _function(text, "check_mode")
    assert "pull|prebuilt)" in modes and "AUTOMATOS_IMAGE_TAG" in modes


def test_image_modes_never_build_and_the_pull_starts_from_an_empty_cache():
    text = _script()
    prepare = _function(text, "prepare_images")
    assert '"") build_images ;;' in prepare and "pull) pull_images ;;" in prepare
    assert "build_images" not in prepare.split("prebuilt)", 1)[1]
    assert not re.search(r"\bup\b[^\n]*--build\b", text), "no `up --build`: image modes run what was pulled or built in the job"
    pull = _function(text, "pull_images")
    assert pull.index("empty_docker_cache") < pull.index('"$IMAGES_TOOL" pull')
    assert 'docker compose "${COMPOSE_ARGS[@]}" pull' in pull
    assert "docker image prune -af" in _function(text, "empty_docker_cache")
    assert "compose up -d --no-build postgres redis minio backend" in _function(text, "boot_fresh")


def test_the_script_asserts_what_booted_before_the_warm_restarts():
    text = _script()
    main = _function(text, "main")
    order = ["prepare_images", "boot_fresh", "check_backend_processes", "boot_frontend", "check_images",
             "warm_restarts"]
    assert [main.index(step) for step in order] == sorted(main.index(step) for step in order)
    workers = _function(text, "check_backend_processes")
    assert '"$IMAGES_TOOL" workers --container "$BACKEND_CONTAINER" --expect "$BACKEND_WORKERS"' in workers
    images = _function(text, "check_images")
    assert "compose config --format json" in images
    assert '--expect-revision "${SMOKE_EXPECT_REVISION:-}"' in images
    assert '"$IMAGES_TOOL" report' in _function(text, "write_report")


def test_the_pull_checks_the_host_before_it_sets_the_teardown_trap():
    text = _script()
    main = _function(text, "main")
    order = [main.index(call) for call in ("refuse_existing_stack", "refuse_emptying_a_kept_image_cache",
                                           "trap finish EXIT", "prepare_images")]
    assert order == sorted(order)
    guard = _function(text, "refuse_emptying_a_kept_image_cache")
    assert '[ "$SMOKE_IMAGES" = "pull" ] || return 0' in guard
    assert '[ "${CI:-}" = "true" ] || [ "${SMOKE_EMPTY_IMAGE_CACHE:-}" = "true" ]' in guard


# A docker that records its calls, lists no container, volume or image, and fails a
# compose pull; python3 is the interpreter running these tests.
_FAKE_DOCKER = """#!/bin/bash
echo "docker $*" >> "$CALL_LOG"
case "$*" in
  "compose "*" pull "*) exit 1 ;;
esac
exit 0
"""


def _run_pull_mode(tmp_path: Path, **extra: str) -> tuple:
    """smoke-fresh-clone.sh with SMOKE_IMAGES=pull against the fake docker: (exit code, calls, output)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in {"docker": _FAKE_DOCKER, "python3": f'#!/bin/bash\nexec "{sys.executable}" "$@"\n'}.items():
        (bin_dir / name).write_text(body, encoding="utf-8")
        (bin_dir / name).chmod((bin_dir / name).stat().st_mode | stat.S_IXUSR)
    log = tmp_path / "calls.log"
    log.touch()
    # Built from scratch, so the CI=true of the runner that runs these tests never leaks in.
    env = {"PATH": f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin", "CALL_LOG": str(log),
           "SMOKE_OUT_DIR": str(tmp_path / "out"), "RUNNER_TEMP": str(tmp_path),
           "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"),
           "SMOKE_IMAGES": "pull", "AUTOMATOS_IMAGE_TAG": "sha-0d9f0ac", **extra}
    proc = subprocess.run(["bash", str(_SCRIPT)], env=env, stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=60)
    return proc.returncode, log.read_text(encoding="utf-8").splitlines(), proc.stdout + proc.stderr


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_outside_ci_the_pull_refuses_to_empty_the_hosts_image_cache(tmp_path):
    code, calls, out = _run_pull_mode(tmp_path)
    assert code == 1, out
    assert "deletes every image on this host" in out and "SMOKE_EMPTY_IMAGE_CACHE=true" in out
    # It stops before pruning, pulling or tearing anything down, and writes no report.
    assert calls == ["docker ps -a --format {{.Names}}", "docker volume ls --format {{.Name}}"]
    assert not (tmp_path / "summary.md").exists()


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize("allowed", [{"CI": "true"}, {"SMOKE_EMPTY_IMAGE_CACHE": "true"}])
def test_in_ci_or_when_allowed_the_pull_empties_the_cache_and_never_builds(tmp_path, allowed):
    code, calls, out = _run_pull_mode(tmp_path, **allowed)
    assert code == 1, out  # the fake docker fails the pull
    assert "docker compose pull failed" in out
    assert "docker image prune -af" in calls
    assert not any(" build" in call or " up " in call for call in calls), calls
    assert calls[-1] == "docker compose -f docker-compose.yml -f docker-compose.images.yml down -v"
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "**FAIL**: docker compose pull failed" in summary
    assert "### Images: pull, first run, and what each container runs" in summary


def test_the_expected_worker_count_is_the_production_commands():
    cmd = [line for line in _DOCKERFILE.read_text(encoding="utf-8").splitlines() if line.startswith("CMD ")][-1]
    workers = re.search(r"--workers (\d+)", cmd)
    assert workers, f"the production CMD names no --workers: {cmd}"
    assert f'BACKEND_WORKERS="${{SMOKE_BACKEND_WORKERS:-{workers.group(1)}}}"' in _script()
