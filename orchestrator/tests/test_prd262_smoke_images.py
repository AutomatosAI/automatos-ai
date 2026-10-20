"""PRD-262 US-003: a lane that boots what we publish.

Static checks on ``.github/workflows/smoke-images.yml``, the smoke script's image
modes and ``images.yml``'s tags. The lane itself runs only in CI:

* ``smoke-images`` runs after images.yml publishes on main (workflow_run) and by hand
  with a tag, on ubuntu-24.04 and ubuntu-24.04-arm, and boots it the way a self-hoster
  does since PRD-262 US-036: ``make up`` on the clean checkout, with no tag exported
  after a push (it must pick the commit's own images), pinned on a weekly or manual run;
* ``make-up-pr`` runs ``make up`` on a pull request with the merge base's ``sha-<7>``
  pinned and checks that nothing was built (scripts/ci/smoke-make-up.sh);
* PRD-262 US-037: the main runs and ``make-up-pr`` export no secret, so ``make up`` must
  write three generated ones to ``.env``; then ``make env`` on a started install (its
  database volume exists) must leave a placeholder ``.env`` as it was and exit non-zero
  (scripts/ci/smoke-make-env.sh);
* the pull-request variant watches compose, the Makefile, ``envs/`` and every
  images.yml input, and builds the three amd64 images with images.yml's context,
  target, build args, labels and GHA cache (read only), tagged as compose expects;
* images.yml's merge job no longer pushes ``edge``; ``promote-edge`` moves it after
  both arches pass, for the images the merge job publishes;
* the script empties the Docker cache before make up, never builds in an image mode,
  times the boots while make up runs, and asserts the backend's worker count:
  envs/api.defaults' WEB_CONCURRENCY, or the pull-request variant's 2 (PRD-262 US-029:
  the production CMD names no --workers);
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
_COMPOSE = _REPO / "docker-compose.yml"
_MAKE_UP_SCRIPT = _REPO / "scripts" / "ci" / "smoke-make-up.sh"
_MAKE_ENV_SCRIPT = _REPO / "scripts" / "ci" / "smoke-make-env.sh"
# The three secrets make env writes (PRD-262 US-037), and the lane's other settings.
_SECRETS = ("POSTGRES_PASSWORD", "REDIS_PASSWORD", "API_KEY")
_SETTINGS = ("AUTH_EDITION", "REQUIRE_AUTH", "DEFAULT_WORKSPACE_ID")
_GENERATED_SECRET = "GENERATED_SECRET='^(POSTGRES_PASSWORD|REDIS_PASSWORD|API_KEY)=[0-9a-f]{48}$'"
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
# docker-compose.yml's service for each published image.
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
           "scripts/ci/smoke_images.py", "scripts/ci/smoke_images_report.py", ".github/workflows/smoke-images.yml",
           "scripts/ci/smoke-make-up.sh", "scripts/stack_images.py",
           # PRD-262 US-037: make env, its CI check, the template it copies and the guard.
           "scripts/stack_env.sh", "scripts/ci/smoke-make-env.sh", ".env.example",
           "orchestrator/docker-secrets-guard.sh"}
    assert own <= set(paths)


# --------------------------------------------------------------------------- the main runs


def test_main_runs_boot_on_both_arches():
    job = JOBS["smoke-images"]
    assert job["runs-on"] == "${{ matrix.runner }}"
    arches = {entry["arch"]: entry["runner"] for entry in job["strategy"]["matrix"]["include"]}
    assert arches == {"amd64": "ubuntu-24.04", "arm64": "ubuntu-24.04-arm"}
    assert job["strategy"]["fail-fast"] is False


def test_main_runs_run_make_up_with_the_tag_pinned_only_when_make_up_must_not_pick_it():
    """After a push the pin is empty: make up picks the commit's sha-<7> itself (PRD-262 US-036)."""
    target = _step("smoke-images", lambda step: step.get("id") == "target")
    assert "smoke_images.py target" in target["run"] and '--outputs "$GITHUB_OUTPUT"' in target["run"]
    assert target["env"]["TAG"] == "${{ inputs.tag }}", "the dispatch input goes through env, never inline"
    env = _boot_step("smoke-images")["env"]
    assert env["SMOKE_IMAGES"] == "pull"
    assert env["AUTOMATOS_IMAGE_TAG"] == "${{ steps.target.outputs.pin }}"
    assert env["SMOKE_EXPECT_REVISION"] == "${{ steps.target.outputs.revision }}"


def _fresh_env() -> dict:
    return next(step for step in _yaml(_FRESH)["jobs"]["fresh-clone-boot"]["steps"]
                if step.get("run") == "bash scripts/ci/smoke-fresh-clone.sh")["env"]


@pytest.mark.parametrize("job", ["smoke-images", "smoke-images-pr"])
def test_both_variants_use_the_fresh_clone_lanes_settings_and_upload_the_records(job):
    fresh = _fresh_env()
    env = _boot_step(job)["env"]
    for key in _SETTINGS:
        assert env[key] == fresh[key], key
    upload = _step(job, lambda step: "upload-artifact" in step.get("uses", ""))
    assert upload["if"] == "always()"
    assert upload["with"]["path"].rstrip("/") == env["SMOKE_OUT_DIR"].rstrip("/")
    assert "continue-on-error" not in _boot_step(job), "a boot that fails fails the job"


def test_the_pr_variant_boots_on_the_fresh_clone_lanes_throwaway_secrets():
    fresh, env = _fresh_env(), _boot_step("smoke-images-pr")["env"]
    for key in _SECRETS:
        assert env[key] == fresh[key], key


def test_the_main_runs_export_no_secret_so_make_up_writes_them():
    """PRD-262 US-037: make up on a clean checkout with no secrets exported."""
    env = _boot_step("smoke-images")["env"]
    assert not set(_SECRETS) & set(env), "make env must generate them, as on a self-hoster's checkout"


# --------------------------------------------------------------------------- PRD-262 US-037: the secrets


def test_the_pull_exports_no_secret_and_finds_three_generated_in_env():
    text = _script()
    block = text.split('if [ "$SMOKE_IMAGES" = "pull" ]; then\n  unset ', 1)[1].split("\nfi\n", 1)[0]
    assert block.startswith("POSTGRES_PASSWORD REDIS_PASSWORD API_KEY\nelse\n"), block
    assert 'export API_KEY="${API_KEY:-smoke_api_key}"' in block, "the other modes keep the throwaway secrets"
    assert _GENERATED_SECRET in text
    check = _function(text, "check_generated_secrets")
    assert '[ "$SMOKE_IMAGES" = "pull" ] || return 0' in check
    assert 'grep -cE "$GENERATED_SECRET" "$ENV_FILE"' in check and '= "3" ]' in check
    main = _function(text, "main")
    assert '[ ! -e "$ENV_FILE" ] || env_existed="true"' in main
    order = ["trap finish EXIT", "env_existed=", "prepare_images", "finish_make", "check_generated_secrets",
             "warm_restarts"]
    assert [main.index(step) for step in order] == sorted(main.index(step) for step in order)


def test_the_make_up_smoke_exports_no_secret_and_finds_three_generated_in_env():
    text = _MAKE_UP_SCRIPT.read_text(encoding="utf-8")
    assert _GENERATED_SECRET in text
    main = _function(text, "main")
    order = ["refuse_existing_stack", "refuse_an_existing_env", "unset POSTGRES_PASSWORD REDIS_PASSWORD API_KEY",
             "trap finish EXIT", "make up 2>&1", "check_generated_secrets", "wait_for fresh"]
    assert [main.index(step) for step in order] == sorted(main.index(step) for step in order)
    assert '[ ! -e "$ENV_FILE" ] || fail' in _function(text, "refuse_an_existing_env")
    check = _function(text, "check_generated_secrets")
    assert 'grep -cE "$GENERATED_SECRET" "$ENV_FILE"' in check and '= "3" ]' in check


def test_make_up_pr_then_checks_make_env_on_a_started_install():
    steps = JOBS["make-up-pr"]["steps"]
    runs = [step.get("run") for step in steps]
    make_up = runs.index("bash scripts/ci/smoke-make-up.sh")
    make_env = runs.index("bash scripts/ci/smoke-make-env.sh")
    assert make_up < make_env, "after the make up smoke's teardown removed the stack's volumes"
    step = steps[make_env]
    assert "if" not in step and "continue-on-error" not in step
    assert not set(_SECRETS) & set(step.get("env") or {})


def test_the_make_env_check_leaves_placeholders_on_a_started_install():
    text = _MAKE_ENV_SCRIPT.read_text(encoding="utf-8")
    assert 'VOLUME="automatos_postgres_data"' in text and "set -uo pipefail" in text
    main = _function(text, "main")
    order = ["refuse_an_existing_volume", "unset POSTGRES_PASSWORD REDIS_PASSWORD API_KEY", "trap finish EXIT",
             "set_env_aside", 'docker volume create "$VOLUME"', "cp .env.example", "check_make_env_changed_nothing"]
    assert [main.index(step) for step in order] == sorted(main.index(step) for step in order)
    assert 'docker volume inspect "$VOLUME"' in _function(text, "refuse_an_existing_volume")
    check = _function(text, "check_make_env_changed_nothing")
    assert "make env > \"$OUTPUT\" 2>&1 || code=$?" in check
    assert '[ "$code" -ne 0 ] || fail' in check
    assert 'cmp -s "$BEFORE" "$ENV_FILE" || fail' in check
    assert 'grep -q "ALTER USER" "$OUTPUT"' in check
    assert 'grep -q "the $VOLUME volume exists, so this install has started" "$OUTPUT"' in check
    finish = _function(text, "finish")
    assert 'docker volume rm "$VOLUME"' in finish and 'mv -f "$SAVED_ENV" "$ENV_FILE"' in finish


# --------------------------------------------------------------------------- edge


def test_the_merge_job_no_longer_pushes_edge():
    merge = IMAGES_WORKFLOW["jobs"]["merge"]
    meta = next(step for step in merge["steps"] if step.get("id") == "meta")
    tags = [line.strip() for line in meta["with"]["tags"].splitlines() if line.strip()]
    # On for every run but the weekly schedule (test_prd262_weekly_rebuild.py pins the switch).
    assert any(tag.split(",", 1)[0] == "type=sha" for tag in tags)
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
    services = _yaml(_COMPOSE)["services"]
    return {service: services[service]["image"] for service in SERVICES.values()}


@pytest.mark.parametrize("image", PUBLISHED, ids=lambda image: image["name"])
def test_the_pr_variant_builds_each_image_as_images_yml_does(image):
    name = image["name"]
    build = _step("smoke-images-pr", lambda step: step.get("name") == f"Build {name} (amd64)")["with"]
    assert build["context"] == image["context"]
    assert build.get("target", "") == image["target"]
    assert build["platforms"] == "linux/amd64"
    assert build.get("build-args", "").split() == image["build_args"], "the API takes none (US-027)"
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


def test_every_mode_runs_docker_compose_yml_alone():
    """docker-compose.yml names the published images itself since PRD-262 US-036: no overlay."""
    text = _script()
    assert "docker-compose.images.yml" not in text
    assert re.search(r'^COMPOSE_ARGS=\(-f "\$COMPOSE_FILE"\)$', text, re.M)
    assert 'docker compose "${COMPOSE_ARGS[@]}" "$@"' in _function(text, "compose")
    modes = _function(text, "check_mode")
    assert '""|pull) ;;' in modes, "make up picks the tag when none is pinned"
    assert "prebuilt)" in modes and "AUTOMATOS_IMAGE_TAG" in modes.split("prebuilt)", 1)[1]
    assert '[ -n "${AUTOMATOS_IMAGE_TAG:-}" ] || unset AUTOMATOS_IMAGE_TAG' in text, \
        "an empty tag would make compose run :edge instead of the tag make up wrote to .env"


def test_the_pull_is_make_up_from_an_empty_cache_and_image_modes_never_build():
    text = _script()
    prepare = _function(text, "prepare_images")
    assert '"") build_images ;;' in prepare and "pull) pull_images ;;" in prepare
    assert "build_images" not in prepare.split("prebuilt)", 1)[1]
    assert not re.search(r"\bup\b[^\n]*--build\b", text), "no `up --build`: image modes run what was pulled or built in the job"
    pull = _function(text, "pull_images")
    assert pull.index("empty_docker_cache") < pull.index('"$IMAGES_TOOL" pull')
    assert '--log "$MAKE_UP_LOG" -- \\\n      make up\n' in pull, "make up runs, its output kept for the 429 search"
    assert 'echo "$?" > "$MAKE_EXIT"' in pull and "make_pid=$!" in pull, "in the background, its exit kept"
    assert "docker image prune -af" in _function(text, "empty_docker_cache")
    assert "compose up -d --no-build postgres redis minio backend" in _function(text, "boot_fresh")


def test_the_boots_are_timed_while_make_up_runs_and_make_up_must_exit_0():
    """make up returns only after the frontend started, so polling after it would overstate both boots.

    make build (the source build, PRD-262 US-039) runs the same way.
    """
    text = _script()
    assert re.search(r'^  pull\) MAKE_TARGET="up" ;;$', text, re.M)
    assert 'if [ -n "$MAKE_TARGET" ]; then\n    boot_fresh_from_make' in _function(text, "boot_fresh")
    fresh = _function(text, "boot_fresh_from_make")
    assert 'wait_backend fresh --wait-start --start-timeout "$start_wait" --stop-file "$MAKE_EXIT"' in fresh
    frontend = _function(text, "boot_frontend_from_make")
    assert '--wait-start --stop-file "$MAKE_EXIT" --published-port 3000' in frontend
    finish = _function(text, "finish_make")
    assert 'wait "$make_pid"' in finish and '= "0" ] || fail "make ${MAKE_TARGET} failed"' in finish
    main = _function(text, "main")
    order = ["start_event_log", "prepare_images", "boot_fresh", "boot_frontend", "finish_make",
             "check_images", "warm_restarts"]
    assert [main.index(step) for step in order] == sorted(main.index(step) for step in order)
    assert 'wait "$make_pid"' in _function(text, "finish"), "the teardown waits for a make up still running"


def test_a_source_build_is_tagged_local_and_labelled_with_head():
    """The fresh-clone lane builds as make build does (PRD-262 US-036, US-039)."""
    text = _script()
    build_mode = text.split('if [ -z "$SMOKE_IMAGES" ]; then', 1)[1].split("\nfi\n", 1)[0]
    assert 'AUTOMATOS_REVISION="${AUTOMATOS_REVISION:-$(git rev-parse HEAD 2>/dev/null)}"' in build_mode
    assert 'export AUTOMATOS_IMAGE_TAG="${AUTOMATOS_IMAGE_TAG:-local-${AUTOMATOS_REVISION:0:7}}"' in build_mode
    assert 'SMOKE_EXPECT_REVISION="${SMOKE_EXPECT_REVISION:-$AUTOMATOS_REVISION}"' in build_mode
    assert re.search(r'^IMAGES_MODE="\$\{SMOKE_IMAGES:-build\}"$', text, re.M)
    images = _function(text, "check_images")
    assert '--mode "$IMAGES_MODE"' in images and "return 0" not in images, "every mode checks its images"


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


# A docker that records its calls and lists no container, volume or image, and a make
# whose `make up` fails; python3 is the interpreter running these tests.
_FAKE_DOCKER = """#!/bin/bash
echo "docker $*" >> "$CALL_LOG"
exit 0
"""
_FAKE_MAKE = """#!/bin/bash
echo "make $*" >> "$CALL_LOG"
echo "make-tag ${AUTOMATOS_IMAGE_TAG-unset}" >> "$CALL_LOG"
exit 1
"""


def _run_pull_mode(tmp_path: Path, **extra: str) -> tuple:
    """smoke-fresh-clone.sh with SMOKE_IMAGES=pull against the fakes: (exit code, calls, output)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stubs = {"docker": _FAKE_DOCKER, "make": _FAKE_MAKE, "python3": f'#!/bin/bash\nexec "{sys.executable}" "$@"\n'}
    for name, body in stubs.items():
        (bin_dir / name).write_text(body, encoding="utf-8")
        (bin_dir / name).chmod((bin_dir / name).stat().st_mode | stat.S_IXUSR)
    log = tmp_path / "calls.log"
    log.touch()
    # Built from scratch, so the CI=true of the runner that runs these tests never leaks in.
    env = {"PATH": f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin", "CALL_LOG": str(log),
           "SMOKE_OUT_DIR": str(tmp_path / "out"), "RUNNER_TEMP": str(tmp_path),
           "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"),
           "SMOKE_IMAGES": "pull", "AUTOMATOS_IMAGE_TAG": "", "SMOKE_MAX_WAIT_SECONDS": "20", **extra}
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
def test_in_ci_or_when_allowed_the_pull_empties_the_cache_and_runs_make_up(tmp_path, allowed):
    code, calls, out = _run_pull_mode(tmp_path, **allowed)
    assert code == 1, out  # the fake make fails make up
    assert "make up failed" in out
    assert "docker image prune -af" in calls
    docker = [call for call in calls if call.startswith("docker ")]
    assert not any(" build" in call or " up " in call or " pull" in call for call in docker), docker
    if Path("/proc/net/route").exists():  # smoke_images.py reads it before it runs make up (Linux, as in CI)
        assert calls.index("docker image prune -af") < calls.index("make up")
        assert "make-tag unset" in calls, "an empty AUTOMATOS_IMAGE_TAG is never passed to make up"
    assert calls[-1] == "docker compose -f docker-compose.yml down -v"
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "**FAIL**: make up failed" in summary
    assert "### Images: pull, first run, and what each container runs" in summary


# --------------------------------------------------------------------------- make-up-pr


def test_make_up_pr_pins_the_merge_bases_published_tag():
    job = JOBS["make-up-pr"]
    assert job["if"] == "github.event_name == 'pull_request'" and job["runs-on"] == "ubuntu-24.04"
    base = _step("make-up-pr", lambda step: step.get("id") == "base")
    assert 'compare/$BASE_SHA...$HEAD_SHA" --jq .merge_base_commit.sha' in base["run"]
    assert "smoke_images.py base --merge-base" in base["run"] and '--outputs "$GITHUB_OUTPUT"' in base["run"]
    assert base["env"]["BASE_SHA"] == "${{ github.event.pull_request.base.sha }}"
    run = _step("make-up-pr", lambda step: step.get("run") == "bash scripts/ci/smoke-make-up.sh")
    assert run["env"]["AUTOMATOS_IMAGE_TAG"] == "${{ steps.base.outputs.tag }}"
    assert run["env"]["SMOKE_EXPECT_REVISION"] == "${{ steps.base.outputs.revision }}"
    fresh = _fresh_env()
    for key in _SETTINGS:
        assert run["env"][key] == fresh[key], key
    assert not set(_SECRETS) & set(run["env"]), "PRD-262 US-037: make up generates the secrets"
    upload = _step("make-up-pr", lambda step: "upload-artifact" in step.get("uses", ""))
    assert upload["if"] == "always()" and upload["with"]["path"].rstrip("/") == run["env"]["SMOKE_OUT_DIR"].rstrip("/")


def test_the_make_up_smoke_checks_what_make_up_promises():
    text = _MAKE_UP_SCRIPT.read_text(encoding="utf-8")
    main = _function(text, "main")
    # The make up call itself: main's first lines also name make up in a message.
    order = [main.index(call) for call in ("refuse_existing_stack", "trap finish EXIT", "make up 2>&1",
                                           "wait_for fresh", "wait_for frontend", "check_nothing_was_built")]
    assert order == sorted(order)
    assert 'make up 2>&1 | tee "$MAKE_UP_LOG" || fail "make up failed"' in main
    assert '"ready=${API_BASE}/health/ready"' in main and '"ui=${FRONTEND_URL}"' in main
    check = _function(text, "check_nothing_was_built")
    assert '"$IMAGES_TOOL" images --mode pull' in check and '--expect-revision "${SMOKE_EXPECT_REVISION:-}"' in check
    assert "--build" not in text and "set -uo pipefail" in text
    assert "docker compose down -v" in _function(text, "finish")


def test_the_expected_worker_count_is_the_one_compose_sets():
    """PRD-262 US-029: the production CMD names no --workers, so compose's WEB_CONCURRENCY sets the count."""
    cmd = [line for line in _DOCKERFILE.read_text(encoding="utf-8").splitlines() if line.startswith("CMD ")][-1]
    assert "--workers" not in cmd, f"the production CMD names --workers, which WEB_CONCURRENCY cannot change: {cmd}"
    defaults = (_REPO / "envs" / "api.defaults").read_text(encoding="utf-8")
    workers = re.search(r"^WEB_CONCURRENCY=(\d+)$", defaults, re.M)
    assert workers, "envs/api.defaults sets no WEB_CONCURRENCY"
    assert f'BACKEND_WORKERS="${{SMOKE_BACKEND_WORKERS:-{workers.group(1)}}}"' in _script()


def test_the_pull_request_variant_runs_enough_workers_to_kill_one():
    """US-008's kill-a-worker step needs a uvicorn manager: WEB_CONCURRENCY=2 or more, through an override."""
    text = _script()
    restart = re.search(r"^RESTART_WORKERS=(\d+)$", text, re.M)
    assert restart and int(restart.group(1)) >= 2
    override = _function(text, "use_restart_workers")
    assert override.lstrip().startswith('[ "$SMOKE_IMAGES" = "prebuilt" ] || return 0'), "the main runs keep compose's count"
    assert "WEB_CONCURRENCY" in override and '"$RESTART_WORKERS"' in override
    assert 'COMPOSE_ARGS+=(-f "$WORKERS_OVERRIDE_FILE")' in override
    assert 'BACKEND_WORKERS="$RESTART_WORKERS"' in override
    assert re.search(r'^WORKERS_OVERRIDE_FILE="\$\{RUNNER_TEMP:-/tmp\}/', text, re.M), "written outside the checkout"
    main = _function(text, "main")
    assert main.index("use_restart_workers") < main.index("boot_fresh")
