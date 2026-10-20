"""PRD-262 US-036: `make up` pulls the published images.

``docker-compose.yml`` names, for backend, frontend and workspace-worker, the image
``.github/workflows/images.yml`` publishes, ``ghcr.io/automatosai/automatos-<x>:
${AUTOMATOS_IMAGE_TAG:-edge}``, next to its ``build:``, so plain compose pulls first and
builds only an image it cannot find, and every local build is labelled
``org.opencontainers.image.revision`` from ``AUTOMATOS_REVISION`` (TESTER, §9-Q1).
make dev's overlay names its development builds ``automatos-<x>:dev`` with
``pull_policy: build``, so it never tags a published name. ``make up`` runs
``scripts/stack_images.py up`` (the rule: test_prd262_make_up_tag_rule.py) and then
starts the stack with ``--no-build``; ``make build`` builds the three under a
``local-*`` tag. images.yml tags each published image ``tree-<12>`` once per tree.
``docker-compose.images.yml`` and ``make up-images`` are gone.

Static reads, plus ``docker compose config`` where the docker CLI is installed: CI's
runners always have it, so there a missing CLI fails instead of skipping.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[2]
_BASE = _REPO / "docker-compose.yml"
_DEV = _REPO / "docker-compose.dev.yml"
_IMAGES_WORKFLOW = _REPO / ".github" / "workflows" / "images.yml"
_MAKEFILE = _REPO / "Makefile"
_REVISION_LABEL = "org.opencontainers.image.revision"
_LOCAL_REVISION = "${AUTOMATOS_REVISION:-unknown}"
_SECRETS = {"POSTGRES_PASSWORD": "test", "REDIS_PASSWORD": "test", "API_KEY": "test"}

# Service in docker-compose.yml -> image name the images workflow publishes.
_PUBLISHED = {
    "backend": "automatos-api",
    "frontend": "automatos-frontend",
    "workspace-worker": "automatos-workspace-worker",
}

_SPEC = importlib.util.spec_from_file_location("stack_images", _REPO / "scripts" / "stack_images.py")
stack = importlib.util.module_from_spec(_SPEC)
sys.modules["stack_images"] = stack
_SPEC.loader.exec_module(stack)

BASE = yaml.safe_load(_BASE.read_text(encoding="utf-8"))["services"]
DEV = yaml.safe_load(_DEV.read_text(encoding="utf-8"))["services"]
WORKFLOW = yaml.safe_load(_IMAGES_WORKFLOW.read_text(encoding="utf-8"))
PICK = next(step for step in WORKFLOW["jobs"]["changes"]["steps"] if step.get("id") == "pick")
IMAGES = [image for image in json.loads(PICK["env"]["IMAGES"]) if image["publish"]]
MERGE = WORKFLOW["jobs"]["merge"]


# --------------------------------------------------------------------------- docker-compose.yml


@pytest.mark.parametrize("service,image", sorted(_PUBLISHED.items()))
def test_each_app_service_names_its_published_image_next_to_its_build(service, image):
    """Pull first, build only when missing: compose's default with both keys and no pull_policy."""
    spec = BASE[service]
    assert spec["image"] == f"ghcr.io/automatosai/{image}:${{AUTOMATOS_IMAGE_TAG:-edge}}"
    assert spec["build"]["context"].startswith("./"), "the build stays, for make build and --build"
    assert "pull_policy" not in spec


@pytest.mark.parametrize("service", sorted(_PUBLISHED))
def test_every_local_build_carries_its_commit(service):
    """TESTER (§9-Q1): a local build is labelled as images.yml labels the published ones."""
    assert BASE[service]["build"]["labels"] == {_REVISION_LABEL: _LOCAL_REVISION}


def test_the_names_and_contexts_match_what_images_yml_publishes():
    """A rename in images.yml, or a moved context, must break compose and the tag picker together."""
    by_name = {f"automatos-{image['name']}": image for image in IMAGES}
    assert set(by_name) == set(_PUBLISHED.values())
    for service, name in _PUBLISHED.items():
        assert BASE[service]["build"]["context"] == f"./{by_name[name]['context']}"
    merged = {entry["name"]: entry["context"] for entry in MERGE["strategy"]["matrix"]["image"]}
    assert merged == {image["name"]: image["context"] for image in IMAGES}, "the merge job tags the same trees"


def test_the_tag_picker_knows_the_same_images():
    assert {app.service: (f"automatos-{app.name}", app.context) for app in stack.APP_IMAGES} == {
        service: (name, BASE[service]["build"]["context"][2:]) for service, name in _PUBLISHED.items()}
    assert stack.REPOSITORY == "ghcr.io/automatosai/automatos-"


def test_the_tag_picker_compares_against_the_published_web_apps_build_args():
    """What images.yml builds the frontend with, and what compose passes when .env sets nothing."""
    frontend = next(image for image in IMAGES if image["name"] == "frontend")
    published = dict(arg.split("=", 1) for arg in frontend["build_args"])
    assert {key: value for key, value in stack.PUBLISHED_FRONTEND.items() if value} == published
    args = BASE["frontend"]["build"]["args"]
    for key, value in stack.PUBLISHED_FRONTEND.items():
        assert args[key] == f"${{{key}:-{value}}}", key


def test_media_render_is_never_named_as_published():
    """§9-Q6: media-render is built and scanned, never published, so compose only builds it."""
    assert "image" not in BASE["media-render"]


# --------------------------------------------------------------------------- make dev's overlay


@pytest.mark.parametrize("service,image", sorted(_PUBLISHED.items()))
def test_make_dev_names_its_builds_locally_and_always_builds_them(service, image):
    """Else make dev would tag a development build ghcr.io/…:edge, which plain compose then runs."""
    assert DEV[service]["image"] == f"{image}:dev"
    assert DEV[service]["pull_policy"] == "build"


# --------------------------------------------------------------------------- the Makefile


def _recipe(target: str) -> list:
    text = _MAKEFILE.read_text(encoding="utf-8")
    body = text.split(f"\n{target}:\n", 1)[1].split("\n\n", 1)[0]
    return [line.strip().lstrip("-@") for line in body.splitlines() if line.startswith("\t")]


def test_the_images_overlay_and_make_up_images_are_gone():
    assert not (_REPO / "docker-compose.images.yml").exists()
    text = _MAKEFILE.read_text(encoding="utf-8")
    for gone in ("up-images", "IMAGES_COMPOSE", "docker-compose.images.yml"):
        assert gone not in text, gone


def test_make_up_picks_the_tag_then_starts_without_building():
    recipe = _recipe("up")
    assert re.search(r"^STACK_IMAGES\s*\?=\s*python3 scripts/stack_images\.py$",
                     _MAKEFILE.read_text(encoding="utf-8"), re.M)
    pick, start = recipe.index("$(STACK_IMAGES) up"), recipe.index("$(COMPOSE) up -d --no-build --remove-orphans")
    assert pick < start
    assert not any("--build" in line.split() for line in recipe), "compose never builds under an official tag"


def test_make_build_builds_the_three_under_its_local_tag_then_starts():
    recipe = _recipe("build")
    order = ["$(STACK_IMAGES) build", "$(COMPOSE) build backend frontend workspace-worker",
             "$(COMPOSE) up -d --no-build --remove-orphans"]
    assert [recipe.index(line) for line in order] == sorted(recipe.index(line) for line in order)


def test_make_exports_the_revision_and_never_an_empty_tag():
    text = _MAKEFILE.read_text(encoding="utf-8")
    contexts = re.search(r"^APP_CONTEXTS := (.+)$", text, re.M).group(1).split()
    assert contexts == [app.context for app in stack.APP_IMAGES]
    assert "git rev-parse --verify -q HEAD" in text and "git status --porcelain -- $(APP_CONTEXTS)" in text
    assert ",-dirty," in text and re.search(r"^export AUTOMATOS_REVISION$", text, re.M)
    assert re.search(r"^ifeq \(\$\(strip \$\(AUTOMATOS_IMAGE_TAG\)\),\)\nunexport AUTOMATOS_IMAGE_TAG\nendif$", text, re.M)


# --------------------------------------------------------------------------- images.yml: tree-<12>


def _tree_step() -> dict:
    return next(step for step in MERGE["steps"] if "tree-<12>" in step.get("name", ""))


def test_the_merge_job_tags_each_tree_once_and_never_on_a_schedule():
    step = _tree_step()
    assert step["if"] == "github.event_name != 'schedule'"
    assert step["env"]["CONTEXT"] == "${{ matrix.image.context }}"
    assert step["env"]["DIGEST"] == "${{ steps.index.outputs.digest }}"
    run = step["run"]
    assert 'python3 scripts/stack_images.py tree-tag --context "$CONTEXT"' in run
    assert 'python3 scripts/stack_images.py published "${IMAGE}:${tag}"' in run
    push = 'docker buildx imagetools create -t "${IMAGE}:${tag}" "${IMAGE}@${DIGEST}"'
    assert run.count(push) == 1 and run.index("1)") < run.index(push) < run.index("*)"), \
        "only a tree the registry says is unpublished is pushed"


def test_the_tree_is_tagged_after_the_index_is_signed_from_a_checkout():
    names = [step.get("name", "") for step in MERGE["steps"]]
    assert names.index("Sign (keyless)") < names.index(_tree_step()["name"])
    checkout = next(step for step in MERGE["steps"] if "actions/checkout" in step.get("uses", ""))
    assert checkout["with"]["persist-credentials"] is False


# --------------------------------------------------------------------------- docker compose config


def _compose_config(*files: str, **env: str) -> dict:
    """``docker compose config --format json`` for ``files`` with only ``env`` and the three secrets."""
    docker = shutil.which("docker")
    if docker is None:
        if os.environ.get("CI") == "true":
            pytest.fail("the docker CLI is missing on this CI runner: these checks need `docker compose config`")
        pytest.skip("needs the docker CLI, which CI's runners have")
    environ = {key: value for key, value in os.environ.items()
               if not key.startswith(("AUTOMATOS_", "NEXT_PUBLIC_", "COMPOSE_", "POSTGRES_", "REDIS_", "API_"))}
    with tempfile.NamedTemporaryFile("w", suffix=".env") as empty:  # never a checkout's own .env
        command = [docker, "compose", "--project-directory", str(_REPO), "--env-file", empty.name]
        for name in files:
            command += ["-f", str(_REPO / name)]
        proc = subprocess.run([*command, "config", "--format", "json"], cwd=str(_REPO), capture_output=True,
                              text=True, timeout=120, env={**environ, **_SECRETS, **env})
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)["services"]


def test_make_devs_stack_names_no_published_image():
    services = _compose_config("docker-compose.yml", "docker-compose.dev.yml")
    assert {name: spec.get("image") for name, spec in services.items()
            if str(spec.get("image", "")).startswith("ghcr.io/")} == {}


def test_a_local_tag_names_the_three_app_images_and_keeps_their_builds():
    """TESTER: `AUTOMATOS_IMAGE_TAG=local-<ref> docker compose up --build` builds under that tag."""
    services = _compose_config("docker-compose.yml", AUTOMATOS_IMAGE_TAG="local-x", AUTOMATOS_REVISION="abc123")
    for service, name in _PUBLISHED.items():
        spec = services[service]
        assert spec["image"] == f"ghcr.io/automatosai/{name}:local-x"
        build = spec["build"]
        assert Path(build["context"]) == _REPO / BASE[service]["build"]["context"][2:]
        assert build.get("target") == BASE[service]["build"].get("target")
        assert build["labels"] == {_REVISION_LABEL: "abc123"}


def test_with_no_tag_compose_runs_edge_and_labels_a_build_unknown():
    services = _compose_config("docker-compose.yml")
    for service, name in _PUBLISHED.items():
        assert services[service]["image"] == f"ghcr.io/automatosai/{name}:edge"
        assert services[service]["build"]["labels"] == {_REVISION_LABEL: "unknown"}
