"""`make up-images`: the local edition from the published images, not a source build.

``docker-compose.images.yml`` overrides ``docker-compose.yml`` so the backend, web
app and workspace worker run the images ``.github/workflows/images.yml`` publishes
to GHCR. Static checks only: the override names the published images, builds
nothing, and never mounts the checkout's source over the image's code (which
would quietly run local code under a published tag).
"""
from __future__ import annotations

import pathlib

import pytest
import yaml

_REPO = pathlib.Path(__file__).resolve().parents[2]
_OVERRIDE = _REPO / "docker-compose.images.yml"
_BASE = _REPO / "docker-compose.yml"
_IMAGES_WORKFLOW = _REPO / ".github" / "workflows" / "images.yml"
_MAKEFILE = _REPO / "Makefile"

# Service in docker-compose.yml -> image name the images workflow publishes.
_PUBLISHED = {
    "backend": "automatos-api",
    "frontend": "automatos-frontend",
    "workspace-worker": "automatos-workspace-worker",
}


class _ComposeLoader(yaml.SafeLoader):
    """SafeLoader that understands compose's !reset and !override merge tags."""


def _tagged(tag: str):
    def construct(loader, node):
        if isinstance(node, yaml.SequenceNode):
            value = loader.construct_sequence(node)
        elif isinstance(node, yaml.MappingNode):
            value = loader.construct_mapping(node)
        else:
            value = loader.construct_scalar(node)
        return {"__tag__": tag, "value": value}
    return construct


_ComposeLoader.add_constructor("!reset", _tagged("reset"))
_ComposeLoader.add_constructor("!override", _tagged("override"))


@pytest.fixture(scope="module")
def override() -> dict:
    """The override file's services, with its merge tags kept visible."""
    return yaml.load(_OVERRIDE.read_text(encoding="utf-8"), Loader=_ComposeLoader)["services"]


def test_override_covers_services_that_exist(override):
    """Every overridden service exists in docker-compose.yml (a typo would add a new one)."""
    base = yaml.safe_load(_BASE.read_text(encoding="utf-8"))["services"]
    assert set(override) == set(_PUBLISHED)
    assert set(override) <= set(base)


@pytest.mark.parametrize("service,image", sorted(_PUBLISHED.items()))
def test_each_service_runs_its_published_image(override, service, image):
    """The image is the GHCR one, tag from AUTOMATOS_IMAGE_TAG (default edge), no build."""
    spec = override[service]
    assert spec["image"] == f"ghcr.io/automatosai/{image}:${{AUTOMATOS_IMAGE_TAG:-edge}}"
    assert spec["build"]["__tag__"] == "reset", "build: !reset — nothing is built from source"


def test_names_match_what_the_images_workflow_publishes():
    """The workflow publishes automatos-<name>; a rename there must break this file's names."""
    workflow = yaml.safe_load(_IMAGES_WORKFLOW.read_text(encoding="utf-8"))
    names = {entry["name"] for entry in workflow["jobs"]["build"]["strategy"]["matrix"]["image"]}
    assert {f"automatos-{name}" for name in names} == set(_PUBLISHED.values())


def test_backend_never_mounts_the_checkout_over_the_image(override):
    """Only data and the workspace folder: the image's code and entrypoint are used as published."""
    volumes = override["backend"]["volumes"]
    assert volumes["__tag__"] == "override"
    sources = [v.split(":", 1)[0] for v in volumes["value"]]
    assert "./orchestrator" not in sources
    assert not any("docker-entrypoint.sh" in v for v in volumes["value"])
    assert {"backend_data", "backend_logs"} <= set(sources)
    assert any(v.startswith("${AUTOMATOS_WORKSPACE_DIR") for v in volumes["value"]), (
        "the workspace folder must stay mounted, or session mode and Deliverables lose it"
    )


def test_make_up_images_pulls_and_never_builds():
    text = _MAKEFILE.read_text(encoding="utf-8")
    body = text.split("\nup-images:\n", 1)[1].split("\n\n", 1)[0]
    assert "docker-compose.images.yml" in text
    assert "pull backend frontend workspace-worker" in body
    assert "up -d --no-build" in body
