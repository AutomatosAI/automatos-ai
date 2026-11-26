"""PRD-280 US-601: the chart's memory store runs the build the compose file runs.

``docker-compose.yml`` runs ``qdrant/qdrant`` pinned by digest, and the chart's ``qdrant.image``
names a repository and a digest. A Helm chart can read only its own files, so the two cannot
share one line; this test holds them equal instead. When the compose pin moves (by hand or in
Dependabot's weekly compose pull request), move ``qdrant.image.digest`` in
``charts/automatos/values.yaml`` in the same change.

Pure file reads: no Docker, no helm, no network.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))["services"]
VALUES = yaml.safe_load((ROOT / "charts" / "automatos" / "values.yaml").read_text(encoding="utf-8"))
# name:tag@sha256:<64 hex>, as test_prd262_dependabot.py pins compose's third-party images.
PINNED = re.compile(r"^(?P<name>[^\s@:]+(?::\d+)?(?:/[^\s@:]+)*):[\w][\w.-]{0,127}@(?P<digest>sha256:[0-9a-f]{64})$")
DOCKER_HUB = "docker.io/"


def qualified(name: str) -> str:
    """An image name as a registry reads it: Docker Hub's host added to a name without one."""
    first = name.split("/", 1)[0]
    has_host = "/" in name and ("." in first or ":" in first or first == "localhost")
    return name if has_host else f"{DOCKER_HUB}{name}"


def _compose_pin() -> re.Match:
    match = PINNED.match(COMPOSE["qdrant"]["image"])
    assert match, f"docker-compose.yml's qdrant image is not name:tag@sha256:<digest>: {COMPOSE['qdrant']['image']}"
    return match


def test_the_chart_runs_the_repository_compose_runs():
    assert VALUES["qdrant"]["image"]["repository"] == qualified(_compose_pin()["name"])


def test_the_chart_runs_the_digest_compose_pins():
    compose_digest = _compose_pin()["digest"]
    assert VALUES["qdrant"]["image"]["digest"] == compose_digest, (
        "charts/automatos/values.yaml qdrant.image.digest differs from docker-compose.yml's qdrant pin: "
        f"set it to {compose_digest} in the same change"
    )


def test_the_memory_store_is_off_by_default():
    assert VALUES["qdrant"]["enabled"] is False


def test_the_name_rule_on_fixtures():
    assert qualified("qdrant/qdrant") == "docker.io/qdrant/qdrant"
    assert qualified("redis") == "docker.io/redis"
    assert qualified("ghcr.io/automatosai/automatos-api") == "ghcr.io/automatosai/automatos-api"
    assert qualified("localhost:5000/qdrant") == "localhost:5000/qdrant"
    assert PINNED.match("qdrant/qdrant:latest@sha256:" + "9" * 64)["name"] == "qdrant/qdrant"
    assert not PINNED.match("qdrant/qdrant@sha256:" + "9" * 64)
    assert not PINNED.match("qdrant/qdrant:latest")
