"""PRD-262 US-042: the kind end-to-end test's datastores are pinned by digest.

``deploy/kind/datastores.yaml`` runs throwaway PostgreSQL (pgvector) and Redis for
``deploy/kind/e2e.sh``. Each image is ``name:tag@sha256:<digest>``, on the tag line
``docker-compose.yml`` runs, so every run gets the same bytes. e2e.sh names exactly the
references the manifests name (``DATASTORE_IMAGES``); it leaves those pinned by digest to
the node's kubelet, which pulls them by that digest, so what runs is what is pinned.

The digests are not held equal to compose's: Dependabot moves compose's every week,
and an equality check would fail its pull requests. The tag line is held.

Pure file reads: no Docker, no cluster, no network.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[2]
KIND = _REPO / "deploy" / "kind"
E2E = (KIND / "e2e.sh").read_text(encoding="utf-8")
COMPOSE = yaml.safe_load((_REPO / "docker-compose.yml").read_text(encoding="utf-8"))["services"]
# name:tag@sha256:<64 hex>, as test_prd262_dependabot.py pins compose's third-party images.
PINNED = re.compile(r"^[^\s@]+:[\w][\w.-]{0,127}@sha256:[0-9a-f]{64}$")
_ARRAY = re.compile(r"^DATASTORE_IMAGES=\((?P<body>.*?)\)$", re.M | re.S)


def _images(name: str) -> dict[str, str]:
    """Deployment name -> its one container's image, for every Deployment in a manifest."""
    found = {}
    for document in yaml.safe_load_all((KIND / name).read_text(encoding="utf-8")):
        if (document or {}).get("kind") == "Deployment":
            (container,) = document["spec"]["template"]["spec"]["containers"]
            found[document["metadata"]["name"]] = container["image"]
    return found


DATASTORES = _images("datastores.yaml")


def _preloaded() -> list[str]:
    match = _ARRAY.search(E2E)
    assert match, "e2e.sh has no DATASTORE_IMAGES=( … ) array"
    words = [line.split("#", 1)[0].strip() for line in match.group("body").splitlines()]
    return [word for line in words for word in line.split()]


def test_the_kind_datastores_are_postgres_and_redis():
    assert set(DATASTORES) == {"postgres", "redis"}


def test_each_kind_datastore_is_pinned_as_tag_at_digest():
    assert {name: image for name, image in DATASTORES.items() if not PINNED.match(image)} == {}


def test_each_kind_datastore_stays_on_composes_tag_line():
    for name, image in DATASTORES.items():
        assert image.split("@", 1)[0] == COMPOSE[name]["image"].split("@", 1)[0], name


def test_e2e_preloads_exactly_what_the_manifests_run():
    expected = sorted([*DATASTORES.values(), *_images("otel-collector.yaml").values()])
    assert sorted(_preloaded()) == expected


def test_the_pin_rule_on_fixtures():
    assert PINNED.match("redis:7-alpine@sha256:" + "8" * 64)
    assert not PINNED.match("redis:7-alpine")
    assert not PINNED.match("redis@sha256:" + "8" * 64)
    assert not PINNED.match("redis:7-alpine@sha256:" + "8" * 63)
