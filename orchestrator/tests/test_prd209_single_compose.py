"""PRD-209 S9 — exactly one canonical compose file (+ one documented override).

The root ``docker-compose.yml`` is the local stack. Six ``infrastructure/docker-compose*.yml``
files (a heavyweight 19-service production-mirror requiring sibling repos; ``.voice``
referenced services decommissioned by #625) predated it and drifted — a second source
of compose truth. They are deleted; this guard asserts one tracked STACK file remains anywhere in
the repo. ``docker-compose.dev.yml`` (PRD-233 slim pass) is allowed alongside it
because it is an OVERRIDE, not a second stack: compose only reads it when it is
passed explicitly (``-f docker-compose.yml -f docker-compose.dev.yml``), and the
guard below proves it declares no service the canonical file does not, and
carries no ``image:`` of its own — so it can never become a rival source of
truth the way the infrastructure/ files did.

``docker-compose.images.yml`` (``make up-images``) is the one override allowed an
``image:``, and only a narrow one: for a service the canonical file BUILDS from
source, the image this repo's own ``.github/workflows/images.yml`` publishes from
that same Dockerfile (``ghcr.io/automatosai/automatos-<name>:${AUTOMATOS_IMAGE_TAG:-edge}``).
It swaps where the same build comes from; it cannot add a service or point a
service at anything else.

Pure/static — reads the git index (`git ls-files`); no Docker.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMPOSE = re.compile(r"(?:^|/)docker-compose[^/]*\.ya?ml$")


def _tracked_compose_files() -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"git ls-files failed: {proc.stderr}"
    return [ln for ln in proc.stdout.splitlines() if _COMPOSE.search(ln)]


_PUBLISHED_IMAGES_OVERRIDE = "docker-compose.images.yml"
_ALLOWED_OVERRIDES = {"docker-compose.dev.yml", _PUBLISHED_IMAGES_OVERRIDE}
# The only image an override may set: this repo's published build of a service.
_PUBLISHED_IMAGE = re.compile(r"^ghcr\.io/automatosai/automatos-[a-z0-9-]+:\$\{AUTOMATOS_IMAGE_TAG:-edge\}$")


def _load_override(path: Path) -> dict:
    """Parse an override, accepting compose's ``!reset`` / ``!override`` merge tags."""
    import yaml

    class _Loader(yaml.SafeLoader):
        pass

    def _tag(loader, node):
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node)
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node)
        return loader.construct_scalar(node)

    for tag in ("!reset", "!override"):
        _Loader.add_constructor(tag, _tag)
    return yaml.load(path.read_text(), Loader=_Loader) or {}


def _image_allowed(name: str, svc: str, image: str, canonical_services: dict) -> bool:
    """Only the published-images override, only this repo's published build of a
    service the canonical file builds from source."""
    builds_from_source = "build" in (canonical_services.get(svc) or {})
    return name == _PUBLISHED_IMAGES_OVERRIDE and builds_from_source and bool(_PUBLISHED_IMAGE.match(image))


def test_exactly_one_tracked_stack_file():
    composes = _tracked_compose_files()
    stacks = [c for c in composes if c not in _ALLOWED_OVERRIDES]
    assert stacks == ["docker-compose.yml"], (
        f"expected exactly one canonical stack file (root docker-compose.yml), found: {stacks}. "
        f"A second stack is the drift PRD-209 S9 deleted; a documented override belongs in "
        f"{sorted(_ALLOWED_OVERRIDES)}."
    )


def test_the_dev_override_is_only_an_override():
    """Overrides may retarget/mount existing services — never define new ones, and
    never pin an image except the published-images override's own builds."""
    import yaml

    canonical = yaml.safe_load((_REPO_ROOT / "docker-compose.yml").read_text())
    canonical_services = canonical.get("services") or {}
    for name in sorted(_ALLOWED_OVERRIDES):
        path = _REPO_ROOT / name
        if not path.exists():
            continue
        override = _load_override(path)
        assert set(override) <= {"services", "volumes", "networks"}, (
            f"{name} may only override services/volumes/networks, found {sorted(override)}"
        )
        unknown = set(override.get("services") or {}) - set(canonical_services)
        assert not unknown, f"{name} defines services absent from the canonical stack: {sorted(unknown)}"
        for svc, body in (override.get("services") or {}).items():
            image = (body or {}).get("image")
            assert image is None or _image_allowed(name, svc, image, canonical_services), (
                f"{name}:{svc} pins image {image!r} — overrides must not choose what the stack runs "
                f"(only {_PUBLISHED_IMAGES_OVERRIDE} may swap a source-built service for its own "
                f"published build, ghcr.io/automatosai/automatos-<name>:${{AUTOMATOS_IMAGE_TAG:-edge}})"
            )


def test_the_image_exception_stays_narrow():
    """The published-images rule refuses everything but this repo's own build."""
    services = {"backend": {"build": {"context": "./orchestrator"}}, "postgres": {"image": "pgvector/pgvector:pg16"}}
    published = "ghcr.io/automatosai/automatos-api:${AUTOMATOS_IMAGE_TAG:-edge}"
    assert _image_allowed(_PUBLISHED_IMAGES_OVERRIDE, "backend", published, services)
    assert not _image_allowed("docker-compose.dev.yml", "backend", published, services)
    assert not _image_allowed(_PUBLISHED_IMAGES_OVERRIDE, "postgres", published, services)
    assert not _image_allowed(_PUBLISHED_IMAGES_OVERRIDE, "backend", "ghcr.io/someone/else:latest", services)
    assert not _image_allowed(_PUBLISHED_IMAGES_OVERRIDE, "backend", "ghcr.io/automatosai/automatos-api:latest", services)
