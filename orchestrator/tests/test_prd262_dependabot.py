"""PRD-262 US-012: every base image is pinned, and Dependabot keeps the pins current.

A pin with nothing to move it freezes security fixes, so the two travel together:

* every FROM line that names an image is ``name:tag@sha256:<digest>``. The tag
  stays, so Dependabot moves the digest along the tag's line; a digest-only pin
  would be moved to the ``latest`` tag's digest. ``scratch`` and earlier stages
  need no pin;
* a ``COPY --from`` or ``RUN --mount=…from=`` names a stage, never an image: an
  updater reads FROM lines only, so Node comes through a named ``FROM … AS node``
  stage (US-007);
* every third-party compose image is ``name:tag@sha256:<digest>``; only this
  repository's published images follow ``AUTOMATOS_IMAGE_TAG``;
* .github/dependabot.yml: docker covers exactly the directories that hold a
  tracked Dockerfile, docker-compose every compose file, github-actions the
  workflows; every entry is weekly and puts every dependency in a group; docker
  and compose ignore tag moves (semver major and minor), so the digest is what
  moves;
* every Python lock file has an updater, or is listed below with the reason
  Dependabot cannot update it. A new lock (US-044) fails here until one or the
  other is true; US-022's three API locks are listed.

Pure file reads: no Docker, no network.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path, PurePosixPath

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEPENDABOT = yaml.safe_load((ROOT / ".github" / "dependabot.yml").read_text())

# name:tag@sha256:<64 hex>; the name may carry a registry host and port.
PINNED = re.compile(r"^[^\s@]+:[\w][\w.-]{0,127}@sha256:[0-9a-f]{64}$")
FIRST_PARTY = re.compile(r"^ghcr\.io/automatosai/automatos-[a-z-]+:\$\{AUTOMATOS_IMAGE_TAG:-edge\}$")
# The same file names test_prd262_production_last_stage.py and test_prd234_s1a_compose_bind.py read.
DOCKERFILE_NAME = re.compile(
    r"(?:^|/)(?:Dockerfile(?:\.(?!md$|txt$|dockerignore$)[\w-]+)?|[\w.-]+\.[Dd]ockerfile)$"
)
COMPOSE_NAME = re.compile(r"(?:^|/)(?:docker-)?compose(?:[.-][^/]*)?\.ya?ml$")
PYTHON_LOCK = re.compile(r"(?:^|/)requirements[^/]*\.lock$")
TAG_MOVES = {"version-update:semver-major", "version-update:semver-minor"}
ALL_UPDATE_TYPES = {"major", "minor", "patch"}

# Lock files no Dependabot updater can regenerate, each with the reason. An entry
# leaves this map when its lock gains an updater or is deleted.
LOCKS_DEPENDABOT_CANNOT_UPDATE = {
    "services/agent-opt-worker/requirements.lock": (
        "Dependabot's pip and uv updaters read only *.txt and *.in files, and both "
        "recompile without uv's --overrides, which keeps the PyPI package fi out "
        "(dependabot-core python/shared_file_fetcher.rb and "
        "uv/file_updater/compile_file_updater.rb, read 18 Oct 2026). How this lock "
        "is kept current is open for Gerard and Dave (PRD-262 US-012)."
    ),
    **dict.fromkeys(
        ("orchestrator/requirements.lock", "orchestrator/requirements-leiden.lock",
         "orchestrator/requirements-dev.lock"),
        "Dependabot's pip and uv updaters fetch only *.txt and *.in files (and uv.lock), so "
        "they never see these locks (dependabot-core uv/lib/dependabot/uv/file_fetcher.rb, "
        "read 19 Oct 2026). Pointed at /orchestrator they would bump requirements.in alone, "
        "which test_prd262_api_locks.py then fails because the lock no longer satisfies it. "
        "A bump is an edit to the input plus the compile command at the top of each lock "
        "(PRD-262 US-022); which updater keeps them current is open for Gerard and Dave.",
    ),
}

_CONTINUATION = re.compile(r"\\[ \t]*\n")
_FROM_WORD = re.compile(r"^FROM\s", re.I)
# FROM [--platform=…] <image or stage> [AS <name>]
_FROM = re.compile(r"^FROM\s+(?:--\S+\s+)*(\S+)(?:\s+AS\s+(\S+))?\s*$", re.I)
_COPY_FROM = re.compile(r"(?:^|\s)--from=(\S+)", re.I)
_MOUNT_FROM = re.compile(r"--mount=\S*?[=,]from=([^,\s]+)", re.I)


def _tracked(pattern: re.Pattern) -> list[str]:
    proc = subprocess.run(["git", "ls-files"], cwd=str(ROOT), capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, f"git ls-files failed: {proc.stderr}"
    return [path for path in proc.stdout.splitlines() if pattern.search(path)]


def _instructions(text: str) -> list[str]:
    """Instructions with continuation lines joined; comment lines dropped."""
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    return [line.strip() for line in _CONTINUATION.sub(" ", code).splitlines() if line.strip()]


def dockerfile_pin_problems(text: str) -> list[str]:
    """Each FROM image that is not tag@digest, and each copy or mount from an image."""
    stages: set[str] = set()
    problems = []
    for line in _instructions(text):
        if _FROM_WORD.match(line):
            match = _FROM.match(line)
            if not match:
                raise ValueError(f"a FROM line this check cannot read: {line}")
            image, name = match.group(1), match.group(2)
            if image.lower() not in stages and image != "scratch" and not PINNED.match(image):
                problems.append(f"FROM {image} is not pinned as name:tag@sha256:<digest>")
            stages.add((name or "").lower())
            continue
        for source in _COPY_FROM.findall(line) + _MOUNT_FROM.findall(line):
            if source.lower() not in stages and not source.isdigit():
                problems.append(f"{line.split()[0]} reads {source}, not an earlier stage: give it a FROM … AS <name>")
    return problems


class _ComposeLoader(yaml.SafeLoader):
    """SafeLoader that reads compose's ``!reset`` / ``!override`` merge tags as plain values."""


def _untagged(loader: yaml.SafeLoader, node: yaml.Node) -> object:
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    return loader.construct_scalar(node)


for _tag in ("!reset", "!override"):
    _ComposeLoader.add_constructor(_tag, _untagged)


def _compose_images() -> dict[str, str]:
    """``file:service`` -> image, for every service of every tracked compose file that names one."""
    images = {}
    for name in _tracked(COMPOSE_NAME):
        services = (yaml.load((ROOT / name).read_text(), Loader=_ComposeLoader) or {}).get("services") or {}
        images.update({f"{name}:{svc}": body["image"] for svc, body in services.items() if (body or {}).get("image")})
    return images


def _entries(ecosystem: str) -> list[dict]:
    return [entry for entry in DEPENDABOT["updates"] if entry["package-ecosystem"] == ecosystem]


def _directories(entry: dict) -> set[str]:
    listed = entry.get("directories") or [entry["directory"]]
    return {"/" + str(directory).strip("/") for directory in listed}


def _directory_of(path: str) -> str:
    parent = PurePosixPath(path).parent.as_posix()
    return "/" if parent == "." else f"/{parent}"


# ── the pins ────────────────────────────────────────────────────────────────


def test_every_tracked_dockerfile_pins_its_images_and_copies_only_from_stages():
    dockerfiles = _tracked(DOCKERFILE_NAME)
    assert "orchestrator/Dockerfile" in dockerfiles and "frontend/Dockerfile" in dockerfiles
    problems = {path: dockerfile_pin_problems((ROOT / path).read_text()) for path in dockerfiles}
    assert {path: found for path, found in problems.items() if found} == {}


@pytest.mark.parametrize(("dockerfile", "expected"), [
    ("FROM python:3.11-slim@sha256:" + "e" * 64 + " AS base\nFROM base AS production\n", 0),
    ("FROM --platform=$BUILDPLATFORM node:24-alpine3.24@sha256:" + "4" * 64 + " AS node\n", 0),
    ("FROM scratch\nCOPY --from=0 /a /b\n", 0),
    ("FROM python:3.11-slim AS base\n", 1),
    ("FROM python@sha256:" + "e" * 64 + "\n", 1),
    ("FROM python:3.11-slim@sha256:" + "e" * 63 + "\n", 1),
    ("FROM alpine:3.24@sha256:" + "2" * 64 + "\nCOPY --from=node:24-alpine /usr/local/bin/node /usr/local/bin/\n", 1),
    ("FROM alpine:3.24@sha256:" + "2" * 64 + "\nRUN --mount=type=bind,from=ghcr.io/x/y:1,source=/a,target=/b true\n", 1),
    ("FROM node:24@sha256:" + "4" * 64 + " AS node\nFROM alpine:3.24@sha256:" + "2" * 64
     + "\nCOPY --from=node \\\n  /usr/local/bin/node /usr/local/bin/node\n", 0),
])
def test_the_dockerfile_rule_on_fixtures(dockerfile, expected):
    assert len(dockerfile_pin_problems(dockerfile)) == expected


def test_a_from_line_the_check_cannot_read_fails_rather_than_skips():
    with pytest.raises(ValueError):
        dockerfile_pin_problems("FROM python:3.11-slim AS\n")


def test_every_third_party_compose_image_is_pinned_as_tag_at_digest():
    images = _compose_images()
    third_party = {where: image for where, image in images.items() if not FIRST_PARTY.match(image)}
    assert len(third_party) >= 6, f"the compose search found too few images: {sorted(images)}"
    assert {where: image for where, image in third_party.items() if not PINNED.match(image)} == {}


# ── the updater ─────────────────────────────────────────────────────────────


def test_dependabot_config_is_version_2_with_the_three_ecosystems():
    assert DEPENDABOT["version"] == 2
    for ecosystem in ("github-actions", "docker", "docker-compose"):
        assert len(_entries(ecosystem)) == 1, ecosystem
    assert _directories(_entries("github-actions")[0]) == {"/"}


def test_docker_covers_exactly_the_directories_that_hold_a_dockerfile():
    expected = {_directory_of(path) for path in _tracked(DOCKERFILE_NAME)}
    assert {"/orchestrator", "/frontend", "/services/workspace-worker", "/services/media-render",
            "/services/agent-opt-worker"} <= expected
    assert _directories(_entries("docker")[0]) == expected, (
        "a Dockerfile with no docker entry keeps its pins frozen, and an entry with no "
        "Dockerfile fails Dependabot's run: update .github/dependabot.yml"
    )


def test_docker_compose_covers_every_compose_file():
    expected = {_directory_of(path) for path in _tracked(COMPOSE_NAME)}
    assert _directories(_entries("docker-compose")[0]) == expected


@pytest.mark.parametrize("entry", DEPENDABOT["updates"], ids=lambda entry: entry["package-ecosystem"])
def test_every_entry_is_weekly_and_groups_every_update(entry):
    assert entry["schedule"]["interval"] == "weekly"
    groups = (entry.get("groups") or {}).values()
    assert groups, "ungrouped: one pull request per dependency"
    covered = set()
    for group in groups:
        assert "*" in group["patterns"], group
        covered |= set(group.get("update-types") or ALL_UPDATE_TYPES)
    assert covered == ALL_UPDATE_TYPES


@pytest.mark.parametrize("ecosystem", ["docker", "docker-compose"])
def test_image_updates_stay_on_the_tags_line(ecosystem):
    (entry,) = _entries(ecosystem)
    ignored = [rule for rule in entry.get("ignore") or [] if rule.get("dependency-name") == "*"]
    assert any(TAG_MOVES <= set(rule.get("update-types") or []) for rule in ignored), (
        "a tag move (Python 3.12, Node 26, Postgres 17) is a change made by hand; "
        "Dependabot moves the digest along the tag"
    )


def test_every_python_lock_has_an_updater_or_a_stated_reason():
    python_dirs = set().union(*(_directories(entry) for entry in DEPENDABOT["updates"]
                                if entry["package-ecosystem"] in {"pip", "uv"}))
    locks = _tracked(PYTHON_LOCK)
    uncovered = [lock for lock in locks if _directory_of(lock) not in python_dirs]
    assert sorted(uncovered) == sorted(LOCKS_DEPENDABOT_CANNOT_UPDATE), (
        "every Python lock needs a Dependabot entry for its directory, or an entry in "
        "LOCKS_DEPENDABOT_CANNOT_UPDATE with the reason (an entry leaves the map when "
        "its lock gains an updater or is deleted)"
    )
    assert all(reason.strip() for reason in LOCKS_DEPENDABOT_CANNOT_UPDATE.values())
