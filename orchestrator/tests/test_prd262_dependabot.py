"""PRD-262 US-012: every base image is pinned, and Dependabot keeps the pins current.

A pin with nothing to move it freezes security fixes, so the two travel together:

* every FROM line that names an image is ``name:tag@sha256:<digest>``. The tag
  stays, so Dependabot moves the digest along the tag's line; a digest-only pin
  would be moved to the ``latest`` tag's digest. ``scratch`` and earlier stages
  need no pin. A FROM line is read with the defaults of the ARGs declared before
  the first FROM, as BuildKit reads it: ``FROM assets-${MEDIA_ASSETS}`` names the
  stage the default picks (media-render, PRD-262 US-053);
* a ``COPY --from`` or ``RUN --mount=…from=`` names a stage, never an image: an
  updater reads FROM lines only, so Node comes through a named ``FROM … AS node``
  stage (US-007);
* every third-party compose image is ``name:tag@sha256:<digest>``; only this
  repository's own images are not: the published ones follow ``AUTOMATOS_IMAGE_TAG``,
  and make dev's local build names (``automatos-<name>:dev``) are never pulled. The
  docker-compose updater ignores both (PRD-262 US-036);
* .github/dependabot.yml: docker covers exactly the directories that hold a
  tracked Dockerfile, docker-compose every compose file, github-actions the
  workflows; every entry is weekly and puts every dependency in a group; docker
  and compose ignore tag moves (semver major and minor), so the digest is what
  moves;
* every Python lock file has an updater, or is listed below with the reason
  Dependabot cannot update it. A new lock (US-044) fails here until one or the
  other is true; US-022's API locks (two since US-027 removed Railway's [leiden]
  lock) and the frozen reference locks of US-023, US-026 and US-027 are listed.

Pure file reads: no Docker, no network.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEPENDABOT = yaml.safe_load((ROOT / ".github" / "dependabot.yml").read_text())

# name:tag@sha256:<64 hex>; the name may carry a registry host and port.
PINNED = re.compile(r"^[^\s@]+:[\w][\w.-]{0,127}@sha256:[0-9a-f]{64}$")
# This repository's own images: the published ones (docker-compose.yml) and make dev's local
# build names (docker-compose.dev.yml), which are never pulled (PRD-262 US-036).
FIRST_PARTY = re.compile(r"^(?:ghcr\.io/automatosai/automatos-[a-z-]+:\$\{AUTOMATOS_IMAGE_TAG:-edge\}"
                         r"|automatos-[a-z-]+:dev)$")
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
    # PRD-262 US-044. A pip entry for the directory would read requirements.in and
    # requirements-browser.in only, and bump a pin there while leaving the lock
    # behind (test_prd262_worker_locks.py then fails the lock against its input).
    "services/workspace-worker/requirements.lock": (
        "Dependabot's pip and uv updaters read only *.txt and *.in files "
        "(dependabot-core python/shared_file_fetcher.rb, read 18 Oct 2026), so they "
        "cannot regenerate a requirements.lock. How this lock is kept current is open "
        "for Gerard and Dave (PRD-262 US-012, US-044)."
    ),
    "services/workspace-worker/requirements-browser.lock": (
        "Dependabot's pip and uv updaters read only *.txt and *.in files "
        "(dependabot-core python/shared_file_fetcher.rb, read 18 Oct 2026), and this "
        "lock is compiled with -c requirements.lock, which neither updater passes. "
        "How it is kept current is open for Gerard and Dave (PRD-262 US-012, US-044)."
    ),
    # PRD-262 (row 27 of the 19 Oct size breakdown): psycopg2's build backend, hash-checked
    # because the worker builds psycopg2 without build isolation.
    "services/workspace-worker/requirements-build.lock": (
        "Dependabot's pip and uv updaters read only *.txt and *.in files "
        "(dependabot-core python/shared_file_fetcher.rb, read 18 Oct 2026), so they "
        "cannot regenerate a requirements.lock. How this lock is kept current is open "
        "for Gerard and Dave (PRD-262 US-012, US-044)."
    ),
    **dict.fromkeys(
        ("orchestrator/requirements.lock", "orchestrator/requirements-dev.lock"),
        "Dependabot's pip and uv updaters fetch only *.txt and *.in files (and uv.lock), so "
        "they never see these locks (dependabot-core uv/lib/dependabot/uv/file_fetcher.rb, "
        "read 19 Oct 2026). Pointed at /orchestrator they would bump requirements.in alone, "
        "which test_prd262_api_locks.py then fails because the lock no longer satisfies it. "
        "A bump is an edit to the input plus the compile command at the top of each lock "
        "(PRD-262 US-022); which updater keeps them current is open for Gerard and Dave.",
    ),
    "scripts/ci/reference/requirements-pdf-tables.lock": (
        "Frozen on purpose: the PDF table extractor the API ran before PRD-262 US-023, at "
        "the versions its W2 lock installed, which test.yml's reference-goldens job runs "
        "in a throwaway venv to re-record the grids pdfplumber is judged against. A bump "
        "would change the reference, not fix anything that ships; no image installs it."
    ),
    "scripts/ci/reference/requirements-text-splitters.lock": (
        "Frozen on purpose: the text splitters the API ran before PRD-262 US-026 "
        "(langchain-text-splitters and langchain-core), at the versions its W2 lock "
        "installed, which test.yml's reference-goldens job runs in a throwaway venv to "
        "re-record the chunks the vendored splitters are judged against. A bump would "
        "change the reference, not fix anything that ships; no image installs it."
    ),
    "scripts/ci/reference/requirements-leiden-partitions.lock": (
        "Frozen on purpose: the Leiden clustering Railway's API ran before PRD-262 US-027 "
        "(graphifyy with graspologic 3.3.0 over graspologic-native), at the versions its "
        "[leiden] lock installed, which test.yml's reference-goldens job runs in a throwaway "
        "venv to re-record the partitions the graspologic-native adapter is judged against. "
        "A bump would change the reference, not fix anything that ships; no image installs it."
    ),
}

_CONTINUATION = re.compile(r"\\[ \t]*\n")
_FROM_WORD = re.compile(r"^FROM\s", re.I)
# FROM [--platform=…] <image or stage> [AS <name>]
_FROM = re.compile(r"^FROM\s+(?:--\S+\s+)*(\S+)(?:\s+AS\s+(\S+))?\s*$", re.I)
_COPY_FROM = re.compile(r"(?:^|\s)--from=(\S+)", re.I)
_MOUNT_FROM = re.compile(r"--mount=\S*?[=,]from=([^,\s]+)", re.I)
_GLOBAL_ARG = re.compile(r"^ARG\s+([A-Za-z_]\w*)(?:=(\S*))?\s*$", re.I)
_VARIABLE = re.compile(r"\$(?:\{([A-Za-z_]\w*)\}|([A-Za-z_]\w*))")


def _tracked(pattern: re.Pattern) -> list[str]:
    proc = subprocess.run(["git", "ls-files"], cwd=str(ROOT), capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, f"git ls-files failed: {proc.stderr}"
    return [path for path in proc.stdout.splitlines() if pattern.search(path)]


def _instructions(text: str) -> list[str]:
    """Instructions with continuation lines joined; comment lines dropped."""
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    return [line.strip() for line in _CONTINUATION.sub(" ", code).splitlines() if line.strip()]


def _with_defaults(word: str, defaults: dict[str, str]) -> str:
    """``word`` with each ``$NAME`` or ``${NAME}`` a global ARG gives a default replaced by it."""
    return _VARIABLE.sub(lambda m: defaults.get(m.group(1) or m.group(2), m.group(0)), word)


def dockerfile_pin_problems(text: str) -> list[str]:
    """Each FROM image that is not tag@digest, and each copy or mount from an image."""
    stages: set[str] = set()
    defaults: dict[str, str] = {}
    problems = []
    for line in _instructions(text):
        global_arg = _GLOBAL_ARG.match(line) if not stages else None
        if global_arg:
            defaults[global_arg.group(1)] = global_arg.group(2) or ""
            continue
        if _FROM_WORD.match(line):
            match = _FROM.match(line)
            if not match:
                raise ValueError(f"a FROM line this check cannot read: {line}")
            image, name = _with_defaults(match.group(1), defaults), match.group(2)
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
    # A global ARG's default picks an earlier stage, or a pinned image; otherwise it is unpinned.
    ("ARG SOURCE=fetch\nFROM python:3.12-slim@sha256:" + "e" * 64 + " AS assets-fetch\nFROM assets-${SOURCE} AS assets\n", 0),
    ("ARG SOURCE=image\nFROM python:3.12-slim@sha256:" + "e" * 64 + " AS assets-fetch\nFROM assets-${SOURCE} AS assets\n", 1),
    ("ARG BASE=python:3.12-slim@sha256:" + "e" * 64 + "\nFROM $BASE AS base\n", 0),
    ("ARG BASE=python:3.12-slim\nFROM ${BASE}\n", 1),
    ("FROM ${UNDECLARED}\n", 1),
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



def test_this_repositorys_own_compose_images_are_its_published_and_dev_names():
    """docker-compose.yml names the three published images, docker-compose.dev.yml their dev builds."""
    own = {where: image for where, image in _compose_images().items() if FIRST_PARTY.match(image)}
    # The registry is the reference's host, the part before its first slash.
    published = {where for where, image in own.items() if urlparse(f"//{image}").netloc == "ghcr.io"}
    dev = {where for where, image in own.items() if image.endswith(":dev")}
    services = ("backend", "frontend", "workspace-worker")
    assert published == {f"docker-compose.yml:{service}" for service in services}
    assert dev == {f"docker-compose.dev.yml:{service}" for service in services}


def test_the_compose_updater_ignores_this_repositorys_own_images():
    """Neither name is a registry pin: the tag moves with AUTOMATOS_IMAGE_TAG, and a dev build is local."""
    (entry,) = _entries("docker-compose")
    whole = {rule["dependency-name"] for rule in entry.get("ignore") or [] if not rule.get("update-types")}
    assert {"automatos-*", "ghcr.io/automatosai/automatos-*"} <= whole


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
