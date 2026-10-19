"""PRD-262 US-009: the frontend lockfile, refreshed once.

- sharp resolves to 0.35.5 or later; 0.34.5 carried three High findings.
- Every native build in the lock that targets one C library records it (`libc`).
  npm then installs only the build the machine can load: the musl sharp on the
  Alpine builder, so Next no longer traces the glibc libvips into the image, and the
  glibc builds on CI's Ubuntu runner. npm 10 (Node 20 and 22) drops the field when
  it rewrites the lock, so the lock is written with Node 24's npm (11).
- Prisma is gone (frontend/lib/db.ts, which nothing imported, frontend/prisma/ and
  the package.json `prisma` block), as are the dependencies nothing imported.
- The lock's root entry is package.json's dependency lists, and every package in the
  lock is reachable from them: the lock was regenerated, not hand-edited.

The built image is checked in CI: test.yml's prod-image-parity job lists the
frontend image's @img packages and reads its sharp version.

Pure file reads: no npm, no network.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"
PACKAGE = json.loads((FRONTEND / "package.json").read_text())
LOCK = json.loads((FRONTEND / "package-lock.json").read_text())["packages"]
PATCHED_SHARP = (0, 35, 5)
DEP_FIELDS = ("dependencies", "optionalDependencies", "peerDependencies")

# Removed by US-009: nothing imported, required or configured them.
REMOVED = (
    "prisma",
    "@prisma/client",
    "@anthropic-ai/sdk",
    "@floating-ui/react",
    "@headlessui/react",
    "@hookform/resolvers",
    "@tiptap/react",
    "@tiptap/starter-kit",
    "dotenv",
    "lodash",
    "react-grid-layout",
    "@types/react-grid-layout",
    "tailwind-scrollbar-hide",
)

# Native builds whose own published manifest declares no libc, so npm has none to record.
NO_LIBC_PUBLISHED = {
    "@unrs/resolver-binding-linux-arm-gnueabihf": "1.11.1 publishes os and cpu only",
    "@unrs/resolver-binding-linux-arm-musleabihf": "1.11.1 publishes os and cpu only",
}

# The sharp builds the images and CI runners load: Alpine (musl) and Ubuntu (glibc).
SHARP_BUILDS = {
    "@img/sharp-linuxmusl-x64": "musl",
    "@img/sharp-linuxmusl-arm64": "musl",
    "@img/sharp-libvips-linuxmusl-x64": "musl",
    "@img/sharp-libvips-linuxmusl-arm64": "musl",
    "@img/sharp-linux-x64": "glibc",
    "@img/sharp-linux-arm64": "glibc",
    "@img/sharp-libvips-linux-x64": "glibc",
    "@img/sharp-libvips-linux-arm64": "glibc",
}


def _name(key: str) -> str:
    """The package name of a lock key such as node_modules/a/node_modules/@s/b."""
    return key.rpartition("node_modules/")[2]


def _release(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("-")[0].split("."))


def _libc_of(name: str) -> str | None:
    """The C library a Linux build's name targets, or None when it names none."""
    if "musl" in name:
        return "musl"
    if "gnu" in name or re.search(r"^@img/sharp-(?:libvips-)?linux-", name):
        return "glibc"
    return None


def _declared() -> dict[str, str]:
    return {**PACKAGE.get("dependencies", {}), **PACKAGE.get("devDependencies", {})}


# ── sharp ───────────────────────────────────────────────────────────────────


def test_sharp_resolves_to_a_patched_release_and_its_builds_move_with_it():
    sharp = LOCK["node_modules/sharp"]["version"]
    assert _release(sharp) >= PATCHED_SHARP, f"sharp {sharp} is older than 0.35.5"
    builds = {
        _name(key): entry["version"]
        for key, entry in LOCK.items()
        if re.fullmatch(r"@img/sharp-(?!libvips-)[a-z0-9-]+", _name(key))
    }
    assert builds and set(builds.values()) == {sharp}, builds


@pytest.mark.parametrize("name", sorted(SHARP_BUILDS))
def test_the_sharp_builds_record_their_c_library(name):
    assert LOCK[f"node_modules/{name}"].get("libc") == [SHARP_BUILDS[name]]


def test_every_native_build_for_one_c_library_records_it():
    wrong = {
        key: entry.get("libc")
        for key, entry in LOCK.items()
        if "linux" in (entry.get("os") or [])
        and _libc_of(_name(key))
        and _name(key) not in NO_LIBC_PUBLISHED
        and entry.get("libc") != [_libc_of(_name(key))]
    }
    assert wrong == {}, (
        "lock entries without the right libc: write the lock with Node 24's npm (11); "
        "npm 10 drops the field"
    )


# ── removed dependencies ────────────────────────────────────────────────────


def test_prisma_is_gone():
    assert "prisma" not in PACKAGE, "the package.json prisma block seeds from a script that does not exist"
    assert not (FRONTEND / "prisma").exists()
    assert not (FRONTEND / "lib" / "db.ts").exists()
    assert [key for key in LOCK if re.fullmatch(r"prisma|@prisma/.+", _name(key))] == []


@pytest.mark.parametrize("name", REMOVED)
def test_an_unused_dependency_stays_removed(name):
    # If code now imports one of these, add it back and drop it from REMOVED.
    assert name not in _declared()
    assert name not in {**LOCK[""].get("dependencies", {}), **LOCK[""].get("devDependencies", {})}


# ── the lock agrees with package.json ───────────────────────────────────────


def test_the_lock_root_is_package_json():
    root = LOCK[""]
    assert root.get("dependencies", {}) == PACKAGE.get("dependencies", {})
    assert root.get("devDependencies", {}) == PACKAGE.get("devDependencies", {})


def _resolve(key: str, name: str) -> str | None:
    """Node's lookup: the nearest node_modules/<name> at or above the package at `key`."""
    base = key
    while True:
        candidate = f"{base}/node_modules/{name}" if base else f"node_modules/{name}"
        if candidate in LOCK:
            return candidate
        if not base:
            return None
        base = base.rpartition("/node_modules/")[0] if "/node_modules/" in base else ""


def _reachable() -> set[str]:
    seen: set[str] = set()
    todo = [""]
    while todo:
        key = todo.pop()
        if key in seen:
            continue
        seen.add(key)
        fields = (*DEP_FIELDS, "devDependencies") if key == "" else DEP_FIELDS
        names = [name for field in fields for name in LOCK[key].get(field) or {}]
        todo.extend(found for found in (_resolve(key, name) for name in names) if found)
    return seen


def test_every_package_in_the_lock_is_reachable_from_package_json():
    assert sorted(set(LOCK) - _reachable()) == []


# ── CI ──────────────────────────────────────────────────────────────────────


def test_ci_checks_the_sharp_build_inside_the_frontend_image():
    steps = yaml.safe_load((ROOT / ".github" / "workflows" / "test.yml").read_text())["jobs"][
        "prod-image-parity"
    ]["steps"]
    parity = "\n".join(step.get("run", "") for step in steps)
    for needle in (
        """jq -r '.packages["node_modules/sharp"].version' frontend/package-lock.json""",
        "ls node_modules/@img",
        "sharp-linux-*|sharp-libvips-linux-*)",
    ):
        assert needle in parity, needle
