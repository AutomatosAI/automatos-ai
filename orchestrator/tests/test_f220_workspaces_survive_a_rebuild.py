"""F220 — a local rebuild never empties a workspace.

Night 6: a report opened fine at 11:35Z and was empty after the owner's
`docker compose up --build` at 13:13Z; "the second time in four days". A
report's body is a file the workspace worker serves from
/workspaces/<workspace_id>/, and docker-compose.yml mounted only the local
workspace's folder there. Every other workspace (the test workspace, a second
one) lived in the worker container's own layer, so recreating the container
discarded its reports, sessions and artifacts while their rows stayed. Now a
named volume holds /workspaces, as Railway's volume does, with the local
workspace's folder still nested inside it.
"""
from __future__ import annotations

import json
import pathlib

import pytest
import yaml

_REPO = pathlib.Path(__file__).resolve().parents[2]
_VOLUME = "workspaces_data"
_LOCAL_ROOT = "/workspaces/${DEFAULT_WORKSPACE_ID:-00000000-0000-0000-0000-0000000000c1}"


class _ComposeLoader(yaml.SafeLoader):
    """SafeLoader that reads compose's !reset and !override tags as plain values."""


_ComposeLoader.add_constructor("!reset", lambda loader, node: None)
_ComposeLoader.add_constructor("!override", lambda loader, node: loader.construct_sequence(node))


def _compose(name: str) -> dict:
    return yaml.load((_REPO / name).read_text(encoding="utf-8"), Loader=_ComposeLoader)


def _fields(entry: str) -> list:
    """A short-syntax volume split on its colons, never on one inside ``${VAR:-default}``."""
    fields, current, depth = [], "", 0
    for char in entry:
        depth += {"{": 1, "}": -1}.get(char, 0)
        if char == ":" and depth == 0:
            fields, current = fields + [current], ""
        else:
            current += char
    return fields + [current]


def _targets(volumes: list) -> dict:
    """``{target: (source, mode)}`` for compose short-syntax volume strings."""
    out = {}
    for entry in volumes:
        source, target, *mode = _fields(entry)
        out[target] = (source, mode[0] if mode else "rw")
    return out


@pytest.fixture(scope="module")
def base() -> dict:
    return _compose("docker-compose.yml")


def test_the_worker_keeps_every_workspace_in_a_named_volume(base):
    mounts = _targets(base["services"]["workspace-worker"]["volumes"])

    assert mounts["/workspaces"] == (_VOLUME, "rw")
    assert base["volumes"][_VOLUME]["name"] == "automatos_workspaces_data"
    # The local workspace is still your folder, nested inside the volume: no layout change.
    assert mounts[_LOCAL_ROOT][0].startswith("${AUTOMATOS_WORKSPACE_DIR")


@pytest.mark.parametrize("compose_file", ["docker-compose.yml", "docker-compose.images.yml"])
def test_the_backend_sees_the_same_files_read_only(compose_file):
    """Session adoption, deliverable sizes and HARNESS read /workspaces/<id> directly."""
    mounts = _targets(_compose(compose_file)["services"]["backend"]["volumes"])

    assert mounts["/workspaces"] == (_VOLUME, "ro")
    assert mounts[_LOCAL_ROOT][1] == "ro"


def test_the_worker_starts_first_so_a_new_volume_is_its_users(base):
    """A fresh volume takes the ownership of the first image that mounts it. The
    worker's /workspaces is uid 1000; the backend image has none, and a root-owned
    volume would make the worker's entrypoint re-own the nested host folder."""
    assert "workspace-worker" in base["services"]["backend"]["depends_on"]


def test_railway_mounts_the_same_root():
    """Hosted already keeps every workspace on its volume; local now matches it."""
    manifest = json.loads((_REPO / "infrastructure" / "railway-manifest.json").read_text(encoding="utf-8"))

    assert "/workspaces" in set(_mounts(manifest))


def _mounts(node) -> list:
    if isinstance(node, dict):
        own = [node["mount"]] if isinstance(node.get("mount"), str) else []
        return own + [m for value in node.values() for m in _mounts(value)]
    if isinstance(node, list):
        return [m for value in node for m in _mounts(value)]
    return []
