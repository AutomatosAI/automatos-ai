"""PRD-170 S6 — codegraph-over-workspace-clone scope + trigger (DB-free).

The pure glue that binds the PRD-165 codegraph tool family to the LOCAL
workspace clone (Q38): the deterministic project name, the clone path (fenced
INSIDE the workspace mount — tenancy), and the on-session-start + on-commit
reindex policy.

The DB-backed indexing + the live "what calls X?" / reindex-on-commit
integration are CI-with-DB — DEFERRED in prd-170.json with this contract written
(the kit forbids faking a DB-backed green). Pure stdlib + pytest.
"""
from __future__ import annotations

import pytest

from modules.codegraph.workspace_clone_index import (
    IndexTrigger,
    clone_index_path,
    should_reindex,
    workspace_clone_project_name,
)


# ---------------------------------------------------------------------------
# Project name — deterministic + workspace-scoped
# ---------------------------------------------------------------------------
def test_project_name_is_deterministic_and_workspace_scoped():
    a = workspace_clone_project_name("ws-1")
    b = workspace_clone_project_name("ws-1")
    assert a == b
    assert "ws-1" in a
    # different workspace → different project
    assert workspace_clone_project_name("ws-2") != a


def test_project_name_distinguishes_repos_within_a_workspace():
    root = workspace_clone_project_name("ws-1")
    app = workspace_clone_project_name("ws-1", "repos/app")
    lib = workspace_clone_project_name("ws-1", "repos/lib")
    assert len({root, app, lib}) == 3


# ---------------------------------------------------------------------------
# Clone path — fenced inside the workspace mount (tenancy)
# ---------------------------------------------------------------------------
def test_clone_path_root_is_the_mount():
    assert clone_index_path("/workspaces/ws-1") == "/workspaces/ws-1"


def test_clone_path_subpath_stays_under_mount():
    p = clone_index_path("/workspaces/ws-1", "repos/app")
    assert p == "/workspaces/ws-1/repos/app"


@pytest.mark.parametrize(
    "bad",
    ["../other", "repos/../../etc", "/etc/passwd", "a/../../b", "x\x00y"],
)
def test_clone_path_rejects_escape_attempts(bad):
    with pytest.raises(ValueError):
        clone_index_path("/workspaces/ws-1", bad)


def test_clone_path_normalises_noise_segments():
    # ./ and empty segments collapse but stay in-mount.
    assert clone_index_path("/workspaces/ws-1", "./repos//app/") == "/workspaces/ws-1/repos/app"


# ---------------------------------------------------------------------------
# Reindex trigger policy — on session start (cold only) + on commit (if changed)
# ---------------------------------------------------------------------------
def test_session_start_indexes_only_when_cold():
    cold = should_reindex(IndexTrigger.SESSION_START, already_indexed=False)
    assert cold.reindex is True
    warm = should_reindex(IndexTrigger.SESSION_START, already_indexed=True)
    assert warm.reindex is False


def test_commit_reindexes_when_files_changed():
    changed = should_reindex(IndexTrigger.COMMIT, already_indexed=True, changed_files=3)
    assert changed.reindex is True
    noop = should_reindex(IndexTrigger.COMMIT, already_indexed=True, changed_files=0)
    assert noop.reindex is False


def test_manual_always_reindexes():
    assert should_reindex(IndexTrigger.MANUAL, already_indexed=True).reindex is True


def test_every_decision_has_a_reason():
    for trig in IndexTrigger:
        d = should_reindex(trig, already_indexed=False, changed_files=1)
        assert isinstance(d.reason, str) and d.reason
