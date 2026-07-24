"""PRD-170 S5 — commit-preview porcelain parsing (DB-free logic).

The commit-preview endpoint turns ``git status --porcelain`` into the changed
paths that feed the (tested) commit-message generator. This proves the pure
parser handles the porcelain shapes — added/modified/untracked/renamed — so the
generated message reflects the real change set. The endpoint's live git call is
CI/container; only the parsing is asserted here.
"""
from __future__ import annotations

from api.workspace_files import _parse_porcelain_paths


def test_parses_modified_and_added_paths():
    out = " M src/app.py\nA  src/new.py\n?? notes.txt\n"
    assert _parse_porcelain_paths(out) == ["src/app.py", "src/new.py", "notes.txt"]


def test_parses_rename_takes_new_path():
    out = "R  old_name.py -> new_name.py\n"
    assert _parse_porcelain_paths(out) == ["new_name.py"]


def test_strips_quotes_around_paths_with_spaces():
    out = ' M "a file.py"\n'
    assert _parse_porcelain_paths(out) == ["a file.py"]


def test_empty_status_yields_no_paths():
    assert _parse_porcelain_paths("") == []
    assert _parse_porcelain_paths("\n\n") == []
