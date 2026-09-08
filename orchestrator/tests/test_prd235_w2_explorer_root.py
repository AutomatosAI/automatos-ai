"""PRD-235 W2: every session knows where the explorer (and Code mode) should open."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from services.cli_host_service import explorer_root_for  # noqa: E402

WS = "00000000-0000-0000-0000-0000000000c1"


def test_default_session_opens_in_its_sessions_folder():
    assert explorer_root_for(71, None, WS, None) == "sessions/71"
    assert explorer_root_for(71, "", WS, "/Users/me/Development") == "sessions/71"


def test_a_repo_under_the_projects_folder_opens_under_projects():
    assert explorer_root_for(72, "/Users/me/Development/repo", WS, "/Users/me/Development") == "projects/repo"


def test_a_folder_inside_the_workspace_volume_maps_by_the_workspace_id():
    assert explorer_root_for(73, f"/w/{WS}/sessions/73", WS, None) == "sessions/73"


def test_anywhere_else_is_not_browsable():
    assert explorer_root_for(74, "/opt/elsewhere/repo", WS, "/Users/me/Development") is None
