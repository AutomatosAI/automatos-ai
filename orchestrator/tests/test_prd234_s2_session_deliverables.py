"""PRD-234 S2: a session's files under the workspace volume become the ticket's deliverables."""
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

from services.cli_host_service import RECENT_TOOLS_KEPT, workspace_relative_path  # noqa: E402

WS = "00000000-0000-0000-0000-0000000000c1"


def test_host_path_maps_onto_the_workers_layout_by_the_workspace_id_segment():
    host = f"/Users/me/Development/automatos-ai/workspaces/{WS}/sessions/68/hello.py"
    assert workspace_relative_path(host, WS) == "sessions/68/hello.py"
    assert workspace_relative_path(f"/somewhere/{WS}/README.md", WS) == "README.md"


def test_files_outside_the_workspace_stay_references():
    assert workspace_relative_path("/Users/me/repo/app.py", WS) is None
    assert workspace_relative_path(f"/w/{WS}/", WS) is None
    assert workspace_relative_path(f"/w/{WS}/../secret", WS) is None
    assert workspace_relative_path(f"/w/{WS}/a/../b", WS) is None


def test_live_log_is_bounded():
    assert RECENT_TOOLS_KEPT == 30


def test_project_files_map_onto_the_workers_projects_view():
    root = "/Users/me/Development"
    assert workspace_relative_path(f"{root}/repo/app.py", WS, root) == "projects/repo/app.py"
    assert workspace_relative_path(f"{root}/repo/app.py", WS, root + "/") == "projects/repo/app.py"
    assert workspace_relative_path(f"{root}", WS, root) is None
    assert workspace_relative_path(f"{root}-other/app.py", WS, root) is None  # prefix, not the folder
    assert workspace_relative_path(f"{root}/../secret", WS, root) is None
    assert workspace_relative_path("/Users/me/elsewhere/app.py", WS, None) is None
