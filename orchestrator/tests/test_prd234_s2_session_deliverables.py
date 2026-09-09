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


# ── the deliverables root as an anchor (2026-09-09: AUTOMATOS_WORKSPACE_DIR is the workspace root) ──

def test_session_files_under_the_deliverables_root_map_without_a_workspace_id_segment():
    root = "/Users/me/Development/deliverables"
    assert workspace_relative_path(f"{root}/sessions/68/hello.py", WS, None, root) == "sessions/68/hello.py"
    assert workspace_relative_path(f"{root}/reports/weekly.md", WS, None, root + "/") == "reports/weekly.md"
    assert workspace_relative_path(root, WS, None, root) is None  # the folder itself is not a file
    assert workspace_relative_path(f"{root}-old/x.md", WS, None, root) is None  # prefix, not the folder
    assert workspace_relative_path(f"{root}/../secret", WS, None, root) is None


def test_the_longer_root_wins_when_the_deliverables_root_sits_inside_the_projects_folder():
    projects = "/Users/me/Development"
    root = f"{projects}/deliverables"
    assert workspace_relative_path(f"{root}/reports/weekly.md", WS, projects, root) == "reports/weekly.md"
    assert workspace_relative_path(f"{projects}/repo/app.py", WS, projects, root) == "projects/repo/app.py"
    # …and the other way round (projects folder inside the deliverables root)
    assert workspace_relative_path(f"{root}/projects/repo/app.py", WS, f"{root}/projects", root) == "projects/repo/app.py"


def test_the_configured_root_is_the_default_anchor_and_must_be_absolute(monkeypatch):
    from services import cli_host_service as svc
    monkeypatch.setattr(svc.config, "AUTOMATOS_WORKSPACE_DIR", "/srv/deliverables", raising=False)
    assert workspace_relative_path("/srv/deliverables/artifacts/a.png", WS) == "artifacts/a.png"
    assert svc.browsable_root("/srv/deliverables", WS, None) == "."
    monkeypatch.setattr(svc.config, "AUTOMATOS_WORKSPACE_DIR", "./workspaces", raising=False)
    assert workspace_relative_path("./workspaces/artifacts/a.png", WS) is None
    assert svc.configured_workspace_dir() is None
