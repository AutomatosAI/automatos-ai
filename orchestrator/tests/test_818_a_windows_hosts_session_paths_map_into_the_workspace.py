"""#818: a session host on native Windows reports C:\\… paths; they map into the workspace.

The stack (Docker Desktop) is configured with ``C:/Users/…`` folders, the host
reports ``C:\\Users\\…``, and NTFS ignores case. Before this, none of a Windows
session's files matched, so none became Deliverables. Unix roots still match
exactly.
"""
from __future__ import annotations

import services.cli_host_service as svc
from services.cli_host_service import configured_workspace_dir, workspace_relative_path
from services.host_paths import is_absolute_host_path, relative_to_root
from services.result_files import worker_paths

WS = "11111111-2222-3333-4444-555555555555"
ROOT = "C:/Users/ada/automatos-deliverables"
PROJECTS = "D:/code"


def test_a_windows_session_file_under_the_deliverables_root_maps():
    host = r"C:\Users\ada\automatos-deliverables\sessions\68\report.md"
    assert workspace_relative_path(host, WS, None, ROOT) == "sessions/68/report.md"


def test_the_drive_and_folders_match_without_regard_to_case_and_the_name_keeps_its_own():
    host = r"c:\USERS\Ada\Automatos-Deliverables\Reports\Weekly.md"
    assert workspace_relative_path(host, WS, None, ROOT) == "Reports/Weekly.md"


def test_the_projects_folder_on_another_drive_maps():
    assert workspace_relative_path(r"D:\code\repo\app.py", WS, PROJECTS, ROOT) == "projects/repo/app.py"


def test_the_workspace_id_anchor_works_with_backslashes():
    assert workspace_relative_path(rf"E:\vol\{WS}\sessions\1\a.py", WS) == "sessions/1/a.py"


def test_a_neighbouring_folder_or_a_way_out_is_not_inside():
    assert workspace_relative_path(r"C:\Users\ada\automatos-deliverables-old\x.md", WS, None, ROOT) is None
    assert workspace_relative_path(r"C:\Users\ada\automatos-deliverables\..\secret.txt", WS, None, ROOT) is None
    assert workspace_relative_path(ROOT.replace("/", "\\"), WS, None, ROOT) is None   # the folder is not a file


def test_unix_roots_still_match_exactly():
    assert workspace_relative_path("/users/me/deliverables/x.md", WS, None, "/Users/me/deliverables") is None
    assert workspace_relative_path("/Users/me/deliverables/x.md", WS, None, "/Users/me/deliverables") == "x.md"


def test_a_windows_deliverables_folder_is_a_configured_root(monkeypatch):
    for configured in ("C:/Users/ada/automatos-deliverables/", r"C:\Users\ada\automatos-deliverables"):
        monkeypatch.setattr(svc.config, "AUTOMATOS_WORKSPACE_DIR", configured, raising=False)
        assert configured_workspace_dir() == ROOT
    monkeypatch.setattr(svc.config, "AUTOMATOS_WORKSPACE_DIR", "./workspaces", raising=False)
    assert configured_workspace_dir() is None


def test_what_counts_as_absolute_and_inside():
    assert is_absolute_host_path(r"C:\x") and is_absolute_host_path("c:/x") and is_absolute_host_path("/x")
    assert not is_absolute_host_path("x/y") and not is_absolute_host_path("./x") and not is_absolute_host_path("C:x")
    assert relative_to_root("C:/a/b", "C:/A") == "/b" and relative_to_root("C:/ab", "C:/a") is None


def test_a_windows_sessions_named_files_are_found(monkeypatch):
    monkeypatch.setattr(svc.config, "AUTOMATOS_WORKSPACE_DIR", ROOT, raising=False)
    ref = {"host_id": "h1", "cwd": r"C:\Users\ada\automatos-deliverables\sessions\68"}
    named = [r"C:\Users\ada\automatos-deliverables\sessions\68\out.md", r"charts\q3.png"]
    assert worker_paths(named, workspace_id=WS, runtime_ref=ref, projects_dir=None) == [
        (named[0], "sessions/68/out.md"),
        (named[1], "sessions/68/charts/q3.png"),
    ]
