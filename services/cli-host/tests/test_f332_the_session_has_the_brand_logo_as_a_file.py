"""F332 (night 10): a session has the brand kit's logo as a file where it saves its work.

Night 10: "None of the tools I have here can read an uploaded image from the Brand kit
page" (BA); "no logo file" (Support). The claim now carries the workspace's uploaded
logo and logo mark (``brand_files``). The host writes them under ``brand/`` in the
ticket's deliverables folder (``<root>/sessions/<ticket>``), names them in the ticket
file, and the gate lets the session read them there without asking anyone. The retest
of #961 had them in the host's own ticket folder, and copying one into the deliverables
folder was held for approval (ticket 2099). A host with no default root keeps them in
the ticket folder. A name the host does not write, bad data or an oversize file is
refused, and nothing is ever written outside that folder.
"""
from __future__ import annotations

import base64

from automatos_cli_host import policy
from automatos_cli_host.adapters import claude as claude_adapter
from automatos_cli_host.brand_files import MAX_BRAND_FILE_BYTES, write_brand_files
from automatos_cli_host.presets import CLAUDE
from automatos_cli_host.session import Session

_CLAUDE = claude_adapter.ClaudeAdapter(CLAUDE)
LOGO = b"\x89PNG\r\n\x1a\n" + b"harbourline-logo"
MARK = b"\xff\xd8\xff" + b"harbourline-mark"


def _file(name, data):
    return {"name": name, "mime": "image/png", "data": base64.b64encode(data).decode("ascii")}


def _session(tmp_path, ticket, default_root="ws"):
    cfg = type("Cfg", (), {"ask_timeout": 1.0, "sessions_dir": tmp_path / "host" / "sessions",
                           "socket_path": tmp_path / "s.sock", "state_dir": tmp_path / "host"})()
    return Session({"task_id": 332, "attempt": 1, "session_id": "sid", "title": "Letter to Tidewater", **ticket},
                   cfg, [str(tmp_path / "ws")], tmp_path / "s.sock",
                   default_root=str(tmp_path / default_root) if default_root else None)


def _deliverables(tmp_path):
    return tmp_path / "ws" / "sessions" / "332"


def test_the_folder_the_session_saves_into_has_the_logo_and_the_ticket_file_names_it(tmp_path):
    s = _session(tmp_path, {"brand_files": [_file("logo.png", LOGO), _file("logo-mark.jpg", MARK)]})
    ticket_path, _ = s._write_session_files("Claude Code")
    brand = _deliverables(tmp_path) / "brand"
    assert (brand / "logo.png").read_bytes() == LOGO
    assert (brand / "logo-mark.jpg").read_bytes() == MARK
    assert not (s.session_dir / "brand").exists()      # not in the host's own state, where a copy out is held
    ticket = ticket_path.read_text(encoding="utf-8")
    assert "Brand files:" in ticket and f"save any file you produce under {_deliverables(tmp_path)}/" in ticket
    assert str(brand / "logo.png") in ticket and str(brand / "logo-mark.jpg") in ticket


def test_the_session_reads_its_logo_without_asking_anyone(tmp_path):
    s = _session(tmp_path, {"brand_files": [_file("logo.png", LOGO)], "cwd": ""})
    s._write_session_files("Claude Code")
    cwd = s._working_dir()                     # a ticket with no folder of its own runs in its deliverables folder
    assert cwd == _deliverables(tmp_path).resolve()
    s._set_policy(cwd, CLAUDE, None)
    logo = cwd / "brand" / "logo.png"
    for tool, tool_input in (("Read", {"file_path": str(logo)}), ("Bash", {"command": "ls brand"})):
        assert policy.decide(_CLAUDE.tool_intent(tool, tool_input), s._policy).behavior == "allow", tool


def test_a_ticket_with_its_own_folder_may_read_the_logo_where_it_saves(tmp_path):
    (tmp_path / "ws" / "repo").mkdir(parents=True)
    s = _session(tmp_path, {"brand_files": [_file("logo.png", LOGO)], "cwd": str(tmp_path / "ws" / "repo")})
    s._write_session_files("Claude Code")
    s._set_policy(s._working_dir(), CLAUDE, None)
    logo = _deliverables(tmp_path) / "brand" / "logo.png"
    assert policy.decide(_CLAUDE.tool_intent("Read", {"file_path": str(logo)}), s._policy).behavior == "allow"


def test_a_host_without_a_default_root_keeps_them_in_the_ticket_folder(tmp_path):
    s = _session(tmp_path, {"brand_files": [_file("logo.png", LOGO)]}, default_root=None)
    ticket_path, _ = s._write_session_files("Claude Code")
    assert (s.session_dir / "brand" / "logo.png").read_bytes() == LOGO
    assert str(s.session_dir / "brand" / "logo.png") in ticket_path.read_text(encoding="utf-8")


def test_a_claim_without_brand_files_says_nothing_of_them(tmp_path):
    s = _session(tmp_path, {})
    ticket_path, _ = s._write_session_files("Claude Code")
    assert "Brand files" not in ticket_path.read_text(encoding="utf-8")
    assert not (_deliverables(tmp_path) / "brand").exists()


def test_only_a_logo_name_is_written_and_never_outside_the_folder(tmp_path):
    folder = tmp_path / "s"
    bad = [_file("../../escape.png", LOGO), _file("logo.svg", LOGO), _file("ticket.md", LOGO),
           {"name": "logo.png", "data": "not base64!"}, _file("logo-mark.png", b"x" * (MAX_BRAND_FILE_BYTES + 1)),
           "logo.png", None]
    assert write_brand_files({"brand_files": bad}, folder) == []
    assert not (tmp_path / "escape.png").exists() and not (folder / "brand" / "ticket.md").exists()


def test_a_replaced_logo_leaves_no_stale_copy(tmp_path):
    folder = tmp_path / "s"
    write_brand_files({"brand_files": [_file("logo.jpg", MARK)]}, folder)
    written = write_brand_files({"brand_files": [_file("logo.png", LOGO)]}, folder)
    assert written == [folder / "brand" / "logo.png"]
    assert sorted(p.name for p in (folder / "brand").iterdir()) == ["logo.png"]


def test_the_logos_uploaded_variants_are_written_beside_it_and_named(tmp_path):
    """PRD-255 (0.13.0): the logo for dark backgrounds and the one-colour logo, as the rules block names them."""
    dark, mono = LOGO + b"-dark", MARK + b"-mono"
    s = _session(tmp_path, {"brand_files": [_file("logo-dark.png", dark), _file("logo-mono.jpg", mono)]})
    ticket_path, _ = s._write_session_files("Claude Code")
    brand = _deliverables(tmp_path) / "brand"
    assert (brand / "logo-dark.png").read_bytes() == dark and (brand / "logo-mono.jpg").read_bytes() == mono
    ticket = ticket_path.read_text(encoding="utf-8")
    assert str(brand / "logo-dark.png") in ticket and str(brand / "logo-mono.jpg") in ticket
    # Only those two variants: any other suffix is not a name this host writes.
    assert write_brand_files({"brand_files": [_file("logo-evil.png", LOGO)]}, tmp_path / "s") == []
