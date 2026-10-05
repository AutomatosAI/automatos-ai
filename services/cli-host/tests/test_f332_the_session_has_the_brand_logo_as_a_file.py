"""F332 (night 10): a session has the brand kit's logo as a file in its ticket folder.

Night 10: "None of the tools I have here can read an uploaded image from the Brand kit
page" (BA); "no logo file" (Support). The claim now carries the workspace's uploaded
logo and logo mark (``brand_files``). The host writes them under ``brand/`` in the
ticket folder, names them in the ticket file, and the gate lets the session read them.
A name the host does not write, bad data or an oversize file is refused, and nothing is
ever written outside that folder.
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


def _session(tmp_path, ticket):
    cfg = type("Cfg", (), {"ask_timeout": 1.0, "sessions_dir": tmp_path / "sessions",
                           "socket_path": tmp_path / "s.sock"})()
    return Session({"task_id": 332, "attempt": 1, "session_id": "sid", "title": "Letter to Tidewater", **ticket},
                   cfg, [str(tmp_path)], tmp_path / "s.sock", default_root=str(tmp_path / "ws"))


def test_the_ticket_folder_has_the_logo_and_the_ticket_file_names_it(tmp_path):
    s = _session(tmp_path, {"brand_files": [_file("logo.png", LOGO), _file("logo-mark.jpg", MARK)]})
    ticket_path, _ = s._write_session_files("Claude Code")
    brand = tmp_path / "sessions" / "332" / "brand"
    assert (brand / "logo.png").read_bytes() == LOGO
    assert (brand / "logo-mark.jpg").read_bytes() == MARK
    ticket = ticket_path.read_text(encoding="utf-8")
    assert "Brand files:" in ticket
    assert str(brand / "logo.png") in ticket and str(brand / "logo-mark.jpg") in ticket


def test_the_session_may_read_its_logo(tmp_path):
    s = _session(tmp_path, {"brand_files": [_file("logo.png", LOGO)]})
    s._write_session_files("Claude Code")
    ctx = policy.PolicyContext(cwd=tmp_path / "ws", extra_dirs=(s.session_dir,), off_limits=(tmp_path / "sessions",))
    read = policy.decide(_CLAUDE.tool_intent("Read", {"file_path": str(s.session_dir / "brand" / "logo.png")}), ctx)
    assert read.behavior == "allow"


def test_a_claim_without_brand_files_says_nothing_of_them(tmp_path):
    s = _session(tmp_path, {})
    ticket_path, _ = s._write_session_files("Claude Code")
    assert "Brand files" not in ticket_path.read_text(encoding="utf-8")
    assert not (tmp_path / "sessions" / "332" / "brand").exists()


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
