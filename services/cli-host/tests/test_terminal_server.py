"""PRD-239 S7 — the Canvas terminal: the RFC 6455 pieces, single-use grants,
directory resolution, one real shell through a real WebSocket on the loopback
(``/bin/sh``, no Claude involved), and — S7 v2, the Runtime Canvas — a launch
grant that starts or resumes the agent's Claude Code session (a fake ``claude``
that prints its argv)."""
from __future__ import annotations

import base64
import os
import re
import socket
import struct
import time
from pathlib import Path

import pytest

from automatos_cli_host import terminal_server as ts


# ── pure pieces ──────────────────────────────────────────────────────────────

def test_accept_key_matches_the_rfc_example():
    assert ts.accept_key("dGhlIHNhbXBsZSBub25jZQ==") == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="


def _client_frame(payload: bytes, opcode: int, mask: bytes = b"\x01\x02\x03\x04") -> bytes:
    """A masked client→server frame, as browsers send them."""
    header = bytearray([0x80 | opcode])
    n = len(payload)
    if n < 126:
        header.append(0x80 | n)
    elif n < 65536:
        header.append(0x80 | 126)
        header += struct.pack("!H", n)
    else:
        header.append(0x80 | 127)
        header += struct.pack("!Q", n)
    masked = bytes(c ^ mask[i % 4] for i, c in enumerate(payload))
    return bytes(header) + mask + masked


def test_frames_round_trip_including_split_and_large_payloads():
    reader = ts.FrameReader()
    small = _client_frame(b"ls\r", ts.OPCODE_BINARY)
    big = _client_frame(b"x" * 70000, ts.OPCODE_BINARY)
    text = _client_frame(b'{"type":"resize","cols":80,"rows":24}', ts.OPCODE_TEXT)
    assert reader.feed(small[:2]) == []          # incomplete header → nothing yet
    assert reader.feed(small[2:]) == [(ts.OPCODE_BINARY, b"ls\r")]
    frames = reader.feed(big + text)
    assert frames == [(ts.OPCODE_BINARY, b"x" * 70000), (ts.OPCODE_TEXT, b'{"type":"resize","cols":80,"rows":24}')]
    server_frame = ts.encode_frame(b"hello", ts.OPCODE_BINARY)
    assert server_frame == b"\x82\x05hello"       # FIN + binary, unmasked, 5 bytes
    assert ts.encode_frame(b"y" * 300)[:4] == b"\x82\x7e\x01\x2c"


def test_handshake_parsing_origin_and_token():
    method, target, headers = ts.parse_handshake(
        b"GET /terminal?token=abc HTTP/1.1\r\nHost: 127.0.0.1\r\nUpgrade: websocket\r\n"
        b"Sec-WebSocket-Key: k\r\nOrigin: http://localhost:3000\r\n\r\n"
    )
    assert method == "GET" and headers["upgrade"] == "websocket" and headers["origin"] == "http://localhost:3000"
    assert ts.token_of(target) == "abc"
    assert ts.token_of("/terminal") is None and ts.token_of("/other?token=abc") is None
    assert ts.origin_allowed("http://localhost:3000") and ts.origin_allowed("http://127.0.0.1:3000")
    assert ts.origin_allowed(None)
    assert not ts.origin_allowed("https://evil.example.com")


def test_grants_are_single_use_and_expire():
    now = [1000.0]
    store = ts.GrantStore(ttl_seconds=60, clock=lambda: now[0])
    assert store.admit([{"token": "a", "cwd": "/w/repo", "task_id": 95}, {"token": "b"}, {"nope": 1}]) == 2
    grant = store.take("a")
    assert grant and grant.cwd == "/w/repo" and grant.task_id == "95"
    assert store.take("a") is None                # single use
    now[0] += 61
    assert store.take("b") is None                # expired
    assert len(store) == 0


def test_directory_resolution_follows_the_allow_list(tmp_path):
    root = tmp_path / "ws"
    (root / "wsid" / "sessions").mkdir(parents=True)
    repo = tmp_path / "repo"
    repo.mkdir()
    server = ts.TerminalServer([str(root), str(repo)], str(root), workspace_id=lambda: "wsid")
    assert server.resolve_directory(ts.Grant("t", str(repo), None, 0)) == repo.resolve()
    assert server.resolve_directory(ts.Grant("t", None, "95", 0)) == (root / "wsid" / "sessions" / "95").resolve()
    assert server.resolve_directory(ts.Grant("t", None, None, 0)) == Path(str(root))
    with pytest.raises(ts.NotAllowed):
        server.resolve_directory(ts.Grant("t", str(tmp_path / "elsewhere"), None, 0))


# ── one real shell through a real WebSocket ─────────────────────────────────

def _handshake(port: int, token: str, origin: str = "http://localhost:3000") -> socket.socket:
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    key = base64.b64encode(os.urandom(16)).decode()
    s.sendall(
        f"GET /terminal?token={token} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nUpgrade: websocket\r\n"
        f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\nOrigin: {origin}\r\n\r\n".encode()
    )
    return s


def _read_response(s: socket.socket) -> bytes:
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = s.recv(4096)
        if not chunk:
            break
        data += chunk
    return data


def test_a_grant_opens_a_shell_in_the_directory_and_the_output_streams_back(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    server = ts.TerminalServer([str(root)], str(root), shell="/bin/sh")
    port = server.start()
    try:
        server.admit([{"token": "good", "cwd": str(root), "expires_at": time.time() + 60}])
        s = _handshake(port, "good")
        head = _read_response(s)
        assert head.startswith(b"HTTP/1.1 101"), head
        s.settimeout(5)
        s.sendall(_client_frame(b'{"type":"resize","cols":100,"rows":30}', ts.OPCODE_TEXT))
        s.sendall(_client_frame(b"pwd; echo AUTOMATOS_OK\r", ts.OPCODE_BINARY))
        reader = ts.FrameReader()
        seen = b""
        deadline = time.time() + 10
        # The PTY echoes the typed command ("$ pwd; echo AUTOMATOS_OK") before the
        # shell answers — the marker must be read on its OWN output line, or the
        # loop stops before `pwd` has printed (flaked in CI and locally, 2026-09-09).
        while not re.search(rb"(^|\n)AUTOMATOS_OK\r?\n", seen) and time.time() < deadline:
            try:
                data = s.recv(65536)
            except socket.timeout:
                continue
            if not data:
                break
            for opcode, payload in reader.feed(data):
                if opcode == ts.OPCODE_BINARY:
                    seen += payload
        assert b"AUTOMATOS_OK" in seen, seen
        assert str(root.resolve()).encode() in seen
        s.sendall(_client_frame(b"", ts.OPCODE_CLOSE))
        s.close()
        deadline = time.time() + 5
        while server.active and time.time() < deadline:
            time.sleep(0.05)
        assert server.active == 0
    finally:
        server.stop()


def test_an_unknown_or_reused_grant_and_a_foreign_origin_are_refused(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    server = ts.TerminalServer([str(root)], str(root), shell="/bin/sh")
    port = server.start()
    try:
        s = _handshake(port, "nope")
        assert _read_response(s).startswith(b"HTTP/1.1 403")
        s.close()
        server.admit([{"token": "once", "cwd": str(root)}])
        s = _handshake(port, "once", origin="https://evil.example.com")
        assert _read_response(s).startswith(b"HTTP/1.1 403")   # origin refused, grant untouched
        s.close()
        s = _handshake(port, "once")
        assert _read_response(s).startswith(b"HTTP/1.1 101")
        s.sendall(_client_frame(b"", ts.OPCODE_CLOSE))
        s.close()
        s = _handshake(port, "once")
        assert _read_response(s).startswith(b"HTTP/1.1 403")   # single use
        s.close()
    finally:
        server.stop()


# ── S7 v2: launch grants — the agent's own Claude Code session in the PTY ────

def test_terminal_args_start_or_resume_and_honour_the_subscription_invariant(tmp_path):
    prompt = tmp_path / "system_prompt.md"
    started = ts.build_terminal_args("/usr/local/bin/claude", session_id="abc", resume=False,
                                     system_prompt_path=prompt, model="opus", task_id="93")
    assert started[:3] == ["/usr/local/bin/claude", "--session-id", "abc"]
    assert started[3:5] == ["--append-system-prompt-file", str(prompt)]
    assert started[-4:] == ["--name", "automatos #93", "--model", "opus"]
    resumed = ts.build_terminal_args("claude", session_id="abc", resume=True, system_prompt_path=None, model=None, task_id=None)
    assert resumed == ["claude", "--resume", "abc"]
    # the operator's own `claude` in that folder: no unattended-lane narrowing,
    # nothing that assumes nobody is at the keyboard, nothing the subscription rules forbid
    for args in (started, resumed):
        assert "--setting-sources" not in args and "--strict-mcp-config" not in args
        assert "--permission-mode" not in args and "--settings" not in args and "--worktree" not in args
        assert "-p" not in args and "--print" not in args and "--bare" not in args
        ts.assert_args_honour_invariant(args)


def test_transcript_lookup_counts_only_the_sessions_own_folder(tmp_path):
    from automatos_cli_host.transcript import transcript_path

    home = tmp_path / "home"
    cwd = tmp_path / "repo"
    cwd.mkdir()
    sid = "11111111-1111-1111-1111-111111111111"
    assert ts.transcript_exists(cwd, sid, home) is False
    elsewhere = home / ".claude" / "projects" / "-some-other-folder"
    elsewhere.mkdir(parents=True)
    (elsewhere / f"{sid}.jsonl").write_text("{}\n")
    assert ts.transcript_exists(cwd, sid, home) is False   # --resume here would say "No conversation found"
    own = transcript_path(str(cwd), sid, home)
    own.parent.mkdir(parents=True, exist_ok=True)
    own.write_text("{}\n")
    assert ts.transcript_exists(cwd, sid, home) is True


def _fake_claude(tmp_path) -> str:
    script = tmp_path / "claude"
    script.write_text('#!/bin/sh\necho "FAKE_CLAUDE_ARGS: $*"\necho "FAKE_CLAUDE_CWD: $(pwd)"\n')
    script.chmod(0o755)
    return str(script)


def _run_launch(server, port, token) -> bytes:
    s = _handshake(port, token)
    head = _read_response(s)
    assert head.startswith(b"HTTP/1.1 101"), head
    s.settimeout(5)
    reader = ts.FrameReader()
    seen = b""
    deadline = time.time() + 10
    while b"FAKE_CLAUDE_CWD" not in seen and time.time() < deadline:
        try:
            data = s.recv(65536)
        except socket.timeout:
            continue
        if not data:
            break
        for opcode, payload in reader.feed(data):
            if opcode == ts.OPCODE_BINARY:
                seen += payload
    s.close()
    deadline = time.time() + 5
    while server.active and time.time() < deadline:
        time.sleep(0.05)
    return seen


def test_a_launch_grant_runs_the_agents_session_and_reports_open_and_close(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    home = tmp_path / "home"
    events = []
    server = ts.TerminalServer(
        [str(root)], str(root), shell="/bin/sh", claude=_fake_claude(tmp_path),
        sessions_dir=tmp_path / "sessions", on_event=lambda task_id, ev, payload: events.append((task_id, ev, payload)),
        claude_home=home,
    )
    port = server.start()
    sid = "22222222-2222-2222-2222-222222222222"
    launch = {"kind": "claude", "session_id": sid, "system_prompt": "You are Bob.", "model": "opus", "agent_name": "Bob"}
    try:
        server.admit([{"token": "first", "cwd": str(root), "task_id": 93, "launch": launch}])
        out = _run_launch(server, port, "first")
        assert b"--session-id " + sid.encode() in out and b"--resume" not in out, out
        assert b"--append-system-prompt-file" in out and b"--model opus" in out and b"automatos #93" in out
        assert (tmp_path / "sessions" / "93" / "system_prompt.md").read_text() == "You are Bob."
        assert str(root.resolve()).encode() in out
        # the transcript now exists in this folder → the next open resumes the same session
        from automatos_cli_host.transcript import transcript_path
        own = transcript_path(str(root.resolve()), sid, home)
        own.parent.mkdir(parents=True, exist_ok=True)
        own.write_text("{}\n")
        server.admit([{"token": "second", "cwd": str(root), "task_id": 93, "launch": launch}])
        out = _run_launch(server, port, "second")
        assert b"--resume " + sid.encode() in out and b"--session-id" not in out, out
    finally:
        server.stop()
    names = [(t, e, p["resumed"]) for t, e, p in events]
    assert names == [("93", "TerminalOpened", False), ("93", "TerminalClosed", False),
                     ("93", "TerminalOpened", True), ("93", "TerminalClosed", True)]
    assert all(p["session_id"] == sid and p["cwd"] == str(root.resolve()) for _, _, p in events)
    # 2026-09-09: the close carries the TURN's usage (the transcript's growth while
    # the terminal was open) and never the host-side snapshot
    closes = [p for _, e, p in events if e == "TerminalClosed"]
    assert all("usage" in p and "usage_before" not in p for p in closes)
    assert closes[1]["usage"]["total_tokens"] == 0 and closes[1]["usage"]["per_model"] == {}
    assert all("usage_before" not in p for _, e, p in events if e == "TerminalOpened")


def test_a_launch_the_host_cannot_honour_is_refused_before_any_shell_runs(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    server = ts.TerminalServer([str(root)], str(root), shell="/bin/sh", claude=str(tmp_path / "missing-claude"))
    port = server.start()
    try:
        server.admit([
            {"token": "kind", "cwd": str(root), "task_id": 1, "launch": {"kind": "codex", "session_id": "x"}},
            {"token": "sid", "cwd": str(root), "task_id": 1, "launch": {"kind": "claude", "session_id": "not-a-uuid"}},
            {"token": "bin", "cwd": str(root), "task_id": 1, "launch": {"kind": "claude", "session_id": "33333333-3333-3333-3333-333333333333"}},
        ])
        for token in ("kind", "sid", "bin"):
            s = _handshake(port, token)
            assert _read_response(s).startswith(b"HTTP/1.1 503"), token
            s.close()
        assert server.active == 0
    finally:
        server.stop()
