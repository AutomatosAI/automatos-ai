"""The hook shim — what Claude Code runs on every lifecycle event.

Reads the hook payload from stdin, forwards it to the host over the loopback
Unix socket (``AUTOMATOS_HOST_SOCK``, set in the session's environment by the
host), prints the host's one-line JSON answer to stdout, exits 0.

Fail posture: for the two events where silence would hand control to the CLI's
own permission prompt (``PreToolUse``, ``PermissionRequest``) an unreachable
host is a DENY with a reason — never a prompt nobody watches. For every other
event silence is fine (exit 0, no output).

CLI adapter design §5: the shim is the same file for every CLI — it forwards
bytes and answers bytes; translation happens in the host, in the adapter. Its
ONE per-CLI fact is the shape of that offline deny, picked by ``AUTOMATOS_CLI``
(set in the session's environment by the host) from the table below. Nothing
else here knows which CLI is running.

Standard library only; it must start fast — the CLI waits for it.
"""
from __future__ import annotations

import json
import os
import socket
import struct
import sys

_GATED_EVENTS = ("PreToolUse", "PermissionRequest")
_CONNECT_TIMEOUT = 3.0
NOT_CONFIGURED = "Automatos CLI host socket is not configured"
UNREACHABLE_MARK = "Automatos CLI host is unreachable"
NOT_THE_HOST_MARK = "Automatos hook socket is not the host's"
UNREACHABLE = f"{UNREACHABLE_MARK} — call denied"
NOT_THE_HOST = f"the {NOT_THE_HOST_MARK} — call denied"
# F234: a CLI that runs its hooks inside its own sandbox (Copilot) is granted the
# socket path read-write so they can reach the host — which would also let a
# sandboxed command unlink it and bind its own. So the process at the other end
# must be the host (``AUTOMATOS_HOST_PID``, set in the session's environment).
_SOL_LOCAL, _LOCAL_PEERPID = 0, 0x002          # macOS <sys/un.h>


def _deny_claude_shaped(event: str, reason: str) -> dict:
    """The Claude Code wire shape — Codex reads the same one."""
    if event == "PermissionRequest":
        return {"hookSpecificOutput": {"hookEventName": "PermissionRequest",
                                       "decision": {"behavior": "deny", "message": reason}}}
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                   "permissionDecision": "deny",
                                   "permissionDecisionReason": reason}}


# The offline deny per CLI id. A CLI whose deny is ``{decision, reason}`` (gemini,
# grok, agy) adds its entry with its adapter; an unknown id gets the Claude shape —
# the shim must always answer SOMETHING that is a refusal.
_OFFLINE_DENY = {
    "claude": _deny_claude_shaped,
    "codex": _deny_claude_shaped,
    "copilot": _deny_claude_shaped,   # Copilot reads Claude-format hook output (PRD-253)
}


def _deny(event: str, reason: str) -> str:
    cli = (os.environ.get("AUTOMATOS_CLI") or "claude").strip().lower()
    render = _OFFLINE_DENY.get(cli, _deny_claude_shaped)
    return json.dumps(render(event, reason))


def _argv_event(argv) -> str:
    """The event a hook entry names on its own command line (``--event <Name>``).

    Design §5 ``event_name_source: argv``: some CLIs send a payload with no event
    name in it — Copilot's ``permissionRequest`` and ``notification`` hooks exist
    only in its camelCase format (PRD-253 S0.4) — so the entry the adapter writes
    says which event it is. A name in the payload always wins."""
    if "--event" in argv:
        i = argv.index("--event")
        return argv[i + 1] if i + 1 < len(argv) else ""
    return ""


def _payload(raw: str, argv) -> dict:
    """The hook payload from stdin — the event named on the command line when the
    payload carries none — tagged with this session's ticket."""
    try:
        payload = json.loads(raw) if raw.strip() else {}
    except ValueError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    if not payload.get("hook_event_name") and _argv_event(argv):
        payload["hook_event_name"] = _argv_event(argv)
    payload.setdefault("automatos_task_id", os.environ.get("AUTOMATOS_TASK_ID"))
    return payload


def _peer_pid(s: socket.socket) -> int:
    """The PID at the other end of ``s``; 0 when the platform cannot tell, or a PID
    namespace hides it (Linux's sandbox bind-mounts the socket, which a sandboxed
    command cannot unlink)."""
    try:
        if sys.platform == "darwin":
            return int.from_bytes(s.getsockopt(_SOL_LOCAL, _LOCAL_PEERPID, 4), sys.byteorder, signed=True)
        if hasattr(socket, "SO_PEERCRED"):
            return struct.unpack("3i", s.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")))[0]
    except OSError:
        return 0
    return 0


def _is_the_host(s: socket.socket) -> bool:
    """The socket's peer is the host that started this session, when it said so."""
    expected = (os.environ.get("AUTOMATOS_HOST_PID") or "").strip()
    peer = _peer_pid(s) if expected else 0
    return peer <= 0 or str(peer) == expected


def main(argv=None) -> int:
    payload = _payload(sys.stdin.read(), sys.argv[1:] if argv is None else argv)
    event = payload.get("hook_event_name") or ""
    sock_path = os.environ.get("AUTOMATOS_HOST_SOCK")
    if not sock_path:
        # The operator's own terminal (the Runtime Canvas, AUTOMATOS_TERMINAL) opens an
        # agent's session in the agent's home, where these hooks are installed: someone
        # is at the keyboard, so the CLI's own prompts apply. Anywhere else a missing
        # socket is a misconfigured session: deny.
        if event in _GATED_EVENTS and not os.environ.get("AUTOMATOS_TERMINAL"):
            sys.stdout.write(_deny(event, NOT_CONFIGURED))
        return 0

    # The host may hold PreToolUse while the approvals inbox answers; wait as
    # long as the hook's own timeout allows (the host answers before that).
    wait = float(os.environ.get("AUTOMATOS_HOOK_WAIT_SECONDS", "560"))
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(_CONNECT_TIMEOUT)
            s.connect(sock_path)
            if not _is_the_host(s):           # nothing is sent to an impostor
                if event in _GATED_EVENTS:
                    sys.stdout.write(_deny(event, NOT_THE_HOST))
                return 0
            s.settimeout(wait)
            s.sendall((json.dumps(payload) + "\n").encode("utf-8"))
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = s.recv(65536)
                if not chunk:
                    break
                buf += chunk
        answer = buf.decode("utf-8", "replace").strip()
        if answer and answer != "{}":
            sys.stdout.write(answer)
        return 0
    except (OSError, socket.timeout):
        if event in _GATED_EVENTS:
            sys.stdout.write(_deny(event, UNREACHABLE))
        return 0


if __name__ == "__main__":
    sys.exit(main())
