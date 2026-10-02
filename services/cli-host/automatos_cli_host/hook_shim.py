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
import sys

_GATED_EVENTS = ("PreToolUse", "PermissionRequest")
_CONNECT_TIMEOUT = 3.0


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


def main(argv=None) -> int:
    payload = _payload(sys.stdin.read(), sys.argv[1:] if argv is None else argv)
    event = payload.get("hook_event_name") or ""
    sock_path = os.environ.get("AUTOMATOS_HOST_SOCK")
    if not sock_path:
        if event in _GATED_EVENTS:
            sys.stdout.write(_deny(event, "Automatos CLI host socket is not configured"))
        return 0

    # The host may hold PreToolUse while the approvals inbox answers; wait as
    # long as the hook's own timeout allows (the host answers before that).
    wait = float(os.environ.get("AUTOMATOS_HOOK_WAIT_SECONDS", "560"))
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(_CONNECT_TIMEOUT)
            s.connect(sock_path)
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
            sys.stdout.write(_deny(event, "Automatos CLI host is unreachable — call denied"))
        return 0


if __name__ == "__main__":
    sys.exit(main())
