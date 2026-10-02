#!/usr/bin/env python3
"""A stand-in ``copilot`` for CI — GitHub Copilot CLI 1.0.91 as the host drives it,
``copilot -p``, in its own spelling (PRD-253 S1.7):

* refuses what the COPILOT preset forbids (every allow flag, ``--config-dir``,
  ``-i``, …) and a launch without ``-p``, with exit 64 — the invariant is executable;
* needs ``COPILOT_HOME`` (66) and the account pointer in its ``config.json`` (67);
* loads the Claude-format hook files in ``$COPILOT_HOME/hooks/`` and fires them the
  way Copilot fires Claude-format hooks: snake_case payloads with Claude tool names
  and Copilot's own input keys, the decision read from ``hookSpecificOutput``;
* a hook that times out FAILS OPEN, like the real CLI — but with no allow flag the
  call then meets Copilot's own refusal in ``-p``, so nothing runs ungated;
* after the gate allows a fetch, raises its OWN ``PermissionRequest`` for it (a URL
  check of its own) and runs it only when that is allowed too;
* ``--session-id <uuid>`` starts a session, ``--resume <id>`` continues one (an
  unknown id exits 1), and the record is ``session-state/<id>/events.jsonl`` in the
  1.0.91 envelope: ``session.start``, ``assistant.usage``, ``assistant.message``,
  ``session.shutdown``;
* exits 0 after ``Stop`` and ``SessionEnd`` — in ``-p`` the turn IS the process.

Scenario knobs: ``FAKE_COPILOT_SCENARIO`` = ``happy`` (default), ``no-hooks`` (the
hooks file ignored), ``ai-credits`` (the plan runs out mid-turn).
"""
from __future__ import annotations

import glob
import json
import os
import shlex
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

FORBIDDEN = {"--allow-all-tools", "--allow-all", "--yolo", "--allow-all-paths", "--allow-all-urls", "--allow-tool",
             "--assisted-approval", "--enable-memory", "--config-dir", "--share-gist", "--remote", "--acp",
             "--headless", "-i", "--interactive", "--continue"}
FETCH_URL = "https://example.com/spec"


def _flag(args, name):
    return args[args.index(name) + 1] if name in args and args.index(name) + 1 < len(args) else None


def _hooks(home: Path) -> dict:
    merged: dict = {}
    for path in sorted(glob.glob(str(home / "hooks" / "*.json"))):
        for event, entries in (json.loads(Path(path).read_text()).get("hooks") or {}).items():
            merged.setdefault(event, []).extend(entries or [])
    return merged


def _run_hook(entry: dict, payload: dict) -> dict:
    try:
        proc = subprocess.run(shlex.split(entry["command"]), input=json.dumps(payload), capture_output=True,
                              text=True, timeout=float(entry.get("timeout", 60)), env=os.environ)
    except subprocess.TimeoutExpired:
        return {}                                     # fails OPEN: no decision
    try:
        return json.loads((proc.stdout or "").strip() or "{}")
    except ValueError:
        return {}


def _fire(hooks: dict, event: str, payload: dict) -> dict:
    answer: dict = {}
    for group in hooks.get(event) or []:
        for entry in group.get("hooks") or []:
            out = _run_hook(entry, {"hook_event_name": event, **payload})
            answer = answer or out
    return answer


def _decision(out: dict, key: str = "permissionDecision") -> str:
    specific = out.get("hookSpecificOutput") or {}
    if key == "permissionDecision":
        return str(specific.get(key) or "")
    return str((specific.get("decision") or {}).get("behavior") or "")


class Record:
    def __init__(self, home: Path, session_id: str):
        self.path = home / "session-state" / session_id / "events.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.parent = None

    def premium_so_far(self) -> int:
        """The session's premium requests before this run — Copilot counts them per session."""
        total = 0
        for line in self.path.read_text().splitlines() if self.path.exists() else []:
            event = json.loads(line)
            if event.get("type") == "session.shutdown":
                total = int(event["data"].get("totalPremiumRequests") or 0)
        return total

    def add(self, kind: str, data: dict) -> None:
        event_id = str(uuid.uuid4())
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"id": event_id, "timestamp": datetime.now(timezone.utc).isoformat(),
                                     "parentId": self.parent, "type": kind, "data": data}) + "\n")
        self.parent = event_id


def _refuse(args) -> int:
    if FORBIDDEN & set(args):
        sys.stderr.write(f"fake copilot: forbidden argument {sorted(FORBIDDEN & set(args))}\n")
        return 64
    if "-p" not in args:
        sys.stderr.write("fake copilot: a supervised session runs with -p\n")
        return 64
    return 0


def _turn(hooks: dict, common: dict, cwd: str) -> tuple:
    """One gated turn: a write, a push (must be denied), a fetch Copilot re-asks for."""
    target = os.path.join(cwd, "hello.txt")
    wrote = _decision(_fire(hooks, "PreToolUse", {**common, "tool_name": "Write",
                                                  "tool_input": {"path": target, "file_text": "hi"}})) == "allow"
    if wrote:
        Path(target).write_text("hi\n")
        _fire(hooks, "PostToolUse", {**common, "tool_name": "Write", "tool_input": {"path": target, "file_text": "hi"},
                                     "tool_response": {"result_type": "success"}})
    pushed = _decision(_fire(hooks, "PreToolUse", {**common, "tool_name": "Bash",
                                                   "tool_input": {"command": "git push origin main"}})) == "allow"
    fetch = {"url": FETCH_URL}
    fetched = False
    if _decision(_fire(hooks, "PreToolUse", {**common, "tool_name": "WebFetch", "tool_input": fetch})) == "allow":
        asked = _fire(hooks, "PermissionRequest", {**common, "tool_name": "WebFetch", "tool_input": fetch})
        fetched = _decision(asked, "decision") == "allow"
    return wrote, pushed, fetched


def main(argv) -> int:
    args = list(argv)
    if "--version" in args:
        print("GitHub Copilot CLI 9.9.9")
        return 0
    refused = _refuse(args)
    if refused:
        return refused
    home = Path(os.environ.get("COPILOT_HOME") or "")
    if not os.environ.get("COPILOT_HOME"):
        sys.stderr.write("fake copilot: COPILOT_HOME is not set (the adapter must prepare a home)\n")
        return 66
    if "lastLoggedInUser" not in json.loads((home / "config.json").read_text() or "{}"):
        sys.stderr.write("fake copilot: not logged in (no account in COPILOT_HOME/config.json)\n")
        return 67
    resume = _flag(args, "--resume")
    session_id = resume or _flag(args, "--session-id") or str(uuid.uuid4())
    if resume and not (home / "session-state" / resume).is_dir():
        sys.stderr.write(f"fake copilot: no session {resume}\n")
        return 1
    scenario = os.environ.get("FAKE_COPILOT_SCENARIO", "happy")
    hooks = {} if scenario == "no-hooks" else _hooks(home)
    cwd = os.getcwd()
    record = Record(home, session_id)
    record.add("session.start", {"sessionId": session_id, "copilotVersion": "9.9.9", "selectedModel": _flag(args, "--model")})
    common = {"session_id": session_id, "cwd": cwd, "transcript_path": str(record.path)}
    _fire(hooks, "SessionStart", {**common, "source": "resume" if resume else "startup"})
    ctx = _fire(hooks, "UserPromptSubmit", {**common, "prompt": _flag(args, "-p") or ""})
    seen = bool((ctx.get("hookSpecificOutput") or {}).get("additionalContext"))
    if scenario == "ai-credits":
        print("Error: You've run out of your AI credits for this month.", flush=True)
        return 1
    wrote, pushed, fetched = _turn(hooks, common, cwd)
    model = _flag(args, "--model") or "auto"
    record.add("assistant.usage", {"model": model, "inputTokens": 100, "outputTokens": 40, "cacheReadTokens": 20,
                                   "cacheWriteTokens": 0, "reasoningTokens": 5,
                                   "copilotUsage": {"model": model, "totalNanoAiu": 2_000_000_000}})
    text = (f"Done.{' Wrote hello.txt.' if wrote else ''}{' Pushed!' if pushed else ' Push was denied by policy.'}"
            f"{' Fetched the spec.' if fetched else ''}{' Contract seen.' if seen else ''}")
    record.add("assistant.message", {"messageId": str(uuid.uuid4()), "model": model, "content": text})
    _fire(hooks, "Stop", {**common, "stop_hook_active": False, "last_assistant_message": text})
    record.add("session.shutdown", {"shutdownType": "routine", "totalPremiumRequests": record.premium_so_far() + 1})
    _fire(hooks, "SessionEnd", {**common, "reason": "exit"})
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
