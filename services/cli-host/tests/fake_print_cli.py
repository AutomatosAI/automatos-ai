#!/usr/bin/env python3
"""A stand-in print-mode CLI for CI (PRD-253 S0.2) — a GATED CLI whose turn ends
when its process exits, the way ``copilot -p`` does.

It talks to the host the way a CLI's hook config would: the shim as a command,
the payload on stdin. ``FAKE_PRINT_SCENARIO`` decides what the host sees:

* ``happy`` (default): SessionStart → Stop → SessionEnd → exit 0;
* ``no-hooks``: the gate never loads (something switched the hooks off) — it
  works anyway, prints, and exits 0;
* ``fail``: SessionStart → Stop → exit 3;
* ``no-stop``: SessionStart → exit 0 without reaching Stop;
* ``linger``: SessionStart → Stop → SessionEnd → never exits;
* ``hang``: no hooks, never exits.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid

SESSION_ID = str(uuid.uuid4())
ANSWER = "Done: the turn's answer."


def _hook(event: str, **fields) -> None:
    payload = {"hook_event_name": event, "session_id": SESSION_ID, "cwd": os.getcwd(), **fields}
    subprocess.run([sys.executable, "-m", "automatos_cli_host.hook_shim"], input=json.dumps(payload),
                   capture_output=True, text=True, timeout=30, env=os.environ)


def _idle() -> None:
    while True:
        time.sleep(0.2)


def main(argv) -> int:
    if "--version" in argv:
        print("printcli 0.0.1 (fake print-mode CLI for CI)")
        return 0
    scenario = os.environ.get("FAKE_PRINT_SCENARIO", "happy")
    print("working on it", flush=True)
    if scenario == "hang":
        _idle()
    if scenario == "no-hooks":
        print("All done, and nobody gated any of it.", flush=True)
        return 0
    _hook("SessionStart", source="startup")
    if scenario == "no-stop":
        return 0
    _hook("Stop", stop_hook_active=False, last_assistant_message=ANSWER)
    if scenario == "fail":
        return 3
    _hook("SessionEnd", reason="complete")
    if scenario == "linger":
        _idle()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
