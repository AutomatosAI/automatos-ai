"""The customer-night persona's hands (PRD-247, the night Gerard asked for).

A Claude Code session plays the operator overnight, in the operator's own
workspace, with the real tools. It talks to Auto, files work, answers the
questions agents raise and reads what came back — through these commands, so
every action is recorded and the platform is used exactly as a customer
would use it (the UI makes the same calls).

    python3 -m tests.sim.customer preflight
    python3 -m tests.sim.customer chat "What's on my board?"        [--chat-id ID]
    python3 -m tests.sim.customer questions | answer ID "text" | grant ID
    python3 -m tests.sim.customer inventory [--tag sim-night-2026-09-18] [--json]
    python3 -m tests.sim.customer cost --since 2026-09-18T22:00:00+00:00
    python3 -m tests.sim.customer judge --brief brief.md --output out.md
    python3 -m tests.sim.customer purge --tag sim-night-2026-09-18 --yes

Unlike ``tests.sim.night`` this deliberately addresses the operator's own
workspace (``SIM_WORKSPACE_ID``, default the local edition's); the persona's
rules, not a guard, keep it to tagged rows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .api import Api, ApiError, Trace
from .config import (DEFAULT_WORKSPACE_ID, LOGS_DIR, MIN_BALANCE_USD, PLATFORM_KEY_WORKSPACE_ID, SIM_HOME, ConfigError,
                     load_settings)
from .customer_ops import (cost_table, inventory, local_time, purge_tagged, question_line, render_inventory,
                           render_prompt)
from .judge import judge_output
from .sse import parse_data_stream

NIGHT_DIR_ENV = "CUSTOMER_NIGHT_DIR"
HOST_LOG = Path.home() / ".automatos" / "cli-host" / "host.log"


def _api(args: argparse.Namespace) -> tuple[Api, Any]:
    settings = load_settings({"api_url": getattr(args, "api_url", None)})
    workspace_id = getattr(args, "workspace_id", None) or os.environ.get("SIM_WORKSPACE_ID") or DEFAULT_WORKSPACE_ID
    return Api(settings.api_url, settings.api_key or None, workspace_id, Trace(), timeout_s=120), settings


def _night_dir() -> Path | None:
    value = os.environ.get(NIGHT_DIR_ENV)
    return Path(value) if value else None


def _append_jsonl(name: str, record: dict[str, Any]) -> None:
    night = _night_dir()
    if night:
        night.mkdir(parents=True, exist_ok=True)
        with (night / name).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")


def cmd_chat(args: argparse.Namespace) -> int:
    api, settings = _api(args)
    response, chat_id = api.stream_chat(args.text, chat_id=args.chat_id, agent_id=args.agent_id, timeout_s=settings.chat_timeout_s)
    turn = parse_data_stream(response.body, chat_id)
    record = {"ts": time.time(), "prompt": args.text, "chat_id": turn.chat_id, "status": response.status, "ms": response.ms,
              "text": turn.text, "tool_calls": [{"name": c.get("toolName"), "args": c.get("args")} for c in turn.tool_calls],
              "errors": list(turn.errors), "usage": turn.usage}
    _append_jsonl("chats.jsonl", record)
    if args.json:
        print(json.dumps(record, default=str, indent=2))
        return 0
    if response.status >= 400:
        print(f"HTTP {response.status}: {response.body[:400]}")
        return 1
    print(turn.text or "(no reply text)")
    print(f"\n--- chat_id={turn.chat_id} · {response.ms} ms · tools: {', '.join(turn.tool_names) or 'none'}"
          + (f" · errors: {'; '.join(e[:120] for e in turn.errors)}" if turn.errors else ""))
    return 0


def cmd_questions(args: argparse.Namespace) -> int:
    api, _ = _api(args)
    inv = inventory(api)
    for q in inv["questions"]:
        how = "answer <id> \"...\"" if q.get("kind") == "question" else "grant <id>"
        print(f"#{q['id']} [{q.get('kind')}] {question_line(q)}" + (f"  options: {q['options']}" if q.get("options") else "")
              + (f"  expires: {local_time(q['expires_at'])}" if q.get("expires_at") else "") + f"  → {how}")
    if not inv["questions"]:
        print("no pending questions or approvals")
    return 0


def cmd_answer(args: argparse.Namespace) -> int:
    api, _ = _api(args)
    body = {"option": args.text} if args.option else {"answer_text": args.text}
    api.post(f"/api/v1/approval-grants/{args.grant_id}/answer", body)
    _append_jsonl("answers.jsonl", {"ts": time.time(), "grant_id": args.grant_id, **body})
    print(f"answered #{args.grant_id}")
    return 0


def cmd_grant(args: argparse.Namespace) -> int:
    api, _ = _api(args)
    api.post(f"/api/v1/approval-grants/{args.grant_id}/grant", {})
    _append_jsonl("answers.jsonl", {"ts": time.time(), "grant_id": args.grant_id, "granted": True})
    print(f"granted #{args.grant_id}")
    return 0


def cmd_inventory(args: argparse.Namespace) -> int:
    api, _ = _api(args)
    inv = inventory(api, args.tag)
    print(json.dumps(inv, indent=2, default=str) if args.json else render_inventory(inv))
    return 0


def cmd_cost(args: argparse.Namespace) -> int:
    api, settings = _api(args)
    print(cost_table(settings, api.workspace_id, args.since))
    return 0


def cmd_judge(args: argparse.Namespace) -> int:
    _, settings = _api(args)
    verdict = judge_output(settings, brief=Path(args.brief).read_text(encoding="utf-8"), expect=args.expect or "",
                           output=Path(args.output).read_text(encoding="utf-8"))
    if verdict is None:
        print("no judge configured (OPENROUTER_API_KEY unset in ~/.automatos-sim/env) — grade it yourself")
        return 1
    print(json.dumps(verdict, indent=2))
    return 0


def cmd_preflight(args: argparse.Namespace) -> int:
    """Everything the night needs, checked as the customer would find it."""
    api, settings = _api(args)
    ok = True
    health = api.request("GET", "/health")
    print(f"backend {settings.api_url}: HTTP {health.status}")
    ok &= health.status == 200
    bridge = api.request("GET", "/api/v1/session-tools/mcp")
    print(f"session tools bridge (PRD-245): {'present' if bridge.status != 404 else 'ABSENT — sessions will have no platform tools'}")
    ok &= bridge.status != 404
    inv = inventory(api)
    cli = [a for a in inv["agents"] if a.get("runtime") == "cli"]
    print(f"workspace {api.workspace_id}: {len(inv['agents'])} agents ({len(cli)} Claude/Codex sessions), "
          f"{len(inv['tasks'])} tasks, {len(inv['questions'])} pending questions")
    for err in inv["errors"]:
        print(f"  ! {err}")
        ok = False
    if HOST_LOG.exists():
        lines = [ln for ln in HOST_LOG.read_text(encoding="utf-8", errors="replace").splitlines() if "CLI host" in ln and "serving" in ln]
        print("cli host: " + (lines[-1].split("INFO")[-1].strip()[:140] if lines else "no 'serving' line in host.log"))
    else:
        print("cli host: no host.log — is the host installed and running? (make cli-host-status)")
    print("deliverables folder: " + (args.deliverables_dir if Path(args.deliverables_dir).is_dir() else f"MISSING {args.deliverables_dir}"))

    # F048: what the night MUTATES and must be restorable, plus the balance that
    # decides whether the night can run at all. Night 1 changed workspace
    # settings and blueprint defaults and nobody had a before-picture.
    snapshot = _write_state_snapshot(api, args)
    print("state snapshot: " + (str(snapshot) if snapshot else "not written (no night dir set)"))
    left, said = _openrouter_balance(api)
    print(f"openrouter balance: {said}")
    if left is None or left < MIN_BALANCE_USD:
        # F208: night 6 launched with the balance unknown and died of it at iteration 10.
        print(f"REFUSED: a night has cost $3-11; launch with at least ${MIN_BALANCE_USD:,.2f} of OpenRouter credit "
              "known (top up at https://openrouter.ai/settings/credits, or set SIM_MIN_BALANCE_USD).")
        ok = False
    return 0 if ok else 2


def _write_state_snapshot(api: Any, args: argparse.Namespace) -> Optional[Path]:
    """Save the rows a night mutates, so the morning can diff or restore them.

    ``workspaces.settings`` and the system settings are GLOBAL: the persona
    changes them in passing and a later night inherits the change as if it were
    the product's own behaviour.
    """
    night = _night_dir()
    if not night:
        return None
    state: dict[str, Any] = {"captured_at": datetime.now(timezone.utc).isoformat()}
    for label, path in (
        ("workspace", "/api/workspaces/current"),
        ("system_settings", "/api/system-settings/by-category"),
        ("orchestrator", "/api/workspaces/current/orchestrator"),
    ):
        try:
            resp = api.request("GET", path)
            state[label] = json.loads(resp.body) if resp.status == 200 else {"_status": resp.status}
        except Exception as exc:  # noqa: BLE001 — a snapshot must not block the night
            state[label] = {"_error": str(exc)[:200]}
    night.mkdir(parents=True, exist_ok=True)
    target = night / "PREFLIGHT-STATE.json"
    target.write_text(json.dumps(state, indent=1, default=str), encoding="utf-8")
    return target


CREDITS_PATH = "/api/analytics/llm/openrouter/credits"


def _openrouter_balance(api: Any) -> tuple[Optional[float], str]:
    """What the night can spend (credits minus usage), and the line that says
    so; None when it could not be read.

    Night 1 ran with the balance unknown. F208: night 6 asked
    /api/v1/providers/openrouter/balance, which does not exist (404), launched
    anyway and died of it at iteration 10. The credits route answers for the
    platform key's workspace (c1), not a sim workspace.
    """
    c1 = Api(api.base_url, api.api_key or None, PLATFORM_KEY_WORKSPACE_ID, api.trace, timeout_s=api.timeout_s)
    try:
        resp = c1.request("GET", CREDITS_PATH)
    except Exception as exc:  # noqa: BLE001 — unreadable is its own answer
        return None, f"unavailable ({type(exc).__name__})"
    if resp.status != 200:
        return None, f"unavailable (HTTP {resp.status} from {CREDITS_PATH} in c1)"
    try:
        data = json.loads(resp.body)
        credits, usage = float(data["total_credits"]), float(data["total_usage"])
    except (ValueError, KeyError, TypeError):
        return None, f"unreadable ({resp.body[:120]})"
    return credits - usage, f"${credits - usage:,.2f} left (credits ${credits:,.2f} - usage ${usage:,.2f})"


def cmd_render_prompt(args: argparse.Namespace) -> int:
    api, settings = _api(args)
    now = datetime.now(timezone.utc)
    values = {
        "DATE": args.date or date.today().isoformat(), "NIGHT_DIR": args.night_dir, "WORKSPACE_ID": api.workspace_id,
        "API_URL": settings.api_url, "ITER": str(args.iter), "MAX_ITERS": str(args.max_iters), "STOP_AT": args.stop_at,
        "NOW": now.isoformat(timespec="minutes"), "DELIVERABLES_DIR": args.deliverables_dir,
        "PERSONA": Path(args.persona).read_text(encoding="utf-8").strip(),
        "INVENTORY": render_inventory(inventory(api)),
    }
    for item in args.set or []:
        key, _, value = item.partition("=")
        values[key] = value
    rendered = render_prompt(Path(args.template).read_text(encoding="utf-8"), values)
    Path(args.out).write_text(rendered, encoding="utf-8")
    print(args.out)
    return 0


def cmd_purge(args: argparse.Namespace) -> int:
    api, _ = _api(args)
    inv = inventory(api, args.tag)
    print(render_inventory(inv))
    if not args.yes:
        print("\n(dry run — add --yes to delete the tagged tasks and agents above)")
        return 0
    for line in purge_tagged(api, inv):
        print(line)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tests.sim.customer", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace-id", help="default SIM_WORKSPACE_ID or the local edition's workspace")
    parser.add_argument("--api-url")
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("chat"); c.add_argument("text"); c.add_argument("--chat-id"); c.add_argument("--agent-id", type=int); c.add_argument("--json", action="store_true"); c.set_defaults(fn=cmd_chat)
    sub.add_parser("questions").set_defaults(fn=cmd_questions)
    a = sub.add_parser("answer"); a.add_argument("grant_id", type=int); a.add_argument("text"); a.add_argument("--option", action="store_true", help="text is one of the offered options"); a.set_defaults(fn=cmd_answer)
    g = sub.add_parser("grant"); g.add_argument("grant_id", type=int); g.set_defaults(fn=cmd_grant)
    i = sub.add_parser("inventory"); i.add_argument("--tag"); i.add_argument("--json", action="store_true"); i.set_defaults(fn=cmd_inventory)
    k = sub.add_parser("cost"); k.add_argument("--since", required=True, help="ISO timestamp"); k.set_defaults(fn=cmd_cost)
    j = sub.add_parser("judge"); j.add_argument("--brief", required=True); j.add_argument("--output", required=True); j.add_argument("--expect"); j.set_defaults(fn=cmd_judge)
    p = sub.add_parser("preflight"); p.add_argument("--deliverables-dir", default=str(Path.home() / "Development" / "deliverables")); p.set_defaults(fn=cmd_preflight)
    r = sub.add_parser("render-prompt"); r.add_argument("--template", required=True); r.add_argument("--persona", required=True); r.add_argument("--out", required=True)
    r.add_argument("--night-dir", required=True); r.add_argument("--iter", type=int, default=1); r.add_argument("--max-iters", type=int, default=10)
    r.add_argument("--stop-at", default="06:30"); r.add_argument("--date"); r.add_argument("--deliverables-dir", default=str(Path.home() / "Development" / "deliverables"))
    r.add_argument("--set", action="append", help="KEY=VALUE extra placeholder"); r.set_defaults(fn=cmd_render_prompt)
    u = sub.add_parser("purge"); u.add_argument("--tag", required=True); u.add_argument("--yes", action="store_true"); u.set_defaults(fn=cmd_purge)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args))
    except ConfigError as exc:
        print(f"cannot run: {exc}", file=sys.stderr)
        return 2
    except ApiError as exc:
        print(f"{exc.method} {exc.path} -> HTTP {exc.status}: {exc.body[:400]}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
