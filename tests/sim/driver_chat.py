"""The *chat* scenario: talk to Auto the way a customer does.

Each turn is one ``POST /api/chat`` (the nightly suite's body shape), chained
by the chat id the stream hands back. The record keeps the reply, every tool
call with its arguments, every tool result, error frames and timings. Effects
are measured before and after — tasks and agents that exist now and did not
before — so a scenario asserts on what Auto *did*, never on which tool it
picked (PRD-247 D10).
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from .answerer import answer_pending
from .api import Api, ApiError, items_of, iter_pages
from .packs import Scenario
from .results import Check, RunContext, ScenarioResult, effect_checks, must_contain_checks, now_iso
from .sse import ChatTurn, parse_data_stream

ARG_CAP = 2_000
RESULT_CAP = 4_000


def _ids(api: Api, path: str, keys: tuple[str, ...], label: str) -> frozenset[Any]:
    try:
        return frozenset(row["id"] for row in iter_pages(api, path, params={}, keys=keys, limit=100)
                         if isinstance(row, Mapping) and "id" in row)
    except ApiError:
        return frozenset()


def snapshot_effects(api: Api, label: str) -> dict[str, frozenset[Any]]:
    return {
        "tasks": _ids(api, "/api/v1/tasks", ("tasks",), f"{label}:tasks"),
        "agents": _ids(api, "/api/agents/", ("agents",), f"{label}:agents"),
    }


def _trim(value: Any, cap: int) -> Any:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= cap else text[:cap] + f"...[+{len(text) - cap}]"


def turn_record(index: int, prompt: str, turn: ChatTurn, status: int, ms: int | None,
                first_byte_ms: int | None) -> dict[str, Any]:
    return {
        "turn": index, "prompt": prompt, "status": status, "ms": ms, "first_byte_ms": first_byte_ms,
        "text": turn.text, "reasoning_chars": len(turn.reasoning), "tool_names": list(turn.tool_names),
        "tool_calls": [{"name": c.get("toolName"), "args": _trim(c.get("args"), ARG_CAP)} for c in turn.tool_calls],
        "tool_results": [_trim(r.get("result"), RESULT_CAP) for r in turn.tool_results],
        "errors": list(turn.errors), "finish_reason": (turn.finish or {}).get("finishReason"),
        "usage": turn.usage, "frames": turn.frames, "unparsed": turn.unparsed, "chat_id": turn.chat_id,
    }


def run_chat(ctx: RunContext, sc: Scenario) -> ScenarioResult:
    started = now_iso()
    api, settings = ctx.api, ctx.settings
    before = snapshot_effects(api, f"{sc.id}:before")
    turns: list[dict[str, Any]] = []
    asks: list[Mapping[str, Any]] = []
    errors: list[str] = []
    chat_id: str | None = None
    for index, prompt in enumerate(sc.turns, start=1):
        response = api.request("POST", "/api/chat", accept="text/event-stream", timeout_s=settings.chat_timeout_s,
                               label=f"{sc.id}:turn{index}", json_body={
                                   "message": {"role": "user", "parts": [{"type": "text", "text": prompt}]},
                                   **({"chatId": chat_id} if chat_id else {})})
        if response.status >= 400:
            errors.append(f"turn {index}: HTTP {response.status} {response.body[:200]}")
            turns.append(turn_record(index, prompt, parse_data_stream(""), response.status, response.ms,
                                     response.first_byte_ms))
            continue
        turn = parse_data_stream(response.body, response.headers.get("x-chat-id") or response.headers.get("X-Chat-Id"))
        chat_id = turn.chat_id or chat_id
        turns.append(turn_record(index, prompt, turn, response.status, response.ms, response.first_byte_ms))
        errors.extend(f"turn {index}: {e[:300]}" for e in turn.errors)
        if not turn.text and not turn.tool_calls:
            errors.append(f"turn {index}: empty reply ({turn.frames} frames, {turn.unparsed} unparsed lines)")
        asks.extend(answer_pending(api, ctx.persona, label=f"{sc.id}:ask"))
    after = snapshot_effects(api, f"{sc.id}:after")
    observed = {
        "tasks_created": len(after["tasks"] - before["tasks"]),
        "agents_created": len(after["agents"] - before["agents"]),
        "tool_calls_min": sum(len(t["tool_calls"]) for t in turns),
        "errors": len(errors), "turns": len(turns), "asks": len(asks),
    }
    reply_text = "\n\n".join(t["text"] for t in turns if t["text"])
    checks = (
        Check("replied", any(t["text"] for t in turns), "" if reply_text else "no turn produced text"),
        *effect_checks(sc.expect_effects, observed),
        *must_contain_checks(sc.must_contain, reply_text),
    )
    outcome = "ok" if not errors else "error"
    return ScenarioResult(
        id=sc.id, kind="chat", started_at=started, ended_at=now_iso(), outcome=outcome,
        ok=all(c.ok for c in checks) and not errors, chat=tuple(turns),
        artifacts={"chat_id": chat_id, "new_task_ids": sorted(after["tasks"] - before["tasks"], key=str),
                   "new_agent_ids": sorted(after["agents"] - before["agents"], key=str)},
        effects=observed, asks=tuple(asks), checks=checks, errors=tuple(errors),
        evidence_text=reply_text, expect=sc.expect, brief="\n".join(sc.turns),
    )


def list_tasks_by_ids(api: Api, ids: list[Any], label: str) -> list[dict[str, Any]]:
    """Fetch the tasks a chat created, for the record."""
    found = []
    for task_id in ids[:20]:
        try:
            found.append(api.get(f"/api/v1/tasks/{task_id}", label=f"{label}:task"))
        except ApiError:
            continue
    return found


__all__ = ["run_chat", "snapshot_effects", "turn_record", "list_tasks_by_ids", "items_of"]
