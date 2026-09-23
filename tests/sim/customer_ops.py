"""Helpers behind ``tests.sim.customer`` — what a customer-night persona reads and writes.

The persona is a Claude Code session playing the operator; these functions give
it reliable eyes (inventory, cost) and a rendered brief. Nothing here decides
what the persona does — that is the prompt's job.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import datetime, timezone, tzinfo
from typing import Any, Mapping, Optional, Sequence

from .api import Api, ApiError, items_of
from .config import Settings
from .cost import CostError, summarise, usage_rows

PLACEHOLDER = re.compile(r"\{\{([A-Z_]+)\}\}")
SLIM_TASK = ("id", "title", "status", "priority", "assigned_agent_id", "review_mode", "created_at", "tags")
SLIM_AGENT = ("id", "name", "status", "job_title", "tags")


def render_prompt(template: str, values: Mapping[str, str]) -> str:
    """Fill ``{{NAME}}`` placeholders; a placeholder without a value is an error, not a blank."""
    wanted = {m.group(1) for m in PLACEHOLDER.finditer(template)}
    missing = sorted(wanted - set(values))
    if missing:
        raise KeyError(f"prompt placeholders without values: {', '.join(missing)}")
    return PLACEHOLDER.sub(lambda m: str(values[m.group(1)]), template)


def has_tag(row: Mapping[str, Any], tag: str) -> bool:
    """Tagged in ``tags``, or carrying the tag in its name/title/description (agents have no free tag field on every route)."""
    tags = row.get("tags")
    if isinstance(tags, list) and tag in tags:
        return True
    haystack = " ".join(str(row.get(k) or "") for k in ("name", "title", "description"))
    return tag in haystack


def _slim(row: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    return {k: row.get(k) for k in keys if k in row}


def _runtime(agent: Mapping[str, Any]) -> str:
    cfg = agent.get("configuration") if isinstance(agent.get("configuration"), Mapping) else {}
    return str(cfg.get("runtime") or "api")


def _fetch(api: Api, path: str, params: Mapping[str, Any], keys: tuple[str, ...]) -> tuple[list[Any], str | None]:
    try:
        return items_of(api.get(path, params=params), *keys), None
    except ApiError as exc:
        return [], f"{path}: HTTP {exc.status} {exc.body[:120]}"


def inventory(api: Api, tag: str | None = None) -> dict[str, Any]:
    """What the workspace holds right now — optionally only the rows carrying ``tag``."""
    agents, e1 = _fetch(api, "/api/agents/", {}, ("agents",))
    tasks, e2 = _fetch(api, "/api/v1/tasks", {"limit": 500}, ("tasks",))
    deliverables, e3 = _fetch(api, "/api/deliverables", {"limit": 100}, ("deliverables",))
    reports, e4 = _fetch(api, "/api/reports", {"period": "1d", "limit": 100}, ("reports",))
    grants, e5 = _fetch(api, "/api/v1/approval-grants", {"status": "pending"}, ("grants",))
    keep = (lambda rows: [r for r in rows if isinstance(r, Mapping) and has_tag(r, tag)]) if tag else \
           (lambda rows: [r for r in rows if isinstance(r, Mapping)])
    return {
        "tag": tag,
        "agents": [{**_slim(a, SLIM_AGENT), "runtime": _runtime(a)} for a in keep(agents)],
        "tasks": [_slim(t, SLIM_TASK) for t in keep(tasks)],
        "deliverables": [_slim(d, ("id", "title", "artifact_type", "source_type", "source_id", "agent_id", "file_path", "created_at")) for d in keep(deliverables)],
        "reports": [_slim(r, ("id", "title", "report_type", "agent_id", "status", "created_at")) for r in reports if isinstance(r, Mapping)],
        "questions": [_slim(g, ("id", "kind", "question_md", "reason", "tool_name", "options", "subject_type", "subject_id", "expires_at", "requested_at")) for g in grants if isinstance(g, Mapping)],
        "errors": [e for e in (e1, e2, e3, e4, e5) if e],
    }


def _span(seconds: float) -> str:
    minutes = int(seconds // 60)
    if minutes < 1:
        return f"{int(seconds)}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 1:
        return f"{minutes}m"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h" if days else f"{hours}h {minutes}m"


def local_time(value: Any, *, now: Optional[datetime] = None, tz: Optional[tzinfo] = None) -> str:
    """An API timestamp in this machine's local time, with how far off it is.

    F091 (night 3): card times came through as UTC ISO strings and the persona
    took four live cards for expired. ``tz`` defaults to the machine's zone; a
    value without an offset is the API's naive UTC.
    """
    if not value:
        return "—"
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return str(value)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    delta = (stamp - (now or datetime.now(timezone.utc))).total_seconds()
    when = f"in {_span(delta)}" if delta >= 0 else f"{_span(-delta)} ago"
    return f"{stamp.astimezone(tz):%Y-%m-%d %H:%M %Z} ({when})"


def question_line(q: Mapping[str, Any]) -> str:
    """The text of an ask (``question_md``) or of a permission hold (``reason``/``tool_name``)."""
    return str(q.get("question_md") or q.get("question") or q.get("reason") or q.get("tool_name") or "").strip()


def render_inventory(inv: Mapping[str, Any]) -> str:
    """A compact markdown snapshot the persona can read at the top of an iteration."""
    lines = [f"### Workspace inventory{' (tag ' + inv['tag'] + ')' if inv.get('tag') else ''}", ""]
    agents = inv.get("agents") or []
    lines.append(f"**Agents ({len(agents)})**: " + (", ".join(f"{a.get('name')} (#{a.get('id')}, {a.get('runtime')})" for a in agents) or "none"))
    by_status: dict[str, list] = defaultdict(list)
    for t in inv.get("tasks") or []:
        by_status[str(t.get("status"))].append(t)
    lines.append(f"**Tasks ({sum(len(v) for v in by_status.values())})**: " + (", ".join(f"{k} {len(v)}" for k, v in sorted(by_status.items())) or "none"))
    for status in ("review", "blocked", "in_progress", "assigned"):
        for t in by_status.get(status, [])[:12]:
            lines.append(f"  - #{t.get('id')} [{status}] {t.get('title')} → agent #{t.get('assigned_agent_id')}")
    lines.append(f"**Deliverables ({len(inv.get('deliverables') or [])})**: " + ", ".join(f"#{d.get('id')} {d.get('title')}" for d in (inv.get("deliverables") or [])[:12]))
    lines.append(f"**Reports today ({len(inv.get('reports') or [])})**: " + ", ".join(f"#{r.get('id')} {r.get('title')}" for r in (inv.get("reports") or [])[:12]))
    qs = inv.get("questions") or []
    lines.append(f"**Pending questions/approvals ({len(qs)})**: " + ("; ".join(
        f"#{q.get('id')} {q.get('kind')}: {question_line(q)[:100]}"
        + (f" (expires {local_time(q.get('expires_at'))})" if q.get("expires_at") else "")
        for q in qs[:8]) or "none"))
    for err in inv.get("errors") or []:
        lines.append(f"  ! {err}")
    return "\n".join(lines)


def cost_table(settings: Settings, workspace_id: str, since_iso: str) -> str:
    """Spend since the night began: per model and per agent, from ``llm_usage``."""
    try:
        rows = usage_rows(settings, workspace_id, since_iso)
    except (CostError, ValueError) as exc:
        return f"cost unavailable: {exc}"
    total = summarise(rows)
    per_agent: Counter = Counter()
    calls_agent: Counter = Counter()
    for r in rows:
        per_agent[str(r.get("agent_id") or "-")] += float(r.get("total_cost") or 0.0)
        calls_agent[str(r.get("agent_id") or "-")] += 1
    lines = [f"Since {since_iso[:19]}: **${total['cost_usd']:.4f}**, {total['calls']} model calls, "
             f"{total['input_tokens']} in / {total['output_tokens']} out tokens, {total['errors']} failed calls", "",
             "| model | calls | cost |", "|---|---|---|"]
    lines += [f"| {m} | {v['calls']} | ${v['cost_usd']:.4f} |" for m, v in sorted(total["by_model"].items(), key=lambda kv: -kv[1]["cost_usd"])]
    lines += ["", "| agent id | calls | cost |", "|---|---|---|"]
    lines += [f"| {a} | {calls_agent[a]} | ${c:.4f} |" for a, c in per_agent.most_common()]
    return "\n".join(lines)


def purge_tagged(api: Api, inv: Mapping[str, Any]) -> list[str]:
    """Delete the tagged tasks and agents the inventory listed; returns one line per attempt."""
    done = []
    for t in inv.get("tasks") or []:
        try:
            api.delete(f"/api/v1/tasks/{t['id']}", label="purge:task")
            done.append(f"task #{t['id']} deleted")
        except ApiError as exc:
            done.append(f"task #{t['id']} NOT deleted: HTTP {exc.status}")
    for a in inv.get("agents") or []:
        try:
            api.delete(f"/api/agents/{a['id']}", label="purge:agent")
            done.append(f"agent #{a['id']} {a.get('name')} deleted")
        except ApiError as exc:
            done.append(f"agent #{a['id']} NOT deleted: HTTP {exc.status}")
    return done
