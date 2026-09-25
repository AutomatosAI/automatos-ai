"""PRD-245 W1 — the Automatos tools a ticket session may call (decision D2).

A ``runtime: cli`` agent's session has Claude Code's own tools and nothing of
ours: the platform actions live in this process, and until now no session could
reach them, while the skills it was given told it to call
``composio_execute``/``platform_*`` by name. This module is the ONE definition of
the short list a session does get — the names, the schema the model reads, and
what each one runs on the backend.

Three properties the rest of the wave leans on:

* **the list is fixed and stable.** Every session of an agent advertises the same
  tools with the same text, or Claude Code's prompt cache misses on every turn
  (PRD-234 §Design 2). Scope is applied per call, never by varying the list;
* **the scope is forced here, not trusted from the call.** ``update_ticket``
  moves THIS ticket and may not close it (the host's result does that);
  ``submit_report`` attaches to THIS ticket and the calling agent;
* **every call goes through the API agents' own executor**
  (``UnifiedExecutor.execute_tool`` via the PRD-64 ``platform_execute``
  dispatcher), so the platform's policy gate, its registry validation, its
  admin gating and its tool telemetry all apply to a session exactly as they do
  to an API turn. No second execution path.

Pure data plus one async call: the HTTP/JSON-RPC layer is ``api/session_tools.py``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple
from uuid import UUID

logger = logging.getLogger(__name__)

# The dispatcher every platform action is reached through (PRD-64): it validates
# the action against the registry and reports a missing parameter with a hint the
# model can act on, instead of a bare failure.
PLATFORM_DISPATCHER = "platform_execute"

# How a tool reaches the executor. Most Automatos tools ARE platform actions, so
# they go through the PRD-64 dispatcher, which validates the action against the
# registry and reports a missing parameter with a hint. ``composio_execute`` is
# not a platform action — it is a tool name the executor routes to the Composio
# router itself — so it is dispatched DIRECTLY under that name.
DISPATCH_PLATFORM_ACTION = "platform_action"
DISPATCH_TOOL_NAME = "tool_name"

# A session may not move its own ticket OUT of ``in_progress`` at all.
#
# "Done" was always refused — it is a fact the host observes when the session
# ends (PRD-245 D3), and a session that could close its own ticket could report
# success for work it did not do. ``blocked`` and ``review`` turned out to be
# worse than that, not milder: ``apply_result`` returns early for a ticket that
# is no longer ``in_progress``, so the moment a session set either one, its own
# turn-end result was DISCARDED — no deliverables registered, no report written,
# no result text, no usage booked. The ticket sat on the board looking like
# finished work with nothing behind it.
#
# A session that is genuinely stuck has ``ask_human``: that parks the ticket the
# supported way, at turn end, keeping everything the session produced.
SESSION_TICKET_STATUSES: Tuple[str, ...] = ("in_progress",)
# A progress note is read by a human on a card; keep it to a couple of sentences.
MAX_NOTE_CHARS = 400
# What ``platform_submit_report`` accepts besides the forced linkage.
SUBMIT_REPORT_FIELDS: Tuple[str, ...] = (
    "title", "content", "summary", "report_type", "status", "metrics",
    "recommendations", "action_items",
)
REFUSED_TICKET_STATUSES: Tuple[str, ...] = (
    "done", "failed", "cancelled", "inbox", "assigned", "blocked", "review",
)

MAX_TOOL_RESULT_CHARS = 40000
# F161: room for read_step_file's header and cut note inside that cap.
STEP_FILE_FRAME_CHARS = 2000
# A recorded fact is read later beside dozens of others; keep it a fact, not a report.
MAX_MEMORY_CHARS = 1500
MAX_MEMORY_KEY_CHARS = 120
# The taxonomy platform_store_memory accepts (PRD-206); anything else is dropped
# and the action's own default applies.
MEMORY_TYPES: Tuple[str, ...] = (
    "tool_outcome", "task_learning", "playbook_pattern", "user_fact",
    "business_fact", "preference", "procedure", "decision", "open_loop",
    "thread_summary",
)
# The operator reads a question on a CARD, not in a terminal: past a short
# paragraph plus its options it is a report, not a question (PRD-225's own rule).
MAX_QUESTION_CHARS = 700
MAX_QUESTION_OPTIONS = 6


@dataclass(frozen=True)
class SessionTool:
    """One tool a session sees: what the model reads, and what it runs."""

    name: str                                   # as the session sees it (mcp__automatos__<name>)
    action: str                                 # the platform action, or the tool name, it runs
    description: str                            # stable text — part of the prompt
    input_schema: Dict[str, Any]
    # params the session sent → params the action gets, with this ticket's scope
    # forced on. ``None`` passes them through unchanged.
    scope: Optional[Callable[[Dict[str, Any], "SessionContext"], Dict[str, Any]]] = None
    # Which of the two ``action`` is (a defaulted field, so it sits with the rest).
    dispatch: str = DISPATCH_PLATFORM_ACTION
    # A tool that only reads is safe to retry and cannot change the board.
    reads_only: bool = True
    tags: Tuple[str, ...] = field(default_factory=tuple)
    # Almost every tool IS its platform action, dispatched. ``ask_human`` is not:
    # the action would park the ticket this instant (it is a board-task ask) while
    # the session is still mid-turn, and a parked ticket cannot take its own
    # result. So it brings a runner that files the question UNPARKED and lets the
    # turn's end do the parking. One exception, named here, not a second path.
    runner: Optional[Callable[[Any, Dict[str, Any], "SessionContext"], Any]] = None
    # What comes BACK, when the action returns more than this tool advertises.
    # The scope functions guard the request; this guards the response.
    project: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None


@dataclass(frozen=True)
class SessionContext:
    """Who is calling: resolved from the ticket's token, never from the call."""

    task_id: int
    agent_id: Optional[int]
    agent_name: Optional[str]
    workspace_id: Any
    # PRD-245 mission parity: the shared field of the mission this ticket belongs
    # to, resolved server-side from OrchestrationRun.config (never from the call).
    # ``None`` for a standalone ticket — then record_memory writes durable memory
    # only, and says so.
    mission_field_id: Optional[str] = None


def _scope_update_ticket(params: Dict[str, Any], ctx: SessionContext) -> Dict[str, Any]:
    """Just the note. A session cannot move its ticket at all.

    ``done`` was always refused — the host records the outcome when the turn
    ends. ``blocked`` and ``review`` were allowed at first and turned out to be
    worse than ``done``, not milder: ``apply_result`` returns early for a ticket
    that is no longer ``in_progress``, so the moment a session set either one its
    own turn was discarded — no deliverables, no report, no result text, no
    usage. A session that is genuinely stuck has ``ask_human``, which parks the
    ticket at turn end and keeps everything.
    """
    status = str(params.get("status") or "").strip().lower()
    if status and status != "in_progress":
        raise SessionToolRefused(
            f"a session cannot move its ticket to {status!r}, or anywhere else. Moving it ends the run, "
            "and your turn's work — your files, your report, your result — is then thrown away. To stop "
            "for an answer use ask_human, which parks the ticket properly and keeps everything. To "
            "finish, just end your turn: the host records the outcome. This tool only leaves a note."
        )
    note = str(params.get("note") or "").strip()
    if not note:
        raise SessionToolRefused(
            "update_ticket leaves a progress note — say what you are doing, in a sentence."
        )
    return {"note": note[:MAX_NOTE_CHARS]}


async def _run_update_ticket(db: Any, params: Dict[str, Any], ctx: SessionContext) -> Dict[str, Any]:
    """Write the note where the operator will see it.

    NOT ``platform_update_task_status``, which this tool used to dispatch to. For
    a ticket that is already ``in_progress`` — which every running session's is —
    that action takes its atomic-claim branch, whose ``UPDATE … WHERE status <>
    'in_progress'`` matches no row; it returned ``{"success": True}`` having
    written nothing at all, and never read ``blocked_reason``. The note vanished
    and the model was told it had landed.
    """
    from services.cli_host_service import record_session_note

    return record_session_note(
        db, task_id=ctx.task_id, workspace_id=ctx.workspace_id,
        agent_name=ctx.agent_name, note=params.get("note") or "",
    )


def _scope_submit_report(params: Dict[str, Any], ctx: SessionContext) -> Dict[str, Any]:
    """This ticket's report, attributed to the calling agent by the executor. A
    session reports on ITS ticket: any other id the call names is dropped."""
    # The whitelist is the ACTION's own optional fields (``handlers_reports``),
    # not a narrower set: WRITER and TRACKER both pass ``status`` and ``metrics``
    # in their skill bodies, and dropping them silently made the session's report
    # poorer than the same skill's report from an API agent for no reason.
    out = {k: v for k, v in params.items() if k in SUBMIT_REPORT_FIELDS}
    out["linked_task_ids"] = [ctx.task_id]
    return out


def _scope_nothing(params: Dict[str, Any], ctx: SessionContext) -> Dict[str, Any]:
    """A tool with no parameters takes none: an argument the schema never
    declared is dropped, not forwarded to the action."""
    return {}


def _scope_list_tasks(params: Dict[str, Any], ctx: SessionContext) -> Dict[str, Any]:
    return {k: v for k, v in params.items() if k in ("status", "assigned_agent_name", "limit")}


# What ``list_tasks`` says it returns, and therefore all it may return. The board
# action hands back every field of every ticket in the workspace, including each
# one's full ``description`` and ``error_message`` — operator-written text that
# routinely carries paths, hostnames and pasted credentials. One call would put
# every other ticket's brief in front of a session whose prompt an injected web
# page or repo file may be steering. The tool promises a few fields; it returns
# those fields.
#
# These are the handler's OWN key names (``handlers_board_tasks``: the list
# handler builds each row by hand, and names the agent ``assigned_agent``), and
# the handler's dict comes back from the executor AS IS — ``tasks`` sits at the
# top level, not under a ``result`` key. The first version of this projection
# assumed both wrongly, and its test fed it the imagined shape: it narrowed
# nothing and passed. A parity test now reads the handler's source.
LIST_TASKS_FIELDS: Tuple[str, ...] = ("id", "title", "status", "priority", "assigned_agent")


def _project_list_tasks(result: Dict[str, Any]) -> Dict[str, Any]:
    """Keep the envelope, narrow each task to the advertised fields."""
    if not isinstance(result, dict) or not result.get("success"):
        return result
    rows = result.get("tasks")
    if not isinstance(rows, list):
        return result
    narrowed = [
        {k: row.get(k) for k in LIST_TASKS_FIELDS if k in row}
        for row in rows if isinstance(row, dict)
    ]
    return {**result, "tasks": narrowed}


async def _run_ask_human(db: Any, params: Dict[str, Any], ctx: SessionContext) -> Dict[str, Any]:
    """File the question through PRD-225's shared internals, UNPARKED, and
    remember it on the ticket. Lazy import: the host service reads this table."""
    from services.cli_host_service import raise_session_ask

    return await raise_session_ask(
        db, task_id=ctx.task_id, workspace_id=ctx.workspace_id,
        agent_id=ctx.agent_id, agent_name=ctx.agent_name,
        question=params.get("question") or "", options=params.get("options"),
    )


def _scope_ask_human(params: Dict[str, Any], ctx: SessionContext) -> Dict[str, Any]:
    """The subject is THIS ticket, always: a session asks about its own work.

    The shared internals only resume a ``board_task`` subject, which is what a
    ticket is — so the ask is answerable and the answer has somewhere to go."""
    question = str(params.get("question") or "").strip()
    if not question:
        raise SessionToolRefused(
            "ask_human needs a question. State the one decision you need, in a sentence or two."
        )
    if len(question) > MAX_QUESTION_CHARS:
        raise SessionToolRefused(
            f"that question is {len(question)} characters; the operator reads it on a card, so keep it "
            f"under {MAX_QUESTION_CHARS}. Cut the narrative and keep the decision."
        )
    out: Dict[str, Any] = {"subject_type": "board_task", "subject_id": str(ctx.task_id), "question": question}
    options = [str(o).strip() for o in (params.get("options") or []) if str(o).strip()]
    if options:
        out["options"] = options[:MAX_QUESTION_OPTIONS]
    return out


def _scope_composio(params: Dict[str, Any], ctx: SessionContext) -> Dict[str, Any]:
    """The action and its parameters, and nothing else.

    There is no ticket scope to force here — WHICH apps are reachable is the
    workspace's own business (its Composio connections, plus any per-agent app
    assignment), enforced where the credential lives. What this does enforce is
    the shape: an action is required, and parameters the session put at the top
    level (the mistake every skill body invites) are folded into ``params``
    rather than silently dropped."""
    action = str(params.get("action") or "").strip()
    if not action:
        raise SessionToolRefused(
            "composio_execute needs an action, e.g. action=\"GMAIL_FETCH_EMAILS\". "
            "Your skill lists the ones it uses."
        )
    nested = params.get("params")
    inner: Dict[str, Any] = dict(nested) if isinstance(nested, Mapping) else {}
    for key, value in params.items():
        if key not in ("action", "params") and key not in inner:
            inner[key] = value
    return {"action": action, "params": inner}


def _scope_search(params: Dict[str, Any], ctx: SessionContext) -> Dict[str, Any]:
    return {k: v for k, v in params.items() if k in ("query", "limit")}


def _scope_record_memory(params: Dict[str, Any], ctx: SessionContext) -> Dict[str, Any]:
    """What a session may say about a fact it wants kept.

    The destination is NOT a parameter: the mission field is written whenever
    this ticket has one, resolved server-side. A session naming its own
    ``field_id`` would be able to write into another mission's memory.
    """
    content = str(params.get("content") or params.get("value") or "").strip()
    if not content:
        raise SessionToolRefused(
            "record_memory needs 'content' — the fact to keep, in one or two sentences."
        )
    scoped: Dict[str, Any] = {"content": content[:MAX_MEMORY_CHARS]}
    key = params.get("key")
    if key:
        scoped["key"] = str(key)[:MAX_MEMORY_KEY_CHARS]
    mem_type = params.get("type")
    if mem_type in MEMORY_TYPES:
        scoped["type"] = mem_type
    importance = params.get("importance")
    if importance is not None:
        try:
            scoped["importance"] = min(max(float(importance), 0.0), 1.0)
        except (TypeError, ValueError):
            pass
    return scoped


async def _run_record_memory(db: Any, params: Dict[str, Any], ctx: SessionContext) -> Dict[str, Any]:
    """Write one fact to durable workspace memory, and to the mission field too.

    Night 1 (2026-09-18): on a 9-task mission the API agent deposited 3 field
    memory points and the session agent deposited 0 — not because it chose not
    to, but because the bridge had a read path into memory and no write path at
    all. Sessions and API turns now leave the same trace.
    """
    from modules.tools.execution.unified_executor import UnifiedToolExecutor

    executor = UnifiedToolExecutor(db)
    agent_id = int(ctx.agent_id or 0)
    content = params.get("content", "")

    durable = await executor.execute_tool(
        tool_name=PLATFORM_DISPATCHER,
        parameters={
            "action": "platform_store_memory",
            "params": {
                "content": content,
                "type": params.get("type", "business_fact"),
                "importance": params.get("importance", 0.6),
                "source_type": "claude_reports",
            },
        },
        agent_id=agent_id,
        workspace_id=ctx.workspace_id,
        trace_id=f"session:{ctx.task_id}:record_memory:durable",
        caller_context=None,
    )
    durable_ok = bool(isinstance(durable, dict) and durable.get("success", True))

    field_ok = False
    field_error: Optional[str] = None
    if ctx.mission_field_id:
        key = params.get("key") or f"ticket-{ctx.task_id}"
        field = await executor.execute_tool(
            tool_name=PLATFORM_DISPATCHER,
            parameters={
                "action": "platform_field_inject",
                "params": {"key": str(key), "value": content},
            },
            agent_id=agent_id,
            workspace_id=ctx.workspace_id,
            trace_id=f"session:{ctx.task_id}:record_memory:field",
            caller_context={"field_context": {"field_id": ctx.mission_field_id}},
        )
        field_ok = bool(isinstance(field, dict) and field.get("success", False))
        if not field_ok and isinstance(field, dict):
            field_error = str(field.get("error") or "")[:200]

    if not durable_ok and not field_ok:
        detail = field_error or (durable.get("error") if isinstance(durable, dict) else "")
        return {"success": False, "error": f"nothing was recorded: {detail or 'memory is unavailable'}"}
    if not durable_ok:
        # F103: the mission field kept it, workspace memory did not — a lost
        # write is never reported as recorded.
        detail = (durable.get("error") if isinstance(durable, dict) else "") or "memory is unavailable"
        return {
            "success": False,
            "stored_durable": False,
            "stored_field": True,
            "error": f"workspace memory did NOT keep it ({detail}); only this mission's shared field has it",
        }

    if ctx.mission_field_id:
        where = "workspace memory and this mission's shared field" if field_ok else (
            "workspace memory (the mission field rejected it: %s)" % (field_error or "unavailable")
        )
    else:
        where = "workspace memory (this ticket is not part of a mission, so there is no shared field)"
    return {
        "success": True,
        "stored_durable": durable_ok,
        "stored_field": field_ok,
        "message": f"Recorded in {where}.",
    }


def _scope_read_step_file(params: Dict[str, Any], ctx: SessionContext) -> Dict[str, Any]:
    """An id from the ticket's list and nothing else: no path, no other field."""
    raw = str(params.get("file_id") or "").strip()
    try:
        return {"file_id": str(UUID(raw))}
    except ValueError:
        raise SessionToolRefused(
            "read_step_file takes the id your ticket lists beside the file "
            "(e.g. 125a3625-1351-435e-a128-6eca909f17b6), never a path or a name."
        ) from None


async def _run_read_step_file(db: Any, params: Dict[str, Any], ctx: SessionContext) -> Dict[str, Any]:
    """F161: read one file another step of this ticket's mission saved. The
    ticket names the mission; the id must be one its other steps registered.
    The read itself is the executor's own ``platform_get_deliverable``."""
    from config import config
    from modules.tools.execution.unified_executor import UnifiedToolExecutor
    from services.step_files import NOT_A_MISSION_STEP, files_for_ticket, not_listed, step_file_text

    files = files_for_ticket(db, ticket_id=ctx.task_id, workspace_id=ctx.workspace_id)
    if files is None:
        raise SessionToolRefused(NOT_A_MISSION_STEP)
    chosen = next((f for f in files if f.deliverable_id == params["file_id"]), None)
    if chosen is None:
        raise SessionToolRefused(not_listed(files))
    result = await UnifiedToolExecutor(db).execute_tool(
        tool_name=PLATFORM_DISPATCHER,
        parameters={
            "action": "platform_get_deliverable",
            "params": {"deliverable_id": chosen.deliverable_id, "include_content": True},
        },
        agent_id=int(ctx.agent_id or 0),
        workspace_id=ctx.workspace_id,
        trace_id=f"session:{ctx.task_id}:read_step_file",
        caller_context=None,
    )
    configured = int(getattr(config, "SESSION_STEP_FILE_MAX_CHARS", 0) or 0)
    ceiling = MAX_TOOL_RESULT_CHARS - STEP_FILE_FRAME_CHARS
    return step_file_text(chosen, result, min(configured, ceiling) if configured > 0 else ceiling)


class SessionToolRefused(Exception):
    """The call is outside what a session may ask for; the reason is for the model."""


SESSION_TOOLS: Tuple[SessionTool, ...] = (
    SessionTool(
        name="board_summary",
        action="platform_board_summary",
        description=(
            "Counts of this workspace's board tasks by status. Use it to orient before you "
            "report on the board, instead of guessing from files on disk."
        ),
        input_schema={"type": "object", "properties": {}, "required": []},
        scope=_scope_nothing,
        tags=("board",),
    ),
    SessionTool(
        name="list_tasks",
        action="platform_list_tasks",
        description=(
            "List this workspace's board tasks — id, title, status, priority, assigned agent. "
            "Filter by status or agent name. The result's 'total_matching' says how many match "
            "in all, so raise 'limit' when you need every one."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "status": {"type": "string", "description": "Only tasks in this status (omit for all)."},
                "assigned_agent_name": {"type": "string", "description": "Only tasks assigned to this agent."},
                "limit": {"type": "integer", "description": "Max results (default 20, max 200)."},
            },
            "required": [],
        },
        scope=_scope_list_tasks,
        project=_project_list_tasks,
        tags=("board",),
    ),
    SessionTool(
        name="update_ticket",
        action="platform_update_task_status",   # the historical name; the runner writes the note
        description=(
            "Leave a progress note on YOUR OWN ticket while you work — the operator sees it live on "
            "the ticket. You cannot move the ticket anywhere: closing it is the host's job when your "
            "turn ends, and any other status would end the run and throw your work away. If you "
            "genuinely cannot proceed without an answer, use ask_human instead."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "note": {"type": "string",
                         "description": "What you are doing, or what changed. One or two sentences."},
            },
            "required": ["note"],
        },
        scope=_scope_update_ticket,
        runner=_run_update_ticket,
        reads_only=False,
        tags=("board",),
    ),
    SessionTool(
        name="submit_report",
        action="platform_submit_report",
        description=(
            "File your report on this ticket: it lands in Deliverables and on the ticket, "
            "attributed to you. Markdown content. This is the record a reviewer reads — "
            "prefer it over leaving your findings only in the final message."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Short title."},
                "content": {"type": "string", "description": "The report, in markdown."},
                "summary": {"type": "string", "description": "One or two sentences for the card."},
                "report_type": {"type": "string",
                                "enum": ["standup", "research", "incident", "summary", "delivery", "audit"],
                                "description": "Category; 'delivery' for completed work, 'research' for a deep dive."},
                "recommendations": {"type": "array", "items": {"type": "string"}, "description": "What you advise."},
                "action_items": {"type": "array", "items": {"type": "string"}, "description": "What still needs doing."},
                # Accepted by the action and passed through by the scope, but not
                # advertised until RESEARCHER guessed a status on the 09-18 re-run
                # and was refused — a field the tool takes is a field it must name.
                "status": {"type": "string", "enum": ["ok", "warning", "critical", "info"],
                           "description": "How things stand: ok, warning, critical or info."},
                "metrics": {"type": "object", "description": "Numbers worth keeping, as key: value.",
                            "additionalProperties": True},
            },
            "required": ["title", "content"],
        },
        scope=_scope_submit_report,
        reads_only=False,
        tags=("report",),
    ),
    SessionTool(
        name="ask_human",
        action="platform_ask_human",
        description=(
            "Ask the operator ONE short question when you genuinely cannot proceed without "
            "an answer — a missing file, a decision only they can make. Your ticket parks "
            "when your turn ends and picks up here, with the answer, once they reply. So "
            "finish everything that does not depend on the answer FIRST, then ask and end "
            "your turn. Never wait, never guess, never ask twice. A sentence or two of "
            "markdown, with options when there are discrete choices."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "question": {"type": "string",
                             "description": "The question, in markdown. One decision, stated plainly."},
                "options": {"type": "array", "items": {"type": "string"},
                            "description": "The discrete choices, when there are some."},
            },
            "required": ["question"],
        },
        scope=_scope_ask_human,
        reads_only=False,
        tags=("ask",),
        runner=_run_ask_human,
    ),
    SessionTool(
        name="composio_execute",
        action="composio_execute",
        dispatch=DISPATCH_TOOL_NAME,
        description=(
            "Run one action on a connected app — Gmail, Google Calendar, Composio Search, "
            "GitHub, whatever this workspace has connected. Pass the action name and its "
            "parameters, exactly as your skills document them, e.g. "
            "action=\"GOOGLECALENDAR_FIND_EVENT\" with params={\"calendar_id\": \"primary\", "
            "\"time_min\": \"2026-09-18T00:00:00Z\"}. Automatos holds the credential and makes "
            "the call: you never see a key and never need one. An app this workspace has not "
            "connected is refused by name, so read the refusal rather than retrying."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string",
                           "description": "The action, e.g. GMAIL_FETCH_EMAILS or GOOGLECALENDAR_CREATE_EVENT."},
                "params": {"type": "object",
                           "description": "The action's own parameters, as your skill documents them."},
            },
            "required": ["action"],
        },
        scope=_scope_composio,
        reads_only=False,
        tags=("composio",),
    ),
    SessionTool(
        name="search_knowledge",
        action="platform_search_memory",
        description=(
            "Search this workspace's REMEMBERED FACTS — decisions, past findings, context "
            "agents and Auto have recorded. Search before re-deriving. This does not search "
            "uploaded documents: use search_documents for those."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for."},
                "limit": {"type": "integer", "description": "Max results (default 10, max 50)."},
            },
            "required": ["query"],
        },
        scope=_scope_search,
        tags=("knowledge",),
    ),
    SessionTool(
        name="search_documents",
        action="platform_search_documents",
        description=(
            "Ask a question of the documents uploaded to this workspace's knowledge base and "
            "get back the passages that answer it. Use this whenever the answer might be in "
            "the owner's own files — policies, specs, product data, briefs."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The question or topic to find."},
                "limit": {"type": "integer", "description": "Max passages (default 8, max 25)."},
            },
            "required": ["query"],
        },
        scope=_scope_search,
        tags=("knowledge", "documents"),
    ),
    SessionTool(
        name="record_memory",
        action="platform_store_memory",
        description=(
            "Record one fact worth keeping — a decision and its reason, a finding, a "
            "correction. It goes into this workspace's long-term memory, and when your "
            "ticket is part of a mission it is also shared with the other agents working "
            "that mission, in their own turn. Record what the next agent would otherwise "
            "have to re-derive; not raw tool output, and never a secret or credential."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The fact, in one or two sentences."},
                "key": {
                    "type": "string",
                    "description": "Short label for the mission field, e.g. 'pricing-decision'.",
                },
                "type": {
                    "type": "string",
                    "enum": list(MEMORY_TYPES),
                    "description": "What kind of fact this is. Default business_fact.",
                },
                "importance": {
                    "type": "number",
                    "description": "0.0-1.0 how load-bearing this is. Default 0.6.",
                },
            },
            "required": ["content"],
        },
        scope=_scope_record_memory,
        runner=_run_record_memory,
        reads_only=False,
        tags=("knowledge", "memory"),
    ),
    SessionTool(
        name="read_step_file",
        action="platform_get_deliverable",
        description=(
            "Read a file an earlier step of THIS mission saved. Your session opens only its own "
            "folder; your ticket lists the files the mission's other steps saved, each with an id. "
            "Pass that id, never a path: only those files can be read, read-only, and a long file "
            "comes back cut with a note saying where."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "file_id": {"type": "string",
                            "description": "The id your ticket lists beside the file."},
            },
            "required": ["file_id"],
        },
        scope=_scope_read_step_file,
        runner=_run_read_step_file,
        tags=("mission", "files"),
    ),
)

_BY_NAME: Mapping[str, SessionTool] = {t.name: t for t in SESSION_TOOLS}


def tool_names() -> Tuple[str, ...]:
    """The names a session may call — the claim payload, the host's policy, the
    agent form and the session prompt all read THIS."""
    return tuple(t.name for t in SESSION_TOOLS)


def get_tool(name: Any) -> Optional[SessionTool]:
    return _BY_NAME.get(str(name or ""))


def equivalent_of(mentioned: Any) -> Optional[str]:
    """The session tool that does what ``mentioned`` names, if any.

    A skill body says ``platform_submit_report`` because that is the API agents'
    action; in a session the same work is ``submit_report``. The table already
    holds both spellings, so the mapping is derived, never a second list to keep
    in step."""
    key = str(mentioned or "").strip()
    if not key:
        return None
    for tool in SESSION_TOOLS:
        if key in (tool.name, tool.action):
            return tool.name
    return None


def definitions() -> Tuple[Dict[str, Any], ...]:
    """The ``tools/list`` payload: stable order, stable text (prompt cache)."""
    return tuple(
        {"name": t.name, "description": t.description, "inputSchema": t.input_schema}
        for t in SESSION_TOOLS
    )


def resolve_parameters(tool: SessionTool, arguments: Any, ctx: SessionContext) -> Dict[str, Any]:
    """The parameters the action runs with: the session's, with this ticket's
    scope forced on. Raises :class:`SessionToolRefused` when the call asks for
    something a session may not have."""
    params = dict(arguments) if isinstance(arguments, Mapping) else {}
    return tool.scope(params, ctx) if tool.scope is not None else params


async def call_tool(db: Any, tool: SessionTool, params: Dict[str, Any], ctx: SessionContext) -> Dict[str, Any]:
    """Run one session tool through the API agents' own executor — the same
    construction the answered-grant resume uses (``api/approval_grants.py``).

    ``params`` are ALREADY scoped: :func:`resolve_parameters` runs on the wire
    (``services/session_tools_rpc.py``), so a refusal happens for every caller
    and no execution path can pass round the ticket's scope.

    Returns the executor's result dict. A tool-level failure is a RESULT, not an
    exception: the caller renders it as tool output the model can read and act
    on, which is what the MCP contract asks for.
    """
    # Every session tool is called BY an agent's ticket. Without an agent id the
    # executor is handed 0, and for ``composio_execute`` that is not a harmless
    # placeholder: an agent with no explicit app assignments inherits every app
    # the workspace has connected. A ticket with no assigned agent must not be
    # the widest caller on the platform.
    if not ctx.agent_id:
        raise SessionToolRefused(
            "this ticket has no agent assigned, so there is nothing to run tools as. "
            "Say so in your result and end your turn."
        )

    if tool.runner is not None:
        return _projected(tool, await tool.runner(db, params, ctx))

    from modules.tools.execution.unified_executor import UnifiedToolExecutor

    direct = tool.dispatch == DISPATCH_TOOL_NAME
    executor = UnifiedToolExecutor(db)
    result = await executor.execute_tool(
        tool_name=tool.action if direct else PLATFORM_DISPATCHER,
        parameters=params if direct else {"action": tool.action, "params": params},
        agent_id=int(ctx.agent_id or 0),
        workspace_id=ctx.workspace_id,
        trace_id=f"session:{ctx.task_id}:{tool.name}",
        # A session speaks as its AGENT, never as a human operator: no user
        # context, so an admin-gated action refuses exactly as it would for any
        # workspace agent. The ONE thing threaded is the mission field this
        # ticket belongs to — resolved server-side from the run, so a session
        # cannot name another mission's field (PRD-178 S1's rule, kept).
        caller_context=(
            {"field_context": {"field_id": ctx.mission_field_id}}
            if ctx.mission_field_id else None
        ),
    )
    if not isinstance(result, dict):
        return {"success": False, "error": "the executor returned no result"}
    return _projected(tool, result)


def _projected(tool: SessionTool, result: Any) -> Any:
    """The tool's response narrowed to what it advertises, if it narrows at all.
    Never raises: a projection fault must not turn a good answer into an error."""
    if tool.project is None or not isinstance(result, dict):
        return result
    try:
        return tool.project(result)
    except Exception:  # noqa: BLE001
        logger.warning("[session-tools] %s: could not narrow the response", tool.name, exc_info=True)
        return {"success": False, "error": "the result could not be prepared for a session"}
