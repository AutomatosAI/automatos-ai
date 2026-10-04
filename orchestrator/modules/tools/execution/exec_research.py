"""Database / research tool executors — workspace-scoped, in-process (PRD-160 S1).

The natural-language-to-SQL tools — ``query_database`` (execute_database_tool)
and ``smart_query_database`` (execute_smart_database_tool) — were turned OFF by
PRD-156 S3 because the old path executed raw LLM-generated SQL against the ENTIRE
main database with no workspace filter (a confirmed cross-tenant leak) and made
unauthenticated HTTP self-calls to the knowledge API.

PRD-160 S1 re-enables them as a first-class Auto tool, but *safely*:

  * In-process — no HTTP self-call. We call ``DatabaseKnowledgeService`` directly
    (the deleted ``query_main_database`` unscoped-SQL helper stays deleted).
  * Workspace-scoped — every call resolves a source *within the caller's
    workspace* (``resolve_source_id``) and threads ``workspace_id`` through to
    ``_get_source``, which fails closed on a cross-workspace source. An agent
    can only ever address sources by name inside its own workspace.
  * Fail-closed — a call with no ``workspace_id`` is refused outright; this is
    the defense-in-depth backstop if the path is reached without scope (e.g. a
    Playbook step).
"""
import functools
import logging
from dataclasses import dataclass, replace
from typing import Any, Awaitable, Callable, Dict, Optional

from modules.tools.execution.nl2sql_card_words import card_words

logger = logging.getLogger(__name__)


# F077 (A, refresh 4): the person's own words for this turn go to NL2SQL beside Auto's
# restatement. Auto turned "How many subscribers ... not counting cancelled ones?" into
# "How many ACTIVE subscribers ...", and the SQL counted active only (383, not 400).
OWNER_WORDS_CHARS = 2000


def owner_words(caller_context: Optional[Dict[str, Any]]) -> Optional[str]:
    """What the person typed this turn (``user_query``, set by the chat server-side;
    never a tool argument), bounded, or None for a lane nobody typed into."""
    if not isinstance(caller_context, dict):
        return None
    return str(caller_context.get("user_query") or "").strip()[:OWNER_WORDS_CHARS] or None


def _error(message: str, *, disabled: bool = False) -> Dict[str, Any]:
    """Structured, leak-free failure response matching the success shape."""
    resp: Dict[str, Any] = {
        "success": False,
        "error": message,
        "data": [],
        "columns": [],
        "row_count": 0,
    }
    if disabled:
        resp["disabled"] = True
    return resp


def _holds_no_caller_connection(
    fn: Callable[..., Awaitable[Dict[str, Any]]],
) -> Callable[..., Awaitable[Dict[str, Any]]]:
    """F330 (night 9c): the query runs holding none of the caller's connection.

    The query makes model calls lasting seconds and opens sessions of its own
    (``modules/nl2sql/service.py``: the source, the credentials, the examples).
    Night 9c's session tools passed the request's session in, and it stayed
    "idle in transaction" through all of that: 26 calls at once held the whole
    pool while each waited, on the event loop, for one more connection, and the
    process froze (``pool._do_get``; /health timed out).

    So the caller's transaction is ended first, if it has only read (one that
    wrote, locked or sent a NOTIFY is kept: ``core.database.read_release``, and
    then the caller's session is used, since only it can see what it wrote),
    and the query is run WITHOUT the caller's session: the source lookup then
    takes a short session of its own and closes it straight away, instead of
    re-opening the caller's transaction and keeping it for the whole query.
    """
    @functools.wraps(fn)
    async def wrapper(*, db_session: Optional[Any] = None, **kwargs: Any) -> Dict[str, Any]:
        released = False
        if db_session is not None:
            from core.database.read_release import release_if_read_only

            released = release_if_read_only(db_session)
        # A caller that has written keeps its transaction (and its connection) either
        # way, and only it can see what it wrote (the owner's card, F301): pass it on.
        return await fn(db_session=None if released or db_session is None else db_session, **kwargs)

    return wrapper


@_holds_no_caller_connection
async def run_nl2sql(
    *,
    method: str,
    parameters: Dict[str, Any],
    agent_id: int,
    workspace_id: Optional[Any],
    caller_context: Optional[Dict[str, Any]],
    db_session: Optional[Any] = None,
) -> Dict[str, Any]:
    """Shared in-process NL2SQL invocation for every database tool —
    ``query_database``, ``smart_query_database`` and ``platform_query_data``
    (F077): one service, one resolver, one audit row.

    ``method`` is ``"smart_query"`` (intelligent router) or ``"query_database"``
    (direct). Resolution and execution are workspace-scoped end to end;
    ``database_name`` may be a source's name or its id.

    The answer is plain JSON (F299: the agent lane serialises it with a bare
    ``json.dumps``, which raised on money, kilos and dates) and carries the
    database's schema (F300) — see ``modules.nl2sql.agent_answer``. On a board
    card the owner wrote, the card's words go to the SQL writer as the owner's
    own (F301: ``nl2sql_card_words``).
    """
    # Fail-closed: NL2SQL must never run without a workspace scope.
    if not workspace_id:
        logger.warning(
            "Agent %s invoked '%s' without a workspace scope — refused (PRD-160 S1)",
            agent_id,
            method,
        )
        return _error(
            "Natural-language database querying requires a workspace context; "
            "none was supplied."
        )

    query = (parameters or {}).get("query")
    if not query or not str(query).strip():
        return _error("A natural-language 'query' is required.")

    database_name = (parameters or {}).get("database_name")
    ws_id = str(workspace_id)

    from modules.nl2sql import get_database_knowledge_service

    service = get_database_knowledge_service()

    # Resolve the target source *within the caller's workspace*. Reuse the
    # executor's request session when present (one fewer pooled connection).
    source_id = await service.resolve_source_id(ws_id, database_name, db_session=db_session)
    if not source_id:
        return await _no_source(service, ws_id, database_name, db_session)

    from modules.nl2sql.agent_answer import ground_source, shape_answer

    schema = await ground_source(service, source_id, ws_id)
    call = NL2SQLCall(
        method=method, query=str(query), source_id=source_id, workspace_id=ws_id, agent_id=agent_id,
        user_id=str((caller_context or {}).get("user_id") or ""),
        owner_question=owner_words(caller_context) or await card_words(caller_context, ws_id, db_session),
    )
    result = await _ask(service, call)
    await _audit(service, call, result)
    result = await _past_the_data(service, call, result, schema, source_id)
    return shape_answer(result, schema, source_id)


async def _no_source(service: Any, ws_id: str, database_name: Any, db_session: Optional[Any]) -> Dict[str, Any]:
    """The failure for a call that names no source of this workspace (or several
    exist and none was named), listing the ones that do exist."""
    available = await _available_sources(service, ws_id, db_session)
    if database_name not in (None, ""):
        return _error(
            f"No active database source named '{str(database_name)[:100]}' is available "
            f"in this workspace.{available}"
        )
    return _error(
        "No database source is configured for this workspace, or several are "
        f"and none was named — pass 'database_name' to choose one.{available}"
    )


@dataclass(frozen=True)
class NL2SQLCall:
    """One database question, as the service is asked it."""

    method: str
    query: str
    source_id: str
    workspace_id: str
    agent_id: Any
    user_id: str
    owner_question: Optional[str]

    @property
    def agent(self) -> Optional[str]:
        """The asking agent's id as the service and the audit row take it."""
        return str(self.agent_id) if self.agent_id is not None else None


async def _ask(service: Any, call: NL2SQLCall) -> Dict[str, Any]:
    """Run the question through the service; a failure becomes a safe message."""
    try:
        if call.method == "smart_query":
            return await service.smart_query(
                source_id=call.source_id, text=call.query, user_id=call.user_id, agent_id=call.agent,
                workspace_id=call.workspace_id, owner_question=call.owner_question,
            )
        return await service.query_database(
            source_id=call.source_id, natural_language_query=call.query, user_id=call.user_id,
            agent_id=call.agent, workspace_id=call.workspace_id, owner_question=call.owner_question,
        )
    except Exception:  # noqa: BLE001 — logged; the agent gets a safe message, never internals
        logger.exception(
            "Agent %s NL2SQL '%s' failed (workspace=%s)", call.agent_id, call.method, call.workspace_id
        )
        return _error("Database query failed.")


async def _audit(service: Any, call: NL2SQLCall, result: Any) -> None:
    """PRD-160 S4: every NL query lands one audit row. Best-effort: a failed write
    is logged and never fails the answer."""
    try:
        await service.write_nl_audit(
            source_id=call.source_id,
            user_id=call.user_id or None,
            agent_id=call.agent,
            nl_query=call.query,
            result=result if isinstance(result, dict) else {},
        )
    except Exception:  # noqa: BLE001 — logged; the audit never decides the answer
        logger.exception("NL2SQL audit row not written for source %s", call.source_id)


async def _past_the_data(
    service: Any, call: NL2SQLCall, result: Any, schema: Dict[str, Any], source_id: str
) -> Any:
    """F301 B1 (build 14, #1895): an empty or all-zero answer whose query filters a date
    past the recorded data is not a count. The SQL writer is asked once more to count
    from the current state that decides it; that answer comes back saying how it was
    worked out, or, when it fails too, the agent gets no count and what to answer from.
    See ``modules.nl2sql.not_recorded``."""
    from modules.nl2sql import not_recorded
    from modules.nl2sql.schema.grounding import cached_facts

    facts = cached_facts(source_id)
    past = not_recorded.past_the_data(result, schema, facts)
    if not past:
        return result
    instruction = not_recorded.redirect_instruction(past, str(result.get("sql") or ""), schema, facts)
    again = replace(
        call,
        query=f"{call.query}\n\n{instruction}",
        owner_question=f"{call.owner_question}\n\n{instruction}" if call.owner_question else None,
    )
    second = await _ask(service, again)
    await _audit(service, again, second)
    if isinstance(second, dict) and second.get("success") and not not_recorded.past_the_data(second, schema, facts):
        return not_recorded.derived(second, past)
    return not_recorded.withheld(result, past, schema, facts)


async def _available_sources(service: Any, ws_id: str, db_session: Optional[Any]) -> str:
    """The ``Available: a (#36), b (#37).`` suffix, so the model can name a
    source instead of asking the owner which database. Best-effort: empty on
    any failure."""
    try:
        sources = await service.active_sources(ws_id, db_session=db_session)
    except Exception:  # noqa: BLE001 — the message is a courtesy, never the failure
        return ""
    if not sources:
        return ""
    return " Available: " + ", ".join(f"{name} (#{sid})" for sid, name in sources[:10]) + "."


async def execute_database_tool(
    executor,
    tool_name: str,
    parameters: Dict[str, Any],
    agent_id: int,
    workspace_id: Optional[Any] = None,
    caller_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """``query_database`` — direct NL→SQL, workspace-scoped, in-process (PRD-160 S1)."""
    return await run_nl2sql(
        method="query_database",
        parameters=parameters,
        agent_id=agent_id,
        workspace_id=workspace_id,
        caller_context=caller_context,
        db_session=getattr(executor, "db", None),
    )


async def execute_smart_database_tool(
    executor,
    tool_name: str,
    parameters: Dict[str, Any],
    agent_id: int,
    workspace_id: Optional[Any] = None,
    caller_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """``smart_query_database`` — intelligent NL→SQL/analysis router, workspace-scoped,
    in-process (PRD-160 S1)."""
    return await run_nl2sql(
        method="smart_query",
        parameters=parameters,
        agent_id=agent_id,
        workspace_id=workspace_id,
        caller_context=caller_context,
        db_session=getattr(executor, "db", None),
    )
