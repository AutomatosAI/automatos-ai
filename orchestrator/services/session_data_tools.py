"""F329 (4 Oct): the owner's database and the Knowledge Graph, for a ticket session.

Smoke ticket #1995 (B6 → Business Analyst) ran as a Claude Code session in 47 s and
answered "There is no database query tool here": PRD-245's fixed session list
(``services/session_tools.py``) had no shop-database tool and no graph tool, so the
data nights could not run on session agents at all. API agents reach both through
the platform dispatcher; a session now reaches the same actions the same way.

* ``query_database`` runs ``platform_query_data``: the in-process NL2SQL path
  (``exec_research.run_nl2sql``) that resolves the source inside the caller's
  workspace, reads its schema metadata, and runs only SQL that passes
  ``SQLValidator`` (one SELECT, known tables, LIMIT injected and capped). The
  session sends a QUESTION, never SQL: there is no field that could carry a write.
* ``query_graph`` runs ``platform_query_graph`` for a question, and
  ``platform_graph_neighbors`` / ``platform_graph_path`` for one thing's links or
  the chain between two. All three are read-permission actions; the executor
  filters the graph to the CALLING agent's team (PRD-124).

Same three rules as the rest of the list: the text is fixed (prompt cache); scope
comes from the session, never the call (the executor is handed the ticket's
workspace and agent, and only the fields each schema declares are forwarded); and
every call goes through ``UnifiedToolExecutor.execute_tool`` via ``platform_execute``.

What comes back is what the actions return (the schema hints and graph direction
other fixes add travel with it), made to fit the bridge's result cap.

These are plain specs: ``session_tools`` builds its ``SessionTool`` rows from them,
so this module imports nothing from it at load time (it would be a cycle).
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover — annotations only
    from services.session_tools import SessionContext

QUERY_DATA_ACTION = "platform_query_data"
QUERY_GRAPH_ACTION = "platform_query_graph"
GRAPH_NEIGHBOURS_ACTION = "platform_graph_neighbors"
GRAPH_PATH_ACTION = "platform_graph_path"
# Room for the cut note and the JSON framing inside the bridge's own cap.
RESULT_FRAME_CHARS = 2000
# What a failed answer keeps beside its error: everything else is context
# (the database's schema, notes on its dates) the session needs to ask again.
FAILURE_ENVELOPE_KEYS: Tuple[str, ...] = ("success", "error", "tool", "hint", "detail")
CUT_NOTE = (
    "This answer was too long for one reply, so {lists} kept only the first rows. "
    "Ask a narrower question for the rest."
)
NO_RESULT = {"success": False, "error": "the executor returned no result"}

QUESTION_NEEDED = (
    "query_database needs a question in plain words, e.g. question=\"How many active "
    "subscribers are on the CLUB plan?\". It writes and runs the query itself."
)
GRAPH_ONE_FORM = (
    "query_graph takes ONE of: question (what the documents say about something), "
    "concept (everything directly linked to one thing), or from + to (the chain "
    "between two things)."
)
GRAPH_BOTH_ENDS = "query_graph needs both 'from' and 'to' to find the chain between two things."


def _bridge() -> Any:
    """The session-tools module, read at CALL time: it imports this one to build
    its list, so importing it at load time would be a cycle."""
    from services import session_tools

    return session_tools


def _refuse(reason: str) -> Exception:
    """The refusal the bridge renders as tool output the model reads."""
    return _bridge().SessionToolRefused(reason)


def _text(params: Dict[str, Any], *keys: str) -> str:
    """The first non-blank text the call gave under any of ``keys``."""
    for key in keys:
        value = params.get(key)
        if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value).strip():
            return str(value).strip()
    return ""


# ── what a session may ask ──────────────────────────────────────────────────

def scope_query_database(params: Dict[str, Any], ctx: "SessionContext") -> Dict[str, Any]:
    """The question, and which database when several are connected. Nothing
    else reaches the action: no SQL, no workspace, no agent (the executor is
    handed the ticket's own)."""
    question = _text(params, "question", "query")
    if not question:
        raise _refuse(QUESTION_NEEDED)
    scoped: Dict[str, Any] = {"question": question}
    database = _text(params, "database", "database_id")
    if database:
        scoped["database_id"] = database
    return scoped


def _graph_forms(params: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """Each lookup the call asked for, as (action, that action's params)."""
    forms: List[Tuple[str, Dict[str, Any]]] = []
    question = _text(params, "question")
    if question:
        forms.append((QUERY_GRAPH_ACTION, {"question": question}))
    concept = _text(params, "concept")
    if concept:
        relation = _text(params, "relation")
        forms.append((GRAPH_NEIGHBOURS_ACTION,
                      {"concept": concept, **({"relation_filter": relation} if relation else {})}))
    start, end = _text(params, "from"), _text(params, "to")
    if bool(start) != bool(end):
        raise _refuse(GRAPH_BOTH_ENDS)
    if start:
        forms.append((GRAPH_PATH_ACTION, {"source": start, "target": end}))
    return forms


def scope_query_graph(params: Dict[str, Any], ctx: "SessionContext") -> Dict[str, Any]:
    """Exactly one lookup, as the dispatcher's own ``{action, params}`` shape;
    the runner sends it as is. Undeclared fields are dropped."""
    forms = _graph_forms(params)
    if len(forms) != 1:
        raise _refuse(GRAPH_ONE_FORM)
    action, inner = forms[0]
    return {"action": action, "params": inner}


async def run_query_graph(db: Any, params: Dict[str, Any], ctx: "SessionContext") -> Dict[str, Any]:
    """The chosen graph action through the API agents' own dispatcher. A runner
    only because one tool reaches three actions; the path is ``call_tool``'s."""
    from modules.tools.execution.unified_executor import UnifiedToolExecutor

    result = await UnifiedToolExecutor(db).execute_tool(
        tool_name=_bridge().PLATFORM_DISPATCHER,
        parameters={"action": params["action"], "params": dict(params["params"])},
        agent_id=int(ctx.agent_id or 0),
        workspace_id=ctx.workspace_id,
        trace_id=f"session:{ctx.task_id}:query_graph",
        caller_context=None,
    )
    return result if isinstance(result, dict) else dict(NO_RESULT)


# ── what comes back ─────────────────────────────────────────────────────────

def _chars(value: Any) -> int:
    """Its length as the bridge writes it (``render_result``)."""
    return len(json.dumps(value, indent=2, default=str))


def _longest_list(result: Dict[str, Any]) -> Optional[str]:
    """The top-level list that takes the most room, if any is non-empty."""
    sizes = [(_chars(v), k) for k, v in result.items() if isinstance(v, list) and v]
    return max(sizes)[1] if sizes else None


def fit_result(result: Dict[str, Any], budget: int) -> Dict[str, Any]:
    """``result`` as plain JSON, halving its longest lists until it fits ``budget``,
    with a note naming what was cut. The bridge would otherwise cut the text
    mid-row; this keeps every key and whole rows."""
    out = json.loads(json.dumps(result, default=str))
    cut: List[str] = []
    while _chars(out) > budget:
        key = _longest_list(out)
        if key is None:
            break
        if key not in cut:
            cut.append(key)
        out = {**out, key: out[key][: len(out[key]) // 2]}
    if cut:
        out = {**out, "cut": CUT_NOTE.format(lists=", ".join(f"'{k}'" for k in cut))}
    return out


def _budget() -> int:
    return int(_bridge().MAX_TOOL_RESULT_CHARS) - RESULT_FRAME_CHARS


def project_answer(result: Dict[str, Any]) -> Dict[str, Any]:
    """An answer, fitted. A failure keeps its context (the schema, notes on the
    data's dates) as the hint, which the bridge shows under the error."""
    if result.get("success"):
        return fit_result(result, _budget())
    context = {k: v for k, v in result.items() if k not in FAILURE_ENVELOPE_KEYS and v not in (None, "", [], {})}
    if not context:
        return result
    shown = json.dumps(fit_result(context, _budget()), indent=2, default=str)
    hint = "\n".join(h for h in (str(result.get("hint") or ""), shown) if h)
    return {"success": False, "error": result.get("error"), "hint": hint}


# ── the two tools ───────────────────────────────────────────────────────────

QUERY_DATABASE_SPEC: Dict[str, Any] = {
    "name": "query_database",
    "action": QUERY_DATA_ACTION,
    "also_runs": ("smart_query_database",),
    "description": (
        "The owner's own business database (their shop's customers, orders, subscriptions, "
        "products, stock). Ask in plain words: pass the question and it writes and runs a "
        "read-only query, then answers with the rows, the query it ran and the tables' schema. "
        "Use it for every count, total, ranking or date: never guess a figure or ask the owner "
        "for a table name. Name a database only when the answer says several are connected."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "question": {"type": "string",
                         "description": "The question, in plain words, e.g. 'How many orders shipped last week?'."},
            "database": {"type": "string",
                         "description": "Which connected database, by name. Omit it when there is one."},
        },
        "required": ["question"],
    },
    "scope": scope_query_database,
    "project": project_answer,
    "tags": ("data", "database"),
}

QUERY_GRAPH_SPEC: Dict[str, Any] = {
    "name": "query_graph",
    "action": QUERY_GRAPH_ACTION,
    "also_runs": (GRAPH_NEIGHBOURS_ACTION, GRAPH_PATH_ACTION),
    "description": (
        "The business Knowledge Graph built from the owner's documents: who supplies, buys "
        "or is part of what, and which rules and processes connect. Pass ONE of: question "
        "(what the documents say about something), concept (everything directly linked to one "
        "thing, each link with its direction), or from + to (the chain between two things). "
        "It holds no live figures: counts, money and dates come from query_database."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "question": {"type": "string",
                         "description": "A question in plain words, e.g. 'who supplies our coffee?'."},
            "concept": {"type": "string",
                        "description": "One thing's name, to list everything directly linked to it."},
            "relation": {"type": "string",
                         "description": "With concept: only links of this kind, e.g. 'part_of'."},
            "from": {"type": "string", "description": "With to: the first of two things to connect."},
            "to": {"type": "string", "description": "With from: the second of two things to connect."},
        },
        "required": [],
    },
    "scope": scope_query_graph,
    "runner": run_query_graph,
    "project": project_answer,
    "tags": ("knowledge", "graph"),
}

# Appended to the fixed list in this order, after every earlier tool.
DATA_TOOL_SPECS: Tuple[Dict[str, Any], ...] = (QUERY_DATABASE_SPEC, QUERY_GRAPH_SPEC)
