"""
Tool Router — Shared Tool Execution Layer
==========================================

Generic tool execution functions used by ALL consumers (chatbot, recipe,
trigger, API).  Promoted from consumers/chatbot/tool_router.py so that
non-chatbot consumers can import without cross-consumer dependencies.

Handles:
- Getting tools from modules.tools.ToolRegistry
- Executing tools via modules.tools.UnifiedToolExecutor
- Formatting results for frontend and LLM
- Capability-based action filtering (PRD-37)
- Execution-time validation (defense in depth)
"""

import asyncio
import concurrent.futures
import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Coroutine, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

# Import from submodules directly to avoid circular import
# (modules/tools/__init__.py imports from this file)
from modules.tools.registry import ToolCategory, get_tool_registry as registry_get_tool_registry
from modules.tools.execution import UnifiedToolExecutor
from modules.tools.formatting.result_formatter import ToolResultFormatter
from modules.tools.discovery.signal_recorder import ToolSignal, get_tool_signal_recorder
from core.database.database import SessionLocal

# Capability-based filtering imports (PRD-37)
try:
    from modules.tools.services.action_capability_filter import (
        ActionCapabilityFilter,
        get_action_capability_filter,
        ActionFilterResult,
    )
    CAPABILITY_FILTER_AVAILABLE = True
except ImportError:
    CAPABILITY_FILTER_AVAILABLE = False

logger = logging.getLogger(__name__)

# =============================================================================
# INTERNAL HELPERS
# =============================================================================


def _get_executor_for_request(db_session):
    """
    Create a new UnifiedToolExecutor per request using the shared registry.
    Avoids mutating a global executor's db_session (race conditions).
    Registry remains singleton; executors are per-request.
    """
    registry = registry_get_tool_registry()
    return UnifiedToolExecutor(db_session, registry=registry)


def _new_trace_id() -> str:
    return uuid4().hex[:12]


def _summarize_args(args: Any) -> str:
    if isinstance(args, dict):
        keys = list(args.keys())
        return f"dict keys={keys[:12]}{'...' if len(keys) > 12 else ''}"
    if isinstance(args, list):
        return f"list len={len(args)}"
    return f"{type(args).__name__}"


def _is_fatal_dependency_error(error: Optional[str]) -> bool:
    if not error:
        return False
    lowered = error.lower()
    return "composio openai sdk not available" in lowered or "composio-openai" in lowered


# =============================================================================
# PRD-138 US-009: Semantic narrowing of platform_execute action enum
# =============================================================================


def _semantic_routing_enabled() -> bool:
    """Read the SEMANTIC_TOOL_ROUTING flag from the canonical config singleton."""
    try:
        from config import config
        return bool(getattr(config, "SEMANTIC_TOOL_ROUTING", False))
    except Exception:
        return False


def _semantic_routing_top_k() -> int:
    """Configured top_k with safe default (matches PlatformActionsSection)."""
    try:
        from config import config
        return int(getattr(config, "SEMANTIC_TOOL_ROUTING_TOP_K", 15))
    except (ImportError, TypeError, ValueError):
        return 15


def _run_coroutine_blocking(coro: Coroutine) -> Any:
    """Run an async coroutine from a sync caller, even when an event loop is
    already running on this thread.

    get_tools_for_agent is sync (called both at module-load and from inside
    async chatbot/agent paths), but ActionSemanticIndex.rank_actions is
    async because the embedding manager is. When we're inside a running
    loop we can't ``asyncio.run`` directly, so we ship the coroutine to a
    helper thread that owns its own loop. Coroutines aren't bound to a
    specific loop until they're awaited, so this transfer is safe.
    """
    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        return asyncio.run(coro)


async def _rank_actions_for_dispatcher_async(
    query: str,
    top_k: int,
    exclude_admin: bool,
    exclude_promoted: bool,
    include_super_admin: bool = False,
) -> Optional[List[str]]:
    """Return the top-K action names for ``query`` from ActionSemanticIndex,
    or None on any failure (caller falls back to the full enum).

    Async-native: awaits rank_actions on the CALLER'S loop — no thread
    bridge, so a slow embedding upstream can neither freeze the event loop
    nor strand httpx clients on ephemeral loops. rank_actions itself bounds
    the live query embed (SEMANTIC_TOOL_ROUTING_EMBED_TIMEOUT_S) and returns
    [] on timeout, which lands in the full-enum fallback here.

    PRD-143: include_super_admin defaults False (fail-closed) — su actions
    are never ranked for an operator caller.

    Empty results also return None — the dispatcher's allowed_names=[] path
    falls back to the full enum, but routing through None is cleaner here:
    we never want a callable schema with zero actions, so we let the
    no-narrowing branch handle it instead of the empty-list defensive
    fallback in to_dispatcher_schema.
    """
    try:
        from modules.tools.discovery.action_semantic_index import (
            get_action_semantic_index,
        )
        index = get_action_semantic_index()
        ranked = await index.rank_actions(
            query,
            top_k=top_k,
            exclude_admin=exclude_admin,
            exclude_promoted=exclude_promoted,
            include_super_admin=include_super_admin,
        )
        if not ranked:
            return None
        return [name for name, _score in ranked]
    except Exception as exc:
        logger.warning(
            "_rank_actions_for_dispatcher failed (query=%r): %s — "
            "falling back to full enum",
            (query or "")[:80],
            exc,
        )
        return None


def _rank_actions_for_dispatcher(
    query: str,
    top_k: int,
    exclude_admin: bool,
    exclude_promoted: bool,
    include_super_admin: bool = False,
) -> Optional[List[str]]:
    """Sync compatibility entry — bridges to the async core.

    Only for genuinely-sync callers (module-load, scripts). Async code must
    await _rank_actions_for_dispatcher_async directly instead of paying the
    thread bridge.
    """
    try:
        return _run_coroutine_blocking(
            _rank_actions_for_dispatcher_async(
                query,
                top_k=top_k,
                exclude_admin=exclude_admin,
                exclude_promoted=exclude_promoted,
                include_super_admin=include_super_admin,
            )
        )
    except Exception as exc:
        logger.warning(
            "_rank_actions_for_dispatcher failed (query=%r): %s — "
            "falling back to full enum",
            (query or "")[:80],
            exc,
        )
        return None


def _narrow_dispatcher_actions_async_inputs(
    query: Optional[str],
) -> Optional[str]:
    """Shared gate for the narrowing paths: a reason string when narrowing
    must be skipped, None when ranking should run."""
    if not _semantic_routing_enabled():
        return "flag SEMANTIC_TOOL_ROUTING=False"
    if not query:
        return "no query supplied"
    return None


def _fallback_pins() -> List[str]:
    """The configured pin set for closed-pins fallback (CSV, whitespace-safe)."""
    try:
        from config import config
        raw = str(getattr(config, "TOOL_FALLBACK_PINS", "") or "")
    except Exception:
        raw = ""
    return [p.strip() for p in raw.split(",") if p.strip()]


def _fallback_mode_closed() -> bool:
    """True when TOOL_FALLBACK_MODE=closed-pins (PR-B; default open-full)."""
    try:
        from config import config
        return str(getattr(config, "TOOL_FALLBACK_MODE", "open-full")).strip().lower() == "closed-pins"
    except Exception:
        return False


def _fallback_narrowing(reason: str) -> Tuple[Optional[List[str]], Optional[str], bool]:
    """What ships when narrowing CAN'T decide (no query / rank fail / timeout).

    open-full (default): (None, reason, False) — today's fail-open full enum.
    closed-pins: the pin set, marked from_pins=True so the dispatcher build
    may admit promoted pins (platform_find_tools et al.) into the enum. A
    slow embed stops meaning maximum context.
    """
    if _fallback_mode_closed():
        pins = _fallback_pins()
        if pins:
            return pins, f"closed-pins ({reason})", True
    return None, reason, False


async def _narrow_dispatcher_actions_async(
    query: Optional[str],
    is_admin: bool,
    is_super_admin: bool,
) -> Tuple[Optional[List[str]], Optional[str], bool]:
    """Resolve (allowed_names, narrow_reason, from_pins) for the dispatcher
    enum — async-native (awaits ranking on the caller's loop).

    from_pins marks a closed-pins fallback surface: the allow-list may then
    contain promoted names that to_dispatcher_schema should admit. A flag-off
    operator choice is always honored as open-full.
    """
    skip_reason = _narrow_dispatcher_actions_async_inputs(query)
    if skip_reason is not None:
        if skip_reason.startswith("flag "):
            return None, skip_reason, False
        return _fallback_narrowing(skip_reason)
    allowed = await _rank_actions_for_dispatcher_async(
        query=query,
        top_k=_semantic_routing_top_k(),
        exclude_admin=not is_admin,
        exclude_promoted=True,
        include_super_admin=is_super_admin,
    )
    if allowed is None:
        return _fallback_narrowing("rank_actions returned empty or raised")
    return allowed, None, False


def _page_action_passes_gate(
    name: str, is_admin: bool, is_super_admin: bool
) -> bool:
    """True iff a page-manifest action clears the SAME role gate as ranking.

    Uses the ActionDefinition metadata directly (admin_only / super_admin_only),
    so page-prior exposure can never surface a gated tool to an unauthorized
    principal (PRD-143 fail-closed). Unknown/unregistered names are dropped.
    """
    try:
        from modules.tools.discovery import get_action_registry
        action = get_action_registry().get(name)
    except Exception:
        return False
    if action is None:
        return False
    if getattr(action, "super_admin_only", False) and not is_super_admin:
        return False
    if getattr(action, "admin_only", False) and not is_admin:
        return False
    return True


def _apply_page_prior(
    narrowing: Tuple[Optional[List[str]], Optional[str], bool],
    page_actions: Optional[List[str]],
    is_admin: bool,
    is_super_admin: bool,
) -> Tuple[Optional[List[str]], Optional[str], bool]:
    """Union gate-cleared page actions into the narrowed enum (PRD-221 S4).

    - No page actions, or a full-enum narrowing (allowed is None → every action
      already exposed): returned unchanged.
    - Otherwise the gate-passing page actions are appended to the ranked
      allow-list (dedup, order-stable) and the result is capped at
      ``TOOL_ROUTING_ENUM_CAP`` to bound prompt cost. from_pins passes through.
    """
    allowed, reason, from_pins = narrowing
    if not page_actions or allowed is None:
        return narrowing
    cleared = [
        n for n in page_actions
        if _page_action_passes_gate(n, is_admin, is_super_admin)
    ]
    if not cleared:
        return narrowing
    merged = list(allowed)
    seen = set(merged)
    for name in cleared:
        if name not in seen:
            merged.append(name)
            seen.add(name)
    try:
        from config import config
        cap = int(getattr(config, "TOOL_ROUTING_ENUM_CAP", 40))
    except (ImportError, TypeError, ValueError):
        cap = 40
    if cap > 0 and len(merged) > cap:
        merged = merged[:cap]
    return merged, (reason or "page_prior"), from_pins


def _narrow_dispatcher_actions_sync(
    query: Optional[str],
    is_admin: bool,
    is_super_admin: bool,
) -> Tuple[Optional[List[str]], Optional[str], bool]:
    """Sync twin of _narrow_dispatcher_actions_async (thread bridge)."""
    skip_reason = _narrow_dispatcher_actions_async_inputs(query)
    if skip_reason is not None:
        if skip_reason.startswith("flag "):
            return None, skip_reason, False
        return _fallback_narrowing(skip_reason)
    allowed = _rank_actions_for_dispatcher(
        query=query,
        top_k=_semantic_routing_top_k(),
        exclude_admin=not is_admin,
        exclude_promoted=True,
        include_super_admin=is_super_admin,
    )
    if allowed is None:
        return _fallback_narrowing("rank_actions returned empty or raised")
    return allowed, None, False


def _resolve_relative_date_window_utc(intent: str) -> Optional[tuple[datetime.date, datetime.date]]:
    """
    Resolve common relative date phrases to an (after_date, before_date) window in UTC.

    NOTE: before_date is exclusive (like Gmail's `before:`).
    """
    t = (intent or "").lower()
    if not t:
        return None

    today = datetime.now(timezone.utc).date()
    start_of_week = today - timedelta(days=today.weekday())  # Monday start (UTC)
    tomorrow = today + timedelta(days=1)

    # Order matters: match more specific phrases first.
    if "this week" in t:
        return (start_of_week, tomorrow)
    if "last week" in t or "previous week" in t:
        return (start_of_week - timedelta(days=7), start_of_week)
    if "yesterday" in t:
        return (today - timedelta(days=1), today)
    if "today" in t:
        return (today, tomorrow)
    if "past 7 days" in t or "last 7 days" in t:
        return (today - timedelta(days=7), tomorrow)

    return None


_AFTER_BEFORE_DATE_RE = re.compile(r"\b(after|before):\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", re.IGNORECASE)


def _rewrite_query_after_before(query: str, after_date: datetime.date, before_date: datetime.date) -> str:
    """
    Replace any existing after:/before: date filters in a query string and append the resolved window.
    Preserves other query terms.
    """
    q = (query or "").strip()
    q = _AFTER_BEFORE_DATE_RE.sub("", q)
    q = re.sub(r"\s+", " ", q).strip()
    window = f"after:{after_date:%Y/%m/%d} before:{before_date:%Y/%m/%d}"
    return f"{q} {window}".strip() if q else window


# =============================================================================
# PUBLIC: Get tools for an agent
# =============================================================================


def _resolve_workspace_id_from_agent(
    session_used,
    agent_id: Optional[int],
    workspace_id: Optional[Any],
    trace_id: str,
) -> Optional[Any]:
    """Resolve workspace_id from the agent row when not provided (needed for
    Composio gating and workspace-admin resolution)."""
    if agent_id is None or workspace_id is not None:
        return workspace_id
    try:
        from core.models import Agent as AgentModel
        agent_row = session_used.query(AgentModel).filter(AgentModel.id == agent_id).first()
        if agent_row and getattr(agent_row, "workspace_id", None):
            workspace_id = agent_row.workspace_id
            logger.info(f"[tool-trace {trace_id}] Resolved workspace_id from agent {agent_id} for tool list gating")
    except Exception as exc:
        logger.warning(f"[tool-trace {trace_id}] Failed to resolve workspace_id for agent {agent_id}: {exc}")
    return workspace_id


def _resolve_workspace_admin(
    session_used,
    workspace_id: Optional[Any],
    is_admin: bool,
    trace_id: str,
) -> bool:
    """PRD-122 fix: auto-resolve admin status from workspace owner role when
    no explicit is_admin was passed (heartbeat, agent factory paths).

    PRD-143: this fallback may flip is_admin ONLY — is_super_admin is never
    derived from workspace roles (su trap #2, PRD-143 §9)."""
    if is_admin or not workspace_id or session_used is None:
        return is_admin
    try:
        from core.workspaces.models import WorkspaceMember
        has_admin = (
            session_used.query(WorkspaceMember)
            .filter(
                WorkspaceMember.workspace_id == workspace_id,
                WorkspaceMember.role.in_(("owner", "admin")),
                WorkspaceMember.is_active.is_(True),
            )
            .first()
        )
        if has_admin:
            logger.info(f"[tool-trace {trace_id}] Workspace {workspace_id} has admin/owner — including admin tools")
            return True
    except Exception as exc:
        logger.debug(f"[tool-trace {trace_id}] Could not resolve workspace admin status: {exc}")
    return is_admin


def _get_tools_for_agent_core(
    *,
    agent_id: Optional[int],
    session_used,
    workspace_id: Optional[Any],
    is_admin: bool,
    is_super_admin: bool,
    query: Optional[str],
    narrowing: Tuple[Optional[List[str]], Optional[str]],
    trace_id: str,
    start_time: float,
) -> List[Dict[str, Any]]:
    """Shared body for get_tools_for_agent / get_tools_for_agent_async.

    Callers resolve workspace_id / is_admin and compute ``narrowing`` =
    (allowed_names, narrow_reason) FIRST — sync callers via the thread
    bridge, async callers by awaiting on their own loop — so this body
    stays loop-free. The session is owned (and closed) by the caller.
    """
    try:
        registry = registry_get_tool_registry(db_session=session_used)
        all_candidates = registry.get_all_tools(active_only=True)
        filtered_tools = all_candidates
        denied: List[Dict[str, Any]] = []

        if agent_id is not None:
            filtered_tools = []
            for tool in all_candidates:
                allowed, reason = registry.validate_tool_access(
                    agent_id=agent_id,
                    tool_name=tool.name,
                    db=session_used,
                    workspace_id=workspace_id
                )
                if allowed:
                    filtered_tools.append(tool)
                else:
                    denied.append({"tool": tool.name, "reason": reason})

        # Convert to OpenAI function format
        openai_tools = []
        for tool in filtered_tools:
            schema = tool.to_openai_format()

            # Enrich composio_execute with available actions from cache
            if tool.name == "composio_execute" and agent_id is not None:
                try:
                    from core.models.composio_cache import AgentAppAssignment, ComposioActionCache

                    # Get assigned apps for this agent
                    assignments = (
                        session_used.query(AgentAppAssignment)
                        .filter(
                            AgentAppAssignment.agent_id == agent_id,
                            AgentAppAssignment.is_active == True,
                            AgentAppAssignment.app_type == "EXTERNAL"
                        )
                        .all()
                    )

                    app_names = [a.app_name for a in assignments if a.app_name]

                    # Workspace inheritance: if no per-agent assignments,
                    # fall back to workspace-connected apps
                    if not app_names and workspace_id:
                        try:
                            from core.composio.entity_manager import EntityManager
                            manager = EntityManager(session_used)
                            entity = manager.get_entity_by_workspace(workspace_id)
                            if entity:
                                app_names = [
                                    (c.get("app_name") or "").upper()
                                    for c in manager.get_entity_connections(str(entity["id"]))
                                    if c.get("status") in ("active", "pending")
                                ]
                                if app_names:
                                    logger.info(f"[tool-trace {trace_id}] Agent {agent_id} inheriting {len(app_names)} workspace apps")
                        except Exception:
                            logger.warning(f"[tool-trace {trace_id}] Workspace app inheritance lookup failed")

                    if app_names:

                        # Get top actions for these apps from cache
                        actions = (
                            session_used.query(ComposioActionCache)
                            .filter(
                                ComposioActionCache.app_name.in_(app_names),
                            )
                            .order_by(ComposioActionCache.app_name, ComposioActionCache.display_name)
                            .limit(50)  # Top 50 actions across all assigned apps
                            .all()
                        )

                        if actions:
                            # Group by app
                            actions_by_app = {}
                            for action in actions:
                                app = action.app_name
                                if app not in actions_by_app:
                                    actions_by_app[app] = []

                                # Use display_name for readability, fallback to action_name
                                desc = action.description or ""
                                actions_by_app[app].append(f"  - {action.action_name}: {desc[:80]}")

                            # Build enriched description
                            action_list = []
                            for app in sorted(actions_by_app.keys()):
                                action_list.append(f"\n{app} actions:")
                                action_list.extend(actions_by_app[app][:10])  # Max 10 per app

                            enriched_desc = (
                                schema["description"] +
                                "\n\nAvailable actions (use exact action_name):" +
                                "\n".join(action_list)
                            )

                            schema["description"] = enriched_desc
                            logger.info(f"[tool-trace {trace_id}] Enriched composio_execute with {len(actions)} actions from {len(app_names)} apps")

                except Exception as e:
                    logger.warning(f"[tool-trace {trace_id}] Failed to enrich composio_execute: {e}")

            openai_tools.append({
                "type": "function",
                "function": schema
            })

        # PRD-64: Single dispatcher for platform actions (reduces 58 schemas → 1)
        # PRD-138 US-009: Narrow the dispatcher's action.enum to the top-K
        # semantically relevant actions when a query is supplied AND
        # SEMANTIC_TOOL_ROUTING is on. Closes the steering-vs-enforcement gap
        # from Phase 1 (prompt-text-only filtering) — same fallback pattern as
        # PlatformActionsSection.
        action_registry = None
        try:
            from modules.tools.discovery import get_action_registry
            action_registry = get_action_registry()

            # Narrowing was resolved by the public entry (sync bridge or
            # native await) BEFORE this loop-free body ran.
            allowed_names, narrow_reason, from_pins = narrowing

            dispatcher_schema = action_registry.to_dispatcher_schema(
                exclude_admin=not is_admin,
                exclude_promoted=True,  # promoted actions have first-class schemas below
                allowed_names=allowed_names,
                include_super_admin=is_super_admin,
                # closed-pins fallback pins include promoted names
                # (platform_find_tools et al.) — admit them into the enum.
                allow_promoted_in_allowlist=from_pins,
            )
            openai_tools.append(dispatcher_schema)
            all_actions = action_registry.get_all()
            dispatcher_count = len([a for a in all_actions if not a.promoted])

            if allowed_names is not None:
                enum_size = len(
                    dispatcher_schema["function"]["parameters"]
                    ["properties"]["action"].get("enum", [])
                )
                logger.info(
                    f"[tool-trace {trace_id}] dispatcher enum narrowed to "
                    f"{enum_size} actions via ActionSemanticIndex "
                    f"(query={(query or '')[:60]!r}, full={dispatcher_count})"
                )
            else:
                logger.info(
                    f"[tool-trace {trace_id}] dispatcher enum NOT narrowed: "
                    f"reason={narrow_reason}; full={dispatcher_count} actions"
                )

            # PRD-143 S14: persist the selection outcome instead of log-only —
            # counter + stash on the existing ToolSignalRecorder so the
            # platform_execute dispatch can attach hit/fallback telemetry.
            get_tool_signal_recorder().record_selection(
                workspace_id=workspace_id,
                agent_id=agent_id,
                narrowed=allowed_names is not None,
                reason=narrow_reason,
                allowed_names=allowed_names,
            )
        except Exception as e:
            logger.debug(f"[tool-trace {trace_id}] Platform actions unavailable: {e}")

        # PRD-122: First-class schemas for promoted actions.
        # Promoted actions get their own OpenAI tool schemas instead of
        # going through the platform_execute dispatcher — the LLM can call
        # them directly. The execution path at unified_executor.py routes
        # platform_* calls correctly regardless of how the schema was defined.
        try:
            if not action_registry:
                raise RuntimeError("action_registry not initialized")
            promoted_schemas = action_registry.to_first_class_schemas(
                exclude_admin=not is_admin,
                include_super_admin=is_super_admin,
            )
            openai_tools.extend(promoted_schemas)
            logger.info(f"[tool-trace {trace_id}] Added {len(promoted_schemas)} promoted action schemas")
        except Exception as e:
            logger.debug(f"[tool-trace {trace_id}] Promoted schemas unavailable: {e}")

        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info(
            f"[tool-trace {trace_id}] Loaded {len(openai_tools)} tools "
            f"(agent_id={agent_id}, denied={len(denied)}, "
            f"candidates={len(all_candidates)}, {elapsed_ms}ms)"
        )
        if denied:
            sample = denied[:10]
            logger.info(f"[tool-trace {trace_id}] Denied sample: {sample}")
        return openai_tools
    except Exception as e:
        logger.error(f"[tool-trace {trace_id}] Error loading tools from registry: {e}")
        return []


_GET_TOOLS_DOC = """
    Get tools from modules.tools.ToolRegistry in OpenAI function format.
    SINGLE SOURCE OF TRUTH — no duplicate definitions.

    Args:
        query: Optional natural-language query (typically the latest user
            turn). When provided AND SEMANTIC_TOOL_ROUTING is on, the
            platform_execute dispatcher's action.enum is narrowed via
            ActionSemanticIndex to the top-K most relevant actions —
            closes the prompt-vs-schema gap from PRD-138 Phase 1 (US-008).
            On any error, empty rank, or embed timeout
            (SEMANTIC_TOOL_ROUTING_EMBED_TIMEOUT_S), falls back to the
            full enum.
        is_super_admin: PRD-143 — True ONLY when the driving principal is
            literally system_role == 'super_admin'. Fail-closed default:
            super_admin_only actions are excluded from the dispatcher enum,
            first-class schemas, and semantic ranking. Unlike ``is_admin``,
            this is NEVER auto-resolved from workspace roles or autonomy.
    """


def get_tools_for_agent(
    agent_id: Optional[int] = None,
    db_session=None,
    workspace_id: Optional[Any] = None,
    is_admin: bool = False,
    query: Optional[str] = None,
    is_super_admin: bool = False,
) -> List[Dict[str, Any]]:
    session_used = db_session or SessionLocal()
    trace_id = _new_trace_id()
    start_time = time.time()
    try:
        workspace_id = _resolve_workspace_id_from_agent(session_used, agent_id, workspace_id, trace_id)
        is_admin = _resolve_workspace_admin(session_used, workspace_id, is_admin, trace_id)
        narrowing = _narrow_dispatcher_actions_sync(query, is_admin, is_super_admin)
        return _get_tools_for_agent_core(
            agent_id=agent_id,
            session_used=session_used,
            workspace_id=workspace_id,
            is_admin=is_admin,
            is_super_admin=is_super_admin,
            query=query,
            narrowing=narrowing,
            trace_id=trace_id,
            start_time=start_time,
        )
    except Exception as e:
        logger.error(f"[tool-trace {trace_id}] Error loading tools from registry: {e}")
        return []
    finally:
        if db_session is None:
            session_used.close()


async def _maybe_log_shadow_surface(
    query: Optional[str],
    is_admin: bool,
    is_super_admin: bool,
    shipped_tools: List[Dict[str, Any]],
    trace_id: str,
) -> None:
    """PR-C eval data (tool-surface review): log — never ship — what the
    relevance-gated surface WOULD have been for this turn.

    One line per turn: how many first-class promoted schemas would survive
    the hybrid cap vs the ~44 shipped today, and the floored enum size. The
    rank call shares the turn's deduped/cached query embed, so this costs a
    cosine pass, not a network call. Any failure is swallowed — shadow may
    never affect a live turn.
    """
    try:
        from config import config
        if not getattr(config, "TOOL_SURFACE_SHADOW", False) or not query:
            return
        from modules.tools.discovery import get_action_registry
        from modules.tools.discovery.action_semantic_index import (
            get_action_semantic_index,
        )

        ranked = await get_action_semantic_index().rank_actions(
            query=query,
            top_k=_semantic_routing_top_k(),
            exclude_admin=not is_admin,
            exclude_promoted=False,  # the would-be surface spans promoted too
            include_super_admin=is_super_admin,
        )
        registry = get_action_registry()
        cap = int(getattr(config, "TOOL_SURFACE_HYBRID_CAP", 6))
        would_first_class = [
            n for n, _ in ranked
            if getattr(registry.get(n), "promoted", False)
        ][: max(cap, 0)]
        logger.info(
            "[tool-shadow %s] shipped_tools=%d would_first_class=%d "
            "would_enum=%d names=%s",
            trace_id,
            len(shipped_tools),
            len(would_first_class),
            len(ranked),
            would_first_class,
        )
    except Exception:
        logger.debug("[tool-shadow %s] failed", trace_id, exc_info=True)


async def get_tools_for_agent_async(
    agent_id: Optional[int] = None,
    db_session=None,
    workspace_id: Optional[Any] = None,
    is_admin: bool = False,
    query: Optional[str] = None,
    is_super_admin: bool = False,
    page_actions: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    session_used = db_session or SessionLocal()
    trace_id = _new_trace_id()
    start_time = time.time()
    try:
        workspace_id = _resolve_workspace_id_from_agent(session_used, agent_id, workspace_id, trace_id)
        is_admin = _resolve_workspace_admin(session_used, workspace_id, is_admin, trace_id)
        narrowing = await _narrow_dispatcher_actions_async(query, is_admin, is_super_admin)
        # PRD-221 S4: fold the current page's manifest actions into the narrowed
        # enum so page-relevant tools survive even when the query ranks them out.
        # Gate-filtered by the SAME predicate as ranking — a manifest can never
        # expose an admin/su tool to an unauthorized principal.
        narrowing = _apply_page_prior(narrowing, page_actions, is_admin, is_super_admin)
        tools = _get_tools_for_agent_core(
            agent_id=agent_id,
            session_used=session_used,
            workspace_id=workspace_id,
            is_admin=is_admin,
            is_super_admin=is_super_admin,
            query=query,
            narrowing=narrowing,
            trace_id=trace_id,
            start_time=start_time,
        )
        tools = _apply_tier_exposure(session_used, workspace_id, tools, trace_id)
        await _maybe_log_shadow_surface(query, is_admin, is_super_admin, tools, trace_id)
        return tools
    except Exception as e:
        logger.error(f"[tool-trace {trace_id}] Error loading tools from registry: {e}")
        return []
    finally:
        if db_session is None:
            session_used.close()


def _apply_tier_exposure(
    session, workspace_id: Optional[Any], tools: List[Dict[str, Any]], trace_id: str
) -> List[Dict[str, Any]]:
    """Trim the assembled tool surface to the workspace tier's capability
    families (PRD-222 US-024). The single seam where per-turn platform tool
    schemas are gated by plan.

    FAIL-OPEN: a missing workspace_id, an unresolvable plan, or any error returns
    the surface unchanged — a lookup fault must never hide tools (the correct
    production posture, and it keeps the mock-based unit suite, which has no real
    workspace row, on the full surface). Only a CONFIRMED gated tier trims.
    """
    if not tools or workspace_id is None:
        return tools
    try:
        from core.models.workspaces import Workspace
        from services.plan_tiers import filter_tools_by_plan

        ws = session.query(Workspace).filter(Workspace.id == workspace_id).first()
        plan = getattr(ws, "plan", None)
        if not plan:
            return tools
        trimmed = filter_tools_by_plan(tools, plan)
        if len(trimmed) != len(tools):
            logger.info(
                f"[tool-trace {trace_id}] tier exposure: {len(tools)}→{len(trimmed)} "
                f"tool schemas for plan={plan} (families trimmed)"
            )
        return trimmed
    except Exception:
        logger.warning(
            f"[tool-trace {trace_id}] tier exposure filter failed — full surface kept",
            exc_info=True,
        )
        return tools


get_tools_for_agent.__doc__ = (
    _GET_TOOLS_DOC
    + """
    Sync entry — for genuinely-sync callers only (module-load, scripts).
    When invoked while an event loop is running, the semantic ranking pays
    a thread bridge. Async code must use get_tools_for_agent_async.
    """
)
get_tools_for_agent_async.__doc__ = (
    _GET_TOOLS_DOC
    + """
    Async-native entry — the hot chat/agent paths. The narrowing embed is
    awaited on the caller's loop (no thread bridge, no loop freeze) and is
    time-bounded so a degraded embedding upstream costs at most
    SEMANTIC_TOOL_ROUTING_EMBED_TIMEOUT_S before the full-enum fallback.
    """
)


# =============================================================================
# PUBLIC: Execute a tool
# =============================================================================


async def execute_tool(
    tool_name: str,
    tool_args: Dict[str, Any],
    agent_id: int = 1,
    workspace_id: Optional[UUID] = None,
    trace_id: Optional[str] = None,
    caller_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute a tool via modules.tools.UnifiedToolExecutor.
    SINGLE ENTRY POINT for all tool execution.

    Uses cached executor to avoid re-initializing tool registry on every call.
    """
    from core.database.database import SessionLocal

    trace = trace_id or _new_trace_id()
    db_session = SessionLocal()
    try:
        if workspace_id is None and agent_id:
            try:
                from core.models import Agent as AgentModel
                agent_row = db_session.query(AgentModel).filter(AgentModel.id == agent_id).first()
                if agent_row and agent_row.workspace_id:
                    workspace_id = agent_row.workspace_id
                    logger.info(f"[tool-trace {trace}] Resolved workspace_id from agent {agent_id}")
                else:
                    logger.warning(f"[tool-trace {trace}] Agent workspace_id missing for agent={agent_id}")
            except Exception as exc:
                logger.warning(f"[tool-trace {trace}] Failed to resolve workspace_id: {exc}")

        executor = _get_executor_for_request(db_session)

        logger.info(
            f"[tool-trace {trace}] execute_tool start tool={tool_name} "
            f"agent={agent_id} workspace={workspace_id} args={_summarize_args(tool_args)}"
        )
        result = await executor.execute_tool(
            tool_name,
            tool_args,
            agent_id,
            workspace_id=workspace_id,
            trace_id=trace,
            caller_context=caller_context,
        )
        db_session.commit()
        logger.info(
            f"[tool-trace {trace}] execute_tool done tool={tool_name} "
            f"success={bool(result.get('success'))}"
        )
        return result
    except Exception as e:
        db_session.rollback()
        logger.error(f"[tool-trace {trace}] execute_tool error tool={tool_name}: {e}")
        raise
    finally:
        db_session.close()


# =============================================================================
# PUBLIC: ToolRouter class — execute + format results
# =============================================================================


class ToolRouter:
    """
    Routes tool execution and formats results.
    Generic — no chatbot-specific logic.
    """

    def __init__(self):
        self.formatter = ToolResultFormatter

    def _filter_frontend_docs_by_scope(
        self,
        frontend_data: Dict[str, Any],
        agent_id: Optional[int],
        workspace_id: Optional[UUID],
    ) -> Dict[str, Any]:
        """PRD-157 S5: drop document widgets the answering agent can't access.

        Honest doc widgets: a scoped agent must not surface ``[View Document]``
        links for documents outside its workspace/team. Documents with no id are
        kept (they carry no working link to leak). Fail-open is avoided — any
        error simply leaves the data unchanged rather than dropping everything.
        """
        if not isinstance(frontend_data, dict):
            return frontend_data
        docs = frontend_data.get("documents")
        if not docs or workspace_id is None:
            return frontend_data
        doc_ids = [d.get("document_id") for d in docs if isinstance(d, dict) and d.get("document_id") is not None]
        if not doc_ids:
            return frontend_data
        try:
            from core.database.database import SessionLocal
            from modules.rag.retrieval_filters import build_retrieval_filters, allowed_document_ids
            from modules.tools.discovery.handlers_documents import _resolve_agent_team

            db = SessionLocal()
            try:
                team = _resolve_agent_team(db, agent_id)
                filters = build_retrieval_filters(workspace_id=str(workspace_id), team=team)
                allowed = allowed_document_ids(db, doc_ids, filters)
            finally:
                db.close()
        except Exception:
            logger.warning("[PRD-157 S5] doc-widget scope filter failed; leaving data unchanged", exc_info=True)
            return frontend_data

        kept = [
            d for d in docs
            if not (isinstance(d, dict) and d.get("document_id") is not None)
            or str(d.get("document_id")) in allowed
        ]
        if len(kept) != len(docs):
            logger.info(
                "[PRD-157 S5] suppressed %d out-of-scope document widget(s)",
                len(docs) - len(kept),
            )
            frontend_data = {**frontend_data, "documents": kept}
        return frontend_data

    async def execute_and_format(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        agent_id: int = 1,
        workspace_id: Optional[UUID] = None,
        original_intent: Optional[str] = None,
        caller_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a tool and return formatted results.

        Returns:
            {
                'success': bool,
                'frontend_data': dict,  # For artifact viewer
                'llm_context': str,     # For LLM injection
                'raw_result': dict      # Original result
            }
        """
        trace_id = _new_trace_id()
        logger.info(
            f"[tool-trace {trace_id}] ToolRouter execute_and_format "
            f"tool={tool_name} agent={agent_id} workspace={workspace_id} args={_summarize_args(tool_args)}"
        )

        try:
            # Deterministic guard: if the user asked for a relative date range,
            # and the tool call includes after:/before: date filters, rewrite them
            # based on current UTC dates to prevent stale/hallucinated years.
            if original_intent and tool_name == "composio_execute" and isinstance(tool_args, dict):
                try:
                    window = _resolve_relative_date_window_utc(original_intent)
                    if window:
                        after_d, before_d = window
                        params = tool_args.get("params") or tool_args.get("parameters")
                        if isinstance(params, dict):
                            q = params.get("query")
                            if isinstance(q, str):
                                params["query"] = _rewrite_query_after_before(q, after_d, before_d)
                except Exception as exc:
                    logger.debug(f"[tool-trace {trace_id}] Relative date rewrite skipped: {exc}")

            # Use validation wrapper for Composio tools to prevent calling unassigned actions
            is_composio = tool_name.startswith("composio_") or tool_name == "composio_execute"
            if is_composio and original_intent:
                result = await execute_tool_with_validation(
                    tool_name=tool_name,
                    tool_args=tool_args if isinstance(tool_args, dict) else {},
                    original_intent=original_intent,
                    agent_id=agent_id,
                    workspace_id=workspace_id,
                    allow_destructive=False,
                )
            else:
                result = await execute_tool(
                    tool_name,
                    tool_args if isinstance(tool_args, dict) else {},
                    agent_id,
                    workspace_id=workspace_id,
                    trace_id=trace_id,
                    caller_context=caller_context,
                )

            # Check success (support multiple executor result shapes)
            success = (
                bool(result.get("success"))
                or result.get("status") == "success"
                or bool(result.get("results"))
            )

            # PRD-141 US-019: fold this outcome into the routing graph via the
            # batched recorder (non-blocking enqueue; no DB / no task per call).
            self._record_tool_signal(
                tool_name, success, agent_id, workspace_id, caller_context, tool_args
            )

            if success:
                frontend_data = self.formatter.format_for_frontend(result, tool_name)
                # PRD-157 S5: suppress document widgets/links the answering agent
                # is not scoped to see (out-of-workspace or out-of-team).
                frontend_data = self._filter_frontend_docs_by_scope(
                    frontend_data, agent_id, workspace_id
                )
                llm_context = self.formatter.format_for_llm(result, tool_name)

                logger.info(f"[tool-trace {trace_id}] {tool_name} succeeded")
                logger.info(f"[tool-trace {trace_id}] frontend_data keys: {list(frontend_data.keys()) if frontend_data else 'EMPTY'}")
                return {
                    "success": True,
                    "frontend_data": frontend_data,
                    "llm_context": llm_context,
                    "raw_result": result,
                    "fatal_error": False,
                    "error_type": None,
                }

            # PRD-143 S15: a confirmation stop is not an opaque failure — the
            # executor's message (action + permission level) must reach the
            # LLM so Auto can relay the ask instead of "Unknown error".
            error = result.get("error") or (
                result.get("message") if result.get("requires_confirmation") else None
            ) or "Unknown error"
            error_type = result.get("error_type")
            fatal_error = bool(result.get("fatal")) or _is_fatal_dependency_error(error)
            llm_error = (
                "Tool execution failed due to a server configuration issue. Please restart the backend and try again."
                if fatal_error
                else error
            )
            logger.warning(f"[tool-trace {trace_id}] {tool_name} failed: {error}")
            # PRD-193 S3 (P2-12): an approval ask is not an opaque failure —
            # when the S1 gate attached a grant, emit the tool_approval card
            # payload so the HUMAN gets an affordance beside the S15 prose
            # (which stays the model's view, above). Best-effort: a formatter
            # fault never changes the envelope shape.
            ask_frontend_data: Dict[str, Any] = {}
            if result.get("requires_confirmation") and result.get("grant_id") is not None:
                try:
                    ask_frontend_data = self.formatter.format_for_frontend(result, tool_name)
                except Exception:
                    logger.debug(
                        f"[tool-trace {trace_id}] tool_approval card formatting skipped",
                        exc_info=True,
                    )
                    ask_frontend_data = {}
            envelope = {
                "success": False,
                "frontend_data": ask_frontend_data,
                "llm_context": f"Tool {tool_name} failed: {llm_error}",
                "raw_result": result,
                "fatal_error": fatal_error,
                "error_type": error_type,
            }
            # PRD-174 §4.2 — errors-as-data: give the model a stable
            # {code, message_for_model, remediation, retryable} block so it can
            # adapt (escalate / approve / drop a field) instead of seeing an
            # opaque failure. A policy deny already carries policy_error; this
            # backfills everything else. Additive + flag-gated (OFF = unchanged).
            return self._maybe_add_error_envelope(envelope, result)

        except Exception as e:
            error_msg = str(e)
            fatal_error = _is_fatal_dependency_error(error_msg)
            llm_error = (
                "Tool execution failed due to a server configuration issue. Please restart the backend and try again."
                if fatal_error
                else error_msg
            )
            logger.error(f"[tool-trace {trace_id}] {tool_name} exception: {error_msg}")
            # PRD-141 US-019: a thrown tool is a failure outcome too.
            self._record_tool_signal(
                tool_name, False, agent_id, workspace_id, caller_context, tool_args
            )
            envelope = {
                "success": False,
                "frontend_data": {},
                "llm_context": f"Tool {tool_name} error: {llm_error}",
                "raw_result": {"success": False, "error": error_msg},
                "fatal_error": fatal_error,
                "error_type": "dependency_missing" if fatal_error else None,
            }
            # PRD-174 §4.2 — errors-as-data (see above); backfill on the thrown path too.
            return self._maybe_add_error_envelope(envelope, {"success": False, "error": error_msg})

    @staticmethod
    def _maybe_add_error_envelope(
        envelope: Dict[str, Any], raw_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Backfill the errors-as-data ``policy_error`` block when the plane is on.

        Additive: the returned dict keeps every existing key; it only adds
        ``policy_error`` (a ``{code, message_for_model, remediation, retryable}``
        payload derived from the failing result). Flag OFF ⇒ returned unchanged,
        so today's callers see byte-for-byte the same shape. Never raises.
        """
        try:
            from modules.policy import policy_plane_enabled

            if not policy_plane_enabled():
                return envelope
            from modules.policy.errors import ensure_error_envelope

            # Preserve a policy_error the raw result already carries (a gate deny).
            source = dict(raw_result) if isinstance(raw_result, dict) else {}
            source.setdefault("success", False)
            enriched = ensure_error_envelope(source)
            if "policy_error" in enriched:
                envelope["policy_error"] = enriched["policy_error"]
        except Exception:
            logger.debug("[tool_router] error-envelope backfill skipped", exc_info=True)
        return envelope

    @staticmethod
    def _record_tool_signal(
        tool_name: str,
        success: bool,
        agent_id: Optional[int],
        workspace_id: Optional[UUID],
        caller_context: Optional[Dict[str, Any]],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Enqueue a routing-graph signal. Best-effort: never raises into the
        tool hot path. ``prior_action`` (the previous tool in the turn) is read
        from caller_context when the caller threads it through.

        PRD-177 S1 (F016): the batched recorder learns the RESOLVED composio
        action (SLACK_SEND_MESSAGE), not the collapsed ``composio_execute`` node
        — same resolution as the ToolExecutionLog path, so both learning paths
        agree on per-action nodes."""
        try:
            from modules.tools.execution.telemetry import resolve_action_name

            action_name = resolve_action_name(tool_name, parameters or {})
            prior_action = (
                caller_context.get("prior_action")
                if isinstance(caller_context, dict)
                else None
            )
            get_tool_signal_recorder().record(
                ToolSignal(
                    action_name=action_name,
                    success=bool(success),
                    agent_id=agent_id,
                    workspace_id=str(workspace_id) if workspace_id else None,
                    prior_action=prior_action,
                )
            )
        except Exception:
            pass  # telemetry is best-effort; never break a tool call

    def truncate_for_llm(
        self,
        result: Dict[str, Any],
        tool_name: str = "unknown",
        max_chars: int = 3000,
    ) -> str:
        """Truncate tool results for LLM context."""
        return self.formatter.format_for_llm(result, tool_name, max_chars)


# Module-level singleton
_tool_router: Optional[ToolRouter] = None


def get_tool_router() -> ToolRouter:
    """Get or create the global ToolRouter instance."""
    global _tool_router
    if _tool_router is None:
        _tool_router = ToolRouter()
    return _tool_router


# =============================================================================
# CAPABILITY-BASED FILTERING (PRD-37)
# =============================================================================


async def get_filtered_composio_actions(
    intent: str,
    enabled_apps: List[str],
    db_session=None,
    include_destructive: bool = False,
    max_actions: int = 15,
) -> Dict[str, Any]:
    """
    Get Composio actions filtered by capability for a given intent.

    This is the core of the capability-based filtering system:
    1. Extracts capabilities from intent text
    2. Filters actions from local metadata (no API calls)
    3. Excludes destructive actions unless explicitly allowed
    4. Returns ranked, relevant actions

    Args:
        intent: User's intent text (e.g., "send a message to #general")
        enabled_apps: List of Composio app IDs user has enabled
        db_session: Optional database session
        include_destructive: Whether to include destructive actions
        max_actions: Maximum number of actions to return

    Returns:
        Dict with filtered actions and metadata
    """
    if not CAPABILITY_FILTER_AVAILABLE:
        logger.warning("Capability filter not available, returning empty result")
        return {
            "success": False,
            "error": "Capability filter not available",
            "actions": [],
            "capabilities": []
        }

    trace_id = _new_trace_id()
    session_used = db_session or SessionLocal()

    try:
        filter_service = get_action_capability_filter(session_used)

        result = await filter_service.get_actions_for_intent(
            intent=intent,
            enabled_apps=enabled_apps,
            include_destructive=include_destructive,
            max_actions=max_actions
        )

        logger.info(
            f"[tool-trace {trace_id}] Capability filter: "
            f"intent='{intent[:50]}...' caps={result.extracted_capabilities} "
            f"matched={result.filtered_count}/{result.total_available} "
            f"fallback={result.fallback_used}"
        )

        # Convert to dict for API response
        actions_data = [
            {
                "action_id": a.action_id,
                "app_id": a.app_id,
                "capabilities": a.capabilities,
                "relevance_score": a.relevance_score,
                "description": a.description,
                "destructive": a.destructive,
                "requires_confirmation": a.requires_confirmation
            }
            for a in result.actions
        ]

        return {
            "success": True,
            "intent": result.intent,
            "extracted_capabilities": result.extracted_capabilities,
            "total_available": result.total_available,
            "filtered_count": result.filtered_count,
            "fallback_used": result.fallback_used,
            "actions": actions_data
        }

    except Exception as e:
        logger.error(f"[tool-trace {trace_id}] Capability filter error: {e}")
        return {
            "success": False,
            "error": str(e),
            "actions": [],
            "capabilities": []
        }
    finally:
        if db_session is None:
            session_used.close()


def _destructive_fail_closed(intent: Optional[str], allow_destructive: bool) -> bool:
    """Should an UNVERIFIABLE action be denied? (PRD-177 S3 / F018)

    True when the fail-closed flag is on, the caller has not explicitly
    authorized destructive actions, and the intent text reads as destructive.
    In that case an action we cannot classify (filter unavailable / errored /
    unsynced) must fail CLOSED rather than fail open.
    """
    if allow_destructive:
        return False
    try:
        from config import config

        if not bool(getattr(config, "COMPOSIO_DESTRUCTIVE_FAIL_CLOSED", True)):
            return False
    except Exception:
        pass  # default to fail-closed behavior if config can't be read
    try:
        from modules.tools.capabilities.taxonomy import intent_is_destructive

        return intent_is_destructive(intent)
    except Exception:
        return False


def validate_action_for_intent(
    action_id: str,
    intent: str,
    db_session=None,
    allow_destructive: bool = False,
) -> Tuple[bool, str]:
    """
    Validate that an action is eligible for execution given an intent.

    This is the EXECUTION-TIME validation gate (per GPT-5.2 recommendation).
    Called before executing any Composio action to ensure the action
    matches the original intent's capabilities.

    Args:
        action_id: The Composio action ID to validate
        intent: The original user intent
        db_session: Optional database session
        allow_destructive: Whether destructive actions are allowed

    Returns:
        (eligible, reason) tuple
    """
    if not CAPABILITY_FILTER_AVAILABLE:
        # PRD-177 S3 (F018): cannot classify — fail CLOSED for destructive intent.
        if _destructive_fail_closed(intent, allow_destructive):
            logger.warning(
                "Capability filter unavailable and intent reads destructive — "
                "failing CLOSED (confirmation required)"
            )
            return False, (
                "Cannot verify this action (capability filter unavailable) and "
                "the request looks destructive — confirmation required."
            )
        logger.warning("Capability filter not available, allowing non-destructive action")
        return True, "Capability filter not available (non-destructive, allowing)"

    session_used = db_session or SessionLocal()

    try:
        filter_service = get_action_capability_filter(session_used)

        eligible, reason = filter_service.check_action_eligibility(
            action_id=action_id,
            intent=intent,
            allow_destructive=allow_destructive
        )

        if not eligible:
            logger.warning(
                f"Action validation failed: action={action_id} "
                f"intent='{intent[:30]}...' reason={reason}"
            )

        return eligible, reason

    except Exception as e:
        # PRD-177 S3 (F018): a validation error means we cannot classify — fail
        # CLOSED for destructive intent rather than silently permitting damage.
        error_str = str(e)
        if "does not exist" in error_str or "UndefinedTable" in error_str:
            logger.debug(f"Action validation skipped (table not created): {e}")
        else:
            logger.warning(f"Action validation error: {e}")
        if _destructive_fail_closed(intent, allow_destructive):
            logger.warning(
                "Action validation errored and intent reads destructive — "
                "failing CLOSED (confirmation required)"
            )
            return False, (
                "Could not verify this action and the request looks destructive "
                "— confirmation required before running."
            )
        # Non-destructive intent: allow so a transient error doesn't block work.
        return True, "Validation skipped (non-destructive, allowing)"
    finally:
        if db_session is None:
            session_used.close()


async def execute_tool_with_validation(
    tool_name: str,
    tool_args: Dict[str, Any],
    original_intent: str,
    agent_id: int = 1,
    workspace_id: Optional[UUID] = None,
    allow_destructive: bool = False,
) -> Dict[str, Any]:
    """
    Execute a tool with intent-based validation.

    This wraps execute_tool() with an additional validation layer
    that checks if the action matches the original intent's capabilities.

    Args:
        tool_name: Tool to execute
        tool_args: Tool arguments
        original_intent: The original user intent (for validation)
        agent_id: Agent ID
        workspace_id: Workspace ID
        allow_destructive: Whether to allow destructive actions

    Returns:
        Tool execution result (with validation info)
    """
    trace_id = _new_trace_id()

    # Only validate Composio actions
    is_composio = tool_name.startswith("composio_") or tool_name == "composio_execute"

    if is_composio and original_intent:
        # Extract action ID from tool name or args
        action_id = None
        if tool_name.startswith("composio_"):
            action_id = tool_name.replace("composio_", "")
        elif isinstance(tool_args, dict):
            action_id = tool_args.get("action") or tool_args.get("action_name")

        if action_id:
            eligible, reason = validate_action_for_intent(
                action_id=action_id,
                intent=original_intent,
                allow_destructive=allow_destructive
            )

            if not eligible:
                logger.warning(
                    f"[tool-trace {trace_id}] Execution blocked by validation: "
                    f"tool={tool_name} action={action_id} reason={reason}"
                )
                return {
                    "success": False,
                    "error": f"Action not allowed for this intent: {reason}",
                    "error_type": "validation_blocked",
                    "blocked_by": "capability_validation",
                    "action_id": action_id,
                    "intent": original_intent[:100]
                }

    # Proceed with normal execution
    return await execute_tool(
        tool_name=tool_name,
        tool_args=tool_args,
        agent_id=agent_id,
        workspace_id=workspace_id,
        trace_id=trace_id
    )


def get_capability_filter_stats(db_session=None) -> Dict[str, Any]:
    """
    Get statistics about the capability filter system.

    Returns info about classified actions, coverage, etc.
    """
    if not CAPABILITY_FILTER_AVAILABLE:
        return {"available": False, "error": "Capability filter not available"}

    session_used = db_session or SessionLocal()

    try:
        filter_service = get_action_capability_filter(session_used)
        stats = filter_service.get_statistics()
        stats["available"] = True
        return stats
    except Exception as e:
        return {"available": False, "error": str(e)}
    finally:
        if db_session is None:
            session_used.close()
