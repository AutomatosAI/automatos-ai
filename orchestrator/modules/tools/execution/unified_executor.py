"""
Unified Tool Executor for PRD-17
=================================

Single entry point for all tool execution, routing to appropriate executors:
- Research tools (search_knowledge, semantic_search, search_codebase)
- File operations (read_file, write_file, list_directory)
- Shell commands (execute_command)

PRD-37: Added capability-based validation for Composio actions.
Enforces capability checks at EXECUTION time (defense in depth).

Executor methods are extracted into separate modules under
modules/tools/execution/exec_*.py for maintainability.
"""

import logging
import time as _time
from typing import Dict, Any, Optional
from uuid import UUID
from sqlalchemy.orm import Session

from modules.agents.services.agent_platform_tools import AgentPlatformTools
from modules.agents.services.agent_action_executor import ActionExecutor
from modules.tools.registry import ToolRegistry

# Extracted executor modules
from modules.tools.execution import exec_platform
from modules.tools.execution import exec_research
from modules.tools.execution import exec_file_ops
from modules.tools.execution import exec_shell
from modules.tools.execution import exec_composio
from modules.tools.execution import exec_document
from modules.tools.execution import exec_multimodal
from modules.tools.execution import exec_workspace
from modules.tools.execution import exec_planning
from modules.tools.execution.telemetry import fire_telemetry
from modules.memory.tool_outcome_capture import capture_tool_outcome
from core.observability.tracer import fire_tool_trace

# PRD-36: Composio Integration (lazy import to avoid startup overhead)
_composio_executor = None


def _get_composio_executor(db):
    """Lazy import of Composio executor."""
    global _composio_executor
    if _composio_executor is None:
        try:
            from core.composio.tool_executor import ComposioToolExecutor
            _composio_executor = ComposioToolExecutor
        except ImportError:
            return None
    return _composio_executor(db) if _composio_executor else None

logger = logging.getLogger(__name__)


class UnifiedToolExecutor:
    """
    Unified tool executor that routes tool calls to the appropriate executor.

    Provides a single interface for all tool execution, simplifying agent code
    and making it easier to add new tools.
    """

    def __init__(
        self,
        db_session: Session,
        workspace_dir: str = "/tmp/automatos_workspace",
        registry: Optional[ToolRegistry] = None,
    ):
        """
        Initialize unified tool executor.

        Args:
            db_session: Database session for this request
            workspace_dir: Directory for file operations
            registry: Optional shared ToolRegistry. If provided, used instead of lazy-loading.
        """
        self.db = db_session
        self.workspace_dir = workspace_dir
        self._tool_registry = registry  # Shared registry when provided; else lazy-load

        # Lazy-loaded executors (only initialize when needed)
        self._platform_tools = None  # For research tools (RAG, CodeGraph)
        self._action_executor = None  # For file/shell operations
        self._composio_executor = None  # PRD-36: Composio tools

        # Per-action Composio tool names (set by agent_factory after SDK schema fetch).
        # When the LLM calls e.g. COMPOSIO_SEARCH_WEB(query="..."), the executor
        # checks this dict to route it to the Composio executor.
        # Maps action_name -> app_name (e.g. "COMPOSIO_SEARCH_WEB" -> "COMPOSIO_SEARCH")
        self.composio_actions: dict = {}

        # Tool routing map -- delegates to extracted executor modules
        self.tool_routes = {
            # Research tools
            'search_knowledge': self._execute_platform_tool,
            'semantic_search': self._execute_platform_tool,
            'search_codebase': self._execute_platform_tool,
            'search_documents': self._execute_platform_tool,  # Alias
            'search_code': self._execute_platform_tool,  # Alias

            # Database tools (natural language SQL)
            'query_database': self._execute_database_tool,
            'smart_query_database': self._execute_smart_database_tool,

            # Multimodal search
            'search_multimodal': self._execute_multimodal_tool,
            'search_tables': self._execute_multimodal_tool,
            'search_images': self._execute_multimodal_tool,
            'search_formulas': self._execute_multimodal_tool,

            # File operations
            'read_file': self._execute_file_op,
            'write_file': self._execute_file_op,
            'list_directory': self._execute_file_op,
            'create_directory': self._execute_file_op,
            'delete_file': self._execute_file_op,

            # Shell commands
            'execute_command': self._execute_shell,

            # HTTP requests (internal API testing)
            'http_request': self._execute_http_request,

            # SSH remote execution
            'ssh_execute': self._execute_ssh,

            # Composio (external apps via DB cache + Composio OAuth)
            'composio_execute': self._execute_composio_execute,

            # PRD-63: Document generation (template-based)
            'generate_document': self._execute_generate_document,

            # PRD-22: Document creation tools (skill-based)
            'create_pdf': self._execute_document_tool,
            'create_docx': self._execute_document_tool,
            'create_xlsx': self._execute_document_tool,
            'create_pptx': self._execute_document_tool,

            # PRD-22 Expansion: Writing & Planning tools
            'create_implementation_plan': self._execute_planning_tool,
            'write_technical_content': self._execute_writing_tool,
            'refine_content': self._execute_writing_tool,

            # PRD-22 Expansion: Analysis tools
            'review_code': self._execute_analysis_tool,
            'security_scan': self._execute_analysis_tool,
            'generate_tests': self._execute_analysis_tool,
            'run_tests': self._execute_analysis_tool,
            'research_topic': self._execute_analysis_tool,
            'analyze_data': self._execute_analysis_tool,
            'write_document': self._execute_writing_tool,

            # PRD-008-A.2: widget UI affordances
            'widget_open_callback_form': self._execute_widget_callback,

            # PRD-36: Composio tools routed dynamically by prefix
        }

        logger.debug("UnifiedToolExecutor initialized (registry=%s)", "injected" if registry is not None else "lazy")

    # ------------------------------------------------------------------
    # Lazy properties
    # ------------------------------------------------------------------

    @property
    def composio_executor(self):
        """Lazy-load Composio executor (PRD-36) only when needed."""
        if self._composio_executor is None:
            logger.debug("  Initializing Composio executor...")
            self._composio_executor = _get_composio_executor(self.db)
        return self._composio_executor

    @property
    def platform_tools(self):
        """Lazy-load platform tools (RAG, CodeGraph) only when needed."""
        if self._platform_tools is None:
            logger.info("  Initializing research tools (RAG, CodeGraph)...")
            self._platform_tools = AgentPlatformTools(self.db)
        return self._platform_tools

    @property
    def action_executor(self):
        """Lazy-load action executor (file/shell ops) only when needed."""
        if self._action_executor is None:
            logger.debug("  Initializing file/shell executor...")
            self._action_executor = ActionExecutor(self.workspace_dir)
        return self._action_executor

    @property
    def tool_registry(self) -> ToolRegistry:
        """Use injected registry or lazy-load global singleton."""
        if self._tool_registry is None:
            from modules.tools.registry import get_tool_registry
            self._tool_registry = get_tool_registry(self.db)
        return self._tool_registry

    # ------------------------------------------------------------------
    # Policy plane chokepoint (PRD-174 W4)
    # ------------------------------------------------------------------

    def _policy_gate_check(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        *,
        agent_id: int,
        workspace_id: Optional[UUID],
        caller_context: Optional[Dict[str, Any]],
        trace: str,
    ) -> Optional[Dict[str, Any]]:
        """Evaluate one tool call through the unified PolicyGate.

        Returns ``None`` when execution may proceed (plane OFF, or an ``allow``
        verdict). Returns an errors-as-data result dict when the plane BLOCKS the
        call (deny/ask) — the caller returns it directly, so the tool never runs.

        Never raises: a fault in the plane must not wedge tool execution, so any
        error here is logged and treated as "proceed" (the downstream per-tool
        gates in ``platform_executor`` remain in force for platform actions).
        """
        try:
            from modules.policy import policy_plane_enabled

            if not policy_plane_enabled():
                return None  # flag OFF — byte-for-byte the legacy per-router gates

            from modules.policy import PolicyGate, ToolCall, Decision
            from modules.policy.errors import verdict_to_result

            # Resolve the effective action for the meta-dispatcher: platform_execute
            # nests the real action under "action" (params may be flat or wrapped).
            effective_name = tool_name
            effective_params = parameters if isinstance(parameters, dict) else {}
            if tool_name == "platform_execute" and isinstance(parameters, dict):
                action_name = (parameters.get("action") or "").strip()
                if action_name:
                    effective_name = action_name
                    effective_params = parameters.get("params") or {
                        k: v for k, v in parameters.items() if k not in ("action", "params")
                    }

            verdict = PolicyGate(self.db).check(
                ToolCall(
                    tool_name=effective_name,
                    parameters=effective_params,
                    workspace_id=workspace_id,
                    agent_id=agent_id,
                    caller_context=caller_context,
                )
            )

            # PRD-181 S1 (Art.12): fire the policy bus for EVERY verdict — allow,
            # ask, and deny — so the attached audit handler records every tool
            # call + policy decision per tenant. The bus is the single audit
            # write point (bus.py:18); this is the only place it is fired. A
            # handler fault is swallowed inside the bus, so audit never wedges
            # or slows the call. The risk tier is recomputed (pure, cheap) so the
            # audit row and the S5 approval card both carry it.
            self._fire_policy_bus(
                effective_name, effective_params, verdict,
                agent_id=agent_id, workspace_id=workspace_id,
                caller_context=caller_context, trace=trace,
            )

            if verdict.decision is Decision.ALLOW:
                return None
            logger.info(
                "[tool-trace %s] policy plane %s '%s': %s",
                trace, verdict.decision.value, effective_name, verdict.reason,
            )
            return verdict_to_result(verdict, tool_name)
        except Exception:
            logger.warning(
                "[tool-trace %s] policy gate errored for '%s' — proceeding "
                "(downstream gates still apply)", trace, tool_name, exc_info=True,
            )
            return None

    def _fire_policy_bus(
        self,
        effective_name: str,
        effective_params: Dict[str, Any],
        verdict: Any,
        *,
        agent_id: int,
        workspace_id: Optional[UUID],
        caller_context: Optional[Dict[str, Any]],
        trace: str,
    ) -> None:
        """Fire ``PRE_TOOL_USE`` on the policy bus with the verdict (PRD-181 S1).

        The attached audit handler reads ``ctx.data['verdict']`` and writes the
        per-tenant Art.12 record. Never raises: audit is a side-effect of the
        chokepoint, so a bus/handler fault must not block or slow the call.
        """
        try:
            from modules.policy import (
                Event,
                EventContext,
                classify_action,
                get_policy_bus,
            )

            # Recompute the risk tier (pure) so the audit row + S5 card carry it.
            risk = None
            try:
                action_def = self._policy_action_def(effective_name)
                permission_level = getattr(action_def, "permission_level", None)
                is_composio = (effective_name or "").startswith("composio_")
                risk = classify_action(
                    effective_name, permission_level=permission_level, is_composio=is_composio
                )
            except Exception:
                risk = None

            ctx = EventContext(
                workspace_id=workspace_id,
                agent_id=agent_id,
                tool_name=effective_name,
                tool_input=effective_params,
                caller_context=caller_context,
            )
            ctx.data["verdict"] = verdict
            ctx.data["risk"] = risk
            ctx.data["trace_id"] = trace
            get_policy_bus().fire(Event.PRE_TOOL_USE, ctx)
        except Exception:
            logger.warning(
                "[tool-trace %s] policy bus fire failed for '%s' — verdict "
                "still enforced, audit skipped for this call", trace, effective_name,
                exc_info=True,
            )

    def _policy_action_def(self, tool_name: str) -> Any:
        """Resolve the ActionDefinition for a tool (or None). Lazy + fail-open."""
        try:
            from modules.tools.discovery import get_action_registry

            return get_action_registry().get(tool_name)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Main dispatch
    # ------------------------------------------------------------------

    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        agent_id: int = 0,
        tenant_id: Optional[UUID] = None,
        workspace_id: Optional[UUID] = None,
        trace_id: Optional[str] = None,
        caller_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a tool by name, routing to the appropriate executor.

        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters
            agent_id: ID of the agent calling the tool
            tenant_id: UUID of the tenant (reserved for future use)
            workspace_id: UUID of the workspace for scoping
            trace_id: Optional trace ID for log correlation
            caller_context: Optional dict with keys user_id, system_role,
                workspace_role. Forwarded to PlatformActionExecutor for
                admin_only gating.  If None, executor falls back to
                workspace-scoped admin check.

        Returns:
            Tool execution result with standard format
        """
        _exec_start = _time.monotonic()
        result: Dict[str, Any] = {"success": False, "error": "Unknown dispatch failure", "tool": tool_name}
        try:
            trace = trace_id or "no-trace"
            logger.info(
                f"[tool-trace {trace}] Executing tool '{tool_name}' for agent={agent_id} "
                f"workspace={workspace_id}"
            )
            logger.info(f"[tool-trace {trace}] Parameters keys={list(parameters.keys()) if isinstance(parameters, dict) else type(parameters).__name__}")

            # PRD-174 W4 — the single policy chokepoint. When the plane is ON,
            # EVERY tool call (platform, workspace, Composio, registry) is
            # evaluated by one typed gate HERE, so Composio/workspace/registry
            # stop routing around the platform gate stack (F085/F060). A deny/ask
            # returns errors-as-data the model can read and never executes.
            # Flag OFF ⇒ this block is a no-op and behaviour is byte-for-byte the
            # per-router gates below.
            _policy_block = self._policy_gate_check(
                tool_name, parameters, agent_id=agent_id,
                workspace_id=workspace_id, caller_context=caller_context, trace=trace,
            )
            if _policy_block is not None:
                return _policy_block

            # PRD-64: Single dispatcher for platform actions
            if tool_name == "platform_execute":
                action_name = (parameters.get("action") or "").strip()
                action_params = parameters.get("params") or {}
                # LLMs often put params at top level instead of nested under "params"
                if not action_params:
                    action_params = {k: v for k, v in parameters.items() if k not in ("action", "params")}
                else:
                    # Merge any required keys the LLM placed at top level but omitted from params
                    for k, v in parameters.items():
                        if k not in ("action", "params") and k not in action_params:
                            action_params[k] = v
                if not action_name:
                    result = {"success": False, "error": "Missing required field: action", "tool": tool_name}
                    return result

                # PRD-143 S14: attach the recorded selection outcome for this
                # (workspace, agent) surface so the universal telemetry hook
                # persists it (router_decision->'selection'). hit = the chosen
                # action came from the narrowed enum; computed BEFORE registry
                # validation so enum-escaping hallucinations count as misses.
                # Best-effort: never blocks or fails the dispatch.
                try:
                    from modules.tools.discovery.signal_recorder import (
                        get_tool_signal_recorder,
                    )
                    _sel = get_tool_signal_recorder().peek_selection(
                        workspace_id=workspace_id, agent_id=agent_id
                    )
                    if _sel is not None:
                        caller_context = {
                            **(caller_context or {}),
                            "selection_outcome": {
                                "action": action_name,
                                "narrowed": _sel["narrowed"],
                                "hit": (action_name in _sel["allowed"]) if _sel["narrowed"] else None,
                                "enum_size": _sel.get("enum_size"),
                                "reason": _sel.get("reason"),
                            },
                        }
                except Exception as _sel_exc:
                    logger.debug(f"[tool-trace {trace}] selection telemetry skipped: {_sel_exc}")

                # Validate action exists in registry
                from modules.tools.discovery import get_action_registry
                registry = get_action_registry()
                action_def = registry.get(action_name)
                if not action_def:
                    available = [a.name for a in registry.get_all()]
                    result = {
                        "success": False,
                        "error": f"Unknown platform action: '{action_name}'. Use one of: {available[:20]}...",
                        "tool": tool_name,
                    }
                    return result

                # Validate required params
                required = action_def.parameters.get("required", [])
                missing = [p for p in required if p not in action_params]
                if missing:
                    # Include param descriptions so the LLM can self-correct
                    props = action_def.parameters.get("properties", {})
                    hints = [
                        f"  {p}: {props[p].get('description', props[p].get('type', '?'))}"
                        for p in missing if p in props
                    ]
                    hint_str = "\n".join(hints)
                    result = {
                        "success": False,
                        "error": (
                            f"Missing required params for '{action_name}': {missing}. "
                            f"Pass them inside params={{...}}.\n{hint_str}"
                        ),
                        "tool": tool_name,
                    }
                    return result

                logger.info(f"[tool-trace {trace}] platform_execute -> {action_name}")
                # Workspace actions registered in ActionRegistry need workspace routing
                if action_name.startswith("workspace_"):
                    result = await self._execute_workspace_action(
                        action_name, action_params, workspace_id=workspace_id, trace_id=trace,
                        agent_id=agent_id, caller_context=caller_context,
                    )
                    return result
                result = await self._execute_platform_action(
                    action_name, action_params, workspace_id=workspace_id, trace_id=trace,
                    caller_context=caller_context, agent_id=agent_id,
                )
                return result

            # PRD-64: Route platform_* actions to PlatformActionExecutor (direct calls)
            if tool_name.startswith("platform_"):
                logger.info(f"[tool-trace {trace}] Routing to PlatformActionExecutor: {tool_name}")
                result = await self._execute_platform_action(
                    tool_name, parameters, workspace_id=workspace_id, trace_id=trace,
                    caller_context=caller_context, agent_id=agent_id,
                )
                return result

            # Workspace tools: proxy to worker via WorkspaceClient
            if tool_name.startswith("workspace_"):
                logger.info(f"[tool-trace {trace}] Routing to WorkspaceClient: {tool_name}")
                result = await self._execute_workspace_action(
                    tool_name, parameters, workspace_id=workspace_id, trace_id=trace,
                    agent_id=agent_id, caller_context=caller_context,
                )
                return result

            # PRD-36: Route Composio per-action tools (SDK-provided schemas).
            # The LLM calls e.g. COMPOSIO_SEARCH_WEB(query="...") directly.
            # Parameters are flat -- no nested action/params wrapping.
            if tool_name in self.composio_actions:
                resolved_app = self.composio_actions[tool_name]
                logger.info(f"[tool-trace {trace}] Routing Composio per-action tool: {tool_name} (app={resolved_app})")
                result = await self._execute_composio_execute(
                    tool_name,
                    {"action": tool_name, "params": parameters, "app_name": resolved_app},
                    agent_id,
                    workspace_id=workspace_id,
                    trace_id=trace,
                )
                return result

            # Check if tool exists in registry
            tool_spec = self.tool_registry.get_tool(tool_name)
            if not tool_spec:
                result = {
                    "success": False,
                    "error": f"Unknown tool: {tool_name}",
                    "tool": tool_name,
                }
                return result

            # PRD-36: Legacy composio_execute meta-tool (fallback for older agents)
            if tool_name == "composio_execute":
                logger.info(f"[tool-trace {trace}] Routing to Composio executor: {tool_name}")
                result = await self._execute_composio_execute(
                    tool_name,
                    parameters,
                    agent_id,
                    workspace_id=workspace_id,
                    trace_id=trace,
                )
                return result

            if tool_spec.metadata and tool_spec.metadata.get("integration_type") == "composio":
                logger.info(f"[tool-trace {trace}] Routing to Composio executor: {tool_name}")
                result = await self._execute_composio_tool(
                    tool_spec,
                    parameters,
                    agent_id,
                    workspace_id,
                    trace_id=trace
                )
                return result

            # Route to appropriate executor
            executor_func = self.tool_routes.get(tool_name)
            if executor_func:
                # Some executors need workspace context (e.g. Composio).
                # Prefer passing workspace_id when supported, otherwise fallback.
                try:
                    result = await executor_func(
                        tool_name,
                        parameters,
                        agent_id,
                        workspace_id=workspace_id,
                        trace_id=trace,
                        caller_context=caller_context,
                    )
                except TypeError:
                    result = await executor_func(tool_name, parameters, agent_id)
                logger.info(f"  Tool '{tool_name}' executed successfully")
                return result
            else:
                result = {
                    "success": False,
                    "error": f"Unknown tool: {tool_name}",
                    "tool": tool_name,
                }
                return result

        except Exception as e:
            logger.error(f"[tool-trace {trace_id or 'no-trace'}] Tool execution failed: {tool_name} - {e}")
            result = {
                "success": False,
                "error": str(e),
                "tool": tool_name
            }
            return result
        finally:
            # PRD-139: Universal telemetry — fire-and-forget, never fails the tool call
            _exec_ms = int((_time.monotonic() - _exec_start) * 1000)
            fire_telemetry(
                self.db,
                tool_name=tool_name,
                parameters=parameters if isinstance(parameters, dict) else {},
                agent_id=agent_id,
                workspace_id=workspace_id,
                result=result,
                execution_time_ms=_exec_ms,
                caller_context=caller_context,
            )
            # PRD-159 S2: capture notable tool outcomes (failures + notable
            # successes) as typed tool_outcome memories — fire-and-forget,
            # content-hash deduped, noise-gated. Never fails the tool call.
            capture_tool_outcome(
                tool_name=tool_name,
                parameters=parameters if isinstance(parameters, dict) else {},
                result=result,
                workspace_id=workspace_id,
                agent_id=agent_id,
            )
            # PRD-185 S9: emit a vendor-neutral trace/score at the tool-dispatch
            # chokepoint (beside telemetry) — "was the tool call good" as a live
            # number over real traffic. Config-gated default-OFF (NoOp) + fully
            # guarded, so it never fails the tool call.
            _ok = bool(result.get("success", result.get("successful"))) if isinstance(result, dict) else False
            fire_tool_trace(
                tool_name=tool_name,
                success=_ok,
                duration_ms=_exec_ms,
                workspace_id=workspace_id,
                agent_id=agent_id,
                error=(result.get("error") if isinstance(result, dict) and not _ok else None),
            )

    # ------------------------------------------------------------------
    # Delegate methods -- thin wrappers calling extracted modules
    # ------------------------------------------------------------------

    async def _execute_platform_tool(self, tool_name, parameters, agent_id, **kw):
        return await exec_platform.execute_platform_tool(self, tool_name, parameters, agent_id)

    async def _execute_platform_action(self, tool_name, parameters, workspace_id=None, trace_id=None, caller_context=None, agent_id=None):
        return await exec_platform.execute_platform_action(
            self, tool_name, parameters,
            workspace_id=workspace_id, trace_id=trace_id, caller_context=caller_context,
            agent_id=agent_id,
        )

    async def _execute_database_tool(self, tool_name, parameters, agent_id, workspace_id=None, caller_context=None, **kw):
        return await exec_research.execute_database_tool(
            self, tool_name, parameters, agent_id,
            workspace_id=workspace_id, caller_context=caller_context,
        )

    async def _execute_smart_database_tool(self, tool_name, parameters, agent_id, workspace_id=None, caller_context=None, **kw):
        return await exec_research.execute_smart_database_tool(
            self, tool_name, parameters, agent_id,
            workspace_id=workspace_id, caller_context=caller_context,
        )

    async def _execute_multimodal_tool(self, tool_name, parameters, agent_id, workspace_id=None, **kw):
        return await exec_multimodal.execute_multimodal_tool(self, tool_name, parameters, agent_id, workspace_id=workspace_id)

    async def _execute_file_op(self, tool_name, parameters, agent_id, workspace_id=None, trace_id=None, caller_context=None, **kw):
        return await exec_file_ops.execute_file_op(
            self, tool_name, parameters, agent_id,
            workspace_id=workspace_id, trace_id=trace_id, caller_context=caller_context,
        )

    async def _execute_shell(self, tool_name, parameters, agent_id, **kw):
        return await exec_shell.execute_shell(self, tool_name, parameters, agent_id)

    async def _execute_http_request(self, tool_name, parameters, agent_id, workspace_id=None, trace_id=None):
        return await exec_shell.execute_http_request(self, tool_name, parameters, agent_id, workspace_id=workspace_id, trace_id=trace_id)

    async def _execute_ssh(self, tool_name, parameters, agent_id, workspace_id=None, trace_id=None):
        return await exec_shell.execute_ssh(self, tool_name, parameters, agent_id, workspace_id=workspace_id, trace_id=trace_id)

    async def _execute_composio_tool(self, tool_spec, parameters, agent_id, workspace_id, trace_id=None):
        return await exec_composio.execute_composio_tool(self, tool_spec, parameters, agent_id, workspace_id, trace_id=trace_id)

    async def _execute_composio_execute(self, tool_name, parameters, agent_id, workspace_id=None, trace_id=None):
        return await exec_composio.execute_composio_execute(self, tool_name, parameters, agent_id, workspace_id=workspace_id, trace_id=trace_id)

    async def _execute_composio_tool_router(self, tool_name, parameters, agent_id, workspace_id=None, trace_id=None):
        return await exec_composio.execute_composio_tool_router(self, tool_name, parameters, agent_id, workspace_id=workspace_id, trace_id=trace_id)

    async def _execute_generate_document(self, tool_name, parameters, agent_id, workspace_id=None, trace_id=None):
        return await exec_document.execute_generate_document(self, tool_name, parameters, agent_id, workspace_id=workspace_id, trace_id=trace_id)

    async def _execute_document_tool(self, tool_name, parameters, agent_id, **kw):
        return await exec_document.execute_document_tool(self, tool_name, parameters, agent_id)

    async def _execute_workspace_action(self, tool_name, parameters, workspace_id=None, trace_id=None, agent_id=None, caller_context=None):
        return await exec_workspace.execute_workspace_action(
            self, tool_name, parameters,
            workspace_id=workspace_id, trace_id=trace_id,
            agent_id=agent_id, caller_context=caller_context,
        )

    async def _execute_widget_callback(self, tool_name, parameters, agent_id, workspace_id=None, trace_id=None):
        from modules.tools.widget_callback import handle_widget_open_callback_form
        return await handle_widget_open_callback_form(
            tool_name, parameters,
            agent_id=agent_id, workspace_id=workspace_id, trace_id=trace_id,
        )

    async def _execute_planning_tool(self, tool_name, parameters, agent_id, **kw):
        return await exec_planning.execute_planning_tool(self, tool_name, parameters, agent_id)

    async def _execute_writing_tool(self, tool_name, parameters, agent_id, **kw):
        return await exec_planning.execute_writing_tool(self, tool_name, parameters, agent_id)

    async def _execute_analysis_tool(self, tool_name, parameters, agent_id, **kw):
        return await exec_planning.execute_analysis_tool(self, tool_name, parameters, agent_id)

    # ------------------------------------------------------------------
    # Tool discovery
    # ------------------------------------------------------------------

    def get_available_tools(self, categories: Optional[list] = None) -> list:
        """
        Get list of available tools, optionally filtered by category.

        Args:
            categories: Optional list of categories to filter by

        Returns:
            List of tool specifications
        """
        if categories:
            tools = []
            for category in categories:
                tools.extend(self.tool_registry.get_tools_by_category(category))
            return tools
        else:
            return list(self.tool_registry.tools.values())

    async def get_tools_for_agent(
        self,
        agent_id: int,
        tenant_id: UUID,
        include_core: bool = True
    ) -> list:
        """
        Get all tools available to an agent.

        Args:
            agent_id: ID of the agent
            tenant_id: UUID of the tenant (reserved for future use)
            include_core: Whether to include core platform tools

        Returns:
            List of tool specifications
        """
        tools = []

        # Add core platform tools
        if include_core:
            core_tools = self.get_available_tools()
            tools.extend(core_tools)

        return tools
