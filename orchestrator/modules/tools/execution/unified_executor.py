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

import json
import logging
import time as _time
from typing import Dict, Any, List, Optional, Tuple
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
from modules.tools.execution.telemetry import fire_telemetry, fire_tool_gap
from modules.memory.tool_outcome_capture import capture_tool_outcome
from core.observability.tracer import fire_tool_trace
from core.database.session_health import rollback_if_aborted

# F088 (night 3): names a model reaches for that are not tools. Auto called
# `search_documents` four times and got "Unknown tool" each time — the registry
# check ran before the old alias entry could — and told the owner the product's
# document search was broken. Resolved before anything gates or routes the call.
TOOL_ALIASES: Dict[str, str] = {
    "search_documents": "platform_search_documents",   # searches the uploads and names the file
    "search_code": "search_codebase",
}

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


# Names a model reaches for when it means one of our required params. Only
# consulted for a param that is MISSING — never to override what was sent.
_PARAM_ALIASES: Dict[str, Tuple[str, ...]] = {
    "title": ("name", "heading", "subject", "report_title", "task_title"),
    # The reverse direction too: platform_create_agent requires `name` and was
    # failing 6/6 with "Missing required parameter: name" from callers that sent
    # `title` or `agent_name` (surfaced by the always-failing-actions check).
    "name": ("title", "agent_name", "label", "display_name"),
    "content": ("body", "markdown", "text", "details", "report_content", "message", "findings"),
    "query": ("q", "search", "question", "prompt"),
    "description": ("desc", "summary", "details"),
    # F027-C (night 3): the playbook-step actions. add_playbook_step's prompt
    # arrived under another name 5 times; update_playbook_step's playbook and
    # step 3 times. ("order" is NOT an alias of step_index: it is update's own
    # parameter — the step's new position.)
    "prompt_template": ("prompt", "template", "instructions", "instruction", "step_prompt", "prompt_text",
                        "text", "content"),
    "playbook_id": ("recipe_id", "workflow_id", "playbookId"),
    "step_index": ("step", "index", "step_number", "step_idx", "stepIndex", "step_position"),
}

# A single dict argument named after the thing itself ({"report": {...}}) is a
# wrapper the model added, not a parameter.
_WRAPPER_KEYS: Tuple[str, ...] = ("report", "task", "document", "payload", "params", "input", "data")


def _fill_required_from_aliases(params: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
    """``params`` with any missing required key filled from an obvious synonym.

    Returns a new dict — the caller's params are never mutated.
    """
    if not required or not isinstance(params, dict):
        return params

    filled = dict(params)

    # Unwrap {"report": {...}} style wrappers first, without losing siblings.
    for key in _WRAPPER_KEYS:
        inner = filled.get(key)
        if isinstance(inner, dict) and any(r in inner for r in required):
            filled = {**inner, **{k: v for k, v in filled.items() if k != key}}
            break

    for name in required:
        if name in filled and filled[name] not in (None, ""):
            continue
        for alias in _PARAM_ALIASES.get(name, ()):
            value = filled.get(alias)
            if value not in (None, "", [], {}):
                filled[name] = value
                break
    return filled


def _placeholder(prop: Dict[str, Any]) -> str:
    """How a value of this schema type is written in the example call."""
    if prop.get("enum"):
        return json.dumps("<one of: " + " | ".join(str(v) for v in prop["enum"]) + ">")
    kind = prop.get("type")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), "string")
    return {"string": '"<string>"', "integer": "<integer>", "number": "<number>", "boolean": "<true|false>",
            "array": "[...]", "object": "{...}"}.get(kind, "<value>")


# F077 (refresh 4): after this refusal Auto asked the owner to "confirm the exact
# question" instead of making the call it was shown. A refused call is Auto's to fix,
# with the values it has. A value only the user can give is still theirs to give, and
# never invented (review LOW).
REFUSED_CALL_IS_YOURS = (
    "Make this call yourself now with the values you have. Ask the user only for a value only they can "
    "give, and never invent one: a refused call is never a question for the user otherwise."
)


def missing_params_error(action_name: str, schema: Dict[str, Any], missing: List[str], sent: Any) -> str:
    """F027-C (night 3): the call to make, spelled out. 69 of the night's 98
    failed platform_execute calls were "Missing required params" — models
    (gemini-2.5-flash: 10 output tokens each) sent params={} and retried the
    same after a hint. The error now names the exact call, every required key
    with its type, from the action's own schema."""
    props = schema.get("properties") or {}
    example = ", ".join(f'"{key}": {_placeholder(props.get(key) or {})}' for key in (schema.get("required") or []))
    lines = [
        f"Missing required params for '{action_name}': {missing}. Pass them inside params={{...}}.",
        f'Call it exactly like this: {{"action": "{action_name}", "params": {{{example}}}}}',
        REFUSED_CALL_IS_YOURS,
    ]
    if not sent:
        lines.append("Your params was empty — the values go inside it.")
    lines += [f"  {p}: {props[p].get('description', props[p].get('type', '?'))}" for p in missing if p in props]
    return "\n".join(lines)


# F182: how much of a refused call the corrected example repeats.
EXAMPLE_CALL_CHARS = 400
# F182: how close a refused key must be to a declared one to be named as it.
CLOSE_KEY_CUTOFF = 0.75


def undeclared_params(action_def: Any, params: Dict[str, Any]) -> List[str]:
    """F182 (night 6): the keys in ``params`` that the action would drop.

    The keys it takes are the schema's, the undeclared ones its handler reads
    (``accepts``), and any known other name of a declared one (``_PARAM_ALIASES``,
    which ``_fill_required_from_aliases`` and ``map_optional_aliases`` fill from).
    Server keys (a leading ``_``) and empty values are not counted, since
    dropping them loses nothing.
    """
    if not isinstance(params, dict):
        return []
    schema = action_def.parameters or {}
    taken = set(schema.get("properties") or {}) | set(getattr(action_def, "accepts", ()) or ())
    for name in schema.get("properties") or {}:
        taken.update(_PARAM_ALIASES.get(name, ()))
    return [key for key, value in params.items()
            if key not in taken and not str(key).startswith("_") and value not in (None, "", [], {})]


def _where_it_goes(
    key: str, value: Any, props: Dict[str, Any], misplaced: Dict[str, str],
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """What the refusal says about one key, and the params that carry its value
    instead (None when nothing in this action does)."""
    import difflib

    inner = {k: v for k, v in value.items() if k in props} if isinstance(value, dict) else {}
    if inner:
        return f"its fields {list(inner)} go straight into params, not inside '{key}'.", inner
    target = misplaced.get(key)
    if target in props:
        return f"put these values under '{target}'.", {target: value}
    if target:
        return target, None
    stray = next((k for k in value if k in misplaced and misplaced[k] not in props), None) \
        if isinstance(value, dict) else None
    if stray:
        return f"inside it, '{stray}': {misplaced[stray]}", None
    close = difflib.get_close_matches(str(key), list(props), n=1, cutoff=CLOSE_KEY_CUTOFF)
    if close:
        return f"did you mean '{close[0]}'?", {close[0]: value}
    return "not a parameter of this action.", None


# F182: the two ways a model calls a platform action, as the refusal log names them.
VIA_DISPATCHER = "platform_execute"
VIA_DIRECT_CALL = "direct call"


def unknown_params_error(action_name: str, action_def: Any, unknown: List[str], sent: Dict[str, Any],
                         via: str = VIA_DISPATCHER) -> str:
    """F182: the refusal names each key the action does not take, where it goes,
    and the call with it moved there, in the form the caller used. Nothing ran."""
    schema = action_def.parameters or {}
    props = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    misplaced = getattr(action_def, "misplaced", None) or {}
    lines = [f"'{action_name}' does not take {unknown}, so nothing was done. Fix the call and send it again:"]
    kept = {k: v for k, v in sent.items() if k not in unknown}
    fixed = kept
    for key in unknown:
        where, moved = _where_it_goes(key, sent[key], props, misplaced)
        lines.append(f"  {key}: {where}")
        fixed = {**fixed, **(moved or {})}
    if via == VIA_DISPATCHER:
        example = json.dumps({"action": action_name, "params": fixed}, default=str, ensure_ascii=False)
    else:
        example = f"{action_name}({json.dumps(fixed, default=str, ensure_ascii=False)})"
    if fixed != kept and len(example) <= EXAMPLE_CALL_CHARS:
        lines.append(f"Call it like this: {example}")
    takes = ", ".join(f"{name} ({(props[name] or {}).get('type', 'any')}{', required' if name in required else ''})"
                      for name in props)
    lines.append(f"'{action_name}' takes: {takes or 'no parameters'}.")
    return "\n".join(lines)


def map_optional_aliases(action_name: str, action_def: Any, params: Dict[str, Any], trace: str,
                         via: str = VIA_DISPATCHER) -> Dict[str, Any]:
    """F182: an optional param sent under a known other name (create_agent's
    "desc") is kept under its declared name, not refused and not dropped; each
    mapping is logged. The required ones are ``_fill_required_from_aliases``'s.
    A key the action itself declares is never taken for another. Returns new
    params; the caller's are not changed."""
    if not isinstance(params, dict):
        return params
    schema = action_def.parameters or {}
    props, required = schema.get("properties") or {}, set(schema.get("required") or [])
    mapped = dict(params)
    for name in props:
        if name in required or mapped.get(name) not in (None, ""):
            continue
        for alias in _PARAM_ALIASES.get(name, ()):
            if alias in props or mapped.get(alias) in (None, "", [], {}):
                continue
            mapped[name] = mapped.pop(alias)
            logger.info(f"[F182] {via} mapped param '{alias}' to '{name}' for {action_name} (trace {trace})")
            break
    return mapped


def undeclared_params_refusal(action_name: str, action_def: Any, params: Dict[str, Any], trace: str,
                              via: str = VIA_DISPATCHER) -> Optional[str]:
    """F182: the refusal for the keys in ``params`` the action does not take,
    each logged to be counted, or None when there are none."""
    unknown = undeclared_params(action_def, params)
    for key in unknown:
        logger.info(f"[F182] {via} refused param '{key}' for {action_name} (trace {trace})")
    return unknown_params_error(action_name, action_def, unknown, params, via) if unknown else None


def unknown_action_error(action_name: str, registry: Any) -> str:
    """F121: the actions an unknown-action error suggests are ones that can
    run here — never one F078 leaves out of every surface."""
    from modules.tools.discovery.action_registry import action_is_available

    runnable = [a.name for a in registry.get_all() if action_is_available(a)]
    return f"Unknown platform action: '{action_name}'. Use one of: {runnable[:20]}..."


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
        """Evaluate one tool call through the unified PolicyGate (PRD-192 S1).

        Returns ``None`` when execution may proceed and an errors-as-data result
        dict when the plane BLOCKS the call — the caller returns it directly, so
        the tool never runs.

        Stage semantics (the ``AUTOMATOS_POLICY_PLANE`` mode dial, ops-flipped):

        - ``off``         — byte-for-byte legacy: no evaluation, no bus fire.
        - ``shadow``      — evaluate + audit EVERY verdict; never block.
        - ``destructive`` — enforce deny/ask only for the fail-closed risk
          classes (destructive / external_side_effect / publish); shadow-log
          blocking verdicts on the open classes.
        - ``on``          — enforce every blocking verdict.

        Fail posture on a plane fault (locked, PRD-192): under enforce modes the
        closed classes ⇒ **deny** errors-as-data (``policy_plane_error``);
        read/internal_write ⇒ proceed with the greppable ``[policy-fail-open]``
        marker; shadow never blocks; an unclassifiable call is treated
        destructive (closed). One classification, computed once, is reused for
        the mode branch, the bus fire, and the fault branch.

        Never raises — every branch resolves to "block with a readable denial"
        or "proceed" (the downstream per-tool gates in ``platform_executor``
        remain in force for platform actions at every stage).
        """
        try:
            from modules.policy import policy_plane_mode

            mode = policy_plane_mode()
        except Exception:
            logger.warning(
                "[tool-trace %s] policy mode read failed for '%s' — plane "
                "treated off", trace, tool_name, exc_info=True,
            )
            return None
        if mode == "off":
            return None  # byte-for-byte the legacy per-router gates

        effective_name, effective_params, is_composio = self._resolve_effective_call(
            tool_name, parameters
        )
        # Computed ONCE (best-effort): reused for the mode branch, the bus fire,
        # and the fault branch. None ⇒ unclassifiable ⇒ treated destructive.
        risk = self._classify_risk(effective_name, is_composio)

        try:
            from modules.policy import PolicyGate, ToolCall, Decision
            from modules.policy.errors import verdict_to_result

            # PRD-192 S3: lift the caller's turn-level estimate (driving model +
            # prompt tokens + output cap) into the ToolCall so budget admission
            # prices THIS call instead of a structural $0. Callers that don't
            # know their model pass nothing — spend-to-date still binds.
            cc = caller_context if isinstance(caller_context, dict) else {}
            try:
                est_in = int(cc.get("est_input_tokens") or 0)
                est_out = int(cc.get("est_output_tokens") or 0)
            except (TypeError, ValueError):
                est_in = est_out = 0

            verdict = PolicyGate(self.db).check(
                ToolCall(
                    tool_name=effective_name,
                    parameters=effective_params,
                    workspace_id=workspace_id,
                    agent_id=agent_id,
                    caller_context=caller_context,
                    model_id=cc.get("model_id"),
                    est_input_tokens=est_in,
                    est_output_tokens=est_out,
                    is_composio=is_composio,
                )
            )

            # PRD-181 S1 (Art.12): fire the policy bus for EVERY verdict — allow,
            # ask, and deny — so the attached audit handler records every tool
            # call + policy decision per tenant. The bus is the single audit
            # write point (bus.py:18). A handler fault is swallowed inside the
            # bus, so audit never wedges or slows the call.
            self._fire_policy_bus(
                effective_name, effective_params, verdict,
                agent_id=agent_id, workspace_id=workspace_id,
                caller_context=caller_context, trace=trace,
                risk=risk, mode=mode, est_tokens=est_in + est_out,
            )

            if verdict.decision is Decision.ALLOW:
                return None

            if mode == "shadow":
                logger.info(
                    "[tool-trace %s] policy plane (shadow) would-%s '%s' "
                    "(risk=%s): %s — proceeding, never blocks",
                    trace, verdict.decision.value, effective_name, risk,
                    verdict.reason,
                )
                return None

            if mode == "destructive" and not self._risk_fails_closed(risk):
                logger.info(
                    "[tool-trace %s] policy plane (destructive stage) shadow-"
                    "logged %s '%s' (risk=%s): %s — open class, proceeding",
                    trace, verdict.decision.value, effective_name, risk,
                    verdict.reason,
                )
                return None

            logger.info(
                "[tool-trace %s] policy plane %s '%s': %s",
                trace, verdict.decision.value, effective_name, verdict.reason,
            )
            return verdict_to_result(verdict, tool_name)
        except Exception:
            return self._policy_fault_result(
                tool_name, effective_name, effective_params, risk, mode,
                agent_id=agent_id, workspace_id=workspace_id,
                caller_context=caller_context, trace=trace,
            )

    def _widget_gate(self, tool_name: str, parameters: Any, trace: str) -> Optional[Dict[str, Any]]:
        """The refusal for a widget turn calling what its key's scopes do not
        grant (core.security.widget_scopes); None when the call may run."""
        from core.security.surface import widget_scopes, widget_turn

        if not widget_turn():
            return None
        from core.security.widget_scopes import WIDGET_REFUSAL, widget_may_call

        effective_name = self._resolve_effective_call(tool_name, parameters)[0]
        if widget_may_call(effective_name, widget_scopes()):
            return None
        logger.warning(
            "[tool-trace %s] widget turn refused '%s' (resolved '%s'): not granted by the key's scopes",
            trace, tool_name, effective_name,
        )
        return {"success": False, "permission_denied": True, "error": WIDGET_REFUSAL, "tool": tool_name}

    def _resolve_effective_call(
        self, tool_name: str, parameters: Any
    ) -> tuple:
        """Resolve ``(effective_name, effective_params, is_composio)`` for the gate.

        - ``platform_execute`` nests the real action under ``"action"`` (params
          may be flat or wrapped) — the gate must judge the ACTION, not the
          dispatcher.
        - ``composio_execute`` nests the real Composio action the same way; the
          resolved per-action name (``GMAIL_SEND_EMAIL``) is what the audit row
          and risk classification carry.
        - Per-action SDK names route via ``self.composio_actions`` / registry
          ``integration_type`` metadata — the executor's own routing knowledge,
          threaded to the gate so external sends never classify internal
          (PRD-192 S1, the honest-classification fix).

        Never raises.
        """
        params = parameters if isinstance(parameters, dict) else {}
        try:
            if tool_name == "platform_execute" and isinstance(parameters, dict):
                action_name = (parameters.get("action") or "").strip()
                if not action_name:
                    return tool_name, params, False
                effective_params = parameters.get("params") or {
                    k: v for k, v in parameters.items() if k not in ("action", "params")
                }
                return action_name, effective_params, False

            if tool_name == "composio_execute" and isinstance(parameters, dict):
                raw_action = parameters.get("action") or parameters.get("action_name")
                effective_name = (
                    str(raw_action).upper().strip() if raw_action else tool_name
                )
                inner = parameters.get("params")
                effective_params = inner if isinstance(inner, dict) else params
                return effective_name, effective_params, True

            return tool_name, params, self._tool_is_composio(tool_name)
        except Exception:
            return tool_name, params, False

    def _tool_is_composio(self, tool_name: str) -> bool:
        """The executor's OWN routing knowledge of "is this a Composio call".

        Mirrors ``execute_tool``'s dispatch order: the ``composio_`` prefix /
        meta-tool, the per-action ``composio_actions`` dict (SDK schema names
        like ``GMAIL_SEND_EMAIL``), and registry ``integration_type`` metadata.
        Builtin-routed tools short-circuit False so the registry is never
        touched for them. Never raises.
        """
        name = tool_name or ""
        if name == "composio_execute" or name.startswith("composio_"):
            return True
        if name in self.composio_actions:
            return True
        if name.startswith(("platform_", "workspace_")) or name in self.tool_routes:
            return False
        try:
            tool_spec = self.tool_registry.get_tool(name)
            return bool(
                tool_spec
                and tool_spec.metadata
                and tool_spec.metadata.get("integration_type") == "composio"
            )
        except Exception:
            return False

    def _classify_risk(self, effective_name: str, is_composio: bool) -> Optional[str]:
        """Best-effort risk class for the call (pure classifier + registry
        permission level). ``None`` ⇒ unclassifiable — the caller treats that
        as destructive (fail closed, locked PRD-192 decision)."""
        try:
            from modules.policy import classify_action

            action_def = self._policy_action_def(effective_name)
            permission_level = getattr(action_def, "permission_level", None)
            return classify_action(
                effective_name,
                permission_level=permission_level,
                is_composio=is_composio,
            )
        except Exception:
            return None

    @staticmethod
    def _risk_fails_closed(risk: Optional[str]) -> bool:
        """True when this risk class fails CLOSED on a plane fault under the
        enforce modes — also the exact set the ``destructive`` stage enforces
        (one frozenset, ``modules.policy.FAIL_CLOSED_RISK_CLASSES``).
        Unclassifiable (``None``) is closed; if even the class set can't be
        read, closed."""
        if risk is None:
            return True
        try:
            from modules.policy import FAIL_CLOSED_RISK_CLASSES

            return risk in FAIL_CLOSED_RISK_CLASSES
        except Exception:
            return True

    def _policy_fault_result(
        self,
        tool_name: str,
        effective_name: str,
        effective_params: Dict[str, Any],
        risk: Optional[str],
        mode: str,
        *,
        agent_id: int,
        workspace_id: Optional[UUID],
        caller_context: Optional[Dict[str, Any]],
        trace: str,
    ) -> Optional[Dict[str, Any]]:
        """The ONE place that decides the plane's fail posture on a fault
        (PRD-192 S1, locked matrix).

        Called from ``_policy_gate_check``'s except while the original
        exception is being handled (``exc_info=True`` captures it). shadow ⇒
        proceed; enforce modes: open classes (read / internal_write) ⇒ proceed
        with the greppable ``[policy-fail-open]`` marker (the G.5 rate the
        shadow report counts); closed classes / unclassifiable ⇒ deny
        errors-as-data, audited via the bus. Never raises.
        """
        if mode == "shadow" or not self._risk_fails_closed(risk):
            logger.warning(
                "[policy-fail-open] [tool-trace %s] policy gate errored for "
                "'%s' (risk=%s, mode=%s) — proceeding (downstream gates still "
                "apply)", trace, effective_name, risk, mode, exc_info=True,
            )
            # PRD-192 S2: fail-opens are COUNTED, not just logged — fire the
            # bus with the marker so the audit row exists and the shadow
            # report can compute the G.5 fail-open rate from data.
            try:
                from modules.policy import Verdict

                self._fire_policy_bus(
                    effective_name, effective_params,
                    Verdict.allow(
                        f"[policy-fail-open] plane fault — proceeded "
                        f"(risk={risk or 'unknown'}, mode={mode})"
                    ),
                    agent_id=agent_id, workspace_id=workspace_id,
                    caller_context=caller_context, trace=trace,
                    risk=risk, mode=mode,
                )
            except Exception:
                logger.debug(
                    "[tool-trace %s] fail-open audit fire skipped", trace,
                    exc_info=True,
                )
            return None

        logger.error(
            "[tool-trace %s] policy gate errored for '%s' (risk=%s, mode=%s) "
            "— FAILING CLOSED, call blocked", trace, effective_name, risk, mode,
            exc_info=True,
        )
        try:
            from modules.policy import PolicyError, Verdict
            from modules.policy.errors import verdict_to_result

            deny = Verdict.deny(
                PolicyError(
                    code="policy_plane_error",
                    message_for_model=(
                        f"The policy plane could not evaluate '{effective_name}' "
                        f"(risk class: {risk or 'unknown'}). High-risk actions "
                        "fail closed on a plane fault, so it was NOT executed."
                    ),
                    remediation=(
                        "Retry shortly. If this persists, a workspace admin "
                        "should check the policy plane or approve the action "
                        "explicitly."
                    ),
                    retryable=True,
                ),
                reason=f"policy plane fault — {risk or 'unclassifiable'} fails closed",
            )
            self._fire_policy_bus(
                effective_name, effective_params, deny,
                agent_id=agent_id, workspace_id=workspace_id,
                caller_context=caller_context, trace=trace,
                risk=risk, mode=mode,
            )
            return verdict_to_result(deny, tool_name)
        except Exception:
            # Even the typed denial failed to build — still block: a closed
            # class must never run on a plane fault. Hand-rolled envelope with
            # the same errors-as-data shape.
            logger.error(
                "[tool-trace %s] policy fault denial construction failed for "
                "'%s' — returning minimal closed block", trace, effective_name,
                exc_info=True,
            )
            message = (
                f"The policy plane could not evaluate '{effective_name}'. "
                "High-risk actions fail closed, so it was NOT executed."
            )
            return {
                "success": False,
                "tool": tool_name,
                "error": message,
                "llm_context": message,
                "permission_denied": True,
                "requires_approval": False,
                "policy_error": {
                    "code": "policy_plane_error",
                    "message_for_model": message,
                    "remediation": "Retry shortly or ask a workspace admin.",
                    "retryable": True,
                },
                "policy_decision": "deny",
                "fatal_error": False,
                "error_type": "policy_plane_error",
            }

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
        risk: Optional[str] = None,
        mode: Optional[str] = None,
        est_tokens: Optional[int] = None,
    ) -> None:
        """Fire ``PRE_TOOL_USE`` on the policy bus with the verdict (PRD-181 S1).

        The attached audit handler reads ``ctx.data['verdict']`` and writes the
        per-tenant Art.12 record; ``risk``, ``mode`` and ``est_tokens`` ride
        along so the audit row (and the PRD-192 S2 shadow report) carry the
        classification, the stage that produced the decision, and the G.2
        priced-call signal. Never raises: audit is a side-effect of the
        chokepoint, so a bus/handler fault must not block or slow the call.
        """
        try:
            from modules.policy import (
                Event,
                EventContext,
                get_policy_bus,
            )

            ctx = EventContext(
                workspace_id=workspace_id,
                agent_id=agent_id,
                tool_name=effective_name,
                tool_input=effective_params,
                caller_context=caller_context,
            )
            ctx.data["verdict"] = verdict
            ctx.data["risk"] = risk
            ctx.data["mode"] = mode
            ctx.data["est_tokens"] = est_tokens
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
            caller_context: The chat's server-built context (``user_id``,
                ``driving_user_id``, ``system_role``, ``conversation_id``).
                Forwarded to PlatformActionExecutor for its super_admin_only
                and admin_only gates; see its execute() for what each reads
                and what happens without one.

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

            # F088: a name a model reaches for that is not a tool runs the tool
            # that does that job — before any gate or route sees it.
            if tool_name in TOOL_ALIASES:
                logger.info(f"[tool-trace {trace}] '{tool_name}' is not a tool — running {TOOL_ALIASES[tool_name]}")
                tool_name = TOOL_ALIASES[tool_name]

            # F155: a public widget turn runs only what its key's scopes grant,
            # judged on the resolved action (platform_execute's included).
            _widget_block = self._widget_gate(tool_name, parameters, trace)
            if _widget_block is not None:
                return _widget_block

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

            # PRD-233 S2: ONE Composio availability seam for EVERY caller. The
            # router refuses routed calls; agent_factory and approval-grant
            # replays call this executor directly and used to fall through to
            # the executor's misleading composio_not_connected error. Without
            # a key ⇒ explicit integrations_unavailable, never a fall-through;
            # with a key ⇒ one cached boolean check.
            from core.composio.client import composio_available
            from modules.tools.tool_router import (
                _integrations_unavailable_result,
                _is_composio_tool_name,
            )
            if (
                _is_composio_tool_name(tool_name) or tool_name in self.composio_actions
            ) and not composio_available():
                return _integrations_unavailable_result(tool_name, trace)

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
                    result = {
                        "success": False,
                        "error": unknown_action_error(action_name, registry),
                        "tool": tool_name,
                    }
                    return result

                # Validate required params — after filling any that the caller
                # supplied under an obvious other name. Night 1: 18 of 89 failed
                # platform_execute calls were "Missing required params for
                # 'platform_submit_report': ['title','content']" from agents that
                # HAD written a report, under 'body' or nested in 'report'. The
                # same schema/handler drift as the onboarding blueprint-rules wall:
                # refuse a call that is genuinely incomplete, not one that used a
                # synonym.
                required = action_def.parameters.get("required", [])
                action_params = _fill_required_from_aliases(action_params, required)
                action_params = map_optional_aliases(action_name, action_def, action_params, trace)
                missing = [p for p in required if p not in action_params]
                # F182 (night 6): a key the action does not take is refused, not
                # dropped. Run 207 started with no inputs because they came nested
                # under "params", and two update_playbook calls changed nothing
                # and reported success. Each refusal is logged, to be counted.
                refused = undeclared_params_refusal(action_name, action_def, action_params, trace)
                if missing or refused:
                    errors = []
                    if missing:
                        # F027-C: the exact call, every required key with its type.
                        errors.append(missing_params_error(action_name, action_def.parameters, missing, action_params))
                    if refused:
                        errors.append(refused)
                    result = {"success": False, "error": "\n".join(errors), "tool": tool_name}
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
                # F182: a direct call is held to platform_execute's rule. Night 6's
                # update_task_status sent "reason", which it does not take, and the
                # reason was dropped without a word.
                from modules.tools.discovery import get_action_registry

                action_def = get_action_registry().get(tool_name)
                if action_def is not None and isinstance(parameters, dict):
                    parameters = _fill_required_from_aliases(parameters, action_def.parameters.get("required", []))
                    parameters = map_optional_aliases(tool_name, action_def, parameters, trace, VIA_DIRECT_CALL)
                    refused = undeclared_params_refusal(tool_name, action_def, parameters, trace, VIA_DIRECT_CALL)
                    if refused:
                        result = {"success": False, "error": refused, "tool": tool_name}
                        return result
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
                    "error": self._unknown_tool_error(tool_name),
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
                    "error": self._unknown_tool_error(tool_name),
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
            # F074: every tool in a turn runs on this one request session. A tool
            # that caught its own failed statement and returned it as data left
            # the transaction aborted, and every tool after it failed "current
            # transaction is aborted" (night 1: query_database 7/7). Never hand
            # the next tool a dead session.
            rollback_if_aborted(getattr(self, "db", None), f"tool '{tool_name}'")
            # PRD-139: Universal telemetry — fire-and-forget, never fails the tool call
            _exec_ms = int((_time.monotonic() - _exec_start) * 1000)
            fire_telemetry(
                tool_name=tool_name,
                parameters=parameters if isinstance(parameters, dict) else {},
                agent_id=agent_id,
                workspace_id=workspace_id,
                result=result,
                execution_time_ms=_exec_ms,
                caller_context=caller_context,
            )
            # PRD-232 US-011a: platform_find_tools IS the model hunting for a
            # capability it wasn't given — a tool_gap. Record it on the same
            # telemetry lane (fire-and-forget, own session) so the nightly
            # resolution join can credit whatever action eventually serves the
            # intent. The turn's real query (caller_context.user_query), not the
            # model's search string, is the intent that clusters the gap. Catch
            # BOTH dispatch shapes: the direct promoted call, and the closed-pins
            # fallback where find_tools rides the platform_execute enum.
            _params = parameters if isinstance(parameters, dict) else {}
            _is_find_tools = tool_name == "platform_find_tools" or (
                tool_name == "platform_execute"
                and _params.get("action") == "platform_find_tools"
            )
            if _is_find_tools:
                try:
                    _nested = _params.get("params") if isinstance(_params.get("params"), dict) else {}
                    # F033: the CAPABILITY the model went looking for is the
                    # gap. This preferred caller_context.user_query — the whole
                    # user prompt — so every row recorded that find_tools was
                    # used, not what was missing: 43 rows of prompts, which is
                    # a usage log wearing a gap's name. The searched term wins;
                    # the prompt is only the fallback when there isn't one.
                    _searched = _params.get("query") or _nested.get("query")
                    _gap_query = _searched or (caller_context or {}).get("user_query")
                    fire_tool_gap(
                        query=_gap_query,
                        workspace_id=workspace_id,
                        agent_id=agent_id,
                        gap_source="find_tools" if _searched else "find_tools_unspecified",
                        caller_context=caller_context,
                    )
                except Exception:
                    logger.debug("[tool-gap] find_tools gap write skipped", exc_info=True)
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

    def _unknown_tool_error(self, tool_name: str) -> str:
        """F088: "Unknown tool" alone came back to the owner as "the document
        search is broken" — say which real tools are nearest, so the model
        retries with one instead."""
        import difflib

        names = set(self.tool_routes)
        try:
            names |= {t.name for t in self.tool_registry.get_all_tools()}
            from modules.tools.discovery.action_registry import action_is_available, get_action_registry

            # F121: never suggest an action that cannot run here (F078)
            names |= {a.name for a in get_action_registry().get_all() if action_is_available(a)}
        except Exception:  # noqa: BLE001 — the suggestions are a courtesy, never a failure
            logger.debug("unknown-tool suggestions unavailable", exc_info=True)
        near = difflib.get_close_matches(tool_name, sorted(names), n=3, cutoff=0.6)
        if not near:
            return f"Unknown tool: {tool_name} — there is no tool by that name; use one from your tool list."
        return (f"Unknown tool: {tool_name}. Nearest real tools: {', '.join(near)} "
                "(a platform_* action runs directly or through platform_execute).")

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
        # F179: a workspace tool clears the gates its definition declares, as a
        # platform action does. A direct call, platform_execute and an approved
        # card's resume all dispatch workspace tools here.
        return await exec_workspace.execute_gated_workspace_action(
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
