"""
Action Registry (PRD-64)
========================

Central registry for platform actions that Auto can discover and execute.
Each ActionDefinition describes a platform operation with its parameters,
permission level, and OpenAI function schema.

Usage:
    registry = get_action_registry()
    all_actions = registry.get_all()
    openai_tools = registry.to_openai_tools()
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Thread-safe singleton
_registry_lock = threading.Lock()
_registry_instance: Optional["ActionRegistry"] = None


@dataclass
class ActionDefinition:
    """Definition of a platform action that Auto can execute."""

    name: str                           # e.g. "platform_list_agents"
    description: str                    # For embedding + LLM tool description
    category: str                       # "agents", "analytics", "recipes", etc.
    parameters: Dict[str, Any]          # OpenAI function parameters schema
    permission_level: str = "read"      # "read" | "write" | "destructive"
    requires_confirmation: bool = False
    workspace_scoped: bool = True
    admin_only: bool = False
    # PRD-143: observability/oversight tier. Fail-closed — every listing/
    # selection path excludes these unless include_super_admin=True is
    # passed explicitly (opt-in include, unlike admin_only's opt-in exclude).
    super_admin_only: bool = False
    promoted: bool = False
    tags: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)

    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ActionRegistry:
    """
    Registry of all platform actions available to Auto.

    Thread-safe. Actions are registered at startup and queried at runtime.
    Supports filtering by category and permission level.
    """

    def __init__(self):
        self._actions: Dict[str, ActionDefinition] = {}
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy-load platform + workspace actions on first access."""
        if not self._initialized:
            from .platform_actions import register_all_actions
            register_all_actions(self)  # includes workspace actions
            self._initialized = True
            logger.info(f"[ActionRegistry] Initialized with {len(self._actions)} actions (platform + workspace)")

    def register(self, action: ActionDefinition) -> None:
        """Register a platform action."""
        if action.name in self._actions:
            logger.warning(f"[ActionRegistry] Overwriting existing action: {action.name}")
        self._actions[action.name] = action

    def get(self, name: str) -> Optional[ActionDefinition]:
        """Get an action by name."""
        self._ensure_initialized()
        return self._actions.get(name)

    def get_all(self) -> List[ActionDefinition]:
        """Get all registered actions."""
        self._ensure_initialized()
        return list(self._actions.values())

    def get_by_category(self, category: str) -> List[ActionDefinition]:
        """Get actions filtered by category."""
        self._ensure_initialized()
        return [a for a in self._actions.values() if a.category == category]

    def get_by_tags(self, tags: List[str]) -> List[ActionDefinition]:
        """Get actions whose tags intersect any of *tags* (OR semantics)."""
        self._ensure_initialized()
        wanted = set(tags)
        return [a for a in self._actions.values() if wanted.intersection(a.tags)]

    def get_by_permission(self, level: str) -> List[ActionDefinition]:
        """Get actions filtered by permission level."""
        self._ensure_initialized()
        return [a for a in self._actions.values() if a.permission_level == level]

    def to_openai_tools(
        self,
        permission_filter: Optional[str] = None,
        include_super_admin: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Convert all actions to OpenAI function calling format.

        Args:
            permission_filter: Optional - only include actions with this permission level
            include_super_admin: Fail-closed — super_admin_only actions are
                excluded unless this is explicitly True.
        """
        self._ensure_initialized()
        actions = [
            a for a in self._actions.values()
            if include_super_admin or not a.super_admin_only
        ]
        if permission_filter:
            actions = [a for a in actions if a.permission_level == permission_filter]
        return [a.to_openai_schema() for a in actions]

    def get_promoted(self) -> List[ActionDefinition]:
        """Get all actions marked as promoted."""
        self._ensure_initialized()
        return [a for a in self._actions.values() if a.promoted]

    def to_first_class_schemas(
        self,
        exclude_admin: bool = False,
        include_super_admin: bool = False,
        first_class_names: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return OpenAI function schemas for promoted actions.

        Promoted actions get their own first-class tool schemas instead of
        going through the platform_execute dispatcher.

        Args:
            exclude_admin: If True, admin_only promoted actions are excluded
                (non-admin callers won't get schemas for admin tools).
            include_super_admin: Fail-closed — super_admin_only actions are
                excluded unless this is explicitly True.
            first_class_names: PRD-232 US-014 (promotion-as-prior) — when given,
                ONLY promoted actions whose name is in this set attach first-class
                (the config pins + whatever ranked into the query surface). Every
                other promoted action is reachable via the platform_execute
                dispatcher enum instead. When None, legacy behaviour: every promoted
                action attaches first-class (unconditional-all-promoted). Role/tier
                filters below ALWAYS run, so a set can never re-admit an admin/su
                action the caller isn't entitled to.
        """
        self._ensure_initialized()
        promoted = [a for a in self._actions.values() if a.promoted]
        if first_class_names is not None:
            promoted = [a for a in promoted if a.name in first_class_names]
        if not include_super_admin:
            promoted = [a for a in promoted if not a.super_admin_only]
        if exclude_admin:
            promoted = [a for a in promoted if not a.admin_only]
        return [a.to_openai_schema() for a in promoted]

    def to_dispatcher_schema(
        self,
        exclude_admin: bool = False,
        exclude_promoted: bool = True,
        allowed_names: Optional[List[str]] = None,
        include_super_admin: bool = False,
        allow_promoted_in_allowlist: bool = False,
        exclude_names: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """
        Return a SINGLE OpenAI tool schema (platform_execute) that wraps
        all platform actions behind one dispatcher.

        Args:
            exclude_admin: If True, admin_only actions are excluded from the
                dispatcher (non-admin callers won't see them).
            exclude_promoted: If True (default), promoted actions are excluded
                from the dispatcher since they have first-class schemas.
            allowed_names: Optional whitelist applied AFTER admin/promoted
                filters. When None, the enum exposes every eligible action
                (legacy behavior). When a non-empty list, the enum is the
                intersection of (admin/promoted-filtered actions) and
                ``allowed_names``. When an empty list, falls back to the full
                enum and logs a WARNING — empty list is treated as "ranker
                returned nothing", not "block everything", so the LLM is
                never left with zero callable actions.
            include_super_admin: Fail-closed — super_admin_only actions are
                excluded from the enum (and from every fallback path)
                unless this is explicitly True.
            allow_promoted_in_allowlist: PR-B (tool-surface review) — when
                True, names in ``allowed_names`` that are promoted may enter
                the enum despite ``exclude_promoted`` (role filters still
                apply first). Used by the closed-pins fallback, whose pin set
                (platform_find_tools et al.) is largely promoted; without
                this the pins would intersect to nothing and fall open to
                the full enum — the exact failure the mode exists to stop.
            exclude_names: PRD-232 US-014 (promotion-as-prior) — names to keep
                OUT of the enum because they are attached FIRST-CLASS this turn
                (the config pins + whatever promoted actions ranked into the
                surface). Applied AFTER the role/su filters, alongside
                ``exclude_promoted=False`` so the remaining (non-first-class)
                promoted actions stay reachable in the enum like any action.
        """
        self._ensure_initialized()

        exclude_set = set(exclude_names or ())

        # Build enum of valid action names AFTER admin/su/promoted filters.
        # The su filter applies here, BEFORE the allow-list, so the
        # empty-intersection fallback below can never re-admit su actions.
        # US-014: exclude_set drops the first-class-attached names (pins + ranked
        # promoted) so they aren't duplicated in the enum.
        valid_actions = sorted(
            a.name for a in self._actions.values()
            if (not exclude_promoted or not a.promoted)
            and (not exclude_admin or not a.admin_only)
            and (include_super_admin or not a.super_admin_only)
            and a.name not in exclude_set
        )

        # PRD-138 US-008: optional allow-list narrows the enum so the LLM only
        # sees the ranker's top-K. Permission filters above always run first.
        if allowed_names is None:
            narrowed_actions = valid_actions
        elif len(allowed_names) == 0:
            logger.warning(
                "[ActionRegistry] to_dispatcher_schema(allowed_names=[]) — "
                "empty allow-list, falling back to full enum"
            )
            narrowed_actions = valid_actions
        else:
            allow_set = set(allowed_names)
            intersect_pool = valid_actions
            if allow_promoted_in_allowlist and exclude_promoted:
                # Same role gates as valid_actions, promoted admitted — the
                # allow-list (pins) is the narrowing here, not the flag.
                intersect_pool = sorted(
                    a.name for a in self._actions.values()
                    if (not exclude_admin or not a.admin_only)
                    and (include_super_admin or not a.super_admin_only)
                    and a.name not in exclude_set
                )
            narrowed_actions = [n for n in intersect_pool if n in allow_set]
            # Defensive: if the intersection is empty (e.g. ranker returned
            # only admin actions for a non-admin caller), fall back to the
            # full eligible set rather than ship a schema with zero options.
            if not narrowed_actions:
                logger.warning(
                    "[ActionRegistry] to_dispatcher_schema: allowed_names "
                    "intersection is empty after permission filters, "
                    "falling back to full enum"
                )
                narrowed_actions = valid_actions

        action_property: Dict[str, Any] = {
            "type": "string",
            "description": "The exact platform action name (e.g. 'platform_configure_agent_heartbeat')",
        }
        if narrowed_actions:
            action_property["enum"] = narrowed_actions

        return {
            "type": "function",
            "function": {
                "name": "platform_execute",
                "description": (
                    "Execute an internal Automatos platform action. "
                    "You MUST pass both 'action' and 'params'. "
                    "Example: platform_execute(action='platform_configure_agent_heartbeat', "
                    "params={'agent_id': 147, 'enabled': true, 'interval_minutes': 15}). "
                    "See the 'Available Platform Actions' section in your system prompt "
                    "for the full list of actions and their required parameters."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": action_property,
                        "params": {
                            "type": "object",
                            "description": (
                                "Parameters for the action as a JSON object. "
                                "Always include required params from the action's definition. "
                                "Example: {'agent_id': 147, 'enabled': true, 'interval_minutes': 60}"
                            ),
                        },
                    },
                    "required": ["action", "params"],
                },
            },
        }

    def build_prompt_summary(
        self,
        exclude_admin: bool = False,
        exclude_promoted: bool = False,
        include_super_admin: bool = False,
        exclude_names: Optional[List[str]] = None,
    ) -> str:
        """
        Build a markdown summary of all platform actions for injection
        into the agent's system prompt.  Grouped by category.

        Args:
            exclude_admin: If True, admin_only actions are skipped from the
                summary (non-admin callers won't see them).
            exclude_promoted: If True, promoted actions are skipped from the
                prompt text. Default False — promoted actions appear in a
                "Direct Tools" section with call-directly instructions.
            include_super_admin: Fail-closed — super_admin_only actions are
                excluded unless this is explicitly True.
            exclude_names: PRD-229 — action names to omit entirely (mode-scoped
                admission, e.g. ask_orchestrator outside execution lanes).
        """
        self._ensure_initialized()
        return self._format_actions_summary(
            list(self._actions.values()),
            exclude_admin=exclude_admin,
            exclude_promoted=exclude_promoted,
            include_super_admin=include_super_admin,
            exclude_names=exclude_names,
        )

    def build_filtered_prompt_summary(
        self,
        action_names: List[str],
        exclude_admin: bool = False,
        exclude_promoted: bool = False,
        include_super_admin: bool = False,
        exclude_names: Optional[List[str]] = None,
    ) -> str:
        """
        Build a markdown summary of only the named subset of platform actions,
        in the same format as build_prompt_summary().

        Intended for callers that have narrowed the action list via semantic
        ranking (PRD-138) and want to inject only the top-K into the prompt.

        Args:
            action_names: Names of actions to include. Names not in the
                registry are silently skipped (no error). An empty list
                yields a summary header with no action lines (does NOT
                fall back to all actions).
            exclude_admin: If True, admin_only actions are skipped.
            exclude_promoted: If True, promoted actions are skipped from
                the prompt text. Default False — promoted actions appear
                in a "Direct Tools" section.
            include_super_admin: Fail-closed — super_admin_only actions are
                excluded even when explicitly named, unless this is True.
        """
        self._ensure_initialized()
        # Preserve registry-membership filtering; unknown names silently skipped.
        # Use a set for O(1) lookup but iterate registry to keep deterministic
        # ordering inside the shared formatter (which sorts by category/name).
        wanted = set(action_names)
        filtered = [a for a in self._actions.values() if a.name in wanted]
        return self._format_actions_summary(
            filtered,
            exclude_admin=exclude_admin,
            exclude_promoted=exclude_promoted,
            include_super_admin=include_super_admin,
            exclude_names=exclude_names,
        )

    @staticmethod
    def _format_action_line(action: ActionDefinition) -> str:
        props = action.parameters.get("properties", {})
        required = action.parameters.get("required", [])
        param_hints = []
        for pname in props:
            req_marker = " (required)" if pname in required else ""
            param_hints.append(f"`{pname}`{req_marker}")
        param_str = f" — params: {', '.join(param_hints)}" if param_hints else ""
        return f"- `{action.name}`: {action.description}{param_str}"

    @staticmethod
    def _format_actions_summary(
        actions: List[ActionDefinition],
        exclude_admin: bool,
        exclude_promoted: bool,
        include_super_admin: bool = False,
        exclude_names: Optional[List[str]] = None,
    ) -> str:
        """
        Render a list of ActionDefinitions as the canonical markdown summary
        used in agent system prompts.  Shared between build_prompt_summary
        and build_filtered_prompt_summary so output format stays identical.

        Fail-closed: super_admin_only actions are skipped unless
        include_super_admin=True is passed explicitly.

        Promoted actions are rendered in a separate section with instructions
        to call them directly by name (they have first-class tool schemas).
        Non-promoted actions are rendered with ``platform_execute`` calling
        instructions.

        PRD-229: ``exclude_names`` drops actions by name entirely (mode-scoped
        admission), mirroring the callable-surface gate.
        """
        blocked = set(exclude_names or ())
        promoted_by_cat: Dict[str, List[ActionDefinition]] = {}
        dispatcher_by_cat: Dict[str, List[ActionDefinition]] = {}

        for action in actions:
            if action.name in blocked:
                continue
            if action.super_admin_only and not include_super_admin:
                continue
            if exclude_admin and action.admin_only:
                continue
            if exclude_promoted and action.promoted:
                continue
            bucket = promoted_by_cat if action.promoted else dispatcher_by_cat
            bucket.setdefault(action.category, []).append(action)

        lines: List[str] = []

        if promoted_by_cat:
            lines.append("\n## Direct Tools\n")
            lines.append(
                "Call these tools directly by name (they have their own "
                "function schemas). Do NOT wrap them in `platform_execute`.\n"
            )
            for category in sorted(promoted_by_cat.keys()):
                lines.append(f"### {category.replace('_', ' ').title()}")
                for action in sorted(promoted_by_cat[category], key=lambda a: a.name):
                    lines.append(ActionRegistry._format_action_line(action))
                lines.append("")

        if dispatcher_by_cat:
            lines.append("\n## Available Platform Actions\n")
            lines.append(
                "Use `platform_execute(action, params)` to call these. "
                "The `action` field must be the exact action name.\n"
            )
            for category in sorted(dispatcher_by_cat.keys()):
                lines.append(f"### {category.replace('_', ' ').title()}")
                for action in sorted(dispatcher_by_cat[category], key=lambda a: a.name):
                    lines.append(ActionRegistry._format_action_line(action))
                lines.append("")

        return "\n".join(lines)


def get_action_registry() -> ActionRegistry:
    """Get or create the global ActionRegistry singleton (thread-safe)."""
    global _registry_instance

    if _registry_instance is not None:
        return _registry_instance

    with _registry_lock:
        if _registry_instance is None:
            _registry_instance = ActionRegistry()

    return _registry_instance
