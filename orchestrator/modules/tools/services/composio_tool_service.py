"""
Composio Tool Service — Per-Action OpenAI Function-Calling Tools
================================================================

Resolves Composio actions into per-action OpenAI function-calling tool schemas.
Each action becomes its own top-level function with correct param names baked in
(e.g. ``JIRA_GET_ISSUE(issue_id_or_key="PILOT-123")``).

Strategy:
  1. Extract explicit action names from the prompt (e.g. GITHUB_CREATE_A_REFERENCE)
     → fetch exact schemas from per-app cache.
  2. No explicit names → use the Composio SDK's semantic search scoped to the
     agent's assigned apps, with a per-app limit to stay within context limits.

Consumers:
  - Recipe executor (Phase 1)
  - Chatbot / external API (Phase 2 — PRD-50)
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from uuid import UUID

from sqlalchemy.orm import Session

from core.composio.client import get_composio_client
from core.composio.entity_manager import EntityManager
from core.models.composio_cache import AgentAppAssignment
from core.models.core import Agent

logger = logging.getLogger(__name__)

# Max tools per app to avoid blowing the LLM context window.
# 15 tools × ~600 tokens each ≈ 9K tokens per app — safe for 64K+ models.
_MAX_TOOLS_PER_APP = 15


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ComposioToolResult:
    """Result from ComposioToolService.get_tools_for_step()."""
    tools: List[Dict[str, Any]] = field(default_factory=list)
    action_set: Set[str] = field(default_factory=set)
    entity_id: str = ""
    app_names: List[str] = field(default_factory=list)
    strategy: str = "none"  # "exact_lookup" | "sdk_search" | "none"
    search_ms: int = 0


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class ComposioToolService:
    """
    Resolves Composio actions into OpenAI function-calling tools.

    Uses the Composio SDK to fetch per-app action schemas and presents
    them as individual tools to the LLM. Reusable across all consumers
    (recipe executor, chatbot, external API — PRD-50 alignment).
    """

    def __init__(self, db: Session):
        self.db = db

    # Regex to extract explicit Composio action names from prompts.
    # Matches patterns like GITHUB_CREATE_A_REFERENCE, JIRA_GET_ISSUE, etc.
    _ACTION_NAME_RE = re.compile(r"\b([A-Z][A-Z0-9]+(?:_[A-Z0-9]+){2,})\b")

    def get_tools_for_step(
        self,
        agent_id: int,
        workspace_id: UUID,
        task_prompt: str,
        limit: int = _MAX_TOOLS_PER_APP,
    ) -> ComposioToolResult:
        """
        Resolve Composio tools for a step.

        Strategy:
          1. Extract explicit action names from the prompt
             (e.g. GITHUB_CREATE_A_REFERENCE) → fetch exact schemas.
          2. No explicit names → SDK semantic search per app, capped
             at ``limit`` tools per app to stay within context limits.

        Returns:
            ComposioToolResult with OpenAI function-calling schemas,
            entity_id, resolved app names, strategy, and search latency.
        """
        result = ComposioToolResult()

        try:
            # 1. Resolve allowed apps for this agent
            allowed_apps = self._resolve_allowed_apps(agent_id, workspace_id)
            if not allowed_apps:
                logger.info(
                    "[ComposioToolService] No allowed apps for agent=%s workspace=%s",
                    agent_id, workspace_id,
                )
                return result
            result.app_names = allowed_apps
            app_prefixes = {a.upper() for a in allowed_apps}

            # 2. Resolve entity_id for this workspace
            entity_id = self._resolve_entity_id(workspace_id)
            if not entity_id:
                logger.warning(
                    "[ComposioToolService] No Composio entity for workspace=%s",
                    workspace_id,
                )
                return result
            result.entity_id = entity_id

            client = get_composio_client()
            t0 = time.monotonic()

            # 3. Try explicit action names first
            explicit_names = self._extract_action_names(task_prompt, app_prefixes)

            if explicit_names:
                lookup_results = client.get_action_schemas_by_name(
                    action_names=list(explicit_names),
                    entity_id=entity_id,
                    app_names=[a.lower() for a in allowed_apps],
                )
                for item in lookup_results:
                    action_name = item.get("action_name", "")
                    schema = item.get("schema")
                    if action_name and schema and action_name not in result.action_set:
                        result.tools.append(schema)
                        result.action_set.add(action_name)

                if result.tools:
                    result.search_ms = int((time.monotonic() - t0) * 1000)
                    result.strategy = "exact_lookup"
                    logger.info(
                        "[ComposioToolService] Exact lookup: agent=%s "
                        "requested=%d resolved=%d actions=%s (%dms)",
                        agent_id, len(explicit_names), len(result.tools),
                        sorted(result.action_set), result.search_ms,
                    )
                    return result

            # 4. SDK semantic search — let Composio find the best actions
            #    for this prompt, scoped to the agent's apps, limited per app.
            search_results = client.search_actions_for_step(
                search_query=task_prompt[:200],
                app_names=[a.lower() for a in allowed_apps],
                entity_id=entity_id,
                limit=limit,
            )
            for item in search_results:
                action_name = item.get("action_name", "")
                schema = item.get("schema")
                if action_name and schema and action_name not in result.action_set:
                    result.tools.append(schema)
                    result.action_set.add(action_name)

            result.search_ms = int((time.monotonic() - t0) * 1000)

            if result.tools:
                result.strategy = "sdk_search"
                logger.info(
                    "[ComposioToolService] SDK search: agent=%s apps=%s "
                    "actions=%d/%d (%dms) → %s",
                    agent_id, allowed_apps, len(result.tools), limit,
                    result.search_ms, sorted(result.action_set),
                )
            else:
                logger.warning(
                    "[ComposioToolService] No tools resolved for agent=%s "
                    "apps=%s prompt=%r (%dms)",
                    agent_id, allowed_apps, task_prompt[:80], result.search_ms,
                )

        except Exception as exc:
            logger.warning(
                "[ComposioToolService] Failed for agent=%s: %s",
                agent_id, exc, exc_info=True,
            )

        return result

    def execute_action(
        self,
        action_name: str,
        params: dict,
        entity_id: str,
    ) -> Dict[str, Any]:
        """
        Execute a Composio action directly.

        Returns:
            Dict with ``success``, ``data``, and ``error`` keys.
        """
        client = get_composio_client()
        return client.execute_action(
            action=action_name,
            params=params,
            entity_id=entity_id,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _extract_action_names(cls, prompt: str, app_prefixes: Set[str]) -> Set[str]:
        """
        Extract explicit Composio action names from a prompt.

        Matches uppercase patterns like GITHUB_CREATE_A_REFERENCE and filters
        to only those whose prefix (e.g. GITHUB) is in the allowed app set.
        """
        candidates = cls._ACTION_NAME_RE.findall(prompt)
        return {
            name for name in candidates
            if name.split("_", 1)[0] in app_prefixes
        }

    def _resolve_allowed_apps(self, agent_id: int, workspace_id: UUID) -> List[str]:
        """
        Query AgentAppAssignment for EXTERNAL active apps, intersect with
        workspace connections.
        """
        assigned = (
            self.db.query(AgentAppAssignment)
            .filter(
                AgentAppAssignment.agent_id == agent_id,
                AgentAppAssignment.is_active.is_(True),
                AgentAppAssignment.app_type == "EXTERNAL",
            )
            .all()
        )
        assigned_apps = [(a.app_name or "").upper() for a in assigned if a.app_name]
        if not assigned_apps:
            return []

        # Cross-reference with connected apps
        connected_apps: List[str] = []
        try:
            manager = EntityManager(self.db)
            entity = manager.get_entity_by_workspace(workspace_id)
            if entity:
                connected_apps = [
                    (c.get("app_name") or "").upper()
                    for c in manager.get_entity_connections(str(entity["id"]))
                    if c.get("status") in ("active", "pending")
                ]
        except Exception as conn_err:
            logger.warning("[ComposioToolService] Connection check failed: %s", conn_err)

        if connected_apps:
            allowed = [a for a in assigned_apps if a in set(connected_apps)]
            return allowed if allowed else assigned_apps

        return assigned_apps

    def _resolve_entity_id(self, workspace_id: UUID) -> Optional[str]:
        """Get the Composio entity_id (string) for this workspace."""
        try:
            manager = EntityManager(self.db)
            entity = manager.get_entity_by_workspace(workspace_id)
            if entity:
                return entity.get("composio_entity_id", "")
        except Exception as exc:
            logger.warning("[ComposioToolService] Entity lookup failed: %s", exc)
        return None
