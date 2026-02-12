"""
Composio Tool Service — SDK Semantic Search → OpenAI Function-Calling Tools
===========================================================================

Reusable service that resolves Composio actions into per-action OpenAI
function-calling tool schemas using the SDK's server-side semantic search.

Instead of the LLM guessing action names inside a single ``composio_execute``
mega-tool, each action becomes its own top-level function with correct param
names baked in (e.g. ``JIRA_GET_ISSUE(issue_id_or_key="PILOT-123")``).

Consumers:
  - Recipe executor (Phase 1 — this PR)
  - Chatbot / external API (Phase 2 — PRD-50)

Falls back to ComposioHintService when SDK search returns empty or errors.
"""

import logging
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
    strategy: str = "none"  # "sdk_search" | "hint_fallback" | "none"
    search_ms: int = 0


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class ComposioToolService:
    """
    Resolves Composio actions into OpenAI function-calling tools using the
    SDK's semantic search. Reusable across all consumers (recipe executor,
    chatbot, external API — PRD-50 alignment).
    """

    def __init__(self, db: Session):
        self.db = db

    def get_tools_for_step(
        self,
        agent_id: int,
        workspace_id: UUID,
        task_prompt: str,
        limit: int = 8,
    ) -> ComposioToolResult:
        """
        Semantic-search Composio for actions relevant to ``task_prompt``.

        Returns:
            ComposioToolResult with OpenAI function-calling schemas, entity_id,
            resolved app names, strategy indicator, and search latency.
        """
        result = ComposioToolResult()

        try:
            # 1. Resolve allowed apps (same logic as ComposioHintService)
            allowed_apps = self._resolve_allowed_apps(agent_id, workspace_id)
            if not allowed_apps:
                logger.info(
                    "[ComposioToolService] No allowed apps for agent=%s workspace=%s",
                    agent_id, workspace_id,
                )
                return result
            result.app_names = allowed_apps

            # 2. Resolve entity_id for this workspace
            entity_id = self._resolve_entity_id(workspace_id)
            if not entity_id:
                logger.warning(
                    "[ComposioToolService] No Composio entity for workspace=%s", workspace_id,
                )
                return result
            result.entity_id = entity_id

            # 3. SDK semantic search
            t0 = time.monotonic()
            client = get_composio_client()
            raw_results = client.search_actions_for_step(
                search_query=task_prompt,
                app_names=[a.lower() for a in allowed_apps],
                entity_id=entity_id,
                limit=limit,
            )
            result.search_ms = int((time.monotonic() - t0) * 1000)

            if raw_results:
                for item in raw_results:
                    action_name = item.get("action_name", "")
                    schema = item.get("schema")
                    if action_name and schema:
                        result.tools.append(schema)
                        result.action_set.add(action_name)
                result.strategy = "sdk_search"
                logger.info(
                    "[ComposioToolService] SDK search OK: agent=%s actions=%s search_ms=%d",
                    agent_id, sorted(result.action_set), result.search_ms,
                )
            else:
                logger.info(
                    "[ComposioToolService] SDK search returned empty for agent=%s "
                    "prompt=%r search_ms=%d — will fall back to hints",
                    agent_id, task_prompt[:80], result.search_ms,
                )

        except Exception as exc:
            logger.warning(
                "[ComposioToolService] SDK search failed for agent=%s: %s",
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
        Execute a Composio action directly — bypasses the composio_execute
        mega-tool and its validation/auto-mapping pipeline.

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

    def _resolve_allowed_apps(self, agent_id: int, workspace_id: UUID) -> List[str]:
        """
        Query AgentAppAssignment for EXTERNAL active apps, intersect with
        workspace connections.  Mirrors ComposioHintService._resolve_allowed_apps().
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
                    if c.get("status") == "active"
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
