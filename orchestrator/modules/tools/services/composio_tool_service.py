"""
Composio Tool Service — Per-Action OpenAI Function-Calling Tools
================================================================

Resolves Composio actions into per-action OpenAI function-calling tool schemas.
Each action becomes its own top-level function with correct param names baked in
(e.g. ``JIRA_GET_ISSUE(issue_id_or_key="PILOT-123")``).

Strategy:
  1. Extract explicit action names from the prompt (e.g. GITHUB_CREATE_A_REFERENCE)
  2. Fetch exact schemas via cached per-app lookup (bypasses broken SDK semantic search)
  3. Fall back to SDK search only when no explicit actions found

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

    # Regex to extract explicit Composio action names from prompts.
    # Matches patterns like GITHUB_CREATE_A_REFERENCE, JIRA_GET_ISSUE, etc.
    _ACTION_NAME_RE = re.compile(r"\b([A-Z][A-Z0-9]+(?:_[A-Z0-9]+){2,})\b")

    # Map conversational verbs → Composio-friendly action queries per app domain.
    # Composio SDK search works best with action-oriented keywords.
    _QUERY_REWRITES: Dict[str, List[tuple]] = {
        "GMAIL": [
            (r"\b(check|read|view|see|show|get|pull up|look at)\b.*(email|mail|inbox)", "list emails fetch inbox messages"),
            (r"\b(send|compose|write|draft)\b.*(email|mail)", "send email create draft message"),
            (r"\b(reply|respond)\b.*(email|mail)", "reply to email send reply"),
            (r"\b(search|find)\b.*(email|mail)", "search emails find messages"),
            (r"\b(delete|remove|trash)\b.*(email|mail)", "delete email trash message"),
        ],
        "SLACK": [
            (r"\b(send|post|write)\b.*(message|slack|channel|dm)", "send message to channel"),
            (r"\b(read|check|see|get)\b.*(message|slack|channel)", "list messages fetch channel"),
            (r"\b(create)\b.*(channel)", "create channel"),
        ],
        "GOOGLE_CALENDAR": [
            (r"\b(check|view|see|show|get|what)\b.*(calendar|schedule|meeting|event)", "list events get calendar"),
            (r"\b(create|add|schedule|book)\b.*(meeting|event|appointment)", "create event schedule meeting"),
            (r"\b(cancel|delete|remove)\b.*(meeting|event)", "delete event cancel meeting"),
        ],
        "GITHUB": [
            (r"\b(create|open|make)\b.*(issue|bug|ticket)", "create issue"),
            (r"\b(create|open|make)\b.*(pull request|pr)", "create pull request"),
            (r"\b(list|show|get)\b.*(issue|pr|pull request)", "list issues pull requests"),
            (r"\b(merge)\b.*(pr|pull request)", "merge pull request"),
        ],
        "JIRA": [
            (r"\b(create|open|make)\b.*(ticket|issue|story|task)", "create issue ticket"),
            (r"\b(list|show|get|find)\b.*(ticket|issue|sprint|backlog)", "list issues search tickets"),
            (r"\b(update|move|assign)\b.*(ticket|issue)", "update issue transition"),
        ],
    }

    @classmethod
    def _rewrite_query(cls, prompt: str, app_names: List[str]) -> str:
        """
        Rewrite a conversational prompt into Composio-friendly action keywords.

        If the user says "check my emails for today", this returns
        "list emails fetch inbox messages" which matches Composio action names
        much better in semantic search.
        """
        prompt_lower = prompt.lower()
        for app in app_names:
            app_upper = app.upper()
            rewrites = cls._QUERY_REWRITES.get(app_upper, [])
            for pattern, replacement in rewrites:
                if re.search(pattern, prompt_lower, re.IGNORECASE):
                    logger.info(
                        "[ComposioToolService] Query rewrite for %s: %r → %r",
                        app_upper, prompt[:60], replacement,
                    )
                    return replacement
        # No rewrite matched — return original (truncated)
        return prompt[:200].strip()

    def get_tools_for_step(
        self,
        agent_id: int,
        workspace_id: UUID,
        task_prompt: str,
        limit: int = 8,
    ) -> ComposioToolResult:
        """
        Resolve Composio tools for a recipe step.

        Strategy:
          1. Extract explicit action names from the prompt (e.g. GITHUB_CREATE_A_REFERENCE).
          2. If found → fetch their exact schemas via cached per-app lookup.
             This bypasses the SDK's semantic search which returns alphabetical results.
          3. If no explicit names → fall back to SDK semantic search (best effort).

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
            app_prefixes = {a.upper() for a in allowed_apps}

            # 2. Resolve entity_id for this workspace
            entity_id = self._resolve_entity_id(workspace_id)
            if not entity_id:
                logger.warning(
                    "[ComposioToolService] No Composio entity for workspace=%s", workspace_id,
                )
                return result
            result.entity_id = entity_id

            client = get_composio_client()
            t0 = time.monotonic()

            # 3. Extract explicit action names from the prompt
            explicit_names = self._extract_action_names(task_prompt, app_prefixes)

            if explicit_names:
                # PRIMARY PATH: Direct schema lookup from per-app cache.
                # This gives the LLM exact parameter schemas for the actions
                # named in the prompt — no guessing.
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

                result.search_ms = int((time.monotonic() - t0) * 1000)

                if result.tools:
                    result.strategy = "exact_lookup"
                    logger.info(
                        "[ComposioToolService] Exact lookup OK: agent=%s "
                        "requested=%d resolved=%d actions=%s search_ms=%d",
                        agent_id, len(explicit_names), len(result.tools),
                        sorted(result.action_set), result.search_ms,
                    )
                else:
                    logger.warning(
                        "[ComposioToolService] Exact lookup returned 0/%d for agent=%s "
                        "— falling through to SDK search",
                        len(explicit_names), agent_id,
                    )

            # FALLBACK: SDK semantic search (when no explicit names or lookup empty)
            if not result.tools:
                search_query = self._rewrite_query(task_prompt, allowed_apps)
                logger.info(
                    "[ComposioToolService] SDK search fallback: query=%r (original=%r)",
                    search_query[:120], task_prompt[:60],
                )
                search_results = client.search_actions_for_step(
                    search_query=search_query,
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
                        "[ComposioToolService] SDK search OK: agent=%s actions=%s search_ms=%d",
                        agent_id, sorted(result.action_set), result.search_ms,
                    )
                else:
                    logger.info(
                        "[ComposioToolService] No tools resolved for agent=%s "
                        "prompt=%r search_ms=%d — will fall back to hints",
                        agent_id, task_prompt[:80], result.search_ms,
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

    @staticmethod
    def _action_names_to_search_query(action_names: Set[str]) -> str:
        """
        Convert explicit action names to a focused semantic search query.

        ``GITHUB_CREATE_A_REFERENCE`` → ``"create a reference"``
        ``JIRA_GET_ISSUE`` → ``"get issue"``

        Strips the app prefix and joins with spaces. This gives Composio's
        semantic search a clean signal instead of a 2000-char prompt dump.
        """
        terms = []
        for name in sorted(action_names):
            # Strip app prefix: GITHUB_CREATE_A_REFERENCE → CREATE_A_REFERENCE
            parts = name.split("_", 1)
            if len(parts) > 1:
                action_part = parts[1].lower().replace("_", " ")
            else:
                action_part = name.lower()
            terms.append(action_part)
        return " ".join(terms)

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
                # Accept active AND pending connections — the agent assignment
                # is the authorization; let the SDK fail gracefully if not truly connected.
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
