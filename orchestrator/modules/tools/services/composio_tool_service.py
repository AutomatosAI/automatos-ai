"""
Composio Tool Service — Per-Action OpenAI Function-Calling Tools
================================================================

Resolves Composio actions into per-action OpenAI function-calling tool schemas.
Each action becomes its own top-level function with correct param names baked in
(e.g. ``JIRA_GET_ISSUE(issue_id_or_key="PILOT-123")``).

Strategy:
  1. Extract explicit action names from the prompt (e.g. GITHUB_CREATE_A_REFERENCE)
     → fetch exact schemas from per-app cache.
  2. No explicit names → SDK semantic search (tools.get with search=).
  3. SDK search returns 0 → load from SDK cache, sort by relevance, cap at limit.

The SDK cache is populated via ``toolset.tools.get(toolkits=[app])`` which
fetches all actions for a toolkit. Semantic search uses the same SDK endpoint
with the ``search=`` parameter.

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

# Max tools to return — keeps tool input under ~18K tokens (30 × ~600 each).
_MAX_TOOLS = 30


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
    strategy: str = "none"  # "exact_lookup" | "sdk_search" | "cache_ranked" | "none"
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
    _ACTION_NAME_RE = re.compile(r"\b([A-Z][A-Z0-9]+(?:_[A-Z0-9]+){2,})\b")

    def get_tools_for_step(
        self,
        agent_id: int,
        workspace_id: UUID,
        task_prompt: str,
        limit: int = _MAX_TOOLS,
    ) -> ComposioToolResult:
        """
        Resolve Composio tools for a step.

        Strategy:
          1. Explicit action names in prompt → exact schema lookup.
          2. SDK semantic search (tools.get with search=query).
          3. SDK search returns 0 → load from cache, rank by keyword
             match on action name, cap at ``limit``.
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

            # 4. SDK semantic search
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

            if len(result.tools) >= 5:
                # SDK returned enough results — use them directly
                result.search_ms = int((time.monotonic() - t0) * 1000)
                result.strategy = "sdk_search"
                logger.info(
                    "[ComposioToolService] SDK search: agent=%s actions=%d (%dms) → %s",
                    agent_id, len(result.tools), result.search_ms,
                    sorted(result.action_set),
                )
                return result

            # SDK returned too few results (<5) — supplement with cache_ranked.
            # Keep whatever SDK found but pad with keyword-ranked cached actions.
            if result.tools:
                logger.info(
                    "[ComposioToolService] SDK search sparse (%d actions) — "
                    "supplementing with cache_ranked for agent=%s",
                    len(result.tools), agent_id,
                )

            # 5. SDK search returned 0 or too few — fall back to cached SDK data.
            #    The cache was populated by tools.get() (the SDK).
            #    Rank actions by keyword match on action name, cap at limit.
            all_schemas = client.get_all_schemas_for_apps(
                app_names=[a.lower() for a in allowed_apps],
                entity_id=entity_id,
            )
            if all_schemas:
                query_words = set(re.findall(r"[a-z]{3,}", task_prompt.lower()))
                query_lower = task_prompt.lower()

                # Map query keywords to likely app prefixes for tiebreaking
                _KEYWORD_TO_APP = {
                    "email": "GMAIL", "emails": "GMAIL", "inbox": "GMAIL",
                    "mail": "GMAIL", "gmail": "GMAIL",
                    "slack": "SLACK", "message": "SLACK", "channel": "SLACK",
                    "calendar": "GOOGLECALENDAR", "event": "GOOGLECALENDAR",
                    "meeting": "GOOGLECALENDAR", "schedule": "GOOGLECALENDAR",
                    "drive": "GOOGLEDRIVE", "file": "GOOGLEDRIVE",
                    "sheet": "GOOGLESHEETS", "spreadsheet": "GOOGLESHEETS",
                    "jira": "JIRA", "ticket": "JIRA", "issue": "JIRA",
                    "github": "GITHUB", "repo": "GITHUB", "pull": "GITHUB",
                    "telegram": "TELEGRAM", "discord": "DISCORDBOT",
                    "doc": "GOOGLEDOCS", "document": "GOOGLEDOCS",
                    "dropbox": "DROPBOX",
                }
                preferred_apps = set()
                for word in query_words:
                    if word in _KEYWORD_TO_APP:
                        preferred_apps.add(_KEYWORD_TO_APP[word])

                def _rank_action(item):
                    name = item.get("action_name", "").lower()
                    # Score 1: keyword overlap
                    keyword_score = sum(1 for w in query_words if w in name)
                    # Score 2: app preference (actions from the right app rank higher)
                    app_prefix = item.get("action_name", "").split("_")[0]
                    app_bonus = 10 if app_prefix in preferred_apps else 0
                    return -(keyword_score + app_bonus)

                ranked = sorted(all_schemas, key=_rank_action)

                # Round-robin across apps to prevent one app monopolizing all slots
                from collections import defaultdict
                per_app: defaultdict = defaultdict(list)
                for item in ranked:
                    action_name = item.get("action_name", "")
                    schema = item.get("schema")
                    if action_name and schema and action_name not in result.action_set:
                        app_prefix = action_name.split("_")[0]
                        per_app[app_prefix].append((action_name, schema))

                # Interleave: take up to 5 per app in round-robin order
                max_per_app = max(5, limit // max(len(per_app), 1))
                added = 0
                round_idx = 0
                while added < limit:
                    any_added = False
                    for app in list(per_app.keys()):
                        items = per_app[app]
                        if round_idx < len(items) and round_idx < max_per_app:
                            action_name, schema = items[round_idx]
                            result.tools.append(schema)
                            result.action_set.add(action_name)
                            added += 1
                            any_added = True
                            if added >= limit:
                                break
                    round_idx += 1
                    if not any_added:
                        break

            result.search_ms = int((time.monotonic() - t0) * 1000)

            if result.tools:
                result.strategy = "cache_ranked"
                logger.info(
                    "[ComposioToolService] Cache ranked: agent=%s apps=%s "
                    "actions=%d/%d (%dms) → %s",
                    agent_id, allowed_apps, len(result.tools), limit,
                    result.search_ms, sorted(result.action_set)[:10],
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

        # Auto-inherit: no explicit assignments → use all workspace-connected apps
        if not assigned_apps:
            if connected_apps:
                logger.info(
                    "[ComposioToolService] Agent %s has no app assignments — "
                    "inheriting %d workspace apps", agent_id, len(connected_apps)
                )
                return connected_apps
            return []

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
