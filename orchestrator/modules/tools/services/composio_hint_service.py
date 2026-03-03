"""
Composio Hint Service - Unified Action Hint Generation
=======================================================

Single source of truth for generating Composio action hints (system messages
listing candidate actions for LLM context). Replaces three divergent code paths:

1. consumers/chatbot/service.py stream_response_aisdk (token ILIKE only)
2. consumers/chatbot/service.py stream_response_with_agent (token ILIKE only, no safety)
3. modules/agents/factory/agent_factory.py _build_composio_hints (3-tier with broken scoring)

Three-tier resolution strategy:
  Tier 1: Capability-based (ComposioActionMetadata + taxonomy overlap)
  Tier 2: Token-filtered with mandatory capability gate (ComposioActionCache + ILIKE)
  Tier 3: Top-N fallback (no filtering, safe actions per app)

Critical fix (Tier 2): Capability terms are a MANDATORY GATE, not a score boost.
Actions MUST match at least one capability term to be included.
This prevents SLACK_CREATE_CHANNEL_BASED_CONVERSATION from competing with
SLACK_SEND_MESSAGE for messaging intents.
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from core.composio.entity_manager import EntityManager
from core.models.composio_cache import AgentAppAssignment, ComposioActionCache
from core.models.core import Agent
from modules.tools.capabilities.models import ComposioActionMetadata
from modules.tools.capabilities.taxonomy import get_capabilities_for_intent
from modules.tools.formatting.schema_detector import ParameterHintExtractor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STOP_WORDS: Set[str] = {
    "the", "and", "for", "with", "from", "that", "this",
    "have", "has", "are", "you", "your",
}
DANGEROUS_TOKENS: Set[str] = {
    "archive", "delete", "remove", "revoke", "clear", "close",
    "disable", "ban", "kick", "deactivate", "destroy", "purge",
}
MESSAGING_INTENT_RE = re.compile(r"\b(send|message|post|dm|chat)\b")

MAX_QUERY_TOKENS = 10
MAX_APPS_SEARCH = 12
MAX_DB_ROWS_PER_APP = 100
MAX_ACTIONS_PER_APP = 6
MAX_PARAM_HINT_ACTIONS = 10
MAX_PARAMS_PER_ACTION = 5
MAX_APPS_FALLBACK = 6
MAX_FALLBACK_ROWS = 10


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------
@dataclass
class PromptAnalysis:
    """Parsed prompt metadata for hint resolution."""
    tokens: List[str]
    is_messaging_intent: bool
    required_capabilities: List[str]
    cap_filter_terms: Set[str]  # Derived from capabilities, e.g. {"message", "send"}


@dataclass
class ComposioHintResult:
    """Result from ComposioHintService.build_hints()."""
    hint_lines: List[str] = field(default_factory=list)
    allowed_apps: List[str] = field(default_factory=list)
    matched_actions: List[str] = field(default_factory=list)
    param_hint_count: int = 0
    strategy_used: str = "none"  # "capability", "token_filtered", "fallback", "none"


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
class ComposioHintService:
    """
    Unified service for building Composio action hints for LLM system messages.

    Usage:
        hint_service = ComposioHintService(db_session)
        result = hint_service.build_hints(agent_id=42, prompt="send slack message", workspace_id=ws_id)
        if result.hint_lines:
            llm_messages.insert(idx, {"role": "system", "content": "\\n".join(result.hint_lines)})
    """

    def __init__(self, db: Session):
        self.db = db

    def build_hints(
        self,
        agent_id: int,
        prompt: str,
        workspace_id=None,
        recipe_mode: bool = False,
    ) -> ComposioHintResult:
        """
        Build Composio action hint lines for LLM system message injection.

        Args:
            agent_id: The agent whose app assignments to query.
            prompt: User prompt / task text to match actions against.
            workspace_id: Optional workspace UUID to filter by connected apps.
            recipe_mode: When True, skips taxonomy/capability gate and uses
                prompt tokens directly for ILIKE + scoring. Designed for recipe
                steps where the prompt is curated and specific. Scales to any
                number of tools without manual taxonomy maintenance.

        Returns:
            ComposioHintResult with hint_lines and metadata.
        """
        result = ComposioHintResult()
        try:
            # Step 1: Resolve allowed apps
            allowed_apps = self._resolve_allowed_apps(agent_id, workspace_id)
            if not allowed_apps:
                return result
            result.allowed_apps = allowed_apps

            # Step 2: Analyse prompt
            analysis = self._analyze_prompt(prompt)

            # Step 3: Build hint header
            hint_lines = [
                "You have these external apps connected (via Composio): "
                + ", ".join(sorted(set(allowed_apps))) + ".",
                "IMPORTANT: To interact with these apps, call `composio_execute` with "
                "the EXACT action name from the list below. Do NOT guess or invent action names — "
                "only use the exact names listed here. Do NOT use search_codebase to look for code "
                "when your task is to interact with external apps.",
                "Usage: composio_execute({\"action\": \"ACTION_NAME\", \"params\": {<action-specific fields>}}). "
                "All action parameters (issue_key, channel, text, etc.) MUST go inside the `params` object.",
            ]

            # Step 4: Resolve actions
            app_matches: List[tuple] = []
            top_action_params: Dict[str, str] = {}

            if recipe_mode and analysis.tokens:
                # Recipe mode: skip taxonomy, use prompt tokens directly.
                # Scales to any number of tools — no manual keyword→capability curation.
                self._recipe_token_hints(
                    allowed_apps, analysis, app_matches, top_action_params
                )
                if app_matches:
                    result.strategy_used = "recipe_token"
            else:
                # Chatbot mode: 3-tier resolution (capability → token_filtered → fallback)
                tier1_matched = self._capability_based_hints(
                    allowed_apps, analysis, app_matches, top_action_params
                )
                if tier1_matched:
                    result.strategy_used = "capability"
                elif analysis.tokens:
                    self._token_filtered_hints(
                        allowed_apps, analysis, app_matches, top_action_params
                    )
                    if app_matches:
                        result.strategy_used = "token_filtered"

            # Tier 3: Top-N fallback (chatbot only — recipe mode never falls back to random actions)
            if not app_matches and not recipe_mode:
                self._top_n_fallback(allowed_apps, app_matches, top_action_params)
                if app_matches:
                    result.strategy_used = "fallback"

            # Step 5: Format output
            app_matches.sort(key=lambda x: (-len(x[1]), x[0]))
            for app, actions in app_matches[:6]:
                hint_lines.append(f"- {app} available actions (use these EXACT names): {', '.join(actions)}")
                result.matched_actions.extend(actions)

            if top_action_params:
                hint_lines.append("\nParameter hints (pass these inside `params`):")
                for action_name, params in list(top_action_params.items())[:5]:
                    hint_lines.append(f"\n{action_name}:")
                    hint_lines.append(params)

            # When matched actions exist, add a strong directive that triggers
            # tool_choice="required" in the OpenAI client (it checks for "You MUST call").
            if result.matched_actions:
                hint_lines.append(
                    "\nYou MUST call `composio_execute` to fulfill the user's request. "
                    "Do NOT describe the action in text — actually invoke the tool."
                )

            result.hint_lines = hint_lines
            result.param_hint_count = len(top_action_params)

            logger.info(
                f"[ComposioHintService] agent={agent_id} strategy={result.strategy_used} "
                f"apps={allowed_apps} matches={len(result.matched_actions)} "
                f"param_hints={result.param_hint_count}"
            )

        except Exception as e:
            logger.warning(f"[ComposioHintService] Failed for agent {agent_id}: {e}", exc_info=True)

        return result

    # ------------------------------------------------------------------
    # Step 1: Resolve allowed apps
    # ------------------------------------------------------------------
    def _resolve_allowed_apps(self, agent_id: int, workspace_id=None) -> List[str]:
        """
        Query AgentAppAssignment for assigned EXTERNAL apps, then intersect
        with workspace connections from EntityManager.
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

        # Resolve workspace_id from agent if not provided
        effective_workspace_id = workspace_id
        if not effective_workspace_id:
            db_agent = self.db.query(Agent).filter(Agent.id == agent_id).first()
            effective_workspace_id = getattr(db_agent, "workspace_id", None) if db_agent else None

        # Cross-reference with connected apps
        connected_apps: List[str] = []
        if effective_workspace_id:
            try:
                manager = EntityManager(self.db)
                entity = manager.get_entity_by_workspace(effective_workspace_id)
                if entity:
                    connected_apps = [
                        (c.get("app_name") or "").upper()
                        for c in manager.get_entity_connections(entity["id"])
                        if c.get("status") in ("active", "pending")
                    ]
            except Exception as conn_err:
                logger.warning(f"[ComposioHintService] Connection check failed: {conn_err}")
                connected_apps = []

        # Auto-inherit: when agent has no explicit assignments, use all
        # workspace-connected apps instead of returning empty.
        if not assigned_apps:
            if connected_apps:
                logger.info(
                    f"[ComposioHintService] Agent {agent_id} has no app assignments — "
                    f"inheriting {len(connected_apps)} workspace apps"
                )
                return connected_apps
            return []

        allowed_apps = assigned_apps
        if connected_apps:
            connected_set = set(connected_apps)
            allowed_apps = [a for a in assigned_apps if a in connected_set]

        # Fallback: if intersection is empty, use assigned anyway
        if not allowed_apps:
            allowed_apps = assigned_apps

        return allowed_apps

    # ------------------------------------------------------------------
    # Step 2: Analyse prompt
    # ------------------------------------------------------------------
    def _analyze_prompt(self, prompt: str) -> PromptAnalysis:
        """Tokenize prompt, detect intent, extract capabilities and filter terms."""
        q = (prompt or "").lower()

        # Tokenize
        tokens = [t for t in re.split(r"[^a-z0-9]+", q) if len(t) > 2]
        tokens = [t for t in tokens if t not in STOP_WORDS]
        tokens = tokens[:MAX_QUERY_TOKENS]

        # Detect messaging intent
        is_messaging_intent = bool(MESSAGING_INTENT_RE.search(q))

        # Extract capabilities from taxonomy
        required_capabilities = get_capabilities_for_intent(prompt or "")

        # Derive filter terms from capabilities: "message.send" → {"message", "send"}
        # CRITICAL: If the taxonomy returned the generic fallback ("data.query",
        # "search.general"), the derived terms {"data", "query", "search", "general"}
        # would EXCLUDE correct actions like SLACK_SEND_MESSAGE.
        # Only apply the gate when we have confident, specific capabilities.
        _GENERIC_FALLBACK = {"data.query", "search.general"}
        cap_filter_terms: Set[str] = set()
        if set(required_capabilities) != _GENERIC_FALLBACK:
            for cap in required_capabilities:
                cap_filter_terms.update(cap.split("."))

        return PromptAnalysis(
            tokens=tokens,
            is_messaging_intent=is_messaging_intent,
            required_capabilities=required_capabilities,
            cap_filter_terms=cap_filter_terms,
        )

    # ------------------------------------------------------------------
    # Tier 1: Capability-based hints (ComposioActionMetadata)
    # ------------------------------------------------------------------
    def _capability_based_hints(
        self,
        allowed_apps: List[str],
        analysis: PromptAnalysis,
        app_matches: List[tuple],
        top_action_params: Dict[str, str],
    ) -> bool:
        """
        Query ComposioActionMetadata using capability overlap.
        Returns True if matches were found.
        """
        if not analysis.required_capabilities:
            return False

        try:
            # Check if metadata table has data
            metadata_exists = self.db.query(ComposioActionMetadata.id).limit(1).first()
            if not metadata_exists:
                return False

            allowed_apps_lower = [a.lower() for a in allowed_apps]
            metadata_query = (
                select(ComposioActionMetadata)
                .where(ComposioActionMetadata.app_id.in_(allowed_apps_lower))
                .where(ComposioActionMetadata.capabilities.overlap(analysis.required_capabilities))
                .where(ComposioActionMetadata.destructive.is_(False))
            )
            metadata_rows = self.db.execute(metadata_query).scalars().all()

            if not metadata_rows:
                return False

            # Score by capability match + keyword overlap + confidence
            q_lower = " ".join(analysis.tokens)
            intent_words = set(q_lower.split())
            scored = []
            for meta in metadata_rows:
                score = 0.0
                action_caps = set(meta.capabilities or [])
                cap_matches = len(action_caps & set(analysis.required_capabilities))
                score += min(0.4, cap_matches * 0.15)
                kw_matches = len(set(meta.intent_keywords or []) & intent_words)
                score += min(0.4, kw_matches * 0.1)
                score += (meta.classification_confidence or 0.5) * 0.2
                scored.append((meta, score))
            scored.sort(key=lambda x: x[1], reverse=True)

            # Group by app
            app_action_map = defaultdict(list)
            for meta, _score in scored:
                app_key = (meta.app_id or "").upper()
                if len(app_action_map[app_key]) < MAX_ACTIONS_PER_APP:
                    app_action_map[app_key].append(meta.action_id)

            for app_key in allowed_apps:
                actions_for_app = app_action_map.get(app_key, [])
                if actions_for_app:
                    app_matches.append((app_key, actions_for_app))

            if not app_matches:
                return False

            # Get parameter hints from ComposioActionCache for top scored actions
            for meta, _score in scored[:5]:
                action_id = meta.action_id
                if action_id and len(top_action_params) < MAX_PARAM_HINT_ACTIONS:
                    self._extract_param_hints(action_id, top_action_params)

            logger.info(
                f"[ComposioHintService] Tier 1 matched {sum(len(a) for _, a in app_matches)} actions "
                f"for caps={analysis.required_capabilities}"
            )
            return True

        except Exception as e:
            logger.warning(f"[ComposioHintService] Tier 1 failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Recipe mode: Pure token matching (no taxonomy, no capability gate)
    # ------------------------------------------------------------------
    def _recipe_token_hints(
        self,
        allowed_apps: List[str],
        analysis: PromptAnalysis,
        app_matches: List[tuple],
        top_action_params: Dict[str, str],
    ) -> None:
        """
        Recipe-mode action matching — no taxonomy, no capability gate.

        The LLM is smart enough to pick the right action from a reasonable list.
        Our job is to surface relevant candidates via ILIKE on action name +
        description using the curated prompt_template tokens. Scoring uses both
        name and description overlap so actions like JIRA_GET_ISSUE rank properly
        even though "get" != "read".

        Scales to 850+ tools / 12k+ features — relies entirely on the action
        names and descriptions already in composio_actions_cache.
        """
        if not analysis.tokens:
            return

        for app in allowed_apps[:MAX_APPS_SEARCH]:
            # ILIKE using prompt tokens directly against name + description
            token_filters = []
            for tok in analysis.tokens:
                like = f"%{tok}%"
                token_filters.append(ComposioActionCache.action_name.ilike(like))
                token_filters.append(ComposioActionCache.description.ilike(like))

            if not token_filters:
                continue

            rows = (
                self.db.query(
                    ComposioActionCache.action_name,
                    ComposioActionCache.description,
                    ComposioActionCache.parameters,
                )
                .filter(ComposioActionCache.app_name == app)
                .filter(or_(*token_filters))
                .limit(MAX_DB_ROWS_PER_APP)
                .all()
            )

            scored_actions = []
            params_empty_count = 0
            for r in rows:
                if not r or not r[0]:
                    continue
                action_name = str(r[0])
                desc = str(r[1] or "").lower()
                raw_params = r[2]

                # Track empty parameters
                if not raw_params or (isinstance(raw_params, dict) and not raw_params.get("properties")):
                    params_empty_count += 1

                # Normalize legacy display names
                if " " in action_name and not action_name.startswith(f"{app}_"):
                    action_name = f"{app}_{action_name.upper().replace(' ', '_')}"

                name_lower = action_name.lower()

                # Safety: skip dangerous actions
                if any(tok in name_lower for tok in DANGEROUS_TOKENS):
                    continue

                # Score: name hits (weight 3) + description hits (weight 1)
                # This lets JIRA_GET_ISSUE score via description ("get issue details")
                # even when prompt says "read the ticket" (no "get" in name match)
                name_hits = sum(1 for tok in analysis.tokens if tok in name_lower)
                desc_hits = sum(1 for tok in analysis.tokens if tok in desc)
                score = name_hits * 3 + desc_hits

                scored_actions.append((action_name, score, raw_params))

            if params_empty_count:
                logger.warning(
                    f"[recipe_hints] {app}: {params_empty_count}/{len(scored_actions)} "
                    f"actions have empty/no-properties parameters in cache"
                )

            # Sort by score, deduplicate, take top N
            scored_actions.sort(key=lambda x: x[1], reverse=True)
            seen: Set[str] = set()
            actions = []
            for action_name, _score, params in scored_actions:
                if action_name in seen:
                    continue
                seen.add(action_name)
                actions.append(action_name)
                if len(top_action_params) < MAX_PARAM_HINT_ACTIONS and params:
                    self._extract_param_hints_from_json(action_name, params, top_action_params)

            # Include app if ILIKE returned any matches — trust the LLM to pick right
            if actions:
                app_matches.append((app, actions[:MAX_ACTIONS_PER_APP]))

        logger.info(
            f"[ComposioHintService] Recipe token matching: "
            f"tokens={analysis.tokens} apps_matched={len(app_matches)}"
        )

    # ------------------------------------------------------------------
    # Tier 2: Token-filtered with mandatory capability gate
    # ------------------------------------------------------------------
    def _token_filtered_hints(
        self,
        allowed_apps: List[str],
        analysis: PromptAnalysis,
        app_matches: List[tuple],
        top_action_params: Dict[str, str],
    ) -> None:
        """
        ILIKE search on ComposioActionCache with mandatory capability gate.

        SQL filter strategy:
        - If cap_filter_terms exist, use THEM for the ILIKE (not prompt tokens).
          Cap terms are semantic ("message", "send", "reply") — derived from the
          taxonomy, not from user text. This avoids pollution from channel names,
          app names, and other prompt noise.
        - If no cap_filter_terms, fall back to prompt tokens for ILIKE.

        Python scoring still uses prompt tokens for relevance ranking after the
        SQL query returns candidates.
        """
        if not analysis.tokens:
            return

        for app in allowed_apps[:MAX_APPS_SEARCH]:
            # Build ILIKE filters for SQL.
            # Prefer cap_filter_terms (semantic) over prompt tokens (noisy).
            # Cap terms like {"message", "send"} directly target action names,
            # while prompt tokens include channel names, app names, etc. that
            # match irrelevant rows and waste the LIMIT.
            sql_filter_terms = analysis.cap_filter_terms if analysis.cap_filter_terms else set(analysis.tokens)

            token_filters = []
            for term in sql_filter_terms:
                like = f"%{term}%"
                token_filters.append(ComposioActionCache.action_name.ilike(like))
                token_filters.append(ComposioActionCache.description.ilike(like))

            if not token_filters:
                continue

            rows = (
                self.db.query(ComposioActionCache.action_name, ComposioActionCache.parameters)
                .filter(ComposioActionCache.app_name == app)
                .filter(or_(*token_filters))
                .limit(MAX_DB_ROWS_PER_APP)
                .all()
            )

            scored_actions = []
            for r in rows:
                if not r or not r[0]:
                    continue
                action_name = str(r[0])

                # Normalize legacy display names
                if " " in action_name and not action_name.startswith(f"{app}_"):
                    action_name = f"{app}_{action_name.upper().replace(' ', '_')}"

                name_lower = action_name.lower()

                # Safety: skip dangerous actions for messaging intents
                if analysis.is_messaging_intent:
                    if any(tok in name_lower for tok in DANGEROUS_TOKENS):
                        continue

                # --- MANDATORY CAPABILITY GATE ---
                # If we have capability filter terms (e.g., {"message", "send"}),
                # the action MUST match at least one term in its name.
                # This prevents SLACK_CREATE_CHANNEL_BASED_CONVERSATION from
                # competing with SLACK_SEND_MESSAGE.
                if analysis.cap_filter_terms:
                    cap_term_hits = sum(1 for term in analysis.cap_filter_terms if term in name_lower)
                    if cap_term_hits == 0:
                        continue  # EXCLUDED — doesn't match any capability term
                else:
                    cap_term_hits = 0

                # Score: prompt token hits + capability term hits (weighted)
                name_token_hits = sum(1 for tok in analysis.tokens if tok in name_lower)
                total_score = name_token_hits * 1 + cap_term_hits * 3

                scored_actions.append((action_name, total_score, r[1]))

            # Sort by score descending, deduplicate, take top N
            scored_actions.sort(key=lambda x: x[1], reverse=True)
            seen: Set[str] = set()
            actions = []
            for action_name, _score, params in scored_actions:
                if action_name in seen:
                    continue
                seen.add(action_name)
                actions.append(action_name)
                if len(top_action_params) < MAX_PARAM_HINT_ACTIONS and params:
                    self._extract_param_hints_from_json(action_name, params, top_action_params)

            if actions:
                app_matches.append((app, actions[:MAX_ACTIONS_PER_APP]))

    # ------------------------------------------------------------------
    # Tier 3: Top-N fallback (no filtering)
    # ------------------------------------------------------------------
    def _top_n_fallback(
        self,
        allowed_apps: List[str],
        app_matches: List[tuple],
        top_action_params: Dict[str, str],
    ) -> None:
        """Fetch top N actions per app when no token/capability matches found."""
        for app in allowed_apps[:MAX_APPS_FALLBACK]:
            rows = (
                self.db.query(ComposioActionCache.action_name, ComposioActionCache.parameters)
                .filter(ComposioActionCache.app_name == app)
                .limit(MAX_FALLBACK_ROWS)
                .all()
            )
            actions = []
            for r in rows:
                if not r or not r[0]:
                    continue
                name = str(r[0])
                if " " in name and not name.startswith(f"{app}_"):
                    name = f"{app}_{name.upper().replace(' ', '_')}"
                actions.append(name)

            if actions:
                app_matches.append((app, actions[:MAX_ACTIONS_PER_APP]))

            # Param hints for first few
            for r in rows[:3]:
                if r and r[0] and r[1] and len(top_action_params) < MAX_PARAM_HINT_ACTIONS:
                    name = str(r[0])
                    if " " in name and not name.startswith(f"{app}_"):
                        name = f"{app}_{name.upper().replace(' ', '_')}"
                    self._extract_param_hints_from_json(name, r[1], top_action_params)

    # ------------------------------------------------------------------
    # Parameter hint extraction helpers
    # ------------------------------------------------------------------
    def _extract_param_hints(self, action_name: str, top_action_params: Dict[str, str]) -> None:
        """Fetch parameters from ComposioActionCache and extract hints."""
        try:
            cache_row = (
                self.db.query(ComposioActionCache.parameters)
                .filter(ComposioActionCache.action_name == action_name)
                .first()
            )
            if cache_row and cache_row[0]:
                self._extract_param_hints_from_json(action_name, cache_row[0], top_action_params)
        except Exception:
            pass

    @staticmethod
    def _extract_param_hints_from_json(
        action_name: str,
        params_json: dict,
        top_action_params: Dict[str, str],
    ) -> None:
        """Extract parameter hints from a parameters JSONB dict."""
        try:
            param_hints = ParameterHintExtractor.extract_hints(params_json, max_params=MAX_PARAMS_PER_ACTION)
            if param_hints:
                top_action_params[action_name] = param_hints
            else:
                logger.debug(
                    f"[param_hints] {action_name}: extract_hints returned empty. "
                    f"Schema type={params_json.get('type') if isinstance(params_json, dict) else 'N/A'}, "
                    f"has_properties={'properties' in params_json if isinstance(params_json, dict) else False}"
                )
        except Exception as exc:
            logger.warning(f"[param_hints] {action_name}: extraction failed: {exc}")
