"""
Universal Router Engine (PRD-50: US-003, US-004).

Core routing engine that resolves a RequestEnvelope to a RoutingDecision
using a tiered strategy:

  Tier 0   — User override (agent_id or workflow_id explicitly set)
  Tier 1   — Cache lookup (RoutingCache hit)
  Tier 2a  — routing_rules table (workspace + explicit source pattern)
  Tier 2b  — TriggerSubscription (for jira_trigger source)
  Tier 2.5 — Semantic similarity (PRD-64: cosine sim on agent embeddings)
  Tier 2c  — IntentClassifier keyword matching against routing rules
  Tier 3   — LLM classification (fallback, uses semantic candidates if available)

Returns None only when all tiers (including LLM) fail to route.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Dict, List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from core.llm.manager import create_llm_manager
from core.models.composio import ComposioEntity, TriggerSubscription
from core.models.composio_cache import AgentAppAssignment
from core.models.core import Agent
from core.models.routing import (
    ChannelSource,
    RequestEnvelope,
    RoutingDecision,
    RoutingDecisionRecord,
    RoutingRule,
    UnroutedEvent,
)
from core.routing.cache import RoutingCache, _normalize_content
from core.services.intent_classifier import IntentClassifier

_LLM_CONFIDENCE_THRESHOLD = float(
    os.getenv("ROUTING_LLM_CONFIDENCE_THRESHOLD", "0.5")
)

logger = logging.getLogger(__name__)


def _envelope_hash(envelope: RequestEnvelope) -> str:
    """Compute a short hash for logging/deduplication."""
    raw = (_normalize_content(envelope.content) + "|" + envelope.source.value)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class UniversalRouter:
    """Tiered routing engine for the universal orchestrator.

    Parameters
    ----------
    db : Session
        SQLAlchemy database session for querying rules/subscriptions.
    cache : RoutingCache | None
        Optional routing cache instance.  When provided, Tier 1 (cache)
        is active.
    """

    def __init__(self, db: Session, cache: Optional[RoutingCache] = None) -> None:
        self._db = db
        self._cache = cache
        self._intent_classifier = IntentClassifier()
        # PRD-64: Semantic candidates from Tier 2.5 (used to narrow Tier 3)
        self._semantic_candidates: list = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def route(self, envelope: RequestEnvelope) -> Optional[RoutingDecision]:
        """Route *envelope* through the tier chain.

        Returns a ``RoutingDecision`` if a match is found at any tier,
        or ``None`` when no tier can resolve the request (the caller
        should fall back to LLM classification or default agent).
        """

        # PRD-64: Reset semantic candidates each routing pass
        self._semantic_candidates = []

        env_hash = _envelope_hash(envelope)
        logger.info(
            "[router] Routing request %s source=%s workspace=%s",
            env_hash,
            envelope.source.value,
            envelope.workspace_id,
        )

        # Tier 0 — explicit user overrides
        decision = self._tier0_override(envelope)
        if decision is not None:
            logger.info("[router] Tier 0 hit (override): %s", decision.reasoning)
            self._log_decision(envelope, decision, env_hash)
            return decision

        # Tier 1 — cache lookup
        decision = self._tier1_cache(envelope)
        if decision is not None:
            logger.info("[router] Tier 1 hit (cache): agent_id=%s", decision.agent_id)
            self._log_decision(envelope, decision, env_hash)
            return decision

        # Tier 2a — routing rules (source pattern match)
        decision = self._tier2a_rules(envelope)
        if decision is not None:
            logger.info("[router] Tier 2a hit (rule): %s", decision.reasoning)
            self._log_decision(envelope, decision, env_hash)
            return decision

        # Tier 2b — TriggerSubscription (jira_trigger)
        decision = self._tier2b_trigger_subscription(envelope)
        if decision is not None:
            logger.info("[router] Tier 2b hit (trigger): %s", decision.reasoning)
            self._log_decision(envelope, decision, env_hash)
            return decision

        # Tier 2.5 — Semantic similarity (PRD-64)
        # Runs BEFORE Tier 2c (intent keywords) because semantic matching
        # understands agent capabilities, while keyword matching is coarse
        # and easily hijacked by overly broad rules.
        decision = await self._tier2_5_semantic(envelope)
        if decision is not None:
            logger.info(
                "[router] Tier 2.5 hit (semantic): agent_id=%s confidence=%.2f",
                decision.agent_id, decision.confidence,
            )
            self._log_decision(envelope, decision, env_hash)
            return decision

        # Tier 2c — IntentClassifier keyword matching against routing rules
        # Skip if Tier 2.5 already found semantic candidates — those go
        # straight to Tier 3 (LLM) which is smarter than keyword matching.
        if not self._semantic_candidates:
            decision = self._tier2c_intent_classifier(envelope)
            if decision is not None:
                logger.info("[router] Tier 2c hit (intent): %s", decision.reasoning)
                self._log_decision(envelope, decision, env_hash)
                return decision

        # Tier 3 — LLM classification (fallback)
        decision = await self._classify_with_llm(envelope)
        if decision is not None:
            logger.info(
                "[router] Tier 3 hit (LLM): route_type=%s agent_id=%s confidence=%.2f",
                decision.route_type,
                decision.agent_id,
                decision.confidence,
            )
            self._log_decision(envelope, decision, env_hash)
            return decision

        # All tiers exhausted — store as unrouted event
        logger.info("[router] No route found for request %s — storing unrouted event", env_hash)
        self._store_unrouted_event(envelope, reason="All routing tiers exhausted (including LLM)")
        return None

    # ------------------------------------------------------------------
    # Tier 0 — User overrides
    # ------------------------------------------------------------------

    def _tier0_override(self, envelope: RequestEnvelope) -> Optional[RoutingDecision]:
        if envelope.override_agent_id is not None:
            return RoutingDecision(
                route_type="agent",
                agent_id=envelope.override_agent_id,
                confidence=1.0,
                reasoning="User override",
            )
        if envelope.override_workflow_id is not None:
            return RoutingDecision(
                route_type="workflow",
                workflow_id=envelope.override_workflow_id,
                confidence=1.0,
                reasoning="User override",
            )
        return None

    # ------------------------------------------------------------------
    # Tier 1 — Cache lookup
    # ------------------------------------------------------------------

    def _tier1_cache(self, envelope: RequestEnvelope) -> Optional[RoutingDecision]:
        if self._cache is None:
            return None
        return self._cache.get(
            envelope.workspace_id, envelope.content, envelope.source
        )

    # ------------------------------------------------------------------
    # Tier 2a — Routing rules (source pattern)
    # ------------------------------------------------------------------

    def _tier2a_rules(self, envelope: RequestEnvelope) -> Optional[RoutingDecision]:
        """Match against routing_rules that have an explicit source_pattern.

        Rules with ``source_pattern=None`` are NOT matched here — they are
        catch-all rules that should only fire via Tier 2c (intent keywords).
        This prevents a single rule from hijacking ALL requests regardless of
        content (the bug that routed everything to Document Generation).
        """
        rules = (
            self._db.query(RoutingRule)
            .filter(
                RoutingRule.workspace_id == envelope.workspace_id,
                RoutingRule.is_active.is_(True),
                RoutingRule.source_pattern.isnot(None),
            )
            .order_by(RoutingRule.priority.desc())
            .all()
        )

        logger.debug(
            "[router] Tier 2a: %d rules with source_pattern for workspace %s",
            len(rules), envelope.workspace_id,
        )

        for rule in rules:
            if rule.source_pattern != envelope.source.value:
                continue

            # Build decision from matching rule
            if rule.target_agent_id is not None:
                return RoutingDecision(
                    route_type="agent",
                    agent_id=rule.target_agent_id,
                    confidence=0.9,
                    reasoning=f"Matched routing rule #{rule.id} (source={rule.source_pattern})",
                )
            if rule.target_workflow_id is not None:
                return RoutingDecision(
                    route_type="workflow",
                    workflow_id=rule.target_workflow_id,
                    confidence=0.9,
                    reasoning=f"Matched routing rule #{rule.id} (source={rule.source_pattern})",
                )
        return None

    # ------------------------------------------------------------------
    # Tier 2b — Trigger subscription (jira_trigger)
    # ------------------------------------------------------------------

    def _tier2b_trigger_subscription(
        self, envelope: RequestEnvelope
    ) -> Optional[RoutingDecision]:
        """For jira_trigger source, check TriggerSubscription table."""
        if envelope.source != ChannelSource.JIRA_TRIGGER:
            return None

        # Resolve workspace_id → entity_id via ComposioEntity
        entity = (
            self._db.query(ComposioEntity)
            .filter(ComposioEntity.workspace_id == envelope.workspace_id)
            .first()
        )
        if entity is None:
            return None

        # Find active trigger subscription for this entity
        trigger_name = envelope.metadata.get("trigger_name", "")
        subscription = (
            self._db.query(TriggerSubscription)
            .filter(
                TriggerSubscription.entity_id == entity.id,
                TriggerSubscription.is_active.is_(True),
            )
            .first()
        )
        if subscription is None:
            return None

        # Optionally narrow to matching trigger_name if available
        if trigger_name and subscription.trigger_name != trigger_name:
            # Try finding an exact trigger_name match
            specific = (
                self._db.query(TriggerSubscription)
                .filter(
                    TriggerSubscription.entity_id == entity.id,
                    TriggerSubscription.trigger_name == trigger_name,
                    TriggerSubscription.is_active.is_(True),
                )
                .first()
            )
            if specific is not None:
                subscription = specific

        if subscription.agent_id is not None:
            return RoutingDecision(
                route_type="agent",
                agent_id=subscription.agent_id,
                confidence=0.95,
                reasoning=f"TriggerSubscription #{subscription.id} (trigger={subscription.trigger_name})",
            )
        if subscription.workflow_id is not None:
            return RoutingDecision(
                route_type="workflow",
                workflow_id=subscription.workflow_id,
                confidence=0.95,
                reasoning=f"TriggerSubscription #{subscription.id} (trigger={subscription.trigger_name})",
            )
        return None

    # ------------------------------------------------------------------
    # Tier 2c — IntentClassifier keyword match
    # ------------------------------------------------------------------

    def _tier2c_intent_classifier(
        self, envelope: RequestEnvelope
    ) -> Optional[RoutingDecision]:
        """Use IntentClassifier to classify the content, then match against
        routing rules that have intent_keywords."""
        classification = self._intent_classifier.classify(envelope.content)

        if classification.confidence < 0.4:
            return None

        # Check if any routing rule has intent_keywords matching the category
        rules = (
            self._db.query(RoutingRule)
            .filter(
                RoutingRule.workspace_id == envelope.workspace_id,
                RoutingRule.is_active.is_(True),
            )
            .order_by(RoutingRule.priority.desc())
            .all()
        )

        category_lower = classification.category.lower()
        for rule in rules:
            keywords = rule.intent_keywords or []
            # Match if any keyword in the rule matches the classified category
            if any(kw.lower() == category_lower for kw in keywords):
                if rule.target_agent_id is not None:
                    return RoutingDecision(
                        route_type="agent",
                        agent_id=rule.target_agent_id,
                        confidence=classification.confidence,
                        reasoning=f"Intent '{classification.category}' matched rule #{rule.id} keywords",
                        intent_category=classification.category,
                    )
                if rule.target_workflow_id is not None:
                    return RoutingDecision(
                        route_type="workflow",
                        workflow_id=rule.target_workflow_id,
                        confidence=classification.confidence,
                        reasoning=f"Intent '{classification.category}' matched rule #{rule.id} keywords",
                        intent_category=classification.category,
                    )
        return None

    # ------------------------------------------------------------------
    # Tier 2.5 — Semantic similarity (PRD-64)
    # ------------------------------------------------------------------

    async def _tier2_5_semantic(
        self, envelope: RequestEnvelope
    ) -> Optional[RoutingDecision]:
        """Use pre-computed agent embeddings to find the best match via cosine
        similarity.  High-confidence match routes directly; ambiguous results
        are stored as candidates for Tier 3 (narrowed LLM prompt)."""
        try:
            from core.routing.semantic_indexer import (
                find_similar_agents,
                SIMILARITY_DIRECT_ROUTE,
                MAX_LLM_CANDIDATES,
            )

            scored = await find_similar_agents(
                envelope.content, envelope.workspace_id, self._db
            )

            if not scored:
                # Check why — are there agents but none with embeddings?
                agent_count = (
                    self._db.query(Agent)
                    .filter(
                        Agent.workspace_id == envelope.workspace_id,
                        Agent.status == "active",
                    )
                    .count()
                )
                embedded_count = (
                    self._db.query(Agent)
                    .filter(
                        Agent.workspace_id == envelope.workspace_id,
                        Agent.status == "active",
                        Agent.semantic_embedding.isnot(None),
                    )
                    .count()
                )
                logger.warning(
                    "[router] Tier 2.5: no scored agents — "
                    "%d active agents, %d with embeddings in workspace %s",
                    agent_count, embedded_count, envelope.workspace_id,
                )
                return None

            # Log top scores for visibility
            top_3 = scored[:3]
            for agent, score in top_3:
                logger.info(
                    "[router] Tier 2.5 candidate: '%s' (id=%d) score=%.3f",
                    agent.name, agent.id, score,
                )

            top_agent, top_score = scored[0]

            if top_score >= SIMILARITY_DIRECT_ROUTE:
                decision = RoutingDecision(
                    route_type="agent",
                    agent_id=top_agent.id,
                    confidence=top_score,
                    reasoning=f"Semantic match '{top_agent.name}' (score={top_score:.2f})",
                )
                if self._cache is not None:
                    self._cache.put(
                        envelope.workspace_id,
                        envelope.content,
                        envelope.source,
                        decision,
                    )
                return decision

            # Below direct threshold — store top candidates for Tier 3
            self._semantic_candidates = [
                agent for agent, _score in scored[:MAX_LLM_CANDIDATES]
            ]
            logger.info(
                "[router] Tier 2.5: no direct match (top=%.2f), "
                "passing %d candidates to Tier 3",
                top_score, len(self._semantic_candidates),
            )
            return None

        except Exception:
            logger.exception("[router] Tier 2.5 failed — falling through")
            return None

    # ------------------------------------------------------------------
    # Tier 3 — LLM classification
    # ------------------------------------------------------------------

    async def _classify_with_llm(
        self, envelope: RequestEnvelope
    ) -> Optional[RoutingDecision]:
        """Use workspace-configured LLM to classify the request and select an agent.

        Queries all active agents in the workspace, builds a lightweight prompt,
        and parses the LLM response to extract agent_id and confidence.

        Returns a RoutingDecision (possibly with route_type='orchestrate' for
        low-confidence results) or None on failure.
        """
        try:
            # Always give the LLM ALL active agents — it's smart enough to
            # pick the right one from 14 agents.  Semantic pre-filtering was
            # removing correct agents before the LLM could see them.
            agents: List[Agent] = (
                self._db.query(Agent)
                .filter(
                    Agent.workspace_id == envelope.workspace_id,
                    Agent.status == "active",
                )
                .all()
            )
            if self._semantic_candidates:
                logger.info(
                    "[router] Tier 3: %d agents (semantic shortlisted %d)",
                    len(agents), len(self._semantic_candidates),
                )

            if not agents:
                logger.warning(
                    "[router] Tier 3: no active agents in workspace %s",
                    envelope.workspace_id,
                )
                return None

            # Build agent descriptions including assigned app names
            agent_descriptions = self._build_agent_descriptions(agents)

            # Log what the LLM will see
            for desc in agent_descriptions:
                apps_str = ", ".join(desc["apps"]) if desc["apps"] else "none"
                logger.info(
                    "[router] Tier 3 agent: id=%s name='%s' apps=[%s] desc='%s'",
                    desc["agent_id"], desc["name"], apps_str,
                    (desc["description"] or "")[:100],
                )

            # Build the classification prompt (pass semantic candidates as hints)
            prompt = self._build_classification_prompt(
                envelope.content,
                agent_descriptions,
                semantic_candidates=self._semantic_candidates or None,
            )

            # Call the LLM
            llm_manager = create_llm_manager(service_name="orchestrator")
            messages = [{"role": "user", "content": prompt}]
            response = await llm_manager.generate_response(messages)

            if not response or not response.content:
                logger.warning("[router] Tier 3: empty LLM response")
                return None

            logger.info(
                "[router] Tier 3: LLM response: %s",
                response.content[:300],
            )

            # Parse the response
            agent_id, confidence = self._parse_llm_routing_response(
                response.content, agents
            )

            if agent_id is None:
                logger.warning(
                    "[router] Tier 3: could not parse agent_id from LLM response: %s",
                    response.content[:300],
                )
                return None

            # Auto sentinel (id=0) → orchestrate with the LLM's own confidence
            if agent_id == 0:
                logger.info(
                    "[router] Tier 3: LLM selected Auto (orchestrate) confidence=%.2f",
                    confidence,
                )
                decision = RoutingDecision(
                    route_type="orchestrate",
                    agent_id=None,
                    confidence=confidence,
                    reasoning=f"LLM selected Auto - orchestrate (confidence={confidence:.2f})",
                )
                if self._cache is not None:
                    self._cache.put(
                        envelope.workspace_id,
                        envelope.content,
                        envelope.source,
                        decision,
                    )
                return decision

            # Low confidence → orchestrate (full decomposition needed)
            if confidence < _LLM_CONFIDENCE_THRESHOLD:
                logger.info(
                    "[router] Tier 3: confidence %.2f < threshold %.2f — returning orchestrate",
                    confidence,
                    _LLM_CONFIDENCE_THRESHOLD,
                )
                decision = RoutingDecision(
                    route_type="orchestrate",
                    agent_id=agent_id,
                    confidence=confidence,
                    reasoning=f"LLM classification below threshold ({confidence:.2f} < {_LLM_CONFIDENCE_THRESHOLD})",
                )
                # Still cache the low-confidence result so we don't re-call LLM
                if self._cache is not None:
                    self._cache.put(
                        envelope.workspace_id,
                        envelope.content,
                        envelope.source,
                        decision,
                    )
                return decision

            # High confidence → route to agent
            decision = RoutingDecision(
                route_type="agent",
                agent_id=agent_id,
                confidence=confidence,
                reasoning=f"LLM classification (confidence={confidence:.2f})",
            )

            # Cache the result for future Tier 1 hits
            if self._cache is not None:
                self._cache.put(
                    envelope.workspace_id,
                    envelope.content,
                    envelope.source,
                    decision,
                )

            return decision

        except Exception as exc:
            try:
                info = llm_manager.get_provider_info()
                logger.exception(
                    "[router] Tier 3: LLM classification failed — provider=%s model=%s error=%s",
                    info.get("provider"), info.get("model"), exc,
                )
            except Exception:
                logger.exception("[router] Tier 3: LLM classification failed")
            return None

    def _build_agent_descriptions(self, agents: List[Agent]) -> List[Dict]:
        """Build a list of agent info dicts for the LLM prompt.

        Includes a synthetic "Auto" entry (id=0) so the LLM can explicitly
        route back to the orchestrator when no specialized agent fits.
        """
        descriptions: List[Dict] = []
        for agent in agents:
            # Look up assigned app names
            app_assignments = (
                self._db.query(AgentAppAssignment)
                .filter(
                    AgentAppAssignment.agent_id == agent.id,
                    AgentAppAssignment.is_active.is_(True),
                )
                .all()
            )
            app_names = [a.app_name for a in app_assignments]

            descriptions.append(
                {
                    "agent_id": agent.id,
                    "name": agent.name,
                    "description": agent.description or "",
                    "apps": app_names,
                    "tags": agent.tags or [],
                }
            )

        # Synthetic "Auto" route — the orchestrator itself.
        # NOTE: By the time we reach Tier 3, AutoBrain has already filtered
        # greetings, platform queries, and memory recalls.  Auto should only
        # be selected here when genuinely NO specialized agent fits.
        descriptions.append(
            {
                "agent_id": 0,
                "name": "Auto",
                "description": (
                    "Fallback orchestrator. Only use when NO specialized agent "
                    "above is a reasonable match for the request."
                ),
                "apps": [],
                "tags": [],
            }
        )

        return descriptions

    @staticmethod
    def _build_classification_prompt(
        content: str,
        agent_descriptions: List[Dict],
        semantic_candidates: Optional[List] = None,
    ) -> str:
        """Build the routing prompt for the LLM.

        The prompt is biased toward specialized agents because AutoBrain
        already filtered greetings, platform queries, and memory recalls.
        Everything reaching Tier 3 is a real task that likely needs an agent.
        """
        agent_lines = []
        for desc in agent_descriptions:
            apps_str = ", ".join(desc["apps"]) if desc["apps"] else "none"
            tags_str = ", ".join(str(t) for t in desc.get("tags", []))
            tags_part = f", Tags: [{tags_str}]" if tags_str else ""
            agent_lines.append(
                f"  - ID: {desc['agent_id']}, Name: {desc['name']}, "
                f"Description: {desc['description']}, Apps: [{apps_str}]{tags_part}"
            )
        agents_block = "\n".join(agent_lines)

        # Semantic hints — give the LLM a head start from Tier 2.5
        semantic_hint = ""
        if semantic_candidates:
            names = [
                f"'{c.name}' (ID {c.id})"
                for c in semantic_candidates[:3]
            ]
            semantic_hint = (
                f"\nSemantic analysis suggests: {', '.join(names)}. "
                "Consider these first, but use your judgment based on the "
                "full agent list.\n"
            )

        # PRD-58: Try PromptRegistry (admin-editable), fallback to hardcoded
        try:
            from core.services.prompt_registry import prompt_registry
            custom = prompt_registry.get_raw("routing-classifier")
            if custom:
                return custom.format_map({
                    "message": content,
                    "available_routes": agents_block,
                })
        except Exception:
            pass

        return (
            "You are a request router for an AI platform. The user's message "
            "has already been screened — it is NOT a greeting, platform query, "
            "or trivial message. It needs a specialized agent.\n\n"
            f"User request: \"{content}\"\n\n"
            f"Available agents:\n{agents_block}\n"
            f"{semantic_hint}\n"
            "Rules:\n"
            "- Match the request to the agent whose description, tools, tags, "
            "or apps are most relevant.\n"
            "- Prefer a specialized agent. Only pick \"Auto\" (ID 0) if "
            "genuinely NO specialized agent fits.\n"
            "- High confidence (0.8-1.0) when description/tools clearly match.\n"
            "- Medium confidence (0.5-0.8) when it's a reasonable but imperfect fit.\n\n"
            "Respond with ONLY a JSON object (no markdown, no explanation):\n"
            '{"agent_id": <int>, "confidence": <float 0-1>}\n'
        )

    @staticmethod
    def _parse_llm_routing_response(
        response_text: str, agents: List[Agent]
    ) -> tuple[Optional[int], float]:
        """Parse the LLM response to extract agent_id and confidence.

        Resilient parser: extracts JSON from anywhere in the response,
        handles markdown fences, extra text, and alternative key names.

        Returns (agent_id, confidence) or (None, 0.0) on parse failure.
        """
        try:
            # Strip markdown code fences if present
            text = response_text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            # Try direct parse first
            parsed = None
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                # Extract first JSON object from the response text
                import re
                json_match = re.search(r'\{[^{}]*\}', text)
                if json_match:
                    parsed = json.loads(json_match.group())

            if not parsed or not isinstance(parsed, dict):
                return None, 0.0

            # Try multiple key names for agent_id
            raw_id = (
                parsed.get("agent_id")
                or parsed.get("agentId")
                or parsed.get("id")
                or parsed.get("agent")
            )

            # Fallback: LLM returned agent name instead of ID
            # e.g. {"route": "Communication", ...} or {"agent": "Research Analyst", ...}
            if raw_id is None or (isinstance(raw_id, str) and not raw_id.isdigit()):
                raw_name = (
                    raw_id  # "agent" key had a name string
                    or parsed.get("route")
                    or parsed.get("name")
                    or parsed.get("agent_name")
                )
                if isinstance(raw_name, str) and raw_name.strip():
                    name_lower = raw_name.strip().lower()
                    # Handle "Auto" virtual route by name
                    if name_lower == "auto":
                        raw_id = 0
                    else:
                        for a in agents:
                            if (a.name or "").lower() == name_lower:
                                raw_id = a.id
                                logger.info(
                                    "[router] Resolved agent name '%s' → id=%d",
                                    raw_name, a.id,
                                )
                                break
                        else:
                            # Fuzzy: check if the name is contained in agent name
                            for a in agents:
                                if name_lower in (a.name or "").lower():
                                    raw_id = a.id
                                    logger.info(
                                        "[router] Fuzzy-resolved agent name '%s' → '%s' id=%d",
                                        raw_name, a.name, a.id,
                                    )
                                    break

            if raw_id is None:
                return None, 0.0

            agent_id = int(raw_id)
            confidence = float(
                parsed.get("confidence")
                or parsed.get("score")
                or 0.7  # default if model omits confidence
            )

            # Auto sentinel (id=0) — valid, skip workspace validation
            if agent_id == 0:
                return 0, confidence

            # Validate agent_id is in the workspace agent list
            valid_ids = {a.id for a in agents}
            if agent_id not in valid_ids:
                logger.warning(
                    "[router] LLM returned agent_id=%d not in workspace agents %s",
                    agent_id,
                    valid_ids,
                )
                return None, 0.0

            # Clamp confidence to [0, 1]
            confidence = max(0.0, min(1.0, confidence))
            return agent_id, confidence

        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            logger.warning("[router] Failed to parse LLM routing response: %s", exc)
            return None, 0.0

    # ------------------------------------------------------------------
    # Unrouted event storage
    # ------------------------------------------------------------------

    def _store_unrouted_event(
        self, envelope: RequestEnvelope, reason: str
    ) -> None:
        """Persist an unrouted event for later analysis."""
        try:
            event = UnroutedEvent(
                workspace_id=envelope.workspace_id,
                source=envelope.source.value,
                content=envelope.content,
                raw_payload=envelope.raw_payload,
                reason=reason,
            )
            self._db.add(event)
            self._db.commit()
        except Exception:
            logger.exception("[router] Failed to store unrouted event")
            self._db.rollback()

    # ------------------------------------------------------------------
    # Decision logging
    # ------------------------------------------------------------------

    def _log_decision(
        self,
        envelope: RequestEnvelope,
        decision: RoutingDecision,
        env_hash: str,
    ) -> None:
        """Persist the routing decision to the routing_decisions table."""
        try:
            record = RoutingDecisionRecord(
                request_id=envelope.id,
                envelope_hash=env_hash,
                workspace_id=envelope.workspace_id,
                source=envelope.source.value,
                content=envelope.content[:2000],
                route_type=decision.route_type,
                agent_id=decision.agent_id,
                workflow_id=decision.workflow_id,
                confidence=decision.confidence,
                cached=decision.cached,
            )
            self._db.add(record)
            self._db.commit()
        except Exception:
            logger.exception("[router] Failed to log routing decision")
            self._db.rollback()
