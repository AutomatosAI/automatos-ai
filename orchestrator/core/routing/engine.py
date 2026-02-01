"""
Universal Router Engine (PRD-50: US-003).

Core routing engine that resolves a RequestEnvelope to a RoutingDecision
using a tiered strategy:

  Tier 0 — User override (agent_id or workflow_id explicitly set)
  Tier 1 — Cache lookup (RoutingCache hit)
  Tier 2 — Rule-based matching:
    2a: routing_rules table (workspace + source pattern)
    2b: TriggerSubscription (for jira_trigger source)
    2c: IntentClassifier keyword matching against routing rules

Returns None when no tier produces a decision (caller handles LLM fallback).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional
from uuid import UUID

from sqlalchemy.orm import Session

from core.models.composio import ComposioEntity, TriggerSubscription
from core.models.routing import (
    ChannelSource,
    RequestEnvelope,
    RoutingDecision,
    RoutingDecisionRecord,
    RoutingRule,
)
from core.routing.cache import RoutingCache, _normalize_content
from core.services.intent_classifier import IntentClassifier

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def route(self, envelope: RequestEnvelope) -> Optional[RoutingDecision]:
        """Route *envelope* through the tier chain.

        Returns a ``RoutingDecision`` if a match is found at any tier,
        or ``None`` when no tier can resolve the request (the caller
        should fall back to LLM classification or default agent).
        """

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

        # Tier 2c — IntentClassifier keyword matching against routing rules
        decision = self._tier2c_intent_classifier(envelope)
        if decision is not None:
            logger.info("[router] Tier 2c hit (intent): %s", decision.reasoning)
            self._log_decision(envelope, decision, env_hash)
            return decision

        # No match at any tier
        logger.info("[router] No route found for request %s", env_hash)
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
        """Match against routing_rules for the workspace + source."""
        rules = (
            self._db.query(RoutingRule)
            .filter(
                RoutingRule.workspace_id == envelope.workspace_id,
                RoutingRule.is_active.is_(True),
            )
            .order_by(RoutingRule.priority.desc())
            .all()
        )

        for rule in rules:
            # Source pattern match (None / empty means "match any")
            if rule.source_pattern and rule.source_pattern != envelope.source.value:
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
                source=envelope.source.value,
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
