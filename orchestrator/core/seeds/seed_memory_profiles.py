"""Seed the platform's memory profiles (PRD draft "Automatos for products", Wave 4, §5.6).

A profile says what distillation keeps as a durable fact in one domain
(``core.models.memory_profiles``). ``general`` is today's prompt unchanged; the others name a
domain and add its focus, what to keep and skip, and worked examples, whose types are the
platform's own (``modules.memory.write_contract.MEMORY_FACT_TYPES``).

Idempotent: inserts or updates each seeded profile (no workspace) on its slug.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from sqlalchemy.orm import Session

from core.models.memory_profiles import MemoryProfile

logger = logging.getLogger(__name__)

GENERAL = "general"

SEEDED_PROFILES: List[Dict[str, Any]] = [
    {
        "slug": GENERAL,
        "name": "General",
        "description": "What Auto keeps today: operational knowledge, and stable facts about the person, "
                       "their business and their preferences.",
        "focus": "", "keep": [], "skip": [], "examples": [],
    },
    {
        "slug": "customer",
        "name": "Customer",
        "description": "For an agent that serves one customer at a time: their account, their issues and "
                       "what they were promised.",
        "focus": "This agent serves a business's customers. Keep what the next conversation with this "
                 "customer would need: who they are, their account, every issue they raised and how it "
                 "ended, what they were promised, and how they like to be dealt with.",
        "keep": [
            "the customer's account: plan, billing cycle, renewal or due dates, and the people on it with their roles",
            "each issue the customer raised, and its outcome or where it stands",
            "every promise or follow-up owed to the customer, with its date (as open_loop)",
            "how the customer prefers to be contacted and spoken to",
        ],
        "skip": [
            "the agent's own generic troubleshooting steps",
            "anything about a different customer",
        ],
        "examples": [
            {"fact": "Northwind is on the Team plan, billed monthly in GBP, renewing on 1 March.", "type": "business_fact"},
            {"fact": "Northwind was promised a refund of the duplicate March invoice by 20 November.", "type": "open_loop"},
            {"fact": "Northwind's billing contact, Priya in finance, prefers email to phone calls.", "type": "preference"},
        ],
    },
    {
        "slug": "incident",
        "name": "Incident",
        "description": "For on-call and operations: an incident as a timeline of what failed, what was tried "
                       "and what fixed it.",
        "focus": "This agent works incidents. Keep the incident as a timeline someone joining later could "
                 "read: what failed and when, each action taken and its result, the cause once known, the "
                 "decisions made and why, and what is still owed.",
        "keep": [
            "each symptom, with the time it was seen (and the timezone)",
            "each action taken, and what it changed",
            "the root cause once it is established, and the evidence for it",
            "decisions, with their reasons, and every follow-up owed (as open_loop)",
        ],
        "skip": [
            "hypotheses that were ruled out (keep only the conclusion)",
            "chatter while waiting",
        ],
        "examples": [
            {"fact": "At 14:05 UTC on 9 November checkout began returning 502s from the eu-west load balancer.",
             "type": "tool_outcome"},
            {"fact": "Rolling back release 2026.11.3 restored checkout at 14:40 UTC.", "type": "task_learning"},
            {"fact": "The team decided to freeze deploys until the connection-pool fix ships, to avoid a repeat.",
             "type": "decision"},
            {"fact": "A postmortem is owed to the incident channel by 12 November.", "type": "open_loop"},
        ],
    },
    {
        "slug": "case",
        "name": "Case",
        "description": "For legal and claims work: the facts of a case, its parties, dates, documents and "
                       "deadlines.",
        "focus": "This agent works a case. Keep the facts someone would need to pick the case up: the "
                 "parties and their roles, the key dates and deadlines, the documents and what they say, "
                 "each party's stated position, and the decisions and next steps.",
        "keep": [
            "the parties and their roles",
            "dates that matter, and every deadline (as open_loop until it is met)",
            "the documents referred to, and what each says on the point at issue",
            "each party's stated position, and the client's decisions with their reasons",
        ],
        "skip": [
            "general legal information that is not about this case",
            "drafts that were discarded",
        ],
        "examples": [
            {"fact": "The lease between Hartwell Ltd and Copperfield Estates began on 1 March 2024, for five years.",
             "type": "business_fact"},
            {"fact": "Copperfield's response to the dilapidations claim is due by 28 November.", "type": "open_loop"},
            {"fact": "The client decided not to contest the service charge, to keep the relationship with the landlord.",
             "type": "decision"},
        ],
    },
]

_FIELDS = ("name", "description", "focus", "keep", "skip", "examples")


def seed_memory_profiles(db: Session) -> Tuple[int, int]:
    """Insert or update each seeded profile on its slug. Returns (created, updated)."""
    created = updated = 0
    for profile in SEEDED_PROFILES:
        row = (db.query(MemoryProfile)
               .filter(MemoryProfile.workspace_id.is_(None), MemoryProfile.slug == profile["slug"]).first())
        if row is None:
            db.add(MemoryProfile(workspace_id=None, slug=profile["slug"], **{f: profile[f] for f in _FIELDS}))
            created += 1
        elif any(getattr(row, f) != profile[f] for f in _FIELDS):
            for f in _FIELDS:
                setattr(row, f, profile[f])
            updated += 1
    db.commit()
    return created, updated
