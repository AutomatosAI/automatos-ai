"""The Socials plan tools' input schemas: one definition for the API agents and a CLI session.

``platform_get_social_plan``, ``platform_add_social_topics`` and
``platform_get_social_history`` are the Content bank research playbook's three tools. The
API agents get them from ``modules/tools/discovery/actions_socials.py``; a ticket session
(``services/session_socials_tools.py``) gets them as ``get_social_plan``,
``add_social_topics`` and ``get_social_history``. Both build their parameters from here, so
the two cannot drift. Like ``core/social_post_schema.py`` it imports only that module's
format list and nothing under ``modules``: a session module may not import ``modules.tools``
at load time.
"""
from __future__ import annotations

from typing import Any, Dict

from core.social_post_schema import POST_FORMATS

# modules/socials/topics.py FACT_SOURCE_KINDS and its limits (PRD-251B B8).
FACT_SOURCE_KINDS = ["knowledge", "deliverable", "web", "github", "note"]
TOPICS_MAX_PER_CALL = 30
FACTS_MAX_PER_TOPIC = 12

# modules/socials/history.py MAX_DAYS and MAX_LIMIT: the most platform_get_social_history reads.
HISTORY_MAX_DAYS = 365
HISTORY_MAX_LIMIT = 200

PLAN_ID = {"type": "string", "description": "The Socials plan's id (the research playbook's plan_id input)."}


def _fact_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The fact, in one sentence."},
            "source": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": FACT_SOURCE_KINDS},
                    # F383 (night 11): where the id comes from, so research never asks the owner for one.
                    "ref": {"type": "string", "description": (
                        "Where the fact comes from, taken from your tool results: a search_knowledge source's "
                        "document id (its 'document N'), a Deliverable's id, or the page's address. A note needs "
                        "none. Never ask the owner for one.")},
                    "label": {"type": "string", "description": "A short name for the source (a note's may be left out)."},
                },
                "required": ["kind"],
            },
        },
        "required": ["text", "source"],
    }


def topic_schema() -> Dict[str, Any]:
    """One researched topic: a title, an angle, its sourced facts and the formats it suits."""
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "The idea, as a post's title."},
            "angle": {"type": "string", "description": "Why it matters to the plan's audience."},
            "facts": {"type": "array", "items": _fact_schema(), "maxItems": FACTS_MAX_PER_TOPIC},
            "formats": {"type": "array", "items": {"type": "string", "enum": POST_FORMATS}},
            # PRD-251C (C9): a dated topic, such as an event's countdown, posted on its day.
            "pinned_on": {
                "type": "string",
                "description": "A day within the plan's dates (YYYY-MM-DD) to post it on: an event's countdown or its day.",
            },
        },
        "required": ["title", "facts"],
    }


def get_plan_properties() -> Dict[str, Any]:
    """The properties of a plan read: the plan's id."""
    return {"plan_id": dict(PLAN_ID)}


def add_topics_properties() -> Dict[str, Any]:
    """The properties of a topics write: the plan's id and the topics to add."""
    return {
        "plan_id": dict(PLAN_ID),
        "topics": {"type": "array", "items": topic_schema(), "minItems": 1, "maxItems": TOPICS_MAX_PER_CALL},
    }


def history_properties() -> Dict[str, Any]:
    """The properties of a history read: how far back, how many, and only what went out."""
    return {
        "days": {
            "type": "integer",
            "minimum": 1,
            "maximum": HISTORY_MAX_DAYS,
            "description": "How many days back to read (the platform's default when left out).",
        },
        "limit": {
            "type": "integer",
            "minimum": 1,
            "maximum": HISTORY_MAX_LIMIT,
            "description": "How many posts at most, newest first (the platform's default when left out).",
        },
        # P256-FIX-SOC-4 (F432-b): what went out, by when, with its receipts.
        "went_out": {
            "type": "boolean",
            "description": (
                "Only the posts that went out in the last days (days 1 for today), newest first by "
                "when they went out, each with when, where, the platform's id and the link."
            ),
        },
    }
