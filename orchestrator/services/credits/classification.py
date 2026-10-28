"""Which model calls are customer work and which are platform overhead (PRD-265 US-006).

Every ``LLMManager`` is built with a purpose (``request_type``, else
``service_name``). Decision D-3 (Gerard, 20 Oct 2026): background and system calls
are ABSORBED — they run on the platform's keys whatever the balance, are recorded
as overhead (``charge_class = 'overhead'``, rolled up nightly) for margin
reporting, and are reviewed after two weeks.

- **Overhead** — work the platform does for itself around an ask: routing and
  classification, memory distillation, retrieval, graph building, verification,
  Watch verdicts, digests. Never gated, never charged.
- **Customer work** — calls that produce what the person asked for. They pass the
  credit gate (``services/credits/gate.py``) and belong to an ask.

An unknown purpose is customer work: a new call site is gated until someone
classifies it here. ``tests/test_prd265_call_site_classes.py`` reads every
construction site in the code and fails on one that is not in either set.
"""
from __future__ import annotations

from typing import Optional

OVERHEAD = "overhead"
CUSTOMER = "customer"

OVERHEAD_PURPOSES = frozenset({
    "action_classifier",      # jobs/sync_composio_actions — tool catalogue batch job
    "complexity_assessor",    # consumers/chatbot/auto — Auto's routing decision
    "consistency_verifier",   # modules/coordination/verification
    "decision_adapter",       # core/llm/decisions — typed routing decisions
    "digest",                 # services/digest_service
    "entity_extraction",      # modules/search — knowledge-graph entities
    "generation_guard",       # core/generation_guard — the Art. 5 check before any image/video/voice
    "graph_community_title",  # modules/knowledge/community_reports
    "graph_extraction",       # modules/knowledge — graph build
    "memory_distill",         # consumers/chatbot/smart_memory
    "rag",                    # modules/rag — query enhancement, chunk annotation
    "thread_checkpoint",      # modules/memory/thread_checkpoint
    "verifier",               # modules/coordination/verification — LLM judge
    "watch",                  # modules/coordination/run_verdict, services/watch_decider
})

CUSTOMER_PURPOSES = frozenset({
    "blog_cover_gen",         # modules/tools/discovery/handlers_blog
    "brand_style_read",       # modules/documents/brand_style
    "check_in",               # services/proactive/runner — one ask per check-in (PRD-268)
    "cover_image",            # modules/tools/discovery/handlers_blog — a post's cover image
    "clarification",          # services/orchestrator_answers — answers a worker mid-run
    "heartbeat",              # services/heartbeat_service — one ask per tick
    "jira_triage",            # modules/workflows/recipes/jira_bug_triage
    "orchestrator",           # every agent's own calls (agent factory), the router's answer, NL2SQL
    "planner",                # modules/coordination/planner — mission planning
    "planning",               # api/board_tasks — task planning assist
    "recipe_fallback",        # api/recipe_executor — a Playbook step's answer
    "socials_compose",        # modules/socials/compose
    "socials_plan_draft",     # modules/socials/plan_draft
    "web_search",             # services/web_search — a tool inside an ask
})


def call_class(purpose: Optional[str]) -> str:
    """``overhead`` or ``customer`` for a manager's purpose (unknown → customer)."""
    return OVERHEAD if (purpose or "") in OVERHEAD_PURPOSES else CUSTOMER


def is_overhead(purpose: Optional[str]) -> bool:
    return call_class(purpose) == OVERHEAD
