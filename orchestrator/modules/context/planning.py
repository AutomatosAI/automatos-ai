"""
Planning Context Pack (PRD-164 S1, Q61)
=======================================

ONE token-budgeted context pack consumed by every planner on the platform:
MissionPlanner (``modules/coordination/planner.py``), the board's ``plan_task``
(``api/board_tasks.py``) and AutoBrain (``consumers/chatbot/auto.py``) all call
``ContextService.build_planning_context`` — never a parallel assembly.

The pack is assembled from the ``ContextMode.PLANNING`` section list
(``planning_knowledge`` → RAG on the goal through the PRD-157 choke point,
``planning_history`` → mission summaries + task failures via the PRD-159
recall path, ``business_graph`` → KG subgraph via the PRD-165 graph service,
``agent_roster`` → roster + agent performance) and capped with the PRD-157
token budgeter (:mod:`modules.rag.budget`) — whole-section selection by
priority, then a hard token cap.

Grep ``build_planning_context`` to enumerate every planning consumer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from core.context_guard import count_tokens
from modules.context.budget import RenderedSection
from modules.rag.budget import select_within_budget, truncate_to_token_budget

# Standing instruction every planning LLM sees ahead of the pack sections.
# Lives here (one place) so all three planners give the model the same
# grounding rules — the "learning demo" depends on the avoid-failures line.
PACK_HEADER = (
    "## What The Platform Knows (planning context)\n"
    "Ground your plan in the context below. Reuse approaches that previously "
    "worked, and do NOT repeat approaches that previously failed — plan around "
    "recorded failures explicitly. Prefer agents with proven performance. "
    "Cite retrieved sources by their [n] markers where relevant."
)

# Priority sections (priority <= this) survive budget pressure first; the
# budgeter ranks sections by ascending priority (1 = most important).
_PRIORITY_SCORE_BASE = 1000


@dataclass(frozen=True)
class PlanningContextPack:
    """Immutable, budget-capped planning context.

    ``content`` is the ready-to-inject prompt block (header + sections).
    ``sections`` maps section name → its (post-budget) rendered content.
    """

    content: str = ""
    sections: dict = field(default_factory=dict)
    token_estimate: int = 0
    token_budget: int = 0
    sections_included: List[str] = field(default_factory=list)
    sections_trimmed: List[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not any(self.sections.values())


def trim_to_tokens(text: str, max_tokens: int) -> str:
    """Trim ``text`` until ``count_tokens`` agrees it fits ``max_tokens``.

    :func:`truncate_to_token_budget` decodes a token prefix; re-encoding a
    decoded prefix can wobble by a token at merge boundaries, so verify with
    the canonical counter and tighten the target until it fits.
    """
    target = max_tokens
    trimmed = truncate_to_token_budget(text, target, suffix="")
    guard = 8
    while trimmed and count_tokens(trimmed) > max_tokens and guard > 0:
        target = max(1, target - max(1, max_tokens // 10))
        trimmed = truncate_to_token_budget(trimmed, target, suffix="")
        guard -= 1
    return trimmed if count_tokens(trimmed) <= max_tokens else ""


def apply_pack_budget(
    rendered: List[RenderedSection],
    budget_tokens: int,
) -> tuple[List[RenderedSection], List[str]]:
    """Cap rendered sections to ``budget_tokens`` using the PRD-157 budgeter.

    1. Token-aware truncation of any section over its own ``max_tokens``
       (:func:`truncate_to_token_budget` — replaces char-slice truncation).
    2. Whole-section selection under the total budget
       (:func:`select_within_budget`), most-important-priority first.
    3. A final hard cap on the single always-included top section when it
       alone exceeds the budget.

    Returns ``(included_sections, trimmed_or_dropped_names)``.
    """
    if budget_tokens <= 0:
        return [], [s.name for s in rendered]

    trimmed: List[str] = []

    # --- 1. Per-section cap (token-aware, PRD-157) ---
    capped: List[RenderedSection] = []
    for section in rendered:
        if not section.content:
            continue
        content = section.content
        tokens = count_tokens(content)
        cap = min(
            section.max_tokens if section.max_tokens else budget_tokens,
            budget_tokens,
        )
        if tokens > cap:
            content = truncate_to_token_budget(content, cap, suffix="")
            tokens = count_tokens(content)
            trimmed.append(section.name)
        capped.append(
            RenderedSection(
                name=section.name,
                priority=section.priority,
                content=content,
                token_estimate=tokens,
                max_tokens=section.max_tokens,
            )
        )

    if not capped:
        return [], trimmed

    # --- 2. Whole-section selection under the total budget (PRD-157) ---
    chunks = [
        {
            "content": s.content,
            "tokens": s.token_estimate,
            "weight": _PRIORITY_SCORE_BASE - s.priority,
            "name": s.name,
        }
        for s in capped
    ]
    selection = select_within_budget(
        chunks, budget_tokens, content_key="content", score_key="weight"
    )
    selected_names = {c["name"] for c in selection.chunks}
    for s in capped:
        if s.name not in selected_names and s.name not in trimmed:
            trimmed.append(s.name)

    # --- 3. Hard cap: the lone top section may still exceed the budget ---
    included: List[RenderedSection] = []
    total = 0
    by_name = {s.name: s for s in capped}
    for chunk in selection.chunks:
        section = by_name[chunk["name"]]
        content = section.content
        tokens = chunk["tokens"]
        if total + tokens > budget_tokens:
            content = trim_to_tokens(content, max(1, budget_tokens - total))
            tokens = count_tokens(content)
            if section.name not in trimmed:
                trimmed.append(section.name)
        if not content:
            continue
        total += tokens
        included.append(
            RenderedSection(
                name=section.name,
                priority=section.priority,
                content=content,
                token_estimate=tokens,
                max_tokens=section.max_tokens,
            )
        )

    # Preserve the mode-config (render) order in the assembled pack.
    order = {s.name: i for i, s in enumerate(capped)}
    included.sort(key=lambda s: order.get(s.name, 99))
    return included, trimmed
