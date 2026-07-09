"""PRD-164 S1 — Planning Context Pack consumed by all three planners (Q61).

Proves the four S1 acceptance criteria with DB-free tests:

1. GOLDEN (learning demo): a seeded prior-mission failure recalled on the
   PRD-159 path flows through the ONE pack into MissionPlanner's prompt and
   visibly changes the resulting plan (the failed approach is avoided).
2. GREP GATE: all three planners — MissionPlanner, board ``plan_task``,
   AutoBrain — call ``ContextService.build_planning_context``; exactly one
   assembler definition exists; no planner assembles RAG/memory/KG context
   itself; the pack's RAG section retrieves through the PRD-157 choke point.
3. BUDGET: the pack stays within its token budget on oversized fixtures
   (both the PRD-157-budgeter cap helper and the full assembler).
4. Wiring smoke for the AutoBrain and board consumers.
"""
from __future__ import annotations

import importlib.util as _ilu
import json
import os
import sys as _sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

# Dummy POSTGRES_* satisfies the config chain (blessed pattern, see
# tests/test_harness_self_management.py) — the port points at nothing so the
# modules.tools import chain's fail-soft DB connect refuses instantly instead
# of hanging on a wedged local proxy. CI exports real POSTGRES_* so these
# setdefaults no-op there. Nothing in this file touches a DB.
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")


# Lean-venv shim: importing modules.tools.* runs modules/tools/__init__, which
# pulls modules.rag's ingestion chain (camelot at module top). Stub the missing
# *leaf* only when truly absent — never the modules.rag package.
def _camelot_unlocatable() -> bool:  # pragma: no cover - env-dependent
    try:
        return _ilu.find_spec("camelot") is None
    except ValueError:
        return False


if _camelot_unlocatable():  # pragma: no cover - env-dependent
    import types as _types

    _sys.modules.setdefault("camelot", _types.ModuleType("camelot"))

from config import config as _config  # noqa: E402,F401
# CI collection-order guard: earlier-collected tests stub modules.*/consumers.*
# in sys.modules (bare ModuleType, no __spec__). On Linux collection order the
# stubs are still live HERE, so the real imports below resolve against them and
# die at collection ("unknown location" ImportError — see PR #434 CI). Purge
# origin-less entries so the real packages import fresh; conftest's autouse
# repair fixture re-binds everything else at test time.
import sys as _sys_guard  # noqa: E402
for _name in [n for n, m in list(_sys_guard.modules.items())
              if (n == "modules" or n.startswith("modules.")
                  or n == "consumers" or n.startswith("consumers."))
              and getattr(m, "__spec__", None) is None]:
    _sys_guard.modules.pop(_name, None)

from core.context_guard import count_tokens  # noqa: E402
from modules.context.budget import DEFAULT_BUDGETS, RenderedSection  # noqa: E402
from modules.context.modes import MODE_CONFIGS, ContextMode  # noqa: E402
from modules.context.planning import (  # noqa: E402
    PACK_HEADER,
    PlanningContextPack,
    apply_pack_budget,
)
from modules.context.sections import SECTION_REGISTRY  # noqa: E402
from modules.context.service import ContextService  # noqa: E402

ORCH_ROOT = Path(__file__).resolve().parent.parent

# The three planners Q61 converges (binding story list, PRD-164 S1).
PLANNER_SOURCES: Dict[str, Path] = {
    "mission_planner": ORCH_ROOT / "modules" / "coordination" / "planner.py",
    "board_plan_task": ORCH_ROOT / "api" / "board_tasks.py",
    "auto_brain": ORCH_ROOT / "consumers" / "chatbot" / "auto.py",
}

FAILURE_MARKER = "LinkedIn blocks automated scraping"
SEEDED_FAILURE = {
    "id": "mem-1",
    "content": (
        "Task failed: Scrape LinkedIn profiles for the lead list. "
        f"Failure detail: HTTP 403 — {FAILURE_MARKER}; all 3 attempts blocked. "
        "Mission goal: Build a lead list of 50 fintech CTOs."
    ),
    "content_type": "task_failure",
    "importance": 0.8,
    "metadata": {"outcome": "failed", "failure_reason_code": "tool_error"},
    "created_at": "2026-06-01T00:00:00+00:00",
}


# ---------------------------------------------------------------------------
# Fakes for the pack's data sources (the sections still run for real)
# ---------------------------------------------------------------------------


class _FakeUnifiedMemory:
    """PRD-159 recall path stand-in: typed L2 + semantic L3."""

    def __init__(self, items: Optional[List[dict]] = None):
        self.items = items or []
        self.short_term_calls: List[dict] = []

    async def search_short_term(self, workspace_id, query="", days=7, limit=20,
                                content_types=None):
        self.short_term_calls.append({
            "workspace_id": workspace_id, "query": query, "days": days,
            "limit": limit, "content_types": content_types,
        })
        if content_types:
            return [i for i in self.items if i.get("content_type") in content_types]
        return list(self.items)

    async def search_long_term(self, workspace_id, query, agent_id=None, limit=5):
        return []


class _FakeRag:
    def __init__(self, formatted_context: str = "", n_chunks: int = 0):
        self._result = SimpleNamespace(
            chunks=[{"content": "c"}] * n_chunks,
            formatted_context=formatted_context,
            total_tokens=0,
            sources=[],
            query="",
        )
        self.calls: List[dict] = []

    async def retrieve(self, **kwargs):
        self.calls.append(kwargs)
        return self._result


class _FakeGraphService:
    async def load_graph(self, workspace_id):
        return None


def _pack_source_patches(unified=None, rag=None):
    """Patch the pack sections' data sources at their lookup sites."""
    unified = unified or _FakeUnifiedMemory()
    rag = rag or _FakeRag()
    return (
        patch(
            "modules.memory.unified_memory_service.get_unified_memory_service",
            return_value=unified,
        ),
        patch("modules.rag.service.get_rag_service", return_value=rag),
        patch(
            "modules.knowledge.graph_service.get_graph_service",
            return_value=_FakeGraphService(),
        ),
    )


def _make_agent(role: str) -> MagicMock:
    agent = MagicMock()
    agent.id = 1
    agent.name = role
    agent.status = "active"
    agent.skills = []
    agent.tags = []
    agent.description = f"A {role} agent"
    return agent


def _plan(task1_title: str, task1_desc: str) -> dict:
    return {
        "tasks": [
            {
                "temp_id": "task_1",
                "title": task1_title,
                "description": task1_desc,
                "agent_role": "researcher",
                "sequence_number": 1,
                "task_type": "llm_generation",
                "complexity": "moderate",
                "parallel_group": None,
                "verification_criteria": [],
                "required_tools": [],
                "dependencies": [],
            },
            {
                "temp_id": "task_2",
                "title": "Compile the lead list",
                "description": "Merge findings into the final lead list",
                "agent_role": "writer",
                "sequence_number": 2,
                "task_type": "synthesis",
                "complexity": "moderate",
                "parallel_group": None,
                "verification_criteria": [],
                "required_tools": [],
                "dependencies": ["task_1"],
            },
        ]
    }


PLAN_SCRAPING = _plan(
    "Scrape LinkedIn profiles",
    "Scrape LinkedIn for fintech CTO profiles",
)
PLAN_AVOIDING = _plan(
    "Collect leads via the licensed data API",
    "Use the licensed B2B data API export instead of scraping LinkedIn "
    "(scraping previously failed with HTTP 403)",
)


class _InstructionFollowingLLM:
    """Deterministic stand-in for the planner LLM.

    Returns the scraping plan unless the prompt carries the recalled failure
    — exactly what an instruction-following model does with the pack's
    'do NOT repeat approaches that previously failed' header. This makes the
    plan delta assertable without a live model.
    """

    def __init__(self):
        self.prompts: List[str] = []

    async def generate_response(self, messages):
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        plan = PLAN_AVOIDING if FAILURE_MARKER in prompt else PLAN_SCRAPING
        return SimpleNamespace(content=json.dumps(plan))


# ===========================================================================
# Mode + registry wiring
# ===========================================================================


class TestPlanningModeRegistered:
    def test_planning_mode_exists_with_config(self):
        assert ContextMode.PLANNING in MODE_CONFIGS

    def test_planning_sections_are_registered(self):
        for name in MODE_CONFIGS[ContextMode.PLANNING].sections:
            assert name in SECTION_REGISTRY, f"section '{name}' not registered"

    def test_planning_sections_cover_the_cheap_routing_signals(self):
        # PERF (2026-07-09): the document-RAG leg ("planning_knowledge",
        # RAG-on-goal) was removed from the classifier's pack — a ~80-113s /
        # ~3k-token document retrieval on every non-heuristic message just to
        # classify was the dominant chat latency/cost drain. The pack now
        # carries only cheap routing signals: mission memory, KG subgraph, the
        # workspace field digest (PRD-179 S1), and roster+performance. Documents
        # are retrieved on demand in the response path via Auto's tools.
        assert MODE_CONFIGS[ContextMode.PLANNING].sections == [
            "planning_history", "business_graph", "field_memory", "agent_roster",
        ]

    def test_planning_budget_exists(self):
        assert ContextMode.PLANNING in DEFAULT_BUDGETS
        assert DEFAULT_BUDGETS[ContextMode.PLANNING].available_for_sections > 0

    def test_history_survives_budget_pressure(self):
        # The learning demo depends on recalled failures never being the
        # section that gets dropped (priority <= 2).
        cls = SECTION_REGISTRY["planning_history"]
        assert cls().priority <= 2


# ===========================================================================
# AC3 — pack stays within budget on oversized fixtures
# ===========================================================================


_OVERSIZED = "the quick brown fox jumps over the lazy dog " * 4000  # ~36k tokens


class TestPackBudget:
    def test_apply_pack_budget_caps_oversized_sections(self):
        budget = 500
        rendered = [
            RenderedSection(
                name=f"s{i}", priority=p, content=_OVERSIZED,
                token_estimate=count_tokens(_OVERSIZED), max_tokens=None,
            )
            for i, p in enumerate((2, 3, 45))
        ]
        included, trimmed = apply_pack_budget(rendered, budget)
        total = sum(s.token_estimate for s in included)
        assert included, "budget cap must never empty a pack that had content"
        assert total <= budget
        assert trimmed, "oversized fixtures must be recorded as trimmed"

    def test_apply_pack_budget_prefers_high_priority_sections(self):
        small_history = "prior failure: scraping was blocked " * 5
        rendered = [
            RenderedSection(
                name="planning_knowledge", priority=3, content=_OVERSIZED,
                token_estimate=count_tokens(_OVERSIZED), max_tokens=None,
            ),
            RenderedSection(
                name="planning_history", priority=2, content=small_history,
                token_estimate=count_tokens(small_history), max_tokens=None,
            ),
        ]
        included, _ = apply_pack_budget(rendered, 200)
        names = [s.name for s in included]
        assert "planning_history" in names

    @pytest.mark.asyncio
    async def test_assembler_respects_budget_on_oversized_sources(self):
        budget = 600
        unified = _FakeUnifiedMemory(
            [dict(SEEDED_FAILURE, id=f"m{i}", content=SEEDED_FAILURE["content"] * 3)
             for i in range(12)]
        )
        rag = _FakeRag(formatted_context=_OVERSIZED, n_chunks=6)
        p1, p2, p3 = _pack_source_patches(unified=unified, rag=rag)
        with p1, p2, p3:
            pack = await ContextService(None).build_planning_context(
                goal="Build a lead list of 50 fintech CTOs",
                workspace_id="ws-budget",
                max_tokens=budget,
            )
        assert not pack.is_empty
        assert pack.token_budget == budget
        assert count_tokens(pack.content) <= budget
        assert pack.token_estimate <= budget

    @pytest.mark.asyncio
    async def test_assembler_default_budget_comes_from_planning_mode(self):
        p1, p2, p3 = _pack_source_patches()
        with p1, p2, p3:
            pack = await ContextService(None).build_planning_context(
                goal="anything", workspace_id="ws-x",
            )
        assert pack.token_budget == (
            DEFAULT_BUDGETS[ContextMode.PLANNING].available_for_sections
        )


# ===========================================================================
# AC1 — GOLDEN: seeded prior failure visibly changes a new plan
# ===========================================================================


class TestGoldenLearningDemo:
    GOAL = "Build a lead list of 50 fintech CTOs with contact details"

    async def _decompose(self, llm, unified) -> Any:
        from modules.coordination.planner import MissionPlanner

        agents = [_make_agent("researcher"), _make_agent("writer")]
        p1, p2, p3 = _pack_source_patches(unified=unified)
        with p1, p2, p3, patch(
            "modules.coordination.planner.match_template", return_value=None
        ), patch(
            "modules.coordination.planner.create_llm_manager", return_value=llm
        ):
            return await MissionPlanner.decompose(
                goal=self.GOAL,
                workspace_id=uuid4(),
                agents=agents,
                db=MagicMock(),
            )

    @pytest.mark.asyncio
    async def test_seeded_prior_failure_visibly_changes_the_plan(self):
        # --- Run 1: no prior failures recorded ---
        llm_fresh = _InstructionFollowingLLM()
        result_fresh = await self._decompose(llm_fresh, _FakeUnifiedMemory([]))

        # --- Run 2: same goal, workspace has the seeded failure ---
        llm_seeded = _InstructionFollowingLLM()
        result_seeded = await self._decompose(
            llm_seeded, _FakeUnifiedMemory([SEEDED_FAILURE])
        )

        # The failure reached the planning LLM verbatim — and only when seeded.
        assert FAILURE_MARKER not in llm_fresh.prompts[0]
        assert FAILURE_MARKER in llm_seeded.prompts[0]

        # The standing pack instruction rode along with it.
        assert "do NOT repeat approaches that previously failed" in (
            llm_seeded.prompts[0]
        )

        # The plan visibly changed: the failed approach is gone, the
        # alternative approach is in.
        fresh_titles = {t.title for t in result_fresh.tasks}
        seeded_titles = {t.title for t in result_seeded.tasks}
        assert "Scrape LinkedIn profiles" in fresh_titles
        assert "Scrape LinkedIn profiles" not in seeded_titles
        assert "Collect leads via the licensed data API" in seeded_titles
        assert fresh_titles != seeded_titles

    @pytest.mark.asyncio
    async def test_recall_uses_prd159_typed_path(self):
        # The pack recalls mission history through search_short_term with the
        # mission content types — the PRD-159 path, not a parallel query.
        unified = _FakeUnifiedMemory([SEEDED_FAILURE])
        llm = _InstructionFollowingLLM()
        await self._decompose(llm, unified)
        assert unified.short_term_calls, "PRD-159 recall path was not consulted"
        types_seen = unified.short_term_calls[0]["content_types"]
        assert "task_failure" in types_seen
        assert "mission_summary" in types_seen


# ===========================================================================
# AC2 — GREP GATE: one assembler, three consumers, no parallel assembly
# ===========================================================================


class TestOneAssemblerGrepGate:
    def test_all_three_planners_call_the_one_assembler(self):
        for name, path in PLANNER_SOURCES.items():
            src = path.read_text()
            assert "build_planning_context(" in src, (
                f"{name} ({path.name}) does not consume "
                "ContextService.build_planning_context — Q61 violation"
            )

    def test_exactly_one_assembler_definition(self):
        definitions: List[Path] = []
        for py in ORCH_ROOT.rglob("*.py"):
            parts = set(py.parts)
            if "tests" in parts or "alembic" in parts or ".venv" in parts:
                continue
            try:
                if "def build_planning_context" in py.read_text(errors="ignore"):
                    definitions.append(py)
            except OSError:  # pragma: no cover
                continue
        assert definitions == [
            ORCH_ROOT / "modules" / "context" / "service.py"
        ], f"expected ONE assembler in modules/context/service.py, found: {definitions}"

    def test_planners_do_not_assemble_context_themselves(self):
        # No planner reaches for the pack's data sources directly — retrieval,
        # memory recall and graph loading happen ONLY inside the pack sections.
        banned = (
            "get_rag_service",
            "search_short_term(",
            "search_long_term(",
            "load_graph(",
            "build_retrieval_filters",
            "get_unified_memory_service",
            "get_graph_service",
        )
        for name, path in PLANNER_SOURCES.items():
            src = path.read_text()
            for token in banned:
                assert token not in src, (
                    f"{name} ({path.name}) assembles planning context itself "
                    f"(found '{token}') — use the one pack"
                )

    def test_pack_rag_goes_through_the_choke_point(self):
        # planning_knowledge retrieves via RAGService.retrieve, and retrieve()
        # derives scope through build_retrieval_filters (PRD-157 choke point).
        # The section may NAME the choke point in prose, but must neither
        # import nor call it — scope derivation belongs to RAGService.retrieve.
        section_src = (
            ORCH_ROOT / "modules" / "context" / "sections" / "planning_knowledge.py"
        ).read_text()
        assert "get_rag_service" in section_src
        assert ".retrieve(" in section_src
        assert "build_retrieval_filters(" not in section_src, (
            "the section must not re-derive scope — RAGService.retrieve owns that"
        )
        assert "retrieval_filters import" not in section_src, (
            "the section must not import the filter builder directly"
        )
        rag_src = (ORCH_ROOT / "modules" / "rag" / "service.py").read_text()
        assert "build_retrieval_filters(" in rag_src


# ===========================================================================
# Consumer wiring smoke — AutoBrain and board plan_task
# ===========================================================================


def _marker_pack() -> PlanningContextPack:
    return PlanningContextPack(
        content="PACK_MARKER_CONTENT prior failure: avoid scraping",
        sections={"planning_history": "prior failure: avoid scraping"},
        token_estimate=12,
        token_budget=2000,
        sections_included=["planning_history"],
    )


class TestAutoBrainConsumesPack:
    @pytest.mark.asyncio
    async def test_tier3_classify_prompt_contains_pack(self):
        from consumers.chatbot.auto import AutoBrain

        brain = AutoBrain(db=MagicMock(), workspace_id="ws-auto")
        brain._redis = None

        captured: Dict[str, Any] = {}

        class _ClassifierLLM:
            async def generate_response(self, messages):
                captured["prompt"] = messages[-1]["content"]
                return SimpleNamespace(content=json.dumps({
                    "complexity": "molecule", "action": "respond",
                    "tool_hints": [], "needs_memory": False,
                    "needs_multi_agent": False, "reasoning": "test",
                }))

        with patch.object(
            ContextService, "build_planning_context",
            new=AsyncMock(return_value=_marker_pack()),
        ), patch(
            "core.llm.create_llm_manager", return_value=_ClassifierLLM()
        ):
            assessment = await brain._llm_classify("build me a lead list", 0)

        assert "PACK_MARKER_CONTENT" in captured["prompt"]
        assert assessment.complexity.value == "molecule"

    @pytest.mark.asyncio
    async def test_pack_failure_never_breaks_classification(self):
        from consumers.chatbot.auto import AutoBrain

        brain = AutoBrain(db=MagicMock(), workspace_id="ws-auto")
        brain._redis = None

        class _ClassifierLLM:
            async def generate_response(self, messages):
                return SimpleNamespace(content=json.dumps({
                    "complexity": "atom", "action": "respond",
                    "tool_hints": [], "needs_memory": False,
                    "needs_multi_agent": False, "reasoning": "test",
                }))

        with patch.object(
            ContextService, "build_planning_context",
            new=AsyncMock(side_effect=RuntimeError("pack exploded")),
        ), patch(
            "core.llm.create_llm_manager", return_value=_ClassifierLLM()
        ):
            assessment = await brain._llm_classify("hello there friend", 0)

        assert assessment is not None
        assert assessment.complexity.value == "atom"


class TestBoardPlanTaskConsumesPack:
    @pytest.mark.asyncio
    async def test_plan_task_injects_pack_as_system_context(self):
        from api.board_tasks import plan_task

        request = MagicMock()
        request.json = AsyncMock(return_value={"raw_prompt": "Build a lead list"})
        ctx = SimpleNamespace(workspace_id=uuid4())

        captured: Dict[str, Any] = {}

        class _BoardLLM:
            def __init__(self, *args, **kwargs):
                pass

            async def generate_response(self, messages):
                captured["messages"] = messages
                return SimpleNamespace(content=json.dumps({
                    "questions": [], "suggested_title": "t",
                    "suggested_priority": "medium",
                }))

        with patch.object(
            ContextService, "build_planning_context",
            new=AsyncMock(return_value=_marker_pack()),
        ), patch("core.llm.manager.LLMManager", _BoardLLM):
            result = await plan_task(request=request, ctx=ctx, db=MagicMock())

        assert result["raw_prompt"] == "Build a lead list"
        system_text = "\n".join(
            m["content"] for m in captured["messages"] if m["role"] == "system"
        )
        assert "PACK_MARKER_CONTENT" in system_text
        # The user message stays clean — context rides in system messages.
        assert "PACK_MARKER_CONTENT" not in captured["messages"][-1]["content"]


# ===========================================================================
# Roster + performance section behavior (pack ingredient)
# ===========================================================================


class TestRosterSection:
    @pytest.mark.asyncio
    async def test_unknown_roster_renders_nothing(self):
        from modules.context.sections.agent_roster import AgentRosterSection
        from modules.context.sections.base import SectionContext

        ctx = SectionContext(agent=None, workspace_id="ws-r", kwargs={})
        assert await AgentRosterSection().render(ctx) == ""

    @pytest.mark.asyncio
    async def test_roster_renders_performance_line(self):
        from modules.context.sections.agent_roster import AgentRosterSection
        from modules.context.sections.base import SectionContext

        agent = SimpleNamespace(
            id=7, name="Vector", agent_type="researcher", status="active",
            description="Research agent", model_config={"model_id": "gpt-4o"},
            skills=[], tags=[],
        )
        ctx = SectionContext(
            agent=None, workspace_id="ws-r",
            kwargs={"roster_agents": [agent], "agent_performance": {7: 0.86}},
        )
        content = await AgentRosterSection().render(ctx)
        assert "Vector" in content
        assert "0.86" in content
        assert "Recent performance" in content
