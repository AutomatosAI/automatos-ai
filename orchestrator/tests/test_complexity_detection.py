"""
Wiring tests for US-004: Complexity detection in planner.
==========================================================

Proves:
1. _detect_complexity returns COMPLEX for multi-domain, multi-deliverable goals
2. _detect_complexity returns SIMPLE for trivial goals
3. _detect_complexity accounts for attachments
4. _complexity_to_max_concurrent maps tiers correctly
5. decompose() sets max_concurrent on DecompositionResult based on complexity
"""
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

# Ensure orchestrator package is importable
_orchestrator_root = str(Path(__file__).resolve().parent.parent)
if _orchestrator_root not in sys.path:
    sys.path.insert(0, _orchestrator_root)

from core.models.orchestration_enums import ComplexityTier
from modules.coordination.planner import (
    DecompositionResult,
    MissionPlanner,
    _complexity_to_max_concurrent,
    _count_deliverables,
    _detect_complexity,
    _estimate_domains,
)


# ---------------------------------------------------------------------------
# Unit tests: helper functions
# ---------------------------------------------------------------------------

class TestCountDeliverables:
    def test_no_keywords(self):
        assert _count_deliverables("summarize this") == 0

    def test_single_keyword(self):
        assert _count_deliverables("Write a report on AI") == 1

    def test_multiple_keywords(self):
        goal = "Build a dashboard and analysis pipeline for the data system"
        assert _count_deliverables(goal) >= 3  # dashboard, analysis, pipeline, system

    def test_case_insensitive(self):
        assert _count_deliverables("Write a REPORT") == 1

    def test_document_not_a_deliverable(self):
        # "document" was removed — too generic
        assert _count_deliverables("Summarize this document") == 0


class TestEstimateDomains:
    def test_no_domains(self):
        assert _estimate_domains("hello world") == 0

    def test_single_domain(self):
        assert _estimate_domains("build a react frontend") == 1

    def test_multiple_domains(self):
        goal = "build a react frontend with aws deployment and machine learning pipeline"
        assert _estimate_domains(goal) >= 3  # web, cloud, ai

    def test_research_domain(self):
        goal = "write a research paper on prior art and experiments"
        assert _estimate_domains(goal) >= 1


class TestDetectComplexity:
    def test_simple_goal(self):
        assert _detect_complexity("Summarize this document") == ComplexityTier.SIMPLE

    def test_moderate_goal_deliverables(self):
        # "report" = 1 deliverable (>= 1, +1), "AI" = 1 domain (< 2, no point) → score 1 → MODERATE
        goal = "Create a report on AI trends"
        result = _detect_complexity(goal)
        assert result == ComplexityTier.MODERATE

    def test_complex_goal(self):
        goal = (
            "Write a 4000 word research paper with 6 sections covering "
            "AI coordination, machine learning experiments, cloud deployment on AWS, "
            "and a web dashboard for data visualization. Include analysis, "
            "a presentation, and a comprehensive report on implications and prior art."
        )
        result = _detect_complexity(goal)
        assert result == ComplexityTier.COMPLEX

    def test_attachments_add_score(self):
        # Simple goal + attachments should bump from SIMPLE to MODERATE
        result = _detect_complexity(
            "Summarize this document",
            attachments=[{"file": "doc.pdf"}],
        )
        assert result == ComplexityTier.MODERATE

    def test_long_goal_adds_score(self):
        # 50+ words
        goal = " ".join(["word"] * 51)
        result = _detect_complexity(goal)
        assert result == ComplexityTier.MODERATE


class TestComplexityToMaxConcurrent:
    def test_simple(self):
        assert _complexity_to_max_concurrent(ComplexityTier.SIMPLE) == 1

    def test_moderate(self):
        assert _complexity_to_max_concurrent(ComplexityTier.MODERATE) == 2

    def test_complex(self):
        assert _complexity_to_max_concurrent(ComplexityTier.COMPLEX) == 3


# ---------------------------------------------------------------------------
# Wiring test: decompose() sets max_concurrent
# ---------------------------------------------------------------------------

class TestDecomposeMaxConcurrent:
    """Prove decompose() calls _detect_complexity and sets max_concurrent."""

    @pytest.mark.asyncio
    async def test_complex_goal_returns_high_max_concurrent(self):
        """
        WIRING TEST: decompose('Write a 4000 word research paper with 6 sections
        covering AI coordination, prior art, experiments, and implications')
        returns max_concurrent >= 2.
        """
        goal = (
            "Write a 4000 word research paper with 6 sections covering "
            "AI coordination, prior art, experiments, and implications"
        )

        # Mock the LLM to return a valid plan
        fake_plan = {
            "tasks": [
                {
                    "temp_id": f"task_{i}",
                    "title": f"Task {i}",
                    "description": f"Do task {i}",
                    "agent_role": "researcher",
                    "sequence_number": i,
                    "task_type": "research" if i < 4 else "review",
                    "verification_criteria": [],
                    "required_tools": [],
                    "depends_on": [f"task_{i-1}"] if i > 1 else [],
                }
                for i in range(1, 5)
            ]
        }

        agents = [_make_agent("researcher"), _make_agent("writer")]

        with patch(
            "modules.coordination.planner.match_template", return_value=None
        ), patch(
            "modules.coordination.planner.create_llm_manager"
        ) as mock_llm_factory:
            import json
            mock_llm = MagicMock()
            mock_llm.generate_response = AsyncMock(
                return_value=MagicMock(content=json.dumps(fake_plan))
            )
            mock_llm_factory.return_value = mock_llm

            result = await MissionPlanner.decompose(
                goal=goal,
                workspace_id=uuid4(),
                agents=agents,
            )

        assert isinstance(result, DecompositionResult)
        assert result.max_concurrent >= 2

    @pytest.mark.asyncio
    async def test_simple_goal_returns_max_concurrent_1(self):
        """
        WIRING TEST: decompose('Summarize this document') returns max_concurrent == 1.
        """
        goal = "Summarize this document"

        fake_plan = {
            "tasks": [
                {
                    "temp_id": f"task_{i}",
                    "title": f"Task {i}",
                    "description": f"Do task {i}",
                    "agent_role": "writer",
                    "sequence_number": i,
                    "task_type": "writing" if i < 3 else "review",
                    "verification_criteria": [],
                    "required_tools": [],
                    "depends_on": [f"task_{i-1}"] if i > 1 else [],
                }
                for i in range(1, 4)
            ]
        }

        agents = [_make_agent("writer"), _make_agent("reviewer")]

        with patch(
            "modules.coordination.planner.match_template", return_value=None
        ), patch(
            "modules.coordination.planner.create_llm_manager"
        ) as mock_llm_factory:
            import json
            mock_llm = MagicMock()
            mock_llm.generate_response = AsyncMock(
                return_value=MagicMock(content=json.dumps(fake_plan))
            )
            mock_llm_factory.return_value = mock_llm

            result = await MissionPlanner.decompose(
                goal=goal,
                workspace_id=uuid4(),
                agents=agents,
            )

        assert isinstance(result, DecompositionResult)
        assert result.max_concurrent == 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(role: str) -> MagicMock:
    """Create a minimal mock Agent with required attributes."""
    agent = MagicMock()
    agent.id = 1
    agent.name = role  # name used for fuzzy matching in _validate_plan
    agent.skill = role
    agent.model = "test-model"
    agent.status = "active"
    agent.skills = []
    agent.tags = []
    return agent
