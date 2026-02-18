"""
Phase Selector — Dynamic Workflow Phase Selection
==================================================

PRD-59: Replaces the fixed 9-stage pipeline with 5 dynamic phases
that expand into specific stages based on task characteristics.

Phases:
  PLAN     → Stages 1 (Decompose), 2 (Select), 2b (Negotiate)
  PREPARE  → Stages 3 (Context), 3b (Prompt Optimization)
  EXECUTE  → Stages 4 (Execution), 4b (Coordination)
  EVALUATE → Stages 5 (Aggregate), 6 (Quality)
  LEARN    → Stages 7 (Learn), 8 (Memory), 9 (Response)

The selector determines which phases and stages to execute based on:
  - Execution mode (AUTONOMOUS / RECIPE / HYBRID)
  - Task complexity (from ComplexityAnalyzer)
  - Number of agents involved
  - Available context sources
"""

import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Workflow execution mode"""
    AUTONOMOUS = "autonomous"   # Full pipeline: decompose + select + execute
    RECIPE = "recipe"           # Pre-defined steps and agents
    HYBRID = "hybrid"           # Pre-defined steps, select agents


class ComplexityLevel(Enum):
    """Progressive Complexity Model levels"""
    ATOM = "atom"           # Simple single-agent task (50-200 tokens)
    MOLECULE = "molecule"   # Needs examples/context (500-2,000 tokens)
    CELL = "cell"           # Agent memory required (2,000-4,000 tokens)
    ORGAN = "organ"         # Multi-agent coordination (4,000-8,000 tokens)
    ORGANISM = "organism"   # Enterprise/cross-workflow (8,000-16,000 tokens)


class Phase(Enum):
    """Workflow phases with their constituent stages"""
    PLAN_FULL = "plan_full"           # Stages 1, 2, 2b
    PLAN_PARTIAL = "plan_partial"     # Stage 2 only (steps pre-defined)
    PREPARE = "prepare"               # Stages 3, 3b
    EXECUTE_SINGLE = "execute_single" # Stage 4 only
    EXECUTE_MULTI = "execute_multi"   # Stages 4, 4b (coordination)
    EVALUATE_FULL = "evaluate_full"   # Stages 5, 6 (aggregate + quality)
    EVALUATE_LIGHT = "evaluate_light" # Stage 5 only (no LLM quality)
    LEARN = "learn"                   # Stages 7, 8, 9


@dataclass
class StageSpec:
    """Specification for a single stage within a phase"""
    stage_number: int
    stage_name: str
    phase: Phase
    is_optional: bool = False
    skip_reason: Optional[str] = None

    def __repr__(self) -> str:
        return f"Stage({self.stage_number}: {self.stage_name})"


@dataclass
class PhaseSpec:
    """A phase with its expanded stages"""
    phase: Phase
    stages: List[StageSpec]
    token_budget: int = 0

    @property
    def stage_count(self) -> int:
        return len(self.stages)

    def __repr__(self) -> str:
        return f"Phase({self.phase.value}: {[s.stage_number for s in self.stages]})"


# Stage definitions
STAGE_DECOMPOSE = lambda: StageSpec(1, "Task Decomposition", Phase.PLAN_FULL)
STAGE_SELECT = lambda: StageSpec(2, "Agent Selection", Phase.PLAN_FULL)
STAGE_NEGOTIATE = lambda: StageSpec(2, "Inter-Agent Negotiation", Phase.PLAN_FULL, is_optional=True)  # 2b
STAGE_CONTEXT = lambda: StageSpec(3, "Context Engineering", Phase.PREPARE)
STAGE_PROMPT_OPT = lambda: StageSpec(3, "Prompt Optimization", Phase.PREPARE, is_optional=True)  # 3b
STAGE_EXECUTE = lambda: StageSpec(4, "Agent Execution", Phase.EXECUTE_SINGLE)
STAGE_COORDINATE = lambda: StageSpec(4, "Inter-Agent Coordination", Phase.EXECUTE_MULTI, is_optional=True)  # 4b
STAGE_AGGREGATE = lambda: StageSpec(5, "Result Aggregation", Phase.EVALUATE_FULL)
STAGE_QUALITY = lambda: StageSpec(6, "Quality Assessment", Phase.EVALUATE_FULL)
STAGE_LEARN = lambda: StageSpec(7, "Learning Update", Phase.LEARN)
STAGE_MEMORY = lambda: StageSpec(8, "Memory Storage", Phase.LEARN)
STAGE_RESPONSE = lambda: StageSpec(9, "Response Generation", Phase.LEARN)


# Token budgets by complexity
TOKEN_BUDGETS = {
    ComplexityLevel.ATOM: 200,
    ComplexityLevel.MOLECULE: 2000,
    ComplexityLevel.CELL: 4000,
    ComplexityLevel.ORGAN: 8000,
    ComplexityLevel.ORGANISM: 16000,
}


class PhaseSelector:
    """
    Determines which phases and stages to execute based on task characteristics.

    Usage:
        selector = PhaseSelector()
        phases = selector.select_phases(
            mode=ExecutionMode.AUTONOMOUS,
            complexity=ComplexityLevel.ORGAN,
            agent_count=4,
            has_context_sources=True,
        )
        # phases is a list of PhaseSpec, each containing its expanded stages
    """

    def select_phases(
        self,
        mode: ExecutionMode,
        complexity: ComplexityLevel,
        agent_count: int,
        has_context_sources: bool = False,
        has_prompt_registry: bool = False,
    ) -> List[PhaseSpec]:
        """
        Select which phases and stages to run.

        Args:
            mode: Execution mode (AUTONOMOUS/RECIPE/HYBRID)
            complexity: Task complexity level
            agent_count: Number of agents that will be involved
            has_context_sources: Whether RAG/memory sources are available
            has_prompt_registry: Whether PRD-58 Prompt Registry is available

        Returns:
            Ordered list of PhaseSpec objects with their stages
        """
        phases: List[PhaseSpec] = []
        token_budget = TOKEN_BUDGETS.get(complexity, 4000)

        # ── PLAN phase ──
        if mode == ExecutionMode.AUTONOMOUS:
            stages = [STAGE_DECOMPOSE(), STAGE_SELECT()]
            if agent_count >= 3:
                stages.append(STAGE_NEGOTIATE())
            phases.append(PhaseSpec(
                phase=Phase.PLAN_FULL,
                stages=stages,
                token_budget=token_budget,
            ))
        elif mode == ExecutionMode.HYBRID:
            # Steps pre-defined, but need agent selection
            phases.append(PhaseSpec(
                phase=Phase.PLAN_PARTIAL,
                stages=[STAGE_SELECT()],
                token_budget=token_budget,
            ))
        # RECIPE mode: skip PLAN entirely

        # ── PREPARE phase ──
        if has_context_sources or complexity.value in (
            ComplexityLevel.MOLECULE.value,
            ComplexityLevel.CELL.value,
            ComplexityLevel.ORGAN.value,
            ComplexityLevel.ORGANISM.value,
        ):
            stages = [STAGE_CONTEXT()]
            if has_prompt_registry:
                stages.append(STAGE_PROMPT_OPT())
            phases.append(PhaseSpec(
                phase=Phase.PREPARE,
                stages=stages,
                token_budget=token_budget,
            ))

        # ── EXECUTE phase (always) ──
        if agent_count > 1:
            phases.append(PhaseSpec(
                phase=Phase.EXECUTE_MULTI,
                stages=[STAGE_EXECUTE(), STAGE_COORDINATE()],
                token_budget=token_budget,
            ))
        else:
            phases.append(PhaseSpec(
                phase=Phase.EXECUTE_SINGLE,
                stages=[STAGE_EXECUTE()],
                token_budget=token_budget,
            ))

        # ── EVALUATE phase ──
        if agent_count > 1 or complexity.value in (
            ComplexityLevel.CELL.value,
            ComplexityLevel.ORGAN.value,
            ComplexityLevel.ORGANISM.value,
        ):
            phases.append(PhaseSpec(
                phase=Phase.EVALUATE_FULL,
                stages=[STAGE_AGGREGATE(), STAGE_QUALITY()],
                token_budget=token_budget,
            ))
        else:
            phases.append(PhaseSpec(
                phase=Phase.EVALUATE_LIGHT,
                stages=[STAGE_AGGREGATE()],
                token_budget=token_budget,
            ))

        # ── LEARN phase (always) ──
        learn_stages = [STAGE_LEARN(), STAGE_MEMORY(), STAGE_RESPONSE()]
        phases.append(PhaseSpec(
            phase=Phase.LEARN,
            stages=learn_stages,
            token_budget=token_budget,
        ))

        total_stages = sum(p.stage_count for p in phases)
        logger.info(
            f"📋 PhaseSelector: mode={mode.value}, complexity={complexity.value}, "
            f"agents={agent_count} → {len(phases)} phases, {total_stages} stages"
        )
        for p in phases:
            logger.debug(f"  {p}")

        return phases

    def get_total_stages(self, phases: List[PhaseSpec]) -> int:
        """Count total stages across all phases"""
        return sum(p.stage_count for p in phases)

    def estimate_token_budget(self, phases: List[PhaseSpec]) -> int:
        """Estimate total token budget"""
        if phases:
            return phases[0].token_budget  # All phases share same budget
        return 4000

    def describe_plan(self, phases: List[PhaseSpec]) -> Dict[str, Any]:
        """Generate a human-readable plan description"""
        return {
            "total_phases": len(phases),
            "total_stages": self.get_total_stages(phases),
            "token_budget": self.estimate_token_budget(phases),
            "phases": [
                {
                    "name": p.phase.value,
                    "stages": [
                        {
                            "number": s.stage_number,
                            "name": s.stage_name,
                            "optional": s.is_optional,
                        }
                        for s in p.stages
                    ],
                }
                for p in phases
            ],
        }
