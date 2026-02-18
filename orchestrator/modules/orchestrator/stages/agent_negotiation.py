"""
Inter-Agent Negotiation Stage (Stage 2b)
=========================================

PRD-59: Before execution, selected agents review the task plan and
propose adjustments. Only triggers for workflows with 3+ agents.

This implements the collaborative problem-solving pattern from PRD-04.
Selected agents review the task decomposition and can propose:
  - Task boundary adjustments
  - Additional context needs
  - Execution order preferences

The negotiation runs as a single LLM call with all agent perspectives
represented in the prompt (not separate calls per agent).

Token budget: 2000 tokens max for the negotiation output.
"""

import json
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class NegotiationResult:
    """Result of inter-agent negotiation"""
    original_plan: List[Dict[str, Any]]
    adjusted_plan: List[Dict[str, Any]]
    adjustments_made: List[Dict[str, Any]] = field(default_factory=list)
    agent_perspectives: Dict[str, str] = field(default_factory=dict)
    consensus_reached: bool = True
    negotiation_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "adjustments_made": self.adjustments_made,
            "agent_perspectives": self.agent_perspectives,
            "consensus_reached": self.consensus_reached,
            "negotiation_tokens": self.negotiation_tokens,
            "num_adjustments": len(self.adjustments_made),
        }


class InterAgentNegotiation:
    """
    Stage 2b: Selected agents review and negotiate the task plan.

    Only triggers for multi-agent workflows (3+ agents).
    Uses a single LLM call with a multi-perspective prompt.
    """

    MAX_NEGOTIATION_TOKENS = 2000
    MIN_AGENTS_FOR_NEGOTIATION = 3

    def __init__(self, llm_client: Optional[Any] = None):
        """
        Args:
            llm_client: LLM client with generate_response(messages) method.
                        If None, negotiation returns the plan unchanged.
        """
        self.llm_client = llm_client

    async def negotiate(
        self,
        subtasks: List[Dict[str, Any]],
        agent_assignments: Dict[str, Dict[str, Any]],
        workflow_description: str = "",
    ) -> NegotiationResult:
        """
        Run inter-agent negotiation on the task plan.

        Args:
            subtasks: List of subtask dicts from decomposer
            agent_assignments: Map of subtask_id → agent assignment info
            workflow_description: Overall task description

        Returns:
            NegotiationResult with potentially adjusted plan
        """
        unique_agents = set()
        for assignment in agent_assignments.values():
            if isinstance(assignment, dict):
                unique_agents.add(assignment.get("agent_name", "Unknown"))
            elif isinstance(assignment, list) and assignment:
                agent = assignment[0]
                if hasattr(agent, "agent_name"):
                    unique_agents.add(agent.agent_name)

        if len(unique_agents) < self.MIN_AGENTS_FOR_NEGOTIATION:
            logger.info(
                f"⏭️ Stage 2b skipped: only {len(unique_agents)} agents "
                f"(need {self.MIN_AGENTS_FOR_NEGOTIATION}+)"
            )
            return NegotiationResult(
                original_plan=subtasks,
                adjusted_plan=subtasks,
                consensus_reached=True,
            )

        if self.llm_client is None:
            logger.info("⏭️ Stage 2b skipped: no LLM client available")
            return NegotiationResult(
                original_plan=subtasks,
                adjusted_plan=subtasks,
                consensus_reached=True,
            )

        logger.info(
            f"🤝 Stage 2b: Inter-agent negotiation with {len(unique_agents)} agents"
        )

        prompt = self._build_negotiation_prompt(
            subtasks, agent_assignments, workflow_description, list(unique_agents)
        )

        try:
            messages = [{"role": "user", "content": prompt}]
            response = await self.llm_client.generate_response(messages)
            response_text = (
                getattr(response, "content", None)
                or getattr(response, "text", None)
                or str(response)
            )

            result = self._parse_negotiation_response(
                response_text, subtasks, agent_assignments
            )
            logger.info(
                f"✅ Negotiation complete: {len(result.adjustments_made)} adjustments, "
                f"consensus={'YES' if result.consensus_reached else 'NO'}"
            )
            return result

        except Exception as e:
            logger.warning(f"Negotiation failed: {e}. Returning original plan.")
            return NegotiationResult(
                original_plan=subtasks,
                adjusted_plan=subtasks,
                consensus_reached=True,
            )

    def _build_negotiation_prompt(
        self,
        subtasks: List[Dict[str, Any]],
        agent_assignments: Dict[str, Dict[str, Any]],
        workflow_description: str,
        agent_names: List[str],
    ) -> str:
        """Build multi-perspective negotiation prompt"""
        task_plan = []
        for i, subtask in enumerate(subtasks):
            sid = subtask.get("subtask_id", f"subtask_{i}")
            assignment = agent_assignments.get(sid, {})
            agent_name = "Unassigned"
            if isinstance(assignment, dict):
                agent_name = assignment.get("agent_name", "Unassigned")
            elif isinstance(assignment, list) and assignment:
                agent = assignment[0]
                agent_name = getattr(agent, "agent_name", "Unassigned")

            task_plan.append(
                f"{i+1}. [{agent_name}] {subtask.get('description', 'No description')}"
                f" (deps: {subtask.get('dependencies', [])})"
            )

        plan_text = "\n".join(task_plan)

        return f"""You are facilitating a brief negotiation between {len(agent_names)} AI agents
about how to execute a multi-step workflow.

WORKFLOW: {workflow_description[:500]}

CURRENT TASK PLAN:
{plan_text}

AGENTS INVOLVED: {', '.join(agent_names)}

For each agent's perspective, consider:
1. Are the task boundaries clear for their assigned subtasks?
2. Do they need additional context from other agents' subtasks?
3. Should the execution order be adjusted?

Respond in JSON:
{{
    "adjustments": [
        {{
            "subtask_index": <0-based>,
            "type": "boundary|context|order",
            "description": "What to change",
            "proposed_by": "agent_name"
        }}
    ],
    "consensus": true/false,
    "agent_perspectives": {{
        "agent_name": "brief perspective (1 sentence)"
    }}
}}

Keep it concise. Max 3 adjustments. If the plan is good as-is, return empty adjustments."""

    def _parse_negotiation_response(
        self,
        response_text: str,
        subtasks: List[Dict[str, Any]],
        agent_assignments: Dict[str, Dict[str, Any]],
    ) -> NegotiationResult:
        """Parse LLM negotiation response and apply adjustments"""
        import re

        # Extract JSON
        json_match = re.search(r"\{[\s\S]*\}", response_text)
        if not json_match:
            return NegotiationResult(
                original_plan=subtasks,
                adjusted_plan=subtasks,
                consensus_reached=True,
            )

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return NegotiationResult(
                original_plan=subtasks,
                adjusted_plan=subtasks,
                consensus_reached=True,
            )

        adjustments = data.get("adjustments", [])
        perspectives = data.get("agent_perspectives", {})
        consensus = data.get("consensus", True)

        # Apply adjustments to subtasks (non-destructive copy)
        adjusted = [dict(s) for s in subtasks]
        applied_adjustments = []

        for adj in adjustments[:3]:  # Cap at 3
            idx = adj.get("subtask_index")
            if idx is not None and 0 <= idx < len(adjusted):
                adj_type = adj.get("type", "unknown")
                desc = adj.get("description", "")

                if adj_type == "context":
                    # Add context need annotation
                    adjusted[idx].setdefault("negotiation_notes", [])
                    adjusted[idx]["negotiation_notes"].append(desc)
                    adjusted[idx]["requires_context"] = True
                elif adj_type == "order":
                    # Just annotate — actual reordering is complex
                    adjusted[idx].setdefault("negotiation_notes", [])
                    adjusted[idx]["negotiation_notes"].append(f"Order suggestion: {desc}")
                elif adj_type == "boundary":
                    adjusted[idx].setdefault("negotiation_notes", [])
                    adjusted[idx]["negotiation_notes"].append(f"Boundary: {desc}")

                applied_adjustments.append(adj)

        return NegotiationResult(
            original_plan=subtasks,
            adjusted_plan=adjusted,
            adjustments_made=applied_adjustments,
            agent_perspectives=perspectives,
            consensus_reached=consensus,
            negotiation_tokens=len(response_text) // 4,  # rough estimate
        )
