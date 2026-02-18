"""
Evaluation Datasets for Stage 1-2 Prompts
==========================================
PRD-59 US-007: Input/expected-output pairs for evaluating and optimizing
task decomposition (Stage 1) and agent selection (Stage 2) prompts.

Each dataset consists of EvalCase objects with:
  - input: The prompt/task fed to the stage
  - expected: Structural expectations for the output
  - tags: Categories for filtering (complexity, domain, etc.)
  - metrics: Which FutureAGI metrics to evaluate against

Usage:
    from modules.orchestrator.eval import TASK_DECOMPOSER_EVAL_DATASET
    for case in TASK_DECOMPOSER_EVAL_DATASET:
        result = await decomposer.decompose_task(case.input)
        score = evaluate(result, case.expected)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvalCase:
    """A single evaluation test case."""
    id: str
    input: str
    expected: Dict[str, Any]
    tags: List[str] = field(default_factory=list)
    metrics: List[str] = field(default_factory=lambda: [
        "instruction_adherence",
        "task_completion",
    ])
    notes: str = ""


# ========== STAGE 1: TASK DECOMPOSER EVALUATION DATASET ==========

TASK_DECOMPOSER_EVAL_DATASET: List[EvalCase] = [
    # --- ATOM complexity: Single subtask, no decomposition needed ---
    EvalCase(
        id="td-atom-01",
        input="What is the current weather in Sydney?",
        expected={
            "min_subtasks": 1,
            "max_subtasks": 1,
            "required_skills_present": ["research"],
            "execution_strategy": "sequential",
            "has_dependencies": False,
        },
        tags=["atom", "simple", "lookup"],
        notes="Trivial single-step task — should NOT over-decompose",
    ),
    EvalCase(
        id="td-atom-02",
        input="Translate 'hello world' to Japanese",
        expected={
            "min_subtasks": 1,
            "max_subtasks": 1,
            "required_skills_present": ["language"],
            "execution_strategy": "sequential",
        },
        tags=["atom", "translation"],
    ),

    # --- MOLECULE complexity: 2-3 subtasks ---
    EvalCase(
        id="td-molecule-01",
        input="Research the top 5 competitors for an AI chatbot startup and summarize their pricing",
        expected={
            "min_subtasks": 2,
            "max_subtasks": 4,
            "required_skills_present": ["research", "analysis"],
            "has_dependencies": True,
        },
        tags=["molecule", "research", "analysis"],
        notes="Research → Summarize pipeline",
    ),
    EvalCase(
        id="td-molecule-02",
        input="Write a Python function that reads a CSV file and generates a bar chart of the top 10 rows by revenue",
        expected={
            "min_subtasks": 2,
            "max_subtasks": 3,
            "required_skills_present": ["coding", "data_analysis"],
            "execution_strategy": "sequential",
        },
        tags=["molecule", "coding", "data"],
    ),

    # --- CELL complexity: 3-5 subtasks, cross-domain ---
    EvalCase(
        id="td-cell-01",
        input="Build a landing page for our new SaaS product. Include hero section, features, pricing table, and testimonials. Use our brand colors (blue/white).",
        expected={
            "min_subtasks": 3,
            "max_subtasks": 6,
            "required_skills_present": ["design", "frontend", "coding"],
            "has_requires_context": True,
        },
        tags=["cell", "frontend", "design"],
        notes="Multi-section page requires parallel subtasks",
    ),
    EvalCase(
        id="td-cell-02",
        input="Create a REST API endpoint for user authentication with JWT tokens, password hashing, and rate limiting. Include tests.",
        expected={
            "min_subtasks": 3,
            "max_subtasks": 6,
            "required_skills_present": ["coding", "security"],
            "has_dependencies": True,
            "execution_strategy_options": ["sequential", "parallel"],
        },
        tags=["cell", "backend", "security", "testing"],
    ),

    # --- ORGAN complexity: 6-10 subtasks, multi-agent coordination ---
    EvalCase(
        id="td-organ-01",
        input="Migrate our PostgreSQL database to a new schema. Need to: analyze current schema, design new schema with proper normalization, write migration scripts, create rollback plan, update ORM models, update all API endpoints, write integration tests, and document changes.",
        expected={
            "min_subtasks": 5,
            "max_subtasks": 10,
            "required_skills_present": ["database", "coding", "documentation"],
            "has_dependencies": True,
            "has_parallel_groups": True,
        },
        tags=["organ", "database", "migration"],
        notes="Complex multi-step migration with parallelizable subtasks",
    ),
    EvalCase(
        id="td-organ-02",
        input="Conduct a full security audit of our web application. Check authentication flows, API endpoints for injection vulnerabilities, CORS configuration, CSP headers, dependency vulnerabilities, and generate a report with severity ratings.",
        expected={
            "min_subtasks": 5,
            "max_subtasks": 10,
            "required_skills_present": ["security", "research", "analysis"],
            "has_parallel_groups": True,
        },
        tags=["organ", "security", "audit"],
    ),

    # --- ORGANISM complexity: 10+ subtasks, enterprise-grade ---
    EvalCase(
        id="td-organism-01",
        input="Design and implement a complete CI/CD pipeline for our microservices architecture. We have 5 services (auth, billing, notifications, core-api, frontend). Each needs: Dockerfile, GitHub Actions workflow, staging + production environments on Railway, database migrations, health checks, monitoring alerts, and rollback procedures.",
        expected={
            "min_subtasks": 8,
            "max_subtasks": 20,
            "required_skills_present": ["devops", "coding", "infrastructure"],
            "has_dependencies": True,
            "has_parallel_groups": True,
        },
        tags=["organism", "devops", "infrastructure"],
        notes="Enterprise-scale task requiring decomposition into parallel service tracks",
    ),

    # --- Edge cases ---
    EvalCase(
        id="td-edge-ambiguous",
        input="Make it better",
        expected={
            "min_subtasks": 1,
            "max_subtasks": 2,
            "should_request_clarification": True,
        },
        tags=["edge", "ambiguous"],
        notes="Ambiguous input — decomposer should flag need for clarification",
    ),
    EvalCase(
        id="td-edge-empty",
        input="",
        expected={
            "min_subtasks": 0,
            "max_subtasks": 1,
            "should_flag_error": True,
        },
        tags=["edge", "empty"],
    ),
]


# ========== STAGE 2: AGENT SELECTOR EVALUATION DATASET ==========

AGENT_SELECTOR_EVAL_DATASET: List[EvalCase] = [
    # --- Exact skill match ---
    EvalCase(
        id="as-exact-01",
        input="Write a Python function to parse JSON",
        expected={
            "top_agent_skills": ["coding", "python"],
            "match_score_min": 0.8,
            "should_prefer_specialized": True,
        },
        tags=["exact_match", "coding"],
        metrics=["instruction_adherence", "task_completion"],
    ),

    # --- Multi-skill requirement ---
    EvalCase(
        id="as-multi-01",
        input="Research market trends and create a data visualization dashboard",
        expected={
            "top_agent_skills": ["research", "data_analysis", "visualization"],
            "requires_multiple_agents": True,
            "min_agents": 2,
        },
        tags=["multi_skill", "research", "visualization"],
    ),

    # --- Performance history matters ---
    EvalCase(
        id="as-perf-01",
        input="Write comprehensive unit tests for a React component",
        expected={
            "should_weight_performance": True,
            "prefer_high_success_rate": True,
            "top_agent_skills": ["testing", "frontend", "react"],
        },
        tags=["performance", "testing"],
        notes="Among equally skilled agents, prefer the one with higher success rate",
    ),

    # --- Load balancing ---
    EvalCase(
        id="as-load-01",
        input="Quick text summarization",
        expected={
            "should_avoid_overloaded": True,
            "workload_threshold": 0.8,
        },
        tags=["load_balance", "simple"],
        notes="Should prefer less loaded agents for simple tasks",
    ),

    # --- No matching agent ---
    EvalCase(
        id="as-nomatch-01",
        input="Compose a symphony in B minor using orchestral notation",
        expected={
            "may_return_no_match": True,
            "fallback_to_general": True,
        },
        tags=["no_match", "edge"],
        notes="Highly specialized task — may not match any agent",
    ),

    # --- Recency preference ---
    EvalCase(
        id="as-recency-01",
        input="Debug a Django REST framework serializer issue",
        expected={
            "top_agent_skills": ["coding", "python", "django"],
            "should_weight_recency": True,
        },
        tags=["recency", "debugging"],
        notes="Agent that recently completed Django tasks should score higher on recency",
    ),

    # --- Formula validation ---
    EvalCase(
        id="as-formula-01",
        input="Analyze our API logs and identify performance bottlenecks",
        expected={
            "scoring_formula": "0.4*skill + 0.3*performance + 0.2*(1-load) + 0.1*recency",
            "top_agent_skills": ["analysis", "devops", "monitoring"],
            "all_four_signals_used": True,
        },
        tags=["formula", "scoring"],
        notes="Verify all 4 scoring signals are contributing to selection",
    ),
]


def get_eval_dataset(stage: str) -> List[EvalCase]:
    """Get evaluation dataset by stage name.

    Args:
        stage: 'task_decomposer' or 'agent_selector'

    Returns:
        List of EvalCase objects for the specified stage
    """
    datasets = {
        "task_decomposer": TASK_DECOMPOSER_EVAL_DATASET,
        "stage1": TASK_DECOMPOSER_EVAL_DATASET,
        "agent_selector": AGENT_SELECTOR_EVAL_DATASET,
        "stage2": AGENT_SELECTOR_EVAL_DATASET,
    }
    return datasets.get(stage, [])
