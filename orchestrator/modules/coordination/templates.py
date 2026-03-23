"""
Decomposition Template Library — PRD-82B US-001
================================================

Pre-built decomposition templates for common mission types.
Template matching runs BEFORE LLM decomposition (wired in US-002)
so common patterns get consistent, high-quality task graphs.

Agent roles MUST align with _ROLE_SYNONYMS categories in agent_matcher.py:
  research, analyst, writer, reviewer, coder, designer, summarizer, search, document
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.models.orchestration_enums import TaskType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Template dataclasses
# ---------------------------------------------------------------------------

KEYWORD_MATCH_THRESHOLD = 2


@dataclass(frozen=True)
class TaskTemplate:
    """Blueprint for a single task within a decomposition template."""

    sequence: int
    agent_role: str
    title_pattern: str
    description_pattern: str
    required_tools: List[str] = field(default_factory=list)
    expected_output: str = ""
    verification_criteria: List[Dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True)
class DecompositionTemplate:
    """A reusable decomposition pattern for common mission types."""

    id: str
    name: str
    description: str
    keywords: List[str]
    task_templates: List[TaskTemplate]
    min_tasks: int
    max_tasks: int
    output_format: str = "markdown"


# ---------------------------------------------------------------------------
# Built-in templates
# ---------------------------------------------------------------------------

TEMPLATE_REGISTRY: List[DecompositionTemplate] = [
    # ── research_and_report (5 tasks) ──────────────────────────────────
    DecompositionTemplate(
        id="research_and_report",
        name="Research & Report",
        description="Research a topic, compare options, and produce a structured report.",
        keywords=[
            "research", "compare", "evaluate", "benchmark",
            "framework", "survey", "review",
        ],
        min_tasks=5,
        max_tasks=7,
        output_format="markdown",
        task_templates=[
            TaskTemplate(
                sequence=1,
                agent_role="researcher",
                title_pattern="Define research scope for: {goal}",
                description_pattern=(
                    "Identify the key dimensions, criteria, and sources "
                    "relevant to: {goal}. Produce a research brief listing "
                    "3-5 evaluation criteria, target sources, and scope boundaries."
                ),
                expected_output="Research brief with criteria and source list",
                verification_criteria=[
                    {"type": "min_length", "value": 200, "must_pass": True},
                    {
                        "type": "required_sections",
                        "value": ["Criteria", "Sources"],
                        "must_pass": False,
                    },
                ],
            ),
            TaskTemplate(
                sequence=2,
                agent_role="search",
                title_pattern="Gather data on: {goal}",
                description_pattern=(
                    "Using the research brief from Task 1, search for and "
                    "collect relevant data, articles, and benchmarks related to: "
                    "{goal}. Compile raw findings with source citations."
                ),
                expected_output="Raw findings with sources",
                verification_criteria=[
                    {"type": "min_length", "value": 300, "must_pass": True},
                ],
            ),
            TaskTemplate(
                sequence=3,
                agent_role="analyst",
                title_pattern="Analyze findings for: {goal}",
                description_pattern=(
                    "Analyze the gathered data from Task 2 against the criteria "
                    "defined in Task 1. Produce a structured comparison with "
                    "pros/cons and scoring for each candidate."
                ),
                expected_output="Comparative analysis with scoring",
                verification_criteria=[
                    {"type": "min_length", "value": 400, "must_pass": True},
                    {
                        "type": "required_sections",
                        "value": ["Comparison", "Scoring"],
                        "must_pass": False,
                    },
                ],
            ),
            TaskTemplate(
                sequence=4,
                agent_role="writer",
                title_pattern="Draft report for: {goal}",
                description_pattern=(
                    "Using the analysis from Task 3, write a polished report "
                    "with executive summary, detailed findings, comparison "
                    "tables, and actionable recommendations for: {goal}."
                ),
                expected_output="Complete research report",
                verification_criteria=[
                    {"type": "min_length", "value": 800, "must_pass": True},
                    {
                        "type": "required_sections",
                        "value": [
                            "Executive Summary",
                            "Findings",
                            "Recommendations",
                        ],
                        "must_pass": True,
                    },
                ],
            ),
            TaskTemplate(
                sequence=5,
                agent_role="reviewer",
                title_pattern="Review and finalize report on: {goal}",
                description_pattern=(
                    "Review the draft report from Task 4 for accuracy, "
                    "completeness, and quality. Check claims against source "
                    "data, fix any inconsistencies, and produce the final version."
                ),
                expected_output="Reviewed and finalized report",
                verification_criteria=[
                    {"type": "min_length", "value": 800, "must_pass": True},
                ],
            ),
        ],
    ),
    # ── content_pipeline (4 tasks) ────────────────────────────────────
    DecompositionTemplate(
        id="content_pipeline",
        name="Content Pipeline",
        description="Research, outline, draft, and edit content.",
        keywords=[
            "write", "blog", "article", "content",
            "draft", "copywrite", "newsletter",
        ],
        min_tasks=4,
        max_tasks=6,
        output_format="markdown",
        task_templates=[
            TaskTemplate(
                sequence=1,
                agent_role="researcher",
                title_pattern="Research topic for content: {goal}",
                description_pattern=(
                    "Research the topic, audience, and key talking points "
                    "for: {goal}. Identify trending angles, competitor "
                    "content, and unique value propositions."
                ),
                expected_output="Research notes with key talking points",
                verification_criteria=[
                    {"type": "min_length", "value": 200, "must_pass": True},
                ],
            ),
            TaskTemplate(
                sequence=2,
                agent_role="writer",
                title_pattern="Create outline for: {goal}",
                description_pattern=(
                    "Using the research from Task 1, create a detailed "
                    "content outline with sections, key points per section, "
                    "and target word count for: {goal}."
                ),
                expected_output="Content outline with sections and key points",
                verification_criteria=[
                    {"type": "min_length", "value": 150, "must_pass": True},
                ],
            ),
            TaskTemplate(
                sequence=3,
                agent_role="writer",
                title_pattern="Write full draft for: {goal}",
                description_pattern=(
                    "Following the outline from Task 2, write the complete "
                    "content piece for: {goal}. Maintain consistent tone, "
                    "include transitions, and ensure all outline points are covered."
                ),
                expected_output="Complete content draft",
                verification_criteria=[
                    {"type": "min_length", "value": 600, "must_pass": True},
                ],
            ),
            TaskTemplate(
                sequence=4,
                agent_role="reviewer",
                title_pattern="Edit and polish: {goal}",
                description_pattern=(
                    "Review the draft from Task 3 for grammar, clarity, "
                    "tone, and factual accuracy. Produce the final polished "
                    "version ready for publication."
                ),
                expected_output="Final polished content",
                verification_criteria=[
                    {"type": "min_length", "value": 600, "must_pass": True},
                ],
            ),
        ],
    ),
    # ── competitive_analysis (4 tasks) ────────────────────────────────
    DecompositionTemplate(
        id="competitive_analysis",
        name="Competitive Analysis",
        description="Analyze competitive landscape and produce strategic insights.",
        keywords=[
            "competitive", "market analysis", "compare companies",
            "industry", "landscape", "players",
        ],
        min_tasks=4,
        max_tasks=6,
        output_format="markdown",
        task_templates=[
            TaskTemplate(
                sequence=1,
                agent_role="search",
                title_pattern="Identify competitors for: {goal}",
                description_pattern=(
                    "Search for and identify the key players, competitors, "
                    "and market participants relevant to: {goal}. List each "
                    "with a brief description, market position, and key differentiators."
                ),
                expected_output="Competitor list with descriptions",
                verification_criteria=[
                    {"type": "min_length", "value": 300, "must_pass": True},
                ],
            ),
            TaskTemplate(
                sequence=2,
                agent_role="researcher",
                title_pattern="Deep-dive competitor analysis for: {goal}",
                description_pattern=(
                    "For each competitor identified in Task 1, research "
                    "their strengths, weaknesses, pricing, features, market "
                    "share, and recent developments related to: {goal}."
                ),
                expected_output="Detailed per-competitor analysis",
                verification_criteria=[
                    {"type": "min_length", "value": 400, "must_pass": True},
                ],
            ),
            TaskTemplate(
                sequence=3,
                agent_role="analyst",
                title_pattern="Synthesize competitive insights for: {goal}",
                description_pattern=(
                    "Synthesize the competitor data from Tasks 1-2 into "
                    "strategic insights: market gaps, opportunities, threats, "
                    "and positioning recommendations for: {goal}."
                ),
                expected_output="Strategic insights and recommendations",
                verification_criteria=[
                    {"type": "min_length", "value": 400, "must_pass": True},
                    {
                        "type": "required_sections",
                        "value": ["Opportunities", "Threats"],
                        "must_pass": False,
                    },
                ],
            ),
            TaskTemplate(
                sequence=4,
                agent_role="writer",
                title_pattern="Write competitive analysis report: {goal}",
                description_pattern=(
                    "Compile the analysis from Tasks 1-3 into a professional "
                    "competitive analysis report with executive summary, "
                    "competitor comparison matrix, and strategic recommendations."
                ),
                expected_output="Complete competitive analysis report",
                verification_criteria=[
                    {"type": "min_length", "value": 800, "must_pass": True},
                    {
                        "type": "required_sections",
                        "value": [
                            "Executive Summary",
                            "Competitor Comparison",
                            "Recommendations",
                        ],
                        "must_pass": True,
                    },
                ],
            ),
        ],
    ),
    # ── data_investigation (3 tasks) ──────────────────────────────────
    DecompositionTemplate(
        id="data_investigation",
        name="Data Investigation",
        description="Investigate, diagnose, and report on a data-related question.",
        keywords=[
            "investigate", "audit", "diagnose", "track",
            "monitor", "analyze data", "find",
        ],
        min_tasks=3,
        max_tasks=5,
        output_format="markdown",
        task_templates=[
            TaskTemplate(
                sequence=1,
                agent_role="search",
                title_pattern="Gather evidence for: {goal}",
                description_pattern=(
                    "Search for and collect all relevant data, logs, metrics, "
                    "and evidence related to: {goal}. Document sources and "
                    "note any anomalies or patterns observed."
                ),
                expected_output="Evidence collection with sources",
                verification_criteria=[
                    {"type": "min_length", "value": 200, "must_pass": True},
                ],
            ),
            TaskTemplate(
                sequence=2,
                agent_role="analyst",
                title_pattern="Analyze and diagnose: {goal}",
                description_pattern=(
                    "Analyze the evidence from Task 1 to identify root "
                    "causes, correlations, and patterns for: {goal}. Provide "
                    "a diagnosis with confidence levels for each finding."
                ),
                expected_output="Diagnosis with findings and confidence levels",
                verification_criteria=[
                    {"type": "min_length", "value": 300, "must_pass": True},
                    {
                        "type": "required_sections",
                        "value": ["Findings", "Diagnosis"],
                        "must_pass": False,
                    },
                ],
            ),
            TaskTemplate(
                sequence=3,
                agent_role="writer",
                title_pattern="Write investigation report: {goal}",
                description_pattern=(
                    "Compile the investigation results from Tasks 1-2 into "
                    "a clear report with findings, root cause analysis, and "
                    "recommended actions for: {goal}."
                ),
                expected_output="Investigation report with recommendations",
                verification_criteria=[
                    {"type": "min_length", "value": 400, "must_pass": True},
                    {
                        "type": "required_sections",
                        "value": ["Findings", "Root Cause", "Recommendations"],
                        "must_pass": True,
                    },
                ],
            ),
        ],
    ),
]


# ---------------------------------------------------------------------------
# Template matching
# ---------------------------------------------------------------------------


def match_template(goal: str) -> Optional[DecompositionTemplate]:
    """
    Match a goal string against the template registry using keyword scoring.

    Returns the highest-scoring template if it meets the threshold
    (KEYWORD_MATCH_THRESHOLD keyword hits), or None.
    """
    if not goal:
        return None

    goal_lower = goal.lower()
    # Tokenize goal into words for word-boundary matching
    goal_words = set(re.findall(r"[a-z]+", goal_lower))

    best_template: Optional[DecompositionTemplate] = None
    best_score = 0

    for template in TEMPLATE_REGISTRY:
        score = 0
        for keyword in template.keywords:
            kw_lower = keyword.lower()
            # Multi-word keywords: check substring
            if " " in kw_lower:
                if kw_lower in goal_lower:
                    score += 1
            else:
                # Single-word: check in tokenized words
                if kw_lower in goal_words:
                    score += 1

        if score >= KEYWORD_MATCH_THRESHOLD and score > best_score:
            best_score = score
            best_template = template

    if best_template is not None:
        logger.info(
            "Template matched: id=%s score=%d for goal='%s'",
            best_template.id,
            best_score,
            goal[:80],
        )
    else:
        logger.debug("No template match for goal='%s'", goal[:80])

    return best_template


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------


def render_template(
    template: DecompositionTemplate,
    goal: str,
) -> List[Dict[str, object]]:
    """
    Render a DecompositionTemplate into a list of task dicts.

    Output format matches planner._parse_plan_response() / PlannedTask fields:
      title, description, agent_role, required_tools, depends_on,
      verification_criteria, temp_id, sequence_number, task_type.
    """
    tasks: List[Dict[str, object]] = []

    for tt in template.task_templates:
        title = tt.title_pattern.replace("{goal}", goal)
        description = tt.description_pattern.replace("{goal}", goal)

        # Build dependencies: each task depends on the previous one
        depends_on: List[str] = []
        if tt.sequence > 1:
            depends_on = [f"task_{tt.sequence - 1}"]

        tasks.append({
            "temp_id": f"task_{tt.sequence}",
            "title": title,
            "description": description,
            "agent_role": tt.agent_role,
            "sequence_number": tt.sequence,
            "task_type": TaskType.LLM_GENERATION.value,
            "required_tools": list(tt.required_tools),
            "dependencies": depends_on,
            "verification_criteria": list(tt.verification_criteria),
        })

    return tasks
