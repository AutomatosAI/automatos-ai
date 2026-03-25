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
    complexity: str = "moderate"
    parallel_group: Optional[str] = None
    depends_on: Optional[List[str]] = None
    task_type: str = "llm_generation"


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
    # ── research_and_report (6 tasks) ──────────────────────────────────
    # seq 1: parallel research tasks → seq 2: synthesis → seq 3: analysis
    # → seq 4: draft → seq 5: review
    DecompositionTemplate(
        id="research_and_report",
        name="Research & Report",
        description="Research a topic, compare options, and produce a structured report.",
        keywords=[
            "research", "compare", "evaluate", "benchmark",
            "framework", "survey", "review",
        ],
        min_tasks=6,
        max_tasks=8,
        output_format="markdown",
        task_templates=[
            TaskTemplate(
                sequence=1,
                agent_role="researcher",
                title_pattern="Research scope and criteria for: {goal}",
                description_pattern=(
                    "Identify the key dimensions, criteria, and sources "
                    "relevant to: {goal}. Produce a research brief listing "
                    "3-5 evaluation criteria, target sources, and scope boundaries."
                ),
                expected_output="Research brief with criteria and source list",
                complexity="moderate",
                parallel_group="research",
                depends_on=[],
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
                sequence=1,
                agent_role="search",
                title_pattern="Gather data and sources for: {goal}",
                description_pattern=(
                    "Search for and collect relevant data, articles, and "
                    "benchmarks related to: {goal}. Compile raw findings "
                    "with source citations."
                ),
                expected_output="Raw findings with sources",
                complexity="moderate",
                parallel_group="research",
                depends_on=[],
                verification_criteria=[
                    {"type": "min_length", "value": 300, "must_pass": True},
                ],
            ),
            TaskTemplate(
                sequence=2,
                agent_role="writer",
                title_pattern="Synthesize research findings for: {goal}",
                description_pattern=(
                    "Merge the research brief and gathered data into a "
                    "unified set of findings. Resolve contradictions, "
                    "deduplicate, and produce a coherent research synthesis."
                ),
                expected_output="Unified research synthesis",
                complexity="moderate",
                task_type="synthesis",
                depends_on=["task_1", "task_2"],
                verification_criteria=[
                    {"type": "min_length", "value": 400, "must_pass": True},
                ],
            ),
            TaskTemplate(
                sequence=3,
                agent_role="analyst",
                title_pattern="Analyze findings for: {goal}",
                description_pattern=(
                    "Analyze the synthesized research against the evaluation "
                    "criteria. Produce a structured comparison with "
                    "pros/cons and scoring for each candidate."
                ),
                expected_output="Comparative analysis with scoring",
                complexity="moderate",
                depends_on=["task_3"],
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
                    "Using the analysis, write a polished report with "
                    "executive summary, detailed findings, comparison "
                    "tables, and actionable recommendations for: {goal}."
                ),
                expected_output="Complete research report",
                complexity="complex",
                depends_on=["task_4"],
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
                    "Review the draft report for accuracy, completeness, "
                    "and quality. Check claims against source data, fix "
                    "any inconsistencies, and produce the final version."
                ),
                expected_output="Reviewed and finalized report",
                complexity="moderate",
                depends_on=["task_5"],
                task_type="review",
                verification_criteria=[
                    {"type": "min_length", "value": 800, "must_pass": True},
                ],
            ),
        ],
    ),
    # ── content_pipeline (6 tasks) ────────────────────────────────────
    # seq 1: parallel research + source gathering → seq 2: synthesis outline
    # → seq 3: parallel section drafts → seq 4: synthesis merge → seq 5: review
    DecompositionTemplate(
        id="content_pipeline",
        name="Content Pipeline",
        description="Research, outline, draft, and edit content.",
        keywords=[
            "write", "blog", "article", "content",
            "draft", "copywrite", "newsletter",
        ],
        min_tasks=6,
        max_tasks=8,
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
                complexity="moderate",
                parallel_group="research",
                depends_on=[],
                verification_criteria=[
                    {"type": "min_length", "value": 200, "must_pass": True},
                ],
            ),
            TaskTemplate(
                sequence=1,
                agent_role="search",
                title_pattern="Gather source material for: {goal}",
                description_pattern=(
                    "Search for reference material, statistics, quotes, "
                    "and examples relevant to: {goal}. Compile a source "
                    "document with citations."
                ),
                expected_output="Source material with citations",
                complexity="moderate",
                parallel_group="research",
                depends_on=[],
                verification_criteria=[
                    {"type": "min_length", "value": 200, "must_pass": True},
                ],
            ),
            TaskTemplate(
                sequence=2,
                agent_role="writer",
                title_pattern="Synthesize research into outline for: {goal}",
                description_pattern=(
                    "Merge the research notes and source material into a "
                    "detailed content outline with sections, key points per "
                    "section, and target word count for: {goal}."
                ),
                expected_output="Content outline with sections and key points",
                complexity="moderate",
                task_type="synthesis",
                depends_on=["task_1", "task_2"],
                verification_criteria=[
                    {"type": "min_length", "value": 200, "must_pass": True},
                ],
            ),
            TaskTemplate(
                sequence=3,
                agent_role="writer",
                title_pattern="Write full draft for: {goal}",
                description_pattern=(
                    "Following the outline, write the complete content piece "
                    "for: {goal}. Maintain consistent tone, include "
                    "transitions, and ensure all outline points are covered."
                ),
                expected_output="Complete content draft",
                complexity="complex",
                depends_on=["task_3"],
                verification_criteria=[
                    {"type": "min_length", "value": 600, "must_pass": True},
                ],
            ),
            TaskTemplate(
                sequence=4,
                agent_role="writer",
                title_pattern="Synthesize and merge final content for: {goal}",
                description_pattern=(
                    "Review the draft against the outline and source material. "
                    "Ensure all key points are covered, resolve any gaps, "
                    "and produce a cohesive final draft."
                ),
                expected_output="Merged final draft",
                complexity="moderate",
                task_type="synthesis",
                depends_on=["task_4"],
                verification_criteria=[
                    {"type": "min_length", "value": 600, "must_pass": True},
                ],
            ),
            TaskTemplate(
                sequence=5,
                agent_role="reviewer",
                title_pattern="Edit and polish: {goal}",
                description_pattern=(
                    "Review the final draft for grammar, clarity, tone, "
                    "and factual accuracy. Produce the final polished "
                    "version ready for publication."
                ),
                expected_output="Final polished content",
                complexity="moderate",
                depends_on=["task_5"],
                task_type="review",
                verification_criteria=[
                    {"type": "min_length", "value": 600, "must_pass": True},
                ],
            ),
        ],
    ),
    # ── competitive_analysis (6 tasks) ────────────────────────────────
    # seq 1: parallel per-competitor research → seq 2: synthesis
    # → seq 3: report → seq 4: review
    DecompositionTemplate(
        id="competitive_analysis",
        name="Competitive Analysis",
        description="Analyze competitive landscape and produce strategic insights.",
        keywords=[
            "competitive", "market analysis", "compare companies",
            "industry", "landscape", "players",
        ],
        min_tasks=6,
        max_tasks=8,
        output_format="markdown",
        task_templates=[
            TaskTemplate(
                sequence=1,
                agent_role="search",
                title_pattern="Identify and research competitor group A for: {goal}",
                description_pattern=(
                    "Search for key players and competitors relevant to: "
                    "{goal}. Focus on market leaders and established players. "
                    "List each with description, market position, strengths, "
                    "weaknesses, pricing, and key differentiators."
                ),
                expected_output="Competitor profiles for group A",
                complexity="moderate",
                parallel_group="competitor_research",
                depends_on=[],
                verification_criteria=[
                    {"type": "min_length", "value": 300, "must_pass": True},
                ],
            ),
            TaskTemplate(
                sequence=1,
                agent_role="researcher",
                title_pattern="Identify and research competitor group B for: {goal}",
                description_pattern=(
                    "Research emerging competitors, startups, and alternative "
                    "solutions relevant to: {goal}. For each, document "
                    "features, market share, recent developments, and "
                    "competitive advantages."
                ),
                expected_output="Competitor profiles for group B",
                complexity="moderate",
                parallel_group="competitor_research",
                depends_on=[],
                verification_criteria=[
                    {"type": "min_length", "value": 300, "must_pass": True},
                ],
            ),
            TaskTemplate(
                sequence=2,
                agent_role="writer",
                title_pattern="Synthesize competitor research for: {goal}",
                description_pattern=(
                    "Merge all competitor research into a unified competitor "
                    "landscape. Deduplicate entries, resolve contradictions, "
                    "and produce a comprehensive competitor database."
                ),
                expected_output="Unified competitor landscape",
                complexity="moderate",
                task_type="synthesis",
                depends_on=["task_1", "task_2"],
                verification_criteria=[
                    {"type": "min_length", "value": 400, "must_pass": True},
                ],
            ),
            TaskTemplate(
                sequence=3,
                agent_role="analyst",
                title_pattern="Analyze competitive insights for: {goal}",
                description_pattern=(
                    "Analyze the unified competitor data to identify market "
                    "gaps, opportunities, threats, and positioning "
                    "recommendations for: {goal}."
                ),
                expected_output="Strategic insights and recommendations",
                complexity="moderate",
                depends_on=["task_3"],
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
                    "Compile the analysis into a professional competitive "
                    "analysis report with executive summary, competitor "
                    "comparison matrix, and strategic recommendations."
                ),
                expected_output="Complete competitive analysis report",
                complexity="complex",
                depends_on=["task_4"],
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
            TaskTemplate(
                sequence=5,
                agent_role="reviewer",
                title_pattern="Review competitive analysis: {goal}",
                description_pattern=(
                    "Review the competitive analysis report for accuracy, "
                    "completeness, and strategic soundness. Verify claims "
                    "against source data and produce the final version."
                ),
                expected_output="Reviewed competitive analysis report",
                complexity="moderate",
                depends_on=["task_5"],
                task_type="review",
                verification_criteria=[
                    {"type": "min_length", "value": 800, "must_pass": True},
                ],
            ),
        ],
    ),
    # ── data_investigation (5 tasks) ──────────────────────────────────
    # seq 1: parallel data gathering → seq 2: analysis → seq 3: report
    DecompositionTemplate(
        id="data_investigation",
        name="Data Investigation",
        description="Investigate, diagnose, and report on a data-related question.",
        keywords=[
            "investigate", "audit", "diagnose", "track",
            "monitor", "analyze data", "find",
        ],
        min_tasks=5,
        max_tasks=7,
        output_format="markdown",
        task_templates=[
            TaskTemplate(
                sequence=1,
                agent_role="search",
                title_pattern="Gather external evidence for: {goal}",
                description_pattern=(
                    "Search for and collect external data, articles, "
                    "benchmarks, and reference material related to: {goal}. "
                    "Document sources and note any relevant patterns."
                ),
                expected_output="External evidence collection with sources",
                complexity="moderate",
                parallel_group="gathering",
                depends_on=[],
                verification_criteria=[
                    {"type": "min_length", "value": 200, "must_pass": True},
                ],
            ),
            TaskTemplate(
                sequence=1,
                agent_role="researcher",
                title_pattern="Gather internal evidence for: {goal}",
                description_pattern=(
                    "Collect all relevant internal data, logs, metrics, and "
                    "evidence related to: {goal}. Document anomalies and "
                    "patterns observed in the data."
                ),
                expected_output="Internal evidence collection",
                complexity="moderate",
                parallel_group="gathering",
                depends_on=[],
                verification_criteria=[
                    {"type": "min_length", "value": 200, "must_pass": True},
                ],
            ),
            TaskTemplate(
                sequence=2,
                agent_role="writer",
                title_pattern="Synthesize gathered evidence for: {goal}",
                description_pattern=(
                    "Merge external and internal evidence into a unified "
                    "evidence base. Resolve contradictions, identify gaps, "
                    "and produce a coherent evidence synthesis."
                ),
                expected_output="Unified evidence synthesis",
                complexity="moderate",
                task_type="synthesis",
                depends_on=["task_1", "task_2"],
                verification_criteria=[
                    {"type": "min_length", "value": 300, "must_pass": True},
                ],
            ),
            TaskTemplate(
                sequence=3,
                agent_role="analyst",
                title_pattern="Analyze and diagnose: {goal}",
                description_pattern=(
                    "Analyze the synthesized evidence to identify root "
                    "causes, correlations, and patterns for: {goal}. Provide "
                    "a diagnosis with confidence levels for each finding."
                ),
                expected_output="Diagnosis with findings and confidence levels",
                complexity="moderate",
                depends_on=["task_3"],
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
                sequence=4,
                agent_role="writer",
                title_pattern="Write investigation report: {goal}",
                description_pattern=(
                    "Compile the investigation results into a clear report "
                    "with findings, root cause analysis, and recommended "
                    "actions for: {goal}."
                ),
                expected_output="Investigation report with recommendations",
                complexity="moderate",
                depends_on=["task_4"],
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
    # ── business_plan (14 tasks) ─────────────────────────────────────
    # Phase 1 (research — 3 parallel tasks + synthesis)
    # Phase 2 (document generation — sequential: exec summary, market
    #          analysis, financial projections, synthesis into full doc)
    # Phase 3 (workspace config — 3 parallel tasks)
    # Phase 4 (verification — summary report)
    DecompositionTemplate(
        id="business_plan",
        name="Business Plan",
        description=(
            "Research, analyze, and produce a comprehensive business plan "
            "with market research, financial modeling, and workspace "
            "configuration for ongoing operations."
        ),
        keywords=[
            "business plan", "business idea", "business model",
            "launch a business", "start a company",
            "business", "plan", "startup", "venture",
            "company", "launch",
        ],
        min_tasks=12,
        max_tasks=14,
        output_format="markdown",
        task_templates=[
            # ── Phase 1: Research (3 parallel + 1 synthesis) ──────────
            TaskTemplate(
                sequence=1,
                agent_role="researcher",
                title_pattern="Market research for: {goal}",
                description_pattern=(
                    "Research the target market for: {goal}. Identify market "
                    "size, growth trends, customer segments, demand drivers, "
                    "and key demographics. Produce a market research brief "
                    "with data-backed findings."
                ),
                expected_output="Market research brief with data and trends",
                complexity="moderate",
                parallel_group="bp_research",
                depends_on=[],
                verification_criteria=[
                    {"type": "min_length", "value": 400, "must_pass": True},
                    {
                        "type": "required_sections",
                        "value": ["Market Size", "Customer Segments"],
                        "must_pass": False,
                    },
                ],
            ),
            TaskTemplate(
                sequence=1,
                agent_role="analyst",
                title_pattern="Competitive analysis for: {goal}",
                description_pattern=(
                    "Identify and analyze competitors relevant to: {goal}. "
                    "For each competitor, document strengths, weaknesses, "
                    "pricing, market position, and differentiators. Identify "
                    "market gaps and competitive advantages."
                ),
                expected_output="Competitive landscape analysis",
                complexity="moderate",
                parallel_group="bp_research",
                depends_on=[],
                verification_criteria=[
                    {"type": "min_length", "value": 400, "must_pass": True},
                    {
                        "type": "required_sections",
                        "value": ["Competitors", "Market Gaps"],
                        "must_pass": False,
                    },
                ],
            ),
            TaskTemplate(
                sequence=1,
                agent_role="analyst",
                title_pattern="Financial modeling for: {goal}",
                description_pattern=(
                    "Build financial projections for: {goal}. Include "
                    "revenue model, cost structure, break-even analysis, "
                    "and 3-year P&L forecast. Identify key assumptions "
                    "and financial risks."
                ),
                expected_output="Financial model with projections and assumptions",
                complexity="complex",
                parallel_group="bp_research",
                depends_on=[],
                verification_criteria=[
                    {"type": "min_length", "value": 400, "must_pass": True},
                    {
                        "type": "required_sections",
                        "value": ["Revenue Model", "Cost Structure"],
                        "must_pass": False,
                    },
                ],
            ),
            TaskTemplate(
                sequence=2,
                agent_role="writer",
                title_pattern="Synthesize research findings for: {goal}",
                description_pattern=(
                    "Merge market research, competitive analysis, and "
                    "financial modeling into a unified research synthesis. "
                    "Resolve contradictions, highlight key insights, and "
                    "produce an integrated research foundation for the "
                    "business plan."
                ),
                expected_output="Integrated research synthesis",
                complexity="moderate",
                task_type="synthesis",
                depends_on=["task_1", "task_2", "task_3"],
                verification_criteria=[
                    {"type": "min_length", "value": 600, "must_pass": True},
                ],
            ),
            # ── Phase 2: Document Generation (sequential) ─────────────
            TaskTemplate(
                sequence=3,
                agent_role="writer",
                title_pattern="Write executive summary for: {goal}",
                description_pattern=(
                    "Using the research synthesis, write a compelling "
                    "executive summary for: {goal}. Cover the business "
                    "concept, value proposition, target market, revenue "
                    "model, and funding requirements. Keep it concise "
                    "and persuasive."
                ),
                expected_output="Executive summary (1-2 pages)",
                complexity="moderate",
                depends_on=["task_4"],
                verification_criteria=[
                    {"type": "min_length", "value": 400, "must_pass": True},
                    {
                        "type": "required_sections",
                        "value": ["Value Proposition", "Revenue Model"],
                        "must_pass": False,
                    },
                ],
            ),
            TaskTemplate(
                sequence=4,
                agent_role="writer",
                title_pattern="Write market analysis section for: {goal}",
                description_pattern=(
                    "Write the detailed market analysis section of the "
                    "business plan for: {goal}. Include industry overview, "
                    "target market analysis, customer personas, competitive "
                    "landscape, and market entry strategy."
                ),
                expected_output="Market analysis section",
                complexity="moderate",
                depends_on=["task_5"],
                verification_criteria=[
                    {"type": "min_length", "value": 600, "must_pass": True},
                    {
                        "type": "required_sections",
                        "value": [
                            "Industry Overview",
                            "Target Market",
                            "Competitive Landscape",
                        ],
                        "must_pass": True,
                    },
                ],
            ),
            TaskTemplate(
                sequence=5,
                agent_role="analyst",
                title_pattern="Write financial projections section for: {goal}",
                description_pattern=(
                    "Write the financial projections section for: {goal}. "
                    "Include detailed revenue forecasts, expense budgets, "
                    "cash flow projections, break-even analysis, and "
                    "funding requirements with use of proceeds."
                ),
                expected_output="Financial projections section",
                complexity="complex",
                depends_on=["task_6"],
                verification_criteria=[
                    {"type": "min_length", "value": 600, "must_pass": True},
                    {
                        "type": "required_sections",
                        "value": [
                            "Revenue Forecast",
                            "Cash Flow",
                            "Break-Even Analysis",
                        ],
                        "must_pass": True,
                    },
                ],
            ),
            TaskTemplate(
                sequence=6,
                agent_role="writer",
                title_pattern="Synthesize full business plan document for: {goal}",
                description_pattern=(
                    "Merge the executive summary, market analysis, and "
                    "financial projections into a complete, cohesive "
                    "business plan document for: {goal}. Add table of "
                    "contents, operations plan, team structure, and "
                    "implementation timeline. Ensure consistent tone "
                    "and formatting throughout."
                ),
                expected_output="Complete business plan document",
                complexity="complex",
                task_type="synthesis",
                depends_on=["task_5", "task_6", "task_7"],
                verification_criteria=[
                    {"type": "min_length", "value": 1500, "must_pass": True},
                    {
                        "type": "required_sections",
                        "value": [
                            "Executive Summary",
                            "Market Analysis",
                            "Financial Projections",
                            "Operations Plan",
                        ],
                        "must_pass": True,
                    },
                ],
            ),
            # ── Phase 3: Workspace Configuration (3 parallel) ────────
            TaskTemplate(
                sequence=7,
                agent_role="admin",
                title_pattern="Create workspace agents from catalog for: {goal}",
                description_pattern=(
                    "Configure the workspace for ongoing operations by "
                    "deploying relevant agents from the marketplace catalog. "
                    "Based on the business plan for: {goal}, deploy agents "
                    "for key roles: marketing strategist, sales outreach, "
                    "content writer, financial analyst, and project manager. "
                    "Use platform_create_agent and platform_install_skill "
                    "to set up each agent."
                ),
                required_tools=[
                    "platform_create_agent",
                    "platform_install_skill",
                ],
                expected_output="List of created agents with roles and skills",
                complexity="moderate",
                parallel_group="bp_workspace_config",
                depends_on=["task_8"],
                verification_criteria=[
                    {"type": "min_length", "value": 200, "must_pass": True},
                ],
            ),
            TaskTemplate(
                sequence=7,
                agent_role="admin",
                title_pattern="Create operational playbooks for: {goal}",
                description_pattern=(
                    "Create playbooks to automate recurring business "
                    "operations for: {goal}. Include playbooks for: "
                    "weekly market monitoring, monthly financial review, "
                    "content publishing pipeline, and customer outreach "
                    "cadence. Use platform_create_playbook for each."
                ),
                required_tools=["platform_create_playbook"],
                expected_output="List of created playbooks with schedules",
                complexity="moderate",
                parallel_group="bp_workspace_config",
                depends_on=["task_8"],
                verification_criteria=[
                    {"type": "min_length", "value": 200, "must_pass": True},
                ],
            ),
            TaskTemplate(
                sequence=7,
                agent_role="admin",
                title_pattern="Create board tasks and configure heartbeats for: {goal}",
                description_pattern=(
                    "Set up the task board with initial action items derived "
                    "from the business plan for: {goal}. Create board tasks "
                    "for key milestones and next steps. Configure agent "
                    "heartbeats for autonomous monitoring. Use "
                    "platform_board_summary and "
                    "platform_configure_agent_heartbeat."
                ),
                required_tools=[
                    "platform_board_summary",
                    "platform_configure_agent_heartbeat",
                ],
                expected_output="Board tasks and heartbeat configurations",
                complexity="moderate",
                parallel_group="bp_workspace_config",
                depends_on=["task_8"],
                verification_criteria=[
                    {"type": "min_length", "value": 200, "must_pass": True},
                ],
            ),
            # ── Phase 4: Verification ─────────────────────────────────
            TaskTemplate(
                sequence=8,
                agent_role="reviewer",
                title_pattern="Review business plan and workspace setup for: {goal}",
                description_pattern=(
                    "Review the complete business plan document and "
                    "workspace configuration for: {goal}. Verify the "
                    "plan's internal consistency, check financial "
                    "projections for reasonableness, confirm all workspace "
                    "agents and playbooks are properly configured, and "
                    "produce a final summary report with any "
                    "recommendations."
                ),
                expected_output="Final review report with summary and recommendations",
                complexity="moderate",
                task_type="review",
                depends_on=["task_9", "task_10", "task_11"],
                verification_criteria=[
                    {"type": "min_length", "value": 400, "must_pass": True},
                    {
                        "type": "required_sections",
                        "value": ["Summary", "Recommendations"],
                        "must_pass": False,
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
      verification_criteria, temp_id, sequence_number, task_type,
      complexity, parallel_group.

    Task numbering: tasks are assigned sequential temp_ids (task_1, task_2, ...)
    based on their position in the task_templates list. Templates use explicit
    depends_on references to these temp_ids instead of auto-chaining.
    """
    tasks: List[Dict[str, object]] = []

    for idx, tt in enumerate(template.task_templates, start=1):
        title = tt.title_pattern.replace("{goal}", goal)
        description = tt.description_pattern.replace("{goal}", goal)
        temp_id = f"task_{idx}"

        # Use explicit depends_on from template; fall back to auto-chain
        if tt.depends_on is not None:
            depends_on = list(tt.depends_on)
        elif tt.sequence > 1:
            # Legacy fallback: chain to previous task
            depends_on = [f"task_{idx - 1}"]
        else:
            depends_on = []

        # Resolve task_type from template field
        task_type = tt.task_type
        if task_type == "llm_generation":
            task_type = TaskType.LLM_GENERATION.value
        elif task_type == "synthesis":
            task_type = TaskType.SYNTHESIS.value
        elif task_type == "review":
            task_type = TaskType.REVIEW.value

        tasks.append({
            "temp_id": temp_id,
            "title": title,
            "description": description,
            "agent_role": tt.agent_role,
            "sequence_number": tt.sequence,
            "task_type": task_type,
            "required_tools": list(tt.required_tools),
            "dependencies": depends_on,
            "verification_criteria": list(tt.verification_criteria),
            "complexity": tt.complexity,
            "parallel_group": tt.parallel_group,
        })

    return tasks
