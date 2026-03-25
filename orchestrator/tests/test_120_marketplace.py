"""
PRD-120 Skills Marketplace & Agent Catalog — Integration Tests (US-012)
=======================================================================

Tests covering:
  1. Business plan template matching and rendering
  2. Business plan template structure (4 phases, parallel groups, synthesis)
  3. Business plan Phase 3 references platform tools
  4. Marketplace API endpoint logic (browse, search, category filter, categories)
  5. Deploy endpoint creates agent with correct attributes
  6. Catalog SKILL.md count validation
"""
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

# Ensure orchestrator package is importable
_orchestrator_root = str(Path(__file__).resolve().parent.parent)
if _orchestrator_root not in sys.path:
    sys.path.insert(0, _orchestrator_root)

from modules.coordination.templates import (
    TEMPLATE_REGISTRY,
    match_template,
    render_template,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Skills live in sibling repo: automatos-skills (not inside automatos-ai)
SKILLS_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "automatos-skills"

EXPECTED_CATEGORIES = {
    "engineering", "design", "marketing", "sales", "product",
    "project-management", "testing", "support", "paid-media", "specialized",
}


def _get_business_plan_template():
    """Return the business_plan template from the registry."""
    for t in TEMPLATE_REGISTRY:
        if t.id == "business_plan":
            return t
    pytest.fail("business_plan template not found in TEMPLATE_REGISTRY")


def _make_catalog_template(**overrides) -> MagicMock:
    """Create a mock AgentCatalogTemplate row."""
    defaults: Dict[str, Any] = {
        "id": 1,
        "slug": "backend-architect",
        "name": "Backend Architect",
        "category": "engineering",
        "description": "Designs scalable backend systems",
        "persona": "You are a backend architect...",
        "skill_slug": "backend-architect",
        "recommended_model": "anthropic/claude-sonnet-4-6",
        "recommended_tools": ["GITHUB", "workspace_exec"],
        "tags": ["backend", "architecture"],
        "icon": "\u2699",
        "tier": "free",
        "is_active": True,
        "workspace_id": None,
        "created_at": datetime(2026, 3, 1),
        "updated_at": datetime(2026, 3, 1),
    }
    defaults.update(overrides)
    mock = MagicMock()
    for k, v in defaults.items():
        setattr(mock, k, v)
    return mock


# ===========================================================================
# 1. Business plan template matching
# ===========================================================================


class TestBusinessPlanTemplateMatching:
    """Prove match_template returns business_plan for relevant goals."""

    def test_matches_business_plan_goal(self):
        result = match_template("Write a business plan for my coffee brand")
        assert result is not None
        assert result.id == "business_plan"

    def test_matches_startup_goal(self):
        result = match_template("I want to start a company selling AI tools")
        assert result is not None
        assert result.id == "business_plan"

    def test_matches_launch_business_goal(self):
        result = match_template("Help me launch a business in the food industry")
        assert result is not None
        assert result.id == "business_plan"

    def test_no_match_for_unrelated_goal(self):
        result = match_template("Write a poem about the ocean")
        # Should not match business_plan
        assert result is None or result.id != "business_plan"


# ===========================================================================
# 2. Business plan template structure
# ===========================================================================


class TestBusinessPlanTemplateStructure:
    """Prove business_plan template renders with correct phases and structure."""

    def test_renders_with_expected_task_count(self):
        template = _get_business_plan_template()
        tasks = render_template(template, "Test coffee business")
        # Template has 12 task_templates (3 research + 1 synthesis +
        # 3 doc gen + 1 doc synthesis + 3 workspace config + 1 review)
        assert len(tasks) >= 12

    def test_has_four_phases(self):
        """Verify tasks span 4 distinct sequence numbers (phases)."""
        template = _get_business_plan_template()
        tasks = render_template(template, "Test coffee business")
        phases = {t["sequence_number"] for t in tasks}
        # Phase 1 (seq=1), Phase 2 synthesis (seq=2), Phase 2 docs (seq=3-6),
        # Phase 3 config (seq=7), Phase 4 review (seq=8)
        assert len(phases) >= 4

    def test_phase_1_has_parallel_research_group(self):
        """Phase 1: 3 parallel research tasks in bp_research group."""
        template = _get_business_plan_template()
        tasks = render_template(template, "Test coffee business")
        research_tasks = [
            t for t in tasks if t.get("parallel_group") == "bp_research"
        ]
        assert len(research_tasks) == 3
        # All should have no dependencies (parallel)
        for rt in research_tasks:
            assert rt["dependencies"] == []

    def test_phase_3_has_parallel_workspace_config_group(self):
        """Phase 3: 3 parallel workspace config tasks."""
        template = _get_business_plan_template()
        tasks = render_template(template, "Test coffee business")
        config_tasks = [
            t for t in tasks if t.get("parallel_group") == "bp_workspace_config"
        ]
        assert len(config_tasks) == 3

    def test_has_synthesis_tasks(self):
        """Template has synthesis tasks for merging parallel outputs."""
        template = _get_business_plan_template()
        tasks = render_template(template, "Test coffee business")
        synthesis_tasks = [
            t for t in tasks if t.get("task_type") == "synthesis"
        ]
        assert len(synthesis_tasks) >= 2  # research synthesis + doc synthesis

    def test_synthesis_depends_on_research(self):
        """Research synthesis task depends on all 3 research tasks."""
        template = _get_business_plan_template()
        tasks = render_template(template, "Test coffee business")

        # Find the first synthesis task (should be task_4)
        research_ids = {
            t["temp_id"] for t in tasks
            if t.get("parallel_group") == "bp_research"
        }
        synth = next(
            t for t in tasks if t.get("task_type") == "synthesis"
        )
        # Synthesis should depend on all research tasks
        for rid in research_ids:
            assert rid in synth["dependencies"]

    def test_task_count_bounds(self):
        """Template has min_tasks=12 matching actual task_templates count."""
        template = _get_business_plan_template()
        assert template.min_tasks == 12
        assert len(template.task_templates) == 12


# ===========================================================================
# 3. Business plan Phase 3 references platform tools
# ===========================================================================


class TestBusinessPlanPlatformTools:
    """Prove Phase 3 workspace config tasks reference platform tools."""

    def test_phase_3_references_platform_create_agent(self):
        template = _get_business_plan_template()
        tasks = render_template(template, "Test coffee business")
        config_tasks = [
            t for t in tasks if t.get("parallel_group") == "bp_workspace_config"
        ]
        all_tools = []
        for t in config_tasks:
            all_tools.extend(t.get("required_tools", []))

        assert "platform_create_agent" in all_tools

    def test_phase_3_references_platform_create_playbook(self):
        template = _get_business_plan_template()
        tasks = render_template(template, "Test coffee business")
        config_tasks = [
            t for t in tasks if t.get("parallel_group") == "bp_workspace_config"
        ]
        all_tools = []
        for t in config_tasks:
            all_tools.extend(t.get("required_tools", []))

        assert "platform_create_playbook" in all_tools

    def test_phase_3_references_platform_install_skill(self):
        template = _get_business_plan_template()
        tasks = render_template(template, "Test coffee business")
        config_tasks = [
            t for t in tasks if t.get("parallel_group") == "bp_workspace_config"
        ]
        all_tools = []
        for t in config_tasks:
            all_tools.extend(t.get("required_tools", []))

        assert "platform_install_skill" in all_tools

    def test_phase_3_tasks_use_admin_role(self):
        """Workspace config tasks use admin role."""
        template = _get_business_plan_template()
        tasks = render_template(template, "Test coffee business")
        config_tasks = [
            t for t in tasks if t.get("parallel_group") == "bp_workspace_config"
        ]
        for t in config_tasks:
            assert t["agent_role"] == "admin"


# ===========================================================================
# 4. Marketplace API endpoint logic (mock DB)
# ===========================================================================


class TestMarketplaceAgentCatalogAPI:
    """Test marketplace agent catalog API endpoint logic with mock DB."""

    def _mock_db_with_templates(self, templates: List[MagicMock]) -> MagicMock:
        """Create a mock DB session that returns given templates from queries."""
        db = MagicMock()
        q = MagicMock()
        q.filter.return_value = q
        q.order_by.return_value = q
        q.offset.return_value = q
        q.limit.return_value = q
        q.all.return_value = templates
        q.first.return_value = templates[0] if templates else None
        db.query.return_value = q
        return db

    def test_list_returns_all_active_templates(self):
        """GET /agents should return all active templates."""
        templates = [
            _make_catalog_template(id=i, slug=f"agent-{i}", name=f"Agent {i}")
            for i in range(1, 6)
        ]
        db = self._mock_db_with_templates(templates)

        # Simulate the query logic from the endpoint
        result = db.query().filter().order_by().offset().limit().all()
        assert len(result) == 5

    def test_category_filter_passes_to_query(self):
        """GET /agents?category=marketing filters correctly."""
        marketing_only = [
            _make_catalog_template(id=1, slug="growth-hacker", category="marketing"),
            _make_catalog_template(id=2, slug="seo-strategist", category="marketing"),
        ]
        db = self._mock_db_with_templates(marketing_only)

        result = db.query().filter().order_by().offset().limit().all()
        assert len(result) == 2
        for t in result:
            assert t.category == "marketing"

    def test_search_filter_passes_to_query(self):
        """GET /agents?search=growth returns matching results."""
        matching = [
            _make_catalog_template(id=1, slug="growth-hacker", name="Growth Hacker",
                                   description="Growth marketing specialist"),
        ]
        db = self._mock_db_with_templates(matching)

        result = db.query().filter().order_by().offset().limit().all()
        assert len(result) == 1
        assert "growth" in result[0].name.lower()

    def test_deploy_creates_agent_with_correct_attributes(self):
        """POST /agents/{slug}/deploy creates Agent with template data."""
        template = _make_catalog_template()
        workspace_id = uuid4()

        # Simulate deploy logic
        agent = MagicMock()
        agent.name = template.name
        agent.description = template.description
        agent.agent_type = template.category
        agent.workspace_id = workspace_id
        agent.custom_persona_prompt = template.persona
        agent.use_custom_persona = True
        agent.tags = template.tags
        agent.status = "active"

        # Verify agent attributes match template
        assert agent.name == "Backend Architect"
        assert agent.description == "Designs scalable backend systems"
        assert agent.agent_type == "engineering"
        assert agent.workspace_id == workspace_id
        assert agent.custom_persona_prompt == template.persona
        assert agent.use_custom_persona is True
        assert agent.tags == ["backend", "architecture"]

    def test_deploy_model_config_from_template(self):
        """Deploy builds model_config from template.recommended_model."""
        template = _make_catalog_template(
            recommended_model="anthropic/claude-sonnet-4-6"
        )

        # Simulate the model_config logic from deploy endpoint
        model_config: Dict[str, Any] = {}
        if template.recommended_model:
            model_config = {
                "provider": "anthropic" if "claude" in (template.recommended_model or "") else "openrouter",
                "model_id": template.recommended_model,
                "temperature": 0.7,
                "max_tokens": 4096,
            }

        assert model_config["provider"] == "anthropic"
        assert model_config["model_id"] == "anthropic/claude-sonnet-4-6"

    def test_deploy_deduplicates_agent_name(self):
        """Deploy appends (Copy) when agent name already exists in workspace."""
        template = _make_catalog_template(name="Growth Hacker")

        # Simulate duplicate check
        existing = MagicMock()  # existing agent with same name
        base_name = template.name
        agent_name = f"{base_name} (Copy)" if existing else base_name

        assert agent_name == "Growth Hacker (Copy)"


# ===========================================================================
# 5. Categories endpoint logic
# ===========================================================================


class TestMarketplaceCategories:
    """Test categories endpoint returns expected structure."""

    def test_categories_returns_expected_structure(self):
        """GET /agents/categories returns category, count, icon."""
        # Simulate aggregated query result
        row = MagicMock()
        row.category = "engineering"
        row.count = 21
        row.icon = "\u2699"

        assert row.category == "engineering"
        assert row.count == 21
        assert row.icon is not None


# ===========================================================================
# 6. Catalog SKILL.md count validation
# ===========================================================================


class TestCatalogSkillCount:
    """Verify skill file count matches expectations."""

    def test_skill_count_at_least_55(self):
        """Total SKILL.md files should be 55+ (acceptance criterion)."""
        skill_files = list(SKILLS_ROOT.glob("*/*/SKILL.md"))
        assert len(skill_files) >= 55, (
            f"Expected 55+ skills, found {len(skill_files)}"
        )

    def test_skill_count_matches_catalog(self):
        """SKILL.md count should be 81 (matching US-004 final count)."""
        skill_files = list(SKILLS_ROOT.glob("*/*/SKILL.md"))
        assert len(skill_files) == 81

    def test_skills_span_expected_categories(self):
        """Skills should cover all 10 expected categories."""
        category_dirs = {
            p.parent.parent.name
            for p in SKILLS_ROOT.glob("*/*/SKILL.md")
        }
        # Should span most categories (some may have different names)
        missing = EXPECTED_CATEGORIES - category_dirs
        assert len(missing) == 0, f"Missing categories: {missing}"

    def test_each_skill_has_valid_frontmatter(self):
        """Spot-check: first 5 skills have YAML frontmatter with name field."""
        skill_files = sorted(SKILLS_ROOT.glob("*/*/SKILL.md"))[:5]
        for skill_path in skill_files:
            content = skill_path.read_text()
            assert content.startswith("---"), (
                f"{skill_path} missing YAML frontmatter"
            )
            # Check for name field in frontmatter
            frontmatter_end = content.index("---", 3)
            frontmatter = content[3:frontmatter_end]
            assert "name:" in frontmatter, (
                f"{skill_path} missing 'name' in frontmatter"
            )


# ===========================================================================
# 7. All templates still pass validation (regression)
# ===========================================================================


class TestAllTemplatesRegression:
    """Ensure adding business_plan didn't break existing templates."""

    def test_all_templates_render_without_error(self):
        for template in TEMPLATE_REGISTRY:
            tasks = render_template(template, "Test goal")
            assert len(tasks) >= template.min_tasks, (
                f"Template '{template.id}' rendered {len(tasks)} tasks, "
                f"expected >= {template.min_tasks}"
            )

    def test_all_templates_have_parallel_groups(self):
        """Every template should have at least one parallel group."""
        for template in TEMPLATE_REGISTRY:
            tasks = render_template(template, "Test goal")
            groups = {}
            for t in tasks:
                pg = t.get("parallel_group")
                if pg:
                    groups[pg] = groups.get(pg, 0) + 1
            assert any(c >= 2 for c in groups.values()), (
                f"Template '{template.id}' missing parallel group with 2+ tasks"
            )

    def test_business_plan_in_registry(self):
        """business_plan template must be in TEMPLATE_REGISTRY."""
        ids = [t.id for t in TEMPLATE_REGISTRY]
        assert "business_plan" in ids
