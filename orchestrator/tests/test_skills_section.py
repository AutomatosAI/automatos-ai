"""PRD-137 Fix #5: Primary skill not truncated, auxiliary skills capped."""
import asyncio
import importlib.util
import pathlib
import sys
import types
from unittest.mock import MagicMock

_ROOT = pathlib.Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Stub estimator so base.py loads without the full module graph
# ---------------------------------------------------------------------------

_estimator_stub = types.ModuleType("modules.context.estimator")


class _FakeEstimator:
    def estimate(self, text):
        return len(text) // 4


_estimator_stub.TokenEstimator = _FakeEstimator
sys.modules.setdefault("modules", types.ModuleType("modules"))
sys.modules["modules"].__path__ = []
sys.modules.setdefault("modules.context", types.ModuleType("modules.context"))
sys.modules["modules.context"].__path__ = []
sys.modules["modules.context.estimator"] = _estimator_stub
sys.modules.setdefault("modules.context.sections", types.ModuleType("modules.context.sections"))
sys.modules["modules.context.sections"].__path__ = []

_base_mod = importlib.util.module_from_spec(
    importlib.util.spec_from_file_location(
        "modules.context.sections.base",
        _ROOT / "modules" / "context" / "sections" / "base.py",
    )
)
sys.modules["modules.context.sections.base"] = _base_mod
_base_mod.__spec__.loader.exec_module(_base_mod)

_skills_mod = importlib.util.module_from_spec(
    importlib.util.spec_from_file_location(
        "modules.context.sections.skills",
        _ROOT / "modules" / "context" / "sections" / "skills.py",
    )
)
_skills_mod.__spec__.loader.exec_module(_skills_mod)

SkillsSection = _skills_mod.SkillsSection
SectionContext = _base_mod.SectionContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_skill(name, content, is_active=True, tools_schema=None, priority=0):
    skill = MagicMock()
    skill.name = name
    skill.prompt_template = content
    skill.is_active = is_active
    skill.tools_schema = tools_schema
    skill.priority = priority
    return skill


def _make_agent(skills):
    agent = MagicMock()
    agent.skills = skills
    return agent


def _make_ctx(agent):
    return SectionContext(agent=agent, workspace_id="ws_test")


def _render(section, ctx):
    return asyncio.get_event_loop().run_until_complete(section.render(ctx))


# ── Primary skill not truncated ─────────────────────────────────────


def test_single_large_skill_not_truncated():
    """A single 11K-token skill should render fully (old cap was 3000)."""
    large_content = "x" * 50000  # ~12,500 tokens at 4 chars/token
    agent = _make_agent([_make_skill("platform-management", large_content)])
    section = SkillsSection()
    result = _render(section, _make_ctx(agent))
    assert len(result) >= 50000


def test_primary_skill_content_preserved():
    content = "# Platform Management\n\nFull skill content here with all sections."
    agent = _make_agent([_make_skill("platform-management", content)])
    section = SkillsSection()
    result = _render(section, _make_ctx(agent))
    assert content in result


# ── Auxiliary skills capped ─────────────────────────────────────────


def test_auxiliary_skills_truncated():
    primary = _make_skill("primary-skill", "Primary content")
    # aux_max_tokens=5000 → 20000 chars max
    aux_content = "y" * 30000  # exceeds 5000 tokens
    aux = _make_skill("aux-skill", aux_content)
    agent = _make_agent([primary, aux])

    section = SkillsSection()
    result = _render(section, _make_ctx(agent))

    assert "Primary content" in result
    # The aux should be truncated to ~5000*4=20000 chars (±1 from boundary)
    y_count = result.count("y")
    assert y_count <= 20001
    assert y_count < 30000  # definitely truncated from original


def test_primary_not_truncated_even_with_aux():
    primary_content = "p" * 50000
    aux_content = "a" * 100
    agent = _make_agent([
        _make_skill("big-primary", primary_content),
        _make_skill("small-aux", aux_content),
    ])

    section = SkillsSection()
    result = _render(section, _make_ctx(agent))
    assert result.count("p") == 50000
    assert "a" * 100 in result


# ── No skills ──────────────────────────────────────────────────────


def test_no_skills_returns_empty():
    agent = _make_agent([])
    section = SkillsSection()
    result = _render(section, _make_ctx(agent))
    assert result == ""


def test_inactive_skills_excluded():
    agent = _make_agent([_make_skill("dead", "content", is_active=False)])
    section = SkillsSection()
    result = _render(section, _make_ctx(agent))
    assert result == ""


# ── Tool names from schema ──────────────────────────────────────────


def test_skill_tool_names_included():
    schema = {"tools": [{"name": "search_knowledge"}, {"name": "write_file"}]}
    agent = _make_agent([_make_skill("scout", "Scout skill", tools_schema=schema)])
    section = SkillsSection()
    result = _render(section, _make_ctx(agent))
    assert "search_knowledge" in result
    assert "write_file" in result
    assert "Using Your Skill Tools" in result
