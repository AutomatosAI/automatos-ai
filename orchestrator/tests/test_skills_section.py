"""PRD-202 S2: trigger-based L2 activation in SkillsSection.

Supersedes the PRD-137 always-inject render (every attached skill's full body,
uncapped-primary + 5k aux budget — both DELETED). New behavior:

  * a ``core`` always-on skill (config.SKILL_CORE_ALWAYS_ON — platform-management)
    renders its full L2 body every turn (Auto's operating manual, Q4);
  * every OTHER attached skill renders only its L1 metadata (name + description)
    plus a load_skill trigger instruction — its body is NOT pre-paid;
  * PRD-191's dedup is preserved.

Isolated module load (fake estimator) so skills.py runs without the full graph.
"""
import asyncio
import importlib.util
import os
import pathlib
import sys
import types
from unittest.mock import MagicMock

for _k, _v in {
    "POSTGRES_USER": "test", "POSTGRES_PASSWORD": "test",
    "POSTGRES_HOST": "localhost", "POSTGRES_PORT": "5432", "POSTGRES_DB": "test",
}.items():
    os.environ.setdefault(_k, _v)

_ROOT = pathlib.Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Stub core.context_guard so base.py loads without the full module graph.
# PRD-201 S2 deleted the char/4 TokenEstimator; base.py now counts + truncates
# via core.context_guard, so the isolated load stubs THAT (a cheap, no-tiktoken
# count_tokens) rather than the removed modules.context.estimator.
# ---------------------------------------------------------------------------

_cg_stub = types.ModuleType("core.context_guard")
_cg_stub.count_tokens = lambda text: len(text or "") // 4


def _cg_truncate(text, max_tokens, *, suffix=""):
    if not text or max_tokens <= 0:
        return text
    limit = max_tokens * 4
    return text if len(text) <= limit else text[:limit] + suffix


_cg_stub.truncate_to_token_budget = _cg_truncate


def _load_sections_isolated():
    _keys = (
        "modules",
        "modules.context",
        "core.context_guard",
        "modules.context.sections",
        "modules.context.sections.base",
    )
    _saved = {k: sys.modules.get(k) for k in _keys}
    try:
        for _name in ("modules", "modules.context", "modules.context.sections"):
            _pkg = types.ModuleType(_name)
            _pkg.__path__ = []
            sys.modules[_name] = _pkg
        sys.modules["core.context_guard"] = _cg_stub

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

        return _skills_mod.SkillsSection, _base_mod.SectionContext
    finally:
        for _k, _v in _saved.items():
            if _v is None:
                sys.modules.pop(_k, None)
            else:
                sys.modules[_k] = _v


SkillsSection, SectionContext = _load_sections_isolated()

# The core always-on skill name (platform-management is the shipped default).
CORE_SKILL = "platform-management"


def _make_skill(name, content, description="", is_active=True, tools_schema=None, sid=None, content_hash=None):
    skill = MagicMock()
    skill.id = sid
    skill.name = name
    skill.prompt_template = content
    skill.description = description
    skill.is_active = is_active
    skill.tools_schema = tools_schema
    skill.content_hash = content_hash
    return skill


def _make_agent(skills):
    agent = MagicMock()
    agent.skills = skills
    return agent


def _make_ctx(agent):
    return SectionContext(agent=agent, workspace_id="ws_test")


def _render(section, ctx):
    return asyncio.run(section.render(ctx))


# ── Core skill: full L2 body always ─────────────────────────────────


def test_core_skill_renders_full_body():
    large = "PLATFORM MANUAL " * 1000
    agent = _make_agent([_make_skill(CORE_SKILL, large, description="core ops")])
    result = _render(SkillsSection(), _make_ctx(agent))
    assert result.count("PLATFORM MANUAL") == 1000  # uncapped, full body


# ── Non-core skill: L1 metadata only (the cost cut) ─────────────────


def test_non_core_skill_renders_l1_only():
    agent = _make_agent([
        _make_skill("growth-hacker", "SECRET BODY " * 500, description="Growth tactics and playbooks"),
    ])
    result = _render(SkillsSection(), _make_ctx(agent))
    # name + description present (L1), full body NOT
    assert "growth-hacker" in result
    assert "Growth tactics and playbooks" in result
    assert "SECRET BODY" not in result


def test_l1_catalog_includes_load_skill_instruction():
    agent = _make_agent([_make_skill("analytics", "body", description="Data analysis")])
    result = _render(SkillsSection(), _make_ctx(agent))
    assert "load_skill" in result  # the trigger instruction is present


def test_untriggered_skill_costs_l1_only():
    """The token delta vs the old always-inject path: a big idle skill costs L1."""
    huge_body = "x" * 40000  # ~10k tokens if inlined
    agent = _make_agent([_make_skill("reporter", huge_body, description="Builds reports")])
    result = _render(SkillsSection(), _make_ctx(agent))
    # The rendered section is L1-sized (name+description+instruction), NOT ~40k.
    assert len(result) < 2000
    assert "x" * 40000 not in result


def test_core_and_non_core_mixed():
    agent = _make_agent([
        _make_skill(CORE_SKILL, "CORE BODY HERE", description="core"),
        _make_skill("side-skill", "SIDE BODY HERE", description="A side capability"),
    ])
    result = _render(SkillsSection(), _make_ctx(agent))
    assert "CORE BODY HERE" in result       # core → full body
    assert "SIDE BODY HERE" not in result    # non-core → L1 only
    assert "side-skill" in result            # but its L1 metadata shows
    assert "A side capability" in result


# ── Dedup preserved (PRD-191) ───────────────────────────────────────


def test_core_skill_deduped_by_id():
    same = _make_skill(CORE_SKILL, "UNIQUE CORE " * 10, sid=7)
    agent = _make_agent([same, same])
    result = _render(SkillsSection(), _make_ctx(agent))
    assert result.count("UNIQUE CORE") == 10  # rendered once, not twice


# ── Empty / inactive ────────────────────────────────────────────────


def test_no_skills_returns_empty():
    assert _render(SkillsSection(), _make_ctx(_make_agent([]))) == ""


def test_inactive_skills_excluded():
    agent = _make_agent([_make_skill("dead", "content", description="x", is_active=False)])
    assert _render(SkillsSection(), _make_ctx(agent)) == ""


# ── Tool names block (kept) ─────────────────────────────────────────


def test_skill_tool_names_included():
    schema = {"tools": [{"name": "search_knowledge"}, {"name": "write_file"}]}
    agent = _make_agent([_make_skill("scout", "Scout skill", description="scout", tools_schema=schema)])
    result = _render(SkillsSection(), _make_ctx(agent))
    assert "search_knowledge" in result
    assert "write_file" in result
    assert "Using Your Skill Tools" in result
