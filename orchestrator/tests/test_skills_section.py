"""PRD-137 Fix #5: Primary skill not truncated, auxiliary skills capped."""
import asyncio
import importlib.util
import pathlib
import sys
import types
from unittest.mock import MagicMock

_ROOT = pathlib.Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Stub core.context_guard so base.py loads without the full module graph.
# PRD-201 S2: base.py counts + truncates via core.context_guard (the char/4
# TokenEstimator was deleted); the char/4 stub preserves the exact truncation
# boundary this file's assertions were written against (aux capped at
# 5000 tokens → 20000 chars).
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
    """Load base.py + skills.py under a fake module graph, then restore.

    The fakes only need to exist while the target files execute — the loaded
    classes capture their dependencies at exec time. Restoring sys.modules
    afterwards stops the fake ``modules`` package (with an emptied ``__path__``)
    from leaking into the collection of sibling test modules. (PRD-142 W2-S2b.)
    """
    _keys = (
        "modules",
        "modules.context",
        "core.context_guard",
        "modules.context.sections",
        "modules.context.sections.base",
    )
    _saved = {k: sys.modules.get(k) for k in _keys}
    try:
        # Assign fresh fake packages — never mutate a real cached package's
        # __path__ in place (setdefault + __path__=[] would corrupt the real
        # ``modules`` package if it was already imported).
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
    # asyncio.run() creates a fresh loop per call, so this is robust when a
    # prior test in the same process already ran asyncio.run() (which leaves
    # no current event loop on 3.10). get_event_loop() would raise there.
    return asyncio.run(section.render(ctx))


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
