"""PRD-202 S2 (P2-21) — trigger-based L2 activation.

Pins:
1. SkillsSection renders L1 metadata (name + description) for non-core skills —
   NOT their full bodies (the per-skill prompt-cost cut).
2. The load_skill tool injects a skill's L2 body (prompt_template) on trigger.
3. An untriggered skill costs L1 only (the measured token delta).
4. Source-grep guard: SkillsSection no longer renders full L2 bodies
   unconditionally — the always-inject path (and the 5k aux budget) is gone.

Pure/mocked — isolated section load with a fake estimator; the load_skill
handler runs against a chainable mock session; no DB, no model.
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
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_SKILLS_SRC = _ROOT / "modules" / "context" / "sections" / "skills.py"


# ---------------------------------------------------------------------------
# Isolated section load (fake estimator)
# ---------------------------------------------------------------------------

_estimator_stub = types.ModuleType("modules.context.estimator")


class _FakeEstimator:
    def estimate(self, text):
        return len(text) // 4


_estimator_stub.TokenEstimator = _FakeEstimator


def _load_skills_section():
    keys = (
        "modules", "modules.context", "modules.context.estimator",
        "modules.context.sections", "modules.context.sections.base",
    )
    saved = {k: sys.modules.get(k) for k in keys}
    try:
        for name in ("modules", "modules.context", "modules.context.sections"):
            pkg = types.ModuleType(name)
            pkg.__path__ = []
            sys.modules[name] = pkg
        sys.modules["modules.context.estimator"] = _estimator_stub
        base = importlib.util.module_from_spec(importlib.util.spec_from_file_location(
            "modules.context.sections.base", _ROOT / "modules/context/sections/base.py"))
        sys.modules["modules.context.sections.base"] = base
        base.__spec__.loader.exec_module(base)
        skills = importlib.util.module_from_spec(importlib.util.spec_from_file_location(
            "modules.context.sections.skills", _SKILLS_SRC))
        skills.__spec__.loader.exec_module(skills)
        return skills.SkillsSection, base.SectionContext
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _skill(name, body, description="", sid=None):
    s = MagicMock()
    s.id = sid
    s.name = name
    s.prompt_template = body
    s.description = description
    s.is_active = True
    s.tools_schema = None
    s.content_hash = None
    return s


def _render(skills_list):
    SkillsSection, SectionContext = _load_skills_section()
    agent = MagicMock()
    agent.skills = skills_list
    ctx = SectionContext(agent=agent, workspace_id="ws_test")
    return asyncio.run(SkillsSection().render(ctx))


# ---------------------------------------------------------------------------
# 1 + 3. L1-only render / token delta
# ---------------------------------------------------------------------------

def test_skills_section_renders_l1_only():
    out = _render([_skill("growth", "FULL BODY " * 400, description="Growth playbooks")])
    assert "growth" in out
    assert "Growth playbooks" in out       # L1 description
    assert "FULL BODY" not in out          # L2 body NOT pre-injected


def test_untriggered_skill_costs_l1_only():
    huge = "z" * 40000
    out = _render([_skill("reporter", huge, description="Builds reports")])
    assert huge not in out
    assert len(out) < 2000  # L1-sized, not ~40k


# ---------------------------------------------------------------------------
# 2. load_skill injects the L2 body
# ---------------------------------------------------------------------------

def _mock_session_returning(skill):
    q = MagicMock()
    q.filter.return_value = q
    q.order_by.return_value = q
    q.first.return_value = skill
    db = MagicMock()
    db.query.return_value = q
    return db


def test_load_skill_tool_injects_l2_body():
    from modules.tools.discovery.handlers_skill_runtime import load_skill

    skill = MagicMock()
    skill.id = 12
    skill.name = "growth"
    skill.prompt_template = "# Growth\n\nDetailed step-by-step body."
    db = _mock_session_returning(skill)

    result = asyncio.run(load_skill(db, "ws-1", {"name": "growth"}))
    assert result["success"] is True
    assert result["skill"] == "growth"
    assert "Detailed step-by-step body." in result["content"]


def test_load_skill_missing_name_errors():
    from modules.tools.discovery.handlers_skill_runtime import load_skill

    result = asyncio.run(load_skill(MagicMock(), "ws-1", {}))
    assert result["success"] is False
    assert "name" in result["error"].lower()


def test_load_skill_not_found_errors():
    from modules.tools.discovery.handlers_skill_runtime import load_skill

    db = _mock_session_returning(None)
    result = asyncio.run(load_skill(db, "ws-1", {"name": "nope"}))
    assert result["success"] is False
    assert "not found" in result["error"].lower()


# ---------------------------------------------------------------------------
# 4. Source-grep guard: no unconditional L2 render
# ---------------------------------------------------------------------------

def test_skills_section_no_unconditional_l2_render():
    src = _SKILLS_SRC.read_text()
    # the 5,000-token aux budget path is DELETED
    assert "aux_max_tokens" not in src, "the superseded aux-budget path must be gone"
    # PRD-191's phantom priority sort stays gone
    assert 'getattr(s, "priority", 0)' not in src
    # the old always-inject helper is gone (renamed/gated)
    assert "_get_skill_content" not in src, "the unconditional L2 render helper must be gone"
    # full-body render is now GATED by the core always-on set, not unconditional
    assert "_core_always_on_names" in src, "the core-gate must exist"
    assert "SKILL_CORE_ALWAYS_ON" in src, "core set must resolve from config"
    # the L1 trigger instruction is wired
    assert "load_skill" in src


def test_load_skill_registered_as_platform_action():
    from modules.tools.discovery.action_registry import get_action_registry

    reg = get_action_registry()
    # platform_-prefixed to satisfy the tool-reachability namespace invariant
    # (test_tool_reachability): every registry action is platform_* / workspace_*.
    assert reg.get("platform_load_skill") is not None, "platform_load_skill must be registered"
    assert reg.get("platform_run_skill_script") is not None, "platform_run_skill_script must be registered"
    assert reg.get("platform_set_skill_script_execution") is not None
