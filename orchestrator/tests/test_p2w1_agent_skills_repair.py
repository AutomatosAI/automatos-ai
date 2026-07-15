"""PRD-191 — agents-skills data repair (P2-10): stop paying the duplicate-skill tax.

Prod state this wave repairs: 205 duplicated active skill names, the global
platform-management skill seeded 5×, the workspace Auto agent linked to it
4× — and SkillsSection rendering every link, taxing every Auto turn with
~5k duplicated tokens. Pins:

1. S1 — the migration's dedupe policy keeps exactly one link per
   (agent_id, skill_id), lowest row wins; the model now CARRIES the unique
   constraint (schema-shape assert, no DB).
2. S4 — priority is REAL: a column on the attachment, and Agent.skills is
   ordered by it at the relationship level (F054 closed with a decision).
3. S2 — SkillsSection never renders the same body twice: dedup by id and by
   (name, content_hash), first (= highest-priority) instance wins.
4. S3 — the seeders are idempotent under concurrency: a lost insert race
   re-selects and returns the existing row (never a silent no-seed), and the
   link assign emits ON CONFLICT (agent_id, skill_id) DO NOTHING.
5. S5 — a foreign workspace's private skill cannot be attached by id on the
   canonical agents router (visibility parity with api/skills.py).

All pure — mocked sessions / in-memory fixtures; no DB, no network.
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

for _k, _v in {
    "POSTGRES_USER": "test", "POSTGRES_PASSWORD": "test",
    "POSTGRES_HOST": "localhost", "POSTGRES_PORT": "5432", "POSTGRES_DB": "test",
}.items():
    os.environ.setdefault(_k, _v)

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# 1. S1 — migration dedupe policy + schema shape
# ---------------------------------------------------------------------------

def _load_migration():
    spec = importlib.util.spec_from_file_location(
        "prd191_migration",
        _ROOT / "alembic" / "versions" / "prd191_agent_skills_unique_and_priority.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_agent_skills_dedupe_keeps_one_link_per_pair():
    mig = _load_migration()
    # Mirrors prod: Auto agent 340 → platform-management (skill 7) linked 4×.
    rows = [
        (1, 340, 7), (2, 340, 7), (3, 340, 7), (4, 340, 7),
        (5, 340, 9),
        (6, 341, 7),
    ]
    assert mig.survivors(rows) == [1, 5, 6]


def test_agent_skills_dedupe_chains_off_prd187_head():
    src = (
        _ROOT / "alembic" / "versions" / "prd191_agent_skills_unique_and_priority.py"
    ).read_text()
    assert 'down_revision = "prd187_s5_drop_memory_relics"' in src, (
        "must chain onto the mainline head — a floating down_revision=None is "
        "exactly what left the earlier skills dedupe a loose fixup"
    )
    # dedupe THEN constrain — the order is load-bearing on live 4× links
    assert src.index("DELETE FROM agent_skills") < src.index("create_unique_constraint")


def test_agent_skills_model_has_unique_and_priority():
    from core.models.core import agent_skills

    uniques = [
        c for c in agent_skills.constraints
        if type(c).__name__ == "UniqueConstraint"
    ]
    assert any(
        [col.name for col in c.columns] == ["agent_id", "skill_id"] for c in uniques
    ), "agent_skills must carry UNIQUE(agent_id, skill_id)"
    assert "priority" in agent_skills.c, "the attachment-level priority column is real now"


def test_agent_skills_relationship_ordered_by_priority():
    from core.models.core import Agent

    order_by = Agent.skills.property.order_by
    assert order_by, "Agent.skills must be ordered (the primary slot is a decision)"
    assert "priority" in str(order_by[0]), (
        "Agent.skills order_by must read agent_skills.priority — not load order"
    )


# ---------------------------------------------------------------------------
# 2. S2 + S4 — SkillsSection dedup + real ordering (isolated module load,
#    same fake-graph technique as tests/test_skills_section.py)
# ---------------------------------------------------------------------------

# PRD-201 S2: base.py counts + truncates via core.context_guard (the char/4
# TokenEstimator was deleted). Stub core.context_guard (cheap, no tiktoken) so
# base loads in isolation with the char/4 behaviour the assertions expect.
_cg_stub = types.ModuleType("core.context_guard")
_cg_stub.count_tokens = lambda text: len(text or "") // 4


def _cg_truncate(text, max_tokens, *, suffix=""):
    if not text or max_tokens <= 0:
        return text
    limit = max_tokens * 4
    return text if len(text) <= limit else text[:limit] + suffix


_cg_stub.truncate_to_token_budget = _cg_truncate


def _load_skills_section():
    keys = (
        "modules", "modules.context", "core.context_guard",
        "modules.context.sections", "modules.context.sections.base",
    )
    saved = {k: sys.modules.get(k) for k in keys}
    try:
        for name in ("modules", "modules.context", "modules.context.sections"):
            pkg = types.ModuleType(name)
            pkg.__path__ = []
            sys.modules[name] = pkg
        sys.modules["core.context_guard"] = _cg_stub
        base = importlib.util.module_from_spec(importlib.util.spec_from_file_location(
            "modules.context.sections.base", _ROOT / "modules/context/sections/base.py"))
        sys.modules["modules.context.sections.base"] = base
        base.__spec__.loader.exec_module(base)
        skills = importlib.util.module_from_spec(importlib.util.spec_from_file_location(
            "skills_section_under_test", _ROOT / "modules/context/sections/skills.py"))
        skills.__spec__.loader.exec_module(skills)
        return skills
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _skill(sid, name, content, content_hash=None):
    s = MagicMock()
    s.id = sid
    s.name = name
    s.prompt_template = content
    s.is_active = True
    s.tools_schema = None
    s.content_hash = content_hash or f"hash-{sid}"
    return s


def _render(skills_list):
    mod = _load_skills_section()
    section = mod.SkillsSection()
    ctx = MagicMock()
    ctx.agent = MagicMock()
    ctx.agent.skills = skills_list
    ctx.kwargs = {}
    return section._build(ctx)


def test_skills_section_dedupes_by_id():
    same = _skill(7, "platform-management", "PLATFORM BODY " * 50)
    out = _render([same, same])
    assert out.count("PLATFORM BODY") == 50, (
        "the same skill row attached twice must render once"
    )


def test_skills_section_dedupes_identical_bodies():
    body = "SHARED 26KB BODY " * 30
    builtin = _skill(7, "platform-management", body, content_hash="same-hash")
    dupe_row = _skill(99, "platform-management", body, content_hash="same-hash")
    out = _render([builtin, dupe_row])
    # one render only — the aux budget is not consumed by the duplicate
    assert out.count("SHARED 26KB BODY") == 30


def test_skills_section_first_skill_is_primary_by_arrival_order():
    # Agent.skills arrives pre-ordered by attachment priority (model order_by);
    # the section must respect that order. PRD-202 S2 supersedes the full-body
    # "primary" render with L1 metadata for non-core skills — so the ordering
    # invariant is now asserted on the L1 catalog (names in arrival order).
    hi = _skill(1, "hi-priority", "HI-BODY unique-hi")
    lo = _skill(2, "lo-priority", "LO-BODY unique-lo")
    out = _render([hi, lo])
    assert out.index("hi-priority") < out.index("lo-priority")
    flipped = _render([lo, hi])
    assert flipped.index("lo-priority") < flipped.index("hi-priority")


def test_skills_section_phantom_priority_sort_is_gone():
    src = (_ROOT / "modules/context/sections/skills.py").read_text()
    assert 'getattr(s, "priority", 0)' not in src, (
        "the phantom Skill.priority sort must not survive (F054)"
    )


# ---------------------------------------------------------------------------
# 3. S3 — seeder idempotency under the race
# ---------------------------------------------------------------------------

def test_platform_skill_upsert_reselects_when_race_lost(monkeypatch):
    from sqlalchemy.exc import IntegrityError

    import core.seeds.seed_auto_agent as seed_mod

    existing = MagicMock(id=42)
    db = MagicMock()
    # first .first() → None (row not there yet); second → the winner's row
    db.query.return_value.filter.return_value.first.side_effect = [None, existing]
    db.flush.side_effect = IntegrityError("stmt", {}, Exception("duplicate key"))
    monkeypatch.setattr(seed_mod, "_PLATFORM_SKILL_PATH", MagicMock(
        exists=lambda: True,
        read_text=lambda encoding: "---\nname: platform-management\n---\nBODY",
    ))

    result = seed_mod._upsert_platform_management_skill(db)

    assert result is existing, (
        "a lost insert race must re-select and return the existing row — "
        "never swallow the IntegrityError into a silent no-seed"
    )


def test_assign_skill_uses_on_conflict():
    from sqlalchemy.dialects import postgresql

    import core.seeds.seed_auto_agent as seed_mod

    db = MagicMock()
    agent = MagicMock(id=340, name="Auto")
    skill = MagicMock(id=7, name="platform-management")

    seed_mod._assign_skill_to_agent(db, agent, skill)

    stmt = db.execute.call_args.args[0]
    sql = str(stmt.compile(dialect=postgresql.dialect()))
    assert "ON CONFLICT (agent_id, skill_id) DO NOTHING" in sql, (
        "the link insert must be conflict-safe on S1's unique constraint"
    )


# ---------------------------------------------------------------------------
# 4. S5 — attach-visibility parity on the canonical router
# ---------------------------------------------------------------------------

def _visibility_fixture(monkeypatch):
    from fastapi import HTTPException

    import api.agents as agents_mod
    import api.skills as skills_mod

    monkeypatch.setattr(skills_mod, "_is_super_admin", lambda ctx: False)
    return agents_mod, HTTPException


def _skill_row(sid, workspace_id):
    row = MagicMock()
    row.id = sid
    row.workspace_id = workspace_id
    return row


def test_attach_rejects_foreign_workspace_skill(monkeypatch):
    agents_mod, HTTPException = _visibility_fixture(monkeypatch)

    ctx = MagicMock()
    ctx.workspace_id = "ws-A"
    foreign = _skill_row(5, "ws-B")
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = [foreign]

    with pytest.raises(HTTPException) as exc:
        agents_mod._fetch_attachable_skills(db, [5], ctx)
    assert exc.value.status_code == 404, (
        "a foreign workspace's private skill must be indistinguishable from a "
        "nonexistent id — never silently attached (and prompt-injected)"
    )


def test_attach_allows_global_and_own_workspace_skills(monkeypatch):
    agents_mod, _ = _visibility_fixture(monkeypatch)

    ctx = MagicMock()
    ctx.workspace_id = "ws-A"
    global_skill = _skill_row(1, None)
    own = _skill_row(2, "ws-A")
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = [global_skill, own]

    out = agents_mod._fetch_attachable_skills(db, [1, 2], ctx)
    assert [s.id for s in out] == [1, 2]
