"""PRD-231 US-003 — seed + assign platform-operations as a NON-core skill.

Pure, LLM-free, Postgres-free. The seeder's upsert/assign are exercised with a
MagicMock session (the same shape ``test_p2w1_agent_skills_repair`` uses); the
handler and loader are exercised against the REAL committed ops seed body so the
"returns the cookbook" and "refresh from disk" claims are real content, not a
stub.
"""

import asyncio
import hashlib
import pathlib
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import core.seeds.seed_auto_agent as seed_mod

_SEEDS = pathlib.Path(__file__).resolve().parents[1] / "core" / "seeds"
_OPS_SEED = _SEEDS / "platform-operations-skill.md"


def _reader_body(path: pathlib.Path) -> str:
    """The frontmatter-stripped body exactly as the platform readers compute it."""
    raw = path.read_text(encoding="utf-8").strip()
    return raw.split("---", 2)[2].strip()


def _skill(name, body, description="", *, sid=None, content_hash=None, is_active=True):
    s = SimpleNamespace()
    s.name = name
    s.prompt_template = body
    s.description = description
    s.is_active = is_active
    s.id = sid
    s.content_hash = content_hash
    s.tools_schema = None
    return s


# ── AC1: fresh seeds+assigns both; existing gains ops, idempotent ────────────

def test_fresh_workspace_seeds_and_assigns_both_skills(monkeypatch):
    charter = SimpleNamespace(id=7, name="platform-management")
    ops = SimpleNamespace(id=9, name="platform-operations")
    monkeypatch.setattr(seed_mod, "_upsert_platform_management_skill", lambda db: charter)
    monkeypatch.setattr(seed_mod, "_upsert_platform_operations_skill", lambda db: ops)
    assigned = []
    monkeypatch.setattr(seed_mod, "_assign_skill_to_agent", lambda db, a, s: assigned.append(s.name))

    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None  # no existing agent

    seed_mod.seed_auto_agent(db, uuid.uuid4())

    assert db.add.called  # a fresh Auto agent row was created
    assert assigned == ["platform-management", "platform-operations"]


def test_existing_workspace_gains_ops_assignment_idempotently(monkeypatch):
    """The ensure path runs on EVERY lazy get-or-seed call, so an existing
    workspace (existing agent, charter already linked) picks up the ops
    assignment on its next chat — and re-running is idempotent (no new agent,
    the ops assignment re-issued through the ON CONFLICT insert each time)."""
    fake_agent = SimpleNamespace(id="a1", name="Auto", slug="auto-ws", workspace_id="ws")
    charter = SimpleNamespace(id=7, name="platform-management")
    ops = SimpleNamespace(id=9, name="platform-operations")

    monkeypatch.setattr(seed_mod, "_backfill_auto_persona", lambda a: "current")
    monkeypatch.setattr(seed_mod, "_upsert_platform_management_skill", lambda db: charter)
    monkeypatch.setattr(seed_mod, "_upsert_platform_operations_skill", lambda db: ops)
    assigned = []
    monkeypatch.setattr(seed_mod, "_assign_skill_to_agent", lambda db, a, s: assigned.append(s.name))

    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = fake_agent  # existing agent

    ws = uuid.uuid4()
    seed_mod.seed_auto_agent(db, ws)  # next chat after deploy
    seed_mod.seed_auto_agent(db, ws)  # idempotent re-run

    assert db.add.call_count == 0  # existing agent reused both times, never re-created
    assert assigned.count("platform-operations") == 2  # ensured every call
    assert assigned.count("platform-management") == 2


def test_ops_upsert_creates_builtin_core_row_from_seed():
    """_upsert_platform_operations_skill creates a global builtin-core row whose
    body is the ops seed sans frontmatter and whose description is the seed's
    frontmatter description (the L1 trigger) — no row yet, so it inserts."""
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None  # not seeded yet

    result = seed_mod._upsert_platform_operations_skill(db)

    added = db.add.call_args.args[0]  # the real Skill instance handed to db.add
    assert result is added
    assert added.name == "platform-operations"
    assert added.skill_source == "builtin-core"
    assert added.workspace_id is None
    assert added.category == "agent-role"
    # body is the seed sans frontmatter; description is the frontmatter trigger.
    assert added.prompt_template == _reader_body(_OPS_SEED)
    assert added.description.startswith("The tool-by-tool operations cookbook")
    assert "LOAD THIS" in added.description
    # version came from the seed frontmatter (v1.0.0), not a silent fallback.
    assert added.skill_version == "1.0.0"


def test_ops_upsert_uses_its_own_advisory_lock_key():
    """A distinct advisory-lock key from the charter's, so the two seeders don't
    serialize against each other."""
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = MagicMock(id=1)  # already seeded
    seed_mod._upsert_platform_operations_skill(db)

    # the advisory lock bound the ops key, never the charter key
    bound = [c.args[1] for c in db.execute.call_args_list if len(c.args) > 1 and isinstance(c.args[1], dict)]
    assert any(d.get("k") == "seed:platform-operations" for d in bound)
    assert all(d.get("k") != "seed:platform-management" for d in bound)


# ── AC2: NON-core; render is charter full body + ops one L1 line ─────────────

def test_platform_operations_absent_from_core_always_on():
    from config import config
    from modules.context.sections.skills import SkillsSection

    core = SkillsSection._core_always_on_names()
    assert "platform-management" in core
    assert "platform-operations" not in core
    assert "platform-operations" not in list(config.SKILL_CORE_ALWAYS_ON)


def test_render_shows_charter_body_and_ops_l1_line_only():
    from modules.context.sections.base import SectionContext
    from modules.context.sections.skills import SkillsSection

    charter = _skill("platform-management", "CHARTER FULL BODY " * 20, "charter", sid=1)
    ops = _skill(
        "platform-operations",
        "OPS COOKBOOK BODY " * 500,
        "The tool-by-tool operations cookbook — LOAD THIS before executing any platform operation.",
        sid=2,
    )
    agent = SimpleNamespace(skills=[charter, ops])
    out = asyncio.run(SkillsSection().render(SectionContext(agent=agent, workspace_id="ws")))

    assert "CHARTER FULL BODY" in out           # charter → full L2 body
    assert "OPS COOKBOOK BODY" not in out         # ops body is NOT pre-paid
    assert "platform-operations" in out           # ops present as an L1 catalog line
    assert "The tool-by-tool operations cookbook" in out  # its L1 trigger = description
    # exactly one L1 line for ops (its name appears once outside the trigger text)
    assert out.count("**platform-operations**") == 1


# ── AC3: platform_load_skill 'platform-operations' returns the cookbook ──────

def test_platform_load_skill_returns_ops_cookbook_body():
    from modules.tools.discovery.handlers_skill_runtime import load_skill

    body = _reader_body(_OPS_SEED)
    fake = SimpleNamespace(name="platform-operations", id=9, prompt_template=body)
    q = MagicMock()
    q.filter.return_value = q
    q.order_by.return_value.first.return_value = fake
    db = MagicMock()
    db.query.return_value = q

    result = asyncio.run(load_skill(db, uuid.uuid4(), {"name": "platform-operations"}))

    assert result["success"] is True
    assert result["skill"] == "platform-operations"
    assert "# Platform Operations Reference" in result["content"]
    assert "## 0." in result["content"] and "## 19." in result["content"]


# ── AC4: _BUILTIN_PATHS both entries; refresh a stale ops row from disk ───────

def test_builtin_paths_has_both_entries():
    from modules.agents.services.skill_loader import SkillLoader

    paths = SkillLoader._BUILTIN_PATHS
    assert "platform-management" in paths
    assert "platform-operations" in paths
    assert paths["platform-operations"].name == "platform-operations-skill.md"
    assert paths["platform-operations"].exists()


def test_refresh_builtin_if_stale_refreshes_ops_from_disk():
    from modules.agents.services.skill_loader import SkillLoader

    loader = SkillLoader(MagicMock())
    disk_body = _reader_body(_OPS_SEED)
    disk_hash = hashlib.sha256(disk_body.encode("utf-8")).hexdigest()

    stale = SimpleNamespace(
        name="platform-operations", content_hash="STALE_HASH", prompt_template="old body"
    )
    db = MagicMock()
    result = loader._refresh_builtin_if_stale(stale, db)

    assert stale.content_hash == disk_hash        # row refreshed to the on-disk hash
    assert stale.prompt_template == disk_body       # …and the on-disk body
    assert result == disk_body
    db.commit.assert_called_once()


def test_refresh_builtin_if_stale_noop_when_ops_current():
    """A row already matching disk is not rewritten and does not commit."""
    from modules.agents.services.skill_loader import SkillLoader

    loader = SkillLoader(MagicMock())
    disk_body = _reader_body(_OPS_SEED)
    disk_hash = hashlib.sha256(disk_body.encode("utf-8")).hexdigest()

    current = SimpleNamespace(
        name="platform-operations", content_hash=disk_hash, prompt_template=disk_body
    )
    db = MagicMock()
    result = loader._refresh_builtin_if_stale(current, db)

    assert result == disk_body
    db.commit.assert_not_called()
