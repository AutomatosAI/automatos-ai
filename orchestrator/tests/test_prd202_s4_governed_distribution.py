"""PRD-202 S4 (P2-21) — tenanted, governed, scanned distribution.

Pins:
1. A spec-conformant folder import runs the 2-stage scanner (static always).
2. Third-party/external provenance (git/plugin) forces the LLM stage ON; a
   trusted source (workspace) stays static-only.
3. L3 execution enablement is a workspace-admin action written to SkillAuditLog
   (import != executable — running is opt-in, gated on a scanner-pass).

Pure/mocked — fixture folder, fake session, the scanner's static + LLM stages
stubbed; no network.
"""
import asyncio
import os
import pathlib
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

for _k, _v in {
    "POSTGRES_USER": "test", "POSTGRES_PASSWORD": "test",
    "POSTGRES_HOST": "localhost", "POSTGRES_PORT": "5432", "POSTGRES_DB": "test",
}.items():
    os.environ.setdefault(_k, _v)

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeQuery:
    def __init__(self, first=None):
        self._first = first

    def filter(self, *a, **k):
        return self

    def order_by(self, *a, **k):
        return self

    def first(self):
        return self._first

    def delete(self):
        return 0


class _FakeSession:
    def __init__(self, first_by_model=None):
        self.added = []
        self._next_id = 0
        self._first_by_model = first_by_model or {}

    def query(self, model, *a, **k):
        name = getattr(model, "__name__", "")
        return _FakeQuery(self._first_by_model.get(name))

    def add(self, obj):
        self.added.append(obj)
        if getattr(obj, "id", None) is None:
            self._next_id += 1
            try:
                obj.id = self._next_id
            except Exception:
                pass

    def flush(self):
        for o in self.added:
            if getattr(o, "id", None) is None:
                self._next_id += 1
                o.id = self._next_id

    def commit(self):
        pass

    def rollback(self):
        pass

    def added_of(self, type_name):
        return [o for o in self.added if type(o).__name__ == type_name]


def _write_folder(root, name="scan-skill"):
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: A clean importable skill\nversion: 1.0.0\n---\n\n# Body\n\nDo the thing.\n",
        encoding="utf-8",
    )
    return folder


def _patch_scanner(monkeypatch, static_findings=None):
    """Spy the static stage; stub the LLM stage. Returns the two mocks."""
    import core.services.plugin_security_scanner as scanner

    quick = MagicMock(return_value=static_findings or [])
    llm = AsyncMock(return_value=SimpleNamespace(status="passed", risk_score=0, findings=[]))
    monkeypatch.setattr(scanner, "quick_scan", quick)
    monkeypatch.setattr(scanner, "llm_security_scan", llm)
    return quick, llm


# ---------------------------------------------------------------------------
# 1 + 2. two-stage scan on import; LLM stage forced for external sources
# ---------------------------------------------------------------------------

def test_standard_import_runs_two_stage_scan(monkeypatch, tmp_path):
    from modules.agents.services.skill_portability import import_standard_skill_folder

    quick, llm = _patch_scanner(monkeypatch)
    folder = _write_folder(tmp_path)
    db = _FakeSession()

    result = asyncio.run(import_standard_skill_folder(
        db, str(folder), source_scheme="workspace", source_ref="user",
    ))
    assert result["success"] is True
    quick.assert_called()  # the static stage always runs on import


def test_external_source_forces_llm_stage(monkeypatch, tmp_path):
    from modules.agents.services.skill_portability import import_standard_skill_folder

    quick, llm = _patch_scanner(monkeypatch)
    folder = _write_folder(tmp_path, name="ext-skill")
    db = _FakeSession()

    # git provenance is third-party/external → LLM stage ON
    asyncio.run(import_standard_skill_folder(
        db, str(folder), source_scheme="git", source_ref="anthropic-official",
    ))
    llm.assert_awaited()  # the LLM stage ran for the external source


def test_trusted_source_skips_llm_stage(monkeypatch, tmp_path):
    from modules.agents.services.skill_portability import import_standard_skill_folder

    quick, llm = _patch_scanner(monkeypatch)
    folder = _write_folder(tmp_path, name="trusted-skill")
    db = _FakeSession()

    # workspace provenance is trusted → static only, LLM stage skipped
    asyncio.run(import_standard_skill_folder(
        db, str(folder), source_scheme="workspace", source_ref="user",
    ))
    llm.assert_not_awaited()


def test_import_blocks_on_critical_static_finding(monkeypatch, tmp_path):
    from modules.agents.services.skill_portability import import_standard_skill_folder

    # NB: the "os.system" strings below are inert scanner-finding FIXTURES (test
    # data describing what the scanner would flag) — no shell is invoked here.
    critical = SimpleNamespace(type="dangerous_call", severity="critical", line=1, description="os.system")
    _patch_scanner(monkeypatch, static_findings=[critical])
    folder = _write_folder(tmp_path, name="bad-skill")
    db = _FakeSession()

    result = asyncio.run(import_standard_skill_folder(
        db, str(folder), source_scheme="git", source_ref="x",
    ))
    assert result["success"] is False
    assert result["verdict"] == "blocked"
    assert db.added_of("Skill") == []  # nothing persisted on a block


# ---------------------------------------------------------------------------
# 3. L3 enablement is audited (SkillAuditLog)
# ---------------------------------------------------------------------------

def test_l3_enablement_is_audited():
    from core.services.skill_l3_execution import set_l3_execution_enabled

    ws = SimpleNamespace(settings={})
    db = _FakeSession(first_by_model={"Workspace": ws})

    set_l3_execution_enabled(db, "ws-1", 42, True, actor="owner-1")

    audits = db.added_of("SkillAuditLog")
    assert len(audits) == 1
    assert audits[0].action == "l3_enable"
    assert audits[0].skill_id == 42
    # the enablement is persisted on the workspace settings
    assert 42 in ws.settings["skills_l3_enabled"]

    # disabling audits too, and removes the id
    set_l3_execution_enabled(db, "ws-1", 42, False, actor="owner-1")
    audits = db.added_of("SkillAuditLog")
    assert audits[-1].action == "l3_disable"
    assert 42 not in ws.settings["skills_l3_enabled"]


def test_enable_refuses_when_scanner_flags_critical(monkeypatch):
    from modules.tools.discovery.handlers_skill_runtime import set_skill_script_execution

    critical = SimpleNamespace(severity="critical", description="os.system", line=3)
    import core.services.plugin_security_scanner as scanner
    monkeypatch.setattr(scanner, "quick_scan", MagicMock(return_value=[critical]))

    skill = MagicMock()
    skill.id = 7
    skill.name = "risky"
    skill.prompt_template = "# body with os.system(...)"
    db = _FakeSession(first_by_model={"Skill": skill})

    result = asyncio.run(set_skill_script_execution(db, "ws-1", {"skill_id": 7, "enabled": True}))
    assert result["success"] is False
    # a critical finding blocks enablement — import != executable, and not even
    # enablable until the skill is clean
    assert db.added_of("SkillAuditLog") == []


def test_enable_succeeds_and_audits_when_clean(monkeypatch):
    from modules.tools.discovery.handlers_skill_runtime import set_skill_script_execution

    import core.services.plugin_security_scanner as scanner
    monkeypatch.setattr(scanner, "quick_scan", MagicMock(return_value=[]))

    skill = MagicMock()
    skill.id = 9
    skill.name = "clean"
    skill.prompt_template = "# clean body"
    ws = SimpleNamespace(settings={})
    db = _FakeSession(first_by_model={"Skill": skill, "Workspace": ws})

    result = asyncio.run(set_skill_script_execution(
        db, "ws-1", {"skill_id": 9, "enabled": True, "_agent_name": "Auto"}
    ))
    assert result["success"] is True
    assert result["l3_execution_enabled"] is True
    assert db.added_of("SkillAuditLog")[-1].action == "l3_enable"
