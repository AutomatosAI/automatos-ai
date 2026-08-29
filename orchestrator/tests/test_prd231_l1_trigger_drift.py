"""PRD-231 RVW-2 — the L1 trigger (Skill.description) must not freeze after
the first seed.

For a NON-core skill (platform-operations) the frontmatter ``description`` IS
the single L1 catalog line the model reads to decide whether to
``platform_load_skill`` it. A frontmatter-description-only edit changes neither
the body nor its content_hash, so the body-hash-only refresh silently no-oped
and the live trigger stayed frozen at its first-seeded value. Two loci close it:

  * skill_loader._refresh_builtin_if_stale — re-syncs description on a body-load,
    detecting a description-only change INDEPENDENTLY of the body hash (AC1/AC2);
  * seed_auto_agent._resync_builtin_description — re-syncs it on the lazy
    get-or-seed path that runs every chat, so existing workspaces self-heal on
    their next turn without any body-load (AC3 option 1: no path leaves it stale).

Pure, LLM-free, Postgres-free — MagicMock sessions + a tmp seed on disk.
"""

import hashlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import core.seeds.seed_auto_agent as seed_mod

_OLD_DESC = "The tool-by-tool operations cookbook — LOAD THIS before any op."
_NEW_DESC = "REVISED trigger — load platform-operations before EVERY platform op. Bite harder."
_BODY = "<!-- banner -->\n\n# Platform Operations Reference\n\n## 0. Setup\n\n## 19. Watches\n"


def _write_seed(tmp_path, *, description: str, body: str = _BODY):
    """A minimal but structurally real generated seed: YAML frontmatter carrying
    the trigger ``description``, then the markdown body the readers hash."""
    text = (
        "---\n"
        "name: platform-operations\n"
        f"description: {description}\n"
        'version: "1.0.0"\n'
        "tags: [platform, operations]\n"
        "category: agent-role\n"
        "---\n\n"
        f"{body}"
    )
    p = tmp_path / "platform-operations-skill.md"
    p.write_text(text, encoding="utf-8")
    return p


def _body_hash(body: str) -> str:
    return hashlib.sha256(body.strip().encode("utf-8")).hexdigest()


# ── AC1/AC2: _refresh_builtin_if_stale re-syncs description independently ─────


def test_refresh_resyncs_description_when_only_the_trigger_drifted(tmp_path, monkeypatch):
    """Body current, frontmatter description changed → the refresh fires on the
    description axis alone (the body-hash early-return must NOT short-circuit it),
    updates the row, and commits once. This is the exact RVW-2 failure."""
    from modules.agents.services.skill_loader import SkillLoader

    seed = _write_seed(tmp_path, description=_NEW_DESC)
    monkeypatch.setitem(SkillLoader._BUILTIN_PATHS, "platform-operations", seed)

    row = SimpleNamespace(
        name="platform-operations",
        content_hash=_body_hash(_BODY),   # body already current
        prompt_template=_BODY.strip(),
        description=_OLD_DESC,             # …but the trigger drifted
    )
    db = MagicMock()
    loader = SkillLoader(MagicMock())

    result = loader._refresh_builtin_if_stale(row, db)

    assert row.description == _NEW_DESC          # trigger re-synced from frontmatter
    assert row.prompt_template == _BODY.strip()  # body untouched (it was current)
    assert result == _BODY.strip()
    db.commit.assert_called_once()               # a description-only change still persists


def test_refresh_noop_when_body_and_description_both_current(tmp_path, monkeypatch):
    """Both axes match disk → no write, no commit (the freshness fast-path holds
    once the trigger has caught up)."""
    from modules.agents.services.skill_loader import SkillLoader

    seed = _write_seed(tmp_path, description=_NEW_DESC)
    monkeypatch.setitem(SkillLoader._BUILTIN_PATHS, "platform-operations", seed)

    row = SimpleNamespace(
        name="platform-operations",
        content_hash=_body_hash(_BODY),
        prompt_template=_BODY.strip(),
        description=_NEW_DESC,
    )
    db = MagicMock()
    result = SkillLoader(MagicMock())._refresh_builtin_if_stale(row, db)

    assert result == _BODY.strip()
    db.commit.assert_not_called()


def test_refresh_updates_both_axes_when_body_and_trigger_both_drift(tmp_path, monkeypatch):
    """A real re-sync of a stale seed (new body AND new trigger) refreshes both."""
    from modules.agents.services.skill_loader import SkillLoader

    new_body = _BODY + "\n## 20. New section\n"
    seed = _write_seed(tmp_path, description=_NEW_DESC, body=new_body)
    monkeypatch.setitem(SkillLoader._BUILTIN_PATHS, "platform-operations", seed)

    row = SimpleNamespace(
        name="platform-operations",
        content_hash="STALE",
        prompt_template="old body",
        description=_OLD_DESC,
    )
    db = MagicMock()
    SkillLoader(MagicMock())._refresh_builtin_if_stale(row, db)

    assert row.content_hash == _body_hash(new_body)
    assert row.prompt_template == new_body.strip()
    assert row.description == _NEW_DESC
    db.commit.assert_called_once()


def test_refresh_does_not_force_a_trigger_onto_a_row_that_has_none(tmp_path, monkeypatch):
    """A body-current row with no description is left alone here — filling the
    initial trigger is the seeder's job, and this keeps the freshness fast-path
    from committing on every load of a legacy row."""
    from modules.agents.services.skill_loader import SkillLoader

    seed = _write_seed(tmp_path, description=_NEW_DESC)
    monkeypatch.setitem(SkillLoader._BUILTIN_PATHS, "platform-operations", seed)

    row = SimpleNamespace(
        name="platform-operations",
        content_hash=_body_hash(_BODY),
        prompt_template=_BODY.strip(),
        description=None,
    )
    db = MagicMock()
    SkillLoader(MagicMock())._refresh_builtin_if_stale(row, db)

    db.commit.assert_not_called()


# ── AC3: lazy get-or-seed re-syncs the trigger on existing rows (self-heal) ───


def _existing_row_db(row):
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = row
    return db


def test_lazy_seed_resyncs_l1_trigger_for_existing_row(tmp_path):
    """The existing-row path of _upsert_builtin_core_skill re-syncs the trigger
    from the current seed — so an existing workspace's L1 line self-heals on its
    NEXT chat, with no body-load required."""
    seed = _write_seed(tmp_path, description=_NEW_DESC)
    row = SimpleNamespace(name="platform-operations", description=_OLD_DESC, id=5)
    db = _existing_row_db(row)

    returned = seed_mod._upsert_builtin_core_skill(
        db,
        name="platform-operations",
        path=seed,
        lock_key="seed:platform-operations",
        description="defensive fallback",
        tags=["platform", "operations"],
    )

    assert returned is row                 # existing row reused, never re-created
    assert not db.add.called
    assert row.description == _NEW_DESC     # trigger re-synced from the seed frontmatter
    db.flush.assert_called()


def test_lazy_seed_trigger_resync_is_idempotent(tmp_path):
    """Once healed, a matching trigger is not rewritten — no per-chat write churn."""
    seed = _write_seed(tmp_path, description=_NEW_DESC)
    row = SimpleNamespace(name="platform-operations", description=_NEW_DESC, id=5)
    db = _existing_row_db(row)

    seed_mod._upsert_builtin_core_skill(
        db,
        name="platform-operations",
        path=seed,
        lock_key="seed:platform-operations",
        description="defensive fallback",
        tags=[],
    )

    assert row.description == _NEW_DESC
    db.flush.assert_not_called()  # nothing changed → no write


# ── AC2 tie-in: the re-synced description IS the rendered L1 line ─────────────


def test_resynced_description_renders_as_the_l1_line():
    """The value re-synced onto the row is exactly what SkillsSection renders as
    the ops L1 catalog line — so healing the row heals the live trigger text."""
    from modules.context.sections.skills import SkillsSection

    row = SimpleNamespace(name="platform-operations", description=_NEW_DESC)
    line = SkillsSection._l1_metadata_line(row)

    assert line == f"- **platform-operations**: {_NEW_DESC}"
    assert _NEW_DESC in line
