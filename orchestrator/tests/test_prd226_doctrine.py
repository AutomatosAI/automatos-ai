"""PRD-226 — The Manager's Doctrine (US-001).

Pure, LLM-free tests for the doctrine seeds and the hash-guarded persona
backfill. No live model calls, no real Postgres (the db session is a MagicMock,
matching the existing seed-test pattern in test_p2w1_agent_skills_repair.py).
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import core.seeds.seed_auto_agent as seed_mod

_SEEDS = Path(__file__).resolve().parent.parent / "core" / "seeds"
_SOUL = _SEEDS / "auto-cto-custom-soul.txt"
_SKILL = _SEEDS / "platform-management-skill.md"

# The nine doctrine points, as anchor phrases present verbatim in every home
# (both seed files and the compact CHATBOT-context block).
DOCTRINE_ANCHORS = [
    "Awareness",
    "Three lanes, chosen deliberately",
    "Delegate, don't implement",
    "Reuse before creating",
    "Dispatch as a contract",
    "Board as ledger",
    "Asks are decisions, not reports",
    "Recurring work becomes a Playbook",
    "Narrate",
]


# ---------------------------------------------------------------------------
# AC1 — both seed files carry the 9 doctrine points; CTO identity retained
# ---------------------------------------------------------------------------

def test_soul_seed_carries_all_nine_doctrine_points():
    soul = _SOUL.read_text(encoding="utf-8")
    missing = [a for a in DOCTRINE_ANCHORS if a not in soul]
    assert not missing, f"soul seed missing doctrine points: {missing}"


def test_skill_seed_carries_all_nine_doctrine_points():
    skill = _SKILL.read_text(encoding="utf-8")
    missing = [a for a in DOCTRINE_ANCHORS if a not in skill]
    assert not missing, f"platform-management skill missing doctrine points: {missing}"


def test_soul_retains_cto_identity_additions_not_a_rewrite():
    """The doctrine is added, not swapped in: the existing CTO soul must survive."""
    soul = _SOUL.read_text(encoding="utf-8")
    for marker in (
        "I *am* Automatos",
        "Irish tech lead",
        "Workspace Operating System",
        "I am every line",
    ):
        assert marker in soul, f"CTO identity marker lost (rewrite, not addition): {marker!r}"


# ---------------------------------------------------------------------------
# AC4 — the CHATBOT-context doctrine block stays under the token budget
# ---------------------------------------------------------------------------

def test_doctrine_block_carries_all_nine_points():
    block = seed_mod.MANAGER_DOCTRINE_BLOCK
    missing = [a for a in DOCTRINE_ANCHORS if a not in block]
    assert not missing, f"compact doctrine block missing points: {missing}"


def test_doctrine_block_character_ceiling():
    """≤ ~350 tokens in CHATBOT context (~4 chars/token → 1400 char ceiling)."""
    block = seed_mod.MANAGER_DOCTRINE_BLOCK
    assert len(block) <= 1400, (
        f"doctrine block is {len(block)} chars (~{len(block) // 4} tokens) — "
        "over the ~350-token CHATBOT-context budget"
    )


# ---------------------------------------------------------------------------
# AC2 — fresh-workspace seed applies the new soul (via the seed path)
# ---------------------------------------------------------------------------

def test_fresh_workspace_seed_applies_new_soul(monkeypatch):
    monkeypatch.setattr(seed_mod, "_upsert_platform_management_skill", lambda db: None)
    monkeypatch.setattr(
        seed_mod, "_get_default_model_config",
        lambda: {"provider": "test", "model_id": "test", "max_tokens": 4000},
    )

    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None  # no existing row

    agent = seed_mod.seed_auto_agent(db, uuid4())

    assert agent.custom_persona_prompt == seed_mod._default_persona()
    # the doctrine actually rode along with the friendly base
    assert "The Manager's Doctrine" in agent.custom_persona_prompt
    for anchor in DOCTRINE_ANCHORS:
        assert anchor in agent.custom_persona_prompt


# ---------------------------------------------------------------------------
# AC2/AC3 — hash-guarded backfill: update known defaults, skip customized
# ---------------------------------------------------------------------------

def _row(persona: str):
    return SimpleNamespace(
        custom_persona_prompt=persona,
        use_custom_persona=True,
        workspace_id=uuid4(),
        slug="auto-x",
    )


def test_backfill_updates_old_friendly_default():
    row = _row(seed_mod._FRIENDLY_FALLBACK)
    assert seed_mod._backfill_auto_persona(row) == "updated"
    assert row.custom_persona_prompt == seed_mod._default_persona()


def test_backfill_updates_alembic_backfilled_default():
    row = _row(seed_mod._ALEMBIC_BACKFILL_PERSONA)
    assert seed_mod._backfill_auto_persona(row) == "updated"
    assert row.custom_persona_prompt == seed_mod._default_persona()


def test_backfill_skips_customized_soul_and_leaves_it_untouched():
    custom = "I am a totally custom Auto persona that the workspace hand-wrote."
    row = _row(custom)
    assert seed_mod._backfill_auto_persona(row) == "skipped"
    assert row.custom_persona_prompt == custom  # untouched


def test_backfill_is_idempotent():
    row = _row(seed_mod._FRIENDLY_FALLBACK)
    first = seed_mod._backfill_auto_persona(row)
    persona_after_first = row.custom_persona_prompt
    second = seed_mod._backfill_auto_persona(row)
    assert (first, second) == ("updated", "current")
    assert row.custom_persona_prompt == persona_after_first  # no second change


def test_sync_auto_personas_reports_each_bucket():
    old = _row(seed_mod._FRIENDLY_FALLBACK)
    custom = _row("hand-written persona, do not touch")
    current = _row(seed_mod._default_persona())

    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = [old, custom, current]

    counts = seed_mod.sync_auto_personas(db)

    assert counts == {"updated": 1, "skipped": 1, "current": 1}
    assert old.custom_persona_prompt == seed_mod._default_persona()
    assert custom.custom_persona_prompt == "hand-written persona, do not touch"
