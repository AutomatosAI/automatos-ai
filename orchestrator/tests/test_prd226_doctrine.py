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
# P226-RVW-3 — pin the SUBSTANTIVE doctrine (not only the nine titles) across
# the two Auto-facing homes: the always-on skill's §17 and the compact
# MANAGER_DOCTRINE_BLOCK. Point 7's ask-length silently drifted between them
# once ("short" vs "≤ ~700"); this fails CI if any substantive value diverges.
# (auto-cto-custom-soul.txt is out of scope here — it feeds the GLOBAL CTO
# agent, not Auto; see seed_auto_agent.py's _PLATFORM_SKILL_PATH comment.)
# ---------------------------------------------------------------------------

def _skill_doctrine_section() -> str:
    """'The Manager's Doctrine' section of the always-on platform-management
    skill, sliced from its header to the next section (or EOF). Located by NAME,
    not number: v2.3.0 (skills repo) moved it from ops §17 to charter §H, and a
    literal "## 17." pin would push the next re-organisation into editing this
    test instead of keeping the doctrine — same lesson as the migration-parent
    pin. The heading match is apostrophe-agnostic (' vs ’)."""
    import re
    skill = _SKILL.read_text(encoding="utf-8")
    m = re.search(r"(?m)^## .*Manager.s Doctrine.*$", skill)
    assert m, "platform-management skill lost 'The Manager's Doctrine' section"
    start = m.start()
    nxt = skill.find("\n## ", start + 1)     # -1 when §17 is the last section
    return skill[start:] if nxt == -1 else skill[start:nxt]


def test_substantive_doctrine_consistent_across_auto_facing_homes():
    """AC3: ask-length, lane names, and awareness tools must read consistently
    across the skill's §17 and MANAGER_DOCTRINE_BLOCK — not just the nine
    titles. A silent drift in either Auto-facing home fails CI."""
    section = _skill_doctrine_section()
    block = seed_mod.MANAGER_DOCTRINE_BLOCK
    for home_name, home in (("skill §17", section), ("doctrine block", block)):
        # point 7 ask-length ceiling — the value that drifted ("short" vs "≤ ~700")
        assert "700" in home, f"{home_name} lost the ~700-char ask ceiling (point 7)"
        # point 2 — the three lanes, named
        for lane in ("DELEGATE", "ASSIGN", "MISSION"):
            assert lane in home, f"{home_name} missing lane {lane!r} (point 2)"
        # point 1 — the awareness tool names
        for tool in (
            "platform_board_summary",
            "platform_list_missions",
            "platform_list_agents",
        ):
            assert tool in home, f"{home_name} missing awareness tool {tool!r} (point 1)"


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


# ---------------------------------------------------------------------------
# P226-RVW-1 — the backfill must actually REACH existing workspace Auto rows
# ---------------------------------------------------------------------------
# The review found the hash-guarded backfill was a production no-op: every
# lazy-seed caller reaches seed_auto_agent ONLY when the Auto row is absent, so
# its existing-row backfill branch never fired for pre-existing rows. The fix
# wires sync_auto_personas into the leader-gated per-deploy boot seed batch.

_MAIN = Path(__file__).resolve().parent.parent / "main.py"


def _boot_phase_1_body() -> str:
    """The source of main.py's _boot_phase_1_core (read as text, no import — the
    module pulls in the whole app and can't load in a bare unit-test env)."""
    src = _MAIN.read_text(encoding="utf-8")
    start = src.index("async def _boot_phase_1_core")
    end = src.index("async def ", start + 1)
    return src[start:end]


def test_boot_phase_invokes_sync_auto_personas():
    """AC1: a NON-test production path runs the backfill when Auto rows already
    exist. Fails if the boot wiring is ever removed (reachability regression)."""
    body = _boot_phase_1_body()
    assert "sync_auto_personas" in body, (
        "_boot_phase_1_core no longer invokes sync_auto_personas — the doctrine "
        "backfill is unreachable for existing workspaces again"
    )


def test_backfill_reaches_pre_existing_alembic_row_via_real_caller():
    """AC2: drive a REAL caller (sync_auto_personas, the fn the boot path calls) —
    NOT _backfill_auto_persona directly — over a pre-existing row carrying the
    _ALEMBIC_BACKFILL_PERSONA default plus a customized row. The default row is
    updated to _default_persona(); the customized row is untouched; a second pass
    through the same caller mutates nothing (idempotent)."""
    old = _row(seed_mod._ALEMBIC_BACKFILL_PERSONA)
    custom = _row("hand-written soul, do not touch")

    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = [old, custom]

    first = seed_mod.sync_auto_personas(db)
    assert first == {"updated": 1, "skipped": 1, "current": 0}
    assert old.custom_persona_prompt == seed_mod._default_persona()
    assert custom.custom_persona_prompt == "hand-written soul, do not touch"

    persona_after_first = old.custom_persona_prompt
    second = seed_mod.sync_auto_personas(db)
    assert second == {"updated": 0, "skipped": 1, "current": 1}
    assert old.custom_persona_prompt == persona_after_first  # no second mutation
    assert custom.custom_persona_prompt == "hand-written soul, do not touch"


def test_cto_soul_apr2026_snapshot_hash_is_a_known_seed():
    """AC3: the transient Irish-CTO soul force-written to every auto-% row by the
    2026-04-13→14 main.py startup migration is recognised as a shipped default —
    so a surviving row is reconciled, not misclassified 'customized' and skipped.
    End-to-end update of any hash-matching row is proven by
    test_backfill_updates_alembic_backfilled_default; this pins the CTO snapshot
    into that eligible set."""
    assert (
        seed_mod._CTO_SOUL_APR2026_SNAPSHOT_HASH
        in seed_mod._KNOWN_SEED_PERSONA_HASHES
    )
