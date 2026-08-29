"""PRD-231 US-004 — the CTO soul's context diet (identity / rulebook split).

Pure, LLM-free, Postgres-free. Guards that the surgical soul slim stays surgical.

ARCHITECTURE NOTE (verified by grep, 2026-08-29): auto-cto-custom-soul.txt feeds
the GLOBAL, admin-only CTO agent (core/seeds/seed_cto_agent.py, slug 'auto-cto',
workspace_id=NULL) — NOT the per-workspace Auto chat agent, whose default persona
is _default_persona() == _FRIENDLY_FALLBACK + doctrine and was ALREADY
identity-only. So this slim does not move the 28k→10.4k per-turn number (US-002/003
did that); it is a de-duplication of the FAT CTO soul: the five verbose rulebook
sections (Role / Authority / How-I-Think / Operating-Rhythm / Routing-Rules) are
superseded by the retained nine-point "How I Manage" doctrine and the
platform-management charter (the single source). The 226 soul tests
(test_prd226_doctrine.py) assert only content OUTSIDE the five removed sections, so
they stay green untouched — this file is the dedicated US-004 guard.
"""

from pathlib import Path

import core.seeds.seed_auto_agent as seed_mod

_SEEDS = Path(__file__).resolve().parent.parent / "core" / "seeds"
_SOUL = _SEEDS / "auto-cto-custom-soul.txt"

# The five rulebook sections PRD-231 removes from the soul DEFAULT — their
# authoritative home is now the platform-management charter (single source).
_REMOVED_HEADERS = (
    "**My Role:**",
    "**My Authority:**",
    "**How I Think:**",
    "**My Operating Rhythm:**",
    "**My Routing Rules:**",
)


def _soul() -> str:
    return _SOUL.read_text(encoding="utf-8")


def test_five_rulebook_sections_removed_from_soul():
    """AC4 anti-duplication guard: none of the five removed headers may reappear
    in the soul seed — the rulebook lives in the charter now, not here."""
    present = [h for h in _REMOVED_HEADERS if h in _soul()]
    assert not present, f"rulebook section(s) leaked back into the soul: {present}"


def test_soul_retains_identity_personality_and_doctrine():
    """The slim is surgical: identity, personality, the CTO markers, and the
    nine-point Manager's Doctrine all survive byte-verbatim. Mirrors the 226
    anchors so a careless re-slim fails HERE too, not only in the 226 suite."""
    soul = _soul()
    for marker in (
        "I *am* Automatos",          # Who I Am (kept)
        "Irish tech lead",           # My Personality (kept)
        "Workspace Operating System",  # Who I Am (kept)
        "I am every line",           # My Promise (kept)
        "one-armed plasterer",       # My Personality (kept)
        "Sacred Ground",             # kept section
        "My Promise",                # kept section
        "The Manager's Doctrine",    # the retained nine-point doctrine
    ):
        assert marker in soul, f"surgical slim removed a KEPT anchor: {marker!r}"


def test_soul_carries_single_source_cross_reference():
    """AC3: the soul points to the platform-management charter as the single
    source of the rulebook it no longer inlines."""
    soul = _soul()
    assert "**How I operate:**" in soul
    assert "single source" in soul
    assert "platform-management" in soul


def test_pre231_fat_soul_hash_frozen_and_lifts_to_slim_default():
    """AC1/AC2: the pre-231 fat CTO soul hash is frozen into the known-seed set
    and lifts to the slim per-workspace default (parallel to the April-2026
    residue guard), so a stray fat-soul row is reconciled, not skipped as
    'customized'. It is a GENUINELY distinct hash — not self-referential."""
    fat = seed_mod._CTO_SOUL_PRE231_FAT_HASH
    assert fat in seed_mod._KNOWN_SEED_PERSONA_HASHES
    target, mode = seed_mod._PERSONA_BACKFILL_LIFTS[fat]
    assert target == seed_mod._default_persona()
    assert mode == "friendly"
    # distinct from both the April snapshot and the (doctrine-carrying) default
    assert fat != seed_mod._CTO_SOUL_APR2026_SNAPSHOT_HASH
    assert fat != seed_mod._persona_hash(seed_mod._default_persona())


def test_pre231_fat_hash_is_historical_not_the_slimmed_file():
    """The frozen hash captures the soul BEFORE the slim — it must NOT equal the
    hash of the now-slim file on disk. Proves the freeze pinned history (so a
    future soul edit can never silently move it), not the current state."""
    current_slim = seed_mod._persona_hash(_soul())
    assert seed_mod._CTO_SOUL_PRE231_FAT_HASH != current_slim
