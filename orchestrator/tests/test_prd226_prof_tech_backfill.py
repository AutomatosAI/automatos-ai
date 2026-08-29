"""PRD-226 — The Manager's Doctrine (P226-RVW-6).

Pure, LLM-free regression tests for the *voice-preserving* persona backfill.

The finding: US-001's hash-guarded backfill kept its 'lift every shipped default'
promise only for the FRIENDLY family. ``_KNOWN_SEED_PERSONA_HASHES`` held the
friendly base voice and the CTO-soul snapshot, but NOT the 'professional' or
'technical' preset voices — both live, UI-selectable, doctrine-FREE shipped
presets. So a workspace that picked 'Professional' or 'Technical' was SKIPPED by
the backfill (never got the persona-level doctrine) and — worse — over-reported
in the once-per-deploy boot log as a hand-customized soul ('skipped … soul is
customized'), corrupting the operational-visibility signal.

The fix makes the backfill voice-aware: a professional/technical base-voice row
is LIFTED to the doctrine-carrying version of ITS OWN voice
(``compose_persona_with_doctrine(base_voice)`` == ``_PERSONALITY_PRESETS[mode]``),
NOT to ``_default_persona()`` (the friendly voice) — a blunt add-hash-then-
replace would silently switch those users to the friendly tone, a worse defect.
The friendly-family behaviour is untouched.

No DB, no async endpoint: the backfill runs through the real
``_backfill_auto_persona`` / ``sync_auto_personas``; the read-side
``_resolve_persona_view`` and write-side ``_resolve_persona_for_mode`` helpers are
the same pure functions the Settings endpoints call.
"""

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import api.workspaces as ws
from api.workspaces import (
    _PERSONALITY_BASE_VOICES as WS_BASE_VOICES,
    _PERSONALITY_PRESETS,
    _resolve_persona_for_mode,
    _resolve_persona_view,
)
import core.seeds.seed_auto_agent as seed_mod
from core.seeds.seed_auto_agent import (
    MANAGER_DOCTRINE_BLOCK,
    _ALEMBIC_BACKFILL_PERSONA,
    _FRIENDLY_FALLBACK,
    _PERSONALITY_BASE_VOICES,
    _backfill_auto_persona,
    _default_persona,
    compose_persona_with_doctrine,
    sync_auto_personas,
)

# Single source for the nine doctrine anchor phrases.
from tests.test_prd226_doctrine import DOCTRINE_ANCHORS

# Distinctive tone markers — proof the row kept its OWN voice, not the friendly one.
_VOICE_MARKERS = {
    "professional": "enterprise-appropriate",
    "technical": "developer-focused",
    "friendly": "knowledgeable friend",
}
_PRESET_MODES = ("professional", "technical")


def _row(persona: str, configuration=None):
    return SimpleNamespace(
        custom_persona_prompt=persona,
        use_custom_persona=True,
        configuration=configuration,
        workspace_id="ws-rvw6",
        slug="auto-x",
    )


# ---------------------------------------------------------------------------
# AC1 — a professional / technical base-voice row is LIFTED to the doctrine-
# carrying version of ITS OWN voice (== _PERSONALITY_PRESETS[mode]), NOT to
# _default_persona() (which would swap it to the friendly voice).
# ---------------------------------------------------------------------------

def test_professional_base_voice_lifts_to_its_own_doctrine_voice():
    row = _row(_PERSONALITY_BASE_VOICES["professional"])

    assert _backfill_auto_persona(row) == "updated"
    # lifted to the doctrine-carrying PROFESSIONAL preset, byte-for-byte …
    assert row.custom_persona_prompt == compose_persona_with_doctrine(
        _PERSONALITY_BASE_VOICES["professional"]
    )
    assert row.custom_persona_prompt == _PERSONALITY_PRESETS["professional"]
    # … carrying the doctrine …
    assert MANAGER_DOCTRINE_BLOCK in row.custom_persona_prompt
    # … while retaining the distinctive professional tone …
    assert _VOICE_MARKERS["professional"] in row.custom_persona_prompt
    # … and it is NOT the friendly voice (the defect a blunt lift would cause).
    assert row.custom_persona_prompt != _default_persona()
    assert _VOICE_MARKERS["friendly"] not in row.custom_persona_prompt
    # mode stamped to name the voice we lifted to (so a later same-mode save
    # converges instead of swapping the tone — P226-RVW-5/RVW-6 anti-flip-flop).
    assert row.configuration["personality_mode"] == "professional"


def test_technical_base_voice_lifts_to_its_own_doctrine_voice():
    row = _row(_PERSONALITY_BASE_VOICES["technical"])

    assert _backfill_auto_persona(row) == "updated"
    assert row.custom_persona_prompt == compose_persona_with_doctrine(
        _PERSONALITY_BASE_VOICES["technical"]
    )
    assert row.custom_persona_prompt == _PERSONALITY_PRESETS["technical"]
    assert MANAGER_DOCTRINE_BLOCK in row.custom_persona_prompt
    assert _VOICE_MARKERS["technical"] in row.custom_persona_prompt
    assert row.custom_persona_prompt != _default_persona()
    assert _VOICE_MARKERS["friendly"] not in row.custom_persona_prompt
    assert row.configuration["personality_mode"] == "technical"


def test_lifted_preset_carries_every_doctrine_anchor():
    for mode in _PRESET_MODES:
        row = _row(_PERSONALITY_BASE_VOICES[mode])
        assert _backfill_auto_persona(row) == "updated"
        missing = [a for a in DOCTRINE_ANCHORS if a not in row.custom_persona_prompt]
        assert not missing, f"lifted '{mode}' persona missing doctrine anchors: {missing}"


# ---------------------------------------------------------------------------
# AC1/anti-swap (constraint a) — end-to-end voice preservation through the real
# GET/save helpers: a lifted professional row reports 'professional' (not
# 'custom', not 'friendly'), and a same-mode save re-writes the SAME voice —
# it converges, it does not flip to the friendly tone.
# ---------------------------------------------------------------------------

def test_lifted_professional_row_round_trips_without_a_voice_swap():
    row = _row(_PERSONALITY_BASE_VOICES["professional"])
    _backfill_auto_persona(row)  # deploy backfill lifts + stamps mode

    # GET reads the stamped mode → 'professional', doctrine NOT leaked into the
    # editable custom-soul field (the stored-mode branch wins over legacy text-match).
    view = _resolve_persona_view(
        row.configuration.get("personality_mode"), row.custom_persona_prompt
    )
    assert view == {"personality_mode": "professional", "custom_soul": ""}

    # A routine settings save round-trips the mode GET returned → writes the SAME
    # doctrine-carrying professional preset. No swap to friendly, doctrine intact.
    written = _resolve_persona_for_mode(view["personality_mode"], view["custom_soul"])
    assert written == _PERSONALITY_PRESETS["professional"]
    assert _VOICE_MARKERS["professional"] in written
    assert MANAGER_DOCTRINE_BLOCK in written
    assert written == row.custom_persona_prompt  # converged — the save is a no-op on the text


# ---------------------------------------------------------------------------
# AC2 — the misleading 'soul is customized' skip is gone for preset rows; the
# skip branch is reached ONLY for genuinely non-preset (hand-written) text, and
# the boot-log skip count excludes every preset base voice.
# ---------------------------------------------------------------------------

def test_no_preset_base_voice_is_ever_skipped():
    for mode, voice in _PERSONALITY_BASE_VOICES.items():
        row = _row(voice)
        assert _backfill_auto_persona(row) != "skipped", (
            f"'{mode}' base voice was skipped — mislabelled a customized soul"
        )


def test_only_genuinely_custom_text_hits_the_skip_branch():
    handwritten = "You are Auto. Speak like a 1920s radio announcer. — a real customization"
    row = _row(handwritten)
    assert _backfill_auto_persona(row) == "skipped"
    assert row.custom_persona_prompt == handwritten  # left byte-identical


def test_boot_log_skip_count_excludes_all_three_preset_base_voices():
    """Drive the REAL boot-path caller (sync_auto_personas) over one row on each
    of the three shipped preset base voices plus a single hand-written soul: only
    the hand-written row is counted 'skipped', so the boot log no longer over-
    reports preset workspaces as 'customized'."""
    rows = [
        _row(_PERSONALITY_BASE_VOICES["friendly"]),
        _row(_PERSONALITY_BASE_VOICES["professional"]),
        _row(_PERSONALITY_BASE_VOICES["technical"]),
        _row("hand-written soul, genuinely customized"),
    ]
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = rows

    counts = sync_auto_personas(db)

    assert counts["skipped"] == 1, f"skip count should exclude preset rows: {counts}"
    assert counts["updated"] == 3, f"all three preset base voices should lift: {counts}"
    # the three presets each lifted to the doctrine-carrying version of their voice
    assert rows[0].custom_persona_prompt == _default_persona()             # friendly family
    assert rows[1].custom_persona_prompt == _PERSONALITY_PRESETS["professional"]
    assert rows[2].custom_persona_prompt == _PERSONALITY_PRESETS["technical"]
    assert rows[3].custom_persona_prompt == "hand-written soul, genuinely customized"


# ---------------------------------------------------------------------------
# AC3 — friendly-family behaviour unchanged; the customized-soul clobber guard
# is intact (this mirrors the anchors named in the finding's constraint b).
# ---------------------------------------------------------------------------

def test_friendly_family_still_lifts_to_default_persona():
    for persona in (_FRIENDLY_FALLBACK, _ALEMBIC_BACKFILL_PERSONA):
        row = _row(persona)
        assert _backfill_auto_persona(row) == "updated"
        # friendly family lifts to the rich onboarding voice + doctrine (unchanged)
        assert row.custom_persona_prompt == _default_persona()
        assert MANAGER_DOCTRINE_BLOCK in row.custom_persona_prompt
        # not swapped to a professional/technical tone
        assert _VOICE_MARKERS["professional"] not in row.custom_persona_prompt
        assert _VOICE_MARKERS["technical"] not in row.custom_persona_prompt


def test_hand_written_soul_still_skipped_and_untouched():
    custom = "I am a totally custom Auto persona that the workspace hand-wrote."
    row = _row(custom)
    assert _backfill_auto_persona(row) == "skipped"
    assert row.custom_persona_prompt == custom


# ---------------------------------------------------------------------------
# AC4 — idempotent: a second pass over a lifted professional / technical row
# returns 'current' with no re-mutation.
# ---------------------------------------------------------------------------

def test_professional_backfill_is_idempotent():
    row = _row(_PERSONALITY_BASE_VOICES["professional"])
    first = _backfill_auto_persona(row)
    after_first = row.custom_persona_prompt
    second = _backfill_auto_persona(row)
    assert (first, second) == ("updated", "current")
    assert row.custom_persona_prompt == after_first  # no second mutation


def test_technical_backfill_is_idempotent():
    row = _row(_PERSONALITY_BASE_VOICES["technical"])
    first = _backfill_auto_persona(row)
    after_first = row.custom_persona_prompt
    second = _backfill_auto_persona(row)
    assert (first, second) == ("updated", "current")
    assert row.custom_persona_prompt == after_first


# ---------------------------------------------------------------------------
# AC5 — no circular import; the preset base voices have exactly ONE definition
# site, reached by the seeder without importing api/workspaces.
# ---------------------------------------------------------------------------

def test_base_voices_are_a_single_shared_object():
    """The single-source fix: api/workspaces and the seeder bind the SAME object,
    so the professional/technical strings are never a second copy that can drift."""
    assert WS_BASE_VOICES is _PERSONALITY_BASE_VOICES


def test_base_voices_map_has_exactly_one_definition_site():
    root = Path(__file__).resolve().parent.parent
    hits = []
    for py in root.rglob("*.py"):
        if "/tests/" in py.as_posix() or py.name.startswith("test_"):
            continue
        if "_PERSONALITY_BASE_VOICES = {" in py.read_text(encoding="utf-8"):
            hits.append(py.relative_to(root).as_posix())
    assert hits == ["core/seeds/seed_auto_agent.py"], (
        f"the base-voices map must have ONE definition site, found: {hits}"
    )


def test_seeder_does_not_import_api_workspaces_at_module_load():
    """No circular import: the seeder must reach the base voices without importing
    api/workspaces (the import edge runs the other way — workspaces → seeder)."""
    src = inspect.getsource(seed_mod)
    tree = ast.parse(src)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    offenders = [m for m in imported if m == "api" or m.startswith("api.")]
    assert not offenders, f"seeder imports api at module load (circular risk): {offenders}"


# ---------------------------------------------------------------------------
# AC1/contract — the seeder's lift target for each preset IS the workspaces
# preset (they cannot drift): compose_persona_with_doctrine(base) on both sides.
# ---------------------------------------------------------------------------

def test_lift_target_equals_the_workspaces_preset_for_each_mode():
    for mode in _PRESET_MODES:
        row = _row(_PERSONALITY_BASE_VOICES[mode])
        _backfill_auto_persona(row)
        assert row.custom_persona_prompt == _PERSONALITY_PRESETS[mode], (
            f"'{mode}' backfill target drifted from the settings-save preset"
        )
