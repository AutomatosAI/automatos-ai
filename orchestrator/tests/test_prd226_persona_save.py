"""PRD-226 — The Manager's Doctrine (P226-RVW-4).

Pure, LLM-free regression tests proving the doctrine survives a Settings >
Orchestrator persona save on the platform-default 'friendly' mode — the
flip-flop where ``save_orchestrator_settings`` wrote the doctrine-FREE
``_PERSONALITY_PRESETS['friendly']`` (byte-identical to the backfill's
``_ALEMBIC_BACKFILL_PERSONA`` known-hash), stripping the doctrine, then the next
deploy's ``sync_auto_personas`` re-applied it — a permanent oscillation.

No DB, no async endpoint: the persona-write branch was extracted into the pure
``_resolve_persona_for_mode`` helper the endpoint calls, and the deploy backfill
is exercised through the real ``_backfill_auto_persona``.
"""

import inspect
from types import SimpleNamespace

import api.workspaces as ws
from api.workspaces import (
    _PERSONALITY_BASE_VOICES,
    _PERSONALITY_PRESETS,
    _resolve_persona_for_mode,
)
from core.seeds.seed_auto_agent import (
    MANAGER_DOCTRINE_BLOCK,
    _ALEMBIC_BACKFILL_PERSONA,
    _KNOWN_SEED_PERSONA_HASHES,
    _backfill_auto_persona,
    _default_persona,
    _persona_hash,
)

# Single source for the nine doctrine anchor phrases.
from tests.test_prd226_doctrine import DOCTRINE_ANCHORS

_MODES = ("friendly", "professional", "technical")


# ---------------------------------------------------------------------------
# AC-1 — the doctrine survives a friendly settings save
# ---------------------------------------------------------------------------

def test_friendly_settings_save_keeps_the_doctrine():
    """A seeded row holds _default_persona() (doctrine present). Saving the
    default 'friendly' mode must NOT strip it — before the fix, workspaces.py
    wrote the doctrine-free preset here."""
    seeded = _default_persona()
    assert MANAGER_DOCTRINE_BLOCK in seeded  # precondition

    written = _resolve_persona_for_mode("friendly", None)
    assert written is not None
    assert MANAGER_DOCTRINE_BLOCK in written, "friendly save stripped the doctrine block"
    missing = [a for a in DOCTRINE_ANCHORS if a not in written]
    assert not missing, f"friendly save dropped doctrine anchors: {missing}"


def test_save_endpoint_uses_the_resolver_helper():
    """Pin the wiring: the tested helper is the REAL write path, not an orphan —
    save_orchestrator_settings must call _resolve_persona_for_mode."""
    src = inspect.getsource(ws.save_orchestrator_settings)
    assert "_resolve_persona_for_mode(" in src, (
        "save_orchestrator_settings no longer routes the persona write through "
        "_resolve_persona_for_mode — the regression guard is bypassed"
    )


# ---------------------------------------------------------------------------
# AC-2 — the collision is resolved: no live preset is in the auto-replace set
# ---------------------------------------------------------------------------

def test_no_live_preset_collides_with_the_backfill_replace_set():
    """The same text must not be simultaneously (a) a settings-writable preset
    and (b) in _KNOWN_SEED_PERSONA_HASHES (the set the backfill auto-replaces)."""
    for mode, text in _PERSONALITY_PRESETS.items():
        assert MANAGER_DOCTRINE_BLOCK in text, f"preset '{mode}' lost the doctrine"
        assert _persona_hash(text) not in _KNOWN_SEED_PERSONA_HASHES, (
            f"preset '{mode}' hash-matches the backfill auto-replace set → flip-flop"
        )


def test_alembic_backfill_persona_is_no_longer_a_live_preset():
    """The specific collision the finding named: the doctrine-free
    _ALEMBIC_BACKFILL_PERSONA was byte-identical to the friendly preset."""
    for mode, text in _PERSONALITY_PRESETS.items():
        assert text != _ALEMBIC_BACKFILL_PERSONA, (
            f"preset '{mode}' is byte-identical to the auto-replace persona"
        )
    # It remains a legitimate known-hash for the doctrine-FREE base voice, so the
    # backfill still lifts genuinely pre-doctrine rows.
    assert _persona_hash(_ALEMBIC_BACKFILL_PERSONA) in _KNOWN_SEED_PERSONA_HASHES


# ---------------------------------------------------------------------------
# AC-4 — every preset carries the doctrine (switching mode never drops it)
# ---------------------------------------------------------------------------

def test_all_three_presets_carry_every_doctrine_anchor():
    for mode in _MODES:
        text = _PERSONALITY_PRESETS[mode]
        missing = [a for a in DOCTRINE_ANCHORS if a not in text]
        assert not missing, f"preset '{mode}' missing doctrine anchors: {missing}"


def test_switching_to_any_mode_writes_doctrine_carrying_text():
    for mode in _MODES:
        written = _resolve_persona_for_mode(mode, None)
        assert MANAGER_DOCTRINE_BLOCK in written, f"'{mode}' save dropped the doctrine"


def test_base_voices_stay_doctrine_free_single_builder():
    """The base voices are the doctrine-FREE tone strings; the doctrine is added
    exactly once, by the shared builder — the preset must equal base + block."""
    from core.seeds.seed_auto_agent import compose_persona_with_doctrine
    for mode, voice in _PERSONALITY_BASE_VOICES.items():
        assert MANAGER_DOCTRINE_BLOCK not in voice, f"base voice '{mode}' pre-carries doctrine"
        assert _PERSONALITY_PRESETS[mode] == compose_persona_with_doctrine(voice)


# ---------------------------------------------------------------------------
# AC-3 — round-trip stability: seed → save → deploy-backfill → save converges
# ---------------------------------------------------------------------------

def test_friendly_round_trip_converges_no_flip_flop():
    agent = SimpleNamespace(
        custom_persona_prompt=_default_persona(),  # 1. seed
        use_custom_persona=True,
        workspace_id="ws-roundtrip",
    )
    assert MANAGER_DOCTRINE_BLOCK in agent.custom_persona_prompt

    # 2. a 'friendly' settings save (any orchestrator change PUTs mode='friendly')
    agent.custom_persona_prompt = _resolve_persona_for_mode("friendly", None)
    assert MANAGER_DOCTRINE_BLOCK in agent.custom_persona_prompt
    after_first_save = agent.custom_persona_prompt

    # 3. the next deploy's backfill must NOT fight the save: a doctrine-carrying
    #    persona is not in the auto-replace set, so it is left untouched.
    result = _backfill_auto_persona(agent)
    assert result != "updated", "deploy backfill re-wrote a doctrine-carrying persona (flip-flop)"
    assert MANAGER_DOCTRINE_BLOCK in agent.custom_persona_prompt

    # 4. a second save is a no-op on the text — the sequence has converged.
    agent.custom_persona_prompt = _resolve_persona_for_mode("friendly", None)
    assert agent.custom_persona_prompt == after_first_save
    assert MANAGER_DOCTRINE_BLOCK in agent.custom_persona_prompt


# ---------------------------------------------------------------------------
# Regression — the refactor preserved the exact custom-mode / partial-payload
# semantics of the original inline branch.
# ---------------------------------------------------------------------------

def test_custom_mode_semantics_unchanged_by_refactor():
    # custom with a real soul → write it
    assert _resolve_persona_for_mode("custom", "You are a stoic advisor.") == "You are a stoic advisor."
    # custom with empty / absent soul → None = leave the existing persona alone
    assert _resolve_persona_for_mode("custom", "") is None
    assert _resolve_persona_for_mode("custom", None) is None


def test_unknown_mode_falls_back_to_friendly_preset():
    assert _resolve_persona_for_mode("nonsense", None) == _PERSONALITY_PRESETS["friendly"]
