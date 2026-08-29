"""PRD-226 — The Manager's Doctrine (P226-RVW-5).

Pure, LLM-free regression tests for the persona-backfill's configuration
invariant. The finding: ``_backfill_auto_persona`` lifted an alembic-seeded row
to the doctrine-carrying ``_default_persona()`` but never stamped
``configuration.personality_mode`` — so the Settings GET's legacy text-match saw
the now ~2670-char doctrine soul, failed to match the doctrine-FREE base voices,
and misreported the never-customized row as 'custom' with the doctrine leaked
into the editable custom-soul field. A routine settings save then stamped
``personality_mode='custom'`` permanently, opting the row out of every future
doctrine backfill (the durable lock-out P226-RVW-4 set out to prevent, via an
entry point RVW-4 never modelled).

No DB, no async endpoint: the backfill runs through the real
``_backfill_auto_persona``; the GET read-side detection was extracted into the
pure ``_resolve_persona_view`` the endpoint now calls, so it is exercised
directly (the write-side already had ``_resolve_persona_for_mode``).
"""

import inspect
from types import SimpleNamespace

import api.workspaces as ws
from api.workspaces import (
    _PERSONALITY_BASE_VOICES,
    _resolve_persona_for_mode,
    _resolve_persona_view,
)
from core.seeds.seed_auto_agent import (
    MANAGER_DOCTRINE_BLOCK,
    _ALEMBIC_BACKFILL_PERSONA,
    _backfill_auto_persona,
    _default_persona,
)

# An alembic-seeded Auto row: friendly base-voice persona, and a configuration
# JSONB that carries the other orchestrator dials but NO personality_mode key —
# exactly what orchestrator/alembic/versions/seed_auto_agents_existing_workspaces
# inserted for pre-226 workspaces.
_ALEMBIC_CONFIG_SHAPE = {
    "thinking_level": "medium",
    "proactive_level": "notify",
    "communication_style": "balanced",
}


def _alembic_row(**overrides):
    row = SimpleNamespace(
        custom_persona_prompt=_ALEMBIC_BACKFILL_PERSONA,
        use_custom_persona=True,
        configuration=dict(_ALEMBIC_CONFIG_SHAPE),
        workspace_id="ws-rvw5",
        slug="auto-x",
    )
    for k, v in overrides.items():
        setattr(row, k, v)
    return row


# ---------------------------------------------------------------------------
# AC1 — the backfill stamps personality_mode='friendly' when it is absent,
# rebuilding the dict (new object, original untouched), and never overrides an
# explicit stored choice.
# ---------------------------------------------------------------------------

def test_backfill_stamps_friendly_mode_when_configuration_lacks_it():
    row = _alembic_row()
    original_cfg = row.configuration  # hold the identity to prove no mutation

    assert _backfill_auto_persona(row) == "updated"
    # persona lifted to the doctrine-carrying default …
    assert row.custom_persona_prompt == _default_persona()
    # … and the CREATE-path invariant restored: the mode is now declared.
    assert row.configuration["personality_mode"] == "friendly"
    # a NEW dict was assigned — the original config object was not mutated.
    assert row.configuration is not original_cfg
    assert "personality_mode" not in original_cfg
    # the other dials survived the rebuild.
    for k, v in _ALEMBIC_CONFIG_SHAPE.items():
        assert row.configuration[k] == v


def test_backfill_leaves_an_explicit_stored_mode_untouched():
    """A row still on a known-seed persona but already carrying an explicit
    personality_mode (a real workspace choice) must keep that choice — the
    backfill lifts the persona but never overrides a stored mode."""
    row = _alembic_row(configuration={**_ALEMBIC_CONFIG_SHAPE, "personality_mode": "professional"})

    assert _backfill_auto_persona(row) == "updated"
    assert row.custom_persona_prompt == _default_persona()   # persona still lifted
    assert row.configuration["personality_mode"] == "professional"  # choice preserved


def test_backfill_stamps_mode_even_when_configuration_is_missing():
    """Defensive: a row with no configuration attribute at all still gets the
    mode stamped (getattr guard) — proves the fix does not depend on the alembic
    shape being present."""
    row = SimpleNamespace(
        custom_persona_prompt=_ALEMBIC_BACKFILL_PERSONA,
        use_custom_persona=True,
        workspace_id="ws-nocfg",
        slug="auto-x",
    )
    assert _backfill_auto_persona(row) == "updated"
    assert row.configuration["personality_mode"] == "friendly"


# ---------------------------------------------------------------------------
# AC2 — the REAL GET read-side detection reports 'friendly' + '' after the
# backfill; the doctrine is NOT leaked into the editable custom-soul field.
# ---------------------------------------------------------------------------

def test_get_view_reports_friendly_and_hides_doctrine_after_backfill():
    row = _alembic_row()
    _backfill_auto_persona(row)  # deploy backfill: persona lifted + mode stamped

    view = _resolve_persona_view(row.configuration.get("personality_mode"), row.custom_persona_prompt)
    assert view == {"personality_mode": "friendly", "custom_soul": ""}


def test_get_view_before_fix_would_leak_doctrine_as_custom():
    """Documents the exact defect the stamp closes: with NO stored mode (the
    pre-fix state), the read-side detection text-matches the ~2670-char doctrine
    soul against the doctrine-FREE base voices, fails, and returns 'custom' with
    the full doctrine text sitting in the editable custom-soul field."""
    leaked = _resolve_persona_view(None, _default_persona())
    assert leaked["personality_mode"] == "custom"
    assert leaked["custom_soul"] == _default_persona()
    assert MANAGER_DOCTRINE_BLOCK in leaked["custom_soul"]  # the leak


def test_resolve_persona_view_is_byte_equivalent_to_the_replaced_branch():
    """Pin the refactor: the extracted helper reproduces every branch of the
    inline GET detection it replaced."""
    # stored non-custom mode → that mode, empty soul (no leak)
    assert _resolve_persona_view("friendly", _default_persona()) == {
        "personality_mode": "friendly", "custom_soul": ""}
    # stored 'custom' → the stored soul is surfaced
    assert _resolve_persona_view("custom", "hand-written soul") == {
        "personality_mode": "custom", "custom_soul": "hand-written soul"}
    # no stored mode + a base-voice persona → detected preset, empty soul
    assert _resolve_persona_view(None, _PERSONALITY_BASE_VOICES["professional"]) == {
        "personality_mode": "professional", "custom_soul": ""}
    # no stored mode + unrecognised text → 'custom' + that text
    assert _resolve_persona_view(None, "totally bespoke") == {
        "personality_mode": "custom", "custom_soul": "totally bespoke"}
    # neither stored mode nor persona → empty dict = keep the orchestrator defaults
    assert _resolve_persona_view(None, None) == {}
    assert _resolve_persona_view(None, "") == {}


def test_get_endpoint_routes_through_resolve_persona_view():
    """Wiring guard: the endpoint must use the extracted helper, so this test is
    not an orphan (mirrors RVW-4's save-endpoint pin)."""
    src = inspect.getsource(ws.get_orchestrator_settings)
    assert "_resolve_persona_view(" in src, (
        "get_orchestrator_settings no longer routes persona detection through "
        "_resolve_persona_view — the read-side regression guard is bypassed"
    )


# ---------------------------------------------------------------------------
# AC3 — the permanent 'custom' lock-out is closed end-to-end: seed → backfill →
# GET → friendly save never flips the mode to 'custom', so the row stays a
# shipped-default and is never mislabelled 'customized'.
# ---------------------------------------------------------------------------

def test_backfill_then_get_then_save_never_locks_row_out_of_doctrine():
    # 1. Pre-existing alembic-seeded row: friendly base voice, NO stored mode.
    row = _alembic_row()

    # 2. The deploy backfill lifts the persona AND stamps the mode (the fix).
    assert _backfill_auto_persona(row) == "updated"
    assert row.custom_persona_prompt == _default_persona()
    assert row.configuration["personality_mode"] == "friendly"
    # The lifted row IS the current shipped default — a re-run reports 'current',
    # never 'skipped' (which would mislabel it 'customized' and lock it out).
    assert _backfill_auto_persona(row) == "current"

    # 3. Settings GET reads the stamped mode → 'friendly', with the doctrine NOT
    #    exposed in the editable custom-soul field. (Pre-fix: 'custom' + full text.)
    view = _resolve_persona_view(row.configuration.get("personality_mode"), row.custom_persona_prompt)
    assert view == {"personality_mode": "friendly", "custom_soul": ""}

    # 4. A routine settings save round-trips the mode GET returned ('friendly'),
    #    resolves a doctrine-carrying persona, and stamps the mode — which stays
    #    'friendly', NEVER silently flipped to 'custom' (the durable lock-out).
    saved_mode, saved_soul = view["personality_mode"], view["custom_soul"]
    resolved = _resolve_persona_for_mode(saved_mode, saved_soul)
    assert resolved is not None and MANAGER_DOCTRINE_BLOCK in resolved
    new_cfg = dict(row.configuration)          # endpoint rebuilds, never mutates
    new_cfg["personality_mode"] = saved_mode   # the real endpoint's stamp
    row.configuration = new_cfg
    row.custom_persona_prompt = resolved
    assert row.configuration["personality_mode"] == "friendly"  # not 'custom'

    # 5. GET still reports 'friendly' — the row is never mislabelled 'customized',
    #    so it stays eligible for future doctrine updates instead of being skipped.
    view2 = _resolve_persona_view(row.configuration["personality_mode"], row.custom_persona_prompt)
    assert view2["personality_mode"] == "friendly"
    assert view2["custom_soul"] == ""
