"""PRD-128 US-002: Default notification preference seeding on workspace provisioning.

Verifies that ``_seed_default_notification_preferences`` issues exactly nine
``INSERT ... WHERE NOT EXISTS`` statements — one per event type in the
``DEFAULT_NOTIFICATION_PREFERENCES`` table — and that every statement carries
the expected ``(event_type, destination)`` pair along with the target
workspace id.

The seeding helper is a plain function that only touches ``db.execute`` so a
``MagicMock`` session is sufficient — no real database is required.
"""

from __future__ import annotations

import os

# hybrid.py imports SessionLocal at module load, which in turn requires
# Postgres credentials. This test never touches a real DB — seed the
# minimum env vars so the import side-effects succeed.
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_DB", "test")

from unittest.mock import MagicMock  # noqa: E402
from uuid import uuid4  # noqa: E402

from core.auth.hybrid import (  # noqa: E402
    DEFAULT_NOTIFICATION_PREFERENCES,
    _seed_default_notification_preferences,
)


EXPECTED_DEFAULTS = {
    ("heartbeat_complete", "in_app"),
    ("task_complete", "in_app"),
    ("mission_step_complete", "silent"),
    ("mission_complete", "in_app"),
    ("playbook_step_complete", "silent"),
    ("playbook_complete", "in_app"),
    ("trigger_fired", "in_app"),
    ("report_submitted", "in_app"),
    ("agent_error", "in_app"),
}


def _make_db(rowcount: int = 1) -> MagicMock:
    """Return a mock session whose execute() returns a result with rowcount=N."""
    db = MagicMock()
    result = MagicMock()
    result.rowcount = rowcount
    db.execute.return_value = result
    return db


def test_default_preferences_table_matches_spec():
    """The module-level constant is the single source of truth for defaults."""
    assert set(DEFAULT_NOTIFICATION_PREFERENCES) == EXPECTED_DEFAULTS
    assert len(DEFAULT_NOTIFICATION_PREFERENCES) == 9


def test_seed_inserts_nine_rows_on_fresh_workspace():
    db = _make_db(rowcount=1)
    ws_id = uuid4()

    inserted = _seed_default_notification_preferences(db, ws_id)

    assert inserted == 9
    assert db.execute.call_count == 9

    seen_pairs: set[tuple[str, str]] = set()
    for call in db.execute.call_args_list:
        # call is call(text_obj, params_dict)
        _, params = call.args
        assert params["ws_id"] == str(ws_id)
        seen_pairs.add((params["event_type"], params["destination"]))

    assert seen_pairs == EXPECTED_DEFAULTS


def test_seed_is_idempotent_when_rows_already_exist():
    """Running twice should not fail and should report 0 new inserts when all rows exist."""
    db = _make_db(rowcount=0)  # simulate WHERE NOT EXISTS skipping every row
    ws_id = uuid4()

    inserted = _seed_default_notification_preferences(db, ws_id)

    # All 9 statements still execute (SQL-level guard), but none insert rows.
    assert db.execute.call_count == 9
    assert inserted == 0


def test_seed_statement_uses_where_not_exists_guard():
    """Sanity check the generated SQL so nobody accidentally removes the idempotency guard."""
    db = _make_db(rowcount=1)
    _seed_default_notification_preferences(db, uuid4())

    first_call_text = str(db.execute.call_args_list[0].args[0])
    assert "notification_preferences" in first_call_text
    assert "WHERE NOT EXISTS" in first_call_text
    assert "user_id IS NULL" in first_call_text
