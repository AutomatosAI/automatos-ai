"""PRD-261 (D5) — ``scripts/theme_publish_actor.py``: the publish key's service user, created once.

Pins: the first run inserts one ``users`` row (the reserved ``.invalid`` email, no Clerk id, active)
and prints its id with the setting to put it in; a second run finds the same row and inserts nothing;
a row with that email linked to a Clerk user is refused (exit 2), unchanged; another row holding the
username leaves nothing written (exit 1). Over SQLite with the ``users`` table's unique columns.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

ORCH = Path(__file__).resolve().parents[1]
if str(ORCH) not in sys.path:
    sys.path.insert(0, str(ORCH))

from scripts import theme_publish_actor as script  # noqa: E402
from tests.helpers_theme_publish import USERS_DDL  # noqa: E402

USERS = sa.text("SELECT id, username, email, clerk_user_id, is_active FROM users ORDER BY id")


@pytest.fixture
def sessions():
    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    with engine.begin() as conn:
        conn.execute(sa.text(USERS_DDL))
        conn.execute(sa.text("INSERT INTO users (id, username, email, is_active) VALUES (3, 'gerard', 'g@example.com', true)"))
    try:
        yield sessionmaker(bind=engine, autoflush=False)
    finally:
        engine.dispose()


def _run(sessions):
    out = io.StringIO()
    code = script.main([], session_factory=sessions, out=out)
    return code, out.getvalue()


def _users(sessions):
    with sessions() as session:
        return [tuple(row) for row in session.execute(USERS).fetchall()]


def test_the_first_run_creates_the_service_user_and_prints_its_id(sessions):
    code, printed = _run(sessions)

    assert code == script.EXIT_OK
    (person, actor) = _users(sessions)
    assert person[0] == 3
    assert actor[1:] == (script.SERVICE_USERNAME, script.SERVICE_EMAIL, None, 1)
    assert f"THEME_PUBLISH_ACTOR={actor[0]}" in printed and "created" in printed


def test_a_second_run_finds_the_same_row_and_writes_nothing(sessions):
    _run(sessions)
    before = _users(sessions)

    code, printed = _run(sessions)

    assert code == script.EXIT_OK and _users(sessions) == before
    assert f"THEME_PUBLISH_ACTOR={before[1][0]}" in printed and "already there" in printed


def test_a_row_with_the_email_linked_to_a_clerk_user_is_refused(sessions):
    with sessions() as session:
        session.execute(sa.text("INSERT INTO users (id, username, email, clerk_user_id, is_active) "
                                "VALUES (9, 'someone', :email, 'user_abc', true)"), {"email": script.SERVICE_EMAIL.upper()})
        session.commit()
    before = _users(sessions)

    code, printed = _run(sessions)

    assert code == script.EXIT_REFUSED and _users(sessions) == before
    assert "THEME_PUBLISH_ACTOR=" not in printed


def test_another_row_holding_the_username_writes_nothing(sessions):
    with sessions() as session:
        session.execute(sa.text("INSERT INTO users (id, username, email, is_active) VALUES (5, :name, 'x@example.com', true)"),
                        {"name": script.SERVICE_USERNAME})
        session.commit()
    before = _users(sessions)

    code, printed = _run(sessions)

    assert code == script.EXIT_FAILED and _users(sessions) == before
    assert printed.strip() == script.NOT_CREATED
