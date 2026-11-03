"""PRD-268 §6.4 — one thumbs rating per person for each brief or Auto's Read (real Postgres).

The digest feedback route added a row on every press, so a remounted brief card,
a double press or a theme's own thumbs beside the card each added another rating.
It now keeps ONE row per (workspace, person, state_hash): a second press updates
it, a double press waits on the first's lock and then updates its row, and the
new read (GET /api/activity/digest/feedback) answers the caller's own rating
only, never another person's or another workspace's. A caller with no person
behind it (an API key) has no rating to read, and its presses overwrite none.
"""
from __future__ import annotations

import threading
import uuid
from types import SimpleNamespace as NS

import pytest
from sqlalchemy import create_engine, text

HASH = "9f2c41d0a7b3"
# How long the second press is watched to prove it waits on the first's lock.
LOCK_WAIT_SECONDS = 0.5
# How long the second press may take once the first has committed.
RELEASE_SECONDS = 10


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the digest rating tests need a reachable Postgres: {exc}")
    yield eng
    eng.dispose()


def _workspace(session) -> uuid.UUID:
    ws = uuid.uuid4()
    session.execute(text("INSERT INTO workspaces (id, name, plan) VALUES (CAST(:id AS uuid), 'prd268-rating', 'basic')"),
                    {"id": str(ws)})
    return ws


@pytest.fixture
def rated(engine, new_session):
    s = new_session()
    ws, other = _workspace(s), _workspace(s)
    s.commit()
    yield NS(ws=ws, other=other, new=new_session)
    s = new_session.sweep()
    for workspace in (ws, other):
        s.execute(text("DELETE FROM digest_feedback WHERE workspace_id = CAST(:w AS uuid)"), {"w": str(workspace)})
        s.execute(text("DELETE FROM workspaces WHERE id = CAST(:w AS uuid)"), {"w": str(workspace)})
    s.commit()


def _caller(workspace_id: uuid.UUID, user_pk: int) -> NS:
    """A request context whose person is ``users.id`` ``user_pk`` (resolve_user_pk reads an int id as it is)."""
    return NS(workspace_id=workspace_id, auth_type="clerk",
              user=NS(id=user_pk, clerk_user_id=None, email=None, raw_claims=None))


def _personless(workspace_id: uuid.UUID) -> NS:
    """A request context with no person behind it (an API key's)."""
    return NS(workspace_id=workspace_id, auth_type="api_key",
              user=NS(id="api_key", clerk_user_id=None, email=None, raw_claims=None))


def _press(rated, ctx, rating: int) -> dict:
    from api.activity import DigestFeedbackRequest, submit_digest_feedback

    return submit_digest_feedback(DigestFeedbackRequest(state_hash=HASH, rating=rating), db=rated.new(), ctx=ctx)


def _read(rated, ctx) -> dict:
    from api.activity import get_digest_feedback

    return get_digest_feedback(state_hash=HASH, db=rated.new(), ctx=ctx)


def _rows(rated, workspace_id: uuid.UUID) -> list:
    found = rated.new().execute(
        text("SELECT user_id, rating FROM digest_feedback WHERE workspace_id = CAST(:w AS uuid) AND state_hash = :h "
             "ORDER BY user_id"),
        {"w": str(workspace_id), "h": HASH},
    )
    return [tuple(row) for row in found]


def test_a_second_press_by_the_same_person_updates_their_one_row(rated):
    sam = _caller(rated.ws, 7)

    assert _press(rated, sam, 1) == {"ok": True}
    assert _press(rated, sam, -1) == {"ok": True}

    assert _rows(rated, rated.ws) == [("7", -1)]          # before: [("…", 1), ("…", -1)], a row per press


def test_another_persons_rating_is_their_own(rated):
    sam, ana = _caller(rated.ws, 7), _caller(rated.ws, 8)

    _press(rated, sam, 1)
    _press(rated, ana, -1)
    _press(rated, ana, 1)

    assert _rows(rated, rated.ws) == [("7", 1), ("8", 1)]
    _press(rated, sam, -1)
    assert _rows(rated, rated.ws) == [("7", -1), ("8", 1)]


def test_the_read_answers_the_callers_own_rating_only(rated):
    sam, ana = _caller(rated.ws, 7), _caller(rated.ws, 8)

    assert _read(rated, sam) == {"state_hash": HASH, "rating": None}
    _press(rated, sam, -1)

    assert _read(rated, sam) == {"state_hash": HASH, "rating": -1}
    assert _read(rated, ana) == {"state_hash": HASH, "rating": None}           # not Sam's
    assert _read(rated, _caller(rated.other, 7)) == {"state_hash": HASH, "rating": None}   # not another workspace's
    _press(rated, ana, 1)
    assert _read(rated, sam)["rating"] == -1 and _read(rated, ana)["rating"] == 1


def test_a_caller_with_no_person_reads_no_rating_and_overwrites_none(rated):
    _press(rated, _personless(rated.ws), 1)
    _press(rated, _personless(rated.ws), -1)

    rows = _rows(rated, rated.ws)
    assert sorted(rating for _, rating in rows) == [-1, 1] and {user for user, _ in rows} == {None}   # a row each
    assert _read(rated, _personless(rated.ws))["rating"] is None         # never another personless caller's
    assert _read(rated, _caller(rated.ws, 7))["rating"] is None


def test_every_row_a_person_has_for_the_state_takes_the_new_rating(rated):
    from core.models.core import DigestFeedback

    s = rated.new()
    s.add_all([DigestFeedback(workspace_id=rated.ws, user_id="7", state_hash=HASH, rating=1) for _ in range(2)])
    s.commit()

    _press(rated, _caller(rated.ws, 7), -1)

    assert _rows(rated, rated.ws) == [("7", -1), ("7", -1)]
    assert _read(rated, _caller(rated.ws, 7))["rating"] == -1


def test_a_double_press_waits_for_the_first_and_updates_its_row(rated):
    from core.models.core import DigestFeedback
    from services.digest_feedback import hold_rating_lock, record_rating

    first, second = rated.new(), rated.new()
    hold_rating_lock(first, rated.ws, "7", HASH)                   # the first press, mid-transaction
    first.add(DigestFeedback(workspace_id=rated.ws, user_id="7", state_hash=HASH, rating=1))
    first.flush()
    landed, failed = threading.Event(), []

    def second_press() -> None:
        try:
            record_rating(second, rated.ws, "7", HASH, -1)
            landed.set()
        except Exception as exc:  # noqa: BLE001 — reported by the assertion below
            failed.append(exc)

    worker = threading.Thread(target=second_press, daemon=True)
    worker.start()
    assert not landed.wait(LOCK_WAIT_SECONDS), "the second press did not wait for the first"
    first.commit()
    worker.join(RELEASE_SECONDS)

    assert failed == [] and landed.is_set()
    assert _rows(rated, rated.ws) == [("7", -1)]                  # before: two rows, one per press
