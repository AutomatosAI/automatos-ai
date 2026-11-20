"""PRD-277 US-402 (§5.11): a sign-in's state is used once, atomically, and lives ten minutes.

``core.auth.oidc_pending`` against real Postgres (``@integration``), with real commits on separate
connections, so two callbacks really race:

* two callbacks with one state, at the same moment: exactly one gets the sign-in, the other
  ``None``, and the row is gone;
* a missing binding, another one, or the state given as the binding gets ``None`` and leaves the
  row for the browser that started it;
* an expired state gets ``None`` and its row is deleted all the same;
* starting a sign-in clears every row whose time is up, in one statement (more than the hundred
  an earlier draft cleared);
* at ``OIDC_PENDING_SIGN_INS_MAX`` live sign-ins, a start is ``SignInsBusy`` and writes nothing;
* an unknown or implausible state gets ``None``; only digests of the state, nonce and binding are
  stored.

Skips when no Postgres is reachable (CI runs it).
"""
from __future__ import annotations

import os
import threading
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, text

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from config import config  # noqa: E402
from core.auth import oidc_pending  # noqa: E402
from core.auth.pairing import hash_secret  # noqa: E402
from core.database.database import get_database_url  # noqa: E402

pytestmark = pytest.mark.integration

RACERS = 2


@pytest.fixture(scope="module")
def engine():
    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT state_hash FROM oidc_pending_sign_ins LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the sign-in state suite needs a reachable Postgres with the W4 table: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def started(new_session):
    """Start sign-ins on committing sessions; every row they wrote is deleted afterwards."""
    states = []

    def start(now=None):
        s = new_session()
        begun = oidc_pending.begin(s, "/after", now=now)
        s.close()
        states.append(begun.state)
        return begun

    yield start
    sweep = new_session.sweep()
    sweep.execute(text("DELETE FROM oidc_pending_sign_ins WHERE state_hash = ANY(:h)"),
                  {"h": [hash_secret(state) for state in states]})
    sweep.commit()
    sweep.close()


def _rows(new_session, state) -> int:
    s = new_session()
    count = s.execute(text("SELECT COUNT(*) FROM oidc_pending_sign_ins WHERE state_hash = :h"),
                      {"h": hash_secret(state)}).scalar_one()
    s.close()
    return count


def test_two_callbacks_with_one_state_exactly_one_wins(started, new_session):
    begun = started()
    barrier, results = threading.Barrier(RACERS), []

    def callback():
        s = new_session()
        barrier.wait()
        results.append(oidc_pending.claim(s, begun.state, begun.binding))
        s.close()

    racers = [threading.Thread(target=callback) for _ in range(RACERS)]
    for racer in racers:
        racer.start()
    for racer in racers:
        racer.join(timeout=30)

    assert len(results) == RACERS
    winners = [claimed for claimed in results if claimed is not None]
    assert len(winners) == 1 and winners[0].return_to == "/after"
    assert winners[0].nonce_hash == hash_secret(begun.nonce)
    assert oidc_pending.code_challenge(winners[0].code_verifier) == begun.code_challenge
    assert _rows(new_session, begun.state) == 0


def test_a_used_state_is_never_used_again(started, new_session):
    begun = started()
    s = new_session()
    assert oidc_pending.claim(s, begun.state, begun.binding) is not None
    assert oidc_pending.claim(s, begun.state, begun.binding) is None
    s.close()


@pytest.mark.parametrize("binding", ["missing", "another", "the-state"])
def test_a_claim_without_its_binding_ends_nothing(started, new_session, binding):
    begun = started()
    presented = {"missing": None, "another": "x" * 43, "the-state": begun.state}[binding]
    s = new_session()

    assert oidc_pending.claim(s, begun.state, presented) is None
    assert _rows(new_session, begun.state) == 1
    assert oidc_pending.claim(s, begun.state, begun.binding) is not None
    s.close()


def test_an_expired_state_is_refused_and_its_row_deleted(started, new_session):
    begun = started(now=datetime.now(timezone.utc) - timedelta(seconds=oidc_pending.SIGN_IN_TTL_SECONDS + 1))
    s = new_session()

    assert oidc_pending.claim(s, begun.state, begun.binding) is None
    s.close()
    assert _rows(new_session, begun.state) == 0


def test_a_state_still_inside_its_ten_minutes_is_accepted(started, new_session):
    begun = started(now=datetime.now(timezone.utc) - timedelta(seconds=oidc_pending.SIGN_IN_TTL_SECONDS - 30))
    s = new_session()
    assert oidc_pending.claim(s, begun.state, begun.binding) is not None
    s.close()


STALE_ROWS = 150  # more than the hundred a batched sweep would clear


def _stale_rows(new_session) -> list:
    """``STALE_ROWS`` sign-ins whose time was up an hour ago; their state digests."""
    hashes = [hash_secret(f"stale-{index}-{uuid.uuid4().hex}") for index in range(STALE_ROWS)]
    s = new_session()
    s.execute(text("INSERT INTO oidc_pending_sign_ins (id, state_hash, nonce_hash, binding_hash, code_verifier, "
                   "return_to, created_at, expires_at) VALUES (:id, :h, :h, :h, :h, '/', :at, :at)"),
              [{"id": uuid.uuid4(), "h": digest, "at": datetime.now(timezone.utc) - timedelta(hours=1)}
               for digest in hashes])
    s.commit()
    s.close()
    return hashes


def test_starting_a_sign_in_clears_every_one_whose_time_is_up(started, new_session):
    hashes = _stale_rows(new_session)

    fresh = started()

    s = new_session()
    left = s.execute(text("SELECT COUNT(*) FROM oidc_pending_sign_ins WHERE state_hash = ANY(:h)"),
                     {"h": hashes}).scalar_one()
    s.close()
    assert left == 0 and _rows(new_session, fresh.state) == 1


def _live(new_session) -> int:
    s = new_session()
    count = s.execute(text("SELECT COUNT(*) FROM oidc_pending_sign_ins WHERE expires_at > now()")).scalar_one()
    s.close()
    return count


def test_at_the_ceiling_a_start_is_busy_and_writes_nothing(started, new_session, monkeypatch):
    monkeypatch.setattr(config, "OIDC_PENDING_SIGN_INS_MAX", _live(new_session) + 1)
    started()
    before = _live(new_session)

    with pytest.raises(oidc_pending.SignInsBusy) as busy:
        started()

    assert _live(new_session) == before and str(busy.value) == oidc_pending.BUSY_MESSAGE


@pytest.mark.parametrize("state", ["unknown-state", "", None, "x" * 300, "état", 42])
def test_an_unknown_or_implausible_state_gets_nothing(started, new_session, state):
    started()
    s = new_session()
    assert oidc_pending.claim(s, state, state) is None
    s.close()


def test_only_digests_of_the_state_and_nonce_are_stored(started, new_session):
    begun = started()
    s = new_session()
    row = s.execute(text("SELECT * FROM oidc_pending_sign_ins WHERE state_hash = :h"),
                    {"h": hash_secret(begun.state)}).mappings().one()
    s.close()

    stored = {str(value) for value in row.values()}
    assert not {begun.state, begun.nonce, begun.binding} & stored
    assert row["nonce_hash"] == hash_secret(begun.nonce) and len(row["state_hash"]) == 64
    assert row["binding_hash"] == hash_secret(begun.binding) and begun.binding not in (begun.state, begun.nonce)
    assert not any(value in repr(begun) for value in (begun.state, begun.nonce, begun.binding))
