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
* one address at ``OIDC_PENDING_SIGN_INS_PER_ADDRESS`` keeps only its newest sign-ins when it starts
  another, and another address's are untouched;
* at ``OIDC_PENDING_SIGN_INS_MAX`` live sign-ins, a start ends the oldest and is stored, never
  refused; a sign-in started before a flood from other addresses that stays under the ceiling
  still completes;
* a row keeps the hash of the address that started it, never the address;
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


def _new_address() -> str:
    """An address in its own IPv6 /64 of the documentation prefix (RFC 3849), so tests share no bucket."""
    bits = uuid.uuid4().int
    return f"2001:db8:{bits & 0xFFFF:x}:{(bits >> 16) & 0xFFFF:x}::1"


def _ago(seconds: float) -> datetime:
    return datetime.now(timezone.utc) - timedelta(seconds=seconds)


@pytest.fixture
def started(new_session):
    """Start sign-ins on committing sessions (from one address of their own unless told another);
    every row they wrote is deleted afterwards."""
    states, own_address = [], _new_address()

    def start(now=None, address=None):
        s = new_session()
        begun = oidc_pending.begin(s, "/after", address or own_address, now=now)
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
    s.execute(text("INSERT INTO oidc_pending_sign_ins (id, state_hash, nonce_hash, binding_hash, client_hash, "
                   "code_verifier, return_to, created_at, expires_at) VALUES (:id, :h, :h, :h, :h, :h, '/', :at, :at)"),
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


PER_ADDRESS = 3


def test_an_address_at_its_limit_keeps_only_its_newest_and_another_address_is_untouched(
        started, new_session, monkeypatch):
    monkeypatch.setattr(config, "OIDC_PENDING_SIGN_INS_PER_ADDRESS", PER_ADDRESS)
    other = _new_address()
    mine = [started(now=_ago(50)), started(now=_ago(40)), started(now=_ago(30))]
    theirs = [started(now=_ago(45), address=other), started(now=_ago(35), address=other)]

    mine += [started(now=_ago(20)), started(now=_ago(10))]

    assert [_rows(new_session, begun.state) for begun in mine] == [0, 0, 1, 1, 1]
    assert [_rows(new_session, begun.state) for begun in theirs] == [1, 1]


def test_at_the_ceiling_the_oldest_sign_in_goes_and_the_new_one_is_stored(started, new_session, monkeypatch):
    # Near its end, so it is older than any other row a test left: it is the oldest overall.
    oldest = started(now=_ago(oidc_pending.SIGN_IN_TTL_SECONDS - 60), address=_new_address())
    younger = started(now=_ago(200), address=_new_address())
    monkeypatch.setattr(config, "OIDC_PENDING_SIGN_INS_MAX", _live(new_session))
    before = _live(new_session)

    newest = started(address=_new_address())

    assert _rows(new_session, oldest.state) == 0
    assert _rows(new_session, younger.state) == 1 and _rows(new_session, newest.state) == 1
    assert _live(new_session) == before


FLOOD_ADDRESSES = 30


def test_a_sign_in_started_before_a_flood_under_the_ceiling_still_completes(started, new_session, monkeypatch):
    monkeypatch.setattr(config, "OIDC_PENDING_SIGN_INS_PER_ADDRESS", PER_ADDRESS)
    person = started(now=_ago(5))
    flooder = _new_address()
    monkeypatch.setattr(config, "OIDC_PENDING_SIGN_INS_MAX", _live(new_session) + FLOOD_ADDRESSES + PER_ADDRESS)

    for _ in range(FLOOD_ADDRESSES):
        started(address=_new_address())
    for _ in range(PER_ADDRESS * 4):
        started(address=flooder)

    s = new_session()
    assert oidc_pending.claim(s, person.state, person.binding) is not None
    count = s.execute(text("SELECT COUNT(*) FROM oidc_pending_sign_ins WHERE client_hash = :c"),
                      {"c": oidc_pending.client_hash(flooder)}).scalar_one()
    s.close()
    assert count == PER_ADDRESS


def test_a_row_keeps_the_hash_of_its_address_never_the_address(started, new_session):
    address = "198.51.100.23"  # TEST-NET-2 (RFC 5737)
    begun = started(address=address)
    s = new_session()
    row = s.execute(text("SELECT * FROM oidc_pending_sign_ins WHERE state_hash = :h"),
                    {"h": hash_secret(begun.state)}).mappings().one()
    s.close()

    assert row["client_hash"] == oidc_pending.client_hash(address) == hash_secret(f"oidc_pending:{address}")
    assert len(row["client_hash"]) == 64 and not any(address in str(value) for value in row.values())
    same_64 = (oidc_pending.client_hash("2001:db8:1:2::a"), oidc_pending.client_hash("2001:db8:1:2::b"))
    assert same_64[0] == same_64[1] != oidc_pending.client_hash("2001:db8:1:3::a")


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
