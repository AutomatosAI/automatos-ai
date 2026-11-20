"""PRD-277 Wave 2, US-201: a person is linked to the (issuer, subject) they sign in with, never by email.

``core.auth.user_link.link_identity`` against real Postgres (``@integration``), where the unique
indexes on ``users`` and ``user_identities`` decide:

* a first sign-in creates one person and one link, and nothing more: no workspace, no
  membership, no role, no Clerk ID;
* a repeat sign-in finds the same person and changes no row, whatever email it carries;
* an email (in any case) or a username already on another row is refused with a message that
  names no issuer, subject or email, and no row changes;
* two first sign-ins for the same pair racing end with one person and one link, and neither
  is an error, whether they carry the same email, no email or different emails;
* a second request whose link read came before the first request committed is never refused by
  the first's row: the pair is read again before any refusal;
* an email in the placeholder domain, or a blank one, is no email; a padded one is refused;
* an email the provider has not verified is no email: it is kept nowhere, is never an address
  conflict, and never holds an address from the person who signs in with it verified;
* every caller states ``email_verified``: there is no default, and it must be a ``bool``;
* an issuer and a subject are compared exactly: another case, a space before or after, a
  prefix or a longer value is another identity.

The checks on the inputs need no database. Skips the rest when no Postgres is reachable (CI
runs it).
"""
from __future__ import annotations

import dataclasses
import inspect
import os
import threading
import time
import uuid

import pytest
from sqlalchemy import create_engine, text

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from core.auth import user_link  # noqa: E402
from core.auth.user_link import (  # noqa: E402
    IDENTITY_LINK_REFUSED_MESSAGE,
    REFUSED_EMAIL_CONFLICT,
    IdentityLinkRefused,
    LinkedPerson,
    link_identity,
    user_id_for_identity,
)
from core.database.database import get_database_url  # noqa: E402
from core.models.user_identities import ISSUER_MAX, SUBJECT_MAX  # noqa: E402

pytestmark = pytest.mark.integration

# How long the second sign-in is watched to prove it waits on the first's unique keys.
LOCK_WAIT_SECONDS = 0.5
# How long the second sign-in may take once the first has committed.
RELEASE_SECONDS = 10
# How often, and how long, the database is asked whether the second sign-in waits on a lock.
LOCK_POLL_SECONDS = 0.05
LOCK_POLL_LIMIT_SECONDS = 10


@pytest.fixture(scope="module")
def engine():
    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            for probe in ("SELECT 1 FROM user_identities LIMIT 1", "SELECT system_role FROM users LIMIT 1",
                          "SELECT 1 FROM workspace_members LIMIT 1", "SELECT 1 FROM workspaces LIMIT 1"):
                c.execute(text(probe))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the identity link suite needs a reachable Postgres with the PRD-277 W2 schema: {exc}")
    yield eng
    eng.dispose()


class _People:
    """The issuer, emails and rows of one test; the sweep deletes every person it made or linked."""

    def __init__(self, new_session):
        self._new_session = new_session
        self.tag = uuid.uuid4().hex[:10]
        self.issuer = "https://" + "idp-" + self.tag + ".example.test"
        self.users = []

    def email(self, label: str) -> str:
        return f"prd277-w2-{label}-{self.tag}@example.test"

    def holder(self, email: str, username: str = None) -> int:
        """A person who signs in with Clerk and already holds ``email`` (and ``username``)."""
        s = self._new_session()
        uid = s.execute(
            text("INSERT INTO users (username, email, clerk_user_id, is_active) "
                 "VALUES (:u, :e, :cid, true) RETURNING id"),
            {"u": username or email, "e": email, "cid": f"user_prd277_{self.tag}_{len(self.users)}"},
        ).scalar_one()
        s.commit()
        self.users.append(uid)
        return uid

    def link(self, subject: str, issuer: str = None, email_verified: bool = True, **profile) -> LinkedPerson:
        """Sign in as ``subject``; the email is verified unless a test says otherwise."""
        db = self._new_session()
        linked = link_identity(db, issuer=issuer or self.issuer, subject=subject,
                               email_verified=email_verified, **profile)
        db.commit()
        self.users.append(linked.user_id)
        return linked

    def sweep(self):
        s = self._new_session.sweep()
        s.execute(
            text("DELETE FROM users WHERE id = ANY(:users) "
                 "OR id IN (SELECT user_id FROM user_identities WHERE issuer LIKE :pattern)"),
            {"users": self.users, "pattern": f"%{self.tag}%"},
        )
        s.commit()


@pytest.fixture
def people(new_session):
    made = _People(new_session)
    yield made
    made.sweep()


def _person(session, user_id: int) -> dict:
    row = session.execute(
        text("SELECT username, email, name, is_active, clerk_user_id, system_role FROM users WHERE id = :id"),
        {"id": user_id},
    ).mappings().one()
    return dict(row)


def _links(session, user_id: int) -> list:
    return [tuple(r) for r in session.execute(
        text("SELECT issuer, subject FROM user_identities WHERE user_id = :id ORDER BY id"), {"id": user_id},
    )]


def _pair_links(session, issuer: str, subject: str) -> int:
    return session.execute(
        text("SELECT COUNT(*) FROM user_identities WHERE issuer = :i AND subject = :s"), {"i": issuer, "s": subject},
    ).scalar_one()


def _count(session, sql: str, **params) -> int:
    return session.execute(text(sql), params).scalar_one()


def _waits_on_a_lock(engine, pid: int) -> bool:
    """Whether backend ``pid`` comes to wait on a lock (a unique key another transaction holds)."""
    deadline = time.monotonic() + LOCK_POLL_LIMIT_SECONDS
    with engine.connect() as probe:
        while time.monotonic() < deadline:
            waiting = probe.execute(
                text("SELECT wait_event_type FROM pg_stat_activity WHERE pid = :pid"), {"pid": pid},
            ).scalar()
            probe.rollback()
            if waiting == "Lock":
                return True
            time.sleep(LOCK_POLL_SECONDS)
    return False


# --------------------------------------------------------------------------- first and repeat sign-in


def test_a_first_sign_in_creates_one_person_and_one_link_and_nothing_more(people, new_session):
    email = people.email("first")

    linked = people.link("person-first", email=email, name="First Person")

    check = new_session()
    assert linked.created is True
    assert _person(check, linked.user_id) == {
        "username": email, "email": email, "name": "First Person", "is_active": True,
        "clerk_user_id": None, "system_role": None,
    }
    assert _links(check, linked.user_id) == [(people.issuer, "person-first")]
    assert _count(check, "SELECT COUNT(*) FROM workspace_members WHERE user_id = :id", id=linked.user_id) == 0
    assert _count(check, "SELECT COUNT(*) FROM workspaces WHERE owner_id = :id", id=linked.user_id) == 0


def test_a_first_sign_in_with_no_email_gets_a_placeholder_of_its_own(people, new_session):
    one, other = people.link("person-a"), people.link("person-b")

    check = new_session()
    assert one.created and other.created and one.user_id != other.user_id
    first, second = _person(check, one.user_id), _person(check, other.user_id)
    assert first["email"].endswith("@pending") and first["email"] != second["email"]
    assert people.tag not in first["email"] and "person-a" not in first["email"]   # no identifier in it


def test_a_repeat_sign_in_finds_the_same_person_and_changes_no_row(people, new_session):
    first = people.link("person-repeat", email=people.email("repeat"), name="Repeat")
    before = _person(new_session(), first.user_id)

    again = people.link("person-repeat", email=people.email("changed"), name="Someone Else")

    check = new_session()
    assert again == LinkedPerson(user_id=first.user_id, created=False)
    assert _person(check, first.user_id) == before, "a repeat sign-in rewrites no profile"
    assert _pair_links(check, people.issuer, "person-repeat") == 1
    assert _count(check, "SELECT COUNT(*) FROM users WHERE email = :e", e=people.email("changed")) == 0


# --------------------------------------------------------------------------- refusals


def _assert_refusal_leaks_nothing(refusal: IdentityLinkRefused, *never: str) -> None:
    assert refusal.reason == REFUSED_EMAIL_CONFLICT
    assert str(refusal) == refusal.message == IDENTITY_LINK_REFUSED_MESSAGE
    for value in never:
        assert value.lower() not in refusal.message.lower()


def test_an_email_held_by_another_row_is_refused_and_changes_no_row(people, new_session):
    held = f"PRD277-W2-Held-{people.tag}@Example.test"
    holder = people.holder(held)
    holder_before = _person(new_session(), holder)
    db = new_session()

    with pytest.raises(IdentityLinkRefused) as refused:
        link_identity(db, issuer=people.issuer, subject="person-held", email_verified=True,
                      email=held.lower(), name="Taker")

    _assert_refusal_leaks_nothing(refused.value, people.issuer, "person-held", held, people.tag)
    assert db.execute(text("SELECT 1")).scalar_one() == 1, "the caller's transaction is still usable"
    db.rollback()
    check = new_session()
    assert _pair_links(check, people.issuer, "person-held") == 0
    assert _count(check, "SELECT COUNT(*) FROM users WHERE LOWER(email) = LOWER(:e)", e=held) == 1
    assert _person(check, holder) == holder_before and _links(check, holder) == []


def test_a_username_held_by_another_row_is_refused_and_changes_no_row(people, new_session):
    email = people.email("username")
    holder = people.holder(people.email("holder"), username=email)

    with pytest.raises(IdentityLinkRefused) as refused:
        link_identity(new_session(), issuer=people.issuer, subject="person-username", email_verified=True,
                      email=email)

    _assert_refusal_leaks_nothing(refused.value, people.issuer, "person-username", email)
    check = new_session()
    assert _pair_links(check, people.issuer, "person-username") == 0
    assert _count(check, "SELECT COUNT(*) FROM users WHERE email = :e", e=email) == 0
    assert _links(check, holder) == []


# --------------------------------------------------------------------------- the race


@pytest.mark.parametrize("emails", ["same", "none", "different"])
def test_two_first_sign_ins_racing_end_with_one_person_and_one_link(engine, people, new_session, emails):
    subject = f"person-race-{emails}"
    first_email, second_email = {
        "same": (people.email("race"), people.email("race")),
        "none": (None, None),
        "different": (people.email("race-1"), people.email("race-2")),
    }[emails]
    first, second = new_session(), new_session()
    second_pid = second.execute(text("SELECT pg_backend_pid()")).scalar_one()
    won = link_identity(first, issuer=people.issuer, subject=subject, email_verified=True, email=first_email)   # not yet committed
    people.users.append(won.user_id)
    landed, failed, results = threading.Event(), [], []

    def second_sign_in() -> None:
        try:
            results.append(link_identity(second, issuer=people.issuer, subject=subject, email_verified=True,
                                         email=second_email))
            second.commit()
            landed.set()
        except Exception as exc:  # noqa: BLE001 — reported by the assertion below
            failed.append(exc)

    worker = threading.Thread(target=second_sign_in, daemon=True)
    worker.start()
    assert _waits_on_a_lock(engine, second_pid), "the second sign-in never waited on the first's keys"
    assert not landed.wait(LOCK_WAIT_SECONDS), "the second sign-in did not wait for the first's keys"
    first.commit()
    worker.join(RELEASE_SECONDS)

    assert failed == [] and landed.is_set()
    assert won.created is True and results == [LinkedPerson(user_id=won.user_id, created=False)]
    check = new_session()
    assert _pair_links(check, people.issuer, subject) == 1
    held = [e for e in (first_email, second_email) if e] or [user_link._pending_identity_email(people.issuer, subject)]
    assert _count(check, "SELECT COUNT(*) FROM users WHERE email = ANY(:e)", e=held) == 1


def test_a_link_committed_between_the_link_read_and_the_email_check_is_returned_not_refused(
        people, new_session, monkeypatch):
    """The order is forced, single-threaded: the email check of the second request (session
    ``second``) first runs the first request to its commit (session ``first``), so the second
    has read no link before the first commits, and finds the first's row by email after."""
    email, subject = people.email("own-row"), "person-own-row"
    first, second = new_session(), new_session()
    real_row_with_email, real_link_read = user_link._row_with_email, user_link.user_id_for_identity
    steps, won = [], []

    def link_read(db, issuer, subj):
        found = real_link_read(db, issuer, subj)
        steps.append(("second reads the link" if db is second else "first reads the link", found))
        return found

    def email_check(db, address):
        if db is second and not won:
            won.append(link_identity(first, issuer=people.issuer, subject=subject, email_verified=True, email=email))
            first.commit()
            people.users.append(won[0].user_id)
            steps.append(("first commits", won[0].user_id))
        found = real_row_with_email(db, address)
        steps.append(("second checks the email" if db is second else "first checks the email", found and found[0]))
        return found

    monkeypatch.setattr(user_link, "user_id_for_identity", link_read)
    monkeypatch.setattr(user_link, "_row_with_email", email_check)
    again = link_identity(second, issuer=people.issuer, subject=subject, email_verified=True, email=email)
    second.commit()

    winner = won[0].user_id
    assert won[0].created and again == LinkedPerson(user_id=winner, created=False)
    second_steps = [step for step in steps if step[0].startswith("second") or step[0] == "first commits"]
    assert second_steps == [
        ("second reads the link", None), ("first commits", winner),
        ("second checks the email", winner), ("second reads the link", winner),
    ]
    check = new_session()
    assert _pair_links(check, people.issuer, subject) == 1
    assert _count(check, "SELECT COUNT(*) FROM users WHERE email = :e", e=email) == 1


# --------------------------------------------------------------------------- emails that are no email


def test_an_email_in_the_placeholder_domain_is_no_email(people, new_session):
    first = people.link("person-placeholder-a")
    taken = _person(new_session(), first.user_id)["email"]

    others = [people.link("person-placeholder-b", email=taken),
              people.link("person-placeholder-c", email=taken.upper()),
              people.link("person-placeholder-d", email="someone@PENDING")]

    check = new_session()
    assert all(other.created and other.user_id != first.user_id for other in others)
    assert _person(check, first.user_id)["email"] == taken
    emails = [_person(check, other.user_id)["email"] for other in others]
    assert emails == [user_link._pending_identity_email(people.issuer, f"person-placeholder-{x}") for x in "bcd"]
    assert _count(check, "SELECT COUNT(*) FROM users WHERE LOWER(email) = LOWER(:e)", e=taken) == 1


@pytest.mark.parametrize("blank", ["", " ", "\t \n"])
def test_a_blank_email_is_no_email(people, new_session, blank):
    linked = people.link("person-blank", email=blank)

    email = _person(new_session(), linked.user_id)["email"]
    assert linked.created and email == user_link._pending_identity_email(people.issuer, "person-blank")


# --------------------------------------------------------------------------- unverified emails


def test_an_unverified_email_is_kept_nowhere_and_its_owner_later_gets_the_address(people, new_session):
    address = people.email("unverified")

    unverified = people.link("person-unverified", email=address, email_verified=False, name="Unverified")
    check = new_session()
    placeholder = user_link._pending_identity_email(people.issuer, "person-unverified")
    assert unverified.created
    assert {k: v for k, v in _person(check, unverified.user_id).items() if k in ("username", "email")} == {
        "username": placeholder, "email": placeholder,
    }
    assert _count(check, "SELECT COUNT(*) FROM users WHERE LOWER(email) = LOWER(:e) OR LOWER(username) = LOWER(:e)",
                  e=address) == 0

    owner = people.link("person-owner", email=address, email_verified=True)

    check = new_session()
    assert owner.created and owner.user_id != unverified.user_id
    assert _person(check, owner.user_id)["email"] == address
    assert _person(check, unverified.user_id)["email"] == placeholder


def test_an_unverified_email_equal_to_another_persons_address_is_no_conflict(people, new_session):
    address = people.email("unverified-held")
    holder = people.holder(address)
    holder_before = _person(new_session(), holder)

    linked = people.link("person-unverified-held", email=address.upper(), email_verified=False)

    check = new_session()
    assert linked.created and linked.user_id != holder
    assert _person(check, linked.user_id)["email"] == user_link._pending_identity_email(
        people.issuer, "person-unverified-held")
    assert _person(check, holder) == holder_before and _links(check, holder) == []
    assert _count(check, "SELECT COUNT(*) FROM users WHERE LOWER(email) = LOWER(:e)", e=address) == 1


def test_an_unverified_email_is_never_checked_against_other_peoples_addresses(people, monkeypatch):
    def no_address_check(db, address):
        pytest.fail("an unverified email was checked against other people's addresses")

    monkeypatch.setattr(user_link, "_row_with_email", no_address_check)
    linked = people.link("person-unverified-unchecked", email=people.email("unchecked"), email_verified=False)
    assert linked.created


# --------------------------------------------------------------------------- exact comparison


def test_an_issuer_and_a_subject_are_compared_exactly(people, new_session):
    base = people.link("Person-1")
    issuer = people.issuer
    others = [
        (issuer, "person-1"), (issuer, "PERSON-1"), (issuer, " Person-1"), (issuer, "Person-1 "),
        (issuer, "Person-"), (issuer, "Person-10"),
        (issuer + "/", "Person-1"), (issuer.upper(), "Person-1"), (" " + issuer, "Person-1"), (issuer[:-1], "Person-1"),
    ]

    check = new_session()
    assert [user_id_for_identity(check, i, s) for i, s in others] == [None] * len(others)
    linked = [people.link(s, issuer=i) for i, s in others]

    check = new_session()
    assert all(other.created for other in linked)
    assert len({other.user_id for other in linked} | {base.user_id}) == len(others) + 1
    assert user_id_for_identity(check, issuer, "Person-1") == base.user_id
    assert _links(check, base.user_id) == [(issuer, "Person-1")]


# --------------------------------------------------------------------------- inputs (no database)


class _NoDatabase:
    """A session that fails the test if it is touched: a refused input reads and writes nothing."""

    def __getattr__(self, name):
        pytest.fail(f"the database was used ({name}) for an input that should be refused first")


@pytest.mark.parametrize("issuer, subject", [
    ("", "person"), ("https://idp.example.test", ""), (None, "person"), ("https://idp.example.test", None),
    (7, "person"), (b"https://idp.example.test", "person"), ("https://idp.example.test", "per\x00son"),
    ("https://" + "i" * ISSUER_MAX, "person"), ("https://idp.example.test", "s" * (SUBJECT_MAX + 1)),
    ("https://idp.example.test", "é" * (SUBJECT_MAX // 2 + 1)),   # within 255 characters, over 255 bytes
])
def test_an_issuer_or_subject_that_cannot_be_stored_exactly_is_refused_before_any_query(issuer, subject):
    for call in (lambda: link_identity(_NoDatabase(), issuer=issuer, subject=subject, email_verified=True),
                 lambda: user_id_for_identity(_NoDatabase(), issuer, subject)):
        with pytest.raises(ValueError) as refused:
            call()
        for value in (issuer, subject):
            if isinstance(value, str) and len(value) > 3:
                assert value not in str(refused.value)


@pytest.mark.parametrize("profile", [{"email": 7}, {"email": "e" * 256}, {"name": ["x"]}, {"email": "a\x00b"}])
def test_an_email_or_name_a_users_row_cannot_hold_is_refused_before_any_query(profile):
    with pytest.raises(ValueError):
        link_identity(_NoDatabase(), issuer="https://idp.example.test", subject="person", email_verified=True,
                      **profile)


@pytest.mark.parametrize("verified", [True, False])
@pytest.mark.parametrize("padded", [" someone@example.test", "someone@example.test ", "someone@example.test\n",
                                    "\tsomeone@example.test"])
def test_an_email_with_leading_or_trailing_whitespace_is_refused_before_any_query(padded, verified):
    with pytest.raises(ValueError) as refused:
        link_identity(_NoDatabase(), issuer="https://idp.example.test", subject="person",
                      email_verified=verified, email=padded)
    assert "someone@example.test" not in str(refused.value)


@pytest.mark.parametrize("stated", ["true", "false", None, 1, 0])
def test_email_verified_that_is_not_a_bool_is_refused_before_any_query(stated):
    with pytest.raises(TypeError):
        link_identity(_NoDatabase(), issuer="https://idp.example.test", subject="person",
                      email_verified=stated, email="someone@example.test")


def test_the_email_is_never_an_identifier_and_the_answer_is_the_person_never_a_token():
    params = inspect.signature(link_identity).parameters
    assert list(params) == ["db", "issuer", "subject", "email_verified", "email", "name"]
    assert all(params[name].kind is inspect.Parameter.KEYWORD_ONLY
               for name in ("issuer", "subject", "email_verified", "email", "name"))
    assert params["email_verified"].default is inspect.Parameter.empty, "every caller states it"
    assert params["email"].default is None and params["name"].default is None
    with pytest.raises(TypeError):
        link_identity(_NoDatabase(), "https://idp.example.test", "person", True, "someone@example.test")
    with pytest.raises(TypeError):
        link_identity(_NoDatabase(), issuer="https://idp.example.test", subject="person", email="someone@example.test")
    assert [f.name for f in dataclasses.fields(LinkedPerson)] == ["user_id", "created"]
    assert LinkedPerson.__dataclass_params__.frozen
