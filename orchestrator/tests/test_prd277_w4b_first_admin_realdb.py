"""PRD-277 US-405 (§5.4) and §5.12 against Postgres and the fake provider: the first admins' rules.

Through the real routes and the in-process provider (the fixtures and helpers of
``test_prd277_w4b_sign_in_rules_realdb.py``), each test in a transaction rolled back:

* **No install admin yet.** A person ``ENTERPRISE_BOOTSTRAP_ADMINS`` names is granted at ANY sign-in
  while the install has no active install admin, so a list mistyped at first and corrected later
  works; the admins are counted under the last-admin guard's lock. (Once one exists, a listed
  person who already exists gets nothing: the sign-in rules file.)
* **One owner stays one owner.** Into a default workspace that already has an owner, a listed person
  becomes an ``admin`` member, and the owner is untouched; the row says which role was given.
* **All or nothing.** A hook that fails after the grant rolls the whole grant back.
* **The email.** An address read from a claim other than ``email`` is not kept unless
  ``OIDC_CLAIM_EMAIL_HOLDS_ONLY_VERIFIED_ADDRESSES`` says the provider verified it, and the list
  reads only the standard ``email`` claim whatever that says.
* **The default workspace before the first row.** With no workspace row, a sign-in still succeeds
  and is audited in it, and so is an access-rule refusal.
"""
from __future__ import annotations

import os
import uuid

from sqlalchemy import event, text

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from config import config  # noqa: E402
from core.auth import sign_in_flow  # noqa: E402
from tests.helpers_fake_oidc_provider import Person  # noqa: E402
from tests.test_prd277_w4b_sign_in_rules_realdb import (  # noqa: E402
    _audits,
    _identity_user,
    _owner_of,
    _refusal,
    _roles_in,
    _sign_in,
    _system_role,
    db_session,
    install,
)

# The install and its session are that file's fixtures, used here by name.
FIXTURES = (db_session, install)


def _no_install_admin(db) -> None:
    """Every install admin the database holds made a plain person, in this test's transaction."""
    db.execute(text("UPDATE users SET system_role = NULL WHERE system_role IN ('admin', 'super_admin')"))
    db.commit()


def _granted(install) -> list:
    return [(uid, d) for a, uid, d in _audits(install.db, install.workspace_id) if a == "auth:bootstrap_admin"]


def _email_of(db, user_id):
    return db.execute(text("SELECT email FROM users WHERE id = :u"), {"u": user_id}).scalar_one()


def _a_person(db) -> int:
    tag = uuid.uuid4().hex[:12]
    return db.execute(text("INSERT INTO users (username, email, is_active) VALUES (:u, :e, true) RETURNING id"),
                      {"u": f"prd277-w4b-{tag}", "e": f"prd277-w4b-{tag}@example.test"}).scalar_one()


# --------------------------------------------------------------------------- no install admin yet


def test_a_corrected_list_grants_at_the_next_sign_in_while_the_install_has_no_admin(install, monkeypatch):
    _no_install_admin(install.db)
    monkeypatch.setattr(config, "ENTERPRISE_BOOTSTRAP_ADMINS", "sub:the-reel-admin")  # mistyped
    install.idp.sign_in_as(Person(subject="the-real-admin"))
    assert _sign_in(install).status_code == 303
    user_id = _identity_user(install.db, "the-real-admin")
    assert _system_role(install.db, user_id) != "admin" and _granted(install) == []

    monkeypatch.setattr(config, "ENTERPRISE_BOOTSTRAP_ADMINS", "sub:the-real-admin")  # corrected
    assert _sign_in(install).status_code == 303

    assert _system_role(install.db, user_id) == "admin"
    assert _roles_in(install.db, user_id, install.workspace_id) == ["owner"]
    assert _owner_of(install.db, install.workspace_id) == user_id
    assert _granted(install) == [(user_id, {"matched_by": "sub", "system_role": "admin", "workspace_role": "owner"})]


def test_the_admins_are_counted_under_the_last_admin_guards_lock(install, monkeypatch):
    _no_install_admin(install.db)
    install.idp.sign_in_as(Person(subject="counted-under-lock"))
    _sign_in(install)
    monkeypatch.setattr(config, "ENTERPRISE_BOOTSTRAP_ADMINS", "sub:counted-under-lock")
    statements = []
    connection = install.db.connection()

    def listener(conn, cursor, statement, params, context, many):
        statements.append(statement)

    event.listen(connection, "before_cursor_execute", listener)
    try:
        assert _sign_in(install).status_code == 303
    finally:
        event.remove(connection, "before_cursor_execute", listener)

    counted = [i for i, sql in enumerate(statements) if "FROM users" in sql and "system_role = ANY" in sql]
    granted = [i for i, sql in enumerate(statements) if sql.lstrip().upper().startswith("UPDATE USERS SET SYSTEM_ROLE")]
    assert counted and granted and counted[0] < granted[0]
    assert all("FOR UPDATE" in statements[i] for i in counted)


# --------------------------------------------------------------------------- one owner stays one owner


def test_into_a_workspace_that_has_an_owner_a_listed_person_becomes_an_admin_member(install, monkeypatch):
    owner = _a_person(install.db)
    install.db.execute(text("UPDATE workspaces SET owner_id = :o WHERE id = CAST(:ws AS uuid)"),
                       {"o": owner, "ws": install.workspace_id})
    install.db.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                            "VALUES (CAST(:ws AS uuid), :o, 'owner', true)"), {"ws": install.workspace_id, "o": owner})
    install.db.commit()
    monkeypatch.setattr(config, "ENTERPRISE_BOOTSTRAP_ADMINS", "sub:second-admin")
    install.idp.sign_in_as(Person(subject="second-admin"))

    assert _sign_in(install).status_code == 303

    user_id = _identity_user(install.db, "second-admin")
    assert _system_role(install.db, user_id) == "admin"
    assert _roles_in(install.db, user_id, install.workspace_id) == ["admin"]
    assert _owner_of(install.db, install.workspace_id) == owner
    assert _roles_in(install.db, owner, install.workspace_id) == ["owner"]
    assert _granted(install) == [(user_id, {"matched_by": "sub", "system_role": "admin", "workspace_role": "admin"})]


# --------------------------------------------------------------------------- all or nothing


def test_a_hook_that_fails_after_the_grant_rolls_the_whole_grant_back(install, monkeypatch):
    monkeypatch.setattr(config, "ENTERPRISE_BOOTSTRAP_ADMINS", "sub:rolled-back-admin")
    sign_in_flow.REGISTRY.register_hook("broken", lambda hook_people, signed_in: 1 / 0)
    install.idp.sign_in_as(Person(subject="rolled-back-admin"))

    assert _refusal(_sign_in(install)) == sign_in_flow.FAILED

    assert _identity_user(install.db, "rolled-back-admin") is None
    assert _owner_of(install.db, install.workspace_id) is None
    assert install.db.execute(text("SELECT COUNT(*) FROM workspace_members WHERE workspace_id = CAST(:ws AS uuid)"),
                              {"ws": install.workspace_id}).scalar_one() == 0
    assert _granted(install) == []


# --------------------------------------------------------------------------- the email


def _address(tag: str) -> str:
    return f"{tag}-{uuid.uuid4().hex[:8]}@example.test"


def test_an_address_from_another_claim_is_not_kept_unless_the_provider_is_said_to_verify_it(install, monkeypatch):
    monkeypatch.setattr(config, "OIDC_CLAIM_EMAIL", "mail")
    off, on = _address("mail-off"), _address("mail-on")
    # email_verified true in the token speaks for the absent standard claim, never for "mail".
    install.idp.sign_in_as(Person(subject="mail-claim-off", email=None, email_verified=True, extra={"mail": off}))
    _sign_in(install)
    kept = _email_of(install.db, _identity_user(install.db, "mail-claim-off"))
    assert kept != off and kept.endswith("@pending")

    monkeypatch.setattr(config, "OIDC_CLAIM_EMAIL_HOLDS_ONLY_VERIFIED_ADDRESSES", "true")
    install.idp.sign_in_as(Person(subject="mail-claim-on", email=None, email_verified=None, extra={"mail": on}))
    _sign_in(install)
    assert _email_of(install.db, _identity_user(install.db, "mail-claim-on")) == on


def test_the_list_reads_only_the_standard_email_claim_whatever_the_setting_says(install, monkeypatch):
    listed = _address("listed")
    monkeypatch.setattr(config, "OIDC_CLAIM_EMAIL", "mail")
    monkeypatch.setattr(config, "OIDC_CLAIM_EMAIL_HOLDS_ONLY_VERIFIED_ADDRESSES", "true")
    monkeypatch.setattr(config, "ENTERPRISE_BOOTSTRAP_ADMINS", "email:" + listed)
    install.idp.sign_in_as(Person(subject="listed-in-mail", email=None, extra={"mail": listed}))

    assert _sign_in(install).status_code == 303

    user_id = _identity_user(install.db, "listed-in-mail")
    assert _email_of(install.db, user_id) == listed  # kept, as the setting says
    assert _system_role(install.db, user_id) != "admin" and _granted(install) == []


# --------------------------------------------------------------------------- the default workspace first


def _workspace_rows(db, workspace_id) -> int:
    return db.execute(text("SELECT COUNT(*) FROM workspaces WHERE id = CAST(:ws AS uuid)"),
                      {"ws": workspace_id}).scalar_one()


def test_with_no_workspace_row_a_sign_in_succeeds_and_is_audited_in_it(install, monkeypatch):
    missing = str(uuid.uuid4())
    monkeypatch.setattr(config, "DEFAULT_WORKSPACE_ID", missing)
    install.idp.sign_in_as(Person(subject="before-the-job"))

    answer = _sign_in(install)

    assert answer.status_code == 303 and answer.headers["location"] == "/settings"
    user_id = _identity_user(install.db, "before-the-job")
    assert _workspace_rows(install.db, missing) == 1 and _owner_of(install.db, missing) is None
    assert [(a, uid) for a, uid, _d in _audits(install.db, missing)] == [
        ("auth:identity_linked", user_id), ("auth:sign_in", user_id)]


def test_with_no_workspace_row_an_access_rule_refusal_is_audited_in_it(install, monkeypatch):
    missing = str(uuid.uuid4())
    monkeypatch.setattr(config, "DEFAULT_WORKSPACE_ID", missing)
    monkeypatch.setattr(config, "OIDC_REQUIRED_CLAIMS", "groups=automatos-users")
    install.idp.sign_in_as(Person(subject="refused-before-the-job"))

    assert _refusal(_sign_in(install)) == sign_in_flow.NOT_ALLOWED

    assert _workspace_rows(install.db, missing) == 1
    assert [(a, uid, d) for a, uid, d in _audits(install.db, missing)] == [
        ("auth:sign_in_refused", None, {"reason": "access_rule"})]
