"""The Clerk cutover's one-time relink (``core/auth/clerk_cutover.py``), unit level.

The hold on PR #1198 set the rules: old-instance IDs only, verified email only, each
row relinked at most once, every relink audited. These pin the decision for each
case, the compare-and-set write, the rollback plan and the operator script's
paging. The real-Postgres journey (moves, records, rollback, re-apply) is in
``test_clerk_cutover_returning_user_realdb.py``.
"""
from __future__ import annotations

import os

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_DB", "test")

from unittest.mock import MagicMock, patch  # noqa: E402

import pytest  # noqa: E402

from core.auth import clerk_cutover as cutover  # noqa: E402
from core.auth.clerk_cutover import AppUser, Decision, ProductionUser, RelinkRecord  # noqa: E402

OLD = "user_dev_2xOld"
NEW = "user_prod_2yNew"
EMAIL = "returning@example.test"


def _prod(external_id=OLD, emails=(EMAIL,)):
    return ProductionUser(clerk_user_id=NEW, external_id=external_id, verified_emails=frozenset(emails))


def _row(clerk_user_id=OLD, email=EMAIL, user_id=7):
    return AppUser(id=user_id, clerk_user_id=clerk_user_id, email=email)


# ── reading the Clerk user object ────────────────────────────────────────

def test_only_verified_addresses_count_and_they_are_lowercased():
    payload = {
        "id": NEW,
        "external_id": OLD,
        "email_addresses": [
            {"email_address": "Returning@Example.TEST", "verification": {"status": "verified"}},
            {"email_address": "other@example.test", "verification": {"status": "unverified"}},
            {"email_address": "none@example.test", "verification": None},
            "not-an-object",
        ],
    }
    user = cutover.production_user(payload)
    assert user == ProductionUser(NEW, OLD, frozenset({EMAIL}))


def test_a_user_without_an_external_id_was_not_imported():
    assert cutover.production_user({"id": NEW, "external_id": None}).external_id is None
    assert cutover.production_user({"id": NEW, "external_id": ""}).external_id is None


def test_a_user_without_an_id_stops_the_run():
    with pytest.raises(ValueError):
        cutover.production_user({"external_id": OLD})


# ── the decision for each case ───────────────────────────────────────────

def test_a_returning_row_is_relinked_by_the_import_link_and_a_verified_email():
    decision = cutover.decide(_prod(), _row(email="Returning@Example.TEST"), None, None)
    assert decision == Decision(cutover.RELINK, NEW, OLD, user_id=7, verified_email=EMAIL)


def test_a_person_new_on_production_is_left_alone():
    assert cutover.decide(_prod(external_id=None), None, None, None).outcome == cutover.NOT_IMPORTED


def test_no_row_holds_the_old_id():
    assert cutover.decide(_prod(), None, None, None).outcome == cutover.NO_ROW


def test_a_row_already_on_the_production_id_is_not_moved_again():
    decision = cutover.decide(_prod(), None, _row(clerk_user_id=NEW), None)
    assert (decision.outcome, decision.user_id) == (cutover.ALREADY_LINKED, 7)


def test_a_production_id_another_row_holds_blocks_the_move():
    decision = cutover.decide(_prod(), _row(), _row(clerk_user_id=NEW, user_id=9), None)
    assert (decision.outcome, decision.user_id) == (cutover.TARGET_TAKEN, 7)


def test_an_email_that_is_not_verified_on_the_production_user_blocks_the_move():
    """The ID link alone is not enough: the row's email must be a verified address too."""
    assert cutover.decide(_prod(emails=()), _row(), None, None).outcome == cutover.EMAIL_NOT_VERIFIED
    other = cutover.decide(_prod(emails=("someone@example.test",)), _row(), None, None)
    assert other.outcome == cutover.EMAIL_NOT_VERIFIED


def test_a_row_is_relinked_at_most_once():
    applied = RelinkRecord(7, OLD, NEW, reverted=False)
    elsewhere = RelinkRecord(7, OLD, "user_prod_other", reverted=True)
    assert cutover.decide(_prod(), _row(), None, applied).outcome == cutover.RELINKED_BEFORE
    assert cutover.decide(_prod(), _row(), None, elsewhere).outcome == cutover.RELINKED_BEFORE


def test_after_a_rollback_only_the_recorded_pair_is_reapplied():
    reverted = RelinkRecord(7, OLD, NEW, reverted=True)
    decision = cutover.decide(_prod(), _row(), None, reverted)
    assert decision == Decision(cutover.REAPPLY, NEW, OLD, user_id=7, verified_email=EMAIL)


def test_two_production_users_naming_one_old_id_move_nothing():
    twin = ProductionUser("user_prod_twin", OLD, frozenset({EMAIL}))
    db = MagicMock()
    decisions = cutover.plan_relinks(db, [_prod(), twin])
    assert [d.outcome for d in decisions] == [cutover.DUPLICATE_EXTERNAL_ID] * 2
    db.execute.assert_not_called()


# ── the writes ───────────────────────────────────────────────────────────

def _db_with_rowcounts(*counts):
    db = MagicMock()
    db.execute.side_effect = [MagicMock(rowcount=c) for c in counts]
    return db


def _sql(db):
    return [" ".join(str(c.args[0]).split()) for c in db.execute.call_args_list]


def test_apply_moves_only_relink_rows_with_a_compare_and_set_and_a_record():
    skipped = Decision(cutover.EMAIL_NOT_VERIFIED, NEW, OLD, user_id=8)
    move = Decision(cutover.RELINK, NEW, OLD, user_id=7, verified_email=EMAIL)
    db = _db_with_rowcounts(1, 1)

    assert cutover.apply_relinks(db, [skipped, move]) == 1

    update, insert = _sql(db)
    assert update == "UPDATE users SET clerk_user_id = :to_id WHERE id = :uid AND clerk_user_id = :from_id"
    assert db.execute.call_args_list[0].args[1] == {"to_id": NEW, "uid": 7, "from_id": OLD}
    assert insert.startswith("INSERT INTO clerk_user_relinks")
    assert db.execute.call_args_list[1].args[1] == {"uid": 7, "old": OLD, "new": NEW, "email": EMAIL}
    db.commit.assert_not_called()


def test_a_row_that_changed_since_the_plan_stops_the_run():
    db = _db_with_rowcounts(0)
    with pytest.raises(cutover.RelinkConflict):
        cutover.apply_relinks(db, [Decision(cutover.RELINK, NEW, OLD, user_id=7, verified_email=EMAIL)])
    assert len(db.execute.call_args_list) == 1, "no record is written for a move that did not happen"


def test_a_reapply_clears_the_rollback_stamp_of_that_exact_pair():
    db = _db_with_rowcounts(1, 1)
    cutover.apply_relinks(db, [Decision(cutover.REAPPLY, NEW, OLD, user_id=7, verified_email=EMAIL)])
    record = _sql(db)[1]
    assert record.startswith("UPDATE clerk_user_relinks SET reverted_at = NULL")
    assert "old_clerk_user_id = :old AND new_clerk_user_id = :new AND reverted_at IS NOT NULL" in record


def test_the_rollback_plan_reverts_only_rows_still_on_their_production_id():
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = [
        (7, OLD, NEW, NEW),
        (8, "user_dev_b", "user_prod_b", "user_dev_b"),
        (9, "user_dev_c", "user_prod_c", "user_somebody_else"),
    ]
    outcomes = [(d.user_id, d.outcome) for d in cutover.plan_rollback(db)]
    assert outcomes == [(7, cutover.REVERT), (8, cutover.ALREADY_REVERTED), (9, cutover.CHANGED_SINCE)]


def test_apply_rollback_moves_back_and_stamps_the_record():
    db = _db_with_rowcounts(1, 1)
    assert cutover.apply_rollback(db, [Decision(cutover.REVERT, NEW, OLD, user_id=7)]) == 1
    update, stamp = _sql(db)
    assert db.execute.call_args_list[0].args[1] == {"to_id": OLD, "uid": 7, "from_id": NEW}
    assert stamp.startswith("UPDATE clerk_user_relinks SET reverted_at = now()")


# ── the operator script ──────────────────────────────────────────────────

def test_the_script_pages_through_every_production_user_once():
    from scripts import clerk_cutover_relink as script

    full = [{"id": f"user_prod_{i}", "external_id": f"user_dev_{i}"} for i in range(script.PAGE_SIZE)]
    last = [{"id": "user_prod_last", "external_id": None}, full[-1]]  # one user seen on both pages
    clerk = MagicMock()
    clerk.list_users_page.side_effect = [full, last]
    with patch.object(script, "get_clerk_auth", return_value=clerk):
        users = script.fetch_production_users()
    assert len(users) == script.PAGE_SIZE + 1
    assert [c.args for c in clerk.list_users_page.call_args_list] == [
        (script.PAGE_SIZE, 0), (script.PAGE_SIZE, script.PAGE_SIZE),
    ]


@patch("core.auth.clerk.httpx.Client")
def test_the_clerk_user_list_reads_one_page_and_fails_loudly(client_cls):
    from core.auth.clerk import ClerkAuth

    auth = ClerkAuth.__new__(ClerkAuth)
    auth.clerk_secret_key = "placeholder-secret"
    auth.clerk_api_base = "https://api.clerk.com"
    client = client_cls.return_value.__enter__.return_value

    client.get.return_value = MagicMock(status_code=200, json=lambda: [{"id": NEW}])
    assert auth.list_users_page(100, 200) == [{"id": NEW}]
    assert client.get.call_args.kwargs["params"] == {"limit": 100, "offset": 200, "order_by": "+created_at"}

    client.get.return_value = MagicMock(status_code=401, json=lambda: {})
    with pytest.raises(ValueError):
        auth.list_users_page(100, 0)
    client.get.return_value = MagicMock(status_code=200, json=lambda: {"data": []})
    with pytest.raises(ValueError):
        auth.list_users_page(100, 0)


# ── the audit table ──────────────────────────────────────────────────────

def test_the_model_and_the_migration_declare_one_shape_with_at_most_one_relink_per_row():
    from pathlib import Path

    from sqlalchemy import UniqueConstraint

    from core.models.clerk_user_relinks import ClerkUserRelink

    table = ClerkUserRelink.__table__
    columns = ("id", "user_id", "old_clerk_user_id", "new_clerk_user_id", "verified_email", "relinked_at", "reverted_at")
    assert tuple(c.name for c in table.columns) == columns
    uniques = {c.name: [col.name for col in c.columns] for c in table.constraints if isinstance(c, UniqueConstraint)}
    assert uniques == {"uq_clerk_user_relinks_user": ["user_id"], "uq_clerk_user_relinks_new_id": ["new_clerk_user_id"]}
    (fk,) = table.c.user_id.foreign_keys
    assert (fk.target_fullname, fk.ondelete) == ("users.id", "CASCADE")

    migration = (Path(__file__).resolve().parents[1] / "alembic" / "versions" / "clerk_user_relinks_20261019.py").read_text()
    assert "CREATE TABLE IF NOT EXISTS clerk_user_relinks" in migration
    for column in columns[1:]:
        assert f"            {column} " in migration, column
    assert "CONSTRAINT uq_clerk_user_relinks_user UNIQUE (user_id)" in migration
    assert "CONSTRAINT uq_clerk_user_relinks_new_id UNIQUE (new_clerk_user_id)" in migration
    assert "REFERENCES users(id) ON DELETE CASCADE" in migration
