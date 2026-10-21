"""PRD-265 US-002 — the credit ledger's pure parts and its schema contract.

How a movement splits across buckets (trial, then monthly, then top-up; an
overdraft lands on the last spendable bucket), what a balance can spend, and the
model/migration pair: the same table, constraint and index names in both, billing
keyed on ``workspace_id`` and never on a Clerk user ID. The real-Postgres half
(idempotency, the same-transaction balance, the trial conversion) is
``test_prd265_credit_ledger_realdb.py``.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from core.models.credits import CreditBalance, CreditLedgerEntry
from services.credits.ledger import (
    BUCKET_MONTHLY,
    BUCKET_OVERHEAD,
    BUCKET_TOPUP,
    BUCKET_TRIAL,
    KIND_DEBIT,
    KIND_GRANT,
    Balance,
    Entry,
    allocate,
)

_MIGRATION = Path(__file__).resolve().parents[1] / "alembic" / "versions" / "prd265_credit_ledger.py"


def _migration():
    spec = importlib.util.spec_from_file_location("_prd265_credit_ledger", _MIGRATION)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _debit(credits: int) -> Entry:
    return Entry(kind=KIND_DEBIT, source_type="usage", source_ref="1", credits=-credits)


# --------------------------------------------------------------------------- #
# allocate — the spend order
# --------------------------------------------------------------------------- #


def test_a_debit_spends_trial_first_then_monthly_then_topup():
    bal = Balance(trial=3, monthly=10, topup=50)
    assert allocate(bal, _debit(5)) == [(BUCKET_TRIAL, -3), (BUCKET_MONTHLY, -2)]
    assert allocate(bal, _debit(20)) == [(BUCKET_TRIAL, -3), (BUCKET_MONTHLY, -10), (BUCKET_TOPUP, -7)]


def test_a_debit_inside_one_bucket_writes_one_row():
    assert allocate(Balance(trial=500), _debit(5)) == [(BUCKET_TRIAL, -5)]


def test_an_overdraft_lands_on_the_last_spendable_bucket():
    # Calls metered after the balance ran out: the rest goes negative, and
    # spending pauses until it is paid back.
    rows = allocate(Balance(trial=2, monthly=0, topup=1), _debit(10))
    assert rows == [(BUCKET_TRIAL, -2), (BUCKET_TOPUP, -8)]
    assert sum(c for _, c in rows) == -10


def test_an_own_keys_plan_spends_only_its_trial():
    # Own keys has no platform lane for monthly/top-up credits: the caller passes
    # only the trial bucket, and an overdraft stays on the trial.
    rows = allocate(Balance(trial=3, topup=100), _debit(5), spend_buckets=(BUCKET_TRIAL,))
    assert rows == [(BUCKET_TRIAL, -5)]


def test_a_negative_bucket_pays_nothing():
    rows = allocate(Balance(trial=-4, monthly=10), _debit(5))
    assert rows == [(BUCKET_MONTHLY, -5)]


def test_a_named_bucket_is_written_as_given():
    grant = Entry(kind=KIND_GRANT, source_type="trial_grant", source_ref="7", credits=500, bucket=BUCKET_TRIAL)
    assert allocate(Balance(), grant) == [(BUCKET_TRIAL, 500)]
    overhead = Entry(kind=KIND_DEBIT, source_type="overhead_day", source_ref="x", credits=0,
                     bucket=BUCKET_OVERHEAD, actual_cost_usd=0.42)
    assert allocate(Balance(), overhead) == [(BUCKET_OVERHEAD, 0)]


def test_an_addition_must_name_its_bucket():
    with pytest.raises(ValueError):
        allocate(Balance(), Entry(kind=KIND_GRANT, source_type="x", source_ref="y", credits=10))


def test_spendable_ignores_negative_buckets_and_honours_the_bucket_set():
    bal = Balance(trial=-5, monthly=20, topup=7)
    assert bal.spendable() == 27
    assert bal.spendable((BUCKET_TRIAL,)) == 0


# --------------------------------------------------------------------------- #
# Model / migration pair
# --------------------------------------------------------------------------- #


def test_the_model_and_the_migration_declare_the_same_names():
    src = _MIGRATION.read_text()
    ledger = CreditLedgerEntry.__table__
    names = {c.name for c in ledger.constraints if c.name} | {i.name for i in ledger.indexes}
    for name in ("uq_credit_ledger_source", "ck_credit_ledger_kind", "ck_credit_ledger_bucket",
                 "ix_credit_ledger_workspace_created"):
        assert name in names, name
        assert name in src, name
    for column in ledger.columns:
        assert column.name in src, column.name
    for column in CreditBalance.__table__.columns:
        assert column.name in src, column.name


def test_the_source_key_includes_the_bucket():
    # A debit that spans buckets writes one row per bucket for the same event.
    uq = next(c for c in CreditLedgerEntry.__table__.constraints if c.name == "uq_credit_ledger_source")
    assert [c.name for c in uq.columns] == ["source_type", "source_ref", "bucket"]


def test_billing_keys_on_the_workspace_never_on_a_clerk_user():
    for table in (CreditLedgerEntry.__table__, CreditBalance.__table__):
        cols = {c.name for c in table.columns}
        assert "workspace_id" in cols
        assert not any("clerk" in c for c in cols), cols


def test_llm_usage_carries_the_ask_the_charge_class_and_their_partial_indexes():
    from core.models.core import LLMUsage

    columns = LLMUsage.__table__.columns
    assert "ask_id" in columns and "charge_class" in columns
    indexes = {i.name: i for i in LLMUsage.__table__.indexes}
    assert [c.name for c in indexes["ix_llm_usage_ask_id"].columns] == ["ask_id"]
    # The meter's index holds only the calls still waiting to be charged.
    assert [c.name for c in indexes["ix_llm_usage_to_meter"].columns] == ["id"]
    src = _MIGRATION.read_text()
    assert "ix_llm_usage_ask_id" in src and "ix_llm_usage_to_meter" in src
    assert "WHERE charge_class = 'customer'" in src
    assert "ADD COLUMN IF NOT EXISTS charge_class" in src


def test_the_migration_chains_onto_the_relinks_revision():
    mod = _migration()
    assert mod.revision == "prd265_credit_ledger"
    assert mod.down_revision == "clerk_user_relinks_20261019"
    # The trial conversion is keyed once per person and never overwrites a grant.
    assert "'trial_grant'" in mod.CONVERT_TRIALS_SQL
    assert "ON CONFLICT ON CONSTRAINT uq_credit_ledger_source DO NOTHING" in mod.CONVERT_TRIALS_SQL
    assert "FLOOR(" in mod.CONVERT_TRIALS_SQL
