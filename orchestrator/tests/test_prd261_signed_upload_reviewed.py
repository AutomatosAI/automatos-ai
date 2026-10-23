"""PRD-261 C-05d with C-05c — QB-24: a private upload Automatos vouched for counts as reviewed.

``choices.is_private`` for a workspace's own upload is ``not (SIGNED_UPLOAD_COUNTS_AS_REVIEWED and
signature_state.signature_vouches(...))``: the stored signature has to verify, on every read, with an
active key that neither the keys file nor the hosted feed revoked. A signature merely being stored
never counts. Pins:

* while QB-24 holds, a verified upload is not private (``GET /api/themes/installed``: ``private``
  false, ``optIn.required`` false) and an unsigned newer version of it is; with the constant off,
  the verified one is private again;
* a retired or revoked key, a key the hosted feed revoked (local edition), a signature no key
  verifies, a keys file that does not load and a row read outside any session all leave it private;
* a Marketplace package is still private exactly while it is unreviewed.

Runs on the SQLite app and keys file of ``test_prd261_theme_import_signed.py`` (its fixture and
helpers; none of its tests).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

ORCH = Path(__file__).resolve().parents[1]
if str(ORCH) not in sys.path:
    sys.path.insert(0, str(ORCH))

from core.models.themes import ThemePackage  # noqa: E402
from services.themes import choices, records, signature_state  # noqa: E402
from tests import test_prd261_theme_import_signed as signed  # noqa: E402  (its fixture and helpers)
from tests.test_prd261_theme_import import _hello, _local_ctx, _upload  # noqa: E402

env = signed.env  # the signed-import app over SQLite, its fake S3 and its keys file
SIGNER, OTHER = signed.SIGNER, signed.OTHER


@pytest.fixture(autouse=True)
def _fresh_answers():
    signature_state.clear_cache()
    yield
    signature_state.clear_cache()


def _private_by_version(env):
    """``is_private`` of every stored row, read through a session of its own (the rule reads the sync row)."""
    session = env.sessions()
    try:
        return {row.version: choices.is_private(row) for row in session.query(ThemePackage).all()}
    finally:
        session.close()


def _signed_upload(env):
    """A local client that uploaded hello 1.0.0 with the signer's verified envelope."""
    data = _hello()
    client = signed._client(env, _local_ctx(uuid4()))
    assert signed._upload_signed(client, data, signed._envelope(data)).status_code == 201
    return client


def test_a_verified_private_upload_counts_as_reviewed_while_qb24_holds(env, monkeypatch):
    client = _signed_upload(env)
    assert _upload(client, _hello(version="1.1.0")).status_code == 201

    assert _private_by_version(env) == {"1.0.0": False, "1.1.0": True}
    row = signed._installed(client)["1.0.0"]
    assert (row["private"], row["optIn"]["required"]) == (False, False)
    assert row["signature"] == {"state": "verified", "keyId": SIGNER["keyId"]}

    monkeypatch.setattr(choices, "SIGNED_UPLOAD_COUNTS_AS_REVIEWED", False)
    assert _private_by_version(env) == {"1.0.0": True, "1.1.0": True}
    row = signed._installed(client)["1.0.0"]
    assert (row["private"], row["optIn"]["required"]) == (True, True)


def test_choices_reads_qb24_from_the_one_named_constant():
    assert choices.SIGNED_UPLOAD_COUNTS_AS_REVIEWED is records.SIGNED_UPLOAD_COUNTS_AS_REVIEWED


@pytest.mark.parametrize("status", ["retired", "revoked"])
def test_a_key_no_longer_active_leaves_the_upload_private(env, status):
    client = _signed_upload(env)
    env.keys(signed._key_entry(SIGNER, status=status))
    assert _private_by_version(env) == {"1.0.0": True}
    assert signed._installed(client)["1.0.0"]["optIn"]["required"] is True


def test_a_key_the_hosted_feed_revoked_leaves_the_upload_private_in_the_local_edition(env):
    _signed_upload(env)
    signed._feed_revokes(env, OTHER["keyId"])
    assert _private_by_version(env) == {"1.0.0": False}
    signed._feed_revokes(env, SIGNER["keyId"])
    assert _private_by_version(env) == {"1.0.0": True}


def test_a_stored_signature_no_key_verifies_never_counts(env):
    _signed_upload(env)
    session = env.sessions()
    try:
        (row,) = session.query(ThemePackage).all()
        row.signature = signed._envelope(_hello(), case=OTHER)["signature"]
        session.commit()
    finally:
        session.close()
    assert _private_by_version(env) == {"1.0.0": True}


def test_a_keys_file_that_does_not_load_fails_closed(env):
    _signed_upload(env)
    env.keys(text="{schema")
    assert _private_by_version(env) == {"1.0.0": True}


def test_a_row_read_outside_any_session_vouches_for_nothing(env):
    _signed_upload(env)
    (detached,) = signed._rows(env, ThemePackage)
    assert detached.signature and choices.is_private(detached) is True
    stand_in = SimpleNamespace(owner_type=choices.WORKSPACE_OWNER, signature=detached.signature, reviewed=True)
    assert choices.is_private(stand_in) is True


@pytest.mark.parametrize("reviewed, private", [(True, False), (False, True)])
def test_a_marketplace_package_is_private_exactly_while_unreviewed(reviewed, private):
    package = SimpleNamespace(owner_type=choices.MARKETPLACE_OWNER, signature=None, reviewed=reviewed)
    assert choices.is_private(package) is private
