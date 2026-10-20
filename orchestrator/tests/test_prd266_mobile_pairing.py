"""PRD-266 US-003b — pairing a phone with a local install: ``services/mobile/pairing.py``.

Studio mints a one-time code (a PENDING ``mobile_devices`` row); the phone exchanges it
once for a device token. Proven here, against Postgres (CI):

* The code's alphabet, length, TTL, hashing and minting are the kernel's
  (``core.auth.pairing``): the service holds no copy of any of them.
* The code and the token are stored only as their SHA-256; neither appears in any column
  or in ``to_dict()``. The token starts ``amd_`` and is returned once.
* A code lives 600 seconds; an expired, unknown or used code is ``None`` and changes
  nothing. ``rotate_after`` is ``paired_at + MOBILE_TOKEN_ROTATE_DAYS``.
* The QR payload is a universal link on the web app (``{web_url}/go/pair?api=…&code=…``),
  never a custom scheme; ``api`` is the request's base URL on a local install still on the
  hosted ``PUBLIC_API_HOST`` and ``https://<host>`` once the operator names one (K15).
* A phone that pairs again keeps its ONE row (the existing row survives; the new code's
  pending row is deleted), a revoked one included; the audit keeps the last 50 events.
* On saas nothing pairs and no paired row is written (§8).
"""
from __future__ import annotations

import inspect
import re
import uuid
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlsplit

import pytest
from sqlalchemy import text

WEB_URL = "https://studio.prd266.test"
REQUEST_BASE = "http://192.0.2.10:8000/"
INSTALL_ID = "prd266-install-0001"
OPERATOR_HOST = "go.prd266.test"
PHONE_NAME = "Studio test phone"


@pytest.fixture
def local(monkeypatch):
    """A local install on the hosted API default, its web app at ``WEB_URL``."""
    from config import HOSTED_PUBLIC_API_HOST, config

    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(config, "FRONTEND_URL", WEB_URL)
    monkeypatch.setattr(config, "PUBLIC_API_HOST", HOSTED_PUBLIC_API_HOST)
    return config


def _seed_user(db) -> int:
    tag = uuid.uuid4().hex[:12]
    return db.execute(
        text("INSERT INTO users (username, email, is_active) VALUES (:u, :e, true) RETURNING id"),
        {"u": f"prd266-pair-{tag}", "e": f"prd266-pair-{tag}@t.test"},
    ).scalar()


@pytest.fixture
def owner(db_session, seed_workspace):
    """A workspace and its person: ``(workspace_id, user_pk)``."""
    return uuid.UUID(seed_workspace()), _seed_user(db_session)


def _mint(db, owner):
    from services.mobile.pairing import create_pairing_code

    return create_pairing_code(db, owner[0], owner[1], request_base_url=REQUEST_BASE)


def _pair(db, code, install_id=INSTALL_ID, name=PHONE_NAME):
    from services.mobile.pairing import pair_device

    return pair_device(db, code, name, "ios", "1.0.0", install_id)


def _rows(db, user_pk):
    from core.models.mobile_devices import MobileDevice

    db.expire_all()
    return db.query(MobileDevice).filter(MobileDevice.user_id == user_pk).all()


def _column_values(row) -> list:
    return [getattr(row, column.name) for column in row.__table__.columns]


# --------------------------------------------------------------------------- the kernel is the only source


def test_the_alphabet_ttl_hashing_and_minting_are_the_kernels_only():
    from core.auth import pairing as kernel
    from services.mobile import pairing as service

    for name in ("new_pairing_code", "hash_secret", "mint_token", "normalize_pairing_code", "secrets_match"):
        assert getattr(service, name) is getattr(kernel, name)
    assert service.PAIRING_CODE_TTL_SECONDS == kernel.PAIRING_CODE_TTL_SECONDS == 600
    source = inspect.getsource(service)
    for copy in ("import secrets", "import hashlib", "import hmac", kernel.PAIRING_ALPHABET, "token_urlsafe"):
        assert copy not in source


def test_a_code_has_the_kernels_format_and_lives_600_seconds(db_session, owner, local):
    from core.auth.pairing import PAIRING_ALPHABET

    before = datetime.now(timezone.utc)
    issued = _mint(db_session, owner)

    letters = re.escape(PAIRING_ALPHABET)
    assert re.fullmatch(rf"[{letters}]{{4}}-[{letters}]{{4}}", issued.code)
    assert timedelta(seconds=600) <= issued.expires_at - before < timedelta(seconds=601)
    (row,) = _rows(db_session, owner[1])
    assert (row.kind, row.status, row.install_id) == ("paired", "pending", None)
    assert row.workspace_id == owner[0]
    assert row.pairing_expires_at == issued.expires_at


def test_the_code_is_stored_only_as_its_hash(db_session, owner, local):
    from core.auth.pairing import hash_secret

    issued = _mint(db_session, owner)

    (row,) = _rows(db_session, owner[1])
    assert row.pairing_code_hash == hash_secret(issued.code)
    assert not any(issued.code in str(value) for value in _column_values(row))
    assert issued.code not in str(row.to_dict())


# --------------------------------------------------------------------------- the exchange


def test_a_valid_code_pairs_once_and_the_token_is_stored_only_as_its_hash(db_session, owner, local):
    from core.auth.pairing import hash_secret

    issued = _mint(db_session, owner)
    paired = _pair(db_session, issued.code)

    assert paired.token.startswith("amd_") and len(paired.token) > 40
    (row,) = _rows(db_session, owner[1])
    assert paired.device_id == row.id and paired.workspace_id == owner[0]
    assert row.status == "active" and row.token_hash == hash_secret(paired.token)
    assert (row.install_id, row.name, row.platform, row.app_version) == (INSTALL_ID, PHONE_NAME, "ios", "1.0.0")
    assert row.pairing_code_hash is None and row.pairing_expires_at is None
    assert row.paired_at is not None and row.last_seen_at == row.paired_at
    assert [event["event"] for event in row.events] == ["enrolled"]
    assert not any(paired.token in str(value) for value in _column_values(row))
    assert paired.token not in str(row.to_dict())


def test_a_used_code_never_pairs_again(db_session, owner, local):
    issued = _mint(db_session, owner)
    assert _pair(db_session, issued.code) is not None

    assert _pair(db_session, issued.code, install_id="prd266-install-0002") is None
    assert len(_rows(db_session, owner[1])) == 1


def test_an_expired_code_is_none_and_changes_nothing(db_session, owner, local):
    issued = _mint(db_session, owner)
    (row,) = _rows(db_session, owner[1])
    row.pairing_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    db_session.flush()

    assert _pair(db_session, issued.code) is None
    (row,) = _rows(db_session, owner[1])
    assert row.status == "pending" and row.token_hash is None and row.pairing_code_hash is not None


def test_a_wrong_or_empty_code_is_none_and_never_spends_the_pending_code(db_session, owner, local):
    issued = _mint(db_session, owner)

    for wrong in ("ABCD-EFGH", "", "   ", issued.code[:-1] + ("2" if issued.code[-1] != "2" else "3")):
        assert _pair(db_session, wrong) is None
    (row,) = _rows(db_session, owner[1])
    assert row.status == "pending"

    assert _pair(db_session, f"  {issued.code.lower()} ") is not None


def test_rotate_after_is_the_pairing_plus_the_rotation_days(db_session, owner, local, monkeypatch):
    monkeypatch.setattr(local, "MOBILE_TOKEN_ROTATE_DAYS", 9)
    paired = _pair(db_session, _mint(db_session, owner).code)

    (row,) = _rows(db_session, owner[1])
    assert paired.rotate_after - row.paired_at == timedelta(days=9)


# --------------------------------------------------------------------------- the QR payload


def test_the_qr_payload_is_a_universal_link_on_the_web_app(db_session, owner, local):
    issued = _mint(db_session, owner)

    link = urlsplit(issued.qr_payload)
    assert issued.qr_payload.startswith(f"{WEB_URL}/go/pair?")
    assert (link.scheme, link.netloc, link.path) == ("https", urlsplit(WEB_URL).netloc, "/go/pair")
    query = parse_qs(link.query)
    assert query == {"api": [REQUEST_BASE.rstrip("/")], "code": [issued.code]}
    assert "api=http%3A%2F%2F192.0.2.10%3A8000" in issued.qr_payload


def test_a_local_install_names_its_own_api_host_once_the_operator_sets_one(db_session, owner, local, monkeypatch):
    monkeypatch.setattr(local, "PUBLIC_API_HOST", OPERATOR_HOST)

    issued = _mint(db_session, owner)

    assert parse_qs(urlsplit(issued.qr_payload).query)["api"] == [f"https://{OPERATOR_HOST}"]


@pytest.mark.parametrize("frontend_url", ["automatos://pair", "studio.prd266.test", ""])
def test_a_web_address_that_is_not_http_refuses_and_leaves_no_row(db_session, owner, local, monkeypatch, frontend_url):
    from services.mobile.urls import MobileUrlError

    monkeypatch.setattr(local, "FRONTEND_URL", frontend_url)

    with pytest.raises(MobileUrlError, match="FRONTEND_URL"):
        _mint(db_session, owner)
    assert _rows(db_session, owner[1]) == []


# --------------------------------------------------------------------------- one phone, one row


def test_a_phone_that_pairs_again_keeps_its_one_row(db_session, owner, local):
    from core.auth.pairing import hash_secret

    first = _pair(db_session, _mint(db_session, owner).code)
    second = _pair(db_session, _mint(db_session, owner).code, name=None)

    (row,) = _rows(db_session, owner[1])
    assert second.device_id == first.device_id == row.id
    assert row.token_hash == hash_secret(second.token) != hash_secret(first.token)
    assert row.prev_token_hash is None and row.name == PHONE_NAME
    assert [event["event"] for event in row.events] == ["enrolled", "enrolled"]


def test_a_revoked_phone_paired_again_is_active_on_the_same_row(db_session, owner, local):
    first = _pair(db_session, _mint(db_session, owner).code)
    (row,) = _rows(db_session, owner[1])
    row.status, row.revoked_at, row.revoked_by = "revoked", datetime.now(timezone.utc), owner[1]
    db_session.flush()

    second = _pair(db_session, _mint(db_session, owner).code)

    (row,) = _rows(db_session, owner[1])
    assert second.device_id == first.device_id
    assert (row.status, row.revoked_at, row.revoked_by) == ("active", None, None)


def test_another_phone_of_the_same_person_gets_its_own_row(db_session, owner, local):
    _pair(db_session, _mint(db_session, owner).code)
    _pair(db_session, _mint(db_session, owner).code, install_id="prd266-install-0002")

    assert sorted(row.install_id for row in _rows(db_session, owner[1])) == [INSTALL_ID, "prd266-install-0002"]


def test_the_audit_keeps_the_newest_50_events():
    from services.mobile.events import MAX_DEVICE_EVENTS, with_event

    at = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)
    old = [{"event": f"e{i}", "at": at.isoformat()} for i in range(MAX_DEVICE_EVENTS)]

    events = with_event(old, "enrolled", at)

    assert MAX_DEVICE_EVENTS == 50 and len(events) == 50
    assert events[0]["event"] == "e1" and events[-1] == {"event": "enrolled", "at": at.isoformat()}
    assert len(old) == 50 and old[0]["event"] == "e0"  # the stored list is never mutated


# --------------------------------------------------------------------------- saas never pairs (§8)


def test_saas_neither_mints_nor_pairs_and_writes_no_paired_row(db_session, owner, local, monkeypatch):
    from services.mobile.pairing import PairingUnavailable

    issued = _mint(db_session, owner)
    monkeypatch.setattr(local, "AUTH_EDITION", "saas")

    with pytest.raises(PairingUnavailable):
        _mint(db_session, owner)
    with pytest.raises(PairingUnavailable):
        _pair(db_session, issued.code)
    (row,) = _rows(db_session, owner[1])
    assert row.status == "pending"


def test_minting_drops_the_persons_expired_codes_and_keeps_live_ones(db_session, owner, local, seed_workspace):
    expired = _mint(db_session, owner)
    live = _mint(db_session, owner)
    stranger = (uuid.UUID(seed_workspace()), _seed_user(db_session))
    theirs = _mint(db_session, stranger)
    for row in _rows(db_session, owner[1]) + _rows(db_session, stranger[1]):
        if row.pairing_expires_at == expired.expires_at or row.pairing_expires_at == theirs.expires_at:
            row.pairing_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    db_session.flush()

    newest = _mint(db_session, owner)

    remaining = {row.pairing_expires_at for row in _rows(db_session, owner[1])}
    assert remaining == {live.expires_at, newest.expires_at}
    assert len(_rows(db_session, stranger[1])) == 1  # another person's expired code is theirs to clear
