"""core.services.api_key_service hashing — CodeQL py/weak-sensitive-data-hashing #92.

A plain SHA-256 of an API key is too fast to brute-force at scale. New keys
are stored as PBKDF2-HMAC-SHA256; a key created before the switch is stored
under its legacy SHA-256 hash and must keep validating (its plaintext was
never stored, so it cannot be rehashed server-side).

Mocked Session (no live DB) — mirrors the style of test_board_sdk_auth.py.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
from uuid import uuid4

from core.services.api_key_service import (
    ApiKeyService,
    _hash_key,
    _legacy_hash_key,
    key_identity_digest,
)

RAW_KEY = "ak_pub_" + "a" * 64


def _record(key_hash: str, *, is_active: bool = True, expires_at=None):
    rec = MagicMock()
    rec.key_hash = key_hash
    rec.is_active = is_active
    rec.expires_at = expires_at
    rec.last_used_at = None
    return rec


def _db_returning(record):
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = record
    return db


class TestHashFunctions:
    def test_hash_key_is_not_plain_sha256(self):
        assert _hash_key(RAW_KEY) != hashlib.sha256(RAW_KEY.encode()).hexdigest()

    def test_hash_key_is_deterministic(self):
        assert _hash_key(RAW_KEY) == _hash_key(RAW_KEY)

    def test_hash_key_differs_from_legacy_hash_key(self):
        assert _hash_key(RAW_KEY) != _legacy_hash_key(RAW_KEY)

    def test_legacy_hash_key_is_plain_sha256(self):
        assert _legacy_hash_key(RAW_KEY) == hashlib.sha256(RAW_KEY.encode()).hexdigest()

    def test_key_identity_digest_matches_legacy_hash_key(self):
        """The rate limiter's bucket digest must stay the fast SHA-256 form:
        it is computed on every request and verifies nothing."""
        assert key_identity_digest(RAW_KEY) == _legacy_hash_key(RAW_KEY)

    def test_hash_key_output_fits_the_stored_column(self):
        # sdk_api_keys.key_hash is String(64); PBKDF2-SHA256's hex digest must
        # still fit it exactly like the SHA-256 hex digest did.
        assert len(_hash_key(RAW_KEY)) == 64


class TestValidateApiKeyAcceptsBothHashForms:
    def test_a_key_stored_under_the_new_pbkdf2_hash_validates(self):
        db = _db_returning(_record(_hash_key(RAW_KEY)))
        assert ApiKeyService.validate_api_key(db, RAW_KEY) is not None

    def test_a_key_stored_under_the_legacy_sha256_hash_still_validates(self):
        """Regression: before this fix, create_api_key wrote SHA-256 and
        validate_api_key looked up SHA-256 — the two matched. After switching
        create_api_key to PBKDF2 alone, a key created under the OLD algorithm
        must still be found, or every key issued before this change breaks."""
        db = _db_returning(_record(_legacy_hash_key(RAW_KEY)))
        assert ApiKeyService.validate_api_key(db, RAW_KEY) is not None

    def test_the_lookup_lists_both_candidate_hashes(self):
        db = _db_returning(None)
        ApiKeyService.validate_api_key(db, RAW_KEY)
        filter_call = db.query.return_value.filter.call_args
        in_clause = filter_call.args[0]
        # SQLAlchemy's in_() binds the candidate list as a single BindParameter.
        assert set(in_clause.right.value) == {_hash_key(RAW_KEY), _legacy_hash_key(RAW_KEY)}

    def test_an_unknown_key_in_neither_form_does_not_validate(self):
        db = _db_returning(None)
        assert ApiKeyService.validate_api_key(db, RAW_KEY) is None

    def test_an_expired_key_does_not_validate_even_if_the_hash_matches(self):
        past = datetime.now(timezone.utc) - timedelta(days=1)
        db = _db_returning(_record(_hash_key(RAW_KEY), expires_at=past))
        assert ApiKeyService.validate_api_key(db, RAW_KEY) is None


class TestCreateApiKeyWritesOnlyTheNewHash:
    def test_create_api_key_stores_the_pbkdf2_hash(self):
        db = MagicMock()
        workspace_id = uuid4()

        result = ApiKeyService.create_api_key(db, workspace_id, name="ci key")

        added = db.add.call_args.args[0]
        assert added.key_hash == _hash_key(result["key"])
        assert added.key_hash != _legacy_hash_key(result["key"])
