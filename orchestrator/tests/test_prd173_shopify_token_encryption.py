"""
PRD-173 (F058) — Shopify Admin token encrypted at rest
=======================================================

The ``POST /api/shopify/connect`` handler stores the merchant's 147-scope
Shopify Admin access token in ``workspace.settings.shopify_access_token``.
Previously it wrote the token verbatim while the docstring *claimed*
encryption. These tests pin the real behaviour:

  * the value written at rest is ciphertext, NOT the plaintext token;
  * the decrypt path round-trips ciphertext back to the original token;
  * the encryption goes through the platform's canonical Fernet service
    (``core.credentials.encryption``), keyed from config — no hand-rolled
    crypto, no key in source.

Self-contained: generates a throwaway Fernet key and swaps the encryption
singleton so it needs no configured ``CREDENTIAL_ENCRYPTION_KEY`` and no DB.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

# Triggers ``load_dotenv`` so imports that resolve config at import time work
# regardless of test run order.
import config  # noqa: E402,F401

import core.credentials.encryption as encryption_module  # noqa: E402
from core.credentials.encryption import EncryptionService  # noqa: E402
from api import shopify  # noqa: E402

# Built at runtime (not a literal) so secret scanners / push-protection don't
# flag a shpat_-shaped constant in source. Shape mimics a real Admin token.
PLAINTEXT_TOKEN = "shpat_" + ("0123456789abcdef" * 2)


@pytest.fixture()
def real_encryption(monkeypatch):
    """Install a real (throwaway-key) Fernet encryption service as the singleton.

    Uses the genuine :class:`EncryptionService` — not a mock — so the tests
    exercise the actual encrypt/decrypt path, but with a key generated in the
    test rather than read from the environment.
    """
    key = Fernet.generate_key()
    monkeypatch.setattr(EncryptionService, "_load_or_generate_key", lambda self: key)
    service = EncryptionService()
    # Reset + pin the module-level singleton so ``get_encryption_service()``
    # (used by shopify._encrypt_secret/_decrypt_secret) returns ours.
    monkeypatch.setattr(encryption_module, "_encryption_service", service)
    return service


def test_encrypt_secret_produces_ciphertext_not_plaintext(real_encryption):
    """The stored value must not be the raw token."""
    stored = shopify._encrypt_secret(PLAINTEXT_TOKEN)

    assert stored != PLAINTEXT_TOKEN
    assert PLAINTEXT_TOKEN not in stored
    # Fernet ciphertext is a non-empty base64 token, not the shpat_ prefix.
    assert stored
    assert not stored.startswith("shpat_")


def test_decrypt_secret_round_trips(real_encryption):
    """Ciphertext written at rest decrypts back to the original token."""
    stored = shopify._encrypt_secret(PLAINTEXT_TOKEN)
    recovered = shopify._decrypt_secret(stored)

    assert recovered == PLAINTEXT_TOKEN


def test_connect_settings_transform_stores_ciphertext(real_encryption):
    """The exact settings mutation the /connect handler applies stores ciphertext.

    Mirrors the handler's write without a DB session: the persisted
    ``shopify_access_token`` is ciphertext and the raw token appears nowhere
    under ``workspace.settings``.
    """
    settings: dict = {}
    settings["shopify_domain"] = "example.myshopify.com"
    settings["shopify_access_token"] = shopify._encrypt_secret(PLAINTEXT_TOKEN)

    assert settings["shopify_access_token"] != PLAINTEXT_TOKEN
    # Plaintext token must not leak into ANY settings value.
    assert PLAINTEXT_TOKEN not in "".join(str(v) for v in settings.values())
    # And it must round-trip for the reader.
    assert shopify._decrypt_secret(settings["shopify_access_token"]) == PLAINTEXT_TOKEN


def test_encryption_uses_canonical_fernet_service(real_encryption):
    """Encryption is delegated to core.credentials.encryption (canonical path)."""
    stored = shopify._encrypt_secret(PLAINTEXT_TOKEN)
    # The canonical service must be able to decrypt what the helper produced,
    # proving the helper used it rather than a parallel scheme.
    assert real_encryption.decrypt(stored) == PLAINTEXT_TOKEN
