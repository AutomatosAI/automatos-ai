"""A product's key, and what a product's record holds back (PRD-274 Wave 5a, §5.1).

- **The key** (``mint_key``): a high-entropy secret shown once, when an admin creates the product or replaces
  its key. Only its PBKDF2-HMAC-SHA256 hash is stored, with a fixed salt of its own, so a key is found by its
  hash and a product key's hash never equals an SDK key's (``core/services/api_key_service``). 5b checks it.
- **The owner workspace can't be purged while it owns a product** (``refused_while_it_owns_a_product``, on the
  purge's first data step): nothing is erased, the workspace stays soft-deleted, and the purge answers
  ``owns_a_product``. An admin deletes the product first. ``products.owner_workspace_id``'s foreign key is
  the backstop.
- **A product isn't deleted while it has workspaces** (``workspaces_of``): every workspace carrying its
  ``settings.product_id``, a closed one inside its grace included (decision 25), would be left with no product.
"""
from __future__ import annotations

import functools
import hashlib
import logging
import secrets
from typing import Any, Callable, NamedTuple

from sqlalchemy import text

logger = logging.getLogger(__name__)

KEY_PREFIX = "ak_prod_"
KEY_PREFIX_SHOWN = len(KEY_PREFIX) + 4  # e.g. "ak_prod_a1b2": enough to tell keys apart, never enough to guess one
_KEY_SALT = b"automatos/product-key/v1"
_KEY_ITERATIONS = 100_000
OWNS_A_PRODUCT = "owns_a_product"  # services/workspace_purge: PurgeResult.skipped_reason

_OWNS = text("SELECT 1 FROM products WHERE owner_workspace_id = CAST(:ws AS uuid) LIMIT 1")
_WORKSPACES = text("SELECT count(*) FROM workspaces WHERE settings->>'product_id' = :product")


class MintedKey(NamedTuple):
    """A new key: the plaintext (returned once), its hash and its first characters (both stored)."""

    plaintext: str
    key_hash: str
    key_prefix: str


def key_hash(plaintext: str) -> str:
    """The key's PBKDF2-HMAC-SHA256 hash, hex: how it is stored and looked up."""
    return hashlib.pbkdf2_hmac("sha256", plaintext.encode(), _KEY_SALT, _KEY_ITERATIONS).hex()


def mint_key() -> MintedKey:
    """A new product key."""
    plaintext = KEY_PREFIX + secrets.token_hex(32)
    return MintedKey(plaintext, key_hash(plaintext), plaintext[:KEY_PREFIX_SHOWN])


def owns_a_product(db: Any, workspace_id: Any) -> bool:
    """The workspace is some product's owner workspace."""
    return db.execute(_OWNS, {"ws": str(workspace_id)}).first() is not None


def workspaces_of(db: Any, product_id: Any) -> int:
    """How many workspaces carry the product's ``settings.product_id``, closed ones included."""
    return int(db.execute(_WORKSPACES, {"product": str(product_id)}).scalar() or 0)


def refused_while_it_owns_a_product(first_data_step: Callable[..., bool]) -> Callable[..., bool]:
    """Wraps the purge's first data step: an owner workspace of a product is not purged, and nothing of it
    is erased (``OWNS_A_PRODUCT``)."""

    @functools.wraps(first_data_step)
    def checked_then_step(workspace_id: Any, result: Any) -> bool:
        from core.database.database import SessionLocal

        db = SessionLocal()
        try:
            owner = owns_a_product(db, workspace_id)
        finally:
            db.close()
        if owner:
            logger.error("Purge refused: workspace %s owns a product; delete the product first", workspace_id)
            result.skipped_reason = OWNS_A_PRODUCT
            return False
        return first_data_step(workspace_id, result)

    return checked_then_step
