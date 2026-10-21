"""PRD-261 C-05c/C-05h seam: the one reader of the signing keys the hosted feed revoked.

The local edition's pull (``revocation_pull``, C-05h) stores the hosted feed's ``revokedKeys`` in
``theme_revocation_sync.revoked_key_ids`` (row ``THEME_REVOCATION_SYNC_ROW_ID``). Everything that
asks which keys the feed revoked reads them here: the signature check on load
(``signature_state``), the upload check and the installed list's signature view (``signed_upload``)
and the admin's sync status (``revocation_pull.sync_status``).

* **Local edition only.** The row is the local edition's own: the hosted edition publishes the
  feed and never pulls it (its revoked keys are its keys file's), so there no key reads as revoked
  by the feed and the row is never read.
* **Validated.** Only key ids (strings) count. A stored value that isn't a list reads as no key;
  a list holding entries that aren't strings keeps its strings. Either is logged as an error once
  per malformed value, not on every read (the files route asks on every file), and again only
  after a well-formed value was read in between.
* **A frozenset**, so a caller can test membership and fingerprint it; ``sorted`` gives the list
  a JSON answer carries.

A leaf: it imports only the models, ``config`` and the standard library, so the revocation rule
(``revocations`` -> ``signature_state``) never reaches the pull or the feed, which would close an
import cycle (``revocation_feed`` imports ``revocations``).
"""

from __future__ import annotations

import hashlib
import logging
import threading
from typing import Any, FrozenSet, Optional

from config import config
from core.models.themes import THEME_REVOCATION_SYNC_ROW_ID, ThemeRevocationSync

logger = logging.getLogger(__name__)

NO_KEYS: FrozenSet[str] = frozenset()
NOT_A_LIST = "theme_revocation_sync.revoked_key_ids is not a list (%s): no key reads as revoked by the feed"
NOT_KEY_IDS = "theme_revocation_sync.revoked_key_ids holds entries that aren't key ids: only its key ids count"


class _MalformedLog:
    """Remembers the malformed stored value it logged last, so one value is logged once."""

    def __init__(self) -> None:
        self._last: Optional[str] = None
        self._lock = threading.Lock()

    def first_time(self, stored: Any) -> bool:
        """True (and remembered) when ``stored`` isn't the malformed value logged last."""
        mark = hashlib.sha256(repr(stored).encode("utf-8")).hexdigest()
        with self._lock:
            if mark == self._last:
                return False
            self._last = mark
            return True

    def forget(self) -> None:
        """A well-formed value was read: the next malformed one is logged again."""
        with self._lock:
            self._last = None


_malformed = _MalformedLog()


def _log_once(stored: Any, message: str, *args: Any) -> None:
    if _malformed.first_time(stored):
        logger.error(message, *args)


def revoked_key_ids(db: Any) -> FrozenSet[str]:
    """The key ids the hosted feed lists as revoked, as the local edition's pull stored them; none in SaaS.

    Malformed stored data is logged once (module docstring); its string entries still count.
    """
    if not config.IS_LOCAL_EDITION:
        return NO_KEYS
    row = db.get(ThemeRevocationSync, THEME_REVOCATION_SYNC_ROW_ID)
    stored = None if row is None else row.revoked_key_ids
    if stored is None:
        return NO_KEYS
    if not isinstance(stored, list):
        _log_once(stored, NOT_A_LIST, type(stored).__name__)
        return NO_KEYS
    if all(isinstance(key_id, str) for key_id in stored):
        _malformed.forget()
    else:
        _log_once(stored, NOT_KEY_IDS)
    return frozenset(key_id for key_id in stored if isinstance(key_id, str))


__all__ = ["NO_KEYS", "revoked_key_ids"]
