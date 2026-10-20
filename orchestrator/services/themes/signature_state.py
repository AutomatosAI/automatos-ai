"""PRD-261 C-05c — a stored signature that no longer checks out disables its version, on load.

QB-28 (answered 20 Oct, D8: disable, fail closed): a version whose stored signature fails, or was
made by a key that has been revoked since, stops running until it is re-signed.
``revocations.revocation_of`` asks ``signature_fails`` about every package it is handed and, when
it is true, answers a **transient** ``disable`` revocation (built there, never added to a session,
never stored) whose reason is ``SIGNATURE_DISABLE_REASON``. So every reader of the one revocation
rule treats the version as withdrawn, with no write: the files route answers 404, the boot route
the base boot, ``active`` the theme's base with the notice, the listings hide the version, and no
install, update, rollback or default moves to it. A new signature (``review.sign`` takes one for a
version whose stored signature no longer verifies) lifts it on the next load.

A stored signature **fails** when:

* no key of the keys file (``signature_verify.KEYS_PATH``) verifies it over the payload the
  version's file list gives: another key's signature, bytes that aren't a signature, or a file
  list that no longer adds up to the digest (``signature_verify.package_verdict`` answers
  ``invalid`` with no key id);
* the key that verifies it is ``revoked`` in the keys file; or
* the key that verifies it is in ``theme_revocation_sync.revoked_key_ids``: the keys the hosted
  revocation feed lists as revoked (``revokedKeys``), as the local edition's pull stores them
  (C-05h). That row is the local edition's own (the pull never runs in SaaS, where the revoked
  keys are the keys file's), so only the local edition reads it.

Not a failure: no signature (an unsigned package reads nothing here and is untouched); a
``retired`` key's signature (still that key's, so the version keeps running; it is no longer
Official, ``listings``); and a keys file that can't be read or parsed (logged as an error: a
broken release file says nothing about one package, and the listings then show every signature
``invalid``).

The answer is cached per (package id, digest, signature, key-set fingerprint), the fingerprint
being the sha256 of the keys file's bytes and the sorted revoked key ids, so the files route, which
asks on every file, verifies once per package per key set, and a changed keys file or revoked-key
list recomputes. The keys file is still read on each check (as ``signature_verify`` reads it), so
a test can point ``KEYS_PATH`` at its own file.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path
from typing import Any, Dict, FrozenSet, Hashable, NamedTuple, Optional, Sequence, Tuple

from config import config
from core.models.themes import THEME_REVOCATION_SYNC_ROW_ID, ThemePackage, ThemeRevocationSync
from services.themes import signature_verify

logger = logging.getLogger(__name__)

SIGNATURE_DISABLE_REASON = "This theme's signature doesn't check out."
# How many (package, signature, key set) answers are kept; the oldest goes first.
SIGNATURE_CACHE_SIZE = 1024
# The fingerprint part of a keys file that can't be read.
UNREADABLE_KEYS = "unreadable"
KEYS_ENCODING = "utf-8"


class KeySet(NamedTuple):
    """What a check is made against: the keys file's bytes (None when unreadable) and the revoked key ids."""

    keys_bytes: Optional[bytes]
    revoked: FrozenSet[str]

    @property
    def fingerprint(self) -> Tuple[str, Tuple[str, ...]]:
        """The keys file's sha256 (or ``UNREADABLE_KEYS``) and the sorted revoked key ids."""
        keys = UNREADABLE_KEYS if self.keys_bytes is None else hashlib.sha256(self.keys_bytes).hexdigest()
        return keys, tuple(sorted(self.revoked))


class _AnswerCache:
    """A small bounded map of ``signature_fails`` answers, safe across threadpool workers."""

    def __init__(self, size: int) -> None:
        self._size = size
        self._items: Dict[Hashable, bool] = {}
        self._lock = threading.Lock()

    def get(self, key: Hashable) -> Optional[bool]:
        with self._lock:
            return self._items.get(key)

    def put(self, key: Hashable, fails: bool) -> None:
        with self._lock:
            if key not in self._items and len(self._items) >= self._size:
                self._items.pop(next(iter(self._items)))
            self._items[key] = fails

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


_answers = _AnswerCache(SIGNATURE_CACHE_SIZE)


def clear_cache() -> None:
    """Forget every cached answer (tests; a process restart does the same)."""
    _answers.clear()


def revoked_key_ids(db: Any) -> FrozenSet[str]:
    """The key ids the hosted feed lists as revoked, as the local edition's pull stored them; none in SaaS.

    A stored value that isn't a list of key ids is logged; its string entries still count.
    """
    if not config.IS_LOCAL_EDITION:
        return frozenset()
    row = db.get(ThemeRevocationSync, THEME_REVOCATION_SYNC_ROW_ID)
    stored = None if row is None else row.revoked_key_ids
    if stored is None:
        return frozenset()
    if not isinstance(stored, list):
        logger.error("theme_revocation_sync.revoked_key_ids is not a list (%s): no key reads as revoked by the feed",
                     type(stored).__name__)
        return frozenset()
    if any(not isinstance(key_id, str) for key_id in stored):
        logger.error("theme_revocation_sync.revoked_key_ids holds entries that aren't key ids: only its key ids count")
    return frozenset(key_id for key_id in stored if isinstance(key_id, str))


def _keys_bytes() -> Optional[bytes]:
    """The keys file's bytes, or None when it can't be read (the check that needs them logs it)."""
    try:
        return Path(signature_verify.KEYS_PATH).read_bytes()
    except OSError:
        return None


def current_key_set(db: Any) -> KeySet:
    """The keys file as it is now, and the feed's revoked key ids."""
    return KeySet(keys_bytes=_keys_bytes(), revoked=revoked_key_ids(db))


def _parsed_keys(package: ThemePackage, key_set: KeySet) -> Optional[Tuple[signature_verify.SigningKey, ...]]:
    """The keys of the key set, or None (logged) when its file can't be read or parsed."""
    if key_set.keys_bytes is None:
        logger.error("The theme signing keys file can't be read: theme package %s's signature isn't checked on load",
                     package.id)
        return None
    try:
        return signature_verify.parse_keys(key_set.keys_bytes.decode(KEYS_ENCODING))
    except UnicodeDecodeError:
        logger.error("The theme signing keys file isn't UTF-8: theme package %s's signature isn't checked on load",
                     package.id)
    except signature_verify.SigningKeysError as bad:
        logger.error("The theme signing keys file doesn't load (%s): theme package %s's signature isn't checked "
                     "on load", bad, package.id)
    return None


def _status_of(keys: Sequence[signature_verify.SigningKey], key_id: str) -> Optional[str]:
    return next((key.status for key in keys if key.key_id == key_id), None)


def _check(package: ThemePackage, key_set: KeySet) -> bool:
    """The uncached answer: True when the stored signature fails or its key is revoked."""
    keys = _parsed_keys(package, key_set)
    if keys is None:
        return False
    verdict = signature_verify.package_verdict(package, keys)
    if verdict.state == signature_verify.STATE_UNSIGNED:
        return False
    if verdict.key_id is None:
        return True
    return verdict.key_id in key_set.revoked or _status_of(keys, verdict.key_id) == signature_verify.STATUS_REVOKED


def signature_fails(db: Any, package: ThemePackage) -> bool:
    """True when the version's stored signature doesn't check out (module docstring); False when unsigned.

    ``revocation_of`` takes any package-shaped object; one with no ``signature`` reads as unsigned.
    """
    signature = getattr(package, "signature", None)
    if not signature:
        return False
    key_set = current_key_set(db)
    key = (package.id, package.digest, signature, key_set.fingerprint)
    cached = _answers.get(key)
    if cached is not None:
        return cached
    fails = _check(package, key_set)
    if fails:
        logger.warning("Theme package %s (%s %s): its stored signature doesn't check out; disabled until re-signed",
                       package.id, package.theme_id, package.version)
    _answers.put(key, fails)
    return fails


__all__ = [
    "KeySet", "SIGNATURE_CACHE_SIZE", "SIGNATURE_DISABLE_REASON", "clear_cache", "current_key_set", "revoked_key_ids",
    "signature_fails",
]
