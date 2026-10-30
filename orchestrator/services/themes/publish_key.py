"""PRD-261 — the theme publish key: who may call ``POST /api/marketplace/themes/publish``.

The Design Kit's publish job (``theme-publish.yml``, GitHub environment ``theme-publish``) sends
``Authorization: Bearer <THEME_PUBLISH_KEY>``, 32 random bytes as hex. Core never holds the key.
Three settings (``config.py``), all needed:

* ``THEME_PUBLISH_KEY_SHA256``: the lower-case hex sha256 of the key string. Two, comma-separated,
  while a rotation runs (add the new hash, swap the secret, drop the old hash); more than two, or
  one that isn't 64 hex characters, reads as unset. Revoking the key is dropping its hash.
* ``THEME_PUBLISH_PUBLISHERS``: the publishers the key may publish under, comma-separated.
* ``THEME_PUBLISH_ACTOR``: the ``users.id`` the key acts as, a service user
  (``scripts/theme_publish_actor.py``), never a person's row.

Any of them unset or unreadable: ``publish_settings`` is None and the route answers 503
``PUBLISH_NOT_CONFIGURED``, so the endpoint is never open without them. The presented key's
sha256 is compared with ``hmac.compare_digest`` to every configured hash, with no early exit.

This is the route's own credential. ``get_request_context_hybrid`` has no lane for it: a bearer
that is neither a Clerk JWT nor a registered prefix is 401 there, and an ``X-API-Key`` is matched
only to ``ORCHESTRATOR_API_KEY``. Nothing here logs the key, its hash or the configured hashes; a
setting that can't be read is logged by its name only.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import re
from dataclasses import dataclass
from typing import FrozenSet, Optional, Sequence, Tuple

from config import config

logger = logging.getLogger(__name__)

KEY_HASHES_SETTING = "THEME_PUBLISH_KEY_SHA256"
PUBLISHERS_SETTING = "THEME_PUBLISH_PUBLISHERS"
ACTOR_SETTING = "THEME_PUBLISH_ACTOR"
MAX_KEY_HASHES = 2  # the live key and, while a rotation runs, the next one
LIST_SEPARATOR = ","
KEY_ENCODING = "utf-8"
SHA256_HEX = re.compile(r"[0-9a-f]{64}")
POSITIVE_ID = re.compile(r"[1-9][0-9]{0,18}")
UNREADABLE = "[Themes] %s can't be read: the theme publish endpoint is off until it is fixed"


@dataclass(frozen=True)
class PublishSettings:
    """What the publish key may do: its accepted hashes, its publishers and the user it acts as."""

    key_hashes: Tuple[str, ...]
    publishers: FrozenSet[str]
    actor_pk: int


def _listed(raw: object) -> Tuple[str, ...]:
    """A comma-separated setting as trimmed, lower-case, distinct entries, in order."""
    parts = (part.strip().lower() for part in str(raw or "").split(LIST_SEPARATOR))
    return tuple(dict.fromkeys(part for part in parts if part))


def _key_hashes() -> Optional[Tuple[str, ...]]:
    hashes = _listed(getattr(config, KEY_HASHES_SETTING, ""))
    if not hashes:
        return None
    if len(hashes) > MAX_KEY_HASHES or not all(SHA256_HEX.fullmatch(value) for value in hashes):
        logger.warning(UNREADABLE, KEY_HASHES_SETTING)
        return None
    return hashes


def _actor_pk() -> Optional[int]:
    raw = str(getattr(config, ACTOR_SETTING, "") or "").strip()
    if not raw:
        return None
    if POSITIVE_ID.fullmatch(raw) is None:
        logger.warning(UNREADABLE, ACTOR_SETTING)
        return None
    return int(raw)


def publish_settings() -> Optional[PublishSettings]:
    """The publish key's settings, or None when any of the three is unset or can't be read (the 503)."""
    hashes = _key_hashes()
    publishers = frozenset(_listed(getattr(config, PUBLISHERS_SETTING, "")))
    actor_pk = _actor_pk()
    if hashes is None or not publishers or actor_pk is None:
        return None
    return PublishSettings(key_hashes=hashes, publishers=publishers, actor_pk=actor_pk)


def key_sha256(key: str) -> str:
    """The lower-case hex sha256 of a key string, as ``THEME_PUBLISH_KEY_SHA256`` holds it."""
    return hashlib.sha256(key.encode(KEY_ENCODING)).hexdigest()


def key_matches(presented: str, hashes: Sequence[str]) -> bool:
    """True when the presented key's sha256 is one of ``hashes``; every hash is compared, in constant time."""
    digest = key_sha256(presented)
    matches = [hmac.compare_digest(digest, expected) for expected in hashes]
    return any(matches)


__all__ = [
    "ACTOR_SETTING", "KEY_HASHES_SETTING", "MAX_KEY_HASHES", "PUBLISHERS_SETTING", "PublishSettings",
    "key_matches", "key_sha256", "publish_settings",
]
