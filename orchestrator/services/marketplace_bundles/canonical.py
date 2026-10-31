"""PRD-270 W2 (K8) — an item bundle's canonical bytes and its revision.

Both sides write a bundle the same way: sorted keys, no spaces between tokens, UTF-8 (no ASCII
escapes). The hosted edition serves exactly these bytes, and a bundle's revision is their sha256,
so a local install that hashes what it fetched knows it got the bundle the catalogue listed. Key
order never changes the revision; any change of content does. A local Cloud row's version carries
both: Automatos Cloud's version and the start of the revision (``local_version``, K10).
"""
from __future__ import annotations

import json
import re
from typing import Any, Mapping

from services.marketplace_bundles.errors import BundleError
from services.marketplace_bundles.schema import LOCAL_REVISION_LENGTH, REVISION_PATTERN, VERSION_PATTERN
from services.themes.digest import sha256_hex

_VERSION = re.compile(VERSION_PATTERN)
_REVISION = re.compile(REVISION_PATTERN)


def canonical_bytes(bundle: Mapping[str, Any]) -> bytes:
    """The bundle as canonical JSON bytes: sorted keys, compact separators, UTF-8.

    Run on a bundle validate_bundle passed. On anything else, a value JSON cannot carry (NaN, a
    lone surrogate, a cycle or a nesting too deep) raises BundleError 'type', never another error.
    """
    try:
        text = json.dumps(bundle, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        return text.encode("utf-8")
    except (TypeError, ValueError, RecursionError) as error:
        raise BundleError("type", "The bundle holds a value JSON cannot carry.") from error


def revision_of(bundle: Mapping[str, Any]) -> str:
    """The bundle's revision: the sha256 of its canonical bytes, as 64 lower-case hex characters."""
    return sha256_hex(canonical_bytes(bundle))


def capped_bytes(bundle: Mapping[str, Any], max_bytes: int) -> bytes:
    """The canonical bytes, or BundleError 'too-large' when they are over ``max_bytes``."""
    data = canonical_bytes(bundle)
    if len(data) > max_bytes:
        raise BundleError("too-large", f"The bundle is over {max_bytes} bytes.")
    return data


def local_version(version: str, revision: str) -> str:
    """The version a local Cloud row carries: ``version``, '+', the revision's first 12 hex
    characters (at most schema.LOCAL_VERSION_MAX, 45). Every clone copies its source's version, so a
    workspace copy records exactly which bundle it was made from (K10).

    Raises BundleError 'pattern' for a version off the format's pattern or a revision that is not
    64 lower-case hex characters.
    """
    if not isinstance(version, str) or _VERSION.fullmatch(version) is None:
        raise BundleError("pattern", "The version is not in the form the bundle format allows.")
    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise BundleError("pattern", "The revision is not a sha256 in lower-case hex.")
    return f"{version}+{revision[:LOCAL_REVISION_LENGTH]}"
