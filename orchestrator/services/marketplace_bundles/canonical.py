"""PRD-270 W2 (K8) — an item bundle's canonical bytes and its revision.

Both sides write a bundle the same way: sorted keys, no spaces between tokens, UTF-8 (no ASCII
escapes). The hosted edition serves exactly these bytes, and a bundle's revision is their sha256,
so a local install that hashes what it fetched knows it got the bundle the catalogue listed. Key
order never changes the revision; any change of content does.
"""
from __future__ import annotations

import json
from typing import Any, Mapping

from services.marketplace_bundles.errors import BundleError
from services.themes.digest import sha256_hex


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
