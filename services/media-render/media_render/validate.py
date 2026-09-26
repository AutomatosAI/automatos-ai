"""Field validators for the service's request bodies (the render bundle, /tts).

Every validator names the field it rejects (``where``), so a refused request
says exactly what to fix. A refusal is a ``BundleError``: the server answers it
with HTTP 400 before any file is fetched or any process is started.
"""

from __future__ import annotations

import base64
import binascii
import math
import re
from typing import AbstractSet, Any, Iterable, Mapping, Optional

LINE_ID = re.compile(r"^[a-z0-9_-]{1,32}$")
VOICE_NAME = re.compile(r"^[a-z]{2}_[a-z0-9]{1,32}$")
LANGUAGE = re.compile(r"^[a-z]{2,3}(?:-[a-z]{2,4})?$")
_PATH = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9._-]*(?:/[A-Za-z0-9_-][A-Za-z0-9._-]*)*$")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MAX_PATH_CHARS = 200
SPEED_RANGE = (0.5, 2.0)


class BundleError(ValueError):
    """The request cannot be rendered as sent (HTTP 400)."""


def mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise BundleError(f"{where} must be an object")
    return value


def items(value: Any, where: str, *, limit: int) -> list:
    if not isinstance(value, list):
        raise BundleError(f"{where} must be a list")
    if len(value) > limit:
        raise BundleError(f"{where} holds {len(value)} entries; the limit is {limit}")
    return value


def keys(value: Mapping[str, Any], where: str, *, required: Iterable[str] = (), optional: Iterable[str] = ()) -> None:
    required, optional = set(required), set(optional)
    missing = sorted(required - set(value))
    if missing:
        raise BundleError(f"{where} is missing {', '.join(missing)}")
    unknown = sorted(set(value) - required - optional)
    if unknown:
        raise BundleError(f"{where} has unknown field(s) {', '.join(unknown)}")


def text(value: Any, where: str, *, max_chars: int, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise BundleError(f"{where} must be text")
    if not allow_empty and not value.strip():
        raise BundleError(f"{where} is empty")
    if len(value) > max_chars:
        raise BundleError(f"{where} is {len(value)} characters; the limit is {max_chars}")
    if _CONTROL.search(value):
        raise BundleError(f"{where} contains control characters")
    return value


def number(
    value: Any,
    where: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    below: Optional[float] = None,
) -> float:
    """A finite number within [minimum, maximum], and strictly under ``below`` when given."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise BundleError(f"{where} must be a number")
    result = float(value)
    if minimum is not None and result < minimum:
        raise BundleError(f"{where} is {result:g}; it must be at least {minimum:g}")
    if maximum is not None and result > maximum:
        raise BundleError(f"{where} is {result:g}; it must be at most {maximum:g}")
    if below is not None and result >= below:
        raise BundleError(f"{where} is {result:g}; it must be under {below:g}")
    return result


def pattern(value: Any, where: str, regex: re.Pattern, expected: str) -> str:
    if not isinstance(value, str) or not regex.match(value):
        raise BundleError(f"{where} must be {expected}")
    return value


def voice_name(value: Any, where: str) -> str:
    return pattern(value, where, VOICE_NAME, "a Kokoro voice name such as af_heart")


def speed(value: Any, where: str) -> float:
    return number(value, where, minimum=SPEED_RANGE[0], maximum=SPEED_RANGE[1])


def language(value: Any, where: str) -> str:
    return pattern(value, where, LANGUAGE, "a language code such as en-us")


def line_id(value: Any, where: str) -> str:
    return pattern(value, where, LINE_ID, "lowercase letters, digits, - or _ (at most 32)")


def asset_path(value: Any, where: str, *, root: str, extensions: AbstractSet[str]) -> str:
    """A relative path under ``root`` made of plain components, with an allowed extension.

    No dot segments, no hidden files (the Hyperframes CLI reads a .env from its
    working directory), nothing that could escape the composition directory.
    """
    if not isinstance(value, str) or len(value) > MAX_PATH_CHARS or not _PATH.match(value):
        raise BundleError(f"{where} must be a relative path of letters, digits, '.', '-' and '_' components")
    if not value.startswith(root) or value == root:
        raise BundleError(f"{where} must be a file under {root}")
    extension = value.rsplit(".", 1)[-1].lower() if "." in value.rsplit("/", 1)[-1] else ""
    if extension not in extensions:
        raise BundleError(f"{where} must end in one of: {', '.join(sorted(extensions))}")
    return value


def data_uri(value: Any, where: str, *, mimes: AbstractSet[str], max_bytes: int) -> bytes:
    """Decode a base64 ``data:`` URI whose media type is one of ``mimes``."""
    if not isinstance(value, str) or not value.startswith("data:") or "," not in value:
        raise BundleError(f"{where} must be a base64 data: URI")
    header, _, payload = value[len("data:") :].partition(",")
    params = [part.strip().lower() for part in header.split(";")]
    if "base64" not in params[1:]:
        raise BundleError(f"{where} must be base64-encoded")
    if params[0] not in mimes:
        raise BundleError(f"{where} has media type {params[0] or 'none'}; expected one of {', '.join(sorted(mimes))}")
    compact = re.sub(r"\s+", "", payload)
    if len(compact) * 3 // 4 > max_bytes:
        raise BundleError(f"{where} is larger than the {max_bytes}-byte limit")
    try:
        data = base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError):
        raise BundleError(f"{where} is not valid base64") from None
    if not data:
        raise BundleError(f"{where} is empty")
    if len(data) > max_bytes:
        raise BundleError(f"{where} is larger than the {max_bytes}-byte limit")
    return data
