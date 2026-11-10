"""Partner tools: from a model's call to the one GET the server will send (PRD-275).

The model chooses an operation and its parameter values, nothing else. The address comes
from the workspace's credential and the path from the pack, so a call can only reach the
partner's own host, on a path the pack declares:

* arguments are checked against the operation's schema, and a name the operation does not
  declare is refused;
* a path value is one plain segment (``PATH_VALUE``), percent-encoded, so it cannot add a
  segment, a query or a host;
* the URL that results must have the base URL's origin, or nothing is sent;
* the key goes in a header the server writes, never in the URL.

Pure: no I/O. Every refusal is a :class:`PartnerRefusal` the caller turns into a result.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Optional, Tuple
from urllib.parse import quote, urlencode, urlsplit

from jsonschema import Draft202012Validator

from modules.tools.partner.pack import OPERATION_KEY, Operation, Tool, placeholders

REQUIRED_SCHEME = "https"
HTTPS_PORT = 443
# One path segment: an id, a number, a slug. Never a slash, a dot segment, a query or a space.
PATH_VALUE = re.compile(r"[A-Za-z0-9._~:@-]{1,200}")
# A key header is named for what it carries (X-API-Key, Api-Key, X-Auth-Token) and nothing else, so a
# credential cannot name a header that changes what the request means: a method override, a forwarded
# address, Host.
KEY_HEADER = re.compile(r"[A-Za-z0-9-]{0,58}(key|token)", re.IGNORECASE)
UNSAFE_HEADER_PREFIXES = ("x-forwarded-", "x-http-method", "x-method", "proxy-", "sec-")
# What a key is made of: visible ASCII. A space or a line break would rewrite the header.
KEY_TEXT = re.compile(r"[\x21-\x7e]{1,4096}")
# A value the model sends is a word, an id or a short phrase: never a document.
MAX_ARGUMENT_CHARS = 1_000
DEFAULT_KEY_HEADER = "X-API-Key"
AUTH_BEARER = "bearer"
AUTH_HEADER = "header"
FAULTS_SHOWN = 5

BAD_ARGUMENTS = "partner_bad_arguments"
BAD_BASE_URL = "partner_bad_base_url"
BAD_CREDENTIAL = "partner_bad_credential"
OTHER_HOST = "partner_other_host"


class PartnerRefusal(Exception):
    """A call the platform will not make, or an answer it will not pass on.

    ``code`` names why (the result's ``error_code``); ``status_code`` is the partner's HTTP
    status when there was one.
    """

    def __init__(self, code: str, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def chosen_operation(tool: Tool, arguments: Mapping[str, Any]) -> Tuple[Operation, Dict[str, Any]]:
    """The operation a call names, and the arguments that are that operation's own."""
    own = {key: value for key, value in arguments.items() if key != OPERATION_KEY}
    if len(tool.operations) == 1:
        return tool.operations[0], own
    asked = arguments.get(OPERATION_KEY)
    names = [operation.name for operation in tool.operations]
    for operation in tool.operations:
        if operation.name == asked:
            return operation, own
    raise PartnerRefusal(BAD_ARGUMENTS, f"'{OPERATION_KEY}' must be one of {names}.")


def _fault(error: Any) -> str:
    where = ".".join(str(part) for part in error.path)
    return f"{where}: {error.message}" if where else error.message


def checked_arguments(operation: Operation, arguments: Mapping[str, Any]) -> Dict[str, Any]:
    """``arguments`` when they fit the operation; a refusal that says what does not."""
    schema = {**operation.parameters.as_json_schema(), "additionalProperties": False}
    errors = sorted(Draft202012Validator(schema).iter_errors(dict(arguments)),
                    key=lambda error: [str(part) for part in error.path])
    if errors:
        faults = "; ".join(_fault(error) for error in errors[:FAULTS_SHOWN])
        raise PartnerRefusal(BAD_ARGUMENTS, f"'{operation.name}' cannot run with these arguments: {faults}")
    long = [name for name, value in arguments.items() if isinstance(value, str) and len(value) > MAX_ARGUMENT_CHARS]
    if long:
        raise PartnerRefusal(BAD_ARGUMENTS, f"{long} must be at most {MAX_ARGUMENT_CHARS} characters.")
    for name in placeholders(operation.path):
        text = _text(arguments[name])
        if not PATH_VALUE.fullmatch(text) or set(text) <= {"."}:
            raise PartnerRefusal(BAD_ARGUMENTS, f"'{name}' must be one plain id: letters, digits and . _ ~ : @ - only.")
    return dict(arguments)


def _text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _origin(url: str) -> Tuple[str, str, int]:
    parts = urlsplit(url)
    return parts.scheme.lower(), (parts.hostname or "").lower(), parts.port or HTTPS_PORT


def same_origin(base_url: str, url: str) -> bool:
    """``url`` goes to the scheme, host and port ``base_url`` names."""
    try:
        return _origin(base_url) == _origin(url)
    except ValueError:  # a port that is not a number: not the same origin as anything
        return False


def checked_base_url(raw: Any) -> str:
    """The credential's base URL when the platform will call it: https, a host, and nothing
    after the path. The refusal never repeats the URL, which may hold what it should not."""
    text = str(raw or "").strip()
    try:
        parts = urlsplit(text)
        port_ok = parts.port is None or 0 < parts.port < 65536
    except ValueError:
        parts, port_ok = None, False
    if parts is None or not port_ok or parts.scheme.lower() != REQUIRED_SCHEME or not parts.hostname:
        raise PartnerRefusal(BAD_BASE_URL, "The pack's credential needs an https base URL.")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise PartnerRefusal(BAD_BASE_URL, "The pack's credential's base URL must be an address only: "
                                           "no user, password, query or fragment.")
    return text.rstrip("/")


def request_url(base_url: str, operation: Operation, arguments: Mapping[str, Any]) -> str:
    """The URL for one call: the base URL, the operation's path with its values encoded, and
    the rest of the arguments as the query. Refused unless it stays on the base URL's origin."""
    in_path = set(placeholders(operation.path))
    path = operation.path
    for name in in_path:
        path = path.replace("{" + name + "}", quote(_text(arguments[name]), safe=""))
    query = {**operation.query, **{key: value for key, value in arguments.items() if key not in in_path}}
    url = base_url.rstrip("/") + path
    if query:
        url = f"{url}?{urlencode({key: _text(value) for key, value in query.items()})}"
    if not same_origin(base_url, url):
        raise PartnerRefusal(OTHER_HOST, "The call would leave the partner's own host, so it was not sent.")
    return url


def auth_headers(credential: Mapping[str, Any]) -> Dict[str, str]:
    """The header that carries the key, as the credential says to send it."""
    key = str(credential.get("api_key") or "").strip()
    if not KEY_TEXT.fullmatch(key):
        raise PartnerRefusal(BAD_CREDENTIAL, "The pack's credential has no usable API key.")
    method = str(credential.get("auth_method") or AUTH_HEADER).strip().lower()
    if method == AUTH_BEARER:
        return {"Authorization": f"Bearer {key}"}
    if method != AUTH_HEADER:
        raise PartnerRefusal(BAD_CREDENTIAL, "Partner tools send the key as a bearer token or in a header: "
                                             "set the credential's auth method to 'bearer' or 'header'.")
    name = str(credential.get("header_name") or DEFAULT_KEY_HEADER).strip()
    if not KEY_HEADER.fullmatch(name) or name.lower().startswith(UNSAFE_HEADER_PREFIXES):
        raise PartnerRefusal(BAD_CREDENTIAL, "The pack's credential names a header the key cannot go in: use a "
                                             "name that ends in Key or Token, such as X-API-Key.")
    return {name: key}


__all__ = ["BAD_ARGUMENTS", "BAD_BASE_URL", "BAD_CREDENTIAL", "OTHER_HOST", "PartnerRefusal", "auth_headers",
           "checked_arguments", "checked_base_url", "chosen_operation", "request_url", "same_origin"]
