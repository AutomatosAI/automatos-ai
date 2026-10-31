"""PRD-270 W2 (K8) — the one error an item bundle check raises, and its codes.

A code says what kind of problem the bundle has; the message says where, by the format's own
field names and list positions. A message never quotes the bundle: a key, a value or a name a
remote sent stays out of it, so it can go into a log or an answer as it is.

Standard library only: the hosted export and the local import share this package.
"""
from __future__ import annotations

# What each code means:
# 'schema'      not a schema 1 bundle of a known type, a required key missing, or a value under
#               its minimum (an empty name, a negative number, a list with too few entries)
# 'unknown-key' a key outside the format, at any depth
# 'type'        a value of the wrong JSON type
# 'too-long'    a text over its length, or a number over its maximum
# 'too-many'    a list or a set of names over its count
# 'pattern'     a text off its pattern, or a value that is not one of the allowed choices
# 'ref'         a reference of the wrong kind or form, or a step naming no agent of the bundle
# 'too-large'   the whole document over the cap its caller gives
BUNDLE_ERROR_CODES = ("schema", "unknown-key", "type", "too-long", "too-many", "pattern", "ref", "too-large")


class BundleError(ValueError):
    """A bundle the format refuses: ``code`` (one of BUNDLE_ERROR_CODES) and a plain ``message``."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
