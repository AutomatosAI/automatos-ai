"""The data a template's preview renders from (F348, night 10b).

F348 (5 Oct): ``POST /api/documents/templates/<id>/preview`` answered 422 "template
variables did not resolve" for every field when it was sent the template's own
sample. The starters (``presets``) and the Template Studio store a sample as
``{"data": {...}}``, the shape a ``generate_document`` call's arguments carry,
while the route handed the body to the renderer as the data itself, so every
``data.<field>`` chip looked one level too high and found nothing. With no body
the route used the stored sample, still wrapped, and failed the same way.

:func:`preview_data` takes either shape: a body (or sample) whose only key is an
object under ``data`` is that object. An empty body previews the stored sample.
The result is a deep copy: generation fills in a title and renames section keys,
and must never write into the stored sample. Pure.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Optional

DATA_KEY = "data"


def unwrapped(payload: Any) -> Dict[str, Any]:
    """``payload`` as the data itself: ``{"data": {...}}`` is its inner object. A copy."""
    if not isinstance(payload, dict):
        return {}
    inner = payload.get(DATA_KEY)
    if set(payload) == {DATA_KEY} and isinstance(inner, dict):
        return copy.deepcopy(inner)
    return copy.deepcopy(payload)


def preview_data(body: Optional[Dict[str, Any]], sample: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """What a preview renders: the body, else the template's stored sample, either shape."""
    return unwrapped(body) or unwrapped(sample)


__all__ = ["DATA_KEY", "preview_data", "unwrapped"]
