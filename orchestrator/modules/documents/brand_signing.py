"""A document is signed by the brand kit's sign-off, not "[Your name]" (night 9b, prep for night 10).

Night 9b: drafts came out signed "[Your name]" (#1971, #0095) and a PDF went out with
"[Your Name]" under it (Auto's 1b39361c). Every document the platform renders passes
through ``DocumentGenerationService.generate`` (an agent's generate_document, a
playbook's document step, a mission's report, the Studio's preview and render routes),
so :func:`a_document_is_signed` fills a placeholder signature in its data there, before
the template renders, with the kit's sign-off (``services.brand_rules.sign_off_name``).
A workspace whose kit names no one keeps the placeholder, and the render's own checks
and the card's notes still see it. Night 10 (F336): "Sincerely, [Your Company Name]"
shipped in two documents; a company placeholder becomes the kit's company name
(``services.brand_rules.fill_placeholders``). Night 10c (F364): a document whose data
names its signer (``data.signer``: "sign it from me, Gerard") is signed by that
name, never the kit's sign-off: its placeholders take the signer too.

``services`` is imported when a document is made, never when this module loads: the
document modules load before the services.
"""
from __future__ import annotations

import functools
from typing import Any, Awaitable, Callable, Dict, Optional

from modules.documents.variables.catalog import signer_of

Async = Callable[..., Awaitable[Any]]


def with_signer(kit: Optional[Dict[str, Any]], data: Any) -> Optional[Dict[str, Any]]:
    """``kit`` with the signer ``data`` names as its sign-off (F364); ``kit`` itself without one."""
    signer = signer_of(data)
    if not signer or kit is None:
        return kit
    return {**kit, "voice": {**(kit.get("voice") or {}), "sign_off": signer}}


def signed(value: Any, by: Any, fill: Callable[[str, Any], str]) -> Any:
    """``value`` (a document's data) with ``fill(text, by)`` applied to every text in it: a new object."""
    if isinstance(value, str):
        return fill(value, by)
    if isinstance(value, dict):
        return {key: signed(item, by, fill) for key, item in value.items()}
    if isinstance(value, list):
        return [signed(item, by, fill) for item in value]
    return value


def a_document_is_signed(generate: Async) -> Async:
    """Wrap ``DocumentGenerationService.generate`` (called with keywords): the data
    renders with the kit's sign-off where it left a placeholder signature, and the
    kit's company name where it left a company placeholder."""
    @functools.wraps(generate)
    async def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        from services import brand_rules as br

        data = kwargs.get("data")
        workspace_id = kwargs.get("workspace_id") or getattr(self, "workspace_id", None)
        has_data = isinstance(data, dict) and bool(data)
        kit = with_signer(await br.kit_off_loop(getattr(self, "db", None), workspace_id), data) if has_data else None
        if br.sign_off_name(kit):   # no sign-off means no company name either
            kwargs = {**kwargs, "data": signed(data, kit, br.fill_placeholders)}
        return await generate(self, *args, **kwargs)
    return wrapped


__all__ = ["a_document_is_signed", "signed", "with_signer"]
