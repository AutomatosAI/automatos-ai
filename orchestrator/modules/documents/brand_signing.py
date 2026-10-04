"""A document is signed by the brand kit's sign-off, not "[Your name]" (night 9b, prep for night 10).

Night 9b: drafts came out signed "[Your name]" (#1971, #0095) and a PDF went out with
"[Your Name]" under it (Auto's 1b39361c). Every document the platform renders passes
through ``DocumentGenerationService.generate`` (an agent's generate_document, a
playbook's document step, a mission's report, the Studio's preview and render routes),
so :func:`a_document_is_signed` fills a placeholder signature in its data there, before
the template renders, with the kit's sign-off (``services.brand_rules.sign_off_name``).
A workspace whose kit names no one keeps the placeholder, and the render's own checks
and the card's notes still see it.

``services`` is imported when a document is made, never when this module loads: the
document modules load before the services.
"""
from __future__ import annotations

import functools
from typing import Any, Awaitable, Callable

Async = Callable[..., Awaitable[Any]]


def signed(value: Any, name: str, fill: Callable[[str, str], str]) -> Any:
    """``value`` (a document's data) with ``fill`` applied to every text in it: a new object."""
    if isinstance(value, str):
        return fill(value, name)
    if isinstance(value, dict):
        return {key: signed(item, name, fill) for key, item in value.items()}
    if isinstance(value, list):
        return [signed(item, name, fill) for item in value]
    return value


def a_document_is_signed(generate: Async) -> Async:
    """Wrap ``DocumentGenerationService.generate`` (called with keywords): the data
    renders with the kit's sign-off where it left a placeholder signature."""
    @functools.wraps(generate)
    async def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        from services import brand_rules as br

        data = kwargs.get("data")
        workspace_id = kwargs.get("workspace_id") or getattr(self, "workspace_id", None)
        has_data = isinstance(data, dict) and bool(data)
        kit = await br.kit_off_loop(getattr(self, "db", None), workspace_id) if has_data else None
        name = br.sign_off_name(kit)
        if name:
            kwargs = {**kwargs, "data": signed(data, name, br.fill_sign_off)}
        return await generate(self, *args, **kwargs)
    return wrapped


__all__ = ["a_document_is_signed", "signed"]
