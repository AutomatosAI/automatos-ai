"""P256-FIX-SOC-5 (F437, socials day 14 Oct): what a plan does with a post it could not finish.

A plan's video draft never got media and its history said nothing: the render was refused before
it started, the plan maker's catch-all rolled back and sent "A planned post was not made" without
the reason, and the post stayed a draft with no entry. Now the reason decides (:func:`held_as`):

* ``TAKEN``: another writer has the post (it is rendering, or changed since it was read:
  ``IllegalTransition``, ``StaleContent``). It is left alone: failing it would make the live
  render drop its result.
* ``REFUSED``: the platform refused the render (no renderer or storage, no render minutes left, a
  report it cannot read). The draft fails with a ``render_failed`` entry every reader shows
  (``service.refuse_render``). A renderer's own words reach no one unless they are one of
  ``render``'s fixed messages: "The renderer is not healthy: <exc>" never does.
* ``WAITS``: anything else the lifecycle refuses (``SocialsError``: a field it cannot carry, a chart
  it cannot bind, a voice it cannot use) waits for the owner as SOC-2's draft does
  (``service.ask_for_input``), saying the route's own 422 words; and a failure of the composer or
  of anything else waits with :data:`COULD_NOT_WRITE`, never the exception's text.

The render's refusal of empty fields or a missing photo (``render.NotRenderable``) is SOC-2's
``socials_plan_maker._wait``, which names the fields.
"""
from __future__ import annotations

from typing import Tuple

from core import media_render_quota as render_quota
from modules.socials import render, report_charts, service

TAKEN, REFUSED, WAITS = "taken", "refused", "waits"
COULD_NOT_WRITE = "The plan could not write this post: write it yourself or ask Auto for another take."
# The renderer's refusals a person may read; any other text becomes the unreachable one.
SAFE_RENDERER_MESSAGES = (
    render.STORAGE_MESSAGE, render.MEDIA_PROFILE_MESSAGE, render.NOT_CONFIGURED_MESSAGE, render.UNREACHABLE_MESSAGE,
)
PLATFORM_REFUSALS = (render_quota.RenderQuotaExceeded, report_charts.ReportUnreadable)


def renderer_text(exc: render.RendererUnavailable) -> str:
    """What a person reads of a renderer outage: its fixed message, else the unreachable one."""
    text = str(exc)
    return text if text in SAFE_RENDERER_MESSAGES else render.UNREACHABLE_MESSAGE


def held_as(exc: BaseException) -> Tuple[str, str]:
    """(``TAKEN``, ``REFUSED`` or ``WAITS``, the user-safe why) for the failure ``exc`` of a plan's post."""
    if isinstance(exc, (service.IllegalTransition, service.StaleContent)):
        return TAKEN, ""
    if isinstance(exc, render.RendererUnavailable):
        return REFUSED, renderer_text(exc)
    if isinstance(exc, PLATFORM_REFUSALS):
        return REFUSED, str(exc)  # the quota's and the report's texts are written for the owner
    if isinstance(exc, service.SocialsError):
        return WAITS, str(exc)  # the texts the routes answer as 422 / 404
    return WAITS, COULD_NOT_WRITE


__all__ = ["COULD_NOT_WRITE", "REFUSED", "SAFE_RENDERER_MESSAGES", "TAKEN", "WAITS", "held_as", "renderer_text"]
