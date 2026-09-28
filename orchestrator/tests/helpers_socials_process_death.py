"""A process that dies mid-render, for the P251W1-RVW-5 tests.

A redeploy's SIGKILL, an OOM or a crash ends the process where it stands: no
``except`` runs and no ``finally`` reaches the database. Python always runs a
``finally``, so the stand-in has two halves. :class:`ProcessDied` is a
``BaseException``, which no ``except Exception`` catches; and once a
:class:`Mortal` has died, no session opens, neither the recipe's (the render's
session factory) nor the usage tracker's (``SessionLocal``). Whatever the code
tries after the death never lands, so what ``llm_usage`` holds afterwards is
what a real death would leave: only what was committed before it.

Used by ``test_prd251w2_footage_booked_first.py`` and
``test_prd251w2_voice_booked_first.py``, on the Wave 1 footage and voice
harnesses.
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List

import httpx
import pytest

import core.database.database as database_mod
import modules.socials.render as render
from core.llm.providers import MEDIA_RENDER_PROVIDER
from core.media_render_client import MediaRenderClient
from core.models.core import LLMUsage


class ProcessDied(BaseException):
    """The process died here."""


class Mortal:
    """Sessions from ``factory`` while the process lives; none once it has died."""

    def __init__(self, factory: Callable[[], Any]):
        self.factory = factory
        self.dead = False

    def __call__(self) -> Any:
        if self.dead:
            raise ProcessDied("the process is gone: nothing after its death reaches the database")
        return self.factory()

    def die(self, *_args: Any) -> None:
        """Kill the process wherever the render awaits it (a Composio answer, say)."""
        self.dead = True
        raise ProcessDied("the process died")


def mortal(env: Any, monkeypatch: Any) -> Mortal:
    """The harness's database as a process that can die; the usage tracker's
    sessions (``SessionLocal``) die with it."""
    process = Mortal(env.factory)
    monkeypatch.setattr(database_mod, "SessionLocal", process)
    return process


def render_until_it_dies(env: Any, post_id: str, process: Mortal, *, renderer: Any, store: Any, token: str) -> None:
    """Start the post's render through the route, then run it in ``process``,
    which dies part-way. The post is left rendering, as a dead process leaves it."""
    resp = env.client.post(f"/api/socials/posts/{post_id}/render")
    assert resp.status_code == 202, resp.text
    job = env.launched[-1]

    async def run() -> None:
        transport = httpx.MockTransport(renderer.handler)
        async with httpx.AsyncClient(transport=transport, headers={"X-Internal-Token": token}) as http:
            await render.run_render(job, client=MediaRenderClient(http), store=store, session_factory=process)

    with pytest.raises(ProcessDied):
        asyncio.run(run())
    assert process.dead, "the process was meant to die part-way through the render"


def paid_media_now(env: Any, post_id: str) -> List[Dict[str, Any]]:
    """The post's paid media in ``llm_usage`` at this moment (not the renderer's
    own seconds): readable from inside a toolkit call, mid-render."""
    env.session.expire_all()
    rows = (
        env.session.query(LLMUsage)
        .filter(LLMUsage.execution_id == f"social_post:{post_id}", LLMUsage.provider != MEDIA_RENDER_PROVIDER)
        .order_by(LLMUsage.id)
        .all()
    )
    return [
        {"provider": row.provider, "usd": row.total_cost, "units": row.input_tokens, "status": row.status}
        for row in rows
    ]
