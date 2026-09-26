"""F208 (a) — the OpenRouter credit balance is read where OpenRouter puts it.

GET /api/analytics/llm/openrouter/credits in c1 answered 0.0 and 0.0 while the
backend log read "can only afford 4013". get_credits read total_credits and
total_usage from the top level of OpenRouter's /api/v1/credits answer, and
OpenRouter nests them under "data". get_key_info, in the same file, already
unwrapped it. So an owner could never see their credit, and F197's balance
display depends on it.
"""
from __future__ import annotations

import asyncio

import pytest


def _answering(monkeypatch, body):
    from core.llm import openrouter_analytics

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return body

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url, headers=None):
            assert url.endswith("/credits")
            return _Response()

    monkeypatch.setattr(openrouter_analytics.httpx, "AsyncClient", _Client)
    return asyncio.run(openrouter_analytics.OpenRouterAnalyticsService().get_credits("k"))


def test_the_balance_is_read_from_openrouters_data_envelope(monkeypatch):
    """The documented shape: {"data": {"total_credits": …, "total_usage": …}}."""
    got = _answering(monkeypatch, {"data": {"total_credits": 25.0, "total_usage": 21.36}})
    assert got == {"total_credits": 25.0, "total_usage": 21.36}


@pytest.mark.parametrize("body", [{"total_credits": 25.0, "total_usage": 21.36}], ids=["unwrapped"])
def test_an_unwrapped_answer_still_reads(monkeypatch, body):
    assert _answering(monkeypatch, body) == {"total_credits": 25.0, "total_usage": 21.36}
