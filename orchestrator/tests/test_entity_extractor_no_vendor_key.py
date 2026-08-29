"""EntityExtractor must construct without a vendor key.

FOUND IN PRODUCTION (2026-08-29, Railway logs during onboarding testing):

    Entity extraction failed for document 69991: Missing credentials. Please
    pass an `api_key` ... or set the `OPENAI_API_KEY` environment variable.
      File "modules/search/services/entity_extractor.py", line 45, in __init__
        self.openai_client = AsyncOpenAI()
      openai.OpenAIError: Missing credentials

``AsyncOpenAI()`` raises at CONSTRUCTION when ``OPENAI_API_KEY`` is unset, and
this platform runs on OpenRouter. So the raise fired on EVERY document
ingestion. The caller catches it and logs a warning, so ingestion appeared to
succeed while entity extraction was silently disabled — every ingested
document since has contributed no entities to the knowledge graph.

Same class as the #645 Harbourline failure: never demand a vendor key that may
not exist. The extractor now builds its LLM lazily through the platform's own
``create_llm_manager``, which resolves provider/model/key (workspace key first,
OpenRouter fallback) instead of hard-coding a vendor.
"""
from __future__ import annotations

from pathlib import Path

import pytest

MODULE = (
    Path(__file__).resolve().parents[1]
    / "modules" / "search" / "services" / "entity_extractor.py"
)


# --------------------------------------------------------------------------- #
# The regression: construction must not need a vendor key
# --------------------------------------------------------------------------- #


def test_constructs_with_no_openai_key_present(monkeypatch):
    """THE PROD FAILURE. With OPENAI_API_KEY absent, construction must still succeed."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)

    from modules.search.services.entity_extractor import EntityExtractor

    extractor = EntityExtractor()  # must not raise
    assert extractor is not None


def test_no_llm_client_is_built_eagerly(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from modules.search.services.entity_extractor import EntityExtractor

    extractor = EntityExtractor()
    # The manager is created on first USE, never at construction.
    assert getattr(extractor, "_llm_manager", "missing") is None
    assert not hasattr(extractor, "openai_client")


def test_module_never_constructs_a_vendor_client_at_import_or_init():
    """Source guard: reintroducing `AsyncOpenAI()` reopens the outage."""
    src = MODULE.read_text()
    code = "\n".join(
        line for line in src.splitlines()
        if not line.strip().startswith("#")
    )
    # The only surviving mention is inside the docstring explaining the bug.
    assert "AsyncOpenAI()" not in code.split('"""')[0]
    assert "self.openai_client" not in code
    assert "from openai import AsyncOpenAI" not in code


def test_routes_through_the_platform_llm_manager():
    src = MODULE.read_text()
    assert "create_llm_manager" in src, (
        "must resolve provider/model/key through platform routing, not a vendor SDK"
    )
    # The hard-coded vendor default is gone with it.
    assert 'or "gpt-4o-mini"' not in src


# --------------------------------------------------------------------------- #
# Response shape handling
# --------------------------------------------------------------------------- #


class _Resp:
    def __init__(self, content):
        self.content = content


@pytest.mark.parametrize(
    "response,expected",
    [(_Resp("  hello  "), "hello"), (_Resp("[]"), "[]"), ("raw string", "raw string")],
)
def test_content_of_handles_shapes(response, expected):
    from modules.search.services.entity_extractor import EntityExtractor

    assert EntityExtractor._content_of(response) == expected
