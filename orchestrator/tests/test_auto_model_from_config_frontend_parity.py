"""The model ids around the platform default stay real, and the UI keeps none (Gerard, 19 Oct 2026).

* frontend/lib/llm-defaults.ts holds no provider or model id: the UI reads the
  platform default from ``GET /api/models/platform-default`` (default_route), so a
  change of the setting needs no frontend release and there is no copy to drift.
* With no Anthropic key the factory sends a Claude through OpenRouter, which lists
  only dotted ids (``anthropic/claude-haiku-5.5``); the hyphenated direct id 404s.
* A saved Anthropic key is checked with the named probe model, not the retired
  ``claude-3-haiku-20240307``.
PURE: files are read as text; the HTTP session is faked.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

from core.llm import defaults

REPO = Path(__file__).resolve().parents[2]
FRONTEND_DEFAULTS = REPO / "frontend" / "lib" / "llm-defaults.ts"


def test_the_frontend_defaults_hold_no_model_id():
    source = FRONTEND_DEFAULTS.read_text(encoding="utf-8")
    assert not re.search(r"(provider|model_id):\s*'[^']+'", source)
    for model_id in (defaults.DEFAULT_LLM_MODEL, "claude-", "gemini", "gpt-"):
        assert model_id not in source
    assert "/api/models/platform-default" in source


# ── the no-Anthropic-key fallback sends OpenRouter's id ─────────────────────

def _factory():
    from modules.agents.factory.agent_factory import AgentFactory
    return AgentFactory(db_session=None)


@pytest.mark.parametrize("direct, twin", [
    ("claude-haiku-5-5", "anthropic/claude-haiku-5.5"),
    ("claude-sonnet-4-5-20250929", "anthropic/claude-sonnet-4.5"),
    ("claude-opus-5", "anthropic/claude-opus-5"),
])
def test_a_direct_claude_id_falls_back_to_its_dotted_openrouter_twin(direct, twin):
    assert _factory()._openrouter_model_id("anthropic", direct) == twin


def test_the_default_model_has_a_dotted_twin_when_only_openrouter_is_keyed():
    routed = _factory()._openrouter_model_id(defaults.DEFAULT_LLM_PROVIDER, defaults.DEFAULT_LLM_MODEL)
    assert routed == "anthropic/claude-haiku-5.5"


def test_the_verifier_defaults_name_openrouters_dotted_claude():
    pairs = dict(item.split("=", 1) for item in defaults.VERIFIER_MODEL_MAPPING.split(","))
    assert pairs["openai"] == "anthropic/claude-haiku-4.5"
    assert not any(re.search(r"anthropic/claude-.*\d-\d", model) for model in pairs.values())


# ── the Anthropic key probe ─────────────────────────────────────────────────

class _Response:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return {"model": defaults.ANTHROPIC_KEY_PROBE_MODEL, "usage": {}}


class _Session:
    def __init__(self):
        self.sent = []

    def post(self, url, headers=None, json=None):
        self.sent.append(json)
        return _Response()


def test_an_anthropic_key_is_probed_with_the_named_current_model():
    import core.credentials.tester as tester_module

    tester = tester_module.CredentialTester()
    tester.session = _Session()
    result = asyncio.run(tester._test_anthropic({"api_key": "probe-test-key"}))
    assert result["success"] is True
    assert [body["model"] for body in tester.session.sent] == [defaults.ANTHROPIC_KEY_PROBE_MODEL]
    assert defaults.ANTHROPIC_KEY_PROBE_MODEL == "claude-haiku-5-5"
    assert "claude-3-haiku-20240307" not in Path(tester_module.__file__).read_text(encoding="utf-8")
