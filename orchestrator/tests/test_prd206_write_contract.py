"""PRD-206 S1 — the ONE memory-write contract.

Covers:
  - the taxonomy gains decision / open_loop / thread_summary and the distiller
    emits them (mocked LLM — no network),
  - BOTH write paths (distill + platform_store_memory tool) produce the same
    canonical metadata keys via ``build_memory_metadata``,
  - the exclusion validator (Q3 silent-everything: it carries ALL the consent
    weight) blocks secrets/credentials/payment strings on both paths,
  - the Q7 split sharing default: user_fact/preference → private (when an
    owner is known), everything else → workspace; explicit override honoured.

Pure/mocked throughout — no DB, no vector store, no LLM.
"""
import os
import sys
import types
from pathlib import Path

import pytest

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

for _k, _v in {
    "POSTGRES_USER": "test",
    "POSTGRES_PASSWORD": "test",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "test",
}.items():
    os.environ.setdefault(_k, _v)

import core.llm as core_llm  # noqa: E402  (patch target for create_llm_manager)

# consumers/__init__.py eagerly imports the chatbot stack → RAG → camelot, an
# optional PDF dep that isn't installed in the test env.
sys.modules.setdefault("camelot", types.ModuleType("camelot"))

from modules.memory.write_contract import (  # noqa: E402
    DEFAULT_FACT_TYPE,
    MEMORY_FACT_TYPES,
    MEMORY_SCOPE_PRIVATE,
    MEMORY_SCOPE_WORKSPACE,
    SOURCE_TYPE_DISTILLED,
    build_memory_metadata,
    default_scope_for_type,
    violates_exclusions,
)
from consumers.chatbot.smart_memory import SmartMemoryManager  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes (same shape as test_l3_distill_input)
# ---------------------------------------------------------------------------

class _FakeResp:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    def __init__(self, content: str):
        self._content = content

    async def generate_response(self, messages, tools=None):
        return _FakeResp(self._content)


def _patch_llm(monkeypatch, content: str) -> None:
    monkeypatch.setattr(core_llm, "create_llm_manager", lambda **kw: _FakeLLM(content))


class _FakeUnified:
    def __init__(self):
        self.two_tier_calls = []
        self.transcript_calls = []

    async def store_two_tier(self, **kwargs):
        self.two_tier_calls.append(kwargs)
        return [("global", {"success": True})]

    async def store_transcript(self, **kwargs):
        self.transcript_calls.append(kwargs)
        return "row-id"


# ---------------------------------------------------------------------------
# New types: taxonomy + distiller
# ---------------------------------------------------------------------------

def test_taxonomy_gains_continuity_types():
    assert {"decision", "open_loop", "thread_summary"} <= MEMORY_FACT_TYPES


def test_distill_prompt_teaches_new_types_and_secret_ban():
    prompt = SmartMemoryManager._build_distill_prompt("u", "a")
    for t in ("decision", "open_loop", "thread_summary"):
        assert t in prompt
    assert "NEVER extract secrets" in prompt


@pytest.mark.asyncio
async def test_distill_emits_new_types(monkeypatch):
    _patch_llm(
        monkeypatch,
        '[{"fact": "Decided to ship the pilot without SSO because of timeline.", '
        '"type": "decision", "importance": 0.9}, '
        '{"fact": "Still need to pick a vector DB for the academy pod.", '
        '"type": "open_loop", "importance": 0.7}]',
    )
    mgr = SmartMemoryManager()
    facts = await mgr._distill_durable_facts(
        "transcript", "reply", workspace_id="ws1", agent_id=3
    )
    assert facts is not None
    assert [f["type"] for f in facts] == ["decision", "open_loop"]
    for f in facts:
        assert f["type"] in MEMORY_FACT_TYPES


# ---------------------------------------------------------------------------
# The unified contract — both paths, same canonical keys
# ---------------------------------------------------------------------------

# Keys the contract guarantees on EVERY L3 write, whichever path produced it.
CANONICAL_REQUIRED = {"type", "category", "importance", "scope", "source_type"}


def test_write_contract_required_keys_always_present():
    meta = build_memory_metadata(fact_type="decision")
    assert CANONICAL_REQUIRED <= set(meta)
    assert meta["type"] == meta["category"] == "decision"
    assert meta["importance"] == 0.5
    assert meta["scope"] == MEMORY_SCOPE_WORKSPACE
    assert meta["source_type"] == SOURCE_TYPE_DISTILLED


@pytest.mark.asyncio
async def test_write_contract_unified_across_both_paths(monkeypatch):
    """The distill path's stored metadata carries the same canonical keys as a
    tool-path build — the PRD-206 scout's split-brain is closed."""
    _patch_llm(
        monkeypatch,
        '[{"fact": "The user prefers dark UI themes.", "type": "preference", '
        '"importance": 0.6}]',
    )
    mgr = SmartMemoryManager()
    fake = _FakeUnified()
    mgr._unified_service = fake

    ok = await mgr.store_conversation(
        workspace_id="ws1",
        agent_id=3,
        user_message="I prefer dark themes everywhere please",
        assistant_response="Noted — dark themes it is.",
        chat_id="chat-42",
        subject_id="user:7",
    )
    assert ok is True
    assert len(fake.two_tier_calls) == 1
    distill_meta = fake.two_tier_calls[0]["metadata"]

    tool_meta = build_memory_metadata(
        fact_type="preference",
        importance=0.6,
        source_type="inference",
        owner="user:7",
        chat_id="chat-42",
    )

    assert CANONICAL_REQUIRED <= set(distill_meta)
    assert CANONICAL_REQUIRED <= set(tool_meta)
    for key in CANONICAL_REQUIRED - {"source_type"}:
        assert distill_meta[key] == tool_meta[key], key
    # Provenance differs by lane, honestly: distilled vs inference.
    assert distill_meta["source_type"] == SOURCE_TYPE_DISTILLED
    assert tool_meta["source_type"] == "inference"
    # Both carry the owner + thread link.
    assert distill_meta["owner"] == tool_meta["owner"] == "user:7"
    assert distill_meta["chat_id"] == tool_meta["chat_id"] == "chat-42"


def test_write_contract_extra_never_overrides_contract_keys():
    extra = {"workspace_id": "ws1", "type": "spoofed", "scope": "spoofed"}
    meta = build_memory_metadata(fact_type="user_fact", owner="user:1", extra=extra)
    assert meta["type"] == "user_fact"
    assert meta["scope"] == MEMORY_SCOPE_PRIVATE
    assert meta["workspace_id"] == "ws1"
    # The caller's dict is not mutated (immutability rule).
    assert extra["type"] == "spoofed"


def test_write_contract_coercion():
    meta = build_memory_metadata(fact_type="not-a-type", importance=7, confidence=-3)
    assert meta["type"] == DEFAULT_FACT_TYPE
    assert meta["importance"] == 1.0        # clamped
    assert meta["confidence"] == 0.0        # clamped
    assert build_memory_metadata(fact_type="decision", importance="junk")["importance"] == 0.5


# ---------------------------------------------------------------------------
# Q7 — split sharing default
# ---------------------------------------------------------------------------

def test_scope_defaults_split():
    # Personal types go private only when an owner is known.
    assert default_scope_for_type("user_fact", has_owner=True) == MEMORY_SCOPE_PRIVATE
    assert default_scope_for_type("preference", has_owner=True) == MEMORY_SCOPE_PRIVATE
    # No owner → private would be visible to no one; default stays workspace.
    assert default_scope_for_type("user_fact", has_owner=False) == MEMORY_SCOPE_WORKSPACE
    # Continuity + operational types are workspace objects.
    for t in ("decision", "open_loop", "thread_summary", "business_fact", "procedure"):
        assert default_scope_for_type(t, has_owner=True) == MEMORY_SCOPE_WORKSPACE


def test_scope_override_honoured_both_ways():
    meta = build_memory_metadata(fact_type="decision", scope="private", owner="user:1")
    assert meta["scope"] == MEMORY_SCOPE_PRIVATE
    meta = build_memory_metadata(fact_type="preference", scope="workspace", owner="user:1")
    assert meta["scope"] == MEMORY_SCOPE_WORKSPACE


# ---------------------------------------------------------------------------
# Q3 — the exclusion validator carries the consent weight
# ---------------------------------------------------------------------------

BLOCKED = [
    ("sk-abcdefghijklmnop1234", "openai_style_key"),
    ("-----BEGIN RSA PRIVATE KEY-----\nMIIE...", "pem_private_key"),
    ("the password is hunter2", "password_assignment"),
    ("postgres://admin:s3cret@db.internal:5432/prod", "credentialed_url"),
    ("ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345", "github_token"),
    ("xoxb-123456789012-abcdefghijklm", "slack_token"),
    ("AKIAIOSFODNN7EXAMPLE", "aws_access_key_id"),
    ("card 4111 1111 1111 1111 expires 09/28", "card_number"),
    ("their IBAN is GB82WEST12345698765432", "iban"),
    ("my ssn is 078-05-1120", "ssn"),
    ("the 2FA verification code is 493021", "otp_code"),
    ("wrote down the wallet seed phrase in the vault", "seed_phrase"),
    # Assembled at runtime so secret scanners (gitleaks) don't flag the
    # validator's own fixture as a leak — the joined string still trips the
    # generic_secret_assignment rule exactly like a pasted literal would.
    ("api_key = " + "9f8e" + "7d6c" + "5b4a" + "3210", "generic_secret_assignment"),
    ("Authorization: Bearer abcdef0123456789abcdef", "bearer_token"),
]


@pytest.mark.parametrize("text,rule", BLOCKED)
def test_exclusion_validator_blocks_sensitive_content(text, rule):
    assert violates_exclusions(text) == rule


ALLOWED = [
    "The password rotation policy is 90 days.",
    "Deploy day is Thursday; the API key rotates quarterly.",
    "Gerard likes black.",
    "Order 8-digit id 12345678 shipped on Friday.",
    "The mission run id is 1234567890123456 in the audit log.",  # non-Luhn digits
    "We decided to adopt PRD-206 phase 1 scope.",
    "",
]


@pytest.mark.parametrize("text", ALLOWED)
def test_exclusion_validator_keeps_benign_content(text):
    assert violates_exclusions(text) is None


@pytest.mark.asyncio
async def test_distill_path_drops_excluded_facts_and_stays_honest(monkeypatch):
    """An excluded fact is never stored, and the PRD-159 S5 honest counter
    counts only the facts that were actually attempted."""
    _patch_llm(
        monkeypatch,
        '[{"fact": "The staging db password is hunter2", "type": "business_fact", '
        '"importance": 0.9}, '
        '{"fact": "Deploys happen on Thursdays.", "type": "procedure", '
        '"importance": 0.7}]',
    )
    mgr = SmartMemoryManager()
    fake = _FakeUnified()
    mgr._unified_service = fake

    ok = await mgr.store_conversation(
        workspace_id="ws1", agent_id=3,
        user_message="staging db password is hunter2, deploys are Thursdays",
        assistant_response="Noted the deploy cadence.",
    )
    assert ok is True
    stored_texts = [
        " ".join(m["content"] for m in c["messages"]) for c in fake.two_tier_calls
    ]
    assert len(stored_texts) == 1
    assert "hunter2" not in stored_texts[0]
    assert "Thursdays" in stored_texts[0]
    assert mgr._last_l3_facts_stored == 1


@pytest.mark.asyncio
async def test_distill_path_all_facts_excluded_keeps_transcript(monkeypatch):
    """Every fact excluded → nothing durable is a SUCCESS (transcript kept),
    not a storage failure."""
    _patch_llm(
        monkeypatch,
        '[{"fact": "api_key = deadbeef01234567", "type": "business_fact", '
        '"importance": 0.9}]',
    )
    mgr = SmartMemoryManager()
    fake = _FakeUnified()
    mgr._unified_service = fake

    ok = await mgr.store_conversation(
        workspace_id="ws1", agent_id=3,
        user_message="store the api key deadbeef01234567 for later",
        assistant_response="I can't keep credentials in memory.",
    )
    assert ok is True
    assert fake.two_tier_calls == []
    assert len(fake.transcript_calls) == 1
    assert mgr._last_l3_facts_stored == 0


@pytest.mark.asyncio
async def test_tool_path_refuses_excluded_content():
    """platform_store_memory refuses BEFORE touching any service, and the
    refusal names the rule without echoing the content."""
    from modules.tools.discovery.handlers_workspace import store_memory

    result = await store_memory(
        None, "00000000-0000-0000-0000-000000000001",
        {"content": "the password is hunter2"},
    )
    assert result["success"] is False
    assert "exclusion policy" in result["error"]
    assert "password_assignment" in result["error"]
    assert "hunter2" not in result["error"]


@pytest.mark.asyncio
async def test_tool_path_validates_type_and_scope():
    from modules.tools.discovery.handlers_workspace import store_memory

    bad_type = await store_memory(
        None, "00000000-0000-0000-0000-000000000001",
        {"content": "x", "type": "nonsense"},
    )
    assert bad_type["success"] is False and "type must be one of" in bad_type["error"]

    bad_scope = await store_memory(
        None, "00000000-0000-0000-0000-000000000001",
        {"content": "x", "scope": "global"},
    )
    assert bad_scope["success"] is False and "scope must be one of" in bad_scope["error"]
