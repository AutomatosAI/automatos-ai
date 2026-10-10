"""PRD-256 FX-015 (night 12, F1, F2, F393): memory the owner can rely on.

- On the local edition a memory written on a chat turn has an owner: the caller context carries
  no Clerk id there, so the executor gives the memory tools the driving person's ``users.id``
  (``memory_owner``); a private row is visible to that owner and to nobody else.
- The owner's standing rules (preferences, and what ``store_memory`` wrote on a turn a person
  drove) ride every chat turn whatever the intent: 'from November we post Tuesdays, remember
  that' is in the prompt of 'write the member note' in a fresh chat, a CREATION turn that
  recalls no memory. Newest first, capped at STANDING_RULES_MAX_TOKENS; none renders nothing.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

OWNER, OTHER = "7", "8"
RULE = "From November we post on Tuesdays, not Mondays."
CREATION_ASK = "Write a member note document for the Harvest Club"


class _FakeDurable:
    """The durable store's write and the standing-rules read, in memory (the scroll's filter
    semantics: the namespace, and at least one ``any_of`` key holding one of its values)."""

    def __init__(self) -> None:
        self.rows: list = []
        self._clock = datetime(2026, 10, 7, 9, tzinfo=timezone.utc)

    async def add(self, messages, user_id, metadata=None, workspace_id=None, subject_id=None):
        self._clock += timedelta(minutes=1)
        self.rows.append({"id": str(uuid.uuid4()), "namespace": user_id, "metadata": metadata or {},
                          "memory": "\n".join(m["content"] for m in messages),
                          "created_at": self._clock.isoformat()})
        return {"success": True, "id": self.rows[-1]["id"]}

    async def get_where_any(self, user_id, any_of, limit=100, where=None):
        def field(row, key):
            meta_key = key.split(".", 1)[1]
            return (row.get("metadata") or {}).get(meta_key)

        hits = [r for r in self.rows if r["namespace"] == user_id
                and any(field(r, key) in values for key, values in any_of.items())
                and all(field(r, key) == value for key, value in (where or {}).items())]
        return hits[:limit]


@pytest.fixture
def durable(monkeypatch):
    """A UnifiedMemoryService over the fake store, as the one the handler and the section read."""
    import modules.memory.unified_memory_service as ums
    from config import config

    store = _FakeDurable()
    service = ums.UnifiedMemoryService.__new__(ums.UnifiedMemoryService)
    service._durable = store
    service._redis_client_getter = lambda: None
    monkeypatch.setattr(ums, "get_unified_memory_service", lambda: service)
    monkeypatch.setattr(config, "QDRANT_URL", "http://qdrant.test")
    return store


def _store(params):
    from modules.tools.discovery.handlers_workspace import store_memory

    return asyncio.run(store_memory(MagicMock(), uuid.UUID(int=1), params))


def _said_by(user, content=RULE, **extra):
    """store_memory's params on a local turn ``user`` drives, as the executor injects them (no Clerk id)."""
    return {"content": content, "_driving_user_id": int(user), **extra}


def _block(workspace_id, viewer, widget_mode=False):
    from modules.context.sections.memory import standing_rules_block

    return asyncio.run(standing_rules_block(workspace_id, viewer_subject_id=viewer, widget_mode=widget_mode))


# ---------------------------------------------------------------------------
# The owner on every stored fact (the local edition)
# ---------------------------------------------------------------------------

def _owner_through_the_executor(monkeypatch, durable, caller_context, params):
    """The owner the real store_memory records when the executor runs it for ``caller_context``."""
    import modules.tools.discovery as discovery_pkg
    from modules.tools.discovery.action_registry import ActionDefinition
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    registry = MagicMock()
    registry.get.return_value = ActionDefinition(
        name="platform_store_memory", description="probe", category="t", permission_level="read",
        parameters={"type": "object", "properties": {}, "required": []},
    )
    monkeypatch.setattr(discovery_pkg, "get_action_registry", lambda: registry)
    db = MagicMock()
    db.execute.return_value.fetchone.return_value = (42,)                 # user_2abc's users.id
    executor = PlatformActionExecutor(db, uuid.UUID(int=1))
    result = asyncio.run(executor.execute("platform_store_memory", params, caller_context=caller_context))
    assert result.get("success") is True, result
    return durable.rows[-1]["metadata"].get("owner")


def test_a_local_chat_turn_gives_the_memory_the_driving_persons_id(monkeypatch, durable):
    local = {"driving_user_id": OWNER, "conversation_id": "c1"}          # no Clerk id on local
    spoofed = {"content": RULE, "_user_id": "user_2spoof", "_driving_user_id": 99}
    assert _owner_through_the_executor(monkeypatch, durable, local, spoofed) == f"user:{OWNER}"


def test_saas_keeps_the_clerk_id(monkeypatch, durable):
    saas = {"user_id": "user_2abc", "driving_user_id": OWNER, "conversation_id": "c1"}
    assert _owner_through_the_executor(monkeypatch, durable, saas, {"content": RULE}) == "user:42"


@pytest.mark.parametrize("headless", [None, {}, {"conversation_id": "c1"}])
def test_a_turn_made_for_nobody_writes_no_owner_and_a_spoofed_one_is_dropped(monkeypatch, durable, headless):
    spoofed = {"content": RULE, "_user_id": OWNER, "_driving_user_id": int(OWNER)}
    assert _owner_through_the_executor(monkeypatch, durable, headless, spoofed) is None


def test_the_drivers_id_fills_only_a_missing_clerk_id():
    from modules.tools.discovery.memory_owner import with_the_drivers_id

    params = {"content": RULE, "_driving_user_id": 7}
    assert with_the_drivers_id(params)["_user_id"] == "7" and "_user_id" not in params
    assert with_the_drivers_id({"_user_id": "user_2abc", "_driving_user_id": 7})["_user_id"] == "user_2abc"
    assert "_user_id" not in with_the_drivers_id({"_driving_user_id": True})
    assert "_user_id" not in with_the_drivers_id({"content": RULE})


def test_resume_context_on_local_sees_as_the_driving_person(monkeypatch):
    import modules.memory.resume_context as resume
    from modules.tools.discovery.handlers_workspace import resume_context
    from modules.tools.discovery.platform_executor import _DRIVER_AWARE_ACTIONS

    seen = {}

    async def _payload(db, *, workspace_id, viewer_user_id):
        seen["viewer"] = viewer_user_id
        return {}

    monkeypatch.setattr(resume, "build_resume_payload", _payload)
    monkeypatch.setattr(resume, "format_resume_for_llm", lambda payload: "")
    asyncio.run(resume_context(MagicMock(), uuid.UUID(int=1), {"_driving_user_id": int(OWNER)}))
    assert "platform_resume_context" in _DRIVER_AWARE_ACTIONS and seen["viewer"] == int(OWNER)


def test_a_private_row_written_on_local_has_an_owner_seen_by_them_alone(durable):
    from modules.memory.injection_filter import visible_to_viewer

    assert _store(_said_by(OWNER, "I take my flat white with oat milk", type="preference"))["success"] is True
    row = durable.rows[-1]
    assert row["metadata"]["scope"] == "private" and row["metadata"]["owner"] == f"user:{OWNER}"
    assert visible_to_viewer(row, f"user:{OWNER}") is True
    assert visible_to_viewer(row, f"user:{OTHER}") is False
    assert "oat milk" in _block(uuid.UUID(int=1), f"user:{OWNER}")
    assert "oat milk" not in _block(uuid.UUID(int=1), f"user:{OTHER}")


# ---------------------------------------------------------------------------
# The standing rules in every turn
# ---------------------------------------------------------------------------

def test_the_rule_the_owner_stated_is_in_the_next_turn_of_a_fresh_chat(durable):
    from modules.context.sections.memory import STANDING_RULES_HEADING

    assert _store(_said_by(OWNER))["success"] is True
    block = _block(uuid.UUID(int=1), f"user:{OWNER}")
    assert block.startswith(STANDING_RULES_HEADING) and f"- {RULE}" in block
    assert _block(uuid.UUID(int=1), f"user:{OTHER}") == ""          # never put to another as theirs
    assert _block(uuid.UUID(int=1), None) == ""                     # nor to a turn with no person


def test_another_workspaces_rule_is_never_read(durable):
    _store(_said_by(OWNER))                                          # workspace 1
    assert _block(uuid.UUID(int=2), f"user:{OWNER}") == ""


def test_reaching_the_scan_cap_is_logged(durable, monkeypatch, caplog):
    from config import config

    monkeypatch.setattr(config, "STANDING_RULES_SCAN_LIMIT", 2)
    for day in ("Monday", "Tuesday", "Wednesday"):
        _store(_said_by(OWNER, f"The roastery is closed on {day}s in December."))
    assert "closed on" in _block(uuid.UUID(int=1), f"user:{OWNER}")
    assert "STANDING_RULES_SCAN_LIMIT" in caplog.text


def test_the_durable_read_is_the_namespace_and_any_of_the_filter():
    from modules.memory.durable_store import DurableMemoryStore
    from modules.memory.injection_filter import STANDING_RULE_FILTER

    store = DurableMemoryStore.__new__(DurableMemoryStore)
    store._enabled, store._bootstrap_done, store._collection = True, True, "durable_memory"
    store._client = MagicMock()
    store._client.scroll = AsyncMock(return_value=([NS(id="p1", payload={"content": RULE, "metadata": {}})], None))
    rows = asyncio.run(store.get_where_any("mem:ws-1", STANDING_RULE_FILTER, limit=5,
                                           where={"metadata.owner": "user:7"}))
    assert [r["memory"] for r in rows] == [RULE]
    flt = store._client.scroll.call_args.kwargs["scroll_filter"]
    assert [(c.key, c.match.value) for c in flt.must] == [("namespace", "mem:ws-1"), ("metadata.owner", "user:7")]
    assert {c.key: list(c.match.any) for c in flt.should} == STANDING_RULE_FILTER


def test_what_is_a_standing_rule():
    from modules.memory.injection_filter import is_standing_rule

    said = {"metadata": {"source": "platform_tool", "owner": "user:7", "type": "business_fact"}}
    headless = {"metadata": {"source": "platform_tool", "type": "business_fact"}}     # a ticket's agent
    distilled_preference = {"metadata": {"source_type": "distilled", "type": "preference", "owner": "user:7"}}
    legacy_preference = {"metadata": {"category": "preference", "owner": "user:7"}}
    nobodys_preference = {"metadata": {"category": "preference"}}                    # no person behind it
    a_learning = {"metadata": {"type": "task_learning", "owner": "user:7"}}
    rows = (said, headless, distilled_preference, legacy_preference, nobodys_preference, a_learning)
    assert [is_standing_rule(m) for m in rows] == [True, False, True, True, False, False]


def test_a_widget_turn_carries_none_of_the_owners_rules(durable):
    _store(_said_by(OWNER))
    assert _block(uuid.UUID(int=1), None, widget_mode=True) == ""


def test_no_rule_renders_nothing(durable):
    from modules.context.sections.memory import render_standing_rules

    _store({"content": "Ticket #12 shipped", "source_type": "claude_reports"})       # no owner: no rule
    assert _block(uuid.UUID(int=1), f"user:{OWNER}") == ""
    assert render_standing_rules([], 600) == ""


def test_a_failed_read_is_logged_and_the_turn_goes_on(monkeypatch, caplog):
    import modules.context.sections.memory as section

    monkeypatch.setattr(section, "_stored_rule_rows", AsyncMock(side_effect=RuntimeError("qdrant down")))
    assert _block(uuid.UUID(int=1), f"user:{OWNER}") == ""
    assert "standing rules not read" in caplog.text


def test_the_cap_holds_newest_first():
    from core.context_guard import count_tokens
    from modules.context.sections.memory import render_standing_rules
    from modules.memory.injection_filter import standing_rules

    rows = [{"memory": f"Rule {i}: the Friday newsletter goes out before ten, with the week's roasts listed.",
             "created_at": f"2026-10-{1 + i // 24:02d}T{i % 24:02d}:00:00+00:00",
             "metadata": {"type": "preference", "owner": "user:7"}} for i in range(120)]
    block = render_standing_rules(standing_rules(rows, "user:7"), 600)
    assert count_tokens(block) <= 600
    lines = [ln for ln in block.splitlines() if ln.startswith("- ")]
    assert lines[0].startswith("- Rule 119:") and lines[1].startswith("- Rule 118:")
    assert 1 < len(lines) < 120
    giant = render_standing_rules(["word " * 5000], 600)
    assert count_tokens(giant) <= 600


def test_the_same_rule_twice_is_said_once():
    from modules.memory.injection_filter import standing_rules

    meta = {"type": "preference", "owner": "user:7"}
    rows = [{"memory": RULE, "created_at": "2026-10-07", "metadata": meta},
            {"memory": f"  {RULE.upper()} ", "created_at": "2026-10-06", "metadata": meta}]
    assert standing_rules(rows, "user:7") == [RULE]


def test_the_chat_mode_carries_the_section_whatever_the_intent():
    from modules.context.modes import MODE_CONFIGS, ContextMode
    from modules.context.sections import SECTION_REGISTRY
    from modules.context.sections.memory import StandingRulesSection

    assert "standing_rules" in MODE_CONFIGS[ContextMode.CHATBOT].sections
    assert SECTION_REGISTRY["standing_rules"] is StandingRulesSection
    assert StandingRulesSection().priority <= 2                     # the budget never drops it


def test_a_creation_turn_in_a_new_chat_has_the_rule_in_its_prompt(durable):
    """The whole assembly: the regex intent says CREATION (no memory), so recall is skipped,
    and the prompt still carries the rule stored in another chat."""
    from consumers.chatbot.intent_classifier import Intent, SmartIntentClassifier
    from modules.context.modes import ContextMode
    from modules.context.sections.platform_actions import PlatformActionsSection
    from modules.context.sections.tools import ToolsSection
    from modules.context.service import ContextService

    intent = SmartIntentClassifier().classify(CREATION_ASK)
    assert intent.primary_intent == Intent.CREATION and intent.requires_memory is False
    _store(_said_by(OWNER))

    db = MagicMock()                       # the chainable session test_context/test_service.py uses
    query = db.query.return_value
    query.join.return_value = query.filter.return_value = query.order_by.return_value = query
    query.all.return_value, query.first.return_value = [], None
    db.execute.return_value.scalars.return_value.all.return_value = []
    agent = NS(id=322, name="Auto", agent_type="assistant", description=None, use_custom_persona=False,
               custom_persona_prompt=None, persona=None, skills=[])
    with patch.object(PlatformActionsSection, "_build", return_value=""), \
            patch.object(ToolsSection, "load_tools", AsyncMock(return_value=([], "auto"))):
        result = asyncio.run(ContextService(db).build_context(
            mode=ContextMode.CHATBOT, agent=agent, workspace_id=str(uuid.UUID(int=1)),
            messages=[{"role": "user", "content": CREATION_ASK}],
            intent_result=intent, skip_memory=True, chat_id=str(uuid.uuid4()), query=CREATION_ASK,
            viewer_subject_id=f"user:{OWNER}",
        ))
    assert RULE in result.system_prompt
    assert "standing_rules" in result.sections_included


def test_the_atom_lane_carries_the_rules_too(durable):
    from consumers.chatbot.atom_prompt import atom_memory_block

    _store(_said_by(OWNER))
    block = asyncio.run(atom_memory_block(None, [{"role": "user", "content": "hi"}], workspace_id=uuid.UUID(int=1),
                                          agent_id=322, widget_mode=False, viewer_subject_id=f"user:{OWNER}"))
    assert RULE in block
    assert asyncio.run(atom_memory_block(None, [{"role": "user", "content": "hi"}], workspace_id=uuid.UUID(int=1),
                                         agent_id=322, widget_mode=True, viewer_subject_id=None)) == ""


# ---------------------------------------------------------------------------
# P256-FIX-RVW-11: the owner's rule is found however many rows the workspace holds, and a
# restated rule is the restating person's (the real DurableMemoryStore over an in-memory Qdrant)
# ---------------------------------------------------------------------------

def _payload_value(payload, key):
    for part in key.split("."):
        payload = payload.get(part) if isinstance(payload, dict) else None
    return payload


def _matches(payload, condition):
    value, match = _payload_value(payload, condition.key), condition.match
    return value in match.any if hasattr(match, "any") else value == match.value


class _InMemoryQdrant:
    """Qdrant's scroll (a filter's must and should, in point order, paged) and upsert."""

    def __init__(self) -> None:
        self.points: list = []

    async def upsert(self, collection_name, points):
        self.points.extend(NS(id=p.id, payload=p.payload) for p in points)

    async def set_payload(self, collection_name, payload, points):
        for point in self.points:
            if point.id in points:
                point.payload = {**point.payload, **payload}

    async def scroll(self, collection_name, scroll_filter, limit, offset=None, **_kwargs):
        hits = [p for p in self.points if all(_matches(p.payload, c) for c in scroll_filter.must or [])
                and (not scroll_filter.should or any(_matches(p.payload, c) for c in scroll_filter.should))]
        start = offset or 0
        page = hits[start:start + limit]
        return page, (start + limit if start + limit < len(hits) else None)


@pytest.fixture
def qdrant(monkeypatch):
    """The real DurableMemoryStore behind the service the handler writes and the section reads."""
    import modules.memory.unified_memory_service as ums
    from config import config
    from modules.memory.durable_store import DurableMemoryStore

    client = _InMemoryQdrant()
    store = DurableMemoryStore.__new__(DurableMemoryStore)
    store._enabled, store._bootstrap_done, store._collection = True, True, "durable_memory"
    store._client = client
    store._embedder = NS(generate_embedding=AsyncMock(return_value=[0.0]))
    service = ums.UnifiedMemoryService.__new__(ums.UnifiedMemoryService)
    service._durable = store
    service._redis_client_getter = lambda: None
    monkeypatch.setattr(ums, "get_unified_memory_service", lambda: service)
    monkeypatch.setattr(config, "QDRANT_URL", "http://qdrant.test")
    return client


def _seed(client, content, metadata):
    """A row already in the store, as an earlier write left it."""
    import hashlib

    client.points.append(NS(id=str(uuid.uuid4()), payload={
        "namespace": f"mem:{uuid.UUID(int=1)}", "workspace_id": str(uuid.UUID(int=1)), "content": content,
        "metadata": metadata, "created_at": "2026-10-06T21:00:00+00:00",
        "content_hash": hashlib.sha256(content.encode()).hexdigest(),
    }))


def test_the_owners_rule_is_read_past_600_other_rows(qdrant):
    from config import config

    assert config.STANDING_RULES_SCAN_LIMIT < 600
    for i in range(300):
        _seed(qdrant, f"Member {i} prefers the dark roast.", {"source": "platform_tool", "owner": f"user:{100 + i}"})
        _seed(qdrant, f"Ticket #{i} shipped on time.", {"source": "platform_tool", "type": "business_fact"})
    assert _store(_said_by(OWNER))["success"] is True                   # written after all 600
    block = _block(uuid.UUID(int=1), f"user:{OWNER}")
    assert f"- {RULE}" in block and "dark roast" not in block


def test_the_owner_restating_a_rule_stored_ownerless_makes_it_theirs(qdrant):
    _seed(qdrant, RULE, {"source": "platform_tool", "type": "business_fact"})   # night 12's ownerless row
    assert _block(uuid.UUID(int=1), f"user:{OWNER}") == ""
    assert _store(_said_by(OWNER))["success"] is True
    assert len(qdrant.points) == 2                                                # a row of the owner's own
    assert f"- {RULE}" in _block(uuid.UUID(int=1), f"user:{OWNER}")
    assert "owner" not in qdrant.points[0].payload["metadata"]                    # the old row is left as it was


def test_another_members_identical_rule_is_theirs_not_the_owners(qdrant):
    assert _store(_said_by(OTHER))["success"] is True
    assert f"- {RULE}" in _block(uuid.UUID(int=1), f"user:{OTHER}")
    assert _block(uuid.UUID(int=1), f"user:{OWNER}") == ""
    assert _store(_said_by(OWNER))["success"] is True                  # the owner's own row, not OTHER's
    assert f"- {RULE}" in _block(uuid.UUID(int=1), f"user:{OWNER}")
    owners = sorted(p.payload["metadata"]["owner"] for p in qdrant.points)
    assert owners == [f"user:{OWNER}", f"user:{OTHER}"]


def test_the_same_person_restating_a_rule_still_dedupes(qdrant):
    assert _store(_said_by(OWNER))["success"] is True and _store(_said_by(OWNER))["success"] is True
    assert len(qdrant.points) == 1
    assert _block(uuid.UUID(int=1), f"user:{OWNER}").count(RULE) == 1


# ---------------------------------------------------------------------------
# P256-FIX-RVW-41: a restated rule is the owner's newest (it leads the block, and beats the
# rule it replaced), however old its first statement
# ---------------------------------------------------------------------------

@pytest.fixture
def clock(monkeypatch):
    """The store's clock, a minute on at every read, so each write is dated after the last."""
    import modules.memory.durable_store as durable_store

    ticks = iter(range(10_000))
    start = datetime(2026, 10, 1, 9, tzinfo=timezone.utc)

    class _Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return start + timedelta(minutes=next(ticks))

    monkeypatch.setattr(durable_store, "datetime", _Clock)


def _rule_lines(viewer=f"user:{OWNER}"):
    return [ln[2:] for ln in _block(uuid.UUID(int=1), viewer).splitlines() if ln.startswith("- ")]


def test_a_rule_restated_after_routine_saves_is_back_in_the_block_and_first(qdrant, clock):
    assert _store(_said_by(OWNER))["success"] is True
    for i in range(30):
        routine = f"Order #{1000 + i} for the Harvest Club went out with two bags of the Ethiopian " \
                  f"natural and a tasting card; the customer asked for a grinder recommendation."
        assert 130 <= len(routine) <= 170
        assert _store(_said_by(OWNER, routine))["success"] is True
    assert RULE not in _rule_lines()                                  # pushed out by the routine saves
    assert _store(_said_by(OWNER))["success"] is True                 # said again, in the same words
    assert len(qdrant.points) == 31                                   # the same row, re-dated
    assert _rule_lines()[0] == RULE


def test_tuesdays_thursdays_tuesdays_leaves_tuesdays_newest(qdrant, clock):
    tuesdays, thursdays = "We post the newsletter on Tuesdays.", "We post the newsletter on Thursdays."
    for said in (tuesdays, thursdays, tuesdays):
        assert _store(_said_by(OWNER, said))["success"] is True
    assert len(qdrant.points) == 2
    lines = _rule_lines()
    assert lines.index(tuesdays) < lines.index(thursdays)


def test_an_ownerless_dedupe_keeps_its_date(qdrant, clock):
    _seed(qdrant, RULE, {"source": "claude_reports"})
    assert _store({"content": RULE, "source_type": "claude_reports"})["success"] is True
    assert len(qdrant.points) == 1 and qdrant.points[0].payload["created_at"] == "2026-10-06T21:00:00+00:00"


def test_a_re_date_that_times_out_keeps_the_row_and_says_so(qdrant, clock, monkeypatch, caplog):
    import httpx

    assert _store(_said_by(OWNER))["success"] is True
    monkeypatch.setattr(qdrant, "set_payload", AsyncMock(side_effect=httpx.ReadTimeout("slow")))
    assert _store(_said_by(OWNER))["success"] is True                 # the rule is stored, once
    assert len(qdrant.points) == 1 and "restated but not re-dated" in caplog.text
