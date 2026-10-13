"""P256-FIX-N14-4 (night 14, F418): every standing rule is the owner's, in their words, and theirs to delete.

Night 14: 'When you tell me to stop querying the database, I stop for that turn' sat in the
standing rules (Auto wrote it on the owner's turn, so it had an owner and counted), cut two
answers short, and both deletes failed ('Memory not found or not owned by this workspace'): Auto's
delete looked for the id among the first 500 rows of the workspace namespace, the Memory page's in
that one namespace, and the page, showing the first rows in scroll order, did not list the rule.

- A memory is a standing rule only in the owner's own words (each sentence is one of their own
  messages in the chat): ``store_memory`` marks it ``stated_by: owner``; what Auto composed is
  kept and recalled, never a rule.
- The Memory page (/browse) and platform_browse_memories list the viewer's standing rules first,
  from the read the prompt's block makes, with the ids delete takes.
- Delete (the page's and Auto's) reads the point's own namespace: a listed rule deletes, past 500
  other rows and from an agent's namespace too, and the next prompt no longer reads it; another
  workspace's memory is not found; another member's private rule is not theirs to delete.

The real DurableMemoryStore over an in-memory Qdrant; the owner's chat messages are given
(``said_instructions.chat_owner_turns``, which reads the chat's user turns, is N13-7's).
"""
from __future__ import annotations

import asyncio
import inspect
import uuid
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock

import pytest

WS, OTHER_WS = uuid.UUID(int=1), uuid.UUID(int=2)
OWNER, MEMBER = 7, 8
RULE = "From November we post on Tuesdays, not Mondays."
SAID = f"{RULE} Remember that please."
F418 = "When you tell me to stop querying the database, I stop for that turn."
F418_SAID = "Stop querying the database, just answer from the notes."


# ---------------------------------------------------------------------------
# An in-memory Qdrant and the one memory service over it
# ---------------------------------------------------------------------------

def _value(payload, key):
    for part in key.split("."):
        payload = payload.get(part) if isinstance(payload, dict) else None
    return payload


def _matches(payload, condition):
    value, match = _value(payload, condition.key), condition.match
    return value in match.any if hasattr(match, "any") else value == match.value


class _Qdrant:
    """Qdrant's scroll (must and should, paged), upsert, set_payload, retrieve and delete."""

    def __init__(self) -> None:
        self.points: list = []
        self.retrieved: list = []

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
        return hits[start:start + limit], (start + limit if start + limit < len(hits) else None)

    async def retrieve(self, collection_name, ids, **_kwargs):
        self.retrieved.extend(ids)
        return [p for p in self.points if p.id in ids]

    async def delete(self, collection_name, points_selector):
        self.points = [p for p in self.points if p.id not in points_selector]


@pytest.fixture
def qdrant(monkeypatch):
    import modules.memory.unified_memory_service as ums
    from config import config
    from modules.memory.durable_store import DurableMemoryStore

    client = _Qdrant()
    store = DurableMemoryStore.__new__(DurableMemoryStore)
    store._enabled, store._bootstrap_done, store._collection = True, True, "durable_memory"
    store._client = client
    store._embedder = NS(generate_embedding=AsyncMock(return_value=[0.0]))
    service = ums.UnifiedMemoryService.__new__(ums.UnifiedMemoryService)
    service._durable = store
    service._redis_client_getter = lambda: None
    monkeypatch.setattr(ums, "get_unified_memory_service", lambda: service)
    monkeypatch.setattr(config, "QDRANT_URL", "http://qdrant.test")
    client.service = service
    return client


@pytest.fixture
def owner_says(monkeypatch):
    """What the owner's messages in the chat say (newest first), as the store reads them."""
    import modules.tools.discovery.said_instructions as said

    messages: list = []
    monkeypatch.setattr(said, "chat_owner_turns", lambda db, workspace_id, params: list(messages))
    return messages


def _store(content, *, by=OWNER, workspace=WS, **extra):
    """platform_store_memory on a local turn ``by`` drives (the executor's injection), or no one's."""
    from modules.tools.discovery.handlers_workspace import store_memory

    params = {"content": content, **({"_driving_user_id": by} if by is not None else {}), **extra}
    return asyncio.run(store_memory(MagicMock(), workspace, params))


def _block(viewer=OWNER, workspace=WS):
    from modules.context.sections.memory import standing_rules_block

    return asyncio.run(standing_rules_block(workspace, viewer_subject_id=f"user:{viewer}", widget_mode=False))


def _seed(client, content, metadata, *, namespace=None, workspace=WS):
    point_id = str(uuid.uuid4())
    client.points.append(NS(id=point_id, payload={
        "namespace": namespace or f"mem:{workspace}", "workspace_id": str(workspace), "content": content,
        "metadata": metadata, "created_at": "2026-10-12T21:00:00+00:00", "content_hash": point_id,
    }))
    return point_id


def _row(client, content):
    return next(p for p in client.points if p.payload["content"] == content)


# ---------------------------------------------------------------------------
# A standing rule only from the owner's own words
# ---------------------------------------------------------------------------

def test_a_rule_from_the_owners_chat_is_a_standing_rule(qdrant, owner_says):
    owner_says.append(SAID)
    answer = _store(RULE)
    assert answer["success"] is True and answer["standing_rule"] is True
    assert _row(qdrant, RULE).payload["metadata"]["stated_by"] == "owner"
    assert f"- {RULE}" in _block()


def test_a_memory_auto_composed_about_itself_is_not_a_standing_rule(qdrant, owner_says):
    owner_says.append(F418_SAID)
    answer = _store(F418)
    assert answer["success"] is True and answer["standing_rule"] is False
    row = _row(qdrant, F418)
    assert row.payload["metadata"]["owner"] == f"user:{OWNER}" and "stated_by" not in row.payload["metadata"]
    assert _block() == ""


def test_an_owned_memory_with_no_owner_words_in_the_chat_is_not_a_rule(qdrant, owner_says):
    assert _store(RULE)["standing_rule"] is False                     # nothing the owner said holds it
    assert _block() == ""


def test_a_turn_no_person_drives_never_writes_a_rule(qdrant, owner_says):
    owner_says.append(SAID)
    assert _store(RULE, by=None)["standing_rule"] is False            # a ticket's agent, a heartbeat
    assert "stated_by" not in _row(qdrant, RULE).payload["metadata"]


def test_a_rule_with_a_sentence_auto_added_is_not_the_owners(qdrant, owner_says):
    owner_says.append(SAID)
    assert _store(f"{RULE} I will remind you every Monday morning.")["standing_rule"] is False


def test_a_model_sent_marker_is_never_taken(qdrant, owner_says):
    from modules.tools.discovery.memory_write import memory_metadata

    owner_says.append(F418_SAID)                                       # the owner said something near it
    assert _store(F418, stated_by="owner", _said_by_owner=True)["standing_rule"] is False
    assert "stated_by" not in _row(qdrant, F418).payload["metadata"]
    params = {"content": F418, "stated_by": "owner", "metadata": {"stated_by": "owner"}}
    assert "stated_by" not in memory_metadata(MagicMock(), str(WS), params, owner=f"user:{OWNER}")


def test_the_answer_says_a_composed_memory_is_not_the_owners_rule(qdrant, owner_says):
    from modules.tools.discovery.memory_write import NOT_THE_OWNERS_RULE

    owner_says.append(F418_SAID)
    assert _store(F418)["rule_note"] == NOT_THE_OWNERS_RULE
    assert "rule_note" not in _store("Ticket #12 shipped.", by=None)            # nobody's: nothing to say
    owner_says.append(SAID)
    assert "rule_note" not in _store(RULE)


def test_no_request_can_store_a_memory_as_someone_elses_rule(monkeypatch):
    """The workspace memory POST takes the caller's metadata: the owner and the marker are the server's."""
    from api import widget_memory
    from modules.memory.injection_filter import without_server_keys

    stored = {}

    async def store_long_term(workspace_id, content, metadata=None, **_kwargs):
        stored["metadata"] = metadata
        return {"success": True, "id": str(uuid.uuid4())}

    monkeypatch.setattr(widget_memory, "_get_memory_service", lambda: NS(store_long_term=store_long_term))
    forged = {"owner": f"user:{OWNER}", "stated_by": "owner", "topic": "posting"}
    body = widget_memory.MemoryCreate(content=RULE, metadata=forged)
    asyncio.run(widget_memory.store_memory(body=body, ctx=NS(workspace_id=WS)))
    assert stored["metadata"] == {"topic": "posting"}
    assert without_server_keys(None) == {} and forged["owner"] == f"user:{OWNER}"   # the caller's dict is untouched


def test_what_auto_wrote_becomes_the_owners_rule_when_they_say_it_in_those_words(qdrant, owner_says):
    assert _store(RULE)["standing_rule"] is False                     # Auto's note first
    owner_says.append(SAID)
    assert _store(RULE)["success"] is True                            # the owner says it: the same row
    assert len(qdrant.points) == 1 and _row(qdrant, RULE).payload["metadata"]["stated_by"] == "owner"
    assert f"- {RULE}" in _block()


def test_in_their_words_needs_every_sentence_wholly_theirs():
    from modules.tools.discovery.memory_write import STANDING_RULE_COVERAGE
    from modules.tools.discovery.rule_wording import in_their_words

    assert in_their_words(RULE, [SAID], STANDING_RULE_COVERAGE) is True
    assert in_their_words(F418, [F418_SAID], STANDING_RULE_COVERAGE) is False
    assert in_their_words(f"{RULE} Noted.", [SAID], STANDING_RULE_COVERAGE) is False   # a word they never said
    assert in_their_words(f"{RULE} !", [SAID], STANDING_RULE_COVERAGE) is True         # no word at all
    assert in_their_words(RULE, [], STANDING_RULE_COVERAGE) is False
    assert in_their_words("", [SAID], STANDING_RULE_COVERAGE) is False


@pytest.mark.parametrize("stored, said", [
    ("We post on Tuesdays.", "We do not post on Tuesdays any more."),                 # the no dropped
    ("We don't post on Tuesdays.", "We post on Tuesdays."),                            # a no added
    ("From November we post on Mondays, not Tuesdays.", RULE),                         # turned round
    ("Never send the newsletter before ten.", "Always send the newsletter before ten."),
    ("Send the newsletter after ten.", "Send the newsletter before ten."),            # one word changed
    ("Post the club boxes at 10 on Wednesdays.", "Post the club boxes at 11 on Wednesdays."),
    ("We post on Tuesdays.", "Do we post on Tuesdays?"),                               # a question
    ("Post on Tuesdays.", "From November, once the new courier starts and the club grows past forty "
                          "members, we will post the club boxes on Tuesdays."),        # too little of it
    ("Send it. " + "word " * 81, "Send it. " + "word " * 81),                         # past the cap
    ("Post on Tuesdays.", "Only post on Tuesdays."),                                   # a condition dropped
    ("Post on Tuesdays.", "Post on Tuesdays unless it rains."),
    ("We post on Tuesdays.", "Do we post on Tuesdays?!"),
    ("Post on Tuesdays. لا تنشر يوم الأحد", "Post on Tuesdays."),                       # a clause in another script
    ("Post on Tuesdays 🚫", "Post on Tuesdays."),                                       # what cannot be weighed
])
def test_a_rule_that_turns_the_owners_words_round_is_not_theirs(stored, said):
    from modules.tools.discovery.memory_write import STANDING_RULE_COVERAGE
    from modules.tools.discovery.rule_wording import in_their_words

    assert in_their_words(stored, [said], STANDING_RULE_COVERAGE) is False


def test_a_no_said_either_way_is_the_same_no():
    from modules.tools.discovery.memory_write import STANDING_RULE_COVERAGE
    from modules.tools.discovery.rule_wording import in_their_words

    assert in_their_words("Do not post on Tuesdays.", ["Please don\u2019t post on Tuesdays."], STANDING_RULE_COVERAGE)
    assert in_their_words("Post on Tuesdays.", ["Remember: post on Tuesdays. Don't forget."], STANDING_RULE_COVERAGE)
    assert in_their_words("We cannot post on Sundays.", ["We can't post on Sundays."], STANDING_RULE_COVERAGE)
    assert in_their_words("From November we post on Tuesdays.",
                          ["Remember that from November we post on Tuesdays."], STANDING_RULE_COVERAGE)
    assert in_their_words("Ｐost on Tuesdays, café first.", ["post on tuesdays, café first"], STANDING_RULE_COVERAGE)
    assert in_their_words("ننشر يوم الثلاثاء.", ["تذكر: ننشر يوم الثلاثاء."], STANDING_RULE_COVERAGE)
    assert in_their_words("हम मंगलवार को पोस्ट करते हैं।", ["हम मंगलवार को पोस्ट करते हैं।"], STANDING_RULE_COVERAGE)


def test_the_checks_split_out_of_store_memory_refuse_as_before(qdrant):
    from modules.tools.discovery.memory_write import memory_params_refusal

    assert "source_type must be one of" in memory_params_refusal({"content": "x", "source_type": "owner"})
    assert memory_params_refusal({"content": "x"}) is None
    assert _store("x", confidence=2)["error"] == "confidence must be a number between 0 and 1"
    assert "type must be one of" in _store("x", type="rule")["error"]
    assert "scope must be one of" in _store("x", scope="global")["error"]
    refused = _store("the password is hunter2")
    assert "exclusion policy" in refused["error"] and "hunter2" not in refused["error"]
    assert qdrant.points == []


# ---------------------------------------------------------------------------
# The Memory page lists the rules the prompt reads, and deletes them
# ---------------------------------------------------------------------------

def _page_ctx(viewer=OWNER, workspace=WS):
    return NS(workspace_id=workspace, user=NS(id=viewer))


def _browse_page(qdrant, monkeypatch, *, viewer=OWNER, **filters):
    """GET /api/v1/memory/browse with the first page of rows not reaching the rule (night 14)."""
    from api import memory_stats

    others = [("global", {"id": str(uuid.uuid4()), "memory": f"Ticket #{i} shipped.", "created_at": "2026-10-13"})
              for i in range(20)]
    qdrant.service.list_short_term_by_type = AsyncMock(return_value=[])
    monkeypatch.setattr(memory_stats, "_get_memory_service", lambda: qdrant.service)
    monkeypatch.setattr(memory_stats, "_get_agent_ids", lambda workspace_id, db: [])
    monkeypatch.setattr(memory_stats, "_fetch_all_scoped_memories", AsyncMock(return_value=others))
    args = {"query": None, "limit": 20, "content_type": None, "tier": None, **filters}
    return asyncio.run(memory_stats.browse_memories(**args, ctx=_page_ctx(viewer), db=MagicMock()))


def _page_delete(qdrant, monkeypatch, memory_id, *, viewer=OWNER, workspace=WS):
    from api import memory_stats

    monkeypatch.setattr(memory_stats, "_get_memory_service", lambda: qdrant.service)
    return asyncio.run(memory_stats.delete_memory(memory_id, ctx=_page_ctx(viewer, workspace), db=MagicMock()))


def test_the_memory_page_lists_the_owners_rule_first_and_deletes_it(qdrant, owner_says, monkeypatch):
    owner_says.append(SAID)
    _store(RULE)
    page = _browse_page(qdrant, monkeypatch)
    first = page["memories"][0]
    assert first["content"] == RULE and first["standing_rule"] is True and page["total"] == 21
    assert _page_delete(qdrant, monkeypatch, first["id"]) == {"success": True,
                                                              "message": f"Memory {first['id']} deleted"}
    assert _block() == ""                                              # the next prompt no longer reads it
    assert _browse_page(qdrant, monkeypatch)["memories"][0]["content"] != RULE


def test_the_page_lists_only_the_viewers_rules_and_only_unfiltered(qdrant, owner_says, monkeypatch):
    owner_says.append(SAID)
    _store(RULE)
    assert all(not m.get("standing_rule") for m in _browse_page(qdrant, monkeypatch, viewer=MEMBER)["memories"])
    assert all(not m.get("standing_rule") for m in _browse_page(qdrant, monkeypatch, query="post")["memories"])
    assert all(not m.get("standing_rule") for m in _browse_page(qdrant, monkeypatch, tier="l2")["memories"])


def test_a_rule_the_page_already_shows_is_listed_once():
    from api.memory_standing_rules import with_rules_first

    rule = {"id": "r1", "memory": RULE, "metadata": {}, "created_at": "2026-10-13"}
    page = {"success": True, "memories": [{"id": "a"}, {"id": "r1", "content": RULE}], "total": 2}
    listed = with_rules_first(page, [rule])
    assert [m["id"] for m in listed["memories"]] == ["r1", "a"] and listed["total"] == 2
    assert page["memories"][0] == {"id": "a"}                         # the page given is not changed


def test_a_failed_rules_read_leaves_the_page_and_says_so(qdrant, monkeypatch, caplog):
    import modules.context.sections.memory as section

    monkeypatch.setattr(section, "_stored_rule_rows", AsyncMock(side_effect=RuntimeError("qdrant down")))
    page = _browse_page(qdrant, monkeypatch)
    assert page["success"] is True and len(page["memories"]) == 20 and page["standing_rules_unavailable"] is True
    assert "standing rules not listed" in caplog.text


def test_fastapi_still_reads_the_browse_routes_parameters():
    from api import memory_stats

    route = next(r for r in memory_stats.router.routes if r.path.endswith("/browse"))
    assert {p.name for p in route.dependant.query_params} >= {"query", "limit", "content_type", "tier"}
    assert {d.name for d in route.dependant.dependencies} >= {"ctx", "db"}
    assert inspect.iscoroutinefunction(route.endpoint)


# ---------------------------------------------------------------------------
# Delete reads the namespace the memory was stored under
# ---------------------------------------------------------------------------

def test_a_rule_past_500_other_rows_is_listed_and_deleted_by_auto(qdrant, owner_says, monkeypatch):
    """The night's failure: Auto's delete looked for the id among the first 500 rows."""
    from modules.tools.discovery.handlers_search import browse_memories, delete_memory

    for i in range(520):
        _seed(qdrant, f"Ticket #{i} shipped on time.", {"source": "platform_tool", "type": "business_fact"})
    owner_says.append(SAID)
    _store(RULE)
    params = {"_driving_user_id": OWNER}
    listed = asyncio.run(browse_memories(MagicMock(), WS, params))["memories"]
    assert listed[0]["content"] == RULE and listed[0]["standing_rule"] is True
    assert sum(1 for m in listed if m["content"] == RULE) == 1
    answer = asyncio.run(delete_memory(MagicMock(), WS, {"memory_id": listed[0]["id"], **params}))
    assert answer["success"] is True and _block() == ""
    assert len(qdrant.points) == 520


def test_auto_browses_without_the_rules_when_they_cannot_be_read(qdrant, owner_says, monkeypatch, caplog):
    import modules.context.sections.memory as section
    from modules.tools.discovery.handlers_search import browse_memories

    _seed(qdrant, "Ticket #1 shipped on time.", {"type": "business_fact"})
    monkeypatch.setattr(section, "_stored_rule_rows", AsyncMock(side_effect=RuntimeError("qdrant down")))
    answer = asyncio.run(browse_memories(MagicMock(), WS, {"_driving_user_id": OWNER}))
    assert answer["success"] is True and answer["standing_rules_unavailable"] is True
    assert [m["content"] for m in answer["memories"]] == ["Ticket #1 shipped on time."]
    assert "standing rules not read for platform_browse_memories" in caplog.text


def test_auto_browsing_for_no_person_lists_no_rules_first(qdrant, owner_says):
    from modules.tools.discovery.handlers_search import browse_memories

    owner_says.append(SAID)
    _store(RULE)
    nobody = asyncio.run(browse_memories(MagicMock(), WS, {}))["memories"]
    assert nobody and not any(m.get("standing_rule") for m in nobody)


def test_a_memory_in_an_agents_namespace_deletes_by_the_pages_id(qdrant, monkeypatch):
    agents = _seed(qdrant, "OPS keeps the courier sheet.", {"type": "task_learning"}, namespace=f"mem:{WS}:agent:5")
    assert _page_delete(qdrant, monkeypatch, agents)["success"] is True and qdrant.points == []


def test_another_workspaces_memory_is_not_found_and_stays(qdrant, monkeypatch):
    theirs = _seed(qdrant, RULE, {"owner": f"user:{OWNER}", "stated_by": "owner"}, workspace=OTHER_WS)
    answer = _page_delete(qdrant, monkeypatch, theirs)
    assert answer == {"success": False, "error": "Memory not found or not owned by this workspace"}
    assert len(qdrant.points) == 1


def test_another_members_private_rule_is_theirs_to_delete(qdrant, monkeypatch):
    from modules.tools.discovery.handlers_search import delete_memory

    private = _seed(qdrant, RULE, {"owner": f"user:{MEMBER}", "scope": "private", "stated_by": "owner"})
    refused = asyncio.run(delete_memory(MagicMock(), WS, {"memory_id": private, "_driving_user_id": OWNER}))
    assert refused["success"] is False and "only they can delete it" in refused["error"]
    assert asyncio.run(delete_memory(MagicMock(), WS, {"memory_id": private}))["success"] is False   # no person
    assert len(qdrant.points) == 1
    assert _page_delete(qdrant, monkeypatch, private, viewer=MEMBER)["success"] is True


def test_an_id_that_is_no_memory_is_not_found_without_asking_qdrant(qdrant, monkeypatch):
    from modules.tools.discovery.handlers_search import delete_memory

    answer = asyncio.run(delete_memory(MagicMock(), WS, {"memory_id": "abc-123", "_driving_user_id": OWNER}))
    assert answer == {"success": False, "error": "Memory not found or not owned by this workspace"}
    assert qdrant.retrieved == []


def test_who_may_delete_a_memory():
    from modules.memory.injection_filter import deletable_by

    private = {"metadata": {"scope": "private", "owner": "user:7"}}
    assert deletable_by(private, "user:7") is True
    assert deletable_by(private, "user:8") is False and deletable_by(private, None) is False
    assert deletable_by({"metadata": {"scope": "workspace", "owner": "user:7"}}, "user:8") is True
    assert deletable_by({"metadata": {"scope": "private"}}, None) is True     # an ownerless row, nobody's
    assert deletable_by({}, None) is True


def test_the_memory_tools_are_told_who_asks():
    from modules.tools.discovery.platform_executor import _DRIVER_AWARE_ACTIONS

    assert {"platform_browse_memories", "platform_delete_memory"} <= set(_DRIVER_AWARE_ACTIONS)
