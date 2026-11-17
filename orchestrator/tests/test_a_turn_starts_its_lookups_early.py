"""Voice latency D-2: a turn starts its two late lookups once its agent is active.

With ``CHAT_EARLY_LOOKUPS`` on, the connected-apps lookup and the knowledge prefetch start as
tasks when the turn's agent is active (``early_lookups.once_the_agent_is_active``, on
``_load_agent_context``) and are collected where each used to run
(``the_turns_app_lookup``, ``the_turns_prefetch``). What is pinned here:

* off by default, and off means the turn passes straight through: its arguments are not read;
* which lookups a turn will ask for, each skip copied from the turn's own;
* through the real ``_inject_composio_tools`` and ``_retrieval_first``, the turn is given the
  same messages, tools, frames, ``prefetched`` and retrieval record early as in place, and
  each lookup ran once;
* an early lookup that failed, raised, was cancelled or was asked for something else is made
  again in place, once; one that gave up in time is used as it is;
* the early prefetch never reads on the turn's session, and closes every session it opens;
* the turn's own stop is raised at the join; when the turn ends, however it ends, its early
  lookups are cancelled and nothing is left unread;
* an early lookup books to the turn's ask and adds its own time to the turn's clock.

Fakes only: no SDK, no network, no database.
"""
from __future__ import annotations

import asyncio
import gc
import json
import logging
import pathlib
import sys
from types import SimpleNamespace as NS

import pytest

from config import config
from consumers.chatbot import early_lookups as el
from consumers.chatbot import knowledge_prefetch as kp
from core.composio.off_loop import ComposioLookupTimeout
from tests.helpers_early_lookups import (
    AGENT, ASK, CHAT_ID, HERE, HINTS_STEP, LIST_AGENTS, LLM_MESSAGES, MESSAGES, RUNTIME, SEARCH_ARGS, SEARCH_STEP,
    SEND_EMAIL, THREAD, TOOLS, WS, World, a_chat, a_turn, a_wrapped_turn, an_ask, drain, given_in_place, late_steps,
    only_the_first_waits, said, the_early_lookups_end, tools_found, until,
)

ORCHESTRATOR = pathlib.Path(__file__).resolve().parents[1]
SURFACE = ORCHESTRATOR / "reports" / "config-surface.json"
EARLY_LOGGER = el.logger.name
INSTRUCTION = "Create an agent called OPS"
EARLY_PATH = ("consumers/chatbot/knowledge_prefetch.py", "services/draft_guides.py", "modules/rag/service.py",
              "modules/rag/owner_leg.py", "services/contested_facts.py")
# What the turn sets in its context between activation and the prefetch's old place.
SET_AFTER_ACTIVATION = ("_narrowed_actions", "turn_narrowing", "_PREFETCHED", "READ_AHEAD", "TRACKERS_MADE")


@pytest.fixture
def world(monkeypatch):
    """Both lookups, faked and counted (``tests.helpers_early_lookups.World``), with the switch on."""
    return World().install(monkeypatch)


# --------------------------------------------------------------------------- the switch

def test_the_switch_is_off_by_default_and_named_in_the_config_surface():
    # The class holds what the environment gave at import; tests patch the instance.
    assert type(config).CHAT_EARLY_LOOKUPS is False
    assert "CHAT_EARLY_LOOKUPS" in json.loads(SURFACE.read_text())["settings"]


def test_with_the_switch_off_the_turn_passes_straight_through(world, monkeypatch):
    read = []

    def _never(*args, **kwargs):
        read.append(args)
        raise AssertionError("nothing of the early half runs while the switch is off")

    async def nothing(chat):
        return None

    monkeypatch.setattr(config, "CHAT_EARLY_LOOKUPS", False)
    monkeypatch.setattr(el, "inspect", NS(signature=_never))
    monkeypatch.setattr(el, "start", _never)
    chat = a_chat(world)

    frames = asyncio.run(drain(a_wrapped_turn(nothing)(chat, CHAT_ID, MESSAGES, AGENT, 1)))

    assert frames == ["chat-id", "answered"]
    assert read == []
    assert not hasattr(chat, "_turn_ask") and not hasattr(chat, "_early")
    assert (world.lookups, world.searches, world.counts, world.sessions) == ([], [], [], [])


def test_the_wrapper_passes_the_turns_arguments_and_frames_through(world, monkeypatch):
    @el.starts_its_lookups_early
    async def turn(chat, chat_id, messages, agent_id, user_id, skip_composio=False, complexity_assessment=None,
                   force_text_only=False, **more):
        yield (chat_id, messages, agent_id, user_id, skip_composio, more)
        yield "second"

    async def frames_of(switch):
        monkeypatch.setattr(config, "CHAT_EARLY_LOOKUPS", switch)
        return await drain(turn(a_chat(world), CHAT_ID, MESSAGES, AGENT, 1, skip_composio=True, spoken_mode=True))

    expected = [(CHAT_ID, MESSAGES, AGENT, 1, True, {"spoken_mode": True}), "second"]
    assert asyncio.run(frames_of(False)) == expected
    assert asyncio.run(frames_of(True)) == expected


@pytest.mark.parametrize("switch", [False, True])
def test_closing_the_wrapper_closes_the_turn_inside_it(world, monkeypatch, switch):
    monkeypatch.setattr(config, "CHAT_EARLY_LOOKUPS", switch)
    closed = []

    @el.starts_its_lookups_early
    async def turn(chat, chat_id, messages, agent_id, user_id, skip_composio=False, complexity_assessment=None,
                   force_text_only=False):
        try:
            yield "first"
            yield "second"
        finally:
            closed.append(True)

    async def go():
        frames = turn(a_chat(world), CHAT_ID, MESSAGES, AGENT, 1)
        first = await frames.__anext__()
        await frames.aclose()
        return first, list(closed)

    assert asyncio.run(go()) == ("first", [True])


def test_a_turn_whose_arguments_cannot_be_read_runs_with_no_early_lookup(world, monkeypatch, caplog):
    def _unreadable(turn):
        raise TypeError("no signature")

    started, seen = [], []

    async def notes_what_started(chat):
        seen.append(chat._early)

    monkeypatch.setattr(el, "inspect", NS(signature=_unreadable))
    monkeypatch.setattr(el, "start", lambda *args: started.append(args))
    chat = a_chat(world)

    with caplog.at_level(logging.INFO, logger=EARLY_LOGGER):
        frames = asyncio.run(drain(a_wrapped_turn(notes_what_started)(chat, CHAT_ID, MESSAGES, AGENT, 1)))

    assert frames == ["chat-id", "answered"]
    assert seen == [None] and started == []
    errors = [r for r in caplog.records if r.name == EARLY_LOGGER and r.levelno == logging.ERROR]
    assert [r.getMessage() for r in errors] == [
        "[early-lookups] could not read the turn's arguments: no early lookup this turn"]
    assert errors[0].exc_info is not None
    assert (world.lookups, world.searches, world.counts) == ([], [], [])


def test_the_real_turns_arguments_are_read_by_name():
    from consumers.chatbot.service import StreamingChatService

    turn = StreamingChatService._stream_response_with_agent_scoped
    while hasattr(turn, "__wrapped__"):
        turn = turn.__wrapped__
    assessment = NS(tool_hints=["gmail"])

    ask = el._the_ask(turn, object(), (CHAT_ID, MESSAGES, AGENT, 1),
                      {"skip_composio": True, "complexity_assessment": assessment, "spoken_mode": True})

    assert ask == el.TurnAsk(chat_id=CHAT_ID, messages=MESSAGES, agent_id=AGENT, skip_composio=True,
                             complexity_assessment=assessment, force_text_only=False)


@pytest.mark.parametrize("switch, starts", [(True, True), (False, False)])
def test_the_real_agent_context_step_starts_the_lookups_only_with_the_switch_on(world, monkeypatch, switch, starts):
    monkeypatch.setattr(config, "CHAT_EARLY_LOOKUPS", switch)
    chat = a_chat(world)

    async def go():
        chat._turn_ask, chat._early = an_ask(), None
        context = await chat._load_agent_context(RUNTIME)
        early = chat._early
        await el._drop(chat)
        return context, early

    context, early = asyncio.run(go())

    assert context["description"].startswith("You are Auto, a specialized assistant agent.")
    assert (early is not None) is starts
    if starts:
        assert early.apps is not None and early.knowledge is not None
        assert world.lookups[0] == SEARCH_STEP


# --------------------------------------------------------------------------- which lookups a turn will ask for

@pytest.mark.parametrize("turn, chat, apps, knowledge", [
    ({}, {}, True, True),                                        # a plain question
    ({"force_text_only": True}, {}, False, False),               # a proactive opener: text only, nothing read
    ({"complexity_assessment": "atom"}, {}, False, True),        # the light path offers no apps
    ({}, {"widget_mode": True}, False, True),                    # a widget visitor's turn: never the owner's apps
    ({}, {"_lean_actions": []}, False, False),                   # a lean agent's turn
    ({"skip_composio": True}, {}, False, True),
    ({"text": INSTRUCTION}, {}, True, False),                    # an instruction: nothing to search first
])
def test_which_lookups_a_turn_will_ask_for(turn, chat, apps, knowledge):
    from consumers.chatbot.auto import Complexity

    given = dict(turn)
    text = given.pop("text", ASK)
    if given.get("complexity_assessment") == "atom":
        given["complexity_assessment"] = NS(complexity=Complexity.ATOM, tool_hints=[])
    ask, service = an_ask(text, **given), NS(workspace_id=WS, **{"widget_mode": False, **chat})

    assert el.apps_key(service, ask, text) == ((AGENT, WS, text, ()) if apps else None)
    assert el.knowledge_how(service, ask, text) == ({} if knowledge else None)


@pytest.mark.parametrize("pool, room", [
    (NS(size=lambda: 10, checkedout=lambda: 6), True),           # four free
    (NS(size=lambda: 10, checkedout=lambda: 7), False),          # three free
    (NS(), False),                                               # a pool that cannot say how full it is
])
def test_the_pool_has_room_with_four_connections_free(monkeypatch, pool, room):
    monkeypatch.setattr("core.database.database.engine", NS(pool=pool))

    assert el.pool_has_room() is room


def test_with_no_room_in_the_pool_only_the_apps_lookup_starts_early(world):
    world.room = False

    _, given, uses = a_turn(world, early=True)

    assert uses == (el.USED, el.NOT_STARTED)
    assert world.counts == [HERE] and world.sessions == []        # counted and searched in place
    assert given["recorded"] == {720}


# --------------------------------------------------------------------------- early, the turn is given the same

def test_a_service_that_never_ran_the_turn_wrapper_looks_up_in_place(world):
    world.found = tools_found()

    chat, given, _ = a_turn(world, early=False)

    assert not hasattr(chat, "_early") and not hasattr(chat, "_turn_ask")
    assert given["tools"] == [LIST_AGENTS, SEND_EMAIL] and given["found"] is world.found
    assert given["prefetched"] == [("search_knowledge", SEARCH_ARGS)] and given["recorded"] == {720}
    assert world.lookups == [SEARCH_STEP] and world.counts == [HERE] and len(world.searches) == 1
    assert world.filtered == [(HERE, chat.db)]                    # on the turn's own session, on the loop
    assert world.sessions == []


def test_early_lookups_give_the_turn_what_it_is_given_in_place_and_each_runs_once(world, caplog):
    world.found = tools_found()
    in_place = given_in_place(world)

    async def the_prompt_is_built(chat):
        await the_early_lookups_end(chat)
        assert chat._turn_document_ids == set() and chat.db.asked == []   # nothing on the turn yet

    with caplog.at_level(logging.INFO, logger=EARLY_LOGGER):
        _, given, uses = a_turn(world, early=True, between=the_prompt_is_built)

    assert given == in_place
    assert uses == (el.USED, el.USED)
    assert world.lookups == [SEARCH_STEP] and len(world.searches) == 1 and world.counts == [THREAD]
    assert given["tools"] == [LIST_AGENTS, SEND_EMAIL] and given["found"] is world.found
    assert [message["role"] for message in given["messages"]] == ["system", "system", "system", "user", "system"]
    assert given["messages"][-1]["content"] == f"{kp.PREFETCH_HEADER}\n\n<passages: brand-voice.md>"
    assert given["prefetched"] == [("search_knowledge", SEARCH_ARGS)]
    assert [frame["type"] for frame in given["frames"]] == ["tool-start", "tool-end"]
    assert given["recorded"] == {720}
    assert "[early-lookups] apps=used knowledge=used" in said(caplog)


def test_with_the_dial_off_the_early_prefetch_searches_nothing_and_the_turn_is_as_it_was(world):
    world.dial = False
    in_place = given_in_place(world)

    _, given, uses = a_turn(world, early=True, between=the_early_lookups_end)

    assert given == in_place and uses[1] == el.USED
    assert (given["frames"], given["prefetched"], given["recorded"]) == ([], [], set())
    assert world.searches == [] and world.counts == []


def test_an_early_lookup_is_in_flight_while_the_turn_builds_its_prompt(world):
    world.found = tools_found()
    order, answer = [], asyncio.Event()

    async def the_lookup_waits():
        order.append("lookup asked")
        await answer.wait()
        order.append("lookup answered")

    async def the_prompt_is_built(chat):
        order.append("prompt built")
        answer.set()

    world.on_lookup = the_lookup_waits
    _, given, uses = a_turn(world, early=True, between=the_prompt_is_built)

    assert order == ["lookup asked", "prompt built", "lookup answered"]
    assert world.lookups == [SEARCH_STEP] and uses[0] == el.USED
    assert given["tools"] == [LIST_AGENTS, SEND_EMAIL]


# --------------------------------------------------------------------------- a failed early lookup is made again in place

def test_an_early_count_that_fails_is_counted_and_searched_in_place(world, caplog):
    in_place = given_in_place(world)
    world.count_answers = [RuntimeError("the count failed")]

    with caplog.at_level(logging.WARNING, logger=EARLY_LOGGER):
        _, given, uses = a_turn(world, early=True, between=the_early_lookups_end)

    assert given == in_place and given["recorded"] == {720}
    assert uses[1] == el.REDONE
    assert world.counts == [THREAD, HERE] and len(world.searches) == 1
    assert "[early-lookups] could not count documents early" in said(caplog)


def test_an_early_search_that_fails_is_searched_again_in_place(world):
    in_place = given_in_place(world)
    world.search_answers = [RuntimeError("embedding timed out")]

    _, given, uses = a_turn(world, early=True, between=the_early_lookups_end)

    assert given == in_place and given["recorded"] == {720}
    assert uses[1] == el.REDONE
    assert world.counts == [THREAD, HERE] and len(world.searches) == 2


def test_an_early_prefetch_that_raises_is_searched_in_place(world, monkeypatch, caplog):
    in_place = given_in_place(world)
    real = kp.prefetch

    async def _breaks_early(*args, **kwargs):
        if kwargs.get("counted") is not None:                    # only the early prefetch is handed its count
            raise RuntimeError("the early prefetch broke")
        return await real(*args, **kwargs)

    monkeypatch.setattr(kp, "prefetch", _breaks_early)
    with caplog.at_level(logging.WARNING, logger=EARLY_LOGGER):
        _, given, uses = a_turn(world, early=True, between=the_early_lookups_end)

    assert given == in_place and given["recorded"] == {720}
    assert uses[1] == el.REDONE
    assert world.counts == [THREAD, HERE] and len(world.searches) == 1
    assert "[early-lookups] the knowledge prefetch raised; the turn searches in place" in said(caplog)


def test_an_early_prefetch_task_that_raises_is_searched_in_place(world, monkeypatch, caplog):
    in_place = given_in_place(world)

    async def _broken(*args, **kwargs):
        raise RuntimeError("the early prefetch broke outside its search")

    monkeypatch.setattr(el, "early_prefetch", _broken)
    with caplog.at_level(logging.WARNING, logger=EARLY_LOGGER):
        _, given, uses = a_turn(world, early=True, between=the_early_lookups_end)

    assert given == in_place and given["recorded"] == {720}
    assert uses[1] == el.REDONE
    assert world.counts == [HERE] and len(world.searches) == 1
    assert "[early-lookups] the knowledge prefetch failed; looking up in place" in said(caplog)


def test_an_early_prefetch_someone_else_cancelled_is_searched_in_place(world, caplog):
    in_place = given_in_place(world)

    async def someone_cancels_it(chat):
        chat._early.knowledge.cancel()
        await asyncio.wait([chat._early.knowledge])

    with caplog.at_level(logging.WARNING, logger=EARLY_LOGGER):
        _, given, uses = a_turn(world, early=True, between=someone_cancels_it)

    assert given == in_place and given["recorded"] == {720}
    assert uses[1] == el.REDONE
    assert world.counts.count(HERE) == 1 and len(world.searches) == 1
    assert "[early-lookups] the knowledge prefetch was cancelled; looking up in place" in said(caplog)


def test_an_early_apps_lookup_that_failed_is_looked_up_again_in_place(world):
    world.found = tools_found()
    in_place = given_in_place(world)
    world.tool_answers = [RuntimeError("the search broke")]

    _, given, uses = a_turn(world, early=True, between=the_early_lookups_end)

    assert given == in_place and given["tools"] == [LIST_AGENTS, SEND_EMAIL]
    assert uses[0] == el.REDONE
    assert world.lookups == [SEARCH_STEP, SEARCH_STEP]


def test_an_early_apps_lookup_someone_else_cancelled_is_looked_up_in_place(world, caplog):
    world.found = tools_found()
    in_place = given_in_place(world)

    async def someone_cancels_it(chat):
        chat._early.apps.cancel()
        await asyncio.wait([chat._early.apps])

    world.on_lookup = only_the_first_waits()
    with caplog.at_level(logging.WARNING, logger=EARLY_LOGGER):
        _, given, uses = a_turn(world, early=True, between=someone_cancels_it)

    assert given == in_place and given["tools"] == [LIST_AGENTS, SEND_EMAIL]
    assert uses[0] == el.REDONE
    assert world.lookups == [SEARCH_STEP, SEARCH_STEP]
    assert "[early-lookups] the apps lookup was cancelled; looking up in place" in said(caplog)


def test_an_early_apps_lookup_that_gave_up_in_time_is_used_as_it_is(world):
    world.tool_answers = [ComposioLookupTimeout("no answer from Composio")]

    _, given, uses = a_turn(world, early=True, between=the_early_lookups_end)

    assert uses[0] == el.USED
    assert world.lookups == [SEARCH_STEP]                         # not asked a second time
    assert given["tools"] == TOOLS and given["found"] is None
    assert given["messages"][:3] == LLM_MESSAGES


def test_an_early_lookup_asked_for_something_else_is_not_used(world, caplog):
    from consumers.chatbot.auto import Complexity

    world.found = tools_found()
    asked_early = NS(complexity=Complexity.MOLECULE, tool_hints=["gmail"])
    asked_late = NS(complexity=Complexity.MOLECULE, tool_hints=["slack"])
    in_place = given_in_place(world, assessment=asked_late)

    with caplog.at_level(logging.INFO, logger=EARLY_LOGGER):
        _, given, uses = a_turn(world, early=True, ask=an_ask(complexity_assessment=asked_early),
                                 between=the_early_lookups_end, assessment=asked_late)

    assert given == in_place
    assert uses == (el.REDONE, el.USED)
    assert [asked["tool_hints"] for asked in world.asked] == [["gmail"], ["slack"]]
    assert "[early-lookups] apps=redone knowledge=used" in said(caplog)


# --------------------------------------------------------------------------- sessions of its own

def test_the_early_prefetch_reads_on_sessions_of_its_own_and_closes_them(world):
    chat = a_chat(world)

    async def go():
        chat._early = early = el.start(chat, an_ask(skip_composio=True), RUNTIME)
        await asyncio.wait([early.knowledge])
        untouched = list(chat.db.asked)
        found = await el.the_turns_prefetch(chat, ASK, RUNTIME, CHAT_ID, {})
        await el._drop(chat)
        return early, untouched, found

    early, untouched, found = asyncio.run(go())

    assert early.apps is None                                     # the turn skips its apps
    assert untouched == []                                        # nothing of the early prefetch on the turn's session
    assert chat.db.asked == ["in_transaction"]                    # then only the turn's own release, at the join
    assert found.passages == 1 and chat._turn_document_ids == {720}
    assert world.counts == [THREAD]
    assert [where for where, _db in world.filtered] == [THREAD]
    assert world.filtered[0][1] in world.sessions
    assert len(world.sessions) == 3 and all(session.closed for session in world.sessions)


def test_an_early_prefetch_cancelled_mid_search_closes_its_sessions(world):
    chat = a_chat(world)

    async def go():
        world.on_search = asyncio.Event().wait                    # the search never answers
        early = el.start(chat, an_ask(skip_composio=True), RUNTIME)
        await until(lambda: world.searches)
        early.knowledge.cancel()
        await asyncio.wait([early.knowledge])
        return early

    early = asyncio.run(go())

    assert early.knowledge.cancelled()
    assert len(world.sessions) == 2 and all(session.closed for session in world.sessions)
    assert chat.db.asked == [] and world.filtered == []


def test_the_dial_and_the_count_are_read_on_a_session_of_their_own(world):
    assert el.dial_and_count(WS) == (True, 2)
    world.dial = False
    assert el.dial_and_count(WS) == (False, 0)                    # the dial off: nothing is counted
    world.dial, world.count_answers = True, [RuntimeError("the count failed")]
    with pytest.raises(RuntimeError, match="the count failed"):
        el.dial_and_count(WS)

    assert world.counts == [HERE, HERE]
    assert len(world.sessions) == 2 and all(session.closed for session in world.sessions)


@pytest.mark.parametrize("counted, searches, counts", [
    (3, 1, []),                  # handed its count: not read again
    (0, 0, []),                  # handed "no documents": nothing is searched
    (None, 1, [HERE]),           # every caller before D-2: the count is read here
])
def test_prefetch_takes_a_count_its_caller_already_read(world, counted, searches, counts):
    async def search(args):
        return await world.execute_and_format(tool_args=args)

    found = asyncio.run(kp.prefetch(None, WS, ASK, search=search, enabled=True, limit=5, min_score=0.3,
                                    counted=counted))

    assert (found is not None) is bool(searches)
    assert len(world.searches) == searches and world.counts == counts


# --------------------------------------------------------------------------- stop, and a turn that ends early

def test_the_turns_own_stop_at_the_join_is_raised_and_nothing_is_searched_in_place(world):
    chat = a_chat(world)

    async def go():
        world.on_search = asyncio.Event().wait                    # the early search never answers
        chat._turn_ask = an_ask()
        chat._early = early = el.start(chat, chat._turn_ask, RUNTIME)
        turn = asyncio.create_task(late_steps(chat))
        await until(lambda: world.searches and "in_transaction" in chat.db.asked)   # the turn waits at the join
        turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await turn
        await el._drop(chat)
        return early

    early = asyncio.run(go())

    assert early.knowledge.cancelled()
    assert len(world.searches) == 1 and world.counts == [THREAD]   # the early ones only: nothing in place
    assert chat._turn_document_ids == set()


def test_stopping_the_turn_cancels_both_early_lookups(world, caplog):
    chat, seen = a_chat(world), []

    async def waits_for_ever(chat):
        seen.append(chat._early)
        await asyncio.Event().wait()

    async def go():
        world.on_lookup = world.on_search = asyncio.Event().wait   # neither lookup answers
        turn = asyncio.create_task(drain(a_wrapped_turn(waits_for_ever)(chat, CHAT_ID, MESSAGES, AGENT, 1)))
        await until(lambda: seen and world.searches)
        turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await turn

    gc.collect()
    caplog.clear()                                                # what earlier tests left behind is not this one's
    with caplog.at_level(logging.INFO):
        asyncio.run(go())
        gc.collect()

    (early,) = seen
    assert early.apps.cancelled() and early.knowledge.cancelled()
    assert chat._early is None and chat._turn_ask is None
    assert "[early-lookups] apps=dropped knowledge=dropped" in said(caplog)
    assert not [line for line in said(caplog) if "never retrieved" in line]


def test_a_turn_that_fails_before_its_agent_is_active_starts_nothing(world, caplog):
    chat = a_chat(world)

    @el.starts_its_lookups_early
    async def turn(chat, chat_id, messages, agent_id, user_id, skip_composio=False, complexity_assessment=None,
                   force_text_only=False):
        yield "chat-id"
        raise RuntimeError("no credits")

    async def go():
        frames = []
        with pytest.raises(RuntimeError, match="no credits"):
            async for frame in turn(chat, CHAT_ID, MESSAGES, AGENT, 1):
                frames.append(frame)
        return frames

    with caplog.at_level(logging.INFO, logger=EARLY_LOGGER):
        assert asyncio.run(go()) == ["chat-id"]

    assert (world.lookups, world.searches, world.counts, world.sessions) == ([], [], [], [])
    assert chat._early is None and chat._turn_ask is None
    assert "[early-lookups] apps=none knowledge=none" in said(caplog)


def test_a_turn_that_fails_after_the_start_cancels_what_runs_and_says_what_had_failed(world, monkeypatch, caplog):
    chat, seen = a_chat(world), []

    async def _broken(*args, **kwargs):
        raise RuntimeError("the early prefetch broke")

    async def fails(chat):
        seen.append(chat._early)
        await asyncio.wait([chat._early.knowledge])               # the early prefetch has failed by now
        raise RuntimeError("the turn failed")

    async def go():
        world.on_lookup = asyncio.Event().wait                    # the apps lookup never answers
        with pytest.raises(RuntimeError, match="the turn failed"):
            await drain(a_wrapped_turn(fails)(chat, CHAT_ID, MESSAGES, AGENT, 1))

    monkeypatch.setattr(el, "early_prefetch", _broken)
    gc.collect()
    caplog.clear()                                                # what earlier tests left behind is not this one's
    with caplog.at_level(logging.INFO):
        asyncio.run(go())
        gc.collect()

    (early,) = seen
    assert early.apps.cancelled()
    assert early.knowledge.done() and not early.knowledge.cancelled()
    assert chat._early is None and chat._turn_ask is None
    assert "[early-lookups] an early lookup the turn never collected had failed" in said(caplog)
    assert not [line for line in said(caplog) if "never retrieved" in line]


# --------------------------------------------------------------------------- the turn's ask and the turn's clock

def test_an_early_lookup_books_to_the_turns_ask(world):
    from core.llm.usage_context import LANE_CHAT, current_usage_scope, usage_scope
    from services.credits.asks import ask_scope

    chat, scopes = a_chat(world), []

    async def notes_its_scope():
        scopes.append(dict(current_usage_scope()))

    async def go():
        world.on_lookup = world.on_search = notes_its_scope
        with usage_scope(request_type=LANE_CHAT, execution_id=f"chat:{CHAT_ID}", agent_id=AGENT), \
                ask_scope(LANE_CHAT, workspace_id=WS) as ask:
            early = el.start(chat, an_ask(), RUNTIME)
        await asyncio.wait([early.apps, early.knowledge])         # outside the scope: each task kept the turn's
        return ask.ask_id

    ask_id = asyncio.run(go())

    assert world.lookups == [SEARCH_STEP, HINTS_STEP] and len(world.searches) == 1
    assert len(scopes) == 3                                       # the tool search, the hints, the search
    assert {scope["ask_id"] for scope in scopes} == {ask_id}
    assert {scope["execution_id"] for scope in scopes} == {f"chat:{CHAT_ID}"}


def test_an_early_lookups_own_time_goes_on_the_turns_clock():
    from core.monitoring import turn_clock

    async def answers():
        return "found"

    async def go():
        previous = turn_clock.current()
        clock = turn_clock.begin(turn_clock.ORIGIN_TURN)
        try:
            return await el._begin(el.APPS_STAGE, answers), clock.snapshot()["stages"]
        finally:
            turn_clock.leave(previous)

    found, stages = asyncio.run(go())

    assert found == "found"
    assert list(stages) == ["apps_lookup"] and stages["apps_lookup"] >= 0


def test_in_a_build_with_no_turn_clock_an_early_lookup_still_answers(monkeypatch):
    monkeypatch.setitem(sys.modules, "core.monitoring.turn_clock", None)   # importing it now fails

    async def answers():
        return "found"

    async def go():
        return await el._begin(el.KNOWLEDGE_STAGE, answers)

    assert asyncio.run(go()) == "found"


# --------------------------------------------------------------------------- structure

def test_the_turn_and_its_agent_context_step_are_wrapped_where_the_plan_puts_them():
    from consumers.chatbot.service import StreamingChatService
    from consumers.chatbot.turn_embeddings import embeds_each_text_once

    async def _stub(chat, *args, **kwargs):
        yield None

    async def _step(chat, agent_runtime):
        return {}

    codes, fn = [], StreamingChatService._stream_response_with_agent_scoped
    while fn is not None:
        codes.append(fn.__code__)
        fn = getattr(fn, "__wrapped__", None)

    early, embeds = el.starts_its_lookups_early(_stub).__code__, embeds_each_text_once(_stub).__code__
    assert early in codes and embeds in codes
    assert codes.index(early) == codes.index(embeds) + 1          # inside the embedding scope (and the clock)
    assert codes.index(early) == len(codes) - 2                   # innermost: only the turn itself is beneath it
    assert StreamingChatService._load_agent_context.__code__ is el.once_the_agent_is_active(_step).__code__
    assert StreamingChatService._retrieval_first.__code__.co_qualname == "its_reads_are_receipted.<locals>.wrapped"


@pytest.mark.parametrize("module", EARLY_PATH)
def test_the_early_search_names_no_context_the_turn_sets_after_activation(module):
    """Pins what was checked by hand: the early search runs in a context copied at activation,
    so nothing on its path may read what the turn sets later. Not the whole executor path."""
    source = (ORCHESTRATOR / module).read_text(encoding="utf-8")

    assert [name for name in SET_AFTER_ACTIVATION if name in source] == []
