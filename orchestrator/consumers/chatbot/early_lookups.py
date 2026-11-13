"""A turn's two late lookups: split from their merges (voice latency D-1), started early (D-2).

Just before its first model call a chat turn makes two lookups that read nothing the turn's
prompt produced: the connected-apps lookup (``consumers.chatbot.app_lookup``) and the
knowledge prefetch (``consumers.chatbot.knowledge_prefetch``, F085). Each was written in one
piece with the code that puts its answer into the turn. This module holds the lookup side of
both, so that ``StreamingChatService._inject_composio_tools`` and ``_retrieval_first`` keep
only the merge:

* ``run_prefetch`` searches and hands back what it found and every search's result. It records
  nothing on the turn and never raises but for a cancel, so it can run on any task.
* ``the_turns_prefetch`` is what ``_retrieval_first`` awaits, and ``the_turns_app_lookup`` what
  ``_inject_composio_tools`` awaits: the lookup's answer, at the place it always ran.

While ``CHAT_EARLY_LOOKUPS`` is false both run in place: a turn searches the same things at the
same point with the same arguments, on its own session, as it did before the split.

While it is true the two lookups start as tasks once the turn's agent is active
(``once_the_agent_is_active``, on ``_load_agent_context``), beside tool loading and the prompt
build, and the two functions above collect them where the lookup used to run:

* An early answer is used only when it was asked for exactly what the turn asks for at the old
  place (its key) and it did not fail. A failed, raised or cancelled early lookup is made again
  in place, so an early failure never takes passages or tools out of a prompt the in-place
  lookup would have put them in. (A failure the in-place lookup would have had too is thereby
  tried once more.)
* No early task touches the turn's session. The apps lookup already runs on threads with a
  session of its own (``core.composio.off_loop``). The early prefetch reads the dial and the
  document count in a thread on a session that thread opens and closes, filters each search in
  a thread the same way, and keeps one session of its own for the one read ``prefetch`` makes
  on the loop (the database note of its header).
* The early prefetch starts only while the worker's pool has room (``pool_has_room``).
* ``starts_its_lookups_early`` (on the turn) cancels both tasks when the turn ends, however it
  ends, waits a moment for them to stop, and writes one line:
  ``[early-lookups] apps=<use> knowledge=<use>`` (``used``, ``redone``, ``dropped`` or ``none``).

Nothing here logs what a person said: the keys hold their words and are only compared.
"""
from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import functools
import inspect
import logging
import time
from typing import TYPE_CHECKING, Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional, Tuple, TypeVar

from config import config
from consumers.chatbot.app_lookup import AppLookup, asks_the_apps, look_up_apps, tool_hints_of
from consumers.chatbot.lean_turn import lean_turn_of
from core.database.read_release import release_if_read_only

if TYPE_CHECKING:
    from consumers.chatbot.knowledge_prefetch import Prefetch

logger = logging.getLogger(__name__)

T = TypeVar("T")
Stream = Callable[..., AsyncGenerator[Any, None]]

EARLY_KNOWLEDGE_FREE_CONNECTIONS = 4   # the turn's, the early session's, the search's, a Composio thread's
DROP_WAIT_SECONDS = 1.0                # how long a turn's end waits for its cancelled lookups to stop
DEFAULT_AGENT_ID = 1                   # the agent a search is run as when the runtime names none
APPS_STAGE = "apps_lookup"             # the turn clock's stages: how long each early lookup itself took
KNOWLEDGE_STAGE = "knowledge_lookup"
APPS = "the apps lookup"
KNOWLEDGE = "the knowledge prefetch"
# What became of an early lookup, for the turn's one line.
NOT_STARTED = "none"
USED = "used"
REDONE = "redone"                      # failed, cancelled or asked for something else: looked up in place
DROPPED = "dropped"                    # started, and the turn ended before it was collected


@dataclasses.dataclass(frozen=True)
class Searched:
    """What a turn's knowledge prefetch did."""

    found: Optional["Prefetch"]              # knowledge_prefetch.prefetch's answer
    results: Tuple[Any, ...]                 # each search's result, for the turn's retrieval record
    failed: bool = False                     # the count, a search, or prefetch itself failed
    error: Optional[BaseException] = None    # what prefetch raised, if it raised


@dataclasses.dataclass(frozen=True)
class TurnAsk:
    """The turn's own arguments the early lookups read."""

    chat_id: str
    messages: List[Dict[str, Any]]
    agent_id: int
    skip_composio: bool
    complexity_assessment: Any
    force_text_only: bool


@dataclasses.dataclass(frozen=True)
class Early:
    """The lookups a turn started early: each task, what it was asked for (its key), and what
    became of it."""

    apps_key: Optional[tuple] = None
    apps: Optional["asyncio.Task[AppLookup]"] = None
    knowledge_key: Optional[tuple] = None
    knowledge: Optional["asyncio.Task[Searched]"] = None
    apps_use: str = NOT_STARTED
    knowledge_use: str = NOT_STARTED


# --- the knowledge prefetch --------------------------------------------------------------------

def _a_session() -> Any:
    from core.database.database import SessionLocal

    return SessionLocal()


def _owners_own_on_its_own_session(result: Any, workspace_id: Any) -> Any:
    """``owners_own`` on a session this call opens and closes. Blocking: run in a thread."""
    from services.draft_guides import owners_own

    db = _a_session()
    try:
        return owners_own(db, result, workspace_id)
    finally:
        db.close()


async def _owners_own(db: Any, off_loop: bool, result: Any, workspace_id: Any) -> Any:
    """The search result without what an agent wrote (F269/F287, night 8: never an agent's
    writing as facts). In place it reads on ``db`` on the loop, as the turn always did;
    ``off_loop`` (the early prefetch) it reads in a thread, on a session that thread opens and
    closes, so a pool wait never holds the loop and a cancel closes no session under a read."""
    from services.draft_guides import owners_own

    if off_loop:
        return await asyncio.to_thread(_owners_own_on_its_own_session, result, workspace_id)
    return owners_own(db, result, workspace_id)


async def run_prefetch(chat: Any, db: Any, latest_text: str, agent_runtime: Any, chat_id: str,
                       how: Dict[str, Any], into: Optional[List[Any]] = None, *,
                       enabled: Optional[bool] = None, counted: Optional[int] = None,
                       on_its_own: bool = False) -> Searched:
    """Search the documents for this turn (``knowledge_prefetch.prefetch``) with ``db`` as the
    session the count and the owner's-documents filter read on. Each search's result is handed
    back, never recorded on ``chat``; ``into``, when given, is the caller's list each result is
    also put in as it arrives, so a caller that is cancelled still holds what was searched.
    What was searched is returned even when ``prefetch`` raised. Never raises but for a cancel.

    ``on_its_own`` is the early prefetch, which never reads on the turn's session: ``db`` is
    not used, ``prefetch`` gets a session this call opens and closes, and each filter runs in a
    thread on a session of that thread's own. ``enabled`` and ``counted`` are the dial and the
    document count when the caller already read them (None: read here, as the turn always did)."""
    from consumers.chatbot.knowledge_prefetch import PREFETCH_TOOL, prefetch

    session = None if on_its_own else db
    results: List[Any] = [] if into is None else into
    search_failed = False

    async def _search(args: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal search_failed
        try:
            result = await _owners_own(session, on_its_own, await chat.tool_router.execute_and_format(
                tool_name=PREFETCH_TOOL, tool_args=args,
                agent_id=agent_runtime.agent_id if hasattr(agent_runtime, "agent_id") else DEFAULT_AGENT_ID,
                workspace_id=chat.workspace_id,
                original_intent=latest_text,
                caller_context={"user_query": latest_text, "conversation_id": chat_id, "retrieval_first": True},
            ), chat.workspace_id)
        except Exception:
            search_failed = True
            raise
        results.append(result)
        return result

    try:
        if on_its_own:
            session = _a_session()
        found = await prefetch(
            session, chat.workspace_id, latest_text, search=_search, viewer=getattr(chat, "_viewer_subject_id", None),
            enabled=config.CHATBOT_KNOWLEDGE_PREFETCH if enabled is None else enabled, **how,
            limit=config.KNOWLEDGE_PREFETCH_PASSAGES, min_score=config.KNOWLEDGE_PREFETCH_MIN_SCORE,
            **({} if counted is None else {"counted": counted}),
        )
    except Exception as exc:  # noqa: BLE001 — handed back as the failure; the caller decides who logs it
        return Searched(None, tuple(results), failed=True, error=exc)
    finally:
        if on_its_own and session is not None:
            session.close()
    return Searched(found, tuple(results), failed=search_failed)


def dial_and_count(workspace_id: Any) -> Tuple[bool, int]:
    """The knowledge-prefetch dial and, while it is on, how many documents the workspace has,
    read on a session this call opens and closes. Blocking (a settings read, a pool checkout,
    a count): run in a thread."""
    from consumers.chatbot.knowledge_prefetch import documents_in

    if not config.CHATBOT_KNOWLEDGE_PREFETCH:
        return False, 0
    db = _a_session()
    try:
        return True, documents_in(db, workspace_id)
    finally:
        db.close()


async def early_prefetch(chat: Any, latest_text: str, agent_runtime: Any, chat_id: str,
                         how: Dict[str, Any]) -> Searched:
    """The knowledge prefetch as an early task: the dial and the count off the loop, then the
    search, none of it on the turn's session. A count that could not be read is a failure, so
    the turn counts and searches in place."""
    try:
        enabled, counted = await asyncio.to_thread(dial_and_count, chat.workspace_id)
    except Exception:  # noqa: BLE001 — handed back as the failure: the turn counts and searches in place
        logger.warning("[early-lookups] could not count documents early", exc_info=True)
        return Searched(None, (), failed=True)
    searched = await run_prefetch(chat, None, latest_text, agent_runtime, chat_id, how,
                                  enabled=enabled, counted=counted, on_its_own=True)
    if searched.error is not None:  # said here, once: the turn may end before it collects this answer
        logger.warning("[early-lookups] %s raised; the turn searches in place", KNOWLEDGE, exc_info=searched.error)
    return searched


# --- which lookups a turn will ask for, and whether there is room -----------------------------------

def _key_of_the_apps(agent_id: Any, workspace_id: Any, latest_text: str, tool_hints: Any) -> tuple:
    return (agent_id, str(workspace_id), latest_text, tuple(tool_hints or ()))


def _key_of_the_search(latest_text: str, how: Dict[str, Any], agent_runtime: Any, chat_id: str) -> tuple:
    return (latest_text, tuple(sorted(how.items())), getattr(agent_runtime, "agent_id", DEFAULT_AGENT_ID), chat_id)


def apps_key(chat: Any, ask: TurnAsk, latest_text: str) -> Optional[tuple]:
    """What the turn's apps lookup will be asked for, or None when the turn will not reach it:
    a text-only or ATOM turn, a widget turn, a lean turn, or one that skips its apps. The skips
    are the turn's own (``_stream_response_with_agent_scoped``, ``no_app_search_on_a_lean_turn``,
    ``asks_the_apps``); the old place still decides, so a skip that drifts costs a wasted read."""
    from consumers.chatbot.auto import Complexity

    assessment = ask.complexity_assessment
    complexity = assessment.complexity if assessment else Complexity.MOLECULE
    if ask.force_text_only or complexity == Complexity.ATOM:
        return None
    if getattr(chat, "widget_mode", False) or lean_turn_of(chat):
        return None
    workspace_id = getattr(chat, "workspace_id", None)
    if not asks_the_apps(latest_text, ask.agent_id, workspace_id, ask.skip_composio):
        return None
    return _key_of_the_apps(ask.agent_id, workspace_id, latest_text, tool_hints_of(assessment))


def knowledge_how(chat: Any, ask: TurnAsk, latest_text: str) -> Optional[Dict[str, Any]]:
    """How the turn will search its documents first (``knowledge_prefetch.searched_first``), or
    None when it will not: a text-only turn, a lean turn, or a message that is neither a
    question nor a drafting turn."""
    from consumers.chatbot.knowledge_prefetch import searched_first

    if ask.force_text_only or lean_turn_of(chat):
        return None
    return searched_first(latest_text)


def pool_has_room(need: int = EARLY_KNOWLEDGE_FREE_CONNECTIONS) -> bool:
    """Whether the worker's base pool has ``need`` connections free. With the early prefetch a
    question turn can hold up to four at once (see the module), so a worker near its base pool
    searches in place, one connection at a time, as before."""
    try:
        from core.database.database import engine

        return engine.pool.size() - engine.pool.checkedout() >= need
    except Exception:  # noqa: BLE001 — a pool that cannot say how full it is means no early start
        logger.debug("[early-lookups] the pool could not say how full it is: no early prefetch", exc_info=True)
        return False


# --- starting the lookups early ----------------------------------------------------------------

def _clocked(stage: str, seconds: float) -> None:
    """Add a lookup's own time to the turn's clock, when this build has one."""
    try:
        import core.monitoring.turn_clock as turn_clock
    except ImportError:  # no turn clock here: the lookup is not timed
        return
    turn_clock.add(stage, seconds)


async def _timed(stage: str, lookup: Callable[..., Awaitable[T]], *args: Any) -> T:
    """``lookup(*args)``, its own time added to the turn's clock. The lookup's coroutine is made
    here, inside its task: a task cancelled before its first step leaves none unawaited."""
    started = time.perf_counter()
    try:
        return await lookup(*args)
    finally:
        _clocked(stage, time.perf_counter() - started)


def _read(task: "asyncio.Task[Any]") -> None:
    """Read a finished early lookup's ending, so one that nobody collected never logs
    'Task exception was never retrieved'. ``_drop`` says which of those failed."""
    if not task.cancelled():
        task.exception()


def _begin(stage: str, lookup: Callable[..., Awaitable[T]], *args: Any) -> "asyncio.Task[T]":
    task = asyncio.create_task(_timed(stage, lookup, *args))
    task.add_done_callback(_read)
    return task


def start(chat: Any, ask: TurnAsk, agent_runtime: Any) -> Early:
    """Start the lookups this turn will ask for, each as a task of its own: the apps lookup
    when the turn will reach it, the knowledge prefetch when it will search and the pool has
    room. Everything that can fail is read before the first task is made, so a failure here
    leaves no task behind."""
    latest_text = chat.prompt_analyzer.extract_latest_user_text(ask.messages)
    hints = tool_hints_of(ask.complexity_assessment)
    for_apps = apps_key(chat, ask, latest_text)
    how = knowledge_how(chat, ask, latest_text)
    for_knowledge = None
    if how is not None and pool_has_room():
        for_knowledge = _key_of_the_search(latest_text, how, agent_runtime, ask.chat_id)
    apps = knowledge = None
    if for_apps is not None:
        apps = _begin(APPS_STAGE, look_up_apps, ask.agent_id, chat.workspace_id, latest_text, hints)
    if for_knowledge is not None:
        knowledge = _begin(KNOWLEDGE_STAGE, early_prefetch, chat, latest_text, agent_runtime, ask.chat_id, how)
    return Early(apps_key=for_apps, apps=apps, knowledge_key=for_knowledge, knowledge=knowledge,
                 apps_use=NOT_STARTED if apps is None else DROPPED,
                 knowledge_use=NOT_STARTED if knowledge is None else DROPPED)


# --- collecting them where the lookups used to run ------------------------------------------------

async def an_early_answer(task: "asyncio.Task[T]", what: str) -> Optional[T]:
    """The early lookup's answer, or None when it raised or was cancelled by anything but the
    turn's own stop: the caller then looks up in place. The turn's own cancellation is raised."""
    try:
        return await task
    except asyncio.CancelledError:
        turn = asyncio.current_task()
        if turn is not None and turn.cancelling():  # the turn itself is being stopped
            raise
        logger.warning("[early-lookups] %s was cancelled; looking up in place", what)
        return None
    except Exception:  # noqa: BLE001 — a failed early lookup is made again in place
        logger.warning("[early-lookups] %s failed; looking up in place", what, exc_info=True)
        return None


def _collected(chat: Any, **use: str) -> None:
    """Note what became of an early lookup (a new ``Early``; the old one is not changed)."""
    early = getattr(chat, "_early", None)
    if early is not None:
        chat._early = dataclasses.replace(early, **use)


async def _the_early_search(chat: Any, latest_text: str, agent_runtime: Any, chat_id: str,
                            how: Dict[str, Any]) -> Optional[Searched]:
    """The early prefetch's answer when the turn can use it; None: search in place."""
    early = getattr(chat, "_early", None)
    if early is None or early.knowledge is None:
        return None
    searched = None
    if early.knowledge_key == _key_of_the_search(latest_text, how, agent_runtime, chat_id):
        # F330: the turn must not hold its connection while it waits (in place, ``prefetch``
        # releases it after its count).
        release_if_read_only(chat.db)
        searched = await an_early_answer(early.knowledge, KNOWLEDGE)
    usable = searched is not None and not searched.failed
    _collected(chat, knowledge_use=USED if usable else REDONE)
    return searched if usable else None


async def the_turns_prefetch(chat: Any, latest_text: str, agent_runtime: Any, chat_id: str,
                             how: Dict[str, Any]) -> Optional["Prefetch"]:
    """The turn's knowledge prefetch, where it always ran. An early search that was asked for
    exactly this and did not fail is used: its results go on the turn's retrieval record here.
    Otherwise the search runs in place on the turn's own session: every search's result goes on
    the record, also when the prefetch found nothing, failed or was cancelled, and what
    ``prefetch`` raised is raised here for ``_retrieval_first`` to log."""
    from consumers.chatbot.knowledge_prefetch import PREFETCH_TOOL

    searched_early = await _the_early_search(chat, latest_text, agent_runtime, chat_id, how)
    if searched_early is not None:
        for result in searched_early.results:
            chat._collect_tool_retrieval(PREFETCH_TOOL, result)
        return searched_early.found
    results: List[Any] = []
    try:
        searched = await run_prefetch(chat, chat.db, latest_text, agent_runtime, chat_id, how, into=results)
    finally:
        for result in results:
            chat._collect_tool_retrieval(PREFETCH_TOOL, result)
    if searched.error is not None:
        raise searched.error
    return searched.found


async def the_turns_app_lookup(chat: Any, agent_id: int, latest_text: str, tool_hints: Any) -> AppLookup:
    """The turn's connected-apps lookup, where it always ran. An early lookup that was asked
    for exactly this and did not fail is used; one that gave up in time is used as it is (it
    had its full ``COMPOSIO_LOOKUP_TIMEOUT_SECONDS``, and asking again would make the turn wait
    up to twice). Otherwise the lookup runs in place."""
    early = getattr(chat, "_early", None)
    if early is not None and early.apps is not None:
        lookup = None
        if early.apps_key == _key_of_the_apps(agent_id, chat.workspace_id, latest_text, tool_hints):
            lookup = await an_early_answer(early.apps, APPS)
        usable = lookup is not None and not lookup.failed
        _collected(chat, apps_use=USED if usable else REDONE)
        if usable:
            return lookup
    return await look_up_apps(agent_id, chat.workspace_id, latest_text, tool_hints)


# --- the turn's two hooks ------------------------------------------------------------------------

def _the_ask(turn: Stream, chat: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Optional[TurnAsk]:
    """The turn's own arguments, by name. None when they could not be read: the turn then runs
    with no early lookup."""
    try:
        bound = inspect.signature(turn).bind(chat, *args, **kwargs)
        bound.apply_defaults()
        given = bound.arguments
        return TurnAsk(chat_id=given["chat_id"], messages=given["messages"], agent_id=given["agent_id"],
                       skip_composio=bool(given["skip_composio"]),
                       complexity_assessment=given["complexity_assessment"],
                       force_text_only=bool(given["force_text_only"]))
    except Exception:  # noqa: BLE001 — the turn runs as it did, every lookup in place
        logger.exception("[early-lookups] could not read the turn's arguments: no early lookup this turn")
        return None


def _left_unread(task: "asyncio.Task[Any]", use: str) -> Optional[BaseException]:
    """What an early lookup that nobody collected raised, if it ended by raising."""
    if use != DROPPED or not task.done() or task.cancelled():
        return None
    return task.exception()


async def _drop(chat: Any) -> None:
    """The turn is over: forget its ask, cancel the early lookups still running and wait, for
    a moment, until they have stopped. A lookup thread already running ends on its own timeout,
    as it always did. One line says what became of each lookup."""
    early = getattr(chat, "_early", None)
    chat._turn_ask, chat._early = None, None
    if early is None:
        logger.info("[early-lookups] apps=%s knowledge=%s", NOT_STARTED, NOT_STARTED)
        return
    started = [(task, use) for task, use in ((early.apps, early.apps_use), (early.knowledge, early.knowledge_use))
               if task is not None]
    for task, _use in started:
        task.cancel()  # a finished task is left as it is
    if started:
        _stopped, running = await asyncio.wait([task for task, _use in started], timeout=DROP_WAIT_SECONDS)
        if running:
            logger.warning("[early-lookups] %d early lookup(s) still stopping after the turn ended", len(running))
    for task, use in started:
        error = _left_unread(task, use)
        if error is not None:
            logger.warning("[early-lookups] an early lookup the turn never collected had failed", exc_info=error)
    logger.info("[early-lookups] apps=%s knowledge=%s", early.apps_use, early.knowledge_use)


def starts_its_lookups_early(turn: Stream) -> Stream:
    """Wrap ``StreamingChatService._stream_response_with_agent_scoped``. While
    ``CHAT_EARLY_LOOKUPS`` is false the turn passes straight through. While it is true the
    turn's arguments are kept for ``once_the_agent_is_active``, and when the turn ends, however
    it ends, its early lookups are cancelled (``_drop``). The turn is closed here, in the task
    that closes this wrapper, before its lookups are dropped."""
    @functools.wraps(turn)
    async def wrapped(chat: Any, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        if not config.CHAT_EARLY_LOOKUPS:  # off: nothing else of the early half runs
            async with contextlib.aclosing(turn(chat, *args, **kwargs)) as chunks:
                async for chunk in chunks:
                    yield chunk
            return
        chat._turn_ask, chat._early = _the_ask(turn, chat, args, kwargs), None
        try:
            async with contextlib.aclosing(turn(chat, *args, **kwargs)) as chunks:
                async for chunk in chunks:
                    yield chunk
        finally:
            await _drop(chat)
    return wrapped


def once_the_agent_is_active(load: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Wrap ``StreamingChatService._load_agent_context``, the turn's first step after its agent
    was activated: a turn whose activation refuses (no credits, a session agent, an unknown
    agent) has started nothing. With ``CHAT_EARLY_LOOKUPS`` on and the turn's arguments kept,
    the lookups start here, and one pass of the loop lets each reach its thread hand-off before
    the turn's synchronous steps take the loop."""
    @functools.wraps(load)
    async def wrapped(chat: Any, agent_runtime: Any, *args: Any, **kwargs: Any) -> Any:
        ask = getattr(chat, "_turn_ask", None)
        if ask is not None and getattr(chat, "_early", None) is None and config.CHAT_EARLY_LOOKUPS:
            try:
                chat._early = start(chat, ask, agent_runtime)
            except Exception:  # noqa: BLE001 — the turn runs as it did, every lookup in place
                logger.exception("[early-lookups] could not start the early lookups: looking up in place")
                chat._early = None
            await asyncio.sleep(0)
        return await load(chat, agent_runtime, *args, **kwargs)
    return wrapped


__all__ = [
    "Early", "EARLY_KNOWLEDGE_FREE_CONNECTIONS", "Searched", "TurnAsk", "an_early_answer", "apps_key",
    "dial_and_count", "early_prefetch", "knowledge_how", "once_the_agent_is_active", "pool_has_room",
    "run_prefetch", "start", "starts_its_lookups_early", "the_turns_app_lookup", "the_turns_prefetch",
]
