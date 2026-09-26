"""PRD-234 S3 — every lane that targets a Claude Code agent files a board ticket.

Heartbeats, scheduled tasks, channel mentions, webhooks and Composio triggers all
used to call ``AgentFactory.execute_with_prompt``; for a ``runtime: cli`` agent the
factory refuses (by design — the user's own ``claude`` runs the work), and each
lane then either logged the refusal, replied nothing, or worse, recorded it as a
success. The honest shape is one ticket on the board: the paired CLI host claims
it, the result lands like any other ticket's, and the lane can say so.

One writer, one shape: the ticket carries the lane's prompt as its description,
``source_type`` names the lane, and ``source_id`` makes the lane's re-fires
idempotent (an open ticket from the same source is reused, never duplicated).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, List, Optional, Sequence, Tuple

from sqlalchemy.orm import Session

from core.cli_runtime import (
    CONFIG_MODEL_KEY, CONFIG_PROVIDER_KEY, CONFIG_WORKING_DIRECTORY_KEY, PROVIDER_CLAUDE, RUNTIME_CLI,
    runtime_kind_of,
)
from core.models.core import Agent, BoardTask

logger = logging.getLogger(__name__)

# PRD-245 S0.5 (D7): the statuses under which a lane's re-fire lands on the ticket
# it already filed. ``review`` is NOT one of them — a ticket in review is finished
# work awaiting sign-off; the next fire files a new ticket (agent 15's #93 absorbed
# 236 heartbeats while it sat in review).
REUSABLE_STATUSES: Sequence[str] = ("inbox", "assigned", "in_progress", "blocked")
QUEUED_LINE = "queued for your Claude Code session as ticket #{task_id}"
NO_HOST_REASON = (
    "Waiting for a CLI host — none is online. Start it with `make cli-host`; "
    "the ticket is claimed on its first poll."
)
# CLI adapter design §8.2: hosts are online, but none of them runs this agent's CLI.
NO_CLI_HOST_PREFIX = "Waiting for a CLI host that runs "
NO_CLI_HOST_REASON = (
    NO_CLI_HOST_PREFIX + "{cli} — the host(s) online serve {served}. Install and log in to "
    "{cli} on a host (or pick another CLI for this agent); the ticket is claimed on the next poll."
)


def is_no_cli_host_reason(reason: Any) -> bool:
    return isinstance(reason, str) and reason.startswith(NO_CLI_HOST_PREFIX)


def agent_cli_provider(db: Session, agent_id: Optional[int]) -> str:
    """The CLI a session agent runs on (``configuration.provider``); ``claude``
    when absent or unreadable — what every session agent ran on before the field."""
    if agent_id is None:
        return PROVIDER_CLAUDE
    try:
        row = db.query(Agent.configuration).filter(Agent.id == agent_id).first()
    except Exception:  # noqa: BLE001 — a test double or a broken session
        logger.debug("[CliTicketLane] provider lookup unavailable for agent %s", agent_id, exc_info=True)
        return PROVIDER_CLAUDE
    configuration = row[0] if isinstance(row, (tuple, list)) else getattr(row, "configuration", row)
    provider = configuration.get(CONFIG_PROVIDER_KEY) if isinstance(configuration, dict) else None
    return provider if isinstance(provider, str) and provider else PROVIDER_CLAUDE


def no_cli_host_reason_for(db: Session, workspace_id: Any, cli: str) -> Optional[str]:
    """The blocked line when hosts are online but none serves ``cli``; ``None``
    when one does — or when it cannot be determined (a line that may be wrong is
    worse than no line; the claim filter is the real guard)."""
    try:
        from services import cli_host_service
        served = cli_host_service.serving_providers(db, workspace_id)
    except Exception:  # noqa: BLE001
        logger.debug("[CliTicketLane] served-CLI lookup unavailable", exc_info=True)
        return None
    if cli in served:
        return None
    return NO_CLI_HOST_REASON.format(cli=cli, served=", ".join(served) or "no CLI")


def is_cli_agent(db: Session, agent_id: Optional[int]) -> bool:
    """True when the agent's configuration says ``runtime: cli``. Lookup failure → False
    (the caller then takes the API path, whose own guard still refuses cli agents)."""
    if agent_id is None:
        return False
    try:
        agent = db.query(Agent).filter(Agent.id == int(agent_id)).first()
    except Exception:  # noqa: BLE001
        return False
    if agent is None:
        return False
    return runtime_kind_of(getattr(agent, "configuration", None) or {}) == RUNTIME_CLI


def open_ticket_for_source(db: Session, workspace_id: Any, source_type: str, source_id: str) -> Optional[BoardTask]:
    """The ticket a lane already filed for this source that can still absorb the
    fire (``REUSABLE_STATUSES``), if any. A ticket in review is not it."""
    try:
        return (
            db.query(BoardTask)
            .filter(
                BoardTask.workspace_id == workspace_id,
                BoardTask.source_type == source_type,
                BoardTask.source_id == source_id,
                BoardTask.status.in_(list(REUSABLE_STATUSES)),
            )
            .order_by(BoardTask.id.desc())
            .first()
        )
    except Exception:  # noqa: BLE001
        return None


def host_online(db: Session, workspace_id: Any) -> bool:
    """Is any paired CLI host of this workspace currently heartbeating?"""
    try:
        from core.models.cli_hosts import CliHost, CliHostStatus
        hosts = db.query(CliHost).filter(
            CliHost.workspace_id == workspace_id,
            CliHost.status == CliHostStatus.PAIRED.value,
        ).all()
        return any(h.is_online() for h in hosts)
    except Exception:  # noqa: BLE001
        return False


def file_cli_ticket(
    db: Session,
    *,
    workspace_id: Any,
    agent_id: int,
    title: str,
    prompt: str,
    source_type: str,
    source_id: str,
    priority: str = "medium",
    review_mode: str = "auto",
    tags: Optional[list] = None,
    actor: Optional[str] = None,
    created_by_type: Optional[str] = None,
    created_by_id: Optional[str] = None,
    resume_session_id: Optional[str] = None,
    resume_host_id: Optional[str] = None,
    orchestration_run_id: Any = None,
    orchestration_task_id: Any = None,
) -> BoardTask:
    """File (or reuse) the board ticket a lane owes a Claude Code agent.

    Returns the ticket. ``task.blocked_reason`` carries the no-host warning when
    no paired host is online — the status stays ``assigned`` so the host claims
    it the moment it is back; nothing needs re-dispatching.

    PRD-239: ``actor`` (``user:<id>``) marks a ticket a human asked for in a live
    chat turn — its consent is the operator's own (D16), not the lane's standing
    grant. ``resume_session_id`` + ``resume_host_id`` ask the host that ran the
    previous session of the same conversation to continue it (``claude --resume``).
    ``orchestration_run_id`` / ``orchestration_task_id`` tie a mission task's
    ticket to its run; the step's own card is that ticket (F094, ``claim_step_card``).
    """
    existing = open_ticket_for_source(db, workspace_id, source_type, source_id)
    if existing is not None:
        logger.info("[CliTicketLane] %s/%s already has open ticket #%s — reusing", source_type, source_id, existing.id)
        return existing
    if orchestration_task_id is not None:
        card = claim_step_card(
            db, workspace_id=workspace_id, agent_id=agent_id, prompt=prompt, source_id=source_id,
            orchestration_run_id=orchestration_run_id, orchestration_task_id=orchestration_task_id,
        )
        if card is not None:
            return card
    task = BoardTask(
        workspace_id=workspace_id,
        title=title[:255],
        description=prompt,
        priority=priority if priority in ("low", "medium", "high", "urgent") else "medium",
        assigned_agent_id=agent_id,
        status="assigned",
        created_by_type=created_by_type or "system",
        created_by_id=created_by_id or source_type,
        source_type=source_type,
        source_id=source_id,
        review_mode=review_mode,
        tags=list(tags or []),
        orchestration_run_id=orchestration_run_id,
        orchestration_task_id=orchestration_task_id,
    )
    if resume_session_id:
        # Read at claim (cli_host_service) and honoured only by the same host —
        # a transcript lives on one machine. The claim rebuilds runtime_ref.
        task.runtime_ref = {
            "resume_session_id": str(resume_session_id),
            "resume_host_id": str(resume_host_id) if resume_host_id else None,
        }
    task.blocked_reason = _waiting_line(db, workspace_id, agent_id)
    db.add(task)
    db.commit()
    db.refresh(task)
    # PRD-234 (2026-09-07): the operator's standing schedule is their approval on
    # the local edition — the dispatcher's claim finds an active grant instead of
    # parking the heartbeat behind 'always_ask'. No-op on SaaS. PRD-239: a ticket
    # a human typed for in chat carries that human's consent instead (D16).
    from services.board_consent import WHY_ASKED_IN_CHAT, consent_for_created_ticket, consent_for_lane_ticket
    if actor:
        consent_for_created_ticket(db, workspace_id=workspace_id, task=task, actor=actor, why=WHY_ASKED_IN_CHAT)
    else:
        consent_for_lane_ticket(db, workspace_id=workspace_id, task=task, source_type=source_type)
    _notify(db, workspace_id, task)
    db.commit()  # F119: the notices ride this commit (after the consent, for the dispatch wake)
    logger.info("[CliTicketLane] filed ticket #%s for agent %s from %s/%s", task.id, agent_id, source_type, source_id)
    return task


def _waiting_line(db: Session, workspace_id: Any, agent_id: int) -> Optional[str]:
    """What a ticket waits on while no host that runs its agent's CLI is online."""
    if not host_online(db, workspace_id):
        return NO_HOST_REASON
    return no_cli_host_reason_for(db, workspace_id, agent_cli_provider(db, agent_id))


def queued_line(task: BoardTask) -> str:
    """The one line a lane replies with."""
    line = QUEUED_LINE.format(task_id=task.id)
    reason = getattr(task, "blocked_reason", None)
    if is_no_cli_host_reason(reason):
        head = reason.split(" — ")[0]                      # "Waiting for a CLI host that runs codex"
        line += f" ({head[0].lower()}{head[1:]} — none online runs it yet)"
    elif reason:
        line += " (no CLI host is online yet — start it with `make cli-host`)"
    return line


def _notify(db: Session, workspace_id: Any, task: BoardTask, event: str = "task_created") -> None:
    """Board SSE + dispatcher wake, both fail-soft (the same two calls the HTTP
    create path makes)."""
    try:
        from services.board_events import notify_board_event
        notify_board_event(db, workspace_id=str(workspace_id), task_id=task.id, status=task.status, event=event)
    except Exception:  # noqa: BLE001
        logger.debug("[CliTicketLane] board notify skipped", exc_info=True)
    try:
        from services.board_dispatcher import notify_task_available
        notify_task_available(db, workspace_id=str(workspace_id), task_id=task.id)
    except Exception:  # noqa: BLE001
        logger.debug("[CliTicketLane] dispatch notify skipped", exc_info=True)


def source_id_for(prefix: str, key: Any, at: Optional[datetime] = None) -> str:
    """A stable per-fire source id: ``heartbeat`` wants ONE open ticket per agent
    (no timestamp); a scheduled task wants one per run (timestamp)."""
    if at is None:
        return f"{prefix}:{key}"
    return f"{prefix}:{key}:{at.astimezone(timezone.utc):%Y%m%dT%H%M}"


# ── PRD-239: the chat lane, and lanes that wait for the session to end ───────

CHAT_SOURCE_TYPE = "chat"
RECIPE_SOURCE_TYPE = "recipe"
MISSION_SOURCE_TYPE = "mission"
TERMINAL_STATUSES: Sequence[str] = ("done", "review", "failed", "cancelled", "closed")
RUNNING_STATUSES: Sequence[str] = ("assigned", "in_progress", "blocked")
DEFAULT_LANE_POLL_SECONDS = 5

# F094: what a mission step's card keeps of its earlier runs.
PREVIOUS_RUNS_KEPT = 10
PREVIOUS_RUN_NOTE_CHARS = 500


def is_lane_owned(card: Any) -> bool:
    """F094: this mission step's card is run by the session lane, which alone
    writes its status (its ``source_id`` is the lane's)."""
    return str(getattr(card, "source_id", None) or "").startswith(f"{MISSION_SOURCE_TYPE}:")


def release_step_card(card: Any) -> None:
    """F094: the step went to an agent the lane does not run; the mission's
    state is the card's status again."""
    card.source_id = None


# F094: the mission's verdict on a step whose session it no longer waits for, as
# a note on the step's card. The card's status stays the session's.
MISSION_NOTE_BY = "the mission"
STILL_WORKING = "The session is still working, and its result will land here."
NOT_STARTED = "No session has started it yet; if one does, its result will land here."
STOPPED_WAITING_NOTE = "The mission stopped waiting for this step{after}. {state}"
CANCELLED_NOTE = "The mission was cancelled while this step ran. {state}"


def _state_line(card_status: str) -> str:
    return NOT_STARTED if card_status == "assigned" else STILL_WORKING


def stopped_waiting_note(waited_s: Any) -> Callable[[str], str]:
    """The verdict when the lane gave up waiting after ``waited_s`` seconds."""
    try:
        minutes = max(1, round(int(waited_s) / 60))
        after = f" after {minutes} minute{'' if minutes == 1 else 's'}"
    except (TypeError, ValueError):
        after = ""
    return lambda status: STOPPED_WAITING_NOTE.format(after=after, state=_state_line(status))


def cancelled_note(card_status: str) -> str:
    """The verdict when the owner cancelled the mission while the step ran."""
    return CANCELLED_NOTE.format(state=_state_line(card_status))


def note_open_step_cards(
    db: Session,
    *,
    note_for: Callable[[str], str],
    run_id: Any = None,
    orchestration_task_id: Any = None,
) -> List[int]:
    """F094: write the mission's verdict on the open cards the lane runs for a
    run or one step (``note_for(card status)`` gives the words). The card's
    status is not touched: it ends on its session's own outcome. Returns the
    cards noted; nothing without a run or a step to look in."""
    from services.cli_host_service import append_session_note, publish_note_line
    from services.orchestration_board_bridge import STEP_CARD_SOURCE_TYPE

    if run_id is None and orchestration_task_id is None:
        return []
    noted: List[int] = []
    # Each read and write runs in its own SAVEPOINT: a statement that fails there
    # rolls back to it, never the caller's transaction, which still holds the
    # cancel or the step's failure it is about to commit.
    try:
        with db.begin_nested():
            query = db.query(BoardTask).filter(
                BoardTask.source_type == STEP_CARD_SOURCE_TYPE, BoardTask.status.in_(list(RUNNING_STATUSES)),
            )
            if run_id is not None:
                query = query.filter(BoardTask.orchestration_run_id == run_id)
            if orchestration_task_id is not None:
                query = query.filter(BoardTask.orchestration_task_id == orchestration_task_id)
            cards = [card for card in query.all() if is_lane_owned(card)]
    except Exception:  # noqa: BLE001 -- a note never stops the mission recording the step or the cancel
        logger.warning("[CliTicketLane] could not find the cards to note (run %s, step %s)",
                       run_id, orchestration_task_id, exc_info=True)
        return noted
    for card in cards:
        try:
            with db.begin_nested():
                entry = append_session_note(db, task_id=card.id, workspace_id=card.workspace_id,
                                            note=note_for(card.status), by=MISSION_NOTE_BY)
        except Exception:  # noqa: BLE001 -- as above
            logger.warning("[CliTicketLane] could not note the mission's verdict on card #%s", card.id,
                           exc_info=True)
            continue
        publish_note_line(card.workspace_id, card.id, entry["note"])
        noted.append(card.id)
    return noted


def _previous_runs(card: BoardTask) -> list:
    """The card's earlier runs, newest last, with the one that just ended."""
    planning = card.planning_data if isinstance(card.planning_data, dict) else {}
    ref = card.runtime_ref if isinstance(card.runtime_ref, dict) else {}
    ended = {
        "status": card.status,
        "finished_at": card.completed_at.isoformat() if card.completed_at else None,
        "note": (card.result or card.error_message or "")[:PREVIOUS_RUN_NOTE_CHARS],
        "deliverable_ids": [d["id"] for d in ref.get("deliverables") or []
                            if isinstance(d, dict) and d.get("id") is not None],
    }
    earlier = planning.get("previous_runs")
    return [*(earlier if isinstance(earlier, list) else []), ended][-PREVIOUS_RUNS_KEPT:]


def claim_step_card(
    db: Session,
    *,
    workspace_id: Any,
    agent_id: int,
    prompt: str,
    source_id: str,
    orchestration_run_id: Any,
    orchestration_task_id: Any,
) -> Optional[BoardTask]:
    """F094 (night 5): a mission step run by a Claude Code agent runs on the
    step's own card, the one dispatch filed, not on a ticket beside it. Step
    2fa5467f had four cards: #964 said done while #980, the ticket its session
    worked, sat in review, and each re-run filed another (#982, #985).

    The lane marks the card with its ``source_id``; from then on it alone writes
    the card's status (``sync_board_status`` skips it). While the card's session
    is still open (``REUSABLE_STATUSES``) it is waited on as it is. A card whose
    run ended is set up for the step's next run, and the ended run is kept in
    ``planning_data['previous_runs']``. ``attempts`` carries on, so a late result
    from an earlier session is refused as a stale attempt. ``None`` when the step
    has no card: the caller files a ticket then.
    """
    from services.orchestration_board_bridge import STEP_CARD_SOURCE_TYPE

    card = (
        db.query(BoardTask)
        .filter(
            BoardTask.workspace_id == workspace_id,
            BoardTask.source_type == STEP_CARD_SOURCE_TYPE,
            BoardTask.orchestration_task_id == orchestration_task_id,
        )
        .order_by(BoardTask.id.asc())
        .first()
    )
    if card is None:
        return None
    if card.source_id == source_id and card.status in REUSABLE_STATUSES:
        logger.info("[CliTicketLane] %s: its card #%s is still open — waiting on it", source_id, card.id)
        return card
    if card.source_id == source_id:
        planning = card.planning_data if isinstance(card.planning_data, dict) else {}
        card.planning_data = {**planning, "previous_runs": _previous_runs(card)}
    card.source_id = source_id
    card.orchestration_run_id = orchestration_run_id
    card.assigned_agent_id = agent_id
    card.description = prompt
    card.status = "assigned"
    card.result = None
    card.error_message = None
    card.started_at = None
    card.completed_at = None
    card.blocked_at = None
    card.lease_until = None
    card.runtime_ref = None
    card.blocked_reason = _waiting_line(db, workspace_id, agent_id)
    db.commit()
    db.refresh(card)
    from services.board_consent import consent_for_lane_ticket
    consent_for_lane_ticket(db, workspace_id=workspace_id, task=card, source_type=MISSION_SOURCE_TYPE)
    _notify(db, workspace_id, card, event="status_changed")
    db.commit()
    logger.info("[CliTicketLane] %s runs on the step's card #%s (agent %s)", source_id, card.id, agent_id)
    return card


SESSION_MODE_TERMINAL = "terminal"


def session_agent_terminal_message(db: Session, agent_id: Any) -> str:
    """PRD-239 S7 v2: what a chat message to a session agent gets back — the
    agent talks in the Runtime Canvas terminal, not through the chat lane."""
    try:
        name = db.query(Agent.name).filter(Agent.id == int(agent_id)).scalar() or "This agent"
    except Exception:  # noqa: BLE001
        name = "This agent"
    return (
        f"{name} runs as a Claude Code session in the Canvas terminal. Pick {name} in the agent menu "
        f"or open the session from the board to talk to {name} there."
    )


def session_source_id(chat_id: Any) -> str:
    """``chat:<chat id>:session`` — ONE session ticket per conversation and agent
    (PRD-239 S7 v2): the Runtime Canvas resumes it every time it opens."""
    return f"{CHAT_SOURCE_TYPE}:{chat_id}:session"


def session_ticket_for(db: Session, workspace_id: Any, chat_id: Any, agent_id: int) -> Optional[BoardTask]:
    """This conversation's session ticket for this agent, whatever its status."""
    try:
        return (
            db.query(BoardTask)
            .filter(
                BoardTask.workspace_id == workspace_id,
                BoardTask.assigned_agent_id == int(agent_id),
                BoardTask.source_type == CHAT_SOURCE_TYPE,
                BoardTask.source_id == session_source_id(chat_id),
            )
            .order_by(BoardTask.id.desc())
            .first()
        )
    except Exception:  # noqa: BLE001 — a broken lookup means "no session yet"
        logger.debug("[CliTicketLane] session ticket lookup failed", exc_info=True)
        return None


def open_session_ticket(
    db: Session, *, workspace_id: Any, agent: Any, chat_id: Any, host: Any, actor: Optional[str],
) -> Tuple[BoardTask, bool]:
    """The ticket behind a Runtime Canvas session with ``agent`` in ``chat_id``:
    the existing one (resumed), or a new one — ``(ticket, created)``.

    The ticket is the session's record: its ``runtime_ref`` carries the Claude
    Code session id the host starts or resumes, the working directory and the
    host. It is NEVER dispatched: no lease (the sweeper leaves ``in_progress``
    tickets without one alone) and no ``assigned`` phase for the claim loop.
    ``TerminalOpened``/``TerminalClosed`` from the host move it between
    ``in_progress`` and ``done``.
    """
    from uuid import uuid4

    from config import config
    from services.cli_host_service import default_session_folder, explorer_root_for

    cfg = getattr(agent, "configuration", None) or {}
    cwd = cfg.get(CONFIG_WORKING_DIRECTORY_KEY) or default_session_folder(db, workspace_id)
    existing = session_ticket_for(db, workspace_id, chat_id, int(agent.id))
    if existing is not None:
        ref = getattr(existing, "runtime_ref", None)
        ref = ref if isinstance(ref, dict) else {}
        if (ref.get("cwd") or None) != cwd:
            # The agent's folder changed: a Claude Code session belongs to the
            # folder its transcript lives in, so this conversation gets a NEW
            # session there (the old one stays resumable from its own folder).
            existing.runtime_ref = {
                **ref,
                "cwd": cwd,
                "explorer_root": explorer_root_for(existing.id, cwd, workspace_id, getattr(config, "LOCAL_PROJECTS_DIR", "") or None),
                "session_id": str(uuid4()),
                "cli_session_id": None,
                "previous_session": {"cwd": ref.get("cwd"), "session_id": ref.get("cli_session_id") or ref.get("session_id")},
            }
            db.commit()
            logger.info("[CliTicketLane] session ticket #%s follows agent %s to %s", existing.id, agent.id, cwd)
        return existing, False

    name = getattr(agent, "name", None) or "the agent"
    task = BoardTask(
        workspace_id=workspace_id,
        title=f"Session with {name}"[:255],
        description=(
            f"Interactive Claude Code session with {name}, opened from the chat. "
            "You type in the Canvas terminal; the session runs on your machine under your own login."
        ),
        priority="medium",
        assigned_agent_id=int(agent.id),
        status="in_progress",
        lease_until=None,
        created_by_type="user",
        created_by_id=actor or "operator",
        source_type=CHAT_SOURCE_TYPE,
        source_id=session_source_id(chat_id),
        review_mode="human",  # F180: the board's word ('manual' was PRD-234's)
        tags=["session"],
    )
    session_id = str(uuid4())
    task.runtime_ref = {
        "runtime": "cli",
        "mode": SESSION_MODE_TERMINAL,
        "host_id": str(host.id),
        "session_id": session_id,
        "cwd": cwd,
        "provider": cfg.get(CONFIG_PROVIDER_KEY),
        "model": cfg.get(CONFIG_MODEL_KEY),
        "explorer_root": None,
    }
    db.add(task)
    db.flush()
    task.runtime_ref = {
        **task.runtime_ref,
        "explorer_root": explorer_root_for(task.id, cwd, workspace_id, getattr(config, "LOCAL_PROJECTS_DIR", "") or None),
    }
    db.commit()
    db.refresh(task)
    from services.board_consent import WHY_ASKED_IN_CHAT, consent_for_created_ticket
    if actor:
        consent_for_created_ticket(db, workspace_id=workspace_id, task=task, actor=actor, why=WHY_ASKED_IN_CHAT)
    try:
        from services.board_events import notify_board_event
        notify_board_event(db, workspace_id=str(workspace_id), task_id=task.id, status=task.status, event="task_created")
    except Exception:  # noqa: BLE001
        logger.debug("[CliTicketLane] board notify skipped", exc_info=True)
    db.commit()  # F119: the notice rides this commit — the route returns without another
    logger.info("[CliTicketLane] opened session ticket #%s for agent %s in chat %s", task.id, agent.id, chat_id)
    return task, True


def exec_result_for(task: Any) -> dict:
    """A finished ticket as the ``execute_with_prompt`` result shape the playbook
    executor and the mission dispatcher already read (status/result/error/tokens)."""
    ref = getattr(task, "runtime_ref", None)
    ref = ref if isinstance(ref, dict) else {}
    usage = ref.get("usage") if isinstance(ref.get("usage"), dict) else {}
    try:
        tokens = int(usage.get("total_tokens") or 0) or (
            int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)
        )
    except (TypeError, ValueError):
        tokens = 0
    status = getattr(task, "status", None)
    from services.cli_host_service import SESSION_CONNECTED_KEY, SESSION_TOOLS_OFFERED_KEY

    base = {
        "runtime": RUNTIME_CLI,
        "task_id": task.id,
        "board_status": status,
        "tokens_used": tokens,
        "execution": {"tokens_used": tokens, "tool_calls": [], "messages": []},
        # F131: True / False when the claim offered Automatos tools, None when it did not say.
        "session_connected": (bool(ref.get(SESSION_CONNECTED_KEY))
                              if ref.get(SESSION_TOOLS_OFFERED_KEY) is True else None),
    }
    if status in ("done", "review"):
        result = getattr(task, "result", None) or ""
        if status == "review":
            result = (result + "\n\n" if result else "") + (
                f"(ticket #{task.id} is held for review on the board)"
            )
        return {"status": "success", "result": result, **base}
    if status == "cancelled":
        return {"status": "cancelled", "result": "", "error": f"ticket #{task.id} was cancelled", **base}
    error = getattr(task, "error_message", None) or f"ticket #{task.id} failed"
    return {"status": "error", "result": "", "error": error, **base}


def _ticket_is_alive(task: Any) -> bool:
    """True when this ticket is still being worked, so giving up on it would
    leave a session running with nothing watching for its result.

    Alive means: a session holds it on a live lease, or it is parked on a
    question the operator has not answered yet, or it is waiting to be picked
    up again. Only a lapsed lease means nobody is on it.
    """
    status = getattr(task, "status", None)
    if status in ("assigned", "blocked"):
        return True
    if status != "in_progress":
        return False
    lease = getattr(task, "lease_until", None)
    if lease is None:
        return True            # in_progress with no lease recorded — assume a live session
    if lease.tzinfo is None:
        lease = lease.replace(tzinfo=timezone.utc)
    return lease > datetime.now(timezone.utc)


def is_database_unreachable(exc: BaseException) -> bool:
    """F114: the database dropped the connection or is restarting (crash recovery,
    "server closed the connection unexpectedly") — not a fault in the query."""
    from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError

    if isinstance(exc, (OperationalError, InterfaceError)):
        return True
    return isinstance(exc, DBAPIError) and bool(getattr(exc, "connection_invalidated", False))


def _db_outage_grace_s() -> float:
    try:
        from config import config

        return float(getattr(config, "CLI_LANE_DB_OUTAGE_GRACE_SECONDS", 180))
    except Exception:  # noqa: BLE001
        return 180.0


async def run_cli_ticket_and_wait(
    db: Session,
    *,
    workspace_id: Any,
    agent_id: int,
    title: str,
    prompt: str,
    source_type: str,
    source_id: str,
    timeout_s: Optional[float] = None,
    hard_timeout_s: Optional[float] = None,
    poll_s: Optional[float] = None,
    on_poll: Optional[Callable[[Any], None]] = None,
    **file_kwargs: Any,
) -> dict:
    """File the ticket a step or mission task owes a session agent and wait for
    it to end (PRD-239 S3). Returns ``exec_result_for`` the ended ticket, or an
    error result that names the still-running ticket when the wait gives up —
    the session carries on; its result lands on the board.

    Two bounds, because a Claude Code session is not an API turn (night 1,
    finding 21). ``timeout_s`` is the soft deadline; reaching it only ends the
    wait if the ticket is NOT alive. A ticket that is genuinely being worked —
    ``in_progress`` on a live lease, or parked on a question a human has yet to
    answer — is waited on up to ``hard_timeout_s``. Declaring a 20-minute
    research session failed at 4 minutes is what re-queued the task and spawned
    a duplicate beside the session that was still running.

    ``on_poll(ticket)`` runs on every poll while waiting (S3b): the caller marks
    progress on its own record so a stall watchdog does not mistake a long
    session for a dead run. Its failures are logged, never raised.
    """
    import asyncio
    import time

    task = file_cli_ticket(
        db, workspace_id=workspace_id, agent_id=agent_id, title=title, prompt=prompt,
        source_type=source_type, source_id=source_id, **file_kwargs,
    )
    task_id = task.id
    if poll_s is None:
        try:
            from config import config

            poll_s = float(getattr(config, "CLI_LANE_POLL_SECONDS", DEFAULT_LANE_POLL_SECONDS))
        except Exception:  # noqa: BLE001
            poll_s = float(DEFAULT_LANE_POLL_SECONDS)
    poll_s = max(0.5, float(poll_s))
    started = time.monotonic()
    extended = False
    outage_since: Optional[float] = None
    while True:
        try:
            db.expire_all()  # see the host's writes, not this session's cache
            current = db.query(BoardTask).filter(BoardTask.id == task_id).first()
        except Exception as exc:  # noqa: BLE001 — only a lost database is waited out
            if not is_database_unreachable(exc):
                raise
            # F114 (run 4): Postgres crash-restarted mid-wait and every run waiting
            # on a session died in the same second while its ticket worked on.
            # The session is not the database: roll back, wait, poll again.
            outage_since = outage_since if outage_since is not None else time.monotonic()
            if time.monotonic() - outage_since > _db_outage_grace_s():
                raise
            logger.warning("[CliTicketLane] database unreachable while waiting on ticket #%s — polling again: %s",
                           task_id, str(exc).splitlines()[0][:160])
            try:
                db.rollback()
            except Exception:  # noqa: BLE001 — the pool replaces the dead connection
                pass
            await asyncio.sleep(min(poll_s, 5.0))
            continue
        if outage_since is not None:
            logger.info("[CliTicketLane] database back after %d s — still waiting on ticket #%s",
                        int(time.monotonic() - outage_since), task_id)
            outage_since = None
        if current is None:
            return {"status": "error", "error": f"ticket #{task_id} disappeared while the session ran",
                    "runtime": RUNTIME_CLI, "task_id": task_id}
        if current.status in TERMINAL_STATUSES:
            return exec_result_for(current)
        if on_poll is not None:
            try:
                on_poll(current)
            except Exception:  # noqa: BLE001 — progress marking must never end the wait
                logger.debug("[CliTicketLane] on_poll failed for ticket #%s", task_id, exc_info=True)
        waited = time.monotonic() - started
        if timeout_s is not None and waited >= timeout_s:
            ceiling = hard_timeout_s if hard_timeout_s is not None else timeout_s
            if _ticket_is_alive(current) and waited < float(ceiling):
                if not extended:
                    extended = True
                    logger.info(
                        "[CliTicketLane] ticket #%s is %s past its %ss deadline — the session is "
                        "alive, waiting up to %ss rather than re-queuing it",
                        task_id, current.status, int(timeout_s), int(ceiling),
                    )
            else:
                return {
                    "status": "error",
                    "error": (
                        f"ticket #{task_id} is still running after {int(waited)} s — the Claude Code "
                        "session carries on and its result lands on the board"
                    ),
                    "runtime": RUNTIME_CLI,
                    "task_id": task_id,
                    "timed_out": True,
                    "still_running": _ticket_is_alive(current),
                    "waited_s": int(waited),
                }
        await asyncio.sleep(poll_s)
