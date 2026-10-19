"""
PRD-258 US-004: the voice-latency runner (the CLI)
===================================================

Times Auto Live's turn without a person on a call: it speaks Retell's
custom-LLM socket protocol to the backend it runs beside. The only part of
this package that does I/O. It never writes a file: the ask set comes on
stdin, the JSON summary and the results line go to stdout.

Simulator (inside the backend container)::

    docker exec -i automatos_backend python -m scripts.eval.voice_latency.run \\
        --workspace <workspace id> --user <users.id> --sha <build sha> < asks.json

It refuses while ``voice.live_enabled`` is off; mints a ``minted`` voice_calls
row with provider ``sim`` bound to a new chat in the workspace; opens
``ws://localhost:<port>/api/voice/retell/llm-websocket/<call_id>``; answers
the server's ``config``; drives the asks in order (each word an
``update_only`` at speaking pace, then ``response_required``); and joins the
client-side times with the call's ``voice_turns`` rows. The sim call is ended
(0 s) when the run stops, so it never holds a minute of the workspace's cap.

Real call (no socket, no insert)::

    docker exec -i automatos_backend python -m scripts.eval.voice_latency.run --call <call_id>

joins that call's voice_turns rows with its ``voice_calls.vendor_latency``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime as dt
import json
import logging
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from time import monotonic
from typing import Any, Dict, List, Optional, Sequence, Tuple

from websockets.client import connect as ws_connect
from websockets.exceptions import WebSocketException

from config import config
from modules.voice.live_socket import BEGIN_RESPONSE_ID
from modules.voice.retell_api import RETELL_LLM_WEBSOCKET_PATH
from scripts.eval.voice_latency import protocol, report
from scripts.eval.voice_latency.protocol import Ask, TurnWatch

logger = logging.getLogger(__name__)

# The port uvicorn listens on INSIDE the backend container: orchestrator/Dockerfile's
# CMD (`uvicorn main:app --host 0.0.0.0 --port 8000`) and orchestrator/start.sh's default.
BACKEND_CONTAINER_PORT = 8000
BACKEND_HOST = "localhost"
SIM_PROVIDER = "sim"
SIM_CALL_PREFIX = "sim_"
CALL_STATUS_MINTED = "minted"
CALL_STATUS_ENDED = "ended"
SIM_END_REASON = "sim_run_complete"
UNKNOWN_SHA = "unknown"
ROW_POLL_SECONDS = 0.25  # how often a turn's voice_turns row is looked for
ROW_WAIT_SECONDS = 20.0  # a turn's row is written off-loop once its reply is saved
EXIT_REFUSED = 2
CLI_DESCRIPTION = "Time Auto Live's turn over Retell's socket protocol, or report a real call."
# Put on the inbox when the server's socket ends, so a waiting turn stops at once.
SOCKET_CLOSED = object()
# The voice_turns columns the report reads.
TURN_COLUMNS = (
    "first_frame_ms", "first_auto_frame_ms", "total_ms", "prep_ms", "tools_before_speech", "audio_delivered",
)


class RunRefused(Exception):
    """The run cannot start or go on; the message says why, for the owner."""


class SocketClosed(RunRefused):
    """The server closed the socket (or it failed) while the run still needed it."""


@dataclass(frozen=True)
class SimCall:
    """The sim call's mint: the ids its dynamic variables carry."""

    call_id: str
    workspace_id: str
    user_id: int
    chat_id: str


def emit(text: str) -> None:
    """The CLI's one output: stdout, for the owner to read or append to results.md."""
    print(text)


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


# ── the database (the backend's own session) ────────────────────────────────


def mint_sim_call(workspace_id: str, user_id: int) -> SimCall:
    """Insert the ``minted`` sim row, bound to a new chat the run owns."""
    from core.database.database import get_db_session
    from core.models.voice_calls import VoiceCall
    from core.models.workspaces import Workspace
    from modules.voice.call_binding import create_voice_chat

    ws_uuid = uuid.UUID(workspace_id)
    call_id = f"{SIM_CALL_PREFIX}{uuid.uuid4().hex}"
    with get_db_session() as db:
        if db.query(Workspace.id).filter(Workspace.id == ws_uuid).first() is None:
            raise RunRefused(f"workspace {workspace_id} does not exist")
        chat_id = str(create_voice_chat(db, user_id=user_id, workspace_id=ws_uuid).id)
        db.add(
            VoiceCall(
                call_id=call_id, provider=SIM_PROVIDER, workspace_id=ws_uuid,
                user_id=user_id, chat_id=chat_id, status=CALL_STATUS_MINTED,
            )
        )
    return SimCall(call_id=call_id, workspace_id=str(ws_uuid), user_id=user_id, chat_id=chat_id)


def end_sim_call(call_id: str) -> None:
    """End the sim row at 0 s: no vendor minutes were spent, none are counted."""
    from core.database.database import get_db_session
    from core.models.voice_calls import VoiceCall

    with get_db_session() as db:
        db.query(VoiceCall).filter(VoiceCall.call_id == call_id).update(
            {
                VoiceCall.status: CALL_STATUS_ENDED,
                VoiceCall.ended_at: dt.datetime.utcnow(),
                VoiceCall.duration_seconds: 0,
                VoiceCall.disconnect_reason: SIM_END_REASON,
            },
            synchronize_session=False,
        )


def read_turn_rows(call_id: str) -> List[Dict[str, Any]]:
    """The call's voice_turns rows, oldest first, as plain dicts of ``TURN_COLUMNS``."""
    from core.database.database import get_db_session
    from core.models.voice_turns import VoiceTurn

    with get_db_session() as db:
        rows = db.query(VoiceTurn).filter(VoiceTurn.call_id == call_id).order_by(VoiceTurn.id).all()
        return [{name: getattr(row, name) for name in TURN_COLUMNS} for row in rows]


def read_vendor_latency(call_id: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """(the call exists, its vendor_latency)."""
    from core.database.database import get_db_session
    from core.models.voice_calls import VoiceCall

    with get_db_session() as db:
        row = db.query(VoiceCall.vendor_latency).filter(VoiceCall.call_id == call_id).first()
        return (row is not None, row[0] if row is not None else None)


# ── the socket ──────────────────────────────────────────────────────────────


class SimSession:
    """One sim call on the socket: the handshake, the pings, the asks in order."""

    def __init__(self, ws: Any, call: SimCall) -> None:
        self._ws = ws
        self._call = call
        self._inbox: asyncio.Queue = asyncio.Queue()
        # The one-time slow-turn ack; cleared once the call has heard it.
        self._ack_text = str(config.VOICE_LIVE_FIRST_FRAME_ACK_TEXT or "").strip()

    async def _send(self, frame: Dict[str, Any]) -> None:
        await self._ws.send(json.dumps(frame))

    async def read_forever(self) -> None:
        """Stamp every server frame with its arrival time, so a busy turn loop can't skew it.
        When the socket ends (closed, failed, or a frame that isn't JSON) the inbox
        gets ``SOCKET_CLOSED``, so the turn waiting on it fails at once."""
        try:
            async for raw in self._ws:
                self._inbox.put_nowait((monotonic(), json.loads(raw)))
        except Exception as exc:  # noqa: BLE001 — logged, then surfaced to the run as SocketClosed
            logger.warning("voice_latency_socket_read_failed call=%s err=%s", self._call.call_id, exc)
        finally:
            self._inbox.put_nowait(SOCKET_CLOSED)

    async def ping_forever(self) -> None:
        while True:
            await self._send(protocol.ping_frame(int(time.time() * protocol.MS_PER_SECOND)))
            await asyncio.sleep(protocol.PING_INTERVAL_SECONDS)

    async def _next(self, deadline: float) -> Tuple[float, Dict[str, Any]]:
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise asyncio.TimeoutError
        item = await asyncio.wait_for(self._inbox.get(), remaining)
        if item is SOCKET_CLOSED:
            raise SocketClosed("the backend closed the call's socket mid-run (see its voice_live_ws log lines)")
        return item

    async def handshake(self) -> bool:
        """Answer the server's config as Retell does; True if it asked for pings."""
        deadline = monotonic() + protocol.HANDSHAKE_TIMEOUT_SECONDS
        _, first = await self._next(deadline)
        if not protocol.is_config(first):
            raise RunRefused(f"the server spoke first with {first.get('response_type')!r}, not a config frame")
        if protocol.wants_call_details(first):
            call = self._call
            await self._send(
                protocol.call_details_frame(
                    call.call_id, workspace_id=call.workspace_id, user_id=call.user_id, chat_id=call.chat_id
                )
            )
            await self._await_begin(deadline)
        return protocol.wants_pings(first)

    async def _await_begin(self, deadline: float) -> None:
        while True:
            _, message = await self._next(deadline)
            if protocol.is_response_to(message, BEGIN_RESPONSE_ID):
                return

    async def ask(self, ask: Ask, response_id: int, history: protocol.Transcript) -> TurnWatch:
        """Speak one ask word by word, then time the reply to ``content_complete``."""
        frames = protocol.ask_frames(history, ask.text, response_id)
        for frame in frames[:-1]:
            await self._send(frame)
            await asyncio.sleep(protocol.WORD_INTERVAL_SECONDS)
        await self._send(frames[-1])
        watch = TurnWatch(response_id=response_id, started=monotonic(), ack_text=self._ack_text)
        deadline = watch.started + protocol.TURN_TIMEOUT_SECONDS
        with contextlib.suppress(asyncio.TimeoutError):
            while not watch.done:
                at, message = await self._next(deadline)
                watch.observe(message, at)
        if watch.ack_heard:
            self._ack_text = ""
        return watch


async def wait_for_rows(call_id: str, count: int) -> int:
    """Wait until the call has ``count`` voice_turns rows (the turn has fully ended)."""
    deadline = monotonic() + ROW_WAIT_SECONDS
    rows = await asyncio.to_thread(read_turn_rows, call_id)
    while len(rows) < count and monotonic() < deadline:
        await asyncio.sleep(ROW_POLL_SECONDS)
        rows = await asyncio.to_thread(read_turn_rows, call_id)
    if len(rows) < count:
        logger.warning("voice_latency_row_missing call=%s expected=%s found=%s", call_id, count, len(rows))
    return len(rows)


async def drive_asks(session: SimSession, call_id: str, asks: Sequence[Ask]) -> List[TurnWatch]:
    """Every ask in order; the next one waits for the last turn's row, so it never barges in."""
    history: protocol.Transcript = ()
    watches: List[TurnWatch] = []
    for index, ask in enumerate(asks):
        watch = await session.ask(ask, protocol.FIRST_RESPONSE_ID + index, history)
        watches.append(watch)
        if not watch.done:
            logger.warning("voice_latency_turn_timed_out call=%s ask=%s", call_id, index)
        history = protocol.after_turn(history, ask.text, watch.reply)
        await wait_for_rows(call_id, index + 1)
        await asyncio.sleep(protocol.BETWEEN_ASKS_SECONDS)
    return watches


async def run_socket(call: SimCall, asks: Sequence[Ask], host: str, port: int) -> List[TurnWatch]:
    """Open the call's socket and drive it; the reader and pinger stop with it."""
    url = f"ws://{host}:{port}{RETELL_LLM_WEBSOCKET_PATH}/{call.call_id}"
    async with ws_connect(url, open_timeout=protocol.CONNECT_TIMEOUT_SECONDS) as ws:
        session = SimSession(ws, call)
        helpers = [asyncio.create_task(session.read_forever())]
        try:
            if await session.handshake():
                helpers.append(asyncio.create_task(session.ping_forever()))
            return await drive_asks(session, call.call_id, asks)
        finally:
            for task in helpers:
                task.cancel()
            await asyncio.gather(*helpers, return_exceptions=True)


# ── the report ──────────────────────────────────────────────────────────────


def turn_results(
    asks: Sequence[Ask], watches: Sequence[TurnWatch], rows: Sequence[Dict[str, Any]]
) -> Tuple[List[report.TurnResult], bool]:
    """Client times per ask, joined by order with the server rows (only when the counts match)."""
    matched = len(rows) == len(asks)
    results = []
    for index, (ask, watch) in enumerate(zip(asks, watches)):
        row = rows[index] if matched else {}
        results.append(
            report.TurnResult(
                kind=ask.kind,
                first_frame_ms=watch.elapsed_ms(watch.first_frame_at),
                first_auto_frame_ms=watch.elapsed_ms(watch.first_auto_frame_at),
                full_ms=watch.elapsed_ms(watch.complete_at),
                prep_ms=row.get("prep_ms"),
                tools_before_speech=row.get("tools_before_speech"),
                silent=watch.first_auto_frame_at is None,
            )
        )
    return results, matched


def simulate(args: argparse.Namespace, asks: Sequence[Ask]) -> Dict[str, Any]:
    """The simulator run: mint, drive, read back, summarise. The sim call always ends."""
    from modules.voice import live_settings

    if not live_settings.voice_live_enabled():
        raise RunRefused("Auto Live is switched off (voice.live_enabled): turn it on in Settings → Voice first")
    call = mint_sim_call(args.workspace, args.user)
    try:
        watches = asyncio.run(run_socket(call, asks, args.host, args.port))
    except (OSError, asyncio.TimeoutError, WebSocketException) as exc:
        logger.warning("voice_latency_socket_failed call=%s", call.call_id, exc_info=True)
        raise RunRefused(f"the call's socket failed: {type(exc).__name__}: {exc}") from exc
    finally:
        end_sim_call(call.call_id)
    rows = read_turn_rows(call.call_id)
    results, matched = turn_results(asks, watches, rows)
    summary = report.summarize(results)
    return {
        "call_id": call.call_id,
        "chat_id": call.chat_id,
        "turns": [asdict(result) for result in results],
        "server_rows": len(rows),
        "server_rows_matched": matched,
        "summary": summary,
        "results_line": report.results_line(_now_iso(), args.sha, summary),
    }


def join_call(args: argparse.Namespace) -> Dict[str, Any]:
    """The real-call report: its voice_turns rows beside Retell's own latency."""
    found, vendor_latency = read_vendor_latency(args.call)
    if not found:
        raise RunRefused(f"no voice_calls row for call {args.call}")
    rows = read_turn_rows(args.call)
    return {
        "call_id": args.call,
        "turns": len(rows),
        "vendor_latency": vendor_latency,
        "results_line": report.call_line(_now_iso(), args.sha, rows, vendor_latency),
    }


# ── the CLI ─────────────────────────────────────────────────────────────────


def read_asks(stream: Any) -> Tuple[Ask, ...]:
    """The ask set from stdin: JSON, every row checked (a bad row is named by its index)."""
    try:
        raw = json.load(stream)
    except json.JSONDecodeError as exc:
        raise protocol.AskSetError(f"the ask set on stdin is not JSON: {exc}") from exc
    return protocol.parse_asks(raw)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m scripts.eval.voice_latency.run", description=CLI_DESCRIPTION)
    parser.add_argument("--workspace", help="the workspace the sim call runs in (its id)")
    parser.add_argument("--user", type=int, help="the caller: a users.id in that workspace")
    parser.add_argument("--call", help="report a real call by its call_id instead of simulating")
    parser.add_argument("--sha", default=UNKNOWN_SHA, help="the build SHA the results line names")
    parser.add_argument("--host", default=BACKEND_HOST, help="the backend's host, seen from where this runs")
    parser.add_argument("--port", type=int, default=BACKEND_CONTAINER_PORT, help="the backend's port")
    return parser


def _validate_sim_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.workspace or args.user is None:
        parser.error("a simulator run needs --workspace and --user (or --call <call_id> to report a real call)")
    try:
        uuid.UUID(args.workspace)
    except ValueError:
        parser.error(f"--workspace {args.workspace!r} is not a workspace id")


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    from core.monitoring.request_log_guard import guard_request_logs
    guard_request_logs()  # no request URL or SDK request body in this script's logs (#1178)
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.call:
            outcome = join_call(args)
        else:
            _validate_sim_args(parser, args)
            outcome = simulate(args, read_asks(sys.stdin))
    except (RunRefused, protocol.AskSetError) as exc:
        logger.error("voice_latency_refused: %s", exc)
        return EXIT_REFUSED
    emit(json.dumps(outcome, indent=2, default=str))
    emit(outcome["results_line"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
