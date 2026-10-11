"""
PRD-207 / PRD-258: one Auto Live call on Retell's custom-LLM WebSocket
=======================================================================

The session behind ``WS /api/voice/retell/llm-websocket/{call_id}``. The route
(``api/voice_retell.py``) refuses unminted and superseded calls before accept;
this class runs the accepted socket. Protocol: the server speaks FIRST with a
``config`` frame; ``call_details`` delivers the dynamic variables once and is
answered by an empty response 0 (the human has the floor); ``ping_pong`` must
be echoed; ``update_only`` owes nothing; ``response_required`` /
``reminder_required`` start a turn — a newer one supersedes a still-streaming
turn (barge-in), so the old turn task is cancelled.

PRD-258 US-001: every turn writes ONE ``voice_turns`` row once its stream ends
(or it is superseded), with the moments this side of the socket sees — the
first content frame sent, and Auto's own first frame — beside the facts the
agent stream reports in-band (``modules/voice/turn_timing.py``).

Moved out of ``api/voice_retell.py`` by PRD-258 US-001 (D6: that file never
grows); the route lends the session its turn stream and the call-status read,
late-bound, so the agent loop stays where it is.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
from dataclasses import dataclass
from time import monotonic
from typing import Any, AsyncIterator, Callable, Optional

from config import config
from modules.voice import telemetry, turn_timing
from modules.voice.providers.retell import (
    INTERACTION_CALL_DETAILS,
    INTERACTION_PING_PONG,
    INTERACTION_REMINDER,
    INTERACTION_RESPONSE_REQUIRED,
    INTERACTION_UPDATE_ONLY,
    RetellLLMRequest,
    parse_llm_request,
    wrap_ws_response,
)

logger = logging.getLogger(__name__)

# On socket exit, an in-flight turn gets this long to finish in the
# foreground; past it the turn keeps running detached (shielded) so the
# reply still persists to the thread.
TURN_EXIT_GRACE_SECONDS = 3.0
# The call status a newer mint gives the caller's older calls.
CALL_STATUS_SUPERSEDED = "superseded"
# Close codes: a normal close, sent when a superseded call tries to speak.
WS_CLOSE_NORMAL = 1000
# Handshake per Retell's own demo: ask for call_details; enable auto_reconnect.
CONFIG_FRAME = {"response_type": "config", "config": {"auto_reconnect": True, "call_details": True}}
# The begin frame: an empty response 0 — the human opens.
BEGIN_RESPONSE_ID = 0


@dataclass(frozen=True)
class SocketPorts:
    """What the route lends a call: Auto's turn stream and the call's live status."""

    stream_turn: Callable[[RetellLLMRequest], AsyncIterator[dict]]
    call_status: Callable[[str], Optional[str]]


def _closing_frame(response_id: int) -> dict:
    return wrap_ws_response({"response_id": response_id, "content": "", "content_complete": True})


class _TurnMarks:
    """One turn's clock readings and facts — a per-turn accumulator owned by its
    ``respond`` task and the ack it arms (never shared across turns)."""

    def __init__(self) -> None:
        self.started = monotonic()
        self.first_any_at: Optional[float] = None  # any Auto frame (the existing log line)
        self.first_frame_at: Optional[float] = None  # first content frame sent, ack included
        self.first_auto_frame_at: Optional[float] = None  # Auto's first content frame
        self.response_chars = 0
        self.auto_delivered = False  # Auto's own words reached the socket (the ack never counts)
        self.sendable = True
        self.facts = turn_timing.TurnFacts()

    def content_sent(self) -> None:
        if self.first_frame_at is None:
            self.first_frame_at = monotonic()

    def auto_content(self, content: str) -> None:
        self.response_chars += len(content)
        if content and self.first_auto_frame_at is None:
            self.first_auto_frame_at = monotonic()


class LiveCallSocket:
    """One accepted Retell custom-LLM socket: the handshake, the turns, the summary."""

    def __init__(self, websocket: Any, call_id: str, ports: SocketPorts) -> None:
        self._ws = websocket
        self._call_id = call_id
        self._ports = ports
        self._dynamic_vars: dict[str, Any] = {}
        self._speaking: Optional[asyncio.Task] = None
        self._begun = False
        # The slow-turn filler is a ONE-TIME courtesy per call. It used to fire
        # on every slow turn, so a 3-turn utterance storm played "One moment.
        # One moment. One moment." over itself — the "five people talking"
        # Gerard heard. Once tells the caller the line is alive; twice is a stutter.
        self._ack_spent = False
        self.stats = {"pings": 0, "updates": 0, "turns": 0, "frames": 0}

    async def safe_send(self, payload: dict) -> bool:
        """Send or say why not — a severed socket must NEVER traceback the
        handler (first live calls died at the ping reply with an uncaught
        ConnectionClosedError, leaving the mic recording into a dead brain)."""
        try:
            await self._ws.send_json(payload)
            return True
        except Exception as exc:  # noqa: BLE001 — closure class varies by stack layer
            logger.warning(
                "voice_live_ws_send_failed call=%s err=%s", self._call_id, type(exc).__name__
            )
            return False

    async def run(self) -> None:
        """Drive the socket until it closes; an in-flight turn gets its exit grace."""
        from fastapi import WebSocketDisconnect

        try:
            if not await self.safe_send(dict(CONFIG_FRAME)):
                return
            while await self._dispatch(await self._ws.receive_json()):
                pass
        except WebSocketDisconnect as exc:
            logger.info("voice_live_ws_closed call=%s code=%s", self._call_id, getattr(exc, "code", "?"))
        except Exception as exc:  # noqa: BLE001 — abnormal severance is a fact of WS life
            logger.warning(
                "voice_live_ws_died call=%s err=%s", self._call_id, type(exc).__name__, exc_info=True
            )
        finally:
            await self._exit_grace()

    async def _exit_grace(self) -> None:
        # The socket is gone but the brain may be mid-reply. Do NOT cancel it:
        # give it a bounded grace to finish (the reply persists to the thread
        # and reaches the open chat over SSE); past the grace the shield lets
        # it run detached to the same end. Killing it here was how a severed
        # socket turned a healthy generation into silence.
        if self._speaking is not None and not self._speaking.done():
            with contextlib.suppress(Exception):
                await asyncio.wait_for(asyncio.shield(self._speaking), TURN_EXIT_GRACE_SECONDS)
        s = self.stats
        logger.info(
            "voice_live_ws_summary call=%s turns=%s frames=%s pings=%s updates=%s begun=%s",
            self._call_id, s["turns"], s["frames"], s["pings"], s["updates"], self._begun,
        )

    async def _dispatch(self, message: dict) -> bool:
        """Answer one inbound message; False ends the socket loop."""
        itype = message.get("interaction_type")
        if itype == INTERACTION_PING_PONG:
            self.stats["pings"] += 1
            return await self.safe_send({"response_type": "ping_pong", "timestamp": message.get("timestamp")})
        if itype == INTERACTION_CALL_DETAILS:
            return await self._on_call_details(message)
        if itype in (INTERACTION_RESPONSE_REQUIRED, INTERACTION_REMINDER):
            return await self._on_turn(message, itype)
        if itype == INTERACTION_UPDATE_ONLY:
            self.stats["updates"] += 1
        return True  # unknown types: no response owed.

    async def _on_call_details(self, message: dict) -> bool:
        self._dynamic_vars = (message.get("call") or {}).get("retell_llm_dynamic_variables") or {}
        logger.info("voice_live_ws_call_details call=%s vars=%s", self._call_id, bool(self._dynamic_vars))
        # …then the begin message — the demo sends it HERE, not at accept.
        if self._begun:
            return True
        self._begun = True
        return await self.safe_send(_closing_frame(BEGIN_RESPONSE_ID))

    async def _on_turn(self, message: dict, itype: str) -> bool:
        self.stats["turns"] += 1
        logger.info(
            "voice_live_ws_turn call=%s rid=%s type=%s", self._call_id, message.get("response_id"), itype
        )
        # ONE LIVE CALL PER USER: a newer mint marked this call 'superseded'.
        # It must NOT speak — several superseded calls on one mic is the "Auto
        # talking over herself" bug. Close quietly (the newest call owns the
        # conversation). Checked at the turn boundary: exactly where a stale
        # call would otherwise talk.
        if self._ports.call_status(self._call_id) == CALL_STATUS_SUPERSEDED:
            logger.info("voice_live_ws_superseded_closing call=%s", self._call_id)
            self._cancel_speaking()  # silence any in-flight reply at once
            await self._ws.close(code=WS_CLOSE_NORMAL)
            return False
        self._cancel_speaking()
        req = parse_llm_request(
            {
                "response_id": message.get("response_id"),
                "interaction_type": INTERACTION_RESPONSE_REQUIRED,
                "transcript": message.get("transcript") or [],
                "call": {"call_id": self._call_id, "retell_llm_dynamic_variables": self._dynamic_vars},
            }
        )
        self._speaking = asyncio.create_task(self.respond(req))
        return True

    def _cancel_speaking(self) -> None:
        if self._speaking is not None and not self._speaking.done():
            self._speaking.cancel()

    async def respond(self, req: RetellLLMRequest) -> None:
        """Stream one turn to Retell, then record where its time went."""
        turn = _TurnMarks()
        ack_task = self._arm_ack(req, turn)
        try:
            await self._relay(req, turn, ack_task)
        except asyncio.CancelledError:
            # Superseded turn (barge-in): Retell has moved on — stop quietly.
            raise
        except Exception:  # noqa: BLE001 — a mid-turn fault closes the turn cleanly
            logger.exception("retell_ws_stream_failed call=%s", self._call_id)
            await self.safe_send(_closing_frame(req.response_id))
        finally:
            if ack_task is not None and not ack_task.done():
                ack_task.cancel()
            self._record_turn(req, turn)

    async def _relay(self, req: RetellLLMRequest, turn: _TurnMarks, ack_task: Optional[asyncio.Task]) -> None:
        async for frame in self._ports.stream_turn(req):
            facts = turn_timing.facts_of(frame)
            if facts is not None:
                turn.facts = turn_timing.merge_facts(turn.facts, facts)
                continue
            if turn.first_any_at is None:
                turn.first_any_at = monotonic()
                if ack_task is not None:
                    ack_task.cancel()
                logger.info(
                    "voice_live_ws_first_frame call=%s rid=%s ms=%d",
                    self._call_id, req.response_id, int((turn.first_any_at - turn.started) * 1000),
                )
            self.stats["frames"] += 1
            content = frame.get("content") or ""
            turn.auto_content(content)
            if not turn.sendable:
                continue
            if await self.safe_send(wrap_ws_response(frame)):
                if content:
                    turn.content_sent()
                    turn.auto_delivered = True
            else:
                # The socket died mid-reply (Retell reconnects live calls).
                # Keep DRAINING: the streaming service persists the reply at
                # stream end, so the words still land in the thread and the
                # open chat hears them over SSE — only the TTS leg is lost.
                turn.sendable = False

    def _arm_ack(self, req: RetellLLMRequest, turn: _TurnMarks) -> Optional[asyncio.Task]:
        """Slow-turn honesty: if the brain has said NOTHING by the deadline, speak
        a short acknowledgment — the caller hears life (and the log gains the
        smoking gun) instead of dead air while tools run. Spoken AT MOST ONCE per
        call; the slow-turn WARNING still logs every time."""
        ack_seconds = float(config.VOICE_LIVE_FIRST_FRAME_ACK_SECONDS or 0)
        if ack_seconds <= 0:
            return None
        return asyncio.create_task(self._slow_turn_ack(req, turn, ack_seconds))

    async def _slow_turn_ack(self, req: RetellLLMRequest, turn: _TurnMarks, ack_seconds: float) -> None:
        await asyncio.sleep(ack_seconds)
        ack_text = str(config.VOICE_LIVE_FIRST_FRAME_ACK_TEXT or "").strip()
        logger.warning(
            "voice_live_ws_turn_slow call=%s rid=%s waited_ms=%d spoken=%s",
            self._call_id, req.response_id, int((monotonic() - turn.started) * 1000), not self._ack_spent,
        )
        if not ack_text or self._ack_spent:
            return
        self._ack_spent = True
        frame = {"response_id": req.response_id, "content": ack_text + " ", "content_complete": False}
        if await self.safe_send(wrap_ws_response(frame)):
            turn.content_sent()

    def _record_turn(self, req: RetellLLMRequest, turn: _TurnMarks) -> None:
        """Hand the turn's ONE voice_turns row to a worker thread (fire-and-forget:
        the write never blocks this loop and never fails the turn). A turn whose
        stream never said where it landed (binding refused, cancelled first) has
        no proven workspace, so no row."""
        facts = turn.facts
        if not facts.workspace_id:
            logger.info("voice_turn_unrecorded call=%s rid=%s reason=no_binding", self._call_id, req.response_id)
            return
        timing = turn_timing.voice_timing_fields(
            started=turn.started,
            first_frame_at=turn.first_frame_at,
            first_auto_frame_at=turn.first_auto_frame_at,
            tools_before_speech=facts.tools_before_speech,
            prep_ms=facts.prep_ms,
        )
        write = functools.partial(
            telemetry.record_voice_turn,
            workspace_id=facts.workspace_id,
            conversation_id=facts.conversation_id,
            message_id=None,
            call_id=req.call_id,
            # STT/TTS run vendor-side: honestly 0, not faked (Retell's own
            # figures land in voice_calls.vendor_latency).
            stt_latency_ms=0,
            tts_latency_ms=0,
            total_ms=(monotonic() - turn.started) * 1000.0,
            transcript=req.user_text,
            response_text="x" * turn.response_chars,  # lengths-only discipline: only len() is stored
            truncated=False,
            # Auto's words actually went to the vendor's TTS (a drain after a
            # dead socket produced text but delivered no audio).
            audio_delivered=turn.auto_delivered,
            timing=timing,
        )
        try:
            asyncio.get_running_loop().run_in_executor(None, write)
        except RuntimeError:  # the loop's executor is already shut down (process exit)
            logger.exception("voice_turn_record_not_scheduled call=%s rid=%s", self._call_id, req.response_id)
