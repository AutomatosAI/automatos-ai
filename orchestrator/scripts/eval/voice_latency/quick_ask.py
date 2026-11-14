"""
Voice latency A-4: the quick-ask runner (the CLI)
=================================================

Times a quick-ask without a person or a phone: per ask it does what Automatos Go's voice
screen does over HTTP. The only quick-ask part of this package that does I/O. It never
writes a file: the ask set comes on stdin, the JSON summary and the results line go to
stdout. Neither holds what was asked or answered.

Inside the backend container (local edition: no sign-in)::

    docker exec -i automatos_backend python -m scripts.eval.voice_latency.quick_ask \\
        --workspace <workspace id> --sha <build sha> < asks.json

Per ask: ``POST /api/chat`` with ``spokenMode`` and a ``turnRef``; the stream is timed
(headers, first text frame, first complete sentence, end); then ``POST /api/voice/speech``
for that first sentence with the same ``turnRef`` (first body byte, last). After the run
the server's own split is read from each reply's ``context_trace["timing"]``.

Against a hosted server, ``--token-file <path>`` names a file holding a bearer token. It
is read again before every request (a session token is short-lived) and is never an
argument, a log line or output. There is no database there, so every server measure is
empty and the results line ends ``server_marks=unavailable``.

What was measured is always reported. If the read after the run fails (no database within
reach), the run is reported with ``server_marks=unavailable``. If the token file cannot be
read part-way, the run stops there and reports the asks already timed (``asks_not_sent``
in the JSON says how many were left).

Refused (exit 2) when discovery lists no ``spoken_mode``, or no ``tts`` without
``--no-speech``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from urllib.parse import urlsplit

import httpx

from scripts.eval.voice_latency import quick_ask_protocol as wire
from scripts.eval.voice_latency import report
from scripts.eval.voice_latency.protocol import (
    BETWEEN_ASKS_SECONDS,
    CONNECT_TIMEOUT_SECONDS,
    TURN_TIMEOUT_SECONDS,
    Ask,
    AskSetError,
)
from scripts.eval.voice_latency.quick_ask_protocol import NO_SPEECH, QuickAskWatch, SpeechTimes
from scripts.eval.voice_latency.run import (
    BACKEND_CONTAINER_PORT,
    BACKEND_HOST,
    EXIT_REFUSED,
    UNKNOWN_SHA,
    RunRefused,
    emit,
    read_asks,
)

logger = logging.getLogger(__name__)

CLI_DESCRIPTION = "Time a quick-ask (a spoken chat turn and its first sentence's speech) over HTTP."
HTTP_OK = 200
POST = "POST"
WORKSPACE_HEADER = "X-Workspace-ID"
AUTHORIZATION_HEADER = "Authorization"
BEARER = "Bearer"
SCHEME_MARK = "://"
PLAIN_HTTP = "http://"
# A token crosses plain http only to this machine (the backend's own container).
THIS_MACHINE = frozenset({"localhost", "127.0.0.1", "::1"})
TOKEN_NEEDS_HTTPS = "a token is sent only over https, or to this machine: give --host an https:// address"
ASSISTANT_ROLE = "assistant"


@dataclass(frozen=True)
class Target:
    """The server a run talks to, and who it talks as."""

    base_url: str
    workspace_id: str
    token_file: Optional[str] = None

    @property
    def hosted(self) -> bool:
        """A run with a token file has no database beside it."""
        return bool(self.token_file)

    def headers(self) -> Dict[str, str]:
        """One request's headers. The token file is read again every time."""
        workspace = {WORKSPACE_HEADER: self.workspace_id}
        if not self.token_file:
            return workspace
        return {**workspace, AUTHORIZATION_HEADER: f"{BEARER} {read_token(self.token_file)}"}


@dataclass(frozen=True)
class Options:
    """How the asks are sent and timed."""

    agent_id: Optional[int] = None
    new_chat_each: bool = False
    no_speech: bool = False
    unit_ends: bool = False


@dataclass(frozen=True)
class Turn:
    """One ask as driven: its kind, its reference, the reply's watch and the speech times."""

    kind: str
    ref: str
    watch: QuickAskWatch
    speech: SpeechTimes


def now() -> float:
    """The runner's clock for intervals."""
    return time.monotonic()


def wall_ms() -> int:
    """Wall time in whole milliseconds: only for the turn reference."""
    return int(time.time() * wire.MS_PER_SECOND)


def pause() -> None:
    """A breath between one reply and the next ask."""
    time.sleep(BETWEEN_ASKS_SECONDS)


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def read_token(path: str) -> str:
    """The bearer in the token file, as it is now. Never logged, printed or kept."""
    try:
        token = Path(path).read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RunRefused(f"the token file cannot be read ({type(exc).__name__})") from exc
    if not token:
        raise RunRefused("the token file is empty")
    return token


def base_url(host: str, port: int) -> str:
    """``http://<host>:<port>``, or ``host`` itself when it already names a scheme (a hosted server)."""
    return host.rstrip("/") if SCHEME_MARK in host else f"http://{host}:{port}"


def sends_token_in_the_clear(target: Target) -> bool:
    """Whether the run's token would cross a network unencrypted: plain http to another machine."""
    if not target.token_file or not target.base_url.startswith(PLAIN_HTTP):
        return False
    return (urlsplit(target.base_url).hostname or "") not in THIS_MACHINE


# ── the requests ────────────────────────────────────────────────────────────


def read_capabilities(client: Any, target: Target) -> Sequence[str]:
    """The capability names the server's discovery document lists (a public route: no token)."""
    try:
        response = client.get(target.base_url + wire.DISCOVERY_PATH)
        if response.status_code != HTTP_OK:
            raise RunRefused(f"the server's discovery answered {response.status_code}")
        return wire.capabilities_of(response.json())
    except (httpx.HTTPError, ValueError) as exc:
        raise RunRefused(f"the server's discovery could not be read ({type(exc).__name__})") from exc


def stream_reply(client: Any, target: Target, body: Mapping[str, Any], at_text_end: bool) -> QuickAskWatch:
    """Send one spoken chat turn and time its stream. A refused or broken turn keeps what
    was timed before it broke; the rest stays empty."""
    headers = target.headers()
    watch = QuickAskWatch(started=now(), at_text_end=at_text_end)
    try:
        with client.stream(POST, target.base_url + wire.CHAT_PATH, json=dict(body), headers=headers) as response:
            watch.headers(now())
            if response.status_code != HTTP_OK:
                logger.warning("quick_ask_chat_refused status=%s", response.status_code)
                return watch
            for line in response.iter_lines():
                watch.observe(line, now())
        watch.end(now())
    except httpx.HTTPError as exc:
        logger.warning("quick_ask_chat_failed err=%s", type(exc).__name__)
    return watch


def speak(client: Any, target: Target, sentence: str, ref: str) -> SpeechTimes:
    """Ask for the first sentence's audio and time its first body byte and its last.
    With no token to send it is not asked for: the reply's own times are kept."""
    try:
        headers = target.headers()
    except RunRefused as exc:
        logger.warning("quick_ask_speech_not_sent: %s", exc)
        return NO_SPEECH
    started, first_byte_at = now(), None
    try:
        with client.stream(
            POST, target.base_url + wire.SPEECH_PATH, json=wire.speech_body(sentence, ref), headers=headers
        ) as response:
            if response.status_code != HTTP_OK:
                logger.warning("quick_ask_speech_refused status=%s", response.status_code)
                return NO_SPEECH
            for chunk in response.iter_bytes():
                if chunk and first_byte_at is None:
                    first_byte_at = now()
        ended = now()
    except httpx.HTTPError as exc:
        logger.warning("quick_ask_speech_failed err=%s", type(exc).__name__)
        return NO_SPEECH
    if first_byte_at is None:
        return NO_SPEECH
    return SpeechTimes(
        first_byte_ms=wire.elapsed_ms(started, first_byte_at), total_ms=wire.elapsed_ms(started, ended)
    )


def drive(client: Any, target: Target, asks: Sequence[Ask], options: Options) -> List[Turn]:
    """Every ask in order, one chat for the run (as the voice screen keeps one) unless
    ``new_chat_each``. The next ask goes out only after the last reply's stream has ended.

    When the token file cannot be read the run stops at that ask and keeps the turns
    already timed; with none timed yet there is nothing to report and the run is refused.
    """
    chat_id: Optional[str] = None
    turns: List[Turn] = []
    for number, ask in enumerate(asks, start=1):
        ref = wire.turn_ref(wall_ms(), number)
        body = wire.chat_body(
            ask.text, ref, chat_id=None if options.new_chat_each else chat_id, agent_id=options.agent_id
        )
        try:
            watch = stream_reply(client, target, body, options.unit_ends)
        except RunRefused as exc:
            if not turns:
                raise
            logger.warning("quick_ask_stopped_early at_ask=%s of=%s: %s", number, len(asks), exc)
            break
        wants_speech = not options.no_speech and watch.sentence is not None
        speech = speak(client, target, watch.sentence, ref) if wants_speech else NO_SPEECH
        turns.append(Turn(kind=ask.kind, ref=ref, watch=watch, speech=speech))
        chat_id = chat_id or watch.chat_id
        pause()
    return turns


# ── the database (the backend's own session; never in hosted mode) ──────────


def _uuids(values: Sequence[str]) -> List[uuid.UUID]:
    found = []
    for value in values:
        try:
            found.append(uuid.UUID(str(value)))
        except ValueError:
            logger.warning("quick_ask_chat_id_not_a_uuid")
    return found


def read_timings(workspace_id: str, chat_ids: Sequence[str]) -> Dict[str, Mapping[str, Any]]:
    """The run's replies' traces, keyed by the turn reference in ``context_trace["timing"]``.

    Read by workspace and chat first (a reference is unique within one run, not across
    callers), then matched by reference in Python.
    """
    from core.database.database import get_db_session
    from core.models.core import Message

    chats = _uuids(chat_ids)
    if not chats:
        return {}
    with get_db_session() as db:
        rows = (
            db.query(Message.context_trace)
            .filter(
                Message.workspace_id == uuid.UUID(workspace_id),
                Message.chat_id.in_(chats),
                Message.role == ASSISTANT_ROLE,
            )
            .all()
        )
        traces = [row[0] for row in rows]
    return wire.timings_by_ref(traces)


def server_timings(target: Target, chat_ids: Sequence[str]) -> Optional[Dict[str, Mapping[str, Any]]]:
    """The server's split for the run, or ``None`` when it cannot be read: a hosted run,
    or a database out of reach. The read comes after the asks, so whatever it raises
    (an import, a connection, a query) must not cost the run its own measures."""
    if target.hosted:
        return None
    try:
        return read_timings(target.workspace_id, chat_ids)
    except Exception as exc:  # noqa: BLE001 - any failure here means "unavailable", never a lost run
        logger.warning("quick_ask_server_marks_unreadable err=%s", type(exc).__name__)
        return None


# ── the report ──────────────────────────────────────────────────────────────


def _options(args: argparse.Namespace, capabilities: Sequence[str]) -> Options:
    return Options(
        agent_id=args.agent,
        new_chat_each=bool(args.new_chat_each),
        no_speech=bool(args.no_speech),
        unit_ends=wire.unit_ends(capabilities, args.unit_ends),
    )


def _report(args: argparse.Namespace, target: Target, options: Options, turns: Sequence[Turn]) -> Dict[str, Any]:
    chat_ids = sorted({turn.watch.chat_id for turn in turns if turn.watch.chat_id})
    found = server_timings(target, chat_ids)
    read = found is not None
    timings = found or {}
    results = [wire.quick_ask_result(t.kind, t.watch, t.speech, timings.get(t.ref)) for t in turns]
    measures = wire.measures_for(read)
    summary = report.summarize(results, measures)
    line = report.results_line(now_iso(), args.sha, summary, measures)
    return {
        "chat_ids": chat_ids,
        "agent_id": options.agent_id,
        "unit_ends": options.unit_ends,
        "speech": not options.no_speech,
        "turns": [{"ref": turn.ref, **asdict(result)} for turn, result in zip(turns, results)],
        "server_marks": wire.SERVER_MARKS_READ if read else wire.SERVER_MARKS_UNAVAILABLE,
        "server_traces": sum(1 for turn in turns if turn.ref in timings),
        "summary": summary,
        "results_line": report.FIELD_SEP.join((line, wire.server_marks_note(read))),
    }


def quick_ask(args: argparse.Namespace, asks: Sequence[Ask], client: Any) -> Dict[str, Any]:
    """The run: check the server can be timed, drive the asks, read back, summarise."""
    target = Target(
        base_url=base_url(args.host, args.port), workspace_id=args.workspace, token_file=args.token_file
    )
    if sends_token_in_the_clear(target):
        raise RunRefused(TOKEN_NEEDS_HTTPS)
    capabilities = read_capabilities(client, target)
    refused = wire.refusal(capabilities, no_speech=bool(args.no_speech))
    if refused:
        raise RunRefused(refused)
    options = _options(args, capabilities)
    turns = drive(client, target, asks, options)
    return {**_report(args, target, options, turns), "asks_not_sent": len(asks) - len(turns)}


# ── the CLI ─────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.eval.voice_latency.quick_ask", description=CLI_DESCRIPTION
    )
    parser.add_argument("--workspace", required=True, help="the workspace the asks run in (its id)")
    parser.add_argument("--agent", type=int, help="send agentId (a chosen agent's route); left out, Auto routes")
    parser.add_argument("--host", default=BACKEND_HOST, help="the backend's host, or a whole https:// base URL")
    parser.add_argument("--port", type=int, default=BACKEND_CONTAINER_PORT, help="the backend's port")
    parser.add_argument("--sha", default=UNKNOWN_SHA, help="the build SHA the results line names")
    parser.add_argument("--token-file", help="a file holding a bearer token (hosted server); read per request")
    parser.add_argument("--new-chat-each", action="store_true", help="a new chat for every ask")
    parser.add_argument("--no-speech", action="store_true", help="time the chat alone; no speech request")
    parser.add_argument(
        "--unit-ends", choices=wire.UNIT_ENDS_CHOICES,
        help="force the sentence rule; left out, it follows the server's spoken_unit_ends capability",
    )
    return parser


def _validate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
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
    _validate(parser, args)
    timeout = httpx.Timeout(TURN_TIMEOUT_SECONDS, connect=CONNECT_TIMEOUT_SECONDS)
    try:
        asks = read_asks(sys.stdin)
        with httpx.Client(timeout=timeout) as client:
            outcome = quick_ask(args, asks, client)
    except (RunRefused, AskSetError) as exc:
        logger.error("voice_latency_refused: %s", exc)
        return EXIT_REFUSED
    emit(json.dumps(outcome, indent=2, default=str))
    emit(outcome["results_line"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
