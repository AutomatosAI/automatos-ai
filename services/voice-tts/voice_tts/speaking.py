"""One ``POST /speak``: the sentences spoken in order, encoded as they come, streamed out.

The first sentence is spoken before anything goes out, so a text Kokoro cannot say is
still a clean 422, never a broken stream. Then LAME starts, and the 200 and its MP3 go
out once LAME writes its first frame; each later sentence is spoken while the earlier
ones play. A sentence Kokoro has nothing to say for (only symbols) is skipped.

Synthesis runs on the request's slot (slots.py), on a thread of the synthesis pool, so
the event loop keeps answering /healthz and refusing past the limit, and a request that
goes away keeps its slot until its sentence is done. No further sentence is spoken and
LAME is killed.

Any other failure (Kokoro, espeak-ng, LAME) is a 500 ``tts_failed`` while nothing has
gone out, and ends the stream cleanly once audio has. Its message and traceback can
carry the text, so a failure is logged by its exception type and the voice alone.

One log line per answer: voice, characters, sentences, the seconds to the first byte
and in all, the audio's bytes. Never the text.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Sequence, Tuple

from aiohttp import web

from .config import Settings
from .engine import silence
from .mp3 import EncoderError, Mp3Encoder
from .request import SpeechRequest, sentences
from .slots import Slot

logger = logging.getLogger(__name__)

CONTENT_TYPE = "audio/mpeg"
VOICE_HEADER = "X-Voice-Id"
FAILED = "tts_failed"
FAILED_DETAIL = "the voice could not speak this text"
StartEncoder = Callable[[Settings], Awaitable[Any]]


class SpeakError(RuntimeError):
    """The answer failed before any audio went out: ``status`` and ``code`` are the refusal's."""

    def __init__(self, status: int, code: str, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail


def failure(exc: BaseException, voice: str, where: str) -> SpeakError:
    """A 500 ``tts_failed``, logged by the exception's type and the voice only: never its message."""
    logger.error("speak failed %s: %s voice=%s", where, type(exc).__name__, voice)
    return SpeakError(500, FAILED, FAILED_DETAIL)


@dataclass
class _Answer:
    started: float
    first_byte: Optional[float] = None
    bytes_out: int = 0


@dataclass(frozen=True)
class _Job:
    """One request's work: what to say, on which slot."""

    spec: SpeechRequest
    slot: Slot


class Speaker:
    """Speaks a request with the engine and streams it through the MP3 encoder."""

    def __init__(self, settings: Settings, engine: Any, *, encoder: StartEncoder = Mp3Encoder.start) -> None:
        self._settings = settings
        self._engine = engine
        self._start_encoder = encoder

    async def _say(self, text: str, job: _Job) -> Optional[bytes]:
        """The sentence as PCM, or None when Kokoro has nothing to say for it (its ValueError)."""
        voice = job.spec.voice
        try:
            return await job.slot.run(self._engine.speak, text, voice.id, job.spec.speed, voice.espeak)
        except ValueError:
            return None

    async def _first(self, pieces: Sequence[str], job: _Job) -> Tuple[bytes, Sequence[str]]:
        """The first sentence that has something to say, and the sentences after it."""
        for index, piece in enumerate(pieces):
            pcm = await self._say(piece, job)
            if pcm:
                return pcm, pieces[index + 1:]
        raise SpeakError(422, "tts_bad_request", "the text has nothing the voice can say")

    async def _start(self, pieces: Sequence[str], job: _Job) -> Tuple[bytes, Sequence[str], Any]:
        """The first sentence spoken and the encoder started. Raises SpeakError: nothing has gone out."""
        try:
            first, rest = await self._first(pieces, job)
            encoder = await self._start_encoder(self._settings)
        except SpeakError:
            raise
        except Exception as exc:
            raise failure(exc, job.spec.voice.id, "before the first byte") from exc
        return first, rest, encoder

    async def stream(self, request: web.Request, spec: SpeechRequest, slot: Slot) -> web.StreamResponse:
        job = _Job(spec=spec, slot=slot)
        answer = _Answer(started=time.monotonic())
        pieces = sentences(spec.text)
        first, rest, encoder = await self._start(pieces, job)
        response = web.StreamResponse(headers={"Cache-Control": "no-store", VOICE_HEADER: spec.voice.id})
        response.content_type = CONTENT_TYPE
        outcome = await self._pipe(request, encoder, response, answer, first, rest, job)
        self._log(spec, len(pieces), answer, outcome)
        if not response.prepared:
            # Nothing went out (LAME failed before its first frame): a clean refusal still can.
            raise SpeakError(500, FAILED, FAILED_DETAIL)
        with contextlib.suppress(ConnectionResetError):
            await response.write_eof()
        return response

    async def _pipe(
        self, request: web.Request, encoder: Any, response: web.StreamResponse, answer: _Answer, first: bytes,
        rest: Sequence[str], job: _Job,
    ) -> str:
        """Speak into the encoder and relay its MP3 at once; the first failure stops both. The outcome."""
        feeder = asyncio.create_task(self._feed(encoder, first, rest, job))
        relay = asyncio.create_task(self._relay(request, encoder, response, answer))
        try:
            await asyncio.gather(feeder, relay)
            return "ok"
        except ConnectionResetError:
            return "client_gone"
        except EncoderError as exc:
            # Our own message: LAME's exit code and stderr, which sees PCM and never the text.
            logger.error("speak: the encoder stopped after %d bytes: %s", answer.bytes_out, exc)
            return "encoder_failed"
        except Exception as exc:  # noqa: BLE001 — its message could carry the text: logged by type, the stream ends
            logger.error("speak: the stream stopped after %d bytes: %s voice=%s", answer.bytes_out,
                         type(exc).__name__, job.spec.voice.id)
            return "failed"
        finally:
            for task in (feeder, relay):
                task.cancel()
            await asyncio.gather(feeder, relay, return_exceptions=True)
            await encoder.close()

    async def _feed(self, encoder: Any, first: bytes, rest: Sequence[str], job: _Job) -> None:
        await encoder.feed(first)
        pause = silence(self._settings.sentence_pause_seconds)
        for piece in rest:
            pcm = await self._say(piece, job)
            if pcm:
                await encoder.feed(pause)
                await encoder.feed(pcm)
        await encoder.end_input()

    @staticmethod
    async def _relay(request: web.Request, encoder: Any, response: web.StreamResponse, answer: _Answer) -> None:
        """LAME's MP3 to the client: the 200 goes out with its first frame."""
        async for chunk in encoder.frames():
            if not response.prepared:
                await response.prepare(request)
                answer.first_byte = time.monotonic()
            answer.bytes_out += len(chunk)
            await response.write(chunk)

    @staticmethod
    def _log(spec: SpeechRequest, sentence_count: int, answer: _Answer, outcome: str) -> None:
        now = time.monotonic()
        first = (answer.first_byte or now) - answer.started
        logger.info(
            "speak %s: voice=%s chars=%d sentences=%d speed=%.2f first_byte_s=%.3f total_s=%.3f bytes=%d",
            outcome, spec.voice.id, len(spec.text), sentence_count, spec.speed, first, now - answer.started,
            answer.bytes_out,
        )
