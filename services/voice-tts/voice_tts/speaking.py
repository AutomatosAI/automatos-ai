"""One ``POST /speak``: the sentences spoken in order, encoded as they come, streamed out.

The first sentence is spoken before the 200 goes out, so a text Kokoro cannot say
is still a clean 422, never a broken stream. Then LAME starts, its MP3 is relayed to
the client as it is written, and each later sentence is spoken while the earlier
ones play. A sentence Kokoro has nothing to say for (only symbols) is skipped.

Synthesis runs in a worker thread, so the event loop keeps answering /healthz and
refusing past the concurrency limit. A client that goes away stops the work: no
further sentence is spoken and LAME is killed.

One log line per answer: voice, characters, sentences, the seconds to the first byte
and in all, the audio's bytes. Never the text.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Sequence

from aiohttp import web

from .config import Settings
from .engine import silence
from .mp3 import EncoderError, Mp3Encoder
from .request import SpeechRequest, sentences

logger = logging.getLogger(__name__)

CONTENT_TYPE = "audio/mpeg"
VOICE_HEADER = "X-Voice-Id"
StartEncoder = Callable[[Settings], Awaitable[Any]]


class SpeakError(RuntimeError):
    """The answer failed before any audio went out: ``status`` and ``code`` are the refusal's."""

    def __init__(self, status: int, code: str, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail


@dataclass
class _Answer:
    started: float
    first_byte: Optional[float] = None
    bytes_out: int = 0


class Speaker:
    """Speaks a request with the engine and streams it through the MP3 encoder."""

    def __init__(self, settings: Settings, engine: Any, *, encoder: StartEncoder = Mp3Encoder.start) -> None:
        self._settings = settings
        self._engine = engine
        self._start_encoder = encoder

    async def _say(self, text: str, spec: SpeechRequest) -> Optional[bytes]:
        """The sentence as PCM, or None when Kokoro has nothing to say for it."""
        try:
            return await asyncio.to_thread(self._engine.speak, text, spec.voice.id, spec.speed, spec.voice.espeak)
        except ValueError:
            return None

    async def _first(self, pieces: Sequence[str], spec: SpeechRequest) -> tuple:
        """The first sentence that has something to say, and the sentences after it."""
        for index, piece in enumerate(pieces):
            pcm = await self._say(piece, spec)
            if pcm:
                return pcm, pieces[index + 1:]
        raise SpeakError(422, "tts_bad_request", "the text has nothing the voice can say")

    async def stream(self, request: web.Request, spec: SpeechRequest) -> web.StreamResponse:
        answer = _Answer(started=time.monotonic())
        pieces = sentences(spec.text)
        first, rest = await self._first(pieces, spec)
        try:
            encoder = await self._start_encoder(self._settings)
        except EncoderError as exc:
            logger.error("speak: %s", exc)
            raise SpeakError(500, "tts_encoder_failed", "the audio could not be encoded") from exc
        response = web.StreamResponse(headers={"Cache-Control": "no-store", VOICE_HEADER: spec.voice.id})
        response.content_type = CONTENT_TYPE
        await response.prepare(request)
        outcome = await self._pipe(encoder, response, answer, first, rest, spec)
        # Logged before the last chunk ends the body, so the line is written when the client has it.
        self._log(spec, len(pieces), answer, outcome)
        if outcome == "ok":
            await response.write_eof()
        return response

    async def _pipe(
        self, encoder: Any, response: web.StreamResponse, answer: _Answer, first: bytes, rest: Sequence[str],
        spec: SpeechRequest,
    ) -> str:
        """Speak into the encoder and relay its MP3 at once; the first failure stops both. The outcome."""
        feeder = asyncio.create_task(self._feed(encoder, first, rest, spec))
        relay = asyncio.create_task(self._relay(encoder, response, answer))
        try:
            await asyncio.gather(feeder, relay)
            return "ok"
        except ConnectionResetError:
            return "client_gone"
        except (EncoderError, asyncio.TimeoutError, OSError) as exc:
            logger.error("speak: the stream stopped after %d bytes: %s", answer.bytes_out, exc)
            return "encoder_failed"
        finally:
            for task in (feeder, relay):
                task.cancel()
            await asyncio.gather(feeder, relay, return_exceptions=True)
            await encoder.close()

    async def _feed(self, encoder: Any, first: bytes, rest: Sequence[str], spec: SpeechRequest) -> None:
        await encoder.feed(first)
        pause = silence(self._settings.sentence_pause_seconds)
        for piece in rest:
            pcm = await self._say(piece, spec)
            if pcm:
                await encoder.feed(pause)
                await encoder.feed(pcm)
        await encoder.end_input()

    @staticmethod
    async def _relay(encoder: Any, response: web.StreamResponse, answer: _Answer) -> None:
        async for chunk in encoder.frames():
            if answer.first_byte is None:
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
