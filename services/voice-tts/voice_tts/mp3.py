"""PCM in, MP3 out, while the rest is still being spoken: LAME's command line through pipes.

MP3 because iOS (AVPlayer, expo-audio) and every browser play it natively. One
encoder per request takes the 16-bit mono PCM of each sentence as Kokoro finishes
it, and its MP3 frames go out as LAME writes them (``--flush``), so the first
sentence is playing before the last is spoken. One encoder for the whole answer
keeps it one continuous stream, with no gap or header between sentences.

``-t`` writes no LAME/Xing tag: its frame count can only be filled in by seeking back
to the start of a file, and a stream has gone out by then. Without it a player reads
the constant bitrate (``-b``) from every frame. The MP3 is MPEG-2 Layer III, mono, at
Kokoro's 24 kHz (``--resample 24`` keeps LAME from choosing another rate).

LAME is LGPL and runs as its own process; nothing links it.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import AsyncIterator, List, Optional

from .config import Settings

READ_CHUNK_BYTES = 4096
SAMPLE_RATE_KHZ = "24"


class EncoderError(RuntimeError):
    """LAME failed: it could not start, or it exited with an error."""


def lame_command(settings: Settings) -> List[str]:
    """LAME's arguments: raw 16-bit little-endian mono PCM at 24 kHz on stdin, MP3 on stdout."""
    return [
        settings.lame_bin,
        "--quiet",
        "-r", "-s", SAMPLE_RATE_KHZ, "--bitwidth", "16", "--signed", "--little-endian",
        "-m", "m",
        "-b", str(settings.mp3_kbps), "--cbr",
        "--resample", SAMPLE_RATE_KHZ,
        "-t", "--noreplaygain", "--flush",
        "-", "-",
    ]


class Mp3Encoder:
    """One LAME process: ``feed`` PCM, iterate ``frames`` for the MP3, ``end_input`` when done.

    ``feed`` and ``frames`` must run at once (two tasks): LAME stops reading its input while
    nobody reads its output, so a feed with no reader blocks once the pipes are full.
    """

    def __init__(self, process: asyncio.subprocess.Process, timeout_seconds: int) -> None:
        self._process = process
        self._timeout = timeout_seconds

    @classmethod
    async def start(cls, settings: Settings) -> "Mp3Encoder":
        try:
            process = await asyncio.create_subprocess_exec(
                *lame_command(settings),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise EncoderError(f"LAME ({settings.lame_bin!r}) could not start: {exc}") from exc
        return cls(process, settings.encode_timeout_seconds)

    async def feed(self, pcm: bytes) -> None:
        """Hand LAME more PCM, waiting while its pipe is full. Read ``frames`` meanwhile, or this blocks."""
        stdin = self._process.stdin
        if stdin is None or not pcm:
            return
        stdin.write(pcm)
        await asyncio.wait_for(stdin.drain(), timeout=self._timeout)

    async def end_input(self) -> None:
        """No more PCM: LAME encodes what it holds, writes its last frames and exits."""
        stdin = self._process.stdin
        if stdin is not None and not stdin.is_closing():
            stdin.close()
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                await stdin.wait_closed()

    async def frames(self) -> AsyncIterator[bytes]:
        """The MP3 as LAME writes it, until it exits. Raises EncoderError when it exits with an error."""
        stdout = self._process.stdout
        if stdout is None:
            return
        while True:
            chunk = await asyncio.wait_for(stdout.read(READ_CHUNK_BYTES), timeout=self._timeout)
            if not chunk:
                break
            yield chunk
        code = await asyncio.wait_for(self._process.wait(), timeout=self._timeout)
        if code != 0:
            raise EncoderError(f"LAME exited with {code}: {await self._stderr()}")

    async def _stderr(self) -> str:
        stderr = self._process.stderr
        if stderr is None:
            return ""
        tail = await stderr.read()
        return tail.decode("utf-8", "replace")[-500:]

    async def close(self) -> Optional[int]:
        """Stop LAME if it is still running (a client that went away); its exit code."""
        if self._process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                self._process.kill()
        return await self._process.wait()
