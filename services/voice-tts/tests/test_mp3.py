"""The image's LAME: PCM in, a continuous MP3 stream out, mono at 24 kHz, while input still arrives."""

from __future__ import annotations

import asyncio
import io
from dataclasses import replace

import numpy as np
import pytest
import soundfile

from voice_tts.engine import SAMPLE_RATE, silence, to_pcm16
from voice_tts.mp3 import EncoderError, Mp3Encoder

# MPEG-2 Layer III, no CRC: the frame sync, then version 2 and layer III.
MPEG2_LAYER3_SYNC = b"\xff\xf3"


def _tone(seconds: float, hertz: float = 440.0) -> bytes:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return to_pcm16(0.3 * np.sin(2 * np.pi * hertz * t))


async def _encode(settings, *pieces: bytes) -> bytes:
    encoder = await Mp3Encoder.start(settings)
    out = bytearray()

    async def read():
        async for chunk in encoder.frames():
            out.extend(chunk)

    reader = asyncio.create_task(read())
    for piece in pieces:
        await encoder.feed(piece)
    await encoder.end_input()
    await reader
    await encoder.close()
    return bytes(out)


def test_pcm_becomes_a_mono_24khz_mp3_of_the_same_length(settings):
    mp3 = asyncio.run(_encode(settings, _tone(1.0), silence(0.25), _tone(1.0, 660.0)))

    assert mp3[:2] == MPEG2_LAYER3_SYNC, "no ID3 or LAME tag before the first frame"
    info = soundfile.info(io.BytesIO(mp3))
    assert (info.samplerate, info.channels) == (SAMPLE_RATE, 1)
    assert info.duration == pytest.approx(2.25, abs=0.15)
    # 64 kbit/s constant: 8000 bytes a second, give or take a frame.
    assert len(mp3) == pytest.approx(2.25 * 8000, rel=0.1)


async def _drain(frames) -> int:
    size = 0
    async for chunk in frames:
        size += len(chunk)
    return size


def test_frames_come_out_while_input_still_arrives(settings):
    async def go():
        encoder = await Mp3Encoder.start(settings)
        frames = encoder.frames()
        first = asyncio.ensure_future(frames.__anext__())  # read while feeding, as speaking.py does
        try:
            await encoder.feed(_tone(1.0))
            # The first frame arrives while LAME's input is still open, more PCM to come.
            assert (await asyncio.wait_for(first, timeout=10))[:2] == MPEG2_LAYER3_SYNC
            rest = asyncio.ensure_future(_drain(frames))
            await encoder.feed(_tone(1.0, 660.0))
            await encoder.end_input()
            assert await asyncio.wait_for(rest, timeout=10) > 0
        finally:
            first.cancel()
            await encoder.close()

    asyncio.run(go())


def test_a_missing_encoder_is_an_encoder_error(settings):
    with pytest.raises(EncoderError, match="could not start"):
        asyncio.run(Mp3Encoder.start(replace(settings, lame_bin="/nonexistent/lame")))


def test_close_stops_an_encoder_whose_client_went_away(settings):
    async def go():
        encoder = await Mp3Encoder.start(settings)
        await encoder.feed(_tone(0.5))
        await asyncio.wait_for(encoder.close(), timeout=10)
        assert encoder._process.returncode is not None

    asyncio.run(go())


def test_pcm16_clips_and_is_little_endian():
    pcm = to_pcm16(np.array([0.0, 1.0, -1.0, 2.0, -2.0], dtype=np.float32))

    assert np.frombuffer(pcm, dtype="<i2").tolist() == [0, 32767, -32767, 32767, -32767]
    assert silence(0.5) == bytes(24000)
