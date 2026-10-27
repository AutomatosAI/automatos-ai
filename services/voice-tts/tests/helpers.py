"""Stand-ins for the model and the encoder, and a voices file made on the spot."""

from __future__ import annotations

import asyncio
import threading
import zipfile
from pathlib import Path
from typing import AsyncIterator, List, Optional, Sequence, Tuple

from voice_tts.voices import Catalogue, load_catalogue

TOKEN = "voice-tts-test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
# A voice of each of Kokoro's language letters, and two English ones.
VOICE_IDS = ("zf_xiaobei", "bm_george", "af_heart", "ef_dora", "am_adam", "jf_alpha", "bf_emma")


def voices_file(path: Path, ids: Sequence[str] = VOICE_IDS) -> Path:
    """A voices file in Kokoro's layout (an npz: one ``<voice>.npy`` per voice)."""
    with zipfile.ZipFile(path, "w") as archive:
        for voice_id in ids:
            archive.writestr(f"{voice_id}.npy", b"\x93NUMPY")
    return path


def catalogue(tmp_path: Path, default: str = "af_heart", ids: Sequence[str] = VOICE_IDS) -> Catalogue:
    return load_catalogue(str(voices_file(tmp_path / "voices.bin", ids)), default)


class FakeEngine:
    """Speaks 100 bytes of PCM per sentence, once loaded; ``hold`` blocks it until released."""

    def __init__(self, loaded: bool = True, refuse: Sequence[str] = ()) -> None:
        self.loaded = loaded
        self.refuse = tuple(refuse)
        self.spoken: List[Tuple[str, str, float, str]] = []
        self.hold: Optional[threading.Event] = None
        self.started = threading.Event()

    def load(self) -> float:
        self.loaded = True
        return 0.5

    def speak(self, text: str, voice: str, speed: float, language: str) -> bytes:
        self.started.set()
        if self.hold is not None:
            assert self.hold.wait(timeout=30), "the test never released the engine"
        if any(word in text for word in self.refuse):
            raise ValueError("nothing to synthesize")
        self.spoken.append((text, voice, speed, language))
        return b"\x01\x00" * 50


class FakeEncoder:
    """Echoes the PCM it is fed as its 'MP3', prefixed by a frame sync, until its input ends."""

    instances: List["FakeEncoder"] = []

    def __init__(self) -> None:
        self.fed: List[bytes] = []
        self.closed = False
        self._queue: "asyncio.Queue[Optional[bytes]]" = asyncio.Queue()

    @classmethod
    async def start(cls, settings) -> "FakeEncoder":
        encoder = cls()
        cls.instances.append(encoder)
        await encoder._queue.put(b"\xff\xf3")
        return encoder

    async def feed(self, pcm: bytes) -> None:
        self.fed.append(pcm)
        await self._queue.put(pcm)

    async def end_input(self) -> None:
        await self._queue.put(None)

    async def frames(self) -> AsyncIterator[bytes]:
        while True:
            chunk = await self._queue.get()
            if chunk is None:
                return
            yield chunk

    async def close(self) -> int:
        self.closed = True
        return 0
