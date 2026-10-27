"""The real voice: the image's Kokoro model speaks, in each of its languages, and POST /speak
answers with an MP3 a player can decode. Run inside the image only (the model is 310 MB).
"""

from __future__ import annotations

import asyncio
import io

import pytest
import soundfile
from aiohttp.test_utils import TestClient, TestServer

from helpers import AUTH
from voice_tts.engine import SAMPLE_RATE, Engine
from voice_tts.server import create_app
from voice_tts.voices import load_catalogue

# One voice per language the image holds, with a short line in that language.
LINES = {
    "af_heart": "Your board has three tasks waiting.",
    "bm_george": "The schedule starts at half past nine.",
    "ef_dora": "Tienes tres tareas pendientes.",
    "ff_siwis": "Vous avez trois tâches en attente.",
    "hf_alpha": "आपके पास तीन काम बाकी हैं।",
    "if_sara": "Hai tre attività in sospeso.",
    "jf_alpha": "こんにちは。",
    "pf_dora": "Você tem três tarefas pendentes.",
    "zf_xiaoxiao": "你有三个任务。",
}


@pytest.fixture(scope="module")
def engine():
    from voice_tts.config import load_settings

    loaded = Engine(load_settings({}, cpus=2))
    loaded.load()
    return loaded


@pytest.mark.parametrize("voice_id", sorted(LINES))
def test_each_language_the_image_holds_is_spoken(engine, settings, voice_id):
    voice = load_catalogue(settings.kokoro_voices_path, settings.default_voice).get(voice_id)

    pcm = engine.speak(LINES[voice_id], voice.id, 1.0, voice.espeak)

    seconds = len(pcm) / 2 / SAMPLE_RATE
    assert 0.3 < seconds < 15, f"{voice_id} spoke {seconds:.2f} s"


def test_speak_answers_with_an_mp3_of_the_whole_text(engine, settings):
    text = "Good morning. Your board has three tasks waiting, and two of them are due today."

    async def go():
        app = create_app(settings, engine=engine)
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/speak", json={"text": text, "voice": "bf_emma"}, headers=AUTH)
            assert response.status == 200, await response.text()
            assert response.headers["Content-Type"] == "audio/mpeg"
            assert response.headers["X-Voice-Id"] == "bf_emma"
            return await response.read()

    mp3 = asyncio.run(go())
    info = soundfile.info(io.BytesIO(mp3))
    assert (info.samplerate, info.channels) == (SAMPLE_RATE, 1)
    assert 3.0 < info.duration < 15.0
