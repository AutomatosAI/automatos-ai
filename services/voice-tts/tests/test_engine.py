"""Engine.load with a stand-in model (no 310 MB load): the warm-up synthesis runs once, in the voice it
is given, before the engine reports loaded, and never keeps a model that loaded from being used.
test_speech.py warms the real model.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

import pytest

from voice_tts import engine as engine_module
from voice_tts.engine import SAMPLE_RATE, WARM_TEXT, Engine

WARM = ("af_heart", "en-us")
SECRET = "the-library-quoted-the-text-here"


class FakeModel:
    """Kokoro's ``create``: records each call; ``hold`` blocks it until released; ``fail`` raises."""

    def __init__(self, fail: Optional[BaseException] = None) -> None:
        self.calls: List[Tuple[str, Dict[str, Any]]] = []
        self.fail = fail
        self.hold: Optional[threading.Event] = None
        self.started = threading.Event()

    def create(self, text: str, **options: Any) -> Tuple[List[float], int]:
        self.calls.append((text, options))
        self.started.set()
        if self.hold is not None:
            assert self.hold.wait(timeout=30), "the test never released the model"
        if self.fail is not None:
            raise self.fail
        return [0.0, 0.5, -0.5], SAMPLE_RATE


@pytest.fixture
def model(monkeypatch):
    fake = FakeModel()
    monkeypatch.setattr(engine_module, "_load_model", lambda _settings: fake)
    return fake


def test_the_warm_text_is_one_plain_sentence_about_a_first_pieces_length():
    assert 80 <= len(WARM_TEXT) <= 120
    assert WARM_TEXT.endswith(".") and WARM_TEXT.count(".") == 1
    assert WARM_TEXT.isascii()


def test_load_warms_once_with_the_warm_text_in_the_given_voice(settings, model):
    engine = Engine(settings)

    seconds = engine.load(WARM)

    assert model.calls == [(WARM_TEXT, {"voice": "af_heart", "speed": 1.0, "lang": "en-us"})]
    assert engine.loaded
    assert seconds >= 0


def test_load_without_a_warm_voice_speaks_nothing(settings, model):
    engine = Engine(settings)

    engine.load()
    engine.load(None)

    assert model.calls == []
    assert engine.loaded


def test_a_loaded_engine_is_not_warmed_again(settings, model):
    engine = Engine(settings)
    engine.load(WARM)

    engine.load(WARM)

    assert len(model.calls) == 1


def test_a_warm_up_that_raises_still_leaves_the_engine_loaded(settings, model, caplog):
    model.fail = RuntimeError(SECRET)
    engine = Engine(settings)

    with caplog.at_level(logging.DEBUG):
        engine.load(WARM)

    assert engine.loaded
    assert len(model.calls) == 1
    assert any("warm-up synthesis failed: RuntimeError voice=af_heart" in record.getMessage()
               for record in caplog.records)
    assert SECRET not in caplog.text
    assert WARM_TEXT not in caplog.text


def test_the_engine_is_not_loaded_while_the_warm_up_runs(settings, model):
    model.hold = threading.Event()
    engine = Engine(settings)
    loading = threading.Thread(target=engine.load, args=(WARM,), daemon=True)

    loading.start()
    try:
        assert model.started.wait(timeout=5), "the warm-up never started"
        assert not engine.loaded
        with pytest.raises(RuntimeError, match="not loaded yet"):
            engine.speak("Hi there.", "af_heart", 1.0, "en-us")
    finally:
        model.hold.set()
        loading.join(timeout=5)

    assert not loading.is_alive()
    assert engine.loaded
    assert len(model.calls) == 1


def test_a_warmed_engine_speaks_through_the_same_model(settings, model):
    engine = Engine(settings)
    engine.load(WARM)

    pcm = engine.speak("Hi there.", "bm_george", 1.1, "en-gb")

    assert len(pcm) == 6
    assert model.calls[-1] == ("Hi there.", {"voice": "bm_george", "speed": 1.1, "lang": "en-gb"})
