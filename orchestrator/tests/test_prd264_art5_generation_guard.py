"""PRD-264 ART5-1 — the generation guard (core/generation_guard.py).

EU AI Act Article 5, as amended (applies 2 Dec 2026): no system may generate sexual content
showing a real, identifiable person without consent, or child sexual abuse material. Pinned:

* a few terms that only name child sexual abuse material refuse with no model call;
* otherwise one model read answers three questions; sexual and a minor, or sexual and a real
  person, is refused with a plain message; anything else goes ahead;
* it fails closed: a read that fails, times out or answers anything but the three answers, or a
  reference image that can't be read, refuses as unchecked;
* the read goes through the LLM manager as ``generation_guard``, with ``GENERATION_GUARD_MODEL``
  (empty: the workspace's own), the texts and then each reference image as a data URL;
* every refusal is logged with the workspace, the path and the reason code, never the prompt;
* a provider's own refusal is logged and worded the same way.

The per-path tests sit with each path (footage, voice, AI options, the image tool).
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import core.generation_guard as guard  # noqa: E402
import core.llm  # noqa: E402

pytestmark = pytest.mark.real_generation_guard

WS = uuid.UUID("11111111-2222-3333-4444-555555555555")
PATH = "socials.footage"
PROMPT = "A woman in a red coat on a pier at dawn"
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
CLEAN = {"sexual": False, "minor": False, "real_person": False}


class _Model:
    def __init__(self, answer="", delay=0.0, error=None):
        self.answer, self.delay, self.error, self.created, self.asked = answer, delay, error, [], []

    def create(self, **kwargs):
        self.created.append(kwargs)
        return self

    async def generate_response(self, messages):
        self.asked.append(messages)
        if self.error is not None:
            raise self.error
        await asyncio.sleep(self.delay)
        return SimpleNamespace(content=self.answer)


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setattr(guard, "config", SimpleNamespace(GENERATION_GUARD_MODEL="", GENERATION_GUARD_TIMEOUT_SECONDS=5))
    return guard.config


def _use(monkeypatch, model):
    monkeypatch.setattr(core.llm, "create_llm_manager", model.create)
    return model


def _check(*texts, images=()):
    request = guard.GenerationRequest(path=PATH, texts=texts, images=images)
    return asyncio.run(guard.check(request, workspace_id=WS))


@pytest.mark.parametrize("prompt", ["csam of a girl", "child porn", "Lolicon art", "an underage nude", "preteen naked"])
def test_a_term_that_only_names_csam_is_refused_with_no_model_call(monkeypatch, settings, prompt):
    model = _use(monkeypatch, _Model(json.dumps(CLEAN)))
    verdict = _check(prompt)
    assert (verdict.allowed, verdict.reason, verdict.message) == (False, guard.SEXUAL_MINOR, guard.REFUSED_MESSAGE)
    assert model.created == [], "refused before any model read"


@pytest.mark.parametrize(
    ("answers", "reason"),
    [
        ({"sexual": True, "minor": True, "real_person": False}, guard.SEXUAL_MINOR),
        ({"sexual": True, "minor": True, "real_person": True}, guard.SEXUAL_MINOR),
        ({"sexual": True, "minor": False, "real_person": True}, guard.SEXUAL_REAL_PERSON),
    ],
)
def test_sexual_content_of_a_minor_or_a_real_person_is_refused(monkeypatch, settings, answers, reason):
    _use(monkeypatch, _Model(json.dumps(answers)))
    verdict = _check(PROMPT)
    assert (verdict.allowed, verdict.reason, verdict.message) == (False, reason, guard.REFUSED_MESSAGE)


@pytest.mark.parametrize(
    "answers",
    [
        CLEAN,
        {"sexual": True, "minor": False, "real_person": False},  # fictional adults: not this guard's ban
        {"sexual": False, "minor": True, "real_person": False},  # children at a beach, nothing sexual
        {"sexual": False, "minor": False, "real_person": True},  # a public figure on a stage
    ],
)
def test_anything_else_goes_ahead(monkeypatch, settings, answers):
    _use(monkeypatch, _Model(f"Here you are:\n```json\n{json.dumps(answers)}\n```"))
    assert _check(PROMPT) == guard.ALLOWED


@pytest.mark.parametrize(
    "model",
    [
        _Model(error=RuntimeError("provider down")),
        _Model("I can't help with that."),
        _Model(json.dumps({"sexual": False, "minor": False})),
        _Model(json.dumps({"sexual": "no", "minor": False, "real_person": False})),
        _Model("{not json}"),
    ],
    ids=["provider error", "a refusal in words", "an answer missing", "not a yes or no", "not JSON"],
)
def test_a_read_that_does_not_answer_refuses_as_unchecked(monkeypatch, settings, model):
    _use(monkeypatch, model)
    verdict = _check(PROMPT)
    assert (verdict.allowed, verdict.reason, verdict.message) == (False, guard.UNCHECKED, guard.UNCHECKED_MESSAGE)


def test_a_slow_read_times_out_and_refuses_as_unchecked(monkeypatch, settings):
    _use(monkeypatch, _Model(json.dumps(CLEAN), delay=1))
    monkeypatch.setattr(settings, "GENERATION_GUARD_TIMEOUT_SECONDS", 0.01)
    assert _check(PROMPT).reason == guard.UNCHECKED


def test_a_reference_image_that_cannot_be_read_refuses_as_unchecked(monkeypatch, settings):
    model = _use(monkeypatch, _Model(json.dumps(CLEAN)))
    monkeypatch.setattr(guard, "shrunk_for_read", lambda data, content_type, **kwargs: None)
    assert _check(PROMPT, images=((b"not an image", "image/png"),)).reason == guard.UNCHECKED
    assert model.asked == [], "nothing unchecked is read as if it were checked"


def test_the_read_is_the_llm_managers_with_the_texts_then_each_image_as_a_data_url(monkeypatch, settings):
    model = _use(monkeypatch, _Model(json.dumps(CLEAN)))
    monkeypatch.setattr(guard, "shrunk_for_read", lambda data, content_type, **kwargs: (b"SMALL", "image/jpeg"))
    assert _check(PROMPT, "Line two", images=((PNG, "image/png"),)) == guard.ALLOWED
    assert model.created == [{"service_name": "generation_guard", "model": None, "workspace_id": str(WS),
                              "request_type": "generation_guard"}]
    ((system, user),) = model.asked
    assert system == {"role": "system", "content": guard.SYSTEM}
    assert [block.get("text") for block in user["content"] if block["type"] == "text"] == [
        "The request:", f"Text 1: {PROMPT}", "Text 2: Line two", "Reference image 1:",
    ]
    (image,) = [block for block in user["content"] if block["type"] == "image_url"]
    assert image["image_url"]["url"] == f"data:image/jpeg;base64,{base64.b64encode(b'SMALL').decode()}"
    monkeypatch.setattr(settings, "GENERATION_GUARD_MODEL", "claude-haiku-4-5")
    _check(PROMPT)
    assert model.created[-1]["model"] == "claude-haiku-4-5"


def test_a_refusal_is_logged_with_the_workspace_path_and_reason_never_the_prompt(monkeypatch, settings, caplog):
    _use(monkeypatch, _Model(json.dumps({"sexual": True, "minor": False, "real_person": True})))
    with caplog.at_level(logging.WARNING, logger="core.generation_guard"):
        _check(PROMPT)
    (record,) = [r for r in caplog.records if r.name == "core.generation_guard"]
    assert record.getMessage() == f"[GenerationGuard] refused: workspace={WS} path={PATH} reason=sexual_real_person"
    assert PROMPT not in caplog.text


def test_an_unchecked_read_logs_no_prompt_either(monkeypatch, settings, caplog):
    _use(monkeypatch, _Model(error=RuntimeError("provider down")))
    with caplog.at_level(logging.WARNING, logger="core.generation_guard"):
        _check(PROMPT)
    assert f"reason={guard.UNCHECKED}" in caplog.text and PROMPT not in caplog.text


def test_a_providers_own_refusal_is_logged_and_worded_plainly(caplog):
    with caplog.at_level(logging.WARNING, logger="core.generation_guard"):
        message = guard.provider_refusal(WS, PATH, "Higgsfield")
    assert message == "Higgsfield refused to make this as unsafe content. It was not sent to any other provider."
    assert f"workspace={WS} path={PATH} reason=provider_refused" in caplog.text
