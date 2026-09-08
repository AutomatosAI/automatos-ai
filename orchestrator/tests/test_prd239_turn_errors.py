"""PRD-239 S4 — a failed chat turn is one plain sentence with a code, and the
error frames carry both. Pure units."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from consumers.chatbot.turn_errors import (  # noqa: E402
    CODE_ACTIVATION_FAILED,
    CODE_MODEL_UNAVAILABLE,
    CODE_NO_API_KEY,
    CODE_RATE_LIMITED,
    CODE_RUNTIME_MISMATCH,
    CODE_TRIAL_EXHAUSTED,
    CODE_TURN_FAILED,
    describe_turn_error,
)
from core.llm.clients.openai_compatible_client import (  # noqa: E402
    ProviderModelUnavailableError,
    ProviderRateLimitError,
    _is_model_unavailable,
)


class TrialExhaustedError(Exception):
    """Stands in for the PRD-230 typed error: matched by name."""


def test_model_unavailable_is_recognised_from_the_providers_wording():
    class _Exc(Exception):
        status_code = 400

    assert _is_model_unavailable(_Exc("Error code: 400 - {'error': {'message': 'deepseek/deepseek-coder is not a valid model ID', 'code': 400}}"))
    assert _is_model_unavailable(_Exc("model not found"))
    not_found = _Exc("nothing here")
    not_found.status_code = 404
    assert not _is_model_unavailable(not_found)  # no 'model' in the text → not a model refusal
    assert not _is_model_unavailable(_Exc("connection reset"))


def test_describe_maps_each_failure_to_a_sentence_and_a_code():
    cases = [
        (ProviderModelUnavailableError("OpenRouter does not offer the model 'x' (any more). Pick another model in the agent's Model tab."), CODE_MODEL_UNAVAILABLE, "Researcher's model is not available"),
        (ProviderRateLimitError("NVIDIA rate limit reached for model 'k'. Free lane."), CODE_RATE_LIMITED, "rate-limited"),
        (ValueError("OpenRouter API key not configured. Set OPENROUTER_API_KEY"), CODE_NO_API_KEY, "no API key"),
        (Exception("Failed to activate agent 15"), CODE_ACTIVATION_FAILED, "could not be started"),
        (RuntimeError("boom"), CODE_TURN_FAILED, "could not finish this reply: boom"),
    ]
    for exc, code, fragment in cases:
        err = describe_turn_error(exc, agent_name="Researcher")
        assert err.code == code, (exc, err)
        assert fragment in err.message, (exc, err)
        assert err.message.startswith("Researcher"), err


def test_trial_exhausted_keeps_the_literal_code_the_client_matches():
    err = describe_turn_error(TrialExhaustedError("trial_exhausted: the trial is spent"))
    assert err.code == CODE_TRIAL_EXHAUSTED and "trial_exhausted" in err.message


def test_runtime_mismatch_by_error_code_attribute():
    exc = RuntimeError("agent 15 runs as a cli session")
    exc.error_code = "runtime_mismatch"  # type: ignore[attr-defined]
    err = describe_turn_error(exc, agent_name="Bob")
    assert err.code == CODE_RUNTIME_MISMATCH and "Claude Code session" in err.message


def test_unknown_agent_reads_as_the_agent_and_long_raw_text_is_cut():
    err = describe_turn_error(RuntimeError("x" * 500))
    assert err.message.startswith("The agent could not finish this reply: ")
    assert len(err.message) < 260 and err.message.endswith("…")


def test_error_frames_carry_message_and_code():
    from consumers.chatbot.streaming import get_streaming_handler
    from services.chat_turns import error_frame

    frame = get_streaming_handler().format_aisdk_error("Researcher's model is not available", code="model_unavailable")
    assert frame.startswith("e:") and json.loads(frame[2:]) == {
        "message": "Researcher's model is not available", "code": "model_unavailable",
    }
    assert json.loads(get_streaming_handler().format_aisdk_error("plain")[2:]) == {"message": "plain"}
    assert json.loads(error_frame("turn died", code="turn_failed")[2:]) == {"message": "turn died", "code": "turn_failed"}
    assert json.loads(error_frame("bare")[2:]) == {"message": "bare"}
