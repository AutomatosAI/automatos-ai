"""PRD-239 S4 — a failed chat turn described in plain words, with a stable code.

Before this, the outer ``except`` of a turn streamed ``str(exc)`` verbatim — a
provider's JSON payload, a bare "Failed to activate agent 15" — and the client
logged it to the console. ``describe_turn_error`` turns the exceptions a turn
can raise into one sentence the person can act on plus a code the client can
branch on; the raw text stays in the server log.

Edition-neutral: nothing here reads config or the database.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

CODE_RATE_LIMITED = "rate_limited"
CODE_MODEL_UNAVAILABLE = "model_unavailable"
CODE_NO_API_KEY = "no_api_key"
CODE_RUNTIME_MISMATCH = "runtime_mismatch"
CODE_TRIAL_EXHAUSTED = "trial_exhausted"
CODE_ACTIVATION_FAILED = "activation_failed"
CODE_TURN_FAILED = "turn_failed"

RAW_MESSAGE_CHARS = 200


@dataclass(frozen=True)
class TurnError:
    code: str
    message: str


def _who(agent_name: Optional[str]) -> str:
    return agent_name.strip() if isinstance(agent_name, str) and agent_name.strip() else "The agent"


def _class_name(exc: BaseException) -> str:
    return exc.__class__.__name__


def _short(text: str) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= RAW_MESSAGE_CHARS else text[:RAW_MESSAGE_CHARS].rstrip() + "…"


def describe_turn_error(exc: BaseException, *, agent_name: Optional[str] = None) -> TurnError:
    """One plain sentence + a code for an exception a chat turn raised."""
    name = _class_name(exc)
    text = str(exc) or name
    who = _who(agent_name)

    if name == "TrialExhaustedError":
        # PRD-222 US-014: the client matches the typed code inside the text.
        return TurnError(CODE_TRIAL_EXHAUSTED, text)
    if name == "ProviderRateLimitError":
        return TurnError(CODE_RATE_LIMITED, f"{who} is rate-limited right now. {_short(text)}")
    if name == "ProviderModelUnavailableError":
        return TurnError(CODE_MODEL_UNAVAILABLE, f"{who}'s model is not available: {_short(text)}")
    if name == "RuntimeMismatchError" or getattr(exc, "error_code", None) == CODE_RUNTIME_MISMATCH:
        return TurnError(
            CODE_RUNTIME_MISMATCH,
            f"{who} runs as a Claude Code session, not an API model — file a ticket for it instead.",
        )
    lowered = text.lower()
    if "api key" in lowered and ("not configured" in lowered or "no api key" in lowered or "missing" in lowered):
        return TurnError(CODE_NO_API_KEY, f"{who} has no API key for its provider. {_short(text)}")
    if text.startswith("Failed to activate agent"):
        return TurnError(
            CODE_ACTIVATION_FAILED,
            f"{who} could not be started — check its model and provider key in the agent's Model tab.",
        )
    return TurnError(CODE_TURN_FAILED, f"{who} could not finish this reply: {_short(text)}")
