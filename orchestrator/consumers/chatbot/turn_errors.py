"""PRD-239 S4 — a failed chat turn described in plain words, with a stable code.

Before this, the outer ``except`` of a turn streamed ``str(exc)`` verbatim — a
provider's JSON payload, a bare "Failed to activate agent 15" — and the client
logged it to the console. ``describe_turn_error`` turns the exceptions a turn
can raise into one sentence the person can act on plus a code the client can
branch on; the raw text stays in the server log.

Edition-neutral: nothing here reads config or the database.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

CODE_RATE_LIMITED = "rate_limited"
CODE_MODEL_UNAVAILABLE = "model_unavailable"
CODE_NO_API_KEY = "no_api_key"
CODE_RUNTIME_MISMATCH = "runtime_mismatch"
CODE_TRIAL_EXHAUSTED = "trial_exhausted"
CODE_ACTIVATION_FAILED = "activation_failed"
CODE_PROVIDER_FAILED = "provider_failed"
CODE_TURN_FAILED = "turn_failed"

RAW_MESSAGE_CHARS = 200

# F169 (night 5, B10/B26): an AI provider's HTTP error reached the reply as the
# SDK's text, "Error: Error code: 502 - {'error': {'message': 'Server tool
# \"openrouter:web_search\" failed: upstream returned an invalid response', …".
# It is said in plain words now; the raw text stays in the log.
PROVIDER_SDK_MODULES = ("openai", "anthropic", "httpx")
_STATUS_IN_TEXT = re.compile(r"^Error code: (\d{3})\b")
_SERVER_TOOL = re.compile(r'Server tool \\?"[\w.-]+:(?P<tool>[\w.-]+)\\?" failed')
ASK_AGAIN = "Nothing needs changing on your side; ask again in a minute."


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


def _provider_status(exc: BaseException) -> Optional[int]:
    """The HTTP status an AI provider answered with, when ``exc`` is its SDK's
    error (``openai.APIStatusError`` and kin); None for anything else."""
    if type(exc).__module__.split(".")[0] not in PROVIDER_SDK_MODULES:
        return None
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        return status
    match = _STATUS_IN_TEXT.match(str(exc) or "")
    return int(match.group(1)) if match else None


def _provider_error(status: int, text: str, who: str) -> TurnError:
    """What an AI provider's HTTP error means for the owner, in one sentence."""
    head = f"{who} could not finish this reply:"
    server_tool = _SERVER_TOOL.search(text)
    if server_tool:
        tool = server_tool.group("tool").replace("_", " ")
        return TurnError(CODE_PROVIDER_FAILED, f"{head} the AI provider's {tool} failed on its side. {ASK_AGAIN}")
    if status == 402:
        return TurnError(CODE_PROVIDER_FAILED, f"{head} the AI provider refused it because the account is out of "
                                               "credits. Top up the provider account, then ask again.")
    if status in (401, 403):
        return TurnError(CODE_PROVIDER_FAILED, f"{head} the AI provider rejected its API key. Check the provider's "
                                               "key in Settings, then ask again.")
    if status == 429:
        return TurnError(CODE_RATE_LIMITED, f"{who} is rate-limited by the AI provider right now. Ask again in a minute.")
    if status in (408, 504):
        return TurnError(CODE_PROVIDER_FAILED, f"{head} the AI provider took too long to answer. {ASK_AGAIN}")
    if status >= 500:
        return TurnError(CODE_PROVIDER_FAILED, f"{head} the AI provider had a problem on its side. {ASK_AGAIN}")
    return TurnError(CODE_PROVIDER_FAILED, f"{head} the AI provider refused the request. Ask again; if it keeps "
                                           "happening, the details are in the server log.")


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
    status = _provider_status(exc)
    if status is not None:
        return _provider_error(status, text, who)
    return TurnError(CODE_TURN_FAILED, f"{who} could not finish this reply: {_short(text)}")
