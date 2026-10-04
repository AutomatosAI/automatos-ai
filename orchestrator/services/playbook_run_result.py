"""What a Playbook run hands back: its last step's answer with what that step
saved, and any step's whole output from the step's log.

F321 (night 9b, build 15): runs #0085 (exec-76879400dd8d) and #0102
(exec-45d8ac862a79) of "Monday green stock" put "The stock report has been
generated and saved to the scratchpad … 2 coffees flagged …" on the card: step 2
wrote the 8-coffee report with scratchpad_write (after six refused
platform_submit_report calls), and the card took only the step's last message.
The scratchpad expires (RECIPE_SCRATCHPAD_TTL), so the report was nowhere the
owner or Auto could read: the run kept a 200-character preview per step, and
platform_get_playbook_execution read a key the stored steps do not have.

Now the card and the run's final output carry what the last step saved, and a
step's whole output (its answer and what it saved) is read from its log.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from modules.tools.builtin.scratchpad_tool import SCRATCHPAD_WRITE_NAME

S3_SCHEME = "s3://"


def _as_text(value: Any) -> str:
    """A saved value as text: JSON for an object or a list."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return "" if value is None else str(value)


def saved_values(tool_calls: Optional[Iterable[Mapping[str, Any]]]) -> List[Tuple[str, str]]:
    """(key, value) for each value a step saved with scratchpad_write, in order."""
    saved: List[Tuple[str, str]] = []
    for call in tool_calls or []:
        if not isinstance(call, Mapping) or call.get("action") != SCRATCHPAD_WRITE_NAME or call.get("success") is False:
            continue
        params = call.get("params")
        if not isinstance(params, Mapping):
            continue
        value = _as_text(params.get("value")).strip()
        if value:
            saved.append((str(params.get("key") or "value"), value))
    return saved


def with_saved_values(answer: str, order: Any, tool_calls: Optional[Iterable[Mapping[str, Any]]]) -> str:
    """``answer`` followed by each value the step saved that the answer does not
    already hold, each under a line naming it."""
    text = answer or ""
    for key, value in saved_values(tool_calls):
        if value in text:
            continue
        heading = f'Step {order} saved this as "{key}":'
        text = f"{text}\n\n{heading}\n{value}" if text.strip() else f"{heading}\n{value}"
    return text


def read_step_log(log_url: str) -> Dict[str, Any]:
    """A step's full log (its answer, tool calls and messages) from ``s3://bucket/key``.
    Raises ValueError for a URL that is not one, and whatever S3 raises."""
    from core.storage import get_s3_client

    if not str(log_url or "").startswith(S3_SCHEME) or "/" not in log_url[len(S3_SCHEME):]:
        raise ValueError(f"not a step log address: {log_url!r}")
    bucket, key = log_url[len(S3_SCHEME):].split("/", 1)
    body = get_s3_client().get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
    return json.loads(body)


__all__ = ["read_step_log", "saved_values", "with_saved_values"]
