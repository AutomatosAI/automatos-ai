"""PRD-256 P256-FIX-RVW-31: the params a ``composio_execute`` call sends, computed once.

``exec_composio.execute_composio_execute`` sends the action's ``params`` (or the
``parameters`` some models emit), with every stray top-level key folded in, the explicit
params winning. The owner's-click card is built from the same params
(``UnifiedToolExecutor._resolve_effective_call`` → ``owner_only.asks_before_a_send``), so
a ``bcc`` or a recipient passed beside ``params`` is on the card the click approves.
"""
from __future__ import annotations

from typing import Any, Dict

# The meta-tool's own keys: every other top-level key is one of the action's params.
COMPOSIO_EXECUTE_KEYS = frozenset({"action", "action_name", "params", "parameters", "app_name", "app"})


def stray_params(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """The action's params a model put at the top level of the call, beside ``params``."""
    return {key: value for key, value in parameters.items() if key not in COMPOSIO_EXECUTE_KEYS}


def sent_params(parameters: Any) -> Dict[str, Any]:
    """The params the action is sent with: ``params`` (or ``parameters``) when it is an
    object, plus the stray top-level keys; an explicit param wins over a stray one."""
    if not isinstance(parameters, dict):
        return {}
    explicit = next((parameters[key] for key in ("params", "parameters") if isinstance(parameters.get(key), dict)), {})
    return {**stray_params(parameters), **explicit}
