"""PRD-143's super-admin predicate, moved out of platform_executor.py (PRD-268 W3 K3: that file
does not grow). The executor imports it as ``_caller_is_super_admin``, the name its gates and
``platform_list_tools``' listing (F122) read."""
from __future__ import annotations

from typing import Any, Dict, Optional

__all__ = ["caller_is_super_admin"]


def caller_is_super_admin(caller_context: Optional[Dict[str, Any]]) -> bool:
    """PRD-143's super-admin predicate: only a literal system_role ==
    'super_admin' passes. There is no fallback, and no caller_context refuses.
    The gate and platform_list_tools' listing (F122) both read this one. A
    public widget turn never passes, whatever its context names (F155)."""
    from core.security.surface import widget_turn

    if widget_turn():
        return False
    return (caller_context or {}).get("system_role") == "super_admin"
