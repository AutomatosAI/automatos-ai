"""
Auth scope constants — single source of truth.

Centralises the permission-scope strings used to gate SDK API keys so the
key-minting allowlist (``api/api_keys.py``) and the board auth gate
(``core/auth/hybrid.py``) reference the *same* value rather than duplicated
string literals (honours the project's no-hardcoded-values rule).

Scope format is ``"<resource>:<action>"``, matching the existing convention in
``VALID_PERMISSIONS`` (e.g. ``"documents:read"``, ``"agents:execute"``).
"""

from __future__ import annotations

# Board task read access for SDK keys (PRD-09 Slice 2 — read-only board).
TASKS_READ = "tasks:read"

# NOTE: ``tasks:write`` / ``tasks:execute`` (board write-back and the
# agent-launching / approval actions) are introduced in Slice 3, where the
# write-vs-execute split is designed alongside the mutation endpoints. They are
# intentionally NOT defined yet — a mintable scope that gates nothing would be a
# misleading half-state.
