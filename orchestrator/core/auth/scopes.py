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

# PRD-274 Wave 1: a product's server key may file a ticket (POST /api/v1/tasks) and start an
# agent's run (POST /api/agents/{id}/execute). Each route declares its scope through
# core/auth/server_key_gate.py; a key without it is refused.
TASKS_WRITE = "tasks:write"
AGENTS_EXECUTE = "agents:execute"
