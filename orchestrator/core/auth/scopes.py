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
# A server key reads (GET /api/memory, /api/memory/search) and stores (POST /api/memory) the
# workspace's memory. It reads only facts no audience limits and none private to a person.
MEMORY_READ = "memory:read"
MEMORY_WRITE = "memory:write"
# A server key starts a Playbook (POST /api/workflow-recipes/{id}/execute) and writes a Deliverable
# (POST /api/documents/generate). A run it starts is an outside product's input, never the owner's
# words (services/playbook_given.TRIGGER_STARTS).
PLAYBOOKS_EXECUTE = "playbooks:execute"
DOCUMENTS_WRITE = "documents:write"
# A server key chats with a lean agent (POST /api/chat, api/chat_server_key.py), as its own
# service user. Not "chat": that scope is the widget's, and would grant this by accident.
AGENTS_CHAT = "agents:chat"
