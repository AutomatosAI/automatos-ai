"""PRD-163 S4 — the mission approval card is emitted to chat (backend half)."""
from __future__ import annotations
import os, sys, types
for _k in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"):
    os.environ.setdefault(_k, "test")
os.environ.setdefault("POSTGRES_HOST", "localhost"); os.environ.setdefault("POSTGRES_PORT", "5432")
sys.modules.setdefault("camelot", types.ModuleType("camelot"))

from modules.tools.formatting.result_formatter import ToolResultFormatter  # noqa: E402


def test_awaiting_approval_emits_card():
    result = {
        "success": True, "mission_id": "abc-123", "state": "awaiting_approval",
        "awaiting_approval": True, "goal": "research X", "task_count": 2,
        "tasks": [{"title": "a", "agent_role": "researcher", "sequence": 1},
                  {"title": "b", "agent_role": "writer", "sequence": 2}],
    }
    fd = ToolResultFormatter.format_for_frontend(result, "platform_create_mission")
    assert "mission_approval" in fd
    card = fd["mission_approval"]
    assert card["mission_id"] == "abc-123" and card["task_count"] == 2
    assert len(card["tasks"]) == 2


def test_running_mission_emits_no_card():
    result = {"success": True, "mission_id": "abc", "state": "running", "awaiting_approval": False}
    fd = ToolResultFormatter.format_for_frontend(result, "platform_create_mission")
    assert "mission_approval" not in fd
