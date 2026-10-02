"""Design §6.9 — the soul of a CLI with no system-prompt flag rides the first prompt.

Claude Code takes the agent's system prompt on the command line
(``--append-system-prompt-file``). Codex and GitHub Copilot have no such flag, so
it rides ``UserPromptSubmit`` → ``additionalContext``, ahead of the ticket. Before
PRD-253 only the ticket rode it, and a Codex session never saw its persona, its
rules or how to reach the Automatos tools.
"""
from __future__ import annotations

from automatos_cli_host.session import Session

SOUL = "SOUL-MARKER: you keep every CSV tidy."
TICKET = {"task_id": 7, "attempt": 1, "session_id": "sid", "agent_name": "Coder", "title": "Tidy the CSV",
          "prompt": "OBJECTIVE: tidy data.csv", "system_prompt": SOUL}


def _first_prompt_context(tmp_path, provider: str) -> str:
    cfg = type("Cfg", (), {"ask_timeout": 1.0, "sessions_dir": tmp_path, "socket_path": tmp_path / "s.sock"})()
    s = Session({**TICKET, "provider": provider}, cfg, [str(tmp_path)], tmp_path / "s.sock",
                default_root=str(tmp_path))
    out = s.handle_hook({"hook_event_name": "UserPromptSubmit", "prompt": "go"})
    assert s.handle_hook({"hook_event_name": "UserPromptSubmit", "prompt": "again"}) == {}   # once per session
    return out["hookSpecificOutput"]["additionalContext"]


def test_a_cli_with_no_system_prompt_flag_gets_its_soul_ahead_of_the_ticket(tmp_path):
    for provider in ("codex", "copilot"):
        said = _first_prompt_context(tmp_path, provider)
        assert SOUL in said and "You are Coder" in said, provider
        assert said.index(SOUL) < said.index("# Ticket #7 — Tidy the CSV"), provider


def test_claude_code_gets_only_the_ticket_there(tmp_path):
    said = _first_prompt_context(tmp_path, "claude")
    assert said.startswith("# Ticket #7 — Tidy the CSV") and SOUL not in said
