"""F288 (night 8): Auto tells the owner the run's card number, not its exec code.

"You can track its progress using execution ID `exec-990b7a5c4296`" (#0440) and
"The execution ID is `exec-ef4c657b537f`. I'll let you know its card number as soon
as it lands on your board": the tool's own message ended "Track with execution_id:
exec-…". It now ends with the card's number, as the board shows it.
"""
from __future__ import annotations

import asyncio
import re
from types import SimpleNamespace as NS
from uuid import UUID, uuid4


def test_a_started_runs_message_names_its_card_not_its_execution_id(db_session, seed_workspace, monkeypatch):
    import modules.tools.discovery.handlers_watches as watches
    import services.concurrency_guard as guard
    import services.playbook_engine as engine
    from core.models.core import WorkflowTemplate
    from modules.tools.discovery.handlers_playbooks import execute_playbook

    async def allowed(workspace_id, db):
        return NS(allowed=True, reason="")

    monkeypatch.setattr(guard, "check_concurrency", allowed)
    monkeypatch.setattr(engine, "get_playbook_engine", lambda: NS(launch=lambda **kw: None))
    monkeypatch.setattr(watches, "auto_create_watch", lambda *a, **k: None)
    ws = UUID(seed_workspace())
    playbook = WorkflowTemplate(template_id=f"f288c-{uuid4().hex[:8]}", name="Harbour Lights welcome",
                                description="Welcome a cafe.", workspace_id=ws, template_definition={"steps": []},
                                steps=[], created_by="f288")
    db_session.add(playbook)
    db_session.flush()

    out = asyncio.run(execute_playbook(db_session, ws, {"playbook_id": playbook.id}))

    assert out["success"] is True and re.fullmatch(r"#\d{4}", out["number"])
    assert out["message"].endswith(f"It runs on card {out['number']}: give the owner that number, as the board "
                                   "shows it, not the execution_id.")
    assert "exec-" not in out["message"] and out["execution_id"].startswith("exec-")
