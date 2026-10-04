"""F321-A (night 9b, build 15) — a Playbook run's card carries its whole result,
and Auto can read any step's whole output.

Runs #0085 (exec-76879400dd8d) and #0102 (exec-45d8ac862a79) of "Monday green
stock" put "The stock report has been generated and saved to the scratchpad … 2
coffees flagged …" on the card. Step 2's last tool call was scratchpad_write with
the 8-coffee report, after six refused platform_submit_report calls (F321-B); the
card took only the step's last message, and the scratchpad expires. Auto then
asked platform_get_playbook_execution for execution_id "0102" (15:52:32Z) and got
"Execution '0102' not found"; by its id it would have got empty step previews
(it read a key the stored steps do not have) and no final output.
"""
from __future__ import annotations

import asyncio
import io
import json
import uuid
from types import SimpleNamespace as NS
from uuid import UUID

from tests import helpers_playbook_run as hp

STOCK = [("Brazil Cerrado", 540.0), ("Guji Shakiso", 118.0), ("Huila La Esperanza", 212.0),
         ("Kirinyaga AA", 41.0), ("Nariño Buesaco", 300.0), ("Sumatra Gayo", 176.0),
         ("Swiss Water Decaf", 64.0), ("Yirgacheffe Konga", 19.1)]
REPORT = "# Green stock\n" + "\n".join(
    f"- {name}: {kg} kg{' (under 50 kg: reorder)' if kg < 50 else ''}" for name, kg in STOCK)
POINTER = ("The stock report has been generated and saved to the scratchpad. The report shows current inventory "
           "levels for all 8 coffee varieties, with 2 coffees flagged as critically low stock requiring immediate "
           "reordering: Yirgacheffe Konga at 19.1kg and Kirinyaga AA at 41.0kg.")
SAVED = {"action": "scratchpad_write", "params": {"key": "stock_report", "value": REPORT},
         "result": "Stored 'stock_report' in scratchpad.", "success": True}
REFUSED = {"action": "platform_submit_report", "params": "{\"title\": \"Green stock report\"}", "success": False,
           "result": "Tool platform_execute failed: platform_submit_report's params must be an object …"}
STEP_LOG = {"step_id": "1c284509b37d", "order": 2, "agent_id": 342, "agent_name": "Shopify Inventory Watchdog",
            "agent_output": POINTER, "tool_calls": [REFUSED, SAVED], "messages": [], "tokens_used": 21888}


def test_the_card_carries_what_the_last_step_saved_not_just_its_pointer(monkeypatch):
    step_2 = {"status": "success", "result": POINTER, "execution": {"tokens_used": 21_888, "tool_calls": [REFUSED, SAVED]}}

    execution, card = hp.run_playbook(monkeypatch, outcomes=[hp.done("Green stock for all 8 coffees."), step_2],
                                      step_seconds=5, exec_config={})

    assert execution.status == "completed"
    assert card.result.startswith(POINTER)
    assert 'Step 2 saved this as "stock_report":' in card.result
    assert all(f"- {name}: {kg} kg" in card.result for name, kg in STOCK)          # all 8, not 2
    assert REPORT in execution.output_data["final_output"]


def test_a_saved_value_the_answer_already_holds_is_not_repeated():
    from services.playbook_run_result import with_saved_values

    assert with_saved_values(f"Here it is:\n{REPORT}", 2, [SAVED]) == f"Here it is:\n{REPORT}"
    assert with_saved_values("", 2, [SAVED]) == f'Step 2 saved this as "stock_report":\n{REPORT}'
    assert with_saved_values(POINTER, 2, [REFUSED]) == POINTER                     # a failed call saved nothing


def _run_and_card(db_session, seed_workspace):
    from datetime import datetime

    from core.models.core import BoardTask, RecipeExecution, WorkflowTemplate
    from services.board_task_bridge import create_recipe_board_task

    ws = UUID(seed_workspace())
    playbook = WorkflowTemplate(template_id=f"f321-green-stock-{uuid.uuid4().hex[:8]}", name="Monday green stock",
                                description="Stock.", workspace_id=ws, template_definition={"steps": []}, steps=[], created_by="f321")
    db_session.add(playbook)
    db_session.flush()
    run = RecipeExecution(
        execution_id="exec-45d8ac862a79", recipe_id=playbook.id, workspace_id=ws, status="completed",
        started_at=datetime(2026, 10, 4, 15, 49, 28), triggered_by="platform_action",
        output_data={"final_output": POINTER, "total_tokens": 35294, "steps_completed": 2},
        step_results=[
            {"order": 1, "agent_name": "Shopify Inventory Watchdog", "status": "completed", "error": None,
             "output_preview": "Perfect! I can see the database automatically understood my request…",
             "log_url": "s3://automatos-ai/workspaces/x/logs/executions/exec-45d8ac862a79/step_1.json"},
            {"order": 2, "agent_name": "Shopify Inventory Watchdog", "status": "completed", "error": None,
             "output_preview": POINTER[:200] + "...",
             "log_url": "s3://automatos-ai/workspaces/x/logs/executions/exec-45d8ac862a79/step_2.json"},
        ])
    db_session.add(run)
    db_session.flush()
    create_recipe_board_task(db_session, playbook, run)
    card = db_session.query(BoardTask).filter(BoardTask.source_id == run.execution_id).one()
    return NS(ws=ws, run=run, card=card)


def test_auto_reads_a_run_by_its_card_number_with_its_final_output_and_previews(db_session, seed_workspace):
    from modules.tools.discovery.handlers_playbooks import get_playbook_execution

    made = _run_and_card(db_session, seed_workspace)
    said = f"{made.card.workspace_seq:04d}"                                         # Auto sent "0102"

    got = asyncio.run(get_playbook_execution(db_session, made.ws, {"execution_id": said}))

    assert got["success"] is True and got["execution"]["execution_id"] == "exec-45d8ac862a79"
    assert got["execution"]["final_output"] == POINTER
    assert [s["output_preview"][:20] for s in got["execution"]["step_results"]] == [
        "Perfect! I can see t", POINTER[:20]]


def test_auto_reads_a_steps_whole_output_and_what_it_saved(db_session, seed_workspace, monkeypatch):
    import core.storage as storage
    from modules.tools.discovery.handlers_playbooks import get_playbook_execution

    made = _run_and_card(db_session, seed_workspace)
    asked = []

    def _get_object(Bucket, Key):
        asked.append((Bucket, Key))
        return {"Body": io.BytesIO(json.dumps(STEP_LOG).encode("utf-8"))}

    monkeypatch.setattr(storage, "get_s3_client", lambda *a, **k: NS(get_object=_get_object))

    got = asyncio.run(get_playbook_execution(db_session, made.ws, {"execution_id": "exec-45d8ac862a79", "step": 2}))

    step = got["execution"]["step_output"]
    assert asked == [("automatos-ai", "workspaces/x/logs/executions/exec-45d8ac862a79/step_2.json")]
    assert step["output"] == POINTER and step["saved"] == {"stock_report": REPORT}
