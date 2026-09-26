"""F140 (night 4, the persona's fix-first #2) — a playbook step that needs the
owner stops its run and asks them.

Night 4: a step could not ask the owner. platform_ask_human wanted a subject no
step is given (B15/B23 were refused; five of six calls came with params={}, and
two of those runs then drafted orders with quantities nobody gave them), and a
step that ended on a question never reached the owner (B4: steps 1, 3 and 4 of
exec-dcb82c78525b ended "Could you please provide…", and the run completed).

A step now needs the owner when (a) it calls platform_ask_human, which the
executor answers itself, (b) its whole answer is short and ends by asking for
what it lacks, or (c) it writes a NEEDS YOU: line. The run stops, failed with
"Needs you: <question>", before F131's rule and without a retry, and the
question goes on the run's card. A draft that ends on a question is the step's
work, not an ask.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from api import recipe_executor as rex
from config import config
from modules.tools.discovery import handlers_asks
from services import playbook_owner_ask as ask
from tests.helpers_playbook_run import run_playbook

# Night 4, exec-dcb82c78525b (B4): the steps' own prompts and answers.
B4_S1_PROMPT = ("Read the contents of 'green-coffee-list-autumn-2026.csv'. If 'green-coffee-list-autumn-2026.csv' "
                "does not exist, ask the human for the correct file path. Parse the CSV data and return it as a "
                "structured list of dictionaries.")
B4_S1 = ("I am unable to directly read the file from Dropbox without a specified path. Could you please provide "
         "the full path to 'green-coffee-list-autumn-2026.csv' in your Dropbox, or confirm if the file exists and "
         "share its exact location?")
B4_S3_PROMPT = ("For each green coffee identified in the previous step, *ask the human for the reorder quantity in "
                "kilograms*. Once provided, draft a reorder email.")
B4_S3 = ("It seems I don't have the `low_stock_coffees` from the previous step. I need to know which green coffees "
         "are low in stock to ask for reorder quantities and draft emails. Can you please provide the list of green "
         "coffees that were identified as low in stock, along with their current stock levels and the importer's "
         "contact information?")
B4_S4_PROMPT = ("Consolidate all drafted reorder emails into a single report, clearly listing each email's "
                "recipient, subject, and body.")
B4_S4 = ("I can consolidate the drafted reorder emails into a single report, but I need the content of those "
         "emails first. Since the previous steps failed to retrieve the low stock coffee information, I don't have "
         "any drafted emails.\n\nCould you please provide the recipient, subject, and body for each reorder email "
         "you would like to include in the report?")


# ── which answers ask the owner ─────────────────────────────────────────────

@pytest.mark.parametrize("output, prompt", [(B4_S1, B4_S1_PROMPT), (B4_S3, B4_S3_PROMPT), (B4_S4, B4_S4_PROMPT)])
def test_b4s_three_steps_asked_the_owner(output, prompt):
    assert ask.owner_question(output, {}, prompt) == {"question": output, "options": None}


def test_a_long_answer_that_ends_on_a_question_is_the_steps_work():
    """Night 4's one other answer ending in '?': a 1,428-character plan whose last
    section is 'To find out'."""
    plan = ("#### Tell them yes:\n\n" + "*   A line of the first month's plan for The Cobbled Yard.\n" * 12
            + "\n#### To find out:\n\n*   Which day is the correct van day for deliveries to Bath?")
    assert len(plan) > config.PLAYBOOK_OWNER_ASK_MAX_CHARS
    assert ask.owner_question(plan, {}, "Plan the café's first month with us.") is None


@pytest.mark.parametrize("draft, prompt", [
    ("Hi Priya,\n\nThanks for your first order. Could you confirm Thursday works for the delivery?",
     "Draft a note to Priya about her first delivery."),
    ("Subject: Reorder of Yirgacheffe Konga\n\nCould you please confirm you can ship 60 kg by Friday?",
     "Draft the reorder email to the importer."),
    ("> Could you let us know which Saturday suits your tasting?", "Write the reply to the café."),
    ("Autumn is here, and so is Harbour Blend. Which one will you try first?",
     "Write an Instagram caption for the autumn blends."),
    ("Could you confirm Thursday works for your first delivery?", "Draft a one-line text message to the café."),
])
def test_a_draft_that_ends_on_a_question_is_not_an_ask(draft, prompt):
    assert ask.owner_question(draft, {}, prompt) is None


def test_an_ask_may_quote_the_error_that_stopped_it():
    output = ("The read returned:\n\n> path not found: /Stock/green-coffee.csv\n\n"
              "I can't go on without the list. Could you give me its right path?")
    assert ask.owner_question(output, {}, "Read the green coffee list.")["question"] == output


def test_a_needs_you_line_asks_what_follows_it():
    output = "Drafted two of the three emails.\n\n**NEEDS YOU:** Which importer gets the Huila order?"
    assert ask.owner_question(output, {}, "Draft the reorder emails.") == {
        "question": "Which importer gets the Huila order?", "options": None}


def test_the_ask_tools_question_comes_first():
    result = {"owner_ask": {"question": "How many kilos of Konga?", "options": ["60 kg", "30 kg"]}}
    assert ask.owner_question("Done.", result, "") == {"question": "How many kilos of Konga?",
                                                       "options": ["60 kg", "30 kg"]}


def test_the_length_limit_is_config(monkeypatch):
    monkeypatch.setattr(config, "PLAYBOOK_OWNER_ASK_MAX_CHARS", 300)
    assert ask.owner_question(B4_S1, {}, B4_S1_PROMPT) is not None     # 231 characters
    assert ask.owner_question(B4_S3, {}, B4_S3_PROMPT) is None         # 334


# ── the run ─────────────────────────────────────────────────────────────────

STEP = {"step_id": "s1", "order": 1, "agent_id": 7, "error_handling": "stop", "max_retries": 1,
        "prompt_template": B4_S1_PROMPT}


def _said(output, tool_calls=(), **extra):
    return {"status": "success", "result": output,
            "execution": {"tokens_used": 900, "tool_calls": list(tool_calls)}, **extra}


@pytest.fixture
def staged(monkeypatch):
    """The question as stage_question receives it; the card parks as it would."""
    questions = []

    async def _stage(db, workspace_id, **kwargs):
        questions.append(kwargs)
        kwargs["park"].status = "blocked"
        return {"success": True, "ask_id": 41, "parked": True}

    monkeypatch.setattr(handlers_asks, "stage_question", _stage)
    return questions


@pytest.fixture
def failures(monkeypatch):
    """_fail_execution is every failure side effect: the bell, report, memory, count."""
    calls = []

    async def _fail(db, execution_id, error_message, step_results=None, review_card=None):
        calls.append(error_message)

    monkeypatch.setattr(rex, "_fail_execution", _fail)
    return calls


def test_a_step_that_asks_stops_its_run_and_asks_the_owner(monkeypatch, staged, failures):
    calls = []
    execution, card = run_playbook(monkeypatch, outcomes=[_said(B4_S1)], step_seconds=5, exec_config={},
                                   steps=[STEP], calls=calls)
    assert len(calls) == 1                              # asked once: never retried
    assert (execution.status, execution.error_message) == ("failed", f"Needs you: {B4_S1}")
    assert card.status == "blocked" and card.blocked_reason == f"Needs you: {B4_S1} (ask #41)"
    (question,) = staged
    assert (question["subject_type"], question["subject_id"]) == ("board_task", "760")
    assert question["question"] == f"{B4_S1}\n\nAnswering runs the whole playbook again from step 1."
    assert question["details"] == {"playbook_ask": {"execution_id": "exec-120", "recipe_id": 79, "step": 1,
                                                    "question": B4_S1}}
    assert execution.execution_metadata["needs_owner"] == {"ask_id": 41, "step": 1, "card_id": 760}
    assert execution.step_results[0]["status"] == "failed"
    assert failures == []                               # no failure bell, report, memory or count


def test_asking_comes_before_f131s_failed_last_call(monkeypatch, staged, failures):
    """B4's step 1 failed to read the file AND asked for its path: the owner can answer that."""
    read = {"action": "DROPBOX_READ_FILE", "result": "path not found", "success": False}
    execution, _card = run_playbook(monkeypatch, outcomes=[_said(B4_S1, [read])], step_seconds=5,
                                    exec_config={}, steps=[STEP])
    assert execution.error_message == f"Needs you: {B4_S1}"


def test_the_question_names_what_the_run_already_changed(monkeypatch, staged, failures):
    draft = {"action": "GMAIL_CREATE_EMAIL_DRAFT", "result": "draft r-17", "success": True}
    read = {"action": "DROPBOX_READ_FILE", "result": "the list", "success": True}
    steps = [{**STEP, "prompt_template": "Draft the reorder emails."},
             {**STEP, "step_id": "s2", "order": 2, "prompt_template": B4_S3_PROMPT}]
    run_playbook(monkeypatch, outcomes=[_said("Drafted both emails.", [read, draft]), _said(B4_S3)],
                 step_seconds=5, exec_config={}, steps=steps)
    (question,) = staged
    assert question["question"] == (
        f"{B4_S3}\n\nAnswering runs the whole playbook again from step 1.\n"
        "This run already made these changes, and the rerun makes them again: step 1: GMAIL_CREATE_EMAIL_DRAFT.")


def test_a_draft_that_ends_on_a_question_completes_its_run(monkeypatch, staged, failures):
    step = {**STEP, "prompt_template": "Draft a one-line text message to the café."}
    execution, card = run_playbook(monkeypatch, outcomes=[_said("Could you confirm Thursday works for you?")],
                                   step_seconds=5, exec_config={}, steps=[step])
    assert execution.status == "completed" and card.status == "done" and staged == []


def test_a_run_that_may_not_ask_fails_as_before(monkeypatch, staged, failures):
    """A run a website visitor started never reaches the owner's Questions (F155)."""
    import core.security.surface as surface

    monkeypatch.setattr(surface, "widget_turn", lambda: True)
    run_playbook(monkeypatch, outcomes=[_said(B4_S1)], step_seconds=5, exec_config={}, steps=[STEP])
    assert staged == [] and failures == [f"Needs you: {B4_S1}"]


def test_a_rerun_carries_the_owners_answer_in_every_step(monkeypatch, staged, failures):
    calls = []
    answers = {"owner_answers": [{"step": 1, "question": B4_S1, "answer": "Dropbox/Stock/green-coffee.csv",
                                  "ask_id": 41}]}
    steps = [STEP, {**STEP, "step_id": "s2", "order": 2, "prompt_template": "Summarise the list."}]
    execution, _card = run_playbook(monkeypatch, outcomes=[_said("Read 14 coffees."), _said("Summary.")],
                                    step_seconds=5, exec_config={}, steps=steps, calls=calls,
                                    execution_metadata=answers)
    assert execution.status == "completed"
    for call in calls:
        assert call["clean_prompt"].endswith(
            "## The owner's answer\nAn earlier run of this playbook stopped to ask the owner, and they "
            "answered. Use the answer; do not ask it again.\n\n"
            f"**Step 1 asked:** {B4_S1}\n**The owner answered:** Dropbox/Stock/green-coffee.csv")


def test_the_step_is_told_how_to_ask():
    from modules.context.sections.base import SectionContext
    from modules.context.sections.playbook_context import PlaybookContextSection

    ctx = SectionContext(agent=None, workspace_id="00000000-0000-0000-0000-0000000000c1",
                         recipe_step={"name": "Monday reorder", "step_number": 1, "total_steps": 2,
                                      "instructions": B4_S1_PROMPT})
    rendered = asyncio.run(PlaybookContextSection().render(ctx))
    assert "`NEEDS YOU: <your question>`" in rendered and "platform_ask_human (only the question)" in rendered


# ── (a) platform_ask_human inside the step ──────────────────────────────────

class _Step:
    """One API step's own loop (api/recipe_executor._execute_step), its model scripted."""

    def __init__(self, monkeypatch, *turns):
        import modules.agents.factory.agent_factory as agent_factory
        import modules.context as context_mod
        import modules.tools.services.composio_hint_service as hint_service
        import modules.tools.services.composio_tool_service as tool_service
        import modules.tools.tool_router as tool_router
        import services.cli_ticket_lane as cli_lane

        responses = [SimpleNamespace(tool_calls=calls or None, content=text, usage=None) for calls, text in turns]
        self.llm_calls = 0
        step = self

        class _LLM:
            async def generate_response(self, messages, tools):
                step.llm_calls += 1
                return responses.pop(0)

        class _Factory:
            def __init__(self, db_session):
                pass

            async def activate_agent(self, agent_id):
                return SimpleNamespace(llm_manager=_LLM())

        class _Context:
            def __init__(self, db):
                pass

            async def build_context(self, **kwargs):
                return SimpleNamespace(system_prompt="system", tools=[])

        class _NoApps:
            def __init__(self, db):
                pass

            def get_tools_for_step(self, **kwargs):
                return SimpleNamespace(tools=[], strategy="none", search_ms=0, entity_id=None, app_names=[],
                                       action_set=set())

            def build_hints(self, **kwargs):
                return SimpleNamespace(hint_lines=[], strategy_used="none", matched_actions=[])

        self.spine = MagicMock(name="tool_router")
        self.spine.execute_and_format = AsyncMock(return_value={"success": True, "llm_context": "sent"})
        monkeypatch.setattr(cli_lane, "is_cli_agent", lambda db, agent_id: False)
        monkeypatch.setattr(agent_factory, "AgentFactory", _Factory)
        monkeypatch.setattr(context_mod, "ContextService", _Context)
        monkeypatch.setattr(tool_service, "ComposioToolService", _NoApps)
        monkeypatch.setattr(hint_service, "ComposioHintService", _NoApps)
        monkeypatch.setattr(tool_router, "get_tool_router", lambda: self.spine)

    def run(self):
        return asyncio.run(rex._execute_step(
            db=MagicMock(name="db"), agent=SimpleNamespace(id=7, name="GREEN COFFEE STOCK"),
            clean_prompt=B4_S3_PROMPT, workspace_id=uuid.uuid4(), max_iterations=4,
            recipe_execution_id="exec-120"))


def _call(name, args, call_id="tc-1"):
    return {"id": call_id, "function": {"name": name, "arguments": json.dumps(args)}}


def test_the_step_asks_through_its_run_and_goes_no_further(monkeypatch):
    asked = _call("platform_execute", {"action": "platform_ask_human",
                                       "params": {"question": "How many kilos of Konga?",
                                                  "options": ["60 kg", "30 kg"],
                                                  "subject_type": "playbook_run", "subject_id": "made-up"}})
    step = _Step(monkeypatch, ([asked], ""), (None, "never reached"))
    result = step.run()
    assert result["owner_ask"] == {"question": "How many kilos of Konga?", "options": ["60 kg", "30 kg"]}
    assert step.llm_calls == 1                          # the step ends with the ask
    step.spine.execute_and_format.assert_not_called()   # the model's subject reaches nothing
    (record,) = result["execution"]["tool_calls"]
    assert (record["action"], record["success"], record["result"]) == ("platform_ask_human", True, ask.ASK_TAKEN)


def test_an_ask_with_no_question_is_told_to_put_one(monkeypatch):
    """Night 4's shape: platform_execute with params={}."""
    empty = _call("platform_execute", {"action": "platform_ask_human", "params": {}})
    retried = _call("platform_ask_human", {"question": "How many kilos of Konga?"}, call_id="tc-2")
    step = _Step(monkeypatch, ([empty], ""), ([retried], ""))
    result = step.run()
    first, second = result["execution"]["tool_calls"]
    assert (first["success"], first["result"]) == (False, ask.ASK_WITHOUT_QUESTION)
    assert result["owner_ask"] == {"question": "How many kilos of Konga?", "options": None}


def test_nothing_acts_after_the_step_asked(monkeypatch):
    asked = _call("platform_ask_human", {"question": "Send the Konga order now?"})
    send = _call("GMAIL_SEND_EMAIL", {"to": "hollis@example.com"}, call_id="tc-2")
    step = _Step(monkeypatch, ([asked, send], ""))
    result = step.run()
    step.spine.execute_and_format.assert_not_called()
    _, not_sent = result["execution"]["tool_calls"]
    assert (not_sent["action"], not_sent["success"], not_sent["result"]) == (
        "GMAIL_SEND_EMAIL", False, "Not run: this step stopped to ask the owner.")
