"""F130 (night 4, the persona's fix-first #1) — a playbook step gets the earlier
steps' answers, whichever kind of agent wrote them, and its blanks are filled.

Night 4: a session (Claude Code) step's ticket held only its own prompt, so a
multi-step playbook needed the owner as courier (B13, B54, B73, B75, B76). An
api step got a 500-character stub of the step before (B21), and the playbook
section cut even that at 2,000 tokens from the end, latest answer first.
`{{date}}`, `{{month}} {{year}}` and `{{roast_log_filename}}` stayed literal
although the run supplied exactly those keys (B8, B27, B57), and `output_key`
never filled a later step's blank (B68: step 1 stored "£22.00" under
price_per_kilo; step 2 got "NO FIGURE FROM STEP 1"). A second step then wrecked
a perfect first one. Where one session may read another's folder is Gerard's
design call; this carries the answers themselves.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from api import recipe_executor as rex
from config import config
from core.services.playbook_scratchpad import PlaybookScratchpad
from tests.helpers_playbook_run import WS, done, run_playbook

PRICE_LIST = ("Harbourline wholesale, autumn 2026. " + "House blend 1 kg £22.00; single origin 1 kg £26.50. " * 30
              + "Minimum order 6 kg, delivery Thursdays, £4.50 under 12 kg.")


class _Hash:
    """The Redis hash the scratchpad writes, in memory."""

    def __init__(self):
        self.rows = {}

    def hset(self, key, field, value):
        self.rows[field] = value

    def hget(self, key, field):
        return self.rows.get(field)

    def hgetall(self, key):
        return dict(self.rows)

    def expire(self, key, ttl):
        pass

    def delete(self, key):
        self.rows.clear()


def _pad_after_step_1(answer):
    pad = PlaybookScratchpad("exec-130", redis_client=_Hash())
    pad.write_step_results(step_order=1, tool_calls=[], agent_output=answer)
    return pad


# ── the answer is kept whole ────────────────────────────────────────────────

def test_a_later_step_reads_the_earlier_answer_whole():
    context = _pad_after_step_1(PRICE_LIST).format_context_for_step(2)
    assert len(PRICE_LIST) > 1500
    assert PRICE_LIST in context


def test_an_answer_past_the_cap_is_cut_where_it_is_read_and_says_so(monkeypatch):
    monkeypatch.setattr(config, "PLAYBOOK_STEP_ANSWER_MAX_CHARS", 40, raising=False)
    context = _pad_after_step_1(PRICE_LIST).format_context_for_step(2)
    assert f"{PRICE_LIST[:40]}\n[Cut at 40 of {len(PRICE_LIST):,} characters.]" in context


# ── a session step's ticket carries them ────────────────────────────────────

def test_a_session_steps_ticket_carries_the_earlier_answers(monkeypatch):
    from services import cli_ticket_lane as lane

    filed = {}

    async def _run_ticket(db, **kwargs):
        filed.update(kwargs)
        return {"status": "success", "result": "quote sent"}

    monkeypatch.setattr(lane, "is_cli_agent", lambda db, agent_id: True)
    monkeypatch.setattr(lane, "run_cli_ticket_and_wait", _run_ticket)
    asyncio.run(rex._execute_step(
        db=None, agent=SimpleNamespace(id=9), clean_prompt="Write Porto Café's quote from the price list.",
        workspace_id=WS, scratchpad=_pad_after_step_1(PRICE_LIST), step_order=2, max_iterations=3,
        recipe_name="Wholesale quote", total_steps=2, recipe_execution_id="exec-130",
    ))
    assert filed["prompt"].startswith("Write Porto Café's quote from the price list.\n\n")
    assert PRICE_LIST in filed["prompt"]


# ── named blanks fill from the run's input and the earlier answers ──────────

def test_named_blanks_fill_from_the_run_input_and_the_earlier_answer(monkeypatch):
    steps = [
        {"step_id": "s1", "order": 1, "agent_id": 7, "output_key": "price_per_kilo",
         "prompt_template": "Read the house blend price for {{ date }}.", "error_handling": "stop", "max_retries": 0},
        {"step_id": "s2", "order": 2, "agent_id": 7,
         "prompt_template": "Quote {{price_per_kilo}} a kilo (confirm {price_per_kilo}) for the {{date}} order.",
         "error_handling": "stop", "max_retries": 0},
    ]
    calls = []
    execution, _card = run_playbook(
        monkeypatch, outcomes=[done("£22.00"), done("quote sent")], step_seconds=5, exec_config={},
        steps=steps, input_data={"date": "2026-09-22"}, calls=calls,
    )
    assert execution.status == "completed"
    assert calls[0]["clean_prompt"] == "Read the house blend price for 2026-09-22."
    assert calls[1]["clean_prompt"] == "Quote £22.00 a kilo (confirm £22.00) for the 2026-09-22 order."


def test_a_blank_that_names_neither_is_left_as_written(monkeypatch):
    """A prompt may be asking for a template: `{{first_name}}` is the author's text."""
    steps = [{"step_id": "s1", "order": 1, "agent_id": 7, "error_handling": "stop", "max_retries": 0,
              "prompt_template": "Draft the café newsletter with a {{first_name}} greeting for {{month}}."}]
    calls = []
    run_playbook(monkeypatch, outcomes=[done("drafted")], step_seconds=5, exec_config={},
                 steps=steps, input_data={"month": "October"}, calls=calls)
    assert calls[0]["clean_prompt"] == "Draft the café newsletter with a {{first_name}} greeting for October."


# ── the playbook section keeps the latest answer ────────────────────────────

def test_the_playbook_section_keeps_the_latest_answer():
    from modules.context.sections.base import SectionContext
    from modules.context.sections.playbook_context import PlaybookContextSection

    earlier = f"## Step 1\n- Answer:\n{PRICE_LIST * 8}\n## Step 2\n- Answer:\nPorto Café accepts: 8 kg, Thursday."
    ctx = SectionContext(agent=None, workspace_id=str(WS), recipe_step={
        "name": "Wholesale quote", "step_number": 3, "total_steps": 3,
        "instructions": "Send the confirmation.", "previous_output": earlier,
    })
    rendered = asyncio.run(PlaybookContextSection().render(ctx))
    assert rendered.endswith("Porto Café accepts: 8 kg, Thursday.")


@pytest.mark.parametrize("name, default", [("PLAYBOOK_CONTEXT_MAX_TOKENS", 12000),
                                           ("PLAYBOOK_STEP_ANSWER_MAX_CHARS", 12000)])
def test_the_caps_are_config(name, default):
    source = (rex.__file__.rsplit("/api/", 1)[0] + "/config.py")
    assert f'{name}: int = int(os.getenv("{name}", "{default}"))' in open(source).read()
