"""F320 (night 9b) — the card's answer is the work, not the agent's working above it.

Cards opened with the agent's narration: "Perfect! Now I have all the information
needed…" (#0046, #0054, #0063, #0070), "Now let me generate the report in the correct
format…" (#0069), "Now I have all the information needed. Let me write the brief as
requested:" (#1994). The owner: "I'd have to trim it before sending anything on." The
result an agent run hands back now starts at the work, and the work is untouched.
Fixtures are the cards' real results from night 9b (board_tasks.result), abridged below
the opening.
"""
from __future__ import annotations

import asyncio

import pytest

from modules.agents.factory.answer_check import said_plainly
from services.answer_working import is_working, the_answer_itself, without_the_working
from services.result_substance import STOPPED_HEADER

WORK_0046 = ("**Contact for reordering Kirinyaga:** Maya Odum at Tidewater Importers (London)  \n"
             "- Email: maya@tidewater-importers.example  \n- Phone: 020 7946 0381\n\n"
             "**Delivery time:** 3 weeks from order to your door")
CARD_0046 = ("Perfect! Now I have all the information needed to answer your question about reordering "
             f"Kirinyaga coffee.\n\n{WORK_0046}")
WORK_0054 = ("Based on the October 2026 club box contents and importer information:\n\n"
             "**Coffees in the October Club Box:**\n- **Guji Shakiso** (Ethiopia, natural)\n"
             "- **Nariño Buesaco** (Colombia, washed)")
CARD_0054 = ("Perfect! Now I have all the information I need. Let me compile the answer with the importers "
             f"and payment terms for the October club box coffees.\n\n{WORK_0054}")
WORK_0063 = ("**Kirinyaga Stock Analysis for 120 Christmas Boxes**\n\n**Requirements:**\n"
             "- Each Christmas box contains: 250g Kirinyaga AA (from christmas-boxes-2026.md)")
CARD_0063 = ("Perfect! Now I have all the information I need. Let me calculate the requirements and show the "
             f"breakdown.\n\n{WORK_0063}")
WORK_0069 = ("## Current Green Coffee Stock Report - October 4, 2026\n\n**⚠️ CRITICAL LOW STOCK ALERTS** "
             "(Under 50kg):\n• **Yirgacheffe Konga**: 19.1 kg - CRITICAL REORDER NEEDED")
CARD_0069 = ("Now let me generate the report in the correct format. Based on the previous step results and "
             f"current database query, here's the stock report:\n\n{WORK_0069}")
WORK_0070 = ("**Current payment terms:** Quay Coffee House is currently on **14-day payment terms**.\n\n"
             "**Planned change:** They will move to **30-day payment terms starting in November**.")
CARD_0070 = ("Perfect! Now I have all the information I need. Based on the owner's corrections, I need to "
             f"provide both current terms and the planned change, with clear sources.\n\n{WORK_0070}")
WORK_1994 = ("## Monday Morning Brief - 5 October 2026\n\n**Club boxes going out Monday:** 63 Harvest Club "
             "boxes (from harbourline_shop database - active subscribers as of today).")
CARD_1994 = f"Now I have all the information needed. Let me write the brief as requested:\n\n{WORK_1994}"


@pytest.mark.parametrize("card, work", [(CARD_0046, WORK_0046), (CARD_0054, WORK_0054), (CARD_0063, WORK_0063),
                                        (CARD_0069, WORK_0069), (CARD_0070, WORK_0070), (CARD_1994, WORK_1994)])
def test_the_card_starts_at_the_work_and_the_work_is_untouched(card, work):
    assert the_answer_itself(card) == work


# #0042 and #0043, the night's best, open on the source; #0045's rerun the same (4/5).
OPENS_ON_ITS_SOURCE = ("Based on your wholesale terms, here's what you charge for delivery:\n\n"
                       "**For a 10 kg café order: £8.50 delivery charge**")
NEWSLETTER = ("Hello, October! This month, your Harvest Club box brings you two distinct coffees to enjoy. We have "
              "the Guji Shakiso from Ethiopia, a naturally processed coffee with notes of blueberry.")
DRAFT = "Hi Rosa,\n\nFor a 10 kg order, delivery costs £8.50.\n\nGerard, Harbourline Coffee Roasters"


@pytest.mark.parametrize("answer", [OPENS_ON_ITS_SOURCE, NEWSLETTER, DRAFT,
                                    "Let me know if you want the October figures too.",
                                    "**63 boxes** go out on Monday, October 5th."])
def test_an_answer_that_opens_on_the_work_is_left_alone(answer):
    assert the_answer_itself(answer) == answer


def test_nothing_but_working_is_left_as_it_was():
    assert without_the_working("Perfect! I have everything I need.") == "Perfect! I have everything I need."


@pytest.mark.parametrize("sentence", ["Perfect!", "Now I have all the information I need.",
                                      "Let me write the brief as requested:", "I'll draft the email following "
                                      "the brand voice guidelines from brand-voice.md.",
                                      "Based on my analysis of the available data, I can now provide the "
                                      "complete calculation with the correct figures."])
def test_the_agents_own_process_is_working(sentence):
    assert is_working(sentence)


def test_every_agent_run_hands_back_the_work_and_a_stopped_step_still_goes_to_review():
    async def run(*_a, **_k):
        return {"status": "success", "result": CARD_1994, "execution": {"tool_iterations": 7}}

    async def stopped(*_a, **_k):
        return {"status": "success", "result": "Now I have the stock. Let me check the payment terms report:"}

    out = asyncio.run(said_plainly(run)())
    assert out["result"] == WORK_1994 and out["execution"] == {"tool_iterations": 7}
    assert asyncio.run(said_plainly(stopped)())["result"].startswith(STOPPED_HEADER)    # F306 first
