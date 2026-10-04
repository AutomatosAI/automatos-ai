"""F320 (night 9b) — taking the working off never takes a fact with it.

#1981's first run (Support, "tell Rosa delivery is free on 10 kg") wrote the draft the
owner asked for, and only its working said the owner was wrong: "it would normally cost
£8.50, but the brief says to tell her delivery is free". The owner: "The one thing I
needed to know (#1981) was in the working, not the answer." So a working sentence goes
only when it carries no figure, name or quote that the rest of the answer lacks, and a
sentence that is about the shop, not the agent's process, always stays. Fixtures are
night 9b's real results: #1981 run 1 from the owner's log, #1970 and #0072 (#1929)
from board_tasks.result, abridged below the opening.
"""
from __future__ import annotations

from services.answer_working import carries_a_fact, the_answer_itself

DRAFT_1981 = ("Hi Rosa,\n\nGood news about your 10 kg order - delivery will be free on this one.\n\n"
              "Your coffee will be roasted fresh and delivered on your usual day. I'll send the invoice through "
              "with the delivery charge removed.\n\nGerard, Harbourline Coffee Roasters")
WHAT_THE_OWNER_NEEDED = ("Based on the wholesale terms document (wholesale-terms-2026.md), I can see that delivery "
                         "is £8.50 for orders under 12 kg, but £5.00 for orders of 12 kg and over. Since Rosa's "
                         "order is 10 kg, it would normally cost £8.50, but the brief says to tell her delivery "
                         "is free.")
CARD_1981_RUN_1 = (f"Perfect! Now I have all the information I need. {WHAT_THE_OWNER_NEEDED} I'll draft the email "
                   f"following the brand voice guidelines from brand-voice.md.\n\n{DRAFT_1981}")


def test_1981_the_owner_still_reads_that_delivery_is_not_free():
    answer = the_answer_itself(CARD_1981_RUN_1)

    assert answer == f"{WHAT_THE_OWNER_NEEDED}\n\n{DRAFT_1981}"
    assert "it would normally cost £8.50" in answer                     # night 9b: lost in the working
    assert "Perfect!" not in answer and "I'll draft the email" not in answer


QUOTE_1970 = ("The cafe notes confirm that **\"Quay Coffee House (Plymouth), Wednesday: Kirinyaga AA for filter, "
              "they love it\"**.")
BODY_1970 = ("From the cafe notes, **Quay Coffee House takes Kirinyaga AA for filter and they love it**, with "
             "Wednesday deliveries.\n\n**Converting roasted to green coffee needed:**\n"
             "- Roast loss: **15%** (from roast-rules.md)")
CARD_1970 = (f"Perfect! {QUOTE_1970} Now I have all the data I need to do the proper calculation using the correct "
             "roast loss of 15% from the roast rules, and the actual Quay Coffee House orders since summer.\n\n"
             "Based on my analysis of the available data, I can now provide the complete calculation with the "
             "correct figures. From the database query, the current Kirinyaga stock is **41 kg** (from the "
             f"harbourline_shop database).\n\n{BODY_1970}")


def test_1970_keeps_what_the_notes_say_and_the_41_kg_and_drops_only_the_narration():
    answer = the_answer_itself(CARD_1970)

    assert answer == (f"{QUOTE_1970}\n\nFrom the database query, the current Kirinyaga stock is **41 kg** "
                      f"(from the harbourline_shop database).\n\n{BODY_1970}")


ONLY_IN_THE_WORKING_0072 = ("I need to identify which coffees are single origins (excluding Cerrado and Sumatra "
                            "which go in blends) and rank them by profitability.")
BODY_0072 = ("From the September margin sheet (margin-sheet-sep-2026.csv), looking at 250g bags and excluding "
             "blend components:\n\n**No, Kirinyaga AA is no longer our most profitable single origin per 250g "
             "bag.**")
CARD_0072 = ("Perfect! Now I have the September margin sheet. I can see the actual costing figures including bag, "
             f"label, labour and roast loss. {ONLY_IN_THE_WORKING_0072}\n\n{BODY_0072}")


def test_0072_a_name_said_only_in_the_working_stays():
    assert the_answer_itself(CARD_0072) == f"{ONLY_IN_THE_WORKING_0072}\n\n{BODY_0072}"


def test_a_fact_is_a_figure_a_name_or_a_quote_the_rest_does_not_repeat():
    rest = "Kirinyaga AA: 41 kg in stock, £9.75 a kilo."
    assert not carries_a_fact("Now I have the Kirinyaga stock of 41.0 kg.", rest)       # 41.0 is 41
    assert carries_a_fact("Now I have the 55 kg from the 1 September list.", rest)
    assert carries_a_fact("Now I have Tidewater's lead time.", rest)
    assert carries_a_fact('Let me quote "on 14 days by agreement" here.', rest)
