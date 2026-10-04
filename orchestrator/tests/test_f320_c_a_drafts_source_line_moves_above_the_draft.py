"""F320 (night 9b) — a source line inside a draft moves above it; it is never dropped.

#0065's draft to Rosa ended "*Source: wholesale-terms-2026.md and brand-voice.md*"
after the sign-off ("ends with '*Source: …*' inside the draft", 4/5); #0002's sat
inside the body, above the sign-off. Either way the owner cuts it before sending. A
source is content to the owner (answers lose marks when a figure has none, #1970 run 2,
and gain them when every figure has one), so it is moved out of the draft, above the
greeting, and the draft runs from greeting to sign-off. Fixtures are the cards' real
results (board_tasks.result).
"""
from __future__ import annotations

from services.answer_working import sources_out_of_the_draft, the_answer_itself

DRAFT_0065 = ("Hi Rosa,\n\nFor a 10 kg order, delivery costs £8.50. This applies to all orders under 12 kg - once "
              "you hit 12 kg or more, delivery drops to £5.00.\n\nGerard, Harbourline Coffee Roasters")
SOURCE_0065 = "*Source: wholesale-terms-2026.md and brand-voice.md*"
NOTE_0065 = ("Based on the wholesale terms document, a 10kg order would fall under the \"under 12 kg\" delivery "
             "charge category, which costs £8.50.")
CARD_0065 = (f"Perfect! Now I have all the information I need. {NOTE_0065} I also have the correct signature "
             f"format from the brand voice document.\n\n{DRAFT_0065}\n\n{SOURCE_0065}")


def test_0065_the_draft_ends_on_its_sign_off_and_the_source_sits_above_it():
    answer = the_answer_itself(CARD_0065)

    assert answer == f"{NOTE_0065}\n\n{SOURCE_0065}\n\n{DRAFT_0065}"
    assert answer.endswith("Gerard, Harbourline Coffee Roasters")


SOURCE_0002 = ("*Source: This information comes from our wholesale-terms-2026.md document, specifically from the "
               "Delivery section which states: \"Carriage is charged per drop: under 12 kg: **£8.50**, 12 kg and "
               "over: **£5.00**\"*")
CARD_0002 = ("Hi Rosa,\n\n**For a 10kg coffee order:**\n- The delivery charge will be **£8.50** (since 10kg is "
             f"under our 12kg threshold)\n\n{SOURCE_0002}\n\nGerard, Harbourline Coffee Roasters")


def test_0002_a_source_inside_the_body_moves_out_whole():
    answer = sources_out_of_the_draft(CARD_0002)

    assert answer == (f"{SOURCE_0002}\n\nHi Rosa,\n\n**For a 10kg coffee order:**\n- The delivery charge will "
                      "be **£8.50** (since 10kg is under our 12kg threshold)\n\nGerard, Harbourline Coffee Roasters")


def test_a_sources_list_under_a_draft_moves_with_its_items():
    card = ("Hi Maya,\n\nPlease send 2 sacks of Kirinyaga AA.\n\nGerard, Harbourline Coffee Roasters\n\n"
            "**Sources:**\n- Price: green-coffee-list-1-sep-2026.csv\n- Lead time: importers-and-green-buying.md")

    assert sources_out_of_the_draft(card) == (
        "**Sources:**\n- Price: green-coffee-list-1-sep-2026.csv\n- Lead time: importers-and-green-buying.md\n\n"
        "Hi Maya,\n\nPlease send 2 sacks of Kirinyaga AA.\n\nGerard, Harbourline Coffee Roasters")


def test_an_answer_with_no_draft_keeps_its_sources_where_they_are():
    answer = ("**Standard payment terms: 30 days** from the invoice date.\n\n"
              "*Sources: wholesale-terms-2026.md document and harbourline_shop database (wholesale_accounts table)*")
    assert the_answer_itself(answer) == answer
