"""P256-FIX-N17-5 (F446, nights 16–17): the not-done line never splits a sentence, a number, a filename or a bullet.

``claims_backed.SENTENCES`` ended a sentence at every "." (a decimal, ".md", a domain) and started the next one with
the closing marks, so the line was spliced mid-number (night 17 idx 181: "…Ask me again if you want it done.8%),
but…"; night 16 idx 222: "done.example."), glued to the text before it (idx 16: "it.Just to be clear"), ate a closing
"**" (idx 155), and took a bullet's bold label with the claim (idx 167 lost "- **#1751 …:**").

Now a full stop ends a sentence only before whitespace, a closing mark or the end, and the sentence keeps its
closing marks; the claims split the reply as written (marks on) so they find the sentence the remover finds; the
line is put one space after text that ends in none; a list marker, and a bold label that ends in a colon, stay in
front of it, and a line left with only those goes. A question closed by a quote or a paren asks, it claims nothing.
"""
from __future__ import annotations

from consumers.chatbot.claim_removed import without_unbacked_claims
from consumers.chatbot.claims_backed import SENTENCES, claims
from tests.helpers_receipts_rule import tracker_of

OK = {"success": True}
QUERIED = ("query_database", {"sql": "select churn from club"}, OK)


def _line(verb):
    return f"Just to be clear: I haven't {verb} anything in this reply. Ask me again if you want it done."


def _removed(answer, *calls):
    """The reply as the owner gets it in Auto's voice; None when nothing is taken out."""
    removed = without_unbacked_claims(answer, tracker_of(calls).outcomes, True)
    return None if removed is None else removed.text


def test_a_decimal_a_filename_and_a_domain_stay_in_their_sentence():
    assert len(SENTENCES.findall("Churn is 2.8% on harbourline-ruling.md, see sales.example for the rest.")) == 1
    assert SENTENCES.findall("**I've sent it.** Here's the list.") == ["**I've sent it.**", " Here's the list."]


def test_the_churn_bullet_is_never_spliced():
    text = _removed("- Churn is up (12.8%), but I've sent the win-back email to the lapsed members.", QUERIED)
    assert "8%)" not in text and "done.8" not in text
    assert text == f"- {_line('sent')}"


def test_a_bolds_closing_marks_stay_with_its_sentence():
    assert _removed("- **Nothing is queued for it.** It has been moved to Done.") == (
        f"- **Nothing is queued for it.** {_line('moved')}")


def test_a_bullets_colon_label_stays_in_front_of_the_line():
    said = "- **#1751 (Tidewater):** I've sent the reorder to CELLAR.\n- **#1752 (Quay):** waiting on Rosa."
    assert _removed(said, "platform_list_tasks") == (
        f"- **#1751 (Tidewater):** {_line('sent')}\n- **#1752 (Quay):** waiting on Rosa.")
    assert _removed("* **Status**: I've sent the invoice to Declan.") == f"* **Status**: {_line('sent')}"


def test_a_bold_span_without_a_colon_goes_with_its_sentence():
    said = "**I've sent it.** Here's the list.\n\n- **Posted the update.**"
    assert _removed(said) == f"{_line('sent or posted')} Here's the list.\n"      # the bullet left bare goes
    assert _removed("- **Done** I've sent the invoice.") == f"- {_line('sent')}"


def test_the_line_is_never_glued_to_the_text_before_it():
    text = _removed("Here's the plan for it.**I've sent it to Declan.**")
    assert "it.Just" not in text and f" {_line('sent')}" in text
    assert "I've sent it" not in text                                     # found in place, not left in


def test_a_question_closed_by_a_quote_or_a_paren_is_no_claim():
    assert claims("Can you confirm it was sent?”") == []
    assert claims("(Was it sent?)") == []
    assert claims("Should I say it's been sent?'") == []
