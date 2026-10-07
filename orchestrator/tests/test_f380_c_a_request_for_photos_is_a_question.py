"""F380 (night 11, 7 Oct): a social ticket that asks for the photos waits for the owner.

#2162 "Before and after: the Guji re-roast" (Social Media Director) answered "Please
provide the two phone photos… Once I have them, I can create the Instagram draft." in
3 s and closed done: no question reached the owner, and a done ticket can't be
answered. F140's polite-request rule skipped every brief that mentions a post, since
there "Could you confirm Thursday?" may be the draft. A request for the material the
work needs, in an answer that drafted nothing, is a question on a drafting brief too.
A drafted caption that ends "Please let me know if you'd like changes" is still the work.
"""
from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from services.playbook_owner_ask import _DRAFTING_STEP, asks_for_material, owner_question
from services.ticket_owner_ask import result_only_asks

BRIEF_2162 = ("Before and after: the Guji re-roast\nAn Instagram before/after post of the Guji re-roast. Ask me "
              "for the photos rather than leaving the boxes empty.")
RESULT_2162 = ("Please provide the two phone photos of the Guji, the first roast and the re-roast. Once I have them, "
               "I can create the Instagram draft.")


def test_the_brief_is_a_drafting_one():
    assert _DRAFTING_STEP.search(BRIEF_2162)             # why night 11's request read as the draft's


@pytest.mark.parametrize("result", [
    RESULT_2162,
    RESULT_2162.replace(". Once", ".\n\nOnce"),            # the promise on its own last line
    "Could you send me the before and after photos? I'll make the post as soon as they arrive.",
    "Please upload Rosa's quote and her photo for the card.",
])
def test_a_request_for_the_material_is_a_question_on_a_post_brief(result):
    assert owner_question(result, {}, BRIEF_2162) == {"question": result, "options": None}


def test_2162_parks_as_a_question_not_done():
    task = NS(id=2162, source_type="user", source_id=None, title=BRIEF_2162.split("\n")[0],
              description=BRIEF_2162.split("\n", 1)[1])

    assert result_only_asks(task, RESULT_2162) == {"question": RESULT_2162, "options": None}


@pytest.mark.parametrize("result", [
    # a drafted caption that ends on a polite line to the owner
    "Caption: Same beans, second chance. Our Guji re-roast came out brighter and sweeter.\n\n"
    "Please let me know if you'd like changes.",
    # a caption that asks its readers for their photos
    "Show us your morning cup! Please share your photos with us. #HarbourlineCoffee #Guji",
    # a report of the work that offers to take more photos
    "I've drafted the before/after post in Socials. Please send any other photos you'd like on it.",
    "Here's the caption for the re-roast post. Could you send it to Rosa as well?",
])
def test_a_drafted_post_that_ends_politely_is_the_work(result):
    assert asks_for_material(result) is False
    assert owner_question(result, {}, BRIEF_2162) is None
