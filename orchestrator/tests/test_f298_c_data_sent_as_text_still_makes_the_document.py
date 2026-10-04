"""F298 (night 8, #0249, 02:27): data sent as text still makes the document.

A playbook step's agent called generate_document with ``data`` as a string, and
the step failed with "Tool generate_document failed: 'str' object has no
attribute 'get'". What an agent sends when its meaning is clear is now read at
the boundary: the checklist itself becomes the PDF's body, an object written out
as JSON is that object, and a list is the PDF's sections.
"""
from __future__ import annotations

import json

import pytest

from tests.f298_fixtures import call_tool, recorded

CHECKLIST = (
    "## Before the collection\n"
    "- [ ] Check Thursday's roast is bagged and labelled\n"
    "- [ ] Count the boxes against the printed labels\n"
    "- [ ] Stack the boxes by the back door for the 3pm collection\n"
)
SECTIONS = [{"title": "Prices", "content": "* Harbour Blend: £19.50 per kilo + VAT"}]


@pytest.mark.parametrize(
    "sent, received",
    [
        pytest.param(CHECKLIST, {"content": CHECKLIST}, id="the-checklist-as-text"),
        pytest.param(json.dumps({"sections": SECTIONS}), {"sections": SECTIONS}, id="an-object-written-as-json"),
        pytest.param(SECTIONS, {"sections": SECTIONS}, id="a-list-of-sections"),
    ],
)
def test_a_pdf_is_made_from_what_the_agent_meant(monkeypatch, sent, received):
    service = recorded(monkeypatch)

    answer = call_tool({"title": "Tom's Monday Dispatch Checklist", "format": "pdf", "data": sent})

    assert answer["success"] is True, answer
    (made,) = service.asked
    assert made["data"] == received
    assert made["format"] == "pdf" and made["title"] == "Tom's Monday Dispatch Checklist"


def test_an_object_is_handed_on_as_a_copy(monkeypatch):
    service = recorded(monkeypatch)
    data = {"sections": SECTIONS}

    call_tool({"title": "Price list", "format": "pdf", "data": data})

    (made,) = service.asked
    assert made["data"] == data and made["data"] is not data
