"""F369 (night 10c, chat 423d83ca 18:31): "Just make it, no questions." and Auto asked anyway.

"Can you make me an invoice as a PDF on my Branded Invoice: to Lantern Kitchen … Just make it,
no questions." got "I need the invoice number, the payment terms, and the due date". The named
template's rules (F351) told Auto to ask for every invoice number, term and date the owner didn't
give. When the owner says not to ask, the rules now say so: make it now with plain defaults for a
number, a date and the terms, say which in one line, and still never make up a name, an address,
a price or a quantity.
"""
from __future__ import annotations

import contextlib
from types import SimpleNamespace as NS
from uuid import UUID

import pytest

from consumers.chatbot import named_template_note as note_module
from consumers.chatbot.named_template import NamedTemplate
from consumers.chatbot.named_template_note import NO_QUESTIONS_RULES, RULES, found_note, no_questions, read_note

WS = UUID("0b9f3c1e-2d4a-4b5c-9e8f-7a6b5c4d3e2f")
BRANDED = NamedTemplate(asked="Branded Invoice", row=NS(id=UUID("199420a1-87da-4018-8e3e-7012dda53f55"),
                                                         name="Branded Invoice"))
SCHEMA = {"data_fields": ["data.client_name", "data.invoice_number", "data.payment_terms", "data.due_date",
                          "data.line_items"], "required_fields": ["client_name"], "list_fields": []}
AT_18_31 = ("Can you make me an invoice as a PDF on my Branded Invoice: to Lantern Kitchen "
            "(accounts@lantern-kitchen.example), 12 kg of Harbour Blend at £22.00 per kg, carriage £5.00, no VAT. "
            "Quantity 12, unit price 22.00. Just make it, no questions.")


@pytest.mark.parametrize("said", [
    AT_18_31,
    "No bank details on it. Show tax as £0.00. Just make it.",                  # chat 76d4753a
    "Don't ask me anything, fill in the rest.",
    "Make the letter without asking, use your judgement.",
])
def test_the_owner_saying_not_to_ask_is_read(said):
    assert no_questions([said]) is True


@pytest.mark.parametrize("said", [
    "Can you make me an invoice on my Branded Invoice for Lantern Kitchen?",
    "Any questions before you start?",
    "Ask me if anything is missing.",
])
def test_an_ordinary_ask_keeps_the_rules(said):
    assert no_questions([said]) is False


def test_told_not_to_ask_the_rules_fill_defaults_and_say_which():
    note = found_note(BRANDED, SCHEMA, told_not_to_ask=True)

    assert RULES not in note and NO_QUESTIONS_RULES in note
    assert "don't ask, make the document now" in note
    assert "payment terms of 30 days and a due date 30 days after it" in note and "INV-YYYYMMDD" in note
    assert "Never make up a name, an address, a price or a quantity" in note
    assert "say in one line which values you filled in" in note
    assert "invoice_number" in note and "due_date" in note                     # the fields are still listed


def test_otherwise_the_rules_still_ask():
    assert RULES in found_note(BRANDED, SCHEMA) and NO_QUESTIONS_RULES not in found_note(BRANDED, SCHEMA)


class _ChatDb:
    def begin_nested(self):
        return contextlib.nullcontext()


@pytest.fixture
def branded_invoice(monkeypatch):
    from modules.documents import template_service

    class _Templates:
        def __init__(self, db):
            pass

        def list_templates(self, workspace_id):
            return [BRANDED.row]

    monkeypatch.setattr(template_service, "DocumentTemplateService", _Templates)
    monkeypatch.setattr(note_module, "designer_note", lambda *a, **k: None)
    monkeypatch.setattr(note_module, "named_in_conversation", lambda texts, rows: BRANDED)
    monkeypatch.setattr(note_module, "_schema", lambda db, workspace_id, row: SCHEMA)


def test_the_turn_at_18_31_gets_the_no_questions_rules(branded_invoice):
    assert NO_QUESTIONS_RULES in read_note(_ChatDb(), WS, [AT_18_31])


def test_a_follow_up_keeps_the_no_questions_said_before_it(branded_invoice):
    follow_up = "Invoice HL-W-1035, payment 30 days, due 5 November 2026."
    assert NO_QUESTIONS_RULES in read_note(_ChatDb(), WS, [follow_up, AT_18_31])
    assert RULES in read_note(_ChatDb(), WS, [follow_up])
