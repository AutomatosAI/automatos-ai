"""F351 (night 10b): asked in plain English for a document on a named template, Auto made a
usable one 1 time in 3. It invented values instead of asking (INV-0043, a blank Bill-to, "Net
30") and made "Harbourline Invoice" on the Branded Invoice, "which I assume is what you mean"
(e786217e, deliverable 557d64d0). With the template studio's "Use with Auto" prompt pasted first
it was 6 of 7. Now a turn that names a template gets that prompt's shape built for it: the exact
name and id, every field, each table's columns, and the rules; a name this workspace hasn't got
gets "not here" and the closest names, never a substitute.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID

import pytest

from consumers.chatbot.named_template import named_in, named_in_conversation
from consumers.chatbot.named_template_note import RULES, fills_the_named_template
from modules.documents.presets import INVOICE, REPORT

PASSAGES = "passages from wholesale-terms.md"
E786217E = ("Hi Auto. Please make an invoice for Lantern Kitchen on my own template called 'Harbourline Invoice'. "
            "Bill to Lantern Kitchen, 14 Wapping Quay, Bristol BS1 4RW. Lines: 6 x Kayon Mountain 1 kg at £31.00.")
LINE_ITEMS = "line_items is a list of rows, each with description, quantity, unit_price, total."


@pytest.fixture
def shop(db_session, seed_workspace):
    from core.models.core import DocumentTemplate

    def template(name, workspace, preset=INVOICE):
        row = DocumentTemplate(workspace_id=workspace, name=name, format="pdf", category=preset["category"],
                               blocks=preset["blocks"], data_schema={}, sample_data={}, created_by="owner")
        db_session.add(row)
        db_session.flush()
        return row

    ws, other = UUID(seed_workspace()), UUID(seed_workspace())
    rows = {name: template(name, ws) for name in ("Harbourline Invoice", "Branded Invoice")}
    rows["Basic Report"] = template("Basic Report", ws, REPORT)
    rows["Quayside Invoice"] = template("Quayside Invoice", other)                 # another workspace's
    return NS(db=db_session, ws=ws, other=other, rows=rows, template=template)


def _turn(shop, said, history=(), widget_mode=False, workspace=None):
    """``_retrieval_first`` wrapped, run for ``said`` after ``history``: the turn's messages."""
    chat = NS(db=shop.db, workspace_id=str(workspace or shop.ws), widget_mode=widget_mode)
    messages = [{"role": "system", "content": "You are Auto."}]
    for i, text in enumerate(history):
        messages += [{"role": "user", "content": text}, {"role": "assistant", "content": f"reply {i}"}]
    messages.append({"role": "user", "content": said})

    async def retrieval_first(self, latest_text, llm_messages, agent_runtime, chat_id, prefetched):
        llm_messages.append({"role": "system", "content": PASSAGES})
        yield "searched"

    async def run():
        return [f async for f in fills_the_named_template(retrieval_first)(chat, said, messages, None, "c", [])]

    assert asyncio.run(run()) == ["searched"]
    return messages


def _the_studio_prompt_for_harbourline(shop, note):
    harbourline = shop.rows["Harbourline Invoice"]
    assert f'"Harbourline Invoice" (template_id {harbourline.id})' in note
    assert f"call generate_document with template_id {harbourline.id}" in note
    assert "Use THIS template" in note and "Never make it on another template" in note
    for field in ("client_name", "client_address", "invoice_number", "due_date", "line_items", "total"):
        assert field in note
    assert LINE_ITEMS in note
    assert RULES in note and "never invent one" in note and "data as an object keyed by these field names" in note
    assert str(shop.rows["Branded Invoice"].id) not in note and "Branded Invoice" not in note


@pytest.mark.parametrize("said", [
    E786217E,                                                                          # in quotes
    "Make the Lantern Kitchen invoice on my harbourline invoice template please.",     # any case, no quotes
    'Use "HARBOURLINE INVOICE" for the Lantern Kitchen invoice.',                      # quoted, upper case
])
def test_a_named_template_gets_its_id_fields_and_columns_after_the_passages(shop, said):
    messages = _turn(shop, said)

    assert messages[-2]["content"] == PASSAGES
    assert messages[-1]["role"] == "system"
    _the_studio_prompt_for_harbourline(shop, messages[-1]["content"])


def test_a_template_named_by_its_id_gets_the_same(shop):
    harbourline = shop.rows["Harbourline Invoice"]

    messages = _turn(shop, f"Please generate a PDF from template_id {harbourline.id} for Lantern Kitchen.")

    _the_studio_prompt_for_harbourline(shop, messages[-1]["content"])


def test_a_name_no_template_has_gets_the_closest_names_and_no_substitute(shop):
    shop.db.delete(shop.rows["Harbourline Invoice"])
    shop.db.flush()

    note = _turn(shop, E786217E)[-1]["content"]

    assert '"Harbourline Invoice"' in note and "no template by that name" in note
    assert "Don't make the document on another template in its place" in note
    assert "The closest names:" in note and '"Branded Invoice"' in note
    assert "template_id" not in note and "Use THIS template" not in note
    assert str(shop.rows["Branded Invoice"].id) not in note


def test_a_message_naming_no_template_gets_nothing(shop):
    messages = _turn(shop, "How many bags of Kayon Mountain did Lantern Kitchen order in September?")

    assert messages[-1]["content"] == PASSAGES


def test_another_workspaces_template_is_never_matched(shop):
    quayside = shop.rows["Quayside Invoice"]

    by_name = _turn(shop, "Make it on my Quayside Invoice template.")[-1]["content"]
    by_id = _turn(shop, f"Use template_id {quayside.id} for the Lantern Kitchen invoice.")[-1]["content"]

    for note in (by_name, by_id):
        assert "no template by that name" in note and "Use THIS template" not in note
        assert f"template_id {quayside.id}" not in note
        assert "Quayside" not in note.split("The closest names:")[1]                # only this workspace's names


def test_a_follow_up_turn_keeps_the_template_named_earlier(shop):
    history = [E786217E]

    messages = _turn(shop, "Invoice HCR-2026-1072, due 4 November 2026, 30 days. Go ahead.", history)

    note = messages[-1]["content"]
    _the_studio_prompt_for_harbourline(shop, note)
    assert "earlier in this conversation" in note


def test_a_template_only_mentioned_later_never_replaces_the_one_asked_for(shop):
    history = [E786217E]

    note = _turn(shop, "The template has the same fields as the Branded Invoice. Go ahead.", history)[-1]["content"]

    _the_studio_prompt_for_harbourline(shop, note)


def test_a_widget_visitors_turn_gets_nothing(shop):
    messages = _turn(shop, E786217E, widget_mode=True)

    assert messages[-1]["content"] == PASSAGES


def test_a_looser_name_never_stands_in_for_the_one_the_owner_gave():
    rows = [NS(id="b0000000-0000-0000-0000-000000000001", name="Invoice"),
            NS(id="b0000000-0000-0000-0000-000000000002", name="Branded Invoice")]

    named = named_in("Make an Invoice for Lantern Kitchen on my 'Harbourline Invoice' template.", rows)

    assert named.row is None and named.asked == "Harbourline Invoice"
    assert set(named.closest) == {"Invoice", "Branded Invoice"}
    assert named_in("Use the plain Invoice template, not the Branded one.", rows).row is rows[0]
    assert named_in("Make a Branded Invoice for Gull & Kettle — 6 kg of Guji Uraga.", rows).row is rows[1]
    assert named_in("make an invoice for Gull & Kettle", rows) is None                # no template named
    assert named_in_conversation(["Thanks!", "Use the Branded Invoice template."], rows).earlier is True


def test_the_chat_runs_retrieval_first_through_it():
    from consumers.chatbot.service import StreamingChatService

    inner = StreamingChatService._retrieval_first
    for _ in range(4):                            # under PRD-256 US-001, F241, F307, F317 (FX-007: F303/F316/F324 gone)
        inner = inner.__wrapped__
    assert inner.__code__ is fills_the_named_template(lambda: None).__code__
