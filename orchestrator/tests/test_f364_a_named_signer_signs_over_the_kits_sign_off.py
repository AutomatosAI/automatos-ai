"""F364 (night 10c): a letter "from me, Gerard" is signed Gerard, and a sign-off card says what it changes.

Chat cb93b9e3: "Sign it from me, Gerard" came back signed "Automatos AI", the kit's
sign-off, which the owner had approved on a designer card that never said it changes
every letter's signature. The Branded Letter signs with ``brand.sign_off``, which had
no way to take a named person. Now a document's data may name its signer
(``data.signer``): it signs in place of the kit's sign-off, the template lists it as a
field the agent may send, Auto's brand rules say to send it, and a sign-off change on
a proposal card says it changes the signature on every letter.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from modules.documents.blocks import collect_variable_paths, render_document_html, validate_blocks
from modules.documents.brand_kit import get_brand_kit
from modules.documents.brand_signing import with_signer
from modules.documents.data_coverage import unused_data_keys
from modules.documents.field_requirements import block_requirements
from modules.documents.presets import LETTER
from modules.documents.variables.resolver import build_context, resolve_paths
from modules.tools.discovery.brand_proposal_card import card_text, changes
from services.brand_rules import rules_for_kit

KIT = get_brand_kit({"brand_kit": {"name": "Automatos AI", "company": {"name": "Automatos AI"},
                                   "voice": {"sign_off": "Automatos AI"}}})
QUAY = {
    "recipient_name": "Sam Okafor", "subject": "Our wholesale payment terms from 1 November",
    "greeting": "Dear Sam,", "body": "From 1 November wholesale invoices move to 30-day terms.",
}


def _sign_off(data):
    context = build_context(None, None, KIT, datetime(2026, 10, 6), data)
    return resolve_paths(context, ["brand.sign_off"]).values["brand.sign_off"]


def _letter(data) -> str:
    doc = validate_blocks(LETTER["blocks"])
    context = build_context(SimpleNamespace(name="Gerard Kavanagh", email="gerard@example.com", username="g"),
                            None, KIT, datetime(2026, 10, 6), data)
    values = resolve_paths(context, collect_variable_paths(doc)).values
    return render_document_html(doc, values, KIT, data=data).html


def test_a_named_signer_signs_in_place_of_the_kits_sign_off():
    assert _sign_off({**QUAY, "signer": "Gerard"}) == "Gerard"
    assert _sign_off(QUAY) == "Automatos AI"  # no signer named: the kit's sign-off, as before
    assert _sign_off({**QUAY, "signer": "   "}) == "Automatos AI"


def test_the_branded_letter_is_signed_by_the_named_signer():
    page = _letter({**QUAY, "signer": "Gerard"})

    assert '<p data-block="sig-name">Gerard</p>' in page
    assert "Automatos AI" not in page.split('data-block="sig-name"')[1].split("</p>")[0]


def test_the_letter_lists_signer_as_a_field_and_never_calls_it_unused():
    assert "data.signer" in block_requirements(LETTER["blocks"])["fallback_fields"]
    template = SimpleNamespace(blocks=LETTER["blocks"], template_content=None)
    assert unused_data_keys(template, {**QUAY, "signer": "Gerard"}, "pdf") == []
    assert "signer" in LETTER["description"]


def test_a_placeholder_signature_takes_the_named_signer_too():
    signed = with_signer(KIT, {"signer": "Gerard"})

    assert signed["voice"]["sign_off"] == "Gerard" and KIT["voice"]["sign_off"] == "Automatos AI"
    assert with_signer(KIT, QUAY) is KIT


def test_autos_brand_rules_say_a_named_signer_wins():
    rules = rules_for_kit(KIT)
    assert 'Sign it "Automatos AI", unless the person names who signs' in rules
    assert "signer" in rules


def test_a_sign_off_change_on_a_proposal_card_says_it_changes_every_letter():
    changed = changes({"voice": {"sign_off": ""}}, {"voice": {"sign_off": "Automatos AI"}}, ["voice"])
    text = card_text("the Brand designer", "Sign as the company.", changed, "brand-board.png")

    assert "- `voice.sign_off`: (none) → `Automatos AI`: this changes the signature on every letter" in text
    other = card_text("the Brand designer", "", changes({"accent_use": "sparing"}, {"accent_use": "bold"},
                                                        ["accent_use"]), "b.png")
    assert "every letter" not in other
