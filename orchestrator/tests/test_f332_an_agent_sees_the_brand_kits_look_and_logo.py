"""F332 (night 10): an agent is given the brand kit's look, not only its voice.

Night 10: session agents had the "## The brand's rules" block (who to write as, tone,
sign-off, banned words) and nothing of the look. Three different navies were guessed in
one night; the BA could not read the uploaded logo, the Analyst "couldn't reach the
Brand kit page", Support had "no logo file"; and Support kept an old sign-off because the
brand-voice document said so. The block now carries the kit's colours (hex), its fonts,
the company's contact details and where the logo is, each only when the kit sets it, and
says the kit wins over any document that says otherwise. A session's claim carries its
own workspace's uploaded logo and logo mark, which the host writes into the ticket folder;
another workspace's logo is never sent.
"""
from __future__ import annotations

import base64
from types import SimpleNamespace as NS
from uuid import UUID, uuid4

import pytest

import modules.documents.brand_logo as bl
from core.models.workspaces import Workspace
from tests import test_brandkit_a_every_drafting_run_is_told_the_brands_rules as kit_a
from tests import test_f249_a_rejects_lesson_reaches_the_agents_next_card as f249a
from tests import test_prd242_brand_logo as logo_tests
from tests import test_prd245_session_token_realdb as t245

engine = f249a.engine
shop = f249a.shop
fresh_kits = kit_a.fresh_kits
ticket = t245.ticket                          # a runtime: cli agent's ticket, swept with its host
quiet_claims = t245._quiet_side_effects      # no approval lookup, no completion fan-out

KIT_WINS = "- The brand kit wins over any document that says otherwise."
KIT = {
    "name": "Harbourline Coffee Roasters",
    "primary_color": "#0b2545", "secondary_color": "#13315c", "accent_color": "#e0a458", "text_color": "#1b1b1b",
    "font_family": "Source Sans 3, sans-serif", "heading_font": "'Playfair Display', serif",
    "company": {"name": "Harbourline Coffee Roasters Ltd", "address": "12 Quay Street,\nCork T12 X2Y3",
                "phone": "+353 21 400 1234", "email": "hello@harbourline.ie", "website": "https://harbourline.ie"},
    "voice": {"tone": ["warm", "plain", "local"], "sign_off": "Gerard, Harbourline Coffee Roasters"},
}
THE_LOOK = (
    "- Colours (hex): primary #0b2545, secondary #13315c, accent #e0a458, text #1b1b1b.",
    "- Fonts: Playfair Display for headings, Source Sans 3 for body text.",
    "- Company: Harbourline Coffee Roasters Ltd. Address: 12 Quay Street, Cork T12 X2Y3. "
    "Phone: +353 21 400 1234. Email: hello@harbourline.ie. Website: https://harbourline.ie.",
)


def _rules(kit):
    from services.brand_rules import rules_for_kit

    return rules_for_kit(kit)


def test_the_rules_carry_the_colours_fonts_contact_details_and_that_the_kit_wins():
    block = _rules(KIT)
    for line in THE_LOOK:
        assert line in block
    assert block.endswith(KIT_WINS)
    # Every rule is a "- " line, so an answer that repeats its prompt repeats one whole block.
    assert all(line.startswith("- ") for line in block.split("\n")[2:])


def test_only_what_the_kit_sets_is_said():
    from modules.documents.brand_kit import get_brand_kit

    voice_only = get_brand_kit({"brand_kit": {"voice": {"tone": ["warm", "plain", "local"]}}})
    block = _rules(voice_only)       # the neutral default palette and font are not the brand's
    assert "Colours" not in block and "Font" not in block and "Company" not in block and "Logo" not in block
    assert block.endswith(KIT_WINS)
    partial = _rules({**voice_only, "accent_color": "#e0a458", "company": {"phone": "+353 21 400 1234"}})
    assert "- Colours (hex): accent #e0a458." in partial
    assert "- Company details: Phone: +353 21 400 1234." in partial
    assert _rules(get_brand_kit({})) is None


def test_the_rules_say_where_the_logo_is():
    uploaded = _rules({**KIT, "logo_path": "ws/brand/logo.png"})
    assert "- Logo: the one uploaded to the brand kit." in uploaded and "your ticket file lists its copy" in uploaded
    linked = _rules({**KIT, "logo_url": "https://harbourline.ie/logo.png"})
    assert "- Logo: https://harbourline.ie/logo.png" in linked


def test_a_sessions_ticket_carries_the_look(shop):
    kit_a._with_kit(shop, KIT)
    card = f249a._ticket(shop, "Content Creator", "Letter to Tidewater")
    for prompt in (f249a._session_prompt(shop, card), f249a._dispatch_prompt(shop, card)):
        for line in (*THE_LOOK, KIT_WINS):
            assert line in prompt
        assert prompt.count("## The brand's rules") == 1


# ── the logo file a session gets ────────────────────────────────────────────────

@pytest.fixture
def storage(tmp_path, monkeypatch):
    monkeypatch.setattr(bl.config, "DOCUMENT_STORAGE_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(bl, "is_storage_configured", lambda: False)
    return tmp_path


def _db(kits):
    """``db.get(Workspace, id)`` → that workspace's settings, as ``stored_kit`` reads them."""
    return NS(get=lambda model, key: NS(settings={"brand_kit": kits[str(key)]})
              if model is Workspace and str(key) in kits else None)


def test_a_session_gets_its_own_workspaces_logo_and_never_another_workspaces(storage):
    from services.session_brand_files import session_brand_files

    ours, theirs = uuid4(), uuid4()
    our_logo, their_logo = logo_tests.png_bytes(120, 40), logo_tests.png_bytes(300, 90)
    our_mark = logo_tests.jpeg_bytes(64, 64)
    kits = {str(ours): {**KIT, "logo_path": bl.save_brand_logo(ours, our_logo),
                        "logo_mark_path": bl.save_brand_logo_mark(ours, our_mark)},
            str(theirs): {**KIT, "logo_path": bl.save_brand_logo(theirs, their_logo)}}
    files = {f["name"]: f for f in session_brand_files(_db(kits), ours)}
    assert sorted(files) == ["logo-mark.jpg", "logo.png"]
    assert base64.b64decode(files["logo.png"]["data"]) == our_logo
    assert base64.b64decode(files["logo-mark.jpg"]["data"]) == our_mark and files["logo.png"]["mime"] == "image/png"
    # A kit that names another workspace's stored logo is never sent it.
    kits[str(ours)] = {**KIT, "logo_path": kits[str(theirs)]["logo_path"]}
    from services.brand_rules import forget_cached_kits

    forget_cached_kits()
    assert session_brand_files(_db(kits), ours) == []
    assert [base64.b64decode(f["data"]) for f in session_brand_files(_db(kits), theirs)] == [their_logo]


def test_no_kit_or_no_upload_sends_nothing(storage):
    from services.session_brand_files import session_brand_files

    ws = uuid4()
    assert session_brand_files(_db({}), ws) == []
    assert session_brand_files(_db({str(ws): {**KIT, "logo_url": "https://harbourline.ie/logo.png"}}), ws) == []


def test_the_claim_carries_the_tickets_own_logo(ticket, new_session, storage):
    from services import cli_host_service as svc

    ws_id, _agent_id, task_id = ticket
    logo = logo_tests.png_bytes(200, 60)
    kit_a._with_kit(NS(new=new_session, ws=ws_id), {**KIT, "logo_path": bl.save_brand_logo(UUID(ws_id), logo)})
    s = new_session()
    claimed = svc.claim_for_host(s, t245._host(s, ws_id), limit=1)["tasks"]
    assert [c["task_id"] for c in claimed] == [task_id]
    assert [(f["name"], base64.b64decode(f["data"])) for f in claimed[0]["brand_files"]] == [("logo.png", logo)]
