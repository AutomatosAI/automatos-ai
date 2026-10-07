"""PRD-255 Wave 1, US-008 — agents and Auto see and change the whole kit.

An owner tells Auto "make the orange an accent only" or "more space between sections"
and the kit changes. Pins:

* **Round trip**: every field ``platform_update_brand_kit`` takes (each of the kit's
  patch fields, the v2 ones included) is saved through the tool and read back by
  ``platform_get_brand_kit``, which marks each colour role ``set`` or ``derived``.
* **The rules block** (``services.brand_rules.rules_for_kit``) every drafting run is
  given carries the roles with their hex and source, the accent's use, a one-line type
  scale, the spacing and margin, the logo's size and uploaded variants (by the file
  name a session holds them under), the currency (only when the kit has one) and the
  date style. A kit of neutral defaults still says nothing of colour (F332).
* **The session's files**: a session gets the uploaded variants as files beside the
  logo (the F332 pattern), under the names the rules block gives.
"""
from __future__ import annotations

import base64
from types import SimpleNamespace as NS
from uuid import uuid4

import pytest

import modules.documents.brand_logo as bl
from core.brand_palette import PALETTE_ROLES, ROLE_DERIVED, ROLE_SET, effective_palette
from core.models.workspaces import Workspace
from modules.documents import brand_kit
from modules.documents.brand_system import DATE_STYLE_MONTH_FIRST
from services.brand_rules import KIT_WINS_LINE, forget_cached_kits, on_brand_text, rules_for_kit
from tests import test_prd242_brand_logo as logo_tests
from tests import test_prd251w1_brand_kit_tools as t251

api = t251.api                # the documents router and the platform tools over one workspace
_dispatch = t251._dispatch
KIT_ROUTE = t251.KIT_ROUTE

# Every field the update tool takes, each with a value the stored kit does not have.
EVERY_FIELD = {
    "name": "Harbourline Coffee Roasters",
    "tagline": "Roasted on the quay",
    "logo_url": "https://harbourline.ie/logo.png",
    "primary_color": "#0b2545",
    "secondary_color": "#13315c",
    "text_color": "#1b1b1b",
    "font_family": "Source Sans 3, sans-serif",
    "company": {"phone": "+353 21 400 1234"},
    "heading_font": "'Playfair Display', serif",
    "logo_mark_url": "https://harbourline.ie/mark.png",
    "social_handles": {"instagram": "harbourline"},
    "voice": {"tone": [{"word": "warm", "meaning": "friendly, never gushing"}, {"word": "plain", "meaning": ""},
                       {"word": "local", "meaning": ""}],
              "banned_phrases": ["exquisite"], "sign_off": "Gerard, Harbourline"},
    "palette": {"accent": "#8a3b12", "heading": "#111111"},
    "palette_source": {"accent": "set", "heading": "set"},
    "accent_use": "bold",
    "type_scale": {"h1": {"size_pt": 24, "line_pt": 30}},
    "spacing_unit_pt": 6,
    "page_margin_mm": 20,
    "logo_rules": {"letterhead_mm": 20},
    "currency": "eur",
    "date_style": DATE_STYLE_MONTH_FIRST,
    "country": "IE",
}

LOOK_KIT = {
    "name": "Harbourline Coffee Roasters",
    "primary_color": "#0b2545", "secondary_color": "#13315c", "text_color": "#1b1b1b",
    "palette": {"heading": "#111111"},
    "currency": "GBP",
    "date_style": DATE_STYLE_MONTH_FIRST,
    "logo_path": "ws/brand/logo.png",
    "logo_dark_path": "ws/brand/logo-dark.png",
    "logo_mono_path": "ws/brand/logo-mono.jpg",
    "voice": {"tone": ["warm", "plain", "local"], "sign_off": "Gerard"},
}
DEFAULT_TYPE_LINE = "- Type scale (pt, size/line): display 32/38 · h1 22/28 · h2 15/20 · h3 12/16 · body 10/15 · small 8.5/12 · caption 7.5/10."


def _kit(stored):
    """``stored`` as the platform reads it: every default filled in."""
    return brand_kit.get_brand_kit({"brand_kit": stored})


# ---------------------------------------------------------------------------
# Round trip through the tools
# ---------------------------------------------------------------------------


def test_every_field_the_tool_takes_round_trips_through_the_get_tool(api):
    assert set(EVERY_FIELD) == set(brand_kit.PATCH_FIELDS)  # the patch below is every field
    result = _dispatch(api.db, "platform_update_brand_kit", EVERY_FIELD)
    assert result["success"] is True, result
    assert result["changed"] == sorted(EVERY_FIELD)

    read = _dispatch(api.db, "platform_get_brand_kit", {})
    assert read["success"] is True, read
    kit = read["brand_kit"]
    assert kit == result["brand_kit"] == api.client.get(KIT_ROUTE).json()
    for field in ("name", "tagline", "logo_url", "primary_color", "secondary_color",
                  "text_color", "font_family", "heading_font", "logo_mark_url", "accent_use", "date_style",
                  "country"):
        assert kit[field] == EVERY_FIELD[field], field
    assert kit["company"]["phone"] == "+353 21 400 1234"
    assert kit["social_handles"] == {"instagram": "harbourline"}
    assert kit["voice"] == EVERY_FIELD["voice"]
    assert (kit["palette"]["accent"], kit["palette"]["heading"]) == ("#8a3b12", "#111111")
    assert kit["type_scale"]["h1"] == {"size_pt": 24.0, "line_pt": 30.0, "weight": 600}
    assert kit["type_scale"]["body"] == {"size_pt": 10.0, "line_pt": 15.0, "weight": 400}  # a step not sent keeps its value
    assert (kit["spacing_unit_pt"], kit["page_margin_mm"]) == (6.0, 20.0)
    assert kit["logo_rules"] == {"letterhead_mm": 20.0, "clear_space": 0.5, "min_mm": 8.0}
    assert kit["currency"] == "EUR"


def test_the_get_tool_marks_each_role_set_or_derived(api):
    read = _dispatch(api.db, "platform_get_brand_kit", {})["brand_kit"]
    assert set(read["palette"]) >= {"ink", "heading", "paper", "surface", "surface_2", "accent", "muted", "rule"}
    assert set(read["palette_source"].values()) == {ROLE_DERIVED}  # a v1 kit: every role derived

    _dispatch(api.db, "platform_update_brand_kit", {"palette": {"accent": "#8a3b12"}})
    read = _dispatch(api.db, "platform_get_brand_kit", {})["brand_kit"]
    assert read["palette"]["accent"] == "#8a3b12" and read["palette_source"]["accent"] == ROLE_SET
    assert {role for role, source in read["palette_source"].items() if source == ROLE_SET} == {"accent"}

    # "Back to derived": a role sent empty.
    _dispatch(api.db, "platform_update_brand_kit", {"palette": {"accent": ""}})
    assert _dispatch(api.db, "platform_get_brand_kit", {})["brand_kit"]["palette_source"]["accent"] == ROLE_DERIVED


def test_the_tool_never_sets_a_logo_variants_file(api):
    before = dict(api.workspace.settings["brand_kit"])
    for field in ("logo_dark_path", "logo_mono_path"):
        result = _dispatch(api.db, "platform_update_brand_kit", {field: "other-ws/brand/logo-dark.png"})
        assert result["success"] is False and field in result["error"]
    assert api.workspace.settings["brand_kit"] == before


def test_an_unreadable_role_is_refused_with_its_ratio_and_nothing_is_saved(api):
    before = dict(api.workspace.settings["brand_kit"])
    result = _dispatch(api.db, "platform_update_brand_kit", {"palette": {"ink": "#eeeeee"}})
    # F366: the refusal names what the colour sits on in plain words, beside the role.
    assert result["success"] is False and "ink is" in result["error"] and "on the page" in result["error"]
    assert ":1" in result["error"]
    assert api.workspace.settings["brand_kit"] == before


def test_the_descriptions_map_plain_words_to_the_fields():
    from modules.tools.discovery.action_registry import get_action_registry

    write = get_action_registry().get("platform_update_brand_kit")
    properties = write.parameters["properties"]
    assert "Less orange" in properties["accent_use"]["description"]
    assert "More space" in properties["spacing_unit_pt"]["description"]
    assert {"make the orange an accent only", "more space between sections"} <= set(write.examples)
    read = get_action_registry().get("platform_get_brand_kit")
    assert "palette_source" in read.description and "type scale" in read.description


# ---------------------------------------------------------------------------
# The rules block
# ---------------------------------------------------------------------------


def test_the_rules_block_lists_every_role_with_its_hex_and_source():
    kit = _kit(LOOK_KIT)
    block = rules_for_kit(kit)
    roles, sources = effective_palette(kit)
    roles_line = next(line for line in block.split("\n") if line.startswith("- Colour roles"))
    for role in roles:
        assert f"{role} {roles[role]} ({sources[role]})" in roles_line, role
    assert "heading #111111 (set)" in roles_line and "ink " in roles_line and "surface_2 " in roles_line
    assert set(roles) <= set(PALETTE_ROLES)
    assert "- Accent use: sparing: the accent only for highlights" in block


def test_the_rules_block_carries_the_type_scale_spacing_currency_dates_and_logo_variants():
    block = rules_for_kit(_kit(LOOK_KIT))
    lines = block.split("\n")
    assert DEFAULT_TYPE_LINE in lines
    assert "- Spacing: a 4 pt grid (every space a multiple of 4 pt); page margins 18 mm." in lines
    assert "- Currency: GBP (£). Print every amount in it, never another currency." in lines
    assert "- Dates: written as October 5, 2026." in lines
    assert "- Logo size: 16 mm high on a letterhead, never under 8 mm, with clear space of 0.5 times its height all round." in lines
    variants = next(line for line in lines if line.startswith("- Logo variants"))
    assert "brand/logo-dark.png (for dark backgrounds), brand/logo-mono.jpg (one colour)" in variants
    assert all(line.startswith("- ") for line in lines[2:]) and block.endswith(KIT_WINS_LINE)


def test_the_rules_block_follows_what_the_owner_changed():
    kit = _kit({**LOOK_KIT, "accent_use": "bold", "type_scale": {"h1": {"size_pt": 24, "line_pt": 30}},
                "spacing_unit_pt": 6, "page_margin_mm": 20, "logo_rules": {"letterhead_mm": 20, "min_mm": 10},
                "date_style": "d MMMM yyyy"})
    block = rules_for_kit(kit)
    assert "- Accent use: bold: the accent for highlights and table header fills" in block
    assert "· h1 24/30 ·" in block
    assert "- Spacing: a 6 pt grid (every space a multiple of 6 pt); page margins 20 mm." in block
    assert "- Logo size: 20 mm high on a letterhead, never under 10 mm" in block
    assert "- Dates: written as 5 October 2026." in block


def test_a_kit_of_neutral_defaults_says_nothing_of_the_look_and_no_currency_is_invented():
    voice_only = _kit({"voice": {"tone": ["warm", "plain", "local"]}})
    block = rules_for_kit(voice_only)
    for absent in ("Colour roles", "Accent use", "Type scale", "Spacing", "Currency", "Logo"):
        assert absent not in block, absent
    assert "- Dates: written as 5 October 2026." in block and block.endswith(KIT_WINS_LINE)
    assert rules_for_kit(_kit({})) is None
    # The owner's accent choice alone is a look: the roles it applies to are said.
    assert "- Colour roles" in rules_for_kit(_kit({**voice_only, "accent_use": "bold"}))
    assert "Currency" not in rules_for_kit(_kit({**LOOK_KIT, "currency": ""}))


def test_a_currency_without_a_symbol_is_named_once():
    assert "- Currency: CHF. Print every amount in it" in rules_for_kit(_kit({**LOOK_KIT, "currency": "CHF"}))


def test_an_answer_that_repeats_the_whole_block_keeps_it_as_it_is():
    kit = _kit(LOOK_KIT)
    block = rules_for_kit(kit)
    out = on_brand_text(f"Write the club note.\n\n{block}\n\nBest,\n[Your name]", kit)
    assert block in out and out.endswith("Best,\nGerard")


# ---------------------------------------------------------------------------
# The session's files
# ---------------------------------------------------------------------------


@pytest.fixture
def storage(tmp_path, monkeypatch):
    monkeypatch.setattr(bl.config, "DOCUMENT_STORAGE_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(bl, "is_storage_configured", lambda: False)
    forget_cached_kits()
    yield tmp_path
    forget_cached_kits()


def test_a_session_gets_the_uploaded_logo_variants_under_the_names_the_rules_give(storage):
    from services.session_brand_files import session_brand_files

    ws = uuid4()
    dark, mono = logo_tests.png_bytes(160, 50), logo_tests.jpeg_bytes(150, 40)
    kit = {**LOOK_KIT, "logo_path": "",
           "logo_dark_path": bl.save_brand_logo(ws, dark, stem=bl.LOGO_DARK_STEM),
           "logo_mono_path": bl.save_brand_logo(ws, mono, stem=bl.LOGO_MONO_STEM)}
    db = NS(get=lambda model, key: NS(settings={"brand_kit": kit}) if model is Workspace and key == ws else None)
    files = {f["name"]: base64.b64decode(f["data"]) for f in session_brand_files(db, ws)}
    assert files == {"logo-dark.png": dark, "logo-mono.jpg": mono}
    variants = next(line for line in rules_for_kit(_kit(kit)).split("\n") if line.startswith("- Logo variants"))
    assert all(f"brand/{name}" in variants for name in files)


def test_a_stored_path_that_climbs_out_of_the_brand_folder_is_never_read(storage):
    from services.session_brand_files import session_brand_files

    ws, other = uuid4(), uuid4()
    theirs = bl.save_brand_logo(other, logo_tests.png_bytes(160, 50), stem=bl.LOGO_DARK_STEM)
    kit = {"logo_dark_path": f"{ws}/brand/../../{theirs}"}
    db = NS(get=lambda model, key: NS(settings={"brand_kit": kit}) if model is Workspace and key == ws else None)
    assert session_brand_files(db, ws) == []
