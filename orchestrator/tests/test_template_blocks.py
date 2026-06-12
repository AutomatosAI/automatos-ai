"""PRD-167 unit tests for the document-template block system.

Pure logic only — no DB, no WeasyPrint, no Docker. Covers the block schema +
validation, variable resolution, brand kit, the HTML renderer, and the legacy mapper.
"""

import pytest

from modules.documents.blocks import (
    BlockValidationError,
    blocks_from_legacy,
    collect_variable_paths,
    render_document_html,
    validate_blocks,
)
from modules.documents.brand_kit import BrandKit, get_brand_kit, validate_brand_kit
from modules.documents.variables.resolver import build_context, resolve_paths

from datetime import datetime

NOW = datetime(2026, 6, 12, 9, 30, 0)


# --------------------------------------------------------------------------- #
# Schema + validation
# --------------------------------------------------------------------------- #


def test_validate_valid_blocks():
    doc = validate_blocks(
        {
            "blocks": [
                {"type": "heading", "id": "h", "level": 2, "content": [{"type": "text", "text": "Hi"}]},
                {"type": "text", "id": "t", "content": [{"type": "variable", "path": "user.name"}]},
            ]
        }
    )
    assert len(doc.blocks) == 2
    assert doc.blocks[0].type == "heading"


def test_validate_accepts_bare_list():
    doc = validate_blocks([{"type": "page_break", "id": "pb"}])
    assert doc.blocks[0].type == "page_break"


def test_validate_malformed_field_level_errors():
    # level 9 is out of range, and an unknown key must be rejected (extra=forbid).
    with pytest.raises(BlockValidationError) as exc:
        validate_blocks({"blocks": [{"type": "heading", "id": "h", "level": 9, "content": []}]})
    locs = {e["loc"] for e in exc.value.errors}
    assert any("level" in loc for loc in locs)
    # discriminator tag stripped from loc
    assert all("heading" not in loc for loc in locs)


def test_validate_image_requires_src_unless_brand_logo():
    with pytest.raises(BlockValidationError):
        validate_blocks({"blocks": [{"type": "image", "id": "i", "source": "url"}]})
    # brand_logo needs no src
    doc = validate_blocks({"blocks": [{"type": "image", "id": "i", "source": "brand_logo", "alt": ""}]})
    assert doc.blocks[0].source == "brand_logo"


def test_collect_variable_paths_walks_tables_and_sections():
    doc = validate_blocks(
        {
            "blocks": [
                {"type": "section", "id": "s", "title": "S", "children": [
                    {"type": "text", "id": "t", "content": [{"type": "variable", "path": "company.name"}]},
                ]},
                {"type": "table", "id": "tb", "header": True, "rows": [
                    [[{"type": "variable", "path": "date.long"}], [{"type": "text", "text": "x"}]],
                ]},
                {"type": "variable", "id": "v", "path": "data.total"},
            ]
        }
    )
    assert collect_variable_paths(doc) == {"company.name", "date.long", "data.total"}


# --------------------------------------------------------------------------- #
# Variable resolution
# --------------------------------------------------------------------------- #


class _User:
    name = "Jane Doe"
    email = "jane@acme.com"
    username = "jane"


class _BP:
    company_name = "Acme Corp"
    domain = "acme.com"


def _context(brand=None, data=None):
    brand_kit = get_brand_kit({"brand_kit": brand} if brand else None)
    return build_context(_User(), _BP(), brand_kit, NOW, extra_data=data)


def test_build_context_splits_name_and_dates():
    ctx = _context()
    assert ctx["user"]["first_name"] == "Jane"
    assert ctx["user"]["last_name"] == "Doe"
    assert ctx["company"]["name"] == "Acme Corp"
    assert ctx["date"]["today"] == "2026-06-12"
    assert ctx["date"]["long"] == "June 12, 2026"
    assert ctx["date"]["year"] == "2026"


def test_resolve_paths_resolved_unresolved_unknown():
    ctx = _context()
    res = resolve_paths(ctx, ["user.name", "company.phone", "bogus.path"])
    assert res.values["user.name"] == "Jane Doe"
    assert "company.phone" in res.unresolved  # known but empty
    assert "bogus.path" in res.unknown        # not in catalog


def test_resolve_data_namespace_dynamic():
    ctx = _context(data={"total": "£2,200", "client": {"name": "Beta Ltd"}})
    res = resolve_paths(ctx, ["data.total", "data.client.name", "data.missing"])
    assert res.values["data.total"] == "£2,200"
    assert res.values["data.client.name"] == "Beta Ltd"
    assert "data.missing" in res.unresolved
    assert res.unknown == []  # data.* is valid even when empty


def test_brand_overrides_resolve():
    ctx = _context(brand={"name": "AcmeBrand", "primary_color": "#0055aa"})
    res = resolve_paths(ctx, ["brand.name", "brand.primary_color"])
    assert res.values["brand.name"] == "AcmeBrand"
    assert res.values["brand.primary_color"] == "#0055aa"


# --------------------------------------------------------------------------- #
# Brand kit
# --------------------------------------------------------------------------- #


def test_brand_kit_defaults_neutral_not_orange():
    kit = get_brand_kit(None)
    assert kit["primary_color"] == "#1a1a2e"
    assert "#ff6b35" not in kit.values()  # the old Automatos orange is gone


def test_brand_kit_hex_validation_rejects_garbage():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        validate_brand_kit({"primary_color": "not-a-color"})


def test_brand_kit_merge_preserves_existing():
    existing = validate_brand_kit({"name": "Acme", "primary_color": "#111111"})
    merged = validate_brand_kit({"tagline": "Go"}, existing)
    assert merged["name"] == "Acme"
    assert merged["primary_color"] == "#111111"
    assert merged["tagline"] == "Go"


# --------------------------------------------------------------------------- #
# HTML renderer
# --------------------------------------------------------------------------- #


def test_html_render_resolved_branded_and_no_orange():
    doc = validate_blocks(
        {"blocks": [
            {"type": "heading", "id": "h", "level": 1, "content": [{"type": "variable", "path": "data.title"}]},
            {"type": "text", "id": "t", "content": [
                {"type": "text", "text": "By "}, {"type": "variable", "path": "user.name"},
            ]},
        ]}
    )
    brand = get_brand_kit({"brand_kit": {"primary_color": "#0055aa"}})
    rendered = render_document_html(doc, {"data.title": "Q2", "user.name": "Jane Doe"}, brand, title="Q2")
    assert "Q2" in rendered.html
    assert "Jane Doe" in rendered.html
    assert "#0055aa" in rendered.html
    assert "#ff6b35" not in rendered.html
    assert rendered.unresolved == []


def test_html_render_escapes_text():
    doc = validate_blocks({"blocks": [{"type": "text", "id": "t", "content": [{"type": "text", "text": "<script>x</script>"}]}]})
    rendered = render_document_html(doc, {}, get_brand_kit(None), title="x")
    assert "<script>x</script>" not in rendered.html
    assert "&lt;script&gt;" in rendered.html


def test_html_render_unresolved_marker_and_list():
    doc = validate_blocks({"blocks": [{"type": "text", "id": "t", "content": [{"type": "variable", "path": "data.missing"}]}]})
    rendered = render_document_html(doc, {}, get_brand_kit(None), title="x")
    assert "[[data.missing]]" in rendered.html
    assert "data.missing" in rendered.unresolved


def test_html_render_fallback_used_when_present():
    doc = validate_blocks({"blocks": [{"type": "text", "id": "t", "content": [{"type": "variable", "path": "user.name", "fallback": "Friend"}]}]})
    rendered = render_document_html(doc, {}, get_brand_kit(None), title="x")
    assert "Friend" in rendered.html
    assert rendered.unresolved == []  # fallback satisfies it


def test_brand_logo_unresolved_when_no_logo():
    doc = validate_blocks({"blocks": [{"type": "image", "id": "i", "source": "brand_logo", "alt": "Logo"}]})
    rendered = render_document_html(doc, {}, get_brand_kit(None), title="x")
    assert "brand.logo_url" in rendered.unresolved


# --------------------------------------------------------------------------- #
# Legacy mapper
# --------------------------------------------------------------------------- #


def test_legacy_mapper_builds_blocks():
    doc = blocks_from_legacy(
        {
            "title": "Monthly Report",
            "author": "Auto",
            "sections": [{"title": "Overview", "content": "All good."}],
            "metrics": {"Revenue": "$1M"},
            "highlights": ["Up 10%"],
        }
    )
    types = [b.type for b in doc.blocks]
    assert types[0] == "heading"  # title
    assert "table" in types        # metrics
    # renders without error
    rendered = render_document_html(doc, {}, get_brand_kit(None), title="Monthly Report")
    assert "Monthly Report" in rendered.html
    assert "Revenue" in rendered.html
