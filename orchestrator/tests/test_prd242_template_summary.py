"""PRD-242 S2 — the gallery entry says what a raw template row does not.

``has_blocks`` (block-editable vs legacy), ``is_starter`` (seeded, copy-on-
customise) and ``data_fields`` (the ``data.*`` chips an agent must supply).
Pure — plain objects in, dicts out.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from modules.documents.template_summary import data_fields_of, summarize_template, variable_paths_of


def _blocks(*paths):
    return {
        "version": 1,
        "blocks": [
            {"type": "text", "id": f"t{i}", "content": [{"type": "variable", "path": p}]}
            for i, p in enumerate(paths)
        ],
    }


def test_variable_paths_sorted_and_deduplicated():
    paths = variable_paths_of(_blocks("data.summary", "user.name", "data.summary", "brand.name"))
    assert paths == ["brand.name", "data.summary", "user.name"]


def test_data_fields_strip_the_dynamic_prefix_and_keep_order():
    assert data_fields_of(["brand.name", "data.summary", "data.title", "user.name"]) == ["summary", "title"]


def test_no_blocks_or_malformed_blocks_yield_empty_lists():
    assert variable_paths_of(None) == []
    assert variable_paths_of({}) == []
    # malformed body (level 9) → not an authoring surface we can read; no crash
    assert variable_paths_of({"blocks": [{"type": "heading", "id": "h", "level": 9, "content": []}]}) == []


def test_summarize_block_starter():
    row = SimpleNamespace(
        id="11111111-1111-1111-1111-111111111111",
        name="Branded Report",
        description="desc",
        format="pdf",
        category="report",
        tags=None,
        version=1,
        data_schema={},
        sample_data={"data": {"title": "Q1"}},
        blocks=_blocks("data.title", "data.summary", "brand.name"),
        created_by="system",
        created_at=datetime(2026, 9, 11, 10, 0, 0),
        updated_at=None,
    )
    out = summarize_template(row)
    assert out["has_blocks"] is True
    assert out["is_starter"] is True
    assert out["data_fields"] == ["summary", "title"]
    assert out["variable_paths"] == ["brand.name", "data.summary", "data.title"]
    assert out["created_at"] == "2026-09-11T10:00:00"
    assert out["updated_at"] is None
    assert out["tags"] == []


def test_summarize_legacy_user_template():
    row = SimpleNamespace(
        id="x", name="Invoice", description=None, format="pdf", category="invoice", tags=["a"],
        version=2, data_schema={"type": "object"}, sample_data={}, blocks=None,
        created_by="user_abc", created_at=None, updated_at=None,
    )
    out = summarize_template(row)
    assert out["has_blocks"] is False
    assert out["is_starter"] is False
    assert out["data_fields"] == []
    assert out["tags"] == ["a"]
