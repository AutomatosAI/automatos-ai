"""P256-FIX-N15-5 (F426): Auto finds the work on the owner's review queue by its name.

Night 15: "show me the Guji blog draft" got "there isn't a Guji blog draft", and "did
the Westmoor invoice come back?" got "I can't find a Westmoor invoice"; with the ticket
number both came back at once. platform_list_tasks had no search, and the Deliverables
search matched the whole phrase. Now both match every word that names the work: a card
by its title or a Deliverable on it (title or file name), a Deliverable by its title,
summary or file path.
"""
from __future__ import annotations

import asyncio
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

from services.named_work import search_patterns, search_words


@pytest.mark.parametrize("raw, words", [
    ("the Guji blog draft", ["guji", "blog"]),
    ("Westmoor invoice", ["westmoor", "invoice"]),
    ("my quote for Hale & Co", ["quote", "hale", "co"]),
    ("Guji's blog post", ["guji", "blog", "post"]),
    ("the draft", ["draft"]),                  # a stage word alone is still searched
    ("the", ["the"]),                          # filler alone never widens to everything
    ("", []), (None, []), (42, []),
])
def test_the_words_that_name_the_work(raw, words):
    assert search_words(raw) == words


def test_a_pattern_never_carries_a_wildcard_from_the_ask():
    assert search_patterns("50%_off promo") == ["%50%", "%off%", "%promo%"]


def _card(db, ws, title, status="review"):
    from core.models.core import BoardTask

    card = BoardTask(workspace_id=ws, title=title, status=status, priority="medium")
    db.add(card)
    db.flush()
    return card


def _deliverable(db, ws, card, title, path, deleted=False):
    db.execute(text(
        "INSERT INTO deliverables (workspace_id, source_type, source_id, artifact_type, title, file_path, deleted_at) "
        "VALUES (CAST(:ws AS uuid), 'task', :card, 'document', :title, :path, "
        "CASE WHEN :deleted THEN now() ELSE NULL END)"),
        {"ws": str(ws), "card": str(card.id), "title": title, "path": path, "deleted": deleted})
    db.flush()


def _search(db, ws, search, **params):
    from modules.tools.discovery.board_task_list import list_board_tasks

    return asyncio.run(list_board_tasks(db, ws, {"search": search, **params}))


def _ids(result):
    return {t["id"] for t in result["tasks"]}


def test_a_review_card_titled_guji_blog_draft_is_found_by_the_guji_blog_draft(db_session, seed_workspace):
    ws, other = UUID(seed_workspace()), UUID(seed_workspace())
    guji = _card(db_session, ws, "Guji blog draft")
    _card(db_session, ws, "Westmoor quote")
    _card(db_session, other, "Guji blog draft")           # another workspace: never seen

    found = _search(db_session, ws, "the Guji blog draft", status="review")

    assert found["success"] is True
    assert _ids(found) == {guji.id} and found["total"] == 1
    assert _ids(_search(db_session, ws, "blog GUJI")) == {guji.id}   # any order, any case


def test_a_card_is_found_by_its_deliverables_title_or_file_name(db_session, seed_workspace):
    ws = UUID(seed_workspace())
    billed = _card(db_session, ws, "Bill the client for March")
    _deliverable(db_session, ws, billed, "March statement", f"invoices/westmoor-invoice-{uuid4().hex}.pdf")
    posted = _card(db_session, ws, "Write this week's post")
    _deliverable(db_session, ws, posted, "Guji: five ways to brew", f"blog/{uuid4().hex}.md")
    dropped = _card(db_session, ws, "Old work")
    _deliverable(db_session, ws, dropped, "Westmoor invoice", "invoices/old.pdf", deleted=True)

    assert _ids(_search(db_session, ws, "the Westmoor invoice")) == {billed.id}
    assert _ids(_search(db_session, ws, "Guji brew")) == {posted.id}


def test_a_widget_turn_searches_titles_only_never_the_deliverables(db_session, seed_workspace, monkeypatch):
    """A visitor holds tasks:read, not the Deliverables: no file name answers their search."""
    import core.security.surface as surface

    ws = UUID(seed_workspace())
    billed = _card(db_session, ws, "Bill the client for March")
    _deliverable(db_session, ws, billed, "March statement", f"invoices/westmoor-invoice-{uuid4().hex}.pdf")
    monkeypatch.setattr(surface, "widget_turn", lambda: True)

    assert _search(db_session, ws, "Westmoor invoice")["tasks"] == []
    assert len(_search(db_session, ws, "Bill March")["tasks"]) == 1


def test_a_card_missing_one_word_of_the_name_is_not_found(db_session, seed_workspace):
    ws = UUID(seed_workspace())
    _card(db_session, ws, "Westmoor quote")

    found = _search(db_session, ws, "Westmoor invoice")

    assert found["tasks"] == [] and found["total"] == 0


def test_the_deliverables_search_matches_every_word_not_the_phrase():
    from services.deliverable_filters import list_filter

    where, params = list_filter(uuid4(), {"search": "the Guji blog draft"})

    assert params["search_0"] == "%guji%" and params["search_1"] == "%blog%"
    assert "o.file_path ILIKE :search_0" in where and "o.title ILIKE :search_1" in where
    assert ":search_2" not in where


def test_the_actions_say_how_to_find_named_work():
    from modules.tools.discovery.action_registry import ActionRegistry
    from modules.tools.discovery.actions_board_tasks import _register_list_tasks

    registry = ActionRegistry()
    _register_list_tasks(registry)
    action = registry.get("platform_list_tasks")

    assert "search" in action.parameters["properties"]
    assert "pass search" in action.description and "Review" in action.description
