"""F105 (4 Nov): a document search the web app tracks is saved, with its results count.

Found while sweeping the routers: ``POST /api/documents/analytics/track`` never saved
anything. The web app (``useTrackSearchQuery``) posts a JSON body with ``event_type``,
``query`` and ``results_count``, but the route read query parameters, so every call
answered 422. And its save read an undefined ``results_count``: the NameError fell into
the log-only fallback, so no row was ever written. The route now takes the body the web
app sends, and the save writes the count it is given.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from api.documents import UsageEventRequest, _track_usage_event_save

SEARCH = {"event_type": "document_searched", "query": "refund policy", "results_count": 7}
SAVED_ID = 42


class _Db:
    """A session stand-in that keeps what the save sends."""

    def __init__(self) -> None:
        self.params = None
        self.committed = False

    def execute(self, _query, params):
        self.params = params
        return SimpleNamespace(fetchone=lambda: (SAVED_ID,))

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        raise AssertionError("the save fell back instead of writing the row")


def test_the_route_reads_the_body_the_web_app_sends():
    event = UsageEventRequest(**SEARCH)
    assert (event.event_type, event.query, event.results_count) == ("document_searched", "refund policy", 7)


def test_the_save_writes_the_row_with_its_results_count():
    db = _Db()
    saved = _track_usage_event_save(db, "document_searched", None, "refund policy", 7, {"page": "search"})
    assert saved["event_id"] == SAVED_ID and db.committed
    assert db.params["results_count"] == 7
    assert json.loads(db.params["metadata"]) == {"page": "search"}


def test_a_search_without_a_count_is_still_saved():
    db = _Db()
    saved = _track_usage_event_save(db, "document_viewed", 3, None, None, None)
    assert saved["event_id"] == SAVED_ID
    assert db.params["results_count"] is None and db.params["metadata"] is None
