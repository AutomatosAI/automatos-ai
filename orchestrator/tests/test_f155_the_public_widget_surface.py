"""F155 — what a public widget key reaches.

(h) The widget data plane runs no caller SQL: only the natural-language query,
which the NL2SQL service scopes to the key's workspace, remains.
"""
from __future__ import annotations


def test_the_widget_data_plane_runs_no_caller_sql():
    from api.widgets import data

    assert sorted(route.path for route in data.router.routes) == ["/data/query"]
