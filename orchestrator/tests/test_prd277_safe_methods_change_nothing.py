"""PRD-277 US-302 (§5.3): no GET, HEAD or OPTIONS route reachable with the session cookie changes state.

US-302's cross-site check holds every cookie-authenticated request with an unsafe method; a safe
method is let through (the cookie is ``SameSite=Lax``), so it changes nothing, and this guard
holds it to that. The live app is read by the boundary sweep's probe
(``tests/authz_sweep_probe.py``: a clean subprocess, DB-free), and every route with a safe method
that the cookie reaches (through ``get_request_context_hybrid``, or ``session_cookie_token`` in its
own body) is held to this:

* **No writes.** Its endpoint and its dependencies make none of the calls
  ``helpers_safe_method_writes`` counts (a commit, a flush, a write on a session or a query, a
  shared helper that commits), unless ``ACCEPTED`` names the route with its reason. Every
  ``ACCEPTED`` route is still served and still makes such a call, so the list cannot go stale.
* **HEAD and OPTIONS.** FastAPI answers HEAD only on a route that declares it, and no route does;
  OPTIONS is the CORS middleware's (the hybrid dependency returns early for it). So the GET
  routes are every safe-method route there is, and a route declaring either would be checked too.
* **The cookie lane's own bookkeeping.** Resolving the cookie writes one thing, the session's
  ``last_seen_at`` and idle clock (``sessions.touch_session``, at most once every five minutes),
  below the level this guard reads. ``LANE_BOOKKEEPING`` names it; every other function of the
  lane makes no write.

What it reads: the endpoint's and each dependency's own source, one level deep. What it does not
see: a write made by a function they call (a service, or a helper beside the endpoint), state
outside the database (rate-limit counters, files, calls to other services), and SQL a route runs
without committing, which ``get_db`` rolls back when the request ends.
"""
from __future__ import annotations

import inspect

from core.auth import session_lane, sessions
from tests.helpers_safe_method_writes import writer_calls
from tests.test_p2w2_authz_boundary_sweep import app_records

# The probe's fixture, used here by name.
FIXTURES = (app_records,)

SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
# No route declares these; FastAPI answers HEAD only where it is declared, OPTIONS is CORS's.
UNDECLARED_SAFE_METHODS = frozenset({"HEAD", "OPTIONS"})
# Safe-method routes the cookie reaches: about 450 today. Below this the guard is not looking.
SCANNED_FLOOR = 300

# Safe-method routes whose own source makes a call the scan counts, each with why it changes nothing.
ACCEPTED = {
    ("GET", "/api/documents/download"): (
        "end_open_transaction ends the request's read-only transaction before the file streams "
        "(F105); nothing is pending, so nothing is committed"
    ),
    ("GET", "/api/documents/{document_id}/download"): (
        "end_open_transaction ends the request's transaction once the row is read (F105); the "
        "row is only read, so nothing is committed"
    ),
    ("GET", "/api/wizard/progress/{profile_id}"): (
        "reads the profile in a session of its own (get_db_session, which commits as it leaves) "
        "before the stream starts; it only reads"
    ),
}

# What resolving the session cookie itself writes, with why it is accepted.
LANE_BOOKKEEPING = {
    sessions.touch_session: (
        "the session's last_seen_at and idle clock, at most once every LAST_SEEN_WRITE_INTERVAL; "
        "it moves no expiry past the absolute one and changes nothing the person can see"
    ),
}


def _cookie_reaches(record) -> bool:
    return record["hybrid"] or record["reads_session_cookie"]


def _safe_cookie_routes(records) -> dict:
    return {(r["method"], r["path"]): r for r in records if r["method"] in SAFE_METHODS and _cookie_reaches(r)}


def test_no_safe_method_route_the_session_cookie_reaches_changes_state(app_records):
    routes = _safe_cookie_routes(app_records)
    assert len(routes) >= SCANNED_FLOOR, f"only {len(routes)} safe-method routes scanned: probe broken?"

    writing = sorted(f"{method} {path} -> {', '.join(record['writes'])}"
                     for (method, path), record in routes.items()
                     if record["writes"] and (method, path) not in ACCEPTED)

    assert not writing, (
        "safe-method routes the session cookie reaches make a write: make them POST (or move the "
        "write out of the GET), or name them in ACCEPTED with why they change nothing:\n  "
        + "\n  ".join(writing)
    )


def test_every_accepted_route_is_served_still_writes_and_says_why(app_records):
    routes = _safe_cookie_routes(app_records)
    for key, reason in ACCEPTED.items():
        assert key in routes, f"{key} is no longer a safe-method route the cookie reaches: drop it"
        assert routes[key]["writes"], f"{key} no longer makes a write the scan counts: drop it"
        assert reason.strip(), f"{key} needs its reason"


def test_no_route_declares_head_or_options_of_its_own(app_records):
    declared = sorted(f"{r['method']} {r['path']}" for r in app_records if r["method"] in UNDECLARED_SAFE_METHODS)
    assert not declared, f"routes declaring HEAD or OPTIONS: check them as GET routes are: {declared}"


def test_the_cookie_lanes_only_write_is_its_named_bookkeeping():
    for writer, reason in LANE_BOOKKEEPING.items():
        assert writer_calls(writer) and reason.strip(), f"{writer.__name__} is named but makes no write"
    assert "sessions.touch_session(" in inspect.getsource(session_lane)
    lane = [f for _, f in inspect.getmembers(session_lane, inspect.isfunction) if f.__module__ == session_lane.__name__]
    assert lane, "the session lane has no functions: import broken?"
    assert {f.__name__: writer_calls(f) for f in lane if writer_calls(f)} == {}


# --------------------------------------------------------------------------- the scan itself


def _writes_a_lot(db, ws, items, end_open_transaction, get_db_session, commit_removal):
    db.add(items)
    db.query(ws).filter(ws.id == 1).delete()
    db.query(ws).filter(ws.id == 1).update({"name": "x"})
    end_open_transaction(db)
    with get_db_session() as other:
        other.flush()
    commit_removal(db, None, None, "x")
    db.commit()


def _writes_nothing(db, ws, result, seen, run):
    """Calls db.commit() only in this docstring."""
    result.update({"name": "x"})  # a dict
    seen.add(1)                   # a set
    seen.delete = None
    run(db, lambda: ws, commit=False)
    return db.query(ws).filter(ws.id == 1).first()


def test_the_scan_counts_every_kind_of_write():
    assert writer_calls(_writes_a_lot) == (
        "add", "commit", "commit_removal", "delete", "end_open_transaction", "flush", "get_db_session", "update",
    )


def test_the_scan_counts_no_read_a_dict_a_set_or_a_docstring():
    assert writer_calls(_writes_nothing) == ()
    assert writer_calls(len) == ()  # a function with no Python source counts nothing
