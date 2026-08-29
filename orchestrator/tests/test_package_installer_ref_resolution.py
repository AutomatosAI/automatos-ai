"""Resolving a marketplace agent by SLUG must not blow up on a UUID column.

LIVE FAILURE (2026-08-29, persona harness → Railway logs). Every package
install failed. The traceback:

    File "services/package_installer.py", line 151, in _find_marketplace_agent
    psycopg2.errors.InvalidTextRepresentation:
        invalid input syntax for type uuid: "shopify-ops"

Package members reference agents by SLUG — ``seed_packages`` builds them that
way deliberately, citing the real roster. ``_find_marketplace_agent`` tried an
``int()`` cast first (ValueError, caught), then fell into a filter comparing
``Agent.public_id`` — a UUID column — against "shopify-ops". Postgres raised at
execute time, INSIDE the driver, where the Python-level ``except (ValueError,
TypeError)`` could never reach it. The exception escaped the installer, escaped
the handler, and surfaced to Auto as the opaque "Action 'platform_install_package'
failed".

Net effect: the seeded Shopify packages could be searched and offered, but
NEVER installed. The fix compares each column only when ``ref`` has the right
shape for it.
"""
from __future__ import annotations

import pytest

from services.package_installer import _find_marketplace_agent, _is_uuid


# --------------------------------------------------------------------------- #
# _is_uuid — the shape test that decides whether public_id is safe to compare
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("good", [
    "3f2504e0-4f89-11d3-9a0c-0305e82c3301",
    "00000000-0000-0000-0000-000000000000",
])
def test_is_uuid_accepts_uuids(good):
    assert _is_uuid(good) is True


@pytest.mark.parametrize("bad", [
    "shopify-ops",                    # the exact prod ref that broke it
    "shopify-business-analyst",
    "", None, 42, [], "not-a-uuid",
])
def test_is_uuid_rejects_everything_else(bad):
    assert _is_uuid(bad) is False


# --------------------------------------------------------------------------- #
# The resolver: a slug must never reach the UUID column
# --------------------------------------------------------------------------- #


class _Recorder:
    """Query stand-in that records how many filters were applied and returns a
    sentinel — enough to assert WHICH predicate path was taken, with no DB."""

    def __init__(self, store):
        self._store = store

    def filter(self, *criteria):
        self._store.append(criteria)
        return self

    def first(self):
        return "AGENT" if len(self._store) > 1 else None


class _Session:
    def __init__(self):
        self.filters = []

    def query(self, _model):
        return _Recorder(self.filters)


def _predicate_sql(session) -> str:
    """The text of the last predicate applied — used to prove public_id is
    absent for a slug lookup and present for a UUID lookup."""
    return " ".join(str(c) for crit in session.filters for c in crit)


def test_slug_lookup_never_references_public_id():
    # THE REGRESSION. A slug must not produce a public_id comparison, or
    # Postgres raises InvalidTextRepresentation and the whole install dies.
    s = _Session()
    _find_marketplace_agent(s, "shopify-ops")
    sql = _predicate_sql(s)
    assert "slug" in sql
    assert "public_id" not in sql, (
        "a slug reached the public_id (uuid) column — this is the exact "
        "comparison that made every package install fail in production"
    )


def test_uuid_lookup_does_reference_public_id():
    s = _Session()
    _find_marketplace_agent(s, "3f2504e0-4f89-11d3-9a0c-0305e82c3301")
    assert "public_id" in _predicate_sql(s)


def test_numeric_ref_uses_the_id_column():
    s = _Session()
    _find_marketplace_agent(s, "123")
    sql = _predicate_sql(s)
    assert "agents.id" in sql or ".id" in sql
    assert "public_id" not in sql


def test_seeded_package_refs_all_resolve_by_slug_path():
    """Every ref in the shipped packages is a slug, so every one of them takes
    the path that used to explode."""
    from core.seeds.seed_packages import PACKAGES

    refs = [m["ref"] for p in PACKAGES for m in p["members"] if m["type"] == "agent"]
    assert refs, "seed carries no agent members — this guard would be vacuous"
    for ref in refs:
        assert not _is_uuid(ref)          # all slugs, none UUIDs
        s = _Session()
        _find_marketplace_agent(s, ref)
        assert "public_id" not in _predicate_sql(s)
