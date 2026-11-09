"""PRD-274 Wave 4 (§8-7): a workspace's managers create, change and delete its own memory profiles.

Seeded profiles stay as shipped; a workspace's own with the same slug overrides one, and deleting it
brings the seeded one back. ``general`` is reserved. Examples use the platform's fact types. A
workspace reaches only its own profiles. Real database (conftest's ``new_session``).
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from api.memory_profiles import (
    ProfileCreate,
    ProfileFields,
    create_memory_profile,
    delete_memory_profile,
    list_memory_profiles,
    update_memory_profile,
)

RUNBOOK = {"name": "Acme on-call", "focus": "Acme's incident runbook: services, owners, fixes.",
           "keep": ["which service failed and why"], "skip": ["status pings"],
           "examples": [{"fact": "The billing API fails over to eu-west-2.", "type": "tool_outcome"}]}


@pytest.fixture(scope="module")
def engine():  # conftest's new_session binds to it
    from sqlalchemy import create_engine

    from core.database.database import get_database_url
    from core.models.memory_profiles import MemoryProfile

    eng = create_engine(get_database_url(), pool_pre_ping=True)
    MemoryProfile.__table__.create(eng, checkfirst=True)  # a test database built before the table
    yield eng
    eng.dispose()


@pytest.fixture
def seeded(new_session):
    from core.seeds.seed_memory_profiles import seed_memory_profiles

    s = new_session()
    seed_memory_profiles(s)
    s.close()


@pytest.fixture
def workspaces(new_session):
    from sqlalchemy import text

    made = []

    def make(settings=None) -> str:
        ws = str(uuid4())
        s = new_session()
        s.execute(text("INSERT INTO workspaces (id, name, settings) VALUES (CAST(:id AS uuid), 'Wave 4c', "
                       "CAST(:settings AS jsonb))"), {"id": ws, "settings": json.dumps(settings or {})})
        s.commit()
        s.close()
        made.append(ws)
        return ws

    yield make
    s = new_session.sweep()
    for ws in made:
        s.execute(text("DELETE FROM memory_profiles WHERE workspace_id = CAST(:id AS uuid)"), {"id": ws})
        s.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": ws})
    s.commit()
    s.close()


def _ctx(ws: str) -> SimpleNamespace:
    return SimpleNamespace(workspace_id=ws)


def _listed(new_session, ws: str) -> dict:
    s = new_session()
    try:
        return {p["slug"]: p for p in list_memory_profiles(ctx=_ctx(ws), db=s)["profiles"]}
    finally:
        s.close()


def _create(new_session, ws: str, slug: str, **fields) -> dict:
    s = new_session()
    try:
        return create_memory_profile(ProfileCreate(slug=slug, **{**RUNBOOK, **fields}), ctx=_ctx(ws), db=s)
    finally:
        s.close()


def test_a_workspace_creates_its_own_profile_and_distillation_uses_it(new_session, seeded, workspaces, monkeypatch):
    from config import config
    from modules.memory import memory_profiles as mp

    monkeypatch.setattr(config, "MEMORY_PROFILES_ENABLED", True)
    ws = workspaces({"memory_profile": "acme-on-call"})
    made = _create(new_session, ws, "acme-on-call")
    assert made["seeded"] is False and made["examples"] == RUNBOOK["examples"]
    assert _listed(new_session, ws)["acme-on-call"]["name"] == "Acme on-call"
    s = new_session()
    profile = mp.profile_for(s, ws, None)
    s.close()
    assert profile is not None and profile.focus == RUNBOOK["focus"] and profile.keep == tuple(RUNBOOK["keep"])


def test_its_own_overrides_a_seeded_one_and_deleting_it_brings_the_seeded_one_back(new_session, seeded, workspaces):
    ws = workspaces()
    seeded_customer = _listed(new_session, ws)["customer"]
    assert seeded_customer["seeded"] is True
    _create(new_session, ws, "customer", name="Acme support")
    assert _listed(new_session, ws)["customer"]["name"] == "Acme support"
    s = new_session()
    delete_memory_profile("customer", ctx=_ctx(ws), db=s)
    s.close()
    assert _listed(new_session, ws)["customer"] == seeded_customer


def test_an_update_replaces_what_its_own_profile_says(new_session, seeded, workspaces):
    ws = workspaces()
    _create(new_session, ws, "acme-on-call")
    s = new_session()
    changed = update_memory_profile("acme-on-call", ProfileFields(name="Acme SRE", focus="Outages only."),
                                    ctx=_ctx(ws), db=s)
    s.close()
    assert (changed["name"], changed["focus"], changed["keep"], changed["examples"]) == ("Acme SRE", "Outages only.", [], [])
    assert _listed(new_session, ws)["acme-on-call"]["name"] == "Acme SRE"


def test_a_slug_it_already_has_is_a_conflict(new_session, seeded, workspaces):
    ws = workspaces()
    _create(new_session, ws, "acme-on-call")
    with pytest.raises(HTTPException) as raised:
        _create(new_session, ws, "acme-on-call", name="Again")
    assert raised.value.status_code == 409
    assert _listed(new_session, ws)["acme-on-call"]["name"] == "Acme on-call"


def test_seeded_profiles_and_other_workspaces_profiles_are_out_of_reach(new_session, seeded, workspaces):
    mine, theirs = workspaces(), workspaces()
    _create(new_session, theirs, "acme-on-call")
    s = new_session()
    try:
        for attempt in (lambda: update_memory_profile("acme-on-call", ProfileFields(name="Mine", focus="x"),
                                                      ctx=_ctx(mine), db=s),
                        lambda: delete_memory_profile("acme-on-call", ctx=_ctx(mine), db=s),
                        lambda: update_memory_profile("customer", ProfileFields(name="Mine", focus="x"),
                                                      ctx=_ctx(mine), db=s),
                        lambda: delete_memory_profile("customer", ctx=_ctx(mine), db=s)):
            with pytest.raises(HTTPException) as raised:
                attempt()
            assert raised.value.status_code == 404
    finally:
        s.close()
    assert _listed(new_session, theirs)["acme-on-call"]["name"] == "Acme on-call"
    assert "acme-on-call" not in _listed(new_session, mine)
    assert _listed(new_session, mine)["customer"]["seeded"] is True


@pytest.mark.parametrize("slug", ["general", "Acme", "-acme", "acme on call", "a" * 65, ""])
def test_general_and_malformed_slugs_are_refused(slug):
    with pytest.raises(ValidationError):
        ProfileCreate(slug=slug, **RUNBOOK)


@pytest.mark.parametrize("change", [
    {"examples": [{"fact": "The billing API fails over to eu-west-2.", "type": "billing_fact"}]},
    {"examples": [{"fact": "", "type": "tool_outcome"}]},
    {"keep": ["x"] * 21},
    {"skip": ["x" * 201]},
    {"keep": ["   "]},
    {"name": ""},
    {"focus": ""},
    {"focus": "x" * 2001},
])
def test_what_a_profile_says_is_bounded_and_uses_the_platforms_fact_types(change):
    with pytest.raises(ValidationError):
        ProfileFields(**{**RUNBOOK, **change})


def test_only_the_workspaces_managers_can_change_its_profiles():
    from fastapi.routing import APIRoute

    from api.memory_profiles import router
    from core.auth.workspace_permission import PERMISSION_MARKER_ATTR

    gates = {}
    for route in router.routes:
        if isinstance(route, APIRoute):
            marks = [getattr(d.call, PERMISSION_MARKER_ATTR, None) for d in route.dependant.dependencies]
            for method in route.methods:
                gates[(method, route.path)] = [m for m in marks if m]
    assert gates[("GET", "/api/v1/memory/profiles")] == []
    for key in (("POST", "/api/v1/memory/profiles"), ("PUT", "/api/v1/memory/profiles/{slug}"),
                ("DELETE", "/api/v1/memory/profiles/{slug}")):
        assert gates[key] == ["workspace:manage"], key
