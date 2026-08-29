"""PRD-230 US-004 — the dependency-closure resolver.

PURE tests: ``resolve_closure`` is a pure walk over a ``DependencyReader``, so the
canonical D2 invariant is proven with a dict-backed fake — no Postgres. The DB
reader's edge mapping is checked with a fake agent object (no session).
"""
from __future__ import annotations

from types import SimpleNamespace

from services.dependency_closure import (
    ClosureResult,
    RequiredConnect,
    TypedRef,
    resolve_closure,
    resolve_many,
)


class FakeReader:
    """Dict-backed DependencyReader for pure tests."""

    def __init__(self, agents=None, playbooks=None):
        self.agents = agents or {}
        self.playbooks = playbooks or {}

    def agent_llm(self, ref):
        return self.agents.get(ref, {}).get("llm")

    def agent_skills(self, ref):
        return self.agents.get(ref, {}).get("skills", [])

    def agent_plugins(self, ref):
        return self.agents.get(ref, {}).get("plugins", [])

    def agent_apps(self, ref):
        return self.agents.get(ref, {}).get("apps", [])

    def playbook_members(self, ref):
        return self.playbooks.get(ref, [])


# --------------------------------------------------------------------------- #
# THE canonical invariant (D2): agent A = 3 tools + 2 skills + 1 LLM ⇒ 7
# --------------------------------------------------------------------------- #


def test_agent_a_closure_is_exactly_seven():
    # "tools" are plugins — the tool-providing marketplace artifact (there is no
    # separate agent→tool edge; tools ride plugins/skills). agent A itself is the
    # 7th registration (installing A registers A + its 6 dependencies).
    reader = FakeReader(agents={
        "A": {"llm": "gpt-4o", "skills": ["s1", "s2"], "plugins": ["t1", "t2", "t3"]},
    })
    result = resolve_closure(TypedRef("agent", "A"), reader)

    assert len(result.members) == 7
    assert len(result.by_type("agent")) == 1
    assert len(result.by_type("llm")) == 1
    assert len(result.by_type("skill")) == 2
    assert len(result.by_type("plugin")) == 3  # the 3 tools
    assert result.members[0] == TypedRef("agent", "A")  # root first


def test_closure_returns_a_closure_result():
    result = resolve_closure(TypedRef("agent", "A"), FakeReader(agents={"A": {}}))
    assert isinstance(result, ClosureResult)
    assert result.members == [TypedRef("agent", "A")]  # a bare agent is just itself


# --------------------------------------------------------------------------- #
# Playbook recursion — a playbook pulls its member agents' full closures
# --------------------------------------------------------------------------- #


def test_playbook_recursion_pulls_member_agent_closures():
    reader = FakeReader(
        agents={
            "A": {"llm": "gpt-4o", "skills": ["s1"], "plugins": ["p1"]},
            "B": {"llm": "claude-haiku", "skills": ["s2"]},
        },
        playbooks={"PB": [TypedRef("agent", "A"), TypedRef("agent", "B")]},
    )
    result = resolve_closure(TypedRef("playbook", "PB"), reader)
    keys = {m.key for m in result.members}

    assert ("playbook", "PB") in keys
    assert {("agent", "A"), ("agent", "B")} <= keys
    assert {("llm", "gpt-4o"), ("llm", "claude-haiku")} <= keys
    assert {("skill", "s1"), ("skill", "s2"), ("plugin", "p1")} <= keys


# --------------------------------------------------------------------------- #
# Cycle safety — a self/mutually-referential playbook must terminate
# --------------------------------------------------------------------------- #


def test_mutual_playbook_cycle_terminates_and_dedups():
    reader = FakeReader(playbooks={
        "P1": [TypedRef("playbook", "P2")],
        "P2": [TypedRef("playbook", "P1")],  # cycle back to P1
    })
    result = resolve_closure(TypedRef("playbook", "P1"), reader)
    keys = [m.key for m in result.members]

    assert sorted(keys) == [("playbook", "P1"), ("playbook", "P2")]
    assert keys.count(("playbook", "P1")) == 1  # visited-guard bit


def test_self_referential_playbook_terminates():
    reader = FakeReader(playbooks={"SELF": [TypedRef("playbook", "SELF")]})
    result = resolve_closure(TypedRef("playbook", "SELF"), reader)
    assert result.members == [TypedRef("playbook", "SELF")]


# --------------------------------------------------------------------------- #
# App assignments → required_connects, distinct from installable members (FR-4)
# --------------------------------------------------------------------------- #


def test_agent_apps_become_required_connects_not_members():
    reader = FakeReader(agents={
        "A": {
            "llm": "gpt-4o",
            "apps": [RequiredConnect("SHOPIFY"), RequiredConnect("GMAIL")],
        },
    })
    result = resolve_closure(TypedRef("agent", "A"), reader)

    member_types = {m.type for m in result.members}
    assert "app" not in member_types and "connect" not in member_types
    assert len(result.members) == 2  # agent + llm only — apps do NOT inflate members
    assert {rc.app_name for rc in result.required_connects} == {"SHOPIFY", "GMAIL"}


def test_required_connects_deduped_case_insensitively():
    reader = FakeReader(agents={
        "A": {"apps": [RequiredConnect("SHOPIFY")]},
        "B": {"apps": [RequiredConnect("shopify")]},
    })
    result = resolve_many([TypedRef("agent", "A"), TypedRef("agent", "B")], reader)
    assert len(result.required_connects) == 1  # keyed by app_name.upper()


# --------------------------------------------------------------------------- #
# resolve_many — a whole package, deduped across shared dependencies
# --------------------------------------------------------------------------- #


def test_resolve_many_dedups_shared_dependencies():
    reader = FakeReader(agents={
        "A": {"llm": "gpt-4o", "skills": ["shared"]},
        "B": {"llm": "gpt-4o", "skills": ["shared"]},  # same llm + skill
    })
    result = resolve_many([TypedRef("agent", "A"), TypedRef("agent", "B")], reader)
    keys = [m.key for m in result.members]

    assert keys.count(("llm", "gpt-4o")) == 1
    assert keys.count(("skill", "shared")) == 1
    assert {("agent", "A"), ("agent", "B")} <= set(keys)
    assert len(result.members) == 4  # A, B, gpt-4o, shared


# --------------------------------------------------------------------------- #
# Leaves + determinism
# --------------------------------------------------------------------------- #


def test_leaf_types_resolve_to_self_only():
    reader = FakeReader()
    for t in ("skill", "plugin", "tool", "llm"):
        result = resolve_closure(TypedRef(t, "x"), reader)
        assert [m.key for m in result.members] == [(t, "x")]


def test_walk_is_deterministic():
    reader = FakeReader(agents={"A": {"llm": "l", "skills": ["s1", "s2"], "plugins": ["p1"]}})
    first = [m.key for m in resolve_closure(TypedRef("agent", "A"), reader).members]
    again = [m.key for m in resolve_closure(TypedRef("agent", "A"), reader).members]
    assert first == again


# --------------------------------------------------------------------------- #
# DbDependencyReader edge mapping (no session — _agent stubbed)
# --------------------------------------------------------------------------- #


def test_db_reader_maps_agent_edges(monkeypatch):
    from services.dependency_closure import DbDependencyReader

    reader = DbDependencyReader(db=None)
    fake_agent = SimpleNamespace(
        id=5,
        model_config={"model_id": "gpt-4o"},
        skills=[SimpleNamespace(id=1), SimpleNamespace(id=2)],
        assigned_plugins=[SimpleNamespace(plugin_id="pl-1"), SimpleNamespace(plugin_id="pl-2")],
    )
    monkeypatch.setattr(reader, "_agent", lambda ref: fake_agent)

    assert reader.agent_llm("5") == "gpt-4o"
    assert reader.agent_skills("5") == ["1", "2"]
    assert reader.agent_plugins("5") == ["pl-1", "pl-2"]


def test_db_reader_missing_agent_degrades_to_empty(monkeypatch):
    from services.dependency_closure import DbDependencyReader

    reader = DbDependencyReader(db=None)
    monkeypatch.setattr(reader, "_agent", lambda ref: None)
    assert reader.agent_llm("999") is None
    assert reader.agent_skills("999") == []
    assert reader.agent_plugins("999") == []
