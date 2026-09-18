"""Packs validate up front; the shipped packs are valid; rotation falls back honestly."""

from datetime import date
from pathlib import Path

import pytest

from tests.sim import packs

GOOD = {
    "pack": {"name": "p", "description": "d", "budget_usd": 1.5, "persona": "default"},
    "agents": [{"key": "r", "name": "R"}],
    "scenarios": [
        {"id": "t", "kind": "task", "agent": "r", "title": "T", "expect_effects": {"deliverables_min": 1}},
        {"id": "c", "kind": "chat", "turns": ["hi"]},
        {"id": "x", "kind": "crud", "steps": ["create", "delete"]},
    ],
}


def _bad(mutate):
    raw = {"pack": dict(GOOD["pack"]), "agents": [dict(a) for a in GOOD["agents"]],
           "scenarios": [dict(s) for s in GOOD["scenarios"]]}
    mutate(raw)
    return raw


def test_good_pack_parses():
    pack = packs.parse_pack(GOOD, Path("good.toml"))
    assert pack.name == "p" and pack.budget_usd == 1.5 and pack.persona == "default"
    assert pack.agent("r").name == "R"
    assert [s.kind for s in pack.scenarios] == ["task", "chat", "crud"]
    assert pack.scenarios[0].expect_effects == {"deliverables_min": 1}


@pytest.mark.parametrize("mutate, message", [
    (lambda r: r["pack"].pop("name"), "needs a name"),
    (lambda r: r["pack"].update(name="../escape"), "must match"),
    (lambda r: r["scenarios"][0].update(id="Bad Id"), "must match"),
    (lambda r: r["agents"][0].update(key="r/x"), "must match"),
    (lambda r: r["pack"].update(budget_usd=-1), "budget_usd"),
    (lambda r: r["scenarios"][0].update(kind="mission"), "kind must be one of"),
    (lambda r: r["scenarios"][0].pop("agent"), "needs an agent"),
    (lambda r: r["scenarios"][0].update(agent="ghost"), "not in \\[\\[agents\\]\\]"),
    (lambda r: r["scenarios"][0].pop("title"), "needs a title"),
    (lambda r: r["scenarios"][0].update(priority="asap"), "priority must be"),
    (lambda r: r["scenarios"][0].update(expect_effects={"wat": 1}), "unknown expect_effects"),
    (lambda r: r["scenarios"][1].update(turns=[]), "at least one turn"),
    (lambda r: r["scenarios"][2].update(steps=["create", "fly"]), "steps must be"),
    (lambda r: r["scenarios"][1].update(id="t"), "duplicate scenario ids"),
    (lambda r: r["agents"].append({"key": "r", "name": "again"}), "duplicate agent keys"),
    (lambda r: r.update(scenarios=[]), "at least one"),
    (lambda r: r["scenarios"][0].update(timeout_s=0), "timeout_s"),
])
def test_bad_packs_name_the_problem(mutate, message):
    with pytest.raises(packs.PackError, match=message):
        packs.parse_pack(_bad(mutate), Path("bad.toml"))


def test_shipped_packs_are_valid():
    names = packs.list_packs()
    assert "smoke" in names and "agents" in names and "rotation" not in names
    for name in names:
        pack = packs.load_pack(name)
        assert pack.scenarios, name
    smoke = packs.load_pack("smoke")
    assert {s.kind for s in smoke.scenarios} == {"task", "chat", "crud"}


def test_unknown_pack_lists_the_known_ones():
    with pytest.raises(packs.PackError, match="smoke"):
        packs.load_pack("no-such-pack")


def test_rotation_falls_back_to_smoke_with_a_reason(tmp_path):
    day = date(2026, 9, 21)
    key = packs.WEEKDAYS[day.weekday()]
    assert packs.rotation_for(day, tmp_path) == ("smoke", "no rotation.toml; running smoke")
    (tmp_path / "rotation.toml").write_text(f'[rotation]\n{key} = "ghost"\n', encoding="utf-8")
    pack, note = packs.rotation_for(day, tmp_path)
    assert pack == "smoke" and "ghost" in note
    (tmp_path / "ghost.toml").write_text("", encoding="utf-8")
    assert packs.rotation_for(day, tmp_path) == ("ghost", None)


def test_shipped_rotation_only_names_packs_or_falls_back():
    for offset in range(7):
        pack, note = packs.rotation_for(date(2026, 9, 21 + offset))
        assert pack in packs.list_packs()
        assert note is None or "no such pack exists yet" in note
