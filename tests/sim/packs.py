"""Scenario packs are data, not code (PRD-247 S0.3).

A pack is a TOML file under ``tests/sim/packs/`` — TOML because ``tomllib`` is
in the standard library from Python 3.11, so the scheduled runner needs no
virtualenv. Shape::

    [pack]
    name = "smoke"
    description = "..."
    budget_usd = 2.0          # optional, overrides the runner's ceiling downwards
    persona = "default"       # optional, the answerer's voice

    [[agents]]                # seeded into the throwaway workspace
    key = "researcher"        # referenced by scenarios
    name = "SIM Researcher"
    agent_type = "custom"
    description = "..."

    [[scenarios]]
    id = "brief"              # unique in the pack
    kind = "task"             # task | chat | crud
    agent = "researcher"
    title = "..."
    description = "..."
    expect = "..."            # plain-English acceptance, read by the judge
    must_contain = ["..."]    # cheap checks on the deliverable text
    [scenarios.expect_effects]
    deliverables_min = 1

Everything is validated up front; a bad pack fails before a workspace exists.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Mapping

from .config import PACKS_DIR

KINDS = ("task", "chat", "crud")
PRIORITIES = ("urgent", "high", "medium", "low")  # api/board_tasks.VALID_PRIORITIES
REVIEW_MODES = ("human", "llm", "auto")  # api/board_tasks.VALID_REVIEW_MODES
CRUD_STEPS = ("create", "update", "get", "list", "delete")
EFFECT_KEYS = ("tasks_created", "agents_created", "tool_calls_min", "deliverables_min",
               "reports_min", "no_errors", "status")
WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
FALLBACK_PACK = "smoke"
# Pack names become run directories and workspace slugs; scenario ids and agent
# keys become labels and tags. Keep them to what a path and a slug accept.
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")


class PackError(ValueError):
    """The pack cannot run as written — the message names the file and the field."""


@dataclass(frozen=True)
class AgentSpec:
    key: str
    name: str
    agent_type: str = "custom"
    description: str = ""
    job_title: str = ""
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Scenario:
    id: str
    kind: str
    agent: str | None = None
    title: str = ""
    description: str = ""
    priority: str = "medium"
    review_mode: str = "auto"
    timeout_s: int | None = None
    turns: tuple[str, ...] = ()
    expect: str = ""
    must_contain: tuple[str, ...] = ()
    expect_effects: Mapping[str, Any] = field(default_factory=dict)
    steps: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Pack:
    name: str
    description: str
    budget_usd: float | None
    persona: str | None
    agents: tuple[AgentSpec, ...]
    scenarios: tuple[Scenario, ...]
    path: Path

    def agent(self, key: str) -> AgentSpec:
        for spec in self.agents:
            if spec.key == key:
                return spec
        raise KeyError(key)


def _str_list(value: Any, where: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise PackError(f"{where} must be a list of strings")
    return tuple(value)


def _ident(value: Any, what: str, where: str) -> str:
    if not isinstance(value, str) or not NAME_RE.match(value):
        raise PackError(f"{where}: {what} must match {NAME_RE.pattern}, got {value!r}")
    return value


def _agent(raw: Mapping[str, Any], where: str) -> AgentSpec:
    key, name = _ident(raw.get("key"), "agents[].key", where), raw.get("name")
    if not isinstance(name, str) or not name:
        raise PackError(f"{where}: agent '{key}' needs a name")
    return AgentSpec(
        key=key, name=name, agent_type=str(raw.get("agent_type") or "custom"),
        description=str(raw.get("description") or ""), job_title=str(raw.get("job_title") or ""),
        tags=_str_list(raw.get("tags"), f"{where}: agent '{key}' tags"),
    )


def _scenario(raw: Mapping[str, Any], agent_keys: set[str], where: str) -> Scenario:
    sid, kind = _ident(raw.get("id"), "scenario id", where), raw.get("kind")
    here = f"{where}: scenario '{sid}'"
    if kind not in KINDS:
        raise PackError(f"{here}: kind must be one of {KINDS}, got {kind!r}")
    agent = raw.get("agent")
    if agent is not None and agent not in agent_keys:
        raise PackError(f"{here}: agent '{agent}' is not in [[agents]]")
    priority = str(raw.get("priority") or "medium")
    review_mode = str(raw.get("review_mode") or "auto")
    if priority not in PRIORITIES:
        raise PackError(f"{here}: priority must be one of {PRIORITIES}")
    if review_mode not in REVIEW_MODES:
        raise PackError(f"{here}: review_mode must be one of {REVIEW_MODES}")
    turns = _str_list(raw.get("turns"), f"{here} turns")
    steps = _str_list(raw.get("steps"), f"{here} steps")
    effects = raw.get("expect_effects") or {}
    if not isinstance(effects, Mapping):
        raise PackError(f"{here}: expect_effects must be a table")
    unknown = sorted(set(effects) - set(EFFECT_KEYS))
    if unknown:
        raise PackError(f"{here}: unknown expect_effects keys {unknown}; known: {EFFECT_KEYS}")
    if kind == "task":
        if not agent:
            raise PackError(f"{here}: a task scenario needs an agent")
        if not raw.get("title"):
            raise PackError(f"{here}: a task scenario needs a title")
    if kind == "chat" and not turns:
        raise PackError(f"{here}: a chat scenario needs at least one turn")
    if kind == "crud":
        bad = [s for s in steps if s not in CRUD_STEPS]
        if bad or not steps:
            raise PackError(f"{here}: steps must be a non-empty subset of {CRUD_STEPS}, got {steps}")
    timeout = raw.get("timeout_s")
    if timeout is not None and (not isinstance(timeout, int) or timeout <= 0):
        raise PackError(f"{here}: timeout_s must be a positive integer")
    return Scenario(
        id=sid, kind=kind, agent=agent, title=str(raw.get("title") or ""),
        description=str(raw.get("description") or ""), priority=priority, review_mode=review_mode,
        timeout_s=timeout, turns=turns, expect=str(raw.get("expect") or ""),
        must_contain=_str_list(raw.get("must_contain"), f"{here} must_contain"),
        expect_effects=dict(effects), steps=steps, tags=_str_list(raw.get("tags"), f"{here} tags"),
    )


def parse_pack(raw: Mapping[str, Any], path: Path) -> Pack:
    """Validate a decoded TOML document into a Pack; every failure names the field."""
    where = path.name
    meta = raw.get("pack")
    if not isinstance(meta, Mapping) or not meta.get("name"):
        raise PackError(f"{where}: [pack] needs a name")
    _ident(meta.get("name"), "[pack].name", where)
    budget = meta.get("budget_usd")
    if budget is not None and (not isinstance(budget, (int, float)) or budget <= 0):
        raise PackError(f"{where}: [pack].budget_usd must be a positive number")
    agents = tuple(_agent(a, where) for a in raw.get("agents") or [])
    keys = [a.key for a in agents]
    if len(keys) != len(set(keys)):
        raise PackError(f"{where}: duplicate agent keys")
    scenarios = tuple(_scenario(s, set(keys), where) for s in raw.get("scenarios") or [])
    if not scenarios:
        raise PackError(f"{where}: a pack needs at least one [[scenarios]] entry")
    ids = [s.id for s in scenarios]
    if len(ids) != len(set(ids)):
        raise PackError(f"{where}: duplicate scenario ids")
    persona = meta.get("persona")
    return Pack(
        name=meta["name"], description=str(meta.get("description") or ""),
        budget_usd=float(budget) if budget is not None else None,
        persona=str(persona) if persona else None, agents=agents, scenarios=scenarios, path=path,
    )


def resolve_pack_path(ref: str | Path, packs_dir: Path = PACKS_DIR) -> Path:
    candidate = Path(ref)
    if candidate.suffix == ".toml" and candidate.exists():
        return candidate
    named = packs_dir / f"{ref}.toml"
    if named.exists():
        return named
    raise PackError(f"no pack named {ref!r} (looked for {named}); known: {', '.join(list_packs(packs_dir))}")


def load_pack(ref: str | Path, packs_dir: Path = PACKS_DIR) -> Pack:
    path = resolve_pack_path(ref, packs_dir)
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise PackError(f"{path.name}: not valid TOML: {exc}") from exc
    return parse_pack(raw, path)


def list_packs(packs_dir: Path = PACKS_DIR) -> tuple[str, ...]:
    return tuple(sorted(p.stem for p in packs_dir.glob("*.toml") if p.stem != "rotation"))


def rotation_for(day: date, packs_dir: Path = PACKS_DIR) -> tuple[str, str | None]:
    """Tonight's pack from ``packs/rotation.toml``; falls back to smoke with a note."""
    table = packs_dir / "rotation.toml"
    if not table.exists():
        return FALLBACK_PACK, "no rotation.toml; running smoke"
    raw = tomllib.loads(table.read_text(encoding="utf-8")).get("rotation") or {}
    wanted = raw.get(WEEKDAYS[day.weekday()])
    if not isinstance(wanted, str) or not wanted:
        return FALLBACK_PACK, f"rotation has no entry for {WEEKDAYS[day.weekday()]}; running smoke"
    if not (packs_dir / f"{wanted}.toml").exists():
        return FALLBACK_PACK, f"rotation names '{wanted}' but no such pack exists yet; running smoke"
    return wanted, None
