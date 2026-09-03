"""PRD-234 S3 — Auto can ask "who is best suited?" and gets a ranked, explained answer."""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from modules.tools.discovery import handlers_agents as ha  # noqa: E402
from modules.tools.discovery.handlers_board_tasks import _parse_deadline  # noqa: E402


class _Q:
    def __init__(self, rows): self._rows = rows
    def filter(self, *a, **k): return self
    def order_by(self, *a, **k): return self
    def all(self): return list(self._rows)


class _DB:
    def __init__(self, rows): self._rows = rows
    def query(self, *a, **k): return _Q(self._rows)


def _agent(i, name, *, runtime="api", model=None, system=False):
    cfg = {"runtime": runtime}
    if runtime == "cli":
        cfg.update({"provider": "claude", "model": model or "sonnet"})
    return SimpleNamespace(id=i, name=name, status="active", agent_type="custom", is_system_agent=system,
                           configuration=cfg, model_config={"model_id": model} if runtime == "api" and model else {})


def _wire(monkeypatch, *, semantic=None, ranked=None):
    import core.routing.semantic_indexer as si
    import modules.coordination.agent_matcher as am

    async def _fake_similar(query, ws, db, *, min_score=0.0):
        return [(a, s) for a, s in (semantic or [])]
    monkeypatch.setattr(si, "find_similar_agents", _fake_similar)

    def _fake_rank(db, task, agents, task_spec=None, semantic=None):
        return [SimpleNamespace(agent_id=aid, agent_name="", total_score=score, availability=avail, reason=reason)
                for aid, score, avail, reason in (ranked or [])]
    monkeypatch.setattr(am.AgentMatcher, "rank", staticmethod(_fake_rank))


def test_ranks_the_roster_with_reasons_and_runtime(monkeypatch):
    bob = _agent(15, "Bob", runtime="cli", model="fable")
    ann = _agent(16, "Ann", model="google/gemini-2.5-flash")
    auto = _agent(1, "Auto", system=True)
    _wire(monkeypatch,
          semantic=[(bob, 0.9), (ann, 0.4)],
          ranked=[(15, 0.7, 1.0, "skills: python, testing"), (16, 0.5, 0.0, "tools: gmail")])
    out = asyncio.run(ha.recommend_agent(_DB([auto, bob, ann]), "ws", {"objective": "fix the failing pytest suite"}))
    assert out["success"] and out["considered"] == 2
    top = out["candidates"][0]
    assert top["name"] == "Bob" and top["runtime"] == "cli" and top["model"] == "fable"
    assert top["score"] == round(0.6 * 0.7 + 0.4 * 0.9, 3) and "python" in top["reason"]
    second = out["candidates"][1]
    assert second["name"] == "Ann" and second["runtime"] == "api" and second["busy"] is True
    assert "let the user confirm" in out["note"]


def test_prefer_runtime_filters_and_missing_signals_fail_soft(monkeypatch):
    bob = _agent(15, "Bob", runtime="cli")
    ann = _agent(16, "Ann")
    import core.routing.semantic_indexer as si
    import modules.coordination.agent_matcher as am

    async def _boom(*a, **k): raise RuntimeError("no embeddings")
    monkeypatch.setattr(si, "find_similar_agents", _boom)
    monkeypatch.setattr(am.AgentMatcher, "rank", staticmethod(lambda *a, **k: (_ for _ in ()).throw(RuntimeError("matcher down"))))
    out = asyncio.run(ha.recommend_agent(_DB([bob, ann]), "ws", {"objective": "anything", "prefer_runtime": "cli"}))
    assert out["success"] and [c["name"] for c in out["candidates"]] == ["Bob"]
    assert out["candidates"][0]["reason"] == "no signal — roster order"
    assert asyncio.run(ha.recommend_agent(_DB([bob]), "ws", {"objective": ""}))["success"] is False


def test_runtime_of_and_deadline_parsing():
    assert ha._runtime_of({"runtime": "cli"}) == "cli" and ha._runtime_of({}) == "api" and ha._runtime_of(None) == "api"
    dt = _parse_deadline("2026-09-05T17:00:00Z")
    assert dt is not None and dt.tzinfo is not None and dt.hour == 17
    assert _parse_deadline("not a date") is None and _parse_deadline(None) is None
