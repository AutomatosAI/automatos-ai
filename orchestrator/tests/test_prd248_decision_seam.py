"""PRD-248 — the decision seam.

Typed questions serialise to the System One shapes; the HTTP backend speaks
both routes and books its receipt; the LLM adapter answers the same contract;
the engine reads its dials fail-soft and never raises into a turn; AutoBrain
runs the engine as a shadow beside every tier without changing a verdict, and
as Tier 2.5 in live mode only above the confidence floor.

PURE tests: HTTP through ``httpx.MockTransport``, the LLM through a fake
manager, AutoBrain over a fake session with the cache and Tier 3 stubbed.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any, Dict, Optional

import httpx
import pytest

from config import config
from consumers.chatbot import auto as auto_mod
from consumers.chatbot import auto_decisions
from consumers.chatbot.auto import Action, AutoBrain, Complexity, ComplexityAssessment
from core.llm.decisions import (
    MODE_LIVE,
    MODE_OFF,
    MODE_SHADOW,
    Choice,
    DecisionAnswer,
    DecisionEngine,
    DecisionResult,
    Dials,
    LLMDecisionAdapter,
    Noul,
    Score,
    TypeSafeDecisionClient,
    parse_answers,
    to_wire,
)
from core.llm.decisions import typesafe_client as ts_mod
from core.llm.usage_tracker import UsageTracker

WS = "11111111-1111-1111-1111-111111111111"


# --------------------------------------------------------------------------- #
# Questions and answers
# --------------------------------------------------------------------------- #


def test_to_wire_matches_system_one_shapes():
    wire = to_wire(
        {
            "dept": Choice("Which team?", {"billing": "Money", "tech": None}),
            "urgency": Score("How urgent?", ["can wait", "soon", "now"]),
            "human": Noul("Needs a human?", {"true": "yes", "false": "no"}),
            "bare": Noul("Bare?"),
        }
    )
    assert wire["dept"] == {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": "Money", "tech": None},
    }
    assert wire["urgency"] == {
        "type": "score",
        "instructions": "How urgent?",
        "criteria": ["can wait", "soon", "now"],
    }
    assert wire["human"]["criteria"] == {"true": "yes", "false": "no"}
    assert "criteria" not in wire["bare"]


def test_question_bounds_are_the_api_limits():
    with pytest.raises(ValueError):
        Choice("one option", {"only": None})
    with pytest.raises(ValueError):
        Choice("too many", {str(i): None for i in range(256)})
    with pytest.raises(ValueError):
        Score("one level", ["only"])
    with pytest.raises(ValueError):
        to_wire({})


def test_parse_answers_is_tolerant_and_certainty_is_one_number():
    answers = parse_answers(
        {
            "typed": {"type": "choice", "choice": "a", "probabilities": {"a": "0.8", "b": 0.2}, "confidence": 0.75},
            "inferred": {"choice": "x"},
            "argmax": {"type": "choice", "probabilities": {"p": 0.3, "q": 0.7}},
            "score": {"type": "score", "probabilities": [0.1, 0.2, 0.7]},
            "no": {"noul": 0.05},
            "junk": {"type": "choice"},
            "notdict": 3,
        }
    )
    assert set(answers) == {"typed", "inferred", "argmax", "score", "no"}
    assert answers["typed"].certainty == 0.75
    assert answers["inferred"].certainty == 0.0
    assert answers["argmax"].choice == "q" and answers["argmax"].certainty == 0.7
    assert answers["score"].score == pytest.approx(1.6)
    assert answers["no"].yes is False and answers["no"].certainty == pytest.approx(0.9)


# --------------------------------------------------------------------------- #
# The HTTP backend
# --------------------------------------------------------------------------- #

CANNED = {
    "model": "jev-1.13.0",
    "answers": {
        "complexity": {"type": "choice", "choice": "atom", "probabilities": {"atom": 0.9, "molecule": 0.1}, "confidence": 0.9},
        "urgent": {"type": "noul", "noul": 0.2},
    },
    "usage": {"input_tokens": 283, "output_tokens": 23},
}


def _client(monkeypatch, handler, provider="typesafe", key="jev_test"):
    monkeypatch.setattr(ts_mod, "resolve_api_key", lambda p: key)
    factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))  # noqa: E731
    return TypeSafeDecisionClient(provider=provider, timeout_s=2.5, client_factory=factory)


@pytest.fixture
def usage(monkeypatch):
    calls = []
    monkeypatch.setattr(UsageTracker, "track", staticmethod(lambda **kw: calls.append(kw)))
    return calls


QUESTIONS = {
    "complexity": Choice("How much?", {"atom": "small", "molecule": "tools"}),
    "urgent": Noul("Urgent?"),
}


@pytest.mark.asyncio
async def test_client_posts_the_native_body_and_books_the_receipt(monkeypatch, usage):
    seen: Dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=CANNED)

    client = _client(monkeypatch, handler)
    from core.llm.usage_context import usage_scope

    # A PRD-247 campaign sets the scope; the receipt must join that campaign.
    with usage_scope(execution_id="sim:night1:s07:2", agent_id=9):
        result = await client.decide(state={"message": "hi"}, questions=QUESTIONS, workspace_id=WS, purpose="t")

    assert seen["url"] == config.TYPESAFE_API_URL
    assert seen["auth"] == "Bearer jev_test"
    assert seen["body"]["model"] == "jev-1.13.0"
    assert seen["body"]["state"] == {"message": "hi"}
    assert seen["body"]["questions"]["urgent"] == {"type": "noul", "instructions": "Urgent?"}
    assert result is not None and result.get("complexity").choice == "atom"
    assert result.input_tokens == 283 and result.model == "jev-1.13.0"

    assert len(usage) == 1
    receipt = usage[0]
    assert receipt["request_type"] == "decision" and receipt["provider"] == "typesafe"
    assert receipt["input_tokens"] == 283 and receipt["status"] == "success"
    assert receipt["cost_override"] == (pytest.approx(283 / 1e6 * config.DECISION_ENGINE_USD_PER_MTOK_IN), 0.0)
    assert receipt["execution_id"] == "sim:night1:s07:2" and receipt["agent_id"] == 9
    assert receipt["workspace_id"] == WS  # an explicit workspace wins over the scope


@pytest.mark.asyncio
async def test_openrouter_route_uses_its_endpoint_and_the_platform_key(monkeypatch, usage):
    seen: Dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["title"] = request.headers.get("x-title")
        seen["model"] = json.loads(request.content)["model"]
        return httpx.Response(200, json=CANNED)

    client = _client(monkeypatch, handler, provider="openrouter", key="sk-or-test")
    result = await client.decide(state="s", questions=QUESTIONS, workspace_id=WS)
    assert seen["url"] == config.OPENROUTER_DECISIONS_URL
    assert seen["title"] == "Automatos" and seen["model"] == "typesafe/jev-1.13"
    assert result is not None and usage[0]["provider"] == "openrouter"


@pytest.mark.asyncio
async def test_http_error_and_timeout_return_none_and_book_an_error(monkeypatch, usage):
    client = _client(monkeypatch, lambda r: httpx.Response(429, text="slow down"))
    assert await client.decide(state="s", questions=QUESTIONS, workspace_id=WS) is None
    assert usage[-1]["status"] == "error" and usage[-1]["error_message"] == "HTTP 429"

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("late", request=request)

    client = _client(monkeypatch, boom)
    assert await client.decide(state="s", questions=QUESTIONS, workspace_id=WS) is None
    assert usage[-1]["status"] == "error" and usage[-1]["error_message"] == "timeout"


@pytest.mark.asyncio
async def test_no_key_means_idle_not_error(monkeypatch, usage):
    client = _client(monkeypatch, lambda r: httpx.Response(200, json=CANNED), key=None)
    assert client.is_available() is False
    assert await client.decide(state="s", questions=QUESTIONS, workspace_id=WS) is None
    assert usage == []


# --------------------------------------------------------------------------- #
# The LLM adapter
# --------------------------------------------------------------------------- #


class _FakeLLM:
    def __init__(self, content: str):
        self.content = content
        self.prompts = []

    async def generate_response(self, messages, tools=None, on_delta=None):
        self.prompts.append(messages[0]["content"])
        return SimpleNamespace(content=self.content, model="fake-flash", usage={"prompt_tokens": 50, "completion_tokens": 9})


@pytest.mark.asyncio
async def test_llm_adapter_answers_the_same_contract():
    payload = json.dumps(
        {
            "complexity": {"choice": "nonsense", "probabilities": {"atom": 3, "molecule": 1}},
            "urgent": {"noul": 1.7},
        }
    )
    llm = _FakeLLM("Sure:\n" + payload)
    adapter = LLMDecisionAdapter(manager_factory=lambda **kw: llm)
    result = await adapter.decide(state={"message": "hi"}, questions=QUESTIONS, workspace_id=WS)

    assert result is not None and result.provider == "llm" and result.model == "fake-flash"
    comp = result.get("complexity")
    assert comp.choice == "atom" and comp.probabilities == {"atom": 0.75, "molecule": 0.25}
    assert comp.certainty == 0.75
    assert result.get("urgent").noul == 1.0
    assert result.input_tokens == 50
    assert '"urgent": {"type": "noul"' in llm.prompts[0]


@pytest.mark.asyncio
async def test_llm_adapter_bad_json_returns_none():
    adapter = LLMDecisionAdapter(manager_factory=lambda **kw: _FakeLLM("no json here"))
    assert await adapter.decide(state="s", questions=QUESTIONS, workspace_id=WS) is None


# --------------------------------------------------------------------------- #
# The engine: dials and the never-raises contract
# --------------------------------------------------------------------------- #


def test_dials_default_off_and_survive_a_broken_settings_read():
    def broken(category, key, default):
        raise RuntimeError("db down")

    engine = DecisionEngine(settings_reader=broken)
    d = engine.dials()
    assert d.classifier_mode == MODE_OFF and d.provider == "openrouter"
    assert d.timeout_seconds == 2.5 and d.min_confidence == 0.7 and d.any_on is False


def test_dials_parse_validate_and_cache_for_a_ttl():
    values = {"classifier_mode": "SHADOW", "provider": "llm", "timeout_seconds": "99", "min_confidence": "abc"}
    reads = []
    clock = {"t": 100.0}

    def reader(category, key, default):
        reads.append(key)
        assert category == "decision_engine"
        return values.get(key)

    engine = DecisionEngine(settings_reader=reader, clock=lambda: clock["t"])
    d = engine.dials()
    assert d.classifier_mode == MODE_SHADOW and d.provider == "llm"
    assert d.timeout_seconds == 30.0  # clamped to the ceiling
    assert d.min_confidence == 0.7  # unparseable → default
    n = len(reads)
    engine.dials()
    assert len(reads) == n  # cached
    clock["t"] += 31
    engine.dials()
    assert len(reads) > n  # refreshed after the TTL


class _Backend:
    provider = "fake"
    model = "fake-1"

    def __init__(self, result=None, exc=None, delay=0.0, timeout_s=0.05):
        self.result, self.exc, self.delay, self.timeout_s = result, exc, delay, timeout_s
        self.calls = []

    async def decide(self, **kw):
        self.calls.append(kw)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.exc:
            raise self.exc
        return self.result


def _engine(mode=MODE_OFF, backend=None, min_confidence=0.7):
    values = {"classifier_mode": mode, "min_confidence": str(min_confidence)}
    return DecisionEngine(
        settings_reader=lambda c, k, d: values.get(k),
        backend_factory=lambda dials: backend,
    )


@pytest.mark.asyncio
async def test_engine_never_raises_into_a_turn():
    assert await _engine(backend=None).decide(state="s", questions=QUESTIONS, workspace_id=WS) is None
    assert await _engine(backend=_Backend(exc=RuntimeError("boom"))).decide(
        state="s", questions=QUESTIONS, workspace_id=WS
    ) is None
    slow = _Backend(result=None, delay=1.5, timeout_s=0.05)
    assert await _engine(backend=slow).decide(state="s", questions=QUESTIONS, workspace_id=WS) is None


def test_record_shadow_appends_json_lines(monkeypatch, tmp_path):
    path = tmp_path / "nested" / "shadow.jsonl"
    monkeypatch.setattr(config, "DECISION_SHADOW_LOG_PATH", str(path))
    engine = _engine()
    engine.record_shadow({"purpose": "classifier", "tier": 3})
    engine.record_shadow({"purpose": "classifier", "tier": 2})
    from core.llm.usage_context import usage_scope

    with usage_scope(execution_id="sim:night1:s07:2"):
        engine.record_shadow({"purpose": "tool_rerank"})
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["tier"] for r in rows[:2]] == [3, 2] and all("ts" in r for r in rows)
    assert "execution_id" not in rows[0] and rows[2]["execution_id"] == "sim:night1:s07:2"

    from scripts.eval.decision_shadow import score as scorer

    assert [r["purpose"] for r in scorer.split_traffic(rows, "sim")] == ["tool_rerank"]
    assert len(scorer.split_traffic(rows, "real")) == 2 and len(scorer.split_traffic(rows, None)) == 3
    assert "tool_rerank rows=1" in scorer.summarize_rerank(scorer.split_traffic(rows, "sim"))
    assert scorer._build_parser().parse_args(["--only", "sim", "--purpose", "tool_rerank"]).only == "sim"
    # the customer-night window: cut by ts the way the night ledger does
    first_ts = rows[0]["ts"]
    assert len(scorer.filter_window(rows, since=first_ts, until=None)) == 3
    assert scorer.filter_window(rows, since=None, until=first_ts - 1) == []
    assert scorer.parse_when("2026-09-18T17:59:45Z") == scorer.parse_when("2026-09-18T18:59:45+01:00")
    assert scorer.parse_when("1758218385") == 1758218385.0 and scorer.parse_when(None) is None
    args = scorer._build_parser().parse_args(["--since", "2026-09-18T17:59:45Z", "--until", "1758300000"])
    assert scorer.parse_when(args.since) < scorer.parse_when(args.until)


# --------------------------------------------------------------------------- #
# The classifier questions (pure)
# --------------------------------------------------------------------------- #


def _result(**answers) -> DecisionResult:
    return DecisionResult(answers=answers, provider="fake", model="fake-1", latency_ms=310, input_tokens=900)


def _choice(option, p=0.9):
    return DecisionAnswer(type="choice", choice=option, probabilities={option: p}, confidence=p)


def _noul(p):
    return DecisionAnswer(type="noul", noul=p)


def test_build_questions_adds_the_roster_choice_only_when_agents_exist():
    assert "target_agent" not in auto_decisions.build_questions([])
    q = auto_decisions.build_questions(["Jim", "jim", "Atlas", "none", ""])
    assert q["target_agent"].options == ["Jim", "Atlas", "none"]
    assert set(q["complexity"].options) == {c.value for c in Complexity}
    # The deprecated "workflow" alias is normalised on read, never offered.
    assert set(q["action"].options) == {a.value for a in Action if a is not Action.WORKFLOW}


def test_verdict_respects_the_confidence_floor_and_maps_fields():
    result = _result(
        complexity=_choice("molecule", 0.8),
        action=_choice("assign", 0.9),
        tool_domain=_choice("platform", 0.6),
        needs_memory=_noul(0.2),
        needs_multi_agent=_noul(0.7),
        target_agent=_choice("Jim", 0.85),
    )
    verdict = auto_decisions.verdict_from_result(result, min_confidence=0.7)
    assert verdict == {
        "complexity": "molecule",
        "action": "assign",
        "confidence": 0.8,
        "needs_memory": False,
        "needs_multi_agent": True,
        "tool_hints": ["platform"],
        "target_agent_name": "Jim",
    }
    assert auto_decisions.verdict_from_result(result, min_confidence=0.85) is None
    assert auto_decisions.verdict_from_result(_result(complexity=_choice("atom")), min_confidence=0.1) is None
    assert auto_decisions.verdict_from_result(
        _result(complexity=_choice("weird"), action=_choice("respond")), min_confidence=0.1
    ) is None


def test_compare_flags_each_field():
    tier = {
        "complexity": "molecule", "action": "respond", "tool_hints": ["platform"],
        "needs_memory": False, "needs_multi_agent": False, "target_agent_name": None,
    }
    result = _result(
        complexity=_choice("molecule"), action=_choice("delegate"), tool_domain=_choice("platform"),
        needs_memory=_noul(0.1), needs_multi_agent=_noul(0.9), target_agent=_choice("none"),
    )
    assert auto_decisions.compare(tier, result) == {
        "complexity": True, "action": False, "needs_memory": True, "needs_multi_agent": False,
        "tool_domain": True, "target_agent": True,
    }
    assert auto_decisions.compare(tier, _result())["complexity"] is None


# --------------------------------------------------------------------------- #
# AutoBrain: off / shadow / live
# --------------------------------------------------------------------------- #


class _Query:
    def __init__(self, agents):
        self._agents = agents

    def filter(self, *a, **k):
        return self

    def limit(self, n):
        return self

    def all(self):
        return self._agents

    def first(self):
        return None


class _Session:
    def __init__(self, agents):
        self._agents = agents

    def query(self, model):
        return _Query(self._agents)


JIM = SimpleNamespace(id=7, name="Jim", role="writer", description="Drafts board packs", status="active", configuration={})

TIER3 = ComplexityAssessment(
    complexity=Complexity.MOLECULE, action=Action.RESPOND, reasoning="tier3", confidence=0.85,
    tool_hints=["platform"],
)


@pytest.fixture
def brain(monkeypatch):
    b = AutoBrain(_Session([JIM]), WS)
    monkeypatch.setattr(b, "_cache_lookup", lambda m: None)
    stored = []
    monkeypatch.setattr(b, "_cache_store", lambda m, a: stored.append(a))
    monkeypatch.setattr(b, "_onboarding_active", lambda: False)
    tier3_calls = []

    async def fake_tier3(message, n):
        tier3_calls.append(message)
        return TIER3

    monkeypatch.setattr(b, "_llm_classify", fake_tier3)
    b._test_stored, b._test_tier3 = stored, tier3_calls  # type: ignore[attr-defined]
    return b


def _install_engine(monkeypatch, tmp_path, mode, backend):
    monkeypatch.setattr(config, "DECISION_SHADOW_LOG_PATH", str(tmp_path / "shadow.jsonl"))
    engine = _engine(mode=mode, backend=backend)
    monkeypatch.setattr(auto_mod, "get_decision_engine", lambda: engine)
    return engine


async def _drain_shadow():
    pending = list(auto_mod._SHADOW_TASKS)
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    for _ in range(3):
        await asyncio.sleep(0)


def _shadow_rows(tmp_path):
    p = tmp_path / "shadow.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


DECIDED_ATOM = _result(
    complexity=_choice("atom", 0.95), action=_choice("respond", 0.92), tool_domain=_choice("none", 0.8),
    needs_memory=_noul(0.1), needs_multi_agent=_noul(0.05), target_agent=_choice("none", 0.9),
)


@pytest.mark.asyncio
async def test_mode_off_leaves_the_tiers_untouched(brain, monkeypatch, tmp_path):
    backend = _Backend(result=DECIDED_ATOM)
    _install_engine(monkeypatch, tmp_path, MODE_OFF, backend)
    verdict = await brain.assess("please draft the board pack for thursday", 3)
    await _drain_shadow()
    assert verdict is TIER3 and brain._test_tier3 == ["please draft the board pack for thursday"]
    assert backend.calls == [] and _shadow_rows(tmp_path) == []


@pytest.mark.asyncio
async def test_shadow_mode_never_changes_the_verdict_and_writes_the_comparison(brain, monkeypatch, tmp_path):
    backend = _Backend(result=DECIDED_ATOM)
    _install_engine(monkeypatch, tmp_path, MODE_SHADOW, backend)

    verdict = await brain.assess("please draft the board pack for thursday", 3)
    assert verdict is TIER3  # Tier 3 still answers, untouched
    await _drain_shadow()

    rows = _shadow_rows(tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["purpose"] == "classifier" and row["tier"] == 3 and row["workspace_id"] == WS
    assert row["verdict"]["complexity"] == "molecule" and row["answers"]["complexity"]["choice"] == "atom"
    assert row["agree"]["complexity"] is False and row["agree"]["action"] is True
    assert row["latency_ms"] == 310 and row["provider"] == "fake"
    assert "message_preview" in row and row["message_sha"]
    # the engine saw the roster, not just the text
    state = backend.calls[0]["state"]
    assert state["agents"] == [{"name": "Jim", "role": "writer", "description": "Drafts board packs"}]
    assert "target_agent" in backend.calls[0]["questions"]


@pytest.mark.asyncio
async def test_shadow_runs_beside_the_regex_tier_too(brain, monkeypatch, tmp_path):
    backend = _Backend(result=DECIDED_ATOM)
    _install_engine(monkeypatch, tmp_path, MODE_SHADOW, backend)
    verdict = await brain.assess("list my agents", 1)
    await _drain_shadow()
    assert verdict.complexity == Complexity.MOLECULE and brain._test_tier3 == []
    rows = _shadow_rows(tmp_path)
    assert len(rows) == 1 and rows[0]["tier"] == 2


@pytest.mark.asyncio
async def test_shadow_failure_is_recorded_not_raised(brain, monkeypatch, tmp_path):
    _install_engine(monkeypatch, tmp_path, MODE_SHADOW, _Backend(exc=RuntimeError("down")))
    verdict = await brain.assess("please draft the board pack for thursday", 3)
    await _drain_shadow()
    assert verdict is TIER3
    rows = _shadow_rows(tmp_path)
    assert len(rows) == 1 and rows[0]["error"] == "no_result"


@pytest.mark.asyncio
async def test_live_mode_replaces_tier3_above_the_floor(brain, monkeypatch, tmp_path):
    backend = _Backend(result=DECIDED_ATOM)
    _install_engine(monkeypatch, tmp_path, MODE_LIVE, backend)
    verdict = await brain.assess("what do you think about tuesday's numbers", 2)
    assert brain._test_tier3 == []
    assert verdict.complexity == Complexity.ATOM and verdict.action == Action.RESPOND
    assert verdict.confidence == pytest.approx(0.92) and verdict.tool_hints == []
    assert "fake/fake-1" in verdict.reasoning
    assert brain._test_stored == [verdict]  # cached like a Tier-3 verdict
    assert _shadow_rows(tmp_path) == []  # live mode writes no shadow rows


@pytest.mark.asyncio
async def test_live_mode_below_the_floor_or_without_a_result_runs_tier3(brain, monkeypatch, tmp_path):
    unsure = _result(complexity=_choice("atom", 0.55), action=_choice("respond", 0.9))
    _install_engine(monkeypatch, tmp_path, MODE_LIVE, _Backend(result=unsure))
    assert await brain.assess("please draft the board pack for thursday", 3) is TIER3
    _install_engine(monkeypatch, tmp_path, MODE_LIVE, _Backend(result=None))
    assert await brain.assess("please draft the board pack for thursday", 3) is TIER3
    assert len(brain._test_tier3) == 2


@pytest.mark.asyncio
async def test_live_assign_resolves_the_named_agent_against_the_roster(brain, monkeypatch, tmp_path):
    decided = _result(
        complexity=_choice("molecule", 0.9), action=_choice("assign", 0.9), tool_domain=_choice("platform", 0.7),
        needs_memory=_noul(0.1), needs_multi_agent=_noul(0.1), target_agent=_choice("Jim", 0.9),
    )
    _install_engine(monkeypatch, tmp_path, MODE_LIVE, _Backend(result=decided))
    verdict = await brain.assess("Get Jim to draft the board pack, no rush", 4)
    assert verdict.action == Action.ASSIGN
    assert verdict.target_agent_id == 7 and verdict.target_agent_name == "Jim"
    assert verdict.tool_hints == ["platform"]
