"""PRD-268 US-C13b (§6.7, K14): the triage, silence and reply chips runners and the command line.

A fake model stands in for the platform's, and the cases are inline, written to ``tmp_path`` (never
a gold case, never a file in the repository). Each runner passes with a model that answers right,
misses with exit 1 with one that does not, and exits 2 when no gold set can be read; the printed
output names the bar. The runners build their models on classified purposes, read every case before
the first model call, and write nothing: the check-in's suggestions are captured, never run, and the
chips turn runs the tool's own handler in the chat's own tool loop.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

from config import config
from core.llm.clients.base import LLMResponse
from core.llm.decisions.questions import TYPE_CHOICE, TYPE_NOUL, DecisionAnswer, DecisionResult
from evals.proactive import reply_chips, silence, triage
from evals.proactive.__main__ import EVALS, main
from evals.proactive.bars import BAR_TEXT
from evals.proactive.gold import GoldSetUnavailable
from evals.proactive.scoring import ACT_TODAY, CAN_WAIT
from services.credits.classification import CUSTOMER_PURPOSES, OVERHEAD_PURPOSES
from services.proactive.prompts import CHECK_IN_RULES

WORKSPACE = str(uuid4())
URGENT = "Card payments fail for every customer"
REASON = "Card payments have failed for every customer since 09:00."
QUESTION = "Shall I book the 15:10 slot or the 16:00 one?"
CHIPS = ["15:10", "16:00"]
ANSWER = "Paris is the capital of France."
SUGGESTION = {"title": "Reconnect Slack", "detail": "Slack has answered 401 on every call since 09:00.",
              "evidence": [{"type": "run", "id": "77"}], "proposed_action": {"kind": "chat"}}
UNTIL = "2026-10-30T09:30:00+00:00"


@pytest.fixture
def gold(tmp_path, monkeypatch):
    """A gold folder outside the repository, and the eval's workspace."""
    folder = tmp_path / "gold"
    folder.mkdir()
    monkeypatch.setattr(config, "PROACTIVE_EVAL_GOLD_DIR", str(folder))
    monkeypatch.setattr(config, "PROACTIVE_EVAL_WORKSPACE_ID", WORKSPACE)
    return folder


def _write(folder, name, cases):
    (folder / f"{name}.jsonl").write_text("".join(json.dumps(case) + "\n" for case in cases), encoding="utf-8")


def _run(capsys, *argv):
    code = main(list(argv))
    return code, capsys.readouterr().out


# ── triage ───────────────────────────────────────────────────────────────────

TRIAGE_CASES = [
    {"expected": "act_today", "kind": "report", "title": URGENT, "summary": "Stripe declines every card.",
     "status": "error", "agent_name": "OPS", "report_type": "incident"},
    {"expected": "can_wait", "kind": "heartbeat:agent", "title": "Weekly digest sent",
     "summary": "18 subscribers, no bounces.", "status": "ok", "agent_name": "OPS"},
]


class _TriageModel:
    """Answers the judgement as PRD-248's LLM adapter asks it, then the reason; ``right`` False says
    'act today' to everything."""

    def __init__(self, right=True, reason=REASON):
        self.right, self.reason, self.prompts, self.tools = right, reason, [], []

    async def generate_response(self, messages, tools=None):
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        self.tools.append(tools)
        if "QUESTIONS (JSON" not in prompt:
            return LLMResponse(content=self.reason)
        yes = URGENT in prompt or not self.right
        return LLMResponse(content=json.dumps({"needs_attention": {"noul": 0.9 if yes else 0.1},
                                               "severity": {"choice": "urgent" if yes else "fyi"}}))


@pytest.fixture
def triage_model(monkeypatch):
    def install(model):
        monkeypatch.setattr(triage, "triage_llm", lambda workspace_id: model)
        return model
    return install


def test_triage_passes_when_the_judgement_is_right(gold, triage_model, capsys):
    _write(gold, "triage", TRIAGE_CASES)
    model = triage_model(_TriageModel())
    code, out = _run(capsys, "triage")
    assert code == 0
    assert "triage: PASS" in out and BAR_TEXT["triage"] in out
    assert "recall: 1.000" in out and "precision: 1.000" in out and "bad_reasons: 0" in out
    # The judgement's own questions, over the report's state; then one reason per verdict, no tools.
    judged = [p for p in model.prompts if "QUESTIONS (JSON" in p]
    assert len(judged) == 2 and all('"needs_attention"' in p and '"severity"' in p for p in judged)
    reasons = [p for p in model.prompts if p.startswith("A report was judged")]
    assert len(reasons) == 2 and "act on it today" in reasons[0] and "it can wait" in reasons[1]
    assert all(tools is None for tools in model.tools)


def test_triage_misses_with_exit_1_when_it_says_act_today_to_everything(gold, triage_model, capsys):
    _write(gold, "triage", TRIAGE_CASES)
    triage_model(_TriageModel(right=False))
    code, out = _run(capsys, "triage")
    assert code == 1
    assert "triage: MISS" in out and BAR_TEXT["triage"] in out and "precision: 0.500" in out


def test_a_reason_over_120_characters_is_a_miss(gold, triage_model, capsys):
    _write(gold, "triage", TRIAGE_CASES)
    triage_model(_TriageModel(reason="x" * 121))
    code, out = _run(capsys, "triage")
    assert code == 1 and "bad_reasons: 2" in out


def test_triage_scores_as_json(gold, triage_model, capsys):
    _write(gold, "triage", TRIAGE_CASES)
    triage_model(_TriageModel())
    code, out = _run(capsys, "triage", "--json")
    score = json.loads(out)
    assert code == 0 and score["eval"] == "triage" and score["verdict"] == "PASS"
    assert score["bar"] == BAR_TEXT["triage"] and score["numbers"]["recall"] == 1.0


def test_a_judgement_call_that_never_answered_exits_2_not_a_miss(gold, triage_model, capsys):
    """The adapter turns a failed call into no result; the eval says it could not run."""
    _write(gold, "triage", TRIAGE_CASES)

    class _Down(_TriageModel):
        async def generate_response(self, messages, tools=None):
            raise ConnectionError("provider down")

    triage_model(_Down())
    code, out = _run(capsys, "triage")
    assert code == 2 and "triage: CANNOT RUN" in out and "case 1 failed or timed out" in out


def _result(**answers):
    return DecisionResult(answers=answers, provider="llm", model="fake", latency_ms=1)


def test_the_verdict_is_needs_attention_else_the_severity():
    assert triage.verdict_of(_result(needs_attention=DecisionAnswer(TYPE_NOUL, noul=0.5))) == ACT_TODAY
    assert triage.verdict_of(_result(needs_attention=DecisionAnswer(TYPE_NOUL, noul=0.49))) == CAN_WAIT
    # needs_attention decides even against the severity.
    assert triage.verdict_of(_result(needs_attention=DecisionAnswer(TYPE_NOUL, noul=0.2),
                                     severity=DecisionAnswer(TYPE_CHOICE, choice="urgent"))) == CAN_WAIT
    assert triage.verdict_of(_result(severity=DecisionAnswer(TYPE_CHOICE, choice="urgent"))) == ACT_TODAY
    assert triage.verdict_of(_result(severity=DecisionAnswer(TYPE_CHOICE, choice="needs_a_decision"))) == CAN_WAIT
    assert triage.verdict_of(_result()) is None and triage.verdict_of(None) is None


def test_no_verdict_asks_no_reason_and_never_passes(gold, triage_model, capsys):
    _write(gold, "triage", TRIAGE_CASES)

    class _Silent(_TriageModel):
        async def generate_response(self, messages, tools=None):
            self.prompts.append(messages[-1]["content"])
            return LLMResponse(content="I cannot say.")

    model = triage_model(_Silent())
    code, out = _run(capsys, "triage")
    assert code == 1 and "no_verdict: 2" in out
    assert not [p for p in model.prompts if p.startswith("A report was judged")]


# ── check-in silence ─────────────────────────────────────────────────────────

def _signal(kind, ref_type, ref_id, summary):
    return {"kind": kind, "ref": {"type": ref_type, "id": ref_id}, "at": "2026-10-30T09:00:00+00:00",
            "summary": summary}


QUIET = {"signals": [_signal("ticket_failed", "ticket", "0846", "The nightly export retried and passed")],
         "since": "2026-10-29T09:30:00+00:00", "until": UNTIL}
PLANTED = {"signals": [_signal("heartbeat_error", "run", "77", "Slack answered 401 on every call")],
           "need": ["Slack", "401"], "since": "2026-10-29T09:30:00+00:00", "until": UNTIL}
NOTHING = {"signals": [], "until": UNTIL}


def _suggest(**arguments):
    return {"id": "call_suggest", "type": "function",
            "function": {"name": "platform_suggest_next_step", "arguments": json.dumps(arguments)}}


class _CheckInModel:
    """Suggests on the planted need; ``right`` False suggests on every snapshot."""

    def __init__(self, right=True):
        self.right, self.calls = right, []

    async def generate_response(self, messages, tools=None):
        self.calls.append(SimpleNamespace(messages=messages, tools=tools))
        if "401" in messages[0]["content"] or not self.right:
            return LLMResponse(content="", tool_calls=[_suggest(**SUGGESTION)])
        return LLMResponse(content="Nothing new needs the owner.")


@pytest.fixture
def check_in_model(monkeypatch):
    def install(model):
        monkeypatch.setattr(silence, "heartbeat_llm", lambda workspace_id, agent_id: model)
        return model
    return install


def _never_run(monkeypatch):
    """A captured suggestion never reaches the platform's executor: nothing is written."""
    from modules.tools.discovery import platform_executor

    async def refuse(*args, **kwargs):
        raise AssertionError("a captured suggestion must never run")

    monkeypatch.setattr(platform_executor.PlatformActionExecutor, "execute", refuse)


def test_silence_passes_when_only_the_planted_need_is_named(gold, check_in_model, monkeypatch, capsys):
    _write(gold, "silence", [QUIET, PLANTED, NOTHING])
    _never_run(monkeypatch)
    model = check_in_model(_CheckInModel())
    code, out = _run(capsys, "silence")
    assert code == 0
    assert "silence: PASS" in out and BAR_TEXT["silence"] in out
    assert "quiet_silent: 1.000" in out and "planted_named: 1.000" in out and "duplicates: 0" in out
    # A snapshot with no signal is silent with no model call, as W1's runner makes it.
    assert len(model.calls) == 2
    first = model.calls[0]
    # W1's prompt and its one message; only the suggestion tool's schema is offered.
    assert first.messages[0]["role"] == "system" and CHECK_IN_RULES in first.messages[0]["content"]
    assert "Auto's workspace check-in for what changed from 2026-10-29 09:30 UTC" in first.messages[0]["content"]
    assert first.messages[1] == {"role": "user", "content": "Run the check-in now. Time: 2026-10-30 09:30 UTC."}
    assert [tool["function"]["name"] for tool in first.tools] == ["platform_suggest_next_step"]
    assert first.tools[0]["function"]["parameters"]["required"] == ["title", "evidence", "proposed_action"]


def test_silence_misses_with_exit_1_when_a_quiet_snapshot_suggests(gold, check_in_model, capsys):
    _write(gold, "silence", [QUIET, PLANTED])
    check_in_model(_CheckInModel(right=False))
    code, out = _run(capsys, "silence")
    assert code == 1 and "silence: MISS" in out and BAR_TEXT["silence"] in out and "quiet_silent: 0.000" in out


def test_a_suggestion_that_repeats_an_open_title_is_a_duplicate(gold, check_in_model, capsys):
    _write(gold, "silence", [QUIET, {**PLANTED, "open_titles": ["reconnect slack."]}])
    check_in_model(_CheckInModel())
    code, out = _run(capsys, "silence")
    assert code == 1 and "duplicates: 1" in out


def test_a_signal_s_ref_is_read_as_given():
    signal = silence._signal({"kind": "heartbeat_error", "ref": {"type": "run", "id": 0}, "at": UNTIL}, 1)
    assert signal.ref == ("run", "0") and signal.subject_key == "run:0" and signal.at.isoformat() == UNTIL
    for ref in ({"type": "run", "id": None}, {"type": " ", "id": "7"}, {"type": "run"}):
        with pytest.raises(GoldSetUnavailable):
            silence._signal({"kind": "heartbeat_error", "ref": ref, "at": UNTIL}, 1)


def test_only_the_calls_the_loop_would_run_are_counted():
    from services.proactive.loop import MAX_CALLS_PER_TURN

    calls = [_suggest(title=f"Suggestion {n}") for n in range(MAX_CALLS_PER_TURN + 2)]
    assert len(silence.suggestions_in(LLMResponse(content="", tool_calls=calls))) == MAX_CALLS_PER_TURN


def test_each_suggestion_call_is_read_as_its_title_and_detail():
    response = LLMResponse(content="", tool_calls=[_suggest(**SUGGESTION), _suggest(title="Second")])
    assert [(s.title, s.detail) for s in silence.suggestions_in(response)] == [
        ("Reconnect Slack", SUGGESTION["detail"]), ("Second", "")]
    assert silence.suggestions_in(LLMResponse(content="Nothing.")) == ()


# ── reply chips ──────────────────────────────────────────────────────────────

CHOICE = {"choice": True, "ask": "Book a slot for the delivery call on Friday."}
OPEN = {"choice": False, "ask": "What is the capital of France?",
        "history": [{"role": "assistant", "content": "Ask me anything."}]}


def _offer(replies):
    return {"id": "call_offer", "type": "function",
            "function": {"name": "platform_offer_replies", "arguments": json.dumps({"replies": replies})}}


class _ChatModel:
    """Asks and offers chips on the slot question; ``chips_on_open`` offers them on the open one too."""

    def __init__(self, chips_on_open=False, replies=CHIPS):
        self.chips_on_open, self.replies, self.calls, self.tools = chips_on_open, replies, 0, []

    async def generate_response(self, messages, tools=None):
        self.calls += 1
        self.tools.append(tools)
        last = messages[-1]
        if last["role"] == "tool":
            return LLMResponse(content=QUESTION)
        if "slot" in last["content"] or self.chips_on_open:
            return LLMResponse(content=QUESTION, tool_calls=[_offer(self.replies)])
        return LLMResponse(content=ANSWER)


@pytest.fixture
def chat_model(monkeypatch):
    def install(model):
        monkeypatch.setattr(reply_chips, "chat_llm", lambda workspace_id: model)
        return model
    return install


def test_reply_chips_pass_on_the_choice_question_only(gold, chat_model, capsys):
    _write(gold, "reply_chips", [CHOICE, OPEN])
    model = chat_model(_ChatModel())
    code, out = _run(capsys, "reply_chips")
    assert code == 0
    assert "reply_chips: PASS" in out and BAR_TEXT["reply_chips"] in out
    assert "on_choice: 1.000" in out and "on_open: 0.000" in out and "out_of_limits: 0" in out
    # The chat's own loop ends the chips turn on its round's text: one model call per question.
    assert model.calls == 2
    assert all([tool["function"]["name"] for tool in tools] == ["platform_offer_replies"] for tools in model.tools)


def test_reply_chips_miss_with_exit_1_when_an_open_question_gets_chips(gold, chat_model, capsys):
    _write(gold, "reply_chips", [CHOICE, OPEN])
    chat_model(_ChatModel(chips_on_open=True))
    code, out = _run(capsys, "reply_chips")
    assert code == 1 and "reply_chips: MISS" in out and BAR_TEXT["reply_chips"] in out and "on_open: 1.000" in out


def test_chips_the_tool_refused_are_outside_the_limits(gold, chat_model, capsys):
    _write(gold, "reply_chips", [CHOICE, OPEN])
    model = chat_model(_ChatModel(replies=["Book the 15:10 slot please", "16:00"]))
    code, out = _run(capsys, "reply_chips")
    assert code == 1 and "on_choice: 0.000" in out and "out_of_limits: 1" in out
    # The refusal is answered and the model asked again, as in chat.
    assert model.calls == 3


def test_an_open_turn_carries_its_history_and_offers_nothing():
    model = _ChatModel()
    question = reply_chips._question(OPEN, 1)
    tools = ({"type": "function", "function": {"name": "platform_offer_replies", "parameters": {}}},)
    turn = asyncio.run(reply_chips.ask(model, question, tools, WORKSPACE))
    assert turn.answer == ANSWER and turn.replies is None and turn.choice is False
    assert [m["role"] for m in question.messages] == ["assistant", "user"]
    assert question.messages[-1] == {"role": "user", "content": "What is the capital of France?"}


# ── the command line ─────────────────────────────────────────────────────────

RUNNERS = [("triage", TRIAGE_CASES), ("silence", [QUIET]), ("reply_chips", [CHOICE])]


@pytest.mark.parametrize("name", [name for name, _ in RUNNERS])
def test_no_gold_directory_exits_2_and_names_the_bar(name, monkeypatch, capsys):
    monkeypatch.setattr(config, "PROACTIVE_EVAL_GOLD_DIR", "")
    monkeypatch.setattr(config, "PROACTIVE_EVAL_WORKSPACE_ID", WORKSPACE)
    code, out = _run(capsys, name)
    assert code == 2
    assert f"{name}: CANNOT RUN" in out and BAR_TEXT[name] in out and "PROACTIVE_EVAL_GOLD_DIR" in out


@pytest.mark.parametrize("name", [name for name, _ in RUNNERS])
def test_a_missing_gold_file_exits_2(name, gold, capsys):
    code, out = _run(capsys, name)
    assert code == 2 and f"{name}.jsonl is missing" in out


@pytest.mark.parametrize("name, cases", RUNNERS)
def test_no_workspace_exits_2_before_any_model_call(name, cases, gold, monkeypatch, capsys):
    _write(gold, name, cases)
    monkeypatch.setattr(config, "PROACTIVE_EVAL_WORKSPACE_ID", "")
    code, out = _run(capsys, name)
    assert code == 2 and "PROACTIVE_EVAL_WORKSPACE_ID is not set" in out


@pytest.mark.parametrize("name, bad, said", [
    ("triage", {"expected": "maybe", "title": "t"}, "triage case 1 is labelled 'maybe'"),
    ("silence", {"signals": [{"kind": "ticket_failed"}]}, "silence case 1: each signal has a kind and a ref"),
    ("silence", {"signals": [], "need": ["!!"]}, "silence case 1:"),
    ("silence", {"signals": [], "need": ["Slack"]}, "silence case 1: a planted need comes with the signal"),
    ("reply_chips", {"choice": "yes", "ask": "Book it"}, "reply_chips case 1: choice is true"),
    ("reply_chips", {"choice": True, "ask": "  "}, "reply_chips case 1: ask is the person's message"),
])
def test_a_malformed_case_exits_2_before_any_model_call(name, bad, said, gold, monkeypatch, capsys):
    _write(gold, name, [bad])
    calls = []
    for module, attr in ((triage, "triage_llm"), (reply_chips, "chat_llm")):
        monkeypatch.setattr(module, attr, lambda *args: calls.append(args))
    monkeypatch.setattr(silence, "heartbeat_llm", lambda *args: calls.append(args))
    code, out = _run(capsys, name)
    assert code == 2 and said in out and calls == []


def test_a_failed_model_call_exits_2(gold, check_in_model, capsys):
    _write(gold, "silence", [QUIET])

    class _Down:
        async def generate_response(self, messages, tools=None):
            raise ConnectionError("provider down")

    check_in_model(_Down())
    code, out = _run(capsys, "silence")
    assert code == 2 and "the run failed (ConnectionError)" in out


def test_the_command_line_takes_the_six_evals_and_nothing_else(capsys):
    assert set(EVALS) == set(BAR_TEXT) and len(EVALS) == 6
    with pytest.raises(SystemExit) as stopped:
        main(["nonsense"])
    assert stopped.value.code == 2


# ── the models: the platform's orchestrator_llm, on classified purposes ──────

def test_every_runner_model_is_on_a_classified_purpose(monkeypatch):
    from core.llm import manager
    from core.llm.decisions.llm_adapter import SERVICE_NAME
    from services.proactive.runner import CHECK_IN_PURPOSE, HEARTBEAT_SERVICE

    built = []
    monkeypatch.setattr(manager, "LLMManager", lambda **kwargs: built.append(kwargs) or kwargs)
    triage.triage_llm(WORKSPACE)
    reply_chips.chat_llm(WORKSPACE)
    assert built == [
        {"service_name": "orchestrator", "workspace_id": WORKSPACE, "request_type": SERVICE_NAME},
        {"service_name": "orchestrator", "workspace_id": WORKSPACE, "request_type": "orchestrator"},
    ]
    # The judgement's own booking is overhead; the check-in's and Auto's own are customer work.
    assert triage.JUDGEMENT_PURPOSE in OVERHEAD_PURPOSES
    assert CHECK_IN_PURPOSE in CUSTOMER_PURPOSES and reply_chips.CHAT_PURPOSE in CUSTOMER_PURPOSES
    # All three run on the orchestrator_llm tier (the heartbeat service's is the check-in's).
    tiers = {manager.SERVICE_CATEGORY_MAP[s] for s in (triage.AUTO_SERVICE, reply_chips.AUTO_SERVICE, HEARTBEAT_SERVICE)}
    assert tiers == {"orchestrator_llm"}
