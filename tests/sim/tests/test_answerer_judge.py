"""The answerer, the judge's parsing, the drivers' safety net, the plist — no network."""

from pathlib import Path

from tests.sim import answerer, judge, schedule
from tests.sim.api import ApiError, Trace
from tests.sim.config import Settings
from tests.sim.driver import run_scenario
from tests.sim.driver_chat import turn_record
from tests.sim.driver_task import _denials
from tests.sim.packs import Scenario
from tests.sim.results import RunContext
from tests.sim.sse import parse_data_stream


class FakeApi:
    def __init__(self, grants=None, fail_post=False):
        self.grants, self.fail_post, self.posts = grants or [], fail_post, []

    def get(self, path, **kw):
        return {"grants": self.grants}

    def post(self, path, body=None, **kw):
        if self.fail_post:
            raise ApiError("POST", path, 500, "down")
        self.posts.append((path, body))
        return {}


def test_pick_answer_prefers_options_then_keywords_then_default():
    p = answerer.DEFAULT_PERSONA
    assert "50 pounds" in answerer.pick_answer(p, "What budget should I keep to?")["answer_text"]
    assert answerer.pick_answer(p, "Which format: PDF or markdown?", ["PDF", "Markdown file"]) == {"option": "PDF"}
    assert answerer.pick_answer(p, "Anything else?", ["Proceed", "Stop"]) == {"option": "Proceed"}
    assert answerer.pick_answer(p, "Something with no keyword at all")["answer_text"] == p["default_answer"]


def test_persona_loading_falls_back_and_reads_the_local_file(tmp_path):
    assert answerer.load_persona("nope", tmp_path / "missing.toml") is answerer.DEFAULT_PERSONA
    f = tmp_path / "personas.toml"
    f.write_text('[personas.x]\ndefault_answer = "fine"\n[[personas.x.answers]]\nmatch = ["a"]\nanswer = "b"\n', encoding="utf-8")
    p = answerer.load_persona("x", f)
    assert p["name"] == "x" and p["default_answer"] == "fine" and p["answers"][0]["answer"] == "b"
    assert answerer.load_persona("y", f) is answerer.DEFAULT_PERSONA


def test_answer_pending_answers_questions_and_grants_the_rest():
    api = FakeApi([{"id": 1, "kind": "question", "question": "What budget do I have?"},
                   {"id": 2, "kind": "board_task", "reason": "run this ticket"}, {"nope": True}])
    decisions = answerer.answer_pending(api, answerer.DEFAULT_PERSONA)
    assert [d["action"] for d in decisions] == ["answered", "granted"]
    assert api.posts[0][0] == "/api/v1/approval-grants/1/answer" and "50 pounds" in api.posts[0][1]["answer_text"]
    assert api.posts[1][0] == "/api/v1/approval-grants/2/grant"
    held = answerer.answer_pending(FakeApi([{"id": 3, "kind": "board_task"}]), answerer.DEFAULT_PERSONA, auto_grant=False)
    assert held[0]["action"] == "left"
    failed = answerer.answer_pending(FakeApi([{"id": 4, "kind": "question", "question": "x"}], fail_post=True), answerer.DEFAULT_PERSONA)
    assert failed[0]["action"] == "error" and failed[0]["status"] == 500


def test_question_text_joins_whatever_field_carried_it():
    assert answerer.question_text({"prompt": "A", "context": {"summary": "B"}, "id": 1}) == "A B"


def test_verdict_parsing_is_tolerant_and_clamped():
    assert judge.parse_verdict('```json\n{"quality": 7, "usefulness": "2", "reasons": ["ok"]}\n```') == {"quality": 5, "usefulness": 2, "reasons": ["ok"]}
    assert judge.parse_verdict("no json here") is None
    assert judge.parse_verdict('{"quality": "high"}') is None
    assert judge.cache_key("m", "b", "e", "o") == judge.cache_key("m", "b", "e", "o") != judge.cache_key("m", "b", "e", "p")


def test_judge_is_skipped_without_a_key_and_rules_on_empty_output(tmp_path):
    assert judge.judge_output(Settings(), brief="b", expect="e", output="o", cache_dir=tmp_path) is None
    keyed = Settings(openrouter_api_key="sk-or-x")
    assert judge.judge_available(keyed) and not judge.judge_available(Settings(openrouter_api_key="sk-or-x", judge=False))
    assert judge.judge_output(keyed, brief="b", expect="e", output="   ", cache_dir=tmp_path)["source"] == "rule"


def test_a_crashing_scenario_becomes_an_error_result_not_an_exception():
    class Boom:
        def post(self, *a, **k):
            raise RuntimeError("no platform here")

    ctx = RunContext(settings=Settings(), api=Boom(), trace=Trace(), workspace_id="w", agents={}, persona={}, run_started="t")
    res = run_scenario(ctx, Scenario(id="x", kind="crud", steps=("create",)))
    assert res.outcome == "error" and res.ok is False and res.errors and "HTTP" not in res.errors[0]
    assert "RuntimeError" in res.errors[0]


def test_task_and_chat_records():
    assert _denials({"runtime_ref": {"session_denials": [1, 2]}}) == ("runtime_ref.session_denials: 2 entries",)
    assert _denials({"runtime_ref": None}) == ()
    turn = parse_data_stream('9:{"toolCallId":"1","toolName":"t","args":{"x":"' + "y" * 3000 + '"}}\n0:"hi"')
    rec = turn_record(1, "prompt", turn, 200, 12, 3)
    assert rec["tool_names"] == ["t"] and rec["tool_calls"][0]["args"].endswith("]") and len(rec["tool_calls"][0]["args"]) < 2100
    assert rec["text"] == "hi" and rec["errors"] == []


def test_launchd_plist_shape():
    plist = schedule.build_plist("agents", 1, 30, python="/usr/bin/python3", repo_root=Path("/repo"), extra=("--budget", "2"))
    args = plist["ProgramArguments"]
    assert args[-7:] == ["-m", "tests.sim.night", "run", "--pack", "agents", "--budget", "2"]
    assert plist["WorkingDirectory"] == "/repo" and plist["StartCalendarInterval"] == {"Hour": 1, "Minute": 30}
    assert plist["Label"] == "app.automatos.sim" and "/usr/bin" in plist["EnvironmentVariables"]["PATH"]
    assert plist["RunAtLoad"] is False
