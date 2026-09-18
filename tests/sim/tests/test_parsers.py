"""Pure parsing and config: no network, no docker, no platform."""

import pytest

from tests.sim import config, cost, sse
from tests.sim.api import Api, Trace, chat_body, items_of
from tests.sim.results import ScenarioResult, effect_checks, must_contain_checks, parse_iso, text_of
from tests.sim import workspace
from tests.sim.workspace import (WorkspaceError, apply_llm_settings, assert_scoped, parse_key_values, provision, purge,
                                 restore_llm_settings, seed_agents)
from tests.sim.packs import AgentSpec

STREAM = "\n".join([
    '0:"Hello "',
    '0:"world"',
    '9:{"toolCallId":"c1","toolName":"list_agents","args":{"limit":5}}',
    'a:{"toolCallId":"c1","result":{"agents":[]}}',
    '2:[{"chatId":"chat-123"}]',
    '3:"boom"',
    'd:{"finishReason":"stop","usage":{"promptTokens":10,"completionTokens":5}}',
    "garbage line without a prefix",
    "x:{}",
])


def test_data_stream_frames_are_all_kept():
    turn = sse.parse_data_stream(STREAM)
    assert turn.text == "Hello world"
    assert turn.tool_names == ("list_agents",)
    assert turn.tool_calls[0]["args"] == {"limit": 5}
    assert len(turn.tool_results) == 1
    assert turn.chat_id == "chat-123"
    assert turn.errors == ("boom",)
    assert turn.finish["finishReason"] == "stop"
    assert turn.usage["promptTokens"] == 10
    assert turn.frames == 7
    assert turn.unparsed == 2


def test_header_chat_id_wins_and_sse_envelope_is_tolerated():
    turn = sse.parse_data_stream('data: 0:"x"\n2:[{"chatId":"c"}]', header_chat_id="hdr")
    assert turn.text == "x"
    assert turn.chat_id == "hdr"
    assert sse.find_chat_id({"foo": {"session_id": "s1"}}) == "s1"
    assert sse.find_chat_id([{"nope": 1}, {"chat_id": "z"}]) == "z"
    assert sse.parse_data_stream("").text == ""


def test_settings_from_env_and_bad_values_name_the_variable():
    s = config.settings_from_mapping({"SIM_BUDGET_USD": "2.5", "SIM_KEEP_WORKSPACE": "true", "SIM_MODEL_ID": "x/y",
                                      "UNRELATED": "1", "SIM_JUDGE": ""})
    assert (s.budget_usd, s.keep_workspace, s.model_id, s.judge) == (2.5, True, "x/y", True)
    with pytest.raises(config.ConfigError, match="SIM_POLL_S"):
        config.settings_from_mapping({"SIM_POLL_S": "abc"})


def test_remote_targets_are_refused_unless_allowed():
    config.check_target(config.Settings(api_url="http://localhost:8000"))
    config.check_target(config.Settings(api_url="http://127.0.0.1:8000"))
    with pytest.raises(config.ConfigError, match="not the local stack"):
        config.check_target(config.Settings(api_url="https://api.example.com"))
    config.check_target(config.Settings(api_url="https://api.example.com", allow_remote=True))


def test_no_key_means_no_key_header_and_the_docker_target_is_checked():
    anon = Api("http://localhost:8000", None, "ws-1", Trace())._headers()
    assert "X-Api-Key" not in anon and anon["X-Workspace-ID"] == "ws-1"
    keyed = Api("http://localhost:8000", "static", "ws-1", Trace())._headers({"Accept": "text/event-stream"})
    assert keyed["X-Api-Key"] == "static" and keyed["Accept"] == "text/event-stream"
    s = config.settings_from_mapping({"SIM_API_KEY": "static"})
    assert s.api_key == "static" and config.public_settings(s)["api_key"] == "set"
    assert config.check_docker_target(config.Settings(), "") == ""
    assert config.check_docker_target(config.Settings(), "unix:///var/run/docker.sock").startswith("unix://")
    with pytest.raises(config.ConfigError, match="DOCKER_HOST"):
        config.check_docker_target(config.Settings(), "tcp://prod-host:2376")
    config.check_docker_target(config.Settings(allow_remote=True), "tcp://prod-host:2376")


def test_chat_body_mints_a_client_side_id_and_reuses_it():
    body, used = chat_body("hello")
    assert body["id"] == used == body["chatId"] and len(used) == 36 and body["message"]["parts"][0]["text"] == "hello"
    body2, used2 = chat_body("again", chat_id=used, agent_id=7)
    assert used2 == used and body2["id"] == used and body2["agentId"] == 7


def test_seeding_stops_when_the_workspace_header_is_not_honoured():
    class Echo:
        workspace_id = "sim-ws"

        def __init__(self, echoed):
            self.echoed, self.posts = echoed, 0

        def post(self, path, body=None, **kw):
            self.posts += 1
            return {"id": self.posts, "name": body["name"], "workspace_id": self.echoed}

    good = Echo("sim-ws")
    created = seed_agents(good, [AgentSpec(key="a", name="A"), AgentSpec(key="b", name="B")])
    assert set(created) == {"a", "b"} and good.posts == 2
    bad = Echo("other-ws")
    with pytest.raises(WorkspaceError, match="X-Workspace-ID is not being honoured"):
        seed_agents(bad, [AgentSpec(key="a", name="A"), AgentSpec(key="b", name="B")])
    assert bad.posts == 1  # stopped at the first row
    assert_scoped(good, {"id": 1}, "row without a workspace field")  # tolerated: nothing to compare


def test_llm_rows_with_null_values_are_neither_applied_nor_restored():
    class Bulk:
        def __init__(self):
            self.sent = []

        def post(self, path, body=None, **kw):
            self.sent.append((path, body))
            return {}

    snapshot = [{"id": 1, "category": "chatbot", "key": "llm_model", "value": "anthropic/claude-opus-5"},
                {"id": 2, "category": "chatbot", "key": "llm_provider", "value": None},
                {"id": 3, "category": "orchestrator_llm", "key": "llm_provider", "value": "openrouter"}]
    api = Bulk()
    applied = apply_llm_settings(api, snapshot, "openrouter", "openai/gpt-4.1-mini")
    assert [u["id"] for u in applied] == [1, 3] and applied[0]["value"] == "openai/gpt-4.1-mini"
    restored = restore_llm_settings(api, snapshot)
    assert [u["id"] for u in restored] == [1, 3] and restored[0]["value"] == "anthropic/claude-opus-5"
    assert all(path == "/api/system-settings/bulk-update" for path, _ in api.sent)


def test_script_refusals_reach_the_error_message(monkeypatch):
    import subprocess

    def fake_run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 2, stdout='{"error": "refused: settings.purpose \'nightly-test-suite\' does not start with \'sim\'"}', stderr="")

    monkeypatch.setattr(workspace.subprocess, "run", fake_run)
    with pytest.raises(WorkspaceError, match="refused: settings.purpose"):
        purge(config.Settings(), "11111111-1111-1111-1111-111111111111")


def test_env_file_parsing_and_key_redaction(tmp_path):
    env = tmp_path / "env"
    env.write_text('# comment\nOPENROUTER_API_KEY="sk-or-test"\nSIM_MODEL_ID=\'a/b\'\nbroken line\n', encoding="utf-8")
    parsed = config.read_env_file(env)
    assert parsed == {"OPENROUTER_API_KEY": "sk-or-test", "SIM_MODEL_ID": "a/b"}
    assert config.read_env_file(tmp_path / "missing") == {}
    shown = config.public_settings(config.settings_from_mapping(parsed))
    assert shown["openrouter_api_key"] == "set" and shown["model_id"] == "a/b"
    assert config.public_settings(config.Settings())["openrouter_api_key"] == ""


def _usage_line(execution_id="board_task:5", status="success", cost_usd="0.0012", model="openai/gpt-4.1-mini"):
    values = [execution_id, "chat", model, "openrouter", "7", "100", "50", "0", cost_usd, "800", status, "2026-09-18 01:00:00"]
    return cost.FIELD_SEP.join(values)


def test_usage_rows_parse_and_summarise():
    rows = cost.parse_rows("\n".join([_usage_line(), _usage_line("chat:9", "error", "0.0008", "other/model"), "short\x1fline", ""]))
    assert len(rows) == 2
    assert rows[0]["input_tokens"] == 100 and rows[0]["total_cost"] == 0.0012 and rows[0]["agent_id"] == 7
    one = cost.summarise(rows, ["board_task:5"])
    assert (one["calls"], one["cost_usd"], one["errors"]) == (1, 0.0012, 0)
    assert one["by_model"]["openai/gpt-4.1-mini"]["tokens"] == 150
    everything = cost.summarise(rows)
    assert everything["calls"] == 2 and everything["errors"] == 1 and everything["latency_ms_max"] == 800
    assert cost.workspace_total(rows) == 0.002
    assert cost.models_seen(rows) == ("openai/gpt-4.1-mini", "other/model")


def test_usage_query_validates_its_inputs():
    with pytest.raises(ValueError):
        cost._sql("not-a-uuid", "2026-09-18T00:00:00")
    with pytest.raises(ValueError):
        cost._sql("00000000-0000-0000-0000-000000000001", "yesterday")
    sql = cost._sql("00000000-0000-0000-0000-000000000001", "2026-09-18T00:00:00")
    assert "llm_usage" in sql and "00000000-0000-0000-0000-000000000001" in sql


def test_workspace_script_output_and_guards():
    assert parse_key_values("# note\nWORKSPACE_ID=abc\nAPI_KEY=ak_srv_x\n") == {"WORKSPACE_ID": "abc", "API_KEY": "ak_srv_x"}
    with pytest.raises(WorkspaceError, match="sim-"):
        provision(config.Settings(), "test-nightly-suite", "x", "sim:x")
    with pytest.raises(WorkspaceError, match="default workspace"):
        purge(config.Settings(), config.DEFAULT_WORKSPACE_ID)


def test_items_of_finds_the_list_whatever_it_is_called():
    assert items_of([1, 2]) == [1, 2]
    assert items_of({"tasks": [1]}, "tasks") == [1]
    assert items_of({"items": [2]}) == [2]
    assert items_of({"x": 1}) == []
    assert items_of(None) == []


def test_text_and_effect_helpers():
    text = text_of({"content": "A", "nested": {"summary": "B"}, "list": [{"body": "C"}], "id": 4})
    assert "A" in text and "B" in text and "C" in text and "4" not in text
    assert [c.ok for c in must_contain_checks(["a", "zzz"], "has A")] == [True, False]
    checks = effect_checks({"deliverables_min": 1, "no_errors": True, "status": "done"},
                           {"deliverables_min": 0, "errors": 0, "status": "done"})
    assert [(c.name, c.ok) for c in checks] == [("effect:deliverables_min", False), ("effect:no_errors", True), ("effect:status", True)]
    assert parse_iso("2026-09-18T01:00:00Z").tzinfo is not None and parse_iso("nope") is None
    res = ScenarioResult(id="s", kind="task", started_at="2026-09-18T01:00:00+00:00", ended_at="2026-09-18T01:00:30+00:00",
                         outcome="done", ok=True)
    assert res.duration_s == 30.0 and res.to_dict()["duration_s"] == 30.0
