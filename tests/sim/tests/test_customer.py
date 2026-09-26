"""The customer-night helpers: prompt rendering, inventory, tagging — no network."""

import pytest

from tests.sim import customer_ops as ops


def test_prompt_rendering_fills_every_placeholder_or_refuses():
    out = ops.render_prompt("night {{DATE}} in {{NIGHT_DIR}} — {{PERSONA}}", {"DATE": "2026-09-18", "NIGHT_DIR": "/n", "PERSONA": "You are…"})
    assert out == "night 2026-09-18 in /n — You are…"
    with pytest.raises(KeyError, match="STOP_AT"):
        ops.render_prompt("stop at {{STOP_AT}}", {})
    assert ops.render_prompt('curl -d \'{"title":"x"}\'', {}) == 'curl -d \'{"title":"x"}\''  # single braces untouched


def test_shipped_prompt_only_uses_placeholders_the_runner_provides():
    from pathlib import Path
    template = (Path(__file__).resolve().parents[3] / "scripts" / "ralph" / "customer-night" / "PROMPT_customer.md").read_text(encoding="utf-8")
    provided = {"DATE", "NIGHT_DIR", "WORKSPACE_ID", "API_URL", "ITER", "MAX_ITERS", "STOP_AT", "NOW", "DELIVERABLES_DIR", "PERSONA", "INVENTORY", "NIGHT_START",
                "AGENDA"}  # the night's theme, nights/<theme>.md (136916278)
    used = {m.group(1) for m in ops.PLACEHOLDER.finditer(template)}
    assert used <= provided, f"placeholders the runner does not fill: {used - provided}"
    assert {"PERSONA", "INVENTORY", "NIGHT_DIR", "WORKSPACE_ID"} <= used


def test_tagging_matches_tags_or_text():
    tag = "sim-night-2026-09-18"
    assert ops.has_tag({"tags": [tag]}, tag)
    assert ops.has_tag({"name": f"Researcher {tag}"}, tag)
    assert ops.has_tag({"description": f"made on {tag}"}, tag)
    assert not ops.has_tag({"tags": ["other"], "name": "RESEARCHER"}, tag)


class FakeApi:
    workspace_id = "ws"

    def __init__(self, payloads):
        self.payloads, self.deleted = payloads, []

    def get(self, path, params=None, **kw):
        return self.payloads.get(path, [])

    def delete(self, path, **kw):
        self.deleted.append(path)
        return {}


def _api():
    tag = "sim-night-2026-09-18"
    return FakeApi({
        "/api/agents/": [{"id": 57, "name": "RESEARCHER", "configuration": {"runtime": "cli"}, "tags": []},
                         {"id": 300, "name": "SIM Ops", "configuration": {"runtime": "api"}, "tags": [tag]}],
        "/api/v1/tasks": {"tasks": [{"id": 1, "title": "old", "status": "done", "tags": []},
                                    {"id": 2, "title": "site", "status": "in_progress", "assigned_agent_id": 57, "tags": [tag]},
                                    {"id": 3, "title": "review me", "status": "review", "assigned_agent_id": 300, "tags": [tag]}]},
        "/api/deliverables": {"deliverables": [{"id": 9, "title": "index.html", "tags": [tag]}]},
        "/api/reports": {"reports": [{"id": 4, "title": "daily", "report_type": "standup"}]},
        "/api/v1/approval-grants": {"grants": [{"id": 11, "kind": "question", "question_md": "Which supplier?", "options": ["A", "B"]}]},
    })


def test_inventory_is_filtered_by_tag_and_rendered():
    inv = ops.inventory(_api(), "sim-night-2026-09-18")
    assert [a["id"] for a in inv["agents"]] == [300] and inv["agents"][0]["runtime"] == "api"
    assert [t["id"] for t in inv["tasks"]] == [2, 3]
    assert inv["reports"] and inv["questions"][0]["options"] == ["A", "B"] and inv["errors"] == []
    text = ops.render_inventory(inv)
    assert "SIM Ops (#300, api)" in text and "#3 [review] review me" in text and "Which supplier?" in text
    everything = ops.inventory(_api())
    assert len(everything["agents"]) == 2 and len(everything["tasks"]) == 3


def test_purge_only_touches_the_inventory_it_was_given():
    api = _api()
    inv = ops.inventory(api, "sim-night-2026-09-18")
    lines = ops.purge_tagged(api, inv)
    assert api.deleted == ["/api/v1/tasks/2", "/api/v1/tasks/3", "/api/agents/300"]
    assert all("deleted" in line for line in lines) and "/api/agents/57" not in api.deleted


# ── F208: the balance a night can spend ────────────────────────────────────

def test_the_balance_is_read_in_the_platform_keys_workspace(monkeypatch):
    """Night 6 asked /api/v1/providers/openrouter/balance (404) and launched blind."""
    from tests.sim import customer
    from tests.sim.api import Response

    asked = []

    class _C1Api:
        def __init__(self, base_url, api_key, workspace_id, trace, timeout_s=60.0):
            self.workspace_id = workspace_id

        def request(self, method, path, **kw):
            asked.append((self.workspace_id, method, path))
            return Response(200, '{"total_credits": 25.0, "total_usage": 21.36}', 12, {})

    monkeypatch.setattr(customer, "Api", _C1Api)
    sim = type("SimApi", (), {"base_url": "http://x", "api_key": "", "trace": None, "timeout_s": 5.0})()

    left, said = customer._openrouter_balance(sim)

    assert asked == [("00000000-0000-0000-0000-0000000000c1", "GET", "/api/analytics/llm/openrouter/credits")]
    assert round(left, 2) == 3.64 and said == "$3.64 left (credits $25.00 - usage $21.36)"


@pytest.mark.parametrize("balance, code", [((3.64, "$3.64 left"), 2), ((None, "unavailable (HTTP 502)"), 2),
                                           ((12.0, "$12.00 left"), 0)], ids=["under-5", "unknown", "enough"])
def test_preflight_refuses_a_night_it_cannot_pay_for(monkeypatch, tmp_path, capsys, balance, code):
    import argparse

    from tests.sim import customer
    from tests.sim.api import Response

    class _Api:
        workspace_id = "sim-ws"

        def request(self, method, path, **kw):
            return Response(200, "{}", 3, {})

    monkeypatch.setattr(customer, "_api", lambda args: (_Api(), type("S", (), {"api_url": "http://x"})()))
    monkeypatch.setattr(customer, "inventory", lambda api: {"agents": [], "tasks": [], "questions": [], "errors": []})
    monkeypatch.setattr(customer, "_write_state_snapshot", lambda api, args: None)
    monkeypatch.setattr(customer, "HOST_LOG", tmp_path / "no-host.log")
    monkeypatch.setattr(customer, "_openrouter_balance", lambda api: balance)

    assert customer.cmd_preflight(argparse.Namespace(deliverables_dir=str(tmp_path))) == code
    assert ("REFUSED" in capsys.readouterr().out) is (code == 2)
