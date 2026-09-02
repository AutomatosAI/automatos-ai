"""
PRD-232 US-012A — eval set refresh + abstain support (build only).
==================================================================

C10 / PRD-223: the eval set was stale (47 rows vs 59 seed queries) and could not
express ABSTAIN — run_eval forced tool_choice="required", so "no tool applies"
was inexpressible and a correct decline was unscorable (Gap 1). US-012A:

- regenerates eval_set.jsonl from eval_seed.yaml (now consistent);
- adds utterance-derived queries incl. the 2026-08-28 VECTOR case;
- adds >=10 abstain rows (no applicable tool);
- makes abstain expressible per-row (tool_choice="auto") and scorable (a no-call
  is correct for abstain rows, any tool call is wrong).

RUNNING the uplift eval is a post-merge human op — these tests touch NO DB and NO
network (a fake OpenAI client captures the tool_choice; the scorer is pure).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

_ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(_ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_ORCH_ROOT))

_EVAL_DIR = _ORCH_ROOT / "scripts" / "eval" / "tool_routing"


def _load_jsonl():
    return [json.loads(line) for line in (_EVAL_DIR / "eval_set.jsonl").read_text().splitlines() if line.strip()]


def _load_seed_queries():
    return yaml.safe_load((_EVAL_DIR / "eval_seed.yaml").read_text())["queries"]


# ── AC: eval_set.jsonl regenerated and consistent with eval_seed.yaml ────────
def test_eval_set_consistent_with_seed():
    rows = _load_jsonl()
    queries = _load_seed_queries()
    assert len(rows) == len(queries), (
        f"eval_set.jsonl ({len(rows)}) is stale vs eval_seed.yaml ({len(queries)}) "
        "— re-run `python -m scripts.eval.tool_routing.seed_eval_set`"
    )
    # every seed query's text appears as a row (order preserved by query_id)
    assert [r["query"] for r in rows] == [q["q"].strip() for q in queries]


# ── AC: >=10 abstain rows, each with an empty correct_actions ────────────────
def test_at_least_ten_abstain_rows():
    rows = _load_jsonl()
    abstain = [r for r in rows if r.get("abstain")]
    assert len(abstain) >= 10, f"only {len(abstain)} abstain rows (need >=10)"
    assert all(not r["correct_actions"] for r in abstain), "abstain rows must have empty correct_actions"


# ── AC: the VECTOR query is present with the expected action ─────────────────
def test_vector_query_present():
    rows = _load_jsonl()
    vec = [r for r in rows if "close all the blocked tickets from vector" == r["query"]]
    assert vec, "the 2026-08-28 VECTOR query is missing from the eval set"
    assert "platform_update_task_status" in vec[0]["correct_actions"]
    assert not vec[0].get("abstain")


# ── AC: scorer counts a correct abstention (no-call) and a wrong tool call ───
def test_scorer_abstain_correct_on_no_call():
    from scripts.eval.tool_routing.score import _is_correct
    assert _is_correct({"abstain": True, "chosen_action": None}) is True
    assert _is_correct({"abstain": True, "chosen_action": ""}) is True


def test_scorer_abstain_wrong_on_any_call():
    from scripts.eval.tool_routing.score import _is_correct
    assert _is_correct({"abstain": True, "chosen_action": "platform_list_agents"}) is False
    assert _is_correct({"abstain": True, "chosen_action": "platform_update_task_status"}) is False


def test_scorer_nonabstain_unchanged():
    from scripts.eval.tool_routing.score import _is_correct
    assert _is_correct({"chosen_action": "platform_list_agents",
                        "correct_actions": ["platform_list_agents"]}) is True
    assert _is_correct({"chosen_action": "wrong_action",
                        "correct_actions": ["platform_list_agents"]}) is False
    assert _is_correct({"chosen_action": None,
                        "correct_actions": ["platform_list_agents"]}) is False


def test_scorer_abstain_counts_in_set():
    """Abstain rows have no correct action to surface — they must not drag the
    in-set metric down (counted in-set)."""
    from scripts.eval.tool_routing.score import _is_in_set
    assert _is_in_set({"abstain": True, "correct_actions": [], "surfaced": []}) is True


# ── AC: run_eval makes abstain expressible (tool_choice="auto"), no network ──
class _FakeCompletions:
    def __init__(self, response, sink):
        self._response = response
        self._sink = sink

    def create(self, **kwargs):
        self._sink["tool_choice"] = kwargs.get("tool_choice")
        return self._response


def _fake_client(response, sink):
    return SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions(response, sink)))


def _fake_response(tool_calls=None):
    msg = SimpleNamespace(tool_calls=tool_calls, content="")
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(
        choices=[choice],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2, total_tokens=7),
    )


def test_abstain_row_uses_auto_tool_choice():
    from scripts.eval.tool_routing.run_eval import _call_model
    sink = {}
    client = _fake_client(_fake_response(tool_calls=None), sink)
    result, err = _call_model(
        client, model="m", system_prompt="s", user_query="good morning!",
        temperature=0.0, max_tokens=16, request_timeout=5, tools=[], abstain=True,
    )
    assert sink["tool_choice"] == "auto", "abstain rows must let the model decline"
    assert result["chosen_action"] is None
    assert err is None


def test_non_abstain_row_uses_required_tool_choice():
    from scripts.eval.tool_routing.run_eval import _call_model
    sink = {}
    tc = SimpleNamespace(function=SimpleNamespace(
        name="platform_execute", arguments='{"action": "platform_list_agents"}'))
    client = _fake_client(_fake_response(tool_calls=[tc]), sink)
    result, err = _call_model(
        client, model="m", system_prompt="s", user_query="list my agents",
        temperature=0.0, max_tokens=16, request_timeout=5, tools=[], abstain=False,
    )
    assert sink["tool_choice"] == "required"
    assert result["chosen_action"] == "platform_list_agents"
    assert err is None
