"""F113 (run 4) — a playbook run's input is key-value pairs, and a run never hands
the owner a Python exception.

Auto triggered "Friday cafe payment chase" through platform_execute_playbook
with input_data as one string (the unpaid invoices as CSV). The run died in
write_inputs on ``'str' object has no attribute 'items'`` and ticket #681
carried that text as its result. Now the tool reads a string as
``{"input": <the text>}`` (the playbook's step reads ``{input}``), refuses a
value that cannot be key-value pairs by naming the parameter, the executor
reads its input the same way whatever the trigger stored, and a programming
error is reported in words while its traceback stays in the log.
"""
from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace

import pytest

from core.services.playbook_inputs import playbook_inputs

CSV = "invoice,cafe,amount_gbp\n1041,The Salt Loft,212.40\n1044,Brew & Bramble,96.00\n"


@pytest.mark.parametrize("value, expected", [
    ({"cafe": "The Salt Loft"}, {"cafe": "The Salt Loft"}),
    (CSV, {"input": CSV}),
    ('{"cafe": "The Salt Loft", "week": 40}', {"cafe": "The Salt Loft", "week": 40}),
    ("{not json", {"input": "{not json"}),
    (None, {}),
    ("", {}),
    ("   ", {}),
], ids=["dict", "string", "json-object-string", "brace-text", "missing", "empty", "blank"])
def test_what_a_run_is_given(value, expected):
    assert playbook_inputs(value) == (expected, None)


@pytest.mark.parametrize("value", [["The Salt Loft"], 42, True])
def test_a_value_that_cannot_be_key_value_pairs_is_refused_by_name(value):
    inputs, problem = playbook_inputs(value)
    assert inputs == {} and problem.startswith("input_data must be key-value pairs")


# ── the tool ────────────────────────────────────────────────────────────────

class _Query:
    def __init__(self, found):
        self.found = found

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self.found


class _DB:
    def __init__(self, playbook):
        self.playbook, self.added = playbook, []

    def query(self, model):
        return _Query(self.playbook)

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        pass

    def rollback(self):
        pass


def _trigger(monkeypatch, **params):
    import modules.tools.discovery.handlers_watches as watches
    import services.concurrency_guard as guard
    import services.playbook_engine as engine
    from modules.tools.discovery.handlers_playbooks import execute_playbook

    launched = []

    async def allowed(workspace_id, db):
        return SimpleNamespace(allowed=True, reason="")

    monkeypatch.setattr(guard, "check_concurrency", allowed)
    monkeypatch.setattr(engine, "get_playbook_engine", lambda: SimpleNamespace(launch=lambda **kw: launched.append(kw)))
    monkeypatch.setattr(watches, "auto_create_watch", lambda *args, **kwargs: None)
    db = _DB(SimpleNamespace(id=12, name="Friday cafe payment chase"))
    result = asyncio.run(execute_playbook(db, "ws-c1", {"playbook_name": "Friday cafe payment chase", **params}))
    return result, db, launched


def test_string_inputs_run_as_the_playbooks_input(monkeypatch):
    result, db, launched = _trigger(monkeypatch, input_data=CSV)
    assert result["success"] is True
    assert db.added[0].input_data == {"input": CSV} and launched[0]["input_data"] == {"input": CSV}


def test_dict_inputs_are_unchanged(monkeypatch):
    result, db, launched = _trigger(monkeypatch, input_data={"cafe": "The Salt Loft", "week": 40})
    assert result["success"] is True and launched[0]["input_data"] == {"cafe": "The Salt Loft", "week": 40}


def test_missing_inputs_run_with_none(monkeypatch):
    result, db, launched = _trigger(monkeypatch)
    assert result["success"] is True and launched[0]["input_data"] == {}


def test_an_inputs_alias_is_read_too(monkeypatch):
    result, db, launched = _trigger(monkeypatch, inputs=CSV)
    assert launched[0]["input_data"] == {"input": CSV}


def test_a_list_is_refused_before_anything_runs(monkeypatch):
    result, db, launched = _trigger(monkeypatch, input_data=["The Salt Loft"])
    assert result["success"] is False and "input_data must be key-value pairs" in result["error"]
    assert db.added == [] and launched == []


# ── the run ─────────────────────────────────────────────────────────────────

def test_a_programming_error_is_reported_in_words():
    from api.recipe_executor import INTERNAL_ERROR_TEXT, owner_error_text

    text = owner_error_text(AttributeError("'str' object has no attribute 'items'"))
    assert text == INTERNAL_ERROR_TEXT and "'str'" not in text and "Error" not in text
    assert owner_error_text(KeyError("cafe")) == INTERNAL_ERROR_TEXT
    # an operational failure keeps what the owner can act on
    assert owner_error_text(RuntimeError("OpenRouter 402: Payment Required")) == "OpenRouter 402: Payment Required"


def test_the_run_reads_its_input_before_using_it_and_never_reports_str_of_an_exception():
    from api import recipe_executor

    inner = inspect.getsource(recipe_executor._execute_recipe_inner)
    assert inner.index("playbook_inputs(input_data)") < inner.index("scratchpad.write_inputs(input_data)")
    assert "_fail_execution(db, recipe_execution_id, str(e))" not in inner
    assert "_fail_execution(db, recipe_execution_id, owner_error_text(e))" in inner
    assert "last_error = str(e)" not in inner
