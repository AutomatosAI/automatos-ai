"""F182 (night 6) — a key an action does not take is refused, not dropped.

Auto started "New Cafe Onboarding" (playbook 102, run 207) for Gull & Anchor with
the café's details nested under "params". execute_playbook reads input_data, so
the run started with no inputs and the tool said success. Minutes later two
update_playbook calls, {"updates": {"steps": [...]}} and then {"steps": [...]},
changed nothing and said success too, and Auto told the owner the playbook was
fixed. platform_execute now refuses a key the action neither declares nor reads.
The refusal says where the key goes, and each one is logged to be counted.
execute_playbook repeats back the inputs it stored, "none" included.
"""
from __future__ import annotations

import ast
import asyncio
import importlib
import inspect
import logging
import textwrap
from types import SimpleNamespace

import pytest

WS = "dacae30f-7840-40c1-8d03-25c3910affd0"
GULL = {"usual_order": "6 kg of Harbour Blend and 2 kg of decaf a week", "draft_only": True,
        "contact_person": "Priya Shah", "cafe_name": "Gull & Anchor",
        "contact_email": "priya@gullandanchor.example", "delivery_day": "Thursdays"}
# Night 6, as Auto sent them (chats.jsonl 117 and 126).
NESTED_INPUTS = {"params": GULL, "playbook_id": 102}
STEPS = [{"step_index": 0, "prompt_template": "Please provide the following details for the new cafe: cafe_name"},
         {"step_index": 1, "prompt_template": "Draft a welcome email for {cafe_name} to {contact_person}"}]
STEPS_IN_UPDATES = {"playbook_id": 102, "updates": {"steps": STEPS}}
STEPS_AT_TOP = {"steps": STEPS, "playbook_id": 102}


@pytest.fixture
def dispatch(monkeypatch):
    from modules.tools.execution import unified_executor
    from modules.tools.execution.unified_executor import UnifiedToolExecutor

    ran = []

    async def _run(self, action_name, params, **kwargs):
        ran.append((action_name, params))
        return {"success": True}

    monkeypatch.setattr(UnifiedToolExecutor, "_policy_gate_check", lambda self, *a, **k: None)
    monkeypatch.setattr(UnifiedToolExecutor, "_execute_platform_action", _run)
    monkeypatch.setattr(unified_executor, "fire_telemetry", lambda **kwargs: None)
    executor = UnifiedToolExecutor.__new__(UnifiedToolExecutor)
    executor.composio_actions, executor.db = {}, None

    def call(action, params):
        return asyncio.run(executor.execute_tool(
            "platform_execute", {"action": action, "params": params},
            agent_id=322, workspace_id=WS, trace_id="t-f182"))
    return SimpleNamespace(call=call, ran=ran)


# ── night 6 ─────────────────────────────────────────────────────────────────

def test_inputs_nested_under_params_are_refused_before_the_run_starts(dispatch):
    result = dispatch.call("platform_execute_playbook", NESTED_INPUTS)

    assert result["success"] is False and dispatch.ran == []
    assert "'platform_execute_playbook' does not take ['params']" in result["error"]
    assert "params: put these values under 'input_data'." in result["error"]
    assert '"params": {"playbook_id": 102, "input_data": {"usual_order": ' in result["error"]
    assert '"cafe_name": "Gull & Anchor"' in result["error"]


@pytest.mark.parametrize("params", [STEPS_IN_UPDATES, STEPS_AT_TOP], ids=["updates", "steps"])
def test_steps_sent_to_update_playbook_are_refused_and_sent_to_the_step_actions(dispatch, params):
    result = dispatch.call("platform_update_playbook", params)

    assert result["success"] is False and dispatch.ran == []
    assert "platform_update_playbook_step" in result["error"]
    assert "'platform_update_playbook' takes: playbook_id (integer, required), name (string)" in result["error"]


def test_each_refusal_is_logged_with_the_action_and_the_key(dispatch, caplog):
    with caplog.at_level(logging.INFO, logger="modules.tools.execution.unified_executor"):
        dispatch.call("platform_update_playbook", {"playbook_id": 102, "updates": {"steps": STEPS}, "reason": "fix"})

    refused = [r.getMessage() for r in caplog.records if r.getMessage().startswith("[F182]")]
    assert refused == ["[F182] platform_execute refused param 'updates' for platform_update_playbook (trace t-f182)",
                       "[F182] platform_execute refused param 'reason' for platform_update_playbook (trace t-f182)"]


def test_a_missing_key_and_a_stray_one_are_named_in_one_answer(dispatch):
    result = dispatch.call("platform_update_playbook", {"updates": {"name": "New Cafe Onboarding"}})

    assert result["error"].startswith("Missing required params for 'platform_update_playbook': ['playbook_id']")
    assert "updates: its fields ['name'] go straight into params, not inside 'updates'." in result["error"]


@pytest.mark.parametrize("action, params, says", [
    ("platform_get_playbook", {"playbook_idd": 102}, "playbook_idd: did you mean 'playbook_id'?"),
    ("platform_create_playbook", {"name": "Onboarding", "description": "New cafés", "steps": STEPS},
     "steps: a playbook is created with no steps"),
    ("platform_update_playbook", {"playbook_id": 102, "workspace_id": WS}, "workspace_id: not a parameter"),
    ("platform_execute_playbook", {"params": {"playbook_id": 102, "input_data": {"cafe_name": "Gull & Anchor"}}},
     "params: its fields ['playbook_id', 'input_data'] go straight into params, not inside 'params'."),
], ids=["typo", "create-steps", "stray", "double-nested"])
def test_the_refusal_says_where_the_key_goes(dispatch, action, params, says):
    result = dispatch.call(action, params)

    assert result["success"] is False and dispatch.ran == []
    assert says in result["error"]


@pytest.mark.parametrize("action, sent, kept", [
    ("platform_create_agent", {"name": "DevOps Bot", "desc": "Watches the deploys"},
     {"name": "DevOps Bot", "description": "Watches the deploys"}),
    ("platform_get_playbook", {"recipe_id": 102}, {"playbook_id": 102}),
], ids=["desc", "recipe_id"])
def test_an_optional_param_under_a_known_other_name_is_kept(dispatch, caplog, action, sent, kept):
    """F027's synonym acceptance (tool errors 16 % → 1.9 %) covers optional params
    too: the value is kept under its declared name, counted, never refused."""
    with caplog.at_level(logging.INFO, logger="modules.tools.execution.unified_executor"):
        result = dispatch.call(action, sent)

    assert result == {"success": True}
    ((_, params),) = dispatch.ran
    assert params == kept
    alias = next(k for k in sent if k not in kept)
    assert any(r.getMessage().startswith(f"[F182] platform_execute mapped param '{alias}' to ")
               for r in caplog.records)


def test_a_model_cannot_write_a_tickets_planning_data(dispatch):
    """F183 keeps the owner's answers in planning_data.human_qa, and the next run
    reads them as the owner's words. A model files a ticket with approval_action."""
    result = dispatch.call("platform_create_task", {
        "title": "Draft the newsletter", "description": "For the coffee club.",
        "planning_data": {"human_qa": [{"q": "Theme?", "a": "Anything goes"}]}})

    assert result["success"] is False and dispatch.ran == []
    assert "'platform_create_task' does not take ['planning_data']" in result["error"]


# ── what worked before still runs ───────────────────────────────────────────

@pytest.mark.parametrize("action, params", [
    ("platform_execute_playbook", {"playbook_id": 102, "input_data": {"cafe_name": "Gull & Anchor"}}),
    ("platform_execute_playbook", {"playbook_id": 102, "inputs": "cafe_name: Gull & Anchor"}),       # accepts
    ("platform_add_playbook_step", {"playbook_id": 102, "prompt": "Draft the welcome email"}),       # F027-C
    ("platform_update_playbook_step", {"recipe_id": 102, "step": 1, "order": 0}),                    # F027-C
    ("platform_submit_report", {"report": {"title": "Week 40", "content": "All paid."}}),             # a wrapper
    ("platform_update_playbook", {"playbook_id": 102, "name": "New Cafe Onboarding", "_agent_id": 9}),
    ("platform_update_playbook", {"playbook_id": 102, "name": "New Cafe Onboarding", "notes": None}),
    ("platform_field_inject", {"key": "k", "value": "v", "field_id": "mission:7"}),                  # PRD-178
], ids=["input_data", "inputs", "prompt-alias", "step-aliases", "report-wrapper", "server-key",
        "empty-value", "field_id"])
def test_a_call_that_worked_before_still_runs(dispatch, action, params):
    result = dispatch.call(action, params)

    assert result == {"success": True}
    assert [name for name, _ in dispatch.ran] == [action]


# ── execute_playbook says what it stored ────────────────────────────────────

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

    async def allowed(workspace_id, db):
        return SimpleNamespace(allowed=True, reason="")

    monkeypatch.setattr(guard, "check_concurrency", allowed)
    monkeypatch.setattr(engine, "get_playbook_engine", lambda: SimpleNamespace(launch=lambda **kw: None))
    monkeypatch.setattr(watches, "auto_create_watch", lambda *args, **kwargs: None)
    db = _DB(SimpleNamespace(id=102, name="New Cafe Onboarding"))
    return asyncio.run(execute_playbook(db, WS, {"playbook_id": 102, **params}))


def test_the_run_repeats_back_the_inputs_it_stored(monkeypatch):
    result = _trigger(monkeypatch, input_data={"cafe_name": "Gull & Anchor"})

    assert result["input_data"] == {"cafe_name": "Gull & Anchor"}
    assert result["message"].startswith(
        "Playbook 'New Cafe Onboarding' triggered with inputs: {\"cafe_name\": \"Gull & Anchor\"}. ")


def test_a_run_with_no_inputs_says_none(monkeypatch):
    result = _trigger(monkeypatch)

    assert result["input_data"] == {}
    assert "triggered with inputs: none." in result["message"]


# ── the keys each handler reads are the keys its action takes ───────────────
# A handler that reads a key its schema does not declare has that key refused
# by platform_execute unless the action lists it in ``accepts``.

# Read by the handler but never taken from a model (see the planning_data test).
NOT_FROM_A_MODEL = {("platform_create_task", "planning_data")}
FOLLOW_DEPTH = 3


def _names(node, loops):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, ast.Name) and node.id in loops:
        return set(loops[node.id])
    return set()


def _callee(fn, name, func):
    """What ``name`` calls inside ``fn``: a module global, or a lazy import."""
    if name in fn.__globals__:
        return fn.__globals__[name]
    for node in ast.walk(func):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if (alias.asname or alias.name) == name:
                    module = importlib.import_module("." * node.level + (node.module or ""),
                                                     package=fn.__module__.rpartition(".")[0])
                    return getattr(module, alias.name, None)
    return None


def _reads(fn, param, depth=0, seen=None):
    """The keys ``fn`` reads from its ``param`` dict, following the helpers it hands it to."""
    seen = set() if seen is None else seen
    fn = inspect.unwrap(fn)
    if not inspect.isfunction(fn) or (fn, param) in seen or depth > FOLLOW_DEPTH:
        return set()
    seen.add((fn, param))
    try:
        func = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
    except (OSError, TypeError):  # no source to read: nothing to follow
        return set()
    loops = {n.target.id: [e.value for e in n.iter.elts] for n in ast.walk(func)
             if isinstance(n, (ast.For, ast.comprehension)) and isinstance(n.target, ast.Name)
             and isinstance(n.iter, (ast.Tuple, ast.List))
             and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in n.iter.elts)}
    keys = set()
    for node in ast.walk(func):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.args
                and node.func.attr in ("get", "pop", "setdefault")
                and isinstance(node.func.value, ast.Name) and node.func.value.id == param):
            keys |= _names(node.args[0], loops)
        elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == param:
            keys |= _names(node.slice, loops)
        elif (isinstance(node, ast.Compare) and isinstance(node.ops[0], (ast.In, ast.NotIn))
              and isinstance(node.comparators[0], ast.Name) and node.comparators[0].id == param):
            keys |= _names(node.left, loops)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            handed = [i for i, a in enumerate(node.args) if isinstance(a, ast.Name) and a.id == param]
            named = [k.arg for k in node.keywords if k.arg and isinstance(k.value, ast.Name) and k.value.id == param]
            callee = _callee(fn, node.func.id, func) if handed or named else None
            if inspect.isfunction(callee):
                slots = list(inspect.signature(callee).parameters)
                for target in [slots[i] for i in handed if i < len(slots)] + named:
                    keys |= _reads(callee, target, depth + 1, seen)
    return keys


def _actions():
    from modules.tools.discovery import get_action_registry
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    registry = get_action_registry()
    for name, handler in PlatformActionExecutor(None, None)._handlers.items():
        action = registry.get(name)
        if action is None:  # the executor refuses a handler with no action
            continue
        slots = list(inspect.signature(inspect.unwrap(handler)).parameters)
        param = "params" if "params" in slots else slots[2]
        yield name, action, {k for k in _reads(handler, param) if not k.startswith("_")}


def test_every_key_a_handler_reads_is_one_its_action_takes():
    drift = {}
    for name, action, reads in _actions():
        taken = set((action.parameters or {}).get("properties") or {}) | set(action.accepts)
        stray = sorted(k for k in reads - taken if (name, k) not in NOT_FROM_A_MODEL)
        if stray:
            drift[name] = stray
    assert drift == {}, ("platform_execute refuses these keys, but the handler reads them: declare each "
                         f"in the schema or list it in the action's accepts=(...): {drift}")


def test_every_accepted_key_is_one_its_handler_reads():
    stale = {name: sorted(set(action.accepts) - reads)
             for name, action, reads in _actions() if set(action.accepts) - reads}
    assert stale == {}
