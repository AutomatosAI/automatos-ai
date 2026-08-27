"""PRD-222 US-011 — tool-schema truth pass + recurrence guard.

A *walker* iterates every registered platform action, resolves its handler from
``PlatformActionExecutor._handlers``, and asserts the LLM-facing schema tells the
truth about what the handler actually needs:

  (A) No ``required[]`` field has a *real* (non-sentinel) default in the handler.
      A handler that does ``params.get("report_type", "standup")`` works fine
      without the caller supplying ``report_type`` — forcing it in ``required[]``
      lies to the model. (This is the actions_reports over-require the story
      fixes; the self-check below proves the walker bites when it is reintroduced.)

  (B) Every either-of the handler enforces (``"Provide X or Y"``) names *both*
      alternatives in the tool description, so the model knows it must supply one.

  (C) Every single-field hard-fail (``"Missing required parameter: X"``) is
      actually in ``required[]`` — otherwise the model is never told to send it.

The walker functions are pure (schema/description/handler-source in, violations
out) so they are unit-testable with fixtures — no DB, no registry — and the same
functions then sweep the live registry.
"""

import inspect
import re

import pytest


# ── Pure walker primitives ───────────────────────────────────────────────────

# Defaults that are sentinels, not usable values: a handler that writes
# ``params.get("x", "")`` and then hard-fails ``if not x`` still *requires* x.
# Only a meaningful default ("standup", "ok", "health") means the field is
# genuinely optional and must not sit in required[].
SENTINEL_DEFAULTS = {'""', "''", "None", "[]", "{}", "()", "0", "0.0", "False"}

_DEFAULT_RE = re.compile(r"""params\.get\(\s*["'](\w+)["']\s*,\s*([^,)\n]+?)\s*\)""")
_EITHER_OF_RE = re.compile(r"Provide (\w+) or (\w+)")
_MISSING_PARAM_RE = re.compile(r"Missing required parameter: (\w+)")


def find_real_default_violations(required, handler_src):
    """(A) required[] fields that carry a real (non-sentinel) handler default."""
    req = set(required)
    out = []
    for field, default in _DEFAULT_RE.findall(handler_src):
        if field in req and default.strip() not in SENTINEL_DEFAULTS:
            out.append(field)
    return sorted(set(out))


def find_undocumented_either_of(description, handler_src):
    """(B) either-of pairs where an alternative is not named in the description."""
    out = []
    for x, y in set(_EITHER_OF_RE.findall(handler_src)):
        for field in (x, y):
            if not re.search(r"\b" + re.escape(field) + r"\b", description):
                out.append((f"{x}|{y}", field))
    return sorted(set(out))


def find_undocumented_required(required, handler_src):
    """(C) single-field hard-fails ("Missing required parameter: X") not required."""
    req = set(required)
    return sorted({f for f in _MISSING_PARAM_RE.findall(handler_src) if f not in req})


# ── Live wiring: map every registered action to its handler source ───────────

def _effective_source(fn, resolve_src):
    """Handler source, with the one shared agent resolver inlined.

    ``resolve_agent`` (handlers_assignments) is the single shared
    agent_id|agent_name resolver used by assign/heartbeat/unassign handlers; its
    either-of error lives there, not in each caller's body, so it is inlined so
    the walker guards those tools too.
    """
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return ""
    if "resolve_agent(" in src:
        src += "\n" + resolve_src
    return src


def _iter_tool_specs():
    """Yield (name, action_def, effective_handler_source) for every registered
    action that has a handler. Pure — the executor __init__ only builds a dict."""
    from modules.tools.discovery.action_registry import get_action_registry
    from modules.tools.discovery.platform_executor import PlatformActionExecutor
    from modules.tools.discovery import handlers_assignments

    registry = get_action_registry()
    actions = {a.name: a for a in registry.get_all()}
    handlers = PlatformActionExecutor(None, None)._handlers  # no DB touched
    resolve_src = inspect.getsource(handlers_assignments.resolve_agent)

    for name, fn in sorted(handlers.items()):
        ad = actions.get(name)
        if ad is None:
            continue
        yield name, ad, _effective_source(fn, resolve_src)


# ── Self-check: the walker demonstrably bites (fixtures, no registry) ─────────

# The exact actions_reports over-require this story removed. If it is ever
# reintroduced, (A) must flag report_type + status.
_REPORT_HANDLER_SRC = '''
async def submit_report(db, workspace_id, params):
    title = params.get("title")
    content = params.get("content")
    report_type = params.get("report_type", "standup")
    status = params.get("status", "ok")
    if not title or not content:
        return {"success": False, "error": "title and content are required"}
'''


def test_walker_bites_on_report_over_require():
    """Regression fixture: the pre-fix required[] must be flagged by (A)."""
    reintroduced = ["title", "content", "report_type", "status"]
    violations = find_real_default_violations(reintroduced, _REPORT_HANDLER_SRC)
    assert violations == ["report_type", "status"], (
        "walker (A) must flag handler-defaulted fields wrongly marked required"
    )


def test_walker_passes_when_report_over_require_fixed():
    """Same handler, corrected required[] → no (A) violation."""
    fixed = ["title", "content"]
    assert find_real_default_violations(fixed, _REPORT_HANDLER_SRC) == []


def test_walker_ignores_sentinel_default_that_is_hard_failed():
    """A ""-sentinel default that is hard-failed is genuinely required (no bite)."""
    src = 'q = params.get("query", "")\n    if not q:\n        return {"error": "query is required"}'
    assert find_real_default_violations(["query"], src) == []


def test_walker_bites_on_undocumented_either_of():
    """(B) flags an either-of whose alternative is absent from the description."""
    src = 'if not a_id and not a_name:\n    return {"error": "Provide a_id or a_name"}'
    assert find_undocumented_either_of("Look up by a_id.", src) == [("a_id|a_name", "a_name")]
    assert find_undocumented_either_of("Look up by a_id or a_name.", src) == []


def test_walker_bites_on_undocumented_required():
    """(C) flags a "Missing required parameter" field absent from required[]."""
    src = 'if not x:\n    return {"error": "Missing required parameter: x"}'
    assert find_undocumented_required([], src) == ["x"]
    assert find_undocumented_required(["x"], src) == []


# ── Live sweep: the fixed tree passes on every registered tool ────────────────

def test_no_real_default_field_is_required():
    """(A) over the live registry — the story's fix must hold platform-wide."""
    offenders = {}
    for name, ad, src in _iter_tool_specs():
        bad = find_real_default_violations(ad.parameters.get("required", []), src)
        if bad:
            offenders[name] = bad
    assert offenders == {}, f"handler-defaulted fields marked required: {offenders}"


def test_every_either_of_is_documented():
    """(B) over the live registry — both alternatives named in the description."""
    offenders = {}
    for name, ad, src in _iter_tool_specs():
        bad = find_undocumented_either_of(ad.description, src)
        if bad:
            offenders[name] = bad
    assert offenders == {}, f"undocumented either-of requirements: {offenders}"


def test_every_hard_required_field_is_in_required():
    """(C) over the live registry — no silently-required field."""
    offenders = {}
    for name, ad, src in _iter_tool_specs():
        bad = find_undocumented_required(ad.parameters.get("required", []), src)
        if bad:
            offenders[name] = bad
    assert offenders == {}, f"hard-failed fields missing from required[]: {offenders}"


# ── Direct assertions of the story's named fixes ─────────────────────────────

def _action(name):
    from modules.tools.discovery.action_registry import get_action_registry
    return get_action_registry().get(name)


def test_submit_report_required_is_title_and_content_only():
    """AC #1 — required[] == ['title', 'content']."""
    assert _action("platform_submit_report").parameters["required"] == ["title", "content"]


def test_create_mission_schema_has_no_source_field():
    """AC #3 — platform_create_mission must not gain a 'source' param (W2 target)."""
    props = _action("platform_create_mission").parameters.get("properties", {})
    assert "source" not in props, "platform_create_mission must not carry a 'source' field"


@pytest.mark.parametrize(
    "name, phrase",
    [
        ("platform_assign_tool_to_agent", "Provide agent_id or agent_name"),
        ("platform_assign_skill_to_agent", "Provide agent_id or agent_name"),
        ("platform_assign_skill_to_agent", "Provide skill_id or skill_name"),
        ("platform_assign_plugin_to_agent", "Provide agent_id or agent_name"),
        ("platform_assign_plugin_to_agent", "Provide plugin_id or plugin_slug"),
        ("platform_configure_agent_heartbeat", "Provide agent_id or agent_name"),
        ("platform_install_plugin", "Provide plugin_id or plugin_slug"),
        ("platform_install_skill", "Provide skill_id or skill_name"),
    ],
)
def test_named_tools_carry_exact_handler_error_copy(name, phrase):
    """AC #2 — the assignment/heartbeat/install descriptions carry the exact
    either-of error copy their handlers raise."""
    assert phrase in _action(name).description
