"""PRD-261 B-03h: the browser lane's dispatch inputs reach a shell only through ``env:``.

``.github/workflows/browser-e2e.yml`` takes ``kit_ref`` and ``kit_theme`` on a manual run: the
Design Kit commit to pack from and the kit theme to pack and install, defaulting to the workflow's
``KIT_REF`` and Play. An input written as an expression inside ``run:`` is pasted into the script
before the shell parses it, so its value could run as code; set in a step's ``env:``, the shell
only ever reads it as a variable. These tests read the workflow and hold it to that:

* the two inputs exist, as optional strings, and every expression that reads one falls back to
  ``env.KIT_REF`` or ``env.KIT_PLAY_WORKSPACE``, so a pull request or a push (no inputs) packs Play
  at ``KIT_REF``;
* an input is read only in a step's ``env:``, ``with:`` or ``if:``: never in ``run:``,
  ``working-directory`` or ``shell``, nor anywhere outside a step. ``kit_theme`` reaches the
  shell through ``env:``; ``kit_ref`` only names the kit checkouts' ref (``with:``);
* both jobs that install the kit do it with ``npm ci --ignore-scripts``, no expression reads a
  secret, and the workflow's permissions only read.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import yaml

_REPO = Path(__file__).resolve().parents[2]
_WORKFLOW = _REPO / ".github" / "workflows" / "browser-e2e.yml"
_INPUTS = ("kit_ref", "kit_theme")
# What each input falls back to when a run has none.
_FALLBACK = {"kit_ref": "env.KIT_REF", "kit_theme": "env.KIT_PLAY_WORKSPACE"}
# Step keys whose values the runner hands to an action or evaluates itself, never to a shell.
_STEP_KEYS_FOR_INPUTS = ("env", "with", "if")
_EXPRESSION = re.compile(r"\$\{\{(.*?)\}\}", re.S)
_INPUT_READ = re.compile(r"\b(?:github\.event\.)?inputs\.(\w+)|\binputs\[")
_SECRET_READ = re.compile(r"\bsecrets\b")
_SHELL_KEYS = ("run", "working-directory", "shell")
_KIT_INSTALL = re.compile(r"\bnpm ci\b")
_READ = "read"


def _workflow() -> Dict[str, Any]:
    return yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))


def _triggers(workflow: Dict[str, Any]) -> Dict[str, Any]:
    return workflow.get("on", workflow.get(True))  # PyYAML reads the bare `on:` key as True


def _strings(node: Any, where: Tuple[str, ...]) -> Iterator[Tuple[Tuple[str, ...], str]]:
    """Every string under ``node`` with the keys that lead to it."""
    if isinstance(node, str):
        yield where, node
    elif isinstance(node, dict):
        for key, value in node.items():
            yield from _strings(value, (*where, str(key)))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _strings(value, (*where, str(index)))


def _input_reads(text: str) -> List[Tuple[str, str]]:
    """``(input name, the expression)`` for each input an expression in ``text`` reads."""
    reads = []
    for expression in _EXPRESSION.findall(text):
        reads += [(match.group(1) or "[]", expression.strip()) for match in _INPUT_READ.finditer(expression)]
    return reads


def _reads_by_place() -> List[Tuple[Tuple[str, ...], str, str]]:
    """Each input read anywhere in the workflow: where it is, which input and the expression."""
    places = []
    for where, text in _strings(_workflow(), ()):
        places += [(where, name, expression) for name, expression in _input_reads(text)]
    return places


def _is_step_key(where: Tuple[str, ...], keys: Tuple[str, ...]) -> bool:
    """``jobs.<job>.steps.<n>.<key>...`` with ``key`` one of ``keys``."""
    return len(where) >= 5 and where[0] == "jobs" and where[2] == "steps" and where[4] in keys


def test_the_dispatch_takes_kit_ref_and_kit_theme_as_optional_strings():
    workflow = _workflow()
    inputs = (_triggers(workflow)["workflow_dispatch"] or {})["inputs"]
    assert set(inputs) == set(_INPUTS)
    for name in _INPUTS:
        assert inputs[name]["type"] == "string" and inputs[name]["required"] is False, name
    assert inputs["kit_ref"]["default"] == ""
    assert inputs["kit_theme"]["default"] == workflow["env"]["KIT_PLAY_WORKSPACE"] == "themes/play"


def test_every_read_of_an_input_falls_back_to_kit_ref_or_play():
    reads = _reads_by_place()
    assert {name for _, name, _ in reads} == set(_INPUTS)
    for where, name, expression in reads:
        assert f"inputs.{name} || {_FALLBACK[name]}" in expression, (".".join(where), expression)


def test_an_input_is_read_only_in_a_steps_env_with_or_if():
    for where, _, expression in _reads_by_place():
        assert _is_step_key(where, _STEP_KEYS_FOR_INPUTS), (".".join(where), expression)


def test_no_run_reads_an_input_and_kit_theme_reaches_the_shell_through_env():
    for job in _workflow()["jobs"].values():
        for step in job.get("steps", []):
            for key in _SHELL_KEYS:
                assert _input_reads(str(step.get(key, ""))) == [], (step.get("name"), key)
    through_env = {name for where, name, _ in _reads_by_place() if _is_step_key(where, ("env",))}
    assert "kit_theme" in through_env


def test_the_kit_installs_without_scripts_the_lane_reads_no_secret_and_only_reads():
    workflow = _workflow()
    installs = [
        step["run"] for job in workflow["jobs"].values() for step in job.get("steps", [])
        if step.get("working-directory") == "kit" and _KIT_INSTALL.search(str(step.get("run", "")))
    ]
    assert len(installs) == 2  # the package job's, and the browser job's for the kit's CLI
    assert all("npm ci --ignore-scripts" in run for run in installs), installs
    expressions = [found for _, text in _strings(workflow, ()) for found in _EXPRESSION.findall(text)]
    assert expressions and not [found for found in expressions if _SECRET_READ.search(found)]
    assert workflow["permissions"] == {"contents": _READ}
    for job in workflow["jobs"].values():
        assert set((job.get("permissions") or {}).values()) <= {_READ}


def test_kit_ref_is_one_full_commit():
    assert re.fullmatch(r"[0-9a-f]{40}", _workflow()["env"]["KIT_REF"])
