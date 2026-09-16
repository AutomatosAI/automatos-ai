"""PRD-242 S2/S4 — static guards for two one-line wirings.

* the local-edition first-run seed calls the starter-template seeder (a local
  install used to open the Template Studio to an empty gallery);
* the playbook ``generate_document`` step passes ``template_id`` and registers
  the file as a Deliverable (it used to accept only ``template_name`` and leave
  the document inside the step's output JSON).

AST reads — no boot, no DB — plus the pure step-config helper.
"""

from __future__ import annotations

import ast
from pathlib import Path
from uuid import UUID

import pytest

_ORCH = Path(__file__).resolve().parent.parent


def _calls_in_function(path: Path, func_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            names = set()
            for call in ast.walk(node):
                if isinstance(call, ast.Call):
                    f = call.func
                    names.add(f.id if isinstance(f, ast.Name) else getattr(f, "attr", ""))
            return names
    raise AssertionError(f"{func_name} not found in {path.name}")


def test_local_first_run_seeds_starter_templates():
    calls = _calls_in_function(_ORCH / "core" / "seeds" / "seed_local_first_run.py", "seed_local_first_run")
    assert "seed_starter_templates" in calls


def test_playbook_document_step_passes_template_id_and_registers_a_deliverable():
    calls = _calls_in_function(_ORCH / "api" / "recipe_executor.py", "_execute_recipe_inner")
    assert "register_as_deliverable" in calls
    assert "_document_step_config" in calls
    assert "share_link" in calls


def _document_step_config():
    try:
        from api.recipe_executor import _document_step_config
    except Exception as e:  # env without the heavy router deps
        pytest.skip(f"api.recipe_executor not importable in this env: {e}")
    return _document_step_config


def test_document_step_config_reads_inline_or_nested_config():
    fn = _document_step_config()
    tid = "11111111-1111-1111-1111-111111111111"
    inline = fn({"type": "generate_document", "title": "Weekly", "format": "docx", "template_id": tid, "data": {"a": 1}})
    nested = fn({"type": "generate_document", "config": {"title": "Weekly", "template_name": "Weekly Report"}})
    assert inline["template_id"] == UUID(tid) and inline["format"] == "docx" and inline["data"] == {"a": 1}
    assert nested["template_id"] is None and nested["template_name"] == "Weekly Report"
    assert nested["format"] == "pdf" and nested["data"] == {}


def test_document_step_config_rejects_a_non_uuid_template_id_loudly():
    fn = _document_step_config()
    with pytest.raises(ValueError, match="not a UUID"):
        fn({"type": "generate_document", "template_id": "Weekly Report"})
