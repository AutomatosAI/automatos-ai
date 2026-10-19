"""PRD-262 US-032: heavy modules load when used, not when the app imports.

Every API process imports `main`, and `main` reaches the multimodal processors
(main → api.knowledge_multimodal → modules.rag → ingestion → multimodal), so whatever
those import at module level, every worker loads at boot. This suite checks:

- modules/rag/ingestion/multimodal/processors.py imports none of pdfplumber, Pillow
  or pytesseract at module level (a `TYPE_CHECKING` import for annotations is fine:
  `from __future__ import annotations` keeps them from being evaluated);
- every function there that uses one of them imports it in its own body, so a later
  edit that drops the module-level import from a caller is caught here, not at the
  first PDF a user uploads;
- the Composio availability probe finds the SDK with `importlib.util.find_spec`
  and never imports it: the boot's TOOL_SYNC stage asks the probe, and only a real
  call needs the SDK.

test.yml's backend job holds `import main` to FR-6 (scripts/ci/import_main_report.py,
tested in test_prd262_import_main_report.py). Pure file reads here, except the probe
tests, which stub the finder.
"""
from __future__ import annotations

import ast
import importlib.machinery
import importlib.util
from pathlib import Path

import pytest

import core.composio.client as cc
from config import config

ORCH = Path(__file__).resolve().parents[1]
PROCESSORS = ORCH / "modules" / "rag" / "ingestion" / "multimodal" / "processors.py"
CLIENT = ORCH / "core" / "composio" / "client.py"
# The module each heavy name comes from, as `import x` or `from x import name`.
HEAVY = {"pdfplumber": "pdfplumber", "Image": "PIL", "pytesseract": "pytesseract"}


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _is_type_checking(node: ast.AST) -> bool:
    test = getattr(node, "test", None)
    return isinstance(node, ast.If) and (
        (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING")
        or (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"))


def _module_level_imports(body: list[ast.stmt]) -> set[str]:
    """Top-level packages imported when the module runs: functions skipped, TYPE_CHECKING too."""
    found: set[str] = set()
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(node, ast.Import):
            found |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
        if _is_type_checking(node):
            found |= _module_level_imports(node.orelse)
            continue
        for field in ("body", "orelse", "finalbody", "handlers"):
            found |= _module_level_imports(getattr(node, field, []) or [])
    return found


def _bound_by_imports(function: ast.AST) -> set[str]:
    """The names the function's own import statements bind."""
    names: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(alias.asname or alias.name).split(".")[0] for alias in node.names}
    return names


def _names_used_in_body(function: ast.FunctionDef) -> set[str]:
    """Names the body reads. The signature's annotations are strings at runtime, so they don't count."""
    return {node.id for statement in function.body for node in ast.walk(statement)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}


def _functions(tree: ast.Module) -> list[ast.FunctionDef]:
    return [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]


# --------------------------------------------------------------------------- the processors


def test_the_processors_import_no_pdf_or_ocr_package_at_module_level():
    imported = _module_level_imports(_tree(PROCESSORS).body)
    assert not imported & set(HEAVY.values()), sorted(imported & set(HEAVY.values()))


def test_the_annotations_that_name_pillow_are_never_evaluated():
    tree = _tree(PROCESSORS)
    future = [node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == "__future__"]
    assert any(alias.name == "annotations" for node in future for alias in node.names)
    guarded = [node for node in tree.body if _is_type_checking(node)]
    assert guarded and {"PIL"} == _module_level_imports([stmt for node in guarded for stmt in node.body])


def test_every_function_that_uses_a_heavy_package_imports_it_itself():
    users = {}
    for function in _functions(_tree(PROCESSORS)):
        used = _names_used_in_body(function) & set(HEAVY)
        missing = used - _bound_by_imports(function)
        assert not missing, f"{function.name} uses {sorted(missing)} without importing it in its body"
        if used:
            users[function.name] = used
    # The four methods that open PDFs, crop images and read text from them.
    assert {"extract_images_from_pdf", "_process_pdf_image", "_extract_text_with_ocr",
            "process_pdf_multimodal"} <= set(users), users


# --------------------------------------------------------------------------- the Composio probe


def _probe_body() -> ast.FunctionDef:
    return next(node for node in _functions(_tree(CLIENT)) if node.name == "_probe_composio_availability")


def test_the_probe_never_calls_the_sdk_importer():
    called = {node.func.id for node in ast.walk(_probe_body())
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "_get_composio" not in called and "_composio_sdk_installed" in called


@pytest.fixture
def keyed(monkeypatch):
    """A Composio key, and an SDK importer that fails the test if anything calls it."""
    def _imported():
        raise AssertionError("the availability probe imported the Composio SDK")

    monkeypatch.setattr(config, "COMPOSIO_API_KEY", "test-key-not-a-secret")
    monkeypatch.setattr(cc, "_get_composio", _imported)
    cc.reset_composio_availability()
    yield
    cc.reset_composio_availability()


def _finder(monkeypatch, found: bool) -> list[str]:
    asked: list[str] = []

    def find_spec(name, package=None):
        asked.append(name)
        return importlib.machinery.ModuleSpec(name, None) if found else None

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    return asked


def test_an_installed_sdk_is_found_without_being_imported(keyed, monkeypatch):
    asked = _finder(monkeypatch, found=True)
    assert cc.composio_available() is True
    assert cc.composio_unavailable_reason() is None
    assert asked == [cc.COMPOSIO_SDK_PACKAGE] == ["composio"]


def test_a_missing_sdk_reads_as_not_installed(keyed, monkeypatch):
    asked = _finder(monkeypatch, found=False)
    assert cc.composio_available() is False
    assert cc.composio_unavailable_reason() == cc.COMPOSIO_UNAVAILABLE_NO_SDK
    assert asked == ["composio"]


def test_without_a_key_the_probe_does_not_look_for_the_sdk(monkeypatch):
    asked = _finder(monkeypatch, found=True)
    monkeypatch.setattr(config, "COMPOSIO_API_KEY", None)
    cc.reset_composio_availability()
    try:
        assert cc.composio_unavailable_reason() == cc.COMPOSIO_UNAVAILABLE_NO_KEY
        assert asked == []
    finally:
        cc.reset_composio_availability()
