"""PRD-185 S5: guard against silently-severed ``core.services`` imports.

A 2026-03 rename (``recipe_*`` -> ``playbook_*``) left three
``from core.services.recipe_*`` imports pointing at deleted modules; the
``ImportError`` was swallowed, so the playbook learning loop wrote memories it
never read for 3.5 months. This test statically resolves every
``from core.services.<mod> import <Name>`` in the orchestrator against the target
module's AST — no runtime import, no heavy deps — so the next rename fails CI
instead of a swallowed log line.
"""
import ast
from pathlib import Path

_ORCH = Path(__file__).resolve().parent.parent
_SERVICES = _ORCH / "core" / "services"
_SKIP_DIRS = {"venv", ".venv", "node_modules", "__pycache__", "model_cache", ".git"}


def _iter_service_imports():
    """Yield (source_file, module, name, lineno) for every
    ``from core.services.<module> import <Name>`` in the orchestrator tree."""
    for py in _ORCH.rglob("*.py"):
        if _SKIP_DIRS & set(py.parts):
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("core.services.")
            ):
                mod = node.module[len("core.services."):]
                if "." in mod:  # only single-segment service modules
                    continue
                for alias in node.names:
                    if alias.name != "*":
                        yield py, mod, alias.name, node.lineno


def _module_defines(mod_path: Path):
    """Top-level class/def/assignment names a module defines (AST only)."""
    names = set()
    try:
        tree = ast.parse(mod_path.read_text(encoding="utf-8"), filename=str(mod_path))
    except (SyntaxError, UnicodeDecodeError):
        return names
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
    return names


def test_core_services_imports_resolve():
    """Every ``from core.services.X import Y`` must resolve to a real module + symbol."""
    unresolved = []
    for src, mod, name, lineno in _iter_service_imports():
        mod_path = _SERVICES / f"{mod}.py"
        if not mod_path.exists():
            pkg_init = _SERVICES / mod / "__init__.py"
            if not pkg_init.exists():
                unresolved.append(
                    f"{src.relative_to(_ORCH)}:{lineno} -> missing module core.services.{mod}"
                )
                continue
            mod_path = pkg_init
        if name not in _module_defines(mod_path):
            unresolved.append(
                f"{src.relative_to(_ORCH)}:{lineno} -> core.services.{mod} has no '{name}'"
            )
    assert not unresolved, "Severed core.services imports:\n" + "\n".join(unresolved)


def test_no_legacy_recipe_service_imports():
    """The specific 2026-03 regression: no import of the deleted recipe_* services."""
    offenders = [
        f"{src.relative_to(_ORCH)}:{lineno} -> core.services.{mod}"
        for src, mod, name, lineno in _iter_service_imports()
        if mod in {"recipe_memory_service", "recipe_learning_service"}
    ]
    assert not offenders, "Legacy recipe_* service imports still present:\n" + "\n".join(offenders)
