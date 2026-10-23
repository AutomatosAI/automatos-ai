"""PRD-265 US-006 — every model-call construction site is classified (decision D-3).

The billing report counted 31 system call sites that built an LLM manager on the
platform's credentials outside any trial or credit check: 28 ``create_llm_manager``
calls and 3 direct ``LLMManager(service_name=…)`` constructions. The credit gate now
sits in ``LLMManager`` itself, and decides by the manager's purpose
(``request_type``, else ``service_name``) whether a call is platform overhead
(absorbed, recorded) or customer work (gated and charged) —
``services/credits/classification.py``.

This test reads every construction site in the orchestrator's code and fails on one
whose purpose is in neither set, so a new call site is classified on purpose, not by
the "unknown means customer" default.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterator, Optional

from services.credits.classification import CUSTOMER_PURPOSES, OVERHEAD_PURPOSES, call_class

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {"tests", "alembic", "scripts", "__pycache__", ".venv", "venv", "node_modules"}
CONSTRUCTORS = {"create_llm_manager", "LLMManager"}
# Construction sites that are not a call site of their own: the factory function's own
# body, the failover sibling (same purpose as the manager it replaces), the agent
# factory's managers (``config=`` given, purpose "orchestrator": every agent's calls),
# and the decisions adapter / orchestrator answers, which pass their purpose through.
PASS_THROUGH = {"core/llm/manager.py", "core/llm/failover.py"}


def _python_files() -> Iterator[Path]:
    for path in ROOT.rglob("*.py"):
        if not SKIP_DIRS.intersection(path.relative_to(ROOT).parts):
            yield path


def _constants(tree: ast.Module) -> dict:
    return {
        target.id: node.value.value
        for node in tree.body if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
    }


def _literal(node: Optional[ast.AST], constants: dict) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    return None


def _purpose(call: ast.Call, constants: dict) -> Optional[str]:
    """The manager's purpose: ``request_type`` if given, else ``service_name``
    (keyword, or the first positional of ``create_llm_manager``), else the default."""
    keywords = {k.arg: k.value for k in call.keywords if k.arg}
    request_type = _literal(keywords.get("request_type"), constants)
    if request_type:
        return request_type
    service = keywords.get("service_name") or (call.args[0] if call.args else None)
    if service is None:
        return "orchestrator"  # both constructors default service_name to "orchestrator"
    return _literal(service, constants)


def _call_sites() -> Iterator[tuple[str, int, Optional[str]]]:
    for path in _python_files():
        rel = path.relative_to(ROOT).as_posix()
        if rel in PASS_THROUGH:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        constants = _constants(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
            if name in CONSTRUCTORS:
                yield rel, node.lineno, _purpose(node, constants)


def test_every_construction_site_is_classified():
    unclassified = [
        f"{rel}:{line} purpose={purpose!r}"
        for rel, line, purpose in _call_sites()
        if purpose is None or purpose not in (OVERHEAD_PURPOSES | CUSTOMER_PURPOSES)
    ]
    assert not unclassified, (
        "LLM manager construction sites whose purpose is not classified as overhead or "
        "customer work (services/credits/classification.py, decision D-3): " + ", ".join(unclassified)
    )


def test_the_report_s_system_call_sites_are_all_still_found():
    sites = list(_call_sites())
    assert len(sites) >= 31, f"expected the 31 system call sites at least, found {len(sites)}"


def test_overhead_and_customer_never_overlap():
    assert not OVERHEAD_PURPOSES & CUSTOMER_PURPOSES


def test_an_unknown_purpose_is_gated_as_customer_work():
    assert call_class("some_new_helper") == "customer"
    assert call_class("memory_distill") == "overhead"
