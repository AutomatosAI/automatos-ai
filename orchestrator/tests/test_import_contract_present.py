"""PRD-211 US-001 — the import-linter topology contract is present and valid.

Thesis T2 measured Automatos as a healthy modular monolith (~3.0% true
feature-to-feature import coupling; ~80.5% shared kernel) and its verdict —
"stay a monolith, harden the boundaries in-repo" — is only a claim until it is
enforced. `orchestrator/.importlinter` holds an `independence` contract that
makes the NEXT new lateral edge between feature modules fail loud in CI.

This test is PURE: it asserts the contract file exists and parses (configparser
only — no graph build, no network). The real enforcement is the `lint-imports`
CI lane (.github/workflows/import-linter.yml); this guard only locks in that the
contract stays present, well-formed, and keeps its two load-bearing invariants:
the permitted routing layer (modules.tools) stays excluded, and the tools
routing wildcard stays in the ignore list.
"""
from __future__ import annotations

import configparser
from pathlib import Path

_ORCH = Path(__file__).resolve().parents[1]
_CONTRACT = _ORCH / ".importlinter"
_CONTRACT_SECTION = "importlinter:contract:feature-module-independence"


def _parse():
    cp = configparser.ConfigParser()
    # ConfigParser.read silently skips a missing file; assert existence first so
    # a deleted contract fails here rather than as a confusing KeyError below.
    assert _CONTRACT.exists(), (
        f".importlinter missing at {_CONTRACT} — PRD-211 topology contract was "
        "deleted; the feature-module independence gate is unenforced"
    )
    cp.read(_CONTRACT)
    return cp


def test_import_contract_present():
    """Canonical PRD-211 US-001 guard: the contract file exists AND parses."""
    cp = _parse()  # asserts existence first, then parses
    assert cp.has_section("importlinter"), "[importlinter] top section missing"
    assert cp["importlinter"]["root_package"].strip() == "modules", (
        "root_package must be `modules` — features are imported as modules.* "
        "(there is no orchestrator/__init__.py)"
    )
    assert cp.has_section(_CONTRACT_SECTION), (
        f"independence contract section [{_CONTRACT_SECTION}] missing"
    )


def test_contract_is_independence_over_feature_modules():
    cp = _parse()
    contract = cp[_CONTRACT_SECTION]
    assert contract["type"].strip() == "independence", (
        "the topology contract must be of type `independence`"
    )
    modules = [m.strip() for m in contract["modules"].splitlines() if m.strip()]
    # A meaningful lock needs the real feature surface, not a token pair.
    assert len(modules) >= 15, (
        f"only {len(modules)} feature modules listed — the contract must cover "
        "the full feature surface under orchestrator/modules/*"
    )
    assert all(m.startswith("modules.") for m in modules), (
        f"every contracted module must be under `modules.`; got {modules}"
    )


def test_routing_layer_is_excluded():
    """modules.tools is the permitted routing layer — it must NOT be a
    contracted feature module, and its dispatch to any feature must stay in the
    ignore list. Losing either turns the router itself into a violation and the
    contract becomes un-green / meaningless."""
    cp = _parse()
    contract = cp[_CONTRACT_SECTION]
    modules = [m.strip() for m in contract["modules"].splitlines() if m.strip()]
    assert "modules.tools" not in modules, (
        "modules.tools is the permitted routing layer and must be EXCLUDED from "
        "the independence set (features route through it)"
    )
    ignore = contract["ignore_imports"]
    assert "modules.tools.** -> modules.**" in ignore, (
        "the tools routing wildcard is missing — the router dispatching to a "
        "feature would be flagged as lateral coupling and the contract breaks"
    )
