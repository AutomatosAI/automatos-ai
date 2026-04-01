#!/usr/bin/env python3
"""Test suite inventory for weekly coverage-gap analysis.

Scans `tests/api`, `tests/regressions`, and `tests/contracts` and produces
a compact JSON summary your weekly "Test Coverage Gap Finder" recipe can consume.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
API_TESTS_DIR = TESTS_DIR / "api"
REGRESSIONS_DIR = TESTS_DIR / "regressions"
CONTRACTS_DIR = TESTS_DIR / "contracts"

EXPECTED_DOMAINS = {
    "workspaces",
    "chat",
    "agents",
    "memory",
    "tools",
    "skills",
    "knowledge",
    "documents",
    "routing",
    "channels",
    "heartbeat",
    "recipes",
    "workflows",
    "analytics",
    "models",
    "keys",
    "personas",
    "webhooks",
    # PRD-123 + PRD-82A additions
    "missions",
    "permissions",
    "health_bootstrap",
    "performance_baselines",
    # Error path coverage
    "agent_errors",
    "document_errors",
    "memory_errors",
    "heartbeat_errors",
    "workflow_errors",
    "channel_errors",
    "routing_errors",
    "persona_errors",
    "key_errors",
    "model_errors",
    "workspace_errors",
    "analytics_errors",
    "chat_errors",
    # Journey tests
    "mission_journeys",
    "onboarding_journey",
    "daily_workflow_journey",
    "admin_config_journey",
    "mission_research_journey",
    "integration_setup_journey",
    "user_journeys",
}


def _module_info(path: Path) -> dict:
    source = path.read_text()
    tree = ast.parse(source)
    test_names = []
    module_doc = ast.get_docstring(tree) or ""

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            test_names.append(node.name)

    domain = path.stem.replace("test_", "")
    return {
        "file": str(path.relative_to(TESTS_DIR.parent)),
        "domain": domain,
        "test_count": len(test_names),
        "tests": test_names,
        "is_journey_file": "Journey" in module_doc or "journey" in module_doc.lower(),
        "is_smoke_file": "smoke" in module_doc.lower(),
        "doc": module_doc.splitlines()[0] if module_doc else "",
    }


def build_summary() -> dict:
    test_files = sorted(API_TESTS_DIR.glob("test_*.py"))

    # Include regressions and contracts directories
    for extra_dir in (REGRESSIONS_DIR, CONTRACTS_DIR):
        if extra_dir.exists():
            test_files.extend(sorted(extra_dir.glob("test_*.py")))

    modules = [_module_info(path) for path in test_files]
    covered_domains = {module["domain"] for module in modules}

    total_tests = sum(module["test_count"] for module in modules)
    journey_files = [module["file"] for module in modules if module["is_journey_file"]]
    smoke_files = [module["file"] for module in modules if module["is_smoke_file"]]
    regression_files = [module["file"] for module in modules if "regressions" in module["file"]]
    contract_files = [module["file"] for module in modules if "contracts" in module["file"]]

    return {
        "total_api_test_files": len(modules),
        "total_api_tests": total_tests,
        "covered_domains": sorted(covered_domains),
        "missing_expected_domains": sorted(EXPECTED_DOMAINS - covered_domains),
        "journey_files": journey_files,
        "smoke_files": smoke_files,
        "regression_files": regression_files,
        "contract_files": contract_files,
        "modules": [
            {
                "file": module["file"],
                "domain": module["domain"],
                "test_count": module["test_count"],
                "is_journey_file": module["is_journey_file"],
                "is_smoke_file": module["is_smoke_file"],
                "doc": module["doc"],
            }
            for module in modules
        ],
    }


def main():
    print(json.dumps(build_summary(), indent=2))


if __name__ == "__main__":
    main()
