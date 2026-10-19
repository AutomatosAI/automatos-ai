"""PRD-263 US-015 — every key DIAGNOSE reads exists in the read-model row.

At 697ca7d6e DIAGNOSE read ``success_rate``, ``cost_7d``, ``efficiency_score`` and
``error_rate`` off ``platform_list_agents`` rows that never carry them, so every delta
was 0 and every agent STABLE. DIAGNOSE (``services.harness_diagnose``, and the DIAGNOSE
method of ``HarnessService``) now reads agent rows only through the names in
``DIAGNOSE_READS``, and those names must be fields ``harness_metrics`` returns. The
tool-failure judgment (PRD-263 US-019) lives in ``harness_diagnose`` and reads the tool
rows, never an agent row, so it is scanned with the rest.

Agent rows are bound to the name ``row`` in the DIAGNOSE code. The AST scan collects
every ``row["..."]`` and ``row.get("...")`` key there; keys read through the metric
specs (``row[metric.x_field]``) are checked from the specs themselves. A second scan,
blind to the receiver's name, checks that no string key DIAGNOSE reads anywhere is one
of the legacy fields, so ``entry["cost_7d"]`` or ``rows[0]["success_rate"]`` fails too.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable, Set

from services import harness_diagnose as hd
from services import harness_metrics as hm

SERVICES = Path(__file__).resolve().parents[1] / "services"
ROW_NAME = "row"
# The DIAGNOSE methods of HarnessService; every function of harness_diagnose.py is DIAGNOSE.
# _diagnose_tool_failures went with US-019: harness_diagnose.tool_failure_issues replaced it.
SERVICE_DIAGNOSE_FUNCTIONS = {"_phase_diagnose"}
# The fields DIAGNOSE read at 697ca7d6e that no agent row carries.
LEGACY_FIELDS = {"success_rate", "cost_7d", "efficiency_score", "error_rate", "model", "heartbeat_cost_7d",
                 "token_usage_7d"}


def _undeclared(reads: Iterable[str]) -> Set[str]:
    """The contract check: names DIAGNOSE reads that the read-model row does not carry."""
    return set(reads) - set(hm.AGENT_ROW_FIELDS)


def _row_keys(node: ast.AST) -> Set[str]:
    """String keys read off ``row`` by subscript or ``.get()`` anywhere under ``node``."""
    keys: Set[str] = set()
    for sub in ast.walk(node):
        if (isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Name) and sub.value.id == ROW_NAME
                and isinstance(sub.slice, ast.Constant) and isinstance(sub.slice.value, str)):
            keys.add(sub.slice.value)
        if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr == "get"
                and isinstance(sub.func.value, ast.Name) and sub.func.value.id == ROW_NAME
                and sub.args and isinstance(sub.args[0], ast.Constant) and isinstance(sub.args[0].value, str)):
            keys.add(sub.args[0].value)
    return keys


def _string_keys(node: ast.AST) -> Set[str]:
    """Every string key read by subscript or ``.get()`` anywhere under ``node``, whatever the receiver."""
    keys: Set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Subscript) and isinstance(sub.slice, ast.Constant) and isinstance(sub.slice.value, str):
            keys.add(sub.slice.value)
        if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr == "get"
                and sub.args and isinstance(sub.args[0], ast.Constant) and isinstance(sub.args[0].value, str)):
            keys.add(sub.args[0].value)
    return keys


def _functions(path: Path, names=None):
    tree = ast.parse(path.read_text())
    return [node for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (names is None or node.name in names)]


def _diagnose_functions():
    """Every function of harness_diagnose.py and the DIAGNOSE methods of HarnessService."""
    service_functions = _functions(SERVICES / "harness_service.py", SERVICE_DIAGNOSE_FUNCTIONS)
    assert {f.name for f in service_functions} == SERVICE_DIAGNOSE_FUNCTIONS
    return _functions(SERVICES / "harness_diagnose.py") + service_functions


def _spec_fields() -> Set[str]:
    fields: Set[str] = set()
    for metric in hd.METRICS:
        if isinstance(metric, hd.RateMetric):
            fields |= {metric.x_field, *metric.n_fields}
        else:
            fields |= {metric.usd_field, metric.tokens_field}
    return fields


def test_every_name_diagnose_reads_is_a_read_model_field():
    assert len(set(hd.DIAGNOSE_READS)) == len(hd.DIAGNOSE_READS)
    assert set(hd.DIAGNOSE_READS) <= set(hm.AGENT_ROW_FIELDS)
    assert _undeclared(hd.DIAGNOSE_READS) == set()


def test_diagnose_reads_no_row_key_outside_diagnose_reads():
    keys = set().union(*(_row_keys(f) for f in _diagnose_functions()))
    assert keys, "the scan found no row reads at all: has the row variable been renamed?"
    assert keys <= set(hd.DIAGNOSE_READS), sorted(keys - set(hd.DIAGNOSE_READS))
    assert _spec_fields() <= set(hd.DIAGNOSE_READS), sorted(_spec_fields() - set(hd.DIAGNOSE_READS))


def test_diagnose_reads_no_legacy_field_through_any_name():
    keys = set().union(*(_string_keys(f) for f in _diagnose_functions()))
    assert keys, "the scan found no string-key reads at all"
    assert not keys & LEGACY_FIELDS, sorted(keys & LEGACY_FIELDS)


def test_the_contract_check_reports_an_undeclared_key():
    widened = (*hd.DIAGNOSE_READS, "success_rate")
    assert _undeclared(widened) == {"success_rate"}


def test_the_scan_catches_a_row_read_of_a_field_the_read_model_never_returns():
    probe = ast.parse("def judge(row):\n    return row['cost_7d'] + row.get('error_rate', 0)\n")
    assert _row_keys(probe) == {"cost_7d", "error_rate"}
    assert _undeclared(_row_keys(probe)) == {"cost_7d", "error_rate"}


def test_the_name_blind_scan_catches_a_legacy_read_whatever_the_receiver_is_called():
    probe = ast.parse("def judge(entry, r, rows):\n"
                      "    return entry['cost_7d'] + r.get('error_rate', 0) + rows[0]['success_rate']\n")
    assert _row_keys(probe) == set()
    assert _string_keys(probe) & LEGACY_FIELDS == {"cost_7d", "error_rate", "success_rate"}
