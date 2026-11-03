"""PRD-270 W2 US-270-12b (K8, K9) — a plugin's scan report on Automatos Cloud, as its bundle carries it.

The report is the PluginSecurityScan row's (plugin_security_scanner.py): its verdict, its static status and
findings, its LLM status, risk score, findings and summary, and when it ran. A static finding keeps only
its type, severity, file, line and description: never its pattern or matched_text, which quote a file's
content. An LLM finding keeps only its severity and description: never its category, file or evidence.

The scanner leaves some of it free, so it is fitted before the strict check: findings highest severity
first and at most FINDINGS_MAX of each kind (the scanner has no cap), a description cut to its cap, an
LLM severity the format doesn't know read as 'medium' (the scanner's own default for a finding with
none), a missing summary as none, a negative risk score as 0. What the format still refuses, a critical
static finding beside a 'safe' verdict or a missing risk score, leaves the plugin out (export_plugins).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, List

from services.marketplace_bundles.schema import FINDING_DESCRIPTION_MAX, FINDINGS_MAX, SCAN_SUMMARY_MAX, SEVERITIES

STATIC_FINDING_KEYS = ("type", "severity", "file", "line", "description")
DEFAULT_LLM_SEVERITY = "medium"  # plugin_security_scanner's LLM parse: f.get("severity", "medium")
_RANK = {severity: rank for rank, severity in enumerate(SEVERITIES)}

Finding = Dict[str, Any]


def scan_report(row: Any) -> Dict[str, Any]:
    """The scan object of a plugin's bundle, from ``row``'s scan columns."""
    risk = row.llm_risk_score
    return {
        "verdict": row.overall_verdict,
        "staticStatus": row.static_scan_status,
        "staticFindings": findings_of(row.static_findings, static_finding),
        "llmStatus": row.llm_scan_status,
        "riskScore": max(risk, 0) if isinstance(risk, int) and not isinstance(risk, bool) else risk,
        "summary": cut(row.llm_summary, SCAN_SUMMARY_MAX),
        "findings": findings_of(row.llm_findings, llm_finding),
        "scannedAt": row.scanned_at.isoformat() if isinstance(row.scanned_at, datetime) else row.scanned_at,
    }


def findings_of(value: Any, shape: Callable[[Dict[str, Any]], Finding]) -> List[Finding]:
    """The stored findings that are objects, each through ``shape``, highest severity first (the stored
    order within one severity), at most FINDINGS_MAX."""
    if not isinstance(value, list):
        return []
    shaped = [shape(finding) for finding in value if isinstance(finding, dict)]
    return sorted(shaped, key=_rank)[:FINDINGS_MAX]


def _rank(finding: Finding) -> int:
    """A finding's place: a known severity by its rank, anything else last (for the validator to refuse)."""
    severity = finding.get("severity")
    return _RANK.get(severity, len(SEVERITIES)) if isinstance(severity, str) else len(SEVERITIES)


def static_finding(finding: Dict[str, Any]) -> Finding:
    """A static finding's type, severity, file, line and description; its description cut to its cap."""
    kept = {key: finding[key] for key in STATIC_FINDING_KEYS if key in finding}
    if isinstance(kept.get("description"), str):
        return {**kept, "description": kept["description"][:FINDING_DESCRIPTION_MAX]}
    return kept


def llm_finding(finding: Dict[str, Any]) -> Finding:
    """An LLM finding's severity, one the format knows, and its description cut to its cap."""
    description = cut(finding.get("description"), FINDING_DESCRIPTION_MAX)
    return {"severity": severity_of(finding.get("severity")), "description": description}


def severity_of(value: Any) -> str:
    """``value`` lower-cased when it is a severity the format knows, else the scanner's default."""
    text = value.strip().lower() if isinstance(value, str) else ""
    return text if text in _RANK else DEFAULT_LLM_SEVERITY


def cut(value: Any, top: int) -> str:
    """``value`` cut to ``top`` characters when it is text, else none."""
    return value[:top] if isinstance(value, str) else ""


__all__ = ["cut", "findings_of", "llm_finding", "scan_report", "severity_of", "static_finding"]
