"""Reading `hyperframes check --json` (0.8.62) into the job report.

The CLI prints one JSON envelope: ``ok`` plus a section each for lint, runtime,
layout, motion and contrast, every section with its counts and findings. A
composition passes when no section has an error (warnings do not block; that is
the CLI's own non-strict verdict). When the check itself fails to run it prints
``{"ok": false, "error": "…"}`` instead: that is a refusal too, since nothing
may render unchecked.

Finding paths are made relative to the composition, so the scratch directory
never leaks into an answer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

SECTIONS = ("lint", "runtime", "layout", "motion", "contrast")
# A refusal carries every error and as many warnings as fit under this cap.
MAX_FINDINGS = 200
# containerSelector and text name the other block of an overlap, and what it reads.
_EXTRA_KEYS = ("selector", "containerSelector", "text", "time", "fixHint", "ratio", "requiredRatio", "suggestedColor")
# The CLI prints the envelope with console.log at the start of a line; a log
# line that carries JSON of its own ("[INFO] … {…}") never starts with a brace.
_LINE_OPENS_OBJECT = re.compile(r"^\{", re.MULTILINE)


class CheckReportError(ValueError):
    """The check printed nothing that reads as a report."""


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    summary: Mapping[str, Any]
    findings: Tuple[Mapping[str, Any], ...]


def _envelope(stdout: str) -> Dict[str, Any]:
    """The first object opening a line that carries a verdict (``ok``)."""
    decoder = json.JSONDecoder()
    problem = "hyperframes check printed no JSON"
    for opening in _LINE_OPENS_OBJECT.finditer(stdout):
        try:
            data, _ = decoder.raw_decode(stdout, opening.start())
        except ValueError as exc:
            problem = f"hyperframes check printed unreadable JSON: {exc}"
            continue
        if isinstance(data, dict) and "ok" in data:
            return data
        problem = "hyperframes check printed JSON without a verdict"
    raise CheckReportError(problem)


def _relative(source: Any, project_dir: Path) -> Optional[str]:
    if not isinstance(source, str) or not source:
        return None
    path = Path(source)
    try:
        return str(path.resolve().relative_to(project_dir.resolve()))
    except ValueError:
        return path.name


def _finding(section: str, raw: Mapping[str, Any], project_dir: Path) -> Dict[str, Any]:
    finding: Dict[str, Any] = {
        "section": section,
        "severity": str(raw.get("severity") or "error"),
        "code": str(raw.get("code") or f"{section}_finding"),
        "message": str(raw.get("message") or ""),
    }
    if not finding["message"] and "ratio" in raw:
        finding["message"] = f"text contrast {raw.get('ratio')}:1, needs {raw.get('requiredRatio')}:1"
    finding.update({key: raw[key] for key in _EXTRA_KEYS if raw.get(key) is not None})
    source = _relative(raw.get("sourceFile"), project_dir)
    if source:
        finding["source"] = source
    return finding


def _count(section: Mapping[str, Any], key: str, findings: List[Dict[str, Any]], severity: str) -> int:
    value = section.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return sum(1 for finding in findings if finding["severity"] == severity)


def parse_check_output(stdout: str, project_dir: Path) -> CheckResult:
    data = _envelope(stdout)
    if "error" in data and not any(name in data for name in SECTIONS):
        message = str(data.get("error") or "hyperframes check failed")
        finding = {"section": "check", "severity": "error", "code": "check_failed", "message": message}
        summary = {"ok": False, "errors": 1, "warnings": 0, "sections": {}}
        return CheckResult(ok=False, summary=summary, findings=(finding,))

    sections: Dict[str, Dict[str, Any]] = {}
    findings: List[Dict[str, Any]] = []
    for name in SECTIONS:
        section = data.get(name) if isinstance(data.get(name), dict) else {}
        raw = section.get("findings") if isinstance(section.get("findings"), list) else []
        parsed = [_finding(name, item, project_dir) for item in raw if isinstance(item, dict)]
        errors = _count(section, "errorCount", parsed, "error")
        warnings = _count(section, "warningCount", parsed, "warning")
        sections[name] = {"ok": bool(section.get("ok", errors == 0)), "errors": errors, "warnings": warnings}
        findings.extend(parsed)

    ok = bool(data.get("ok")) and all(section["ok"] for section in sections.values())
    findings.sort(key=lambda finding: finding["severity"] != "error")
    meta = data.get("_meta") if isinstance(data.get("_meta"), dict) else {}
    summary = {
        "ok": ok,
        "errors": sum(section["errors"] for section in sections.values()),
        "warnings": sum(section["warnings"] for section in sections.values()),
        "sections": sections,
        "hyperframes": meta.get("version"),
    }
    return CheckResult(ok=ok, summary=summary, findings=tuple(findings[:MAX_FINDINGS]))
