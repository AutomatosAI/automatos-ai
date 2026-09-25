"""Reading `hyperframes check --json` (US-102). PASSED is the envelope 0.8.62
printed for the fixture in CI run 36131814406; FAILED adds a lint error and a
contrast error the way the CLI reports them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from media_render.check_report import CheckReportError, parse_check_output

PROJECT = Path("/tmp/media-render/0123/project")

PASSED = {
    "ok": True,
    "strict": False,
    "lint": {
        "ok": True,
        "errorCount": 0,
        "warningCount": 1,
        "infoCount": 0,
        "findings": [
            {
                "code": "nested_structure_needs_subcomposition",
                "severity": "warning",
                "message": '<section id="scene"> is a timeline element that contains nested <h1 id="title">.',
                "selector": "#scene",
                "sourceFile": str(PROJECT / "index.html"),
                "bbox": {"x": 0, "y": 0, "width": 0, "height": 0},
                "time": 0,
                "fixHint": "Move <section id=\"scene\"> into a sub-composition file.",
            }
        ],
        "filesScanned": 1,
    },
    "runtime": {"ok": True, "errorCount": 0, "warningCount": 0, "infoCount": 0, "findings": []},
    "layout": {"ok": True, "errorCount": 0, "warningCount": 0, "infoCount": 0, "findings": [], "duration": 3},
    "motion": {"ok": True, "errorCount": 0, "warningCount": 0, "infoCount": 0, "findings": [], "enabled": False},
    "contrast": {"ok": True, "errorCount": 0, "warningCount": 0, "infoCount": 0, "findings": [], "checked": 4, "passed": 4},
    "_meta": {"version": "0.8.62", "updateAvailable": False},
}

LINT_ERROR = {
    "code": "non_deterministic_code",
    "severity": "error",
    "message": "Script contains `Math.random()` which produces non-deterministic output.",
    "fixHint": "Remove time-dependent code.",
}
CONTRAST_ERROR = {
    "selector": "#tagline",
    "ratio": 2.9,
    "requiredRatio": 4.5,
    "time": 1.5,
    "suggestedColor": "#F07A50",
    "sourceFile": str(PROJECT / "index.html"),
    "severity": "error",
}


def failed():
    report = json.loads(json.dumps(PASSED))
    report["ok"] = False
    report["lint"].update(ok=False, errorCount=1, findings=report["lint"]["findings"] + [LINT_ERROR])
    report["contrast"].update(ok=False, errorCount=1, findings=[CONTRAST_ERROR])
    return report


def test_a_passing_check_passes_with_its_warning_kept():
    result = parse_check_output(json.dumps(PASSED, indent=2), PROJECT)
    assert result.ok
    assert result.summary["errors"] == 0 and result.summary["warnings"] == 1
    assert result.summary["hyperframes"] == "0.8.62"
    assert result.summary["sections"]["lint"] == {"ok": True, "errors": 0, "warnings": 1}
    (warning,) = result.findings
    assert warning["section"] == "lint" and warning["severity"] == "warning"
    assert warning["source"] == "index.html", "the scratch path never leaks into an answer"
    assert "bbox" not in warning


def test_errors_fail_the_check_and_come_first():
    result = parse_check_output(json.dumps(failed()), PROJECT)
    assert not result.ok
    assert result.summary["errors"] == 2
    assert [f["code"] for f in result.findings][:2] == ["non_deterministic_code", "contrast_finding"]
    contrast = result.findings[1]
    assert contrast["section"] == "contrast" and contrast["ratio"] == 2.9 and contrast["suggestedColor"] == "#F07A50"
    assert "4.5:1" in contrast["message"]


def test_a_check_that_could_not_run_is_a_refusal():
    result = parse_check_output('{"ok": false, "error": "No composition root found", "_meta": {}}', PROJECT)
    assert not result.ok
    assert result.findings == ({"section": "check", "severity": "error", "code": "check_failed", "message": "No composition root found"},)


def test_text_around_the_envelope_is_ignored():
    assert parse_check_output("warming up\n" + json.dumps(PASSED) + "\ndone\n", PROJECT).ok


def test_json_inside_a_log_line_never_stands_in_for_the_envelope():
    # A stray {"ok": true} in a log line must not turn a failed check into a pass.
    noisy = '[INFO] [Compiler] resolved {"ok": true, "fonts": 1}\n' + json.dumps(failed(), indent=2)
    result = parse_check_output(noisy, PROJECT)
    assert not result.ok and result.summary["errors"] == 2


@pytest.mark.parametrize("stdout", ["", "no json here", '{"lint": {}}', "[1, 2]"])
def test_no_report_is_an_error_not_a_pass(stdout):
    with pytest.raises(CheckReportError):
        parse_check_output(stdout, PROJECT)


def test_an_overlap_names_both_text_blocks():
    report = json.loads(json.dumps(PASSED))
    overlap = {
        "code": "content_overlap",
        "severity": "error",
        "message": "Two text blocks overlap and may render unreadable.",
        "selector": "div.colon",
        "containerSelector": "#mm1",
        "text": ":",
        "time": 21.5,
    }
    report["ok"] = False
    report["layout"].update(ok=False, errorCount=1, findings=[overlap])
    (finding, _) = parse_check_output(json.dumps(report), PROJECT).findings
    assert (finding["selector"], finding["containerSelector"], finding["text"]) == ("div.colon", "#mm1", ":")
