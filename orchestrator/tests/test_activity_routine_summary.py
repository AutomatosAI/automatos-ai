"""PRD-244 review — a routine row says what the heartbeat is for and what it found."""
from services.activity_service import ActivityService


def test_purpose_and_finding_make_the_line():
    findings = [
        {"check": "agent_health", "detail": "Agent responsive"},
        {"check": "llm_analysis", "detail": "Two invoices are overdue; flagged both.\nMore detail here."},
    ]
    assert (
        ActivityService._routine_summary(findings, "Watch the invoice queue every hour.")
        == "Watch the invoice queue every hour — Two invoices are overdue; flagged both"
    )


def test_purpose_alone_and_finding_alone():
    assert ActivityService._routine_summary([], "Check the build") == "Check the build"
    assert ActivityService._routine_summary([{"check": "checklist", "detail": "3 of 3 passed"}]) == "3 of 3 passed"


def test_counts_when_no_informative_check_and_never_fabricates():
    assert ActivityService._routine_summary([{"check": "error", "detail": "boom"}, {"check": "x"}]) == "Checked 2 items"
    assert ActivityService._routine_summary(None) == "Routine check completed"
    assert ActivityService._routine_summary("not json") == "Routine check completed"


def test_long_lines_are_trimmed_to_one_line():
    line = ActivityService._routine_summary([{"check": "llm_analysis", "detail": "x" * 300}], "p" * 100)
    head, finding = line.split(" — ")
    assert head.endswith("…") and len(head) <= 60
    assert finding.endswith("…") and len(finding) <= 140
