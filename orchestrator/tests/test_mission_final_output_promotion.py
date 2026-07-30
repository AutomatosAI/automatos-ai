"""Tests for the mission final-output deliverable promotion (2026-07-30).

Covers the pure selection helper — the registration path itself is fail-soft
and exercised by the coordinator integration suite.
"""

from types import SimpleNamespace

from services.coordinator_service import pick_final_output_task


def _task(seq, output, title="t"):
    return SimpleNamespace(sequence_number=seq, output=output, title=title)


def test_picks_last_sequence_with_output():
    tasks = [
        _task(1, "extraction"),
        _task(2, "research"),
        _task(6, "the final report"),
        _task(5, "qa notes"),
    ]
    assert pick_final_output_task(tasks).output == "the final report"


def test_skips_trailing_tasks_without_output():
    tasks = [
        _task(4, "draft"),
        _task(5, None),
        _task(6, "   "),
    ]
    assert pick_final_output_task(tasks).output == "draft"


def test_parallel_group_tie_resolves_to_later_task():
    tasks = [
        _task(4, "draft report", title="Draft"),
        _task(4, "validation appendix", title="Appendix"),
    ]
    assert pick_final_output_task(tasks).title == "Appendix"


def test_no_outputs_returns_none():
    assert pick_final_output_task([_task(1, None), _task(2, "")]) is None


def test_empty_and_none_inputs():
    assert pick_final_output_task([]) is None
    assert pick_final_output_task(None) is None


def test_none_sequence_sorts_first():
    tasks = [_task(None, "early"), _task(3, "late")]
    assert pick_final_output_task(tasks).output == "late"
