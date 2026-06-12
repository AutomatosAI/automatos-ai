"""PRD-161 S4 — StuckDetector (same-action-loop breaker). Pure, no DB."""
import pytest

from core.utils.stuck_detector import StuckDetector, action_key


def test_same_action_three_times_is_stuck():
    d = StuckDetector(threshold=3)
    k = action_key("SLACK_SEND_MESSAGE", {"channel": "x", "text": "hi"})
    d.record(k)
    d.record(k)
    assert d.is_stuck() is False  # only 2 in a row
    d.record(k)
    assert d.is_stuck() is True   # 3 in a row → stuck


def test_varied_actions_never_stuck():
    d = StuckDetector(threshold=3)
    for i in range(6):
        d.record(action_key("TOOL", {"i": i}))
    assert d.is_stuck() is False


def test_breaks_within_bounds_not_after_full_budget():
    """An induced identical-failing-action loop is flagged the moment the
    threshold is hit — not after exhausting the iteration budget."""
    d = StuckDetector(threshold=3)
    k = action_key("GITHUB_LIST_REPOS", {"q": "fail"})
    iters = 0
    for _ in range(50):  # would-be iteration budget
        d.record(k)
        iters += 1
        if d.is_stuck():
            break
    assert iters == 3, "must break exactly when the 3rd identical action lands"


def test_recovered_loop_then_repeat_restarts_detection():
    d = StuckDetector(threshold=3)
    k1 = action_key("A", {})
    k2 = action_key("B", {})
    d.record(k1); d.record(k2); d.record(k1)  # not 3 in a row
    assert d.is_stuck() is False
    d.record(k1); d.record(k1)  # now k1 three in a row (last three)
    assert d.is_stuck() is True


def test_action_key_stable_across_dict_ordering():
    assert action_key("T", {"a": 1, "b": 2}) == action_key("T", {"b": 2, "a": 1})


def test_action_key_differs_on_args():
    assert action_key("T", {"a": 1}) != action_key("T", {"a": 2})


def test_threshold_must_be_at_least_two():
    with pytest.raises(ValueError):
        StuckDetector(threshold=1)
