"""F309 (night 9): a repeat of a refused call is told it was refused, never that it ran.

#0027.2: sending one card back took four messages. In the last one Auto called
platform_update_task_status twice with the same words, and the second call was
skipped as "already executed with identical parameters". Auto read the skip as proof
("This means the action *was* indeed taken in the previous turn"). A skip of a call
that FAILED this turn would say the same, though nothing was done. The skip now says
which: a refused call's repeat names the refusal and that nothing was done; a call that
worked is said to have run once. A changed call (the owner's own words) still runs.
"""
from __future__ import annotations

from modules.tools.execution.tool_execution_tracker import ToolExecutionTracker

PARAPHRASE = {"action": "platform_update_task_status",
              "params": {"task_id": "0027.2", "status": "assigned", "note": "Please revise the intro."}}
THEIR_WORDS = {"action": "platform_update_task_status",
               "params": {"task_id": "0027.2", "status": "assigned",
                          "note": "'delightful' and 'exquisite' are banned in our brand voice."}}
REFUSAL = "A note on the card is signed as the owner's, so it is their own words. Nothing was done."


def _ran(tracker, args, result):
    tracker.record_execution("platform_execute", args)
    tracker.record_outcome("platform_execute", args, result)


def test_a_repeat_of_a_refused_call_says_it_was_refused_and_nothing_was_done():
    tracker = ToolExecutionTracker()
    _ran(tracker, PARAPHRASE, {"success": False, "error": REFUSAL})

    skipped, reason = tracker.should_skip_execution("platform_execute", PARAPHRASE)

    assert skipped and "already failed" in reason and REFUSAL in reason and "Nothing was done" in reason
    assert "executed" not in reason                                  # night 9: "already executed"
    assert tracker.should_skip_execution("platform_execute", THEIR_WORDS) == (False, "")   # the owner's words go


def test_a_repeat_of_a_call_that_worked_says_it_ran_once():
    tracker = ToolExecutionTracker()
    _ran(tracker, THEIR_WORDS, {"success": True, "status": "in_progress"})

    skipped, reason = tracker.should_skip_execution("platform_execute", THEIR_WORDS)

    assert skipped and "identical parameters" in reason and "not run a second time" in reason
