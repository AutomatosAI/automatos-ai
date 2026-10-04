"""F297 (night 8): a mission step whose run wrote no answer is a failed attempt.

#0408.3's Operations Manager wrote the packing notes into the mission field, then
answered with 2 tokens; the step's output was "Based on the tool results:
**platform_execute**: {"success": true, "message": "Pattern
'packing_notes_completed' shared with the mission field."}" and it was marked
done. Such a step is now recorded as a failed attempt (retried while it has
attempts), with the plain result as its reason.
"""
from __future__ import annotations

from services.result_substance import TOOL_RESULTS_HEADER, as_step_failure, plain_no_answer

DUMP_0408_3 = (f"{TOOL_RESULTS_HEADER}\n\n**platform_execute**: {{\"success\": true, \"message\": "
               "\"Pattern 'packing_notes_completed' shared with the mission field.\"}")


def test_0408_3_is_a_failed_attempt_with_the_plain_reason():
    plain = plain_no_answer(DUMP_0408_3)
    out = as_step_failure({"status": "success", "result": plain, "execution": {"tokens_used": 12}})
    assert out["status"] == "error" and out["error"] == plain
    assert out["execution"] == {"tokens_used": 12}


def test_a_step_with_an_answer_or_an_error_is_left_as_it_is():
    answer = {"status": "success", "result": "Pack each tin with two 250 g bags and a card."}
    error = {"status": "error", "error": "Execution timed out after 300s"}
    assert as_step_failure(answer) is answer and as_step_failure(error) is error


def test_the_coordinator_records_the_step_through_it():
    import inspect

    from services.coordinator_service import CoordinatorService

    assert "record_task_completion(db, task, as_step_failure(result))" in inspect.getsource(
        CoordinatorService._record_task_result)
