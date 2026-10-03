"""platform_create_task's approval_action names the actions an approval can run.

It was a bare object whose only contract was one inline example, and
auto_approve marked a ticket done, reporting action_executed, for any type while
only publish_blog ran.
"""
from modules.tools.discovery.actions_board_tasks import _create_task_properties


def test_approval_action_lists_the_actions_an_approval_runs():
    prop = _create_task_properties()["approval_action"]
    assert prop["properties"]["type"]["enum"] == ["publish_blog", "create_blog"]
    assert prop["required"] == ["type"]
