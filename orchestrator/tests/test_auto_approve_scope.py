"""platform_create_task's auto_approve holds only for the action it can run.

create_board_task ran an approval_action on the spot only for publish_blog, yet
marked the ticket done and answered action_executed for any type. Any other
approval action now waits in Review for the owner.
"""
import asyncio

from modules.tools.discovery.auto_approve_scope import auto_approves_only_what_it_runs


def _seen(params):
    seen = {}

    async def handler(db, workspace_id, params):
        seen.update(params)
        return {"success": True}

    asyncio.run(auto_approves_only_what_it_runs(handler)(None, "ws", params))
    return seen


def test_publish_blog_keeps_auto_approve():
    params = {"auto_approve": True, "approval_action": {"type": "publish_blog", "post_id": "p1"}}
    assert _seen(params)["auto_approve"] is True


def test_create_blog_waits_for_the_owner():
    params = {"auto_approve": True, "approval_action": {"type": "create_blog", "topic": "Oat milk"}}
    assert _seen(params)["auto_approve"] is False


def test_an_invented_action_waits_for_the_owner():
    params = {"auto_approve": True, "planning_data": {"approval_action": {"type": "send_email"}}}
    assert _seen(params)["auto_approve"] is False


def test_a_call_without_auto_approve_is_passed_on_as_it_came():
    params = {"title": "t", "description": "d", "approval_action": {"type": "create_blog"}}
    assert _seen(params) == params


def test_create_board_task_carries_it():
    from modules.tools.discovery import handlers_board_tasks

    async def handler(db, workspace_id, params):
        return {}

    probe = auto_approves_only_what_it_runs(handler).__code__
    codes, fn = [], handlers_board_tasks.create_board_task
    while fn is not None:
        codes.append(fn.__code__)
        fn = getattr(fn, "__wrapped__", None)
    assert probe in codes
