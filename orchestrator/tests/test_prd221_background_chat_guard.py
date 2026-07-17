"""PRD-221 S11 — background→chat delivery regression guard (test-only).

PRD-205 Auto Speaks (#560/#561) is canonical for background→chat: scheduled
and watcher output is delivered through services.chat_messenger to the per-user
Auto thread, fixing the historic PRD-77 log-and-drop. This PRD does NOT rebuild
any of that — it only guards it.

Behavioural coverage already lives in #560's suite:
    test_prd205_auto_speaks.py::test_scheduled_output_delivered_to_origin_chat
    test_prd205_auto_speaks.py::test_scheduled_output_without_origin_stays_log_only
    test_prd205_auto_speaks.py::test_scheduled_empty_output_posts_nothing

What we add here is a structural tripwire that fails loudly if a future change
reverts the scheduled path to log-only, or grows a rival chat-creation path
inside scheduled_task_service (chat_messenger owns chat/message creation).
"""
from __future__ import annotations

# CI collection-order guard (see PR #434).
import sys as _sys_guard  # noqa: E402
for _name in [n for n, m in list(_sys_guard.modules.items())
              if (n == "modules" or n.startswith("modules.")
                  or n == "consumers" or n.startswith("consumers.")
                  or n == "services" or n.startswith("services."))
              and getattr(m, "__spec__", None) is None]:
    _sys_guard.modules.pop(_name, None)

import inspect  # noqa: E402


def test_scheduled_path_delivers_via_chat_messenger_seam():
    from services import scheduled_task_service as sts
    src = inspect.getsource(sts)
    # Routes output through the canonical seam — not a bare log line (PRD-77).
    assert "deliver_background_message" in src, (
        "scheduled_task_service must deliver output via the ChatMessenger seam; "
        "reverting to log-only reintroduces the PRD-77 discard bug"
    )


def test_scheduled_service_owns_no_rival_chat_creation():
    from services import scheduled_task_service as sts
    src = inspect.getsource(sts).lower()
    # chat_messenger owns chat/message creation; the scheduler must not grow a
    # second path that inserts chats/messages directly.
    assert "insert into chats" not in src
    assert "insert into messages" not in src


def test_chat_messenger_seam_is_importable():
    from services.chat_messenger import deliver_background_message
    assert callable(deliver_background_message)
