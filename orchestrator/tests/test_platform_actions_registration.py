"""Regression guard: register_all_actions() wires every Wave-4 platform tool.

Two PRs touched platform_actions.py's registration block — #428 (power_mode S5,
routing S6) and #429 (autonomy S14/S15). A merge resolved the conflict by keeping
the autonomy registration and silently dropping power + routing, while their
handlers stayed wired in platform_executor — so the actions vanished from the
tool registry (and the LLM tool schema) with no test failure.

This pins that the single entry point actually registers all of them, so a
future conflict can't drop one unnoticed. Dummy POSTGRES_* + the apscheduler stub
let the action-module import chain load without a DB or the prod-only scheduler.
"""
import os
import sys
import types

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_DB", "test")


def _install_fake_apscheduler():
    if "apscheduler" in sys.modules:
        return
    aps = types.ModuleType("apscheduler")
    schedulers = types.ModuleType("apscheduler.schedulers")
    asyncio_mod = types.ModuleType("apscheduler.schedulers.asyncio")
    asyncio_mod.AsyncIOScheduler = type("AsyncIOScheduler", (), {})
    jobstores = types.ModuleType("apscheduler.jobstores")
    memory_mod = types.ModuleType("apscheduler.jobstores.memory")
    memory_mod.MemoryJobStore = type("MemoryJobStore", (), {})
    aps.schedulers = schedulers
    aps.jobstores = jobstores
    schedulers.asyncio = asyncio_mod
    jobstores.memory = memory_mod
    sys.modules.update({
        "apscheduler": aps,
        "apscheduler.schedulers": schedulers,
        "apscheduler.schedulers.asyncio": asyncio_mod,
        "apscheduler.jobstores": jobstores,
        "apscheduler.jobstores.memory": memory_mod,
    })


_install_fake_apscheduler()

from modules.tools.discovery.action_registry import ActionRegistry  # noqa: E402
from modules.tools.discovery.platform_actions import register_all_actions  # noqa: E402


def test_register_all_actions_wires_wave4_tools():
    reg = ActionRegistry()
    register_all_actions(reg)
    # _actions is the registry's name->definition map; read it directly rather
    # than get() (which would trigger a second full init).
    for name in (
        "platform_set_power_mode",        # W4-S5
        "platform_create_routing_rule",   # W4-S6
        "platform_set_autonomy_level",    # W4-S14/S15
        "platform_get_autonomy_level",    # W4-S14/S15
    ):
        assert reg._actions.get(name) is not None, (
            f"{name} is not registered by register_all_actions — a merge likely "
            "dropped its register_*_actions() call from platform_actions.py"
        )
