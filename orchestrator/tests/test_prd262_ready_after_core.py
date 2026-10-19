"""PRD-262 US-034: the API answers once its core is up; the rest boots in the background.

uvicorn 0.24 awaits the lifespan's start-up before it listens, so ``/health`` and
``/health/ready`` used to first answer at the same moment, after the whole boot. Now:

* the lifespan yields after phase-1 core and the trust gate (``_boot_core``), so uvicorn
  listens and ``/health`` answers;
* the starters pass, the Composio catalogue, Phase 2 (scheduler, channels) and board
  dispatch run as one background task (``_boot_in_background``), which records their
  stages in the bootstrap report and sets ``app.state.ready`` once all of them finished,
  failed or not; ``/health/ready`` answers 503 until then;
* board dispatch keeps its place: after Phase 2, before READY;
* only the boot leader (the worker whose core phase returned seeds) embeds the agents;
* a shutdown during the background boot cancels it.

The starters pass itself (``main._run_starters``) is held in test_prd262_seed_stages.py.

These tests drive ``main.lifespan`` on a bare FastAPI app with every boot step stubbed,
so nothing touches a database, and read ``/health/ready`` through its handler.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI

from core.models.bootstrap import BootstrapReport, BootstrapStage, DeferredInitResult, run_stage
from core.seeds.leader_stage import LeaderSeeds

CORE_STAGES = ["database_init", "semantic_embeddings", "trust_gate"]
BACKGROUND_STAGES = ["seed_system_data", "tool_sync", "scheduler_init", "ready"]
SEEDS = LeaderSeeds(fingerprint="f" * 64, ran=True, clean=True)
WAIT_S = 5


class _Boot:
    """main's boot steps, stubbed: each records its call, and the Composio stage waits
    on ``gate`` so a test can hold the background boot where it likes."""

    def __init__(self, monkeypatch, *, seeds=SEEDS, trust=True, catalog_error=None, dispatch=False):
        import core.loop_watchdog as loop_watchdog
        import main

        self.calls: list = []
        self.gate: asyncio.Event | None = None
        self.app = FastAPI()
        self._seeds, self._trust, self._catalog_error = seeds, trust, catalog_error
        for name in ("HEARTBEAT_ENABLED", "RECIPE_SCHEDULER_ENABLED", "COORDINATOR_ENABLED", "CHANNELS_ENABLED"):
            monkeypatch.setattr(main.config, name, False)
        monkeypatch.setattr(main.config, "BOARD_DISPATCH_ENABLED", dispatch)
        monkeypatch.setattr(main.config, "validate_security", lambda: None)
        monkeypatch.setattr("modules.policy.policy_plane_enabled", lambda: False)
        monkeypatch.setattr(main, "_boot_phase_1_core", self._core)
        monkeypatch.setattr(main, "_seed_semantic_embeddings", self._semantic)
        monkeypatch.setattr(main, "_trust_gate", self._trust_gate)
        monkeypatch.setattr(main, "_run_starters", self._starters)
        monkeypatch.setattr(main, "_bootstrap_composio_catalog", self._catalog)
        monkeypatch.setattr(main, "_boot_phase_2_extensions", self._phase_2)
        monkeypatch.setattr(main, "shutdown_dashboard", self._shutdown_dashboard)
        monkeypatch.setattr(loop_watchdog, "start_loop_watchdog",
                            lambda: SimpleNamespace(stop=lambda: self.calls.append("watchdog stopped")))
        if dispatch:
            monkeypatch.setattr("services.board_dispatcher.run_dispatch_loop", self._dispatch)
        self.main = main

    async def _core(self):
        self.calls.append("core")
        return self._seeds

    async def _semantic(self, leader):
        self.calls.append(("semantic", leader))

    def _trust_gate(self):
        self.calls.append("trust gate")
        if not self._trust:
            raise self.main.TrustGateError("database unreachable")

    async def _starters(self, seeds):
        self.calls.append(("starters", seeds))

    async def _catalog(self):
        self.calls.append("catalog")
        await self.gate.wait()
        if self._catalog_error is not None:
            raise self._catalog_error

    async def _phase_2(self, app):
        self.calls.append("phase 2")
        return DeferredInitResult(scheduler_started=True)

    def _dispatch(self, stop_event):
        """Records the boot's state when the lifespan starts the dispatch loop."""
        self.calls.append(("dispatch", self.app.state.ready, tuple(self.stages())))
        return self._dispatch_loop(stop_event)

    @staticmethod
    async def _dispatch_loop(stop_event):
        try:
            await stop_event.wait()
        except asyncio.CancelledError:
            return  # the shutdown cancels it; this stand-in ends quietly

    async def _shutdown_dashboard(self, app):
        self.calls.append("dashboard shut down")

    async def ready_probe(self):
        """``/health/ready`` through its handler: the status code and body."""
        answer = await self.main.readiness_probe(SimpleNamespace(app=self.app))
        return (answer.status_code, answer.body) if hasattr(answer, "status_code") else (200, answer)

    def stages(self):
        return [stage.stage.value for stage in self.app.state.bootstrap_report.stages]


def _settle():
    """Let the background boot run up to the gate."""
    return asyncio.sleep(0.05)


def test_the_lifespan_yields_after_the_core_and_ready_waits_for_the_background_boot(monkeypatch):
    boot = _Boot(monkeypatch)

    async def scenario():
        boot.gate = asyncio.Event()
        async with boot.main.lifespan(boot.app):
            seen = {"stages_at_yield": boot.stages(), "ready_at_yield": await boot.ready_probe()}
            await _settle()
            seen["held"] = (boot.stages(), boot.app.state.ready, (await boot.ready_probe())[0])
            boot.gate.set()
            await asyncio.wait_for(asyncio.shield(boot.app.state.background_boot), WAIT_S)
            seen["after"] = (boot.stages(), await boot.ready_probe(), boot.app.state.bootstrap_report)
        return seen

    seen = asyncio.run(scenario())

    assert seen["stages_at_yield"] == CORE_STAGES  # the server listens with the core done, nothing more
    assert seen["ready_at_yield"][0] == 503
    assert seen["held"] == (CORE_STAGES + ["seed_system_data"], False, 503)  # stages show as they finish
    stages, ready, report = seen["after"]
    assert stages == CORE_STAGES + BACKGROUND_STAGES
    assert ready == (200, {"status": "ready"})
    assert report.ready_at is not None and report.failed_stages == []
    assert boot.calls[:3] == ["core", ("semantic", True), "trust gate"]
    assert boot.calls[3:6] == [("starters", SEEDS), "catalog", "phase 2"]
    assert boot.calls[-2:] == ["watchdog stopped", "dashboard shut down"]


def test_a_failed_background_stage_still_ends_in_ready_and_shows_in_the_report(monkeypatch):
    boot = _Boot(monkeypatch, catalog_error=RuntimeError("composio down"))

    async def scenario():
        boot.gate = asyncio.Event()
        boot.gate.set()
        async with boot.main.lifespan(boot.app):
            await asyncio.wait_for(asyncio.shield(boot.app.state.background_boot), WAIT_S)
            return await boot.ready_probe(), boot.app.state.bootstrap_report.as_dict()

    ready, report = asyncio.run(scenario())

    assert ready == (200, {"status": "ready"})
    assert report["failed_stages"] == ["tool_sync"]
    assert [stage["stage"] for stage in report["stages"]] == CORE_STAGES + BACKGROUND_STAGES
    assert "composio down" in next(s["error"] for s in report["stages"] if s["stage"] == "tool_sync")


def test_a_failed_trust_gate_skips_phase_2_and_still_ends_in_ready(monkeypatch):
    boot = _Boot(monkeypatch, trust=False, dispatch=True)

    async def scenario():
        boot.gate = asyncio.Event()
        boot.gate.set()
        async with boot.main.lifespan(boot.app):
            await asyncio.wait_for(asyncio.shield(boot.app.state.background_boot), WAIT_S)
            return boot.app.state.bootstrap_report.as_dict(), boot.app.state.trust_passed

    report, trust_passed = asyncio.run(scenario())

    assert trust_passed is False
    statuses = {stage["stage"]: stage["status"] for stage in report["stages"]}
    assert statuses["trust_gate"] == "failed" and statuses["scheduler_init"] == "skipped"
    assert statuses["ready"] == "success"
    assert "phase 2" not in boot.calls and not any(isinstance(c, tuple) and c[0] == "dispatch" for c in boot.calls)


def test_board_dispatch_starts_after_phase_2_and_before_ready(monkeypatch):
    """It keeps the place it had in the lifespan: after Phase 2, before READY."""
    boot = _Boot(monkeypatch, dispatch=True)

    async def scenario():
        boot.gate = asyncio.Event()
        boot.gate.set()
        async with boot.main.lifespan(boot.app):
            await asyncio.wait_for(asyncio.shield(boot.app.state.background_boot), WAIT_S)
            await _settle()  # the dispatch loop's first step
            return boot.app.state.board_dispatch_task

    task = asyncio.run(scenario())

    dispatch = next(call for call in boot.calls if isinstance(call, tuple) and call[0] == "dispatch")
    assert boot.calls.index("phase 2") < boot.calls.index(dispatch)
    assert dispatch[1:] == (False, tuple(CORE_STAGES + ["seed_system_data", "tool_sync", "scheduler_init"]))
    assert task.done()  # the shutdown stopped it


def test_a_worker_that_did_not_lead_neither_embeds_the_agents_nor_runs_the_starters(monkeypatch):
    boot = _Boot(monkeypatch, seeds=None)

    async def scenario():
        boot.gate = asyncio.Event()
        boot.gate.set()
        async with boot.main.lifespan(boot.app):
            await asyncio.wait_for(asyncio.shield(boot.app.state.background_boot), WAIT_S)

    asyncio.run(scenario())

    assert ("semantic", False) in boot.calls
    assert ("starters", None) in boot.calls  # _run_starters returns at once for None


def test_a_shutdown_during_the_background_boot_cancels_it(monkeypatch):
    boot = _Boot(monkeypatch)

    async def scenario():
        boot.gate = asyncio.Event()  # never set: the boot is held at the catalogue
        async with boot.main.lifespan(boot.app):
            await _settle()
            task = boot.app.state.background_boot
        return task, boot.app.state.ready

    task, ready = asyncio.run(scenario())

    assert task.cancelled() and ready is False
    assert "phase 2" not in boot.calls
    assert boot.calls[-2:] == ["watchdog stopped", "dashboard shut down"]


# --------------------------------------------------------------------------- the pieces


@pytest.mark.parametrize("leader,embedded", [(True, True), (False, False)])
def test_only_the_boot_leader_embeds_the_agents_and_every_worker_warms_its_index(monkeypatch, leader, embedded):
    import core.boot.startup_tasks as startup_tasks
    import main

    started = []

    def recorder(name):
        async def run():
            started.append(name)
        return run

    for name in ("embed_all_agents_on_startup", "ensure_field_memory_collection",
                 "warm_action_index_on_startup", "warm_output_budgets_on_startup"):
        monkeypatch.setattr(startup_tasks, name, recorder(name))

    async def scenario():
        await main._seed_semantic_embeddings(leader=leader)
        await asyncio.sleep(0)  # the launched tasks run

    asyncio.run(scenario())

    assert ("embed_all_agents_on_startup" in started) is embedded
    assert {"warm_output_budgets_on_startup", "ensure_field_memory_collection",
            "warm_action_index_on_startup"} <= set(started)


def test_the_composio_catalogue_stage_runs_off_the_event_loop(monkeypatch):
    import threading

    import core.composio.bootstrap as composio_bootstrap
    import main

    threads = []
    monkeypatch.setattr(composio_bootstrap, "ensure_catalog_on_boot", lambda: threads.append(threading.get_ident()))

    async def scenario():
        await main._bootstrap_composio_catalog()
        return threading.get_ident()

    loop_thread = asyncio.run(scenario())

    assert threads and threads[0] != loop_thread


def test_a_stage_result_carries_what_its_callable_returned_and_the_report_leaves_it_out():
    async def returns():
        return SEEDS

    def fails():
        raise RuntimeError("boom")

    async def scenario():
        report = BootstrapReport()
        ran = await run_stage(report, BootstrapStage.DATABASE_INIT, returns)
        failed = await run_stage(report, BootstrapStage.TOOL_SYNC, fails)
        skipped = await run_stage(report, BootstrapStage.SCHEDULER_INIT, returns, skip_condition=True)
        return report, ran, failed, skipped

    report, ran, failed, skipped = asyncio.run(scenario())

    assert (ran.value, failed.value, skipped.value) == (SEEDS, None, None)
    assert all("value" not in stage for stage in report.as_dict()["stages"])


def test_the_readiness_comments_say_what_the_boot_now_does():
    """main.py's readiness probe comment, the smoke's READY_URL comment and QUICKSTART say
    that /health answers once the API listens and /health/ready waits for the rest."""
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    main_py = (repo / "orchestrator" / "main.py").read_text(encoding="utf-8")
    probe = main_py[:main_py.index('@app.get("/health/ready"')].rsplit("\n\n", 1)[-1]
    smoke = (repo / "scripts" / "ci" / "smoke-fresh-clone.sh").read_text(encoding="utf-8")
    ready_url = smoke[:smoke.index('READY_URL="${API_BASE}/health/ready"')].rsplit("\n\n", 1)[-1]
    quickstart = (repo / "QUICKSTART.md").read_text(encoding="utf-8")
    readiness = next(paragraph for paragraph in quickstart.split("\n\n") if "8000/health/ready" in paragraph)

    for text in (probe, ready_url, readiness):
        assert "background" in text and "503" in text, text
        assert "same moment" not in text and "full boot has" not in text, text
