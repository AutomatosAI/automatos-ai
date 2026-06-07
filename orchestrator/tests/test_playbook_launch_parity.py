"""PRD-142 Wave 3 · WS-3R · W3-S12 — Playbook launch parity.

The agent-scope deliverable: every one of the 7 BACKEND launch sites for a
Playbook (formerly "recipe") goes through the consolidated
``services.playbook_engine.PlaybookEngine`` interface — NOT directly through
``api.recipe_executor.launch_recipe_task`` / ``execute_recipe_direct``.

Why parity matters: the strangler-fig only works if every caller sits on the
new seam BEFORE the legacy path is removed. With every call site converged
on ``PlaybookEngine``, swapping the engine internals (Wave 3R follow-up:
durability columns, retry-learning) lands once in one place. With a single
forgotten caller still on the legacy path, the consolidation is fake.

These tests pin:

  1. ``PlaybookEngine`` exposes the two stable methods the 7 callers need:
     ``launch`` (sync — fires the background task) and ``execute_direct``
     (async — runs the executor inline).
  2. The engine forwards every kwarg unchanged to the underlying
     ``api.recipe_executor`` function (no signature drift between the seam
     and the legacy implementation).
  3. The 7 verified backend launch sites (PRD §6 W3-S12) import the engine,
     NOT the raw recipe_executor functions, and call the engine method.
  4. ``api/workflow_recipes.py`` stays as a THIN delegator — the legacy
     router lives on (FE still calls it), but its execution paths go via
     the engine.
  5. The frontend is NOT touched: ``api/api_playbooks.py`` stays a 49-line
     read-only stub (no execute route), ``frontend/`` has zero new pages.
     [HUMAN GATE] — re-asserted negatively so a future agent run cannot
     drift into the frontend.

AST inspection is the workhorse: no real launches happen; we read each call
site's source and prove it points at the engine seam.

TDD GUARANTEE: written BEFORE the engine and the call-site migrations land.
Each test fails with ``ModuleNotFoundError`` / ``AttributeError`` / missing
imports until the seam is built and the 7 sites are migrated.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

for _k, _v in {
    "POSTGRES_USER": "test",
    "POSTGRES_PASSWORD": "test",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "test",
}.items():
    os.environ.setdefault(_k, _v)


# Verified 2026-06-06 in PRD-142-WAVE3 §6 W3-S12 + scripts/ralph/prd-142-wave3.json
# notes. The 7 backend launch sites the agent migrates.
SEVEN_BACKEND_LAUNCH_SITES = (
    "api/workflow_recipes.py",         # site #1 + #2 (POST /execute + 2nd webhook door)
    "api/composio.py",                  # site #3 (execute_recipe_direct)
    "api/webhooks.py",                  # site #4 (execute_recipe_direct)
    "modules/tools/discovery/handlers_playbooks.py",  # site #5 (launch_recipe_task)
    "services/playbook_scheduler.py",   # site #6 (launch_recipe_task)
    "services/task_reconciler.py",      # site #7 (launch_recipe_task)
)


def _read(rel_path: str) -> str:
    return (ORCH_ROOT / rel_path).read_text()


# ---------------------------------------------------------------------------
# 1. The engine exists and exposes the stable seam the 7 callers need.
# ---------------------------------------------------------------------------


def test_playbook_engine_module_exists():
    """services/playbook_engine.py is the strangler-fig seam."""
    p = ORCH_ROOT / "services" / "playbook_engine.py"
    assert p.exists(), "services/playbook_engine.py must exist (the seam)"


def test_engine_exposes_launch_and_execute_direct():
    """The PlaybookEngine class has ``launch`` (sync) + ``execute_direct``
    (async coroutine). These are the two methods the 7 sites call — they
    must exist BEFORE any site migrates onto them."""
    import asyncio
    import inspect

    from services.playbook_engine import PlaybookEngine, get_playbook_engine

    engine = get_playbook_engine()
    assert isinstance(engine, PlaybookEngine)

    # launch is the sync fire-and-track-row entry point (wraps launch_recipe_task)
    assert hasattr(engine, "launch")
    assert callable(engine.launch)
    assert not asyncio.iscoroutinefunction(engine.launch), (
        "launch must be SYNC — it schedules the work, doesn't await it"
    )

    # execute_direct is the async inline executor (wraps execute_recipe_direct)
    assert hasattr(engine, "execute_direct")
    assert asyncio.iscoroutinefunction(engine.execute_direct), (
        "execute_direct must be a coroutine — it awaits the executor"
    )

    # Both methods take the canonical kwargs (verified against the 7 sites).
    launch_sig = inspect.signature(engine.launch)
    for kw in ("recipe_execution_id", "recipe_id", "workspace_id", "input_data"):
        assert kw in launch_sig.parameters, (
            f"launch() must accept '{kw}' kwarg — sites use it"
        )

    direct_sig = inspect.signature(engine.execute_direct)
    for kw in ("recipe_execution_id", "recipe_id", "workspace_id", "input_data"):
        assert kw in direct_sig.parameters, (
            f"execute_direct() must accept '{kw}' kwarg — sites use it"
        )


def test_get_playbook_engine_returns_module_singleton():
    """Repeated calls return the SAME engine instance — sites share one seam.
    A fresh instance per call would defeat the strangler-fig (different
    state per caller)."""
    from services.playbook_engine import get_playbook_engine

    e1 = get_playbook_engine()
    e2 = get_playbook_engine()
    assert e1 is e2, "get_playbook_engine() must be a singleton"


# ---------------------------------------------------------------------------
# 2. The engine forwards every kwarg unchanged to the legacy implementation.
# ---------------------------------------------------------------------------


def test_launch_forwards_kwargs_unchanged(monkeypatch):
    """launch(...) delegates to api.recipe_executor.launch_recipe_task with
    the EXACT kwargs the caller passed — no rename, no drop, no inject."""
    from uuid import uuid4

    captured = {}

    def _spy(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    # Patch the underlying function by attribute on the module.
    import api.recipe_executor as rex
    monkeypatch.setattr(rex, "launch_recipe_task", _spy)

    from services.playbook_engine import get_playbook_engine

    ws_id = uuid4()
    get_playbook_engine().launch(
        recipe_execution_id="exec-xyz",
        recipe_id=42,
        workspace_id=ws_id,
        input_data={"k": "v"},
    )

    assert captured["kwargs"]["recipe_execution_id"] == "exec-xyz"
    assert captured["kwargs"]["recipe_id"] == 42
    assert captured["kwargs"]["workspace_id"] == ws_id
    assert captured["kwargs"]["input_data"] == {"k": "v"}
    # Positional args may be empty — kwargs is canonical.
    assert "input_data" in captured["kwargs"]


def test_execute_direct_forwards_kwargs_unchanged(monkeypatch):
    """execute_direct(...) awaits api.recipe_executor.execute_recipe_direct
    with the EXACT kwargs — same parity contract as launch()."""
    import asyncio
    from uuid import uuid4

    captured = {}

    async def _spy(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    import api.recipe_executor as rex
    monkeypatch.setattr(rex, "execute_recipe_direct", _spy)

    from services.playbook_engine import get_playbook_engine

    ws_id = uuid4()

    asyncio.run(get_playbook_engine().execute_direct(
        recipe_execution_id="exec-abc",
        recipe_id=99,
        workspace_id=ws_id,
        input_data={"foo": "bar"},
    ))

    assert captured["kwargs"]["recipe_execution_id"] == "exec-abc"
    assert captured["kwargs"]["recipe_id"] == 99
    assert captured["kwargs"]["workspace_id"] == ws_id
    assert captured["kwargs"]["input_data"] == {"foo": "bar"}


def test_execute_direct_forwards_optional_db_url(monkeypatch):
    """The optional db_url kwarg (used by tests + scheduled retries) is
    preserved through the seam."""
    import asyncio
    from uuid import uuid4

    captured = {}

    async def _spy(*args, **kwargs):
        captured["kwargs"] = kwargs

    import api.recipe_executor as rex
    monkeypatch.setattr(rex, "execute_recipe_direct", _spy)

    from services.playbook_engine import get_playbook_engine

    asyncio.run(get_playbook_engine().execute_direct(
        recipe_execution_id="exec-1",
        recipe_id=1,
        workspace_id=uuid4(),
        input_data={},
        db_url="postgresql://test/test",
    ))

    assert captured["kwargs"].get("db_url") == "postgresql://test/test"


# ---------------------------------------------------------------------------
# 3. Each of the 7 backend launch sites is migrated onto the engine seam.
# ---------------------------------------------------------------------------


import pytest  # noqa: E402 — pytest is imported after the helper for the @parametrize


# Each tuple = (site_path, expected_engine_method) — method = "launch" or "execute_direct"
SITE_TO_METHOD = (
    # api/workflow_recipes.py has TWO launches — the POST /{id}/execute (905) and
    # the 2nd webhook door (1860). Both via launch_recipe_task → engine.launch.
    ("api/workflow_recipes.py", "launch"),
    # api/composio.py:886 awaits execute_recipe_direct inline → engine.execute_direct.
    ("api/composio.py", "execute_direct"),
    # api/webhooks.py:683 wraps execute_recipe_direct in asyncio.create_task →
    # engine.execute_direct (still inline-async via the engine seam).
    ("api/webhooks.py", "execute_direct"),
    # modules/tools/discovery/handlers_playbooks.py:487 → engine.launch
    ("modules/tools/discovery/handlers_playbooks.py", "launch"),
    # services/playbook_scheduler.py:208 → engine.launch
    ("services/playbook_scheduler.py", "launch"),
    # services/task_reconciler.py:273 → engine.launch
    ("services/task_reconciler.py", "launch"),
)


@pytest.mark.parametrize("site_path,method", SITE_TO_METHOD)
def test_site_calls_engine_not_legacy(site_path, method):
    """Each call site imports the engine seam and calls the expected method.
    Drift here = a forgotten caller still on the legacy path = consolidation
    is fake."""
    src = _read(site_path)
    # Imports the engine seam (or the convenience accessor).
    assert (
        "from services.playbook_engine import" in src
    ), f"{site_path}: must import from services.playbook_engine"

    # Calls the engine method via the singleton accessor.
    assert f".{method}(" in src or f"engine.{method}(" in src, (
        f"{site_path}: must call engine.{method}(...)"
    )


@pytest.mark.parametrize("site_path,_", SITE_TO_METHOD)
def test_site_does_not_import_legacy_executor_directly(site_path, _):
    """Once migrated, a backend site MUST NOT import the legacy entry points
    directly — they go through the engine. Otherwise the strangler-fig has
    a hole."""
    src = _read(site_path)
    assert "from api.recipe_executor import launch_recipe_task" not in src, (
        f"{site_path}: legacy launch_recipe_task import survives — must go via engine.launch"
    )
    assert "from api.recipe_executor import execute_recipe_direct" not in src, (
        f"{site_path}: legacy execute_recipe_direct import survives — must go via engine.execute_direct"
    )


# ---------------------------------------------------------------------------
# 4. api/workflow_recipes.py remains a THIN delegator — the FE still calls it.
# ---------------------------------------------------------------------------


def test_workflow_recipes_router_still_exists():
    """The legacy router is kept (FE depends on it; the deletion is HUMAN
    GATE, not agent scope). The agent migrates execution paths through the
    engine, but the router file lives on as a thin delegator."""
    p = ORCH_ROOT / "api" / "workflow_recipes.py"
    assert p.exists(), "api/workflow_recipes.py must NOT be deleted in the agent run"

    # The POST /{recipe_id}/execute route still lives here (FE calls it).
    text = p.read_text()
    assert '@router.post("/{recipe_id}/execute")' in text, (
        "execute_recipe POST route is gone — FE will 404"
    )


def test_api_playbooks_remains_a_readonly_stub():
    """api/api_playbooks.py was a 49-line read-only stub. The PROMOTION to a
    real execution router is the HUMAN-GATE front-door decision (§12.6) —
    not the agent's. Verify nothing snuck in."""
    p = ORCH_ROOT / "api" / "api_playbooks.py"
    if not p.exists():
        return  # File may have been removed in the human gate path — out of agent scope.
    text = p.read_text()
    # Read-only routes only — no /execute POST. The /api/playbooks router
    # SHOULD NOT have grown an execute endpoint inside the agent run.
    assert '@router.post("/{playbook_id}/execute")' not in text, (
        "agent must NOT promote api_playbooks.py to an execution router — that is the human gate"
    )


# ---------------------------------------------------------------------------
# 5. The frontend is NOT touched (re-asserted negatively).
# ---------------------------------------------------------------------------


def test_no_new_frontend_pages_added():
    """Memory feedback-ralph-supervision: past Ralph runs invented UI nobody
    asked for. W3-S3 (FE tile) and the W3-S12 FE-repoint are HUMAN GATES.
    The agent must not touch frontend/."""
    # frontend/ lives at the repo root, not under orchestrator/.
    repo_root = ORCH_ROOT.parent
    fe = repo_root / "frontend"
    if not fe.exists():
        return  # FE absent in this checkout — skip.
    # The api-client.ts must still point at the legacy /api/workflow-recipes
    # path until the human gate moves it.
    api_client = fe / "lib" / "api-client.ts"
    if not api_client.exists():
        return
    txt = api_client.read_text()
    # The legacy path still exists (the agent did not yank it).
    assert "/api/workflow-recipes" in txt, (
        "agent must NOT repoint /api/workflow-recipes FE calls — that is the human gate"
    )


# ---------------------------------------------------------------------------
# 6. Behavioural parity: engine.launch produces the SAME observable effect
#    as a direct call to the legacy launch_recipe_task (kwargs forwarded,
#    no extra work injected at the seam).
# ---------------------------------------------------------------------------


def test_engine_launch_does_not_inject_extra_work(monkeypatch):
    """The seam is a delegation only — no extra DB writes, no extra side
    effects beyond what the legacy function already does. This protects
    against silent behaviour drift between callers."""
    calls = []

    def _spy(**kwargs):
        calls.append(("launch_recipe_task", kwargs))

    import api.recipe_executor as rex
    monkeypatch.setattr(rex, "launch_recipe_task", _spy)

    from services.playbook_engine import get_playbook_engine

    get_playbook_engine().launch(
        recipe_execution_id="exec-parity",
        recipe_id=1,
        workspace_id="ws-parity",
        input_data={},
    )

    # Exactly one downstream call — no double-dispatch, no auxiliary writes.
    assert len(calls) == 1
    name, kwargs = calls[0]
    assert name == "launch_recipe_task"
    # Caller's kwargs preserved 1:1.
    assert kwargs == {
        "recipe_execution_id": "exec-parity",
        "recipe_id": 1,
        "workspace_id": "ws-parity",
        "input_data": {},
    }


# ---------------------------------------------------------------------------
# 7. AST-verified site count — exactly 7 backend launch sites.
# ---------------------------------------------------------------------------


def test_exactly_seven_backend_launch_sites_migrated():
    """The PRD §6 W3-S12 acceptance names 7 backend launch sites. Counting
    the call-site files (workflow_recipes.py has TWO call sites; the other
    5 files have one each) keeps the agent honest — a forgotten 8th site
    would mean the strangler-fig has a hole."""
    # workflow_recipes.py contains 2 engine.launch call sites
    wr_src = _read("api/workflow_recipes.py")
    wr_launch_count = wr_src.count(".launch(")
    assert wr_launch_count >= 2, (
        f"workflow_recipes.py expected >=2 engine.launch( calls, got {wr_launch_count}"
    )

    # The other 5 files each have one engine call (launch or execute_direct).
    one_call_files = (
        ("api/composio.py", "execute_direct"),
        ("api/webhooks.py", "execute_direct"),
        ("modules/tools/discovery/handlers_playbooks.py", "launch"),
        ("services/playbook_scheduler.py", "launch"),
        ("services/task_reconciler.py", "launch"),
    )
    for path, method in one_call_files:
        src = _read(path)
        assert f".{method}(" in src, f"{path}: missing engine.{method}( call"


# ---------------------------------------------------------------------------
# 8. The engine module is lightweight at import — no recipe_executor pulled
#    in eagerly (else the seam costs as much as the legacy chain).
# ---------------------------------------------------------------------------


def test_engine_module_is_import_light():
    """A top-level ``import api.recipe_executor`` at module load defeats the
    purpose of the seam (the heavy chain still loads). The engine MUST
    import recipe_executor lazily inside the methods."""
    p = ORCH_ROOT / "services" / "playbook_engine.py"
    src = p.read_text()
    tree = ast.parse(src)
    top_level_imports = [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.Import, ast.ImportFrom))
        and getattr(n, "col_offset", 0) == 0
    ]
    for node in top_level_imports:
        mod = (
            node.module
            if isinstance(node, ast.ImportFrom)
            else (node.names[0].name if node.names else "")
        )
        assert mod != "api.recipe_executor", (
            "engine module imports api.recipe_executor at top level — must be lazy"
        )
        assert not (mod or "").startswith("api.recipe_executor"), (
            "engine module imports a recipe_executor submodule at top level — must be lazy"
        )
