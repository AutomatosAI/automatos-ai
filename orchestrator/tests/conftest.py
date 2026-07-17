"""Shared fixtures for plugin system tests."""
import os
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest


def _module_is_real(mod) -> bool:
    """True iff ``mod`` is a genuinely-loaded module/package, not a test stub.

    Realness is decided by the filesystem, not by ``__spec__`` presence:
      * a real module/regular package has ``__file__`` pointing at a file that
        EXISTS on disk;
      * a real namespace package has a ``__spec__`` carrying
        ``submodule_search_locations``.
    Everything else — bare ``ModuleType`` stubs, ``module_from_spec`` stubs with
    ``origin=None``, and (critically) path-bearing stubs that set
    ``__path__=[realdir]`` but no ``__file__`` (these import as
    "(unknown location)") — is a stub. A ``__spec__``- or ``__path__``-only
    check misses the last shape, which is the PR #434 CI failure.
    """
    f = getattr(mod, "__file__", None)
    if f and os.path.exists(f):
        return True
    spec = getattr(mod, "__spec__", None)
    if spec is not None and getattr(spec, "submodule_search_locations", None):
        return True
    return False


# The real app chain that the heavy test_prd143_* modules import at COLLECTION
# time. Preloaded once while sys.modules is still clean (pytest_configure, below)
# and snapshotted, so those modules resolve from a real cache instead of
# cold-re-importing against a sys.modules already polluted by sibling stubs.
_PRELOAD_TARGETS = (
    "modules.tools.discovery.platform_executor",
    "modules.tools.discovery.action_registry",
    "modules.tools.discovery.platform_actions",
    "modules.tools.tool_router",
    "modules.tools.execution.telemetry",
    "consumers.chatbot.service",
    "core.auth.super_admin",
)
_REAL_APP_SNAPSHOT: dict = {}


def _restore_real_app_modules() -> None:
    """Put the preloaded real modules back over any stub a sibling left.

    Only entries that are currently MISSING or a stub are overwritten — a real
    module already in place (even a different instance) is left untouched, so
    this cannot disturb a test that legitimately holds the real module. No
    package is re-imported, so the cold-import collision the purge approach hit
    ("cannot import name 'tools' from 'modules'") cannot occur.
    """
    for _name, _real in _REAL_APP_SNAPSHOT.items():
        _cur = sys.modules.get(_name)
        if _cur is _real:
            continue
        if not _module_is_real(_cur):
            sys.modules[_name] = _real

# Ensure orchestrator package is importable
_orchestrator_root = str(Path(__file__).resolve().parent.parent)
if _orchestrator_root not in sys.path:
    sys.path.insert(0, _orchestrator_root)


@pytest.fixture(autouse=True)
def _repair_stubbed_package_bindings():
    """Repair parent->child module attribute bindings that sibling tests broke.

    Many unit tests inject ``types.ModuleType`` stubs straight into
    ``sys.modules`` (for ``consumers``, ``consumers.chatbot``, ``modules``,
    ``modules.tools`` ...) to dodge a heavy package __init__ that pulls optional
    deps (asyncpg / pdfplumber / tiktoken / camelot). A real ``import a.b`` binds
    ``b`` as an attribute on package ``a``; a manual ``sys.modules['a.b'] = stub``
    does NOT. Worse, replacing ``sys.modules['a']`` with a bare stub drops every
    real child attribute it used to carry.

    pytest's ``monkeypatch.setattr("a.b.c.attr", ...)`` resolves its target by
    walking attributes from the top package (``getattr(a, 'b')`` ...). When a
    sibling has left ``a`` as a stub missing ``b``, that walk raises
    ``"'module' object at a.b has no attribute 'b'"`` in whatever test is
    collected after the leak — e.g. test_w1s9 patching
    ``modules.tools.discovery.platform_executor.PlatformActionExecutor`` trips on
    ``modules.tools``; an earlier sibling tripped on ``consumers.chatbot``.

    This autouse fixture re-binds every dotted ``sys.modules`` entry onto its
    parent package when the parent is missing that attribute. It is
    *fill-missing only* — it never overwrites an existing binding, so it cannot
    change a target a healthy test already resolves — and is idempotent / a
    no-op once the real packages are loaded. A failed repair must never break
    collection, so setattr errors are swallowed.
    """
    for name, mod in list(sys.modules.items()):
        if mod is None or "." not in name:
            continue
        parent_name, _, child_attr = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None and not hasattr(parent, child_attr):
            try:
                setattr(parent, child_attr, mod)
            except Exception:
                pass
    yield


def pytest_collectstart(collector):
    """Collection-time stub purge for heavy-import test modules.

    The autouse ``_repair_stubbed_package_bindings`` fixture above runs at
    TEST-SETUP time and is fill-missing only — too late and too gentle for a
    module whose *top level* imports the real ``modules.*`` / ``consumers.*``
    app packages at COLLECTION time (e.g. ``test_prd143_boundary_sweep``
    imports ``modules.tools.discovery.platform_executor`` at import). By the
    time such a module is collected, earlier siblings have injected
    non-file-backed ``ModuleType`` stubs (e.g.
    ``test_platform_actions_section_graph`` stubs
    ``modules.tools.discovery.*`` at module level), so the real import resolves
    against a stub and dies at collection — pytest then aborts the WHOLE run
    ("Interrupted: 1 error during collection"). The stubs come in two shapes:
    bare (``__spec__ is None``) AND spec'd-but-pathless ("unknown location"),
    so a ``__spec__``-based guard misses half of them — gate on file-backing.
    Linux collection order makes the stubs live here; macOS order hides it,
    which is why this passed locally and failed only on CI (PR #434).

    Right before one of these modules collects, restore the preloaded real app
    modules over any sibling's stub (see ``_restore_real_app_modules``), so the
    module's collection-time imports resolve to the real, fully-initialised
    packages instead of a stub ("(unknown location)") or a doomed cold re-import
    ("cannot import name 'tools' from 'modules'").
    """
    name = getattr(collector, "name", "")
    if not name.startswith("test_prd143"):
        return
    _restore_real_app_modules()


def pytest_configure(config):
    """Preload the real app chain while sys.modules is still clean.

    Runs once, before any test module is collected (so before sibling tests
    inject their ``modules.*`` / ``consumers.*`` stubs). Importing the chain now
    populates sys.modules with the real, fully-initialised packages and lets us
    snapshot them; ``pytest_collectstart`` then restores that snapshot in front
    of the heavy ``test_prd143_*`` modules. Best-effort: a preload failure (e.g.
    an optional dep missing in a lean dev venv) just leaves the snapshot partial
    — never breaks the run.
    """
    import importlib
    for _tgt in _PRELOAD_TARGETS:
        try:
            importlib.import_module(_tgt)
        except Exception:
            pass
    for _name, _mod in list(sys.modules.items()):
        if _name.split(".")[0] in ("modules", "consumers") and _module_is_real(_mod):
            _REAL_APP_SNAPSHOT[_name] = _mod


# ---- Database mock ----

@pytest.fixture
def mock_db():
    """Mock SQLAlchemy session with chainable query API."""
    db = MagicMock()
    # Make .query().join().filter().order_by().all() chainable
    q = MagicMock()
    q.join.return_value = q
    q.filter.return_value = q
    q.order_by.return_value = q
    q.all.return_value = []
    q.first.return_value = None
    q.delete.return_value = 0
    db.query.return_value = q
    return db


# ---- Plugin model mocks ----

@pytest.fixture
def sample_plugin():
    """A MarketplacePlugin-like object with realistic data."""
    plugin = MagicMock()
    plugin.id = uuid4()
    plugin.slug = "email-automation"
    plugin.name = "Email Automation"
    plugin.version = "1.2.0"
    plugin.description = "Send and manage emails across providers"
    plugin.tags = ["email", "automation", "notification"]
    plugin.skills_count = 5
    plugin.commands_count = 3
    plugin.token_estimate = 1500
    plugin.s3_bucket = "automatos-marketplace"
    plugin.s3_path = "plugins/email-automation/1.2.0/"
    return plugin


@pytest.fixture
def sample_assignment(sample_plugin):
    """An AgentAssignedPlugin-like object."""
    aap = MagicMock()
    aap.agent_id = 42
    aap.plugin_id = sample_plugin.id
    aap.priority = 0
    aap.assigned_at = datetime(2025, 1, 15, 12, 0, 0)
    return aap


@pytest.fixture
def plugin_rows(sample_assignment, sample_plugin):
    """List of (assignment, plugin) tuples — single plugin."""
    return [(sample_assignment, sample_plugin)]


def _make_plugin(slug, name, description=None, tags=None, priority=0, **overrides):
    """Helper to create (assignment, plugin) tuples for multi-plugin tests."""
    plugin = MagicMock()
    plugin.id = uuid4()
    plugin.slug = slug
    plugin.name = name
    plugin.version = overrides.get("version", "1.0.0")
    plugin.description = description
    plugin.tags = tags
    plugin.skills_count = overrides.get("skills_count", 2)
    plugin.commands_count = overrides.get("commands_count", 1)
    plugin.token_estimate = overrides.get("token_estimate", 500)
    plugin.s3_bucket = "automatos-marketplace"
    plugin.s3_path = f"plugins/{slug}/1.0.0/"

    aap = MagicMock()
    aap.agent_id = overrides.get("agent_id", 42)
    aap.plugin_id = plugin.id
    aap.priority = priority
    aap.assigned_at = datetime(2025, 1, 15, 12, 0, 0)
    return aap, plugin


@pytest.fixture
def three_plugins():
    """Three plugins with different priorities and themes."""
    return [
        _make_plugin("db-manager", "Database Manager", "Query and manage databases", ["database", "sql"], priority=2),
        _make_plugin("email-sender", "Email Sender", "Send email notifications", ["email", "notification"], priority=0),
        _make_plugin("slack-bot", "Slack Bot", "Post messages to Slack channels", ["slack", "messaging"], priority=1),
    ]


# ---- Agent mock ----

@pytest.fixture
def mock_agent():
    """A realistic Agent-like mock."""
    agent = MagicMock()
    agent.id = 42
    agent.name = "Test Agent"
    agent.workspace_id = uuid4()
    agent.skills = []
    agent.assigned_plugins = []
    agent.model_config = {"model_id": "gpt-4", "temperature": 0.7}
    agent.description = "A test agent"
    agent.agent_type = "code_architect"
    agent.status = "active"
    agent.configuration = {}
    agent.priority_level = "medium"
    agent.max_concurrent_tasks = 5
    agent.auto_start = False
    agent.tags = ["test"]
    agent.created_at = datetime(2025, 1, 1)
    agent.updated_at = datetime(2025, 1, 2)
    agent.created_by = "user_123"
    agent.performance_metrics = {}
    agent.use_custom_persona = False
    agent.custom_persona_prompt = None
    agent.persona_id = None
    return agent


# ---- Playbook mocks ----

@pytest.fixture
def mock_playbook():
    """A WorkflowTemplate (Playbook)-like mock with cron schedule_config."""
    playbook = MagicMock()
    playbook.id = 42
    playbook.template_id = "test-cron-playbook"
    playbook.name = "Test Cron Playbook"
    playbook.workspace_id = uuid4()
    playbook.steps = [
        {"step_id": "s1", "order": 1, "agent_id": 101, "prompt_template": "Do the thing"},
    ]
    playbook.schedule_config = {
        "type": "cron",
        "cron_expression": "0 9 * * *",
    }
    playbook.owner_type = "workspace"
    playbook.is_system = False
    return playbook


@pytest.fixture
def mock_playbook_manual():
    """A WorkflowTemplate (Playbook)-like mock with manual schedule (no cron)."""
    playbook = MagicMock()
    playbook.id = 99
    playbook.template_id = "test-manual-playbook"
    playbook.name = "Test Manual Playbook"
    playbook.workspace_id = uuid4()
    playbook.steps = [
        {"step_id": "s1", "order": 1, "agent_id": 101, "prompt_template": "Manual task"},
    ]
    playbook.schedule_config = {
        "type": "manual",
    }
    playbook.owner_type = "workspace"
    playbook.is_system = False
    return playbook


# ---- Request context mock ----

@pytest.fixture
def mock_ctx(mock_agent):
    """A RequestContext-like mock matching the agent's workspace."""
    ctx = MagicMock()
    ctx.workspace_id = mock_agent.workspace_id
    ctx.user = MagicMock()
    ctx.user.id = "user_123"
    ctx.auth_type = "clerk"
    return ctx


# ---- Real-DB teardown guard: leaked sessions + bounded sweep locks ----
#
# Anatomy of the silent-hang class this kills (first hit: a 30-minute CI hang,
# fixed for one file in 2b86dfeda): a test that dies mid-transaction (e.g. an
# ImportError between flush and commit) abandons its session; pytest pins the
# failed frames -- and with them the session and its uncommitted row locks --
# via ``sys.last_traceback``. pytest-timeout cancels its per-item timer on any
# failure (pytest_exception_interact), so when a later fixture teardown's
# DELETE sweep blocks on the leaked lock, nothing kills the wait: the lane
# hangs silently to the job cap.
#
# ``new_session`` below is the shared cure: it REMEMBERS every session it
# hands out, and ``new_session.sweep()`` rolls them all back before returning
# a fresh session whose transaction carries a bounded lock_timeout -- so a
# future leak fails LOUDLY in seconds instead of hanging the lane.

TEARDOWN_LOCK_TIMEOUT = "5s"


def _release_sessions(sessions) -> None:
    """Roll back + close every recorded session. Never raises: a teardown
    must always reach its DELETE sweep."""
    for leaked in list(sessions):
        try:
            leaked.rollback()
            leaked.close()
        except Exception:  # noqa: BLE001
            pass


@pytest.fixture
def new_session(engine):
    """Tracking session factory shared by the real-DB test files.

    Each file keeps its own module-scoped ``engine`` fixture (schema probes
    and skip messages differ per suite); pytest resolves it per-module.

    * ``new_session()`` -- an independent, committing session
      (``expire_on_commit=False``), recorded so teardown can reclaim it.
    * ``new_session.sweep()`` -- for fixture teardowns that DELETE-sweep
      rows: rolls back + closes every session this factory issued (releasing
      locks a failed test left behind), then returns a fresh session whose
      transaction runs under ``SET LOCAL lock_timeout`` so any residual
      blocker raises instead of hanging. Keep the sweep to a single commit;
      ``SET LOCAL`` lasts only until the first COMMIT/ROLLBACK.

    SQLAlchemy is imported lazily so pure-stdlib collection stays possible
    (see the root conftest).
    """
    from sqlalchemy import text
    from sqlalchemy.orm import sessionmaker

    maker = sessionmaker(bind=engine, expire_on_commit=False)
    created = []

    def factory(**kwargs):
        s = maker(**kwargs)
        created.append(s)
        return s

    def sweep():
        _release_sessions(created)
        s = factory()
        s.execute(text(f"SET LOCAL lock_timeout = '{TEARDOWN_LOCK_TIMEOUT}'"))
        return s

    factory.created = created
    factory.sweep = sweep

    yield factory

    # Files with no sweep of their own must still not pin row locks for the
    # rest of the run -- a leak here would hang a LATER module's sweep.
    _release_sessions(created)
