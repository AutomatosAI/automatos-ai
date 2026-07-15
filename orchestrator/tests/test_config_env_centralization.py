"""
PRD-142 Wave 3 · W3-S5 — Centralize env reads through config.py (G7)

Asserts:
 1. config.py exposes typed accessors for every env var moved out of a
    runtime module (defaults match the call site's prior default — pure
    widening, no behaviour change).
 2. The swept modules read their setting from ``config`` (or a config
    helper) — NOT from ``os.getenv`` / ``os.environ`` / ``load_dotenv``.

The Wave 2 CI grep gate guards regressions globally; these tests guard
each swept module specifically and lock the new accessors' defaults.
"""

from __future__ import annotations

import ast
import importlib
import os
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Test plumbing — mirrors tests/test_config.py
# ---------------------------------------------------------------------------

_ORCHESTRATOR_ROOT = Path(__file__).resolve().parent.parent
if str(_ORCHESTRATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_ORCHESTRATOR_ROOT))


# Env vars touched by this sweep — wiped + restored around each reload so a
# value set elsewhere in CI (Railway has SERVICE_NAME=automatos-backend, etc.)
# does not bleed into a "default" assertion.
_SWEPT_ENV_VARS = (
    "LOG_RELAY_URL",
    "LOG_RELAY_ENABLED",
    "SERVICE_NAME",
    "ENVIRONMENT",
    "RAILWAY_ENVIRONMENT",
    "LOG_RELAY_BATCH_SIZE",
    "LOG_RELAY_FLUSH_INTERVAL",
    "RATE_LIMIT_GIT_CLONE_MAX",
    "RATE_LIMIT_GIT_CLONE_WINDOW_SECONDS",
    "LOKI_QUERY_URL",
    "ALERT_INGEST_TOKEN",
    "PUBLIC_API_HOST",
    # PRD-176 F068 — the nine railway.internal defaults + log-relay toggle that
    # W6 makes local-safe. Swept so a Railway env value can't mask the new
    # localhost/off defaults these tests assert.
    "LOKI_URL",
    "PROMETHEUS_URL",
    "INTERNAL_API_HOSTNAME",
    "INTERNAL_FRONTEND_HOSTNAME",
    "AGENT_OPT_WORKER_URL",
    "VOICE_SERVICE_URL",
)


@pytest.fixture(autouse=True)
def _restore_config_module():
    """Contain the reload blast radius (PRD-142 W2-S/WS-F).

    ``_reload_config`` mutates the shared ``config`` module (``importlib.reload``
    rebuilds its ``Config`` class + singleton). Downstream suites co-run after
    this file — which sorts first — must not inherit a test's env-derived config
    state (the leak that broke test_harness_commands / test_prd143_concierge).

    Restore the exact pre-test ``config`` module object into ``sys.modules`` so
    the class identity the rest of the session already bound (via
    ``from config import config`` / ``type(config.config)``) is the one that
    survives. Do NOT reload here: a reload would swap ``Config`` for a fresh
    class object, desyncing downstream fixtures that patch ``type(config.config)``
    class-level properties (e.g. CHATBOT_MAX_TOOL_ITERATIONS). Also snapshot and
    restore ``os.environ`` so the swept vars / neutralized ``load_dotenv`` the
    reload helper set cannot bleed past this file, independent of monkeypatch
    teardown ordering.
    """
    env_snapshot = dict(os.environ)
    saved = sys.modules.get("config")
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(env_snapshot)
        if saved is not None:
            sys.modules["config"] = saved
        else:
            sys.modules.pop("config", None)


def _reload_config(monkeypatch, env: dict[str, str | None]):
    """Apply the env delta, then drop+reimport ``config``.

    Neutralizes ``dotenv.load_dotenv`` for the duration of the reload so the
    on-disk ``orchestrator/.env`` cannot silently restore the values we just
    deleted (which would mask the "default when unset" branch of the new
    config accessors).
    """
    for name in _SWEPT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **kw: False)
    sys.modules.pop("config", None)
    import config  # noqa: WPS433 — intentional re-import after env change
    importlib.reload(config)
    return config


# ---------------------------------------------------------------------------
# 1. Static checks — swept modules do not call os.getenv / os.environ
# ---------------------------------------------------------------------------


def _ast_env_reads(path: Path) -> list[tuple[int, str]]:
    """Return real (non-docstring) env read attributes in a file."""
    src = path.read_text()
    tree = ast.parse(src)
    docstring_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            first = node.body[0] if node.body else None
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                for ln in range(first.lineno, first.end_lineno + 1):
                    docstring_lines.add(ln)
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "os":
            if node.attr in {"getenv", "environ"} and node.lineno not in docstring_lines:
                hits.append((node.lineno, node.attr))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "load_dotenv":
            if node.lineno not in docstring_lines:
                hits.append((node.lineno, "load_dotenv"))
    return hits


SWEPT_MODULES = [
    "core/monitoring/automatos_logging.py",
    "core/security/rate_limiter.py",
    "core/monitoring/automatos_metrics.py",
    "core/monitoring/automatos_logs_api.py",
    "core/monitoring/automatos_alerts.py",
    "core/database/database.py",
    "api/wizard.py",
    "api/channels.py",
]


@pytest.mark.parametrize("relpath", SWEPT_MODULES)
def test_swept_module_has_no_env_reads(relpath):
    """No os.getenv / os.environ / load_dotenv in any swept runtime file (G7)."""
    path = _ORCHESTRATOR_ROOT / relpath
    hits = _ast_env_reads(path)
    assert hits == [], (
        f"{relpath} still reads env directly: "
        + ", ".join(f"line {ln} os.{attr}" for ln, attr in hits)
    )


@pytest.mark.parametrize("relpath", SWEPT_MODULES)
def test_swept_module_imports_config(relpath):
    """Each swept module must import ``config`` so the centralized accessors are reachable.

    database.py already imports ``from config import config``; this test pins
    that contract across the sweep.
    """
    path = _ORCHESTRATOR_ROOT / relpath
    src = path.read_text()
    assert "from config import config" in src, (
        f"{relpath} does not import 'from config import config' — env reads "
        "should be routed through the centralized config module."
    )


# ---------------------------------------------------------------------------
# 2. Config accessors — exist + default value preserved
# ---------------------------------------------------------------------------


def test_log_relay_url_default_local_safe(monkeypatch):
    """PRD-176 F068: default log-relay target is local-safe, not railway.internal."""
    cfg = _reload_config(monkeypatch, {})
    assert hasattr(cfg.config, "LOG_RELAY_URL")
    assert "railway.internal" not in cfg.config.LOG_RELAY_URL
    assert cfg.config.LOG_RELAY_URL == "http://localhost:8080/push"


def test_log_relay_url_override(monkeypatch):
    cfg = _reload_config(monkeypatch, {"LOG_RELAY_URL": "http://custom:9090/push"})
    assert cfg.config.LOG_RELAY_URL == "http://custom:9090/push"


def test_log_relay_url_override_to_railway(monkeypatch):
    """SaaS still points log relay at the railway host via env."""
    cfg = _reload_config(
        monkeypatch,
        {"LOG_RELAY_URL": "http://log-relay.railway.internal:8080/push"},
    )
    assert cfg.config.LOG_RELAY_URL == "http://log-relay.railway.internal:8080/push"


def test_log_relay_enabled_default_false(monkeypatch):
    """PRD-176 F068: log relay is OFF by default (local); SaaS opts in via env."""
    cfg = _reload_config(monkeypatch, {})
    assert cfg.config.LOG_RELAY_ENABLED is False


def test_log_relay_enabled_on_when_true(monkeypatch):
    cfg = _reload_config(monkeypatch, {"LOG_RELAY_ENABLED": "true"})
    assert cfg.config.LOG_RELAY_ENABLED is True


def test_log_relay_service_name_default_unknown(monkeypatch):
    """Logging-side SERVICE_NAME default is 'unknown' (was os.environ default)."""
    cfg = _reload_config(monkeypatch, {})
    assert cfg.config.LOG_RELAY_SERVICE_NAME == "unknown"


def test_log_relay_service_name_from_env(monkeypatch):
    cfg = _reload_config(monkeypatch, {"SERVICE_NAME": "orchestrator"})
    assert cfg.config.LOG_RELAY_SERVICE_NAME == "orchestrator"


def test_log_relay_environment_default_development(monkeypatch):
    """ENVIRONMENT unset, RAILWAY_ENVIRONMENT unset → 'development'."""
    cfg = _reload_config(monkeypatch, {})
    assert cfg.config.LOG_RELAY_ENVIRONMENT == "development"


def test_log_relay_environment_railway_fallback(monkeypatch):
    """ENVIRONMENT unset, RAILWAY_ENVIRONMENT='staging' → 'staging'."""
    cfg = _reload_config(monkeypatch, {"RAILWAY_ENVIRONMENT": "staging"})
    assert cfg.config.LOG_RELAY_ENVIRONMENT == "staging"


def test_log_relay_environment_env_overrides_railway(monkeypatch):
    cfg = _reload_config(
        monkeypatch,
        {"ENVIRONMENT": "production", "RAILWAY_ENVIRONMENT": "staging"},
    )
    assert cfg.config.LOG_RELAY_ENVIRONMENT == "production"


def test_log_relay_batch_size_default_int(monkeypatch):
    cfg = _reload_config(monkeypatch, {})
    assert cfg.config.LOG_RELAY_BATCH_SIZE == 50
    assert isinstance(cfg.config.LOG_RELAY_BATCH_SIZE, int)


def test_log_relay_batch_size_override(monkeypatch):
    cfg = _reload_config(monkeypatch, {"LOG_RELAY_BATCH_SIZE": "200"})
    assert cfg.config.LOG_RELAY_BATCH_SIZE == 200


def test_log_relay_flush_interval_default_float(monkeypatch):
    cfg = _reload_config(monkeypatch, {})
    assert cfg.config.LOG_RELAY_FLUSH_INTERVAL == pytest.approx(2.0)
    assert isinstance(cfg.config.LOG_RELAY_FLUSH_INTERVAL, float)


def test_log_relay_flush_interval_override(monkeypatch):
    cfg = _reload_config(monkeypatch, {"LOG_RELAY_FLUSH_INTERVAL": "0.5"})
    assert cfg.config.LOG_RELAY_FLUSH_INTERVAL == pytest.approx(0.5)


def test_metrics_service_name_default(monkeypatch):
    """Metrics-side SERVICE_NAME default is 'automatos-backend' (was os.environ default)."""
    cfg = _reload_config(monkeypatch, {})
    assert cfg.config.METRICS_SERVICE_NAME == "automatos-backend"


def test_metrics_service_name_from_env(monkeypatch):
    cfg = _reload_config(monkeypatch, {"SERVICE_NAME": "orchestrator"})
    assert cfg.config.METRICS_SERVICE_NAME == "orchestrator"


def test_metrics_environment_default_unknown(monkeypatch):
    """Metrics-side ENVIRONMENT default is 'unknown' (was os.environ default)."""
    cfg = _reload_config(monkeypatch, {})
    assert cfg.config.METRICS_ENVIRONMENT == "unknown"


def test_metrics_environment_from_env(monkeypatch):
    cfg = _reload_config(monkeypatch, {"ENVIRONMENT": "production"})
    assert cfg.config.METRICS_ENVIRONMENT == "production"


def test_loki_query_url_default_local_safe(monkeypatch):
    """PRD-176 F068: default resolves local-safe, not railway.internal."""
    cfg = _reload_config(monkeypatch, {})
    assert "railway.internal" not in cfg.config.LOKI_QUERY_URL
    assert cfg.config.LOKI_QUERY_URL == "http://localhost:3100"


def test_loki_query_url_override(monkeypatch):
    cfg = _reload_config(monkeypatch, {"LOKI_QUERY_URL": "http://custom:4100"})
    assert cfg.config.LOKI_QUERY_URL == "http://custom:4100"


def test_alert_ingest_token_default_empty(monkeypatch):
    cfg = _reload_config(monkeypatch, {})
    assert cfg.config.ALERT_INGEST_TOKEN == ""


def test_alert_ingest_token_from_env(monkeypatch):
    cfg = _reload_config(monkeypatch, {"ALERT_INGEST_TOKEN": "tok-123"})
    assert cfg.config.ALERT_INGEST_TOKEN == "tok-123"


def test_public_api_host_default(monkeypatch):
    cfg = _reload_config(monkeypatch, {})
    assert cfg.config.PUBLIC_API_HOST == "api.automatos.app"


def test_public_api_host_strips_trailing_slash(monkeypatch):
    """channels.py rstrip('/') behaviour is preserved by the config accessor."""
    cfg = _reload_config(monkeypatch, {"PUBLIC_API_HOST": "api.staging.automatos.app/"})
    assert cfg.config.PUBLIC_API_HOST == "api.staging.automatos.app"


# ---------------------------------------------------------------------------
# 2b. PRD-176 F068 — the nine railway.internal defaults are local-safe
# ---------------------------------------------------------------------------

# Every config attribute that carried a ``*.railway.internal`` default at
# 37fdecc4e. With no env override, none may dial a railway host by default
# (roadmap deployability bar: a fresh clone reaches nothing SaaS-topology).
_RAILWAY_DEFAULT_ATTRS = (
    "INTERNAL_API_HOSTNAME",
    "INTERNAL_FRONTEND_HOSTNAME",
    "LOKI_URL",
    "PROMETHEUS_URL",
    "LOG_RELAY_URL",
    "LOKI_QUERY_URL",
    "AGENT_OPT_WORKER_URL",
    "VOICE_SERVICE_URL",
)


def test_no_railway_internal_in_any_default(monkeypatch):
    """PRD-176 F068: with no env, no config default contains 'railway.internal'."""
    cfg = _reload_config(monkeypatch, {})
    offenders = {
        attr: getattr(cfg.config, attr)
        for attr in _RAILWAY_DEFAULT_ATTRS
        if "railway.internal" in (getattr(cfg.config, attr) or "")
    }
    assert offenders == {}, f"railway.internal leaked into local defaults: {offenders}"


@pytest.mark.parametrize("attr", _RAILWAY_DEFAULT_ATTRS)
def test_railway_default_is_localhost(monkeypatch, attr):
    """Each former railway.internal default now resolves to a localhost value."""
    cfg = _reload_config(monkeypatch, {})
    value = getattr(cfg.config, attr) or ""
    assert "localhost" in value, f"{attr} default is not local-safe: {value!r}"


@pytest.mark.parametrize(
    "attr,env_name,saas_value",
    [
        ("INTERNAL_API_HOSTNAME", "INTERNAL_API_HOSTNAME", "automatos-ai.railway.internal"),
        ("INTERNAL_FRONTEND_HOSTNAME", "INTERNAL_FRONTEND_HOSTNAME", "automatos-ai-frontend.railway.internal"),
        ("LOKI_URL", "LOKI_URL", "http://loki.railway.internal:3100"),
        ("PROMETHEUS_URL", "PROMETHEUS_URL", "http://prometheus.railway.internal:9090"),
        ("AGENT_OPT_WORKER_URL", "AGENT_OPT_WORKER_URL", "http://agent-opt-worker.railway.internal:8080"),
        ("VOICE_SERVICE_URL", "VOICE_SERVICE_URL", "http://voice-service.railway.internal:8300"),
    ],
)
def test_railway_host_still_settable_via_env(monkeypatch, attr, env_name, saas_value):
    """SaaS supplies the real Railway host via env — only the default moved."""
    cfg = _reload_config(monkeypatch, {env_name: saas_value})
    assert getattr(cfg.config, attr) == saas_value


# ---------------------------------------------------------------------------
# 3. rate_limit_for helper — preserves per-operation overrides
# ---------------------------------------------------------------------------


def test_rate_limit_for_default(monkeypatch):
    cfg = _reload_config(monkeypatch, {})
    max_req, window = cfg.config.rate_limit_for("git_clone", default_max=5, default_window=3600)
    assert (max_req, window) == (5, 3600)


def test_rate_limit_for_env_override(monkeypatch):
    cfg = _reload_config(
        monkeypatch,
        {"RATE_LIMIT_GIT_CLONE_MAX": "12", "RATE_LIMIT_GIT_CLONE_WINDOW_SECONDS": "1800"},
    )
    max_req, window = cfg.config.rate_limit_for("git_clone", default_max=5, default_window=3600)
    assert (max_req, window) == (12, 1800)


def test_rate_limit_for_invalid_env_falls_back(monkeypatch):
    cfg = _reload_config(
        monkeypatch,
        {"RATE_LIMIT_GIT_CLONE_MAX": "not-a-number"},
    )
    max_req, window = cfg.config.rate_limit_for("git_clone", default_max=5, default_window=3600)
    assert (max_req, window) == (5, 3600)


def test_rate_limit_for_clamps_to_minimum_one(monkeypatch):
    """Existing rate_limiter._env_limit clamped max(1, …) — preserve that."""
    cfg = _reload_config(
        monkeypatch,
        {"RATE_LIMIT_GIT_CLONE_MAX": "0", "RATE_LIMIT_GIT_CLONE_WINDOW_SECONDS": "0"},
    )
    max_req, window = cfg.config.rate_limit_for("git_clone", default_max=5, default_window=3600)
    assert (max_req, window) == (1, 1)
