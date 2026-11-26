"""
Worker-local config seam (PRD-203 C·S8)
=======================================

The workspace-worker is a self-contained image (``COPY . .`` / ``PYTHONPATH=/app``)
— at runtime it has NO access to the orchestrator's ``config.py``. This module
centralises the worker's environment reads so call sites never do ``os.getenv``
inline (mirrors the orchestrator's config discipline).

Named ``worker_config`` — deliberately NOT ``config`` — so it can never collide
with the orchestrator's ``config.py`` that ``main.py`` transiently places on
``sys.path`` (``sys.path.insert(0, "../../orchestrator")``).
"""

from __future__ import annotations

import os
import ssl
from pathlib import Path
from typing import Dict, Mapping, Optional

# Workspace mount root inside the worker container (PRD-233 S1). Compose
# bind-mounts the host directory ``${AUTOMATOS_WORKSPACE_DIR:-./workspaces}``
# here; Railway mounts its persistent volume here. Per-workspace roots are
# ``workspace_root() / <workspace_id>`` (``WorkspaceManager.root``) — the
# boundary every canvas tool call is re-bound against (canvas_confinement).
WORKSPACE_ROOT_ENV = "WORKSPACE_VOLUME_PATH"
DEFAULT_WORKSPACE_ROOT = "/workspaces"


def model_auth_env() -> Dict[str, str]:
    """The model credential for the headless Claude Agent SDK subprocess.

    Returns the subset of ``{CLAUDE_CODE_OAUTH_TOKEN, ANTHROPIC_API_KEY}`` that is
    actually set — merged into ``ClaudeAgentOptions.env`` by ``start_session`` so
    the SDK subprocess is authenticated. An OAuth token (subscription) is
    preferred when both are present; either alone is sufficient.

    Empty dict → no credential configured. ``start_session`` treats that as a
    fail-fast condition on the real SDK path (never a silent idle).
    """
    env: Dict[str, str] = {}
    token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "").strip()
    if token:
        env["CLAUDE_CODE_OAUTH_TOKEN"] = token
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        env["ANTHROPIC_API_KEY"] = key
    return env


# The coding agent's environment. The Claude Agent SDK starts the CLI with the worker's own
# environment, with ``ClaudeAgentOptions.env`` laid over it, and every command the agent runs
# inherits the result. The worker holds the install's database and Redis addresses, its token
# for the API and its cloud role, none of which the agent needs. So every variable the worker
# holds is blanked for the agent, except these names and prefixes: what a CLI and its commands
# need to run (paths, locale, time zone, terminal, temp folders, proxies, CA bundles) and the
# CLI's own switches. The model credential is added back by ``model_auth_env``.
AGENT_ENV_KEPT_NAMES = frozenset({
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "PWD", "LANG", "LANGUAGE", "TZ", "TERM",
    "TMPDIR", "TMP", "TEMP",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "NODE_EXTRA_CA_CERTS",
    "ANTHROPIC_BASE_URL",
})
AGENT_ENV_KEPT_PREFIXES = ("LC_", "DISABLE_")
# The CLI's own switches, except its credentials (``CLAUDE_CODE_OAUTH_TOKEN``, the OAuth refresh
# token, any later ``*_TOKEN``/``*_KEY``/``*_SECRET``): only ``model_auth_env`` passes a credential.
AGENT_ENV_CLI_PREFIX = "CLAUDE_CODE_"
AGENT_ENV_CLI_SECRET_SUFFIXES = ("_TOKEN", "_KEY", "_SECRET")


def _kept_for_agent(name: str) -> bool:
    """True when the coding agent may see the worker's own value of ``name``."""
    if name in AGENT_ENV_KEPT_NAMES or name.startswith(AGENT_ENV_KEPT_PREFIXES):
        return True
    return name.startswith(AGENT_ENV_CLI_PREFIX) and not name.endswith(AGENT_ENV_CLI_SECRET_SUFFIXES)


def agent_environment(inherited: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """What the coding agent's ``ClaudeAgentOptions.env`` lays over the worker's environment.

    Every variable in ``inherited`` (the worker's own environment by default) that the agent is
    not to see is set to the empty string. The SDK cannot drop a variable from the environment it
    passes on, only replace its value, so blanking is how one is withheld. The result never adds
    a value of its own: the caller adds ``CLAUDE_CONFIG_DIR`` and the model credential after it.
    """
    names = os.environ if inherited is None else inherited
    return {name: "" for name in names if not _kept_for_agent(name)}


def has_model_auth() -> bool:
    """True iff a model credential is configured for the SDK subprocess."""
    return bool(model_auth_env())


def workspace_root() -> Path:
    """The workspace mount root (``WORKSPACE_VOLUME_PATH``, default ``/workspaces``).

    Read at call time, not import time, so it always reflects the process
    environment. Blank/whitespace counts as unset: a relative root silently
    anchored to the process cwd is the one thing this must never return.
    """
    raw = os.environ.get(WORKSPACE_ROOT_ENV, "").strip()
    return Path(raw or DEFAULT_WORKSPACE_ROOT)


# The largest binary file the platform may write through /files/write a piece at a
# time (the Socials copies of approved pictures and videos, executor._write_chunk).
MAX_BINARY_WRITE_ENV = "WORKER_MAX_BINARY_WRITE_BYTES"
DEFAULT_MAX_BINARY_WRITE_BYTES = 500 * 1024 * 1024


def max_binary_write_bytes() -> int:
    """The cap on one binary file written in pieces (``WORKER_MAX_BINARY_WRITE_BYTES``)."""
    raw = os.environ.get(MAX_BINARY_WRITE_ENV, "").strip()
    return int(raw) if raw.isdigit() else DEFAULT_MAX_BINARY_WRITE_BYTES


# The internal HTTP server (worker_http.py): the shared secret the platform sends in
# ``X-Internal-Token`` (empty = not enforced), and the address it binds.
INTERNAL_TOKEN_ENV = "WORKER_INTERNAL_TOKEN"
BIND_HOST_ENV = "WORKER_BIND_HOST"
DEFAULT_BIND_HOST = "0.0.0.0"


def worker_internal_token() -> str:
    """``WORKER_INTERNAL_TOKEN``; empty when unset."""
    return os.environ.get(INTERNAL_TOKEN_ENV, "")


def worker_bind_host() -> str:
    """``WORKER_BIND_HOST``, default ``0.0.0.0``."""
    return os.environ.get(BIND_HOST_ENV, DEFAULT_BIND_HOST)


# PRD-256 O2b (#847): OpenTelemetry traces (worker_otel.py), the same names and
# defaults as the platform's (orchestrator/config.py), and default OFF.
DEFAULT_OTEL_SERVICE_NAME = "automatos-workspace-worker"
DEFAULT_OTEL_ENDPOINT = "http://localhost:4318"
DEFAULT_OTEL_SAMPLER_RATIO = "1.0"


def otel_enabled() -> bool:
    """``OTEL_ENABLED``: traces for this worker's HTTP requests."""
    return os.environ.get("OTEL_ENABLED", "false").strip().lower() in ("true", "1", "yes", "on")


def otel_service_name() -> str:
    """``OTEL_SERVICE_NAME``, default ``automatos-workspace-worker``."""
    return os.environ.get("OTEL_SERVICE_NAME", "").strip() or DEFAULT_OTEL_SERVICE_NAME


def otel_endpoint() -> str:
    """``OTEL_EXPORTER_OTLP_ENDPOINT``: the collector's OTLP/HTTP base address."""
    return os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip() or DEFAULT_OTEL_ENDPOINT


def otel_headers() -> str:
    """``OTEL_EXPORTER_OTLP_HEADERS`` (``key=value,key2=value2``, URL-encoded values)."""
    return os.environ.get("OTEL_EXPORTER_OTLP_HEADERS", "")


def otel_sampler_ratio() -> str:
    """``OTEL_TRACES_SAMPLER_RATIO``, as text: parsed when tracing starts."""
    return os.environ.get("OTEL_TRACES_SAMPLER_RATIO", DEFAULT_OTEL_SAMPLER_RATIO)


# The platform's Postgres, for the worker's one write: a task's lifecycle in
# task_executions (task_executions_db.py). SQLAlchemy's driver form
# (postgresql+psycopg2://) is accepted; the write normalises it for libpq.
DATABASE_URL_ENV = "DATABASE_URL"


def database_url() -> str:
    """``DATABASE_URL``; empty when unset or blank."""
    return os.environ.get(DATABASE_URL_ENV, "").strip()


# PRD-278 W3: TLS on that connection, from the same two settings the platform reads
# (orchestrator/core/database/tls.py): libpq's sslmode, and the CA file a verify-ca or
# verify-full check uses, which must be a file (a PEM bundle) and is required with either.
# Both unset: no keyword is added, and the write connects as before.
DATABASE_SSLMODE_ENV = "DATABASE_SSLMODE"
DATABASE_SSLROOTCERT_ENV = "DATABASE_SSLROOTCERT"
DATABASE_SSLMODES = ("disable", "allow", "prefer", "require", "verify-ca", "verify-full")
DATABASE_CERTIFICATE_CHECKING_MODES = ("verify-ca", "verify-full")
_UNUSABLE_CA = f"{DATABASE_SSLROOTCERT_ENV} must name a readable file holding at least one PEM certificate"
_CA_REQUIRED = (
    f"{DATABASE_SSLMODE_ENV} verify-ca and verify-full need {DATABASE_SSLROOTCERT_ENV}: the path of a file "
    "holding the CA certificate(s) that signed the database server's certificate"
)


def _require_usable_ca(cafile: str) -> None:
    """Raise ValueError, naming the setting only, unless ``cafile`` loads at least one certificate."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    try:
        context.load_verify_locations(cafile=cafile)
    except OSError:  # ssl.SSLError included: missing, unreadable, or no certificate in it
        raise ValueError(_UNUSABLE_CA) from None
    if not context.cert_store_stats().get("x509"):
        raise ValueError(_UNUSABLE_CA)


def database_tls_params() -> Dict[str, str]:
    """psycopg2's ``sslmode`` and ``sslrootcert`` keywords from the two settings, checked.

    Empty when neither is set. Raises ValueError, naming the setting and never the CA
    file's path, for a mode libpq doesn't know, a CA file without a mode that checks it, a
    mode that checks the certificate with no CA file, or a CA file that is missing,
    unreadable or holds no certificate.
    """
    mode = os.environ.get(DATABASE_SSLMODE_ENV, "").strip().lower()
    rootcert = os.environ.get(DATABASE_SSLROOTCERT_ENV, "").strip()
    if mode and mode not in DATABASE_SSLMODES:
        raise ValueError(f"{DATABASE_SSLMODE_ENV} must be one of {', '.join(DATABASE_SSLMODES)}; got {mode!r}")
    if rootcert and mode not in DATABASE_CERTIFICATE_CHECKING_MODES:
        raise ValueError(
            f"{DATABASE_SSLROOTCERT_ENV} is used only with {DATABASE_SSLMODE_ENV} verify-ca or verify-full"
        )
    if mode in DATABASE_CERTIFICATE_CHECKING_MODES and not rootcert:
        raise ValueError(_CA_REQUIRED)
    params: Dict[str, str] = {}
    if mode:
        params["sslmode"] = mode
    if rootcert:
        _require_usable_ca(rootcert)
        params["sslrootcert"] = rootcert
    return params
