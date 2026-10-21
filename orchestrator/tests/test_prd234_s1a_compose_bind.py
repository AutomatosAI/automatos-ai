"""PRD-234 S1a — source guards for the session-mode security posture.

* every compose file: each ``ports:`` entry of every service publishes on
  ``${BIND_ADDRESS:-127.0.0.1}`` — this machine only, unless the operator sets
  ``BIND_ADDRESS`` deliberately. In session mode the API can run commands on the
  machine, and the local edition serves whoever reaches it as the owner; a
  LAN-wide default would hand both to the LAN. PRD-262 US-005 widened the check
  from five named variables to every entry: the frontend (3000), Adminer and
  Gotenberg had slipped through on all interfaces, and a visitor on the network
  could chat with the backend as the owner through the web app's chat proxy.
  PRD-266 US-005a (K10): the backend may publish only on
  ``${API_PUBLISH_ADDRESS:-${BIND_ADDRESS:-127.0.0.1}}`` and the frontend only on
  ``${WEB_PUBLISH_ADDRESS:-${BIND_ADDRESS:-127.0.0.1}}``, each defaulting to BIND and so
  to loopback; every other service (the datastores) keeps exactly BIND and can never
  follow a publish variable. The backend's boot check (``services.mobile.remote_access``)
  then refuses a published install unless ``LOCAL_REMOTE_ACCESS`` says who may call it.
* the fresh-clone smoke lane proves the same at run time
  (``docker compose port frontend 3000`` must print ``127.0.0.1:3000``).
* the pairing token can never be an empty default (the ``WORKER_INTERNAL_TOKEN=""``
  pattern is not copied): ``resolve_host_by_token`` refuses empty/None.
* the boot gate lives OUTSIDE ``run_stage`` (the only place an abort is real).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

REPO = _ORCH.parent
COMPOSE = REPO / "docker-compose.yml"
SMOKE_SH = REPO / "scripts" / "ci" / "smoke-fresh-clone.sh"
LOOPBACK_SH = REPO / "scripts" / "ci" / "smoke-loopback-ports.sh"
SMOKE_YML = REPO / ".github" / "workflows" / "smoke-fresh-clone.yml"

BIND = "${BIND_ADDRESS:-127.0.0.1}"
# PRD-266 (K10): the API's and the web app's own publish variables, each defaulting to BIND.
API_PUBLISH = "${API_PUBLISH_ADDRESS:-" + BIND + "}"
WEB_PUBLISH = "${WEB_PUBLISH_ADDRESS:-" + BIND + "}"
# The one service each publish variable may appear on; every other service publishes on BIND.
_PUBLISH_HOST = {"backend": API_PUBLISH, "frontend": WEB_PUBLISH}
# Compose file names: docker-compose*.yml and compose*.yml, stack files and overrides alike.
_COMPOSE_NAME = re.compile(r"(?:^|/)(?:docker-)?compose(?:[.-][^/]*)?\.ya?ml$")
# Short syntax: the service's host address, then an optional host port (a number, a range
# or ${VAR:-n}), then the container port with an optional protocol.
_HOST_PORT = r"(?:\$\{[A-Z][A-Z0-9_]*(?::-\d+)?\}|\d+(?:-\d+)?)?"
_CONTAINER_PORT = r"\d+(?:-\d+)?(?:/(?:tcp|udp|sctp))?"


def _short_syntax(host: str) -> "re.Pattern[str]":
    return re.compile("^" + re.escape(host) + ":" + _HOST_PORT + ":" + _CONTAINER_PORT + "$")


class _ComposeLoader(yaml.SafeLoader):
    """SafeLoader that reads compose's ``!reset`` / ``!override`` merge tags as plain values."""


def _untagged(loader: yaml.SafeLoader, node: yaml.Node) -> object:
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    return loader.construct_scalar(node)


for _tag in ("!reset", "!override"):
    _ComposeLoader.add_constructor(_tag, _untagged)


def _compose_files() -> list[Path]:
    """Every tracked compose file in the repo, read from the git index."""
    proc = subprocess.run(
        ["git", "ls-files"], cwd=str(REPO), capture_output=True, text=True, timeout=30
    )
    assert proc.returncode == 0, f"git ls-files failed: {proc.stderr}"
    return [REPO / name for name in proc.stdout.splitlines() if _COMPOSE_NAME.search(name)]


def _published_ports(path: Path) -> list[tuple[str, object]]:
    """``(service, entry)`` for every ``ports:`` entry the file declares."""
    data = yaml.load(path.read_text(encoding="utf-8"), Loader=_ComposeLoader) or {}
    services = (data.get("services") if isinstance(data, dict) else None) or {}
    return [
        (name, entry)
        for name, service in services.items()
        for entry in ((service or {}).get("ports") or [])
    ]


def _bind_violation(entry: object, service: str | None = None) -> str | None:
    """Why a ``ports:`` entry would listen beyond this machine; None when it is bound to the
    service's host: its own publish variable for the backend and the frontend, BIND for
    every other service (and when no service is named)."""
    host = _PUBLISH_HOST.get(service, BIND)
    if isinstance(entry, dict):
        host_ip = entry.get("host_ip")
        return None if host_ip == host else f"host_ip is {host_ip!r}, expected {host!r}"
    if isinstance(entry, str) and _short_syntax(host).match(entry):
        return None
    return f"{entry!r} does not publish on {host}:<host port>:<container port>"


def test_every_published_port_listens_on_bind_address():
    """PRD-262 US-005: every ``ports:`` entry of every service, in every compose file.
    PRD-266 K10: the backend and the frontend on their own publish variable, every other
    service on BIND exactly."""
    offenders = [
        f"{path.relative_to(REPO)} · {service}: {reason}"
        for path in _compose_files()
        for service, entry in _published_ports(path)
        if (reason := _bind_violation(entry, service))
    ]
    assert not offenders, "ports that listen beyond this machine:\n" + "\n".join(offenders)


def test_the_port_check_reads_the_stack_it_guards():
    """Non-vacuity: a discovery or parse that found no ports would pass the test above
    for nothing. The canonical file is found, its core services publish, and the
    frontend line this story fixed is the one compose reads (PRD-266: on its own publish
    variable, as the backend's is)."""
    assert COMPOSE in _compose_files()
    ports = _published_ports(COMPOSE)
    assert {"frontend", "backend", "postgres", "redis", "minio"} <= {svc for svc, _ in ports}
    assert ("frontend", WEB_PUBLISH + ":${FRONTEND_PORT:-3000}:3000") in ports
    assert ("backend", API_PUBLISH + ":${API_PORT:-8000}:8000") in ports
    assert ("postgres", BIND + ":${POSTGRES_PORT:-5432}:5432") in ports


@pytest.mark.parametrize(
    "entry",
    [
        "${FRONTEND_PORT:-3000}:3000",  # the frontend line before US-005
        "3000:3000",
        "3000",
        3000,
        "0.0.0.0:3000:3000",
        "127.0.0.1:3000:3000",  # loopback, but the owner can no longer open it
        "${BIND_ADDRESS}:3000:3000",  # no loopback default
        "${BIND_ADDRESS:-0.0.0.0}:3000:3000",
        {"target": 3000, "published": 3000},
        {"target": 3000, "published": 3000, "host_ip": "0.0.0.0"},
    ],
)
def test_the_port_check_refuses_an_all_interfaces_publish(entry):
    assert _bind_violation(entry)


@pytest.mark.parametrize(
    "entry",
    [
        BIND + ":${FRONTEND_PORT:-3000}:3000",
        BIND + ":8080:8080",
        BIND + ":9000:9000/tcp",
        BIND + "::3000",
        {"target": 3000, "published": "${FRONTEND_PORT:-3000}", "host_ip": BIND},
    ],
)
def test_the_port_check_accepts_a_bind_address_publish(entry):
    assert _bind_violation(entry) is None


@pytest.mark.parametrize(
    "service,entry",
    [
        ("backend", "${API_PUBLISH_ADDRESS:-0.0.0.0}:8000:8000"),
        ("backend", "${API_PUBLISH_ADDRESS}:8000:8000"),  # no default
        ("backend", "${API_PUBLISH_ADDRESS:-127.0.0.1}:8000:8000"),  # ignores BIND
        ("backend", "${API_PUBLISH_ADDRESS:-${BIND_ADDRESS:-0.0.0.0}}:8000:8000"),
        ("backend", "0.0.0.0:8000:8000"),
        ("backend", WEB_PUBLISH + ":8000:8000"),  # the web app's variable
        ("backend", {"target": 8000, "published": 8000, "host_ip": "${API_PUBLISH_ADDRESS:-0.0.0.0}"}),
        ("frontend", "${WEB_PUBLISH_ADDRESS:-0.0.0.0}:3000:3000"),
        ("frontend", API_PUBLISH + ":3000:3000"),  # the API's variable
        ("postgres", API_PUBLISH + ":${POSTGRES_PORT:-5432}:5432"),
        ("postgres", WEB_PUBLISH + ":5432:5432"),
        ("redis", API_PUBLISH + ":6379:6379"),
        ("minio", {"target": 9000, "published": 9000, "host_ip": WEB_PUBLISH}),
        ("qdrant", API_PUBLISH + ":6333:6333"),
        ("adminer", WEB_PUBLISH + ":8080:8080"),
        ("searxng", API_PUBLISH + ":8888:8080"),
    ],
)
def test_the_port_check_refuses_a_publish_variable_off_its_service_or_its_default(service, entry):
    """PRD-266 K10: each publish variable only on its own service, and only with BIND
    (and so loopback) as its default; a datastore never follows one."""
    assert _bind_violation(entry, service)


@pytest.mark.parametrize(
    "service,entry",
    [
        ("backend", API_PUBLISH + ":${API_PORT:-8000}:8000"),
        ("frontend", WEB_PUBLISH + ":${FRONTEND_PORT:-3000}:3000"),
        ("backend", {"target": 8000, "published": "${API_PORT:-8000}", "host_ip": API_PUBLISH}),
        ("postgres", BIND + ":${POSTGRES_PORT:-5432}:5432"),
    ],
)
def test_the_port_check_accepts_each_publish_variable_on_its_own_service(service, entry):
    assert _bind_violation(entry, service) is None


def test_the_smoke_lane_asserts_the_frontend_listens_on_loopback():
    """US-005's run-time half: once the frontend answers, the fresh-clone smoke runs
    smoke-loopback-ports.sh, which requires ``docker compose port frontend 3000`` to
    print 127.0.0.1:3000, then refuses any running service that publishes on another
    address. A failed check fails the smoke (``fail`` exits 1 through the report trap)."""
    smoke = SMOKE_SH.read_text(encoding="utf-8")
    assert re.search(r'bash scripts/ci/smoke-loopback-ports\.sh "\$COMPOSE_FILE" \\\n\s+\|\| fail ', smoke)
    # Runs once the frontend is up. Other steps may sit between the two since PRD-262
    # US-036 (make up's exit) and US-037/US-038 (checks of what make up left behind).
    main = re.search(r"^main\(\) \{\n(.*?)^\}", smoke, re.M | re.S).group(1)
    steps = [line.strip() for line in main.splitlines()]
    assert "boot_frontend" in steps and "check_loopback_ports" in steps
    assert steps.index("boot_frontend") < steps.index("check_loopback_ports"), "runs once the frontend is up"
    check = LOOPBACK_SH.read_text(encoding="utf-8")
    assert 'LOOPBACK="127.0.0.1"' in check
    assert 'FRONTEND_EXPECTED="${LOOPBACK}:${FRONTEND_PORT:-3000}"' in check
    assert re.search(r'docker compose -f "\$COMPOSE_FILE" port frontend 3000', check)
    assert "Publishers" in check, "every running service's published ports are checked too"
    triggers = yaml.safe_load(SMOKE_YML.read_text(encoding="utf-8"))[True]  # `on:` parses as True
    for event in ("push", "pull_request"):
        assert "scripts/ci/smoke-loopback-ports.sh" in triggers[event]["paths"], event


def test_compose_documents_the_bind_dial():
    text = COMPOSE.read_text(encoding="utf-8")
    assert "BIND_ADDRESS=0.0.0.0" in text, "the deliberate way to expose must be documented in the file"


def test_host_token_resolution_refuses_empty_tokens():
    from services import cli_host_service as svc

    class _NeverQueried:
        def query(self, *a, **k):  # pragma: no cover - must not be reached
            raise AssertionError("an empty token must short-circuit before any lookup")

    assert svc.resolve_host_by_token(_NeverQueried(), "") is None
    assert svc.resolve_host_by_token(_NeverQueried(), None) is None


def test_boot_gate_runs_outside_run_stage():
    """The CLI gate is inside validate_auth_edition, which main.py calls directly —
    never wrapped by run_stage (the wrapper that swallowed the PRD-175 guard)."""
    cfg = (_ORCH / "config.py").read_text(encoding="utf-8")
    start = cfg.index("def validate_auth_edition(self)")
    body = cfg[start:cfg.index("def validate(self)", start)]
    assert "CLI_RUNTIME_ENABLED" in body and "AUTH_EDITION=local" in body
    main = (_ORCH / "main.py").read_text(encoding="utf-8")
    assert "config.validate_security()" in main
    assert "run_stage(report, BootstrapStage.DATABASE_INIT" in main


def test_claude_driver_invariants_are_not_yet_contradicted_in_backend_code():
    """S1b owns the host; the backend must not pre-empt the §Terms invariant by
    handling credentials itself. No backend module reads Claude Code credentials
    or sets the API key into a session environment."""
    forbidden = ("CLAUDE_CODE_OAUTH_TOKEN", ".credentials.json", "CLAUDE_CODE_ENTRYPOINT")
    for path in (
        _ORCH / "services" / "cli_host_service.py",
        _ORCH / "api" / "cli_hosts.py",
        _ORCH / "core" / "cli_runtime.py",
    ):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path.name} must not reference {token}"


def test_compose_passes_the_session_mode_dial_into_the_backend():
    """A .env value only reaches a container through `environment:` (or the
    api.local lane). CLI_RUNTIME_ENABLED must be wired, default off."""
    text = COMPOSE.read_text(encoding="utf-8")
    assert "CLI_RUNTIME_ENABLED: ${CLI_RUNTIME_ENABLED:-false}" in text
