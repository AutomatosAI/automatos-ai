"""PRD-234 S1a — source guards for the session-mode security posture.

* every compose file: each ``ports:`` entry of every service publishes on
  ``${BIND_ADDRESS:-127.0.0.1}`` — this machine only, unless the operator sets
  ``BIND_ADDRESS`` deliberately. In session mode the API can run commands on the
  machine, and the local edition serves whoever reaches it as the owner; a
  LAN-wide default would hand both to the LAN. PRD-262 US-005 widened the check
  from five named variables to every entry: the frontend (3000), Adminer and
  Gotenberg had slipped through on all interfaces, and a visitor on the network
  could chat with the backend as the owner through the web app's chat proxy.
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
# Compose file names: docker-compose*.yml and compose*.yml, stack files and overrides alike.
_COMPOSE_NAME = re.compile(r"(?:^|/)(?:docker-)?compose(?:[.-][^/]*)?\.ya?ml$")
# Short syntax: BIND, then an optional host port (a number, a range or ${VAR:-n}), then
# the container port with an optional protocol.
_HOST_PORT = r"(?:\$\{[A-Z][A-Z0-9_]*(?::-\d+)?\}|\d+(?:-\d+)?)?"
_CONTAINER_PORT = r"\d+(?:-\d+)?(?:/(?:tcp|udp|sctp))?"
_SHORT_SYNTAX = re.compile("^" + re.escape(BIND) + ":" + _HOST_PORT + ":" + _CONTAINER_PORT + "$")


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


def _bind_violation(entry: object) -> str | None:
    """Why a ``ports:`` entry would listen beyond this machine; None when it is bound to BIND."""
    if isinstance(entry, dict):
        host_ip = entry.get("host_ip")
        return None if host_ip == BIND else f"host_ip is {host_ip!r}, expected {BIND!r}"
    if isinstance(entry, str) and _SHORT_SYNTAX.match(entry):
        return None
    return f"{entry!r} does not publish on {BIND}:<host port>:<container port>"


def test_every_published_port_listens_on_bind_address():
    """PRD-262 US-005: every ``ports:`` entry of every service, in every compose file."""
    offenders = [
        f"{path.relative_to(REPO)} · {service}: {reason}"
        for path in _compose_files()
        for service, entry in _published_ports(path)
        if (reason := _bind_violation(entry))
    ]
    assert not offenders, "ports that listen beyond this machine:\n" + "\n".join(offenders)


def test_the_port_check_reads_the_stack_it_guards():
    """Non-vacuity: a discovery or parse that found no ports would pass the test above
    for nothing. The canonical file is found, its core services publish, and the
    frontend line this story fixed is the one compose reads."""
    assert COMPOSE in _compose_files()
    ports = _published_ports(COMPOSE)
    assert {"frontend", "backend", "postgres", "redis", "minio"} <= {svc for svc, _ in ports}
    assert ("frontend", BIND + ":${FRONTEND_PORT:-3000}:3000") in ports


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
