"""PRD-233 S1 — workspace-worker joins the local profile (container-free).

The Code Canvas runtime (``services/workspace-worker``) ships to the laptop:
compose runs it by default against a designated HOST directory
(``${AUTOMATOS_WORKSPACE_DIR:-./workspaces}`` bind-mounted at ``/workspaces``),
and the worker resolves its confinement root through ONE seam,
``worker_config.workspace_root()`` — env reads live only in ``worker_config``
(the worker's mirror of the orchestrator's ``config.py`` discipline).

Proven here:
  * root resolution: default when unset (Railway parity: ``/workspaces``),
    override honoured, blank == unset (never a cwd-relative root), read at
    CALL time rather than import time;
  * the per-workspace root ``<root>/<workspace_id>`` (``WorkspaceManager.root``)
    derives from that seam, and ``evaluate_tool_confinement`` bound to it
    denies escapes from the resolved host directory (outside the dial, ``..``
    traversal, a sibling workspace inside the same host dir, bash references)
    while re-binding in-root relative paths. The dial moves WHERE the root is,
    never what a session may reach. ``test_prd170_canvas_session_manager``
    covers the confinement primitives against a literal root; this file
    covers the root-from-config seam that feeds them;
  * compose source guard: worker in the default profile (no ``profiles``),
    the host bind mount on the worker (rw) and the backend (ro) with the same
    source, no named ``workspace_data`` volume left behind, mount target ==
    the worker's ``WORKSPACE_VOLUME_PATH`` == the config default, credential
    passthrough + resource limits kept, and the backend's ``WORKER_INTERNAL_URL``
    (``envs/api.defaults``) pointing at the compose service on the worker's
    health port;
  * the default host directory is gitignored (public repo, user deliverables).

Pure stdlib + yaml + pytest: no DB, no docker, no SDK.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKER_DIR = _REPO_ROOT / "services" / "workspace-worker"
_COMPOSE = _REPO_ROOT / "docker-compose.yml"
_API_DEFAULTS = _REPO_ROOT / "envs" / "api.defaults"
_GITIGNORE = _REPO_ROOT / ".gitignore"

# The worker is a flat-module service (hyphenated dir, not a package) — put
# its directory on sys.path just long enough to import the modules under
# test, then remove the entry so nothing else resolves against it.
sys.path.insert(0, str(_WORKER_DIR))
try:
    import worker_config as wc
    from canvas_confinement import evaluate_tool_confinement
    from workspace_manager import WorkspaceManager
finally:
    sys.path.remove(str(_WORKER_DIR))

WORKER_SERVICE = "workspace-worker"
BACKEND_SERVICE = "backend"
MOUNT_TARGET = "/workspaces"
# The designated host directory — Q1 owner decision: one dial, safe default.
HOST_DIR_SOURCE = "${AUTOMATOS_WORKSPACE_DIR:-./workspaces}"
RETIRED_VOLUME = "workspace_data"
CREDENTIAL_PASSTHROUGH = ("ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN")


# ---------------------------------------------------------------------------
# worker_config.workspace_root — the ONE env seam for the mount root
# ---------------------------------------------------------------------------


def test_workspace_root_defaults_to_railway_parity_value(monkeypatch):
    monkeypatch.delenv(wc.WORKSPACE_ROOT_ENV, raising=False)
    assert wc.workspace_root() == Path(wc.DEFAULT_WORKSPACE_ROOT) == Path("/workspaces")


def test_workspace_root_honours_override(monkeypatch, tmp_path):
    monkeypatch.setenv(wc.WORKSPACE_ROOT_ENV, str(tmp_path / "host-dir"))
    assert wc.workspace_root() == tmp_path / "host-dir"


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_workspace_root_blank_counts_as_unset(monkeypatch, blank):
    monkeypatch.setenv(wc.WORKSPACE_ROOT_ENV, blank)
    root = wc.workspace_root()
    assert root == Path(wc.DEFAULT_WORKSPACE_ROOT)
    assert root.is_absolute(), "a cwd-relative root must never be returned"


def test_workspace_root_is_read_at_call_time(monkeypatch, tmp_path):
    monkeypatch.setenv(wc.WORKSPACE_ROOT_ENV, str(tmp_path / "a"))
    first = wc.workspace_root()
    monkeypatch.setenv(wc.WORKSPACE_ROOT_ENV, str(tmp_path / "b"))
    assert first == tmp_path / "a"
    assert wc.workspace_root() == tmp_path / "b"


def test_workspace_manager_root_derives_from_worker_config(monkeypatch, tmp_path):
    monkeypatch.setenv(wc.WORKSPACE_ROOT_ENV, str(tmp_path / "host-dir"))
    assert WorkspaceManager("ws-a").root == tmp_path / "host-dir" / "ws-a"
    # An explicit root still wins (the health server + canvas manager pass one).
    explicit = WorkspaceManager("ws-a", str(tmp_path / "other"))
    assert explicit.root == tmp_path / "other" / "ws-a"


# ---------------------------------------------------------------------------
# Confinement against the RESOLVED host root (extends test_prd170 coverage)
# ---------------------------------------------------------------------------


@pytest.fixture
def host_dir(monkeypatch, tmp_path) -> Path:
    """A designated host directory holding two workspaces, wired as the root."""
    host = tmp_path / "host-dir"
    for workspace_id in ("ws-a", "ws-b"):
        (host / workspace_id / "reports").mkdir(parents=True)
    (tmp_path / "outside.txt").write_text("host file outside the dial\n")
    monkeypatch.setenv(wc.WORKSPACE_ROOT_ENV, str(host))
    return host


def _root(workspace_id: str) -> Path:
    """Same derivation the canvas manager uses: ``WorkspaceManager(...).root.resolve()``."""
    return WorkspaceManager(workspace_id).root.resolve()


@pytest.mark.parametrize(
    "tool_name, tool_input",
    [
        ("Read", {"file_path": "{tmp}/outside.txt"}),  # host path outside the dial
        ("Read", {"file_path": "../ws-b/reports/x.md"}),  # traversal into a sibling
        ("Write", {"file_path": "{host}/ws-b/reports/x.md"}),  # absolute sibling, same host dir
        ("Read", {"file_path": "../../outside.txt"}),  # traversal out of the host dir
        ("Bash", {"command": "cat {host}/ws-b/reports/x.md"}),  # bash reference to a sibling
        ("Bash", {"command": "ls ../"}),  # bash parent-directory component
    ],
)
def test_confinement_denies_escapes_from_resolved_host_root(
    host_dir, tmp_path, tool_name, tool_input
):
    filled = {k: v.format(tmp=tmp_path, host=host_dir) for k, v in tool_input.items()}
    verdict = evaluate_tool_confinement(tool_name, filled, _root("ws-a"))
    assert not verdict.allowed, f"{tool_name} {filled} must be denied"
    assert verdict.reason


def test_confinement_rebinds_in_root_relative_paths(host_dir):
    root = _root("ws-a")
    verdict = evaluate_tool_confinement(
        "Write", {"file_path": "reports/out.md", "content": "x"}, root
    )
    assert verdict.allowed
    assert verdict.updated_input["file_path"] == str(root / "reports" / "out.md")
    assert verdict.updated_input["content"] == "x"


def test_confinement_follows_the_dial_not_a_baked_in_path(monkeypatch, tmp_path):
    """Move the dial: the same absolute path flips from denied to allowed."""
    first, second = tmp_path / "first", tmp_path / "second"
    for host in (first, second):
        (host / "ws-a").mkdir(parents=True)
    target = {"file_path": str(second / "ws-a" / "notes.md")}

    monkeypatch.setenv(wc.WORKSPACE_ROOT_ENV, str(first))
    assert not evaluate_tool_confinement("Read", target, _root("ws-a")).allowed

    monkeypatch.setenv(wc.WORKSPACE_ROOT_ENV, str(second))
    assert evaluate_tool_confinement("Read", target, _root("ws-a")).allowed


# ---------------------------------------------------------------------------
# Compose source guard — the local profile carries the worker
# ---------------------------------------------------------------------------

# Short-syntax volume entry: SOURCE:TARGET[:MODE]. The source may itself hold
# ':' (``${VAR:-default}``), so the target is anchored on the first ':/'.
_SHORT_VOLUME_RE = re.compile(r"^(?P<source>.+?):(?P<target>/[^:]*)(?::(?P<mode>[a-zA-Z,]+))?$")


def _compose() -> dict:
    return yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))


def _mounts(service: dict) -> list[dict]:
    """Normalise short- and long-syntax volume entries to source/target/read_only."""
    out: list[dict] = []
    for entry in service.get("volumes", []) or []:
        if isinstance(entry, dict):
            out.append({
                "source": entry.get("source"),
                "target": entry.get("target"),
                "read_only": bool(entry.get("read_only")),
            })
            continue
        match = _SHORT_VOLUME_RE.match(entry)
        if not match:  # anonymous volume, e.g. "/app/node_modules"
            out.append({"source": None, "target": entry, "read_only": False})
            continue
        mode = (match.group("mode") or "").split(",")
        out.append({
            "source": match.group("source"),
            "target": match.group("target"),
            "read_only": "ro" in mode,
        })
    return out


def _mount(service: dict, target: str) -> dict:
    hits = [m for m in _mounts(service) if m["target"] == target]
    assert len(hits) == 1, f"expected exactly one mount at {target}, got {hits}"
    return hits[0]


def _env(service: dict) -> dict[str, str]:
    env = service.get("environment", {}) or {}
    if isinstance(env, list):
        env = dict(item.split("=", 1) for item in env)
    return {k: "" if v is None else str(v) for k, v in env.items()}


def _api_defaults() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in _API_DEFAULTS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def test_worker_runs_in_the_default_profile():
    svc = _compose()["services"][WORKER_SERVICE]
    assert "profiles" not in svc, (
        "workspace-worker must not sit behind a compose profile — it is the local edition's runtime"
    )


def test_worker_mounts_the_designated_host_directory_read_write():
    mount = _mount(_compose()["services"][WORKER_SERVICE], MOUNT_TARGET)
    assert mount["source"] == HOST_DIR_SOURCE
    assert mount["read_only"] is False


def test_backend_mounts_the_same_host_directory_read_only():
    mount = _mount(_compose()["services"][BACKEND_SERVICE], MOUNT_TARGET)
    assert mount["source"] == HOST_DIR_SOURCE
    assert mount["read_only"] is True


def test_named_workspace_volume_is_gone():
    compose = _compose()
    assert RETIRED_VOLUME not in (compose.get("volumes") or {}), (
        f"top-level volume {RETIRED_VOLUME} must be deleted with its last reference"
    )
    for name, svc in compose["services"].items():
        for mount in _mounts(svc):
            assert mount["source"] != RETIRED_VOLUME, f"{name} still mounts {RETIRED_VOLUME}"


def test_mount_target_matches_worker_config_default():
    compose = _compose()
    worker_env = _env(compose["services"][WORKER_SERVICE])
    assert worker_env[wc.WORKSPACE_ROOT_ENV] == MOUNT_TARGET == wc.DEFAULT_WORKSPACE_ROOT
    # The backend's view of the same directory (config.WORKSPACE_VOLUME_PATH).
    assert _api_defaults()[wc.WORKSPACE_ROOT_ENV] == MOUNT_TARGET


def test_worker_keeps_credential_passthrough_and_limits():
    svc = _compose()["services"][WORKER_SERVICE]
    env = _env(svc)
    for key in CREDENTIAL_PASSTHROUGH:
        assert env.get(key) == f"${{{key}:-}}", f"{key} must pass through from .env without blocking boot"
    limits = svc.get("deploy", {}).get("resources", {}).get("limits", {})
    assert limits.get("cpus") and limits.get("memory"), "worker resource limits must stay"


def test_backend_reaches_worker_at_compose_service_url():
    worker_env = _env(_compose()["services"][WORKER_SERVICE])
    expected = f"http://{WORKER_SERVICE}:{worker_env['WORKER_HEALTH_PORT']}"
    assert _api_defaults().get("WORKER_INTERNAL_URL") == expected, (
        "envs/api.defaults must point WORKER_INTERNAL_URL at the compose service "
        "(config.py's localhost default reaches nothing inside the backend container)"
    )


def test_default_host_directory_is_gitignored():
    default_dir = re.fullmatch(r"\$\{AUTOMATOS_WORKSPACE_DIR:-\./([^}]+)\}", HOST_DIR_SOURCE).group(1)
    lines = [line.strip() for line in _GITIGNORE.read_text(encoding="utf-8").splitlines()]
    assert f"/{default_dir}/" in lines, f"/{default_dir}/ must be gitignored (user deliverables, public repo)"
