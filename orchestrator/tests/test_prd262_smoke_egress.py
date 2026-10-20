"""PRD-262 US-033: the egress smoke boots the backend with no internet.

The lane runs only in CI, against a real stack. These tests hold its parts:

* ``scripts/ci/smoke_boot.py --within`` turns a boot into a bar: a URL that first
  answered 200 later than the bar, timed from the container's start, fails the boot
  though its record shows it answered;
* ``scripts/ci/smoke-egress.sh`` cuts the egress the way SMOKE_EGRESS names. Refused:
  every backend container it starts reads a compose override that sends HTTP(S)
  through a proxy at 127.0.0.1:9 and keeps the stack's services direct. Blackholed:
  after the datastores have created the network, and before the backend starts, it adds
  ``iptables -I DOCKER-USER -s <subnet> ! -d <subnet> -p tcp -j DROP`` for the stack's
  subnet, and deletes it again on exit. It proves the cut is the kind it says before the
  backend starts, holds a new container to the 30 s fresh-boot bar, and loads the
  tokenizer inside it with names core.context_guard defines. Before any backend container,
  a one-off backend under docker-compose.core.yml creates the local workspace's mount
  point in the workspaces volume, which the read-only mount of docker-compose.yml's
  backend cannot do on a fresh volume and the absent worker would have done;
* ``smoke-fresh-clone.yml``'s ``egress-boot`` job runs it once per way and uploads the
  records.

The script runs here against a fake ``docker`` and ``sudo`` that record their calls.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts" / "ci" / "smoke-egress.sh"
_WORKFLOW = _REPO / ".github" / "workflows" / "smoke-fresh-clone.yml"
_CONTEXT_GUARD = _REPO / "orchestrator" / "core" / "context_guard.py"
NS = 1_000_000_000
BACKEND = "automatos_backend"
SUBNET = "172.18.0.0/16"
NETWORK = "automatos_network"
RULE = f"DOCKER-USER -s {SUBNET} ! -d {SUBNET} -p tcp -j DROP"
ENCODING_URL = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
# The one-off backend under the core override that creates the local workspace's mount point.
MOUNTPOINT = ("docker compose -f docker-compose.yml -f docker-compose.core.yml "
              "run --rm -T --no-deps --entrypoint true backend")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / "scripts" / "ci" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # before exec, as the scripts register their siblings
    spec.loader.exec_module(module)
    return module


boot = _load("smoke_boot")


def _started_at(ns: int) -> str:
    """Like ``State.StartedAt``: RFC 3339, UTC, with a nine-digit fraction."""
    seconds, fraction = divmod(ns, NS)
    return datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + f".{fraction:09d}Z"


# --------------------------------------------------------------------------- smoke_boot --within


def test_late_probes_are_the_ones_that_answered_after_the_bar():
    probes = {
        "never": {"seconds": None},
        "early": {"seconds": 29.9},
        "on_the_bar": {"seconds": 30.0},
        "late": {"seconds": 30.5},
    }
    assert boot.late_probes(probes, 30) == ["late"]
    assert boot.late_probes(probes, None) == []


def _wait_with_bar(tmp_path: Path, monkeypatch, booted_seconds_ago: float) -> tuple[int, dict]:
    """A boot whose /health answers 200 at once, ``booted_seconds_ago`` after the container started."""
    started_at = _started_at(time.time_ns() - int(booted_seconds_ago * NS))
    monkeypatch.setattr(boot, "container_started_at", lambda container: started_at)
    monkeypatch.setattr(boot, "probe", lambda url: (200, url))
    monkeypatch.setattr(boot, "fetch_json", lambda url: {"stages": []})
    out = tmp_path / "boots.jsonl"
    code = boot.main(["--label", "offline-new-container", "--container", BACKEND,
                      "--url", "health=http://x/health", "--timeout", "5", "--within", "30", "--out", str(out)])
    return code, json.loads(out.read_text(encoding="utf-8"))


def test_a_boot_within_the_bar_passes(tmp_path, monkeypatch, capsys):
    code, record = _wait_with_bar(tmp_path, monkeypatch, booted_seconds_ago=12)
    assert code == 0
    assert record["ok"] is True and record["probes"]["health"]["seconds"] < 30
    out = capsys.readouterr().out
    assert "OK: health" in out and "bar" not in out


def test_a_boot_past_the_bar_fails_though_it_answered(tmp_path, monkeypatch, capsys):
    code, record = _wait_with_bar(tmp_path, monkeypatch, booted_seconds_ago=95)
    assert code == 1
    # The record says what happened: it answered, 95 s in. The bar fails the boot.
    assert record["ok"] is True and record["probes"]["health"]["seconds"] > 30
    out = capsys.readouterr().out
    assert "FAIL: health (http://x/health) answered 200" in out
    assert "over the 30s bar [offline-new-container]" in out


def test_a_boot_that_never_answered_fails_before_the_bar_is_read(tmp_path, monkeypatch):
    monkeypatch.setattr(boot, "container_started_at", lambda container: "0001-01-01T00:00:00Z")
    out = tmp_path / "boots.jsonl"
    code = boot.main(["--label", "offline-new-container", "--container", BACKEND,
                      "--url", "health=http://x/health", "--within", "30", "--out", str(out)])
    assert code == 1
    assert not out.exists()


# --------------------------------------------------------------------------- the script's text


def _script() -> str:
    return _SCRIPT.read_text(encoding="utf-8")


def _function(text: str, name: str) -> str:
    match = re.search(rf"^{name}\(\) \{{\n(.*?)^\}}", text, re.M | re.S)
    assert match, f"smoke-egress.sh has no {name}() function"
    return match.group(1)


def test_the_script_holds_a_new_container_to_the_fresh_boot_bar():
    text = _script()
    assert re.search(r"^FRESH_BOOT_BAR_SECONDS=30$", text, re.M), "PRD-262 §8's fresh-boot bar is 30 s"
    new_container = _function(text, "boot_new_container")
    assert "compose up -d --no-build --no-deps --force-recreate backend" in new_container
    assert '--within "$FRESH_BOOT_BAR_SECONDS"' in new_container
    # The first container's log is kept: the recreate removes it.
    assert new_container.index('docker logs -t "$BACKEND_CONTAINER" > "$FIRST_CONTAINER_LOG"') \
        < new_container.index("--force-recreate")


def test_the_script_cuts_egress_the_two_ways_the_prd_names():
    text = _script()
    assert re.search(r'^REFUSING_PROXY="http://127\.0\.0\.1:9"$', text, re.M)
    blackhole = _function(text, "blackhole_egress")
    assert 'sudo "$tool" -I DOCKER-USER -s "$subnet" ! -d "$subnet" -p tcp -j DROP' in blackhole
    restore = _function(text, "restore_egress")
    assert '-D DOCKER-USER -s "${rule#* }" ! -d "${rule#* }" -p tcp -j DROP' in restore
    direct = re.search(r'^DIRECT_HOSTS="([^"]+)"$', text, re.M).group(1).split(",")
    for host in ("localhost", "127.0.0.1", "postgres", "redis", "minio", "workspace-worker"):
        assert host in direct, f"NO_PROXY must keep {host} direct"


def test_the_tokenizer_check_imports_what_the_context_guard_defines():
    snippet = re.search(r"^TOKENIZER_CHECK='(.*?)'$", _script(), re.M | re.S).group(1)
    tree = ast.parse(snippet)
    (imported,) = [node for node in tree.body if isinstance(node, ast.ImportFrom)]
    assert imported.module == "core.context_guard"
    guard = ast.parse(_CONTEXT_GUARD.read_text(encoding="utf-8"))
    defined = {node.name for node in guard.body if isinstance(node, ast.FunctionDef)}
    defined |= {target.id for node in guard.body if isinstance(node, ast.Assign)
                for target in node.targets if isinstance(target, ast.Name)}
    assert {alias.name for alias in imported.names} <= defined


def test_the_script_checks_the_host_before_it_sets_the_teardown_trap():
    main = _function(_script(), "main")
    calls = ["check_mode", "refuse_existing_stack", "trap finish EXIT", "build_backend", "start_datastores",
             "make_workspace_mountpoint", "cut_egress", "boot_first", "boot_new_container", "check_tokenizer"]
    positions = [re.search(rf"^[ \t]+{re.escape(call)}$", main, re.M).start() for call in calls]
    assert positions == sorted(positions)


def _backend_volumes(path: Path) -> list:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["services"]["backend"]["volumes"]


def test_a_backend_booted_without_the_worker_first_gets_the_local_workspaces_mount_point():
    """docker-compose.yml's backend mounts the workspaces volume read-only, with a folder
    nested in it; on a fresh volume Docker cannot create that mount point and refuses the
    container (CI, 20 Oct: "mkdirat .../workspaces/<id>: read-only file system"). The
    smoke has no worker to create it, so a one-off backend under the core override, which
    mounts the volume writable, does."""
    base = _backend_volumes(_REPO / "docker-compose.yml")
    assert "workspaces_data:/workspaces:ro" in base
    assert any(isinstance(entry, str) and ":/workspaces/${DEFAULT_WORKSPACE_ID" in entry for entry in base)
    core = [entry for entry in _backend_volumes(_REPO / "docker-compose.core.yml")
            if isinstance(entry, dict) and entry.get("target") == "/workspaces"]
    assert core == [{"type": "volume", "source": "workspaces_data", "target": "/workspaces", "read_only": False}]

    text = _script()
    assert re.search(r'^CORE_OVERRIDE="docker-compose\.core\.yml"$', text, re.M)
    step = _function(text, "make_workspace_mountpoint")
    assert 'compose -f "$CORE_OVERRIDE" run --rm -T --no-deps --entrypoint true backend' in step
    assert "|| fail " in step


# --------------------------------------------------------------------------- the script, run on fakes


# A docker that records its calls: no stack exists, the datastores' network has one
# subnet, the one-off probes exit as FAKE_GETENT_EXIT / FAKE_CURL_EXIT say, and the
# backend never starts (smoke_boot.py then fails the first boot at once).
_FAKE_DOCKER = """#!/bin/bash
echo "docker $*" >> "$CALL_LOG"
case "$*" in
  "inspect -f {{.State.StartedAt}}"*) echo "0001-01-01T00:00:00Z" ;;
  "inspect -f {{range"*) echo "%(network)s " ;;
  "network inspect"*) echo "%(subnet)s " ;;
  *"--entrypoint getent"*) exit "${FAKE_GETENT_EXIT:-0}" ;;
  *"--entrypoint curl"*) exit "${FAKE_CURL_EXIT:-0}" ;;
esac
exit 0
""" % {"network": NETWORK, "subnet": SUBNET}
_FAKE_SUDO = '#!/bin/bash\necho "sudo $*" >> "$CALL_LOG"\nexit 0\n'


def _run_script(tmp_path: Path, **env: str) -> tuple[int, list[str], str]:
    """Run smoke-egress.sh against the fakes; (exit code, docker and sudo calls, output)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stubs = {"docker": _FAKE_DOCKER, "sudo": _FAKE_SUDO, "python3": f'#!/bin/bash\nexec "{sys.executable}" "$@"\n'}
    for name, body in stubs.items():
        (bin_dir / name).write_text(body, encoding="utf-8")
        (bin_dir / name).chmod((bin_dir / name).stat().st_mode | stat.S_IXUSR)
    log = tmp_path / "calls.log"
    log.touch()
    full_env = {
        "PATH": f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin",
        "CALL_LOG": str(log),
        "SMOKE_OUT_DIR": str(tmp_path / "out"),
        "RUNNER_TEMP": str(tmp_path),
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"),
        **env,
    }
    proc = subprocess.run(["bash", str(_SCRIPT)], env=full_env, stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=60)
    calls = [call for call in log.read_text(encoding="utf-8").splitlines() if not call.startswith("docker events")]
    return proc.returncode, calls, proc.stdout + proc.stderr


def _index(calls: list[str], fragment: str) -> int:
    matches = [i for i, call in enumerate(calls) if fragment in call]
    assert matches, f"no call contains {fragment!r}: {calls}"
    return matches[0]


needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


@needs_bash
@pytest.mark.parametrize("env", [{"SMOKE_EGRESS": ""}, {"SMOKE_EGRESS": "offline"},
                                 {"SMOKE_EGRESS": "blackholed"}])  # blackholed outside CI
def test_the_script_refuses_a_mode_it_cannot_run_before_touching_the_host(tmp_path, env):
    code, calls, out = _run_script(tmp_path, **env)
    assert code == 1, out
    assert calls == []
    assert "FAIL: SMOKE_EGRESS" in out


@needs_bash
def test_refused_egress_reaches_every_backend_container_through_the_override(tmp_path):
    code, calls, out = _run_script(tmp_path, SMOKE_EGRESS="refused", FAKE_CURL_EXIT="7")
    assert code == 1, out  # the fake backend never starts
    override = tmp_path / "smoke-egress-proxy.yml"
    environment = yaml.safe_load(override.read_text(encoding="utf-8"))["services"]["backend"]["environment"]
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        assert environment[name] == "http://127.0.0.1:9"
    assert "minio" in environment["NO_PROXY"].split(",") and environment["no_proxy"] == environment["NO_PROXY"]

    with_override = f"docker compose -f docker-compose.yml -f {override}"
    datastores = _index(calls, "up -d --wait postgres redis minio")
    mountpoint = calls.index(MOUNTPOINT)
    probe = _index(calls, f"{with_override} run --rm -T --no-deps --entrypoint curl backend")
    backend = _index(calls, f"{with_override} up -d --no-build --no-deps backend")
    assert datastores < mountpoint < probe < backend
    assert ENCODING_URL in calls[probe] and "--max-time 10" in calls[probe]
    assert calls[-1] == f"{with_override} down -v"
    assert not any(call.startswith("sudo") for call in calls), "the refused way changes no firewall"
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "Egress refused: the backend boots with no internet (PRD-262 US-033)" in summary
    assert "**FAIL**: first boot with egress refused" in summary


@needs_bash
def test_blackholed_egress_drops_the_stack_subnet_between_the_datastores_and_the_backend(tmp_path):
    code, calls, out = _run_script(tmp_path, SMOKE_EGRESS="blackholed", CI="true", FAKE_CURL_EXIT="28")
    assert code == 1, out  # the fake backend never starts
    datastores = _index(calls, "up -d --wait postgres redis minio")
    mountpoint = calls.index(MOUNTPOINT)
    drop = calls.index(f"sudo iptables -I {RULE}")
    dns = _index(calls, "--entrypoint getent backend ahosts openaipublic.blob.core.windows.net")
    probe = _index(calls, "--entrypoint curl backend")
    backend = _index(calls, "up -d --no-build --no-deps backend")
    restore = calls.index(f"sudo iptables -D {RULE}")
    assert datastores < _index(calls, "network inspect -f {{range .IPAM.Config}}") < drop
    assert datastores < mountpoint < dns and drop < dns < probe < backend < restore
    assert calls[-1] == "docker compose -f docker-compose.yml down -v"
    assert "-f " + str(tmp_path / "smoke-egress-proxy.yml") not in " ".join(calls), "no proxy in this way"


@needs_bash
def test_a_cut_that_refuses_instead_of_dropping_fails_before_the_backend_starts(tmp_path):
    code, calls, out = _run_script(tmp_path, SMOKE_EGRESS="blackholed", CI="true", FAKE_CURL_EXIT="7")
    assert code == 1, out
    assert "ended with curl exit 7, not 28: egress is not blackholed" in out
    assert not any("up -d --no-build --no-deps backend" in call for call in calls)
    assert f"sudo iptables -D {RULE}" in calls, "the rule is deleted on every exit"


@needs_bash
def test_a_blackhole_with_no_dns_is_not_the_case_the_prd_tests(tmp_path):
    code, calls, out = _run_script(tmp_path, SMOKE_EGRESS="blackholed", CI="true", FAKE_GETENT_EXIT="2")
    assert code == 1, out
    assert "the blackholed case needs DNS to answer" in out
    assert not any("--entrypoint curl" in call for call in calls)


# --------------------------------------------------------------------------- the workflow


def _workflow() -> dict:
    return yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))


def _triggers() -> dict:
    data = _workflow()
    return data.get("on", data.get(True))  # PyYAML reads the bare `on:` key as True


def test_the_workflow_runs_the_egress_smoke_once_per_way():
    job = _workflow()["jobs"]["egress-boot"]
    assert job["runs-on"] == "ubuntu-24.04"
    assert job["strategy"]["fail-fast"] is False
    assert sorted(job["strategy"]["matrix"]["egress"]) == ["blackholed", "refused"]
    run = next(step for step in job["steps"] if step.get("run") == "bash scripts/ci/smoke-egress.sh")
    assert run["env"]["SMOKE_EGRESS"] == "${{ matrix.egress }}"
    assert "continue-on-error" not in run
    upload = next(step for step in job["steps"] if "upload-artifact" in step.get("uses", ""))
    assert upload["if"] == "always()"
    assert upload["with"]["path"].rstrip("/") == run["env"]["SMOKE_OUT_DIR"].rstrip("/")
    assert upload["with"]["name"] == "smoke-egress-${{ matrix.egress }}"


def test_the_workflow_runs_when_the_egress_smoke_changes():
    triggers = _triggers()
    for event in ("push", "pull_request"):
        assert "scripts/ci/smoke-egress.sh" in triggers[event]["paths"], event
