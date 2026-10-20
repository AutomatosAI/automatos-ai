"""PRD-262 US-041: MinIO runs as its own user.

``docker-compose.yml`` no longer runs MinIO as root (``user: "0:0"``): the Chainguard
image runs as its own user, uid 65532. A one-shot service, ``minio-volume-owner``, runs
before it on the stack's own digest-pinned redis image, as root, with the minio_data
volume. It re-owns the volume to 65532:65532 only when it is not MinIO's already, logs
that it did, and exits 0; MinIO waits for it (``service_completed_successfully``). So
``make up``, plain ``docker compose up`` and every documented compose command repair a
volume the official image wrote as root, and the backend, which waits for
``minio: service_healthy``, still starts after a plain-compose upgrade.

This suite checks, from the files:

* the two services in docker-compose.yml, and that no override or profile leaves the
  one-shot out;
* the one-shot's script, run by ``sh`` with stand-ins for ``stat`` and ``chown``: it
  re-owns a root or redis-owned volume, leaves MinIO's alone, and fails loudly when the
  chown fails;
* ``docker compose config`` (where the docker CLI is installed; CI's runners always have
  it, so there a missing CLI fails instead of skipping): MinIO has no user in the
  resolved model, and the one-shot is in the stack with no profile, with every profile
  the guides name, and with the core override.

No Docker daemon, no network. scripts/ci/smoke-minio-upgrade.sh proves it on a real stack.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[2]
_SECRETS = {"POSTGRES_PASSWORD": "test", "REDIS_PASSWORD": "test", "API_KEY": "test"}
_SPEC = importlib.util.spec_from_file_location("smoke_minio", _REPO / "scripts" / "ci" / "smoke_minio.py")
tool = importlib.util.module_from_spec(_SPEC)
sys.modules["smoke_minio"] = tool
_SPEC.loader.exec_module(tool)


def _compose(name: str) -> dict:
    return yaml.safe_load((_REPO / name).read_text(encoding="utf-8"))


COMPOSE = _compose("docker-compose.yml")
SERVICES = COMPOSE["services"]
OWNER_NAME = "minio-volume-owner"
OWNER = SERVICES[OWNER_NAME]
MINIO = SERVICES["minio"]
RAW_SCRIPT = OWNER["entrypoint"][2]
# What sh runs: compose turns each $$ into a literal $.
SCRIPT = RAW_SCRIPT.replace("$$", "$")
# The profiles the guides start the stack with (QUICKSTART.md, self-hosting.md).
GUIDE_PROFILES = ("all", "media", "memory", "search")


# --------------------------------------------------------------------------- compose


def test_minio_runs_as_the_images_own_user():
    assert "user" not in MINIO, "MinIO runs as the Chainguard image's user, uid 65532, never root"
    assert MINIO["image"].startswith("cgr.dev/chainguard/minio:latest@sha256:")
    assert MINIO["volumes"] == ["minio_data:/data"]


def test_minio_waits_for_the_one_shot_to_finish():
    assert MINIO["depends_on"] == {OWNER_NAME: {"condition": "service_completed_successfully"}}


def test_the_backend_still_waits_for_a_healthy_minio():
    """Why the repair lives in compose: a plain-compose upgrade must still start the backend."""
    assert SERVICES["backend"]["depends_on"]["minio"] == {"condition": "service_healthy"}


def test_the_one_shot_runs_as_root_on_the_stacks_own_redis_image():
    assert OWNER["image"] == SERVICES["redis"]["image"], (
        "the one-shot reuses the redis image the stack already pulled; Dependabot moves both digests")
    assert "@sha256:" in OWNER["image"]
    assert OWNER["user"] == "0:0"
    assert OWNER["volumes"] == MINIO["volumes"] == ["minio_data:/data"]
    assert OWNER["entrypoint"][:2] == ["/bin/sh", "-c"] and len(OWNER["entrypoint"]) == 3
    assert "command" not in OWNER, "the entrypoint holds the whole script; redis-server never runs"


def test_the_one_shot_has_no_profile_and_only_what_chown_needs():
    assert "profiles" not in OWNER, "plain compose and every --profile must run it"
    assert OWNER["restart"] == "no", "it exits once the volume is MinIO's"
    assert OWNER["network_mode"] == "none"
    assert OWNER["cap_drop"] == ["ALL"]
    assert sorted(OWNER["cap_add"]) == ["CHOWN", "DAC_OVERRIDE"]
    assert "ports" not in OWNER and "healthcheck" not in OWNER
    assert OWNER["container_name"] == "automatos_minio_volume_owner", "the smokes' ^automatos_ refusal sees it"


def test_no_override_touches_the_one_shot_or_minios_user():
    for name in ("docker-compose.core.yml", "docker-compose.dev.yml"):
        services = _compose(name).get("services") or {}
        assert OWNER_NAME not in services, name
        assert "user" not in (services.get("minio") or {}), name


def test_compose_reads_every_dollar_in_the_script_as_a_literal_one():
    """A lone $ is a compose variable: `$owner` would reach sh as an empty string."""
    assert "$" not in RAW_SCRIPT.replace("$$", ""), "write each $ in the one-shot's script as $$"


def test_the_one_shot_logs_the_line_the_smokes_look_for():
    assert tool.REOWN_MARK in SCRIPT
    assert f'= "{tool.MINIO_OWNER}"' in SCRIPT and f"chown -R {tool.MINIO_OWNER} /data" in SCRIPT


# --------------------------------------------------------------------------- the script


_STAT = """#!/bin/sh
echo "stat $*" >> "$CALL_LOG"
[ -z "${STAT_FAILS:-}" ] || exit 1
echo "$FAKE_OWNER"
"""
_CHOWN = """#!/bin/sh
echo "chown $*" >> "$CALL_LOG"
[ -z "${CHOWN_FAILS:-}" ] || exit 1
"""


def _run_script(tmp_path: Path, owner: str, **fake: str) -> tuple:
    """The one-shot's script under sh, with stat answering ``owner``; (process, calls)."""
    sh = shutil.which("sh")
    if sh is None:
        pytest.skip("needs sh")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("stat", _STAT), ("chown", _CHOWN)):
        (bin_dir / name).write_text(body, encoding="utf-8")
        (bin_dir / name).chmod(0o755)
    log = tmp_path / "calls.log"
    log.touch()
    env = {"PATH": f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin", "CALL_LOG": str(log), "FAKE_OWNER": owner,
           **fake}
    proc = subprocess.run([sh, "-c", SCRIPT], env=env, capture_output=True, text=True, timeout=30)
    return proc, log.read_text(encoding="utf-8").splitlines()


@pytest.mark.parametrize("owner", ["0:0", "999:1000", "65532:0"])
def test_a_volume_not_minios_is_re_owned_once_and_the_log_says_so(tmp_path, owner):
    proc, calls = _run_script(tmp_path, owner)
    assert proc.returncode == 0, proc.stderr
    assert calls == ["stat -c %u:%g /data", f"chown -R {tool.MINIO_OWNER} /data"]
    assert f"/data was owned by {owner}; {tool.REOWN_MARK}" in proc.stdout


def test_a_volume_already_minios_is_left_alone_and_quietly(tmp_path):
    proc, calls = _run_script(tmp_path, tool.MINIO_OWNER)
    assert proc.returncode == 0, proc.stderr
    assert calls == ["stat -c %u:%g /data"], "no chown on a volume that is MinIO's"
    assert proc.stdout == ""


def test_a_failed_chown_fails_the_one_shot_so_minio_never_starts(tmp_path):
    proc, calls = _run_script(tmp_path, "0:0", CHOWN_FAILS="1")
    assert proc.returncode == 1
    assert calls[-1] == f"chown -R {tool.MINIO_OWNER} /data"
    assert f"could not re-own /data (owned by 0:0) to {tool.MINIO_OWNER}" in proc.stderr
    assert tool.REOWN_MARK not in proc.stdout


def test_a_failed_stat_fails_the_one_shot_without_a_chown(tmp_path):
    proc, calls = _run_script(tmp_path, "", STAT_FAILS="1")
    assert proc.returncode != 0
    assert calls == ["stat -c %u:%g /data"]


@pytest.mark.skipif(shutil.which("sh") is None, reason="needs sh")
def test_the_script_parses():
    assert subprocess.run(["sh", "-n", "-c", SCRIPT], capture_output=True, timeout=30).returncode == 0


# --------------------------------------------------------------------------- docker compose config


def _config(*args: str, files: tuple = ("docker-compose.yml",)) -> dict:
    """``docker compose config --format json``'s services, with only the three secrets set."""
    docker = shutil.which("docker")
    if docker is None:
        if os.environ.get("CI") == "true":
            pytest.fail("the docker CLI is missing on this CI runner: these checks need `docker compose config`")
        pytest.skip("needs the docker CLI, which CI's runners have")
    environ = {key: value for key, value in os.environ.items()
               if not key.startswith(("AUTOMATOS_", "NEXT_PUBLIC_", "COMPOSE_", "POSTGRES_", "REDIS_", "API_",
                                      "MINIO_"))}
    with tempfile.NamedTemporaryFile("w", suffix=".env") as empty:  # never a checkout's own .env
        command = [docker, "compose", "--project-directory", str(_REPO), "--env-file", empty.name]
        for name in files:
            command += ["-f", str(_REPO / name)]
        proc = subprocess.run([*command, *args, "config", "--format", "json"], cwd=str(_REPO),
                              capture_output=True, text=True, timeout=120, env={**environ, **_SECRETS})
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)["services"]


def test_the_resolved_stack_runs_minio_as_its_image_user_after_the_one_shot():
    services = _config()
    assert {"minio", OWNER_NAME} <= set(services)
    assert "user" not in services["minio"]
    assert services["minio"]["depends_on"][OWNER_NAME]["condition"] == "service_completed_successfully"
    assert services[OWNER_NAME]["user"] == "0:0"
    assert services[OWNER_NAME]["image"] == services["redis"]["image"]


@pytest.mark.parametrize("profile", GUIDE_PROFILES)
def test_every_profile_the_guides_name_runs_the_one_shot(profile):
    assert OWNER_NAME in _config("--profile", profile)


def test_the_core_stack_runs_the_one_shot():
    """make up CORE=1 and the §1b command (PRD-262 US-040)."""
    services = _config(files=("docker-compose.yml", "docker-compose.core.yml"))
    assert OWNER_NAME in services and "user" not in services["minio"]
