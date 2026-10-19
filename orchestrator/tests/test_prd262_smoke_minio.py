"""PRD-262 US-041: what the smokes check about MinIO's user, and where they run it.

``scripts/ci/smoke_minio.py`` runs inside the smoke lanes against a real stack; these
tests feed it recorded shapes and a fake ``docker`` instead:

* ``server`` passes MinIO healthy as uid 65532 after minio-volume-owner exited 0, and
  fails MinIO as root (a ``user: "0:0"`` back in compose), a root process, an unhealthy
  MinIO, a one-shot that failed or is missing, and a re-own that was or was not
  expected;
* ``volume`` lists every path's owner from a throwaway container with the volume
  read-only and no network, and fails any path not owned as expected;
* ``put`` and ``get`` run the image's own mc inside the container, with the root
  credentials only in the docker CLI's environment, and ``get`` needs the bytes written;
* ``scripts/ci/smoke-minio-upgrade.sh`` seeds a root-owned volume, then upgrades through
  make up and through plain docker compose up, and checks each; smoke-images-pr runs it,
  and smoke-fresh-clone.sh checks MinIO's user on the fresh volume before the document
  round trip.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[2]
_CI = _REPO / "scripts" / "ci"
_UPGRADE = _CI / "smoke-minio-upgrade.sh"
_FRESH = _CI / "smoke-fresh-clone.sh"
_WORKFLOWS = _REPO / ".github" / "workflows"
_SPEC = importlib.util.spec_from_file_location("smoke_minio", _CI / "smoke_minio.py")
tool = importlib.util.module_from_spec(_SPEC)
sys.modules["smoke_minio"] = tool
_SPEC.loader.exec_module(tool)

COMPOSE = yaml.safe_load((_REPO / "docker-compose.yml").read_text(encoding="utf-8"))
UPGRADE = _UPGRADE.read_text(encoding="utf-8")
FRESH = _FRESH.read_text(encoding="utf-8")
HOST = tool.mc_host("minioadmin", "s3cret/pass")

HEALTHY_MINIO = {"State": {"Status": "running", "Health": {"Status": "healthy"}}, "Config": {"User": "65532"}}
EXITED_OWNER = {"State": {"Status": "exited", "ExitCode": 0}}
TOP = "PID    UID    COMMAND\n4101   65532  /usr/bin/minio server /data --console-address :9001\n"
REOWNED_LOG = f"minio-volume-owner: /data was owned by 0:0; {tool.REOWN_MARK}, the user MinIO runs as\n"


class _FakeRunner:
    """subprocess.run for ``docker``: records each call and answers from a script."""

    def __init__(self, *answers: tuple):
        self.answers = list(answers)
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append({"argv": list(argv), **kwargs})
        code, out, err = self.answers.pop(0) if self.answers else (0, b"", b"")
        return subprocess.CompletedProcess(argv, code, out, err)


# --------------------------------------------------------------------------- server


def test_docker_tops_rows_give_each_processs_uid():
    rows = tool.parse_uids(TOP + "4102   0      sh\nnot a row\n")
    assert rows == [{"pid": 4101, "uid": "65532", "args": "/usr/bin/minio server /data --console-address :9001"},
                    {"pid": 4102, "uid": "0", "args": "sh"}]


def test_minio_healthy_as_65532_after_the_one_shot_passes():
    record = tool.check_server(HEALTHY_MINIO, tool.parse_uids(TOP), EXITED_OWNER, REOWNED_LOG, "yes")
    assert record["failures"] == []
    assert record["reowned"] is True and record["minio_user"] == "65532"


@pytest.mark.parametrize("user", ["65532", "65532:65532"])
def test_the_images_user_in_either_spelling_passes(user):
    minio = {**HEALTHY_MINIO, "Config": {"User": user}}
    assert tool.minio_problems(minio, tool.parse_uids(TOP)) == []


@pytest.mark.parametrize(("minio", "top", "expected"), [
    ({**HEALTHY_MINIO, "Config": {"User": "0:0"}}, TOP, "runs as user 0:0, not 65532"),
    ({**HEALTHY_MINIO, "Config": {"User": ""}}, TOP, "runs as user root (none set)"),
    (HEALTHY_MINIO, TOP + "4102   0      /usr/bin/minio\n", "processes run as uid 0"),
    ({**HEALTHY_MINIO, "State": {"Status": "restarting"}}, TOP, "MinIO is restarting, not healthy"),
    ({**HEALTHY_MINIO, "State": {"Status": "running", "Health": {"Status": "unhealthy"}}}, TOP, "is unhealthy"),
    (HEALTHY_MINIO, "PID UID COMMAND\n", "docker top lists no MinIO process"),
    ({}, TOP, "there is no MinIO container"),
])
def test_minio_as_root_or_unhealthy_fails(minio, top, expected):
    problems = tool.minio_problems(minio, tool.parse_uids(top))
    assert any(expected in problem for problem in problems), problems


@pytest.mark.parametrize(("owner", "log", "reowned", "expected"), [
    (EXITED_OWNER, "", "yes", "does not say it re-owned the volume"),
    (EXITED_OWNER, REOWNED_LOG, "no", "re-owned a volume that was already MinIO's"),
    ({"State": {"Status": "exited", "ExitCode": 1}}, "", "any", "exited with exit code 1, not exited 0"),
    ({"State": {"Status": "running", "ExitCode": 0}}, "", "any", "is running"),
    ({}, "", "any", "there is no minio-volume-owner container"),
])
def test_a_one_shot_that_failed_or_did_not_do_what_was_expected_fails(owner, log, reowned, expected):
    problems = tool.owner_problems(owner, log, reowned)
    assert any(expected in problem for problem in problems), problems


@pytest.mark.parametrize("log", ["", REOWNED_LOG])
def test_any_accepts_a_fresh_volume_re_owned_or_not(log):
    assert tool.owner_problems(EXITED_OWNER, log, "any") == []


def test_the_server_command_writes_its_record_and_fails_on_a_root_minio(tmp_path, monkeypatch):
    root = {**HEALTHY_MINIO, "Config": {"User": "0:0"}}
    monkeypatch.setattr(tool, "read_server", lambda *_: (root, tool.parse_uids(TOP), EXITED_OWNER, ""))
    out, summary = tmp_path / "minio.json", tmp_path / "summary.md"
    code = tool.main(["server", "--container", "automatos_minio", "--owner-container", "automatos_minio_volume_owner",
                      "--reowned", "any", "--out", str(out), "--summary", str(summary)])
    assert code == 1
    assert "runs as user 0:0" in json.loads(out.read_text(encoding="utf-8"))["failures"][0]
    assert summary.read_text(encoding="utf-8").startswith(f"**{tool.TITLE}**: ")


# --------------------------------------------------------------------------- volume


def test_the_owners_come_from_a_read_only_mount_with_no_network():
    runner = _FakeRunner()
    tool.list_owners("automatos_minio_data", "redis:7-alpine@sha256:abc", runner=runner)
    argv = runner.calls[0]["argv"]
    assert argv[:2] == ["docker", "run"] and "--rm" in argv
    assert argv[argv.index("--network") + 1] == "none"
    assert argv[argv.index("-v") + 1] == "automatos_minio_data:/data:ro"
    assert argv[argv.index("--entrypoint") + 1:] == [
        "find", "redis:7-alpine@sha256:abc", "/data", "-exec", "stat", "-c", "%u:%g %n", "{}", ";"]


def test_every_path_must_have_the_owner():
    listing = "65532:65532 /data\n65532:65532 /data/us041-upgrade\n0:0 /data/us041-upgrade/old.txt\n"
    record = tool.check_volume(listing, tool.MINIO_OWNER)
    assert record["paths"] == 3
    assert record["offenders"] == ["/data/us041-upgrade/old.txt (0:0)"]
    assert "1 of 3 paths in the volume are not owned by 65532:65532" in record["failures"][0]
    assert tool.check_volume(listing.replace("0:0 ", "65532:65532 "), tool.MINIO_OWNER)["failures"] == []


def test_an_empty_listing_fails():
    assert tool.check_volume("", "0:0")["failures"] == ["the volume listing is empty, so its owners are unknown"]


def test_paths_listed_are_judged_when_find_exits_1_on_a_file_minio_removed(tmp_path, monkeypatch):
    listed = subprocess.CompletedProcess([], 1, b"65532:65532 /data\n65532:65532 /data/.minio.sys\n",
                                         b"find: /data/.minio.sys/tmp/x: No such file or directory\n")
    monkeypatch.setattr(tool, "list_owners", lambda *_: listed)
    out = tmp_path / "volume.json"
    code = tool.main(["volume", "--volume", "automatos_minio_data", "--image", "redis", "--owner", "65532:65532",
                      "--out", str(out), "--summary", str(tmp_path / "summary.md")])
    record = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0 and record["paths"] == 2 and record["listing_exit_code"] == 1
    assert "No such file" in record["listing_errors"]


# --------------------------------------------------------------------------- objects


def test_the_credentials_are_the_containers_own_or_minios_defaults():
    env = ["PATH=/usr/bin", "MINIO_ROOT_USER=admin", "MINIO_ROOT_PASSWORD=pa=ss"]
    assert tool.root_credentials(env) == ("admin", "pa=ss")
    assert tool.root_credentials(["PATH=/usr/bin"]) == ("minioadmin", "minioadmin")
    assert HOST == "http://minioadmin:s3cret%2Fpass@127.0.0.1:9000"


def test_put_makes_the_bucket_then_pipes_the_text_through_the_containers_mc():
    runner = _FakeRunner()
    record = tool.put_object("automatos_minio", HOST, "us041", "old.txt", "hello", runner=runner)
    assert record["failures"] == [] and record["bytes"] == 5
    make, pipe = runner.calls
    assert make["argv"] == ["docker", "exec", "-e", tool.MC_HOST_ENV, "automatos_minio", "mc", "--quiet",
                            "mb", "--ignore-existing", "smoke/us041"]
    assert pipe["argv"] == ["docker", "exec", "-i", "-e", tool.MC_HOST_ENV, "automatos_minio", "mc", "--quiet",
                            "pipe", "smoke/us041/old.txt"]
    assert pipe["input"] == b"hello"
    for call in runner.calls:
        assert call["env"][tool.MC_HOST_ENV] == HOST
        assert not any("s3cret" in part for part in call["argv"]), "credentials never on a command line"


def test_put_stops_at_a_bucket_it_could_not_make():
    runner = _FakeRunner((1, b"", b"mc: <ERROR> Unable to make bucket: Access Denied."))
    record = tool.put_object("automatos_minio", HOST, "us041", "new.txt", "hello", runner=runner)
    assert len(runner.calls) == 1
    assert record["failures"] == ["`mc mb` exited 1: mc: <ERROR> Unable to make bucket: Access Denied."]


@pytest.mark.parametrize(("answer", "failures"), [
    ((0, b"hello", b""), []),
    ((0, b"hellp", b""), ["us041/old.txt came back as 5 bytes that are not the 5 written"]),
    ((1, b"", b"mc: <ERROR> Object does not exist."), ["`mc cat` exited 1: mc: <ERROR> Object does not exist."]),
])
def test_get_needs_the_bytes_written(answer, failures):
    runner = _FakeRunner(answer)
    record = tool.get_object("automatos_minio", HOST, "us041", "old.txt", "hello", runner=runner)
    assert record["failures"] == failures
    assert runner.calls[0]["argv"][-2:] == ["cat", "smoke/us041/old.txt"]


# --------------------------------------------------------------------------- the upgrade smoke


def _shell_value(text: str, name: str) -> str:
    return re.search(rf'^{name}="([^"]*)"$', text, re.M).group(1)


def _shell_function(text: str, name: str) -> str:
    match = re.search(rf"^{name}\(\) \{{\n(.*?)^\}}", text, re.M | re.S)
    assert match, f"no {name}() function"
    return match.group(1)


def test_the_upgrade_smoke_names_what_compose_defines():
    services = COMPOSE["services"]
    assert _shell_value(UPGRADE, "VOLUME") == COMPOSE["volumes"]["minio_data"]["name"]
    assert _shell_value(UPGRADE, "MINIO_CONTAINER") == services["minio"]["container_name"]
    assert _shell_value(UPGRADE, "OWNER_CONTAINER") == services["minio-volume-owner"]["container_name"]
    assert _shell_value(UPGRADE, "OWNER_SERVICE") in services
    assert _shell_value(UPGRADE, "MINIO_OWNER") == tool.MINIO_OWNER
    assert _shell_value(UPGRADE, "ROOT_OWNER") == "0:0"
    assert _shell_value(UPGRADE, "MINIO_TOOL") == "scripts/ci/smoke_minio.py"
    assert re.match(r"\^automatos_", re.search(r"^STACK_NAME_PATTERN='([^']*)'$", UPGRADE, re.M).group(1))


def test_the_upgrade_smoke_runs_make_up_then_plain_compose_each_from_a_root_volume():
    main = _shell_function(UPGRADE, "main")
    order = ["refuse_existing_stack", "trap finish EXIT", "read_owner_image", "upgrade_with_make_up", "teardown",
             "upgrade_with_compose_up", 'outcome="pass"']
    assert [main.index(step) for step in order] == sorted(main.index(step) for step in order)
    make, plain = _shell_function(UPGRADE, "upgrade_with_make_up"), _shell_function(UPGRADE, "upgrade_with_compose_up")
    for body, command in ((make, "make up >"), (plain, "docker compose up -d >")):
        assert body.index("seed_root_volume") < body.index(command) < body.index("check_upgraded"), command
    assert "make " not in plain, "the second upgrade is plain compose, no make"


def test_the_seed_is_a_volume_root_wrote_and_checked_as_roots():
    seed = _shell_function(UPGRADE, "seed_root_volume")
    assert 'compose_any run -d --no-deps --user 0:0 --name "$SEED_CONTAINER" minio' in seed
    steps = ["wait_seed_healthy", "put --container \"$SEED_CONTAINER\"", 'docker stop -t 30 "$SEED_CONTAINER"',
             "'chown -R 0:0 /data && chmod 755 /data'", '--owner "$ROOT_OWNER"']
    assert [seed.index(step) for step in steps] == sorted(seed.index(step) for step in steps)


def test_each_upgrade_needs_the_backend_minio_as_65532_the_volume_and_both_objects():
    checks = _shell_function(UPGRADE, "check_upgraded")
    steps = ["ready=${API_BASE}/health/ready", "server --container", "--reowned yes", '--owner "$MINIO_OWNER"',
             'get --container "$MINIO_CONTAINER" --bucket "$BUCKET" --key "$OLD_KEY"',
             'put --container "$MINIO_CONTAINER" --bucket "$BUCKET" --key "$NEW_KEY"',
             'get --container "$MINIO_CONTAINER" --bucket "$BUCKET" --key "$NEW_KEY"']
    assert [checks.index(step) for step in steps] == sorted(checks.index(step) for step in steps)
    assert checks.count("|| fail ") == 6


_FAKE_DOCKER = """#!/bin/bash
echo "docker $*" >> "$CALL_LOG"
case "$*" in
  "ps -a --format"*) [ -z "${FAKE_CONTAINERS:-}" ] || printf '%s\\n' $FAKE_CONTAINERS ;;
  "volume ls --format"*) [ -z "${FAKE_VOLUMES:-}" ] || printf '%s\\n' $FAKE_VOLUMES ;;
esac
exit 0
"""


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_the_upgrade_smoke_refuses_a_host_with_its_own_automatos_volume(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "docker").write_text(_FAKE_DOCKER, encoding="utf-8")
    (bin_dir / "docker").chmod((bin_dir / "docker").stat().st_mode | stat.S_IXUSR)
    log = tmp_path / "calls.log"
    log.touch()
    env = {"PATH": f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin", "CALL_LOG": str(log),
           "RUNNER_TEMP": str(tmp_path), "SMOKE_OUT_DIR": str(tmp_path / "out"), "AUTOMATOS_IMAGE_TAG": "pr-0d9f0ac",
           "FAKE_VOLUMES": "automatos_minio_data"}
    proc = subprocess.run(["bash", str(_UPGRADE)], env=env, stdin=subprocess.DEVNULL, capture_output=True,
                          text=True, timeout=60)
    assert proc.returncode == 1
    assert "already has Automatos containers or volumes (automatos_minio_data)" in proc.stderr
    assert log.read_text(encoding="utf-8").splitlines() == [
        "docker ps -a --format {{.Names}}", "docker volume ls --format {{.Name}}"], "it seeds and tears down nothing"


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_the_upgrade_smoke_parses():
    assert subprocess.run(["bash", "-n", str(_UPGRADE)], capture_output=True, timeout=30).returncode == 0


# --------------------------------------------------------------------------- the fresh volume and the workflows


def test_the_fresh_clone_smoke_checks_minios_user_before_the_document_round_trip():
    assert _shell_value(FRESH, "MINIO_TOOL") == "scripts/ci/smoke_minio.py"
    assert _shell_value(FRESH, "MINIO_CONTAINER") == COMPOSE["services"]["minio"]["container_name"]
    assert _shell_value(FRESH, "MINIO_OWNER_CONTAINER") == COMPOSE["services"]["minio-volume-owner"]["container_name"]
    step = _shell_function(FRESH, "check_minio_user")
    assert "--reowned any" in step and "|| fail " in step, "a fresh volume is root's until the one-shot runs"
    main = _shell_function(FRESH, "main")
    order = [re.search(rf"^[ \t]+{call}$", main, re.M).start()
             for call in ("check_images", "check_minio_user", "check_document_storage")]
    assert order == sorted(order)


def _workflow(name: str) -> dict:
    return yaml.safe_load((_WORKFLOWS / name).read_text(encoding="utf-8"))


def _paths(workflow: dict, event: str) -> list:
    return workflow.get("on", workflow.get(True))[event]["paths"]  # PyYAML reads `on` as True


def test_smoke_images_pr_runs_the_upgrade_on_the_images_it_built():
    workflow = _workflow("smoke-images.yml")
    steps = workflow["jobs"]["smoke-images-pr"]["steps"]
    runs = [step.get("run") for step in steps]
    core = runs.index("bash scripts/ci/smoke-core-stack.sh")
    upgrade = runs.index("bash scripts/ci/smoke-minio-upgrade.sh")
    upload = next(i for i, step in enumerate(steps) if "upload-artifact" in step.get("uses", ""))
    assert core < upgrade < upload
    env = steps[upgrade]["env"]
    assert env["AUTOMATOS_IMAGE_TAG"] == "${{ steps.target.outputs.tag }}"
    assert not {"POSTGRES_PASSWORD", "REDIS_PASSWORD", "API_KEY"} & set(env), "make env writes them"
    assert env["SMOKE_OUT_DIR"].startswith(steps[upload]["with"]["path"].rstrip("/"))
    assert {"scripts/ci/smoke-minio-upgrade.sh", "scripts/ci/smoke_minio.py", "docker-compose*.yml"} <= set(
        _paths(workflow, "pull_request"))


def test_the_fresh_clone_workflow_watches_the_tool():
    workflow = _workflow("smoke-fresh-clone.yml")
    for event in ("push", "pull_request"):
        assert "scripts/ci/smoke_minio.py" in _paths(workflow, event), event
