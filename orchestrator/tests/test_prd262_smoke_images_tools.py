"""PRD-262 US-003: the published-image smoke's measurements and assertions.

``scripts/ci/smoke_images.py`` runs inside smoke-images.yml against a real stack; these
tests feed it recorded shapes instead of Docker:

* ``target`` picks ``sha-<7>`` after a main publish, ``pr-<7>`` on a pull request, and a
  manual run's tag, refusing anything that is not a tag, and the tag to pin for
  ``make up`` (none after a push: make up picks it, PRD-262 US-036);
* ``base`` finds the merge base's published ``sha-<7>``, or its nearest published ancestor;
* ``pull`` times the pull (``make up``) and counts the bytes the default-route interface
  received; with ``--log`` it keeps the output and counts registry rate limits (429);
* ``workers`` reads ``docker top``: the production command runs 4 uvicorn workers
  without ``--reload``; the development stage's ``--reload`` and a wrong count fail;
* ``images`` passes a pulled image whose ``RepoDigests`` holds the GHCR index digest,
  and fails a locally built one, a ``local-*`` tag, a compose-built image, an app image
  also present under another tag, and a missing or wrong revision label; a source build
  (mode ``build``) must be tagged ``local-*``;
* ``report`` works out the first run (pull start to ``/health/ready`` and ``:3000``)
  and writes the summary section, also for a run that stopped early;
* ``image_registry.resolve`` reads the digest a tag points at.
"""
from __future__ import annotations

import hashlib
import http.client
import importlib.util
import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "prd262_edge_images.json"
NS = 1_000_000_000
T0 = 1_792_100_000  # 2026-10-15 21:33:20 UTC
SHA = "0d9f0ac594c3783e62a30e6c5ce75164028850db"  # the fixture images' revision
TAG = "sha-0d9f0ac"
INDEX = {"backend": "sha256:" + "a1" * 32, "frontend": "sha256:" + "b2" * 32, "workspace-worker": "sha256:" + "c3" * 32}
NAMES = {"backend": "api", "frontend": "frontend", "workspace-worker": "workspace-worker"}
CONTAINERS = {"backend": "automatos_backend", "frontend": "automatos_frontend",
              "workspace-worker": "automatos_workspace_worker"}


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / "scripts" / "ci" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # before exec, as the scripts register their siblings
    spec.loader.exec_module(module)
    return module


tool = _load("smoke_images")
section = _load("smoke_images_report")
EDGE = json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _ref(service: str, tag: str = TAG) -> str:
    return f"ghcr.io/automatosai/automatos-{NAMES[service]}:{tag}"


# --------------------------------------------------------------------------- target


def test_a_main_publish_boots_its_sha_tag_and_expects_its_commit():
    assert tool.target("workflow_run", SHA) == {"tag": TAG, "revision": SHA}


def test_a_pull_request_boots_pr_and_its_seven_characters():
    assert tool.target("pull_request", SHA) == {"tag": "pr-0d9f0ac", "revision": SHA}


@pytest.mark.parametrize("tag,revision", [("sha-0d9f0ac", "0d9f0ac"), ("edge", ""), ("weekly-20261019", "")])
def test_a_manual_run_boots_the_tag_it_was_given(tag, revision):
    assert tool.target("workflow_dispatch", tag=tag) == {"tag": tag, "revision": revision}


@pytest.mark.parametrize("event,sha,tag", [
    ("workflow_dispatch", "", "edge; rm -rf /"),
    ("workflow_dispatch", "", ""),
    ("workflow_run", "0d9f0ac", ""),
    ("push", SHA, ""),
])
def test_target_refuses_what_it_cannot_boot(event, sha, tag):
    with pytest.raises(ValueError):
        tool.target(event, sha, tag)


def test_the_target_command_writes_the_job_outputs(tmp_path, capsys):
    outputs = tmp_path / "out"
    assert tool.main(["target", "--event", "workflow_run", "--sha", SHA, "--outputs", str(outputs)]) == 0
    assert outputs.read_text().splitlines() == [f"tag={TAG}", f"revision={SHA}", "pin="]
    assert f"tag={TAG}" in capsys.readouterr().out


@pytest.mark.parametrize("event,trigger,tag,pin", [
    ("workflow_run", "push", TAG, ""),  # make up picks the commit's own images
    ("workflow_run", "workflow_dispatch", TAG, ""),
    ("workflow_run", "schedule", "weekly-20261019", "weekly-20261019"),  # make up would pick the sha, not the rebuild
    ("workflow_dispatch", "", "edge", "edge"),
    ("pull_request", "", "pr-0d9f0ac", "pr-0d9f0ac"),
])
def test_make_up_picks_the_tag_only_after_a_push(event, trigger, tag, pin):
    assert tool.pin_for(event, trigger, tag) == pin


# --------------------------------------------------------------------------- base


def _published_sha_tags(monkeypatch, published: set):
    def resolve(ref):
        if ref.rsplit(":", 1)[1] in published:
            return "sha256:" + "a" * 64
        raise urllib.error.HTTPError(ref, 404, "manifest unknown", {}, io.BytesIO(b""))

    monkeypatch.setattr(tool.registry, "resolve", resolve)


OLDER = "1" * 40


def test_the_merge_base_runs_its_own_sha_when_ci_published_it(monkeypatch):
    _published_sha_tags(monkeypatch, {TAG})
    assert tool.published_base([SHA, OLDER]) == {"tag": TAG, "revision": SHA}


def test_a_merge_base_without_images_falls_back_to_its_nearest_published_ancestor(monkeypatch, tmp_path, capsys):
    _published_sha_tags(monkeypatch, {"sha-1111111"})
    candidates, outputs = tmp_path / "candidates", tmp_path / "out"
    candidates.write_text(f"{SHA}\n{OLDER}\n")
    assert tool.main(["base", "--merge-base", SHA, "--candidates", str(candidates), "--outputs", str(outputs)]) == 0
    assert outputs.read_text().splitlines() == ["tag=sha-1111111", f"revision={OLDER}"]
    assert "::warning title=smoke-images::CI published no images for the merge base 0d9f0ac594c3" in capsys.readouterr().out


def test_no_published_candidate_fails_and_an_unreadable_registry_raises(monkeypatch, tmp_path):
    _published_sha_tags(monkeypatch, set())
    candidates = tmp_path / "candidates"
    candidates.write_text(f"{OLDER}\n")
    assert tool.main(["base", "--merge-base", SHA, "--candidates", str(candidates)]) == 1

    def unreachable(ref):
        raise urllib.error.HTTPError(ref, 503, "unavailable", {}, io.BytesIO(b""))

    monkeypatch.setattr(tool.registry, "resolve", unreachable)
    with pytest.raises(urllib.error.HTTPError):
        tool.published_base([SHA])


# --------------------------------------------------------------------------- pull

ROUTE = """Iface\tDestination\tGateway \tFlags\tRefCnt\tUse\tMetric\tMask\t\tMTU\tWindow\tIRTT
docker0\t000011AC\t00000000\t0001\t0\t0\t0\t0000FFFF\t0\t0\t0
eth0\t00000000\t0100000A\t0003\t0\t0\t100\t00000000\t0\t0\t0
eth0\t0000000A\t00000000\t0001\t0\t0\t100\t00FFFFFF\t0\t0\t0
"""


def _dev(eth0_rx: int) -> str:
    return ("Inter-|   Receive                            |  Transmit\n"
            " face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets\n"
            "    lo:  123456     100    0    0    0     0          0         0   123456     100\n"
            f"  eth0: {eth0_rx}  9000    0    0    0     0          0         0   555 10\n"
            "docker0: 999999999  10    0    0    0     0          0         0   1 1\n")


def test_the_default_route_names_the_interface_the_pull_comes_in_on():
    assert tool.default_interface(ROUTE) == "eth0"
    assert tool.received_bytes(_dev(1_000), "eth0") == 1_000
    with pytest.raises(LookupError):
        tool.default_interface(ROUTE.splitlines()[0])
    with pytest.raises(LookupError):
        tool.received_bytes(_dev(1), "eth1")


def test_the_pull_is_timed_and_its_bytes_counted():
    tables = iter([ROUTE, _dev(5_000_000), _dev(1_335_000_000)])
    clock = iter([T0 * NS, (T0 + 41) * NS + 250_000_000])
    ran = []
    code, record = tool.run_pull(["docker", "compose", "pull"], images_before=2,
                                 reader=lambda path: next(tables), clock=lambda: next(clock),
                                 runner=lambda command: ran.append(command) or 0)
    assert (code, ran) == (0, [["docker", "compose", "pull"]])
    assert record == {"started_ns": T0 * NS, "seconds": 41.25, "received_bytes": 1_330_000_000,
                      "interface": "eth0", "images_before": 2, "exit_code": 0}


def test_a_failed_pull_passes_its_exit_code_on():
    tables = iter([ROUTE, _dev(0), _dev(10)])
    code, record = tool.run_pull(["false"], 0, reader=lambda path: next(tables), clock=lambda: T0 * NS,
                                 runner=lambda command: 18)
    assert code == 18 and record["exit_code"] == 18


def test_the_pull_command_runs_what_follows_the_separator(monkeypatch, tmp_path):
    seen = {}

    def fake_run_pull(command, images_before):
        seen.update(command=command, images_before=images_before)
        return 0, {"seconds": 1.0, "received_bytes": 2_000_000, "interface": "eth0"}

    monkeypatch.setattr(tool, "run_pull", fake_run_pull)
    out = tmp_path / "pull.json"
    args = ["pull", "--out", str(out), "--images-before", "3", "--", "docker", "compose", "-f", "a.yml", "pull", "--quiet"]
    assert tool.main(args) == 0
    assert seen == {"command": ["docker", "compose", "-f", "a.yml", "pull", "--quiet"], "images_before": 3}
    assert json.loads(out.read_text())["received_bytes"] == 2_000_000


LIMITED = "Error response from daemon: toomanyrequests: You have reached your pull rate limit."


def test_the_pull_keeps_make_ups_output_and_records_a_docker_hub_429(monkeypatch, tmp_path, capsys):
    def fake_run_pull(command, images_before, runner):
        return runner(command), {"seconds": 1.0, "received_bytes": 5, "interface": "eth0"}

    monkeypatch.setattr(tool, "run_pull", fake_run_pull)
    out, log = tmp_path / "pull.json", tmp_path / "make-up.log"
    script = f"print('pulling redis'); print({LIMITED!r}); raise SystemExit(3)"
    assert tool.main(["pull", "--out", str(out), "--log", str(log), "--", sys.executable, "-c", script]) == 3
    assert log.read_text().splitlines() == ["pulling redis", LIMITED]
    record = json.loads(out.read_text())
    assert (record["rate_limited"], record["rate_limit_lines"]) == (1, [LIMITED])
    shown = capsys.readouterr().out
    assert "pulling redis" in shown, "make up's output still shows in the job log"
    assert f"::warning title=smoke-images::a registry refused a pull over its rate limit: {LIMITED}" in shown


def test_only_rate_limit_lines_count():
    text = "Pulled\n429 Too Many Requests\nredis Pulled\nTOOMANYREQUESTS: slow down\n"
    assert tool.rate_limited(text) == ["429 Too Many Requests", "TOOMANYREQUESTS: slow down"]


# --------------------------------------------------------------------------- workers

SPAWN = "/usr/local/bin/python3.11 -c from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=5, pipe_handle={}) --multiprocessing-fork"
PRODUCTION_TOP = "\n".join([
    "PID                 PPID                COMMAND",
    "4101                4080                /bin/sh /usr/local/bin/docker-entrypoint.sh /bin/sh -c sh -c "
    "\"alembic upgrade heads && uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 4\"",
    "4190                4101                sh -c alembic upgrade heads && uvicorn main:app --host 0.0.0.0 "
    "--port ${PORT:-8000} --workers 4",
    "4191                4101                /bin/sh /usr/local/bin/docker-entrypoint.sh /bin/sh -c sh -c x",
    "4250                4190                /usr/local/bin/python3.11 /usr/local/bin/uvicorn main:app "
    "--host 0.0.0.0 --port 8000 --workers 4",
    "4251                4250                /usr/local/bin/python3.11 -c from multiprocessing.resource_tracker "
    "import main;main(4)",
] + [f"{pid}                4250                {SPAWN.format(pid)}" for pid in (4252, 4253, 4254, 4255)])
DEVELOPMENT_TOP = "\n".join([
    "PID                 PPID                COMMAND",
    "301                 280                 /usr/local/bin/python3.11 /usr/local/bin/uvicorn main:app --host "
    "0.0.0.0 --port 8000 --reload --timeout-graceful-shutdown 5",
    f"302                 301                 {SPAWN.format(9)}",
])


def test_the_production_command_runs_four_workers_without_reload():
    found = tool.backend_processes(tool.parse_top(PRODUCTION_TOP), expected_workers=4)
    assert found["failures"] == []
    assert (found["workers"], found["reload"]) == (4, False)
    assert found["command"].endswith("uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4")


def test_the_development_stage_fails_on_reload():
    found = tool.backend_processes(tool.parse_top(DEVELOPMENT_TOP), expected_workers=4)
    assert found["reload"] is True and found["workers"] == 1
    assert any("--reload" in failure for failure in found["failures"])
    assert any("1 uvicorn worker(s) serve, expected 4" in failure for failure in found["failures"])


def test_a_worker_count_that_does_not_match_fails():
    rows = [row for row in tool.parse_top(PRODUCTION_TOP) if row["pid"] != 4255]
    assert tool.backend_processes(rows, 4)["failures"] == ["3 uvicorn worker(s) serve, expected 4"]


def test_one_uvicorn_process_with_no_children_is_one_worker():
    rows = [{"pid": 7, "ppid": 1, "args": "/usr/local/bin/python /usr/local/bin/uvicorn main:app --workers 1"}]
    assert tool.backend_processes(rows, 1)["failures"] == []


def test_no_uvicorn_process_fails():
    rows = [{"pid": 7, "ppid": 1, "args": "sh -c alembic upgrade heads && uvicorn main:app"}]
    found = tool.backend_processes(rows, 4)
    assert found["workers"] is None and found["failures"] == ["expected one uvicorn process, found 0"]


def test_the_workers_command_reads_docker_top_with_wide_output(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(tool, "docker_text", lambda *args: calls.append(args) or PRODUCTION_TOP)
    out = tmp_path / "processes.json"
    assert tool.main(["workers", "--container", "automatos_backend", "--expect", "4", "--out", str(out)]) == 0
    assert calls == [("top", "automatos_backend", "-eo", "pid,ppid,args", "-ww")]
    assert json.loads(out.read_text())["workers"] == 4
    assert tool.main(["workers", "--container", "automatos_backend", "--expect", "2", "--out", str(out)]) == 1


# --------------------------------------------------------------------------- images


def _config(tag: str = TAG) -> dict:
    return {"name": "automatos-ai",
            "services": {service: {"image": _ref(service, tag), "container_name": CONTAINERS[service]}
                         for service in tool.APP_SERVICES}}


def _docker(monkeypatch, *, digests=None, revision=SHA, listing=None, tag=TAG):
    """Fake ``docker inspect``, ``docker image inspect`` and ``docker image ls``."""
    by_container = {CONTAINERS[service]: service for service in tool.APP_SERVICES}

    def docker_json(*args):
        if args[0] == "inspect":
            service = by_container[args[1]]
            return [{"Config": {"Image": _ref(service, tag)}, "Image": f"sha256:id-{service}"}]
        service = args[2].removeprefix("sha256:id-")
        held = digests(service) if digests else [f"ghcr.io/automatosai/automatos-{NAMES[service]}@{INDEX[service]}"]
        labels = {tool.REVISION_LABEL: revision} if revision else {}
        return [{"RepoDigests": held, "Config": {"Labels": labels}}]

    rows = listing if listing is not None else (
        [{"Repository": _ref(s, tag).rsplit(":", 1)[0], "Tag": tag, "ID": s, "Size": "1GB"} for s in tool.APP_SERVICES]
        + [{"Repository": "pgvector/pgvector", "Tag": "pg16"}, {"Repository": "redis", "Tag": "7-alpine"}])
    monkeypatch.setattr(tool, "docker_json", docker_json)
    monkeypatch.setattr(tool, "docker_lines", lambda *args: rows)


def _registry(monkeypatch, resolved=None):
    def resolve(ref):
        service = next(s for s in tool.APP_SERVICES if f"automatos-{NAMES[s]}:" in ref)
        return (resolved or INDEX)[service]

    def image(ref, arch):
        name = ref.split("/automatos-", 1)[1].rsplit(":", 1)[0]
        return EDGE[name]["manifest"], EDGE[name]["config"], None

    monkeypatch.setattr(tool.registry, "resolve", resolve)
    monkeypatch.setattr(tool.registry, "image", image)


def test_pulled_images_pass_and_their_layers_are_counted_once(monkeypatch):
    _docker(monkeypatch)
    _registry(monkeypatch)
    found = tool.check_images(_config(), "pull", SHA, "amd64")
    assert found["failures"] == []
    assert [row["index_digest"] for row in found["services"]] == [INDEX[s] for s in tool.APP_SERVICES]
    # api 520.5 + frontend 93.9 + worker 506.6 MB, the Debian layer the API and worker share counted once
    assert found["app_layer_bytes"] == 1_091_050_360


def test_layer_sizes_cut_short_are_warned_not_fatal(monkeypatch, capsys):
    def cut_short(ref, arch):
        raise http.client.IncompleteRead(b"")

    monkeypatch.setattr(tool.registry, "image", cut_short)
    assert tool.app_layer_bytes([_ref("backend")], "amd64") is None
    assert "::warning title=smoke-images::" in capsys.readouterr().out


def test_a_locally_built_image_has_no_repo_digest_and_fails(monkeypatch):
    _docker(monkeypatch, digests=lambda service: [] if service == "backend" else
            [f"ghcr.io/automatosai/automatos-{NAMES[service]}@{INDEX[service]}"])
    _registry(monkeypatch)
    failures = tool.check_images(_config(), "pull", SHA, "amd64")["failures"]
    assert len(failures) == 1 and failures[0].startswith("backend: RepoDigests holds nothing (a locally built image)")


def test_an_image_pulled_from_another_digest_fails(monkeypatch):
    _docker(monkeypatch)
    _registry(monkeypatch, resolved={**INDEX, "frontend": "sha256:" + "ff" * 32})
    failures = tool.check_images(_config(), "pull", SHA, "amd64")["failures"]
    assert len(failures) == 1 and failures[0].startswith("frontend: RepoDigests holds ghcr.io/automatosai/")


@pytest.mark.parametrize("row,expected", [
    ({"Repository": "ghcr.io/automatosai/automatos-api", "Tag": "local-feat-x"}, "carries a local-* tag"),
    ({"Repository": "automatos-ai-backend", "Tag": "latest"}, "was built by docker compose"),
])
def test_a_local_tag_or_a_compose_build_in_the_image_list_fails(monkeypatch, row, expected):
    _docker(monkeypatch, listing=[row])
    _registry(monkeypatch)
    found = tool.check_images(_config(), "pull", SHA, "amd64")
    assert len(found["stray"]) == 1 and expected in found["stray"][0]
    assert found["failures"] == found["stray"]


@pytest.mark.parametrize("revision,expected", [
    ("", "no org.opencontainers.image.revision label naming a commit"),
    ("1" * 40, "org.opencontainers.image.revision is 111111111111, expected 0d9f0ac594c3"),
])
def test_a_missing_or_wrong_revision_label_fails(monkeypatch, revision, expected):
    _docker(monkeypatch, revision=revision)
    _registry(monkeypatch)
    failures = tool.check_images(_config(), "pull", SHA, "amd64")["failures"]
    assert failures == [f"{service}: {expected}" for service in tool.APP_SERVICES]


def test_a_manual_sha_tag_checks_the_revision_prefix(monkeypatch):
    _docker(monkeypatch)
    _registry(monkeypatch)
    assert tool.check_images(_config(), "pull", "0d9f0ac", "amd64")["failures"] == []


def test_a_container_running_another_image_than_compose_names_fails(monkeypatch):
    _docker(monkeypatch, tag="sha-1234567")
    _registry(monkeypatch)
    failures = tool.check_images(_config(), "pull", SHA, "amd64")["failures"]
    assert "backend: runs ghcr.io/automatosai/automatos-api:sha-1234567, but compose names " \
           "ghcr.io/automatosai/automatos-api:sha-0d9f0ac" in failures


def test_an_app_image_also_present_under_another_tag_fails_a_pull(monkeypatch):
    """make up ran one tag: the app images exist only under it (PRD-262 US-036)."""
    rows = [{"Repository": _ref(s, TAG).rsplit(":", 1)[0], "Tag": TAG} for s in tool.APP_SERVICES]
    extra = {"Repository": "ghcr.io/automatosai/automatos-api", "Tag": "edge"}
    dangling = {"Repository": "ghcr.io/automatosai/automatos-api", "Tag": "<none>"}
    _docker(monkeypatch, listing=rows + [extra, dangling])
    _registry(monkeypatch)
    found = tool.check_images(_config(), "pull", SHA, "amd64")
    assert found["other_tags"] == [f"ghcr.io/automatosai/automatos-api:edge is here too: the app images must "
                                   f"exist only under {TAG}"]
    assert found["failures"] == found["other_tags"] and found["stray"] == []


def test_a_source_build_must_carry_a_local_tag_and_head(monkeypatch):
    local = "local-0d9f0ac"
    _docker(monkeypatch, digests=lambda service: [], tag=local)

    def no_registry(*args):
        raise AssertionError("a source build is never looked up on the registry")

    monkeypatch.setattr(tool.registry, "resolve", no_registry)
    found = tool.check_images(_config(local), "build", SHA, "amd64")
    assert found["failures"] == [] and found["other_tags"] == []
    _docker(monkeypatch, digests=lambda service: [])
    failures = tool.check_images(_config(), "build", SHA, "amd64")["failures"]
    assert failures == [f"{service}: {_ref(service)} was built here under an official tag: a local build is "
                        f"tagged local-*" for service in tool.APP_SERVICES]


def test_images_built_in_the_job_need_no_repo_digest_and_nothing_is_resolved(monkeypatch):
    _docker(monkeypatch, digests=lambda service: [], tag="pr-0d9f0ac")

    def no_registry(*args):
        raise AssertionError("a prebuilt run must not read the registry")

    monkeypatch.setattr(tool.registry, "resolve", no_registry)
    monkeypatch.setattr(tool.registry, "image", no_registry)
    found = tool.check_images(_config("pr-0d9f0ac"), "prebuilt", SHA, "amd64")
    assert found["failures"] == [] and found["app_layer_bytes"] is None


def test_the_images_command_reads_compose_config_and_fails_on_a_finding(monkeypatch, tmp_path):
    _docker(monkeypatch, revision="")
    _registry(monkeypatch)
    config, out = tmp_path / "config.json", tmp_path / "images.json"
    config.write_text(json.dumps(_config()))
    args = ["images", "--mode", "pull", "--config", str(config), "--expect-revision", SHA, "--arch", "amd64",
            "--out", str(out)]
    assert tool.main(args) == 1
    assert len(json.loads(out.read_text())["failures"]) == 3


# --------------------------------------------------------------------------- report


def _stamp(ns: int) -> str:
    from datetime import datetime, timezone
    seconds, fraction = divmod(ns, NS)
    return datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + f".{fraction:09d}Z"


def _boots() -> list:
    """The pull starts at T0 and takes 40 s; the backend starts at T0+45, the frontend at T0+80."""
    fresh = {"label": "fresh", "container": "automatos_backend", "started_at": _stamp((T0 + 45) * NS),
             "probes": {"health": {"seconds": 20.0}, "ready": {"seconds": 21.5}}}
    frontend = {"label": "frontend", "container": "automatos_frontend", "started_at": _stamp((T0 + 80) * NS),
                "probes": {"ui": {"seconds": 3.0}}}
    return [fresh, frontend]


PULL = {"started_ns": T0 * NS, "seconds": 40.0, "received_bytes": 1_372_400_000, "interface": "eth0",
        "images_before": 0, "exit_code": 0}


def test_the_first_run_runs_from_the_start_of_the_pull():
    assert section.first_run(_boots(), T0 * NS, "pull") == {"since": "pull", "ready_s": 66.5, "ui_s": 83.0}
    assert section.first_run([], T0 * NS, "pull") == {"since": "pull", "ready_s": None, "ui_s": None}


def test_the_report_writes_the_section_and_the_json(monkeypatch, tmp_path):
    _docker(monkeypatch)
    _registry(monkeypatch)
    (tmp_path / "pull.json").write_text(json.dumps(PULL))
    (tmp_path / "processes.json").write_text(json.dumps(
        tool.backend_processes(tool.parse_top(PRODUCTION_TOP), 4)))
    (tmp_path / "images.json").write_text(json.dumps(tool.check_images(_config(), "pull", SHA, "amd64")))
    boots = tmp_path / "boots.jsonl"
    boots.write_text("\n".join(json.dumps(record) for record in _boots()) + "\n")
    summary, out = tmp_path / "summary.md", tmp_path / "report.json"
    args = ["report", "--mode", "pull", "--tag", TAG, "--dir", str(tmp_path), "--results", str(boots),
            "--up-started-ns", str((T0 + 41) * NS), "--json", str(out), "--summary", str(summary)]
    assert tool.main(args) == 0
    data = json.loads(out.read_text())
    assert data["first_run"] == {"since": "pull", "ready_s": 66.5, "ui_s": 83.0}
    assert data["pull"]["received_bytes"] == 1_372_400_000
    text = summary.read_text()
    assert "**40.0 s**, **1,372.4 MB** received on `eth0`" in text
    assert "The app images' compressed layers on GHCR: 1,091.1 MB" in text
    assert "`/health/ready` 200 after **66.5 s**, :3000 200 after **83.0 s**" in text
    assert "workers: 4 (expected 4). ok." in text
    assert f"| backend | `{_ref('backend')}` | `0d9f0ac594c3` | `{INDEX['backend'][:19]}…` | ok |" in text
    assert "- No `local-*` tag and no compose-built image: ok." in text


def test_the_report_says_make_up_ran_and_what_rate_limits_it_met():
    pull = {**PULL, "rate_limited": 1, "rate_limit_lines": ["toomanyrequests: rate limit"]}
    data = section.build({"mode": "pull", "tag": TAG}, pull=pull, processes=None, images=None, boots=_boots())
    text = section.markdown(data)
    assert "- `make up` into an empty Docker cache" in text and "from the start of `make up`" in text
    assert "Registry rate limits (Docker Hub's 429): **1 line(s)**: `toomanyrequests: rate limit`" in text
    clean = section.markdown(section.build({"mode": "pull", "tag": TAG}, PULL | {"rate_limited": 0,
                                                                                 "rate_limit_lines": []},
                                           None, None, []))
    assert "Registry rate limits (Docker Hub's 429): none." in clean


def test_a_source_build_reports_no_pull_and_its_images(monkeypatch):
    _docker(monkeypatch, digests=lambda service: [], tag="local-0d9f0ac")
    images = tool.check_images(_config("local-0d9f0ac"), "build", SHA, "amd64")
    text = section.markdown(section.build({"mode": "build", "tag": "local-0d9f0ac"}, None, None, images, []))
    assert "- Pull: none. The app images were built from this checkout." in text
    assert "- No compose-built image: ok." in text and "only under the tag" not in text


def test_a_source_build_times_the_first_run_from_the_start_of_make_build():
    """PRD-262 US-039: the fresh-clone lane runs make build, and its first run starts with it."""
    data = section.build({"mode": "build", "tag": "local-0d9f0ac", "up_started_ns": (T0 + 41) * NS},
                         pull=None, processes=None, images=None, boots=_boots())
    assert data["first_run"] == {"since": "build", "ready_s": 25.5, "ui_s": 42.0}
    assert "- First run, from the start of `make build`: `/health/ready` 200 after **25.5 s**" in section.markdown(data)


def test_a_prebuilt_run_times_the_first_run_from_compose_up():
    data = section.build({"mode": "prebuilt", "tag": "pr-0d9f0ac", "up_started_ns": (T0 + 41) * NS},
                         pull=None, processes=None, images=None, boots=_boots())
    assert data["first_run"] == {"since": "up", "ready_s": 25.5, "ui_s": 42.0}
    text = section.markdown(data)
    assert "- Pull: none. The images were built in this job" in text
    assert "from `docker compose up`" in text


def test_a_run_that_stopped_early_still_gets_a_section():
    data = section.build({"mode": "pull", "tag": TAG, "up_started_ns": None}, None, None, None, [])
    text = section.markdown(data)
    assert "- Pull: not reached." in text
    assert "Backend processes: not checked" in text
    assert "The app images were not checked" in text


# --------------------------------------------------------------------------- registry


class _Response(io.BytesIO):
    def __init__(self, body: bytes = b"", headers: dict | None = None):
        super().__init__(body)
        self.headers = headers or {}


def _fake_open(monkeypatch, registry, responses):
    seen = []

    def fake_open(request):
        seen.append((request.get_method(), request.full_url))
        return responses.pop(0)

    monkeypatch.setattr(registry, "connect", lambda ref: (registry.Session("https://ghcr.io", "o/a", "t"), ref.rsplit(":", 1)[1]))
    monkeypatch.setattr(registry, "_open", fake_open)
    return seen


def test_resolve_reads_the_digest_header_with_a_head_request(monkeypatch):
    registry = tool.registry
    seen = _fake_open(monkeypatch, registry, [_Response(headers={"Docker-Content-Digest": INDEX["backend"]})])
    assert registry.resolve("ghcr.io/o/a:sha-0d9f0ac") == INDEX["backend"]
    assert seen == [("HEAD", "https://ghcr.io/v2/o/a/manifests/sha-0d9f0ac")]


def test_resolve_hashes_the_manifest_when_the_header_is_missing(monkeypatch):
    registry = tool.registry
    body = b'{"schemaVersion": 2}'
    seen = _fake_open(monkeypatch, registry, [_Response(), _Response(body)])
    assert registry.resolve("ghcr.io/o/a:edge") == "sha256:" + hashlib.sha256(body).hexdigest()
    assert [method for method, _ in seen] == ["HEAD", "GET"]


def test_resolve_returns_a_digest_reference_as_it_is(monkeypatch):
    registry = tool.registry
    monkeypatch.setattr(registry, "connect", lambda ref: (registry.Session("https://ghcr.io", "o/a", None),
                                                           ref.split("@", 1)[1]))
    assert registry.resolve(f"ghcr.io/o/a@{INDEX['frontend']}") == INDEX["frontend"]
