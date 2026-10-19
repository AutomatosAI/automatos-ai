"""PRD-262 US-011: a weekly uncached rebuild brings in OS fixes.

images.yml runs every Monday at 05:00 UTC. On that schedule, and on a manual run with
``no_cache``, each image's build skips the cache for the stages whose OS packages ship
(``no-cache-filters``), so Debian and Alpine fixes arrive without a code change. The
schedule publishes ``weekly-<yyyymmdd>`` and never ``sha-<7>``; smoke-images boots that
tag, and promote-edge moves ``edge`` to it only after it boots.

Static checks on the two workflows and the Dockerfiles, plus the picker
(scripts/ci/image_matrix.py), the smoke's target (smoke_images.py) and promote_edge.py
fed recorded values:

* the schedule, the ``no_cache`` input, and the NO_CACHE switch that hands each build
  its filters (and only the filters: the rest of the build keeps its cache);
* each image's filters are exactly the stages of its shipped image that install apt or
  apk packages, read from its Dockerfile. The frontend runtime runs ``apk upgrade`` in
  the same RUN as ``apk add libstdc++``; the worker's stage has a name;
* a branch run never publishes. A schedule run tags only ``weekly-<yyyymmdd>``: no other
  tag (``sha-<7>``, or US-036's ``tree-<12>`` when it lands) is pushed on a schedule;
* both workflows derive the weekly tag from the images run's created_at, so the smoke
  boots what images.yml published, even after a re-run on another day.

Pure file reads: no Docker, no network.
"""
from __future__ import annotations

import importlib.util
import io
import json
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
CREATED = "2026-10-19T05:03:12Z"  # a Monday, as GitHub writes created_at
WEEKLY = "weekly-20261019"
SHA = "0d9f0ac594c3783e62a30e6c5ce75164028850db"
OWNER = "AutomatosAI"
LOCAL = "registry:2@sha256:" + "0" * 64
PREFIX = "ghcr.io/automatosai/automatos-"
NOT_ON_SCHEDULE = "enable=${{ github.event_name != 'schedule' }}"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "ci" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # before exec, as the scripts register their siblings
    spec.loader.exec_module(module)
    return module


picker = _load("image_matrix")
smoke = _load("smoke_images")
promote = _load("promote_edge")


def _yaml(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


IMAGES_WORKFLOW = _yaml("images.yml")
SMOKE_JOBS = _yaml("smoke-images.yml")["jobs"]
TRIGGERS = IMAGES_WORKFLOW.get("on", IMAGES_WORKFLOW.get(True))  # PyYAML reads the `on` key as True
JOBS = IMAGES_WORKFLOW["jobs"]
PICK = next(step for step in JOBS["changes"]["steps"] if step.get("id") == "pick")
IMAGES = json.loads(PICK["env"]["IMAGES"])
SHARED = json.loads(PICK["env"]["SHARED_INPUTS"])
RUNNERS = json.loads(PICK["env"]["RUNNERS"])
BUILD_STEP = next(step for step in JOBS["build"]["steps"] if step.get("id") == "build")
MERGE = JOBS["merge"]
MERGE_META = next(step for step in MERGE["steps"] if step.get("id") == "meta")


def _image(name: str) -> dict:
    return next(image for image in IMAGES if image["name"] == name)


def _expression(value: str) -> str:
    """The body of a ``${{ … }}`` expression, whitespace collapsed."""
    match = re.fullmatch(r"\$\{\{\s*(.*?)\s*\}\}", value.strip(), re.S)
    assert match, value
    return " ".join(match.group(1).split())


# --------------------------------------------------------------------------- Dockerfiles

_CONTINUATION = re.compile(r"\\[ \t]*\n")
_FROM_WORD = re.compile(r"^FROM\s", re.I)
# FROM [--platform=…] <image or stage> [AS <name>]
_FROM = re.compile(r"^FROM\s+(?:--\S+\s+)*(\S+)(?:\s+AS\s+(\S+))?\s*$", re.I)
# apt-get/apt install or upgrade, apk add or upgrade, with any flags before the verb.
_OS_PACKAGES = re.compile(
    r"\bapt(?:-get)?(?:\s+-\S+)*\s+(?:install|upgrade|dist-upgrade)\b|\bapk(?:\s+-\S+)*\s+(?:add|upgrade)\b"
)


def _instructions(text: str) -> list[str]:
    """Instructions with continuation lines joined; comment lines dropped."""
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    return [line.strip() for line in _CONTINUATION.sub(" ", code).splitlines() if line.strip()]


def stages(text: str) -> list[tuple[str, str, list[str]]]:
    """(name, FROM image or stage, RUN instructions) per stage in build order; '' when unnamed."""
    found: list[tuple[str, str, list[str]]] = []
    for line in _instructions(text):
        if _FROM_WORD.match(line):
            match = _FROM.match(line)
            if not match:
                raise ValueError(f"a FROM line this check cannot read: {line}")
            found.append(((match.group(2) or "").lower(), match.group(1).lower(), []))
        elif found and line.upper().startswith("RUN "):
            found[-1][2].append(line)
    return found


def os_package_stages(text: str, target: str) -> list[str]:
    """The stages of the shipped image's FROM chain that install OS packages.

    The chain starts at ``target`` (the last stage when empty) and follows ``FROM
    <stage>`` back to an image. A stage another stage only copies from (the API's
    pybuild, the frontend's builder) is not in it: its packages never ship. An unnamed
    stage shows as '', which no filter can name.
    """
    parsed = stages(text)
    index = {name: position for position, (name, _, _) in enumerate(parsed) if name}
    current = index[target.lower()] if target else len(parsed) - 1
    chain: list[int] = []
    while current is not None and current not in chain:
        chain.append(current)
        current = index.get(parsed[current][1])
    return sorted(parsed[i][0] for i in chain if any(_OS_PACKAGES.search(run) for run in parsed[i][2]))


def _dockerfile(image: dict) -> str:
    return (ROOT / image["context"] / "Dockerfile").read_text(encoding="utf-8")


def test_the_stage_reader_follows_the_from_chain_and_shows_an_unnamed_stage():
    text = (
        "# syntax=docker/dockerfile:1\n"
        "FROM debian:13 AS build\nRUN apt-get -y install gcc\n"
        "FROM debian:13 as base\nRUN apt-get update && \\\n    apt-get install -y curl\n"
        "FROM base AS production\nCOPY --from=build /out /usr/local\nRUN useradd -m app\n"
    )
    assert os_package_stages(text, "production") == ["base"]
    assert os_package_stages(text, "") == ["base"]
    assert os_package_stages(text, "build") == ["build"]
    assert os_package_stages("FROM alpine:3.24\nRUN apk --no-cache add tini\n", "") == [""]
    with pytest.raises(ValueError):
        stages("FROM python:3.12-slim AS one two\n")


@pytest.mark.parametrize("image", IMAGES, ids=lambda image: image["name"])
def test_each_image_rebuilds_exactly_the_stages_whose_os_packages_ship(image):
    expected = os_package_stages(_dockerfile(image), image["target"])
    assert "" not in expected, f"{image['name']}: an unnamed stage installs OS packages; name it so it can be filtered"
    assert sorted(image["no_cache_filters"]) == expected


def test_the_api_rebuilds_its_runtime_packages_and_not_its_compilers():
    assert _image("api")["no_cache_filters"] == ["base"]


def test_the_workers_stage_has_a_name_so_it_can_be_filtered():
    worker = _image("workspace-worker")
    assert stages(_dockerfile(worker))[-1][0] == "production"
    assert worker["no_cache_filters"] == ["production"]


def test_the_frontend_runtime_upgrades_alpine_in_the_same_run_as_libstdcpp():
    frontend = _image("frontend")
    runtime = {name: runs for name, _, runs in stages(_dockerfile(frontend))}["production"]
    runs = [run for run in runtime if re.search(r"\bapk add\b[^\n]*\blibstdc\+\+", run)]
    assert len(runs) == 1, runtime
    assert "apk upgrade --no-cache" in runs[0], runs[0]
    assert runs[0].index("apk upgrade") < runs[0].index("apk add"), "upgrade first, then add against it"
    assert frontend["no_cache_filters"] == ["production"]


# --------------------------------------------------------------------------- images.yml


def test_images_rebuilds_every_monday_at_five_utc():
    assert TRIGGERS["schedule"] == [{"cron": "0 5 * * 1"}]


def test_a_manual_run_can_ask_for_the_same_rebuild():
    no_cache = TRIGGERS["workflow_dispatch"]["inputs"]["no_cache"]
    assert no_cache["type"] == "boolean" and no_cache["default"] is False


def test_the_filters_apply_to_schedule_runs_and_no_cache_dispatches_only():
    env = IMAGES_WORKFLOW["env"]
    assert _expression(env["NO_CACHE"]) == "github.event_name == 'schedule' || inputs.no_cache == true"
    assert '--no-cache "$NO_CACHE"' in PICK["run"]
    assert BUILD_STEP["with"]["no-cache-filters"] == "${{ matrix.no_cache_filters }}"
    assert "no-cache" not in BUILD_STEP["with"], "only the OS-package stages skip the cache, never the whole build"


def _entries(no_cache: bool, publish: bool = True) -> list[dict]:
    names = picker.selected(IMAGES, SHARED, [], every=True)
    return picker.matrix(IMAGES, names, RUNNERS, publish, OWNER, LOCAL, no_cache)["include"]


def test_a_no_cache_run_hands_each_build_its_filters_and_any_other_run_none():
    entries = _entries(no_cache=True)
    assert len(entries) == sum(len(image["arches"]) for image in IMAGES)
    for entry in entries:
        assert entry["no_cache_filters"] == ",".join(_image(entry["name"])["no_cache_filters"])
    assert {entry["no_cache_filters"] for entry in _entries(no_cache=False)} == {""}


def test_a_branch_run_never_publishes_and_the_merge_job_agrees():
    publish = _expression(IMAGES_WORKFLOW["env"]["PUBLISH"])
    assert "github.ref == 'refs/heads/main'" in publish
    assert "startsWith(github.ref, 'refs/tags/v')" in publish
    assert " ".join(MERGE["if"].split()) == publish, "the merge job's if: repeats PUBLISH"
    for entry in _entries(no_cache=True, publish=False):
        assert entry["publish"] is False and entry["registry"] == LOCAL
        assert entry["image"] == f"localhost:5000/automatos-{entry['name']}"


def _merge_tags() -> list[str]:
    return [line.strip() for line in MERGE_META["with"]["tags"].splitlines() if line.strip()]


def test_a_schedule_run_tags_weekly_and_never_re_pushes_a_sha_tag():
    tags = _merge_tags()
    assert f"type=sha,{NOT_ON_SCHEDULE}" in tags
    assert "type=schedule,pattern=${{ needs.changes.outputs.weekly_tag }}" in tags
    assert set(MERGE["needs"]) == {"changes", "build"}


def test_no_other_tag_is_pushed_on_a_schedule_run():
    """sha-<7>, and US-036's tree-<12> when it lands, stay immutable: a schedule run pushes neither."""
    for tag in _merge_tags():
        kind = tag.split(",", 1)[0]
        if kind in ("type=schedule", "type=semver"):  # semver applies to v* tag pushes only
            continue
        assert NOT_ON_SCHEDULE in tag, f"{tag} would be re-pushed by the weekly rebuild"


def test_the_changes_job_names_the_weekly_tag_from_the_runs_created_at():
    job = JOBS["changes"]
    assert job["permissions"]["actions"] == "read"
    assert job["outputs"]["weekly_tag"] == "${{ steps.pick.outputs.weekly_tag }}"
    schedule_only = PICK["run"].split('if [ "$EVENT" = "schedule" ]; then', 1)[1].split("\nfi", 1)[0]
    assert 'gh api "repos/$REPO/actions/runs/$GITHUB_RUN_ID" --jq .created_at' in schedule_only
    assert '"${weekly[@]}"' in PICK["run"]


@pytest.mark.parametrize("created,tag", [
    (CREATED, WEEKLY),
    ("2026-10-19T01:30:00+02:00", "weekly-20261018"),  # the UTC day, not the offset's
    (" 2026-10-26T05:00:00Z\n", "weekly-20261026"),
])
def test_the_weekly_tag_is_the_utc_day_the_run_was_created(created, tag):
    assert picker.weekly_tag(created) == tag


@pytest.mark.parametrize("created", ["", "2026-10-19T05:03:12", "Monday"])
def test_a_creation_time_without_a_zone_is_refused(created):
    with pytest.raises(ValueError):
        picker.weekly_tag(created)


def test_the_picker_writes_the_weekly_tag_only_when_given_a_creation_time(monkeypatch):
    common = ["--images", json.dumps(IMAGES), "--shared-inputs", json.dumps(SHARED), "--runners", json.dumps(RUNNERS),
              "--owner", OWNER, "--publish", "true", "--local-registry", LOCAL, "--every"]
    for extra, weekly in ((["--no-cache", "true", "--created", CREATED], WEEKLY), ([], None)):
        monkeypatch.setattr(sys, "stdin", io.StringIO(""))
        out = io.StringIO()
        with redirect_stdout(out):
            assert picker.main(common + extra) == 0
        lines = dict(line.split("=", 1) for line in out.getvalue().splitlines())
        assert lines.get("weekly_tag") == weekly
        filters = {entry["no_cache_filters"] for entry in json.loads(lines["matrix"])["include"]}
        assert (filters != {""}) is bool(extra)


# --------------------------------------------------------------------------- smoke-images and edge


def test_smoke_images_boots_what_a_schedule_run_published():
    condition = " ".join(SMOKE_JOBS["smoke-images"]["if"].split())
    assert "github.event.workflow_run.event == 'schedule'" in condition
    target = next(step for step in SMOKE_JOBS["smoke-images"]["steps"] if step.get("id") == "target")
    assert target["env"]["TRIGGER"] == "${{ github.event.workflow_run.event }}"
    assert target["env"]["CREATED"] == "${{ github.event.workflow_run.created_at }}"
    assert '--trigger "$TRIGGER" --created "$CREATED"' in target["run"]
    assert SMOKE_JOBS["promote-edge"]["needs"] == "smoke-images"


def test_the_smoke_boots_the_tag_images_yml_published():
    chosen = smoke.target("workflow_run", SHA, trigger="schedule", created=CREATED)
    assert chosen == {"tag": picker.weekly_tag(CREATED), "revision": SHA}
    assert chosen["tag"] == WEEKLY
    assert smoke.target("workflow_run", SHA, trigger="push", created=CREATED)["tag"] == "sha-0d9f0ac"
    assert smoke.target("workflow_run", SHA, trigger="workflow_dispatch")["tag"] == "sha-0d9f0ac"
    with pytest.raises(ValueError):
        smoke.target("workflow_run", SHA, trigger="schedule")


def test_the_target_command_takes_the_trigger_and_the_creation_time(tmp_path):
    outputs = tmp_path / "out"
    argv = ["target", "--event", "workflow_run", "--sha", SHA, "--trigger", "schedule", "--created", CREATED,
            "--outputs", str(outputs)]
    assert smoke.main(argv) == 0
    assert outputs.read_text().splitlines() == [f"tag={WEEKLY}", f"revision={SHA}"]


def test_edge_moves_to_the_weekly_rebuild_of_the_commit_it_already_runs(monkeypatch):
    weekly_digest, edge_digest = "sha256:" + "a" * 64, "sha256:" + "e" * 64
    tags = {f"{PREFIX}api:{WEEKLY}": weekly_digest, f"{PREFIX}api:edge": edge_digest}
    config = {"config": {"Labels": {promote.REVISION_LABEL: SHA}}}
    monkeypatch.setattr(promote.registry, "resolve", lambda ref: tags[ref])
    monkeypatch.setattr(promote.registry, "image", lambda ref, arch: ({"layers": []}, config, None))
    monkeypatch.setattr(promote, "compare", lambda repo, base, head: "identical")
    commands = []

    def run(command, check):
        commands.append(command)
        tags[command[-2]] = tags[command[-1]]

    args = promote.build_parser().parse_args(
        ["--sha", SHA, "--tag", WEEKLY, "--repo", "AutomatosAI/automatos-ai", "--prefix", PREFIX, "--image", "api"]
    )
    assert promote.promote(args, run=run) == 0
    assert commands == [["docker", "buildx", "imagetools", "create", "-t", f"{PREFIX}api:edge", f"{PREFIX}api:{WEEKLY}"]]
    assert tags[f"{PREFIX}api:edge"] == weekly_digest
