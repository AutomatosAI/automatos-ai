"""PRD-262 US-001: which images a PR builds, where they push, and what they are judged by.

Static checks on ``.github/workflows/images.yml`` and ``.github/image-budgets.json``,
plus ``scripts/ci/image_matrix.py`` (the ``changes`` job's picker) fed changed-file
lists:

* the PR paths include the frontend's ``public/`` files, build config and ``.npmrc``,
  and the budgets, and are exactly the images' inputs plus the shared ones;
* a frontend-only PR builds only the frontend, on both arches;
* PR and branch builds push to the job's ``registry:2`` (buildx on the host
  network), main pushes to GHCR, and only publishing runs write the GHA cache,
  ``mode=min`` for the API (the picker writes the cache settings);
* every image and arch has a budget with a reason.
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[2]
_WORKFLOW = _REPO / ".github" / "workflows" / "images.yml"
_BUDGETS = _REPO / ".github" / "image-budgets.json"
OWNER = "AutomatosAI"
LOCAL = "registry:2@sha256:" + "0" * 64


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / "scripts" / "ci" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


picker = _load("image_matrix")
WORKFLOW = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
TRIGGERS = WORKFLOW.get("on", WORKFLOW.get(True))  # PyYAML reads the `on` key as True
PICK = next(step for step in WORKFLOW["jobs"]["changes"]["steps"] if step.get("id") == "pick")
IMAGES = json.loads(PICK["env"]["IMAGES"])
SHARED = json.loads(PICK["env"]["SHARED_INPUTS"])
RUNNERS = json.loads(PICK["env"]["RUNNERS"])
BUILD = WORKFLOW["jobs"]["build"]
BUILD_STEP = next(step for step in BUILD["steps"] if step.get("id") == "build")


def _pick(files, publish=False, every=False):
    names = picker.selected(IMAGES, SHARED, files, every)
    return picker.matrix(IMAGES, names, RUNNERS, publish, OWNER, LOCAL)["include"]


@pytest.mark.parametrize("path", [
    "frontend/public/**", "frontend/next.config.js", "frontend/csp-sources.js", "frontend/.npmrc",
    ".github/image-budgets.json", ".grype.yaml",
])
def test_the_pull_request_paths_include(path):
    assert path in TRIGGERS["pull_request"]["paths"]


def test_the_pull_request_paths_are_exactly_the_image_inputs():
    """A path only the trigger lists would start a run that builds nothing, and vice versa."""
    inputs = set(SHARED).union(*(image["inputs"] for image in IMAGES))
    assert set(TRIGGERS["pull_request"]["paths"]) == inputs


def test_a_frontend_only_pr_builds_only_the_frontend_on_both_arches():
    entries = _pick(["frontend/public/auto/new-pose.png", "frontend/app/page.tsx"])
    assert [(entry["name"], entry["arch"]) for entry in entries] == [("frontend", "amd64"), ("frontend", "arm64")]
    assert {entry["runner"] for entry in entries} == {"ubuntu-24.04", "ubuntu-24.04-arm"}
    for entry in entries:
        assert entry["publish"] is False
        assert entry["image"] == "localhost:5000/automatos-frontend"
        assert entry["registry"] == LOCAL
        assert entry["edge"] == "ghcr.io/automatosai/automatos-frontend:edge"
        assert "NEXT_PUBLIC_AUTH_EDITION=local\nNEXT_PUBLIC_API_URL=http://localhost:8000" in entry["build_args"]


EVERY_IMAGE = ["api", "frontend", "workspace-worker", "agent-opt-worker", "media-render"]


@pytest.mark.parametrize("path,names", [
    ("orchestrator/requirements.txt", ["api"]),
    ("services/workspace-worker/executor.py", ["workspace-worker"]),
    ("services/agent-opt-worker/main.py", ["agent-opt-worker"]),
    ("services/media-render/Dockerfile", ["media-render"]),
    (".github/image-budgets.json", EVERY_IMAGE),
    ("scripts/ci/image_report.py", EVERY_IMAGE),
    ("scripts/ci/image_scan_policy.py", EVERY_IMAGE),
    ("frontend/app/page.tsx", []),
    ("scripts/ci/sub/image_x.py", []),
])
def test_each_change_builds_the_images_that_read_it(path, names):
    assert picker.selected(IMAGES, SHARED, [path], every=False) == names


def test_a_push_publishes_every_published_image_to_ghcr():
    entries = _pick([], publish=True, every=True)
    assert len(entries) == sum(len(image["arches"]) for image in IMAGES)
    published = {image["name"] for image in IMAGES if image["publish"]}
    assert published == {"api", "frontend", "workspace-worker"}
    for entry in entries:
        if entry["name"] in published:
            assert entry["publish"] is True and entry["registry"] == ""
            assert entry["image"] == f"ghcr.io/automatosai/automatos-{entry['name']}"
        else:
            assert entry["publish"] is False and entry["registry"] == LOCAL
            assert entry["image"] == f"localhost:5000/automatos-{entry['name']}"


def test_an_image_that_is_never_published_has_no_edge_and_stays_local():
    image = {**IMAGES[0], "name": "media-render", "publish": False}
    entry = picker.entry(image, "amd64", "ubuntu-24.04", True, OWNER, LOCAL)
    assert (entry["publish"], entry["edge"], entry["registry"]) == (False, "", LOCAL)
    assert entry["image"] == "localhost:5000/automatos-media-render"


def test_the_picker_writes_the_job_outputs(monkeypatch):
    args = ["--images", json.dumps(IMAGES), "--shared-inputs", json.dumps(SHARED), "--runners", json.dumps(RUNNERS),
            "--owner", OWNER, "--publish", "false", "--local-registry", LOCAL]
    for files, any_ in (("frontend/.npmrc\n", "true"), ("docs/README.md\n", "false")):
        monkeypatch.setattr(sys, "stdin", io.StringIO(files))
        out = io.StringIO()
        with redirect_stdout(out):
            assert picker.main(args) == 0
        lines = dict(line.split("=", 1) for line in out.getvalue().splitlines())
        assert lines["any"] == any_
        assert isinstance(json.loads(lines["matrix"])["include"], list)


def test_the_build_job_takes_its_matrix_from_the_changes_job():
    assert BUILD["needs"] == "changes"
    assert BUILD["if"] == "needs.changes.outputs.any == 'true'"
    assert BUILD["strategy"]["matrix"] == "${{ fromJSON(needs.changes.outputs.matrix) }}"


def test_pr_builds_push_to_the_jobs_own_registry():
    assert BUILD["services"]["registry"]["image"] == "${{ matrix.registry }}"
    assert BUILD["services"]["registry"]["ports"] == ["5000:5000"]
    assert PICK["env"]["LOCAL_REGISTRY"].startswith("registry:2@sha256:")
    buildx = next(step for step in BUILD["steps"] if step.get("uses", "").startswith("docker/setup-buildx-action"))
    assert buildx["with"]["driver-opts"] == "network=host"
    outputs = BUILD_STEP["with"]["outputs"]
    assert "name=${{ matrix.image }}" in outputs and "push-by-digest=true" in outputs and "push=true" in outputs


def test_only_publishing_runs_write_the_cache_and_the_api_writes_min():
    """The picker writes both cache settings; the GHA images keep one scope per image and arch."""
    assert BUILD_STEP["with"]["cache-from"] == "${{ matrix.cache_from }}"
    assert BUILD_STEP["with"]["cache-to"] == "${{ matrix.cache_to }}"
    modes = {image["name"]: image["cache_mode"] for image in IMAGES}
    assert modes["api"] == "min" and modes["frontend"] == "max"
    gha = {image["name"] for image in IMAGES if image.get("cache", "gha") == "gha"}
    assert gha == {"api", "frontend", "workspace-worker", "agent-opt-worker"}
    for publish in (False, True):
        for entry in _pick([], publish=publish, every=True):
            if entry["name"] not in gha:
                continue  # media-render's registry cache: test_prd262_media_render_images.py
            scope = f"scope={entry['name']}-{entry['arch']}"
            assert entry["cache_from"] == f"type=gha,{scope}"
            assert entry["cache_to"] == (f"type=gha,mode={modes[entry['name']]},{scope}" if publish else "")


def test_every_build_measures_and_checks_its_budget():
    runs = "\n".join(step.get("run", "") for step in BUILD["steps"])
    assert "scripts/ci/image_report.py measure" in runs
    assert "scripts/ci/image_report.py budget" in runs
    uploads = [step["with"]["name"] for step in BUILD["steps"] if step.get("uses", "").startswith("actions/upload-artifact")]
    assert "layers-${{ matrix.name }}-${{ matrix.arch }}" in uploads


def test_the_report_job_reads_every_build():
    job = WORKFLOW["jobs"]["report"]
    assert job["needs"] == "build"
    download = next(step for step in job["steps"] if step.get("uses", "").startswith("actions/download-artifact"))
    assert download["with"]["pattern"] == "layers-*"
    assert {"shared_layers_amd64", "shared_layers_arm64"} <= set(job["outputs"])


def test_every_image_and_arch_has_a_budget_with_a_reason():
    budgets = json.loads(_BUDGETS.read_text(encoding="utf-8"))
    assert budgets["mode"] in ("warn", "fail")
    assert set(budgets["budgets"]) == {image["name"] for image in IMAGES}
    for image in IMAGES:
        entries = budgets["budgets"][image["name"]]
        assert set(entries) == set(image["arches"])
        for entry in entries.values():
            assert entry["max_mb"] > 0 and entry["reason"].strip()


def test_glob_patterns_follow_github_path_filters():
    assert picker.glob_regex("scripts/ci/image_*.py").match("scripts/ci/image_matrix.py")
    assert not picker.glob_regex("scripts/ci/image_*.py").match("scripts/ci/a/image_x.py")
    assert picker.glob_regex("frontend/public/**").match("frontend/public/auto/poses/a.webm")
    assert not picker.glob_regex(".grype.yaml").match("xgrype.yaml")


@pytest.mark.parametrize("pattern", ["frontend/*.config.[jt]s", "frontend/next.config.?s", "!docs/**", "a+b"])
def test_filter_syntax_the_picker_does_not_copy_is_refused(pattern):
    """GitHub reads these differently from a literal match, so the picker must not guess."""
    with pytest.raises(ValueError, match="does not support"):
        picker.glob_regex(pattern)
