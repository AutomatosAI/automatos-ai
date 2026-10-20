"""PRD-262 US-054: the renderer is built and scanned in images.yml, and never published.

Static checks on ``.github/workflows/images.yml``, ``.github/image-budgets.json`` and
``media-render.yml``, plus ``scripts/ci/image_matrix.py`` (the changes job's picker)
and ``scripts/ci/image_report.py`` fed recorded values:

* images.yml builds ``services/media-render`` (target ``production``) for amd64 and
  arm64 on native runners, on pull requests that touch it and on every main run. It
  pushes only to the job's ``registry:2`` and is never published (§9-Q6: it carries
  GPL-3.0 espeak-ng and phonemizer and GPL x264);
* its build cache is a private GHCR ref, ``automatos-media-render:buildcache-<arch>``:
  every run reads it, only publishing runs write it, and the job logs in to GHCR to
  read it. The other images keep the GHA cache (§9-Q13);
* its budget: a compressed size per arch, and §8's on-disk bar on amd64, which the
  build checks on the image pulled from the job's registry and media-render.yml's
  proof run reads from the same entry;
* its first scans are report only, whatever POLICY_MODE says, and the switch can
  only lower the mode.

Pure file reads, plus the scan step's mode line run through bash: no Docker, no network.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
BUDGETS_FILE = ROOT / ".github" / "image-budgets.json"
OWNER = "AutomatosAI"
LOCAL = "registry:2@sha256:" + "0" * 64
CACHE_REF = "type=registry,ref=ghcr.io/automatosai/automatos-media-render:buildcache-{arch}"
ON_DISK_BAR_MB = 1300.0  # §8: the amd64 image at most 1.30 GB on disk


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "ci" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # before exec, as the scripts register their siblings
    spec.loader.exec_module(module)
    return module


picker = _load("image_matrix")
report = _load("image_report")
layers = _load("image_layers")

WORKFLOW = yaml.safe_load((WORKFLOWS / "images.yml").read_text(encoding="utf-8"))
TRIGGERS = WORKFLOW.get("on", WORKFLOW.get(True))  # PyYAML reads the `on` key as True
PICK = next(step for step in WORKFLOW["jobs"]["changes"]["steps"] if step.get("id") == "pick")
IMAGES = json.loads(PICK["env"]["IMAGES"])
SHARED = json.loads(PICK["env"]["SHARED_INPUTS"])
RUNNERS = json.loads(PICK["env"]["RUNNERS"])
BUILD_STEPS = WORKFLOW["jobs"]["build"]["steps"]
BUDGETS = json.loads(BUDGETS_FILE.read_text(encoding="utf-8"))
MEDIA_RENDER = next(image for image in IMAGES if image["name"] == "media-render")


def _entries(files, publish=False, every=False):
    names = picker.selected(IMAGES, SHARED, files, every)
    return picker.matrix(IMAGES, names, RUNNERS, publish, OWNER, LOCAL)["include"]


def _media_render(publish: bool) -> list[dict]:
    return [entry for entry in _entries([], publish=publish, every=True) if entry["name"] == "media-render"]


def _index(name: str) -> int:
    return next(position for position, step in enumerate(BUILD_STEPS) if step.get("name") == name)


def _step(name: str) -> dict:
    return BUILD_STEPS[_index(name)]


# --------------------------------------------------------------------------- built, never published


def test_images_yml_builds_it_from_its_directory_for_production_on_both_arches():
    assert (MEDIA_RENDER["context"], MEDIA_RENDER["target"]) == ("services/media-render", "production")
    assert MEDIA_RENDER["build_args"] == [], "Railway builds it with no build args"
    assert MEDIA_RENDER["arches"] == ["amd64", "arm64"]
    assert MEDIA_RENDER["inputs"] == ["services/media-render/**"]
    assert "services/media-render/**" in TRIGGERS["pull_request"]["paths"]


def test_a_pr_that_touches_it_builds_it_alone_on_native_runners_into_the_jobs_registry():
    entries = _entries(["services/media-render/media_render/server.py"])
    assert [(entry["name"], entry["arch"]) for entry in entries] == [("media-render", "amd64"), ("media-render", "arm64")]
    assert [entry["runner"] for entry in entries] == [RUNNERS["amd64"], RUNNERS["arm64"]]
    assert RUNNERS == {"amd64": "ubuntu-24.04", "arm64": "ubuntu-24.04-arm"}, "native runners, no QEMU"
    for entry in entries:
        assert (entry["publish"], entry["edge"], entry["registry"]) == (False, "", LOCAL)
        assert entry["image"] == "localhost:5000/automatos-media-render"


def test_main_builds_it_too_and_still_pushes_it_only_to_the_jobs_registry():
    """§9-Q6: built and scanned, not published, on main as on a pull request."""
    assert MEDIA_RENDER["publish"] is False
    entries = _media_render(publish=True)
    assert [entry["arch"] for entry in entries] == ["amd64", "arm64"]
    for entry in entries:
        assert (entry["publish"], entry["registry"]) == (False, LOCAL)
        assert entry["image"] == "localhost:5000/automatos-media-render"
    merged = [image["name"] for image in WORKFLOW["jobs"]["merge"]["strategy"]["matrix"]["image"]]
    assert "media-render" not in merged, "the merge job never tags, publishes or signs it"


def test_the_weekly_rebuild_refreshes_its_runtime_packages():
    """The runtime stage's apt packages ship; the builder stages' never do (US-011)."""
    assert MEDIA_RENDER["no_cache_filters"] == ["runtime"]


# --------------------------------------------------------------------------- the registry cache


def test_its_cache_is_a_private_ghcr_ref_that_publishing_runs_write_and_every_run_reads():
    assert (MEDIA_RENDER["cache"], MEDIA_RENDER["cache_mode"]) == ("registry", "max")
    for publish in (False, True):
        for entry in _media_render(publish):
            ref = CACHE_REF.format(arch=entry["arch"])
            assert entry["cache_from"] == ref
            assert entry["cache_to"] == (f"{ref},mode=max,ignore-error=true" if publish else "")
            assert entry["ghcr_login"] is True, "a private package: reading it needs the login"


def test_the_other_images_keep_the_gha_cache_and_log_in_only_to_publish():
    for publish in (False, True):
        for entry in _entries([], publish=publish, every=True):
            if entry["name"] == "media-render":
                continue
            assert entry["cache_from"].startswith("type=gha,"), entry["name"]
            assert entry["ghcr_login"] is entry["publish"], entry["name"]


def test_the_build_takes_the_pickers_cache_and_logs_in_before_it():
    build = next(step for step in BUILD_STEPS if step.get("id") == "build")
    assert build["with"]["cache-from"] == "${{ matrix.cache_from }}"
    assert build["with"]["cache-to"] == "${{ matrix.cache_to }}"
    login = _step("Log in to GHCR")
    assert login["if"] == "matrix.ghcr_login"
    assert login["with"]["password"] == "${{ secrets.GITHUB_TOKEN }}"
    assert _index("Log in to GHCR") < BUILD_STEPS.index(build)


def test_a_cache_backend_the_picker_does_not_know_is_refused():
    with pytest.raises(ValueError, match="is not one of gha, registry"):
        picker.cache_refs({**MEDIA_RENDER, "cache": "s3"}, "amd64", True, "ghcr.io/o/automatos-media-render")


# --------------------------------------------------------------------------- size: compressed and on disk


def test_it_has_a_compressed_budget_per_arch_and_the_on_disk_bar_on_amd64():
    entries = BUDGETS["budgets"]["media-render"]
    assert set(entries) == {"amd64", "arm64"}
    for entry in entries.values():
        assert entry["max_mb"] > 0 and "US-054" in entry["reason"]
    assert entries["amd64"]["on_disk_max_mb"] == ON_DISK_BAR_MB
    assert "on_disk_max_mb" not in entries["arm64"], "§8 sets an amd64 bar only"
    on_disk = {name for name, arches in BUDGETS["budgets"].items()
               for entry in arches.values() if "on_disk_max_mb" in entry}
    assert on_disk == {"media-render"}


def test_only_media_render_is_measured_on_disk():
    assert {entry["name"] for entry in _entries([], every=True) if entry["on_disk"]} == {"media-render"}


def test_the_build_pulls_it_from_the_jobs_registry_and_reads_dockers_size_before_the_budget():
    step = _step("On-disk size")
    assert step["if"] == "matrix.on_disk"
    assert step["env"]["IMAGE"] == "${{ matrix.image }}" and step["env"]["DIGEST"] == "${{ steps.build.outputs.digest }}"
    assert step["env"]["REPORT"] == "${{ runner.temp }}/layers/${{ matrix.name }}-${{ matrix.arch }}.json"
    run = step["run"]
    assert 'docker pull --quiet "${IMAGE}@${DIGEST}"' in run
    assert "size=$(docker image inspect -f '{{.Size}}' \"${IMAGE}@${DIGEST}\")" in run
    assert 'scripts/ci/image_report.py disk --report "$REPORT" --bytes "$size"' in run
    order = [_index(name) for name in ("Size, layers and update delta", "On-disk size", "Size and update budgets",
                                       "Upload the layer report")]
    assert order == sorted(order), "measured after the report is written, before it is judged and kept"


def test_the_proof_run_reads_the_same_on_disk_bar():
    workflow = yaml.safe_load((WORKFLOWS / "media-render.yml").read_text(encoding="utf-8"))
    runs = "\n".join(step.get("run", "") for step in workflow["jobs"]["media-render"]["steps"])
    assert ".budgets[\"media-render\"].amd64.on_disk_max_mb * 1000000" in runs
    assert "budget=1300000000" not in runs, "one bar, in .github/image-budgets.json"


def _report(tmp_path: Path, **extra) -> Path:
    path = tmp_path / "media-render-amd64.json"
    path.write_text(json.dumps({"name": "media-render", "arch": "amd64", "bytes": 700_000_000, **extra}))
    return path


def test_disk_adds_the_on_disk_size_and_keeps_the_rest_of_the_report(tmp_path):
    path, summary = _report(tmp_path, layers=[{"digest": "sha256:" + "ab" * 32}]), tmp_path / "summary.md"
    assert report.main(["disk", "--report", str(path), "--bytes", "1250000000", "--summary", str(summary)]) == 0
    written = json.loads(path.read_text())
    assert written["on_disk_bytes"] == 1_250_000_000 and written["bytes"] == 700_000_000
    assert written["layers"] == [{"digest": "sha256:" + "ab" * 32}]
    assert "**On disk** (media-render, amd64): 1,250.0 MB" in summary.read_text()


@pytest.mark.parametrize("value", ["0", "-5", "1.3GB"])
def test_disk_refuses_a_size_that_is_not_a_byte_count(tmp_path, value):
    with pytest.raises(SystemExit):
        report.main(["disk", "--report", str(_report(tmp_path)), "--bytes", value])


@pytest.mark.parametrize("mode,size,level,code", [
    ("warn", 1_250_000_000, "ok", 0),
    ("warn", 1_400_000_000, "warning", 0),
    ("fail", 1_300_000_000, "ok", 0),
    ("fail", 1_300_000_001, "error", 1),
])
def test_the_budget_holds_the_amd64_image_to_its_on_disk_bar_in_the_files_mode(tmp_path, capsys, mode, size,
                                                                             level, code):
    budgets = tmp_path / "budgets.json"
    budgets.write_text(json.dumps({**BUDGETS, "mode": mode}))
    path = _report(tmp_path, on_disk_bytes=size)
    assert report.main(["budget", "--report", str(path), "--budgets", str(budgets)]) == code
    verdict = layers.on_disk_verdict("media-render", "amd64", size, {**BUDGETS, "mode": mode})
    assert verdict.level == level
    out = capsys.readouterr().out
    if level == "ok":
        assert "on disk" not in out
    else:
        assert f"::{level} title=Image size budget::media-render (amd64) on disk is" in out


def test_an_on_disk_bar_with_no_measurement_is_a_finding_not_a_pass():
    verdict = layers.on_disk_verdict("media-render", "amd64", None, {**BUDGETS, "mode": "fail"})
    assert verdict.level == "error" and "was not measured" in verdict.message


def test_arm64_reports_its_on_disk_size_without_a_bar_and_other_images_say_nothing():
    verdict = layers.on_disk_verdict("media-render", "arm64", 1_100_000_000, BUDGETS)
    assert verdict.level == "ok" and "with no on-disk budget" in verdict.message
    assert layers.on_disk_verdict("api", "amd64", None, BUDGETS) is None


# --------------------------------------------------------------------------- the first scans


def test_only_media_render_is_scanned_report_only():
    assert MEDIA_RENDER["scan_report_only"] is True
    entries = _entries([], publish=True, every=True)
    assert {entry["name"] for entry in entries if entry["scan_report_only"]} == {"media-render"}


def _mode_line() -> str:
    step = _step("Scan policy")
    assert step["env"]["SCAN_REPORT_ONLY"] == "${{ matrix.scan_report_only }}"
    first, rest = step["run"].split("\n", 1)
    assert '--mode "$POLICY_MODE"' in rest, "the policy runs in the mode the first line leaves"
    assert "POLICY_MODE=" not in rest, "nothing else changes the mode"
    return first


@pytest.mark.parametrize("report_only,policy_mode,expected", [
    ("true", "fail", "report"),
    ("true", "report", "report"),
    ("false", "fail", "fail"),
    ("false", "report", "report"),
])
def test_report_only_lowers_the_mode_and_never_raises_it(report_only, policy_mode, expected):
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "SCAN_REPORT_ONLY": report_only,
           "POLICY_MODE": policy_mode}
    result = subprocess.run(["bash", "-c", f'{_mode_line()}\necho "$POLICY_MODE"'], env=env,
                            capture_output=True, text=True, check=True)
    assert result.stdout.strip() == expected
