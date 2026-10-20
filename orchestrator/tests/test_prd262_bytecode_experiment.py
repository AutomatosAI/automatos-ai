"""PRD-262 US-032 (§9-Q7): the one-off experiment that decides whether the app bytecode stays.

.github/workflows/bytecode-experiment.yml builds the API's production stage as committed
and with the compile step removed, each again after a code-only change, and boots both
in new containers against one seeded database (scripts/ci/bytecode-experiment.sh).
scripts/ci/bytecode_experiment.py strips the step, reads the layers back, times the
boots and reports the outcome. This suite checks:

- strip-compile removes exactly the compile stage's RUN from the committed Dockerfile,
  and refuses a Dockerfile without that one compileall step;
- a build's app layer and code-only update delta, judged against US-016's bar;
- the boot timer, from the container's start to the first /health 200;
- the medians, the gain and the §9-Q7 outcome in each of its branches, and the report;
- the workflow builds the four images the way images.yml builds the API, strips and
  restores the checkout between them, and reports even when a boot fails; the boot
  script recreates containers (never `restart`) with one worker, in the order
  with, without, without, with, after creating the local workspace's mount point in
  the fresh workspaces volume.

Pure file reads and stand-ins: no Docker, no registry.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "ci" / "bytecode_experiment.py"
SHELL = (ROOT / "scripts" / "ci" / "bytecode-experiment.sh").read_text(encoding="utf-8")
WORKFLOW = yaml.safe_load((ROOT / ".github" / "workflows" / "bytecode-experiment.yml").read_text(encoding="utf-8"))
DOCKERFILE = (ROOT / "orchestrator" / "Dockerfile").read_text(encoding="utf-8")
BUDGETS = json.loads((ROOT / ".github" / "image-budgets.json").read_text(encoding="utf-8"))
MB = 1_000_000


def _load():
    spec = importlib.util.spec_from_file_location("bytecode_experiment", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["bytecode_experiment"] = module
    spec.loader.exec_module(module)
    return module


experiment = _load()


# --------------------------------------------------------------------------- strip-compile


def _stage_lines(text: str, name: str) -> list[str]:
    lines = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    start = next(i for i, line in enumerate(lines) if re.fullmatch(rf"FROM \S+ as {name}", line.strip(), re.I))
    end = next((i for i in range(start + 1, len(lines)) if lines[i].upper().startswith("FROM ")), len(lines))
    return lines[start:end]


def test_strip_compile_removes_the_compile_step_and_nothing_else():
    stripped = experiment.without_compile(DOCKERFILE)
    assert _stage_lines(stripped, "compile") == ["FROM base as compile", "COPY . ."]
    assert "compileall -q --invalidation-mode unchecked-hash -i" not in stripped
    removed = [line for line in DOCKERFILE.splitlines() if line not in stripped.splitlines()]
    assert len(removed) == 2 and removed[0].startswith("RUN python scripts/boot_modules.py"), removed
    assert _stage_lines(stripped, "production") == _stage_lines(DOCKERFILE, "production")


@pytest.mark.parametrize("text", [
    "FROM base as production\nRUN true\n",
    "FROM base as compile\nCOPY . .\nFROM base as production\n",
    "FROM base as compile\nRUN echo one\nRUN python -m compileall -q .\n",
    "FROM base as compile\nRUN echo no compile here\n",
])
def test_strip_compile_refuses_anything_but_the_one_compile_step(text):
    with pytest.raises(ValueError):
        experiment.without_compile(text)


def test_the_command_rewrites_the_file_in_place(tmp_path):
    path = tmp_path / "Dockerfile"
    path.write_text(DOCKERFILE, encoding="utf-8")
    assert experiment.main(["strip-compile", str(path)]) == 0
    assert path.read_text(encoding="utf-8") == experiment.without_compile(DOCKERFILE)


# --------------------------------------------------------------------------- layers


def _image(app_digest: str, app_size: int) -> tuple[dict, dict]:
    texts = ["COPY /install /usr/local", "RUN useradd -m -u 1000 automatos", "COPY /app /app"]
    digests = ["sha256:" + "a" * 64, "sha256:" + "b" * 64, app_digest]
    sizes = [300 * MB, 4096, app_size]
    manifest = {"layers": [{"mediaType": "application/vnd.oci.image.layer.v1.tar+gzip", "digest": digest,
                            "size": size} for digest, size in zip(digests, sizes)]}
    return manifest, {"history": [{"created_by": f"{text} # buildkit"} for text in texts]}


@pytest.mark.parametrize(("app_size", "verdict"), [(5 * MB, "ok"), (11 * MB, "error")])
def test_a_builds_app_layer_and_its_code_only_update_against_the_bar(app_size, verdict):
    base, edited = _image("sha256:" + "c" * 64, app_size), _image("sha256:" + "d" * 64, app_size + 1)
    found = experiment.build_layers(*base, *edited, BUDGETS)
    assert found["app_layer_bytes"] == app_size and found["app_layer_instruction"] == "COPY /app /app"
    assert (found["delta_bytes"], found["delta_change"]) == (app_size + 1, "code-only")
    assert found["verdict"] == verdict, found["verdict_message"]
    assert found["image_bytes"] == 300 * MB + 4096 + app_size


# --------------------------------------------------------------------------- boot


def test_the_timer_counts_from_the_containers_start_to_the_first_200():
    statuses = iter([0, 503, 200])
    now = [10 * experiment.NS_PER_SECOND]

    def sleep(seconds):
        now[0] += int(seconds * experiment.NS_PER_SECOND)

    seconds = experiment.seconds_to_200("http://x/health", 4 * experiment.NS_PER_SECOND, 60,
                                        probe_fn=lambda url: (next(statuses), url), clock=lambda: now[0], sleep=sleep)
    assert seconds == 6.5  # started at 4 s, two 0.25 s waits after 10 s
    assert experiment.seconds_to_200("http://x/health", 0, 1, probe_fn=lambda url: (0, url),
                                     clock=lambda: now[0], sleep=sleep) is None


# --------------------------------------------------------------------------- report

BOOTS = [
    {"label": "seed", "variant": "", "seconds": 40.0},
    {"label": "with-1", "variant": "with", "seconds": 9.0},
    {"label": "without-2", "variant": "without", "seconds": 12.5},
    {"label": "without-3", "variant": "without", "seconds": 11.5},
    {"label": "with-4", "variant": "with", "seconds": 10.0},
]


def _builds(with_verdict: str = "ok") -> dict:
    build = {"image_bytes": 320 * MB, "delta_change": "code-only", "app_layer_instruction": "COPY /app /app"}
    return {
        "with": {**build, "app_layer_bytes": 11 * MB, "delta_bytes": 11 * MB, "verdict": with_verdict,
                 "verdict_message": "within its bar" if with_verdict == "ok" else "5.0 MB over its 6.0 MB bar"},
        "without": {**build, "app_layer_bytes": 5 * MB, "delta_bytes": 5 * MB, "verdict": "ok",
                    "verdict_message": "within its bar"},
    }


def test_medians_count_the_measured_boots_only():
    assert experiment.medians(BOOTS) == {"with": 9.5, "without": 12.0}
    assert experiment.medians(BOOTS[:2]) == {"with": 9.0, "without": None}
    failed = [*BOOTS, {"label": "with-5", "variant": "with", "seconds": None}]
    assert experiment.medians(failed)["with"] is None


@pytest.mark.parametrize(("gain", "verdict", "starts"), [
    (None, "ok", "no outcome"),
    (1.99, "ok", "W5's push removes the compile step"),
    (2.0, "ok", "the bytecode stays"),
    (2.5, "error", "Gerard chooses in W5's push"),
])
def test_the_outcome_follows_section_9_q7(gain, verdict, starts):
    assert experiment.outcome(gain, {"verdict": verdict}).startswith(starts)


def test_the_report_gives_the_gain_the_bytecodes_share_and_the_outcome(tmp_path):
    summary = experiment.summarise(_builds("error"), BOOTS)
    assert (summary["gain_seconds"], summary["bytecode_bytes"]) == (2.5, 6 * MB)
    text = experiment.report_markdown(summary)
    assert "| New container to /health 200, median | 9.50 s (9.00, 10.00) | 12.00 s (12.50, 11.50) |" in text
    assert "Gain: **2.50 s**." in text and "The bytecode adds 6.0 MB to the app layer" in text
    assert "**Outcome:** Gerard chooses in W5's push" in text


def test_the_report_command_writes_its_record_and_fails_without_a_gain(tmp_path):
    layers, boots, out, summary = (tmp_path / name for name in ("layers.json", "boots.jsonl", "r.json", "s.md"))
    layers.write_text(json.dumps(_builds()), encoding="utf-8")
    boots.write_text("\n".join(json.dumps(record) for record in BOOTS) + "\n", encoding="utf-8")
    args = ["report", "--layers", str(layers), "--boots", str(boots), "--out", str(out), "--summary", str(summary)]
    assert experiment.main(args) == 0
    assert json.loads(out.read_text())["outcome"].startswith("the bytecode stays")
    boots.write_text(json.dumps(BOOTS[1]) + "\n", encoding="utf-8")
    assert experiment.main(args) == 1
    boots.unlink()
    assert experiment.main(args) == 1


# --------------------------------------------------------------------------- the workflow and the boot script

STEPS = WORKFLOW["jobs"]["bytecode"]["steps"]


def _index(predicate) -> int:
    (found,) = [i for i, step in enumerate(STEPS) if predicate(step)]
    return found


def _build(tag: str) -> int:
    return _index(lambda step: (step.get("with") or {}).get("tags") == "${{ env.REPO }}:" + tag)


def test_it_runs_on_its_own_files_and_by_hand():
    triggers = WORKFLOW.get("on", WORKFLOW.get(True))  # PyYAML reads the `on` key as True
    assert set(triggers) == {"pull_request", "workflow_dispatch"}
    assert set(triggers["pull_request"]["paths"]) == {
        ".github/workflows/bytecode-experiment.yml", "scripts/ci/bytecode_experiment.py",
        "scripts/ci/bytecode-experiment.sh"}


def test_the_four_builds_are_images_yml_api_build_pushed_to_the_jobs_registry():
    assert WORKFLOW["jobs"]["bytecode"]["env"]["REPO"] == "localhost:5000/automatos-api"
    for tag in ("with", "with-edit", "without", "without-edit"):
        build = STEPS[_build(tag)]["with"]
        assert (build["context"], build["target"], build["platforms"]) == ("orchestrator", "production", "linux/amd64")
        assert "build-args" not in build, "images.yml's api build passes none since PRD-262 US-027"
        assert build["push"] is True and build["cache-from"] == "type=gha,scope=api-amd64"
        assert "cache-to" not in build, "a pull request reads the cache and never writes it (§9-Q13)"


def test_the_checkout_is_edited_between_the_builds_and_restored_before_the_boots():
    edit = "printf '\\n# PRD-262 US-032 bytecode experiment, a code-only change\\n' >> orchestrator/main.py"
    edits = [i for i, step in enumerate(STEPS) if step.get("run", "").strip() == edit]
    strip = _index(lambda step: "strip-compile orchestrator/Dockerfile" in step.get("run", ""))
    restore = _index(lambda step: step.get("run", "").strip()
                     == "git checkout -- orchestrator/main.py orchestrator/Dockerfile")
    boots = _index(lambda step: step.get("run") == "bash scripts/ci/bytecode-experiment.sh")
    assert _build("with") < edits[0] < _build("with-edit") < strip < _build("without") < edits[1]
    assert edits[1] < _build("without-edit") < restore < boots
    assert STEPS[strip]["run"].startswith("git checkout -- orchestrator/main.py\n")


def test_the_report_runs_even_when_a_boot_fails():
    report = _index(lambda step: "bytecode_experiment.py report" in step.get("run", ""))
    assert STEPS[report]["if"] == "${{ !cancelled() }}"
    assert '--summary "$GITHUB_STEP_SUMMARY"' in STEPS[report]["run"]


def test_the_boots_are_new_containers_with_one_worker_in_a_balanced_order():
    assert "compose up -d --no-deps --no-build --force-recreate backend" in SHELL
    assert not re.search(r"\b(?:compose|docker) restart\b", SHELL), "a restart keeps the first boot's .pyc"
    assert 'WEB_CONCURRENCY: "1"' in SHELL
    assert "ORDER=(with without without with)" in SHELL
    assert "compose up -d --wait --no-build postgres redis minio" in SHELL
    assert 'boot "" seed' in SHELL and 'boot "" settle' in SHELL
    assert "volumes: !override" in SHELL and "./orchestrator:/app" not in SHELL


def test_the_local_workspaces_mount_point_exists_before_the_first_boot():
    """The boots mount the workspaces volume read-only with a folder nested in it, and run
    without the worker that creates that mount point in the default stack: on the fresh
    volume Docker refused the first one (CI, 20 Oct: "read-only file system"). A one-off
    backend under the core override, where the volume is writable, creates it first."""
    assert 'CORE_OVERRIDE="docker-compose.core.yml"' in SHELL
    step = re.search(r"^make_workspace_mountpoint\(\) \{\n(.*?)^\}", SHELL, re.M | re.S)
    assert step, "bytecode-experiment.sh has no make_workspace_mountpoint() function"
    assert 'compose -f "$CORE_OVERRIDE" run --rm -T --no-deps --entrypoint true backend' in step.group(1)
    main = SHELL[SHELL.index("\nmain() {"):]
    order = [main.index(call) for call in ("compose up -d --wait --no-build postgres redis minio",
                                           "make_workspace_mountpoint", 'boot "" seed')]
    assert order == sorted(order)
    core = yaml.safe_load((ROOT / "docker-compose.core.yml").read_text(encoding="utf-8"))
    assert {"type": "volume", "source": "workspaces_data", "target": "/workspaces",
            "read_only": False} in core["services"]["backend"]["volumes"]
