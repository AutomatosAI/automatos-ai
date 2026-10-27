"""PRD-262 US-039: source builds keep their cache and overlap with startup.

* ``make clean`` (which ``make up``, ``make build`` and ``make dev`` end with) prunes only
  build cache unused for a week (``--filter until=168h``). A plain ``docker builder prune
  -f`` deletes every cache record no image holds, and the frontend's npm install and the
  backend's pip install run in stages no image holds, so a rebuild after a ``git pull``
  reinstalled every dependency.
* ``make build`` starts postgres, redis and minio before it builds, so their first-run
  initdb and health gate pass while the images build.
* No Railway-built Dockerfile has a BuildKit cache mount: Railway fails one without its
  service id, and the ids are unknown (PRD-262 §10). The experiment measures the layer
  cache alone.
* CI: ``smoke-fresh-clone.yml``'s source-build-cache job runs
  ``scripts/ci/build-cache-experiment.sh``: a build, a plain prune, a one-line change and
  a rebuild (recorded by ``scripts/ci/build_cache.py``), then ``make clean``, another
  change and a rebuild, which must leave both installs CACHED. The fresh-clone smoke runs
  ``make build`` itself, and ``scripts/ci/smoke_build_overlap.py`` reads Docker's events:
  postgres must turn healthy before the backend image's tag event, the moment its build
  finished.

The experiment script runs here against a fake docker and make, in a copy of the
checkout's layout, so the repository's own sources are never touched.
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
_MAKEFILE = _REPO / "Makefile"
_GUIDE = _REPO / "docs" / "getting-started" / "self-hosting.md"
_SMOKE = _REPO / "scripts" / "ci" / "smoke-fresh-clone.sh"
_EXPERIMENT = _REPO / "scripts" / "ci" / "build-cache-experiment.sh"
_WORKFLOW = _REPO / ".github" / "workflows" / "smoke-fresh-clone.yml"
_WEEK_PRUNE = "docker builder prune -f --filter until=168h"
NS = 1_000_000_000
T0 = 1_792_100_000  # a whole epoch second


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / "scripts" / "ci" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # before exec, as the scripts register their siblings
    spec.loader.exec_module(module)
    return module


cache = _load("build_cache")
overlap = _load("smoke_build_overlap")
boot = _load("smoke_boot")


def _recipe(target: str) -> list:
    text = _MAKEFILE.read_text(encoding="utf-8")
    body = text.split(f"\n{target}:\n", 1)[1].split("\n\n", 1)[0]
    return [line.strip().lstrip("-@") for line in body.splitlines() if line.startswith("\t")]


def _function(text: str, name: str) -> str:
    match = re.search(rf"^{name}\(\) \{{\n(.*?)^\}}", text, re.M | re.S)
    assert match, f"no {name}() function"
    return match.group(1)


def _steps(text: str) -> list:
    return [line.strip() for line in _function(text, "main").splitlines()]


# --------------------------------------------------------------------------- the Makefile


def test_make_build_starts_the_datastores_before_it_builds():
    recipe = _recipe("build")
    order = ["$(STACK_IMAGES) build", "$(COMPOSE) up -d postgres redis minio",
             "$(COMPOSE) build backend frontend workspace-worker", "$(COMPOSE) up -d --no-build --remove-orphans"]
    assert [recipe.index(line) for line in order] == sorted(recipe.index(line) for line in order)
    assert recipe[0] == "$(MAKE) --no-print-directory env", "make env stays first (PRD-262 US-037)"


def test_make_clean_keeps_a_week_of_build_cache_and_never_prunes_images_in_use():
    recipe = " ".join(_recipe("clean"))
    assert f"{_WEEK_PRUNE} >/dev/null;" in recipe
    assert "docker builder prune -f >/dev/null" not in recipe, "a plain prune drops the dependency installs"
    assert "docker image prune -f >/dev/null;" in recipe
    assert not re.search(r"prune[^;]*(?:\s-a\b|--all\b)", recipe), "never every image or every cache record"
    assert "volume prune" not in recipe


def test_every_target_that_builds_ends_with_make_clean():
    for target in ("up", "build", "dev"):
        assert "$(MAKE) --no-print-directory clean" in _recipe(target), target


# Railway builds these with its own BuildKit, which fails a cache mount whose id is not
# `s/<service-id>-<path>`. The service ids are unknown (PRD-262 §10), so the experiment
# measures the layer cache alone and none of them gets a cache mount until they are known.
_RAILWAY_DOCKERFILES = ("orchestrator/Dockerfile", "frontend/Dockerfile", "services/workspace-worker/Dockerfile",
                        "services/media-render/Dockerfile", "services/agent-opt-worker/Dockerfile",
                        "services/voice-tts/Dockerfile")


@pytest.mark.parametrize("dockerfile", _RAILWAY_DOCKERFILES)
def test_no_railway_built_dockerfile_has_a_cache_mount(dockerfile):
    code = [line for line in (_REPO / dockerfile).read_text(encoding="utf-8").splitlines()
            if not line.lstrip().startswith("#")]  # a comment may name the rule
    mounts = [line for line in code if re.search(r"--mount=\S*type=cache", line)]
    assert mounts == [], f"{dockerfile}: no cache mount until §10's service id is known"


def test_the_guide_gives_the_week_and_the_datastores_first():
    guide = _GUIDE.read_text(encoding="utf-8")
    section = guide[guide.index("## 1b. "):]
    section = section[:section.index("\n## ", 1)]
    assert section.count(_WEEK_PRUNE) == 2, "make up's tidy-up and make clean"
    assert "docker builder prune -f\n" not in section
    build = section[section.index("# make build:"):section.index("# make dev:")]
    assert build.index("docker compose up -d postgres redis minio") < build.index("docker compose build")
    keeping = guide[guide.index("### Keeping it small"):]
    keeping = keeping[:keeping.index("\n### ", 1)]
    assert f"docker image prune -f && {_WEEK_PRUNE}" in keeping
    assert "npm install" in keeping and "pip install" in keeping, "the guide says why the week is kept"


# --------------------------------------------------------------------------- build_cache.py


def _log(npm: str, pip: str, npm_command: str = "npm install --legacy-peer-deps",
         pip_stage: str = "pybuild 5/8") -> str:
    """A piped `docker compose build backend frontend`, its install steps ending in ``npm`` and ``pip``."""
    pip_run = 'RUN pip install --no-cache-dir --prefix=/install -r requirements.txt'
    return "\n".join([
        "#1 [internal] load local bake definitions",
        "#1 DONE 0.0s",
        "#12 [backend internal] load build context",
        "#12 transferring context: 19.81MB 1.1s done",
        "#12 DONE 1.2s",
        f"#14 [frontend builder 4/8] RUN {npm_command}",
        f"#15 [backend {pip_stage}] {pip_run}",
        "#14 12.3 npm warn deprecated inflight@1.0.6",
        f"#15 {pip}",
        f"#14 [frontend builder 4/8] RUN {npm_command}",
        f"#14 {npm}",
        "#20 [frontend builder 8/8] RUN npm run build",
        "#20 DONE 90.2s",
        "#24 [backend production 9/9] COPY --chown=1000:1000 . .",
        "#24 DONE 1.0s",
    ]) + "\n"


def test_both_installs_cached_pass_the_make_clean_arm():
    record = cache.check(_log("CACHED", "CACHED"), "make clean", expect_cached=True)
    assert record["failures"] == []
    assert record["installs"]["npm install"]["step"] == "[frontend builder 4/8] RUN npm install --legacy-peer-deps"
    assert {state["state"] for state in record["installs"].values()} == {cache.CACHED}


def test_an_install_that_ran_again_fails_the_make_clean_arm_and_is_only_recorded_after_a_plain_prune():
    log = _log("DONE 238.3s", "CACHED")
    record = cache.check(log, "make clean", expect_cached=True)
    assert record["failures"] == ["npm install after make clean: ran (238.3 s), not CACHED"]
    plain = cache.check(log, "docker builder prune -f", expect_cached=False)
    assert plain["failures"] == []
    assert plain["installs"]["npm install"] == {
        "step": "[frontend builder 4/8] RUN npm install --legacy-peer-deps", "state": cache.RAN, "seconds": 238.3}


def test_a_failed_or_missing_install_is_not_cached():
    assert cache.install_states(_log("ERROR: process did not complete", "CACHED"))["npm install"]["state"] == cache.FAILED
    no_pip = "".join(line + "\n" for line in _log("CACHED", "CACHED").splitlines() if "#15" not in line)
    record = cache.check(no_pip, "make clean", expect_cached=True)
    assert record["installs"]["pip install"]["state"] == cache.MISSING
    assert record["failures"][0].startswith("pip install after make clean: not in the log")


def test_the_installs_are_found_in_any_stage_and_as_npm_ci_or_uv_pip():
    """npm ci moves to the base stage with the lock (US-047); the API's pip may run through uv."""
    log = _log("CACHED", "CACHED", npm_command="npm ci --no-audit --no-fund", pip_stage="pybuild 6/9")
    log = log.replace("[frontend builder 4/8]", "[frontend base 4/4]").replace("pip install", "uv pip install")
    states = cache.install_states(log)
    assert states["npm install"]["step"] == "[frontend base 4/4] RUN npm ci --no-audit --no-fund"
    assert {state["state"] for state in states.values()} == {cache.CACHED}


def test_an_install_of_another_service_or_another_command_does_not_count():
    assert cache.install_named("[workspace-worker production 5/9] RUN pip install --no-cache-dir uv") is None
    assert cache.install_named("[frontend builder 8/8] RUN npm run build") is None
    assert cache.install_named("[backend pybuild 2/8] RUN apt-get update") is None


def test_the_command_writes_the_record_and_the_summary(tmp_path):
    log, out, summary = tmp_path / "rebuild.log", tmp_path / "rebuild.json", tmp_path / "summary.md"
    log.write_text(_log("DONE 238.3s", "CACHED"), encoding="utf-8")
    assert cache.main([str(log), "--arm", "docker builder prune -f", "--out", str(out), "--summary", str(summary)]) == 0
    assert cache.main([str(log), "--arm", "make clean", "--expect-cached", "--out", str(out),
                       "--summary", str(summary)]) == 1
    assert json.loads(out.read_text(encoding="utf-8"))["expect_cached"] is True
    text = summary.read_text(encoding="utf-8")
    assert "### Source rebuild after `docker builder prune -f` (PRD-262 US-039)" in text
    assert "| npm install | `[frontend builder 4/8] RUN npm install --legacy-peer-deps` | ran (238.3 s) |" in text
    assert "Recorded only" in text and "**FAIL**: npm install after make clean" in text


# --------------------------------------------------------------------------- the experiment


def test_the_experiment_prunes_plainly_then_runs_make_clean_and_needs_cached_only_after_it():
    text = _EXPERIMENT.read_text(encoding="utf-8")
    steps = _steps(text)
    order = ["refuse_pruning_a_kept_build_cache", "refuse_changing_uncommitted_sources", "trap restore_sources EXIT",
             "build first-build",
             'docker builder prune -f || fail "${PLAIN_PRUNE} failed"', 'change_a_line "$PLAIN_PRUNE"',
             "build after-plain-prune", 'make clean || fail "make clean failed"', 'change_a_line "make clean"',
             "build after-make-clean"]
    assert [steps.index(step) for step in order] == sorted(steps.index(step) for step in order)
    plain = next(step for step in steps if step.startswith("read_installs after-plain-prune"))
    assert "--expect-cached" not in plain
    cleaned = next(step for step in steps if step.startswith("read_installs after-make-clean"))
    assert "--expect-cached" in cleaned and steps.index(cleaned) > steps.index("build after-make-clean")
    assert "--mount=type=cache" not in text and "cache-from" not in text, "the layer cache only (PRD-262 §10)"
    assert re.search(r"^SERVICES=\(backend frontend\)$", text, re.M)
    for source in ("orchestrator/main.py", "frontend/app/layout.tsx"):
        assert f'"{source}"' in text and (_REPO / source).is_file(), source


# A docker that records its calls, prints the next prepared build log for each
# `compose build`, with the line each source ended on, and a make that records its call.
_FAKE_DOCKER = """#!/bin/bash
echo "docker $*" >> "$CALL_LOG"
case "$*" in
  "compose build backend frontend")
    n=$(grep -c '^docker compose build' "$CALL_LOG")
    echo "backend source ends: $(tail -n 1 orchestrator/main.py)" >> "$CALL_LOG"
    cat "$FAKE_LOGS/$n.log" ;;
esac
exit 0
"""
_FAKE_MAKE = """#!/bin/bash
echo "make $*" >> "$CALL_LOG"
exit 0
"""


def _checkout(tmp_path: Path) -> Path:
    """The experiment, its two tools and the two sources it edits, committed in a git repo."""
    root = tmp_path / "checkout"
    for relative in ("scripts/ci/build-cache-experiment.sh", "scripts/ci/build_cache.py", "scripts/ci/build_context.py"):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(_REPO / relative, root / relative)
    for relative, body in (("orchestrator/main.py", "app = None\n"), ("frontend/app/layout.tsx", "export {}\n")):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(body, encoding="utf-8")
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false"]
    for command in (["init", "-q"], ["add", "."], ["commit", "-q", "-m", "fixture"]):
        subprocess.run([*git, *command], cwd=root, check=True, capture_output=True)
    return root


def _run_experiment(tmp_path: Path, logs: list, uncommitted: str = "", **extra: str) -> tuple:
    """Run the experiment in a fixture checkout; ``uncommitted`` is appended to its main.py first."""
    root = _checkout(tmp_path)
    if uncommitted:
        with open(root / "orchestrator/main.py", "a", encoding="utf-8") as source:
            source.write(uncommitted)
    bin_dir, logs_dir = tmp_path / "bin", tmp_path / "logs"
    bin_dir.mkdir()
    logs_dir.mkdir()
    for index, body in enumerate(logs, start=1):
        (logs_dir / f"{index}.log").write_text(body, encoding="utf-8")
    stubs = {"docker": _FAKE_DOCKER, "make": _FAKE_MAKE, "python3": f'#!/bin/bash\nexec "{sys.executable}" "$@"\n'}
    for name, body in stubs.items():
        (bin_dir / name).write_text(body, encoding="utf-8")
        (bin_dir / name).chmod((bin_dir / name).stat().st_mode | stat.S_IXUSR)
    calls = tmp_path / "calls.log"
    calls.touch()
    env = {"PATH": f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin", "CALL_LOG": str(calls),
           "FAKE_LOGS": str(logs_dir), "BUILD_CACHE_OUT_DIR": str(tmp_path / "out"),
           "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"), **extra}
    proc = subprocess.run(["bash", str(root / "scripts/ci/build-cache-experiment.sh")], cwd=root, env=env,
                          stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60)
    return proc.returncode, calls.read_text(encoding="utf-8").splitlines(), proc.stdout + proc.stderr, root


_GIT = shutil.which("git") is not None and shutil.which("bash") is not None


@pytest.mark.skipif(not _GIT, reason="needs bash and git")
def test_outside_ci_the_experiment_refuses_to_prune_the_hosts_build_cache(tmp_path):
    code, calls, out, root = _run_experiment(tmp_path, [])
    assert code == 1, out
    assert "BUILD_CACHE_PRUNE_OK=true" in out
    assert calls == [], "nothing built, pruned or changed"
    assert (root / "orchestrator/main.py").read_text(encoding="utf-8") == "app = None\n"


@pytest.mark.skipif(not _GIT, reason="needs bash and git")
def test_the_experiment_refuses_to_discard_an_uncommitted_change_to_a_source_it_edits(tmp_path):
    """It puts the two sources back from git on exit, so it never starts on a changed one."""
    code, calls, out, root = _run_experiment(tmp_path, [], uncommitted="# mine\n", CI="true")
    assert code == 1, out
    assert "has uncommitted changes" in out
    assert calls == [], "nothing built, pruned or changed"
    assert (root / "orchestrator/main.py").read_text(encoding="utf-8") == "app = None\n# mine\n", "the change is kept"


@pytest.mark.skipif(not _GIT, reason="needs bash and git")
def test_the_experiment_records_the_plain_prune_and_passes_when_make_clean_keeps_the_installs(tmp_path):
    logs = [_log("DONE 238.3s", "DONE 98.0s"), _log("DONE 240.1s", "CACHED"), _log("CACHED", "CACHED")]
    code, calls, out, root = _run_experiment(tmp_path, logs, CI="true")
    assert code == 0, out
    assert [call for call in calls if not call.startswith("backend source ends")] == [
        "docker compose build backend frontend", "docker builder prune -f", "docker compose build backend frontend",
        "make clean", "docker compose build backend frontend"]
    ends = [call for call in calls if call.startswith("backend source ends")]
    assert ends[0] == "backend source ends: app = None"
    assert "after docker builder prune -f" in ends[1] and "after make clean" in ends[2], "a line changed before each rebuild"
    assert (root / "orchestrator/main.py").read_text(encoding="utf-8") == "app = None\n", "the sources are put back"
    assert (root / "frontend/app/layout.tsx").read_text(encoding="utf-8") == "export {}\n"
    plain = json.loads((tmp_path / "out" / "after-plain-prune.json").read_text(encoding="utf-8"))
    assert plain["installs"]["npm install"]["state"] == cache.RAN and plain["expect_cached"] is False
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "## Source build cache (PRD-262 US-039)" in summary and "Both installs CACHED: pass." in summary


@pytest.mark.skipif(not _GIT, reason="needs bash and git")
def test_the_experiment_fails_when_make_clean_lets_an_install_run_again(tmp_path):
    logs = [_log("DONE 238.3s", "DONE 98.0s"), _log("DONE 240.1s", "CACHED"), _log("DONE 236.0s", "CACHED")]
    code, _, out, root = _run_experiment(tmp_path, logs, CI="true")
    assert code == 1, out
    assert "after make clean, a rebuild ran a dependency install again" in out
    assert (root / "orchestrator/main.py").read_text(encoding="utf-8") == "app = None\n"


# --------------------------------------------------------------------------- the workflow


def _workflow() -> dict:
    return yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))


def test_the_source_build_cache_job_runs_the_experiment_and_uploads_its_logs():
    job = _workflow()["jobs"]["source-build-cache"]
    assert job["if"] == "github.event_name != 'schedule'" and job["runs-on"] == "ubuntu-24.04"
    run = next(step for step in job["steps"] if step.get("run") == "bash scripts/ci/build-cache-experiment.sh")
    checkout = next(step for step in job["steps"] if "actions/checkout" in step.get("uses", ""))
    assert checkout["with"]["persist-credentials"] is False
    upload = next(step for step in job["steps"] if "upload-artifact" in step.get("uses", ""))
    assert upload["if"] == "always()"
    assert upload["with"]["path"].rstrip("/") == run["env"]["BUILD_CACHE_OUT_DIR"].rstrip("/")


def test_the_workflow_watches_what_make_build_and_the_experiment_run():
    triggers = _workflow().get("on", _workflow().get(True))  # PyYAML reads the bare `on:` key as True
    wanted = {"Makefile", "scripts/ci/build-cache-experiment.sh", "scripts/ci/build_cache.py",
              "scripts/ci/build_context.py", "scripts/ci/smoke_build_overlap.py", "scripts/stack_images.py",
              "scripts/stack_env.sh", "scripts/stack_docker_floor.sh", ".env.example",
              "orchestrator/Dockerfile", "frontend/Dockerfile", "frontend/package-lock.json"}
    for event in ("push", "pull_request"):
        assert wanted <= set(triggers[event]["paths"]), event


# --------------------------------------------------------------------------- the fresh-clone smoke


def test_the_fresh_clone_smoke_runs_make_build_in_the_background():
    text = _SMOKE.read_text(encoding="utf-8")
    assert re.search(r'^  ""\) MAKE_TARGET="build" ;;$', text, re.M)
    build = _function(text, "build_images")
    assert build.index('up_started_ns="$(date +%s%N)"') < build.index("make build 2>&1 | tee")
    assert 'echo "${PIPESTATUS[0]}" > "$MAKE_EXIT"' in build and "make_pid=$!" in build
    assert "compose build" not in text, "the smoke builds through make build, not a compose build of its own"
    fresh = _function(text, "boot_fresh_from_make")
    assert '[ "$MAKE_TARGET" != "build" ] || start_wait="$BUILD_MAX_WAIT_SECONDS"' in fresh, \
        "make build builds three images before the backend starts"


def test_the_fresh_clone_smoke_streams_image_tag_events_and_checks_the_overlap_after_make_build():
    text = _SMOKE.read_text(encoding="utf-8")
    events = _function(text, "start_event_log")
    assert "--filter type=container --filter type=image" in events
    assert "--filter event=start --filter event=health_status --filter event=tag" in events
    steps = _steps(text)
    order = [steps.index(step) for step in ("start_event_log", "prepare_images", "finish_make",
                                             "check_health_gates", "check_build_overlap", "check_loopback_ports")]
    assert order == sorted(order)
    check = _function(text, "check_build_overlap")
    assert check.lstrip().startswith('[ "$MAKE_TARGET" = "build" ] || return 0')
    assert '"$OVERLAP_TOOL" --events "$EVENTS_FILE" --tag "$(image_tag)" --started-ns "$up_started_ns"' in check
    assert 'build_seconds="$(cat "$BUILD_SECONDS_FILE" 2>/dev/null)"' in check and "|| fail " in check
    assert 'OVERLAP_TOOL="scripts/ci/smoke_build_overlap.py"' in text
    assert '--build-seconds "$build_seconds"' in _function(text, "write_report")


def test_the_backend_waits_for_make_builds_start_on_its_own_timeout(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(boot, "wait_for_start", lambda container, timeout, stop_file: seen.append(timeout) or False)
    common = ["--label", "fresh", "--container", "automatos_backend", "--url", "health=http://x", "--wait-start",
              "--timeout", "5", "--out", str(tmp_path / "boots.jsonl")]
    assert boot.main([*common, "--start-timeout", "1200"]) == 1
    assert boot.main(common) == 1
    assert seen == [1200.0, 5.0], "the boot's own timeout is the default"


# --------------------------------------------------------------------------- smoke_build_overlap.py


def _at(seconds: float) -> int:
    return T0 * NS + round(seconds * NS)


def _container(service: str, action: str, at_s: float) -> dict:
    attributes = {"name": f"automatos_{service}", "com.docker.compose.service": service}
    return {"Type": "container", "Action": action, "Actor": {"ID": "c0ffee", "Attributes": attributes},
            "time": T0 + int(at_s), "timeNano": _at(at_s)}


def _tag(name: str, tag: str, at_s: float) -> dict:
    return {"Type": "image", "Action": "tag", "status": "tag", "id": "sha256:feed",
            "Actor": {"ID": "sha256:feed", "Attributes": {"name": f"ghcr.io/automatosai/automatos-{name}:{tag}"}},
            "time": T0 + int(at_s), "timeNano": _at(at_s)}


TAG = "local-0123abc"


def _make_build(postgres_healthy=21.0, backend_built=140.0) -> list:
    """make build: the datastores start at 8 s, the images build until 300 s, the stack starts."""
    events = [_container("postgres", "start", 8.0), _container("redis", "start", 8.0),
              _tag("workspace-worker", TAG, 100.0), _tag("frontend", TAG, 300.0),
              _tag("api", "tree-0123456789ab", 50.0), _tag("api", "local-other", 60.0)]
    if postgres_healthy is not None:
        events.append(_container("postgres", "health_status: healthy", postgres_healthy))
    if backend_built is not None:
        events.append(_tag("api", TAG, backend_built))
    return events + [_container("backend", "start", 301.0)]


def test_postgres_healthy_while_the_backend_still_builds_passes():
    record = overlap.check(_make_build(), TAG, T0 * NS)
    assert record["failures"] == []
    assert record["postgres_healthy_s"] == 21.0 and record["lead_s"] == 119.0
    assert record["images"]["backend"] == {"ref": f"ghcr.io/automatosai/automatos-api:{TAG}", "built_s": 140.0}
    assert record["build_s"] == 300.0, "the image build ends with the last of the three"


def test_postgres_healthy_after_the_backend_was_built_fails():
    record = overlap.check(_make_build(postgres_healthy=150.0), TAG, T0 * NS)
    assert record["failures"] == ["postgres turned healthy 10.0 s after the backend image was built: make build "
                                  "must start postgres, redis and minio before it builds"]


@pytest.mark.parametrize("field,problem", [
    ("backend_built", "Docker reported no image tag event for ghcr.io/automatosai/automatos-api:local-0123abc"),
    ("postgres_healthy", "postgres never reported healthy"),
])
def test_a_missing_event_fails(field, problem):
    record = overlap.check(_make_build(**{field: None}), TAG, T0 * NS)
    assert record["failures"][0].startswith(problem)
    assert record["build_s"] is None if field == "backend_built" else record["build_s"] == 300.0


def test_another_tag_of_the_backend_image_is_not_its_build():
    events = [event for event in _make_build(backend_built=None)] + [_tag("api", TAG, 140.0)]
    assert overlap.tagged_ns(events, overlap.image_refs(TAG))["backend"] == _at(140.0)
    assert overlap.tagged_ns(_make_build(backend_built=None), overlap.image_refs(TAG))["backend"] is None


def test_the_command_writes_the_record_the_seconds_and_the_summary(tmp_path):
    events, out = tmp_path / "events.jsonl", tmp_path / "overlap.json"
    seconds, summary = tmp_path / "seconds.txt", tmp_path / "summary.md"
    events.write_text("".join(json.dumps(event) + "\n" for event in _make_build()), encoding="utf-8")
    args = ["--events", str(events), "--tag", TAG, "--started-ns", str(T0 * NS), "--out", str(out),
            "--seconds-out", str(seconds), "--summary", str(summary)]
    assert overlap.main(args) == 0
    assert seconds.read_text(encoding="utf-8") == "300\n"
    assert json.loads(out.read_text(encoding="utf-8"))["lead_s"] == 119.0
    text = summary.read_text(encoding="utf-8")
    assert "### make build: the datastores start while the images build (PRD-262 US-039)" in text
    assert "| postgres healthy | 21.0 s |" in text
    assert "postgres turned healthy 119.0 s before the backend image was built: pass." in text
    events.write_text("".join(json.dumps(event) + "\n" for event in _make_build(postgres_healthy=None)),
                      encoding="utf-8")
    assert overlap.main(args) == 1
