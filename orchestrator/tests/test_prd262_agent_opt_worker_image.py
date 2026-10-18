"""PRD-262 US-006: agent-opt-worker answers only the platform, from a locked image.

The `agent-opt-worker` job in test.yml builds the image and runs the service's
own TestClient suite inside it. This suite runs in the required orchestrator job
and pins what a later edit could quietly lose:

- the dependencies: requirements.in plus a hash-locked requirements.lock, the
  PyPI package `fi` kept out by overrides.txt, plain uvicorn, starlette past
  GHSA-86qp-5c8j-p5mr;
- the Dockerfile: production is the last stage (Railway builds no target), no
  BuildKit cache mount (Railway refuses one without a service id), the lock
  installed with --require-hashes, the user created before the code is copied in;
- main.py: no API docs, no /test route, every route but /health on the token router;
- the CI lane that builds and tests the image;
- images.yml builds, measures and scans it for amd64 and never publishes it, and
  litellm's three Criticals are allowlisted against the version the lock installs.

Pure file reads: no Docker, no database.
"""
from __future__ import annotations

import ast
import json
import re
from datetime import date, timedelta
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "services" / "agent-opt-worker"
DOCKERFILE = (WORKER / "Dockerfile").read_text()
# The build instructions alone: a comment may name what the file no longer does.
INSTRUCTIONS = "\n".join(line for line in DOCKERFILE.splitlines() if not line.lstrip().startswith("#"))
MAIN = (WORKER / "main.py").read_text()
LOCK = (WORKER / "requirements.lock").read_text()
REQUIREMENTS_IN = (WORKER / "requirements.in").read_text()
PIN = re.compile(r"^([A-Za-z0-9._-]+)==([^\s\\]+)", re.M)


def _pins() -> dict[str, str]:
    return {name.lower().replace("_", "-"): version for name, version in PIN.findall(LOCK)}


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", text)[:3])


def test_the_requirements_are_an_input_and_a_hash_lock():
    assert not (WORKER / "requirements.txt").exists()
    assert "--generate-hashes" in LOCK.splitlines()[1]
    blocks = re.split(r"\n(?=[A-Za-z0-9._-]+==)", LOCK)
    pinned = [block for block in blocks if PIN.match(block)]
    assert pinned, "the lock pins nothing"
    unhashed = [block.split()[0] for block in pinned if "--hash=sha256:" not in block]
    assert not unhashed, f"pins without a hash: {unhashed}"


def test_the_pypi_package_fi_is_kept_out():
    assert not re.search(r"^fi==", LOCK, re.M), "the lock installs the PyPI package fi"
    overrides = (WORKER / "overrides.txt").read_text()
    assert re.search(r"^fi\s*;\s*sys_platform\s*==\s*'never'", overrides, re.M)
    assert "--overrides services/agent-opt-worker/overrides.txt" in LOCK.splitlines()[1]
    # The modules the worker imports come from these two distributions.
    assert {"agent-opt", "ai-evaluation"} <= set(_pins())


def test_uvicorn_is_plain_and_starlette_is_past_the_host_header_advisory():
    pins = _pins()
    assert re.search(r"^uvicorn==", REQUIREMENTS_IN, re.M)
    assert "uvicorn[" not in REQUIREMENTS_IN
    for extra in ("uvloop", "httptools", "watchfiles", "websockets"):
        assert extra not in pins, f"{extra} is back in the lock (uvicorn[standard])"
    assert _version(pins["starlette"]) >= (1, 0, 1), pins["starlette"]
    assert "fastapi" in pins


def test_production_is_the_last_stage_and_there_is_a_test_stage():
    stages = re.findall(r"^FROM\s+\S+(?:\s+AS\s+(\S+))?", DOCKERFILE, re.M | re.I)
    assert stages[-1].lower() == "production"
    assert "test" in [s.lower() for s in stages]


def test_no_buildkit_cache_mount():
    assert "--mount=type=cache" not in DOCKERFILE


def test_the_lock_is_installed_with_hashes_and_no_resolution():
    install = re.search(r"^RUN pip install[^\n]*requirements\.lock", DOCKERFILE, re.M)
    assert install, "the image does not install requirements.lock"
    assert "--require-hashes" in install.group(0) and "--no-deps" in install.group(0)
    assert "requirements.txt" not in DOCKERFILE


def test_the_user_exists_before_the_code_is_copied_in_its_name():
    assert "chown -R" not in INSTRUCTIONS
    useradd = INSTRUCTIONS.index("useradd -m -u 1000 worker")
    copy = INSTRUCTIONS.index("COPY --chown=1000:1000 . .")
    assert useradd < copy
    assert INSTRUCTIONS.index("USER worker") > copy


def test_newer_dockerfile_flags_carry_the_syntax_line():
    if re.search(r"--start-interval|COPY --chmod|COPY --link", DOCKERFILE):
        assert DOCKERFILE.startswith("# syntax=docker/dockerfile:1")


def test_main_serves_no_docs_and_keeps_only_health_off_the_token_router():
    tree = ast.parse(MAIN)
    app_call = next(
        node.value for node in tree.body
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "app" for t in node.targets)
    )
    keywords = {kw.arg: kw.value for kw in app_call.keywords}
    for name in ("docs_url", "redoc_url", "openapi_url"):
        assert isinstance(keywords.get(name), ast.Constant) and keywords[name].value is None, name
    app_routes = re.findall(r"^@app\.(?:get|post|put|patch|delete|api_route)\(\s*\"([^\"]+)\"", MAIN, re.M)
    assert app_routes == ["/health"]
    router_routes = set(re.findall(r"^@router\.(?:get|post|put|patch|delete)\(\s*\"([^\"]+)\"", MAIN, re.M))
    assert router_routes == {"/assess", "/safety", "/score", "/optimize", "/optimize/{job_id}"}
    assert '"/test"' not in MAIN
    assert "Depends(require_internal_token(internal_token()))" in MAIN
    last = tree.body[-1]
    assert ast.unparse(last) == "app.include_router(router)"


def test_the_ci_lane_builds_and_tests_the_image():
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "test.yml").read_text())
    changes = workflow["jobs"]["changes"]
    match = next(step for step in changes["steps"] if step.get("id") == "match")
    assert match["env"]["AGENT_OPT_PATHS"].startswith("^services/agent-opt-worker/")
    assert "agent_opt" in changes["outputs"]
    job = workflow["jobs"]["agent-opt-worker"]
    assert job["needs"] == "changes"
    assert job["if"] == "needs.changes.outputs.agent_opt == 'true'"
    script = "\n".join(step.get("run", "") for step in job["steps"])
    for needle in ("'^fi=='", "--target test", "-m pytest", "import fi.evals, fi.opt.optimizers",
                   "AGENT_OPT_WORKER_TOKEN is not set"):
        assert needle in script, needle


def _images_workflow() -> tuple[dict, dict]:
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "images.yml").read_text())
    pick = next(step for step in workflow["jobs"]["changes"]["steps"] if step.get("id") == "pick")
    image = next(item for item in json.loads(pick["env"]["IMAGES"]) if item["name"] == "agent-opt-worker")
    return workflow, image


def test_images_yml_builds_and_scans_it_on_amd64_and_never_publishes_it():
    workflow, image = _images_workflow()
    assert (image["context"], image["target"]) == ("services/agent-opt-worker", "production")
    assert image["arches"] == ["amd64"] and image["publish"] is False
    assert image["inputs"] == ["services/agent-opt-worker/**"]
    triggers = workflow.get("on", workflow.get(True))  # PyYAML reads the `on` key as True
    assert "services/agent-opt-worker/**" in triggers["pull_request"]["paths"]
    merged = [entry["name"] for entry in workflow["jobs"]["merge"]["strategy"]["matrix"]["image"]]
    assert "agent-opt-worker" not in merged, "never published to GHCR (PRD-262 D1, §9-Q4)"
    budgets = json.loads((ROOT / ".github" / "image-budgets.json").read_text())
    assert budgets["budgets"]["agent-opt-worker"]["amd64"]["reason"].strip()


def test_litellms_criticals_are_allowlisted_for_the_version_the_lock_installs():
    """litellm stays at 1.80.0 while ai-evaluation needs openai<2; a bump past it must drop these."""
    allowlist = yaml.safe_load((ROOT / ".grype.yaml").read_text())["ignore"]
    litellm = [entry for entry in allowlist if (entry.get("package") or {}).get("name") == "litellm"]
    assert {entry["vulnerability"] for entry in litellm} == {
        "GHSA-6wvf-77m9-58rm", "GHSA-4xpc-pv4p-pm3w", "GHSA-jjhc-v7c2-5hh6"}
    for entry in litellm:
        assert entry["package"]["version"] == _pins()["litellm"]
        assert "openai<2" in entry["reason"] and "openai>=2.8.0" in entry["reason"]
        assert entry["expires"] <= date.today() + timedelta(days=90)
