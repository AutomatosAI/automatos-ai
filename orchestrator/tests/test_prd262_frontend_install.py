"""PRD-262 US-047: the frontend installs once, and its public settings come after.

- frontend/.npmrc sets legacy-peer-deps: the one install config the Dockerfile and
  frontend-ci share. The lock is written with it, so it holds no peer-only package, and
  `npm ci` installs the same tree in both (the peer-only @types/dom-mediacapture-record
  goes; frontend-ci's tsc baseline shows whether that changes anything).
- The Dockerfile's base stage, which the development and builder stages build from, runs
  `npm ci` right after copying package.json, package-lock.json and .npmrc. The builder
  reads its NEXT_PUBLIC_* args after `COPY . .`, just above the build, so a changed
  setting rebuilds only the bundle. Next's telemetry is off.
- No BuildKit cache mount: Railway builds this Dockerfile and fails a cache mount whose
  id lacks the frontend service's id, which is not known yet (§10). Until then the build
  RUN deletes .next/cache. FRONTEND_CACHE_BUST stays.
- frontend-ci installs with `npm ci`, so a lock out of step with package.json fails there
  as it fails in Docker.
- A PR that touches only .npmrc or next.config.js runs prod-image-parity and builds the
  frontend image (images.yml's PR paths: test_prd262_image_matrix.py).
- prod-image-parity's local build differs from the SaaS build only in its build args, and
  a step fails unless BuildKit printed CACHED for its npm ci. Both build times go to the
  run summary, report-only (§7).

File reads, plus that step's own bash run against sample BuildKit logs. No Docker, no npm.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"
DOCKERFILE = (FRONTEND / "Dockerfile").read_text()
LOCK = json.loads((FRONTEND / "package-lock.json").read_text())["packages"]
WORKFLOWS = ROOT / ".github" / "workflows"
TEST_YML = (WORKFLOWS / "test.yml").read_text()
JOBS = yaml.safe_load(TEST_YML)["jobs"]

INSTALL = "RUN npm ci --no-audit --no-fund"
INSTALL_INPUTS = "COPY package.json package-lock.json .npmrc ./"
BUILD = "RUN npm run build && rm -rf .next/cache"
LOCAL_LOG = '"$RUNNER_TEMP/parity-fe-local.log"'
CACHED_STEP = "Frontend — a build that changes only build args reuses the npm ci layer"
LOCAL_STEP = "Frontend — the local build serves /chat with no network, without sharp, libvips or TypeScript"
SAAS_STEP = "Frontend — edition baked with no edition arg must be saas"


def _instructions(text: str) -> list[str]:
    """Instructions with continuation lines joined; comment lines dropped."""
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    return [line.strip() for line in code.replace("\\\n", " ").splitlines() if line.strip()]


def _stage(name: str) -> list[str]:
    """The instructions of stage `name`, without its FROM line."""
    match = re.search(rf"^FROM\s+\S+\s+AS\s+{re.escape(name)}\s*$", DOCKERFILE, re.M | re.I)
    assert match, f"no stage named {name}"
    rest = DOCKERFILE[match.end():]
    following = re.search(r"^FROM\s", rest, re.M)
    return _instructions(rest[: following.start()] if following else rest)


def _npmrc() -> dict[str, str]:
    lines = (FRONTEND / ".npmrc").read_text().splitlines()
    pairs = [line.split("=", 1) for line in lines if line.strip() and line.lstrip()[0] not in "#;"]
    return {key.strip(): value.strip() for key, value in pairs}


def _parity_step(name: str) -> tuple[int, str]:
    steps = JOBS["prod-image-parity"]["steps"]
    found = [(i, step["run"]) for i, step in enumerate(steps) if step.get("name") == name]
    assert len(found) == 1, f"expected one prod-image-parity step named {name!r}"
    return found[0]


# ── one install config, one tree ────────────────────────────────────────────


def test_the_install_config_sets_legacy_peer_deps():
    assert _npmrc().get("legacy-peer-deps") == "true"


def test_the_lock_is_written_with_that_config():
    peer_only = sorted(key for key, entry in LOCK.items() if entry.get("peer"))
    assert peer_only == [], (
        f"{peer_only}: rewrite the lock from frontend/ with Node 24's npm "
        "(`npm install --package-lock-only`), which reads frontend/.npmrc"
    )


def test_no_install_passes_its_own_peer_flag():
    for name, text in (("frontend/Dockerfile", DOCKERFILE), (".github/workflows/test.yml", TEST_YML)):
        assert "--legacy-peer-deps" not in text, f"{name}: frontend/.npmrc is the one place for it"


def test_frontend_ci_installs_with_npm_ci_from_the_frontend_directory():
    job = JOBS["frontend-ci"]
    assert job["defaults"]["run"]["working-directory"] == "frontend"
    install = [step for step in job["steps"] if step.get("name") == "Install dependencies"]
    assert [step["run"] for step in install] == ["npm ci --no-audit --no-fund"]
    assert "out of sync with package.json today" not in TEST_YML


# ── the Dockerfile ──────────────────────────────────────────────────────────


def test_the_base_stage_installs_from_the_lock_and_the_config_alone():
    base = _stage("base")
    copies = [line for line in base if line.startswith("COPY")]
    assert copies == [INSTALL_INPUTS]
    assert base[base.index(INSTALL_INPUTS) + 1] == INSTALL
    assert "ENV NEXT_TELEMETRY_DISABLED=1" in base[: base.index(INSTALL)]


def test_the_development_and_builder_stages_reuse_that_install():
    runs = [line for line in _instructions(DOCKERFILE) if line.startswith("RUN") and "npm" in line]
    assert [line for line in runs if re.search(r"\bnpm (ci|install|i)\b", line)] == [INSTALL]
    for stage in ("development", "builder"):
        assert re.search(rf"^FROM base AS {stage}$", DOCKERFILE, re.M), stage


def test_the_builder_reads_the_public_settings_after_the_source_copy():
    builder = _stage("builder")
    public = [i for i, line in enumerate(builder) if re.match(r"(ARG|ENV) NEXT_PUBLIC_", line)]
    assert public, "the builder declares no NEXT_PUBLIC_* settings"
    copy, build = builder.index("COPY . ."), builder.index(BUILD)
    assert copy < min(public) and max(public) < build
    assert build == len(builder) - 1, "the build is the builder's last instruction"
    names = {kind: {re.match(r"\w+ (NEXT_PUBLIC_\w+)", builder[i])[1] for i in public if builder[i].startswith(kind)}
             for kind in ("ARG", "ENV")}
    assert names["ARG"] == names["ENV"], "every public build arg is set for the build"


def test_the_cache_bust_still_reruns_the_source_copy():
    builder = _stage("builder")
    bust = next(i for i, line in enumerate(builder) if line.startswith("ARG FRONTEND_CACHE_BUST="))
    assert bust < builder.index("COPY . .")


def test_no_cache_mount_until_railway_gives_the_frontend_service_id():
    assert not [line for line in _instructions(DOCKERFILE) if "--mount=type=cache" in line]
    assert BUILD in _stage("builder"), "without a .next/cache mount, the build RUN deletes the cache"


# ── CI ──────────────────────────────────────────────────────────────────────


def test_a_change_to_the_install_config_or_next_config_runs_the_parity_job():
    match = next(step for step in JOBS["changes"]["steps"] if step.get("id") == "match")
    for changed in ("frontend/.npmrc", "frontend/next.config.js"):
        assert re.search(match["env"]["PROD_IMAGE_PATHS"], changed), changed


def test_the_local_build_keeps_its_log_for_the_cached_check():
    local, run = _parity_step(LOCAL_STEP)
    cached, _ = _parity_step(CACHED_STEP)
    assert cached == local + 1
    assert run.startswith("set -o pipefail\n")
    assert "docker build --progress=plain --target production" in run
    assert f"-t parity-fe-local frontend/ 2>&1 | tee {LOCAL_LOG}" in run


def test_both_frontend_builds_report_their_time_to_the_summary():
    _, saas = _parity_step(SAAS_STEP)
    _, local = _parity_step(LOCAL_STEP)
    for run in (saas, local):
        assert "start=$SECONDS" in run and "$((SECONDS - start)) s" in run
        assert '>> "$GITHUB_STEP_SUMMARY"' in run
    assert "336-370 s before PRD-262 US-047" in saas


BUILDKIT_LOG = """\
#7 [base 3/4] COPY package.json package-lock.json .npmrc ./
#7 CACHED

#8 [base 4/4] RUN npm ci --no-audit --no-fund
{npm_ci}

#9 [builder 1/4] RUN echo "Frontend cache bust: x"
#9 CACHED
"""


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize(
    ("npm_ci", "status", "says"),
    [
        ("#8 CACHED", 0, "npm ci: CACHED (#8)"),
        ("#8 3.105 added 1005 packages in 3m\n#8 DONE 201.4s", 1, "npm ci ran again"),
        (None, 1, "shows no npm ci step"),
    ],
)
def test_the_cached_check_reads_the_buildkit_log(tmp_path, npm_ci, status, says):
    log = BUILDKIT_LOG.format(npm_ci=npm_ci) if npm_ci else "#1 [internal] load build definition\n#1 DONE 0.1s\n"
    (tmp_path / "parity-fe-local.log").write_text(log)
    _, run = _parity_step(CACHED_STEP)
    proc = subprocess.run(
        ["bash", "-e", "-c", run], env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "RUNNER_TEMP": str(tmp_path)},
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == status, proc.stdout + proc.stderr
    assert says in proc.stdout
