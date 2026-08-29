"""PRD-209 S3 — the boot lanes fail red on boot death (de-masked).

Both fresh-clone boot lanes reported green while broken because they ran under
``continue-on-error: true``. Now that US-001/002/003 make the boot actually reach a
green /health, the mask comes off so a future boot death fails the lane. This guard
asserts, purely (YAML-parse of the workflows + a text read of the smoke script):

* the smoke-fresh-clone lane's boot step carries NO ``continue-on-error``;
* the ``alembic-from-zero`` job's steps carry NO ``continue-on-error`` (scoped to
  that job — the eval lanes in test.yml keep their own masks, correctly);
* the smoke script exports ``DEFAULT_WORKSPACE_ID`` at the CI-seed convention value
  (``validate_auth_edition()`` hard-requires it in local mode).

It also proves the required lanes' definitions are untouched by name.
"""
from __future__ import annotations

import pathlib
import re

import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SMOKE_YML = _REPO_ROOT / ".github" / "workflows" / "smoke-fresh-clone.yml"
_TEST_YML = _REPO_ROOT / ".github" / "workflows" / "test.yml"
_SMOKE_SH = _REPO_ROOT / "scripts" / "ci" / "smoke-fresh-clone.sh"

CANONICAL_WORKSPACE_ID = "00000000-0000-0000-0000-0000000000c1"


def _load(path: pathlib.Path) -> dict:
    # PyYAML parses the `on:` key as boolean True; that's fine, we read `jobs`.
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _steps_with_mask(job: dict) -> list[str]:
    return [
        s.get("name", "<unnamed>")
        for s in job.get("steps", [])
        if s.get("continue-on-error") is True
    ]


def test_smoke_lane_boot_step_is_hard():
    data = _load(_SMOKE_YML)
    offenders = []
    for job_name, job in data["jobs"].items():
        offenders += [f"{job_name}:{s}" for s in _steps_with_mask(job)]
    assert not offenders, (
        f"smoke-fresh-clone.yml still masks boot death with continue-on-error: {offenders}"
    )


def test_alembic_from_zero_steps_are_hard():
    data = _load(_TEST_YML)
    job = data["jobs"].get("alembic-from-zero")
    assert job is not None, "alembic-from-zero job disappeared from test.yml"
    masked = _steps_with_mask(job)
    assert not masked, f"alembic-from-zero still masks steps with continue-on-error: {masked}"


def test_smoke_script_exports_default_workspace_id():
    text = _SMOKE_SH.read_text(encoding="utf-8")
    m = re.search(r"^export DEFAULT_WORKSPACE_ID=", text, re.M)
    assert m, "smoke-fresh-clone.sh must export DEFAULT_WORKSPACE_ID (local-mode requirement)"
    assert CANONICAL_WORKSPACE_ID in text, (
        f"smoke script's DEFAULT_WORKSPACE_ID must match the api.defaults / CI-seed value "
        f"{CANONICAL_WORKSPACE_ID}"
    )


def test_required_lane_definitions_present_and_untouched_by_name():
    # Non-vacuity: the required lanes still exist (we only edited the non-required
    # from-zero job). orchestrator-tests lives in test.yml; ioc-scan is its own
    # workflow — assert the one we edited still carries the required job intact.
    data = _load(_TEST_YML)
    assert "orchestrator-tests" in data["jobs"], "required orchestrator-tests job must remain"
