"""PRD-279 US-203: the chart is checked against each supported Kubernetes version.

``.github/workflows/chart.yml`` renders charts/automatos and checks the manifests with
kubeconform against the oldest and the newest Kubernetes the chart supports, not one.
``charts/automatos/README.md`` states the same two minors, and this test fails when the
README and the workflow disagree.

Pure file reads: no helm, no kubeconform, no network.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CHART_WORKFLOW = yaml.safe_load((ROOT / ".github" / "workflows" / "chart.yml").read_text(encoding="utf-8"))
README = (ROOT / "charts" / "automatos" / "README.md").read_text(encoding="utf-8")
STATED = re.compile(r"Kubernetes \*\*(\d+\.\d+)\*\* \(the\s+oldest it supports\) and \*\*(\d+\.\d+)\*\* \(the newest\)")


def _minor(version: str) -> tuple:
    major, minor, _patch = (int(part) for part in str(version).split("."))
    return major, minor


def _workflow_versions() -> tuple:
    env = CHART_WORKFLOW["env"]
    return env["KUBERNETES_OLDEST"], env["KUBERNETES_NEWEST"]


def test_the_workflow_checks_an_oldest_and_a_newer_newest():
    oldest, newest = _workflow_versions()
    assert _minor(oldest) < _minor(newest)
    assert "KUBERNETES_VERSION" not in CHART_WORKFLOW["env"], "one version, not two, is what this replaced"


def test_every_render_is_checked_against_both():
    step = next(step for step in CHART_WORKFLOW["jobs"]["chart"]["steps"]
                if "kubeconform -strict" in step.get("run", ""))
    run = step["run"]
    assert 'for version in "$KUBERNETES_OLDEST" "$KUBERNETES_NEWEST"; do' in run
    assert '--kube-version "$version"' in run and '-kubernetes-version "$version"' in run
    assert "set -o pipefail" in run, "a failed render must fail the step, not feed kubeconform nothing"


def test_the_readme_states_the_same_two_versions_as_the_workflow():
    match = STATED.search(" ".join(README.split()))
    assert match, "charts/automatos/README.md no longer states the supported Kubernetes versions"
    stated = tuple(tuple(int(part) for part in minor.split(".")) for minor in match.groups())
    assert stated == tuple(_minor(version) for version in _workflow_versions())
