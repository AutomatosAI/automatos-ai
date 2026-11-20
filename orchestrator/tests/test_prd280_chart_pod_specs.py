"""PRD-280 US-100: the install lane checks every pod the release makes against "restricted".

``scripts/ci/chart_pod_specs.py`` turns a release's manifest and hooks (as kubectl's JSON)
into bare Pods, which ``deploy/kind/strict.sh`` creates with a server-side dry run in a
namespace that enforces the level. These tests pin the conversion: one Pod per object
that makes pods, its template's spec unchanged but for the service account, and nothing
for an object that makes none.

Pure: no cluster, no network.
"""
from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location("chart_pod_specs", _ROOT / "scripts" / "ci" / "chart_pod_specs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


specs = _load()

POD_SPEC = {
    "serviceAccountName": "demo-automatos-api",
    "automountServiceAccountToken": False,
    "securityContext": {"runAsNonRoot": True},
    "containers": [{"name": "api", "image": "automatos-api:ci",
                    "securityContext": {"readOnlyRootFilesystem": True}}],
}


def _deployment(name: str) -> dict:
    return {"kind": "Deployment", "metadata": {"name": name}, "spec": {"template": {"spec": POD_SPEC}}}


def test_a_deployment_becomes_one_pod_named_after_it_without_its_account():
    ((name, pod),) = list(specs.probe_pods(_deployment("demo-automatos-api")))

    assert name == "demo-automatos-api"
    assert pod["kind"] == "Pod" and pod["metadata"] == {"name": "demo-automatos-api"}
    assert "serviceAccountName" not in pod["spec"]
    assert pod["spec"]["containers"] == POD_SPEC["containers"]
    assert pod["spec"]["securityContext"] == {"runAsNonRoot": True}


def test_the_template_it_reads_is_left_as_it_was():
    deployment = _deployment("demo-automatos-api")

    list(specs.probe_pods(deployment))

    assert deployment["spec"]["template"]["spec"]["serviceAccountName"] == "demo-automatos-api"


def test_a_list_gives_one_pod_per_object_that_makes_pods_and_skips_the_rest():
    job = {"kind": "Job", "metadata": {"name": "demo-automatos-migrate"},
           "spec": {"template": {"spec": {**POD_SPEC, "restartPolicy": "Never"}}}}
    service = {"kind": "Service", "metadata": {"name": "demo-automatos-api"}, "spec": {}}
    account = {"kind": "ServiceAccount", "metadata": {"name": "demo-automatos-migrate"}}
    document = {"kind": "List", "items": [_deployment("demo-automatos-api"), service, job, account]}

    names = [name for name, _ in specs.probe_pods(document)]

    assert names == ["demo-automatos-api", "demo-automatos-migrate"]


@pytest.mark.parametrize("kind, path", [
    ("StatefulSet", ("spec", "template", "spec")),
    ("CronJob", ("spec", "jobTemplate", "spec", "template", "spec")),
    ("Pod", ("spec",)),
])
def test_every_kind_that_makes_pods_is_read_where_it_keeps_its_template(kind, path):
    obj: dict = {"kind": kind, "metadata": {"name": "probe"}}
    node = obj
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = POD_SPEC

    assert specs.pod_spec(obj) == POD_SPEC


def test_main_prints_one_tab_separated_line_per_pod(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(_deployment("demo-automatos-worker"))))

    assert specs.main() == 0
    name, pod = capsys.readouterr().out.rstrip("\n").split("\t")
    assert name == "demo-automatos-worker"
    assert json.loads(pod)["metadata"]["name"] == "demo-automatos-worker"


def test_main_reads_the_objects_kubectl_create_prints_one_after_another(monkeypatch, capsys):
    stream = json.dumps(_deployment("demo-automatos-api"), indent=4) + "\n" + json.dumps(
        {"kind": "List", "items": [_deployment("demo-automatos-worker")]}, indent=4)
    monkeypatch.setattr("sys.stdin", io.StringIO(stream))

    assert specs.main() == 0
    lines = capsys.readouterr().out.splitlines()
    assert [line.split("\t")[0] for line in lines] == ["demo-automatos-api", "demo-automatos-worker"]


def test_main_fails_when_nothing_makes_a_pod(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"kind": "List", "items": []})))

    assert specs.main() == 1
    assert "no object that makes a pod" in capsys.readouterr().err
