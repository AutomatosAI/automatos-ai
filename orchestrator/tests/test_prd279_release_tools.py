"""PRD-279 Waves 1 and 2: the scripts a `v*` tag runs.

* ``scripts/ci/release_version.py`` (US-101): the version a tag names, whether it is a
  release candidate, and the refusal of a tag that is not a version or goes backwards.
* ``scripts/ci/release_signing.py`` (US-103, US-202): the ``cosign verify`` a release
  runs pins its signer exactly, so a signature from a pull request build, a branch,
  another tag, another workflow or a fork does not pass it.
* ``scripts/ci/release_chart.py`` (US-201): the packaged chart's values name every
  image by digest and change nothing else, the web app's one slot takes the default
  edition's image, and a rendered chart that names an image by a tag fails.

Pure file reads and temporary files: no Docker, no helm, no cosign, no network.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
CHART = ROOT / "charts" / "automatos"
REPO = "AutomatosAI/automatos-ai"
SHA = "0d9f0ac594c3783e62a30e6c5ce75164028850db"
TAG = "refs/tags/v0.8.0-rc.1"
DIGESTS = {"api": "sha256:" + "a" * 64, "frontend": "sha256:" + "b" * 64, "workspace-worker": "sha256:" + "c" * 64}
PREFIX = "ghcr.io/automatosai/automatos-"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "ci" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


version = _load("release_version")
signing = _load("release_signing")
chart = _load("release_chart")


# --------------------------------------------------------------------------- the version (US-101)


@pytest.mark.parametrize("ref,expected,candidate,line", [
    ("refs/tags/v0.7.0", "0.7.0", False, "0.7"),
    ("refs/tags/v0.8.0-rc.1", "0.8.0-rc.1", True, "0.8"),
    ("refs/tags/v1.12.3-beta.2", "1.12.3-beta.2", True, "1.12"),
])
def test_a_version_tag_names_its_version_and_whether_it_is_a_candidate(ref, expected, candidate, line):
    release = version.parse_ref(ref)
    assert (release.version, release.candidate, release.major_minor) == (expected, candidate, line)
    assert version.outputs(release) == [f"version={expected}", f"candidate={str(candidate).lower()}",
                                        f"major_minor={line}"]


@pytest.mark.parametrize("ref", [
    "refs/tags/v0.7", "refs/tags/v0.7.0+build.1", "refs/tags/v01.7.0", "refs/tags/v0.7.0-",
    "refs/tags/vnext", "refs/tags/0.7.0", "refs/heads/v0.7.0", "refs/pull/12/merge", "refs/tags/v0.7.0 ",
])
def test_anything_else_is_refused(ref):
    with pytest.raises(version.ReleaseError):
        version.parse_ref(ref)


def test_a_version_below_the_charts_is_refused_and_the_same_or_above_passes():
    with pytest.raises(version.ReleaseError, match="below the chart's version 0.6.0: a release never goes backwards"):
        version.check_not_backwards(version.parse_version("0.5.9"), (0, 6, 0))
    with pytest.raises(version.ReleaseError, match="never goes backwards"):
        version.check_not_backwards(version.parse_version("0.6.0-rc.1"), (0, 7, 0))
    for allowed in ("0.6.0", "0.7.0-rc.1", "1.0.0"):
        version.check_not_backwards(version.parse_version(allowed), (0, 6, 0))


def test_the_charts_version_is_read_from_its_chart_yaml():
    core = version.chart_version((CHART / "Chart.yaml").read_text(encoding="utf-8"))
    assert len(core) == 3 and all(isinstance(part, int) for part in core)
    assert version.chart_version("apiVersion: v2\nname: x\nversion: \"1.2.3\"\nappVersion: edge\n") == (1, 2, 3)


def test_the_cli_writes_the_outputs_and_fails_on_a_bad_tag(tmp_path, capsys):
    outputs = tmp_path / "outputs"
    args = ["--chart-yaml", str(CHART / "Chart.yaml"), "--outputs", str(outputs)]
    assert version.main(["--ref", "refs/tags/v99.0.0-rc.1", *args]) == 0
    assert outputs.read_text(encoding="utf-8").splitlines() == [
        "version=99.0.0-rc.1", "candidate=true", "major_minor=99.0"]
    assert version.main(["--ref", "refs/tags/v0.0.1", *args]) == 1
    assert "::error title=Release version::" in capsys.readouterr().out


def test_the_chart_in_the_repository_keeps_edge():
    """US-201: only the packaged copy carries the version; the repository's chart follows main."""
    assert yaml.safe_load((CHART / "Chart.yaml").read_text(encoding="utf-8"))["appVersion"] == "edge"


# --------------------------------------------------------------------------- the signature check (US-103)


def _release_check(reference: str = "ghcr.io/automatosai/automatos-api@sha256:" + "d" * 64) -> list:
    return signing.verify_args(reference, signing.release_signer(REPO, TAG, SHA))


def _certificate(**changes) -> "signing.Signer":
    claims = {"repository": REPO, "workflow": ".github/workflows/images.yml", "ref": TAG, "sha": SHA,
              "workflow_name": "images", **changes}
    return signing.Signer(**claims)


def test_the_check_pins_identity_and_issuer_exactly():
    args = _release_check()
    assert args[:2] == ["cosign", "verify"] and args[-1].endswith("@sha256:" + "d" * 64)
    pinned = dict(zip(args[2:-1:2], args[3:-1:2]))
    assert pinned["--certificate-identity"] == (
        "https://github.com/AutomatosAI/automatos-ai/.github/workflows/images.yml@refs/tags/v0.8.0-rc.1")
    assert pinned["--certificate-oidc-issuer"] == "https://token.actions.githubusercontent.com"
    assert pinned["--certificate-github-workflow-repository"] == REPO
    assert pinned["--certificate-github-workflow-ref"] == TAG
    assert pinned["--certificate-github-workflow-sha"] == SHA
    assert not [arg for arg in args if arg.endswith("-regexp")], "never a pattern"


def test_the_release_run_passes_its_own_check():
    assert signing.accepts(_release_check(), _certificate())


@pytest.mark.parametrize("who,changes", [
    ("a pull request build", {"ref": "refs/pull/1577/merge"}),
    ("main", {"ref": "refs/heads/main"}),
    ("another branch", {"ref": "refs/heads/feat/prd279-w1-releases"}),
    ("another tag", {"ref": "refs/tags/v0.8.0-rc.2"}),
    ("another workflow", {"workflow": ".github/workflows/smoke-images.yml", "workflow_name": "smoke-images"}),
    ("another workflow under the same name", {"workflow": ".github/workflows/other.yml"}),
    ("a fork", {"repository": "someone/automatos-ai"}),
    ("another commit", {"sha": "1" * 40}),
    ("another issuer", {"issuer": "https://accounts.example.com"}),
])
def test_a_signature_from_anywhere_else_does_not_pass_the_release_check(who, changes):
    assert not signing.accepts(_release_check(), _certificate(**changes)), who


def test_a_check_by_pattern_or_with_a_claim_left_out_is_refused():
    loose = [arg.replace("--certificate-identity", "--certificate-identity-regexp") for arg in _release_check()]
    with pytest.raises(ValueError, match="pins the identity exactly"):
        signing.accepts(loose, _certificate())
    with pytest.raises(ValueError, match="unpinned"):
        signing.accepts(_release_check()[:4] + _release_check()[-1:], _certificate())


def test_the_check_runs_on_a_digest_never_a_tag():
    with pytest.raises(ValueError, match="verify a digest"):
        _release_check("ghcr.io/automatosai/automatos-api:0.8.0-rc.1")


def test_verify_runs_cosign_and_a_failed_check_fails(capsys):
    ran = []
    args = SimpleNamespace(image="ghcr.io/automatosai/automatos-api@sha256:" + "d" * 64,
                           repository=REPO, ref=TAG, sha=SHA)
    assert signing.verify(args, run=lambda command, check: ran.append(command) or SimpleNamespace(returncode=0)) == 0
    assert ran == [_release_check()]
    assert signing.verify(args, run=lambda command, check: SimpleNamespace(returncode=1)) == 1
    assert "is not signed by https://github.com/AutomatosAI/automatos-ai/.github/workflows/images.yml@" in (
        capsys.readouterr().out)


# --------------------------------------------------------------------------- the packaged chart (US-201)


def _prepared(tmp_path: Path) -> tuple[dict, str, list]:
    references = chart.prepare(CHART, tmp_path, DIGESTS, PREFIX, "0.8.0-rc.1")
    text = (tmp_path / "automatos" / "values.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text), text, references


def test_the_packaged_values_name_every_image_by_its_digest(tmp_path):
    values, _, references = _prepared(tmp_path)
    for key, name in (("api", "api"), ("frontend", "frontend"), ("worker", "workspace-worker")):
        assert values[key]["image"]["repository"] == f"{PREFIX}{name}"
        assert values[key]["image"]["digest"] == DIGESTS[name]
    assert references == sorted(f"{PREFIX}{name}@{digest}" for name, digest in DIGESTS.items())


def test_nothing_else_in_the_values_changes_and_the_comments_stay(tmp_path):
    values, text, _ = _prepared(tmp_path)
    original_text = (CHART / "values.yaml").read_text(encoding="utf-8")
    original = yaml.safe_load(original_text)
    for key in ("api", "frontend", "worker"):
        values[key]["image"] = {**values[key]["image"], "digest": "", "repository": original[key]["image"]["repository"]}
    assert values == original
    comments = [line for line in original_text.splitlines() if line.lstrip().startswith("#")]
    assert [line for line in text.splitlines() if line.lstrip().startswith("#")][2:] == comments
    assert 'digest: "sha256:' + "a" * 64 + '"         # sha256:<64 hex>; wins over the tag' in text
    assert (CHART / "values.yaml").read_text(encoding="utf-8") == original_text, "the repository's chart is untouched"


def test_the_web_slot_takes_the_default_editions_image_and_refuses_an_edition_it_cannot_hold():
    assert chart.plan("local") == {"api": "api", "worker": "workspace-worker", "frontend": "frontend"}
    with pytest.raises(chart.ChartError, match="the editions \\['enterprise'\\] need a slot each"):
        chart.web_image("local", {"local": "frontend", "enterprise": "frontend-enterprise"})
    with pytest.raises(chart.ChartError, match="no published web image"):
        chart.web_image("saas")


def test_a_missing_or_malformed_digest_publishes_nothing(tmp_path):
    (tmp_path / "api").write_text(DIGESTS["api"] + "\n", encoding="utf-8")
    (tmp_path / "frontend").write_text("sha256:short\n", encoding="utf-8")
    with pytest.raises(chart.ChartError, match="frontend: no sha256 digest.*workspace-worker: no sha256 digest"):
        chart.read_digests(tmp_path, ["api", "frontend", "workspace-worker"])


def test_a_value_the_chart_does_not_hold_once_is_refused():
    with pytest.raises(chart.ChartError, match="exactly once: api.image.nothing"):
        chart.set_values((CHART / "values.yaml").read_text(encoding="utf-8"), {("api", "image", "nothing"): "x"})
    with pytest.raises(chart.ChartError, match="a mapping, not a value"):
        chart.set_values((CHART / "values.yaml").read_text(encoding="utf-8"), {("api", "image"): "x"})


def _rendered(images: list, version_label: str = "0.8.0-rc.1") -> list:
    labels = {"helm.sh/chart": f"automatos-{version_label}", "app.kubernetes.io/version": version_label}
    return [{"kind": "Deployment", "metadata": {"labels": labels},
             "spec": {"template": {"spec": {"containers": [{"name": "c", "image": image}]}}}} for image in images]


def test_a_render_that_runs_exactly_the_published_digests_passes():
    expected = sorted(f"{PREFIX}{name}@{digest}" for name, digest in DIGESTS.items())
    assert chart.check_rendered(_rendered(expected), expected, "0.8.0-rc.1") == []


@pytest.mark.parametrize("image", [
    "ghcr.io/automatosai/automatos-api:0.8.0-rc.1",
    "ghcr.io/automatosai/automatos-api:edge",
    "ghcr.io/automatosai/automatos-api",
    "ghcr.io/automatosai/automatos-api:0.8@sha256:" + "a" * 64,
])
def test_a_render_with_an_image_named_by_a_tag_fails(image):
    expected = sorted(f"{PREFIX}{name}@{digest}" for name, digest in DIGESTS.items())
    problems = chart.check_rendered(_rendered([*expected[1:], image]), expected, "0.8.0-rc.1")
    assert f"named by a tag, not a digest: {image}" in problems
    assert f"published but not run by the chart: {expected[0]}" in problems


def test_a_render_with_another_image_or_the_wrong_version_fails():
    expected = sorted(f"{PREFIX}{name}@{digest}" for name, digest in DIGESTS.items())
    stranger = "ghcr.io/someone/else@sha256:" + "e" * 64
    assert chart.check_rendered(_rendered([*expected, stranger]), expected, "0.8.0-rc.1") == [
        f"run by the chart but not published by this release: {stranger}"]
    problems = chart.check_rendered(_rendered(expected, "edge"), expected, "0.8.0-rc.1")
    assert any(problem.startswith("label helm.sh/chart is") for problem in problems)
    assert chart.check_rendered([], expected, "0.8.0-rc.1")[-1] == "the render names no image at all"


def test_the_cli_prepares_the_chart_and_lists_its_images(tmp_path):
    digests = tmp_path / "digests"
    digests.mkdir()
    for name, digest in DIGESTS.items():
        (digests / name).write_text(digest + "\n", encoding="utf-8")
    images = tmp_path / "images.json"
    assert chart.main(["prepare", "--chart", str(CHART), "--out", str(tmp_path / "out"), "--digests", str(digests),
                       "--prefix", PREFIX, "--version", "0.8.0-rc.1", "--images", str(images)]) == 0
    assert json.loads(images.read_text(encoding="utf-8")) == sorted(f"{PREFIX}{n}@{d}" for n, d in DIGESTS.items())
    assert (tmp_path / "out" / "automatos" / "templates" / "_helpers.tpl").is_file()
