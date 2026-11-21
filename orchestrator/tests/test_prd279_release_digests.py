"""PRD-279 US-101, US-103, US-202: each digest a release signs is the one its own push made.

``scripts/ci/release_digests.py``:

* ``index``: the index digest comes from ``imagetools create``'s metadata file, and the
  index, read by that digest, must hold exactly the members of this run's builds;
* ``chart``: the chart digest comes from ``helm push``'s output, and the manifest it
  names must hold the packaged chart byte for byte;
* every read is by digest, and bytes that do not hash to the digest are refused.

A fake registry stands in for GHCR: no network, no Docker, no Helm.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
IMAGE = "ghcr.io/automatosai/automatos-api"
CHART = "ghcr.io/automatosai/charts/automatos"
OCI_INDEX = "application/vnd.oci.image.index.v1+json"
OCI_MANIFEST = "application/vnd.oci.image.manifest.v1+json"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "ci" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


digests = _load("release_digests")
registry = _load("image_registry")


def _sha(body: bytes) -> str:
    return "sha256:" + hashlib.sha256(body).hexdigest()


def _index(entries) -> bytes:
    return json.dumps({"schemaVersion": 2, "mediaType": OCI_INDEX,
                       "manifests": [{"mediaType": OCI_MANIFEST, "digest": entry, "size": 1} for entry in entries]}).encode()


class FakeRegistry:
    """Manifests by digest; tags are never read. ``reads`` records each (repository, reference)."""

    def __init__(self, blobs=()):
        self.blobs = {_sha(body): body for body in blobs}
        self.reads = []

    def add(self, body: bytes) -> str:
        self.blobs = {**self.blobs, _sha(body): body}
        return _sha(body)

    def __call__(self, repository: str, reference: str) -> bytes:
        self.reads.append((repository, reference))
        return self.blobs[reference]


def _builds(fake: FakeRegistry):
    """Two per-arch builds as BuildKit pushes them: an index of the image and its attestation."""
    members = {arch: [_sha(f"{arch}-image".encode()), _sha(f"{arch}-attestation".encode())] for arch in ("amd64", "arm64")}
    built = {arch: fake.add(_index(entries)) for arch, entries in members.items()}
    return built, members


def _run_index(tmp_path, fake, index_digest, built, name=IMAGE, media_type=OCI_INDEX):
    built_dir = tmp_path / "digests"
    built_dir.mkdir(exist_ok=True)
    for digest in built:
        (built_dir / digest.removeprefix("sha256:")).touch()
    metadata = tmp_path / "index-metadata.json"
    metadata.write_text(json.dumps({"containerimage.descriptor": {"mediaType": media_type, "digest": index_digest,
                                                                   "size": 1}, "image.name": name}))
    outputs = tmp_path / "outputs"
    code = digests.main(["index", "--metadata", str(metadata), "--built", str(built_dir), "--image", IMAGE,
                         "--outputs", str(outputs)], fetch=fake)
    return code, outputs.read_text() if outputs.exists() else ""


# --------------------------------------------------------------------------- the images


def test_the_index_digest_is_the_create_steps_and_its_members_are_this_runs_builds(tmp_path):
    fake = FakeRegistry()
    built, members = _builds(fake)
    index = fake.add(_index(members["amd64"] + members["arm64"]))
    code, outputs = _run_index(tmp_path, fake, index, built.values())
    assert code == 0 and outputs == f"digest={index}\n"
    assert all(digests.DIGEST.match(reference) for _, reference in fake.reads), "read by digest, never by tag"
    assert {reference for _, reference in fake.reads} == {index, *built.values()}


def test_an_index_holding_a_manifest_this_run_did_not_build_signs_nothing(tmp_path, capsys):
    fake = FakeRegistry()
    built, members = _builds(fake)
    stranger = _sha(b"an image this run never scanned")
    index = fake.add(_index(members["amd64"] + [stranger, members["arm64"][1]]))
    code, outputs = _run_index(tmp_path, fake, index, built.values())
    out = capsys.readouterr().out
    assert code == 1 and outputs == ""
    assert f"holds {stranger}, which this run did not build" in out
    assert f"lacks {members['arm64'][0]}, which this run built and scanned" in out


def test_an_index_missing_an_architecture_signs_nothing(tmp_path, capsys):
    fake = FakeRegistry()
    built, members = _builds(fake)
    index = fake.add(_index(members["amd64"]))
    code, outputs = _run_index(tmp_path, fake, index, built.values())
    assert code == 1 and outputs == ""
    assert "lacks" in capsys.readouterr().out


def test_bytes_that_do_not_hash_to_the_digest_are_refused(tmp_path, capsys):
    fake = FakeRegistry()
    built, members = _builds(fake)
    index = _sha(_index(members["amd64"] + members["arm64"]))
    fake.blobs = {**fake.blobs, index: _index([_sha(b"swapped")])}
    code, outputs = _run_index(tmp_path, fake, index, built.values())
    assert code == 1 and outputs == ""
    assert "the registry served bytes that hash to" in capsys.readouterr().out


@pytest.mark.parametrize("name, media_type, digest, message", [
    ("ghcr.io/automatosai/automatos-frontend", OCI_INDEX, "sha256:" + "a" * 64, "not 'ghcr.io/automatosai/automatos-api'"),
    (IMAGE, OCI_MANIFEST, "sha256:" + "a" * 64, "not an index"),
    (IMAGE, OCI_INDEX, "sha256:short", "not a sha256 digest"),
])
def test_metadata_for_another_image_or_not_an_index_signs_nothing(tmp_path, capsys, name, media_type, digest, message):
    fake = FakeRegistry()
    built, _ = _builds(fake)
    code, outputs = _run_index(tmp_path, fake, digest, built.values(), name=name, media_type=media_type)
    assert code == 1 and outputs == "" and fake.reads == []
    assert message in capsys.readouterr().out


def test_no_builds_or_a_file_not_named_by_a_digest_signs_nothing(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(digests.DigestError, match="no build digests"):
        digests.built_digests(tmp_path / "empty")
    (tmp_path / "odd").mkdir()
    (tmp_path / "odd" / "not-a-digest").touch()
    with pytest.raises(digests.DigestError, match="not a sha256 digest"):
        digests.built_digests(tmp_path / "odd")


def test_a_build_pushed_as_a_plain_manifest_is_its_own_member():
    fake = FakeRegistry()
    plain = fake.add(json.dumps({"schemaVersion": 2, "mediaType": OCI_MANIFEST, "layers": []}).encode())
    index = fake.add(_index([plain]))
    assert digests.check_index(fake, IMAGE, index, [plain]) == []


# --------------------------------------------------------------------------- the chart


def _chart_manifest(package: bytes, layers=None) -> bytes:
    layer = {"mediaType": digests.CHART_LAYER, "digest": _sha(package), "size": len(package)}
    return json.dumps({"schemaVersion": 2, "mediaType": OCI_MANIFEST,
                       "config": {"mediaType": digests.CHART_CONFIG, "digest": _sha(b"{}"), "size": 2},
                       "layers": [layer] if layers is None else layers}).encode()


def _run_chart(tmp_path, fake, log: str, package: bytes):
    (tmp_path / "push.log").write_text(log)
    (tmp_path / "automatos-0.7.0.tgz").write_bytes(package)
    outputs = tmp_path / "outputs"
    code = digests.main(["chart", "--push-log", str(tmp_path / "push.log"), "--chart", CHART, "--version", "0.7.0",
                         "--package", str(tmp_path / "automatos-0.7.0.tgz"), "--outputs", str(outputs)], fetch=fake)
    return code, outputs.read_text() if outputs.exists() else ""


def test_the_chart_digest_is_the_one_helm_push_printed_and_holds_the_package(tmp_path):
    package = b"the packaged chart"
    fake = FakeRegistry()
    manifest = fake.add(_chart_manifest(package))
    code, outputs = _run_chart(tmp_path, fake, f"Pushed: {CHART}:0.7.0\nDigest: {manifest}\n", package)
    assert code == 0 and outputs == f"digest={manifest}\n"
    assert fake.reads == [(CHART, manifest)], "read by the printed digest, never by the version tag"


def test_a_pushed_chart_that_is_not_the_package_signs_nothing(tmp_path, capsys):
    package = b"the packaged chart"
    fake = FakeRegistry()
    manifest = fake.add(_chart_manifest(b"another chart"))
    code, outputs = _run_chart(tmp_path, fake, f"Pushed: {CHART}:0.7.0\nDigest: {manifest}\n", package)
    assert code == 1 and outputs == ""
    assert "not the package" in capsys.readouterr().out


@pytest.mark.parametrize("log, message", [
    (f"Pushed: {CHART}:0.6.0\nDigest: sha256:{'a' * 64}\n", "not ['ghcr.io/automatosai/charts/automatos:0.7.0']"),
    (f"Pushed: {CHART}:0.7.0\n", "printed 0 digests"),
    (f"Pushed: {CHART}:0.7.0\nDigest: sha256:{'a' * 64}\nDigest: sha256:{'b' * 64}\n", "printed 2 digests"),
    ("", "reported pushing []"),
])
def test_a_push_output_that_does_not_name_the_version_once_signs_nothing(tmp_path, capsys, log, message):
    fake = FakeRegistry()
    code, outputs = _run_chart(tmp_path, fake, log, b"chart")
    assert code == 1 and outputs == "" and fake.reads == []
    assert message in capsys.readouterr().out


def test_a_chart_with_an_extra_layer_or_another_config_is_refused():
    package = b"chart"
    extra = [{"mediaType": digests.CHART_LAYER, "digest": _sha(package), "size": len(package)},
             {"mediaType": "application/vnd.cncf.helm.chart.provenance.v1.prov", "digest": _sha(b"p"), "size": 1}]
    assert digests.check_chart(json.loads(_chart_manifest(package, extra)), package)
    other = {**json.loads(_chart_manifest(package)), "config": {"mediaType": "application/json"}}
    assert digests.check_chart(other, package) == [
        f"the pushed chart's config is a 'application/json', not {digests.CHART_CONFIG}"]


# --------------------------------------------------------------------------- the registry read


def test_manifest_bytes_are_the_bytes_the_registry_served(monkeypatch):
    body = b'{"schemaVersion":  2}'
    seen = []

    def fake_open(request):
        seen.append(request.full_url)
        return io.BytesIO(body)

    monkeypatch.setattr(registry, "_open", fake_open)
    session = registry.Session("https://ghcr.io", "automatosai/automatos-api", "t")
    assert registry.manifest_bytes(session, _sha(body)) == body
    assert registry.manifest(session, _sha(body)) == {"schemaVersion": 2}
    assert seen == [f"https://ghcr.io/v2/automatosai/automatos-api/manifests/{_sha(body)}"] * 2
