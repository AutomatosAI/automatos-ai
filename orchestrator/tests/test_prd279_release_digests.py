"""PRD-279 US-101, US-103, US-202: each digest a release signs is the one its own push made.

``scripts/ci/release_digests.py``:

* ``index``: the index digest comes from ``imagetools create``'s metadata file; there is
  one build per architecture the build matrix published, and the index, read by that
  digest, holds exactly those builds' entries;
* ``chart``: the chart digest comes from ``helm push``'s output, and the manifest it
  names must hold the packaged chart byte for byte;
* every read is by digest, bytes that do not hash to it are refused, and anything
  malformed fails with the same ``::error`` line and exit code as any other problem.

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
ARCHES = ("amd64", "arm64")


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
    """An OCI index of ``entries``: digests, or (digest, platform) pairs."""
    rows = [entry if isinstance(entry, tuple) else (entry, None) for entry in entries]
    manifests = [{"mediaType": OCI_MANIFEST, "digest": digest, "size": 1, **({"platform": platform} if platform else {})}
                 for digest, platform in rows]
    return json.dumps({"schemaVersion": 2, "mediaType": OCI_INDEX, "manifests": manifests}).encode()


def _metadata(digest: str, media_type: str = OCI_INDEX, size: int = 1) -> str:
    """The file ``imagetools create --metadata-file`` writes, as buildx v0.39.0 builds it.

    commands/imagetools/create.go:270-276 writes ``map[string]any{
    exptypes.ExporterImageDescriptorKey: desc, exptypes.ExporterImageNameKey:
    strings.Join(repoNames, ",")}`` through ``json.MarshalIndent(dt, "", "  ")`` (:426-432).
    The keys are "containerimage.descriptor" and "image.name" (vendored buildkit
    exporter/containerimage/exptypes/types.go:12,16); ``desc`` is the OCI descriptor
    ``combine`` returns (util/imagetools/create.go): mediaType, digest and size only;
    ``repoNames`` are the tags' untagged repository names, sorted (create.go:81-90).
    Go writes a map's keys sorted.
    """
    return json.dumps({"containerimage.descriptor": {"mediaType": media_type, "digest": digest, "size": size},
                       "image.name": IMAGE}, indent=2)


def _matrix(arches=ARCHES, image: str = IMAGE) -> str:
    """The build job's matrix, as scripts/ci/image_matrix.py writes an entry (the fields read here)."""
    rows = [{"name": "api", "arch": arch, "publish": True, "image": image} for arch in arches]
    rows += [{"name": "frontend", "arch": "amd64", "publish": True, "image": "ghcr.io/automatosai/automatos-frontend"},
             {"name": "media-render", "arch": "arm64", "publish": False, "image": "localhost:5000/automatos-media-render"}]
    return json.dumps({"include": rows})


class FakeRegistry:
    """Manifests by digest; tags are never read. ``reads`` records each (repository, reference)."""

    def __init__(self):
        self.blobs = {}
        self.reads = []

    def add(self, body: bytes) -> str:
        self.blobs = {**self.blobs, _sha(body): body}
        return _sha(body)

    def __call__(self, repository: str, reference: str) -> bytes:
        self.reads.append((repository, reference))
        return self.blobs[reference]


def _builds(fake: FakeRegistry, arches=ARCHES):
    """Per-arch builds as BuildKit pushes them: an index of the linux image and its attestation."""
    members = {arch: [_sha(f"{arch}-image".encode()), _sha(f"{arch}-attestation".encode())] for arch in arches}
    built = {arch: fake.add(_index([(entries[0], {"architecture": arch, "os": "linux"}),
                                    (entries[1], {"architecture": "unknown", "os": "unknown"})]))
             for arch, entries in members.items()}
    return built, members


def _run_index(tmp_path, fake, metadata: str, built, matrix: str = ""):
    built_dir = tmp_path / "digests"
    built_dir.mkdir(exist_ok=True)
    for digest in built:
        (built_dir / digest.removeprefix("sha256:")).touch()
    (tmp_path / "index-metadata.json").write_text(metadata)
    outputs = tmp_path / "outputs"
    code = digests.main(["index", "--metadata", str(tmp_path / "index-metadata.json"), "--built", str(built_dir),
                         "--image", IMAGE, "--matrix", matrix or _matrix(), "--outputs", str(outputs)], fetch=fake)
    return code, outputs.read_text() if outputs.exists() else ""


def _all(members, arches=ARCHES):
    return [entry for arch in arches for entry in members[arch]]


# --------------------------------------------------------------------------- the images


def test_the_index_digest_is_the_create_steps_and_holds_one_build_per_architecture(tmp_path):
    fake = FakeRegistry()
    built, members = _builds(fake)
    index = fake.add(_index(_all(members)))
    code, outputs = _run_index(tmp_path, fake, _metadata(index), built.values())
    assert code == 0 and outputs == f"digest={index}\n"
    assert all(digests.DIGEST.fullmatch(reference) for _, reference in fake.reads), "read by digest, never by tag"
    assert {reference for _, reference in fake.reads} == {index, *built.values()}


def test_a_one_architecture_run_passes_when_the_matrix_built_one(tmp_path):
    fake = FakeRegistry()
    built, members = _builds(fake, ("amd64",))
    index = fake.add(_index(members["amd64"]))
    code, outputs = _run_index(tmp_path, fake, _metadata(index), built.values(), _matrix(("amd64",)))
    assert code == 0 and outputs == f"digest={index}\n"


def test_a_missing_architecture_signs_nothing_though_the_index_matches_the_builds_present(tmp_path, capsys):
    fake = FakeRegistry()
    built, members = _builds(fake, ("amd64",))
    index = fake.add(_index(members["amd64"]))
    code, outputs = _run_index(tmp_path, fake, _metadata(index), built.values())
    assert code == 1 and outputs == ""
    assert "builds for ['amd64'], not one for each of ['amd64', 'arm64']" in capsys.readouterr().out


def test_an_extra_build_signs_nothing(tmp_path, capsys):
    fake = FakeRegistry()
    built, members = _builds(fake)
    spare = fake.add(_index([(_sha(b"another amd64 image"), {"architecture": "amd64", "os": "linux"})]))
    index = fake.add(_index(_all(members) + [_sha(b"another amd64 image")]))
    code, outputs = _run_index(tmp_path, fake, _metadata(index), [*built.values(), spare])
    assert code == 1 and outputs == ""
    assert "builds for ['amd64', 'amd64', 'arm64']" in capsys.readouterr().out


def test_an_index_holding_a_manifest_this_run_did_not_build_signs_nothing(tmp_path, capsys):
    fake = FakeRegistry()
    built, members = _builds(fake)
    stranger = _sha(b"an image this run never scanned")
    index = fake.add(_index(members["amd64"] + [stranger, members["arm64"][1]]))
    code, outputs = _run_index(tmp_path, fake, _metadata(index), built.values())
    out = capsys.readouterr().out
    assert code == 1 and outputs == ""
    assert f"holds {stranger}, which this run did not build" in out
    assert f"lacks {members['arm64'][0]}, which this run built and scanned" in out


def test_an_entry_listed_twice_signs_nothing(tmp_path, capsys):
    fake = FakeRegistry()
    built, members = _builds(fake)
    index = fake.add(_index(_all(members) + [members["amd64"][0]]))
    code, outputs = _run_index(tmp_path, fake, _metadata(index), built.values())
    assert code == 1 and outputs == ""
    assert "an entry is listed twice" in capsys.readouterr().out


def test_an_empty_index_or_an_empty_build_signs_nothing(tmp_path, capsys):
    fake = FakeRegistry()
    built, _ = _builds(fake)
    empty = fake.add(_index([]))
    (tmp_path / "index").mkdir()
    (tmp_path / "build").mkdir()
    assert _run_index(tmp_path / "index", fake, _metadata(empty), built.values()) == (1, "")
    assert "0 entries, the builds 4; neither may be empty" in capsys.readouterr().out
    assert _run_index(tmp_path / "build", fake, _metadata(empty), [empty], _matrix(("amd64",))) == (1, "")
    assert f"build {empty} holds 0 linux images, not one" in capsys.readouterr().out


def test_bytes_that_do_not_hash_to_the_digest_are_refused(tmp_path, capsys):
    fake = FakeRegistry()
    built, members = _builds(fake)
    index = _sha(_index(_all(members)))
    fake.blobs = {**fake.blobs, index: _index([_sha(b"swapped")])}
    code, outputs = _run_index(tmp_path, fake, _metadata(index), built.values())
    assert code == 1 and outputs == ""
    assert "the registry served bytes that hash to" in capsys.readouterr().out


@pytest.mark.parametrize("body", [
    json.dumps({"schemaVersion": 2, "manifests": ["not an object"]}).encode(),
    json.dumps({"schemaVersion": 2, "manifests": {"digest": "x"}}).encode(),
    json.dumps({"schemaVersion": 2, "mediaType": OCI_MANIFEST, "layers": []}).encode(),
    b'["a list, not an object"]',
])
def test_a_malformed_index_fails_with_the_same_error_line_and_no_traceback(tmp_path, capsys, body):
    fake = FakeRegistry()
    built, _ = _builds(fake)
    index = fake.add(body)
    code, outputs = _run_index(tmp_path, fake, _metadata(index), built.values())
    out = capsys.readouterr().out
    assert code == 1 and outputs == "" and out.count("::error title=Release digest::") == 1


def test_a_build_pushed_as_a_plain_manifest_is_refused(tmp_path, capsys):
    fake = FakeRegistry()
    plain = fake.add(json.dumps({"schemaVersion": 2, "mediaType": OCI_MANIFEST, "layers": []}).encode())
    index = fake.add(_index([plain]))
    assert _run_index(tmp_path, fake, _metadata(index), [plain], _matrix(("amd64",))) == (1, "")
    assert f"{plain} is not an index of descriptors" in capsys.readouterr().out


@pytest.mark.parametrize("metadata, message", [
    (_metadata("sha256:" + "a" * 64, media_type=OCI_MANIFEST), "names no pushed index"),
    (_metadata("sha256:short"), "not a sha256 digest"),
    (_metadata("sha256:" + "a" * 64 + "\n"), "not a sha256 digest"),
    (json.dumps({"image.name": IMAGE}), "names no pushed index"),
    ("[]", "names no pushed index"),
    ("not json", "Expecting value"),
])
def test_metadata_that_names_no_pushed_index_signs_nothing(tmp_path, capsys, metadata, message):
    fake = FakeRegistry()
    built, _ = _builds(fake)
    code, outputs = _run_index(tmp_path, fake, metadata, built.values())
    assert code == 1 and outputs == "" and fake.reads == []
    assert message in capsys.readouterr().out


def test_the_metadata_files_image_name_is_not_relied_on(tmp_path):
    """buildx writes every tagged repository there, comma-joined: the digest is what is checked."""
    fake = FakeRegistry()
    built, members = _builds(fake)
    index = fake.add(_index(_all(members)))
    listed = json.dumps({**json.loads(_metadata(index)), "image.name": f"{IMAGE},ghcr.io/automatosai/other"})
    assert _run_index(tmp_path, fake, listed, built.values()) == (0, f"digest={index}\n")


@pytest.mark.parametrize("matrix, message", [
    (_matrix(image="ghcr.io/automatosai/automatos-frontend-other"), "publishes no build of"),
    (json.dumps({"include": "nothing"}), "is not {\"include\": [entries]}"),
    (json.dumps([]), "is not {\"include\": [entries]}"),
])
def test_a_matrix_without_this_images_builds_signs_nothing(tmp_path, capsys, matrix, message):
    fake = FakeRegistry()
    built, members = _builds(fake)
    index = fake.add(_index(_all(members)))
    assert _run_index(tmp_path, fake, _metadata(index), built.values(), matrix) == (1, "")
    assert message in capsys.readouterr().out


def test_no_builds_or_a_file_not_named_by_a_digest_signs_nothing(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(digests.DigestError, match="no build digests"):
        digests.built_digests(tmp_path / "empty")
    (tmp_path / "odd").mkdir()
    (tmp_path / "odd" / "not-a-digest").touch()
    with pytest.raises(digests.DigestError, match="not a sha256 digest"):
        digests.built_digests(tmp_path / "odd")


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


def test_a_pushed_chart_that_is_not_the_package_signs_nothing_and_the_push_output_is_shown(tmp_path, capsys):
    package = b"the packaged chart"
    fake = FakeRegistry()
    manifest = fake.add(_chart_manifest(b"another chart"))
    code, outputs = _run_chart(tmp_path, fake, f"Pushed: {CHART}:0.7.0\nDigest: {manifest}\n", package)
    out = capsys.readouterr().out
    assert code == 1 and outputs == ""
    assert "not the package" in out and f"helm push printed:\nPushed: {CHART}:0.7.0\nDigest: {manifest}" in out


@pytest.mark.parametrize("log, message", [
    (f"Pushed: {CHART}:0.6.0\nDigest: sha256:{'a' * 64}\n", "not ['ghcr.io/automatosai/charts/automatos:0.7.0']"),
    (f"Pushed: {CHART}:0.7.0\n", "printed 0 digests"),
    (f"Pushed: {CHART}:0.7.0\nDigest: sha256:{'a' * 64}\nDigest: sha256:{'b' * 64}\n", "printed 2 digests"),
    (f"Pushed: {CHART}:0.7.0\nDigest: sha256:{'a' * 63}\n", "not a sha256 digest"),
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
    assert digests.check_chart({**other, "config": "a string", "layers": ["a string"]}, package) == [
        "the pushed chart's layers are not a list of descriptors"]


# --------------------------------------------------------------------------- the registry read


def _serve(monkeypatch, body: bytes) -> list:
    seen = []

    def fake_open(request):
        seen.append(request.full_url)
        return io.BytesIO(body)

    monkeypatch.setattr(registry, "_open", fake_open)
    return seen


def test_manifest_bytes_are_the_bytes_the_registry_served(monkeypatch):
    body = b'{"schemaVersion":  2}'
    seen = _serve(monkeypatch, body)
    session = registry.Session("https://ghcr.io", "automatosai/automatos-api", "t")
    assert registry.manifest_bytes(session, _sha(body)) == body
    assert registry.manifest(session, _sha(body)) == {"schemaVersion": 2}
    assert seen == [f"https://ghcr.io/v2/automatosai/automatos-api/manifests/{_sha(body)}"] * 2


def test_an_oversized_manifest_is_refused_not_read_whole(monkeypatch):
    assert registry.MAX_MANIFEST_BYTES == 4 * 1024 * 1024
    monkeypatch.setattr(registry, "MAX_MANIFEST_BYTES", 16)
    session = registry.Session("https://ghcr.io", "automatosai/automatos-api", None)
    _serve(monkeypatch, b"x" * 16)
    assert registry.manifest_bytes(session, "sha256:" + "a" * 64) == b"x" * 16
    _serve(monkeypatch, b"x" * 17)
    with pytest.raises(ValueError, match="over 16 bytes"):
        registry.manifest_bytes(session, "sha256:" + "a" * 64)
    assert ValueError in registry.READ_ERRORS, "the callers that report a failed read catch it"
