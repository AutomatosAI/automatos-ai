"""PRD-262 US-001: every image build reports its size, its layers and its update cost.

``scripts/ci/image_layers.py`` and ``scripts/ci/image_report.py`` are fed manifests
and configs recorded from the published ``edge`` images
(``tests/fixtures/prd262_edge_images.json``), so no registry is needed:

* the platform pick skips BuildKit's ``unknown/unknown`` attestation manifest;
* each layer gets the instruction that made it and a kind;
* ``edge`` plus a 10 MB ``public/`` layer is a +10 MB code-only update, and the
  budget check (the committed ``.github/image-budgets.json``) warns on it;
* a requirements change is reported apart from a code change;
* the API and the worker share only the Debian layer today;
* a layer download cut short is a warning in the report, never a failed build.
"""
from __future__ import annotations

import copy
import gzip
import http.client
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "prd262_edge_images.json"
_BUDGETS = _REPO / ".github" / "image-budgets.json"
TEN_MB = 10_000_000
NEW_DIGEST = "sha256:" + "ab" * 32


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / "scripts" / "ci" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # before exec, as the scripts register their siblings
    spec.loader.exec_module(module)
    return module


layers = _load("image_layers")
registry = _load("image_registry")
report = _load("image_report")
EDGE = json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _with_public_layer(image: dict) -> tuple[dict, dict]:
    """``image``'s manifest and config plus one 10 MB ``public/`` layer on top."""
    manifest = copy.deepcopy(image["manifest"])
    config = copy.deepcopy(image["config"])
    manifest["layers"].append({"mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
                               "digest": NEW_DIGEST, "size": TEN_MB})
    config["history"].append({"created_by": "COPY --chown=nextjs:nodejs /app/public ./public # buildkit"})
    return manifest, config


def _digests(image: dict) -> list[str]:
    return [layer["digest"] for layer in image["manifest"]["layers"]]


def test_the_platform_pick_skips_the_attestation_manifest():
    index = EDGE["frontend"]["index"]
    assert any(entry["platform"]["architecture"] == "unknown" for entry in index["manifests"])
    for arch in ("amd64", "arm64"):
        digest = registry.platform_digest(index, arch)
        entry = next(entry for entry in index["manifests"] if entry["digest"] == digest)
        assert entry["platform"] == {"architecture": arch, "os": "linux"}
    assert registry.platform_digest(EDGE["frontend"]["manifest"], "amd64") is None
    with pytest.raises(LookupError):
        registry.platform_digest(index, "s390x")


def test_each_layer_names_its_instruction_and_kind():
    rows = layers.layer_rows(EDGE["api"]["manifest"], EDGE["api"]["config"])
    assert len(rows) == 14
    by_instruction = {row["instruction"]: row["kind"] for row in rows}
    assert by_instruction["COPY /install /usr/local"] == "dependencies"  # the pip layer
    assert by_instruction["COPY . ."] == "code"
    assert rows[0]["kind"] == "system"  # the Debian root filesystem
    assert any(text.startswith("RUN pip uninstall") and kind == "code" for text, kind in by_instruction.items())
    frontend = layers.layer_rows(EDGE["frontend"]["manifest"], EDGE["frontend"]["config"])
    assert frontend[0]["kind"] == "system" and frontend[-1]["instruction"].endswith("/app/public ./public")


def test_a_history_that_does_not_match_the_layers_is_unclassified():
    manifest, config = EDGE["api"]["manifest"], {"history": []}
    rows = layers.layer_rows(manifest, config)
    assert {row["kind"] for row in rows} == {"unknown"}
    assert layers.update_delta(rows, [])["change"] == "unclassified"


def test_a_public_layer_on_edge_is_a_ten_mb_code_only_update():
    manifest, config = _with_public_layer(EDGE["frontend"])
    delta = layers.update_delta(layers.layer_rows(manifest, config), _digests(EDGE["frontend"]))
    assert delta["bytes"] == TEN_MB
    assert delta["layers"] == 1
    assert delta["change"] == "code-only"
    assert delta["by_kind"] == {"system": 0, "dependencies": 0, "code": TEN_MB}


def test_edge_against_itself_is_unchanged():
    rows = layers.layer_rows(EDGE["frontend"]["manifest"], EDGE["frontend"]["config"])
    delta = layers.update_delta(rows, _digests(EDGE["frontend"]))
    assert (delta["bytes"], delta["layers"], delta["change"]) == (0, 0, "unchanged")


def test_a_requirements_change_is_reported_apart_from_a_code_change():
    manifest = copy.deepcopy(EDGE["api"]["manifest"])
    rows = layers.layer_rows(manifest, EDGE["api"]["config"])
    changed = {"COPY /install /usr/local": "sha256:" + "01" * 32, "COPY . .": "sha256:" + "02" * 32}
    for position, row in enumerate(rows):
        if row["instruction"] in changed:
            manifest["layers"][position] = {**manifest["layers"][position], "digest": changed[row["instruction"]]}
    delta = layers.update_delta(layers.layer_rows(manifest, EDGE["api"]["config"]), _digests(EDGE["api"]))
    sizes = {row["instruction"]: row["size"] for row in rows}
    assert delta["change"] == "dependencies"
    assert delta["by_kind"]["dependencies"] == sizes["COPY /install /usr/local"]
    assert delta["by_kind"]["code"] == sizes["COPY . ."]


def test_the_budget_check_warns_on_the_larger_build():
    budgets = json.loads(_BUDGETS.read_text(encoding="utf-8"))
    assert budgets["mode"] == "warn", "report only until US-004 switches it to fail"
    edge_size = sum(layer["size"] for layer in EDGE["frontend"]["manifest"]["layers"])
    manifest, _ = _with_public_layer(EDGE["frontend"])
    larger = sum(layer["size"] for layer in manifest["layers"])
    assert layers.budget_verdict("frontend", "amd64", edge_size, budgets).level == "ok"
    verdict = layers.budget_verdict("frontend", "amd64", larger, budgets)
    assert verdict.level == "warning" and "over its" in verdict.message
    assert layers.budget_verdict("frontend", "amd64", larger, {**budgets, "mode": "fail"}).level == "error"
    assert layers.budget_verdict("nope", "amd64", 1, budgets).level == "warning"


def test_the_api_and_the_worker_share_only_the_debian_layer():
    shared = layers.shared_layers(EDGE["api"]["manifest"]["layers"], EDGE["workspace-worker"]["manifest"]["layers"])
    assert (shared["layers"], shared["bytes"]) == (1, 29_843_350)
    assert shared["trivial_layers"] == 1  # WORKDIR /app: a directory entry, not left out silently


def test_the_first_pull_counts_each_layer_once():
    lists = [EDGE[name]["manifest"]["layers"] for name in ("api", "frontend", "workspace-worker")]
    total = sum(layer["size"] for layer_list in lists for layer in layer_list)
    assert layers.unique_bytes(lists) == 1_091_050_360 < total


def _layer_tar(files: dict[str, bytes], whiteouts: tuple[str, ...] = ()) -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        directory = tarfile.TarInfo("usr/local/lib")
        directory.type = tarfile.DIRTYPE
        tar.addfile(directory)
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        for name in whiteouts:
            tar.addfile(tarfile.TarInfo(name))
    return gzip.compress(raw.getvalue())


SITE = "usr/local/lib/python3.11/site-packages"
LAYER = {f"{SITE}/pkg_a/x.py": b"a" * 100, f"{SITE}/pkg_a/y.py": b"b" * 50,
         f"{SITE}/pkg_b/z.py": b"c" * 10, "./app/main.py": b"d" * 5}


def test_a_layers_paths_are_summed_by_prefix():
    listing = layers.list_paths(io.BytesIO(_layer_tar(LAYER, ("app/.wh.old.py",))),
                                "application/vnd.oci.image.layer.v1.tar+gzip")
    assert (listing["files"], listing["bytes"], listing["whiteouts"]) == (4, 165, 1)
    assert listing["paths"][SITE] == [160, 3]
    assert listing["paths"]["app/main.py"] == [5, 1]
    assert listing["heaviest"][0] == [f"{SITE}/pkg_a", 150]


def test_a_zstd_layer_is_refused_not_misread():
    with pytest.raises(ValueError):
        layers.list_paths(io.BytesIO(b""), "application/vnd.oci.image.layer.v1.tar+zstd")


class _CutShort(io.BytesIO):
    """A blob download that breaks off mid-body, the way http.client reports it."""

    def read(self, *args):
        raise http.client.IncompleteRead(b"")


def test_a_layer_download_cut_short_is_warned_not_fatal(monkeypatch, capsys):
    """IncompleteRead is neither a URLError nor an OSError; it must not fail the build job."""
    assert issubclass(http.client.IncompleteRead, report.READ_ERRORS)
    monkeypatch.setattr(report.registry, "blob", lambda session, digest: _CutShort())
    row = {"digest": NEW_DIGEST, "media_type": "application/vnd.oci.image.layer.v1.tar+gzip"}
    assert set(report._listing(None, row)) == {"error"}
    assert "Layer not listed" in capsys.readouterr().out


def _fake_registry(monkeypatch, pushed: tuple[dict, dict], blobs: dict[str, bytes]):
    def image(ref, arch):
        source = EDGE["frontend"] if ref.endswith(":edge") else {"manifest": pushed[0], "config": pushed[1]}
        return {**source["manifest"], "digest": "sha256:" + "cd" * 32}, source["config"], None

    monkeypatch.setattr(report.registry, "image", image)
    monkeypatch.setattr(report.registry, "blob", lambda session, digest: io.BytesIO(blobs.get(digest, b"")))


def test_measure_writes_the_report_and_the_summary(monkeypatch, tmp_path, capsys):
    manifest, config = _with_public_layer(EDGE["frontend"])
    _fake_registry(monkeypatch, (manifest, config), {NEW_DIGEST: _layer_tar({"app/public/a.png": b"x" * 7})})
    out, summary = tmp_path / "frontend-amd64.json", tmp_path / "summary.md"
    assert report.main(["measure", "--name", "frontend", "--arch", "amd64", "--image", "localhost:5000/automatos-frontend",
                        "--digest", "sha256:" + "ef" * 32, "--edge", "ghcr.io/o/automatos-frontend:edge",
                        "--out", str(out), "--summary", str(summary)]) == 0
    written = json.loads(out.read_text())
    assert written["delta"]["bytes"] == TEN_MB and written["delta"]["change"] == "code-only"
    assert written["revision"] == "0d9f0ac594c3783e62a30e6c5ce75164028850db"
    assert [row["new"] for row in written["layers"]].count(True) == 1
    assert written["layers"][-1]["listing"]["paths"]["app/public/a.png"] == [7, 1]
    assert "Layer not listed" in capsys.readouterr().out  # the other fake blobs are empty: warned, not hidden
    text = summary.read_text()
    assert "**10.0 MB** in 1 new layers, a **code-only** change" in text
    assert "| 11 | 10.0 MB |" in text


def test_an_unpublished_image_reports_its_size_only(monkeypatch, tmp_path):
    _fake_registry(monkeypatch, (EDGE["api"]["manifest"], EDGE["api"]["config"]), {})
    out = tmp_path / "x.json"
    report.main(["measure", "--name", "agent-opt-worker", "--arch", "amd64", "--image", "localhost:5000/x",
                 "--digest", "sha256:" + "ef" * 32, "--edge", "", "--out", str(out)])
    written = json.loads(out.read_text())
    assert written["delta"] is None and written["edge"] is None
    assert written["bytes"] == 520_480_102 and "not published" in written["edge_note"]


def test_the_budget_command_warns_and_passes_in_warn_mode(tmp_path, capsys):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"name": "frontend", "arch": "amd64", "bytes": 103_851_176}))
    assert report.main(["budget", "--report", str(path), "--budgets", str(_BUDGETS)]) == 0
    assert capsys.readouterr().out.startswith("::warning title=Image size budget::frontend (amd64) is 103.9 MB")


def test_the_report_job_uses_edge_for_an_image_it_did_not_build(monkeypatch, tmp_path):
    def image(ref, arch):
        name = ref.rsplit("/automatos-", 1)[1].removesuffix(":edge")
        return EDGE[name]["manifest"], EDGE[name]["config"], None

    monkeypatch.setattr(report.registry, "image", image)
    built = {"name": "frontend", "arch": "amd64", "bytes": 1, "unpacked_bytes": None, "revision": "", "delta": None,
             "layers": [{"digest": NEW_DIGEST, "size": TEN_MB}]}
    (tmp_path / "frontend-amd64.json").write_text(json.dumps(built))
    outputs = tmp_path / "outputs"
    args = ["report", "--dir", str(tmp_path), "--edge-prefix", "ghcr.io/o/automatos-",
            "--out", str(tmp_path / "out.json"), "--outputs", str(outputs)]
    assert report.main(args) == 0
    stack = json.loads((tmp_path / "out.json").read_text())["stack"]
    assert stack["amd64"]["sources"] == {"api": "edge", "frontend": "this build", "workspace-worker": "edge"}
    assert stack["amd64"]["shared"]["layers"] == 1
    assert "shared_layers_amd64=1" in outputs.read_text().splitlines()


def test_markdown_cells_survive_pipes_in_an_instruction():
    rows = [{"size": 1, "kind": "system", "new": None, "listing": {}, "instruction": "RUN curl x | bash -"}]
    table = report._layer_table({"layers": rows})
    assert "`RUN curl x \\| bash -`" in table[-1]
